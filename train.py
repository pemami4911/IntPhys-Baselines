import argparse
import torch.utils.data
import random
import time
import numpy as np
from pydoc import locate

import option
import models
import datasets
import utils

from tqdm import tqdm

opt = option.make(argparse.ArgumentParser())

d1 = datasets.IntPhys(opt, 'paths_train')
trainLoader = torch.utils.data.DataLoader(
    d1,
    opt.bsz,
    num_workers=opt.nThreads,
    sampler=torch.utils.data.sampler.SubsetRandomSampler(d1.indices)
)
d2 = datasets.IntPhys(opt, 'paths_val')
valLoader = torch.utils.data.DataLoader(
    d2,
    opt.bsz,
    num_workers=opt.nThreads,
    sampler=torch.utils.data.sampler.SubsetRandomSampler(d2.indices)
)
opt.nbatch_train = len(trainLoader) 
opt.nbatch_val = len(valLoader)
print(opt)
np.random.seed(opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.gpu:
    torch.cuda.manual_seed_all(opt.manualSeed)

model = locate('models.%s' %opt.model)(opt)
if opt.load:
    model.load(opt.load, 'train')
if opt.gpu:
    model.gpu()

print('n parameters: %d' %sum([m.numel() for m in model.parameters()]))

viz = utils.Viz(opt)
def process_batch(batch, loss, i, k, set_, t0):
    """Optimization step.

    batch = [input, target]: contains data for optim step [input, target]
    loss: dict containing statistics about optimization
    i: epoch
    k: index of the current batch
    set_: type of batch (\"train\" or \"dev\")
    t0: time of the beginning of epoch
    """

    nbatch = vars(opt)['nbatch_' + set_]
    res = model.step(batch, set_)
    for key, value in res.items():
        try:
            loss[key].append(value)
        except KeyError:
            loss[key] = [value]
    if opt.verbose:
        batch_time = (time.time() - t0) / (k + 1)
        eta = nbatch * batch_time
        out = ' %s %d: batch %.5d/%.5d |' %(set_, i, k, nbatch - 1)
        for key, value in res.items():
            out += ' %s: %.2e |' %(key, value)
        out += ' batch time: %.2fs | %s eta: %.2dH%.2dm%.2ds' \
            %(batch_time, set_, eta / (60 * 60), (eta / 60) % 60, eta % 60)
        print(out, end='\r')
    if opt.image_save or opt.visdom:
        if 'bev' not in opt.input:
            to_plot = []
            nviz = min(10, opt.bsz)
            to_plot.append(utils.stack(batch[0], nviz, opt.input_len))
            to_plot.append(utils.stack(batch[1], nviz, opt.target_len))
            to_plot.append(utils.stack(model.output(), nviz, opt.target_len))
            img = np.concatenate(to_plot, 2)
        elif opt.input == 'bev-depth':
            to_plot = []
            bev, fv, bev_targets, fv_targets = utils.bev_batch_viz(batch)
            to_plot.append((bev, 'BEV'))
            to_plot.append((fv, 'FV'))
            to_plot.append((bev_targets, 'BEV targets'))
            to_plot.append((fv_targets, 'FV targets'))
            to_plot.append((model.output(), 'BEV and FV predictions'))
            img = to_plot
        elif opt.input == 'bev-crop':
            img = []
            bev_crop, fv_crop, fv_full, label = utils.bev_crop_viz(batch)
            #img.append(bev_crop)
            img.append((fv_full, label))
            img.append((fv_crop, label))
        viz(img, loss, i, k, nbatch, set_)
    return loss

loss_train, loss_val, log = {}, {}, []

try:
    for i in range(opt.n_epochs):
        log.append([])
        t_optim = 0
        t0 = time.time()
        train_slices = utils.slice_epoch(opt.nbatch_train, opt.n_slices)
        val_slices = utils.slice_epoch(opt.nbatch_val, opt.n_slices)
        for ts, vs, j in zip(train_slices, val_slices, range(opt.n_slices)):
            log[i].append({})
            model.train()
            for k, batch in tqdm(zip(ts, trainLoader)):
                t = time.time()
                loss_train = process_batch(batch, loss_train, i, k, 'train', t0)
                t_optim += time.time() - t
            for key, value in loss_train.items():
                log[i][j][key] = float(np.mean(value[-opt.nbatch_train:]))
            log[i][j]['train_batch'] = k
            model.eval()
            for k, batch in zip(vs, valLoader):
                t = time.time()
                loss_val = process_batch(batch, loss_val, i, k, 'val', t0)
                t_optim += time.time() - t
            for key, value in loss_val.items():
                log[i][j][key] = float(np.mean(value[-opt.nbatch_val:]))
            # optionally update LR after each epoch/minibatch
            model.lr_step()
            utils.checkpoint('%d_%d' %(i, j), model, log, opt)
            log[i][j]['time(optim)'] = '%.2f(%.2f)' %(time.time() - t0, t_optim)
            print(log[i][j])

except KeyboardInterrupt:
    time.sleep(2) # waiting for all threads to stop
    print('-' * 89)
    save = input('Exiting early, save the last model?[y/n]')
    if save == 'y':
        print('Saving...')
        utils.checkpoint('final', model, log, opt)

