import argparse
import torch.utils.data
import random
import time
import numpy as np
from pydoc import locate
import scipy.misc

import option
import models
import datasets
import utils
from point_cloud import Depth2BEV

from tqdm import tqdm

opt = option.make(argparse.ArgumentParser())

d = datasets.IntPhys(opt, 'paths_test')
indices = list(range(1200, 1301))
valLoader = torch.utils.data.DataLoader(
    d,
    1,
    num_workers=opt.nThreads,
    sampler=datasets.SubsetSampler(indices)
)
opt.nbatch_val = len(valLoader)
print(opt)
np.random.seed(opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.gpu:
    torch.cuda.manual_seed_all(opt.manualSeed)

model = locate('models.%s' %opt.model)(opt)
if opt.load:
    model.load(opt.load, 'test')
if opt.gpu:
    model.gpu()

print('n parameters: %d' %sum([m.numel() for m in model.parameters()]))

viz = utils.Viz(opt)
model.eval()
i = indices[0]
for data in tqdm(valLoader):
    pc = data[0]['point_cloud'].squeeze()
    # numpy array [N,4]
    detections, scores = model.predict(data, d.depth2bev)
    print(detections)
    if opt.image_save or opt.visdom:
        Depth2BEV.display_point_cloud(pc, '3d', detections, opt.ball_radius, True, name='eval_imgs/{}.png'.format(i))
        #scipy.misc.imsave('eval_imgs/scores_{}.png'.format(i), scores * 255)
        #tmp = frame[0,:,:] * 255
        #for j in range(1,frame.shape[0]):
        #    tmp |= frame[j,:,:] * 255
        #scipy.misc.imsave('eval_imgs/bev_{}.png'.format(i), tmp)
    i += 1
print('Done')
        
