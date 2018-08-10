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
    #detections, bev_scores, fv_scores, labeled_bev_scores, labeled_fv_scores = model.predict(data, d.depth2bev)
    detections = model.predict(data, d.depth2bev)
    print(detections)
    if opt.image_save or opt.visdom:
        Depth2BEV.display_point_cloud(pc, '3d', detections, opt.ball_radius, True, name='eval_imgs/{}.png'.format(i))
        """ 
        bev_scores = scipy.misc.imresize(bev_scores * 255, 200)
        scipy.misc.imsave('eval_imgs/bev_scores_{}.png'.format(i), bev_scores)
        fv_scores = scipy.misc.imresize(fv_scores * 255, 200)
        scipy.misc.imsave('eval_imgs/fv_scores_{}.png'.format(i), fv_scores)
        labeled_bev_scores = scipy.misc.imresize(labeled_bev_scores, 200)
        scipy.misc.imsave('eval_imgs/labeled_bev_scores_{}.png'.format(i), labeled_bev_scores)
        labeled_fv_scores = scipy.misc.imresize(labeled_fv_scores, 200)
        scipy.misc.imsave('eval_imgs/labeled_fv_scores_{}.png'.format(i), labeled_fv_scores)
        
        tmp = data[0]['FV'][0][0] * 255
        for j in range(1,data[0]['FV'][0].shape[0]):
            tmp |= data[0]['FV'][0][j,:,:] * 255
        scipy.misc.imsave('eval_imgs/fv_{}.png'.format(i), tmp)
        """
    i += 1
    #exit(0)
print('Done')
        
