import argparse
import torch.utils.data
import random
import time
import numpy as np
from pydoc import locate
import scipy.misc
import csv
import os
import option
import models
import datasets
import utils
from point_cloud import Depth2BEV

from tqdm import tqdm

opt = option.make(argparse.ArgumentParser())

#d = datasets.IntPhys(opt, 'paths_test')
d = datasets.IntPhys(opt, 'paths_val')
indices = [4246]
#indices = list(range(1, 5000))
#indices = list(range(1200,1300))
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

false_positives = 0
false_negatives = 0
true_positives = 0

# precision = true_positives / (true_positives + false_positives)
# recall = true_positives / (true_positives + false_negatives)
overlap_ratios = [1.0]
#conf_threshs = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]
conf_threshs = [0.9]
ball_radii = [opt.ball_radius]

#outfile = open(os.path.join(opt.results, 'detector_eval_results.csv'), 'w+')
#fieldnames = ['conf thresh', 'overlap ratio', 'ball radius', 'precision', 'recall']
#csv_writer = csv.DictWriter(outfile, fieldnames=fieldnames)
#csv_writer.writeheader()

if opt.save_detections:
    #detections_file = open(os.path.join('/data/pemami/intphys/val_detections.csv'), 'w+')
    detections_file = open(os.path.join(opt.results, 'val_detections.csv'), 'w+')
    #fieldnames = ['height','depth','width','p(i-1;j-1)', 'p(i;j-1)', 'p(i+1;j-1)', 'p(i-1;j)', 'p(i;j)', 'p(i+1;j)', 'p(i-1;j+1)', 'p(i;j+1)', 'p(i+1;j+1)']
    fieldnames = ['idx', 'height', 'depth', 'width', 'p']
    det_csv_writer = csv.DictWriter(detections_file, fieldnames=fieldnames)
    det_csv_writer.writeheader()

viz = utils.Viz(opt)
model.eval()
i = d.indices[0]
for radius in ball_radii:
    opt.ball_radius = radius
    model.bev_pixor.ball_radius = radius
    model.fv_pixor.ball_radius = radius
    for overlap_ratio in overlap_ratios:
        for conf_th in conf_threshs:
            opt.conf_thresh = conf_th
            model.bev_pixor.conf_thresh = conf_th
            model.fv_pixor.conf_thresh = conf_th
            for data in tqdm(valLoader):
                #pc = data[0]['point_cloud'].squeeze()
                # numpy array [N,4]
                #detections, bev_scores, fv_scores, labeled_bev_scores, labeled_fv_scores = model.predict(data, d.depth2bev)
                detections, dets_px, bev_scores, _ = model.predict(data, d.depth2bev)
                print(detections)
                objects = data[1]['objects']
                found_objs = 0
                for det, dp in zip(detections, dets_px):
                    j = -1; found = False
                    for obj in objects:
                        obj = obj.numpy()[0]
                        j += 1
                        #dist_bev = np.linalg.norm(obj[0:2] - det[0:2])
                        #dist_fv = abs(obj[2] - det[2])
                        #if dist_bev/(overlap_ratio * 2 * opt.ball_radius) <= 1. and \
                        #        dist_fv/(overlap_ratio * 4 * opt.ball_radius) <= 1.:
                        dist = np.linalg.norm(obj - det)
                        if dist/(overlap_ratio * 2 * opt.ball_radius) <= 1.:
                            found = True
                            true_positives += 1
                            found_objs += 1
                            print(det)
                            print(obj)
                            x, y, z=dp[0], dp[1], dp[2]
                            x = int(round(x/4))
                            y = int(round(y/4))
                            z = int(round(z/4))
                            print(bev_scores[x,y])
                            if opt.save_detections:
                                x, y, z=dp[0], dp[1], dp[2]
                                x = int(round(x/4))
                                y = int(round(y/4))
                                z = int(round(z/4))
                                det_csv_writer.writerow({'idx': i, 'height': round(det[2],3), 'depth': round(det[0],3), 'width': round(det[1],3), 'p': bev_scores[x,y]})
                            break
                    if found:
                        del objects[j]
                # this should be 0 if all objects were accounted for
                false_negatives += len(objects)
                false_positives += len(detections) - found_objs
                
                # store bev scores for false negatives
                if opt.save_detections:
                    for obj in objects:
                        y,x = d.depth2bev.point_2_grid_cell(obj[0], scale=4)
                        det_csv_writer.writerow({'idx': i, 'height': round(obj[0,2].item(),3), 'depth': round(obj[0,0].item(),3), 'width': round(obj[0,1].item(),3), 'p': bev_scores[x,y]})

                if opt.image_save or opt.visdom:
                    start = time.time()
                    Depth2BEV.display_point_cloud(pc, '3d', detections, opt.ball_radius, True, name='eval_imgs/{}.png'.format(i))
                    diff = time.time() - start
                    print("display pc time: {}".format(diff))
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
            
            prec = 0
            tp_fp = true_positives + false_positives
            if tp_fp > 0:
                prec = true_positives / tp_fp

            recall = 0
            tp_fn = true_positives + false_negatives
            if tp_fn > 0:
                recall = true_positives / tp_fn
            
            print("threshold: {}, ball radius: {}, overlap ratio: {}, precision: {}, recall: {}".format(
                opt.conf_thresh, opt.ball_radius, overlap_ratio, prec, recall))
            csv_writer.writerow({'conf thresh': opt.conf_thresh, 'overlap ratio': overlap_ratio,
                'ball radius': opt.ball_radius, 'precision': prec, 'recall': recall})
            false_positives = 0
            false_negatives = 0
            true_positives = 0
outfile.close()
print('Done')
        
