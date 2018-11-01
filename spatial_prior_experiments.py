import argparse
import torch.utils.data
import random
import time
import numpy as np
from pydoc import locate
import scipy.misc
import csv
import copy
import os
import option
import models
import datasets
import utils
from point_cloud import Depth2BEV
from tqdm import tqdm

opt = option.make(argparse.ArgumentParser())

d = datasets.IntPhys(opt, 'paths_test')
indices = list(range(9999))
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

def get_visibility_probabilities(tp_dets, fn_dets, vg, cam, ball_radius=17.5):
    """
    Args:
        tp_dets: List of (i,j,k) pixel coordinates in [250,348,35] of TP dets for an image
        fn_dets: List of (i,j,k) pixel coordinates in [250,348,35] of FN dets (gt objects 
            missed by the detector)
        vg: The visibility grid with dims [1/35,250,348]
        cam: the coordinates of the camera in [250,348]
    Return:
        tp_probs, fn_probs: Lists of probabilities
    """
    # c, h, w = vg.shape
    h, w = vg.shape
    def shifted(cam_x, cam_y, obj_x, obj_y):
        if cam_y == obj_y:
            theta = np.pi/2
        else:
            theta = np.arctan((cam_x - obj_x) / (cam_y - obj_y))
        xa = ball_radius * np.cos(theta)
        yb = ball_radius * np.sin(theta)
        if obj_y <= cam_y:
            y = obj_y + yb
        else:
            y = obj_y - yb
        x = obj_x - xa
        x = min(max(int(np.floor(x)), 0), h-2)
        y = min(max(int(np.floor(y)), 0), w-2)
        return x,y 
    
    def average(x, y, vg_):
        vg_ = vg_.squeeze()
        u = vg_[x, y+1].item()
        d = vg_[x, y-1].item()
        l = vg_[x-1, y].item()
        r = vg_[x+1, y].item()
        return np.mean(np.array([u, d, l, r, vg_[x,y].item()]))

    tp_probs, fn_probs = [], []
    for t in tp_dets:
        if type(t) == tuple:
            t = t[1]
        x = t[0]
        y = t[1]
        if type(x) == torch.Tensor:
            x = x.item()
        if type(y) == torch.Tensor:
            y = y.item()
        # shift towards camera by ball radius
        x, y = shifted(cam[0].item(), cam[1].item(), x, y)
        # take average of neighborhood
        #tp_probs.append(average(x,y,vg[t[2]]))
        tp_probs.append(average(x,y,vg))
    for f in fn_dets:
        x = f[0]
        y = f[1]
        if type(x) == torch.Tensor:
            x = x.item()
        if type(y) == torch.Tensor:
            y = y.item()
        # shift towards camera by ball radius
        x, y = shifted(cam[0].item(), cam[1].item(), x, y)
        #fn_probs.append(average(x,y,vg[f[2]]))
        fn_probs.append(average(x,y,vg))
    return tp_probs, fn_probs

def get_proximity(source_obj, idx, other_obj, other_occluders, cam,
        thresholds=[10,20,40]):
    """
    Given a source object, compute the minimum angle to 
    the other objects and occluders. 
    Args: 
        source_obj: (i,j,k) tuple
        other_obj: List of (i,j,k) tuples (potentially empty)
        other_occluders: List of (i,j,k) tuples (potentially empty)
    """
    cam = (cam[0].item(), cam[1].item())
    def angle_to_cam(obj, cam):
        if cam[1] - obj[1] == 0:
            return 0
        return np.rad2deg(np.arctan((cam[0] - obj[0])/(cam[1] - obj[1])))
    
    angles = []
    if type(source_obj[0]) == torch.Tensor:
        source_obj[0] = source_obj[0].item()
        source_obj[1] = source_obj[1].item()
    src_obj_angle_to_cam = angle_to_cam(source_obj, cam)
    for idx_, o in enumerate(other_obj):
        if type(o[0]) == torch.Tensor:
            o[0] = o[0].item()
            o[1] = o[1].item()
        if idx == idx_:
            continue
        if (source_obj[0]-o[0]) == 0 and (source_obj[1]-o[1]) == 0:
            continue
        a = angle_to_cam(o, cam)
        angles.append(abs(src_obj_angle_to_cam - a))
    for o in other_occluders:
        if type(o[0]) == torch.Tensor:
            o[0] = o[0].item()
            o[1] = o[1].item()
        a = angle_to_cam(o, cam)
        angles.append(abs(src_obj_angle_to_cam - a))
    
    if len(angles) == 0:
        return -1
    return min(angles)

    #min_angle = min(angles)
    #if min_angle <= thresholds[0]:
    #    return thresholds[0]
    #elif min_angle <= thresholds[1]:
    #    return thresholds[1]
    #elif min_angle <= thresholds[2]:
    #    return thresholds[2]
    #else:
    #    return -1

overlap_ratios = [1.0]
conf_threshs = [0.9]
ball_radii = [opt.ball_radius]

outfile = open(os.path.join(opt.results, 'spatial_prior_experiment_results.csv'), 'w+')
fieldnames = ['frame', 'tp', 'detector_conf', 'prior_conf', 'angular_proximity', 'num_objects', 'num_occluders', 'distance_to_camera']
csv_writer = csv.DictWriter(outfile, fieldnames=fieldnames)
csv_writer.writeheader()

viz = utils.Viz(opt)
model.eval()
#i = indices[0]
n = 0
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
                if not opt.use_occluded:
                    pc = data[0]['point_cloud'].squeeze()
                    vg = data[0]['VG'].squeeze()
                    i = data[0]['frame'][0]
                    detections, dets_px, bev_scores, fv_scores = model.predict(data, d.depth2bev)
                    objects = data[1]['objects']
                    gt_balls_px = copy.deepcopy(data[1]['objects_px']['balls'])
                    gt_walls_px = copy.deepcopy(data[1]['objects_px']['walls'])
                    found_objs = 0
                    tp_objs = []
                    for det, det_px in zip(detections, dets_px):
                        j = -1; found = False
                        for obj in objects:
                            obj = obj.numpy()[0]
                            j += 1
                            dist = np.linalg.norm(obj - det)
                            if dist/(overlap_ratio * 2 * opt.ball_radius) <= 1.:
                                found = True
                                found_objs += 1
                                break
                        if found:
                            del objects[j]
                            # get the index into the gt objs
                            # TODO: test
                            for idx, b in enumerate(data[1]['objects_px']['balls']):
                                if gt_balls_px[j] == b:
                                    break
                            del gt_balls_px[j]
                            tp_objs.append((idx,det_px))
                    # gathering statistics for experiment below
                    num_objects = len(data[1]['objects_px']['balls'])
                    num_occluders = len(gt_walls_px)
                    # either there are 2+ objects or there is at least 1 object and 1 occluder
                    if num_objects <= 1 and num_occluders == 0:
                        continue
                    tp_det_probs, fn_det_probs = [], []
                    tp_min_angle_cat, fn_min_angle_cat = [], []
                    tp_cam_dist, fn_cam_dist = [], []
                    # true positives
                    for (idx,o) in tp_objs:
                        tp_det_probs.append(o[3].item())
                        tp_min_angle_cat.append(get_proximity(
                            o[:3], idx, data[1]['objects_px']['balls'],
                            gt_walls_px, data[0]['cam'], thresholds=opt.angular_thresh))
                        tp_cam_dist.append(np.sqrt(np.square((o[0] - data[0]['cam'][0])) + np.square(o[1] - data[0]['cam'][1])).item())
                    # false negatives
                    for k in range(len(gt_balls_px)):
                        a, b = gt_balls_px[k][0], gt_balls_px[k][1]
                        a_ = int(np.floor(a / 4)) # TODO
                        b_ = int(np.floor(b / 4)) # TODO
                        bev_score = bev_scores[a_, b_]
                        fn_det_probs.append(bev_score)
                        fn_min_angle_cat.append(get_proximity(
                            gt_balls_px[k], -1, data[1]['objects_px']['balls'],
                            gt_walls_px, data[0]['cam'], thresholds=opt.angular_thresh))
                        fn_cam_dist.append(np.sqrt(np.square((a - data[0]['cam'][0])) + np.square(b - data[0]['cam'][1])).item())
                    tp_vg_probs, fn_vg_probs = get_visibility_probabilities(
                            tp_objs, gt_balls_px, vg, data[0]['cam'])
                    for tp_det_prob, tp_vg_prob, tp_angle, tp_cam in zip(tp_det_probs, tp_vg_probs, tp_min_angle_cat, tp_cam_dist):
                        #if tp_angle == -1:
                        #    tp_angle = '{}'.format(1+opt.angular_thresh[-1])
                        csv_writer.writerow({'frame': i, 'tp': 1, 'detector_conf': round(tp_det_prob,3),
                            'prior_conf': round(tp_vg_prob,3), 'angular_proximity': round(tp_angle,3),
                            'num_objects': num_objects, 'num_occluders': num_occluders, 'distance_to_camera': round(tp_cam, 3)})
                    for fn_det_prob, fn_vg_prob, fn_angle, fn_cam in zip(fn_det_probs, fn_vg_probs, fn_min_angle_cat, fn_cam_dist):
                        #if fn_angle == -1:
                        #    fn_angle = '{}'.format(1+opt.angular_thresh[-1])
                        csv_writer.writerow({'frame': i, 'tp': 0, 'detector_conf': round(fn_det_prob,3),
                            'prior_conf': round(fn_vg_prob,3), 'angular_proximity': round(fn_angle,3),
                            'num_objects': num_objects, 'num_occluders': num_occluders, 'distance_to_camera': round(fn_cam, 3)})
                    if opt.image_save:
                        #start = time.time()
                        #Depth2BEV.display_point_cloud(pc, '3d', detections, opt.ball_radius, True, name='eval_imgs/{}.png'.format(i))
                        #diff = time.time() - start
                        #print("display pc time: {}".format(diff))
                        for o in tp_objs:
                            vg_ = np.zeros((vg.shape[2], vg.shape[3], 3))
                            for a in range(vg.shape[2]):
                                for b in range(vg.shape[3]):
                                    vg_[a,b,:] = np.array([0, vg[o[2], a, b] * 255, 0])
                            
                            vg_[o[0]-5:o[0]+5, o[1]-5:o[1]+5] = np.array([255, 0, 0])
                            vg_resized = scipy.misc.imresize(vg_, 200)
                            scipy.misc.imsave('eval_imgs/vg/{}.png'.format(n), vg_resized)
                            n += 1
                else:
                    vg = data[0]['VG'].squeeze()
                    i = data[0]['frame'][0]
                    gt_balls_px = copy.deepcopy(data[1]['objects_px']['balls'])
                    gt_walls_px = copy.deepcopy(data[1]['objects_px']['walls'])
                    # gathering statistics for experiment below
                    num_objects = len(data[1]['objects_px']['balls'])
                    num_occluders = len(gt_walls_px)
                    # either there are 2+ objects or there is at least 1 object and 1 occluder
                    if num_objects <= 1 and num_occluders == 0:
                        continue
                    _, vg_probs = get_visibility_probabilities([], gt_balls_px, vg, data[0]['cam'])
                    for k in range(len(gt_balls_px)):
                        a, b = gt_balls_px[k][0], gt_balls_px[k][1]
                        a_ = int(np.floor(a / 4)) # TODO
                        b_ = int(np.floor(b / 4)) # TODO
                        min_angle = get_proximity(
                            gt_balls_px[k], -1, data[1]['objects_px']['balls'],
                            gt_walls_px, data[0]['cam'])
                        cam_dist = np.sqrt(np.square((a - data[0]['cam'][0])) + np.square(b - data[0]['cam'][1])).item()
                        csv_writer.writerow({'frame': i, 'tp': -1, 'detector_conf': -1,
                            'prior_conf': round(vg_probs[k],3), 'angular_proximity': round(min_angle,3),
                            'num_objects': num_objects, 'num_occluders': num_occluders, 'distance_to_camera': round(cam_dist, 3)})
                        
                #i += 1
            
outfile.close()
print('Done')
        
