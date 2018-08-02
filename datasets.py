import torch.utils.data
import random
import scipy.misc
import numpy as np
import os
import math

import utils
from tqdm import tqdm
from point_cloud import Depth2BEV
import time

class SubsetSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

class IntPhys(torch.utils.data.Dataset):

    def __init__(self, opt, split):
        self.opt = opt
        self.index = 0
        self.test = 'test' in split
        self.depth2bev = Depth2BEV(opt)
        if opt.list:
            self.file = os.path.join(opt.list, split + '.npy')
            self.paths = np.load(self.file).tolist()
            if opt.remove_no_objects:
                self.no_object_indices = \
                        np.load(os.path.join(opt.list,
                        split + '_no_object_indices.npy')).tolist()
            else:
                self.no_object_indices = []
            count = min(opt.count, len(self.paths)) * self.opt.m
            #count = min(opt.count, len(self.paths))
        else:
            self.pattern = opt.pattern
            count = opt.count * opt.m
            count = count * 0.9 if split == 'train' else count * 0.1
            count  = int(count)
            self.i0 = 1 if split == 'train'  else int(0.9 * opt.count + 1)
        # Create list of valid indices for subset sampler
        self.indices = list(range(count))
        if len(self.no_object_indices) > 1:
            self.indices = list(set(self.indices) - set(self.no_object_indices))
        self.count = len(self.indices)
        self.count = self.count - (self.count % opt.bsz)
        vars(opt)['n_sample_%s' %split] = self.count
        vars(opt)['nbatch_%s' %split] = int(self.count / opt.bsz)
        print('n_sample_%s: %s' %(split, self.count))
        self.last_offsets = None
        self.last_flip = False
        self.manhattan_dist = 3

    def __getitem__(self, index):
        video_idx = math.floor(index / self.opt.m)
        video_path = self._getpath(video_idx)
        frame_idx = index % self.opt.m

        def extract_random_image_patch(obj_array, bev):
            c, h, w = bev.shape
            obj = obj_array[np.random.randint(0, len(obj_array))]
            i = obj[0]; j = obj[1]
            # randomly shift 
            i += np.random.randint(-5, 5)
            j += np.random.randint(-5, 5)
            h_min = max(0, j-24); h_max = min(h-1, j+24)
            w_min = max(0, i-24); w_max = min(w-1, i+24)
            crop = bev[:, h_min:h_max, w_min:w_max]
            # if crop is not 48x48, pad with zeros
            crop_c, crop_h, crop_w = crop.shape
            final_crop = np.zeros((crop_c, 48, 48), dtype=np.uint8)
            padding = []
            for chw in [crop_h, crop_w]:
                if chw < 48:
                    short = 48 - chw
                    if short % 2 == 0:
                        pad_width = (int(short/2), int(short/2))
                    else:
                        l = np.floor(short/2)
                        pad_width = (int(l+1), int(l))
                    padding.append(pad_width)
                else:
                    padding.append((0,0))
            for r in range(crop_c):
                final_crop[r] = np.pad(crop[r], padding, 'constant', constant_values=0)
            # randomly flip
            if np.random.uniform() < 0.5:
                for r in range(crop_c):
                    final_crop[r] = np.flipud(final_crop[r]).copy()
            return final_crop

        def load_BEV_crop(idx, label):
            idx += 1
            # if idx is even it's positive example
            # if odd, negative example
            if label:
                if idx % 2 == 1:
                    return {'binary_target': np.array([0.])}
                else:
                    return {'binary_target': np.array([1.])}
            else:
                with open('%s/annotations/%03d.txt' %(video_path, idx), 'r') as f:
                    max_depth = float(f.readline())
                    # this cast to np.float32 is very important...
                    depth_img = np.float32(scipy.misc.imread(
                        '%s/depth/depth_%03d.png' %(video_path, idx)))
                    # uses the fixed depth, not the actual max depth
                    # need to rescale objects
                    pc = self.depth2bev.depth_2_point_cloud(depth_img)
                    # Note that this BEV map is already in PyTorch CHW format
                    bev, offsets = self.depth2bev.point_cloud_2_BEV(pc)
                    c, h, w = bev.shape
                    self.last_offsets = offsets
                    gt_objects = []
                    for line in f:
                        pos = np.array(list(map(float, line.split(" "))))
                        rescaled_pos = self.depth2bev.backproject_and_rescale(
                                pos, self.depth2bev.fixed_depth / max_depth)
                        for r in range(3): rescaled_pos[r] += self.last_offsets[r]
                        # pixel location of the object
                        i, j = self.depth2bev.point_2_grid_cell(rescaled_pos, scale=1)
                        # if object is in view
                        if 0 <= i < w and 0 <= j < h:
                            gt_objects.append((i,j))
                    # grab occluders too
                    gt_occluders = []
                    _, _, occluders = self.depth2bev.parse_status('%s/status.json' %(video_path))
                    frame = occluders[idx-1]
                    for pos in frame:
                        rescaled_pos = self.depth2bev.backproject_and_rescale(
                                pos, self.depth2bev.fixed_depth / max_depth)
                        for r in range(3): rescaled_pos[r] += self.last_offsets[r]
                        # pixel location of the object
                        i, j = self.depth2bev.point_2_grid_cell(rescaled_pos, scale=1)
                        # if object is in view
                        if 0 <= i < w and 0 <= j < h:
                            gt_occluders.append((i,j))
                    # if positive
                    if idx % 2 == 0 and len(gt_objects) > 0:
                        # extract a 48 x 48 crop around a random item in the video
                        final_crop = extract_random_image_patch(gt_objects, bev)
                    # negative examples
                    elif idx % 2 == 1 or (idx % 2 == 0 and len(gt_objects) == 0):
                        # If there is an occluder, randomly sample patch from it, otherwise use a random patch
                        if len(gt_occluders) > 0:
                            final_crop = extract_random_image_patch(gt_occluders, bev)
                        else:
                            # Sample a 48x48 patch from the image that doesn't overlap gt patches
                            done = False
                            while not done:
                                i = np.random.randint(24, w-24)
                                j = np.random.randint(24, h-24)
                                if len(gt_objects) == 0:
                                    done = True
                                else:
                                    for obj in gt_objects:
                                        gt_i = obj[0]; gt_j = obj[1];
                                        if abs(gt_i - i) > 24 and abs(gt_j - j) > 24:
                                            done = True
                            h_min = max(0, j-24); h_max = min(h-1, j+24)
                            w_min = max(0, i-24); w_max = min(w-1, i+24)
                            crop = bev[:, h_min:h_max, w_min:w_max]
                            crop_c, crop_h, crop_w = crop.shape
                            # randomly flip
                            if np.random.uniform() < 0.5:
                                for r in range(crop_c):
                                    crop[r] = np.flipud(crop[r]).copy()
                            final_crop = crop
                    return final_crop

        def load_BEV(idx, label):
            # The output BEV prediction maps is downsampled
            # by some amount
            pixor_downsample = 4
            idx += 1
            if not label:
                if np.random.uniform() < 0.5:
                    self.last_flip = True
                else:
                    self.last_flip = False
                # this cast to np.float32 is very important...
                depth_img = np.float32(scipy.misc.imread(
                    '%s/depth/depth_%03d.png' %(video_path, idx)))
                # uses the fixed depth, not the actual max depth
                # need to rescale objects
                pc = self.depth2bev.depth_2_point_cloud(depth_img)
                # Note that this BEV map is already in PyTorch CHW format
                bev, offsets = self.depth2bev.point_cloud_2_BEV(pc)
                self.last_offsets = offsets
                if self.last_flip and not self.test:
                    for r in range(bev.shape[0]):
                        bev[r] = np.flipud(bev[r]).copy()
                if self.test:
                    return {'BEV': bev, 'point_cloud': pc}
                else:
                    return bev
            else:
                with open('%s/annotations/%03d.txt' %(video_path, idx), 'r') as f:
                    max_depth = float(f.readline())
                    # 87
                    grid_x = int(np.ceil((self.depth2bev.bev_x_dim / self.depth2bev.bev_x_res) / pixor_downsample))
                    # 63
                    grid_y = int(np.ceil((self.depth2bev.bev_y_dim / self.depth2bev.bev_y_res) / pixor_downsample))
                    # 9
                    grid_z = int(np.ceil(self.depth2bev.bev_z_channels / pixor_downsample))
                    # Use H x W for easy integration with PyTorch
                    binary_map = np.zeros((grid_y, grid_x))
                    height_map = np.zeros((grid_y, grid_x))
                    regression_map = np.zeros((3, grid_y, grid_x))                     
                    for line in f:
                        pos = np.array(list(map(float, line.split(" "))))
                        rescaled_pos = self.depth2bev.backproject_and_rescale(
                                pos, self.depth2bev.fixed_depth / max_depth)
                        for r in range(3): rescaled_pos[r] += self.last_offsets[r]
                        # pixel location of the object
                        i, j = self.depth2bev.point_2_grid_cell(rescaled_pos, scale=pixor_downsample)
                        k = self.depth2bev.z_2_grid(rescaled_pos[2], scale=pixor_downsample)
                        # set pixels in ~50 pixel radius to 1 (50 / 10 / 4 ~ 2)
                        if 0 <= i < grid_x and 0 <= j < grid_y and 0 <= k < grid_z:
                            if k < 0:
                                print('%s/annotations/%03d.txt' %(video_path, idx), pos[2], rescaled_pos[2], k)
                            c = (i, j) 
                            px = utils.get_nearby_pixels(c, self.manhattan_dist, (grid_y, grid_x))
                            for p in px: # positives
                                binary_map[p[1], p[0]] = 1
                                height_map[p[1], p[0]] = k
                                # compute dx, dy, and dz for each grid cell in the set of positives 
                                x, y, z = self.depth2bev.grid_cell_2_point(p[0], p[1], scale=pixor_downsample, k=k)
                                dx = rescaled_pos[0] - x; dy = rescaled_pos[1] - y; dz = rescaled_pos[2] - z
                                for r,d in enumerate([dx, dy, dz]):
                                    # normalize to N(0,1)
                                    d = (d - self.depth2bev.regression_stats[r][0]) / self.depth2bev.regression_stats[r][1]
                                    regression_map[r, p[1], p[0]] = d
                    if self.last_flip:
                        binary_map = np.flipud(binary_map).copy()
                        height_map = np.flipud(height_map).copy()
                        for r in range(3):
                            regression_map[r] = np.flipud(regression_map[r]).copy()
                    return {'binary_target': binary_map, 'z_target': height_map, 'regression_target': regression_map}

        def load(x, nc, start, seq, interp, c):
            out = []
            for j,f in enumerate(seq):
                if f == 'z':
                    f = 'x' if random.random() < self.opt.px else 'y'
                if f == 'y':
                    ri = random.randint(0, len(self.paths) - 1)
                    v = self.paths[ri].decode('UTF-8')
                else:
                    v = os.path.join(video_path, c)
                if f == 'x' or f == 'y':
                    f = random.randint(1, self.opt.n_frames) - start
                    assert not self.test
                img = scipy.misc.imread(
                    '%s/%s/%s_%03d.png' %(v, x, x, start + f),
                    #mode='RGB'
                )
                out.append(scipy.misc.imresize(
                    img,
                    (self.opt.frame_height, self.opt.frame_width),
                    interp))
            return np.array(out)

        def load_diff(x, nc, start, seq, interp, c):
            if self.opt.residual == 0:
                return load(x, nc, start, seq, interp, c)
            else:
                out0 = load(x, nc, start, seq, interp, c)
                out1 = load(x, nc, start + self.opt.residual, seq, interp, c)
                return out1 - out0

        def make_output(x, start, seq, c='.'):
            if x == 'edge':
                raise NotImplementedError
            elif x == 'depth':
                return load_diff('depth', 1, start, seq, 'bilinear', c)
            elif x == 'bev-depth':
                return load_BEV(start, label=False)
            elif x == 'bev-label':
                return load_BEV(start, label=True)
            elif x == 'bev-crop':
                return load_BEV_crop(start, label=False)
            elif x == 'bev-crop-label':
                return load_BEV_crop(start, label=True)
            elif x == 'mask':
                mask_value = utils.get_mask_index(
                    os.path.join(video_path, str(c), 'status.json'),
                    self.opt.mask_object
                )
                raw_mask = load_diff('mask', 1, start, seq, 'nearest', c)
                mask = raw_mask.astype(int)
                out = [np.ones(mask.shape, dtype=bool)]
                for o in self.opt.mask_object:
                    m = np.zeros(mask.shape, dtype=bool)
                    for v in mask_value[o]:
                        m[mask == v] = True
                        out[0][mask == v] = False
                    out.append(m)
                return np.transpose(np.array(out, dtype=int), (1, 0, 2, 3))
            elif x == 'scene':
                out = load_diff('scene', self.opt.num_channels, start, seq,
                               'bilinear', c).astype(float) / 255
                return np.transpose(out, (0, 3, 1, 2))
            else:
                print('Unknown opt.input or opt.target: ' + x)
                return None
        if self.test:
            # TODO
            #input_, target = [], []
            #for c in range(1, 5):
            #    input_.append(make_output(
            #        self.opt.input, frame_idx, self.opt.input_seq, str(c)
            #    ))
            #    target.append(make_output(
            #        self.opt.target, frame_idx, self.opt.target_seq, str(c)
            #    ))
            #input_ = np.array(input_)
            #target = np.array(target)
            input_ = make_output(
                self.opt.input, frame_idx, self.opt.input_seq
            )
            target = make_output(
                self.opt.target, frame_idx, self.opt.target_seq
            )
        else:
            input_ = make_output(
                self.opt.input, frame_idx, self.opt.input_seq
            )
            target = make_output(
                self.opt.target, frame_idx, self.opt.target_seq
            )

        #out.video_path = video_path
        return input_, target

    def __len__(self):
        return self.count

    def _getpath(self, video_idx):
        if hasattr(self, 'paths'):
            try:
                video_path = self.paths[video_idx].decode('UTF-8')
            except AttributeError:
                video_path = self.paths[video_idx]
        else:
            video_path = self.pattern %(self.i0 + video_idx)
        return video_path
