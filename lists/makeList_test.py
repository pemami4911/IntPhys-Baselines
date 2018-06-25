import os
from subprocess import call
import argparse
from copy import deepcopy

import makeList_train

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', default='.')
#parser.add_argument('--blocks', nargs='+', default=['occluded_dynamic_1_nObj1', 'occluded_dynamic_1_nObj2', 'occluded_dynamic_1_nObj3', 'occluded_dynamic_2_nObj1', 'occluded_dynamic_2_nObj2', 'occluded_dynamic_2_nObj3', 'occluded_static_nObj1', 'occluded_static_nObj2', 'occluded_static_nObj3', 'visible_dynamic_1_nObj1', 'visible_dynamic_1_nObj2', 'visible_dynamic_1_nObj3', 'visible_dynamic_2_nObj1', 'visible_dynamic_2_nObj2', 'visible_dynamic_2_nObj3', 'visible_static_nObj1', 'visible_static_nObj2', 'visible_static_nObj3'])
parser.add_argument('--blocks', nargs='+', default=[])
parser.add_argument('--prop_train', type=float, default=0)
parser.add_argument('--prop_val', type=float, default=0)
parser.add_argument('--prop_test', type=float, default=1)
parser.add_argument('--train_file', default='train.npy')
parser.add_argument('--val_file', default='val.npy')
parser.add_argument('--test_file', default='test.npy')
parser.add_argument('--datadir', nargs='+', default=[])
opt = parser.parse_args()
#for b in opt.blocks:
#    tmpOpt = deepcopy(opt)
    #vars(tmpOpt)['pattern'] = os.path.join(tmpOpt.datadir[0], b)
#    vars(tmpOpt)['pattern'] = ''
    #vars(tmpOpt)['outdir'] = 
#    makeList_train.makeList(tmpOpt)
vars(opt)['pattern'] = ''
makeList_train.makeList(opt)
