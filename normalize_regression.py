import argparse
import torch.utils.data
import random
import time
import numpy as np

import option
import models
import datasets
import utils

from tqdm import tqdm

def update_regression_stat(d, mean, var_Mk, var_Sk, k):
    d = d[d != 0.].numpy()
    M = len(d)
    mean = (k/(k+M)) * mean + (np.sum(d) / (k+M))
    var_Mk_1 = var_Mk
    for i in range(len(d)):
        if k == 0:
            var_Mk = d[i]
        k += 1
        var_Mk_1 = var_Mk + (d[i] - var_Mk)/k
        var_Sk += (d[i] - var_Mk) * (d[i] - var_Mk_1)
    return mean, var_Mk_1, var_Sk, kf


if __name__ == '__main__':
    opt = option.make(argparse.ArgumentParser())

    trainLoader = torch.utils.data.DataLoader(
	datasets.IntPhys(opt, 'paths_train'),
	opt.bsz,
	num_workers=opt.nThreads,
	shuffle=True
    )
    opt.nbatch_train = len(trainLoader)
    print(opt)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    counter = 0
    k_dx = 0; k_dy = 0; k_dz = 0;
    mean_dx = 0.; var_dx_Mk = 0.; var_dx_Sk = 0.;
    mean_dy = 0.; var_dy_Mk = 0.; var_dy_Sk = 0.;
    mean_dz = 0.; var_dz_Mk = 0.; var_dz_Sk = 0.;
    # iterate over training and val sets and compute
    for b in tqdm(trainLoader):
        regression = b[1]['regression_target']
        for t in range(opt.bsz):
            dxs = regression[t][0]
            dys = regression[t][1]
            dzs = regression[t][2]
            mean_dx, var_dx_Mk, var_dx_Sk, k_dx \
                    = update_regression_stat(dxs, mean_dx, var_dx_Mk, var_dx_Sk, k_dx)
            mean_dy, var_dy_Mk, var_dy_Sk, k_dy \
                    = update_regression_stat(dys, mean_dy, var_dy_Mk, var_dy_Sk, k_dy)
            mean_dz, var_dz_Mk, var_dz_Sk, k_dz \
                    = update_regression_stat(dzs, mean_dz, var_dz_Mk, var_dz_Sk, k_dz)
        counter += 1
        if counter % 1000 == 0:
            for x,y in zip([mean_dx, mean_dy, mean_dz], [var_dx_Sk/(k_dx - 1), \
                    var_dy_Sk/(k_dy-1), var_dz_Sk/(k_dz-1)]):
                print(x, y)
    with open(opt.regression_statistics_file, 'w') as rs:
        for x,y in zip([mean_dx, mean_dy, mean_dz], [var_dx_Sk/(k_dx - 1), \
                var_dy_Sk/(k_dy-1), var_dz_Sk/(k_dz-1)]):
            rs.write("3 {} {}\n".format(x, y))
            print(x, y)
