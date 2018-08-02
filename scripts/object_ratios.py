import os
import numpy as np
from tqdm import tqdm

def get_no_object_indices(dir_list, set_):
    indices = []
    dirs = np.load(dir_list).tolist()
    i = 0
    for subdir in tqdm(dirs):
        for j in range(100):
            with open(os.path.join(subdir, "annotations", "%03d.txt" %(j+1))) as f:
                lines = f.read().strip().split("\n")
                if len(lines) == 1:
                    indices.append(i)    
            i += 1
    print(len(indices))
    np.save("0.7_train/{}_no_object_indices.npy".format(set_), np.array(indices)) 


def count_object_ratios():
    basedir = "/data/pemami/intphys/train"
    n0 = 0; n1 = 0; n2 = 0; n3 = 0; n = 0
    for subdir in tqdm(sorted(os.listdir(basedir))):
        if '.json' in subdir:
            continue
        for i in range(100):
            with open(os.path.join(basedir, subdir, "annotations", "%03d.txt" %(i+1))) as f:
                lines = f.read().strip().split("\n")
                if len(lines) == 1:
                    n0 += 1
                elif len(lines) == 2:
                    n1 += 1
                elif len(lines) == 3:
                    n2 += 1
                elif len(lines) == 4:
                    n3 += 1
                n += 1
    print("0 objects: {}, 1 object: {}, 2 objects: {}, 3 objects: {}".format(n0/n, n1/n, n2/n, n3/n))

if __name__ == '__main__':
    get_no_object_indices("0.7_train/paths_train.npy", "train")
    get_no_object_indices("0.7_train/paths_val.npy", "val")
