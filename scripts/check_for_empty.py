import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import numpy as np

basedir = '/data/pemami/intphys/train'

subdirs = os.listdir(basedir)
max_depths = []
for subdir in tqdm(subdirs):
    if '.json' in subdir:
        continue
    with open(os.path.join(basedir, subdir, 'status.json'), 'r') as f:
        stuff = json.load(f)
        max_depths.append(float(stuff['max_depth']))

print(np.mean(max_depths))
print(np.var(max_depths))
plt.hist(max_depths)
plt.show()
