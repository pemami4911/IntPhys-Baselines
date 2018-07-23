#!/bin/sh
python train.py --verbose --visdom --model PIXOR --input_seq 1 --target_seq 1 --input bev-depth --target bev-label --bsz 16 --gpu --n_slices 10 --nThreads 12 --n_gpus 4 --manualSeed 1337 
