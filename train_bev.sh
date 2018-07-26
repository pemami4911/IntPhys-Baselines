#!/bin/sh
CUDA_VISIBLE_DEVICES=1,2 python train.py --verbose --visdom --model PIXOR --input_seq 1 --target_seq 1 --input bev-depth --target bev-label --bsz 32 --gpu --n_slices 10 --nThreads 8 --n_gpus 2 --manualSeed 1341 --normalize_regression --remove_no_objects --lr 0.01 --n_epochs 5 
