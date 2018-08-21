#!/bin/sh
CUDA_VISIBLE_DEVICES=1,2 python train.py --verbose --visdom --model PIXORBinary --input_seq 1 --target_seq 1 --input bev-crop --target bev-crop-label --bsz 32 --gpu --n_slices 10 --nThreads 8 --n_gpus 2 --manualSeed 1714 --remove_no_objects --lr 0.01 --n_epochs 3 --pixor_head binary --view FV --crop_sz 48
