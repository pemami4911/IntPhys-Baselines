#!/bin/sh
CUDA_VISIBLE_DEVICES=1,2 python train.py --verbose --visdom --model PIXOR --input_seq 1 --target_seq 1 --input bev-crop --target bev-crop-label --bsz 32 --gpu --n_slices 10 --nThreads 8 --n_gpus 2 --manualSeed 1713 --remove_no_objects --lr 0.01 --n_epochs 10 --pixor_head full --load checkpoints/PIXORBinary_180731_080934/pixorbinary_0_5.pth
