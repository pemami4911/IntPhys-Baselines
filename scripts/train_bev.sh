#!/bin/sh
cd ..
CUDA_VISIBLE_DEVICES=1,2 python train.py --verbose --visdom --model MVPIXOR --input_seq 1 --target_seq 1 --input bev-depth --target bev-label --bsz 32 --gpu --n_slices 10 --nThreads 8 --n_gpus 2 --manualSeed 16360 --normalize_regression --remove_no_objects --lr 0.001 --n_epochs 10 --load bev=checkpoints/PIXORBinary_180731_164016/pixorbinary_1_1.pth --load fv=checkpoints/PIXORBinary_180820_140731/pixorbinary_1_0.pth --pixor_head full
