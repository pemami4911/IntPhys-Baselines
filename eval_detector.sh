#!/bin/sh
CUDA_VISIBLE_DEVICES=0 python eval_detector.py --verbose --image_save --model PIXOR --input_seq 1 --target_seq 1 --input bev-depth --target bev-label --bsz 1 --gpu --nThreads 0 --n_gpus 1 --manualSeed 3434 --normalize_regression --load 'checkpoints/PIXOR_180801_234124/pixor_0_6.pth' --pixor_head full --conf_thresh 0.7 --disable_checkpoint --IOU_thresh 0.0 --ball_radius 100
