#!/bin/sh
cd ..
CUDA_VISIBLE_DEVICES=2 python eval_detector.py --verbose --model MVPIXOR --input_seq 1 --target_seq 1 --input bev-depth --target bev-label --bsz 1 --gpu --nThreads 8 --n_gpus 1 --manualSeed 3435 --normalize_regression --load bev=checkpoints/MVPIXOR_180810_133249/bev_pixor_0_8.pth --load fv=checkpoints/MVPIXOR_180810_133249/fv_pixor_0_8.pth --pixor_head full --conf_thresh 0.9 --disable_checkpoint --IOU_thresh 0.0 --ball_radius 175 --image_save
