#!/bin/sh
cd ..
CUDA_VISIBLE_DEVICES=1,2 python train.py --verbose --model VNet --input_seq 1 --target_seq 1 --input bev-prior --target bev-prior-label --bsz 8 --gpu --n_slices 10 --nThreads 8 --n_gpus 1 --manualSeed 16360 --normalize_regression --remove_images_with_no_objects --lr 0.0001 --n_epochs 2 --random_flip 0.0 --pretrained_bev 'checkpoints/MVPIXOR_180810_133249/bev_pixor_0_8.pth' --train_detections_file '/data/pemami/intphys/train_detections.csv' --val_detections_file '/data/pemami/intphys/val_detections.csv' --pixor_head viz  --visdom
