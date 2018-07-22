#!/bin/sh
python train.py --verbose --visdom --model PIXOR --input_seq 1 --target_seq 1 --input bev-depth --target bev-label --bsz 4 --gpu --n_slices 5
