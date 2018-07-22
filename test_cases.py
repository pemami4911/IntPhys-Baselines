""" Test focal loss, NMS"""
import torch
import argparse
import numpy as np
from models.pixor import PIXOR

def test_focal_loss(model):
    x = torch.FloatTensor([[[
        [-0.023, -0.1, -1],
        [-3, 0.004, 1],
        [0.31, -0.65, -0.89]
        ]]])
    y = torch.FloatTensor([[
        [0, 0, 0],
        [0, 0, 0],
        [0, 1, 0]]])
    y = y.view(1,1,9)
    fl = model.focal_loss(x, y)
    assert np.allclose(fl.item(), 0.1400, rtol=1e-2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--bev_dims', nargs='+', default=[348, 250, 35], type=int)
    parser.add_argument('--conf_thresh', type=float, default=0.6)
    parser.add_argument('--ball_radius', type=float, default=50)
    parser.add_argument('--IOU_thresh', type=float, default=0.5)
    parser.add_argument('--bsz', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.01)

    opt = parser.parse_args()

    torch.manual_seed(opt.seed)
    pxor = PIXOR(opt)

    test_focal_loss(pxor)
