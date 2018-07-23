import torch
import torch.nn as nn
from .model import Model

class DataParallel(Model):
    def __init__(self, opt, m, device_ids):
        super(Model, self).__init__(opt)
        self.m = nn.DataParallel(m, device_ids)

    def step(self, batch):
        return self.m.step(batch)


