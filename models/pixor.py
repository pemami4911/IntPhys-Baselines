import torch
import torch.nn as nn
from torchvision.models.resnet import Bottleneck

class PIXOR(nn.Module):
    """
    Some conv arithmetic: when using 3x3 convs with 
    stride 1 or 2, use padding of 1 to preserve dimensions!
    """

    def __init__(self, image_dims):
        super(PIXOR, self).__init__()
        self.x_dim, self.y_dim, self.z_dim = image_dims
        # Initial processing by 2 Conv2d layers with
        # 3x3 kernel, stride 1, padding 1
        self.block1 = nn.Sequential(
            nn.Conv2d(self.z_dim, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32))   
        self.resnet_inplanes = 32
        # Blocks 2-5
        self.block2 = self._make_layer(Bottleneck, 24, 3, stride=2)
        self.block3 = self._make_layer(Bottleneck, 48, 6, stride=2)
        self.block4 = self._make_layer(Bottleneck, 64, 6, stride=2)
        self.block5 = self._make_layer(Bottleneck, 96, 4, stride=2)
        # Blocks 6-7
        self.u_bend = nn.Conv2d(384, 196, 1, 1)
        self.block6 = self._make_deconv_layer(196, 128, output_padding=1)
        self.block7 = self._make_deconv_layer(128, 96, output_padding=(1,0))
        self.block4_6 = nn.Conv2d(256, 128, 1, 1)
        self.block3_7 = nn.Conv2d(192, 96, 1, 1)

        # Head network
        self.header = nn.Sequential(
                nn.Conv2d(96, 96, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(96),
                nn.Conv2d(96, 96, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(96),
                nn.Conv2d(96, 96, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(96),
                nn.Conv2d(96, 96, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(96))
        self.classification_out = nn.Sequential(
                nn.Conv2d(96, 1, 3, 1, 1),
                nn.Sigmoid())
        self.regression_out = nn.Conv2d(96, 3, 3, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_deconv_layer(self, in_planes, out_planes, output_padding=0):
        upsample = nn.Sequential(
                nn.ConvTranspose2d(in_planes, out_planes, 3, 2, 1, 
                    output_padding=output_padding),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_planes))
        return upsample

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.resnet_inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    nn.Conv2d(self.resnet_inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.resnet_inplanes, planes, stride, downsample))
        self.resnet_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.resnet_inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        x: input tensor of [bsz, C, H, W], where C=35,
          H=384, W=250
        """
        # Backbone network
        x = self.block1(x)
        # input [32,384,250]
        # output [96,192,125]
        x = self.block2(x)
        # output [192,96,63]
        x3 = self.block3(x)
        # output [256,48,32]
        x4 = self.block4(x3)
        # output [384,24,16]
        x = self.block5(x4)
        # Upsampling fusion
        # output [196, 24, 16]
        x = self.u_bend(x)
        # output is [128, 48, 32]
        x = self.block4_6(x4) + self.block6(x)
        # output is [96, 96, 63]
        x = self.block3_7(x3) + self.block7(x)

        # Head network
        x = self.header(x)
        # [1, 96, 63] - 0-1
        co = self.classification_out(x)
        # [3, 96, 63] - x,y,z
        ro = self.regression_out(x)
        
        return co, ro

if __name__ == '__main__':
    import numpy as np
    pix = PIXOR([348, 250, 35])
    pix = pix.cuda()
    model_parameters = filter(lambda p: p.requires_grad, pix.parameters())
    print("# of trainable actor parameters: {}".format(sum([np.prod(p.size()) for p in model_parameters])))

    x = torch.zeros(1, 35, 384, 250).cuda()
    co, ro = pix(x)
    print(co.shape, ro.shape)
