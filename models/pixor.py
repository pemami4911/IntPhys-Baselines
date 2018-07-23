import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models.resnet import Bottleneck
import torch.optim as optim
from .model import Model

class PIXOR(Model):
    """
    Separate model (the IntPhys model API) and PIXORNet to use nn.DataParallel :( 
    """
    def __init__(self, opt, test=False):
        super(Model).__init__()
        self.x_dim = opt.bev_dims[0] # width
        self.y_dim = opt.bev_dims[1] # height
        self.z_dim = opt.bev_dims[2] # channels
        self.target_x_dim = 63
        self.target_y_dim = 87
        self.conf_thresh = opt.conf_thresh
        self.ball_radius = opt.ball_radius
        self.IOU_thresh = opt.IOU_thresh
        self.bsz = 1 if test else opt.bsz
        self.pixor = PIXORNet(opt)
        # stuff for forward pass
        # N C H W format for Pytorch
        self.input = Variable(torch.FloatTensor(self.bsz, self.z_dim, self.y_dim, self.x_dim))
        self.regression_target = Variable(torch.FloatTensor(self.bsz, 3, 
            self.target_x_dim * self.target_y_dim))
        self.binary_class_target = Variable(torch.FloatTensor(self.bsz, 1,
            self.target_x_dim * self.target_y_dim))
        self.categorical_target = Variable(torch.FloatTensor(
            self.bsz * self.target_x_dim * self.target_y_dim, 1))
        self.input.pin_memory()
        self.regression_target.pin_memory()
        self.binary_class_target.pin_memory()
        self.categorical_target.pin_memory()
        # optimization
        self.smoothL1 = nn.SmoothL1Loss()
        self.nll = nn.NLLLoss()
        self.optimizer = optim.SGD(self.pixor.parameters(), lr=opt.lr, momentum=0.9)
        self.optim_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[20,30], gamma=0.1)
        if opt.n_gpus > 1:
            self.pixor = nn.DataParallel(self.pixor, device_ids=list(range(opt.n_gpus)))

    def parameters(self):
        return self.pixor.parameters()
    
    def train(self):
        self.pixor.train()

    def eval(self):
        self.pixor.eval()

    def step(self, batch, set_):
        """
        Args:
            batch[0] is batch of input maps of size [N,C,H,W]
            batch[1]['regression_target'] is regression targets of size [N,3,H,W]
            batch[1]['binary_target'] is binary class target of size [N,1,H,W]
            batch[1]['z_target'] is categorical target of size [N,1,H,W]
        """
        self.input.data.copy_(batch[0].squeeze())
        # flatten last two dimensions
        self.regression_target.data.copy_(batch[1]['regression_target'].view(self.bsz, 3, self.target_x_dim * self.target_y_dim))
        self.binary_class_target.data.copy_(batch[1]['binary_target'].view(self.bsz, 1, self.target_x_dim * self.target_y_dim))
        self.categorical_target.data.copy_(batch[1]['z_target'].view(self.bsz * self.target_x_dim * self.target_y_dim, 1))
        self.reg_out, self.binary_out, self.cat_out = self.pixor(self.input)
        binary_classification_loss = self.focal_loss(self.binary_out, self.binary_class_target)
        # If there are no objects in the scene, no regression or cat loss
        if not self.binary_class_target.byte().any():
            total_loss = binary_classification_loss
            result = {'{}_total_loss'.format(set_): total_loss.item(),
                    '{}_binary_classification_loss'.format(set_): binary_classification_loss.item()}
        else:
            # regression loss only computed over positive samples
            reg_out = self.reg_out.view(self.bsz, 3, self.target_x_dim * self.target_y_dim)
            # zero out predictions not at positive samples
            reg_out = reg_out * self.binary_class_target
            regression_loss = self.smoothL1(reg_out, self.regression_target)
            # compute categorical loss
            # zero out predictions not a positive samples
            # channel-wise softmax, in shape [N,C] (9 classes)
            # TODO: don't hardcode 9
            cat_out = self.cat_out.view(-1, 9)
            binary_target_flat = self.binary_class_target.view(-1)
            # select only positive samples
            cat_out = cat_out[binary_target_flat.byte()]
            cat_target_out = self.categorical_target[binary_target_flat.byte()]
            height_logits = F.log_softmax(cat_out, dim=1)
            categorical_loss = self.nll(height_logits, cat_target_out.long().squeeze())
            total_loss = binary_classification_loss + regression_loss + categorical_loss
            result = {'{}_regression_loss'.format(set_): regression_loss.item(),
                    '{}_binary_classification_loss'.format(set_): binary_classification_loss.item(),
                    '{}_categorical_loss'.format(set_): categorical_loss.item(),
                    '{}_total_loss'.format(set_): total_loss.item()}
        if set_ == 'train':
            self.pixor.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        return result 
    
    def gpu(self):
        self.pixor.cuda()
        self.input = self.input.cuda(async=True)
        self.regression_target = self.regression_target.cuda(async=True)
        self.binary_class_target = self.binary_class_target.cuda(async=True)
        self.categorical_target = self.categorical_target.cuda(async=True)

    def output(self):
        with torch.no_grad():
            bo = self.binary_out[0].sigmoid().round() * 255
            _, co = torch.max(F.softmax(self.cat_out[0,:]), dim=0)
            co = co.float() * (255 / 9.)
            cat1 = torch.cat([bo, self.reg_out[0], co.unsqueeze(0)], 0)
            d1, d2, d3 = cat1.shape
            cat1 = cat1.view(d1 * d2, d3).cpu().numpy()
            return cat1

    def lr_step(self):
        self.optim_scheduler.step()
    
    def predict(self, x, depth2bev):
        """
        TODO: Get rid of depth2bev arg

        At test time, input is just the BEV representation of the scene.
        Outputs object detection predictions

        Args: 
            x: [1,35,250,348] BEV map of scene
        """
        with torch.no_grad():
            self.input.data.copy_(x)
            reg_out, binary_out, cat_out = self.pixor.forward(self.input)
            reg_out = reg_out.squeeze()
            binary_out = binary_out.squeeze()
            cat_out = cat_out.squeeze()
            # compute confidence scores
            conf_scores = binary_out.squeeze().sigmoid()
            # assume this is [N,2], where N is the # of pixels
            pixels = (conf_scores > self.conf_thresh).nonzero()
            dxs = reg_out[0]; dys = reg_out[1]; dzs = reg_out[2]
            dets = np.zeros((pixels.shape[0], 4))
            for i, p in enumerate(pixels):
                # decode pixels
                _, k = torch.max(F.softmax(cat_out[:,p]), dim=0)
                # pixels is [height x width], so [j, i]
                x, y, z = depth2bev.grid_cell_2_point(p[1], p[0], scale=4, k=k)
                dets[i,0] = x + dxs[p]
                dets[i,1] = y + dys[p]
                dets[i,2] = z + dzs[p]
                dets[i,3] = conf_scores[p]
            return self.NMS(dets)
    
    # TODO: test
    def NMS(self, detections):
        """
        Apply NMS to set of detections with corresponding predictions
        """
        # 1. Grab the detection with highest pred
        # 2. Discard other detections with overlap between circles higher than 
        # self.IOU_thresh (0.5)
        # Repeat 1-2 until all detections have been handled
        results = []
        # TODO: vectorize this
        def IOU(x1, y1, r1, x2, y2, r2):
            """ IOU between two circles """
            # distance between circle centers
            d = np.linalg.norm(np.array([y2 - y1, x2 - x1]))
            if d >= r1 + r2: # the circles do not intersect
                return 0.
            elif d == 0.:
                intersection = np.pi * r1 ** 2
            else:
                alpha = np.arccos((r1 ** 2 + d ** 2 - r2 ** 2) / (2 * r1 * d))
                beta = np.arccos((r2 ** 2 + d ** 2 - r1 ** 2) / (2 * r2 * d))
                intersection = alpha * r1 ** 2 + beta * r2 ** 2 - 0.5 * r1 ** 2 \
                        * np.sin(2 * alpha) - 0.5 * r2 ** 2 * np.sin(2 * beta)
            union = (np.pi * r1 ** 2) + (np.pi * r2 ** 2) - intersection
            return intersection / union
        # sort in decreasing order of confidence
        x = detections[:,0]
        y = detections[:,1]
        p = detections[:,3]
        idxs = np.argsort(p)
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            results.append(i)
            scores = []
            for l in range(last-1, -1, -1):
                j = idxs[j]
                score = IOU(x[i], y[i], self.ball_radius, x[j], y[j], self.ball_radius)
                scores.append(score)
            idxs = np.delete(idxs, np.concatenate(([last],
                np.where(scores > self.IOU_thresh)[0])))
        return detections[results]


    def focal_loss(self, x, y):
        '''Focal loss.
	Based on https://github.com/kuangliu/pytorch-retinanet/blob/master/loss.py
        
	Args:
          x: (tensor) sized [N,1,H,W] where HxW is the BEV grid dims.
          y: (tensor) sized [N,1,H*W].

        Return:
          (tensor) focal loss.
        '''
        alpha = 0.25
        gamma = 2
        dx1, dx2, dx3, dx4 = x.shape
        dy1, dy2, dy3 = y.shape
        # flatten both x and y int [N,H*W]
        x_ = x.view(dx1, dx3 * dx4)
        t = y.view(dy1, dy3)
        p = x_.sigmoid()
        pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
        w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
        w = w * (1-pt).pow(gamma)
        return F.binary_cross_entropy_with_logits(x_, t, w)

class PIXORNet(nn.Module):
    """
    Some conv arithmetic: when using 3x3 convs with 
    stride 1 or 2, use padding of 1 to preserve dimensions!
    """

    def __init__(self, opt):
        super(PIXORNet, self).__init__()
        # Initial processing by 2 Conv2d layers with
        # 3x3 kernel, stride 1, padding 1
        self.block1 = nn.Sequential(
            nn.Conv2d(opt.bev_dims[2], 32, 3, 1, 1),
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
        self.block7 = self._make_deconv_layer(128, 96)
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
        # binary occupation over BEV grid map
        # this can be directly interpreted as logits for Sigmoid
        self.classification_out = nn.Conv2d(96, 1, 3, 1, 1)
        # dx, dy, dz offsets
        self.regression_out = nn.Conv2d(96, 3, 3, 1, 1)
        # multi-class classification for z-dimension occupation
        # this can be directly interpreted as logits for Softmax2d
        self.categorical_out = nn.Conv2d(96, 9, 3, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # use class imbalance prior from RetinaNet
        nn.init.constant_(self.classification_out.bias, -2.69810055)
    
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
          H=250, W=348
        """
        # Backbone network
        x = self.block1(x)
        # input [32,250,348]
        # output [96,125,174]
        x = self.block2(x)
        # output [192,63,87]
        x3 = self.block3(x)
        # output [256,32,44]
        x4 = self.block4(x3)
        # output [384,16,22]
        x = self.block5(x4)
        # Upsampling fusion
        # output [196, 16,22]
        x = self.u_bend(x)
        # output is [128,32,44]
        x = self.block4_6(x4) + self.block6(x)
        # output is [96,63,87]
        x = self.block3_7(x3) + self.block7(x)

        # Head network
        x = self.header(x)
        # [1,63,87] - 0-1 class logits
        c = self.classification_out(x)
        # [3,63,87] - dx,dy,dz
        r = self.regression_out(x)
        # [9,63,87] - categorical logits
        # ceil(35 / 4) = 9
        m = self.categorical_out(x)
        return r, c, m

if __name__ == '__main__':
    
    pix = PIXOR([348, 250, 35])
    pix = pix.cuda()
    model_parameters = filter(lambda p: p.requires_grad, pix.parameters())
    print("# of trainable actor parameters: {}".format(sum([np.prod(p.size()) for p in model_parameters])))
    
    # [NCHW]
    x = torch.zeros(1, 35, 250, 348).cuda()
    r, c, m = pix(x)
    print(r.shape, c.shape, m.shape)
