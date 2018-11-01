import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models.resnet import Bottleneck
import torch.optim as optim
from .model import Model

class VNet(Model):
    """ Visibility-Net """
    def __init__(self, opt):
        self.__name__ = 'vnet'
        opt.pixor_head = 'full'
        self.bev_pixor = PIXORNet(opt, 'BEV', input_channels=opt.view_dims['BEV'][2])
        opt.pixor_head = 'viz'
        self.vnet = PIXORNet(opt, 'BEV', input_channels=1, header_inplanes=384)
        self.downsample = 0.25
        self.optimizer = optim.SGD(self.vnet.parameters(), lr=opt.lr, momentum=opt.momentum)
        self.optim_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                milestones=opt.milestones, gamma=opt.lr_decay)
        self.x_dim = opt.view_dims['BEV'][0] # width
        self.y_dim = opt.view_dims['BEV'][1] # height
        self.z_dim = opt.view_dims['BEV'][2] # channels
        self.bsz = opt.bsz
        self.bev_input = torch.FloatTensor(self.bsz, self.z_dim, self.y_dim, self.x_dim) 
        self.vnet_input = torch.FloatTensor(self.bsz, 1, self.y_dim, self.x_dim)
        self.bev_input.pin_memory()
        self.vnet_input.pin_memory()
        self.use_gpu = opt.gpu
        self.entropy_coeff = opt.entropy_coeff

        opt.pixor_head = 'full'
        x = torch.load(opt.pretrained_bev)
        opt.pixor_head = 'viz'
        new_state_dict = {}
        for k,v in x.items():
            new_state_dict[k.replace('module.', '')] = v            
        self.bev_pixor.load_state_dict(new_state_dict)

        # disable update for the BEV embedding branch
        for m in self.bev_pixor.modules():
            m.requires_grad = False
        self.bev_input.requires_grad = False
        self.vnet_input.requires_grad = False
        
        if opt.n_gpus > 1:
            self.bev_pixor = nn.DataParallel(self.bev_pixor, device_ids=list(range(opt.n_gpus)))
            self.vnet = nn.DataParallel(self.vnet, device_ids=list(range(opt.n_gpus)))
    
    def train(self):
        self.bev_pixor.train()
        self.vnet.train()

    def eval(self):
        self.bev_pixor.eval()
        self.vnet.eval()

    def gpu(self):
        self.bev_pixor.cuda()
        self.vnet.cuda()
        self.bev_input = self.bev_input.cuda(async=True)
        self.vnet_input = self.vnet_input.cuda(async=True)
        
    def load(self, x, set_):
        self.load_state_dict(torch.load(x))
    
    def lr_step(self):
        self.optim_scheduler.step()
    
    def parameters(self):
        params = []
        for m in self.bev_pixor.parameters():
            params.append(m)
        for m in self.vnet.parameters():
            params.append(m)
        return params 
    
    # TODO: delete
    def _make_deconv_layer(self, in_planes, out_planes, output_padding=0):
        upsample = nn.Sequential(
                nn.ConvTranspose2d(in_planes, out_planes, 3, 2, 1, 
                    output_padding=output_padding),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_planes))
        return upsample

    def l2(self, viz_posterior):
        with torch.no_grad():
            b, c, h, w = viz_posterior.shape
            viz_prior = torch.clamp(F.interpolate(self.vnet_input, size=(h,w)), 1e-6, 1).log()
        return F.mse_loss(viz_posterior, viz_prior)

    # TODO: Delete
    def hellinger_loss(self, viz_posterior):
        """
        Implements Hellinger(viz_posterior, viz_prior)
        Args:
            viz_posterior: [N,63,87]
        """
        # resize viz_prior to 63, 87
        b, c, h, w = viz_posterior.shape
        viz_prior = torch.clamp(F.interpolate(self.vnet_input, size=(h,w)), 0, 1)
        return torch.sqrt(torch.sum((torch.sqrt(viz_posterior) - \
                torch.sqrt(viz_prior)) ** 2)) / np.sqrt(2)

    # TODO: Delete
    def kl_loss(self, viz_posterior):
        """
        Implements KL(viz_posterior||viz_prior)
        Args:
            viz_posterior: [N,63,87]
        """
        # resize viz_prior to 63, 87
        b, c, h, w = viz_posterior.shape
        viz_prior = torch.clamp(F.interpolate(self.vnet_input, size=(h,w)), 0, 1)
        return F.kl_div(viz_posterior, viz_prior)

    def posterior_error(self, viz_posterior, detections, occluded):
        """
        Loss measuring disagreement between the detector scores and the viz posterior,
        as well as btwn the occluded object (score = 0) and the viz posterior.
        Args:
            viz_posterior: [n,1, 63, 87]
            detections: [n, 3, 4] (c,h,w indices ordering)
            occluded: [n, 3, 3] (c,h,w indices ordering)
        """
        b, c, h, w = viz_posterior.shape
        zed = torch.LongTensor([0])
        h_lim = torch.LongTensor([h-1])
        w_lim = torch.LongTensor([w-1])
        if self.use_gpu:
            detections = detections.cuda()
            occluded = occluded.cuda()
            zed = zed.cuda()
            h_lim = h_lim.cuda()
            w_lim = w_lim.cuda()
        def get_local_patches(dets, posterior):
            """
            dets is [n,3] or [n,4]
            posterior is [1,63,87]
            """
            dets = dets.long()
            # assert coord.shape[0] > 0
            c, h, w = posterior.shape
            # compute the 9 coords
            post_patches = []
            for i in range(-1, 2):
                for j in range(-1, 2):
                    j_ = torch.min(torch.max(j + dets[:,1], zed), h_lim) # j
                    i_ = torch.min(torch.max(i + dets[:,2], zed), w_lim) # i
                    post_patches.append(posterior[:,j_, i_])
            post_patches = torch.stack(post_patches).view(-1, 9).contiguous()
            return post_patches
        
        
        det_preds = []
        det_targets = []
        occ_preds = []
        for i in range(b):
            # count # of detections
            n_detections = (detections[i,:,0] != -1.).sum().item()
            if n_detections > 0:
                d_post_patches = get_local_patches(detections[i,:n_detections], viz_posterior[i])
                if d_post_patches is not None:
                    s = detections[i,:n_detections,3].unsqueeze(1).repeat(1,9).float().log()
                    if self.use_gpu:
                        s = s.cuda()
                    det_preds.append(d_post_patches.view(-1))
                    det_targets.append(s.view(-1))
                #pred = viz_posterior[i,:,detections[i,:n_detections,1].long(),
                #        detections[i,:n_detections,2].long()]
                #target = detections[i,:n_detections,3].float().expand_as(pred)
                #loss= loss + F.mse_loss(pred, target)
            n_occluded = (occluded[i,:,0] != -1.).sum()
            if n_occluded > 0:
                o_post_patches = get_local_patches(occluded[i,:n_occluded], viz_posterior[i])
                if o_post_patches is not None:
                    #loss = loss + o_post_patches.mean()
                    occ_preds.append(o_post_patches.view(-1))
                #pred = viz_posterior[i,:,occluded[i,:n_occluded,1].long(),
                #        occluded[i,:n_occluded,2].long()]
                #loss = loss + pred.mean()
        loss =  F.mse_loss(torch.cat(det_preds), torch.cat(det_targets))
        if len(occ_preds) > 0:
            loss = loss + torch.cat(occ_preds).mean()
        return loss

    def forward(self, bev_x, viz_x):
        bev_x = self.bev_pixor(bev_x, get_embedding=True)
        return self.vnet(viz_x, bev_x)

    def step(self, batch, set_):
        """
        Args:
            batch['input'] is batch of input maps of size [N,C,H,W]
                * batch['input']['BEV'] is BEV grid [N,35,250,348]
                * batch['input']['visibility'] is v grid [N,1,250,348]
            batch['labels']['BEV_dets'] is [N,3,4] (FloatTensor)
            batch['labels']['occluded'] is [N,3,3] (LongTensor)
        """
        if set_ == 'train':
            env = torch.enable_grad
        else:
            env = torch.no_grad
        with env():
            self.bev_input.data.copy_(batch[0]['BEV'])
            self.vnet_input.data.copy_(batch[0]['VG'])
            BEV_dets = batch[1]['detections']
            occluded = batch[1]['occluded']
            self.viz_posterior = self.forward(self.bev_input, self.vnet_input)
            #ent = self.hellinger_loss(self.viz_posterior.sigmoid())
            #ent = self.kl_loss(F.logsigmoid(self.viz_posterior))
            #ent = self.l2(F.logsigmoid(self.viz_posterior))
            posterior_err = self.posterior_error(F.logsigmoid(self.viz_posterior),
                    BEV_dets, occluded)
            if set_ == 'train':
                self.vnet.zero_grad()
                #(self.entropy_coeff * ent + posterior_err).backward()
                posterior_err.backward()
                self.optimizer.step()
            #result = {'{}_entropy_loss'.format(set_): ent.item()}
            #result['{}_posterior_err'.format(set_)] = posterior_err.item()
            result = {'{}_posterior_err'.format(set_): posterior_err.item()}
            return result
        
    def output(self):
        with torch.no_grad():
            _, h, w = self.viz_posterior[0].shape
            prior = self.vnet_input[0].clone()
            prior = F.interpolate(prior.unsqueeze(0), size=(h,w)).squeeze(0)
            posterior = self.viz_posterior[0].sigmoid().clone()
            prior = prior * 255
            posterior = posterior * 255
            cat1 = torch.cat([prior, posterior], 0)
            d1, d2, d3 = cat1.shape
            cat1 = cat1.view(d1 * d2, d3).cpu().numpy()
            return cat1

class MVPIXOR(Model):
    """ Multi-view PIXOR """
    def __init__(self, opt, test=False):
        self.__name__ = 'mvpixor'
        if opt.pixor_head == "full":
            self.bev_pixor = PIXOR(opt, 'BEV')
            self.fv_pixor = PIXOR(opt, 'FV')
        else:
            self.bev_pixor = PIXORBinary(opt, 'BEV')
            self.fv_pixor = PIXORBinary(opt, 'FV')

    def load(self, x, set_):
        for e in x:
            if 'bev' in e[0]:
                print('loading bev_pixor at %s' %(e[1]))
                self.bev_pixor.load_state_dict(torch.load(e[1]), set_)
            elif 'fv' in e[0]:
                print('loading fv_pixor at %s' %(e[1]))
                self.fv_pixor.load_state_dict(torch.load(e[1]), set_)
    
    def save(self, path, epoch):
        f = open(os.path.join(path, 'bev_pixor.txt'), 'w')
        f.write(str(self.bev_pixor))
        f.close()
        torch.save(
            self.bev_pixor.state_dict(),
            os.path.join(path, 'bev_pixor_%s.pth' %epoch)
        )
        f = open(os.path.join(path, 'fv_pixor.txt'), 'w')
        f.write(str(self.fv_pixor))
        f.close()
        torch.save(
            self.fv_pixor.state_dict(),
            os.path.join(path, 'fv_pixor_%s.pth' %epoch)
        )

    def parameters(self):
        params = []
        for m in self.bev_pixor.parameters():
            params.append(m)
        for m in self.fv_pixor.parameters():
            params.append(m)
        return params 

    def train(self):
        self.bev_pixor.train()
        self.fv_pixor.train()

    def eval(self):
        self.bev_pixor.eval()
        self.fv_pixor.eval()

    def lr_step(self):
        self.bev_pixor.lr_step()
        self.fv_pixor.lr_step()

    def step(self, batch, set_):
        r1 = self.bev_pixor.step(batch, set_, 'BEV')
        r2 = self.fv_pixor.step(batch, set_, 'FV')
        return {**r1, **r2}

    def gpu(self):
        self.bev_pixor.gpu()
        self.fv_pixor.gpu()

    def output(self):
        o1 = self.bev_pixor.output()
        o2 = self.fv_pixor.output()
        return np.concatenate([o1, o2], axis=0)

    # TODO: remove depth2bev arg
    def predict(self, x, depth2bev):
        with torch.no_grad():
            bev_detections, bev_conf_scores = self.bev_pixor.predict(x, 'BEV', depth2bev)
            fv = x[0]['FV'].squeeze()
            self.fv_pixor.input.data.copy_(fv)
            reg_out, binary_out = self.fv_pixor.pixor(self.fv_pixor.input)
            reg_out = reg_out.squeeze().cpu().numpy()
            binary_out = binary_out.squeeze()
            fv_conf_scores = binary_out.sigmoid().cpu().numpy()
            # for each bev detection, try to grab z
            dets = np.zeros((bev_detections.shape[0], 3))
            dets_px = []
            bev_det_cells = []
            fv_det_cells = []
            for k, bd in enumerate(bev_detections):
                # get the x coord of the detection
                x_cord = int(np.floor(bd[0] / (depth2bev.grid_y_res * 4)))
                # get the y coord 
                y_cord = int(np.floor(bd[1] / (depth2bev.grid_x_res * 4)))
                # clamp
                # TODO: 62 is x_dim/4 - 1 (63 - 1)
                x_cord = max(min(62, x_cord), 0)
                # TODO: 87 is y_dim/4 - 1
                y_cord = max(min(86, y_cord), 0)
                #print("BEV: ", x_cord, y_cord)
                bev_det_cells.append((x_cord, y_cord))
                # the score map is in [hxw] format - [63,87]
                bev_conf = bev_conf_scores[x_cord, y_cord]
                best_z_conf = -1
                best_z = -1
                for i in range(max(0, y_cord), min(y_cord+1, fv_conf_scores.shape[1])):
                    # compute the max across the z values in 3 cell neighborhood (this is in 4x downsample)
                    z_idx = np.argmax(fv_conf_scores[:,i])
                    tmp = fv_conf_scores[z_idx, i]
                    if tmp > best_z_conf:
                        best_z_conf = tmp
                        best_z = int(z_idx)
                # k coord
                #print("FV: ", y_cord, inv_best_z)
                fv_det_cells.append((best_z, y_cord))
                # convert best_z to point
                dz = reg_out[1, best_z, y_cord]
                z = best_z * depth2bev.grid_y_res * 4
                dz = dz * depth2bev.regression_stats['FV'][1][1] + depth2bev.regression_stats['FV'][1][0]
                #final_pred_z = z + dz
                final_pred_z = z
                dets[k,0] = bd[0]
                dets[k,1] = bd[1]
                # need to invert z 
                #dets[k,2] = depth2bev.pc_z_dim - final_pred_z
                #dets[k,2] = final_pred_z
                # Compute the i,j,k coords in [348,250] space
                x_dim = depth2bev.pc_x_dim / depth2bev.grid_x_res
                y_dim = depth2bev.pc_y_dim / depth2bev.grid_y_res
                x_px = int(np.floor(bd[0] / depth2bev.grid_x_res))
                y_px = int(np.floor(bd[1] / depth2bev.grid_y_res))
                x_px = max(min(x_dim, x_px), 0)
                y_px = max(min(y_dim, y_px), 0)
                # convert the z coord from FV to a BEV height channel index
                z_px = int(np.floor(dets[k,2] / (depth2bev.pc_z_dim / depth2bev.grid_height_channels)))
                dets_px.append((x_px, y_px, z_px, bev_conf))
        """ 
        labeled_bev_conf = 255 * np.ones((bev_conf_scores.shape[0], bev_conf_scores.shape[1], 3))
        for i in range(bev_conf_scores.shape[0]):
            for j in range(bev_conf_scores.shape[1]):
                labeled_bev_conf[i, j, :] = np.array([0, bev_conf_scores[i,j] * 255, 0])
        for bd in bev_det_cells:
            labeled_bev_conf[bd[0], bd[1], :] = np.array([255, 0, 0])
        labeled_fv_conf = 255 * np.ones((fv_conf_scores.shape[0], fv_conf_scores.shape[1], 3))
        for i in range(fv_conf_scores.shape[0]):
            for j in range(fv_conf_scores.shape[1]):
                labeled_fv_conf[i, j, :] = np.array([0, fv_conf_scores[i,j] * 255, 0])
        for bd in fv_det_cells:
            labeled_fv_conf[int(bd[0]), int(bd[1]), :] = np.array([255, 0, 0])
        #for bd in bev_det_cells:
        #    labeled_fv_conf[:, bd[1], :] = np.array([0, 0, 255])
        return dets, bev_conf_scores, fv_conf_scores, labeled_bev_conf, labeled_fv_conf
        """
        return dets[:3], dets_px, bev_conf_scores.cpu().numpy(), fv_conf_scores

# TODO runnable standalone or as Multi-view
class PIXOR(Model):
    """
    Separate model (the IntPhys model API) and PIXORNet to use nn.DataParallel :( 
    """
    def __init__(self, opt, view=None, test=False):
        super(Model).__init__()
        self.__name__ = 'pixor'
        view = opt.view if view is None else view
        self.x_dim = opt.view_dims[view][0] # width
        self.y_dim = opt.view_dims[view][1] # height
        self.z_dim = opt.view_dims[view][2] # channels
        self.target_x_dim = int(np.ceil(self.x_dim / 4)) 
        self.target_y_dim = int(np.ceil(self.y_dim / 4))
        self.conf_thresh = opt.conf_thresh
        self.ball_radius = opt.ball_radius
        self.IOU_thresh = opt.IOU_thresh
        self.bsz = 1 if test else opt.bsz
        self.pixor = PIXORNet(opt, view, input_channels=self.z_dim)
        # stuff for forward pass
        # N C H W format for Pytorch
        self.input = Variable(torch.FloatTensor(self.bsz, self.z_dim, self.y_dim, self.x_dim))
        self.regression_target = Variable(torch.FloatTensor(self.bsz, 2, 
            self.target_x_dim * self.target_y_dim))
        self.binary_class_target = Variable(torch.FloatTensor(self.bsz, 1,
            self.target_x_dim * self.target_y_dim))
        #self.categorical_target = Variable(torch.FloatTensor(
        #    self.bsz * self.target_x_dim * self.target_y_dim, 1))
        self.input.pin_memory()
        self.regression_target.pin_memory()
        self.binary_class_target.pin_memory()
        #self.categorical_target.pin_memory()
        # optimization
        self.smoothL1 = nn.SmoothL1Loss()
        #self.nll = nn.NLLLoss()
        self.optimizer = optim.SGD(self.pixor.parameters(), lr=opt.lr, momentum=0.9)
        #self.optim_scheduler = CyclicLR(self.optimizer, base_lr=0.0001, max_lr=0.001, step_size=int(opt.nbatch_train / 2.))
        self.optim_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[3, 6], gamma=0.1)
        if opt.n_gpus > 1:
            self.pixor = nn.DataParallel(self.pixor, device_ids=list(range(opt.n_gpus)))
        
    def load_state_dict(self, x, set_):
        if 'test' not in set_:
            model_dict = self.pixor.state_dict()
            pretrained = {k: v for k, v in x.items() if k in model_dict}
            model_dict.update(pretrained)
            self.pixor.load_state_dict(model_dict)
        else:
            # TODO: fix this
            fixed_state_dict = {}
            for k,v in x.items():
                fixed_state_dict[k.replace('module.', '')] = v            
            self.pixor.load_state_dict(fixed_state_dict)

    def state_dict(self):
        return self.pixor.state_dict()

    def parameters(self):
        return self.pixor.parameters()
    
    def train(self):
        self.pixor.train()

    def eval(self):
        self.pixor.eval()
    
    def lr_step(self):
        #self.optim_scheduler.batch_step()
        self.optim_scheduler.step()

    def step(self, batch, set_, view):
        """
        Args:
            batch[0] is batch of input maps of size [N,C,H,W]
            batch[1]['regression_target'] is regression targets of size [N,3,H,W]
            batch[1]['binary_target'] is binary class target of size [N,1,H,W]
            -- DEPRECATED: batch[1]['z_target'] is categorical target of size [N,1,H,W]
        """
        if set_ == 'train':
            env = torch.enable_grad
        else:
            env = torch.no_grad
        with env():
            self.input.data.copy_(batch[0][view].squeeze())
            # flatten last two dimensions
            self.regression_target.data.copy_(batch[1][view]['regression_target'].view(
                self.bsz, 2, self.target_x_dim * self.target_y_dim))
            self.binary_class_target.data.copy_(batch[1][view]['binary_target'].view(
                self.bsz, 1, self.target_x_dim * self.target_y_dim))
            #self.categorical_target.data.copy_(batch[1]['z_target'].view(
            #    self.bsz * self.target_x_dim * self.target_y_dim, 1))
            #self.reg_out, self.binary_out, self.cat_out = self.pixor(self.input)
            self.reg_out, self.binary_out = self.pixor(self.input)
            binary_classification_loss = self.focal_loss(self.binary_out, self.binary_class_target)
            # If there are no objects in the scene, no regression or cat loss
            if not self.binary_class_target.byte().any():
                total_loss = binary_classification_loss
                result = {'{}_{}_total_loss'.format(view, set_): total_loss.item(),
                        '{}_{}_binary_classification_loss'.format(view, set_): binary_classification_loss.item()}
            else:
                # regression loss only computed over positive samples
                reg_out = self.reg_out.view(self.bsz, 2, self.target_x_dim * self.target_y_dim)
                # zero out predictions not at positive samples
                reg_out = reg_out * self.binary_class_target
                regression_loss = self.smoothL1(reg_out, self.regression_target)
                # compute categorical loss
                # zero out predictions not a positive samples
                # channel-wise softmax, in shape [N,C] (9 classes)
                # TODO: don't hardcode 9
                #cat_out = self.cat_out.view(-1, 9)
                #binary_target_flat = self.binary_class_target.view(-1)
                # select only positive samples
                #cat_out = cat_out[binary_target_flat.byte()]
                #cat_target_out = self.categorical_target[binary_target_flat.byte()]
                #height_logits = F.log_softmax(cat_out, dim=1)
                #categorical_loss = self.nll(height_logits, cat_target_out.long().squeeze())
                #total_loss = binary_classification_loss + regression_loss + categorical_loss
                total_loss = binary_classification_loss + regression_loss
                result = {'{}_{}_regression_loss'.format(view, set_): regression_loss.item(),
                        '{}_{}_binary_classification_loss'.format(view, set_): binary_classification_loss.item(),
                #        '{}_categorical_loss'.format(set_): categorical_loss.item(),
                        '{}_{}_total_loss'.format(view, set_): total_loss.item()}
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
        #self.categorical_target = self.categorical_target.cuda(async=True)

    def output(self):
        with torch.no_grad():
            po = self.binary_out[0].sigmoid()
            bo = po.round() * 255
            po *= 255
            #_, co = torch.max(F.softmax(self.cat_out[0,:]), dim=0)
            #co = co.float() * (255 / 9.)
            #cat1 = torch.cat([po, bo, self.reg_out[0], co.unsqueeze(0)], 0)
            cat1 = torch.cat([po, bo, self.reg_out[0]], 0)
            d1, d2, d3 = cat1.shape
            cat1 = cat1.view(d1 * d2, d3).cpu().numpy()
            return cat1

    
    def predict(self, x, view, depth2bev):
        """
        TODO: Get rid of depth2bev arg

        At test time, input is just the BEV representation of the scene.
        Outputs object detection predictions

        Args: 
            x[0]['BEV']: [1,35,250,348] BEV map of scene
        """
        with torch.no_grad():
            frame = x[0][view].squeeze()
            #regression_target = x[1][view]['regression_target'].squeeze()
            self.input.data.copy_(frame)
            #reg_out, binary_out, cat_out = self.pixor(self.input)
            reg_out, binary_out = self.pixor(self.input)
            reg_out = reg_out.squeeze()
            binary_out = binary_out.squeeze()
            #cat_out = cat_out.squeeze()
            # compute confidence scores
            conf_scores = binary_out.sigmoid()
            # assume this is [N,2], where N is the # of pixels
            pixels = (conf_scores > self.conf_thresh).nonzero()
            dets = np.zeros((pixels.shape[0], 3))
            for i, p in enumerate(pixels):
                #dxs = reg_out[0][p[0], p[1]]
                #dys = reg_out[1][p[0], p[1]]
                #dzs = reg_out[2][p[0], p[1]]
                # gt
                #gt_dxs = regression_target[0][p[0], p[1]]
                #gt_dys = regression_target[1][p[0], p[1]]
                #gt_dzs = regression_target[2][p[0], p[1]]
                #if gt_dxs != 0.0 and gt_dys != 0.0 and gt_dzs != 0.0:
                #if gt_dxs != 0.0 and gt_dys != 0.0:
                    #print("dx = {}, gt dx = {}".format(dxs, gt_dxs))
                    #print("dy = {}, gt dy = {}".format(dys, gt_dys))
                    #print("dz = {}, gt dz = {}".format(dzs, gt_dzs))
                
                # do \sigma * dx + \mu to add back regression stats
                #ds = [dxs, dys, dzs]
                #ds =  [dxs, dys]
                #for j in range(len(ds)):
                #    ds[j] = depth2bev.regression_stats[view][j][1] * ds[j] + depth2bev.regression_stats[view][j][0]
                # decode pixels
                #_, k = torch.max(F.softmax(cat_out[:,p[0], p[1]]), dim=0)
                # pixels is [height x width], so [j, i]
                x, y = depth2bev.grid_cell_2_point(p[1], p[0], scale=4, view=view)
                #dets[i,0] = (x.double().cpu() + ds[0]).numpy()
                #dets[i,1] = (y.double().cpu() + ds[1]).numpy()
                dets[i,0] = x.double().cpu().numpy()
                dets[i,1] = y.double().cpu().numpy()
                dets[i,2] = conf_scores[p[0], p[1]].cpu().numpy()
            return self.NMS(dets), conf_scores
    
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
        def radius_overlap(x1, y1, x2, y2, r):
            """ distance between centers ratio """
            return max(0., (2*r - np.linalg.norm(np.array([y2 - y1, x2 - x1]))) / (2*r))

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
        p = detections[:,2]
        idxs = np.argsort(p)
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            results.append(i)
            delete = []
            for l in range(last-1, -1, -1):
                j = idxs[l]
                #score = IOU(x[i], y[i], self.ball_radius, x[j], y[j], self.ball_radius)
                score = radius_overlap(x[i], y[i], x[j], y[j], self.ball_radius)
                if score > self.IOU_thresh:
                    delete.append(l)
            #idxs = np.delete(idxs, np.concatenate(([last],
            #    np.where(np.array(scores) > self.IOU_thresh)[0])))
            idxs = np.delete(idxs, np.concatenate(([last], np.array(delete))))
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

class PIXORBinary(Model):
    def __init__(self, opt, view=None, test=False):
        super(Model).__init__()
        self.__name__ = 'pixorbinary'
        self.view = opt.view if view is None else view
        self.x_dim = opt.view_dims[self.view][0] # width
        self.y_dim = opt.view_dims[self.view][1] # height
        self.z_dim = opt.view_dims[self.view][2] # channels
        self.bsz = 1 if test else opt.bsz
        self.pixor = PIXORNet(opt, self.view, input_channels=self.z_dim)
        # stuff for forward pass
        # N C H W format for Pytorch
        self.input = Variable(torch.FloatTensor(self.bsz, self.z_dim, opt.crop_sz, opt.crop_sz))
        self.binary_class_target = Variable(torch.FloatTensor(self.bsz, 1))
        self.input.pin_memory()
        self.binary_class_target.pin_memory()
        # opt
        self.loss = nn.BCEWithLogitsLoss()
        self.optimizer = optim.SGD(self.pixor.parameters(), lr=opt.lr, momentum=0.9)
        #self.optim_scheduler = CyclicLR(self.optimizer, base_lr=0.0001, max_lr=0.001, step_size=int(opt.nbatch_train / 2.))
        self.optim_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[3, 6], gamma=0.1)
        if opt.n_gpus > 1:
            self.pixor = nn.DataParallel(self.pixor, device_ids=list(range(opt.n_gpus)))
        
    def load_state_dict(self, x):
        self.pixor.load_state_dict(x)

    def state_dict(self):
        return self.pixor.state_dict()

    def parameters(self):
        return self.pixor.parameters()
    
    def train(self):
        self.pixor.train()

    def eval(self):
        self.pixor.eval()
    
    def lr_step(self):
        #self.optim_scheduler.batch_step()
        self.optim_scheduler.step()

    def step(self, batch, set_):
        """
        Args:
            batch[0][view] is batch of input maps of size [N,C,H,W]
            batch[1][view]['binary_target'] is binary class target of size [N,1]
        """
        if set_ == 'train':
            env = torch.enable_grad
        else:
            env = torch.no_grad
        with env():
            self.input.data.copy_(batch[0][self.view].squeeze())
            self.binary_class_target.data.copy_(batch[1][self.view]['binary_target'].view(self.bsz, 1))
            self.logits = self.pixor(self.input)
            binary_loss = self.loss(self.logits, self.binary_class_target)
            if set_ == 'train':
                self.pixor.zero_grad()
                binary_loss.backward()
                self.optimizer.step()
                result = {'{}_{}_BCEWithLogitsLoss'.format(self.view, set_): binary_loss.item()}
            else:
                predictions = (self.logits.sigmoid() > 0.5).float()
                acc = (1 - torch.abs(predictions - self.binary_class_target).mean()) * 100.
                result = {'{}_{}_BCEWithLogitsLoss'.format(self.view, set_): binary_loss.item(),
                        '{}_{}_pred_acc'.format(self.view, set_): acc.item()}
            return result

    def gpu(self):
        self.pixor.cuda()
        self.input = self.input.cuda(async=True)
        self.binary_class_target = self.binary_class_target.cuda(async=True)

    def output(self):
       return None

class PIXORNet(nn.Module):
    """
    Some conv arithmetic: when using 3x3 convs with 
    stride 1 or 2, use padding of 1 to preserve dimensions!
    """

    def __init__(self, opt, view, input_channels=35, header_inplanes=192):
        super(PIXORNet, self).__init__()
        # Initial processing by 2 Conv2d layers with
        # 3x3 kernel, stride 1, padding 1
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32))   
        self.resnet_inplanes = 32
        # Blocks 2-5
        self.block2 = self._make_layer(Bottleneck, 24, 3, stride=2)
        self.block3 = self._make_layer(Bottleneck, 48, 6, stride=2)
        #self.block4 = self._make_layer(Bottleneck, 64, 6, stride=2)
        #self.block5 = self._make_layer(Bottleneck, 96, 4, stride=2)
        # Blocks 6-7
        #self.u_bend = nn.Conv2d(384, 196, 1, 1)
        #self.block6 = self._make_deconv_layer(196, 128, output_padding=1)
        #self.block7 = self._make_deconv_layer(128, 96)
        #self.block4_6 = nn.Conv2d(256, 128, 1, 1)
        #self.block3_7 = nn.Conv2d(192, 96, 1, 1)
        self.pixor_head = opt.pixor_head
        # Head network
        self.header = nn.Sequential(
            #nn.Conv2d(96, 96, 3, 1, 1),
            nn.Conv2d(header_inplanes, 96, 3, 1, 1),
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
        if opt.pixor_head == "full":
            # binary occupation over BEV grid map
            # this can be directly interpreted as logits for Sigmoid
            self.binary_class_out = nn.Conv2d(96, 1, 3, 1, 1)
            # dx, dy, dz offsets
            self.regression_out = nn.Conv2d(96, 2, 3, 1, 1)
        elif opt.pixor_head == "binary":
            # single scalar output for binary classification
            self.out1 = nn.Sequential(
                nn.Conv2d(96,1,3,1,1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(1))
            self.classification_out = nn.Linear(int((opt.crop_sz/4) ** 2), 1)
        elif opt.pixor_head == "viz":
            self.out = nn.Conv2d(96,1,3,1,1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if opt.pixor_head == "full":
            # use class imbalance prior from RetinaNet
            nn.init.constant_(self.binary_class_out.bias, -2.1518512)
    

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
    
    def forward(self, x, y=None, get_embedding=False):
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
        x = self.block3(x)
        if get_embedding:
            return x
        if y is not None:
            # element-wise mean
            #x = torch.sum(x,y).div(2)
            # concat - output channel # is 384
            x = torch.cat([x,y], dim=1)
        # output [256,32,44]
        #x4 = self.block4(x3)
        # output [384,16,22]
        #x = self.block5(x4)
        # Upsampling fusion
        # output [196, 16,22]
        #x = self.u_bend(x)
        # output is [128,32,44]
        #x = self.block4_6(x4) + self.block6(x)
        # output is [96,63,87]
        #x = self.block3_7(x3) + self.block7(x)

        # Head network
        x = self.header(x)
        # [1,63,87] - 0-1 class logits
        if self.pixor_head == "binary":
            x = self.out1(x)
            c = self.classification_out(x.view(x.shape[0], -1))
            return c
        if self.pixor_head == "full":
            c = self.binary_class_out(x)
            # [3,63,87] - dx,dy,dz
            r = self.regression_out(x)
            # [9,63,87] - categorical logits
            # ceil(35 / 4) = 9
            #m = self.categorical_out(x)
            return r, c
        if self.pixor_head == "viz":
            return self.out(x)

if __name__ == '__main__':
    
    pix = PIXOR([348, 250, 35])
    pix = pix.cuda()
    model_parameters = filter(lambda p: p.requires_grad, pix.parameters())
    print("# of trainable actor parameters: {}".format(sum([np.prod(p.size()) for p in model_parameters])))
    
    # [NCHW]
    x = torch.zeros(1, 35, 250, 348).cuda()
    r, c, m = pix(x)
    print(r.shape, c.shape, m.shape)
