from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import numpy as np
import cv2

from models.losses import FocalLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.decode import pano_decode
from models.utils import _sigmoid
from utils.debugger import Debugger
#from utils.post_process import pano_post_process
from .base_trainer import BaseTrainer


class PanoLoss(torch.nn.Module):

    def __init__(self, opt):
        super(PanoLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
                        RegLoss() if opt.reg_loss == 'sl1' else None
        self.opt = opt

    def forward(self, outputs, batch):
        opt = self.opt
        hm_loss = 0
        wh_loss = 0
        off_loss = 0

        for s in range(opt.num_stacks):
            #TODO: upgrade to support mutiple extreme points
            output = outputs[s]
            if not opt.mse_loss:
                output['hm_center'] = _sigmoid(output['hm_center'])

            hm_loss += self.crit(output['hm_center'], batch['hm_center']) / opt.num_stacks

            if opt.wh_weight > 0:
                wh_loss += self.crit_reg(
                    output['bbox_wh'], batch['reg_mask'],
                    batch['ind_center'], batch['bbox_wh']) / opt.num_stacks

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(
                    output['reg_center'], batch['reg_mask'],
                    batch['ind_center'], batch['reg_center']) / opt.num_stacks

        loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss
        loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'wh_loss': wh_loss, 'off_loss': off_loss}
        return loss, loss_stats


class PanoTrainer(BaseTrainer):

    def __init__(self, opt, model, optimizer=None):
        super(PanoTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss']
        loss = PanoLoss(opt)
        return loss_states, loss

    def pano_center_crop_and_resize(self, img):
        h, w = img.shape
        new_w = int(h * 1.5)
        margin = (w - new_w) // 2
        img = img[:, margin: margin + new_w]
        img = cv2.resize(img, dsize=(self.opt.input_w, self.opt.input_h))
        return img

    def draw_result(self, batch, det):
        img_id = batch['img_id'][0]
        split = 'val'
        img_path = os.path.join(self.opt.data_dir, split, img_id + '.jpg')
        if not os.path.exists(img_path):
            return

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = self.pano_center_crop_and_resize(img)
        H, W = img.shape
        det[:, :4] *= 4

        for d in det:
            if d[5] == 0:
                continue

            x1 = int(d[0])
            y1 = int(d[1])
            x2 = int(d[2])
            y2 = int(d[3])
            cv2.line(img, (x1, y1), (x1, y2), 255)
            cv2.line(img, (x1, y2), (x2, y2), 255)
            cv2.line(img, (x2, y2), (x2, y1), 255)
            cv2.line(img, (x2, y1), (x1, y1), 255)

        cv2.imwrite(os.path.join(self.opt.save_dir, 'vis_' + split, img_id + '.jpg'), img)

    def save_result(self, output, batch, results):
        reg_center = output['reg_center'] if self.opt.reg_offset else None
        dets = pano_decode(output['hm_center'], output['bbox_wh'], reg_center, K=35) # (batch_size, K, 6)
        self.draw_result(batch, dets[0].detach().cpu().numpy())
