from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import numpy as np
import cv2
import math
import json

from models.losses import FocalLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.decode import pano_decode, pano_decode_crown, pano_decode_aligned, pano_decode_det_and_cls
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
            output = outputs[s]
            if not opt.mse_loss:
                output['hm_center'] = _sigmoid(output['hm_center'])

            hm_loss += self.crit(output['hm_center'], batch['hm_center']) / opt.num_stacks

            if opt.wh_weight > 0:
                wh_loss += self.crit_reg(
                    output['tooth_wh'], batch['reg_mask'],
                    batch['ind_center'], batch['tooth_wh']) / opt.num_stacks

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(
                    output['reg_center'], batch['reg_mask'],
                    batch['ind_center'], batch['reg_center']) / opt.num_stacks
                off_loss += self.crit_reg(
                    output['reg_crown'], batch['reg_mask'],
                    batch['ind_center'], batch['reg_crown']) / opt.num_stacks

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
        h, w, _ = img.shape
        new_w = int(h * 1.5)
        margin = (w - new_w) // 2
        img = img[:, margin: margin + new_w, :]
        img = cv2.resize(img, dsize=(self.opt.input_w, self.opt.input_h))
        return img, 512 * margin / h

    def draw_result(self, batch, det, save_json=False, save_img=False):
        img_id = batch['img_id'][0]
        if self.opt.val:
            split = 'val'
            epoch = self.opt.val
        elif self.opt.test:
            split = 'test'
            epoch = self.opt.test

        vis_dir = os.path.join(self.opt.save_dir, 'vis_' + split + '_' + epoch)
        img_path = os.path.join(self.opt.data_dir, split, img_id)
        if os.path.exists(img_path + '.jpg'):
            img_path = img_path + '.jpg'
        elif os.path.exists(img_path + '.bmp'):
            img_path = img_path + '.bmp'
        elif os.path.exists(img_path + '.BMP'):
            img_path = img_path + '.BMP'
        else:
            return

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img, margin = self.pano_center_crop_and_resize(img)
        H, W, _ = img.shape
        det[:, :2] *= 4
        det[:, 2] *= W
        det[:, 3] *= H
        det[:, 4] *= W
        det[:, 5] *= H

        img_list = []
        if save_img:
            for i in range(5):
                img_copy = np.copy(img)
                img_list.append(img_copy)

        colors = [
            (0, 0, 255), # red
            (0, 255, 255), # yellow
            (0, 255, 0), # green
            (255, 255, 0), #
            (255, 0, 0),  # blue
            (255, 0, 255),
            (255, 255, 255),  # white
            (0, 0, 0) # black
        ]

        output_list = []
        for d in det:
            det_score = float(d[6])
            if det_score < 0.55:
                continue

            '''
            cls_scores = d[7:15]
            loc_scores = d[15:]
            pred_cls = np.argmax(cls_scores)
            pred_loc = np.argmax(loc_scores) + 1
            pred_tooth_num = int(pred_loc * 10 + pred_cls + 1)
            color = colors[pred_cls]
            '''
            cls_scores = d[7:]
            pred_cls = np.argmax(cls_scores)
            pred_tooth_num = (pred_cls // 8 + 1) * 10 + pred_cls % 8 + 1

            tooth_w = float(d[4])
            tooth_h = float(d[5])
            x_center = float(d[0])
            y_center = float(d[1])
            x_crown = x_center + float(d[2])
            y_crown = y_center + float(d[3])

            v_x = x_center - x_crown
            v_y = y_center - y_crown
            v_d = math.sqrt(v_x**2 + v_y**2)
            v_x /= v_d
            v_y /= v_d

            x1 = x_crown - v_y * tooth_w / 2
            y1 = y_crown + v_x * tooth_w / 2
            x2 = x1 + v_y * tooth_w
            y2 = y1 - v_x * tooth_w
            x3 = x2 + v_x * tooth_h
            y3 = y2 + v_y * tooth_h
            x4 = x3 - v_y * tooth_w
            y4 = y3 + v_x * tooth_w

            if save_img:
                '''
                cv2.line(img_list[pred_loc], (int(x1), int(y1)), (int(x2), int(y2)), color)
                cv2.line(img_list[pred_loc], (int(x2), int(y2)), (int(x3), int(y3)), color)
                cv2.line(img_list[pred_loc], (int(x3), int(y3)), (int(x4), int(y4)), color)
                cv2.line(img_list[pred_loc], (int(x4), int(y4)), (int(x1), int(y1)), color)
                cv2.circle(img_list[pred_loc], (int(x_center), int(y_center)), 2, color, -1)
                cv2.circle(img_list[pred_loc], (int(x_crown), int(y_crown)), 4, color, 1)
                '''

                cv2.line(img_list[0], (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0))
                cv2.line(img_list[0], (int(x2), int(y2)), (int(x3), int(y3)), (255,0,0))
                cv2.line(img_list[0], (int(x3), int(y3)), (int(x4), int(y4)), (255,0,0))
                cv2.line(img_list[0], (int(x4), int(y4)), (int(x1), int(y1)), (255,0,0))
                cv2.circle(img_list[0], (int(x_center), int(y_center)), 2, (0,255,0), -1)
                cv2.circle(img_list[0], (int(x_crown), int(y_crown)), 4, (255,0,0), 1)
                cv2.putText(img_list[0], str(pred_tooth_num), (int(x_center - 15), int(y_center - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

            if save_json:
                x1 += margin
                x2 += margin
                x3 += margin
                x4 += margin
                x_center += margin
                x1 /= 768 + 2 * margin
                x2 /= 768 + 2 * margin
                x3 /= 768 + 2 * margin
                x4 /= 768 + 2 * margin
                x_center /= 768 + 2 * margin
                y1 /= 512
                y2 /= 512
                y3 /= 512
                y4 /= 512
                y_center /= 512
                cur_dict = {}
                cur_dict['alveolar'] = (x_center, y_center)
                cur_dict['box_coords'] = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

                '''
                all_scores = []
                for l in loc_scores:
                    for c in cls_scores:
                        all_scores.append(float(l * c))

                cur_dict['class_scores'] = all_scores
                '''
                cur_dict['class_scores'] = cls_scores.tolist()
                output_list.append(cur_dict)

        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

        if save_img:
            cv2.imwrite(os.path.join(vis_dir, img_id + '.jpg'), img_list[0])
            #for i in range(5):
            #    cv2.imwrite(os.path.join(vis_dir, img_id + '_' + str(i) + '.jpg'), img_list[i])

        if save_json:
            with open(os.path.join(vis_dir, img_id + '.json'), 'w') as f:
                json.dump(output_list, f)


    def save_result(self, output, batch, results):
        hm_center = _sigmoid(output['hm_center'])
        reg_center = output['reg_center'] if self.opt.reg_offset else None
        reg_crown = output['reg_crown'] if self.opt.reg_offset else None

        dets = pano_decode_det_and_cls(hm_center, output['tooth_wh'], reg_center, reg_crown, K=32)
        self.draw_result(batch, dets[0].detach().cpu().numpy(), save_json=True, save_img=True)
