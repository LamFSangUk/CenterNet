from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import pandas as pd
import torch
import json
import cv2
import os
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
import math

class PanoDataset(data.Dataset):
    def pano_center_crop_and_resize(self, img):
        h, w = img.shape
        new_w = int(h * 1.5)
        margin = (w - new_w) // 2
        img = img[:, margin: margin + new_w]
        img = cv2.resize(img, dsize=self.default_resolution)

        img = np.float64(img) / 255
        img = (img - self.mean) / self.std
        return img    


    def get_tooth_class(self, tooth_num):
        if tooth_num % 10 < 3:
            return 1 # incisor
        if tooth_num % 10 == 3:
            return 2 # canine
        if tooth_num % 10 < 6:
            return 3 # premolar            

        return 4 # molar


    def change_coords(self, x, y, H, W):
        new_w = int(H * 1.5)
        margin = (W - new_w) // 2
        x -= margin
        x /= new_w
        y /= H

        return (x, y)


    def process_anno(self, anno_file_name, H, W):        
        w = W // 2
        h = H // 2
        annos = []
        df = pd.read_csv(os.path.join(self.data_dir, anno_file_name), header=None)        

        for idx, row in df.iterrows():
            tooth_num = int(row[0])
            tooth_class = self.get_tooth_class(tooth_num)            
            
            x_max = y_max = -1
            x_min = y_min = math.inf
            j = 3

            while j < 19:
                x = row[j]
                y = row[j+1]
                j += 2
                x_max = max(x_max, x)
                y_max = max(y_max, y)
                x_min = min(x_min, x)
                y_min = min(y_min, y)

            x_min += w
            x_max += w
            y_min += h
            y_max += h

            x_center = (x_min + x_max) // 2
            y_center = (y_min + y_max) // 2
            x_crown = w + (int(row[3]) + int(row[17])) // 2
            y_crown = h + (int(row[4]) + int(row[18])) // 2
            x_root = w + (int(row[9]) + int(row[11])) // 2
            y_root = h + (int(row[10]) + int(row[12])) // 2

            tooth_width = (int(row[5]) - int(row[15])) ** 2 + (int(row[6]) - int(row[16])) ** 2
            tooth_width = math.sqrt(tooth_width) / (H * 1.5)
            tooth_height = (x_crown - x_root) ** 2 + (y_crown - y_root) ** 2
            tooth_height = math.sqrt(tooth_height) / H

            x_center, y_center = self.change_coords(x_center, y_center, H, W)
            x_crown, y_crown = self.change_coords(x_crown, y_crown, H, W)
            x_root, y_root = self.change_coords(x_root, y_root, H, W)

            annos.append({
                'tooth_class': tooth_class,
                'tooth_size': (tooth_height, tooth_width),
                'extreme_points': [[x_center, y_center],
                                   [x_crown, y_crown],
                                   [x_root, y_root]]
            })

        return annos


    def __getitem__(self, index):
        img_file_name = self.img_file_names[index]
        img_path = os.path.join(self.data_dir, img_file_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        H, W = img.shape

        '''
        flipped = False
        if self.split == 'train':
            if not self.opt.not_rand_crop:
                s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
                w_border = self._get_border(128, img.shape[1])
                h_border = self._get_border(128, img.shape[0])
                c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
                c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
            else:
                sf = self.opt.scale
                cf = self.opt.shift
                s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
                c[0] += img.shape[1] * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
                c[1] += img.shape[0] * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
            if np.random.random() < self.opt.flip:
                flipped = True
                img = img[:, ::-1, :]

        trans_input = get_affine_transform(
            c, s, 0, [self.opt.input_res, self.opt.input_res])
        inp = cv2.warpAffine(img, trans_input,
                                                 (self.opt.input_res, self.opt.input_res),
                                                 flags=cv2.INTER_LINEAR)
        '''

        inp = self.pano_center_crop_and_resize(img)
        inp = np.expand_dims(inp, 0)
        output_h = self.opt.output_h
        output_w = self.opt.output_w

        hm_center = np.zeros((self.num_classes, output_h, output_w), dtype=np.float32)
        hm_crown = np.zeros((self.num_classes, output_h, output_w), dtype=np.float32)
        hm_root = np.zeros((self.num_classes, output_h, output_w), dtype=np.float32)
        reg_center = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg_crown = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg_root = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind_center = np.zeros((self.max_objs), dtype=np.int64)
        ind_crown = np.zeros((self.max_objs), dtype=np.int64)
        ind_root = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)

        anno_file_name = img_file_name[:-3] + 'txt'
        annos = self.process_anno(anno_file_name, H, W)
        num_objs = min(len(annos), self.max_objs)
        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
                        draw_umich_gaussian

        for k in range(num_objs):
            anno = annos[k]
            cls_id = anno['tooth_class']
            pts = np.array(anno['extreme_points'], dtype=np.float32) * [output_w, output_h]
            pt_int = pts.astype(np.int32)
            tooth_height, tooth_width = anno['tooth_size']
            tooth_height = math.ceil(tooth_height * output_h)
            tooth_width = math.ceil(tooth_width * output_w)
            radius = gaussian_radius((tooth_height, tooth_width))
            radius = max(0, int(radius))

            draw_gaussian(hm_center[cls_id], pt_int[0], radius)
            draw_gaussian(hm_crown[cls_id], pt_int[1], radius)
            draw_gaussian(hm_root[cls_id], pt_int[2], radius)
            reg_center[k] = pts[0] - pt_int[0]
            reg_crown[k] = pts[1] - pt_int[1]
            reg_root[k] = pts[2] - pt_int[2]
            ind_center[k] = pt_int[0, 1] * output_w + pt_int[0, 0]
            ind_crown[k] = pt_int[1, 1] * output_w + pt_int[1, 0]
            ind_root[k] = pt_int[2, 1] * output_w + pt_int[2, 0]            
            reg_mask[k] = 1

        ret = {
            'input': inp,
            'hm_center': hm_center, 'hm_crown': hm_crown, 'hm_root': hm_root,
            'reg_mask': reg_mask,
            'reg_w': anno['tooth_size'][1],
            'reg_center': reg_center, 'reg_crown': reg_crown, 'reg_root': reg_root,
            'ind_center': ind_center, 'ind_crown': ind_crown, 'ind_root': ind_root
        }
        
        return ret
