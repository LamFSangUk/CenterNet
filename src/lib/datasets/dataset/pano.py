from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import json
import os
import torch.utils.data as data


class PANO(data.Dataset):
    default_resolution = (512, 768)
    num_classes = 2
    mean = 0.4046034371726307
    std = 0.20749317242254753

    def __init__(self, opt, split):
        # needed: self.images, self.img_dir, self.split, self.mean/std, self.max_obj, self.anno
        super(PANO, self).__init__()
        #self.num_classes = 5
        #self.default_resolution = (300, 450)
        self.split = split # split = test or train?
        self.opt = opt
        self.data_dir = os.path.join(opt.data_dir, split)
        self.img_file_names = []

        for f in os.listdir(self.data_dir):
            if f[-3:] != 'txt' and 'thum' not in f:
                self.img_file_names.append(f)
        
        self.num_samples = len(self.img_file_names)
        self.max_objs = 32
        self.class_name = [
            '__background__',
            'tooth'
            #'incisor',  # front tooth (_1, _2)
            #'canine',   # sharp tooth (_3)
            #'premolar', # small molar (_4, _5)
            #'molar'     # big molar   (_6, _7, _8)
        ]

        # calculated manually
        #self.mean = 0.40984170839779815
        #self.std = 0.20721967617834552

    def __len__(self):
        return self.num_samples

    '''
    def _to_float(self, x):
        return float("{:.2f}".format(x))
    
    def convert_eval_format(self, all_bboxes):
        # import pdb; pdb.set_trace()
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = self._valid_ids[cls_ind - 1]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = bbox[4]
                    bbox_out  = list(map(self._to_float, bbox[0:4]))

                    detection = {
                            "image_id": int(image_id),
                            "category_id": int(category_id),
                            "bbox": bbox_out,
                            "score": float("{:.2f}".format(score))
                    }
                    if len(bbox) > 5:
                            extreme_points = list(map(self._to_float, bbox[5:13]))
                            detection["extreme_points"] = extreme_points
                    detections.append(detection)
        return detections


    def save_results(self, results, save_dir):
        json.dump(self.convert_eval_format(results), 
                                open('{}/results.json'.format(save_dir), 'w'))
    
    def run_eval(self, results, save_dir):
        # result_json = os.path.join(save_dir, "results.json")
        # detections  = self.convert_eval_format(results)
        # json.dump(detections, open(result_json, "w"))
        self.save_results(results, save_dir)
        coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    '''
