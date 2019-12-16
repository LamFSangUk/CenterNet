import os
import sys
import cv2
import json
import math
import numpy as np
import pandas as pd
import argparse


def get_intersection(subject_polygon, clip_polygon):
    """
        Get coords of intersection of two polygons

        Sutherland-Hodgman clipping
        https://en.wikipedia.org/wiki/Sutherland%E2%80%93Hodgman_algorithm
        Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python
    """
    def inside(p):
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

    def compute_intersection():
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

    output_list = subject_polygon
    cp1 = clip_polygon[-1]

    for clip_vertex in clip_polygon:
        cp2 = clip_vertex
        input_list = output_list
        output_list = []

        if len(input_list) == 0:
            break
        s = input_list[-1]

        for subjectVertex in input_list:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    output_list.append(compute_intersection())
                output_list.append(e)
            elif inside(s):
                output_list.append(compute_intersection())
            s = e
        cp1 = cp2

    return (output_list)


def get_polygon_area(vertices):
    """
        Get area of polygon using shoelace formula
    """
    n = len(vertices)  # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]

    area = abs(area) / 2.0
    return area


def get_squared_error(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2


def calculate_AP(det_list, target_iou):
    """
        calculate AP(Average Precision) with 11-point interpolation
        Ref: https://github.com/rafaelpadilla/Object-Detection-Metrics
    """
    target_recall = 100
    max_precision = 0
    precision_sum = 0

    for det in reversed(det_list):
        while det['recall_' + target_iou] * 100 < target_recall and target_recall >= 0:
            precision_sum += max_precision
            target_recall -= 10

        max_precision = max(max_precision, det['precision_' + target_iou])

    while target_recall >= 0:
        precision_sum += max_precision
        target_recall -= 10

    return precision_sum / 11


def resolve_duplicate(dets):
    """
        If two detections are at same class, choose one with higher confidence,
        then change the other one to next-highest class
    """
    det_count = len(dets)
    prob_list = []
    for d in dets:
        class_scores = d['class_scores']
        prob_list += class_scores

    for _ in range(det_count):
        max_idx = np.argmax(prob_list)
        det_idx = max_idx // 32
        cls_idx = max_idx % 32

        dets[det_idx]['pred_cls_idx'] = cls_idx
        for i in range(32):
            prob_list[det_idx * 32 + i] = -1
        for i in range(det_count):
            prob_list[32 * i + cls_idx] = -1

    return


def main(configs, epoch, do_postprocess=False):
    gt_path = 'D:\\osstem_pano_data\\gendata\\gt_object_aligned\\'
    exp_path = 'D:\\osstem_pano_data\\exp\\pano_det\\' + configs.exp_id
    results_path = exp_path + '\\vis_test_' + str(epoch)
    ignore_missing_tooth = True if configs.ignore_missing_tooth == 'True' else False

    class_list = ['11', '12', '13', '14', '15', '16', '17', '18',
                  '21', '22', '23', '24', '25', '26', '27', '28',
                  '31', '32', '33', '34', '35', '36', '37', '38',
                  '41', '42', '43', '44', '45', '46', '47', '48']
    class_num = len(class_list)
    gt_counts = {}
    class_det_list = {}
    AP_50 = []
    AP_75 = []
    AP_95 = []
    alveolar_error_sum = 0
    det_count = 0

    for c in class_list:
        class_det_list[c] = []
        gt_counts[c] = 0

    for f in os.listdir(results_path):
        if f[-4:] != 'json':
            continue        

        # load gt file and detection result file
        det_file = os.path.join(results_path, f)
        gt_file = os.path.join(gt_path, f)
        with open(det_file) as json_file_det:
            dets = json.load(json_file_det)
        with open(gt_file) as json_file_gt:
            gts = json.load(json_file_gt)
        orig_h, orig_w = gts['dim']

        # add gt count
        for c in class_list:
            if not c in gts:
                continue
            if gts[c]['exists'] == 'False':
                continue
            gt_counts[c] += 1

        # get rid of duplicate detections
        if do_postprocess:
            resolve_duplicate(dets)

        # add det to class_det_list
        for d in dets:
            class_scores = d['class_scores']
            pred_cls_idx = d['pred_cls_idx'] if do_postprocess else np.argmax(class_scores)
            pred_cls = class_list[pred_cls_idx]
            this_gt = gts[pred_cls]

            cur_det = {}
            cur_det['confidence'] = class_scores[pred_cls_idx]

            if this_gt['exists'] == 'False' and (this_gt['space'] == 'False' or ignore_missing_tooth):
                cur_det['TP_50'] = False
                cur_det['TP_75'] = False
                cur_det['TP_95'] = False
            else:
                # detemine TP_50, TP_75, TP_95
                polygon_gt = this_gt['box_coords']
                polygon_det = d['box_coords']
                for p in polygon_det:
                    p[0] *= orig_w
                    p[1] *= orig_h

                polygon_intersection = get_intersection(polygon_gt, polygon_det)
                area_gt = get_polygon_area(polygon_gt)
                area_det = get_polygon_area(polygon_det)
                area_intersection = get_polygon_area(polygon_intersection)
                iou = area_intersection / (area_gt + area_det - area_intersection)

                cur_det['TP_50'] = iou >= 0.5
                cur_det['TP_75'] = iou >= 0.75
                cur_det['TP_95'] = iou >= 0.95

                # calculate alveolar error(squared error)
                if iou >= 0.5:
                    det_alveolar = d['alveolar']
                    det_alveolar[0] *= orig_w
                    det_alveolar[1] *= orig_h
                    alveolar_error_sum += get_squared_error(det_alveolar, this_gt['alveolar'])
                    det_count += 1

            # add this det
            class_det_list[pred_cls].append(cur_det)

    # get AP for each class
    for c in class_list:
        # sort detections by confidence
        class_det_list[c].sort(key=lambda d: d['confidence'], reverse=True)
        accumulated_TP_50 = 0
        accumulated_TP_75 = 0
        accumulated_TP_95 = 0

        for i, d in enumerate(class_det_list[c]):
            # set accumulated TP
            if d['TP_50'] == True:
                accumulated_TP_50 += 1
            if d['TP_75'] == True:
                accumulated_TP_75 += 1
            if d['TP_95'] == True:
                accumulated_TP_95 += 1

            # set precision
            d['precision_50'] = accumulated_TP_50 / (i + 1)
            d['precision_75'] = accumulated_TP_75 / (i + 1)
            d['precision_95'] = accumulated_TP_95 / (i + 1)

            # set recall
            gt_count = gt_counts[c]
            d['recall_50'] = accumulated_TP_50 / gt_count
            d['recall_75'] = accumulated_TP_75 / gt_count
            d['recall_95'] = accumulated_TP_95 / gt_count

        # calculate AP with 11-point interpolation
        AP_50.append(calculate_AP(class_det_list[c], '50'))
        AP_75.append(calculate_AP(class_det_list[c], '75'))
        AP_95.append(calculate_AP(class_det_list[c], '95'))

    print('-------------------------------------------')
    print("epoch: ", epoch)
    for i in range(4):
        print(["{0:0.4f}".format(AP_50[j]) for j in range(i*8, i*8+8)])    

    # calculate mAP
    mAP_50 = sum(AP_50) / class_num
    mAP_75 = sum(AP_75) / class_num
    mAP_95 = sum(AP_95) / class_num

    # calculate MSE of alveolar point
    alveolar_mse = math.sqrt(alveolar_error_sum / det_count)

    print('mAP:', mAP_50, alveolar_mse)
    return


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', default='naive_numbering')
    parser.add_argument('--ignore_missing_tooth', default='True')
    configs = parser.parse_args()

    epoch = 140
    while epoch <= 140:
        main(configs, epoch, do_postprocess=True)
        epoch += 10
