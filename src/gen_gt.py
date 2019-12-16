import os
import sys
import math
import cv2
import numpy as np
import json
import pandas as pd

def is_true(s):
    if 'T' in s or 't' in s:
        return True
    return False

data_path = 'D:\\osstem_pano_data\\'
raw_path = data_path + 'rawdata\\all\\'
output_path = data_path + 'gendata\\gt_object_aligned\\'
cols = ['tooth_num', 'x_alveolar', 'y_alveolar', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']
class_list = ['11', '12', '13', '14', '15', '16', '17', '18',
              '21', '22', '23', '24', '25', '26', '27', '28',
              '31', '32', '33', '34', '35', '36', '37', '38',
              '41', '42', '43', '44', '45', '46', '47', '48']

group_list = ['train', 'val', 'test']
ignore_list = ['000590', '000596', '000598', '000664', '000695', '000591', '000800']

for group in group_list:
    cur_path = raw_path + group
    file_list = os.listdir(cur_path)

    for f in file_list:
        if f[-3:] == 'txt':
            continue
        if f[:-4] in ignore_list:
            continue

        print(f)

        img_path = os.path.join(cur_path, f)
        txt_path = os.path.join(cur_path, f[:-3] + 'txt')
        out_dict = {}

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        h, w, _ = img.shape
        out_dict['dim'] = (h, w)
        h //= 2
        w //= 2

        df = pd.read_csv(txt_path, header=None)
        for i, row in df.iterrows():
            tooth_num = int(row[0])
            x_alveolar = w + int(row[27])
            y_alveolar = h + int(row[28])
            x_crown = w + (int(row[3]) + int(row[17])) // 2
            y_crown = h + (int(row[4]) + int(row[18])) // 2
            x_root = w + (int(row[9]) + int(row[11])) // 2
            y_root = h + (int(row[10]) + int(row[12])) // 2

            tooth_w = (int(row[5]) - int(row[15])) ** 2 + (int(row[6]) - int(row[16])) ** 2
            tooth_w = math.sqrt(tooth_w)
            tooth_h = (x_crown - x_root) ** 2 + (y_crown - y_root) ** 2
            tooth_h = math.sqrt(tooth_h)

            v_x = x_alveolar - x_crown
            v_y = y_alveolar - y_crown
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

            x1 = int(x1)
            x2 = int(x2)
            x3 = int(x3)
            x4 = int(x4)
            y1 = int(y1)
            y2 = int(y2)
            y3 = int(y3)
            y4 = int(y4)

            this_dict = {}
            this_dict['exists'] = 'True' if is_true(row[1]) else 'False'
            this_dict['space'] = 'True' if is_true(row[2]) else 'False'
            this_dict['alveolar'] = (x_alveolar, y_alveolar)
            this_dict['box_coords'] = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
            out_dict[str(tooth_num)] = this_dict

            color = (255,0,0)
            if not is_true(row[1]):
                if is_true(row[2]):
                    color = (0,255,255)
                else:
                    color = (0,0,255)

            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color)
            cv2.line(img, (int(x2), int(y2)), (int(x3), int(y3)), color)
            cv2.line(img, (int(x3), int(y3)), (int(x4), int(y4)), color)
            cv2.line(img, (int(x4), int(y4)), (int(x1), int(y1)), color)
            cv2.circle(img, (x_alveolar, y_alveolar), 3, (0,255,0), -1)
            cv2.putText(img, str(int(row[0])), (x_alveolar - 15, y_alveolar - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1)

        for c in class_list:
            if not c in out_dict:
                print(group, f)
                break

        output_file_name = os.path.join(output_path, f[:-3] + 'json')
        with open(output_file_name, "w") as json_file:
            json.dump(out_dict, json_file)

        output_img_name = os.path.join(output_path, f[:-3] + 'jpg')
        cv2.imwrite(output_img_name, img)
