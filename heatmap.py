import json

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


class HeatMap:
    def __init__(self, plates_path):
        self.plates_path = Path(plates_path)

    def get_max_plate_shape(self):
        plate_shapes_y = []
        plate_shapes_x = []
        for folder_path in self.plates_path.iterdir():
            with open(str(folder_path / 'plate_info.json'), 'r') as f:
                plate_info = json.load(f)
            plate_shape = [int(plate_info['info']['plate_info']['plate_shape'][0] -
                               plate_info['info']['plate_info']['margin_boundaries']['upper_boundary'] -
                               plate_info['info']['plate_info']['margin_boundaries']['bottom_boundaries']),
                           int(plate_info['info']['plate_info']['plate_shape'][1] -
                               plate_info['info']['plate_info']['margin_boundaries']['left_boundary'] -
                               plate_info['info']['plate_info']['margin_boundaries']['right_boundary'])]
            plate_shapes_y.append(plate_shape[0])
            plate_shapes_x.append(plate_shape[1])
        return [max(plate_shapes_y), max(plate_shapes_x)]

    @staticmethod
    def get_plate_info(df, plate_number):
        results_dict = {}
        temp_dict = {}
        for _, row in df.iterrows():
            temp_dict[row['crop_id_plate']] = {'x1': row['bounding_box_x'], 'y1': row['bounding_box_y'],
                                               'w': row['defect_width'], 'h': row['defect_height'],
                                               'area': row['defect_width'] * row['defect_height'],
                                               'defect': row['is_defect'], 'type': row['defect_type'],
                                               'b_nb': row['buffer_number'], 'c_nb': row['crop_id_buffer']}
        results_dict[plate_number] = temp_dict
        return results_dict

    @staticmethod
    def draw_defects_per_plate(info_plate, temp):
        for plate_number, val in info_plate.items():
            for id in val.keys():
                x1 = info_plate[plate_number][id]['x1']
                x2 = x1 + info_plate[plate_number][id]['w']
                y1 = info_plate[plate_number][id]['y1']
                y2 = y1 + info_plate[plate_number][id]['h']
                temp = cv2.rectangle(temp, (x1, y1), (x2, y2), 1, -1)
        return temp

    def _sum_defects_of_plates(self, path_list, background):
        drawn_defects = background.copy()
        temp = background.copy()
        for folder_path in path_list:
            defects = pd.read_csv(folder_path / 'defects_in_plate.csv')
            plate_info = self.get_plate_info(defects, folder_path.name)
            print(folder_path.name)
            temp = self.draw_defects_per_plate(plate_info, temp)
            drawn_defects = drawn_defects + temp
        return drawn_defects

    def draw_plate_defects(self, max_shape):
        background = np.zeros(max_shape, dtype='int32')
        path_list = [p for p in self.plates_path.iterdir()]
        results = self._sum_defects_of_plates(path_list, background)
        return results

    @staticmethod
    def draw_heatmap(plates_defects):
        plt.imshow(plates_defects * 1000, cmap='YlOrRd', interpolation='bilinear')
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    run_path = '/home/schneian/data/fis_knb/08sep20/daimler-mfa-2-brush-3w177-0001-2b45'
    heatmap = HeatMap(run_path)
    shape = heatmap.get_max_plate_shape()
    plates_sum = heatmap.draw_plate_defects(shape)
    heatmap.draw_heatmap(plates_sum)
