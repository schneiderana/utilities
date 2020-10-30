import pickle
import time
from collections import defaultdict
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from heatmap import HeatMap
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class Reproducibility:
    def __init__(self, plates_path):
        self.plates_path = Path(plates_path)
        self.heatmap = HeatMap(self.plates_path)
        self.save_folder = Path(Path.cwd() / 'output')
        self.save_folder.mkdir(exist_ok=True, parents=True)

    def get_plate_info(self):
        results_dict = {}
        for plate_path in self.plates_path.iterdir():
            temp_dict = {}
            df = pd.read_csv(plate_path / 'defects_in_plate.csv')
            for _, row in df.iterrows():
                temp_dict[row['crop_id_plate']] = {'x1': row['bounding_box_x'], 'y1': row['bounding_box_y'],
                                                   'w': row['defect_width'], 'h': row['defect_height'],
                                                   'defect': row['is_defect'], 'type': row['defect_type']}
            results_dict[plate_path.name] = temp_dict
        return results_dict

    @staticmethod
    def _draw_plate(x1a, x1b, wa, wb, y1a, y1b, ha, hb):
        x1 = min(x1a, x1b)
        y1 = min(y1a, y1b)
        x2 = max(x1a + wa, x1b + wb) - x1
        y2 = max(y1a + ha, y1b + hb) - y1
        base = np.zeros((y2, x2), dtype='uint8')
        plate_draw = cv2.rectangle(base, (x1a - x1, y1a - y1), (x1a - x1 + wa, y1a - y1 + ha), 1, -1)
        base = np.zeros((y2, x2), dtype='uint8')
        p_draw = cv2.rectangle(base, (x1b - x1, y1b - y1), (x1b - x1 + wb, y1b - y1 + hb), 1, -1)
        # area1 = wa * ha
        # area2 = wb * hb
        # intersection = (min(x1a - x1 + wa, x1b - x1 + wb) - max(x1a - x1, x1b - x1)) * \
        #                (min(y1a - y1 + ha, y1b - y1 + hb) - max(y1a - y1, y1b - y1))
        # union = area1 + area2 - intersection
        # return intersection, union
        return plate_draw, p_draw

    def evaluate_reproducibility(self):
        plates_info = self.get_plate_info()
        plates_seen = []
        results = {}
        start = time.time()
        for plate in plates_info.keys():
            for p in plates_info.keys():
                print(time.time() - start)
                start = time.time()
                if [plate, p] not in plates_seen and plate != p:
                    plates_seen.append([plate, p])
                    plates_seen.append([p, plate])
                    temp = defaultdict(list)
                    id_seen = []
                    print(plate, p)
                    for id in plates_info[plate].keys():
                        for i in plates_info[p].keys():
                            if [id, i] in id_seen:
                                continue
                            id_seen.append([id, i])
                            id_seen.append([i, id])
                            x1_plate = plates_info[plate][id]['x1']
                            w_plate = plates_info[plate][id]['w']
                            y1_plate = plates_info[plate][id]['y1']
                            h_plate = plates_info[plate][id]['h']
                            defect_plate = plates_info[plate][id]['type']
                            x1_p = plates_info[p][i]['x1']
                            w_p = plates_info[p][i]['w']
                            y1_p = plates_info[p][i]['y1']
                            h_p = plates_info[p][i]['h']
                            defect_p = plates_info[p][i]['type']
                            if max(0, x1_plate - w_plate) <= x1_p <= x1_plate + w_plate:
                                if max(0, y1_plate - h_plate) <= y1_p <= y1_plate + h_plate:
                                    draw_plate, draw_p = self._draw_plate(x1_plate, x1_p, w_plate, w_p, y1_plate, y1_p,
                                                                          h_plate, h_p)
                                    # intersection, union = self._draw_plate(x1_plate, x1_p, w_plate, w_p, y1_plate, y1_p,
                                    #                                        h_plate, h_p)
                                    intersection = np.count_nonzero(draw_plate == draw_p)
                                    union = np.count_nonzero(draw_plate) + np.count_nonzero(draw_p) - intersection
                                    temp['iou'].append(intersection / union)
                                    temp['defects'].append(f'{defect_plate}-{defect_p}')
                                else:
                                    continue
                            else:
                                continue
                    results[f'{plate}-{p}'] = temp
        with open(str(self.save_folder / 'results.pkl'), 'wb') as file:
            pickle.dump(results, file)
        return results

    def get_obs_pred(self):
        no_defects_list = ['ASPECT', 'BG', 'DOT_DUST', 'DOT_RED']
        defects_list = ['INCLUSION', 'COLOUR_DRY', 'SCRATCH', 'STAIN']
        with open(str(self.save_folder / 'results.pkl'), 'rb') as file:
            results = pickle.load(file)
        full_pred = []
        full_obs = []
        obs = []
        pred = []
        for key in results.keys():
            for index in range(len(results[key]['defects'])):
                if results[key]['iou'][index] > 0:
                    obs_name = results[key]['defects'][index].split('-')[0]
                    pred_name = results[key]['defects'][index].split('-')[1]
                    full_obs.append(obs_name)
                    full_pred.append(pred_name)
                    if obs_name in defects_list:
                        obs.append('DEFECT')
                    elif obs_name in no_defects_list:
                        obs.append('NO_DEFECT')
                    if pred_name in defects_list:
                        pred.append('DEFECT')
                    elif pred_name in no_defects_list:
                        pred.append('NO_DEFECT')
        return full_obs, full_pred, obs, pred

    def write_results(self, observed, predicted, name):
        conf_matrix = confusion_matrix(observed, predicted, labels=np.unique(observed + predicted))
        cm_sum = np.sum(conf_matrix, axis=1, keepdims=True)
        cm_perc = conf_matrix / cm_sum.astype(float) * 100
        annot = np.empty_like(conf_matrix).astype(str)
        nrows, ncols = conf_matrix.shape
        for i in range(nrows):
            for j in range(ncols):
                c = conf_matrix[i, j]
                p = cm_perc[i, j]
                if i == j:
                    s = cm_sum[i]
                    annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
                elif c == 0:
                    annot[i, j] = ''
                else:
                    annot[i, j] = '%.1f%%\n%d' % (p, c)
        cm_norm = confusion_matrix(observed, predicted, labels=np.unique(observed + predicted), normalize='true')
        cm = pd.DataFrame(cm_norm * 100, index=np.unique(predicted), columns=np.unique(observed + predicted))
        cm.index.name = ''
        cm.columns.name = ''
        fig, ax = plt.subplots(figsize=(11, 11))
        sns.heatmap(cm, cmap="YlGnBu", annot=annot, fmt='', ax=ax)
        save_name = self.save_folder / f'confusion_matrix_{name}.png'
        plt.savefig(str(save_name))


    # def _write_kpis(self):
#


if __name__ == '__main__':
    run_path = '/home/schneian/data/fis_knb/08sep20/daimler-mfa-2-brush-3w177-0001-2b45'
    reproducibility = Reproducibility(run_path)
    # var = reproducibility.evaluate_reproducibility()
    obs_class, pred_class, obs, pred = reproducibility.get_obs_pred()
    reproducibility.write_results(obs_class, pred_class, 'classes')
    reproducibility.write_results(obs, pred, 'defect_not_defect')