import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


class ConfusionMatrix:
    def __init__(self, true_path, pred_path, save_path):
        self.t_path = true_path
        self.p_path = pred_path
        self.save_path = save_path

    def prep_data(self):
        dirs_true = ['DOT_DUST', 'ASPECT', 'BG', 'INCLUSION', 'STAIN', 'SCRATCH']
        dirs_pred = ['DOT_DUST', 'DOT_RED', 'COLOUR_DRY', 'ASPECT', 'BG', 'INCLUSION', 'STAIN',
                     'SCRATCH']
        true_dict = {}
        pred_dict = {}
        for dirs in dirs_true:
            folder = self.t_path + f'{dirs}/'
            for file_name in os.listdir(folder):
                if file_name.split('.')[-1] == 'png':
                    true_dict[file_name.split('.')[0]] = dirs

        for dirs in dirs_pred:
            folder = self.p_path + f'{dirs}/'
            for file_name in os.listdir(folder):
                new_name = '_'.join(file_name.split('_')[:6])
                pred_dict[new_name] = dirs
        return true_dict, pred_dict

    def get_true_pred(self):
        true = []
        pred = []
        true_fn, pred_fn = self.get_true_pred()
        for key in pred_fn.keys():
            pred.append(pred_fn[key])
            true.append(true_fn[key])
        return true, pred

    def plot_cm(self, figsize=(10, 10)):
        y_true, y_pred = self.get_true_pred()
        cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_pred))
        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_perc = cm / cm_sum.astype(float) * 100
        annot = np.empty_like(cm).astype(str)
        nrows, ncols = cm.shape
        for i in range(nrows):
            for j in range(ncols):
                c = cm[i, j]
                p = cm_perc[i, j]
                if i == j:
                    s = cm_sum[i]
                    annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
                elif c == 0:
                    annot[i, j] = ''
                else:
                    annot[i, j] = '%.1f%%\n%d' % (p, c)
        cm = pd.DataFrame(cm, index=np.unique(y_pred), columns=np.unique(y_pred))
        cm.index.name = 'Actual'
        cm.columns.name = 'Predicted'
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cm, cmap="YlGnBu", annot=annot, fmt='', ax=ax)
        plt.savefig(self.save_path + 'confusion_matrix.png')

    def calculate_kpis(self):
        true_dict, pred_dict = self.prep_data()
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        defects = ['COLOUR_DRY', 'INCLUSION', 'SCRATCH', 'STAIN']
        for file in pred_dict.keys():
            if true_dict[file] in defects:
                if pred_dict[file] in defects:
                    tp += 1
                else:
                    fn += 1
            else:
                if pred_dict[file] in defects:
                    fp += 1
                else:
                    tn += 1

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        print(tp, tn, fp, fn)
        return precision, recall, accuracy


if __name__ == '__main__':
    t_path = '/home/schneian/data/fis_knb/20jul20/classification_B9_1NK/'
    p_path = '/home/schneian/data/fis_knb/20jul20/to_classify_20jul20/'
    save_path = '/home/schneian/data/fis_knb/20jul20/'
    cm = ConfusionMatrix(t_path, p_path, save_path)
    # cm.plot_cm((12, 12))

    p, r, a = calculate_kpis(d_true, d_pred)
    print(f"Precision: {'%.2f' % (p * 100)}\n"
          f"Recall: {'%.2f' % (r * 100)}\n"
          f"Accuracy: {'%.2f' % (a * 100)}")