import numpy as np
from medpy.metric.binary import hd95

def fuzzy_dice_coeff(out, label):
    epsilon = 1e-6
    TP = np.minimum(out, label).sum()
    FP = np.maximum(out-label, 0).sum()
    TN = np.minimum(1-out, 1-label).sum()
    FN = np.maximum((1-out)-(1-label), 0).sum()
    return (2*TP + epsilon) / ((2*TP + FP + FN) + epsilon)

def fuzzy_precision(out, label):
    epsilon = 1e-6
    TP = np.minimum(out, label).sum()
    FP = np.maximum(out-label, 0).sum()
    TN = np.minimum(1-out, 1-label).sum()
    FN = np.maximum((1-out)-(1-label), 0).sum()
    return (TP + epsilon) / ((TP + FP) + epsilon)

def fuzzy_recall(out, label):
    epsilon = 1e-6
    TP = np.minimum(out, label).sum()
    FP = np.maximum(out-label, 0).sum()
    TN = np.minimum(1-out, 1-label).sum()
    FN = np.maximum((1-out)-(1-label), 0).sum()
    return (TP + epsilon) / ((TP + FN) + epsilon)

def fuzzy_hd95(out, label):
    threshold_num = 5
    fuzzy_hd95 = 0
    for i in range(1,threshold_num+1):
        threshold = (1.0 / (threshold_num+1)) * i
        out_tem = out.copy()
        label_tem = label.copy()
        out_tem[out_tem>=threshold] = 1.0
        out_tem[out_tem<threshold] = 0.0
        label_tem[label_tem>=threshold] = 1.0
        label_tem[label_tem<threshold] = 0.0
        if out_tem.max() == 1.0:
            fuzzy_hd95 = fuzzy_hd95 + hd95(out_tem, label_tem)
        else:
            return np.inf
    return fuzzy_hd95/threshold_num