import os
from evaluation_utils import fuzzy_dice_coeff, fuzzy_hd95, fuzzy_precision, fuzzy_recall
import numpy as np
import nibabel as nib
import pandas as pd

out_base_dir = ''
label_base_dir = ''
save_dir = 'results_all.csv'
matrics = ['dice', 'hd95', 'precision', 'recall']

if os.path.exists(save_dir):
    os.remove(save_dir)

file_list = os.listdir(out_base_dir)
case_list = []
for file in file_list:
    if os.path.isdir(os.path.join(out_base_dir, file)):
        case_list.append(file)
case_list.sort()
dataframe = pd.DataFrame(columns=case_list, index=matrics)

for case in case_list:
    file_list = os.listdir(os.path.join(out_base_dir, case))
    out_list = []
    for file in file_list:
        if os.path.splitext(file)[1] == '.npz':
            out_list.append(file)
    
    label = np.load(os.path.join(label_base_dir, case + '.npz'))['arr_0']
    num = label.shape[0]
    label_gtv = label[:,3,:,:].copy().reshape([num,512,512])

    out_data = np.load(os.path.join(out_base_dir, case, out_list[0]))['arr_0'].reshape([num,512,512])
    out_data[out_data<0] = 0
    out_data[out_data>1] = 1
    assert out_data.shape == label_gtv.shape
    dice_img = fuzzy_dice_coeff(out_data, label_gtv)
    hd95_img = fuzzy_hd95(out_data, label_gtv)
    precision_img = fuzzy_precision(out_data, label_gtv)
    recall_img = fuzzy_recall(out_data, label_gtv)

    dataframe.loc['dice',case] = dice_img
    dataframe.loc['hd95',case] = hd95_img
    dataframe.loc['precision',case] = precision_img
    dataframe.loc['recall',case] = recall_img
dataframe.to_csv(save_dir)