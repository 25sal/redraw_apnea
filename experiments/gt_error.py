# il dataset 5 originale  si trova nella cartalla data/dataset5
import numpy as np
import os
import json
from generation import gt_gaussian
import pandas as pd

def gen_gt_error(filename,profiles, error_mean=-7, error_std=15, offset=[0,0]):
    
    with open(filename, 'w') as fout:
        gt_json = {}
        for key in profiles.keys():
            gt_intervals = profiles[key]['gt']
            for fileid in profiles[key]['files']:
                gt_intervals_new = []
                for (gt_start, gt_end) in gt_intervals:
                    shift = np.random.normal(error_mean, error_std)
                    gt_intervals_new.append((gt_start + shift + offset[0], gt_end + shift + offset[1]))
                    gt_json
        json.dump(gt_json, fout, indent=4)
        




if __name__ == "__main__":

    window_size = 30000
    # if the folder does not exist, create it
    gaussian_mean = -7
    gaussian_std = 8
    
    # set to 0 for no error (only for gaussian metrics)
    error_mean = -7
    error_std = 15
    
    profiles = {'v1': {'files': [1,3,6,7,8,9,11,13,17], 'gt': [(60000, 90000), (120000, 150000), (180000, 210000), (240000, 270000), (480000, 510000), (540000, 570000), (600000, 630000), (660000, 690000)]},
                    'v2': {'files': [2,4,5,10,12,14,15,16,18, 19,20], 'gt': [(60000, 90000), (150000, 180000), (240000, 270000), (330000, 360000)]}
        }

    dataset_path = 'data/dataset5'
    exp_folder = f'experiments/smin10000_smax150000_seed42'
    filename = f'{dataset_path}/{exp_folder}/gt_error_{error_mean}_{error_std}.json'
    if not os.path.exists(os.path.dirname(filename)):
        gen_gt_error(filename, profiles, error_mean=error_mean, error_std=error_std)
    
    gt_intervals = json.load(open(filename))
    for key in gt_intervals.keys():
            window_file = f'{dataset_path}/{exp_folder}/{key}_samples.csv'
            if os.path.exists(filename):
                samples = pd.read_csv(window_file)
                windows = samples.iloc[:,0]
                score_list = []
                
                for window in windows:
                    wstart = window
                    wend = window + window_size
                    
                    score_list.append(gt_gaussian(gt_intervals[key], (wstart, wend)))
                    
                samples['gt_gaussian_error'] = np.array(score_list)
                samples.to_csv(window_file, index=False)