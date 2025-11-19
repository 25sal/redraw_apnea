
import json
import matplotlib.pyplot as plt
import glob
import pandas as pd
import os

# set to 0 for no error (only for gaussian metrics)
error_mean = -7000
error_std = 15000

profiles = {'v1': {'files': [1,3,6,7,8,9,11,13,17], 'gt': [(60000, 90000), (120000, 150000), (180000, 210000), (240000, 270000), (480000, 510000), (540000, 570000), (600000, 630000), (660000, 690000)]},
                'v2': {'files': [2,4,5,10,12,14,15,16,18, 19,20], 'gt': [(60000, 90000), (150000, 180000), (240000, 270000), (330000, 360000)]}
    }

dataset_path = 'data/dataset5'
exp_folder = f'experiments/smin10000_smax10000_seed42'
filename = f'{dataset_path}/{exp_folder}/gt_error_{error_mean}_{error_std}.json'


def plot_gt_error():
    jsonlist = json.load(open(filename))

    for key in profiles.keys():
            gt_intervals = profiles[key]['gt']
            for fileid in profiles[key]['files']:
                gt_intervals_error = jsonlist[f'{fileid}']
                plt.figure(figsize=(10,5))
                for (gt_start, gt_end) in gt_intervals:
                    plt.axvspan(gt_start, gt_end, color='green', alpha=0.3)
                for (gt_start, gt_end) in gt_intervals_error:
                    plt.axvspan(gt_start, gt_end, color='yellow', alpha=0.3)
                plt.savefig(f'{dataset_path}/{exp_folder}/gt_error_visualization_file{fileid}.png')
                plt.close() 

def diff_labels(tresholds):
    apnea_intervals = [0,0,0]
    diff_count = [0,0]
    total_count = 0
    for key in profiles.keys():
            gt_intervals = profiles[key]['gt']
            for fileid in profiles[key]['files']:
                filename = f'{dataset_path}/{exp_folder}/{fileid}_samples.csv'
                if os.path.exists(filename):
                    df = pd.read_csv(filename)
                    df = df[['score_intersection','score_gaussian','gt_gaussian_error']]
                    # put 1 in each column if above treshold
                    df['pred_intersection'] = (df['score_intersection'] >= tresholds[0]).astype(int)
                    df['pred_gaussian'] = (df['score_gaussian'] >= tresholds[1]).astype(int)
                    df['pred_gaussian_error'] = (df['gt_gaussian_error'] >= tresholds[2]).astype(int)
                    # compare pred_intersection with gt_label
                    diff_count[0] += (df['pred_intersection'] != df['pred_gaussian']).sum()
                    diff_count[1] += (df['pred_gaussian_error'] != df['pred_gaussian']).sum()
                    apnea_intervals[0] += df['pred_intersection'].sum()
                    apnea_intervals[1] += df['pred_gaussian'].sum()
                    apnea_intervals[2] += df['pred_gaussian_error'].sum()
                    total_count += len(df)
    return diff_count, total_count, apnea_intervals

#plot_gt_error()
print(diff_labels([15000,0.6, 0.6]))