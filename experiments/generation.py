# il dataset 5 originale  si trova nella cartalla data/dataset5
import numpy as np
import os

def sample_windows(stride_min, stride_max, len, width=30000, seed=42):
    x = [0]
    np.random.seed(seed)
    while x[-1] + width < len:
        stride =np.random.randint(stride_min, stride_max)
        x.append(x[-1] + stride)
    return x 

def gt_gaussian(gt_intervals, window, offset=[10000,5000]):
    # define a gaussian centered at left side of window
    
    score = 0
    # compute the are of the gaussian that intersects with the gt intervals
    wstart, wend = window
    for (gt_start, gt_end) in gt_intervals:
        # compute the intersection between the gaussian and the gt interval
        if not (wend <= gt_start+offset[0] or wstart >= gt_end + offset[1]):
            # compute the area of the gaussian that intersects with the gt interval
            intersect_start = max(gt_start+offset[0], wstart)
            intersect_end = min(gt_end+offset[1], wend)
            # compute the gussian values centering it in gt_end
            x = np.linspace(gt_start+offset[0], gt_end+offset[1], 1000)
            gaussian_mean = gt_end
            gaussian_std = 10000
            gaussian = (1/(gaussian_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - gaussian_mean)/gaussian_std)**2)
            dx = x[1] - x[0]
            intersection_mask = (x >= intersect_start) & (x <= intersect_end)
            score += np.sum(gaussian[intersection_mask]) * dx
            # drow the window the gaussian and the gt interval for debugging
            #import matplotlib.pyplot as plt
            #plt.plot(x, gaussian)
            #plt.axvspan(gt_start, gt_end, color='red', alpha=0.3)
            #plt.axvspan(wstart, wend, color='green', alpha=0.3)
            #plt.legend(['gaussian', 'gt interval', 'window'])
            #plt.show()
        
    return score

def gt__intersections_count(gt_intervals, window, offset=[10000,4000]):
    count = 0
    wstart, wend = window
    for (gt_start, gt_end) in gt_intervals:
        if not (wend <= gt_start+offset[0] or wstart >= gt_end+offset[1]):
            count = min(gt_end+offset[1], wend) - max(gt_start + offset[0], wstart)
    return count

if __name__ == "__main__":
    # if the folder does not exist, create it
    gaussian_mean = -7
    gaussian_std = 8
    
    # set equals for a constant strid
    stride_min = 10000
    stride_max = 15000
    
    # set to 0 for no error (only for gaussian metrics)
    error_mean = -7
    error_std = 15
    
    # for reproducibility
    seed = 42
    
    # should be not changed
    window_size = 30000

    metrics = {'intersection': gt__intersections_count, 'gaussian': gt_gaussian}
   

    dataset_path = 'data/dataset5'
    exp_folder = f'{dataset_path}/experiments/smin{stride_min}_smax{stride_max}_seed{seed}'
    if  os.path.exists(exp_folder):
        print(f"Folder {exp_folder} already exists. Remove it first!")
    else:
    
        os.makedirs(exp_folder)
        profiles = {'v1': {'files': [1,3,6,7,8,9,11,13,17], 'gt': [(60000, 90000), (120000, 150000), (180000, 210000), (240000, 270000), (480000, 510000), (540000, 570000), (600000, 630000), (660000, 690000)]},
                    'v2': {'files': [2,4,5,10,12,14,15,16,18, 19,20], 'gt': [(60000, 90000), (150000, 180000), (240000, 270000), (330000, 360000)]}
        }



        
        for profile in profiles:
            for data_file in profiles[profile]['files']:
                
                filename = f'{dataset_path}/{data_file}.txt'
                if os.path.exists(filename):
                    # how many samples in the file
                    len_file = sum(1 for line in open(filename))
                    windows = sample_windows(stride_min, stride_max, len_file, width=window_size,seed=seed)
                    score_list = {}
                    for metric in metrics.keys():
                        if metric not in score_list:
                            score_list[metric] = []
                    
                        for window in windows:
                            wstart = window
                            wend = min(window + window_size, len_file)
                            
                            score_list[metric].append(metrics[metric](profiles[profile]['gt'], (wstart, wend)))
                            
                    # save the window column and the score column in a csv file
                    output_filename = f'{exp_folder}/{data_file}_samples.csv'
                    m= np.array(windows)
                    for metric in score_list.keys():
                        m = np.column_stack((m, np.array(score_list[metric])))
                    header = 'window_start'
                    for metric in score_list.keys():
                        header += f',score_{metric}'
                    np.savetxt(output_filename, m, delimiter=',', header=header, comments='', fmt='%d')
                    print(f'Saved {output_filename}')