import pandas as pd
import numpy as np
import sys
from datetime import datetime
import glob


folder_path= 'data/ppg'
subfolders = [1,2]
for subfolder in subfolders:
    folder = f"{folder_path}/{subfolder}/"
    #read bcg file
    raw_files  = glob.glob(f'{folder}raw_*.csv')
    if len(raw_files) == 1:
        bcg = pd.read_csv(raw_files[0])
        # interpolate time column each 9 samples
        new_time = []
        for i in range(0,len(bcg)-9,9):
            v = np.arange(bcg['t_local'].iloc[i], bcg['t_local'].iloc[i+9],(bcg['t_local'].iloc[i+9]-bcg['t_local'].iloc[i])/9)
            new_time = np.concatenate((new_time, v), axis=0)

        # last 9 samples are ignored, and cut
        bcg = bcg[['bcg_raw']][:len(new_time)]
        # convert to datetime to interpolate to 20ms
        bcg['datetime'] = pd.to_datetime(new_time, unit='s')
        bcg.index = bcg['datetime']
        bcg = bcg.drop(columns=['datetime'])
        #bcg.reset_index(drop=True, inplace=True)
        bcg=bcg['bcg_raw']
        bcg = bcg.resample('20L').mean().interpolate()

        # read ppg file
        raw_files  = glob.glob(f'{folder}spo2_*.csv')
        if len(raw_files) == 1:
            ppg = pd.read_csv(raw_files[0])

            # convert to datetime to interpolate to 20ms
            ppg['datetime'] = pd.to_datetime(ppg['pc_time_ms'], unit='ms')
            ppg.index = ppg['datetime']
            ppg = ppg[['wave']].resample('20L').mean().interpolate()

            # align start time removing first samples of the earlier started signal
            last_start = max(ppg.index[0], bcg.index[0])
            if ppg.index[0] < last_start:
                ppg = ppg[last_start:]
            if bcg.index[0] < last_start:
                bcg = bcg[last_start:]
            min_len = min(len(ppg), len(bcg))

            # truncate to the same length
            ppg = ppg[:min_len]
            bcg = bcg[:min_len]

            # save aligned signals
            aligned = pd.concat([ppg, bcg], axis=1)
            aligned.columns = ['ppg_wave', 'bcg_raw']
            # convert last_start to string for filename %Y-%m-%d_%H-%M-%S
            filename = datetime.strftime(last_start, '%Y-%m-%d_%H-%M-%S')
            aligned.to_csv(f'{folder_path}/{subfolder}.csv', index=False)

