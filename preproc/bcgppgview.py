import numpy as np
import matplotlib.pyplot as plt


trattenute = {"v1": [(60000, 90000), (120000, 150000), (180000, 210000), (240000, 270000),
                 (480000, 510000), (540000, 570000), (600000, 630000), (660000, 690000)],
              "v2": [(60000, 90000), (150000, 180000), (240000, 270000), (330000, 360000)]
            }
trattenute = trattenute["v2"] 
v2 = [2, 4, 5, 10,12,14,15,16, 18, 19,20]  # Example V2 values for the plot
v2 = [1,2,3]  # Example V2 values for the plot
# Reload the uploaded ECG data (file 5.csv)
folder_path = 'data/ppg/'

wleft = [50, 140, 230, 320]
wsize = 60
for id in v2:
    
    print(f"generating plot {id}")
    file_path = f'{folder_path}{id}.csv'
    
    signal = np.loadtxt(file_path, delimiter=",", skiprows=1)
    time = np.arange(len(signal)/50, step=0.02)  # Assuming 50 Hz sampling rate
    
    for left in wleft:
        fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        axs[0].plot(time[left*50:(left+wsize)*50], signal[left*50:(left+wsize)*50,0], label='BCG Signal', color='tab:blue')
        axs[0].set_ylabel('Amplitude')
        axs[0].set_xlabel('Time (s)')
        axs[0].set_title('PPG Signal Segment')
        for start, end in trattenute:
            if start/1000 >= time[left] and end/1000 <= time[left+wsize]:
                    axs[0].axvspan(start/1000, end/1000, color='yellow', alpha=0.5)     
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(time[left*50:(left+wsize)*50], signal[left*50:(left+wsize)*50, 1], label='PPG Signal', color='tab:blue')
        axs[1].set_ylabel('Amplitude')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_title('BCG Signal Segment')

        for start, end in trattenute:
            if start/1000 >= time[left] and end/1000 <= time[left+wsize]:
                axs[1].axvspan(start/1000, end/1000, color='yellow', alpha=0.5)
        axs[1].legend()
        axs[1].grid(True)
        plt.tight_layout()
        plt.savefig(f'{folder_path}visualization_ppg_bcg_{id}_{left}.png')