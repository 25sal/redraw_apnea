# Re-import required packages after state reset
import numpy as np
import matplotlib.pyplot as plt
import sys


    
trattenute = {"v1": [(60000, 90000), (120000, 150000), (180000, 210000), (240000, 270000),
                 (480000, 510000), (540000, 570000), (600000, 630000), (660000, 690000)],
              "v2": [(60000, 90000), (150000, 180000), (240000, 270000), (330000, 360000)]
            }
trattenute = trattenute["v2"] 
v2 = [2, 4, 5, 10,12,14,15,16, 18, 19,20]  # Example V2 values for the plot

# Reload the uploaded ECG data (file 5.csv)
file_path = 'data/dataset5/'
PAUSE_MIN_SEC = 10              # min length of pause in EDR

events_detected = 0

for id in v2:
    print(f"generating plot {id}")
    file_path = f'data/dataset5/{id}.csv'
    
    # Compute parameters

    #ecg_vals = np.loadtxt(file_path)
    #ecg_time = np.arange(0,len(ecg_vals)/1000, 0.001)
    #axs[3].plot(ecg_time, ecg_vals)


    offset = 5
    delta_hr_vals = np.loadtxt(f"data/dataset5/{id}_delta_hr_vals.csv", delimiter=",") 
    delta_hr_times = np.arange(len(delta_hr_vals))
    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(12, 6), sharex=True)

    axs[0].plot(delta_hr_times[offset:], delta_hr_vals[offset:], label='ΔHR (bpm)', color='tab:blue')
    
    axs[0].axhline(3, color='gray', linestyle='--', label='ΔHR Treshold= 3 bpm')
    axs[0].set_ylabel('ΔHR (bpm)')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_title('ΔHR between successive beats')
    
    for start, end in trattenute:
        axs[0].axvspan(start/1000, end/1000, color='yellow', alpha=0.5)
    
    axs[0].legend()
    axs[0].grid(True)
        
   

    #Plotting lh/hf
    lfhf_vals = np.loadtxt(f"data/dataset5/{id}_delta_lfhf_vals.csv", delimiter=",")
    lfhf_times = np.arange(len(lfhf_vals))
    

    axs[1].plot(lfhf_times[offset:], lfhf_vals[offset:], label='LF/HF', color='tab:red')
    axs[1].axhline(2, color='gray', linestyle='--', label=' LF/HF Treshold = 2')
    axs[1].set_ylabel('LF/HF')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_title('LF/HF on moving window (60s width, 1s step)')
    
    for start, end in trattenute:
        axs[1].axvspan(start/1000, end/1000, color='yellow', alpha=0.5)
    
    '''
    if lfhf_intervals is not None:
        for start, end in lfhf_intervals:
            axs[1].axvspan(start, end, alpha=0.1, color='red', label='LF/HF increasing' if start == lfhf_intervals[0][0] else None)
    '''
   
    
    edr_vals = np.loadtxt(f"data/dataset5/{id}_edr_vals.csv", delimiter=",")
    edr_times = np.arange(len(edr_vals))
    
    axs[1].legend()
    axs[1].grid(True)
    axs[2].plot(edr_times[offset:], edr_vals[offset:], label='EDR', color='tab:green')
    axs[2].set_ylabel('EDR (mV)')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_title('EDR (R‑peak amplitudes)')
    for start, end in trattenute:
        axs[2].axvspan(start/1000, end/1000, color='yellow', alpha=0.5)
        
    '''
    for s, e in edr_pauses:
        axs[2].axvspan(s, e, alpha=0.2, color='green', label='EDR pause' if s == edr_pauses[0][0] else None)
    '''
    
    axs[2].legend()
    axs[2].grid(True)

    
 
    plt.tight_layout()
    plt.savefig(f'data/dataset5/plot_{id}.png')
