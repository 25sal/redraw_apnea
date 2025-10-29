import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks


# ---------- Filters and helpers ----------

def butter_bandpass(data, fs, low=5.0, high=15.0, order=3):
    nyq = fs/2
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, data)

def moving_average(x, w):
    if w <= 1:
        return x.astype(float)
    c = np.convolve(x, np.ones(w, dtype=float)/w, mode='same')
    return c

def robust_rr_clean(rr, rr_min=0.3, rr_max=2.5, z_thresh=3.5):
    rr = np.asarray(rr, dtype=float)
    # Physiologic clamp
    keep = (rr >= rr_min) & (rr <= rr_max)
    rr_clipped = rr[keep]
    # Outlier by robust z-score
    if len(rr_clipped) < 5:
        return rr_clipped
    med = np.median(rr_clipped)
    mad = np.median(np.abs(rr_clipped - med)) + 1e-9
    z = 0.6745*(rr_clipped - med)/mad
    return rr_clipped[np.abs(z) <= z_thresh]

def interp_to_grid(x_times, x_vals, t_grid, left=None, right=None):
    if len(x_times) == 0:
        return np.full_like(t_grid, np.nan, dtype=float)
    if left is None:
        left = x_vals[0]
    if right is None:
        right = x_vals[-1]
    return np.interp(t_grid, x_times, x_vals, left=left, right=right)

# ---------- R-peak detection (robust, reproducible) ----------


def detect_rpeaks_twostep(ecg, fs):
    """
    Two-step approach:
    1. Liberal detection (catch all candidates)
    2. Median-based outlier removal
    """
    # Step 1: Preprocessing
    ecg_f = butter_bandpass(ecg, fs, low=5.0, high=15.0, order=3)
    sq = ecg_f**2
    ma_win = int(0.150 * fs)
    integ = moving_average(sq, ma_win)
    
    # Step 2: Liberal peak detection
    distance = int(0.4 * fs)
    prom = np.percentile(integ, 65)  # Lower threshold (75th percentile)
    
    peaks, _ = find_peaks(integ, distance=distance, prominence=prom)
    
    if len(peaks) < 3:
        return peaks
    
    # Step 3: Compute all RR intervals
    rr_intervals = np.diff(peaks) / fs
    
    # Step 4: Remove outliers using median absolute deviation (MAD)
    rr_median = np.median(rr_intervals)
    mad = np.median(np.abs(rr_intervals - rr_median))
    
    # Accept RR intervals within 3 MAD of median
    threshold_lower = max(0.4, rr_median - 3 * mad)
    threshold_upper = min(2.0, rr_median + 3 * mad)
    
    # Rebuild peak list
    peaks_clean = [peaks[0]]
    for i in range(len(rr_intervals)):
        if threshold_lower <= rr_intervals[i] <= threshold_upper:
            peaks_clean.append(peaks[i+1])
        else:
            # Check if gap can be bridged
            if len(peaks_clean) > 0:
                gap = (peaks[i+1] - peaks_clean[-1]) / fs
                if gap <= threshold_upper:
                    peaks_clean.append(peaks[i+1])
    
    return np.array(peaks_clean)



def detect_rpeaks_simple_robust(ecg, fs):
    """
    Simplified Pan-Tompkins - avoids complex adaptive feedback.
    Best for BCG/sleep apnea monitoring.
    """
    # Preprocessing
    ecg_f = butter_bandpass(ecg, fs, low=5.0, high=15.0, order=3)
    sq = ecg_f**2
    ma_win = int(0.150 * fs)
    integ = moving_average(sq, ma_win)
    
    # Conservative peak detection
    distance = int(0.4 * fs)  # 400ms minimum (150 bpm max)
    prom = np.percentile(integ, 50)  # 60th percentile
    
    peaks, _ = find_peaks(integ, distance=distance, prominence=prom)
    
    # Post-filter: keep only physiologically valid intervals
    if len(peaks) < 2:
        return peaks
    
    peaks_clean = [peaks[0]]
    for i in range(1, len(peaks)):
        rr = (peaks[i] - peaks_clean[-1]) / fs
        if 0.4 <= rr <= 2.0:  # 30-150 bpm
            peaks_clean.append(peaks[i])
    
    return np.array(peaks_clean)




def detect_rpeaks_adaptive_rr(ecg, fs, min_distance_s=0.3, ma_win_s=0.150, dyn_prom_pct=50):
    ecg_f = butter_bandpass(ecg, fs, low=5.0, high=15.0, order=3)
    sq = ecg_f**2
    ma_win = max(1, int(round(ma_win_s*fs)))
    integ = moving_average(sq, ma_win)
    distance = max(1, int(round(min_distance_s*fs)))
    
    prom = np.percentile(integ, dyn_prom_pct)
    peaks_raw, _ = find_peaks(integ, distance=distance, prominence=prom)
    
    # Adaptive filtering based on RR-interval average
    peaks_filtered = [peaks_raw[0]]
    rr_history = []
    
    for i in range(1, len(peaks_raw)):
        rr_interval = (peaks_raw[i] - peaks_filtered[-1]) / fs
        
        # Calculate running average of last 8 RR intervals
        if len(rr_history) >= 8:
            rr_avg = np.mean(rr_history[-8:])
        elif len(rr_history) > 0:
            rr_avg = np.mean(rr_history)
        else:
            rr_avg = 1.0  # Default ~60 bpm
        
        # Reject if too short (< 360ms OR < 0.5 * average)
        if rr_interval >= 0.36 and rr_interval >= 0.5 * rr_avg:
            peaks_filtered.append(peaks_raw[i])
            rr_history.append(rr_interval)
    
    return np.array(peaks_filtered)



def detect_rpeaks(ecg, fs, min_distance_s=0.4, ma_win_s=0.150, dyn_prom_pct=50):
    """
    Simple Pan–Tompkins-like pipeline:
      1) bandpass 5–15 Hz to emphasize QRS
      2) square the signal
      3) moving-average integration (~150 ms)
      4) find_peaks with minimum distance and a dynamic prominence threshold
    """
    # 1) Bandpass
    ecg_f = butter_bandpass(ecg, fs, low=5.0, high=15.0, order=3)
    # 2) Square
    sq = ecg_f**2
    # 3) Moving-average integration
    ma_win = max(1, int(round(ma_win_s*fs)))
    integ = moving_average(sq, ma_win)
    # 4) Peaks with distance and dynamic threshold
    distance = max(1, int(round(min_distance_s*fs)))
    
    
    #prom = np.percentile(integ, dyn_prom_pct)  # adaptive
    #prom = max(prom, np.std(integ)*0.5)        # floor on variability
    prom = np.percentile(integ, dyn_prom_pct)
    peaks, _ = find_peaks(integ, distance=distance, prominence=prom)
 
    return peaks





def compute_hr_dhr_edr_fixed_grid(ecg, fs=100, grid_hz=1, rr_min=0.3, rr_max=2.5):
    """
    Returns consistent-length arrays on a fixed time grid:
      - t_grid_s: np.ndarray [N]
      - hr_grid: np.ndarray [N] (bpm)
      - dhr_grid: np.ndarray [N] (abs first diff of HR on grid)
      - edr_grid: np.ndarray [N] (R-peak amplitudes interpolated on grid; lowpassed optional)
    """
    ecg = np.asarray(ecg, dtype=float)
    T = len(ecg)/fs
    # Fixed grid over full duration, 0..floor(T) seconds
    N = int(np.floor(T*grid_hz)) + 1
    t_grid_s = np.arange(N, dtype=float)/grid_hz

    # Detect R-peaks and build RR/HR
    r_idx = detect_rpeaks_adaptive_rr(ecg, fs)
    if len(r_idx) < 2:
        # not enough beats; return NaNs
        return t_grid_s, np.full(N, np.nan), np.full(N, np.nan), np.full(N, np.nan)

    # RR and HR (instantaneous)
    rr_s = np.diff(r_idx)/fs
    rr_s = robust_rr_clean(rr_s, rr_min=rr_min, rr_max=rr_max)
    # If cleaning removed edges, rebuild times accordingly:
    # Use midpoints of each valid RR interval.
    # First, compute RR from original r_idx; then mask the ones kept
    rr_all = np.diff(r_idx)/fs
    rr_keep = (rr_all >= rr_min) & (rr_all <= rr_max)
    # mid-time for each RR interval at second R-peak index
    rr_mid_times = (r_idx[1:]/fs)
    rr_mid_times = rr_mid_times[rr_keep]
    hr_bpm = 60.0/rr_all[rr_keep]
    if len(hr_bpm) == 0:
        return t_grid_s, np.full(N, np.nan), np.full(N, np.nan), np.full(N, np.nan)

    # Interpolate HR to fixed grid
    hr_grid = interp_to_grid(rr_mid_times, hr_bpm, t_grid_s, left=hr_bpm[0], right=hr_bpm[-1])

    # Compute DHR on the same grid (abs first difference)
    dhr_grid = np.abs(np.diff(hr_grid, prepend=hr_grid[0]))

    # EDR from R-peak amplitudes (simple proxy): use bandpassed absolute peak height
    # Sample amplitude at R-peak indices from bandpassed ECG
    ecg_f = butter_bandpass(ecg, fs, low=5.0, high=15.0, order=3)
    r_amp = np.abs(ecg_f[r_idx])
    r_times = r_idx/fs
    edr_grid = interp_to_grid(r_times, r_amp, t_grid_s, left=r_amp[0], right=r_amp[-1])

    return t_grid_s, hr_grid, dhr_grid, edr_grid

# ---------- LF/HF (expected shorter due to windowing) ----------

def compute_lfhf(hr_grid_bpm, t_grid_s, psd_fs=4.0, win_s=60, step_s=1):
    """
    Standard HRV LF/HF on a resampled HR series:
      1) Interpolate HR to 4 Hz
      2) Sliding Welch PSD over win_s, step_s
      3) Compute LF/HF per window
    Returns: times_s, lfhf_vals
    """
    # Build uniform 4 Hz grid over available HR time span
    if len(t_grid_s) < 2 or np.all(np.isnan(hr_grid_bpm)):
        return np.array([]), np.array([])
    t4 = np.arange(t_grid_s[0], t_grid_s[-1]+1e-9, 1.0/psd_fs)
    hr4 = interp_to_grid(t_grid_s, hr_grid_bpm, t4, left=hr_grid_bpm[0], right=hr_grid_bpm[-1])

    # Basic cleaning: replace NaNs if any
    if np.any(np.isnan(hr4)):
        # simple forward-fill then back-fill
        s = pd.Series(hr4).fillna(method='ffill').fillna(method='bfill')
        hr4 = s.values

    from scipy.signal import welch

    nperseg = int(win_s*psd_fs)
    hop = int(step_s*psd_fs)
    vals, times = [], []
    for start in range(0, len(hr4)-nperseg+1, hop):
        seg = hr4[start:start+nperseg]
        f, pxx = welch(seg, fs=psd_fs, nperseg=nperseg)
        lf_band = (f>=0.04)&(f<0.15)
        hf_band = (f>=0.15)&(f<0.40)
        lf = np.trapz(pxx[lf_band], f[lf_band])
        hf = np.trapz(pxx[hf_band], f[hf_band])
        ratio = lf/hf if hf > 0 else np.nan
        vals.append(ratio)
        # center time of the window in seconds
        t_center = t4[start] + 0.5*win_s
        times.append(t_center)
    return np.array(times), np.array(vals)

# ---------- Example usage ----------
if __name__ == "__main__":
    # Example: read raw ECG, fs=100 Hz, produce aligned outputs
    # Replace '2.csv' and '19.csv' with your filenames (single-column ECG)
    import pandas as pd

    def run_one(ecg_file, fs=100):
        ecg = pd.read_csv(ecg_file, header=None).iloc[:,0].values.astype(float)
        t_s, hr, dhr, edr = compute_hr_dhr_edr_fixed_grid(ecg, fs=fs, grid_hz=1)
        
        t_lfhf, lfhf = compute_lfhf(hr, t_s, psd_fs=4.0, win_s=60, step_s=1)
        print(f"{ecg_file}: duration={len(ecg)/fs:.1f}s "
              f"| HR_len={len(hr)} DHR_len={len(dhr)} EDR_len={len(edr)} "
              f"| LFHF_len={len(lfhf)}")
        # Save if needed
        pd.Series(dhr).to_csv(Path(ecg_file).stem + "_delta_hr_vals.csv", index=False)
        pd.Series(lfhf).to_csv(Path(ecg_file).stem + "_delta_lfhf_vals.csv", index=False)
        pd.Series(edr).to_csv(Path(ecg_file).stem + "_edr_vals.csv", index=False)
        return t_s, hr, dhr, edr, t_lfhf, lfhf


    v2 = [2, 4, 5, 10,12,14,15,16, 18, 19,20]  # Example V2 values for the plot
   
    
    from pathlib import Path

    
    for id in v2:
        df = pd.read_csv(f"data/dataset5/{id}.txt", header=None, delimiter='\t')
        ecg = df.iloc[:,0-1].values.astype(float)
        pd.Series(ecg).to_csv(f"data/dataset5/{id}.csv", index=False, header=False)
        run_one(f"data/dataset5/{id}.csv", fs=1000)
