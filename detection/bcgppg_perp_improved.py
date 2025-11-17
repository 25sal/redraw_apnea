import pandas as pd
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt

# ============================================================================
# IMPROVED APNEA DETECTION USING BCG AMPLITUDE ANALYSIS
# ============================================================================
# Key improvements based on visual inspection:
# 1. BCG amplitude reduction is the strongest indicator (75.8% during apnea)
# 2. Use amplitude-based detection with sliding windows
# 3. Combine amplitude with peak count for robustness
# ============================================================================

# Configuration

SAMPLING_FREQUENCY = 50  # Hz
WINDOW_SIZE = 10  # seconds - window for analysis
STRIDE = 2  # seconds - sliding window stride
BASELINE_DURATION = 50  # seconds - use first 50s as baseline
AMPLITUDE_THRESHOLD = 50  # % - minimum amplitude reduction to detect apnea
MIN_PEAK_COUNT = 8  # peaks per window - below this suggests apnea
IRREGULARITY_THRESHOLD = 0.5  # coefficient of variation threshold

v2 = [1,2]
FOLDER_PATH= 'data/ppg/'

for id in v2:
    FILENAME = f'{FOLDER_PATH}{id}.csv'

    # Load data
    print("Loading data...")
    df = pd.read_csv(FILENAME)
    ppg = df['ppg_wave'].values
    bcg = df['bcg_raw'].values
    time = np.arange(len(ppg)) / SAMPLING_FREQUENCY

    print(f"Data loaded: {len(ppg)} samples at {SAMPLING_FREQUENCY} Hz")
    print(f"Recording duration: {len(ppg)/SAMPLING_FREQUENCY:.1f} seconds\n")

    # Bandpass filter function
    def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
        """Apply bandpass filter to isolate cardiac frequency components"""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, data)
        return y

    # Filter signals
    print("Filtering signals...")
    bcg_filtered = butter_bandpass_filter(bcg, 0.5, 10, SAMPLING_FREQUENCY)
    ppg_filtered = butter_bandpass_filter(ppg, 0.5, 8, SAMPLING_FREQUENCY)

    # Calculate baseline metrics from first 50 seconds
    baseline_end_idx = BASELINE_DURATION * SAMPLING_FREQUENCY
    baseline_amplitude = np.std(bcg_filtered[:baseline_end_idx])

    # Detect baseline peaks
    baseline_peaks, _ = find_peaks(bcg_filtered[:baseline_end_idx], 
                                distance=SAMPLING_FREQUENCY*0.4, 
                                prominence=50)
    baseline_peak_rate = len(baseline_peaks) / BASELINE_DURATION * 60  # peaks per minute

    print(f"Baseline metrics (0-{BASELINE_DURATION}s):")
    print(f"  BCG amplitude: {baseline_amplitude:.2f}")
    print(f"  Peak rate: {baseline_peak_rate:.1f} peaks/min\n")

    # Sliding window analysis
    print("Analyzing windows for apnea detection...")
    window_samples = WINDOW_SIZE * SAMPLING_FREQUENCY
    stride_samples = STRIDE * SAMPLING_FREQUENCY

    results = []
    for start_idx in range(0, len(bcg) - window_samples, stride_samples):
        end_idx = start_idx + window_samples
        window_time_start = start_idx / SAMPLING_FREQUENCY
        window_time_end = end_idx / SAMPLING_FREQUENCY
        
        # Amplitude analysis
        window_amplitude = np.std(bcg_filtered[start_idx:end_idx])
        amplitude_reduction = ((baseline_amplitude - window_amplitude) / baseline_amplitude) * 100
        
        # Peak analysis
        window_peaks, _ = find_peaks(bcg_filtered[start_idx:end_idx], 
                                    distance=SAMPLING_FREQUENCY*0.4, 
                                    prominence=50)
        peak_count = len(window_peaks)
        window_peak_rate = peak_count / WINDOW_SIZE * 60
        peak_reduction = ((baseline_peak_rate - window_peak_rate) / baseline_peak_rate) * 100
        
        # Calculate signal irregularity
        if len(window_peaks) > 2:
            peak_intervals = np.diff(window_peaks)
            irregularity = np.std(peak_intervals) / np.mean(peak_intervals) if np.mean(peak_intervals) > 0 else 0
        else:
            irregularity = 999  # High value for missing/very few peaks
        
        # Apnea detection criteria:
        # Strong: amplitude reduction > 50% OR (amplitude > 40% AND few peaks)
        is_apnea = (amplitude_reduction > AMPLITUDE_THRESHOLD) or \
                (amplitude_reduction > 40 and (peak_count < MIN_PEAK_COUNT or irregularity > IRREGULARITY_THRESHOLD))
        
        results.append({
            'start_time': window_time_start,
            'end_time': window_time_end,
            'amplitude_reduction': amplitude_reduction,
            'peak_reduction': peak_reduction,
            'peak_count': peak_count,
            'irregularity': irregularity if irregularity < 999 else 'high',
            'is_apnea': is_apnea
        })

    results_df = pd.DataFrame(results)

    # Group consecutive apnea windows into events
    apnea_windows = results_df[results_df['is_apnea'] == True]
    print(f"Detected {len(apnea_windows)} apnea windows\n")

    # Merge consecutive windows into apnea events
    def merge_consecutive_windows(windows_df, max_gap=5):
        """Merge consecutive apnea windows into distinct events"""
        if len(windows_df) == 0:
            return []
        
        events = []
        current_start = windows_df.iloc[0]['start_time']
        current_end = windows_df.iloc[0]['end_time']
        
        for i in range(1, len(windows_df)):
            gap = windows_df.iloc[i]['start_time'] - current_end
            
            if gap <= max_gap:  # Continue current event
                current_end = windows_df.iloc[i]['end_time']
            else:  # Start new event
                events.append((current_start, current_end))
                current_start = windows_df.iloc[i]['start_time']
                current_end = windows_df.iloc[i]['end_time']
        
        # Add last event
        events.append((current_start, current_end))
        return events

    apnea_events = merge_consecutive_windows(apnea_windows)

    # Print results
    print("=" * 70)
    print("APNEA DETECTION RESULTS (IMPROVED METHOD)")
    print("=" * 70)
    print(f"\nTotal apnea events detected: {len(apnea_events)}\n")

    for i, (start, end) in enumerate(apnea_events, 1):
        duration = end - start
        print(f"Event {i:2d}: {start:6.1f}s - {end:6.1f}s (duration: {duration:5.1f}s)")

    # Calculate and display statistics
    total_apnea_time = sum([end - start for start, end in apnea_events])
    apnea_index = len(apnea_events) / (len(ppg) / SAMPLING_FREQUENCY / 3600)  # events per hour

    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    print(f"Total recording time: {len(ppg)/SAMPLING_FREQUENCY:.1f} seconds")
    print(f"Total apnea time: {total_apnea_time:.1f} seconds ({total_apnea_time/(len(ppg)/SAMPLING_FREQUENCY)*100:.1f}%)")
    print(f"Average event duration: {total_apnea_time/len(apnea_events):.1f} seconds")
    print(f"Apnea-Hypopnea Index (AHI): {apnea_index:.1f} events/hour")

    # Save detailed results to CSV
    results_df.to_csv('apnea_detection_detailed.csv', index=False)
    print(f"\nDetailed window analysis saved to: apnea_detection_detailed.csv")

    # Save event summary
    events_list = []
    for i, (start, end) in enumerate(apnea_events, 1):
        events_list.append({
            'event_number': i,
            'start_time_sec': round(start, 1),
            'end_time_sec': round(end, 1),
            'duration_sec': round(end - start, 1)
        })

    events_df = pd.DataFrame(events_list)
    events_df.to_csv(f'{FOLDER_PATH}{id}_apnea_events_improved.csv', index=False)
    print(f"Event summary saved to: {FOLDER_PATH}{id}_apnea_events_improved.csv\n")
