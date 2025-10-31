import pandas as pd
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt

v2 = [1,2]
FOLDER_PATH= 'data/ppg/'

for id in v2:
    # Configuration
    FILENAME = f'{FOLDER_PATH}{id}.csv'
    SAMPLING_FREQUENCY = 50  # Hz
    APNEA_RR_THRESHOLD = 2.0  # seconds - RR intervals longer than this suggest apnea
    MAX_EVENT_GAP = 10  # seconds - max gap to group RR intervals into same event

    # Load data
    df = pd.read_csv(FILENAME)
    ppg = df['ppg_wave'].values
    bcg = df['bcg_raw'].values
    time = np.arange(len(ppg)) / SAMPLING_FREQUENCY

    print(f"Data loaded: {len(ppg)} samples at {SAMPLING_FREQUENCY} Hz")
    print(f"Recording duration: {len(ppg)/SAMPLING_FREQUENCY:.1f} seconds")

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
    bcg_filtered = butter_bandpass_filter(bcg, 0.5, 10, SAMPLING_FREQUENCY)
    ppg_filtered = butter_bandpass_filter(ppg, 0.5, 8, SAMPLING_FREQUENCY)

    print("\nSignals filtered:")
    print(f"  BCG: 0.5-10 Hz bandpass")
    print(f"  PPG: 0.5-8 Hz bandpass")

    # Detect peaks for heart rate calculation
    peaks_bcg, _ = find_peaks(bcg_filtered, distance=SAMPLING_FREQUENCY*0.4, prominence=50)
    peaks_ppg, _ = find_peaks(ppg_filtered, distance=SAMPLING_FREQUENCY*0.4, prominence=5)

    print(f"\nPeaks detected:")
    print(f"  BCG: {len(peaks_bcg)} peaks")
    print(f"  PPG: {len(peaks_ppg)} peaks")

    # Calculate RR intervals and heart rate
    if len(peaks_bcg) > 1:
        rr_intervals_bcg = np.diff(peaks_bcg) / SAMPLING_FREQUENCY  # in seconds
        heart_rate_bcg = 60 / rr_intervals_bcg  # in bpm
        time_hr_bcg = time[peaks_bcg[1:]]
        print(f"  BCG mean HR: {heart_rate_bcg.mean():.1f} bpm")
    else:
        rr_intervals_bcg = np.array([])
        heart_rate_bcg = np.array([])
        time_hr_bcg = np.array([])

    if len(peaks_ppg) > 1:
        rr_intervals_ppg = np.diff(peaks_ppg) / SAMPLING_FREQUENCY
        heart_rate_ppg = 60 / rr_intervals_ppg
        time_hr_ppg = time[peaks_ppg[1:]]
        print(f"  PPG mean HR: {heart_rate_ppg.mean():.1f} bpm")
    else:
        rr_intervals_ppg = np.array([])
        heart_rate_ppg = np.array([])
        time_hr_ppg = np.array([])

    # Detect long RR intervals (potential apnea indicators)
    long_rr_bcg = np.where(rr_intervals_bcg > APNEA_RR_THRESHOLD)[0]
    long_rr_ppg = np.where(rr_intervals_ppg > APNEA_RR_THRESHOLD)[0]

    print(f"\nLong RR intervals detected (>{APNEA_RR_THRESHOLD}s):")
    print(f"  BCG: {len(long_rr_bcg)} intervals")
    print(f"  PPG: {len(long_rr_ppg)} intervals")

    # Group consecutive long RR intervals into apnea events
    def group_events(indices, time_points, intervals, max_gap=10):
        """
        Group consecutive long RR intervals into distinct apnea events.
        
        Parameters:
        - indices: array of indices where long RR intervals occur
        - time_points: time values corresponding to each RR interval
        - intervals: RR interval durations
        - max_gap: maximum time gap (seconds) to consider intervals part of same event
        
        Returns:
        - List of tuples (start_time, end_time) for each apnea event
        """
        if len(indices) == 0:
            return []
        
        events = []
        current_event_start = time_points[indices[0]]
        current_event_end = time_points[indices[0]] + intervals[indices[0]]
        
        for i in range(1, len(indices)):
            time_gap = time_points[indices[i]] - current_event_end
            
            if time_gap < max_gap:  # Continue current event
                current_event_end = time_points[indices[i]] + intervals[indices[i]]
            else:  # Start new event
                events.append((current_event_start, current_event_end))
                current_event_start = time_points[indices[i]]
                current_event_end = time_points[indices[i]] + intervals[indices[i]]
        
        # Add last event
        events.append((current_event_start, current_event_end))
        return events

    # Detect apnea events
    bcg_apnea_events = group_events(long_rr_bcg, time_hr_bcg, rr_intervals_bcg, MAX_EVENT_GAP)
    ppg_apnea_events = group_events(long_rr_ppg, time_hr_ppg, rr_intervals_ppg, MAX_EVENT_GAP)

    # Print results
    print(f"\n{'='*60}")
    print(f"APNEA DETECTION RESULTS")
    print(f"{'='*60}")

    print(f"\nBCG-detected apnea events: {len(bcg_apnea_events)}")
    for i, (start, end) in enumerate(bcg_apnea_events, 1):
        duration = end - start
        print(f"  Event {i:2d}: {start:6.1f}s - {end:6.1f}s (duration: {duration:5.1f}s)")

    print(f"\nPPG-detected apnea events: {len(ppg_apnea_events)}")
    for i, (start, end) in enumerate(ppg_apnea_events, 1):
        duration = end - start
        print(f"  Event {i:2d}: {start:6.1f}s - {end:6.1f}s (duration: {duration:5.1f}s)")

    # Save results to CSV
    results = []
    for i, (start, end) in enumerate(bcg_apnea_events, 1):
        results.append({
            'event_number': i,
            'signal': 'BCG',
            'start_time_sec': round(start, 1),
            'end_time_sec': round(end, 1),
            'duration_sec': round(end - start, 1)
        })

    for i, (start, end) in enumerate(ppg_apnea_events, 1):
        results.append({
            'event_number': i,
            'signal': 'PPG',
            'start_time_sec': round(start, 1),
            'end_time_sec': round(end, 1),
            'duration_sec': round(end - start, 1)
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{FOLDER_PATH}{id}_apnea_events_detected.csv', index=False)
    print(f"\nResults saved to: {FOLDER_PATH}{id}_apnea_events_detected.csv")
