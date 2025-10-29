#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Apnea full pipeline:
- Ground truth loader from dataset_summary.csv (long or compact formats)
- Dataset builder from *_delta_hr_vals.csv, *_delta_lfhf_vals.csv, *_edr_vals.csv
- Methods:
  * LF/HF: sustained elevation (threshold, min_duration_s)
  * DHR/EDR: single-peak (threshold)
  * DHR/EDR: min-peaks (threshold, min_peaks in window_s)
- Event matching with displacement (max_time_s - GT_right_border)
- Grid search optimization across subjects
- Per-subject and aggregate evaluation
- CSV outputs:
  * results_aggregate.csv
  * results_per_subject.csv
"""

import re, ast, glob
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# ==========================
# Configuration (edit here)
# ==========================
folder = 'data/dataset5/'
FILE_PATTERNS = {
    'DHR': f'{folder}*_delta_hr_vals.csv',
    'LF/HF': f'{folder}*_delta_lfhf_vals.csv',
    'EDR': f'{folder}*_edr_vals.csv'
}

GT_CSV_PATH = f'{folder}gt_summary.csv'

# Sampling: the *_vals.csv series are assumed at 1 Hz (one value per second).
# If different, change SAMPLE_MS below or adapt per-parameter.
SAMPLE_MS = 1000  # 1 second per sample

# Grid search defaults
GRIDS = {
    'LF/HF': {
        'threshold_percentiles': (40, 90, 15),   # start, end, n_points
        'durations_s': [10, 15, 20, 30]
    },
    'DHR_single': {
        'threshold_percentiles': (50, 99, 20)
    },
    'EDR_single': {
        'threshold_percentiles': (50, 99, 20)
    },
    'DHR_minpeaks': {
        'threshold_percentiles': (60, 95, 12),
        'window_s': [15, 20, 30],
        'min_peaks': [2, 3, 4]
    },
    'EDR_minpeaks': {
        'threshold_percentiles': (60, 95, 12),
        'window_s': [20, 30, 40],
        'min_peaks': [2, 3]
    }
}

# ==========================
# Ground-truth from CSV
# ==========================
def load_ground_truth_from_csv(csv_path: str) -> dict:
    """
    Returns:
        gt dict: {'subject_id': [(start_ms, end_ms), ...], ...}

    Supported formats:

    A) Long (one row per interval):
       subject_id,start_ms,end_ms
       5,60000,90000
       5,150000,180000
       Aliases:
         subject_id: subject_id|subject|file_id|id
         start: start_ms|start|begin_ms|t_start_ms|start_s
         end:   end_ms|end|stop_ms|t_end_ms|end_s
       If in seconds (_s or values <1e4), convert to ms.

    B) Compact (one row per subject with list of tuples as string):
       subject_id,apnea_intervals_ms
       5,"[(60000, 90000), (150000, 180000), ...]"
       19,"[(60000, 90000), ...]"
       Aliases: apnea_intervals_ms|intervals_ms|gt_intervals_ms|apnea_intervals
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    def find_col(aliases):
        for a in aliases:
            if a in df.columns:
                return a
        return None

    subj_col = find_col(['subject_id','subject','file_id','id'])
    list_col = find_col(['apnea_intervals_ms','intervals_ms','gt_intervals_ms','apnea_intervals'])
    gt = {}

    # Compact
    if subj_col and list_col:
        for _, row in df.iterrows():
            sid = str(row[subj_col]).strip()
            raw = row[list_col]
            intervals = ast.literal_eval(raw) if isinstance(raw, str) else raw
            fixed = []
            for s,e in intervals:
                if s < 1e4 and e < 1e4:  # likely seconds
                    s, e = int(round(s*1000)), int(round(e*1000))
                else:
                    s, e = int(s), int(e)
                if e > s:
                    fixed.append((s,e))
            gt[sid] = fixed
        return gt

    # Long
    start_col = find_col(['start_ms','start','begin_ms','t_start_ms','start_s'])
    end_col   = find_col(['end_ms','end','stop_ms','t_end_ms','end_s'])
    if not (subj_col and start_col and end_col):
        raise ValueError("gt_summary.csv: colonne GT non riconosciute.")

    def to_ms(v, name):
        v = float(v)
        if name.endswith('_s') or v < 1e4:
            return int(round(v*1000))
        return int(round(v))

    for sid, grp in df.groupby(subj_col):
        sid = str(sid).strip()
        ints = []
        for _, r in grp.iterrows():
            s = to_ms(r[start_col], start_col)
            e = to_ms(r[end_col], end_col)
            if e > s:
                ints.append((s,e))
        ints.sort()
        # merge overlaps
        merged = []
        for s,e in ints:
            if not merged or s > merged[-1][1]:
                merged.append([s,e])
            else:
                merged[-1][1] = max(merged[-1][1], e)
        gt[sid] = [(s,e) for s,e in merged]
    return gt

# ==========================
# Dataset builder
# ==========================
def extract_subject_id(path: str) -> str:
    stem = Path(path).stem
    token = stem.split('_')[0]
    if not re.match(r'^[0-9]+$', token):
        m = re.search(r'([0-9]+)', stem)
        token = m.group(1) if m else stem
    return token

def build_dataset(file_patterns: dict, gt_csv: str) -> dict:
    gt = load_ground_truth_from_csv(gt_csv)
    ds = {k:{} for k in file_patterns.keys()}
    for param, patt in file_patterns.items():
        for fp in sorted(glob.glob(patt)):
            sid = extract_subject_id(fp)
            if sid not in gt:
                print(f"[warn] no GT for subject {sid}; skip {fp}")
                continue
            vals = pd.read_csv(fp).iloc[:,0].astype(float).values
            t_ms = (np.arange(len(vals))*SAMPLE_MS).astype(int)
            ds[param][sid] = dict(values=vals, time_ms=t_ms, apnea_intervals_ms=gt[sid])
            print(f"Loaded {param} | subject={sid} | n={len(vals)} | GT={len(gt[sid])}")
    return ds

# ==========================
# Detection primitives
# ==========================
@dataclass
class Event:
    start_s: float
    end_s: float
    max_time_s: float
    n_peaks: int = 1

def detect_peaks(values: np.ndarray, time_ms: np.ndarray, threshold: float) -> list:
    """Single-peak detector: each peak over threshold -> event at that instant."""
    t = time_ms/1000.0
    peaks, _ = find_peaks(values, height=threshold)
    return [Event(start_s=float(t[i]), end_s=float(t[i]), max_time_s=float(t[i]), n_peaks=1) for i in peaks]

def detect_min_peaks(values: np.ndarray, time_ms: np.ndarray, threshold: float,
                     window_s: int = 30, min_peaks: int = 2) -> list:
    t = time_ms/1000.0
    peaks, _ = find_peaks(values, height=threshold)
    if len(peaks)==0: return []
    pk_t = t[peaks]
    used = np.zeros(len(pk_t), dtype=bool)
    events = []
    i = 0
    while i < len(pk_t):
        if used[i]: i += 1; continue
        start = pk_t[i]; end = start + window_s
        idx = np.where((pk_t >= start) & (pk_t < end) & (~used))[0]
        if len(idx) >= min_peaks:
            used[idx] = True
            events.append(Event(start_s=float(start),
                                end_s=float(pk_t[idx[-1]]),
                                max_time_s=float(start),
                                n_peaks=int(len(idx))))
        i += 1
    return events

def detect_sustained(values: np.ndarray, time_ms: np.ndarray, threshold: float,
                     min_duration_s: int = 10) -> list:
    """LF/HF sustained elevation: contiguous run above threshold for >= min_duration_s."""
    t = time_ms/1000.0
    above = values >= threshold
    events = []
    i = 0
    while i < len(above):
        if not above[i]: i += 1; continue
        s = i
        while i < len(above) and above[i]:
            i += 1
        e = i - 1
        dur = t[e] - t[s]
        if dur >= min_duration_s:
            seg = values[s:i]
            max_idx = s + int(np.argmax(seg))
            events.append(Event(start_s=float(t[s]), end_s=float(t[e]), max_time_s=float(t[max_idx])))
    return events

# ==========================
# Matching & metrics
# ==========================
def match_events_to_gt(events: list, gt_ms: list, method: str,
                       tolerance_s: int = 30) -> dict:
    """
    Returns metrics and displacement statistics.
      - For sustained: match on overlap.
      - For peaks/min-peaks: match if event max_time_s within [gt_start - tol, gt_end + tol].
    Displacement for matched events: max_time_s - gt_end (right border).
    """
    gt_s = [(s/1000.0, e/1000.0) for s,e in gt_ms]
    gt_hit = [False]*len(gt_s)
    displacements = []

    if method == 'sustained':
        for ev in events:
            matched_idx = None
            for k,(gs,ge) in enumerate(gt_s):
                if not (ev.end_s < gs or ev.start_s > ge):
                    matched_idx = k
                    break
            if matched_idx is not None:
                gt_hit[matched_idx] = True
                _, ge = gt_s[matched_idx]
                displacements.append(ev.max_time_s - ge)
    else:
        for ev in events:
            matched_idx = None
            best_dist = float('inf')
            for k,(gs,ge) in enumerate(gt_s):
                if ev.max_time_s >= (gs - tolerance_s) and ev.max_time_s <= (ge + tolerance_s):
                    dist = min(abs(ev.max_time_s - gs), abs(ev.max_time_s - ge))
                    if dist < best_dist:
                        best_dist = dist
                        matched_idx = k
            if matched_idx is not None:
                gt_hit[matched_idx] = True
                _, ge = gt_s[matched_idx]
                displacements.append(ev.max_time_s - ge)

    tp = sum(gt_hit)
    fn = len(gt_s) - tp
    fp = max(0, len(events) - tp)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0.0

    disp_mean = float(np.mean(displacements)) if len(displacements)>0 else np.nan
    disp_std  = float(np.std(displacements))  if len(displacements)>0 else np.nan

    return dict(tp=tp, fp=fp, fn=fn,
                precision=precision, recall=recall, f1=f1,
                n_events=len(events),
                disp_mean_s=disp_mean, disp_std_s=disp_std)

# ==========================
# Evaluation wrappers
# ==========================
def evaluate_subject(values: np.ndarray, time_ms: np.ndarray, gt_ms: list,
                     method_name: str, **kwargs) -> tuple:
    """
    method_name in {'LF/HF_sustained', 'peak', 'minpeaks'}
    kwargs for:
      - sustained: threshold, min_duration_s
      - peak: threshold
      - minpeaks: threshold, window_s, min_peaks
    Returns (events, metrics_dict)
    """
    if method_name == 'LF/HF_sustained':
        events = detect_sustained(values, time_ms, kwargs['threshold'], kwargs['min_duration_s'])
        metrics = match_events_to_gt(events, gt_ms, method='sustained')
    elif method_name == 'peak':
        events = detect_peaks(values, time_ms, kwargs['threshold'])
        metrics = match_events_to_gt(events, gt_ms, method='peak')
    elif method_name == 'minpeaks':
        events = detect_min_peaks(values, time_ms, kwargs['threshold'], kwargs['window_s'], kwargs['min_peaks'])
        metrics = match_events_to_gt(events, gt_ms, method='minpeaks')
    else:
        raise ValueError("Unknown method_name")
    return events, metrics

# ==========================
# Grid search optimizers
# ==========================
def percentile_grid(data: np.ndarray, p0: int, p1: int, n: int) -> np.ndarray:
    lo = float(np.percentile(data, p0))
    hi = float(np.percentile(data, p1))
    if hi <= lo:
        hi = lo + 1e-6
    return np.linspace(lo, hi, n)

def optimize_across_dataset(dataset: dict, param_name: str, method_name: str, grids: dict) -> dict:
    """
    Optimize parameters for method on given param across all subjects.
    Returns best configuration and aggregate metrics.
    """
    if param_name not in dataset or len(dataset[param_name]) == 0:
        return {}

    # Gather values for threshold grid
    all_vals = np.concatenate([d['values'] for d in dataset[param_name].values()])

    # Build search space
    if method_name == 'LF/HF_sustained':
        ps = grids['threshold_percentiles']
        thr_grid = percentile_grid(all_vals, ps[0], ps[1], ps[2])
        dur_grid = grids['durations_s']
        search = [{'threshold': thr, 'min_duration_s': dur} for thr in thr_grid for dur in dur_grid]
    elif method_name == 'peak':
        ps = grids['threshold_percentiles']
        thr_grid = percentile_grid(all_vals, ps[0], ps[1], ps[2])
        search = [{'threshold': thr} for thr in thr_grid]
    elif method_name == 'minpeaks':
        ps = grids['threshold_percentiles']
        thr_grid = percentile_grid(all_vals, ps[0], ps[1], ps[2])
        search = [{'threshold': thr, 'window_s': w, 'min_peaks': m}
                  for thr in thr_grid for w in grids['window_s'] for m in grids['min_peaks']]
    else:
        raise ValueError("Unknown method_name")

    best = None
    for cfg in search:
        agg_tp=agg_fp=agg_fn=0
        disp_all = []
        for sid, d in dataset[param_name].items():
            _, m = evaluate_subject(d['values'], d['time_ms'], d['apnea_intervals_ms'], method_name, **cfg)
            agg_tp += m['tp']; agg_fp += m['fp']; agg_fn += m['fn']
            if not np.isnan(m['disp_mean_s']):
                disp_all.append((m['disp_mean_s'], m['disp_std_s']))
        precision = agg_tp/(agg_tp+agg_fp) if (agg_tp+agg_fp)>0 else 0
        recall    = agg_tp/(agg_tp+agg_fn) if (agg_tp+agg_fn)>0 else 0
        f1        = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0

        disp_mean = np.nan
        disp_std  = np.nan
        if len(disp_all)>0:
            disp_mean = float(np.mean([x[0] for x in disp_all]))
            disp_std  = float(np.mean([x[1] for x in disp_all]))

        candidate = dict(Parameter=param_name, Method=method_name, Precision=precision, Recall=recall, F1=f1,
                         TP=agg_tp, FP=agg_fp, FN=agg_fn, DispMean_s=disp_mean, DispStd_s=disp_std, **cfg)
        if (best is None) or (f1 > best['F1']):
            best = candidate

    return best if best is not None else {}

# ==========================
# Per-subject evaluation
# ==========================
def evaluate_per_subject(dataset: dict, best_cfgs: list) -> pd.DataFrame:
    """
    For each best configuration (per param+method), evaluate per subject and return a detailed DataFrame.
    """
    rows = []
    for best in best_cfgs:
        if not best:
            continue
        pname = best['Parameter']; mname = best['Method']
        cfg_keys = set(best.keys()) - set(['Parameter','Method','Precision','Recall','F1','TP','FP','FN','DispMean_s','DispStd_s'])
        cfg = {k: best[k] for k in cfg_keys}
        for sid, d in dataset[pname].items():
            _, m = evaluate_subject(d['values'], d['time_ms'], d['apnea_intervals_ms'], mname, **cfg)
            rows.append(dict(Subject=sid, Parameter=pname, Method=mname,
                             Precision=m['precision'], Recall=m['recall'], F1=m['f1'],
                             TP=m['tp'], FP=m['fp'], FN=m['fn'],
                             N_Events=m['n_events'],
                             DispMean_s=m['disp_mean_s'], DispStd_s=m['disp_std_s'],
                             **cfg))
    return pd.DataFrame(rows)

# ==========================
# Main pipeline
# ==========================
def run_pipeline():
    # Build dataset from files and GT CSV
    dataset = build_dataset(FILE_PATTERNS, GT_CSV_PATH)

    best_cfgs = []

    # LF/HF sustained
    if len(dataset.get('LF/HF', {}))>0:
        best_lfhf = optimize_across_dataset(dataset, 'LF/HF', 'LF/HF_sustained', GRIDS['LF/HF'])
        print(f"[LF/HF] Best: {best_lfhf}")
        best_cfgs.append(best_lfhf)

    # DHR: single peak and min-peaks
    if len(dataset.get('DHR', {}))>0:
        best_dhr_single = optimize_across_dataset(dataset, 'DHR', 'peak', GRIDS['DHR_single'])
        print(f"[DHR single-peak] Best: {best_dhr_single}")
        best_cfgs.append(best_dhr_single)

        best_dhr_minp = optimize_across_dataset(dataset, 'DHR', 'minpeaks', GRIDS['DHR_minpeaks'])
        print(f"[DHR min-peaks] Best: {best_dhr_minp}")
        best_cfgs.append(best_dhr_minp)

    # EDR: single peak and min-peaks
    if len(dataset.get('EDR', {}))>0:
        best_edr_single = optimize_across_dataset(dataset, 'EDR', 'peak', GRIDS['EDR_single'])
        print(f"[EDR single-peak] Best: {best_edr_single}")
        best_cfgs.append(best_edr_single)

        best_edr_minp = optimize_across_dataset(dataset, 'EDR', 'minpeaks', GRIDS['EDR_minpeaks'])
        print(f"[EDR min-peaks] Best: {best_edr_minp}")
        best_cfgs.append(best_edr_minp)

    # Aggregate results
    agg = pd.DataFrame([b for b in best_cfgs if b])
    agg.to_csv('results_aggregate.csv', index=False)
    print("✓ Saved: results_aggregate.csv")

    # Per-subject results with displacement stats
    detailed = evaluate_per_subject(dataset, [b for b in best_cfgs if b])
    detailed.sort_values(['Parameter','Method','Subject'], inplace=True)
    detailed.to_csv('results_per_subject.csv', index=False)
    print("✓ Saved: results_per_subject.csv")

    # Console preview
    print("\nAGGREGATE BEST CONFIGS:")
    if not agg.empty:
        print(agg.to_string(index=False))
    print("\nPER-SUBJECT (head):")
    if not detailed.empty:
        print(detailed.head(12).to_string(index=False))

if __name__ == '__main__':
    run_pipeline()
