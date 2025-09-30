#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COLET JSON → Random Forest (LOW/MEDIUM/HIGH)

Tailored for files like:
  participant_01_01_gaze_data.json
  participant_01_02_gaze_data.json
  ...
Each JSON contains a list of dicts with keys: t, x, y, pupil_size
- t: timestamp (seconds)
- x, y: normalized gaze coordinates in [0,1]
- pupil_size: pupil diameter (arbitrary units)

Labels CSV (compiled_rtlx_workload.csv) with columns:
  participant_id (e.g., P01), activity_id (1..4), mean, cognitive_workload (LOW/MEDIUM/HIGH)

Outputs:
- features.json.csv           (one row per JSON file)
- features_with_labels.csv    (joined with labels)
- model_report.txt            (classification report)
- model.joblib                (persisted best RandomForest)

Usage:
  python colet_json_rf.py --json_dir /path/to/jsons --labels_csv /path/to/compiled_rtlx_workload.csv --workdir ./out --train
"""
import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, cohen_kappa_score
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler


# Display assumptions for pixel geometry to convert degrees:
RES_W = 1280
RES_H = 720
SCREEN_WIDTH_CM = 53.1
SCREEN_HEIGHT_CM = 29.9
VIEW_DIST_CM = 80.0
IVT_VEL_THRESH_DEG_S = 45.0
MIN_FIX_DUR_S = 0.055

def pixels_to_degrees(dx_pix: np.ndarray, dy_pix: np.ndarray) -> np.ndarray:
    cm_per_px_x = SCREEN_WIDTH_CM / RES_W
    cm_per_px_y = SCREEN_HEIGHT_CM / RES_H
    dx_cm = dx_pix * cm_per_px_x
    dy_cm = dy_pix * cm_per_px_y
    disp_cm = np.sqrt(dx_cm**2 + dy_cm**2)
    angle_rad = 2.0 * np.arctan2(disp_cm / 2.0, VIEW_DIST_CM)
    return np.degrees(angle_rad)

def ivt_segmentation(t_s: np.ndarray, gx_px: np.ndarray, gy_px: np.ndarray):
    order = np.argsort(t_s)
    t = t_s[order]
    x = gx_px[order]
    y = gy_px[order]
    if t.size < 3:
        return [], []
    dt = np.diff(t)
    dt[dt == 0] = 1e-6
    dx = np.diff(x)
    dy = np.diff(y)
    ddeg = pixels_to_degrees(dx, dy)
    vel = ddeg / dt
    is_fix = vel < IVT_VEL_THRESH_DEG_S
    fix_intervals, sac_intervals = [], []
    def push_interval(si, ei, is_fixation):
        t_start = t[si]
        t_end = t[ei]
        dur = t_end - t_start
        if is_fixation and dur >= MIN_FIX_DUR_S:
            fix_intervals.append((t_start, t_end))
        elif not is_fixation:
            sac_intervals.append((t_start, t_end))
    curr = is_fix[0]
    start = 0
    for i in range(1, len(is_fix)):
        if is_fix[i] != curr:
            push_interval(start, i, curr)
            curr = is_fix[i]
            start = i
    push_interval(start, len(is_fix) - 1, curr)
    return fix_intervals, sac_intervals

def interval_stats(intervals: List[Tuple[float,float]]) -> Dict[str, float]:
    if not intervals:
        return dict(median=np.nan, mean=np.nan, variation=np.nan, skew=np.nan, kurtosis=np.nan, count=0)
    d = np.array([e-s for s,e in intervals], dtype=float)
    mean = d.mean()
    std = d.std(ddof=1) if d.size > 1 else 0.0
    return dict(
        median=float(np.median(d)),
        mean=float(mean),
        variation=float(std/mean) if mean>0 else np.nan,
        skew=float(skew(d)) if d.size>2 else np.nan,
        kurtosis=float(kurtosis(d)) if d.size>3 else np.nan,
        count=int(d.size),
    )

def velocity_stats(t_s: np.ndarray, gx_px: np.ndarray, gy_px: np.ndarray) -> Dict[str, float]:
    order = np.argsort(t_s)
    t = t_s[order]
    x = gx_px[order]
    y = gy_px[order]
    if t.size < 3:
        return {k: np.nan for k in [
            "saccade_velocity_mean","saccade_velocity_variation","saccade_velocity_skew","saccade_velocity_kurtosis",
            "peak_saccade_velocity_mean","peak_saccade_velocity_variation","peak_saccade_velocity_skew","peak_saccade_velocity_kurtosis",
            "saccade_amplitude_median_deg"
        ]}
    dt = np.diff(t); dt[dt==0]=1e-6
    dx = np.diff(x); dy = np.diff(y)
    ddeg = pixels_to_degrees(dx, dy)
    vel = ddeg / dt
    # peak proxy
    peaks = []
    win=5
    for i in range(win, len(vel)-win):
        if vel[i] == np.max(vel[i-win:i+win+1]):
            peaks.append(vel[i])
    peaks = np.array(peaks) if peaks else np.array([])
    def stats(v):
        if v.size==0:
            return np.nan, np.nan, np.nan, np.nan
        mean = float(np.mean(v))
        std = float(np.std(v, ddof=1)) if v.size>1 else 0.0
        return mean, (std/mean if mean>0 else np.nan), (float(skew(v)) if v.size>2 else np.nan), (float(kurtosis(v)) if v.size>3 else np.nan)
    v_mean, v_var, v_sk, v_ku = stats(vel)
    p_mean, p_var, p_sk, p_ku = stats(peaks)
    amp_med = float(np.median(ddeg)) if ddeg.size else np.nan
    return dict(
        saccade_velocity_mean=v_mean, saccade_velocity_variation=v_var, saccade_velocity_skew=v_sk, saccade_velocity_kurtosis=v_ku,
        peak_saccade_velocity_mean=p_mean, peak_saccade_velocity_variation=p_var, peak_saccade_velocity_skew=p_sk, peak_saccade_velocity_kurtosis=p_ku,
        saccade_amplitude_median_deg=amp_med
    )

def pupil_features(p: np.ndarray) -> Dict[str, float]:
    if p.size==0:
        return dict(pupil_mean=np.nan, pupil_variation=np.nan, pupil_skew=np.nan, pupil_kurtosis=np.nan)
    mean = float(np.mean(p))
    std = float(np.std(p, ddof=1)) if p.size>1 else 0.0
    return dict(
        pupil_mean=mean,
        pupil_variation=(std/mean if mean>0 else np.nan),
        pupil_skew=(float(skew(p)) if p.size>2 else np.nan),
        pupil_kurtosis=(float(kurtosis(p)) if p.size>3 else np.nan),
    )

def parse_ids_from_name(path: Path) -> Tuple[str,int]:
    # participant_01_03_gaze_data.json -> P01, 3
    m = re.search(r'participant[_-]?(\d+)[_-]+(\d+)', path.stem, flags=re.IGNORECASE)
    if m:
        pid = f"P{int(m.group(1)):02d}"
        aid = int(m.group(2))
        return pid, aid
    # fallback
    return "P00", -1

def extract_from_json(path: Path) -> Dict[str, float]:
    data = json.loads(path.read_text())
    # extract with fallbacks and tolerate missing records
    t_list, x_list, y_list, p_list = [], [], [], []
    for row in data:
        # accept 't' or 'timestamp'
        t_val = row.get("t", row.get("timestamp", None))
        x_val = row.get("x", None)
        y_val = row.get("y", None)
        p_val = row.get("pupil_size", row.get("pupil", None))
        # keep only rows with essential fields t,x,y
        if t_val is not None and x_val is not None and y_val is not None:
            t_list.append(float(t_val))
            x_list.append(float(x_val))
            y_list.append(float(y_val))
            p_list.append(float(p_val) if p_val is not None else np.nan)

    t = np.array(t_list, dtype=float)
    x = np.array(x_list, dtype=float)
    y = np.array(y_list, dtype=float)
    p = np.array(p_list, dtype=float)

    # Normalize time to start at 0
    if t.size>0:
        t = t - t.min()

    # Convert normalized gaze to pixels
    gx = x * RES_W
    gy = y * RES_H

    fix_int, sac_int = ivt_segmentation(t, gx, gy)
    fix_stats = interval_stats(fix_int)
    sac_stats = interval_stats(sac_int)

    total_dur = float(t.max() - t.min()) if t.size>1 else np.nan
    fix_freq = (fix_stats["count"]/total_dur) if total_dur and total_dur>0 else 0.0
    sac_freq = (sac_stats["count"]/total_dur) if total_dur and total_dur>0 else 0.0

    kin = velocity_stats(t, gx, gy)
    pup = pupil_features(p[~np.isnan(p)])

    feats = dict(
        fixation_frequency_per_sec=float(fix_freq),
        fixation_duration_median_s=float(fix_stats["median"]),
        fixation_duration_variation=float(fix_stats["variation"]),
        saccade_frequency_per_sec=float(sac_freq),
        saccade_duration_median_s=float(sac_stats["median"]),
        saccade_duration_variation=float(sac_stats["variation"]),
        saccade_amplitude_median_deg=float(kin["saccade_amplitude_median_deg"]),
        saccade_velocity_mean_deg_s=float(kin["saccade_velocity_mean"]),
        saccade_velocity_variation=float(kin["saccade_velocity_variation"]),
        saccade_velocity_skew=float(kin["saccade_velocity_skew"]),
        saccade_velocity_kurtosis=float(kin["saccade_velocity_kurtosis"]),
        peak_saccade_velocity_mean_deg_s=float(kin["peak_saccade_velocity_mean"]),
        peak_saccade_velocity_variation=float(kin["peak_saccade_velocity_variation"]),
        peak_saccade_velocity_skew=float(kin["peak_saccade_velocity_skew"]),
        peak_saccade_velocity_kurtosis=float(kin["peak_saccade_velocity_kurtosis"]),
        pupil_mean=float(pup["pupil_mean"]),
        pupil_variation=float(pup["pupil_variation"]),
        pupil_skew=float(pup["pupil_skew"]),
        pupil_kurtosis=float(pup["pupil_kurtosis"]),
        trial_duration_s=total_dur,
    )
    return feats

def derive_label_from_text(s: str) -> str:
    s = str(s).strip().upper()
    if s in {"LOW","MEDIUM","HIGH"}:
        return s
    return "UNKNOWN"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_dir", required=True, help="Directory that contains participant_*_*.json files")
    ap.add_argument("--labels_csv", required=True, help="compiled_rtlx_workload.csv")
    ap.add_argument("--workdir", default="./colet_out_json", help="Output dir")
    ap.add_argument("--train", action="store_true", help="Train RF")
    args = ap.parse_args()

    outdir = Path(args.workdir); outdir.mkdir(parents=True, exist_ok=True)

    # 1) Scan JSONs
    rows = []
    for p in sorted(Path(args.json_dir).glob("*.json")):
        pid, aid = parse_ids_from_name(p)
        feats = extract_from_json(p)
        feats.update(dict(participant_id=pid, activity_id=aid, source=str(p.name)))
        rows.append(feats)
    feat_df = pd.DataFrame(rows)
    feat_df.to_csv(outdir/"features.json.csv", index=False)

    # 2) Join labels
    labels = pd.read_csv(args.labels_csv)
    # normalize columns if present
    if "cognitive_workload" in labels.columns:
        labels["label"] = labels["cognitive_workload"].apply(derive_label_from_text)
    elif "label" in labels.columns:
        labels["label"] = labels["label"].apply(derive_label_from_text)
    else:
        raise SystemExit("Labels CSV must have 'cognitive_workload' or 'label'.")

    merged = feat_df.merge(labels[["participant_id","activity_id","label"]], on=["participant_id","activity_id"], how="inner")
    merged = merged[merged["label"].isin(["LOW","MEDIUM","HIGH"])].copy()
    merged.to_csv(outdir/"features_with_labels.csv", index=False)

    if not args.train:
        print("Feature extraction complete. Run with --train to train the RF model.")
        return

    # 3) Train RF
    X = merged.drop(columns=["participant_id","activity_id","label","source"])
    y = merged["label"]
    scaler = MinMaxScaler()
    Xs = scaler.fit_transform(X)

    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, stratify=y, random_state=42)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42, class_weight='balanced'),
        {
            "n_estimators": [200, 500, 800],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 4],
            "min_samples_leaf": [1, 2, 4],
        },
        scoring="f1_macro", cv=cv, n_jobs=1
    )
    grid.fit(Xtr, ytr)
    ypred = grid.predict(Xte)
    rep = classification_report(yte, ypred, digits=3, zero_division=0)
    cm = confusion_matrix(yte, ypred, labels=["LOW","MEDIUM","HIGH"])  # fixed label order


    acc = accuracy_score(yte, ypred)
    f1_macro = f1_score(yte, ypred, average="macro")
    kappa = cohen_kappa_score(yte, ypred)

    metrics_summary = (
        f"\nOverall metrics:\n"
        f"  Accuracy:   {acc:.3f}\n"
        f"  F1-macro:   {f1_macro:.3f}\n"
        f"  Cohen's κ:  {kappa:.3f}\n"
    )

    report_text = (
        f"Best params: {grid.best_params_}\n\n"
        f"{rep}\n\n"
        f"Confusion matrix (rows=true, cols=pred) [LOW,MEDIUM,HIGH]:\n{cm}\n"
        f"{metrics_summary}\n"
    )

    (outdir / "model_report.txt").write_text(report_text)
    dump(grid.best_estimator_, outdir / "model.joblib")
    print(report_text)


if __name__ == "__main__":
    main()
