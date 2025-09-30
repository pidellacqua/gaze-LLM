#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_cw_from_semantic.py
---------------------------
Estimate Cognitive Workload (LOW / MEDIUM / HIGH) for each row of an EMOA-style
semantic gaze dataset that includes:
- Total Fixation count
- Total Duration (ms)
- Question fixations/duration (columns like "Q fix", "Q dur")
- Answer option fixations/duration (columns like "A fix","A dur",... "K fix","K dur")
- Correctness flag (column like "Correctly Answerd" with values yes/no)

Usage:
    python compute_cw_from_semantic.py --input EMOASemanticData.csv --output EMOASemanticData_cw_estimates.csv

Notes:
- Robust z-scores are used (median/MAD), with a fallback to std if MAD=0.
- Composite score weights and thresholds can be tuned via CLI args if desired.
"""

import argparse
import re
import numpy as np
import pandas as pd


def robust_z(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    med = np.nanmedian(s)
    mad = np.nanmedian(np.abs(s - med))
    if np.isnan(mad) or mad == 0:
        std = np.nanstd(s)
        if std == 0 or np.isnan(std):
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - med) / std
    return 0.6745 * (s - med) / mad


def auto_detect_answer_columns(columns):
    """Return (answer_fix_cols, answer_dur_cols) lists from dataframe columns.
    Matches labels like 'A fix','B dur',... including 'K' if present.
    """
    answer_labels = list("ABCDEK")  # extend if needed
    fix_cols, dur_cols = [], []
    for label in answer_labels:
        fix_name = f"{label} fix"
        dur_name = f"{label} dur"
        if fix_name in columns:
            fix_cols.append(fix_name)
        if dur_name in columns:
            dur_cols.append(dur_name)
    return fix_cols, dur_cols


def ensure_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def classify_band(score: float, low_thr: float, high_thr: float) -> str:
    if score < low_thr:
        return "LOW"
    elif score < high_thr:
        return "MEDIUM"
    else:
        return "HIGH"


def build_explanation(row) -> str:
    parts = []
    total_fix = int(row["Total Fixation"]) if pd.notna(row["Total Fixation"]) else 0
    total_dur = int(row["Total Duration"]) if pd.notna(row["Total Duration"]) else 0
    q_dur = int(row["Q dur"]) if pd.notna(row["Q dur"]) else 0
    q_share = row.get("q_share", np.nan)
    ans_visited = int(row.get("answers_visited", 0)) if pd.notna(row.get("answers_visited", np.nan)) else 0
    ans_sum = int(row.get("answers_dur_sum", 0)) if pd.notna(row.get("answers_dur_sum", np.nan)) else 0
    ans_share = row.get("ans_share", np.nan)
    ans_max_share = row.get("ans_max_share", np.nan)
    is_incorrect = bool(row.get("is_incorrect", False))

    parts.append(f"{total_fix} fixations, total gaze duration {total_dur} ms.")
    if pd.notna(q_share):
        parts.append(f"Question read for {q_dur} ms ({q_share:.0%} of total).")
    else:
        parts.append("Question duration unavailable.")
    if pd.notna(ans_share):
        parts.append(f"Answers viewed: {ans_visited} options; time on answers {ans_sum} ms ({ans_share:.0%} of total).")
    else:
        parts.append("Answer viewing time unavailable.")
    if pd.notna(ans_max_share):
        parts.append(f"Attention concentration on a single option: {ans_max_share:.0%}.")
    if is_incorrect:
        parts.append("Incorrect response adds uncertainty/effort.")

    # Rationale summary
    band = row.get("Cognitive_Workload", "MEDIUM")
    if band == "HIGH":
        parts.append("Prolonged viewing, broader exploration and/or indecision patterns indicate overload.")
    elif band == "LOW":
        parts.append("Shorter viewing with focused option selection suggests light processing.")
    else:
        parts.append("Moderate viewing and exploration indicate engaged but manageable effort.")
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser(description="Estimate Cognitive Workload from EMOA semantic gaze data.")
    ap.add_argument("--input", required=True, help="Path to input CSV (e.g., EMOASemanticData.csv)")
    ap.add_argument("--output", required=True, help="Path to output CSV to write results")
    # Tuning options
    ap.add_argument("--low-threshold", type=float, default=-0.5, help="cw_score threshold below which label is LOW")
    ap.add_argument("--high-threshold", type=float, default=0.75, help="cw_score threshold above which label is HIGH")
    ap.add_argument("--incorrect-penalty", type=float, default=0.3, help="Additive penalty to cw_score for incorrect answers")
    ap.add_argument("--w-fix", type=float, default=0.25, help="Weight for z(Total Fixation)")
    ap.add_argument("--w-dur", type=float, default=0.25, help="Weight for z(Total Duration)")
    ap.add_argument("--w-ans-share", type=float, default=0.15, help="Weight for z(answers share of total duration)")
    ap.add_argument("--w-q-share", type=float, default=0.15, help="Weight for z(question share of total duration)")
    ap.add_argument("--w-spread", type=float, default=0.15, help="Weight for z(# of answers visited)")
    ap.add_argument("--w-conc-low", type=float, default=0.05, help="Weight for z(1 - max per-answer share)")
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    # Detect columns
    required = ["Total Fixation", "Total Duration", "Q fix", "Q dur", "Correctly Answerd"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    answer_fix_cols, answer_dur_cols = auto_detect_answer_columns(df.columns)

    # Ensure numeric
    numeric_cols = ["Total Fixation", "Total Duration", "Q fix", "Q dur"] + answer_fix_cols + answer_dur_cols
    df = ensure_numeric(df, numeric_cols)

    # Aggregate features
    df["answers_fix_sum"] = df[answer_fix_cols].sum(axis=1, skipna=True) if answer_fix_cols else 0
    df["answers_dur_sum"] = df[answer_dur_cols].sum(axis=1, skipna=True) if answer_dur_cols else 0

    # Shares
    df["q_share"] = np.where(df["Total Duration"] > 0, df["Q dur"] / df["Total Duration"], np.nan)
    df["ans_share"] = np.where(df["Total Duration"] > 0, df["answers_dur_sum"] / df["Total Duration"], np.nan)

    # Spread and concentration
    df["answers_visited"] = df[answer_fix_cols].gt(0).sum(axis=1) if answer_fix_cols else 0
    if answer_dur_cols:
        per_answer_shares = df[answer_dur_cols].div(df["answers_dur_sum"].replace(0, np.nan), axis=0)
        df["ans_max_share"] = per_answer_shares.max(axis=1)
    else:
        df["ans_max_share"] = np.nan

    # Correctness flag
    df["is_incorrect"] = df["Correctly Answerd"].astype(str).str.strip().str.lower().isin(["no", "false", "0"])

    # Robust z-scores
    df["z_fix"] = robust_z(df["Total Fixation"])
    df["z_dur"] = robust_z(df["Total Duration"])
    df["z_ans_share"] = robust_z(df["ans_share"])
    df["z_q_share"] = robust_z(df["q_share"])
    df["z_spread"] = robust_z(df["answers_visited"])
    df["z_concentration_low"] = robust_z(1 - df["ans_max_share"])  # lower concentration -> higher indecision

    # Composite score
    df["cw_score_raw"] = (
        args.w_fix * df["z_fix"] +
        args.w_dur * df["z_dur"] +
        args.w_ans_share * df["z_ans_share"] +
        args.w_q_share * df["z_q_share"] +
        args.w_spread * df["z_spread"] +
        args.w_conc_low * df["z_concentration_low"]
    )
    df["cw_score"] = df["cw_score_raw"] + np.where(df["is_incorrect"], args.incorrect_penalty, 0.0)

    # Bands
    df["Cognitive_Workload"] = df["cw_score"].apply(lambda s: classify_band(s, args.low_threshold, args.high_threshold))

    # Explanations
    df["Explanation"] = df.apply(build_explanation, axis=1)

    # Output
    # Keep original columns plus derived results for transparency
    out_cols = list(df.columns)
    # Reorder so results are at the end but visible
    result = df[out_cols]
    result.to_csv(args.output, index=False)
    print(f"Saved results to: {args.output}")


if __name__ == "__main__":
    main()
