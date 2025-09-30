#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
emoa_evaluate.py
-----------------------------
Compare two CSV files that both contain a column "Cognitive_Workload"
and compute evaluation metrics: accuracy, F1 (macro/weighted), Cohen's kappa,
classification report (per-class), and confusion matrix.
Saves all results to files and also prints a short summary.

Usage:
  python emoa_evaluate.py \
      --pred emoa_symbolic_sequences_cw_GPT.csv \
      --true EMOASemanticData_cw_estimates.csv \
      --outdir results_eval

Outputs in outdir:
  - cw_comparison_paired.csv          (side-by-side labels + optional IDs)
  - cw_classification_report.csv      (precision/recall/F1/support)
  - cw_confusion_matrix.csv           (confusion matrix with labels)
  - cw_metrics_summary.json           (accuracy, macro/weighted F1, kappa, #samples)
"""
import os
import json
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    cohen_kappa_score, confusion_matrix
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="CSV with predicted Cognitive_Workload")
    ap.add_argument("--true", required=True, help="CSV with true/reference Cognitive_Workload")
    ap.add_argument("--outdir", default=".", help="Output directory for results files")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load
    df_pred = pd.read_csv(args.pred)
    df_true = pd.read_csv(args.true)

    # Extract labels
    y_pred = df_pred["Cognitive_Workload"].astype(str).str.strip().str.upper()
    y_true = df_true["Cognitive_Workload"].astype(str).str.strip().str.upper()

    # Align length
    n = min(len(y_pred), len(y_true))
    y_pred = y_pred.iloc[:n].reset_index(drop=True)
    y_true = y_true.iloc[:n].reset_index(drop=True)

    # Build paired (try to keep some identifiers)
    paired = pd.DataFrame({
        "True_Cognitive_Workload": y_true,
        "Pred_Cognitive_Workload": y_pred
    })
    for cand in ["Participant Code", "Trial", "symbolic_sequence"]:
        if cand in df_true.columns and cand not in paired.columns:
            paired[cand] = df_true.loc[:n-1, cand].values
        elif cand in df_pred.columns and cand not in paired.columns:
            paired[cand] = df_pred.loc[:n-1, cand].values

    # Metrics
    labels = ["LOW", "MEDIUM", "HIGH"]
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    conf_mat = confusion_matrix(y_true, y_pred, labels=labels)

    # Save files
    paired_path = os.path.join(args.outdir, "cw_comparison_paired.csv")
    report_csv_path = os.path.join(args.outdir, "cw_classification_report.csv")
    confusion_csv_path = os.path.join(args.outdir, "cw_confusion_matrix.csv")
    summary_json_path = os.path.join(args.outdir, "cw_metrics_summary.json")

    paired.to_csv(paired_path, index=False)
    pd.DataFrame(report_dict).T.to_csv(report_csv_path, index=True)
    pd.DataFrame(conf_mat, index=[f"True_{l}" for l in labels], columns=[f"Pred_{l}" for l in labels]).to_csv(confusion_csv_path, index=True)

    summary = {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "cohen_kappa": kappa,
        "num_samples": int(n),
        "labels_order": labels,
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Saved:")
    print(" -", paired_path)
    print(" -", report_csv_path)
    print(" -", confusion_csv_path)
    print(" -", summary_json_path)
    print("\nSummary:", json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
