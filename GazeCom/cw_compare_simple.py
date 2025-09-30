#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys

''' 
python cw_compare_simple.py \
  --entropy_csv ./results_entropy.csv \
  --llm_csv ./cognitive_workload_results_gazecom_detailed_GPT.csv \
  --entropy_col cognitive_workload \
  --llm_col cognitive_workload_llm \
  --out_metrics ./cw_compare_metrics.csv \
  --out_cm ./cw_confusion_matrix.csv \
  --out_pairs ./cw_pairs.csv

'''

LABELS = ["LOW","MEDIUM","HIGH"]

def normalize_label(x):
    if isinstance(x, str):
        x = x.strip().upper()
        if x in {"LOW","L"}: return "LOW"
        if x in {"MEDIUM","MID","M"}: return "MEDIUM"
        if x in {"HIGH","H"}: return "HIGH"
    return x

def confusion_matrix(y_true, y_pred, labels):
    idx = {lab:i for i,lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t,p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1
    return cm

def metrics_from_cm(cm):
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.where(tp+fp>0, tp/(tp+fp), 0.0)
        recall    = np.where(tp+fn>0, tp/(tp+fn), 0.0)
        f1        = np.where(precision+recall>0, 2*precision*recall/(precision+recall), 0.0)
        specificity = np.where(tn+fp>0, tn/(tn+fp), 0.0)

    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()
    macro_specificity = specificity.mean()

    support = cm.sum(axis=1).astype(float)
    total = support.sum()
    weights = np.where(total>0, support/total, 0.0)
    weighted_precision = (precision*weights).sum()
    weighted_recall = (recall*weights).sum()
    weighted_f1 = (f1*weights).sum()

    total_tp = tp.sum()
    total_fp = fp.sum()
    total_fn = fn.sum()
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp+total_fp)>0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp+total_fn)>0 else 0.0
    micro_f1 = 2*micro_precision*micro_recall/(micro_precision+micro_recall) if (micro_precision+micro_recall)>0 else 0.0

    accuracy = tp.sum() / cm.sum() if cm.sum() > 0 else 0.0

    row_m = cm.sum(axis=1)
    col_m = cm.sum(axis=0)
    pe = (row_m * col_m).sum() / (cm.sum()**2) if cm.sum() > 0 else 0.0
    kappa = (accuracy - pe) / (1 - pe) if (1 - pe) != 0 else 0.0

    # Multiclass MCC (Gorodkin)
    c = len(cm)
    s = 0.0
    for k in range(c):
        for l in range(c):
            for m in range(c):
                s += cm[k,k]*cm[l,m] - cm[k,l]*cm[m,k]
    t_sum = cm.sum(axis=1)
    p_sum = cm.sum(axis=0)
    denom1 = 0.0
    denom2 = 0.0
    for k in range(c):
        denom1 += t_sum[k]*p_sum[k] - cm[k,k]*cm[k,k]
        denom2 += t_sum[k]*p_sum[k] - cm[k,k]*cm[k,k]
    mcc = s / np.sqrt(denom1*denom2) if denom1>0 and denom2>0 else 0.0

    bal_acc = macro_recall

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "macro_specificity": macro_specificity,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "cohen_kappa": kappa,
        "mcc": mcc,
        "balanced_accuracy": bal_acc,
        "support_total": int(total)
    }

def main():
    ap = argparse.ArgumentParser(description="Simple comparison of CW labels (Entropy vs LLM) by row order alignment.")
    ap.add_argument("--entropy_csv", required=True, help="CSV containing column with entropy-based labels.")
    ap.add_argument("--llm_csv", required=True, help="CSV containing column with LLM-based labels.")
    ap.add_argument("--entropy_col", default="cognitive_workload", help="Column name in entropy CSV (default: cognitive_workload).")
    ap.add_argument("--llm_col", default="cognitive_workload_llm", help="Column name in LLM CSV (default: cognitive_workload_llm).")
    ap.add_argument("--out_metrics", required=True, help="Output metrics CSV path.")
    ap.add_argument("--out_cm", required=True, help="Output confusion matrix CSV path.")
    ap.add_argument("--out_pairs", required=True, help="Output aligned pairs CSV path.")
    ap.add_argument("--dropna", action="store_true", help="Drop rows with missing labels before aligning.")
    args = ap.parse_args()

    df_e = pd.read_csv(args.entropy_csv)
    df_l = pd.read_csv(args.llm_csv)

    if args.entropy_col not in df_e.columns:
        print(f"[ERROR] Column '{args.entropy_col}' not found in {args.entropy_csv}. Found: {list(df_e.columns)}", file=sys.stderr)
        sys.exit(1)
    if args.llm_col not in df_l.columns:
        print(f"[ERROR] Column '{args.llm_col}' not found in {args.llm_csv}. Found: {list(df_l.columns)}", file=sys.stderr)
        sys.exit(1)

    a = df_e[[args.entropy_col]].copy()
    b = df_l[[args.llm_col]].copy()
    if args.dropna:
        a = a.dropna()
        b = b.dropna()

    n = min(len(a), len(b))
    if n == 0:
        print("[ERROR] No rows to compare after trimming. Check inputs.", file=sys.stderr)
        sys.exit(1)

    a = a.iloc[:n].reset_index(drop=True)
    b = b.iloc[:n].reset_index(drop=True)

    y_pred = a[args.entropy_col].apply(normalize_label)
    y_true = b[args.llm_col].apply(normalize_label)

    valid = {"LOW","MEDIUM","HIGH"}
    mask = y_pred.isin(valid) & y_true.isin(valid)
    y_pred = y_pred[mask]
    y_true = y_true[mask]
    if y_true.empty:
        print("[WARN] No valid overlapping labels after filtering. Check label values.", file=sys.stderr)

    # Re-trim both to the filtered mask indices
    pairs = pd.DataFrame({
        "true_llm": y_true.values,
        "pred_entropy": y_pred.values
    })

    cm = confusion_matrix(pairs["true_llm"].tolist(), pairs["pred_entropy"].tolist(), LABELS)
    metrics = metrics_from_cm(cm)

    Path(args.out_metrics).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_cm).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_pairs).parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([metrics]).to_csv(args.out_metrics, index=False)
    pd.DataFrame(cm, index=[f"true_{l}" for l in LABELS], columns=[f"pred_{l}" for l in LABELS]).to_csv(args.out_cm)
    pairs.to_csv(args.out_pairs, index=False)

    print(f"[OK] Metrics saved to: {args.out_metrics}")
    print(f"[OK] Confusion matrix saved to: {args.out_cm}")
    print(f"[OK] Pairs saved to: {args.out_pairs}")

if __name__ == "__main__":
    main()
