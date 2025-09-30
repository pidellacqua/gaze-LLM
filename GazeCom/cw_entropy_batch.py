#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

'''
python cw_entropy_batch.py \
  --root /percorso/cartella_principale \
  --out /percorso/output/results_entropy.csv \
  --bins 48 --low_thr 0.30 --high_thr 0.70



python cw_entropy_batch.py \
  --root /percorso/cartella_principale \
  --out /percorso/output/results_entropy.csv \
  --normalize \
  --conf_col Confidence --conf_thr 0.6


::Comando utilizzato per le misure::

python cw_entropy_batch.py \
  --root ./gazecom \
  --out ./results_entropy.csv \
  --bins 48 --low_thr 0.70 --high_thr 0.80


'''


# ------------------ Entropy utilities ------------------
def normalized_shannon_entropy_2d(x, y, bins=32):
    H2, xedges, yedges = np.histogram2d(x, y, bins=bins)
    total = H2.sum()
    if total <= 0:
        return 0.0, 0.0
    p = H2 / total
    p_nz = p[p > 0]
    H = -(p_nz * np.log2(p_nz)).sum()
    K = max(1, p_nz.size)
    H_norm = H / np.log2(K) if K > 1 else 0.0
    return float(H), float(H_norm)

def classify_workload_from_entropy(H_norm, low_thr=0.33, high_thr=0.66):
    if H_norm < low_thr:
        return "LOW"
    elif H_norm < high_thr:
        return "MEDIUM"
    else:
        return "HIGH"

# ------------------ File reading helpers ------------------
SEPARATORS = ["\t", ",", ";", "|"]

def try_read(path, sep=None, use_python_engine=False):
    try:
        if sep is None:
            # pandas automatic detection via python engine with sep=None
            df = pd.read_csv(path, sep=None, engine="python")
            return df, df.attrs.get("delimiter", None) or "auto"
        else:
            df = pd.read_csv(path, sep=sep)
            return df, sep
    except Exception as e:
        return e, None

def autodetect_read(path):
    # First try pandas sep=None (python engine)
    df, used = try_read(path, sep=None, use_python_engine=True)
    if isinstance(df, pd.DataFrame) and df.shape[1] >= 2:
        return df, used
    # Then try a list of common separators
    for sep in SEPARATORS:
        df, used = try_read(path, sep=sep)
        if isinstance(df, pd.DataFrame) and df.shape[1] >= 2:
            return df, used
    # If all failed, return the last error
    return df, used

def pick_xy_columns(df, x_candidates=("X_coord","X","x","x_coord"),
                          y_candidates=("Y_coord","Y","y","y_coord")):
    cols_lower = {c.lower(): c for c in df.columns}
    x_col = None
    y_col = None
    for xc in x_candidates:
        if xc.lower() in cols_lower:
            x_col = cols_lower[xc.lower()]
            break
    for yc in y_candidates:
        if yc.lower() in cols_lower:
            y_col = cols_lower[yc.lower()]
            break
    # Fallback: take the first two numeric columns if needed
    if x_col is None or y_col is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) >= 2:
            if x_col is None: x_col = numeric_cols[0]
            if y_col is None: y_col = numeric_cols[1] if numeric_cols[1] != x_col else (numeric_cols[2] if len(numeric_cols) > 2 else None)
    return x_col, y_col

def normalize_01(arr):
    a = np.asarray(arr, dtype=float)
    mn = np.nanmin(a)
    mx = np.nanmax(a)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn == 0:
        return a  # no change if degenerate
    return (a - mn) / (mx - mn)

# ------------------ Main processing ------------------
def process_file(path, bins, low_thr, high_thr, conf_col=None, conf_thr=None, do_normalize=False):
    df, sep_used = autodetect_read(path)
    if not isinstance(df, pd.DataFrame):
        raise RuntimeError(f"Unable to read file: {path} (last error: {df})")
    # Select X/Y
    x_col, y_col = pick_xy_columns(df)
    if x_col is None or y_col is None:
        raise RuntimeError(f"Unable to identify X/Y columns in {path}. Columns={list(df.columns)}")
    # Confidence filtering (optional)
    if conf_col and conf_col in df.columns and conf_thr is not None:
        df = df[df[conf_col] >= conf_thr]
    # Extract and clean
    xy = df[[x_col, y_col]].dropna()
    if xy.empty:
        raise RuntimeError(f"No valid X/Y data after cleaning in {path}")
    x = xy[x_col].to_numpy().astype(float)
    y = xy[y_col].to_numpy().astype(float)
    # Optional normalization
    if do_normalize:
        x = normalize_01(x)
        y = normalize_01(y)
    # Entropy + classification
    H_bits, H_norm = normalized_shannon_entropy_2d(x, y, bins=bins)
    cw = classify_workload_from_entropy(H_norm, low_thr=low_thr, high_thr=high_thr)
    return {
        "entropy_bits": H_bits,
        "entropy_norm": H_norm,
        "classification": cw,
        "n_points": len(x),
        "sep_used": sep_used,
        "x_col": x_col,
        "y_col": y_col
    }

def main():
    ap = argparse.ArgumentParser(description="Batch Shannon-entropy CW estimation from gaze CSV/TSV files (recursive).")
    ap.add_argument("--root", required=True, help="Root directory to scan recursively for files.")
    ap.add_argument("--out", required=True, help="Output CSV path (single consolidated results file).")
    ap.add_argument("--bins", type=int, default=32, help="2D histogram bins per axis (default: 32).")
    ap.add_argument("--low_thr", type=float, default=0.33, help="LOW threshold on normalized entropy (default: 0.33).")
    ap.add_argument("--high_thr", type=float, default=0.66, help="HIGH threshold on normalized entropy (default: 0.66).")
    ap.add_argument("--conf_col", default=None, help="Optional confidence column name (e.g., 'Confidence').")
    ap.add_argument("--conf_thr", type=float, default=None, help="Keep rows with confidence >= this value.")
    ap.add_argument("--normalize", action="store_true", help="Normalize X/Y to [0,1] before entropy.")
    ap.add_argument("--ext", nargs="+", default=[".csv", ".tsv", ".txt"], help="File extensions to include (default: .csv .tsv .txt).")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"[ERROR] Root folder does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    results = []
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {e.lower() for e in args.ext}]
    if not files:
        print(f"[WARN] No files with extensions {args.ext} found under {root}", file=sys.stderr)

    for idx, path in enumerate(sorted(files)):
        rel = path.relative_to(root)
        try:
            res = process_file(
                path, bins=args.bins, low_thr=args.low_thr, high_thr=args.high_thr,
                conf_col=args.conf_col, conf_thr=args.conf_thr, do_normalize=args.normalize
            )
            results.append({
                "filepath": str(path),
                "relative_path": str(rel),
                "filename": path.name,
                "n_points": res["n_points"],
                "entropy_bits": f"{res['entropy_bits']:.6f}",
                "entropy_norm": f"{res['entropy_norm']:.6f}",
                "cognitive_workload": res["classification"],
                "bins": args.bins,
                "sep_used": res["sep_used"],
                "x_col": res["x_col"],
                "y_col": res["y_col"],
                "error": ""
            })
        except Exception as e:
            results.append({
                "filepath": str(path),
                "relative_path": str(rel),
                "filename": path.name,
                "n_points": 0,
                "entropy_bits": "",
                "entropy_norm": "",
                "cognitive_workload": "",
                "bins": args.bins,
                "sep_used": "",
                "x_col": "",
                "y_col": "",
                "error": str(e)
            })

    # Save consolidated CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(results, columns=[
        "filepath","relative_path","filename","n_points",
        "entropy_bits","entropy_norm","cognitive_workload",
        "bins","sep_used","x_col","y_col","error"
    ])
    df_out.to_csv(out_path, index=False)
    print(f"[OK] Saved results to: {out_path}")

if __name__ == "__main__":
    main()
