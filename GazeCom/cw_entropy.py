import numpy as np
import pandas as pd

#python cw_entropy.py --input /path/al/tuo_file.csv --sep "\t" --x_col X_coord --y_col Y_coord --bins 32

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
    if H_norm < low_thr: return "LOW"
    elif H_norm < high_thr: return "MEDIUM"
    else: return "HIGH"

def main(input_csv, sep="\t", x_col="X_coord", y_col="Y_coord", bins=32):
    df = pd.read_csv(input_csv, sep=sep)
    xy = df[[x_col, y_col]].dropna()
    x = xy[x_col].to_numpy().astype(float)
    y = xy[y_col].to_numpy().astype(float)
    H_bits, H_norm = normalized_shannon_entropy_2d(x, y, bins=bins)
    cw = classify_workload_from_entropy(H_norm)
    print(f"Entropy bits: {H_bits:.6f}")
    print(f"Normalized entropy: {H_norm:.6f}")
    print(f"Cognitive Workload: {cw}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to CSV/TSV with X_coord, Y_coord")
    ap.add_argument("--sep", default="\t", help="Field separator (default: tab)")
    ap.add_argument("--x_col", default="X_coord")
    ap.add_argument("--y_col", default="Y_coord")
    ap.add_argument("--bins", type=int, default=32)
    args = ap.parse_args()
    main(args.input, args.sep, args.x_col, args.y_col, args.bins)
