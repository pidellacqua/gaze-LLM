"""
detect_gaze_events.py
Rileva: saccade, microsaccade, regression, smooth pursuit, fixation, blink
Formato atteso: colonne X_coord, Y_coord (opz. Confidence), sampling 250 Hz.
Supporta CSV o TSV (auto-separatore).
"""

import argparse
import numpy as np
import pandas as pd
import os

FPS = 250.0
DT = 1.0 / FPS

# --- Parametri (modificabili da CLI) ---
DEFAULTS = dict(
    fix_quantile=0.35,       # quantile velocità per fissazioni
    sac_quantile=0.90,       # quantile velocità per saccadi
    min_fix_dur=0.08,        # s
    min_sac_dur=0.012,       # s
    min_purs_dur=0.10,       # s
    micro_amp=0.02,          # ampiezza in unità delle coordinate (normalizzate)  ~2% dello schermo
    micro_max_dur=0.040,     # s
    pursuit_dir_std=30.0,    # deviazione std circolare max (gradi)
    conf_thr=0.5,            # Confidence < conf_thr -> blink
    smooth_win=5             # media mobile (campioni)
)

def read_table(path):
    # prova con TAB, poi con virgola
    try:
        df = pd.read_csv(path, sep="\t")
    except Exception:
        df = pd.read_csv(path)
    return df

def central_diff_signal(x, win=5):
    xs = pd.Series(x).rolling(win, center=True, min_periods=1).mean().to_numpy()
    v = np.zeros_like(xs, dtype=float)
    v[1:-1] = (xs[2:] - xs[:-2]) / (2*DT)
    v[0] = (xs[1] - xs[0]) / DT
    v[-1] = (xs[-1] - xs[-2]) / DT
    return xs, v

def segment_bool(mask):
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    segs = []
    s = idx[0]; p = idx[0]
    for i in idx[1:]:
        if i == p + 1:
            p = i; continue
        segs.append((s, p)); s = i; p = i
    segs.append((s, p))
    return segs

def dur_s(seg):
    s,e = seg; return (e - s + 1) * DT

def angle_deg(x0, y0, x1, y1):
    return float(np.degrees(np.arctan2(y1 - y0, x1 - x0)))

def amp_units(x0, y0, x1, y1):
    return float(np.hypot(x1 - x0, y1 - y0))

def angdiff(a, b):
    d = (a - b + 180) % 360 - 180
    return abs(d)

def circ_std_deg(angles_deg):
    ang = np.radians(angles_deg)
    s = float(np.mean(np.sin(ang))); c = float(np.mean(np.cos(ang)))
    R = float(np.hypot(s, c))
    if R <= 0:
        return 180.0
    return float(np.degrees(np.sqrt(-2*np.log(max(R, 1e-12)))))

def detect_events(df, args):
    # colonne
    assert "X_coord" in df.columns and "Y_coord" in df.columns, \
        "Il file deve contenere colonne 'X_coord' e 'Y_coord'."
    x = df["X_coord"].astype(float).to_numpy()
    y = df["Y_coord"].astype(float).to_numpy()
    conf = df[args.col_conf].astype(float).to_numpy() if (args.col_conf in df.columns) else np.ones_like(x)

    # velocità
    _, vx = central_diff_signal(x, win=args.smooth_win)
    _, vy = central_diff_signal(y, win=args.smooth_win)
    speed = np.hypot(vx, vy)
    direction = np.degrees(np.arctan2(vy, vx))

    # soglie adattive
    fix_thr = np.quantile(speed, args.fix_quantile)
    sac_thr = np.quantile(speed, args.sac_quantile)

    # blink
    blink_mask = (np.isnan(x) | np.isnan(y)) | (conf < args.conf_thr)
    blink_segs = segment_bool(blink_mask)
    blinks = [{"type":"blink","start_idx":s,"end_idx":e,"start_s":s*DT,"end_s":e*DT,
               "duration_s":dur_s((s,e))} for (s,e) in blink_segs]

    # saccadi
    sac_mask = (speed >= sac_thr) & (~blink_mask)
    sac_segs = [seg for seg in segment_bool(sac_mask) if dur_s(seg) >= args.min_sac_dur]
    saccades = []
    for (s,e) in sac_segs:
        amp = amp_units(x[s], y[s], x[e], y[e])
        ang = angle_deg(x[s], y[s], x[e], y[e])
        ev = {"type":"saccade","start_idx":s,"end_idx":e,"start_s":s*DT,"end_s":e*DT,
              "duration_s":dur_s((s,e)),"amplitude":amp,"angle_deg":ang}
        # microsaccade
        if (amp < args.micro_amp) and (ev["duration_s"] <= args.micro_max_dur):
            ev["type"] = "microsaccade"
        saccades.append(ev)
    # regression (flip >135° vs saccade precedente)
    for i in range(1, len(saccades)):
        if angdiff(saccades[i]["angle_deg"], saccades[i-1]["angle_deg"]) > 135:
            saccades[i]["type"] += "+regression"

    # fissazioni
    fix_mask = (speed <= fix_thr) & (~blink_mask)
    fix_segs = [seg for seg in segment_bool(fix_mask) if dur_s(seg) >= args.min_fix_dur]
    fixations = [{"type":"fixation","start_idx":s,"end_idx":e,"start_s":s*DT,"end_s":e*DT,
                  "duration_s":dur_s((s,e)),
                  "cx":float(np.nanmean(x[s:e+1])),"cy":float(np.nanmean(y[s:e+1]))} for (s,e) in fix_segs]

    # pursuit (vel. intermedia + direzione stabile)
    mid_mask = (speed > fix_thr) & (speed < sac_thr) & (~blink_mask)
    for s,e in sac_segs: mid_mask[s:e+1] = False
    for s,e in fix_segs: mid_mask[s:e+1] = False
    purs_segs = segment_bool(mid_mask)
    pursuits = []
    for (s,e) in purs_segs:
        if dur_s((s,e)) < args.min_purs_dur:
            continue
        if circ_std_deg(direction[s:e+1]) <= args.pursuit_dir_std:
            pursuits.append({"type":"pursuit","start_idx":s,"end_idx":e,"start_s":s*DT,"end_s":e*DT,
                             "duration_s":dur_s((s,e)),"mean_speed":float(np.mean(speed[s:e+1]))})

    # unione
    events = []
    events.extend(blinks)
    events.extend(fixations)
    events.extend(saccades)
    events.extend(pursuits)
    events = sorted(events, key=lambda d: d["start_idx"])
    events_df = pd.DataFrame(events)

    # etichette per campione
    labels = np.array(["none"] * len(df), dtype=object)
    for s,e in blink_segs: labels[s:e+1] = "blink"
    for s,e in fix_segs:  labels[s:e+1] = "fixation"
    for s,e in sac_segs:  labels[s:e+1] = "saccade"
    for ev in saccades:
        if "microsaccade" in ev["type"]:
            labels[ev["start_idx"]:ev["end_idx"]+1] = ev["type"]
    for p in pursuits: labels[p["start_idx"]:p["end_idx"]+1] = "pursuit"

    samples = df.copy()
    samples["time_s"] = np.arange(len(df)) * DT
    samples["speed"] = speed
    samples["vx"] = vx
    samples["vy"] = vy
    samples["dir_deg"] = direction
    samples["event_label"] = labels

    meta = {
        "fix_vel_thr": float(fix_thr),
        "sac_vel_thr": float(sac_thr),
        "params": vars(args)
    }
    return events_df, samples, meta

def build_argparser():
    p = argparse.ArgumentParser(description="Rilevamento eventi oculomotori (250 Hz, X_coord/Y_coord).")
    p.add_argument("input", help="File CSV/TSV con colonne X_coord, Y_coord (opzionale Confidence).")
    p.add_argument("--out-events", default="gaze_events.csv", help="CSV con eventi.")
    p.add_argument("--out-samples", default="gaze_samples_labeled.csv", help="CSV per-campione etichettato.")
    p.add_argument("--col-conf", default="Confidence", dest="col_conf")
    # soglie
    p.add_argument("--fix-quantile", type=float, default=DEFAULTS["fix_quantile"])
    p.add_argument("--sac-quantile", type=float, default=DEFAULTS["sac_quantile"])
    p.add_argument("--min-fix-dur", type=float, default=DEFAULTS["min_fix_dur"])
    p.add_argument("--min-sac-dur", type=float, default=DEFAULTS["min_sac_dur"])
    p.add_argument("--min-purs-dur", type=float, default=DEFAULTS["min_purs_dur"])
    p.add_argument("--micro-amp", type=float, default=DEFAULTS["micro_amp"])
    p.add_argument("--micro-max-dur", type=float, default=DEFAULTS["micro_max_dur"])
    p.add_argument("--pursuit-dir-std", type=float, default=DEFAULTS["pursuit_dir_std"])
    p.add_argument("--conf-thr", type=float, default=DEFAULTS["conf_thr"])
    p.add_argument("--smooth-win", type=int, default=DEFAULTS["smooth_win"])
    return p

def main():
    args = build_argparser().parse_args()
    df = read_table(args.input)
    events_df, samples_df, meta = detect_events(df, args)
    events_df.to_csv(args.out_events, index=False)
    samples_df.to_csv(args.out_samples, index=False)
    print(f"[OK] Saved events -> {args.out_events}  | samples -> {args.out_samples}")
    print(f"Adaptive thresholds: fix_vel_thr={meta['fix_vel_thr']:.4f}, sac_vel_thr={meta['sac_vel_thr']:.4f}")

if __name__ == "__main__":
    main()
