"""
Genera sequenze testuali (Start → ... → End) da file eventi *_events.csv.

Eventi supportati:
- fixation           -> "FIX R_i_j (Xms)"
- saccade            -> "SAC R_i_j" (R_i_j = regione della fissazione successiva)
- saccade+regression -> "REG R_i_j (Xms)"
- blink              -> "BLINK"
- microsaccade       -> "+MS" (appeso alla FIX precedente se adiacente, altrimenti token autonomo)
- pursuit            -> "JMP R_i_j" (mappato alla regione della fissazione successiva)

Assunzioni colonne tipiche nei file eventi:
    type, start_idx, end_idx, start_s, end_s, duration_s, [cx, cy]
Dove cx, cy (se presenti) sono le coordinate (normalizzate 0..1) del baricentro fissazione.

Uso:
    python make_sequences_from_events.py --in AAF_beach_events.csv
    python make_sequences_from_events.py --in gazeom_out --summary sequences_summary.csv
"""

from pathlib import Path
import argparse
import pandas as pd
import numpy as np

ARROW = " \u2192 "  # " → "

# ------------------------- Utilità -------------------------

def autodetect_read_csv(path: Path) -> pd.DataFrame:
    """Tenta lettura con separatore auto (csv/tsv)."""
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        # fallback comune
        return pd.read_csv(path)

def to_region(cx: float, cy: float, rows: int, cols: int) -> str | None:
    """Mappa (cx, cy) normalizzati [0,1] in regione R_row_col (row 0=top, col 0=left)."""
    if pd.isna(cx) or pd.isna(cy):
        return None
    c = int(np.clip(np.floor(cx * cols), 0, cols - 1))
    r = int(np.clip(np.floor(cy * rows), 0, rows - 1))
    return f"R_{r}_{c}"

def ms_from_seconds(d: float | None) -> str:
    if d is None or pd.isna(d):
        return "0ms"
    return f"{int(round(float(d) * 1000))}ms"

def next_fixation_region(events: pd.DataFrame, i: int, rows: int, cols: int, fallback: str = "R_1_1") -> str:
    """Ritorna la regione della prima fissazione successiva all'evento i (altrimenti fallback)."""
    n = len(events)
    for j in range(i + 1, n):
        if str(events.iloc[j]["type"]) == "fixation":
            cx = events.iloc[j].get("cx", np.nan)
            cy = events.iloc[j].get("cy", np.nan)
            reg = to_region(cx, cy, rows, cols)
            return reg or fallback
    return fallback

# ------------------------- Sequenza da eventi -------------------------

def build_sequence_from_events(df: pd.DataFrame, rows: int = 3, cols: int = 3) -> str:
    """
    Converte la tabella eventi (ordinata temporalmente) in una stringa:
        Start → ... → End
    """
    events = df.sort_values(["start_idx", "start_s"], na_position="last").reset_index(drop=True)
    tokens: list[str] = ["Start"]

    for i, ev in events.iterrows():
        etype = str(ev.get("type", "")).lower()

        if etype == "fixation":
            region = to_region(ev.get("cx", np.nan), ev.get("cy", np.nan), rows, cols) or "R_1_1"
            token = f"FIX {region} ({ms_from_seconds(ev.get('duration_s'))})"
            # Se il prossimo evento è un microsaccade, appendi "+MS" a questo token
            if i + 1 < len(events) and "microsaccade" in str(events.iloc[i + 1].get("type", "")).lower():
                token += " +MS"
            tokens.append(token)

        elif "microsaccade" in etype:
            # Se non è già stato appeso alla FIX precedente, emetti token autonomo
            if not (len(tokens) > 1 and tokens[-1].startswith("FIX " ) and tokens[-1].endswith("+MS")):
                tokens.append("+MS")

        elif etype.startswith("saccade"):
            region = next_fixation_region(events, i, rows, cols)
            if "regression" in etype:
                tokens.append(f"REG {region} ({ms_from_seconds(ev.get('duration_s'))})")
            else:
                tokens.append(f"SAC {region}")

        elif etype == "blink":
            tokens.append("BLINK")

        elif "pursuit" in etype:
            # mappiamo il pursuit a un "JMP" verso la prossima fissazione
            region = next_fixation_region(events, i, rows, cols)
            tokens.append(f"JMP {region}")

        else:
            # Eventuali altri tipi: lasciare in chiaro in maiuscolo
            tokens.append(etype.upper())

    tokens.append("End")
    return ARROW.join(tokens)

# ------------------------- Batch processing -------------------------

def collect_event_files(in_path: Path) -> list[Path]:
    """Restituisce la lista dei file eventi da processare.
    - Se in_path è un file, restituisce [in_path]
    - Se è una cartella, cerca ricorsivamente *_events.csv e *.csv
    """
    if in_path.is_file():
        return [in_path]
    files = []
    # Preferisci pattern *_events.csv, ma consenti anche qualunque .csv
    for p in in_path.rglob("*_events.csv"):
        files.append(p)
    if not files:
        files = list(in_path.rglob("*.csv"))
    return files

def main():
    ap = argparse.ArgumentParser(description="Genera sequenze testuali da file eventi gaze.")
    ap.add_argument("--in", dest="in_path", required=True,
                    help="File eventi CSV/TSV o cartella contenente più file.")
    ap.add_argument("--out-dir", default=None,
                    help="Cartella di output. Se non specificata, scrive accanto ai file input.")
    ap.add_argument("--rows", type=int, default=3, help="Numero righe griglia (default 3).")
    ap.add_argument("--cols", type=int, default=3, help="Numero colonne griglia (default 3).")
    ap.add_argument("--summary", default=None,
                    help="(Opzionale) Salva un CSV riassuntivo con colonne id,sequence.")
    args = ap.parse_args()

    in_path = Path(args.in_path).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else None

    files = collect_event_files(in_path)
    if not files:
        print(f"[WARN] Nessun file trovato in: {in_path}")
        return

    summary_rows = []

    for fp in sorted(files):
        try:
            df = autodetect_read_csv(fp)
        except Exception as e:
            print(f"[SKIP] {fp}: errore lettura -> {e}")
            continue

        # controllo colonne minime
        if "type" not in df.columns:
            print(f"[SKIP] {fp}: colonna 'type' assente.")
            continue

        # build sequence
        seq = build_sequence_from_events(df, rows=args.rows, cols=args.cols)

        # prepara output path
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            dst = out_dir / f"{fp.stem.replace('.csv','')}_sequence.txt"
        else:
            dst = fp.with_name(f"{fp.stem.replace('.csv','')}_sequence.txt")

        # salva
        dst.write_text(seq, encoding="utf-8")
        print(f"[OK] {fp.name} → {dst.name}")

        # summary row
        summary_rows.append({"id": str(fp.stem), "sequence": seq})

    if args.summary:
        summary_path = Path(args.summary).expanduser().resolve()
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f"[OK] Summary salvato in: {summary_path}")

if __name__ == "__main__":
    main()
