import pandas as pd

#FIX ×6 (avg: 190ms) → SAC ×5 (avg: 150ms, amp: 4.9°) → BLINK ×1

# === CONFIG ===
INPUT_FILE = "./emoa/EMOAEventData.csv"
OUTPUT_FILE = "emoa_symbolic_sequences.csv"

def build_symbolic_sequence(row):
    parts = []

    # --- Fixations
    fix_count = int(row["fixation_count"])
    fix_dur = float(row["fixation_duration"])
    if fix_count > 0:
        avg_fix = round(fix_dur / fix_count, 1)
        parts.append(f"FIX ×{fix_count} (avg: {avg_fix}ms)")

    # --- Saccades
    sac_count = int(row["saccade_count"])
    sac_dur = float(row["saccade_duration"])
    sac_amp = float(row["saccade_amplitude"])
    if sac_count > 0:
        avg_sac = round(sac_dur / sac_count, 1)
        parts.append(f"SAC ×{sac_count} (avg: {avg_sac}ms, amp: {sac_amp:.1f}°)")

    # --- Blinks
    blink_count = int(row["blink_count"])
    if blink_count > 0:
        parts.append(f"BLINK ×{blink_count}")

    return " → ".join(parts)

def main():
    # Legge i dati EMOA
    df = pd.read_csv(INPUT_FILE)

    # Crea la colonna delle sequenze simboliche
    df["symbolic_sequence"] = df.apply(build_symbolic_sequence, axis=1)

    # Esporta il file simbolico
    df[["symbolic_sequence"]].to_csv(OUTPUT_FILE, index=False)
    print(f"✅ File salvato: {OUTPUT_FILE} ({len(df)} sequenze)")

if __name__ == "__main__":
    main()
