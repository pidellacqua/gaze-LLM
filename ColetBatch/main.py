import math
import pandas as pd
import json
import os

def gaze_to_text_full_events_norm_ms(gaze_data, grid_x=3, grid_y=3, baseline_pupil=40.0):
    """
    Converts normalized gaze data (x, y ∈ [0,1]) with pupil dilation,
    regressions, jumps, and special events (blink, microsaccade)
    into a symbolic text sequence.
    Consecutive FIX on same region are merged.
    Durations are expressed in milliseconds.
    """
    sequence = ["Start"]
    visited_regions = []
    last_region = None
    last_time = gaze_data[0]["t"]

    def get_region(x, y):
        return int(x * grid_x), int(y * grid_y)

    def region_token(region):
        return f"R_{region[0]}_{region[1]}"

    i = 0
    while i < len(gaze_data):
        current = gaze_data[i]

        # BLINK event
        if current.get("blink", False):
            sequence.append("BLINK")
            last_region = None  # reset context after blink
            i += 1
            continue

        # Get current region
        region = get_region(current["x"], current["y"])
        region_str = region_token(region)
        time = current["t"]
        pupil_values = [current.get("pupil_size", baseline_pupil)]
        microsaccade = current.get("microsaccade", False)

        # Merge consecutive samples in same region
        j = i + 1
        while j < len(gaze_data):
            next_event = gaze_data[j]
            if "blink" in next_event:
                break
            next_region = get_region(next_event["x"], next_event["y"])
            if next_region != region:
                break
            pupil_values.append(next_event.get("pupil_size", baseline_pupil))
            if next_event.get("microsaccade", False):
                microsaccade = True
            j += 1

        # Calculate duration and PD
        end_time = gaze_data[j - 1]["t"]
        duration_ms = int(round((end_time - time) * 1000))
        avg_pupil = sum(pupil_values) / len(pupil_values)
        pupil_change = ((avg_pupil - baseline_pupil) / baseline_pupil) * 100
        pupil_token = f"PD_{pupil_change:+.0f}%"
        micro_token = " +MS" if microsaccade else ""

        # Decide the type of transition
        if i == 0 or last_region is None:
            action = f"FIX {region_str} (start) [{pupil_token}]{micro_token}"
        else:
            dx = abs(region[0] - last_region[0])
            dy = abs(region[1] - last_region[1])
            is_jump = dx > 1 or dy > 1
            is_regression = region in visited_regions

            if region == last_region:
                action = f"FIX {region_str} ({duration_ms}ms) [{pupil_token}]{micro_token}"
            elif is_regression:
                action = f"REG {region_str} ({duration_ms}ms) [{pupil_token}]{micro_token}"
            elif is_jump:
                action = f"JMP {region_str} → FIX {region_str} ({duration_ms}ms) [{pupil_token}]{micro_token}"
            else:
                action = f"SAC {region_str} → FIX {region_str} ({duration_ms}ms) [{pupil_token}]{micro_token}"

        sequence.append(action)
        visited_regions.append(region)
        last_region = region
        last_time = end_time
        i = j  # Passa al prossimo evento non ancora gestito

    sequence.append("End")
    return " → ".join(sequence)




# === PARAMETRI MICROSACCADE ===
MIN_DURATION = 0.02  # in secondi
VELOCITY_THRESHOLD = 0.05  # in norm units / sec

def load_gaze_data(gaze_path):
    df = pd.read_csv(gaze_path)
    df = df.rename(columns={"gaze_timestamp": "t", "norm_pos_x": "x", "norm_pos_y": "y"})
    return df[["t", "x", "y"]].sort_values("t")

def load_pupil_data(pupil_path):
    df = pd.read_csv(pupil_path)
    df = df.rename(columns={"pupil_timestamp": "t", "diameter": "pupil_size"})
    return df[["t", "pupil_size"]].sort_values("t")

def load_blink_data(blink_path):
    df = pd.read_csv(blink_path)
    df = df.rename(columns={"start_timestamp": "t"})
    return df[["t"]].sort_values("t")

def merge_gaze_and_pupil(gaze_df, pupil_df):
    df = pd.merge_asof(gaze_df.sort_values("t"), pupil_df.sort_values("t"), on="t", direction='nearest')
    return df

def convert_to_gaze_objects(merged_df):
    gaze_events = []
    for _, row in merged_df.iterrows():
        event = {
            "t": float(row["t"]),
            "x": float(row["x"]),
            "y": float(row["y"]),
            "pupil_size": float(row["pupil_size"])
        }
        gaze_events.append(event)
    return gaze_events

def insert_blinks(gaze_events, blink_df):
    blink_events = [{"t": float(row["t"]), "blink": True} for _, row in blink_df.iterrows()]
    all_events = gaze_events + blink_events
    return sorted(all_events, key=lambda e: e["t"])

def detect_microsaccades(events):
    for i in range(1, len(events)):
        prev = events[i - 1]
        curr = events[i]

        if all(k in prev for k in ["x", "y", "t"]) and all(k in curr for k in ["x", "y", "t"]):
            dx = curr["x"] - prev["x"]
            dy = curr["y"] - prev["y"]
            dt = curr["t"] - prev["t"]

            if dt >= MIN_DURATION:
                v = ((dx**2 + dy**2)**0.5) / dt
                if v > VELOCITY_THRESHOLD:
                    curr["microsaccade"] = True
    return events

def process_participant(gaze_file, pupil_file, blink_file, output_file):
    print(f"🔄 Processing participant files...")
    gaze_df = load_gaze_data(gaze_file)
    pupil_df = load_pupil_data(pupil_file)
    blink_df = load_blink_data(blink_file)

    merged_df = merge_gaze_and_pupil(gaze_df, pupil_df)
    gaze_events = convert_to_gaze_objects(merged_df)


    all_events = insert_blinks(gaze_events, blink_df)
    enriched_events = detect_microsaccades(all_events)
    
    with open(output_file, "w") as f:
        json.dump(enriched_events, f, indent=2)
    print(f"✅ Gaze data salvato in: {output_file} ({len(enriched_events)} eventi)")


if __name__ == "__main__":
    # === CONFIG ===
    DATA_DIR = "./data_csv"
    OUTPUT_CSV = "./data_csv/symbolic_sequences_all.csv"
    N_PARTICIPANTS = 47

    results = []

    for pid in range(1, N_PARTICIPANTS + 1):
        pid_str = f"{pid:02d}"  # 01, 02, ..., 47

        for i in range(1, 5):
            mes_str = f"{i:02d}"
            # === File path
            gaze_file = f"{DATA_DIR}/participant_{pid_str}_{mes_str}_gaze.csv"
            pupil_file = f"{DATA_DIR}/participant_{pid_str}_{mes_str}_pupil.csv"
            blink_file = f"{DATA_DIR}/participant_{pid_str}_{mes_str}_blinks.csv"
            output_json = f"{DATA_DIR}/participant_{pid_str}_{mes_str}_gaze_data.json"



            # === Verifica esistenza file
            if not (os.path.exists(gaze_file) and os.path.exists(pupil_file) and os.path.exists(blink_file)):
                print(f"⚠️  File mancanti per participant {pid_str} , misura {mes_str} . Salto...")
                continue

            # === Elabora e salva gaze_data JSON
            print(f"🔄 Processing P{pid_str} M{mes_str} ...")
            process_participant(gaze_file, pupil_file, blink_file, output_json)

            # === Carica gaze_data e genera sequenza simbolica
            with open(output_json, "r") as f:
                gaze_data = json.load(f)
            text_seq = gaze_to_text_full_events_norm_ms(gaze_data)

            # === Salva riga nel CSV finale
            results.append({
                "participant_id": f"P{pid_str}_M{mes_str}",
                "symbolic_sequence": text_seq
            })

    # === Esporta tutte le sequenze in CSV
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Tutte le {len(results)} sequenze simboliche salvate in: {OUTPUT_CSV}")