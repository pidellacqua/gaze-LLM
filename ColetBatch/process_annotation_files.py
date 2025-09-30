import os
import csv
import pandas as pd

# === Configuration ===
INPUT_FOLDER = "./data_csv"  # folder with all 188 files
OUTPUT_FILE = "./data_csv/compiled_rtlx_workload.csv"

# === Thresholds ===
def classify_workload(mean_score):
    if mean_score <= 10:
        return "LOW"
    elif mean_score <= 20:
        return "MEDIUM"
    else:
        return "HIGH"

# === Process files ===
rows = []

for pid in range(1, 48):  # participant_01 to participant_47
    for activity in range(1, 5):  # activity_01 to activity_04
        filename = f"participant_{pid:02d}_{activity:02d}_annotation.csv"
        filepath = os.path.join(INPUT_FOLDER, filename)

        if not os.path.exists(filepath):
            print(f"⚠️ File not found: {filename}")
            continue

        with open(filepath, "r", newline='') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            try:
                values = next(reader)  # read the actual row
            except StopIteration:
                print(f"⚠️ No data row in: {filename}")
                continue

            if len(values) < 7:
                print(f"⚠️ Invalid row in: {filename}")
                continue

            try:
                mean = float(values[6])  # assuming 'mean' is the 7th column
                workload = classify_workload(mean)
            except ValueError:
                print(f"⚠️ Invalid mean value in: {filename}")
                continue

            rows.append({
                "participant_id": f"P{pid:02d}",
                "activity_id": activity,
                "mean": mean,
                "cognitive_workload": workload
            })

# === Save output CSV ===
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ File salvato: {OUTPUT_FILE} ({len(df)} rows)")

