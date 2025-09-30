import pandas as pd
import json
import time
import subprocess

def ask_ollama(prompt_model):
    process = subprocess.Popen(
        ['ollama', 'run', 'phi3-cw2'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    stdout, stderr = process.communicate(input=prompt_model.encode())

    if stderr:
        print("Error:", stderr.decode())

    return stdout.decode()




# === CONFIG ===
INPUT_FILE = "./data_csv/symbolic_sequences_all.csv"
OUTPUT_CSV = "./data_csv/results_ollama.csv"
OUTPUT_JSON = "./data_csv/results_ollama.json"



# === Load input ===
df = pd.read_csv(INPUT_FILE)
results = []

print("🚀 Starting batch evaluation using Ollama...")

for idx, row in df.iterrows():
    gaze_seq = row["symbolic_sequence"]
    sample_id = row.get("participant_id", idx)
    prompt = gaze_seq
    print(f"[{sample_id}] → {prompt}")
    output = ask_ollama(prompt)
    lines = output.split("\n")
    label = next((l.split(":")[-1].strip().upper() for l in lines if "Cognitive Workload" in l), "UNKNOWN")
    label.strip("*")
    explanation = next((l.split(":", 1)[-1].strip() for l in lines if "Explanation" in l), "")
    explanation.strip("*")
    result = {
        "id": sample_id,
        "gaze_sequence": gaze_seq,
        "predicted_label": label,
        "explanation": explanation
    }

    results.append(result)
    print(f"[{sample_id}] → {label} | {explanation}")


# === Save output ===
df_out = pd.DataFrame(results)
df_out.to_csv(OUTPUT_CSV, index=False)

with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved to:\n - {OUTPUT_CSV}\n - {OUTPUT_JSON}")