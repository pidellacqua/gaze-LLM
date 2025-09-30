from sklearn.metrics import classification_report, accuracy_score, cohen_kappa_score, f1_score
import pandas as pd

# === File di input ===
file_rtlx = "./data_csv/compiled_rtlx_workload.csv"
file_llm = "./data_csv/Cognitive_Workload_Results_GPT5.csv"
output_file = "./data_csv/merged_workload_results.csv"

# === Carica i due file ===
df_rtlx = pd.read_csv(file_rtlx)
df_llm = pd.read_csv(file_llm)




# Aggiungi una colonna di df2 a df1 (ad esempio la prima colonna)
# Se vuoi un'altra colonna, cambia 'df2.columns[0]' con il nome della colonna
col_to_add = df_llm['cognitive_workload_llm']

# Unisci i due DataFrame
df_rtlx["cognitive_workload_llm"] = col_to_add

# Salva il risultato
df_rtlx.to_csv(output_file, index=False)

print(f"✅ File unito salvato come {output_file}")


# Evaluation
y_true = df_rtlx["cognitive_workload"]
y_pred = df_rtlx["cognitive_workload_llm"]
print("\n📊 Evaluation Metrics:")
print("Accuracy:", accuracy_score(y_true, y_pred))
print("F1 (macro):", f1_score(y_true, y_pred, average='macro'))
print("Cohen's Kappa:", cohen_kappa_score(y_true, y_pred))
print("\nDetailed Report:\n", classification_report(y_true, y_pred))
