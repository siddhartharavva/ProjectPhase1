import pandas as pd

# Load your results file
df = pd.read_csv("rag_results_techqa.csv")

# Ensure numeric columns (important if CSV saved weirdly)
cols = ["EM", "F1", "Recall@K", "Hallucination", "Confidence"]
for c in cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Group by model and compute mean
summary = df.groupby("model").agg({
    "EM": "mean",
    "F1": "mean",
    "Recall@K": "mean",
    "Hallucination": "mean",
    "Confidence": "mean",
}).reset_index()

# Print nicely
print("\n===== SUMMARY =====")
print(summary)

# Save
summary.to_csv("summary_techqa.csv", index=False)