from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    csv_path = BASE_DIR / "rag_results_techqa.csv"
    df = pd.read_csv(csv_path)
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    summary = df.groupby("generator_model")[numeric].mean(numeric_only=True).reset_index()
    print(summary)
    summary.to_csv(BASE_DIR / "summary_techqa.csv", index=False)
    (BASE_DIR / "summary_techqa.json").write_text(summary.to_json(orient="records", indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
