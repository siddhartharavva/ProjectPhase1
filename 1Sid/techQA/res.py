from __future__ import annotations

from pathlib import Path

import pandas as pd


RESULTS_PATH = Path(__file__).resolve().parent / "rag_results_techqa.csv"


def main() -> None:
    df = pd.read_csv(RESULTS_PATH)
    numeric_columns = [
        "exact_match",
        "partial_exact_match",
        "token_f1",
        "token_precision",
        "token_recall",
        "rouge1",
        "rouge2",
        "rougeL",
        "bleu",
        "mrr",
        "precision@1",
        "recall@1",
        "hit@1",
        "ndcg@1",
        "precision@3",
        "recall@3",
        "hit@3",
        "ndcg@3",
        "precision@5",
        "recall@5",
        "hit@5",
        "ndcg@5",
        "precision@7",
        "recall@7",
        "hit@7",
        "ndcg@7",
        "bertscore_f1",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    summary = df.groupby("generator_model")[numeric_columns].mean(numeric_only=True).reset_index()
    print("\n===== SUMMARY =====")
    print(summary)
    summary.to_csv(Path(__file__).resolve().parent / "summary_techqa.csv", index=False)


if __name__ == "__main__":
    main()
