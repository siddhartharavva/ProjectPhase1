from __future__ import annotations

import argparse
import gc
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from techqa_common import (  # noqa: E402
    build_result_row,
    build_rag_prompt,
    build_techqa_corpus_and_samples,
    configure_runtime,
    filter_answerable_samples,
    finalize_with_bertscore,
    generate_from_causal_lm,
    load_causal_lm,
    load_techqa_split,
    save_run_outputs,
    tokenize_for_bm25,
)


DATASET_NAME = "nvidia/TechQA-RAG-Eval"
TOP_K = 3
LIMIT = 50
MODELS = [
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
]


configure_runtime()


class BM25Retriever:
    def __init__(self, corpus: List[Dict[str, str]]) -> None:
        self.corpus = corpus
        self.bm25 = BM25Okapi([tokenize_for_bm25(doc["text"]) for doc in corpus])

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Dict[str, str]]:
        scores = self.bm25.get_scores(tokenize_for_bm25(query))
        top_indices = np.argsort(scores)[::-1][:top_k]
        results: List[Dict[str, str]] = []
        for rank, idx in enumerate(top_indices, start=1):
            doc = dict(self.corpus[int(idx)])
            doc["retrieval_score"] = float(scores[int(idx)])
            doc["rank"] = rank
            results.append(doc)
        return results


def generate_answer(tokenizer, model, question: str, docs: List[Dict[str, str]]) -> str:
    prompt = build_rag_prompt(question, docs, max_docs=len(docs))
    prediction = generate_from_causal_lm(
        tokenizer,
        model,
        prompt,
        max_new_tokens=160,
        do_sample=True,
        temperature=0.3,
        top_p=0.9,
    )
    return re.sub(r"\[/?INST\]|</s>", "", prediction).strip()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BM25 RAG evaluation on TechQA.")
    parser.add_argument("--limit", type=int, default=LIMIT)
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--skip_bertscore", action="store_true")
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parent))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    split = load_techqa_split(split="train", dataset_name=DATASET_NAME)
    corpus, samples = build_techqa_corpus_and_samples(split)
    samples = filter_answerable_samples(samples, args.limit)

    retriever = BM25Retriever(corpus)
    rows: List[Dict[str, object]] = []

    for model_name in MODELS:
        print(f"\n{'=' * 60}\nModel: {model_name}\n{'=' * 60}")
        tokenizer, model = load_causal_lm(model_name, use_4bit=args.use_4bit)

        for index, sample in enumerate(samples):
            docs = retriever.retrieve(sample["question"], top_k=args.top_k)
            prediction = generate_answer(tokenizer, model, sample["question"], docs)
            row = build_result_row(
                rag_variant="bm25_rag",
                question_id=sample["qid"],
                question=sample["question"],
                gold_answer=sample["answer"],
                prediction=prediction,
                retrieved_docs=docs,
                relevant_ids=sample["relevant_ids"],
                generator_model=model_name,
            )
            rows.append(row)

            print(
                f"[{index}] EM:{row['exact_match']:.0f} "
                f"F1:{row['token_f1']:.2f} "
                f"MRR:{row['mrr']:.2f} "
                f"Q: {sample['question'][:60]}"
            )

        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    if not args.skip_bertscore:
        finalize_with_bertscore(rows)
    save_run_outputs(
        rows,
        args.output_dir,
        results_stem="bm25_techqa_results",
        summary_stem="bm25_techqa_summary",
    )
    print("\nSaved TechQA BM25 RAG results and summary.")


if __name__ == "__main__":
    main()
