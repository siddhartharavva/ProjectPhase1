from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from techqa_common import (  # noqa: E402
    build_context,
    build_dense_retrieval_bundle,
    build_result_row,
    build_techqa_corpus_and_samples,
    configure_runtime,
    filter_answerable_samples,
    finalize_with_bertscore,
    generate_from_causal_lm,
    load_causal_lm,
    load_techqa_split,
    save_run_outputs,
)


DATASET_NAME = "nvidia/TechQA-RAG-Eval"
TOP_K = 3
LIMIT = 150
MODELS = [
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
]


configure_runtime()


def retrieve(bundle: Dict[str, Any], question: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    from techqa_common import dense_retrieve

    return dense_retrieve(bundle, question, top_k=top_k)


def generate_answer(tokenizer, model, question: str, docs: List[Dict[str, Any]]) -> str:
    context = build_context(docs)
    prompt = (
        "You are a helpful technical assistant.\n"
        "Answer the question using ONLY the context below.\n"
        "If the answer is not in the context, say: I don't know.\n"
        "Be concise and factual.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    return generate_from_causal_lm(tokenizer, model, prompt, max_new_tokens=80, do_sample=False)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vanilla RAG evaluation on TechQA.")
    parser.add_argument("--limit", type=int, default=LIMIT)
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--skip_bertscore", action="store_true")
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parent))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    split = load_techqa_split(split="train", dataset_name=DATASET_NAME)
    corpus, samples = build_techqa_corpus_and_samples(split)
    samples = filter_answerable_samples(samples, args.limit)

    dense_bundle = build_dense_retrieval_bundle(corpus, batch_size=args.batch_size)
    rows: List[Dict[str, Any]] = []

    for model_name in MODELS:
        print(f"\n{'=' * 60}\nModel: {model_name}\n{'=' * 60}")
        tokenizer, model = load_causal_lm(model_name, use_4bit=args.use_4bit)

        for index, sample in enumerate(samples):
            docs = retrieve(dense_bundle, sample["question"], top_k=args.top_k)
            prediction = generate_answer(tokenizer, model, sample["question"], docs)
            row = build_result_row(
                rag_variant="vanilla_rag",
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
        results_stem="rag_results_techqa",
        summary_stem="summary_techqa",
    )
    print("\nSaved TechQA vanilla RAG results and summary.")


if __name__ == "__main__":
    main()
