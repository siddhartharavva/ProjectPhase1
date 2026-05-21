from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from sentence_transformers import CrossEncoder

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from techqa_common import (  # noqa: E402
    build_bm25,
    build_dense_retrieval_bundle,
    build_rag_prompt,
    build_result_row,
    build_techqa_corpus_and_samples,
    configure_runtime,
    dense_retrieve,
    filter_answerable_samples,
    finalize_with_bertscore,
    generate_from_causal_lm,
    load_causal_lm,
    load_techqa_split,
    reciprocal_rank_fusion,
    save_run_outputs,
    tokenize_for_bm25,
)


DATASET_NAME = "nvidia/TechQA-RAG-Eval"
GENERATOR_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
TOP_K_PER_STAGE = 15
RERANK_TOP_K = 7
LIMIT = 100


configure_runtime()


def build_hybrid_bundle(corpus: List[Dict[str, str]], batch_size: int = 64) -> Dict[str, object]:
    dense_bundle = build_dense_retrieval_bundle(corpus, batch_size=batch_size)
    bm25 = build_bm25(corpus)
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
    return {
        "dense": dense_bundle,
        "bm25": bm25,
        "docs": corpus,
        "reranker": reranker,
    }


def hybrid_retrieve(
    bundle: Dict[str, object],
    question: str,
    top_k: int = TOP_K_PER_STAGE,
    rerank_top_k: int = RERANK_TOP_K,
) -> List[Dict[str, str]]:
    dense_docs = dense_retrieve(bundle["dense"], question, top_k=top_k)
    dense_rank = {int(doc["doc_id"].split("_")[-1]): rank for rank, doc in enumerate(dense_docs)}

    bm25_scores = bundle["bm25"].get_scores(tokenize_for_bm25(question))
    bm25_top = np.argsort(bm25_scores)[::-1][:top_k]
    bm25_rank = {int(idx): rank for rank, idx in enumerate(bm25_top)}

    fused = reciprocal_rank_fusion((dense_rank, bm25_rank))
    top_ids = sorted(fused, key=fused.get, reverse=True)[:top_k]
    candidates = [dict(bundle["docs"][idx]) for idx in top_ids]

    if not candidates:
        return []
    scores = bundle["reranker"].predict([[question, doc["text"]] for doc in candidates])
    ranked = [doc for doc, _ in sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)]
    return ranked[:rerank_top_k]


def generate_answer(tokenizer, model, question: str, docs: List[Dict[str, str]]) -> str:
    prompt = build_rag_prompt(question, docs, max_docs=5, max_chars_per_doc=1200)
    return generate_from_causal_lm(tokenizer, model, prompt, max_new_tokens=200, do_sample=False)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid RAG evaluation on TechQA.")
    parser.add_argument("--limit", type=int, default=LIMIT)
    parser.add_argument("--top_k_per_stage", type=int, default=TOP_K_PER_STAGE)
    parser.add_argument("--rerank_top_k", type=int, default=RERANK_TOP_K)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--skip_bertscore", action="store_true")
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parent))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    split = load_techqa_split(split="train", dataset_name=DATASET_NAME)
    corpus, samples = build_techqa_corpus_and_samples(split)
    samples = filter_answerable_samples(samples, args.limit)

    bundle = build_hybrid_bundle(corpus, batch_size=args.batch_size)
    tokenizer, model = load_causal_lm(GENERATOR_MODEL, use_4bit=args.use_4bit)
    rows: List[Dict[str, object]] = []

    for index, sample in enumerate(samples):
        docs = hybrid_retrieve(
            bundle,
            sample["question"],
            top_k=args.top_k_per_stage,
            rerank_top_k=args.rerank_top_k,
        )
        prediction = generate_answer(tokenizer, model, sample["question"], docs)
        row = build_result_row(
            rag_variant="hybrid_rag",
            question_id=sample["qid"],
            question=sample["question"],
            gold_answer=sample["answer"],
            prediction=prediction,
            retrieved_docs=docs,
            relevant_ids=sample["relevant_ids"],
            generator_model=GENERATOR_MODEL,
        )
        rows.append(row)
        print(
            f"[{index}] EM:{row['exact_match']:.0f} "
            f"F1:{row['token_f1']:.2f} "
            f"MRR:{row['mrr']:.2f} "
            f"Q: {sample['question'][:60]}"
        )

    if not args.skip_bertscore:
        finalize_with_bertscore(rows)
    save_run_outputs(
        rows,
        args.output_dir,
        results_stem="hybrid_rag_techqa_results",
        summary_stem="hybrid_rag_techqa_summary",
        group_by=None,
    )
    print("\nSaved TechQA hybrid RAG results and summary.")


if __name__ == "__main__":
    main()
