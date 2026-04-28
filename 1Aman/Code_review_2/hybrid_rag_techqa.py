from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from techqa_common import (  # noqa: E402
    build_bm25,
    build_dense_retrieval_bundle,
    build_techqa_corpus_and_samples,
    compute_answer_metrics,
    compute_retrieval_metrics,
    dense_retrieve,
    finalize_with_bertscore,
    load_techqa_split,
    numeric_summary,
    reciprocal_rank_fusion,
    tokenize_for_bm25,
    write_jsonl,
)


DATASET_NAME = "nvidia/TechQA-RAG-Eval"
TOP_K_PER_STAGE = 15
RERANK_TOP_K = 7
LIMIT = 100

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


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


def hybrid_retrieve(bundle: Dict[str, object], question: str, top_k: int = TOP_K_PER_STAGE, rerank_top_k: int = RERANK_TOP_K) -> List[Dict[str, str]]:
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


_generator = None


def load_generator():
    global _generator
    if _generator is not None:
        return _generator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    if device == "cuda":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    _generator = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False)
    return _generator


def generate_answer(question: str, docs: List[Dict[str, str]]) -> str:
    generator = load_generator()
    passages = "\n\n".join(f"[{i}] {doc['text'][:1200].strip()}" for i, doc in enumerate(docs[:5], start=1))
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise technical assistant. "
                "Answer questions using ONLY the provided passages. "
                "Be specific and concise."
            ),
        },
        {"role": "user", "content": f"Passages:\n{passages}\n\nQuestion: {question}"},
    ]
    prompt = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    output = generator(prompt, max_new_tokens=200, do_sample=False, temperature=None)[0]["generated_text"].strip()
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid RAG evaluation on TechQA.")
    parser.add_argument("--limit", type=int, default=LIMIT)
    parser.add_argument("--top_k_per_stage", type=int, default=TOP_K_PER_STAGE)
    parser.add_argument("--rerank_top_k", type=int, default=RERANK_TOP_K)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--skip_bertscore", action="store_true")
    parser.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parent))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    split = load_techqa_split(split="train", dataset_name=DATASET_NAME)
    corpus, samples = build_techqa_corpus_and_samples(split)
    samples = [sample for sample in samples if sample["answer"]][: args.limit]

    bundle = build_hybrid_bundle(corpus, batch_size=args.batch_size)
    rows: List[Dict[str, object]] = []

    for index, sample in enumerate(samples):
        docs = hybrid_retrieve(
            bundle,
            sample["question"],
            top_k=args.top_k_per_stage,
            rerank_top_k=args.rerank_top_k,
        )
        prediction = generate_answer(sample["question"], docs)
        row: Dict[str, object] = {
            "rag_variant": "hybrid_rag",
            "dataset": DATASET_NAME,
            "generator_model": "mistralai/Mistral-7B-Instruct-v0.3",
            "question_id": sample["qid"],
            "question": sample["question"],
            "gold_answer": sample["answer"],
            "prediction": prediction,
            "retrieved_doc_ids": [doc["doc_id"] for doc in docs],
            "relevant_doc_ids": sorted(sample["relevant_ids"]),
        }
        row.update(compute_retrieval_metrics(row["retrieved_doc_ids"], set(sample["relevant_ids"])))
        row.update(compute_answer_metrics(prediction, sample["answer"]))
        rows.append(row)
        print(
            f"[{index}] EM:{row['exact_match']:.0f} "
            f"F1:{row['token_f1']:.2f} "
            f"MRR:{row['mrr']:.2f} "
            f"Q: {sample['question'][:60]}"
        )

    if not args.skip_bertscore:
        finalize_with_bertscore(rows)
    summary = {"rag_variant": "hybrid_rag", "generator_model": "mistralai/Mistral-7B-Instruct-v0.3"}
    summary.update(numeric_summary(rows, exclude=("question", "prediction", "gold_answer", "question_id", "retrieved_doc_ids", "relevant_doc_ids", "dataset", "generator_model", "rag_variant")))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "hybrid_rag_techqa_results.jsonl", rows)
    pd.DataFrame(rows).to_csv(out_dir / "hybrid_rag_techqa_results.csv", index=False)
    pd.DataFrame([summary]).to_csv(out_dir / "hybrid_rag_techqa_summary.csv", index=False)
    (out_dir / "hybrid_rag_techqa_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSaved TechQA hybrid RAG results and summary.")


if __name__ == "__main__":
    main()
