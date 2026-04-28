from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from techqa_common import (  # noqa: E402
    build_dense_retrieval_bundle,
    build_techqa_corpus_and_samples,
    compute_answer_metrics,
    compute_retrieval_metrics,
    finalize_with_bertscore,
    load_techqa_split,
    numeric_summary,
    write_jsonl,
)


DATASET_NAME = "nvidia/TechQA-RAG-Eval"
TOP_K = 3
LIMIT = 150
MODELS = [
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
]


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def load_generator(model_name: str, use_4bit: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
    }
    if use_4bit and torch.cuda.is_available():
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()
    return tokenizer, model


def generate_answer(tokenizer, model, question: str, context: str) -> str:
    prompt = (
        "You are a helpful technical assistant.\n"
        "Answer the question using ONLY the context below.\n"
        "If the answer is not in the context, say: I don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = generated[0][encoded["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def retrieve(bundle: Dict[str, Any], question: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    from techqa_common import dense_retrieve

    return dense_retrieve(bundle, question, top_k=top_k)


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
    samples = [sample for sample in samples if not sample["is_impossible"]][: args.limit]

    dense_bundle = build_dense_retrieval_bundle(corpus, batch_size=args.batch_size)
    results: List[Dict[str, Any]] = []

    for model_name in MODELS:
        print(f"\n{'=' * 60}\nModel: {model_name}\n{'=' * 60}")
        tokenizer, model = load_generator(model_name, use_4bit=args.use_4bit)

        for index, sample in enumerate(samples):
            docs = retrieve(dense_bundle, sample["question"], top_k=args.top_k)
            context = "\n\n".join(doc["text"] for doc in docs)
            prediction = generate_answer(tokenizer, model, sample["question"], context)

            row = {
                "rag_variant": "vanilla_rag",
                "dataset": DATASET_NAME,
                "generator_model": model_name,
                "question_id": sample["qid"],
                "question": sample["question"],
                "gold_answer": sample["answer"],
                "prediction": prediction,
                "retrieved_doc_ids": [doc["doc_id"] for doc in docs],
                "relevant_doc_ids": sorted(sample["relevant_ids"]),
            }
            row.update(compute_retrieval_metrics(row["retrieved_doc_ids"], set(sample["relevant_doc_ids"])))
            row.update(compute_answer_metrics(prediction, sample["answer"]))
            results.append(row)

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
        finalize_with_bertscore(results)
    summary_rows = []
    for model_name, frame in pd.DataFrame(results).groupby("generator_model"):
        row = {"generator_model": model_name, "rag_variant": "vanilla_rag"}
        row.update(numeric_summary(frame.to_dict(orient="records"), exclude=("question", "prediction", "gold_answer", "question_id", "retrieved_doc_ids", "relevant_doc_ids", "dataset", "generator_model", "rag_variant")))
        summary_rows.append(row)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "rag_results_techqa.jsonl", results)
    pd.DataFrame(results).to_csv(output_dir / "rag_results_techqa.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output_dir / "summary_techqa.csv", index=False)
    (output_dir / "summary_techqa.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    print("\nSaved TechQA vanilla RAG results and summary.")


if __name__ == "__main__":
    main()
