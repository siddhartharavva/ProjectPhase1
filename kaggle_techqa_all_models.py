from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import string
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence

import faiss
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForQuestionAnswering, AutoTokenizer, BitsAndBytesConfig


DATASET_NAME = "nvidia/TechQA-RAG-Eval"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
K_VALUES = (1, 3, 5, 7)
OUTPUT_ROOT = Path("/kaggle/working/kaggle_outputs")

VANILLA_MODELS = [
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
]
BM25_MODELS = VANILLA_MODELS
HYBRID_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
CRAG_QA_MODEL = "deepset/roberta-base-squad2"


def configure_runtime() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"[Runtime] CUDA available: {torch.cuda.device_count()} GPU(s)")
        for idx in range(torch.cuda.device_count()):
            print(f"  GPU {idx}: {torch.cuda.get_device_name(idx)}")
    else:
        print("[Runtime] CUDA not available. This will be slow.")


def normalize_text(text: str) -> str:
    text = str(text or "").lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def tokenize_for_bm25(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_+-]+", str(text or "").lower())


def load_techqa_split(split: str = "train"):
    print(f"[Dataset] Loading {DATASET_NAME}:{split}")
    return load_dataset(DATASET_NAME, split=split)


def build_techqa_corpus_and_samples(rows: Iterable[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    docs: List[Dict[str, Any]] = []
    samples: List[Dict[str, Any]] = []
    seen_docs: dict[str, str] = {}

    for row in rows:
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "") or "").strip()
        qid = str(row.get("id", "") or f"q_{len(samples)}").strip()
        is_impossible = bool(row.get("is_impossible", False))
        relevant_ids: set[str] = set()

        for ctx in row.get("contexts", []) or []:
            text = str(ctx.get("text", "")).strip()
            if len(text) < 20:
                continue
            dedupe_key = text[:300]
            doc_id = seen_docs.get(dedupe_key)
            if doc_id is None:
                doc_id = f"tech_{len(docs)}"
                seen_docs[dedupe_key] = doc_id
                docs.append(
                    {
                        "doc_id": doc_id,
                        "title": str(ctx.get("filename") or ctx.get("title") or question[:80] or doc_id),
                        "text": text,
                        "source": "TechQA",
                        "domain": "it_tech",
                    }
                )
            relevant_ids.add(doc_id)

        if question:
            samples.append(
                {
                    "qid": qid,
                    "question": question,
                    "answer": answer,
                    "is_impossible": is_impossible,
                    "relevant_ids": relevant_ids,
                }
            )

    print(f"[Dataset] Corpus docs: {len(docs)}")
    print(f"[Dataset] QA samples: {len(samples)}")
    return docs, samples


def filter_answerable_samples(samples: Sequence[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    answerable = [sample for sample in samples if not sample.get("is_impossible") and sample.get("answer")]
    return answerable[:limit] if limit > 0 else answerable


def encode_texts(
    model: SentenceTransformer,
    texts: Sequence[str],
    *,
    batch_size: int = 64,
    normalize: bool = True,
    show_progress_bar: bool = False,
) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype="float32")
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=show_progress_bar,
    )
    return np.asarray(embeddings, dtype="float32")


def build_faiss_ip_index(embeddings: np.ndarray) -> faiss.Index:
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError(f"Expected non-empty 2D embeddings, got {embeddings.shape}")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


class DenseRetriever:
    def __init__(self, docs: Sequence[Dict[str, Any]], batch_size: int = 96) -> None:
        self.docs = list(docs)
        self.embedder = SentenceTransformer(EMBED_MODEL)
        texts = [doc["text"] for doc in self.docs]
        embeddings = encode_texts(self.embedder, texts, batch_size=batch_size, normalize=True, show_progress_bar=True)
        self.index = build_faiss_ip_index(embeddings)

    def retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        query_vec = encode_texts(self.embedder, [query], batch_size=1, normalize=True)
        scores, idxs = self.index.search(query_vec, top_k)
        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            doc = dict(self.docs[int(idx)])
            doc["retrieval_score"] = float(score)
            results.append(doc)
        return results


class BM25Retriever:
    def __init__(self, docs: Sequence[Dict[str, Any]]) -> None:
        self.docs = list(docs)
        self.bm25 = BM25Okapi([tokenize_for_bm25(doc["text"]) for doc in self.docs])

    def retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        scores = self.bm25.get_scores(tokenize_for_bm25(query))
        top_indices = np.argsort(scores)[::-1][:top_k]
        results: List[Dict[str, Any]] = []
        for rank, idx in enumerate(top_indices, start=1):
            doc = dict(self.docs[int(idx)])
            doc["retrieval_score"] = float(scores[int(idx)])
            doc["rank"] = rank
            results.append(doc)
        return results


class HybridRetriever:
    def __init__(self, docs: Sequence[Dict[str, Any]], batch_size: int = 64) -> None:
        self.docs = list(docs)
        self.dense = DenseRetriever(docs, batch_size=batch_size)
        self.bm25 = BM25Retriever(docs)
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)

    @staticmethod
    def reciprocal_rank_fusion(rank_maps: Sequence[Dict[int, int]], k: int = 60) -> Dict[int, float]:
        all_ids = set()
        for rank_map in rank_maps:
            all_ids.update(rank_map)
        return {
            item_id: sum(1.0 / (k + rank_map[item_id]) for rank_map in rank_maps if item_id in rank_map)
            for item_id in all_ids
        }

    def retrieve(self, query: str, top_k_per_stage: int, rerank_top_k: int) -> List[Dict[str, Any]]:
        dense_docs = self.dense.retrieve(query, top_k=top_k_per_stage)
        dense_rank = {int(doc["doc_id"].split("_")[-1]): rank for rank, doc in enumerate(dense_docs)}

        bm25_scores = self.bm25.bm25.get_scores(tokenize_for_bm25(query))
        bm25_top = np.argsort(bm25_scores)[::-1][:top_k_per_stage]
        bm25_rank = {int(idx): rank for rank, idx in enumerate(bm25_top)}

        fused = self.reciprocal_rank_fusion((dense_rank, bm25_rank))
        top_ids = sorted(fused, key=fused.get, reverse=True)[:top_k_per_stage]
        candidates = [dict(self.docs[idx]) for idx in top_ids]
        if not candidates:
            return []

        scores = self.reranker.predict([[query, doc["text"]] for doc in candidates])
        ranked = [doc for doc, _ in sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)]
        return ranked[:rerank_top_k]


def build_context(docs: Sequence[Dict[str, Any]], max_docs: int = 5, max_chars_per_doc: int = 900) -> str:
    chunks = []
    for index, doc in enumerate(list(docs)[:max_docs], start=1):
        text = str(doc.get("text") or doc.get("abstract") or "").strip()
        if text:
            chunks.append(f"[Document {index}]\n{text[:max_chars_per_doc]}")
    return "\n\n".join(chunks)


def build_rag_prompt(question: str, docs: Sequence[Dict[str, Any]], max_docs: int = 5, max_chars_per_doc: int = 900) -> str:
    context = build_context(docs, max_docs=max_docs, max_chars_per_doc=max_chars_per_doc)
    return (
        "You are a helpful technical assistant.\n"
        "Answer the question using ONLY the context below.\n"
        "If the answer is not in the context, say: I don't know.\n"
        "Be concise and factual.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def load_causal_lm(model_name: str, use_4bit: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs: Dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": True,
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
    }
    if use_4bit and torch.cuda.is_available():
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.max_length = None
    model.eval()
    return tokenizer, model


def generate_from_causal_lm(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int,
    do_sample: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
) -> str:
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    with torch.no_grad():
        generated = model.generate(**encoded, **kwargs)
    new_tokens = generated[0][encoded["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_text(prediction) == normalize_text(reference))


def partial_exact_match(prediction: str, reference: str, n_words: int = 10) -> float:
    pred_norm = normalize_text(prediction)
    ref_tokens = normalize_text(reference).split()
    if not ref_tokens:
        return 0.0
    for start in range(max(1, len(ref_tokens) - n_words + 1)):
        window = " ".join(ref_tokens[start : start + n_words])
        if window and window in pred_norm:
            return 1.0
    return 0.0


def token_precision(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_set = set(normalize_text(reference).split())
    if not pred_tokens:
        return 0.0
    return sum(1 for token in pred_tokens if token in ref_set) / len(pred_tokens)


def token_recall(prediction: str, reference: str) -> float:
    ref_tokens = normalize_text(reference).split()
    pred_set = set(normalize_text(prediction).split())
    if not ref_tokens:
        return 0.0
    return sum(1 for token in ref_tokens if token in pred_set) / len(ref_tokens)


def token_f1(prediction: str, reference: str) -> float:
    precision = token_precision(prediction, reference)
    recall = token_recall(prediction, reference)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def compute_ndcg(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(rank + 1) for rank, doc_id in enumerate(retrieved_ids[:k], 1) if doc_id in relevant_ids)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant_ids), k) + 1))
    return dcg / idcg if idcg > 0 else 0.0


def compute_retrieval_metrics(retrieved_ids: Sequence[str], relevant_ids: set[str]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics["mrr"] = next((1.0 / rank for rank, doc_id in enumerate(retrieved_ids, 1) if doc_id in relevant_ids), 0.0)
    for k in K_VALUES:
        hits = len(set(retrieved_ids[:k]) & relevant_ids)
        metrics[f"precision@{k}"] = hits / k
        metrics[f"recall@{k}"] = hits / len(relevant_ids) if relevant_ids else 0.0
        metrics[f"hit@{k}"] = float(hits > 0)
        metrics[f"ndcg@{k}"] = compute_ndcg(retrieved_ids, relevant_ids, k)
    return metrics


def compute_answer_metrics(prediction: str, reference: str) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    bleu = BLEU(effective_order=True)
    rouge = scorer.score(reference, prediction)
    return {
        "exact_match": exact_match(prediction, reference),
        "partial_exact_match": partial_exact_match(prediction, reference),
        "accuracy": exact_match(prediction, reference),
        "token_precision": token_precision(prediction, reference),
        "token_recall": token_recall(prediction, reference),
        "token_f1": token_f1(prediction, reference),
        "rouge1": rouge["rouge1"].fmeasure,
        "rouge2": rouge["rouge2"].fmeasure,
        "rougeL": rouge["rougeL"].fmeasure,
        "bleu": bleu.sentence_score(prediction, [reference]).score / 100.0,
        "prediction_length": float(len(str(prediction or "").split())),
        "reference_length": float(len(str(reference or "").split())),
    }


def build_result_row(
    rag_variant: str,
    question_id: str,
    question: str,
    gold_answer: str,
    prediction: str,
    retrieved_docs: Sequence[Dict[str, Any]],
    relevant_ids: set[str],
    generator_model: str,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "rag_variant": rag_variant,
        "dataset": DATASET_NAME,
        "generator_model": generator_model,
        "question_id": question_id,
        "question": question,
        "gold_answer": gold_answer,
        "prediction": prediction,
        "retrieved_doc_ids": [str(doc.get("doc_id")) for doc in retrieved_docs],
        "relevant_doc_ids": sorted(str(doc_id) for doc_id in relevant_ids),
    }
    row.update(compute_retrieval_metrics(row["retrieved_doc_ids"], set(row["relevant_doc_ids"])))
    row.update(compute_answer_metrics(prediction, gold_answer))
    return row


def maybe_add_bertscore(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    try:
        from bert_score import score as bert_score_fn

        _, _, f1 = bert_score_fn(
            [str(row["prediction"]) for row in rows],
            [str(row["gold_answer"]) for row in rows],
            lang="en",
            model_type="distilbert-base-uncased",
            verbose=False,
            batch_size=16,
        )
        for row, score in zip(rows, f1.cpu().numpy().tolist()):
            row["bertscore_f1"] = float(score)
    except Exception as exc:
        print(f"[BERTScore] Skipped: {exc}")


def numeric_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    excluded = {
        "question",
        "prediction",
        "gold_answer",
        "question_id",
        "retrieved_doc_ids",
        "relevant_doc_ids",
        "dataset",
        "generator_model",
        "rag_variant",
    }
    keys = sorted({key for row in rows for key, value in row.items() if key not in excluded and isinstance(value, (int, float, np.floating))})
    return {key: mean(float(row[key]) for row in rows if isinstance(row.get(key), (int, float, np.floating))) for key in keys}


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_outputs(rows: Sequence[Dict[str, Any]], output_dir: Path, results_stem: str, summary_stem: str, group_by: str | None = "generator_model") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_list = list(rows)
    write_jsonl(output_dir / f"{results_stem}.jsonl", rows_list)
    pd.DataFrame(rows_list).to_csv(output_dir / f"{results_stem}.csv", index=False)

    summaries: List[Dict[str, Any]]
    if not rows_list:
        summaries = []
    elif group_by and group_by in rows_list[0]:
        summaries = []
        frame = pd.DataFrame(rows_list)
        for value, group in frame.groupby(group_by):
            summary = {group_by: value}
            if "rag_variant" in group.columns:
                summary["rag_variant"] = group["rag_variant"].iloc[0]
            summary.update(numeric_summary(group.to_dict(orient="records")))
            summaries.append(summary)
    else:
        summary = {}
        if "rag_variant" in rows_list[0]:
            summary["rag_variant"] = rows_list[0]["rag_variant"]
        summary.update(numeric_summary(rows_list))
        summaries = [summary]

    pd.DataFrame(summaries).to_csv(output_dir / f"{summary_stem}.csv", index=False)
    (output_dir / f"{summary_stem}.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")


def run_vanilla(corpus: List[Dict[str, Any]], samples: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    print("\n========== VANILLA RAG ==========")
    retriever = DenseRetriever(corpus, batch_size=args.batch_size)
    rows: List[Dict[str, Any]] = []
    for model_name in VANILLA_MODELS:
        tokenizer, model = load_causal_lm(model_name, use_4bit=args.use_4bit)
        for idx, sample in enumerate(samples):
            docs = retriever.retrieve(sample["question"], args.top_k)
            prompt = build_rag_prompt(sample["question"], docs, max_docs=args.top_k)
            prediction = generate_from_causal_lm(tokenizer, model, prompt, max_new_tokens=80, do_sample=False)
            row = build_result_row("vanilla_rag", sample["qid"], sample["question"], sample["answer"], prediction, docs, sample["relevant_ids"], model_name)
            rows.append(row)
            print(f"[vanilla:{idx}] F1={row['token_f1']:.3f} MRR={row['mrr']:.3f}")
        del model, tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
    if not args.skip_bertscore:
        maybe_add_bertscore(rows)
    save_outputs(rows, args.output_dir / "vanilla", "rag_results_techqa", "summary_techqa")


def run_bm25(corpus: List[Dict[str, Any]], samples: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    print("\n========== BM25 RAG ==========")
    retriever = BM25Retriever(corpus)
    rows: List[Dict[str, Any]] = []
    for model_name in BM25_MODELS:
        tokenizer, model = load_causal_lm(model_name, use_4bit=args.use_4bit)
        for idx, sample in enumerate(samples):
            docs = retriever.retrieve(sample["question"], args.top_k)
            prompt = build_rag_prompt(sample["question"], docs, max_docs=args.top_k)
            prediction = generate_from_causal_lm(tokenizer, model, prompt, max_new_tokens=160, do_sample=True, temperature=0.3, top_p=0.9)
            prediction = re.sub(r"\[/?INST\]|</s>", "", prediction).strip()
            row = build_result_row("bm25_rag", sample["qid"], sample["question"], sample["answer"], prediction, docs, sample["relevant_ids"], model_name)
            rows.append(row)
            print(f"[bm25:{idx}] F1={row['token_f1']:.3f} MRR={row['mrr']:.3f}")
        del model, tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
    if not args.skip_bertscore:
        maybe_add_bertscore(rows)
    save_outputs(rows, args.output_dir / "bm25", "bm25_techqa_results", "bm25_techqa_summary")


def run_hybrid(corpus: List[Dict[str, Any]], samples: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    print("\n========== HYBRID RAG ==========")
    retriever = HybridRetriever(corpus, batch_size=args.batch_size)
    tokenizer, model = load_causal_lm(HYBRID_MODEL, use_4bit=args.use_4bit)
    rows: List[Dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        docs = retriever.retrieve(sample["question"], args.top_k_per_stage, args.rerank_top_k)
        prompt = build_rag_prompt(sample["question"], docs, max_docs=5, max_chars_per_doc=1200)
        prediction = generate_from_causal_lm(tokenizer, model, prompt, max_new_tokens=200, do_sample=False)
        row = build_result_row("hybrid_rag", sample["qid"], sample["question"], sample["answer"], prediction, docs, sample["relevant_ids"], HYBRID_MODEL)
        rows.append(row)
        print(f"[hybrid:{idx}] F1={row['token_f1']:.3f} MRR={row['mrr']:.3f}")
    if not args.skip_bertscore:
        maybe_add_bertscore(rows)
    save_outputs(rows, args.output_dir / "hybrid", "hybrid_rag_techqa_results", "hybrid_rag_techqa_summary", group_by=None)


STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "to", "of", "in", "on", "for", "by", "with", "as",
    "is", "are", "was", "were", "be", "been", "being", "do", "does", "did", "can", "could", "should", "would", "will",
    "what", "which", "who", "whom", "where", "when", "why", "how", "that", "this", "these", "those", "it", "its",
    "from", "at", "into", "about", "after", "before", "between", "than", "your", "you", "i", "we", "they", "he", "she",
}


def question_keywords(question: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9-]+", question.lower())
    return [token for token in tokens if token not in STOPWORDS and len(token) > 2]


def split_sentences(text: str) -> List[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", str(text or "").strip()) if len(part.strip()) >= 20]


def rerank_docs(question: str, docs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keywords = set(question_keywords(question))
    scored = []
    for doc in docs:
        text_norm = normalize_text(f"{doc.get('title', '')} {doc.get('text', '')}")
        overlap = len(keywords & set(text_norm.split())) / max(len(keywords), 1)
        score = 0.72 * float(doc.get("retrieval_score", 0.0)) + 0.28 * overlap
        scored.append((score, doc))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored]


def rewrite_query(question: str) -> str:
    keywords = question_keywords(question)
    seen = set()
    deduped = []
    for keyword in keywords:
        if keyword not in seen:
            seen.add(keyword)
            deduped.append(keyword)
    return " ".join(deduped[:10]).strip() or question


def retrieval_quality(docs: Sequence[Dict[str, Any]]) -> float:
    if not docs:
        return 0.0
    scores = [float(doc.get("retrieval_score", 0.0)) for doc in docs]
    return 0.75 * (sum(scores) / len(scores)) + 0.25 * (min(len(scores), 8) / 8.0)


def answer_with_extractive_qa(tokenizer, model, device, question: str, docs: Sequence[Dict[str, Any]]) -> str:
    context = build_context(docs, max_docs=4, max_chars_per_doc=1200)
    if not context.strip():
        return "UNANSWERABLE"
    try:
        encoded = tokenizer(
            question,
            context,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            truncation="only_second",
            max_length=512,
            stride=64,
        )
        best_text = ""
        best_score = -1e9
        for feature_index, input_ids in enumerate(encoded["input_ids"]):
            feature = {"input_ids": torch.tensor([input_ids], device=device)}
            if "attention_mask" in encoded:
                feature["attention_mask"] = torch.tensor([encoded["attention_mask"][feature_index]], device=device)
            if "token_type_ids" in encoded:
                feature["token_type_ids"] = torch.tensor([encoded["token_type_ids"][feature_index]], device=device)
            with torch.no_grad():
                out = model(**feature)
            offsets = encoded["offset_mapping"][feature_index]
            sequence_ids = encoded.sequence_ids(feature_index)
            context_indexes = [idx for idx, sid in enumerate(sequence_ids) if sid == 1]
            for start_idx in context_indexes:
                for end_idx in context_indexes:
                    if end_idx < start_idx or end_idx - start_idx + 1 > 40:
                        continue
                    start_offset = offsets[start_idx]
                    end_offset = offsets[end_idx]
                    if start_offset is None or end_offset is None:
                        continue
                    start_char, _ = start_offset
                    _, end_char = end_offset
                    if end_char <= start_char:
                        continue
                    score = float(out.start_logits[0][start_idx].item() + out.end_logits[0][end_idx].item())
                    if score > best_score:
                        candidate = context[start_char:end_char].strip()
                        if candidate:
                            best_score = score
                            best_text = candidate
        answer = best_text.strip(" .;:\n\t\"")
        return " ".join(answer.split()[:32]) if answer else "UNANSWERABLE"
    except Exception:
        return "UNANSWERABLE"


def run_crag(corpus: List[Dict[str, Any]], samples: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    print("\n========== CRAG ==========")
    retriever = DenseRetriever(corpus, batch_size=args.batch_size)
    qa_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qa_tokenizer = AutoTokenizer.from_pretrained(args.crag_qa_model)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(args.crag_qa_model).to(qa_device)
    qa_model.eval()

    rows: List[Dict[str, Any]] = []
    for idx, sample in enumerate(tqdm(samples, desc="CRAG")):
        initial_docs = rerank_docs(sample["question"], retriever.retrieve(sample["question"], args.top_k))
        initial_quality = retrieval_quality(initial_docs)
        rewritten = sample["question"]
        final_docs = initial_docs
        if initial_quality < 0.55:
            rewritten = rewrite_query(sample["question"])
            rewritten_docs = rerank_docs(sample["question"], retriever.retrieve(rewritten, args.top_k))
            final_docs = rewritten_docs if initial_quality < 0.35 else (initial_docs + rewritten_docs)[:8]

        prediction = answer_with_extractive_qa(qa_tokenizer, qa_model, qa_device, sample["question"], final_docs)
        row = build_result_row("crag", sample["qid"], sample["question"], sample["answer"], prediction, final_docs, sample["relevant_ids"], args.crag_qa_model)
        row["rewritten_query"] = rewritten
        row["retrieval_improvement"] = retrieval_quality(final_docs) - initial_quality
        rows.append(row)
        print(f"[crag:{idx}] F1={row['token_f1']:.3f} MRR={row['mrr']:.3f}")

    if not args.skip_bertscore:
        maybe_add_bertscore(rows)
    save_outputs(rows, args.output_dir / "crag", "techqa_eval_results", "techqa_eval_summary", group_by=None)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-file Kaggle runner for TechQA BM25, Vanilla RAG, Hybrid RAG, and CRAG.")
    parser.add_argument("--mode", choices=["fast", "full"], default="fast")
    parser.add_argument("--models", nargs="+", choices=["vanilla", "bm25", "hybrid", "crag"], default=["vanilla", "bm25", "hybrid", "crag"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_k_per_stage", type=int, default=None)
    parser.add_argument("--rerank_top_k", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--skip_bertscore", action="store_true")
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--crag_qa_model", type=str, default=CRAG_QA_MODEL)
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_ROOT)
    return parser


def apply_mode_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.mode == "fast":
        args.limit = args.limit or 40
        args.top_k = args.top_k or 3
        args.top_k_per_stage = args.top_k_per_stage or 10
        args.rerank_top_k = args.rerank_top_k or 5
        args.batch_size = args.batch_size or 96
        args.skip_bertscore = True if not args.skip_bertscore else args.skip_bertscore
    else:
        args.limit = args.limit or 100
        args.top_k = args.top_k or 3
        args.top_k_per_stage = args.top_k_per_stage or 15
        args.rerank_top_k = args.rerank_top_k or 7
        args.batch_size = args.batch_size or 64
    return args


def main() -> None:
    configure_runtime()
    args = apply_mode_defaults(build_arg_parser().parse_args())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    split = load_techqa_split()
    corpus, all_samples = build_techqa_corpus_and_samples(split)
    samples = filter_answerable_samples(all_samples, args.limit)
    print(f"[Run] Evaluating {len(samples)} answerable samples")
    print(f"[Run] Models: {', '.join(args.models)}")
    print(f"[Run] Output: {args.output_dir}")

    if "vanilla" in args.models:
        run_vanilla(corpus, samples, args)
    if "bm25" in args.models:
        run_bm25(corpus, samples, args)
    if "hybrid" in args.models:
        run_hybrid(corpus, samples, args)
    if "crag" in args.models:
        run_crag(corpus, samples, args)

    elapsed = (time.time() - start) / 60
    print(f"\n[Done] Completed selected runs in {elapsed:.1f} minutes.")


if __name__ == "__main__":
    main()
