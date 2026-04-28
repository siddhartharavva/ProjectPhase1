from __future__ import annotations

import json
import math
import re
import string
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence

import faiss
import numpy as np
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from sentence_transformers import SentenceTransformer


DATASET_NAME = "nvidia/TechQA-RAG-Eval"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_K_VALUES = (1, 3, 5, 7)


@dataclass
class TechQADocument:
    doc_id: str
    title: str
    text: str
    source: str = "TechQA"
    domain: str = "it_tech"


@dataclass
class TechQASample:
    qid: str
    question: str
    answer: str
    is_impossible: bool
    relevant_ids: set[str]


def normalize_text(text: str) -> str:
    text = str(text or "").lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def tokenize_for_bm25(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_+-]+", str(text or "").lower())


def load_techqa_split(
    split: str = "train",
    data_json: str | None = None,
    dataset_name: str = DATASET_NAME,
):
    if data_json:
        json_path = Path(data_json)
        if not json_path.exists():
            raise FileNotFoundError(f"TechQA JSON file not found: {json_path}")
        dataset = load_dataset("json", data_files={split: str(json_path)})
        return dataset[split]
    return load_dataset(dataset_name, split=split, trust_remote_code=True)


def build_techqa_corpus_and_samples(rows: Iterable[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    docs: List[Dict[str, Any]] = []
    samples: List[Dict[str, Any]] = []
    seen_docs: dict[str, str] = {}

    for row in rows:
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "") or "").strip()
        qid = str(row.get("id", "")).strip()
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
                    asdict(
                        TechQADocument(
                            doc_id=doc_id,
                            title=str(ctx.get("filename") or ctx.get("title") or question[:80] or doc_id),
                            text=text,
                        )
                    )
                )
            relevant_ids.add(doc_id)

        if question:
            samples.append(
                {
                    "qid": qid or f"q_{len(samples)}",
                    "question": question,
                    "answer": answer,
                    "is_impossible": is_impossible,
                    "relevant_ids": relevant_ids,
                }
            )

    return docs, samples


def get_embedder(model_name: str = DEFAULT_EMBED_MODEL, device: str | None = None) -> SentenceTransformer:
    kwargs = {}
    if device:
        kwargs["device"] = device
    return SentenceTransformer(model_name, **kwargs)


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
        raise ValueError(f"Expected non-empty 2D embeddings, got shape={embeddings.shape}")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def build_dense_retrieval_bundle(
    docs: Sequence[Dict[str, Any]],
    *,
    model_name: str = DEFAULT_EMBED_MODEL,
    device: str | None = None,
    batch_size: int = 64,
    text_key: str = "text",
) -> Dict[str, Any]:
    embedder = get_embedder(model_name=model_name, device=device)
    texts = [str(doc.get(text_key, "")) for doc in docs]
    embeddings = encode_texts(embedder, texts, batch_size=batch_size, normalize=True, show_progress_bar=True)
    index = build_faiss_ip_index(embeddings)
    return {
        "embedder": embedder,
        "index": index,
        "texts": texts,
        "docs": list(docs),
        "model_name": model_name,
    }


def build_bm25(docs: Sequence[Dict[str, Any]], text_key: str = "text") -> BM25Okapi:
    return BM25Okapi([tokenize_for_bm25(str(doc.get(text_key, ""))) for doc in docs])


def dense_retrieve(bundle: Dict[str, Any], query: str, top_k: int) -> List[Dict[str, Any]]:
    query_vec = encode_texts(bundle["embedder"], [query], batch_size=1, normalize=True)
    scores, idxs = bundle["index"].search(query_vec, top_k)
    results: List[Dict[str, Any]] = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue
        row = dict(bundle["docs"][int(idx)])
        row["retrieval_score"] = float(score)
        results.append(row)
    return results


def reciprocal_rank_fusion(rank_maps: Sequence[Dict[int, int]], k: int = 60) -> Dict[int, float]:
    all_ids = set()
    for rank_map in rank_maps:
        all_ids.update(rank_map)
    fused: Dict[int, float] = {}
    for item_id in all_ids:
        fused[item_id] = sum(1.0 / (k + rank_map[item_id]) for rank_map in rank_maps if item_id in rank_map)
    return fused


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_text(prediction) == normalize_text(reference))


def partial_exact_match(prediction: str, reference: str, n_words: int = 10) -> float:
    pred_norm = normalize_text(prediction)
    ref_tokens = normalize_text(reference).split()
    if not ref_tokens:
        return 0.0
    max_start = max(1, len(ref_tokens) - n_words + 1)
    for start in range(max_start):
        window = " ".join(ref_tokens[start : start + n_words])
        if window and window in pred_norm:
            return 1.0
    return 0.0


def token_precision(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_set = set(normalize_text(reference).split())
    if not pred_tokens:
        return 0.0
    hits = sum(1 for token in pred_tokens if token in ref_set)
    return hits / len(pred_tokens)


def token_recall(prediction: str, reference: str) -> float:
    ref_tokens = normalize_text(reference).split()
    pred_set = set(normalize_text(prediction).split())
    if not ref_tokens:
        return 0.0
    hits = sum(1 for token in ref_tokens if token in pred_set)
    return hits / len(ref_tokens)


def token_f1(prediction: str, reference: str) -> float:
    precision = token_precision(prediction, reference)
    recall = token_recall(prediction, reference)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_ndcg(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(rank + 1) for rank, doc_id in enumerate(retrieved_ids[:k], 1) if doc_id in relevant_ids)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant_ids), k) + 1))
    return dcg / idcg if idcg > 0 else 0.0


def compute_retrieval_metrics(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    *,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if not relevant_ids:
        metrics["mrr"] = 0.0
        for k in k_values:
            metrics[f"precision@{k}"] = 0.0
            metrics[f"recall@{k}"] = 0.0
            metrics[f"hit@{k}"] = 0.0
            metrics[f"ndcg@{k}"] = 0.0
        return metrics

    metrics["mrr"] = next((1.0 / rank for rank, doc_id in enumerate(retrieved_ids, 1) if doc_id in relevant_ids), 0.0)
    for k in k_values:
        hits = len(set(retrieved_ids[:k]) & relevant_ids)
        metrics[f"precision@{k}"] = hits / k
        metrics[f"recall@{k}"] = hits / len(relevant_ids)
        metrics[f"hit@{k}"] = float(hits > 0)
        metrics[f"ndcg@{k}"] = compute_ndcg(retrieved_ids, relevant_ids, k)
    return metrics


def compute_answer_metrics(prediction: str, reference: str) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    bleu = BLEU(effective_order=True)
    rouge_scores = scorer.score(reference, prediction)

    metrics = {
        "exact_match": exact_match(prediction, reference),
        "partial_exact_match": partial_exact_match(prediction, reference),
        "accuracy": exact_match(prediction, reference),
        "token_precision": token_precision(prediction, reference),
        "token_recall": token_recall(prediction, reference),
        "token_f1": token_f1(prediction, reference),
        "rouge1": rouge_scores["rouge1"].fmeasure,
        "rouge2": rouge_scores["rouge2"].fmeasure,
        "rougeL": rouge_scores["rougeL"].fmeasure,
        "bleu": bleu.sentence_score(prediction, [reference]).score / 100.0,
        "prediction_length": float(len(str(prediction or "").split())),
        "reference_length": float(len(str(reference or "").split())),
    }
    return metrics


def maybe_compute_bertscore(predictions: Sequence[str], references: Sequence[str]) -> List[float] | None:
    if not predictions:
        return []
    try:
        from bert_score import score as bert_score_fn
    except Exception:
        return None

    _, _, f1 = bert_score_fn(
        list(predictions),
        list(references),
        lang="en",
        model_type="distilbert-base-uncased",
        verbose=False,
        batch_size=16,
    )
    return [float(x) for x in f1.cpu().numpy().tolist()]


def finalize_with_bertscore(rows: List[Dict[str, Any]], prediction_key: str = "prediction", reference_key: str = "gold_answer") -> None:
    scores = maybe_compute_bertscore(
        [str(row.get(prediction_key, "")) for row in rows],
        [str(row.get(reference_key, "")) for row in rows],
    )
    if scores is None:
        return
    for row, score in zip(rows, scores):
        row["bertscore_f1"] = score


def numeric_summary(rows: Sequence[Dict[str, Any]], *, exclude: Sequence[str] = ()) -> Dict[str, float]:
    excluded = set(exclude)
    keys = sorted({key for row in rows for key, value in row.items() if key not in excluded and isinstance(value, (int, float, np.floating))})
    summary: Dict[str, float] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float, np.floating))]
        if values:
            summary[key] = mean(values)
    return summary


def write_jsonl(path: str | Path, rows: Sequence[Dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
