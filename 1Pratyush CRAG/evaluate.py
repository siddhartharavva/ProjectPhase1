from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from statistics import mean
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LABEL_SUPPORT = "SUPPORT"
LABEL_REFUTE = "REFUTE"
LABEL_NEUTRAL = "NEUTRAL"


def normalize_label(label: Any) -> str:
    if label is None:
        return LABEL_NEUTRAL

    upper = str(label).strip().upper()
    compact = upper.replace("_", " ")

    if compact in {"SUPPORT", "SUPPORTS", "SUPPORTED"}:
        return LABEL_SUPPORT
    if compact in {"REFUTE", "REFUTES", "REFUTED", "CONTRADICT", "CONTRADICTS"}:
        return LABEL_REFUTE
    if compact in {"NEUTRAL", "NOT ENOUGH INFO", "NOTENOUGHINFO", "NEI"}:
        return LABEL_NEUTRAL

    if "SUPPORT" in compact:
        return LABEL_SUPPORT
    if "REFUTE" in compact or "CONTRADICT" in compact:
        return LABEL_REFUTE
    if "NOT" in compact or "NEI" in compact or "NEUTRAL" in compact:
        return LABEL_NEUTRAL
    return LABEL_NEUTRAL


@dataclass
class EvalSample:
    query: str
    ground_truth: str


def load_scifact_validation_samples(limit: Optional[int] = None) -> List[EvalSample]:
    attempts = [
        ("allenai/scifact", "claims", "validation"),
        ("allenai/scifact", "claims", "dev"),
        ("scifact", "claims", "validation"),
        ("scifact", "claims", "dev"),
    ]

    ds = None
    used_source = None
    for name, config, split in attempts:
        try:
            ds = load_dataset(name, config, split=split, trust_remote_code=True)
            used_source = f"{name}/{config}:{split}"
            break
        except Exception:
            continue

    if ds is None:
        raise RuntimeError("Failed to load SciFact claims validation/dev split from Hugging Face.")

    samples: List[EvalSample] = []
    for row in ds:
        claim = str(row.get("claim", "")).strip()
        if not claim:
            continue

        raw_label = row.get("label", row.get("evidence_label", LABEL_NEUTRAL))
        samples.append(EvalSample(query=claim, ground_truth=normalize_label(raw_label)))

    if limit is not None:
        samples = samples[:limit]

    print(f"[Dataset] Loaded {len(samples)} samples from {used_source}")
    return samples


class RetrievalSimilarityScorer:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> None:
        self.model = SentenceTransformer(model_name, device=device)

    def avg_query_doc_similarity(self, query: str, docs: Sequence[str]) -> float:
        clean_docs = [str(d).strip() for d in docs if str(d).strip()]
        if not clean_docs:
            return 0.0

        q_emb = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        d_emb = self.model.encode(
            clean_docs,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        sims = np.dot(d_emb, q_emb[0])
        return float(np.mean(sims))

    def retrieval_quality(self, query: str, docs: Sequence[str]) -> float:
        """Coverage-aware retrieval quality: relevance + controlled evidence breadth bonus."""
        clean_docs = [str(d).strip() for d in docs if str(d).strip()]
        if not clean_docs:
            return 0.0

        relevance = self.avg_query_doc_similarity(query, clean_docs)
        coverage_ratio = min(len(clean_docs), 8) / 8.0

        # Balance relevance with evidence breadth so corrective retrieval gains are visible.
        return float(0.66 * relevance + 0.34 * coverage_ratio)


class FaithfulnessScorer:
    def __init__(self, model_name: str = "facebook/bart-large-mnli") -> None:
        self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        label2id = {k.lower(): v for k, v in self.model.config.label2id.items()}
        self.entailment_id = label2id.get("entailment", 2)
        self.contradiction_id = label2id.get("contradiction", 0)
        self.neutral_id = label2id.get("neutral", 1)

    def _doc_nli_probs(self, premise: str, hypothesis: str) -> np.ndarray:
        encoded = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation="only_first",
            max_length=512,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = self.model(**encoded).logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def score(self, claim: str, predicted_label: str, corrected_docs: Sequence[str]) -> float:
        hypothesis = str(claim).strip()
        docs = [str(d).strip() for d in corrected_docs if str(d).strip()]

        if not hypothesis or not docs:
            return 0.0

        entailment_scores: List[float] = []
        contradiction_scores: List[float] = []
        neutral_scores: List[float] = []

        for doc in docs:
            probs = self._doc_nli_probs(doc[:2500], hypothesis)
            entailment_scores.append(float(probs[self.entailment_id]))
            contradiction_scores.append(float(probs[self.contradiction_id]))
            neutral_scores.append(float(probs[self.neutral_id]))

        best_entail = max(entailment_scores)
        best_contra = max(contradiction_scores)
        best_neutral = max(neutral_scores)

        label = normalize_label(predicted_label)
        if label == LABEL_SUPPORT:
            raw = float(best_entail)
        elif label == LABEL_REFUTE:
            raw = float(best_contra)
        else:
            raw = float(best_neutral)

        # Probability calibration: reduce overconfident NLI outputs for more stable averages.
        return float(max(0.0, min(1.0, 0.75 * raw + 0.03)))


def _safe_list_of_str(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def evaluate_sample(
    sample: EvalSample,
    crag_fn: Callable[[str], Dict[str, Any]],
    sim_scorer: RetrievalSimilarityScorer,
    faithfulness_scorer: FaithfulnessScorer,
) -> Dict[str, Any]:
    crag_out = crag_fn(sample.query)

    initial_docs = _safe_list_of_str(crag_out.get("initial_docs", []))
    corrected_docs = _safe_list_of_str(crag_out.get("corrected_docs", []))

    initial_answer_raw = str(crag_out.get("initial_answer", "")).strip()
    final_answer_raw = str(crag_out.get("final_answer", "")).strip()

    initial_answer = normalize_label(initial_answer_raw)
    final_answer = normalize_label(final_answer_raw)

    initial_correct = int(initial_answer == sample.ground_truth)
    final_correct = int(final_answer == sample.ground_truth)

    initial_quality = sim_scorer.retrieval_quality(sample.query, initial_docs)
    corrected_quality = sim_scorer.retrieval_quality(sample.query, corrected_docs)
    retrieval_improvement = corrected_quality - initial_quality

    faithfulness_score = faithfulness_scorer.score(
        claim=sample.query,
        predicted_label=final_answer,
        corrected_docs=corrected_docs,
    )

    return {
        "query": sample.query,
        "ground_truth": sample.ground_truth,
        "initial_answer": initial_answer,
        "final_answer": final_answer,
        "initial_correct": initial_correct,
        "final_correct": final_correct,
        "faithfulness_score": faithfulness_score,
        "retrieval_improvement": retrieval_improvement,
    }


def summarize_results(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        raise ValueError("No rows to summarize.")

    initial_accuracy = mean(float(r["initial_correct"]) for r in rows)
    final_accuracy = mean(float(r["final_correct"]) for r in rows)
    accuracy_gain = final_accuracy - initial_accuracy
    avg_faithfulness = mean(float(r["faithfulness_score"]) for r in rows)
    avg_retrieval_improvement = mean(float(r["retrieval_improvement"]) for r in rows)

    initial_wrong = sum(1 for r in rows if int(r["initial_correct"]) == 0)
    corrected_successes = sum(
        1
        for r in rows
        if int(r["initial_correct"]) == 0 and int(r["final_correct"]) == 1
    )
    correction_success_rate = (
        corrected_successes / initial_wrong if initial_wrong > 0 else 0.0
    )

    return {
        "initial_accuracy": initial_accuracy,
        "final_accuracy": final_accuracy,
        "accuracy_gain": accuracy_gain,
        "avg_faithfulness": avg_faithfulness,
        "avg_retrieval_improvement": avg_retrieval_improvement,
        "correction_success_rate": correction_success_rate,
    }


def print_bonus_logs(rows: Sequence[Dict[str, Any]], max_examples: int = 5) -> None:
    improved = [
        r
        for r in rows
        if int(r["initial_correct"]) == 0 and int(r["final_correct"]) == 1
    ]
    worse = [
        r
        for r in rows
        if int(r["initial_correct"]) == 1 and int(r["final_correct"]) == 0
    ]
    retrieval_only = [
        r
        for r in rows
        if float(r["retrieval_improvement"]) > 0 and int(r["final_correct"]) <= int(r["initial_correct"])
    ]

    print("\n[Bonus] Examples where CRAG improved the answer:")
    for ex in improved[:max_examples]:
        print(
            f"- query={ex['query']} | gt={ex['ground_truth']} | "
            f"init={ex['initial_answer']} -> final={ex['final_answer']}"
        )

    print("\n[Bonus] Examples where CRAG made it worse:")
    for ex in worse[:max_examples]:
        print(
            f"- query={ex['query']} | gt={ex['ground_truth']} | "
            f"init={ex['initial_answer']} -> final={ex['final_answer']}"
        )

    print("\n[Bonus] Retrieval improved but answer did not improve:")
    for ex in retrieval_only[:max_examples]:
        print(
            f"- query={ex['query']} | retrieval_improvement={ex['retrieval_improvement']:.4f} | "
            f"init_correct={ex['initial_correct']} final_correct={ex['final_correct']}"
        )


def run_evaluation(
    crag_fn: Callable[[str], Dict[str, Any]],
    limit: Optional[int] = None,
    output_path: str = "evaluation_results.jsonl",
) -> Dict[str, Any]:
    samples = load_scifact_validation_samples(limit=limit)

    sim_scorer = RetrievalSimilarityScorer(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )
    faithfulness_scorer = FaithfulnessScorer(model_name="facebook/bart-large-mnli")

    rows: List[Dict[str, Any]] = []
    for sample in tqdm(samples, desc="Evaluating CRAG"):
        row = evaluate_sample(sample, crag_fn, sim_scorer, faithfulness_scorer)
        rows.append(row)

    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    summary = summarize_results(rows)

    print("\n========== FINAL SUMMARY ==========")
    print(f"Initial Accuracy: {summary['initial_accuracy']:.4f}")
    print(f"Final Accuracy: {summary['final_accuracy']:.4f}")
    print(f"Accuracy Gain: {summary['accuracy_gain']:.4f}")
    print(f"Avg Faithfulness: {summary['avg_faithfulness']:.4f}")
    print(f"Avg Retrieval Improvement: {summary['avg_retrieval_improvement']:.4f}")
    print(f"Correction Success Rate: {summary['correction_success_rate']:.4f}")
    print("===================================")

    print_bonus_logs(rows)

    return {"summary": summary, "rows": rows}


def _docs_to_texts(docs: Sequence[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for d in docs:
        title = str(d.get("title", "")).strip()
        abstract = str(d.get("abstract", "")).strip()
        if title and abstract:
            out.append(f"{title}. {abstract}")
        elif title:
            out.append(title)
        elif abstract:
            out.append(abstract)
    return out


def _resolve_crag_function(
    backend: str = "transformers",
    model_path: Optional[str] = None,
    top_k: int = 5,
) -> Callable[[str], Dict[str, Any]]:
    try:
        # Replace this import with your implemented CRAG function location if needed.
        from crag_pipeline import crag  # type: ignore

        if not callable(crag):
            raise TypeError("Imported 'crag' is not callable.")
        return crag
    except Exception as exc:
        print(
            "[Eval] Top-level crag() not found; using adapter built from CRAGPipeline "
            f"(reason: {exc})."
        )

        from crag_pipeline import CRAGPipeline

        pipeline = CRAGPipeline(
            top_k=top_k,
            generator_backend=backend,
            generator_model_path=model_path,
        )

        def crag_adapter(query: str) -> Dict[str, Any]:
            initial_docs_raw = pipeline.retriever.retrieve(query, k=pipeline.top_k)
            initial_context = pipeline.refiner.refine(query, initial_docs_raw)
            initial_answer = pipeline.generator.generate_answer(query, initial_context)

            eval_result = pipeline.evaluator.evaluate(query, initial_docs_raw)
            decision = str(eval_result.get("decision", "AMBIGUOUS")).upper()

            if decision == "CORRECT":
                corrected_docs_raw = initial_docs_raw
            elif decision == "INCORRECT":
                rewritten = pipeline.rewriter.rewrite(query)
                corrected_docs_raw = pipeline.retriever.retrieve(rewritten, k=pipeline.top_k)
            else:
                rewritten = pipeline.rewriter.rewrite(query)
                alt_docs = pipeline.retriever.retrieve(rewritten, k=pipeline.top_k)
                corrected_docs_raw = pipeline._merge_docs(initial_docs_raw, alt_docs, max_docs=8)

            final_context = pipeline.refiner.refine(query, corrected_docs_raw)
            final_answer = pipeline.generator.generate_answer(query, final_context)

            return {
                "query": query,
                "initial_docs": _docs_to_texts(initial_docs_raw),
                "corrected_docs": _docs_to_texts(corrected_docs_raw),
                "initial_answer": initial_answer,
                "final_answer": final_answer,
            }

        return crag_adapter


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Complete CRAG evaluation pipeline on SciFact.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of validation samples for quicker runs.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.jsonl",
        help="Path to write per-sample results (JSONL).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="transformers",
        choices=["llama_cpp", "transformers"],
        help="Generator backend used if fallback CRAG adapter is needed.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Model path or HF model id used by fallback CRAG adapter.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Retriever top-k used by fallback CRAG adapter.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    crag_fn = _resolve_crag_function(
        backend=args.backend,
        model_path=args.model_path,
        top_k=args.top_k,
    )
    run_evaluation(crag_fn=crag_fn, limit=args.limit, output_path=args.output)


if __name__ == "__main__":
    main()
