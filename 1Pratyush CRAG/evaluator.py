from __future__ import annotations

from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer


class RetrievalEvaluator:
    """Lightweight retrieval evaluator for CRAG decision routing."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        high_threshold: float = 0.68,
        low_threshold: float = 0.45,
    ) -> None:
        self.encoder = SentenceTransformer(model_name)
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold

    def evaluate(self, query: str, docs: List[Dict]) -> Dict:
        if not docs:
            return {"scores": [], "decision": "INCORRECT"}

        query_vec = self.encoder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        doc_texts = [f"{d.get('title', '')}. {d.get('abstract', '')}" for d in docs]
        doc_vecs = self.encoder.encode(doc_texts, normalize_embeddings=True, convert_to_numpy=True)

        # Cosine similarity on normalized vectors equals dot product.
        sims = np.dot(doc_vecs, query_vec[0]).tolist()

        score_items = []
        for d, s in zip(docs, sims):
            score_items.append({"doc_id": d.get("doc_id"), "score": float(s)})

        max_score = max(sims)
        if max_score >= self.high_threshold:
            decision = "CORRECT"
        elif max_score < self.low_threshold:
            decision = "INCORRECT"
        else:
            decision = "AMBIGUOUS"

        print("[Evaluator] Document relevance scores:")
        for s in score_items:
            print(f"  doc_id={s['doc_id']} score={s['score']:.4f}")
        print(f"[Evaluator] Decision: {decision} (max_score={max_score:.4f})")

        return {"scores": score_items, "decision": decision}
