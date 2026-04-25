from __future__ import annotations

import re
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer


class KnowledgeRefiner:
    """Sentence-level context refinement for CRAG corrective generation."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        keep_top_n: int = 8,
        min_score: float = 0.25,
    ) -> None:
        self.encoder = SentenceTransformer(model_name)
        self.keep_top_n = keep_top_n
        self.min_score = min_score

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p.strip() for p in parts if len(p.strip()) > 10]

    def refine(self, query: str, docs: List[Dict]) -> str:
        chunks: List[Dict] = []
        for doc in docs:
            title = doc.get("title", "")
            abstract = doc.get("abstract", "")
            for sent in self._split_sentences(abstract):
                chunk = f"{title}: {sent}" if title else sent
                chunks.append({"doc_id": doc.get("doc_id"), "text": chunk})

        if not chunks:
            print("[Refiner] No chunks found; returning empty context.")
            return ""

        query_vec = self.encoder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        chunk_vecs = self.encoder.encode(
            [c["text"] for c in chunks],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        sims = np.dot(chunk_vecs, query_vec[0]).tolist()

        scored = []
        for chunk, score in zip(chunks, sims):
            if score >= self.min_score:
                scored.append({"doc_id": chunk["doc_id"], "text": chunk["text"], "score": float(score)})

        scored.sort(key=lambda x: x["score"], reverse=True)
        scored = scored[: self.keep_top_n]

        context = "\n".join([f"- {item['text']}" for item in scored])

        print(f"[Refiner] Kept {len(scored)} chunks for final context.")
        for i, item in enumerate(scored, start=1):
            preview = item["text"][:120].replace("\n", " ")
            print(f"  {i}. score={item['score']:.4f} {preview}")

        return context
