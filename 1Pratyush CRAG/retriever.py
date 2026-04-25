from __future__ import annotations

from typing import Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from dataset import CorpusItem


class Retriever:
    """Embedding retriever over SciFact abstracts using FAISS."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.encoder = SentenceTransformer(model_name)
        self.index = None
        self.doc_ids: List[int] = []
        self.corpus_map: Dict[int, CorpusItem] = {}

    def build_index(self, corpus: List[CorpusItem]) -> None:
        self.corpus_map = {doc.doc_id: doc for doc in corpus}
        self.doc_ids = [doc.doc_id for doc in corpus]
        texts = [f"{doc.title}. {doc.abstract}" for doc in corpus]

        embeddings = self.encoder.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).astype("float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        print(f"[Retriever] FAISS index built with {len(corpus)} docs (dim={dim}).")

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        if self.index is None:
            raise RuntimeError("Retriever index is not built. Call build_index(corpus) first.")

        query_vec = self.encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, idxs = self.index.search(query_vec, k)

        results: List[Dict] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            doc_id = self.doc_ids[idx]
            doc = self.corpus_map[doc_id]
            results.append(
                {
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "abstract": doc.abstract,
                    "retrieval_score": float(score),
                }
            )

        print("[Retriever] Top retrieved documents:")
        for i, doc in enumerate(results, start=1):
            print(
                f"  {i}. doc_id={doc['doc_id']} score={doc['retrieval_score']:.4f} "
                f"title={doc['title'][:80]}"
            )

        return results
