from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from datasets import load_dataset


@dataclass
class ClaimItem:
    claim_id: int
    claim: str
    label: Optional[str] = None


@dataclass
class CorpusItem:
    doc_id: int
    title: str
    abstract: str


class SciFactDataset:
    """Loads SciFact claims/corpus only and exposes normalized accessors."""

    def __init__(self) -> None:
        self._claims: Optional[List[ClaimItem]] = None
        self._corpus: Optional[List[CorpusItem]] = None

    @staticmethod
    def _try_load(name: str, config: str):
        return load_dataset(name, config, trust_remote_code=True)

    def _load_raw_claims(self):
        attempts = [
            ("allenai/scifact", "claims"),
            ("scifact", "claims"),
        ]
        for name, config in attempts:
            try:
                ds = self._try_load(name, config)
                return ds
            except Exception:
                continue
        raise RuntimeError("Could not load SciFact claims split from Hugging Face datasets.")

    def _load_raw_corpus(self):
        attempts = [
            ("allenai/scifact", "corpus"),
            ("scifact", "corpus"),
        ]
        for name, config in attempts:
            try:
                ds = self._try_load(name, config)
                return ds
            except Exception:
                continue
        raise RuntimeError("Could not load SciFact corpus split from Hugging Face datasets.")

    def get_claims(self, split: str = "train") -> List[ClaimItem]:
        if self._claims is not None:
            return self._claims

        raw_claims = self._load_raw_claims()
        if split not in raw_claims:
            # Some SciFact claim configs expose only one split; fallback to first available.
            split = list(raw_claims.keys())[0]

        items: List[ClaimItem] = []
        for row in raw_claims[split]:
            claim_id = int(row.get("id", row.get("claim_id", len(items))))
            claim_text = str(row.get("claim", "")).strip()
            # SciFact claims commonly use evidence_label; empty means NOT ENOUGH INFO.
            label = row.get("label")
            if label is None:
                label = row.get("evidence_label")
            if label is not None:
                label = str(label).strip()
                if not label:
                    label = "NOT ENOUGH INFO"
            items.append(ClaimItem(claim_id=claim_id, claim=claim_text, label=label))

        self._claims = items
        print(f"[Dataset] Loaded {len(items)} claims from split='{split}'.")
        return items

    def get_corpus(self, split: str = "train") -> List[CorpusItem]:
        if self._corpus is not None:
            return self._corpus

        raw_corpus = self._load_raw_corpus()
        if split not in raw_corpus:
            split = list(raw_corpus.keys())[0]

        items: List[CorpusItem] = []
        for row in raw_corpus[split]:
            doc_id = int(row.get("doc_id", row.get("id", len(items))))
            title = str(row.get("title", "")).strip()
            abstract_field = row.get("abstract", "")
            if isinstance(abstract_field, list):
                abstract = " ".join([str(x).strip() for x in abstract_field if str(x).strip()])
            else:
                abstract = str(abstract_field).strip()
            items.append(CorpusItem(doc_id=doc_id, title=title, abstract=abstract))

        self._corpus = items
        print(f"[Dataset] Loaded {len(items)} corpus documents from split='{split}'.")
        return items

    def corpus_as_dict(self) -> Dict[int, CorpusItem]:
        corpus = self.get_corpus()
        return {item.doc_id: item for item in corpus}
