from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LABEL_SUPPORT = "SUPPORT"
LABEL_REFUTE = "REFUTE"
LABEL_NEUTRAL = "NEUTRAL"


class ClaimVerifier:
    """NLI-based verifier to classify claim as SUPPORT/REFUTE/NEUTRAL from docs."""

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        max_sentences: int = 48,
        max_nli_sentences: int = 14,
    ) -> None:
        self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        label2id = {k.lower(): v for k, v in self.model.config.label2id.items()}
        self.entailment_id = label2id.get("entailment", 2)
        self.contradiction_id = label2id.get("contradiction", 0)
        self.neutral_id = label2id.get("neutral", 1)
        self.max_sentences = max_sentences
        self.max_nli_sentences = max_nli_sentences
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p.strip() for p in parts if len(p.strip()) >= 20]

    @staticmethod
    def _tokens(text: str) -> List[str]:
        return re.findall(r"[a-z0-9-]+", text.lower())

    @staticmethod
    def _directional_cues(tokens_claim: List[str], tokens_sent: List[str]) -> Tuple[int, int]:
        pairs = [
            ("increase", "decrease"),
            ("increases", "decreases"),
            ("higher", "lower"),
            ("high", "low"),
            ("induces", "suppresses"),
            ("activate", "inhibit"),
            ("activation", "inhibition"),
            ("improves", "worsens"),
            ("accelerates", "delays"),
            ("protects", "vulnerability"),
        ]

        claim_set = set(tokens_claim)
        sent_set = set(tokens_sent)
        support = 0
        contra = 0

        for a, b in pairs:
            claim_has_a = a in claim_set
            claim_has_b = b in claim_set
            sent_has_a = a in sent_set
            sent_has_b = b in sent_set

            if (claim_has_a and sent_has_a) or (claim_has_b and sent_has_b):
                support += 1
            if (claim_has_a and sent_has_b) or (claim_has_b and sent_has_a):
                contra += 1

        return support, contra

    def _to_sentences(self, docs: Sequence[Dict]) -> List[str]:
        sentences: List[str] = []
        for d in docs:
            title = str(d.get("title", "")).strip()
            abstract = str(d.get("abstract", "")).strip()
            source = f"{title}. {abstract}".strip(". ")
            if not source:
                continue
            for sent in self._split_sentences(source):
                sentences.append(sent)
                if len(sentences) >= self.max_sentences:
                    return sentences
        return sentences

    def _nli_probs(self, premise: str, hypothesis: str) -> Tuple[float, float, float]:
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
            probs = torch.softmax(logits, dim=-1)

        entail = float(probs[self.entailment_id].item())
        contra = float(probs[self.contradiction_id].item())
        neutral = float(probs[self.neutral_id].item())
        return entail, contra, neutral

    def predict_with_scores(self, claim: str, docs: Sequence[Dict]) -> Dict[str, float | str]:
        sents = self._to_sentences(docs)
        if not sents:
            return {
                "label": LABEL_NEUTRAL,
                "best_entail": 0.0,
                "best_contra": 0.0,
                "best_neutral": 1.0,
                "confidence": 0.0,
            }

        # Keep only the most query-relevant sentences to avoid unrelated contradictions.
        claim_vec = self.embedder.encode([claim], normalize_embeddings=True, convert_to_numpy=True)
        sent_vecs = self.embedder.encode(sents, normalize_embeddings=True, convert_to_numpy=True)
        sims = sent_vecs @ claim_vec[0]
        ranked = sorted(zip(sents, sims.tolist()), key=lambda x: x[1], reverse=True)
        sents = [s for s, _ in ranked[: self.max_nli_sentences]]

        best_entail = 0.0
        best_contra = 0.0
        best_neutral = 0.0
        entail_scores: List[float] = []
        contra_scores: List[float] = []

        for sent in sents:
            entail, contra, neutral = self._nli_probs(sent, claim)
            entail_scores.append(entail)
            contra_scores.append(contra)
            if entail > best_entail:
                best_entail = entail
            if contra > best_contra:
                best_contra = contra
            if neutral > best_neutral:
                best_neutral = neutral

        entail_top = sorted(entail_scores, reverse=True)[:3]
        contra_top = sorted(contra_scores, reverse=True)[:3]
        entail_mean_top = sum(entail_top) / max(len(entail_top), 1)
        contra_mean_top = sum(contra_top) / max(len(contra_top), 1)

        support_signal = 0.6 * best_entail + 0.4 * entail_mean_top
        contra_signal = 0.6 * best_contra + 0.4 * contra_mean_top

        # Lexical consistency adjustment for low-confidence NLI ties.
        claim_tokens = [t for t in self._tokens(claim) if len(t) > 2]
        lexical_support = 0
        lexical_contra = 0
        for sent, sim in ranked[:3]:
            sent_tokens = [t for t in self._tokens(sent) if len(t) > 2]
            if not sent_tokens or not claim_tokens:
                continue
            overlap = len(set(claim_tokens) & set(sent_tokens)) / max(len(set(claim_tokens)), 1)
            if overlap < 0.35 or sim < 0.45:
                continue
            s_up, c_up = self._directional_cues(claim_tokens, sent_tokens)
            lexical_support += s_up
            lexical_contra += c_up

        if lexical_support >= lexical_contra + 2:
            support_signal += 0.08
        elif lexical_contra >= lexical_support + 2:
            contra_signal += 0.08

        confidence = abs(support_signal - contra_signal)

        # Aggregate signal thresholds reduce single-sentence spike errors.
        if support_signal >= 0.48 and (support_signal - contra_signal) >= 0.04:
            label = LABEL_SUPPORT
        elif contra_signal >= 0.60 and (contra_signal - support_signal) >= 0.10:
            label = LABEL_REFUTE
        else:
            label = LABEL_NEUTRAL

        return {
            "label": label,
            "best_entail": float(best_entail),
            "best_contra": float(best_contra),
            "best_neutral": float(best_neutral),
            "support_signal": float(support_signal),
            "contra_signal": float(contra_signal),
            "confidence": float(max(confidence, 0.0)),
        }

    def predict(self, claim: str, docs: Sequence[Dict]) -> str:
        return str(self.predict_with_scores(claim, docs)["label"])
