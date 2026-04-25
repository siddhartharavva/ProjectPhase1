from __future__ import annotations

import re
from typing import List


class QueryRewriter:
    """Rule-based query rewriting used only when retrieval quality is low."""

    STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
        "does",
        "do",
        "did",
        "can",
        "could",
        "should",
        "would",
        "this",
        "these",
        "those",
    }

    def rewrite(self, claim: str) -> str:
        tokens = re.findall(r"[A-Za-z0-9-]+", claim.lower())
        keywords: List[str] = [t for t in tokens if t not in self.STOPWORDS and len(t) > 2]

        # Keep stable order while removing duplicates.
        seen = set()
        deduped = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                deduped.append(kw)

        rewritten = " ".join(deduped[:10]).strip()
        if not rewritten:
            rewritten = claim

        print(f"[Rewriter] Rewritten query: {rewritten}")
        return rewritten
