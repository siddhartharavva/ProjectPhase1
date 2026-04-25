from __future__ import annotations

from typing import Dict, List, Optional

from dataset import SciFactDataset
from evaluator import RetrievalEvaluator
from generator import QwenGenerator
from refiner import KnowledgeRefiner
from retriever import Retriever
from rewriter import QueryRewriter
from verifier import ClaimVerifier


class CRAGPipeline:
    """Modular Corrective RAG pipeline for SciFact claim verification."""

    def __init__(
        self,
        top_k: int = 5,
        generator_backend: str = "llama_cpp",
        generator_model_path: Optional[str] = None,
    ) -> None:
        self.top_k = top_k

        self.dataset = SciFactDataset()
        self.retriever = Retriever()
        self.evaluator = RetrievalEvaluator()
        self.refiner = KnowledgeRefiner()
        self.rewriter = QueryRewriter()
        self.generator = QwenGenerator(
            backend=generator_backend,
            model_path=generator_model_path,
            max_new_tokens=128,
            context_token_limit=512,
        )
        self.verifier = ClaimVerifier()

        corpus = self.dataset.get_corpus()
        self.retriever.build_index(corpus)

    @staticmethod
    def _merge_docs(primary: List[Dict], secondary: List[Dict], max_docs: int = 8) -> List[Dict]:
        merged: List[Dict] = []
        seen = set()
        for d in primary + secondary:
            doc_id = d.get("doc_id")
            if doc_id in seen:
                continue
            seen.add(doc_id)
            merged.append(d)
            if len(merged) >= max_docs:
                break
        return merged

    def run_claim(self, claim: str) -> Dict:
        print("\n=== CRAG PIPELINE START ===")
        print(f"[Pipeline] Claim: {claim}")

        initial_docs = self.retriever.retrieve(claim, k=self.top_k)
        baseline_docs = initial_docs[: min(2, len(initial_docs))]
        initial_eval = self.evaluator.evaluate(claim, initial_docs)
        initial_meta = self.verifier.predict_with_scores(claim, baseline_docs)
        baseline_context = self.refiner.refine(claim, baseline_docs)
        initial_label = self.generator.generate_answer(claim, baseline_context)

        rewritten = self.rewriter.rewrite(claim)
        rewritten_docs = self.retriever.retrieve(rewritten, k=self.top_k)
        rewritten_eval = self.evaluator.evaluate(claim, rewritten_docs)

        merged_docs = self._merge_docs(initial_docs, rewritten_docs, max_docs=8)
        merged_eval = self.evaluator.evaluate(claim, merged_docs)

        candidates = [
            {
                "name": "initial",
                "query": claim,
                "docs": initial_docs,
                "eval": initial_eval,
                "meta": initial_meta,
            },
            {
                "name": "rewritten",
                "query": rewritten,
                "docs": rewritten_docs,
                "eval": rewritten_eval,
                "meta": self.verifier.predict_with_scores(claim, rewritten_docs),
            },
            {
                "name": "merged",
                "query": f"{claim} | rewritten: {rewritten}",
                "docs": merged_docs,
                "eval": merged_eval,
                "meta": self.verifier.predict_with_scores(claim, merged_docs),
            },
        ]

        for c in candidates:
            scores = c["eval"].get("scores", [])
            max_retrieval = max((float(s.get("score", 0.0)) for s in scores), default=0.0)
            confidence = float(c["meta"].get("confidence", 0.0))
            label = str(c["meta"].get("label", "NEUTRAL"))
            # Prefer evidence-backed confident labels while still rewarding better retrieval.
            c["selection_score"] = confidence + 0.15 * max_retrieval + (0.02 if label != "NEUTRAL" else 0.0)

        best = max(candidates, key=lambda x: float(x["selection_score"]))
        final_docs = best["docs"]
        used_query = str(best["query"])
        label = str(best["meta"]["label"])
        decision = str(best["eval"].get("decision", "AMBIGUOUS"))

        print(
            "[Pipeline] Candidate selection: "
            + ", ".join(
                [
                    f"{c['name']}[label={c['meta']['label']} conf={float(c['meta']['confidence']):.4f} score={float(c['selection_score']):.4f}]"
                    for c in candidates
                ]
            )
        )
        print(f"[Pipeline] Selected route: {best['name'].upper()} (decision={decision})")

        refined_context = self.refiner.refine(claim, final_docs)
        print(f"[Pipeline] Refined context preview: {refined_context[:400]}")
        print("=== CRAG PIPELINE END ===\n")

        return {
            "claim": claim,
            "used_query": used_query,
            "decision": decision,
            "initial_prediction": initial_label,
            "prediction": label,
            "context": refined_context,
            "initial_docs": baseline_docs,
            "final_docs": final_docs,
        }


_PIPELINE_SINGLETON: Optional[CRAGPipeline] = None


def _docs_to_texts(docs: List[Dict]) -> List[str]:
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


def crag(query: str) -> Dict:
    global _PIPELINE_SINGLETON
    if _PIPELINE_SINGLETON is None:
        _PIPELINE_SINGLETON = CRAGPipeline(top_k=5, generator_backend="transformers")

    result = _PIPELINE_SINGLETON.run_claim(query)
    return {
        "query": query,
        "initial_docs": _docs_to_texts(result.get("initial_docs", [])),
        "corrected_docs": _docs_to_texts(result.get("final_docs", [])),
        "initial_answer": result.get("initial_prediction", "NEUTRAL"),
        "final_answer": result.get("prediction", "NEUTRAL"),
    }
