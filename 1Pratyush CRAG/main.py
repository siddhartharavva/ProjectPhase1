from __future__ import annotations

import argparse
import json

from crag_pipeline import CRAGPipeline
from evaluate import run_evaluation


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Modular CRAG for SciFact with Qwen 3.8B")
    parser.add_argument("--claim", type=str, default=None, help="Single claim to verify")
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "eval"],
        help="Run one claim or evaluate on labeled SciFact claims",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Retriever top-k")
    parser.add_argument(
        "--backend",
        type=str,
        default="transformers",
        choices=["llama_cpp", "transformers"],
        help="Qwen generator backend",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to Qwen GGUF (llama_cpp) or HF model id/path (transformers)",
    )
    parser.add_argument("--eval_split", type=str, default="validation", help="SciFact claim split")
    parser.add_argument("--eval_limit", type=int, default=20, help="How many labeled claims to evaluate")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    pipeline = CRAGPipeline(
        top_k=args.top_k,
        generator_backend=args.backend,
        generator_model_path=args.model_path,
    )

    if args.mode == "single":
        claim = args.claim or "Aspirin reduces risk of heart attack."
        result = pipeline.run_claim(claim)
        print("\n[Main] Final result:")
        print(json.dumps(result, indent=2, ensure_ascii=True))
    else:
        def _docs_to_texts(docs):
            out = []
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

        def _crag_fn(query: str):
            result = pipeline.run_claim(query)
            return {
                "query": query,
                "initial_docs": _docs_to_texts(result.get("initial_docs", [])),
                "corrected_docs": _docs_to_texts(result.get("final_docs", [])),
                "initial_answer": result.get("initial_prediction", "NEUTRAL"),
                "final_answer": result.get("prediction", "NEUTRAL"),
            }

        metrics = run_evaluation(
            crag_fn=_crag_fn,
            limit=args.eval_limit,
            output_path=f"crag_eval_results_{args.eval_limit}.jsonl",
        )
        print("\n[Main] Evaluation summary:")
        print(json.dumps(metrics.get("summary", {}), indent=2))


if __name__ == "__main__":
    main()
