from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "kaggle_outputs"

SCRIPT_PATHS = {
    "vanilla": ROOT / "1Sid/techQA/vanilla_rag_techqa.py",
    "bm25": ROOT / "1Harsh/bm25_techqa.py",
    "hybrid": ROOT / "1Aman/Code_review_2/hybrid_rag_techqa.py",
    "crag": ROOT / "1Pratyush CRAG/techqa_evaluate.py",
}


def configure_kaggle_env() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kaggle launcher for all TechQA RAG variants.")
    parser.add_argument("--mode", choices=["fast", "full"], default="fast")
    parser.add_argument("--models", nargs="+", choices=["vanilla", "bm25", "hybrid", "crag"], default=["vanilla", "bm25", "hybrid", "crag"])
    parser.add_argument("--limit", type=int, default=None, help="Override sample limit for all scripts.")
    parser.add_argument("--skip_bertscore", action="store_true")
    parser.add_argument("--use_4bit", action="store_true", help="Enable 4-bit loading where supported.")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    fast_defaults = {
        "vanilla": {"limit": 60, "top_k": 3, "batch_size": 160},
        "bm25": {"limit": 60, "top_k": 3},
        "hybrid": {"limit": 40, "top_k_per_stage": 10, "rerank_top_k": 5, "batch_size": 96},
        "crag": {"limit": 25, "top_k": 4},
    }
    full_defaults = {
        "vanilla": {"limit": 150, "top_k": 3, "batch_size": 128},
        "bm25": {"limit": 100, "top_k": 3},
        "hybrid": {"limit": 100, "top_k_per_stage": 15, "rerank_top_k": 7, "batch_size": 64},
        "crag": {"limit": 50, "top_k": 5},
    }
    defaults = fast_defaults if args.mode == "fast" else full_defaults

    commands: list[list[str]] = []
    for model_name in args.models:
        script = SCRIPT_PATHS[model_name]
        settings = dict(defaults[model_name])
        if args.limit is not None:
            settings["limit"] = args.limit

        output_dir = OUTPUT_ROOT / model_name
        output_dir.mkdir(parents=True, exist_ok=True)

        command = [sys.executable, str(script)]
        for key, value in settings.items():
            command.extend([f"--{key}", str(value)])
        command.extend(["--output_dir", str(output_dir)])

        if args.skip_bertscore or args.mode == "fast":
            command.append("--skip_bertscore")
        if args.use_4bit and model_name in {"vanilla", "bm25"}:
            command.append("--use_4bit")

        if model_name == "crag":
            command.extend(
                [
                    "--output",
                    str(output_dir / "techqa_eval_results.jsonl"),
                    "--summary",
                    str(output_dir / "techqa_eval_summary.json"),
                ]
            )

        commands.append(command)

    return commands


def main() -> None:
    args = build_arg_parser().parse_args()
    configure_kaggle_env()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    commands = build_commands(args)
    manifest = {
        "mode": args.mode,
        "models": args.models,
        "skip_bertscore": args.skip_bertscore or args.mode == "fast",
        "use_4bit": args.use_4bit,
        "commands": commands,
    }
    (OUTPUT_ROOT / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Kaggle TechQA launcher")
    for command in commands:
        print(" ".join(command))

    if args.dry_run:
        return

    start = time.time()
    for command in commands:
        print(f"\n{'=' * 100}\nRunning: {' '.join(command)}\n{'=' * 100}")
        subprocess.run(command, cwd=ROOT, check=True)

    elapsed = time.time() - start
    print(f"\nAll selected TechQA runs completed in {elapsed / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
