from __future__ import annotations

import os
import re
from typing import Optional


DEFAULT_TRANSFORMERS_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class QwenGenerator:
    """
    Qwen 3.8B generator wrapper with two local backends:
    - llama_cpp: recommended for 4GB VRAM with GGUF quantized model.
    - transformers: optional 4-bit backend.
    """

    def __init__(
        self,
        backend: str = "llama_cpp",
        model_path: Optional[str] = None,
        max_new_tokens: int = 128,
        context_token_limit: int = 512,
    ) -> None:
        self.backend = backend
        self.model_path = model_path or os.getenv("QWEN_MODEL_PATH")
        self.max_new_tokens = min(max_new_tokens, 128)
        self.context_token_limit = context_token_limit

        self._llm = None
        self._tokenizer = None
        self._hf_model = None

        if self.backend == "llama_cpp":
            self._init_llama_cpp()
        elif self.backend == "transformers":
            self._init_transformers()
        else:
            raise ValueError("backend must be 'llama_cpp' or 'transformers'")

    def _init_llama_cpp(self) -> None:
        if not self.model_path:
            raise ValueError(
                "Set model_path or QWEN_MODEL_PATH for llama.cpp backend, or switch to "
                "backend='transformers' to use a default HF Qwen model."
            )
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ImportError("Install llama-cpp-python for llama_cpp backend.") from exc

        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=2048,
            n_gpu_layers=24,  # Safe default for 4GB VRAM; lower if OOM.
            n_threads=max(2, os.cpu_count() or 2),
            verbose=False,
        )
        print(f"[Generator] Loaded llama.cpp model from: {self.model_path}")

    def _init_transformers(self) -> None:
        if not self.model_path:
            self.model_path = DEFAULT_TRANSFORMERS_MODEL
            print(f"[Generator] No model_path provided. Using default: {self.model_path}")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Install transformers, bitsandbytes, accelerate for transformers backend."
            ) from exc

        use_cuda = torch.cuda.is_available()

        quantization_config = None
        model_kwargs = {}
        if use_cuda:
            try:
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                model_kwargs["device_map"] = "auto"
                model_kwargs["dtype"] = torch.float16
                print("[Generator] CUDA detected. Loading 4-bit quantized model.")
            except Exception:
                model_kwargs["device_map"] = "auto"
                model_kwargs["dtype"] = torch.float16
                print("[Generator] bitsandbytes unavailable; loading non-quantized model on CUDA.")
        else:
            model_kwargs["device_map"] = "cpu"
            model_kwargs["dtype"] = torch.float32
            print("[Generator] CUDA not detected. Loading model on CPU in float32.")

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
        try:
            self._hf_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=quantization_config,
                **model_kwargs,
            )
        except ImportError as exc:
            if quantization_config is None:
                raise
            # Some environments expose BitsAndBytesConfig but do not have bitsandbytes installed.
            print(f"[Generator] Quantized load failed ({exc}); retrying without quantization.")
            fallback_kwargs = dict(model_kwargs)
            fallback_kwargs.pop("dtype", None)
            self._hf_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=None,
                **fallback_kwargs,
            )
        # Some chat-tuned checkpoints ship sampling params by default; clear them for greedy decode.
        self._hf_model.generation_config.temperature = None
        self._hf_model.generation_config.top_p = None
        self._hf_model.generation_config.top_k = None
        print(f"[Generator] Loaded transformers model from: {self.model_path}")

    def _truncate_context(self, context: str) -> str:
        # Approximate 512-token truncation by words to stay lightweight and backend-agnostic.
        words = context.split()
        if len(words) <= self.context_token_limit:
            return context
        return " ".join(words[: self.context_token_limit])

    @staticmethod
    def _build_prompt(claim: str, context: str) -> str:
        return (
            f"Claim: {claim}\n"
            f"Context: {context}\n\n"
            "Question: Is the claim supported or refuted?\n\n"
            "Answer with one word: SUPPORTED / REFUTED / NOT ENOUGH INFO"
        )

    @staticmethod
    def _build_qa_prompt(question: str, context: str) -> str:
        return (
            "You are a concise technical QA assistant. Use only the provided context.\n"
            "If the answer is not present in the context, answer exactly: UNANSWERABLE.\n\n"
            f"Question: {question}\n"
            f"Context:\n{context}\n\n"
            "Final answer:"
        )

    @staticmethod
    def _normalize_label(text: str) -> str:
        upper = text.upper()

        # Choose the first explicit label mention in generated text.
        candidates = []
        for label, pattern in [
            ("NOT ENOUGH INFO", r"\bNOT\s*ENOUGH\s*INFO\b|\bNEI\b|\bNOTENOUGHINFO\b"),
            ("SUPPORTED", r"\bSUPPORTED\b"),
            ("REFUTED", r"\bREFUTED\b"),
        ]:
            match = re.search(pattern, upper)
            if match:
                candidates.append((match.start(), label))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]

        return "NOT ENOUGH INFO"

    def _generate_llama_cpp(self, prompt: str) -> str:
        out = self._llm.create_completion(
            prompt=prompt,
            max_tokens=self.max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            stop=["\n"],
        )
        return out["choices"][0]["text"].strip()

    def _generate_transformers(self, prompt: str) -> str:
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True).to(self._hf_model.device)
        with torch.no_grad():
            outputs = self._hf_model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = text[len(prompt) :].strip() if text.startswith(prompt) else text.strip()
        return generated

    def generate_answer(self, claim: str, context: str) -> str:
        context = self._truncate_context(context)
        prompt = self._build_prompt(claim, context)

        if self.backend == "llama_cpp":
            raw = self._generate_llama_cpp(prompt)
        else:
            raw = self._generate_transformers(prompt)

        # Keep first line/sentence to avoid verbose spill.
        first_piece = re.split(r"[\n\.]", raw, maxsplit=1)[0].strip()
        label = self._normalize_label(first_piece if first_piece else raw)

        print(f"[Generator] Raw output: {raw}")
        print(f"[Generator] Final label: {label}")
        return label

    def generate_qa_answer(self, question: str, context: str) -> str:
        context = self._truncate_context(context)
        prompt = self._build_qa_prompt(question, context)

        if self.backend == "llama_cpp":
            raw = self._generate_llama_cpp(prompt)
        else:
            raw = self._generate_transformers(prompt)

        answer = raw.strip()
        if answer.lower().startswith("final answer:"):
            answer = answer[len("final answer:") :].strip()

        # Keep the first non-empty line to avoid long-form drift.
        for line in answer.splitlines():
            line = line.strip()
            if line:
                answer = line
                break

        print(f"[Generator-QA] Raw output: {raw}")
        print(f"[Generator-QA] Final answer: {answer}")
        return answer or "UNANSWERABLE"
