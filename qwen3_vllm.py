"""
Qwen3 vLLM grant scorer.

Retrieval and section scoring run via configurable vLLM models.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from src.scoring.pipeline import score_application_base

MODEL_NAME = os.environ.get("QWEN3_MODEL", "cyankiwi/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit")
RETRIEVAL_MODEL = os.environ.get("QWEN3_RETRIEVAL_MODEL", MODEL_NAME)
MODEL_A = os.environ.get("QWEN3_MODEL_A", MODEL_NAME)
MODEL_B = os.environ.get("QWEN3_MODEL_B", MODEL_A)
QUANTIZATION = os.environ.get("QWEN3_QUANTIZATION", "none")


class _Scorer:
    def __init__(self, model_name: str = MODEL_NAME):
        from vllm import LLM, SamplingParams
        try:
            from vllm.sampling_params import GuidedDecodingParams
        except ImportError:
            GuidedDecodingParams = None  # type: ignore

        self.model_name = model_name
        self._SamplingParams = SamplingParams
        self._GuidedDecodingParams = GuidedDecodingParams
        print(f"[qwen3_vllm] loading {model_name} via vLLM", flush=True)
        kwargs = {
            "model": model_name,
            "trust_remote_code": True,
            "dtype": "auto",
            "gpu_memory_utilization": float(os.environ.get("QWEN3_GPU_UTIL", "0.9")),
            "max_model_len": int(os.environ.get("QWEN3_MAX_LEN", "32768")),
        }
        if QUANTIZATION and QUANTIZATION.lower() != "none":
            kwargs["quantization"] = QUANTIZATION
        self.llm = LLM(**kwargs)
        self.tokenizer = self.llm.get_tokenizer()

    def generate_json(self, messages: list[dict[str, str]], *, schema: dict[str, Any], max_tokens: int) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        kwargs = {"temperature": 0.1, "top_p": 0.9, "max_tokens": max_tokens}
        if self._GuidedDecodingParams is not None:
            kwargs["guided_decoding"] = self._GuidedDecodingParams(json=schema)
        sampling = self._SamplingParams(**kwargs)
        output = self.llm.generate([prompt], sampling)[0]
        return output.outputs[0].text.strip()


def score_application(
    application: dict[str, Any],
    criteria_path: str | Path,
    *,
    doc_id: str | None = None,
    scorer: _Scorer | None = None,
    retrieval_client: _Scorer | None = None,
    scorer_client_a: _Scorer | None = None,
    scorer_client_b: _Scorer | None = None,
    artifacts_dir: str | Path | None = None,
) -> dict[str, Any]:
    scorer_client_a = scorer_client_a or scorer or _Scorer(model_name=MODEL_A)
    retrieval_client = retrieval_client or (
        scorer_client_a
        if RETRIEVAL_MODEL == getattr(scorer_client_a, "model_name", None)
        else _Scorer(model_name=RETRIEVAL_MODEL)
    )
    scorer_client_b = scorer_client_b or (
        scorer_client_a
        if MODEL_B == getattr(scorer_client_a, "model_name", None)
        else _Scorer(model_name=MODEL_B)
    )
    return score_application_base(
        application=application,
        criteria_path=criteria_path,
        doc_id=doc_id,
        retrieval_client=retrieval_client,
        scorer_client_a=scorer_client_a,
        scorer_client_b=scorer_client_b,
        artifacts_dir=artifacts_dir,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("application_json")
    parser.add_argument("--criteria", default=str(Path(__file__).parent / "criteria_points.json"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    in_path = Path(args.application_json)
    out_path = Path(args.out) if args.out else in_path.with_name(in_path.stem + "_scored.json")
    application = json.loads(in_path.read_text(encoding="utf-8"))
    result = score_application(
        application,
        args.criteria,
        doc_id=in_path.stem,
        artifacts_dir=out_path.parent,
    )
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[qwen3_vllm] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
