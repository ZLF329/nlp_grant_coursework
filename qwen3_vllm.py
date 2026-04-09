"""
Qwen3 vLLM grant scorer.

Stage 1 runs in-process via vLLM.
Stage 2 faithfulness auditing uses the configured Ollama Gemma model.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from src.scoring.pipeline import score_application_base
from src.verify.faithfulness import OllamaFaithfulnessJudge

MODEL_NAME = os.environ.get("QWEN3_MODEL", "cyankiwi/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit")
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
    artifacts_dir: str | Path | None = None,
) -> dict[str, Any]:
    scorer = scorer or _Scorer()
    judge = OllamaFaithfulnessJudge()
    return score_application_base(
        application=application,
        criteria_path=criteria_path,
        doc_id=doc_id,
        stage1_client=scorer,
        judge=judge,
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
