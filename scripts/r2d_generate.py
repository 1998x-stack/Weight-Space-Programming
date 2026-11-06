"""Run a bounded pilot of genuine Qwen3.5 generations on diagnostic-only prompts.

Requires explicit compute acknowledgement; never overwrites saved traces or
uses MATH-500. Inference artifacts include generated text/token IDs for audit.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from wsp.adapters.qwen import MODEL_REVISION, ProbeError, inspect_official_base, load_local_qwen  # noqa: E402
from wsp.evaluate.trace_generation import load_public_diagnostic_prompts, run_greedy_pilot  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description="R2-D real Qwen diagnostic generation pilot (not MATH-500)")
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--prompts", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--device", choices=("cpu", "cuda"), required=True)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--max-input-tokens", type=int, default=160)
    p.add_argument("--confirm-compute", action="store_true")
    a = p.parse_args(argv)
    try:
        if not a.confirm_compute:
            raise ProbeError("Explicit --confirm-compute is mandatory")
        if a.out.exists():
            raise ProbeError("Refusing to overwrite existing output")
        prompts = load_public_diagnostic_prompts(a.prompts)
        attestation = inspect_official_base(a.model)
        model, tokenizer = load_local_qwen(a.model, device=a.device)
        report = run_greedy_pilot(model, tokenizer, prompts, device=a.device,
                                  max_new_tokens=a.max_new_tokens,
                                  max_input_tokens=a.max_input_tokens)
        import torch
        import transformers
        report["provenance"] = {
            "model_revision_declared": MODEL_REVISION,
            "model_attestation_scope": attestation["attestation_scope"],
            "base_target_bf16_sha256": attestation["base_target_bf16_sha256"],
            "model_config_sha256": hashlib.sha256((a.model/"config.json").read_bytes()).hexdigest(),
            "input_prompts_sha256": hashlib.sha256(a.prompts.read_bytes()).hexdigest(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": a.device,
        }
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with a.out.open("x", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2, allow_nan=False)
            fh.write("\n")
        print(json.dumps({
            "status": report["status"], "output": str(a.out),
            "summary": report["summary"], "max_new_tokens": a.max_new_tokens
        }))
        return 0
    except (ProbeError, ValueError, OSError, KeyError, ImportError) as exc:
        print(f"R2-D GENERATION UNVERIFIED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
