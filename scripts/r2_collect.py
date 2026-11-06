"""R2-B: collect local Qwen3.5-0.8B teacher-forced diagnostic arrays.

No network request or weight write; official BF16 base target hashes are mandatory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wsp.adapters.qwen import (  # noqa: E402
    MODEL_REVISION, ProbeError, inspect_official_base, load_local_qwen)
from wsp.read.activation_probe import (  # noqa: E402
    collect, load_diagnostic_pairs, save_probe_bundle)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Pinned Qwen R2-B diagnostic probe")
    parser.add_argument("--model", type=Path, required=True,
                        help="Local official unedited checkpoint, fixed revision")
    parser.add_argument("--pairs", type=Path, required=True,
                        help="Diagnostic-only JSON with no benchmark answers")
    parser.add_argument("--out", type=Path, required=True,
                        help="New directory for NPZ and provenance manifest")
    parser.add_argument("--layer", type=int, choices=(15, 19), required=True)
    parser.add_argument("--last-k", type=int, default=16)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--confirm-compute", action="store_true",
                        help="Acknowledge up to 256 forward passes on a model")
    a = parser.parse_args(argv)
    try:
        if not a.confirm_compute:
            raise ProbeError("Pass --confirm-compute to authorize model inference")
        if a.out.exists():
            raise ProbeError("Refusing to overwrite probe output")
        pairs = load_diagnostic_pairs(a.pairs)
        evidence = inspect_official_base(a.model)
        model, tokenizer = load_local_qwen(a.model, device=a.device)
        arrays, meta = collect(model, tokenizer, pairs, layer_index=a.layer,
                               window=a.last_k, max_tokens=a.max_input_tokens,
                               device=a.device)
        import torch
        import transformers
        meta.update({
            "model_revision": MODEL_REVISION,
            "model_attestation": evidence,
            "model_config_sha256": hashlib.sha256(
                (a.model / "config.json").read_bytes()).hexdigest(),
            "pairs_json_sha256": hashlib.sha256(a.pairs.read_bytes()).hexdigest(),
            "tokenizer_class": type(tokenizer).__name__,
            "tokenization": "raw_input_text_add_special_tokens_no_chat_template",
            "execution": {"device": a.device, "torch_version": torch.__version__,
                          "transformers_version": transformers.__version__},
        })
        npz, manifest = save_probe_bundle(a.out, arrays, meta)
        print(json.dumps({
            "status": "teacher_forced_diagnostic_probes_collected_not_r2_validated",
            "layer": a.layer, "pairs": len(pairs), "npz": str(npz),
            "manifest": str(manifest),
            "model_attestation": evidence["attestation_scope"],
        }))
        return 0
    except (ProbeError, ValueError, KeyError, OSError, ImportError) as exc:
        print(f"R2-B COLLECT FAILED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
