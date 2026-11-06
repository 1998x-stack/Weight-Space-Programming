"""R2-B: compare fixed-input logits under a diagnostic-only MLP ablation.

Never reads benchmark answers or writes a model checkpoint. Uses a previously
selected R2-A candidate direction; no automatic search over test data.
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
from wsp.read.activation_probe import load_diagnostic_pairs  # noqa: E402
from wsp.read.causal_intervention import compare_next_token_logits  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fixed-input Qwen MLP output ablation")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--pair-id", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layer", type=int, choices=(15, 19), required=True)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--device", choices=("cpu", "cuda"), required=True)
    ap.add_argument("--confirm-compute", action="store_true")
    args = ap.parse_args(argv)
    try:
        if not args.confirm_compute:
            raise ProbeError("Pass --confirm-compute to authorize model inference")
        if args.out.exists():
            raise ProbeError("Refusing to overwrite result")
        pairs = load_diagnostic_pairs(args.pairs)
        matches = [r for r in pairs if r["id"] == args.pair_id]
        if len(matches) != 1:
            raise ProbeError("Pair ID absent from diagnostic-only corpus")
        candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
        p = candidate.get("provenance", {})
        if (candidate.get("status") != "candidate_only_not_a_reproduction"
                or p.get("model_revision") != MODEL_REVISION
                or p.get("split") != "diagnostic"
                or p.get("tensor_name") !=
                    f"model.language_model.layers.{args.layer}.mlp.down_proj.weight"):
            raise ProbeError("Candidate provenance mismatch; no holdout/test candidates")
        direction = candidate["evidence"]["direction"]
        evidence = inspect_official_base(args.model)
        model, tokenizer = load_local_qwen(args.model, device=args.device)
        import torch
        ids = tokenizer.encode(matches[0]["repetitive"], add_special_tokens=True)
        if not isinstance(ids, list) or not 1 <= len(ids) <= 512:
            raise ProbeError("Invalid/out-of-budget input")
        input_ids = torch.tensor([ids], dtype=torch.long, device=args.device)
        report = compare_next_token_logits(model, input_ids, layer_index=args.layer,
                                           direction=direction, alpha=args.alpha)
        report.update({
            "status": "diagnostic_fixed_input_intervention_not_task_success",
            "pair_id": args.pair_id,
            "prompt_sha256": hashlib.sha256(
                matches[0]["repetitive"].encode("utf-8")).hexdigest(),
            "candidate_json_sha256": hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
            "model_attestation": evidence,
            "model_revision": MODEL_REVISION, "split": "diagnostic",
        })
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
        print(json.dumps({"status": report["status"], "output": str(args.out),
                          "next_token_kl": report["next_token_kl_base_to_intervened"]}))
        return 0
    except (ProbeError, ValueError, KeyError, OSError, ImportError) as exc:
        print(f"R2-B INTERVENTION FAILED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
