"""R2-C one-shot authentic Qwen3.5 CPU/GPU smoke run with sanitized audit only.

Downloads are intentionally OUTSIDE this script. Never releases diagnostic NPZ,
model weights or raw prompt text. It is not mathematical benchmark reproduction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wsp.adapters.qwen import (  # noqa: E402
    MODEL_REVISION, ProbeError, inspect_official_base, load_local_qwen)
from wsp.read.activation_probe import (  # noqa: E402
    collect, load_diagnostic_pairs, save_probe_bundle)
from wsp.read.causal_intervention import compare_next_token_logits  # noqa: E402
from wsp.read.diagnostic_controls import control_directions  # noqa: E402
from r2_offline import run as analyze_offline  # noqa: E402


def _hash_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def execute(model_path: Path, pairs_path: Path, *, layer: int,
            window: int, max_tokens: int, device: str) -> dict:
    """Collect actual teacher-forced activations and compare diagnostic controls.

    Model loading happens exactly once. Only aggregate evidence is returned;
    no prompt strings or raw checkpoint arrays are included in the audit.
    """
    if layer not in (15, 19):
        raise ProbeError("Only layers 15/19 are allowed")
    pairs = load_diagnostic_pairs(pairs_path)
    official = inspect_official_base(model_path)
    model, tokenizer = load_local_qwen(model_path, device=device)
    arrays, meta = collect(model, tokenizer, pairs, layer_index=layer,
                           window=window, max_tokens=max_tokens, device=device)
    import torch
    import transformers
    meta.update({
        "model_revision": MODEL_REVISION,
        "model_attestation": official,
        "source_pairs_sha256": _hash_file(pairs_path),
        "execution": {
            "device": device, "torch": torch.__version__,
            "transformers": transformers.__version__
        }
    })
    # R2-A requires an NPZ/manifest; both are destroyed with the temporary
    # directory. Do not upload these proprietary model-weight copies as artifacts.
    with TemporaryDirectory(prefix="wsp-r2c-") as t:
        dirname = Path(t) / "probe"
        npz, manifest = save_probe_bundle(dirname, arrays, meta)
        result = analyze_offline(npz, manifest, top_k=64, alpha=1)
    direction = np.asarray(result["evidence"]["direction"], dtype=np.float64)
    controls = control_directions(
        arrays["productive_states"], arrays["repetitive_states"],
        direction, seed=20261106)
    ids = tokenizer.encode(pairs[0]["repetitive"], add_special_tokens=True)
    if not 1 <= len(ids) <= max_tokens:
        raise ProbeError("Diagnostic intervention exceeds max tokens")
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    zero = compare_next_token_logits(model, tokens, layer_index=layer,
                                     direction=direction, alpha=0.)
    target = compare_next_token_logits(model, tokens, layer_index=layer,
                                       direction=direction, alpha=1.)
    random = compare_next_token_logits(model, tokens, layer_index=layer,
                                       direction=controls["random_direction"], alpha=1.)
    for metric in (zero, target, random):
        for k in ("next_token_kl_base_to_intervened", "next_token_logit_l2"):
            if not np.isfinite(metric[k]):
                raise ProbeError("Non-finite intervention metric")
    if zero["next_token_logit_l2"] > 1e-6:
        raise ProbeError("Alpha-zero control changed logits")
    # Status is carefully limited: these are manually crafted diagnostic
    # strings, not actual generations observed entering a repetition loop.
    return {
        "schema_version": "r2c-real-smoke-v1",
        "status": "real_qwen_forward_smoke_not_mechanism_replication",
        "evidence_scope": "illustrative_diagnostic_pairs_only_not_observed_model_failure",
        "model_revision": MODEL_REVISION,
        "target_base_bf16_sha256": official["base_target_bf16_sha256"],
        "model_attestation_scope": official["attestation_scope"],
        "model_config_sha256": _hash_file(model_path / "config.json"),
        "diagnostic_pairs_sha256": _hash_file(pairs_path),
        "diagnostic_pair_count": len(pairs),
        "layer": layer,
        "token_window_last_k": window,
        "tokens_per_pair": meta["token_lengths_per_pair"],
        "output_tensor_shape": meta["weight_shape"],
        "model_probe_npz_sha256": result["provenance"]["probes_sha256"],
        "candidate": {
            "top_k": result["candidate"]["top_k"],
            "selected_column_count": len(result["candidate"]["selected_columns"]),
            "selected_columns": result["candidate"]["selected_columns"],
            "loo_min_cosine": result["evidence"]["loo_min_cosine"],
            "loo_mean_cosine": result["evidence"]["loo_mean_cosine"],
            "candidate_direction_l2_norm": float(np.linalg.norm(direction)),
            "random_direction_cosine": controls["random_cosine_with_observed"],
            "label_swapped_direction_cosine":
                controls["label_swapped_cosine_with_observed"],
            "label_swapped_pair_indices": controls["swapped_pair_indices"],
        },
        "fixed_input_last_token": {
            "alpha_zero_kl": zero["next_token_kl_base_to_intervened"],
            "target_kl": target["next_token_kl_base_to_intervened"],
            "random_kl": random["next_token_kl_base_to_intervened"],
            "target_l2": target["next_token_logit_l2"],
            "random_l2": random["next_token_logit_l2"],
        },
        "environment": {"device": device, "torch": torch.__version__,
                        "transformers": transformers.__version__},
        "unmet_gates": [
            "independently observed productive/repetitive model traces",
            "minimum four independent contrast groups and genuine failure labels",
            "position-matched causal interventions and off-target negative controls",
            "untouched holdout generation-level evaluation",
            "R1 full MATH-500 base-versus-edited GPU reproduction"
        ]
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="Official Qwen3.5 minimal R2-C smoke")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--layer", type=int, choices=(15,19), default=15)
    p.add_argument("--last-k", type=int, default=4)
    p.add_argument("--max-input-tokens", type=int, default=128)
    p.add_argument("--device", choices=("cpu","cuda"), required=True)
    p.add_argument("--confirm-compute", action="store_true")
    a = p.parse_args(argv)
    try:
        if not a.confirm_compute:
            raise ProbeError("--confirm-compute is required")
        if a.out.exists():
            raise ProbeError("Refusing to overwrite existing audit")
        report = execute(a.model, a.pairs, layer=a.layer, window=a.last_k,
                         max_tokens=a.max_input_tokens, device=a.device)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with a.out.open("x", encoding="utf8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
        print(json.dumps({"status":report["status"],"audit":str(a.out),
                          "diagnostic_pair_count":report["diagnostic_pair_count"],
                          "target_kl":report["fixed_input_last_token"]["target_kl"]}))
        return 0
    except (ProbeError, ValueError, KeyError, OSError, ImportError) as exc:
        print(f"R2-C NOT VERIFIED: {exc}",file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
