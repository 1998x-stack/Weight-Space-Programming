"""R2-A offline, CPU-only candidate analysis on explicitly diagnostic NPZ probes.

Never loads a model, never writes checkpoints, never uses holdout answers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wsp.read.contrastive_direction import DirectionError, estimate_direction  # noqa: E402
from wsp.read.neuron_score import rank_neurons  # noqa: E402
from wsp.read.layer_attribution import observational_alignment  # noqa: E402
from wsp.repair.sparse_edit import propose_projection_edit  # noqa: E402

NPZ_FIELDS = {
    "productive_states", "repetitive_states",
    "productive_activations", "repetitive_activations", "down_proj_weight",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(probes: Path, manifest_path: Path, *, top_k: int, alpha: float) -> dict:
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise DirectionError("Manifest must be a JSON object")
    if metadata.get("split") != "diagnostic" or metadata.get("allowed_for_selection") is not True:
        raise DirectionError("Only explicit diagnostic selection data are allowed")
    rev = metadata.get("model_revision")
    if not isinstance(rev, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", rev):
        raise DirectionError("Model revision must be an exact 40-character commit")
    if not isinstance(metadata.get("tensor_name"), str) or not metadata["tensor_name"]:
        raise DirectionError("Missing target tensor_name")
    if not isinstance(metadata.get("token_selection"), str) or not metadata["token_selection"]:
        raise DirectionError("Missing matched token selection declaration")
    with np.load(probes, allow_pickle=False) as f:
        if set(f.files) != NPZ_FIELDS:
            raise DirectionError("Unexpected or missing NPZ keys: no labels/holdout allowed")
        p, r = f["productive_states"], f["repetitive_states"]
        pa, ra = f["productive_activations"], f["repetitive_activations"]
        w = f["down_proj_weight"]
    if (not isinstance(metadata.get("weight_sha256"), str)
            or metadata["weight_sha256"] != hashlib.sha256(w.tobytes(order="C")).hexdigest()):
        raise DirectionError("Weight bytes differ from manifest")
    if metadata.get("weight_shape") != list(w.shape) or metadata.get("weight_dtype") != str(w.dtype):
        raise DirectionError("Weight dtype/shape differ from manifest")
    if p.shape[0] != pa.shape[0]:
        raise DirectionError("Hidden-state and activation pair counts differ")
    direction = estimate_direction(p, r, pair_ids=metadata.get("pair_ids", []),
                                   split=metadata["split"])
    rank = rank_neurons(w, pa, ra, direction["direction"], top_k=top_k)
    proposal = propose_projection_edit(w, direction["direction"],
                                       rank["selected_columns"], alpha=alpha)
    alignment = observational_alignment(direction["direction"], p, r)
    return {
        "schema_version": "r2a-candidate-v1",
        "status": "candidate_only_not_a_reproduction",
        "provenance": {
            "probes_sha256": sha256(probes),
            "manifest_sha256": sha256(manifest_path),
            "model_revision": rev,
            "tensor_name": metadata["tensor_name"],
            "split": metadata["split"],
            "token_selection": metadata["token_selection"],
        },
        "evidence": {
            "pair_count": direction["pair_count"],
            "raw_direction_norm": direction["raw_l2_norm"],
            "loo_min_cosine": direction["loo_min_cosine"],
            "loo_mean_cosine": direction["loo_mean_cosine"],
            "observational_alignment": alignment["paired_mean_alignment"],
            "causal_supported": False,
            "direction": direction["direction"].tolist(),
        },
        "candidate": {
            "top_k": top_k,
            "selected_columns": rank["selected_columns"],
            "selected_scores": rank["selected_scores"],
            "alpha": proposal["alpha"],
            "changed_entries": proposal["changed_entries"],
            "touched_entries": proposal["touched_entries"],
            "delta_frobenius_norm": proposal["delta_frobenius_norm"],
            "candidate_weight_bytes_sha256": hashlib.sha256(
                proposal["edited_weight"].tobytes(order="C")).hexdigest(),
        },
        "unmet_gates": [
            "real model teacher-forced activations",
            "independent negative controls and held-out evaluation",
            "layer attribution via controlled forward interventions",
            "R1 full GPU benchmark",
        ],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="R2-A CPU-only observational candidate analysis")
    ap.add_argument("--probes", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args(argv)
    try:
        if args.out.exists():
            raise DirectionError("Refusing to overwrite existing result")
        report = run(args.probes, args.manifest, top_k=args.top_k, alpha=args.alpha)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
        print(json.dumps({"output": str(args.out), "status": report["status"],
                          "selected_columns": report["candidate"]["selected_columns"]}))
        return 0
    except (DirectionError, ValueError, KeyError, OSError) as exc:
        print(f"R2-A FAILED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
