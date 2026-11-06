"""R2-A: paired observational direction estimation on pre-collected CPU probes.

This module does NOT collect activations or rediscover the published p50c mask.
Each row must be a matched pair of token-aggregated hidden states, in the same
model revision, layer, and token-selection rule. Holdout use is forbidden.
"""
from __future__ import annotations

import numpy as np


class DirectionError(ValueError):
    """Invalid, contaminated, or degenerate paired directional evidence."""


def _paired_arrays(productive, repetitive) -> tuple[np.ndarray, np.ndarray]:
    try:
        p = np.asarray(productive, dtype=np.float64)
        r = np.asarray(repetitive, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise DirectionError("Paired states must be numeric") from exc
    if p.ndim != 2 or p.shape != r.shape or p.shape[0] < 3 or p.shape[1] < 1:
        raise DirectionError("Expected matching [at least 3 pairs, hidden_dim] matrices")
    if not np.isfinite(p).all() or not np.isfinite(r).all():
        raise DirectionError("Non-finite paired states")
    return p, r


def estimate_direction(productive, repetitive, *, pair_ids, split: str) -> dict:
    """Return the normalized R-minus-P mean direction and leave-one-pair-out checks.

    Split is restricted to training/diagnostic data. This is observational
    association, not intervention-based causal proof.
    """
    if split not in ("train", "diagnostic"):
        raise DirectionError("Holdout/validation data must not select a direction")
    p, r = _paired_arrays(productive, repetitive)
    if isinstance(pair_ids, (str, bytes)):
        raise DirectionError("pair_ids must be a sequence of unique strings")
    ids = list(pair_ids)
    if (len(ids) != len(p) or any(not isinstance(i, str) or not i for i in ids)
            or len(set(ids)) != len(ids)):
        raise DirectionError("Missing, duplicate, or mismatched pair IDs")
    differences = r - p
    raw = differences.mean(axis=0)
    norm = float(np.linalg.norm(raw))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise DirectionError("No nonzero reproducible mean contrast direction")
    direction = raw / norm
    leave_out = (differences.sum(axis=0) - differences) / (len(ids) - 1)
    norms = np.linalg.norm(leave_out, axis=1)
    cosines = [
        float(np.clip(np.dot(direction, d) / n, -1.0, 1.0))
        if n > 1e-12 else 0.0
        for d, n in zip(leave_out, norms)
    ]
    return {
        "direction": direction,
        "raw_l2_norm": norm,
        "pair_count": len(ids),
        "loo_min_cosine": min(cosines),
        "loo_mean_cosine": float(np.mean(cosines)),
        "split": split,
        "interpretation": "observational_association_only",
    }
