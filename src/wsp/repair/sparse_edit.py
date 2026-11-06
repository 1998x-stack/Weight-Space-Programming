"""R2-A pure CPU sparse projection proposal, not a checkpoint patch writer.

No model file is read or written. This float32/float64 candidate operator is
not the exact BF16 replay used by R1, and cannot be labelled an official edit.
"""
from __future__ import annotations

import numpy as np
from wsp.read.contrastive_direction import DirectionError


def propose_projection_edit(weight, direction, selected_columns, *, alpha: float) -> dict:
    """Return an isolated candidate W' = W - alpha*u*(u.T@W_selected).

    Enforce finite values, unit u, unique in-bounds columns and alpha 0..6.
    Input weights remain byte-identical; unselected output columns must match.
    """
    w = np.asarray(weight)
    if w.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise DirectionError("CPU candidate weights must be float32 or float64")
    try:
        u = np.asarray(direction, dtype=np.float64)
        a = float(alpha)
    except (TypeError, ValueError) as exc:
        raise DirectionError("Direction/alpha must be numeric") from exc
    if (w.ndim != 2 or u.ndim != 1 or w.shape[0] != u.size
            or not np.isfinite(w).all() or not np.isfinite(u).all()
            or not np.isfinite(a) or not 0 <= a <= 6):
        raise DirectionError("Invalid weights, direction, or alpha (allowed 0..6)")
    if not np.isclose(np.linalg.norm(u), 1.0, rtol=1e-6, atol=1e-8):
        raise DirectionError("Direction must have unit L2 norm")
    cols = list(selected_columns)
    if (not cols or any(type(c) is not int or c < 0 or c >= w.shape[1] for c in cols)
            or len(set(cols)) != len(cols)):
        raise DirectionError("Selected columns must be unique in-bounds integers")
    edited = w.copy()
    selected = w[:, cols].astype(np.float64, copy=True)
    projected = u @ selected
    edited[:, cols] = (selected - a * np.outer(u, projected)).astype(w.dtype)
    outside = np.ones(w.shape[1], dtype=bool)
    outside[cols] = False
    if not np.array_equal(edited[:, outside], w[:, outside]):
        raise DirectionError("Unselected columns changed")
    delta = edited.astype(np.float64) - w.astype(np.float64)
    return {
        "edited_weight": edited,
        "selected_columns": cols,
        "alpha": a,
        "touched_entries": len(cols) * w.shape[0],
        "changed_entries": int(np.count_nonzero(delta)),
        "delta_frobenius_norm": float(np.linalg.norm(delta)),
        "input_unchanged": True,
        "status": "candidate_only_not_applied",
    }
