"""R2-A: rank candidate MLP down-projection columns by contrastive writing score."""
from __future__ import annotations

import numpy as np
from wsp.read.contrastive_direction import DirectionError, _paired_arrays


def rank_neurons(weight, productive_activations, repetitive_activations,
                 direction, *, top_k: int) -> dict:
    """score_j = mean(a_repetitive - a_productive)_j * (u @ W[:,j]).

    W must be [d_model, d_ff], not transposed. Sort descending with a stable
    lowest-column-index tie-break. Result does NOT justify applying an edit.
    """
    p, r = _paired_arrays(productive_activations, repetitive_activations)
    try:
        w = np.asarray(weight, dtype=np.float64)
        u = np.asarray(direction, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise DirectionError("Non-numeric weight or direction") from exc
    if (w.ndim != 2 or u.ndim != 1 or w.shape[0] != u.size
            or w.shape[1] != p.shape[1] or not np.isfinite(w).all()
            or not np.isfinite(u).all()):
        raise DirectionError("Weight/direction/activation dimensions or values invalid")
    if not np.isclose(np.linalg.norm(u), 1.0, rtol=1e-6, atol=1e-8):
        raise DirectionError("Direction must have unit L2 norm")
    if type(top_k) is not int or not 1 <= top_k <= w.shape[1]:
        raise DirectionError("Invalid top_k")
    activation_delta = (r - p).mean(axis=0)
    column_alignment = u @ w
    scores = activation_delta * column_alignment
    order = np.lexsort((np.arange(scores.size), -scores))
    return {
        "selected_columns": [int(j) for j in order[:top_k]],
        "selected_scores": [float(scores[j]) for j in order[:top_k]],
        "all_scores": scores,
        "activation_delta": activation_delta,
        "column_alignment": column_alignment,
        "ordering": "descending_signed_score_then_ascending_index",
    }
