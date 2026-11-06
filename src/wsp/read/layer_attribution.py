"""R2-A: observational component alignment; NOT causal layer attribution."""
from __future__ import annotations

import numpy as np
from wsp.read.contrastive_direction import DirectionError, _paired_arrays


def observational_alignment(direction, productive_components,
                            repetitive_components) -> dict:
    """Project matched component differences on an independently chosen u.

    This is a descriptive correlation across pairs. Controlled forward
    interventions are mandatory before claiming that a layer caused a failure.
    """
    p, r = _paired_arrays(productive_components, repetitive_components)
    try:
        u = np.asarray(direction, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DirectionError("Direction must be numeric") from exc
    if (u.ndim != 1 or u.size != p.shape[1] or not np.isfinite(u).all()
            or not np.isclose(np.linalg.norm(u), 1.0, rtol=1e-6, atol=1e-8)):
        raise DirectionError("Direction must be a matching unit vector")
    projected = (r - p) @ u
    return {
        "paired_mean_alignment": float(np.mean(projected)),
        "paired_std_alignment": float(np.std(projected, ddof=1)),
        "pairs": len(projected),
        "causal_supported": False,
        "interpretation": "observational_alignment_only",
    }
