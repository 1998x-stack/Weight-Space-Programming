"""R2-C deterministic descriptive negative controls, not hypothesis testing.

No test/holdout labels or published author masks are used. Shuffling sign
assignments within matched diagnostic pairs is a *sanity check*, not an
exchangeable permutation p-value unless stronger assumptions are justified.
"""
from __future__ import annotations
import numpy as np
from wsp.read.contrastive_direction import DirectionError, _paired_arrays


def control_directions(productive, repetitive, observed, *, seed: int = 20261106) -> dict:
    """Construct a fixed random unit direction and a label-swap direction.

    Swaps the productive/repetitive labels for half the pairs, rounded down.
    Returns no significance claims. Degenerate swapped means are reported as
    null rather than inventing a direction.
    """
    p, r = _paired_arrays(productive, repetitive)
    u = np.asarray(observed, dtype=np.float64)
    if (u.ndim != 1 or u.shape != (p.shape[1],)
            or not np.isfinite(u).all()
            or not np.isclose(np.linalg.norm(u), 1., rtol=1e-6, atol=1e-8)):
        raise DirectionError("Observed direction must be a finite matching unit vector")
    if type(seed) is not int or seed < 0:
        raise DirectionError("Invalid control seed")
    rng = np.random.default_rng(seed)
    random = rng.standard_normal(len(u))
    if len(u) > 1:
        random = random - np.dot(random, u) * u
    magnitude = np.linalg.norm(random)
    if not np.isfinite(magnitude) or magnitude < 1e-12:
        raise DirectionError("Degenerate random control")
    random /= magnitude
    contrasts = r - p
    swapped = contrasts.copy()
    flip_ids = sorted(int(j) for j in rng.choice(len(p), size=max(1, len(p)//2), replace=False))
    swapped[flip_ids] *= -1
    swap_mean = swapped.mean(axis=0)
    swap_norm = float(np.linalg.norm(swap_mean))
    swap_u = swap_mean/swap_norm if swap_norm > 1e-12 else None
    return {
        "random_direction": random,
        "label_swapped_direction": swap_u,
        "random_cosine_with_observed": float(np.dot(u, random)),
        "label_swapped_cosine_with_observed":
            (float(np.dot(u, swap_u)) if swap_u is not None else None),
        "swapped_pair_indices": flip_ids,
        "control_seed": seed,
        "status": "descriptive_controls_only_no_significance_claim",
    }
