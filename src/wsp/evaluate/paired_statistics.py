"""R1: deterministic paired-per-problem evaluation summaries; no model inference."""
from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any


class PairingError(ValueError):
    """Sample pairing cannot be established."""


def _load(path: str | Path) -> dict[str, list[int]]:
    records = json.loads(Path(path).read_text())
    if not isinstance(records, list) or not records:
        raise PairingError("Expected nonempty per_problem JSON array")
    rows: dict[str, list[int]] = {}
    for r in records:
        key, answers = r["id"], r["correct"]
        if not isinstance(key, str) or not key or key in rows:
            raise PairingError("Missing or duplicate problem id")
        if not isinstance(answers, list) or not answers or any(x not in (0, 1) or type(x) is not int for x in answers):
            raise PairingError("Each problem needs binary per-sample correctness")
        rows[key] = answers
    return rows


def _quantile(xs: list[float], q: float) -> float:
    pos = (len(xs) - 1) * q
    a = int(pos)
    return xs[a] + (xs[min(a + 1, len(xs) - 1)] - xs[a]) * (pos - a)


def compare(base_path: str | Path, edited_path: str | Path, *, seed: int = 20261106,
            bootstrap: int = 2000, require_samples: int | None = None) -> dict[str, Any]:
    if bootstrap < 100:
        raise PairingError("At least 100 resamples required")
    base, edit = _load(base_path), _load(edited_path)
    if base.keys() != edit.keys():
        raise PairingError("Problem IDs differ; cannot claim paired benchmark")
    ids = sorted(base)
    n_samples = len(base[ids[0]])
    if require_samples is not None and n_samples != require_samples:
        raise PairingError("Unexpected samples per problem")
    if any(len(base[k]) != len(edit[k]) or len(base[k]) != n_samples for k in ids):
        raise PairingError("Per-problem sample counts differ")
    n = len(ids)
    b = [sum(base[k]) / n_samples for k in ids]
    e = [sum(edit[k]) / n_samples for k in ids]
    diff = [y - x for x, y in zip(b, e)]
    rng = random.Random(seed)
    draws = sorted(sum(diff[rng.randrange(n)] for _ in range(n)) / n for _ in range(bootstrap))
    flips = {"wrong_to_right": 0, "right_to_wrong": 0, "wrong_to_wrong": 0, "right_to_right": 0}
    for k in ids:
        for x, y in zip(base[k], edit[k]):
            flips[('right' if x else 'wrong') + '_to_' + ('right' if y else 'wrong')] += 1
    return {"problems": n, "samples_per_problem": n_samples, "base_accuracy": sum(b) / n,
            "edited_accuracy": sum(e) / n, "absolute_gain": sum(diff) / n,
            "paired_cluster_bootstrap_95ci": [_quantile(draws, .025), _quantile(draws, .975)],
            "sample_index_flips": flips, "bootstrap_replicates": bootstrap, "seed": seed,
            "pairing_note": "Pairs by problem ID and sample index; matching model seeds/config require external manifests."}
