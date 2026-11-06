"""CPU-only tests: synthetic evidence is NOT a real Qwen mechanism result."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from wsp.read.contrastive_direction import DirectionError, estimate_direction  # noqa: E402
from wsp.read.neuron_score import rank_neurons  # noqa: E402
from wsp.read.layer_attribution import observational_alignment  # noqa: E402
from wsp.repair.sparse_edit import propose_projection_edit  # noqa: E402
from scripts.r2_offline import main as offline_main  # noqa: E402


P = np.zeros((3, 2), dtype=np.float64)
R = np.array([[1, 0], [2, 0], [3, 0]], dtype=np.float64)
IDS = ["x", "y", "z"]


def test_direction_and_loo_exact():
    report = estimate_direction(P, R, pair_ids=IDS, split="diagnostic")
    np.testing.assert_allclose(report["direction"], [1, 0])
    assert report["raw_l2_norm"] == 2
    assert report["loo_min_cosine"] == 1


@pytest.mark.parametrize("split", ["holdout", "validation", "test"])
def test_holdout_and_validation_never_select(split):
    with pytest.raises(DirectionError, match="Holdout"):
        estimate_direction(P, R, pair_ids=IDS, split=split)


def test_duplicate_pair_ids_and_zero_direction_refused():
    with pytest.raises(DirectionError, match="duplicate"):
        estimate_direction(P, R, pair_ids=["x", "x", "z"], split="diagnostic")
    with pytest.raises(DirectionError, match="nonzero"):
        estimate_direction(P, P, pair_ids=IDS, split="diagnostic")


def test_direction_rejects_nonfinite_or_unmatched():
    bad = R.copy()
    bad[0, 0] = float("nan")
    with pytest.raises(DirectionError, match="Non-finite"):
        estimate_direction(P, bad, pair_ids=IDS, split="diagnostic")
    with pytest.raises(DirectionError, match="matching"):
        estimate_direction(P, R[:2], pair_ids=IDS, split="diagnostic")


def test_rank_signed_scores_and_stable_ties():
    w = np.array([[1, -1, 0, 0], [0, 0, 1, 0]], dtype=float)
    pa = np.zeros((3, 4), dtype=float)
    ra = np.tile([2, 1, 0, 0], (3, 1)).astype(float)
    result = rank_neurons(w, pa, ra, np.array([1., 0.]), top_k=3)
    assert result["selected_columns"] == [0, 2, 3]
    np.testing.assert_allclose(result["all_scores"], [2, -1, 0, 0])


def test_invalid_direction_or_topk_refused():
    w = np.eye(2)
    with pytest.raises(DirectionError, match="unit"):
        rank_neurons(w, P, R, [2, 0], top_k=1)
    with pytest.raises(DirectionError, match="top_k"):
        rank_neurons(w, P, R, [1, 0], top_k=3)


def test_sparse_projection_is_pure_masked_and_reflects_at_alpha6():
    w = np.array([[2., 8.], [1., 9.]], dtype=np.float32)
    source = w.copy()
    report = propose_projection_edit(w, [1, 0], [0], alpha=6)
    np.testing.assert_array_equal(w, source)
    np.testing.assert_array_equal(report["edited_weight"][:, 1], source[:, 1])
    np.testing.assert_allclose(report["edited_weight"][:, 0], [-10, 1])
    assert report["touched_entries"] == 2
    assert report["changed_entries"] == 1


def test_alpha_zero_is_no_edit_and_bad_mask_refused():
    w = np.eye(2, dtype=np.float32)
    report = propose_projection_edit(w, [1, 0], [0], alpha=0)
    np.testing.assert_array_equal(w, report["edited_weight"])
    assert report["changed_entries"] == 0
    for cols in ([0, 0], [2], [], [-1]):
        with pytest.raises(DirectionError, match="columns"):
            propose_projection_edit(w, [1, 0], cols, alpha=1)
    with pytest.raises(DirectionError, match="alpha"):
        propose_projection_edit(w, [1, 0], [0], alpha=7)


def test_alignment_is_not_misrepresented_as_causal():
    res = observational_alignment([1, 0], P, R)
    assert res["paired_mean_alignment"] == 2
    assert res["causal_supported"] is False


def _inputs(folder: Path, split: str = "diagnostic"):
    npz = folder / "probe.npz"
    w = np.array([[1., 0., 0.], [0., 1., 0.]], dtype=np.float32)
    a = np.zeros((3, 3))
    b = np.tile([2., 0., 0.], (3, 1))
    np.savez(npz, productive_states=P, repetitive_states=R,
             productive_activations=a, repetitive_activations=b,
             down_proj_weight=w)
    manifest = folder / "manifest.json"
    manifest.write_text(json.dumps({
        "split": split, "allowed_for_selection": True,
        "pair_ids": IDS, "model_revision": "a" * 40,
        "tensor_name": "model.layers.15.mlp.down_proj.weight",
        "token_selection": "matched:last-16-token-average",
        "weight_sha256": hashlib.sha256(w.tobytes(order="C")).hexdigest(),
        "weight_shape": list(w.shape), "weight_dtype": str(w.dtype),
    }))
    return npz, manifest


def test_offline_cli_produces_provenance_and_refuses_overwrite(tmp_path):
    probes, manifest = _inputs(tmp_path)
    out = tmp_path / "result.json"
    argv = ["--probes", str(probes), "--manifest", str(manifest),
            "--out", str(out), "--top-k", "1", "--alpha", "1"]
    assert offline_main(argv) == 0
    result = json.loads(out.read_text())
    assert result["candidate"]["selected_columns"] == [0]
    assert result["provenance"]["split"] == "diagnostic"
    assert result["status"] == "candidate_only_not_a_reproduction"
    assert offline_main(argv) == 2


def test_offline_cli_rejects_holdout_and_tampered_weight_hash(tmp_path):
    probes, manifest = _inputs(tmp_path, "holdout")
    argv = ["--probes", str(probes), "--manifest", str(manifest),
            "--out", str(tmp_path / "result.json"), "--top-k", "1"]
    assert offline_main(argv) == 2
    data = json.loads(manifest.read_text())
    data["split"] = "diagnostic"
    data["weight_sha256"] = "0" * 64
    manifest.write_text(json.dumps(data))
    assert offline_main(argv) == 2
    assert not (tmp_path / "result.json").exists()
