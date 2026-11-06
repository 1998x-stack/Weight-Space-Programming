"""Deterministic synthetic descriptive controls; no Qwen benchmark claims."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from wsp.read.diagnostic_controls import control_directions  # noqa: E402
from wsp.read.contrastive_direction import DirectionError  # noqa: E402

def test_negative_controls_deterministic_and_unit():
    p=np.zeros((4,3))
    r=np.array([[1,1,0],[2,1,0],[3,2,0],[4,2,0]],dtype=float)
    u=(r-p).mean(axis=0)
    u/=np.linalg.norm(u)
    one=control_directions(p,r,u,seed=5)
    two=control_directions(p,r,u,seed=5)
    assert one["swapped_pair_indices"]==two["swapped_pair_indices"]
    np.testing.assert_allclose(one["random_direction"],two["random_direction"])
    np.testing.assert_allclose(np.linalg.norm(one["random_direction"]),1.)
    assert abs(one["random_cosine_with_observed"])<1e-12
    assert len(one["swapped_pair_indices"])==2
    assert one["status"].startswith("descriptive_controls_only")

def test_input_validation_rejects_nonfinite_mismatched_and_bad_seed():
    p=np.zeros((4,2))
    r=np.ones((4,2))
    with pytest.raises(DirectionError,match="unit"):
        control_directions(p,r,[0,0])
    with pytest.raises(DirectionError,match="seed"):
        control_directions(p,r,[1,0],seed=-1)
    with pytest.raises(DirectionError,match="matching"):
        control_directions(p,r[:3],[1,0])

def test_degenerate_swap_yields_null_not_pvalue():
    p=np.zeros((4,2))
    r=np.tile([1.,0.],(4,1))
    out=control_directions(p,r,[1,0],seed=1)
    assert out["label_swapped_direction"] is None
    assert out["label_swapped_cosine_with_observed"] is None
