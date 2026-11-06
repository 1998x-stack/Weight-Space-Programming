"""Paired problem-level statistics tests."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from wsp.evaluate.paired_statistics import compare, PairingError  # noqa: E402


def write(p, rows):
    p.write_text(json.dumps(rows))


def test_paired_statistics_are_deterministic(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    write(a, [{"id": "z", "correct": [0, 1]}, {"id": "a", "correct": [0, 0]}])
    write(b, [{"id": "a", "correct": [1, 1]}, {"id": "z", "correct": [1, 1]}])
    result = compare(a, b, bootstrap=400, require_samples=2)
    assert result == compare(a, b, bootstrap=400, require_samples=2)
    assert result["base_accuracy"] == .25
    assert result["edited_accuracy"] == 1
    assert result["absolute_gain"] == .75
    assert result["sample_index_flips"]["wrong_to_right"] == 3
    assert result["sample_index_flips"]["right_to_wrong"] == 0


def test_different_ids_fail(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    write(a, [{"id": "1", "correct": [1]}])
    write(b, [{"id": "2", "correct": [1]}])
    with pytest.raises(PairingError, match="Problem IDs"):
        compare(a, b)


def test_samples_mismatch_fail(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    write(a, [{"id": "1", "correct": [1, 1]}])
    write(b, [{"id": "1", "correct": [0]}])
    with pytest.raises(PairingError, match="sample counts"):
        compare(a, b)


def test_duplicates_fail(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    write(a, [{"id": "1", "correct": [1]}, {"id": "1", "correct": [0]}])
    write(b, [{"id": "1", "correct": [1]}])
    with pytest.raises(PairingError, match="duplicate"):
        compare(a, b)
