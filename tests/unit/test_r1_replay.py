"""CPU-only synthetic integration tests; no claim of testing real Qwen weights."""
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from wsp.repair.replay import replay, ReplayError, _read_bits  # noqa: E402


def write_st(path, tensors):
    offset = 0
    header = {}
    chunks = []
    for name, ar in tensors.items():
        b = ar.astype("<u2").tobytes()
        header[name] = {"dtype": "BF16", "shape": list(ar.shape),
                        "data_offsets": [offset, offset + len(b)]}
        offset += len(b)
        chunks.append(b)
    h = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"".join(chunks))


def fixture(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    edit = tmp_path / "edit"
    edit.mkdir()
    w0 = np.arange(12, dtype=np.uint16).reshape(3, 4)
    w1 = np.arange(12, 24, dtype=np.uint16).reshape(3, 4)
    n0 = "model.language_model.layers.15.mlp.down_proj.weight"
    n1 = "model.language_model.layers.19.mlp.down_proj.weight"
    write_st(base / "model.safetensors", {n0: w0, n1: w1})
    (base / "config.json").write_text('{"model_type":"fixture"}')
    arrays, recs = {}, []
    for i, (name, base_w) in enumerate([(n0, w0), (n1, w1)]):
        out = base_w.copy()
        out[0, 1] += 9
        arrays[f"t{i}_idx"] = np.array([1], dtype=np.int32)
        arrays[f"t{i}_val"] = np.array([out[0, 1]], dtype=np.uint16)
        h = lambda x: hashlib.sha256(x.tobytes()).hexdigest()
        recs.append(dict(name=name, shape=[3, 4], dtype="bfloat16", form="sparse",
                         changed_entries=1, changed_columns=1, changed_rows=1,
                         base_sha256=h(base_w), edited_sha256=h(out)))
    np.savez(edit / "edit.npz", **arrays)
    (edit / "manifest.json").write_text(
        json.dumps(dict(tensors=recs, changed_tensors=2, changed_entries_total=2)))
    return base, edit, n0, n1


def test_exact_replay_and_base_immutable(tmp_path):
    base, edit, n0, n1 = fixture(tmp_path)
    raw_before = (base / "model.safetensors").read_bytes()
    report = replay(base, edit, tmp_path / "output", strict=False)
    assert report["changed_entries"] == 2
    assert (base / "model.safetensors").read_bytes() == raw_before
    for name in (n0, n1):
        assert _read_bits(tmp_path / "output" / "model.safetensors", name)[0, 1] == (
            _read_bits(base / "model.safetensors", name)[0, 1] + 9)
    assert (tmp_path / "output" / "config.json").read_text() == '{"model_type":"fixture"}'


def test_strict_rejects_fixture(tmp_path):
    base, edit, *_ = fixture(tmp_path)
    with pytest.raises(ReplayError, match="matrix shape"):
        replay(base, edit, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_modified_base_rejected_without_partial_output(tmp_path):
    base, edit, *_ = fixture(tmp_path)
    data = bytearray((base / "model.safetensors").read_bytes())
    data[-1] ^= 1
    (base / "model.safetensors").write_bytes(data)
    with pytest.raises(ReplayError, match="Original BF16"):
        replay(base, edit, tmp_path / "output", strict=False)
    assert not (tmp_path / "output").exists()


def test_corrupt_edit_sha_rejected(tmp_path):
    base, edit, *_ = fixture(tmp_path)
    p = edit / "manifest.json"
    data = json.loads(p.read_text())
    data["tensors"][0]["edited_sha256"] = "0" * 64
    p.write_text(json.dumps(data))
    with pytest.raises(ReplayError, match="Edited SHA256"):
        replay(base, edit, tmp_path / "out", strict=False)
    assert not (tmp_path / "out").exists()


def test_reject_duplicate_indices(tmp_path):
    base, edit, *_ = fixture(tmp_path)
    with np.load(edit / "edit.npz") as z:
        arrays = {k: z[k] for k in z.files}
    arrays["t0_idx"] = np.array([1, 1], dtype=np.int32)
    arrays["t0_val"] = np.array([22, 22], dtype=np.uint16)
    np.savez(edit / "edit.npz", **arrays)
    with pytest.raises(ReplayError, match="Repeated patch index"):
        replay(base, edit, tmp_path / "out", strict=False)


def test_no_overwrite_or_nested_destination(tmp_path):
    base, edit, *_ = fixture(tmp_path)
    with pytest.raises(ReplayError, match="overlap"):
        replay(base, edit, base / "edited", strict=False)
    with pytest.raises(ReplayError, match="output must not exist"):
        replay(base, edit, base, strict=False)
