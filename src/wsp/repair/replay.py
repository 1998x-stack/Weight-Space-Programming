"""R1: byte-for-byte p50c BF16 sparse patch replay; CPU-only.

This independent implementation consumes the *published* manifest/NPZ without
copying upstream source. It never downloads assets and never modifies the base.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
from typing import Any

import numpy as np

EXPECTED_NAMES = (
    "model.language_model.layers.15.mlp.down_proj.weight",
    "model.language_model.layers.19.mlp.down_proj.weight",
)
EXPECTED_SHAPE = (1024, 3584)
EXPECTED_TOTAL = 130188
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"


class ReplayError(ValueError):
    """A mismatch invalidates the faithful-replay claim."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _header(filename: Path) -> tuple[dict[str, Any], int]:
    with filename.open("rb") as fh:
        prefix = fh.read(8)
        if len(prefix) != 8:
            raise ReplayError(f"Invalid safetensors header: {filename}")
        n = struct.unpack("<Q", prefix)[0]
        if n > 100_000_000 or n == 0:
            raise ReplayError("Invalid safetensors header length")
        raw = fh.read(n)
        if len(raw) != n:
            raise ReplayError("Truncated safetensors header")
    head = json.loads(raw)
    if not isinstance(head, dict):
        raise ReplayError("Invalid safetensors metadata")
    return head, 8 + n


def _read_bits(shard: Path, name: str) -> np.ndarray:
    header, off = _header(shard)
    if name not in header:
        raise ReplayError(f"Missing tensor: {name} in {shard}")
    info = header[name]
    if info["dtype"] != "BF16":
        raise ReplayError(f"Expected BF16: {name}")
    start, end = info["data_offsets"]
    size = int(np.prod(info["shape"])) * 2
    if start < 0 or end - start != size or end + off > shard.stat().st_size:
        raise ReplayError(f"Invalid tensor range: {name}")
    with shard.open("rb") as fh:
        fh.seek(off + start)
        raw = fh.read(size)
    if len(raw) != size:
        raise ReplayError(f"Truncated tensor: {name}")
    return np.frombuffer(raw, dtype="<u2").copy().reshape(info["shape"])


def _write_bits(shard: Path, name: str, bits: np.ndarray) -> None:
    header, off = _header(shard)
    info = header[name]
    start, end = info["data_offsets"]
    raw = np.ascontiguousarray(bits.astype("<u2", copy=False)).tobytes()
    if len(raw) != end - start or list(bits.shape) != info["shape"]:
        raise ReplayError(f"Write shape mismatch: {name}")
    with shard.open("r+b") as fh:
        fh.seek(off + start)
        fh.write(raw)


def _shards(base_dir: Path) -> dict[str, Path]:
    index = base_dir / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        if not isinstance(weight_map, dict):
            raise ReplayError("Invalid weight map")
        if any(Path(s).name != s for s in weight_map.values()):
            raise ReplayError("Non-local safetensors shard path")
        return {n: base_dir / s for n, s in weight_map.items()}
    # The author's text-only checkpoint is unsharded.
    single = base_dir / "model.safetensors"
    if not single.is_file():
        raise ReplayError("Missing model.safetensors or model.safetensors.index.json")
    return {n: single for n in _header(single)[0] if n != "__metadata__"}


def _check_manifest(man: dict[str, Any], strict: bool) -> list[dict[str, Any]]:
    recs = man.get("tensors")
    if not isinstance(recs, list) or not recs:
        raise ReplayError("Empty or invalid patch manifest")
    names = [r["name"] for r in recs]
    if len(names) != len(set(names)):
        raise ReplayError("Duplicate patch tensor")
    if strict and set(names) != set(EXPECTED_NAMES):
        raise ReplayError("Not the p50c layer-15/layer-19 patch")
    total = 0
    for r in recs:
        if r.get("form") != "sparse" or r.get("dtype") != "bfloat16":
            raise ReplayError("Only published BF16 sparse edits are supported")
        if strict and tuple(r["shape"]) != EXPECTED_SHAPE:
            raise ReplayError("Unexpected p50c matrix shape")
        for key in ("base_sha256", "edited_sha256"):
            h = r.get(key, "")
            if not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
                raise ReplayError(f"Invalid {key} for {r['name']}")
        total += int(r["changed_entries"])
        if strict and int(r["changed_columns"]) != 64:
            raise ReplayError("Unexpected column count")
    if total != man.get("changed_entries_total") or len(recs) != man.get("changed_tensors"):
        raise ReplayError("Manifest totals inconsistent")
    if strict and total != EXPECTED_TOTAL:
        raise ReplayError("Unexpected p50c changed-entry count")
    return recs


def _make_patch(base: np.ndarray, z: Any, i: int, record: dict[str, Any]) -> np.ndarray:
    idx, val = z[f"t{i}_idx"], z[f"t{i}_val"]
    if idx.ndim != 1 or val.ndim != 1 or len(idx) != len(val):
        raise ReplayError("Patch index/value dimensions invalid")
    if idx.dtype.kind not in "iu" or val.dtype != np.dtype("uint16"):
        raise ReplayError("Patch requires integer indices and raw uint16 BF16 values")
    if len(idx) and (idx.min() < 0 or idx.max() >= base.size):
        raise ReplayError("Patch index out of bounds")
    if len(np.unique(idx)) != len(idx):
        raise ReplayError("Repeated patch index")
    edited = base.copy()
    edited.reshape(-1)[idx] = val
    diff = edited != base
    if int(diff.sum()) != int(record["changed_entries"]):
        raise ReplayError("Patch changed-entry count mismatch")
    if "changed_columns" in record and int(diff.any(axis=0).sum()) != int(record["changed_columns"]):
        raise ReplayError("Patch column count mismatch")
    if "changed_rows" in record and int(diff.any(axis=1).sum()) != int(record["changed_rows"]):
        raise ReplayError("Patch row count mismatch")
    if sha256(edited.tobytes()) != record["edited_sha256"]:
        raise ReplayError("Edited SHA256 mismatch")
    return edited


def replay(base_dir: Path | str, edit_dir: Path | str, output_dir: Path | str,
           *, strict: bool = True) -> dict[str, Any]:
    """Validate all tensors before writing an atomic, isolated checkpoint copy.

    strict=True is mandatory for a faithful p50c replay. strict=False is only
    for small synthetic test fixtures, and must never be used to claim R1 pass.
    """
    base_dir, edit_dir, output_dir = map(lambda p: Path(p).resolve(), (base_dir, edit_dir, output_dir))
    if not base_dir.is_dir() or not edit_dir.is_dir() or output_dir.exists():
        raise ReplayError("Base/edit directories must exist; output must not exist")
    if base_dir == output_dir or base_dir in output_dir.parents or output_dir in base_dir.parents:
        raise ReplayError("Output cannot overlap or nest in the immutable base directory")
    if any(p.is_symlink() for p in base_dir.rglob("*")):
        raise ReplayError("Refusing symlinks in model directory")
    man = json.loads((edit_dir / "manifest.json").read_text())
    recs = _check_manifest(man, strict)
    shards = _shards(base_dir)
    edits: dict[str, np.ndarray] = {}
    with np.load(edit_dir / "edit.npz", allow_pickle=False) as z:
        for i, rec in enumerate(recs):
            name = rec["name"]
            if name not in shards or not shards[name].is_file():
                raise ReplayError(f"Base tensor unavailable: {name}")
            b = _read_bits(shards[name], name)
            if list(b.shape) != rec["shape"] or sha256(b.tobytes()) != rec["base_sha256"]:
                raise ReplayError(f"Original BF16 bits do not match manifest: {name}")
            edits[name] = _make_patch(b, z, i, rec)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".wsp-r1-", dir=output_dir.parent))
    try:
        # Copy configuration/tokenizer as well as all sharded weights without modifying base.
        for src in base_dir.iterdir():
            if src.is_file():
                shutil.copy2(src, temp / src.name)
        for name, bits in edits.items():
            _write_bits(temp / shards[name].name, name, bits)
        for rec in recs:
            if sha256(_read_bits(temp / shards[rec["name"]].name, rec["name"]).tobytes()) != rec["edited_sha256"]:
                raise ReplayError("Post-write hash differs")
        audit = {"status": "bit_exact_patch_replayed", "strict_p50c": strict,
                 "base_revision_required": MODEL_REVISION if strict else None,
                 "changed_tensors": len(recs), "changed_entries": man["changed_entries_total"],
                 "tensors": [{"name": r["name"], "base_sha256": r["base_sha256"],
                              "edited_sha256": r["edited_sha256"]} for r in recs],
                 "warning": "This verifies edited tensors, not full-model revision or benchmark."}
        (temp / "wsp_r1_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        os.rename(temp, output_dir)
        return audit
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
