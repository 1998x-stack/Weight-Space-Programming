"""R2-B: actual teacher-forced hidden/MLP activation collection via PyTorch hooks.

Only explicitly labelled, user-authored diagnostic pairs are accepted. This
produces the exact five-array NPZ contract consumed by scripts/r2_offline.py.
No labels, generation, training, or persistent checkpoint edits occur.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from wsp.adapters.qwen import ProbeError, resolve_layer


def load_diagnostic_pairs(path: str | Path) -> list[dict[str, str]]:
    """Read a bounded, strict diagnostic JSON file; reject benchmark/answer fields."""
    path = Path(path)
    if path.stat().st_size > 1_048_576:
        raise ProbeError("Diagnostic dataset exceeds 1 MiB limit")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(doc, dict) or set(doc) != {"split", "pairs"}
            or doc["split"] != "diagnostic" or not isinstance(doc["pairs"], list)
            or not 3 <= len(doc["pairs"]) <= 128):
        raise ProbeError("Expected diagnostic split with 3..128 matched pairs")
    ids = set()
    for p in doc["pairs"]:
        if not isinstance(p, dict) or set(p) != {"id", "productive", "repetitive"}:
            raise ProbeError("Pairs may contain only id/productive/repetitive (no answers)")
        if (any(not isinstance(p[k], str) or not p[k].strip() for k in p)
                or len(p["productive"]) > 8192 or len(p["repetitive"]) > 8192
                or p["productive"] == p["repetitive"] or p["id"] in ids):
            raise ProbeError("Invalid, duplicate, or identical diagnostic pair")
        ids.add(p["id"])
    return doc["pairs"]


def _tokens(tokenizer, text: str, *, window: int, max_tokens: int) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=True)
    if (not isinstance(ids, list) or not all(type(i) is int and i >= 0 for i in ids)
            or len(ids) < window or len(ids) > max_tokens):
        raise ProbeError("Tokenizer IDs invalid or input length outside window/max_tokens")
    return ids


def probe_one(model, tokenizer, text: str, *, layer_index: int, window: int,
              max_tokens: int, device: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Capture decoder-input hidden states and down_proj *input* activations.

    Both use the mean of the last k actual input tokens. Hooks are unregistered
    even after model failure. Captures are read-only and detached to CPU FP32.
    """
    import torch
    if type(window) is not int or not 1 <= window <= 128:
        raise ProbeError("Window must be 1..128")
    layer, down = resolve_layer(model, layer_index)
    ids = _tokens(tokenizer, text, window=window, max_tokens=max_tokens)
    captures = {"hidden": [], "activation": []}

    def record(kind):
        def hook(_module, args):
            if not args or not hasattr(args[0], "ndim") or args[0].ndim != 3:
                raise ProbeError("Expected [batch, sequence, feature] hook input")
            value = args[0]
            if value.shape[0] != 1 or value.shape[1] != len(ids):
                raise ProbeError("Unexpected hook sequence dimensions")
            pooled = value[0, -window:, :].detach().float().mean(dim=0).cpu().numpy()
            if not np.isfinite(pooled).all():
                raise ProbeError("Non-finite probe activations")
            captures[kind].append(pooled)
        return hook

    handles = [layer.register_forward_pre_hook(record("hidden")),
               down.register_forward_pre_hook(record("activation"))]
    try:
        tokens = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            model(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    if len(captures["hidden"]) != 1 or len(captures["activation"]) != 1:
        raise ProbeError("Hook expected exactly once; model backend may bypass hooks")
    h = captures["hidden"][0]
    a = captures["activation"][0]
    if h.shape != (1024,) or a.shape != (3584,):
        raise ProbeError("Unexpected Qwen hidden/MLP activation shape")
    return h, a, len(ids)


def collect(model, tokenizer, pairs: list[dict[str, str]], *, layer_index: int,
            window: int, max_tokens: int, device: str) -> tuple[dict, dict]:
    """Collect strictly aligned pairs and report token lengths for confound audits."""
    _, down = resolve_layer(model, layer_index)
    if (type(max_tokens) is not int or max_tokens < window or max_tokens > 8192
            or type(window) is not int or not 1 <= window <= 128):
        raise ProbeError("Invalid max_tokens/window")
    if not 3 <= len(pairs) <= 128:
        raise ProbeError("At least 3, at most 128 diagnostic pairs required")
    out = {k: [] for k in ("productive_states", "repetitive_states",
                           "productive_activations", "repetitive_activations")}
    lengths = []
    for pair in pairs:
        ph, pa, pn = probe_one(model, tokenizer, pair["productive"],
                               layer_index=layer_index, window=window,
                               max_tokens=max_tokens, device=device)
        rh, ra, rn = probe_one(model, tokenizer, pair["repetitive"],
                               layer_index=layer_index, window=window,
                               max_tokens=max_tokens, device=device)
        out["productive_states"].append(ph)
        out["repetitive_states"].append(rh)
        out["productive_activations"].append(pa)
        out["repetitive_activations"].append(ra)
        lengths.append([pn, rn])
    arrays = {k: np.stack(v).astype(np.float32) for k, v in out.items()}
    weight = down.weight.detach().float().cpu().numpy().copy()
    arrays["down_proj_weight"] = weight
    meta = {
        "split": "diagnostic", "allowed_for_selection": True,
        "pair_ids": [p["id"] for p in pairs],
        "tensor_name": f"model.language_model.layers.{layer_index}.mlp.down_proj.weight",
        "token_selection": f"matched:right_aligned_last_{window}_tokens_mean",
        "window_tokens": window, "max_input_tokens": max_tokens,
        "token_lengths_per_pair": lengths,
        "alignment_warning": "right_aligned_windows_may_have_different_absolute_positions",
        "weight_sha256": hashlib.sha256(weight.tobytes(order="C")).hexdigest(),
        "weight_shape": list(weight.shape), "weight_dtype": str(weight.dtype),
        "model_revision": None,
        "model_attestation": "caller_must_verify",
        "probe_kind": "teacher_forced_no_labels",
    }
    return arrays, meta


def save_probe_bundle(out_dir: str | Path, arrays: dict, meta: dict) -> tuple[Path, Path]:
    """Write only in a fresh directory. No sensitive prompt text is saved."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    target = out_dir / "diagnostic.npz"
    manifest = out_dir / "diagnostic.manifest.json"
    np.savez(target, **arrays)
    meta = {**meta, "probe_npz_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    manifest.write_text(json.dumps(meta, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    return target, manifest
