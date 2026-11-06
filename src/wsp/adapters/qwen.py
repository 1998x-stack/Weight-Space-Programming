"""Qwen3.5 text-layer adapter for offline R2-B teacher-forced probes.

No remote downloads occur here. This module handles both the official multimodal
conditional-generation wrapper and a text-only Qwen3.5 causal-LM wrapper.
"""
from __future__ import annotations

from pathlib import Path

from wsp.repair.replay import (
    EXPECTED_HASHES, EXPECTED_NAMES, EXPECTED_SHAPE, MODEL_REVISION,
    _read_bits, _shards, sha256,
)


class ProbeError(ValueError):
    """R2-B evidence, model topology, or intervention precondition failed."""


def resolve_layer(model, layer_index: int):
    """Return a decoder block and its down projection after checking topology."""
    if type(layer_index) is not int or layer_index not in (15, 19):
        raise ProbeError("Only Qwen p50c diagnostic layers 15 and 19 are allowed")
    backbone = getattr(model, "model", None)
    text_model = getattr(backbone, "language_model", backbone)
    layers = getattr(text_model, "layers", None)
    if layers is None or len(layers) != 24:
        raise ProbeError("Qwen3.5 text model must expose exactly 24 decoder layers")
    layer = layers[layer_index]
    down = getattr(getattr(layer, "mlp", None), "down_proj", None)
    weight = getattr(down, "weight", None)
    if weight is None or tuple(weight.shape) != EXPECTED_SHAPE:
        raise ProbeError("Qwen down_proj weight must have shape [1024, 3584]")
    return layer, down


def inspect_official_base(model_dir: str | Path) -> dict:
    """Fail closed on both published official base-tensor BF16 hashes.

    This verifies the two edited target tensors, NOT every other checkpoint
    tensor, tokenizer, environment or the full Hugging Face repository revision.
    """
    model_dir = Path(model_dir)
    if not (model_dir / "config.json").is_file():
        raise ProbeError("Model directory must contain local config.json")
    try:
        import json
        cfg = json.loads((model_dir / "config.json").read_text())
        text_cfg = cfg.get("text_config", cfg)
        if (text_cfg.get("hidden_size") != 1024
                or text_cfg.get("intermediate_size") != 3584
                or text_cfg.get("num_hidden_layers") != 24):
            raise ProbeError("Wrong pinned Qwen3.5 topology")
        shards = _shards(model_dir)
        results = {}
        for name in EXPECTED_NAMES:
            if name not in shards:
                raise ProbeError(f"Missing target tensor: {name}")
            bits = _read_bits(shards[name], name)
            actual = sha256(bits.astype("<u2", copy=False).tobytes(order="C"))
            if actual != EXPECTED_HASHES[name][0]:
                raise ProbeError(f"Not the published BF16 base tensor: {name}")
            results[name] = actual
    except (OSError, KeyError, TypeError, IndexError, json.JSONDecodeError) as exc:
        raise ProbeError(f"Official base inspection failed: {exc}") from exc
    return {"revision_declared": MODEL_REVISION, "base_target_bf16_sha256": results,
            "attestation_scope": "two_target_tensors_only_not_complete_model"}


def load_local_qwen(model_dir: str | Path, *, device: str):
    """Offline pretrained loader; use CUDA BF16 or CPU FP32 for probe arithmetic."""
    if device not in ("cpu", "cuda"):
        raise ProbeError("Explicit device must be cpu or cuda")
    try:
        import torch
        from transformers import AutoConfig, AutoTokenizer
        from transformers import Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration
    except ImportError as exc:
        raise ProbeError("Install R2-B optional torch/transformers dependencies") from exc
    if device == "cuda" and not torch.cuda.is_available():
        raise ProbeError("CUDA requested but unavailable")
    cfg = AutoConfig.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
    if cfg.model_type == "qwen3_5":
        cls = Qwen3_5ForConditionalGeneration
    elif cfg.model_type == "qwen3_5_text":
        cls = Qwen3_5ForCausalLM
    else:
        raise ProbeError(f"Unsupported model type: {cfg.model_type}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False)
    model = cls.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device).eval()
    resolve_layer(model, 15)
    resolve_layer(model, 19)
    return model, tokenizer
