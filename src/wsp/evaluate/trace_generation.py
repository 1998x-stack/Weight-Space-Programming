"""Bounded, real-model autoregressive trace collection for R2-D.

Greedy diagnostic pilot (not the original authors' stochastic Exp8 protocol).
No model editing, benchmark answer scoring, or holdout data.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from wsp.adapters.qwen import ProbeError
from wsp.repair.failure_classifier import classify_generation


def load_public_diagnostic_prompts(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if path.stat().st_size > 100_000:
        raise ProbeError("Too-large public diagnostic prompt dataset")
    data = json.loads(path.read_text(encoding="utf8"))
    if (not isinstance(data, dict) or set(data) != {"split", "prompts"}
            or data["split"] != "diagnostic" or not isinstance(data["prompts"], list)
            or not 2 <= len(data["prompts"]) <= 12):
        raise ProbeError("Expected diagnostic-only array of 2..12 prompts")
    ids = set()
    for row in data["prompts"]:
        if (not isinstance(row, dict) or set(row) != {"id", "prompt"}
                or not all(isinstance(row[k], str) and row[k].strip() for k in row)
                or len(row["prompt"]) > 2000 or row["id"] in ids):
            raise ProbeError("Invalid prompt/ID or prohibited extra fields (e.g. answers)")
        ids.add(row["id"])
    return data["prompts"]



def normalize_prompt_ids(raw: Any, *, max_input_tokens: int) -> list[int]:
    """Handle standard HF list, BatchEncoding/dict or one-row tensor outputs.

    Multiple tokenized examples and multimodal extra-input fields are refused
    because this pilot intentionally supports one text prompt at a time.
    """
    original_type = type(raw).__name__
    if hasattr(raw, "keys"):
        keys = set(raw.keys())
        if keys != {"input_ids"}:
            # Some tokenizers also return an attention mask, which can be
            # safely regenerated from the validated single text sequence.
            if keys != {"input_ids", "attention_mask"}:
                raise ProbeError(f"Unsupported tokenizer fields: {sorted(keys)}")
        raw = raw["input_ids"]
    if hasattr(raw, "tolist") and not isinstance(raw, list):
        raw = raw.tolist()
    if isinstance(raw, tuple):
        raw = list(raw)
    if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], (list, tuple)):
        raw = list(raw[0])
    if (not isinstance(raw, list) or
            any(type(i) is not int or i < 0 for i in raw)):
        raise ProbeError(f"Tokenizer output not a 1D int sequence: {original_type}")
    if not 1 <= len(raw) <= max_input_tokens:
        raise ProbeError(
            f"Tokenized prompt length {len(raw)} outside max_input_tokens "
            f"{max_input_tokens} (source type {original_type})")
    return raw

def run_greedy_pilot(model, tokenizer, prompts: list[dict[str, str]], *,
                     device: str, max_new_tokens: int = 96,
                     max_input_tokens: int = 160) -> dict[str, Any]:
    """Run each diagnostic prompt through a genuine model.generate forward pass.

    Default greedy decoding is explicitly distinct from think35 sampling.
    Traces are never used as independently labelled productive/repetitive pairs.
    """
    import torch

    if model.training:
        raise ProbeError("Inference requires model.eval()")
    if not (type(max_new_tokens) is int and 1 <= max_new_tokens <= 256
            and type(max_input_tokens) is int and 1 <= max_input_tokens <= 512):
        raise ProbeError("Pilot token limits out of range")
    if not 2 <= len(prompts) <= 12:
        raise ProbeError("Pilot requires 2..12 diagnostic prompts")
    eos_ids = getattr(model.generation_config, "eos_token_id", None)
    if eos_ids is None:
        eos_ids = tokenizer.eos_token_id
    allowed_eos = set(eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids])
    if not allowed_eos or None in allowed_eos:
        raise ProbeError("No known EOS token")
    records = []
    for prompt in prompts:
        try:
            source_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt["prompt"]}],
                tokenize=True, add_generation_prompt=True, enable_thinking=True)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ProbeError("Incompatible pinned tokenizer chat template") from exc
        source_ids = normalize_prompt_ids(source_ids, max_input_tokens=max_input_tokens)
        ids = torch.tensor([source_ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids),
                max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        if (generated.ndim != 2 or generated.shape[0] != 1 or
                generated.shape[1] <= len(source_ids) or
                not torch.equal(generated[0, :len(source_ids)], ids[0])):
            raise ProbeError("Generated sequence is not an input-prefixed completion")
        new_ids = [int(i) for i in generated[0, len(source_ids):].tolist()]
        finish = ("eos" if new_ids[-1] in allowed_eos else
                  "length" if len(new_ids) == max_new_tokens else "other_stop")
        content = tokenizer.decode(new_ids, skip_special_tokens=False,
                                   clean_up_tokenization_spaces=False)
        flags = classify_generation(new_ids, content, finish_reason=finish,
                                    max_new_tokens=max_new_tokens)
        records.append({
            "prompt_id": prompt["id"],
            "prompt_sha256": hashlib.sha256(prompt["prompt"].encode()).hexdigest(),
            "input_tokens": len(source_ids),
            "generated_token_ids": new_ids,
            "generated_text": content,
            "generated_text_sha256": hashlib.sha256(content.encode()).hexdigest(),
            "classification": flags,
        })
    return {
        "schema_version": "r2d-traces-v1",
        "status": "real_model_generated_diagnostic_traces_no_scientific_loop_claim",
        "decoding": {
            "mode": "greedy",
            "thinking_template": True,
            "max_new_tokens": max_new_tokens,
            "max_input_tokens": max_input_tokens,
            "do_sample": False,
            "sampling_note": "NOT author think35 (temperature 0.6/presence_penalty 1.5)",
        },
        "records": records,
        "summary": {
            "prompts": len(records),
            "cut_count": sum(r["classification"]["cut_at_limit"] for r in records),
            "repetition_candidate_count": sum(
                r["classification"]["consecutive_repetition_candidate"] for r in records),
            "boxed_marker_count": sum(r["classification"]["boxed_marker_present"] for r in records),
            "independently_labelled_productive_repetitive_pairs": 0,
            "scientific_gate_r2_passed": False,
        },
    }
