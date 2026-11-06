"""R2-B: forward-hook ablation of MLP output direction (not weight editing).

Reports a *functional* causal effect on the next-token distribution for fixed
inputs, not evidence of mathematical ability improvement or solved looping.
"""
from __future__ import annotations

import numpy as np

from wsp.adapters.qwen import ProbeError, resolve_layer


def compare_next_token_logits(model, input_ids, *, layer_index: int,
                              direction, alpha: float) -> dict:
    """Compare fixed-token logits with and without a down-projection output hook.

    The hook intervenes only on the LAST position of the selected MLP output.
    Both forward passes use exactly the same tokens, no cache, eval/inference mode.
    The source model parameters are never mutated.
    """
    import torch
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
        raise ProbeError("Expected a single nonempty [1, token_count] input")
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ProbeError("Intervention alpha must lie in [0, 1]")
    layer, down = resolve_layer(model, layer_index)
    u = torch.as_tensor(direction, dtype=torch.float32, device=input_ids.device)
    if (u.ndim != 1 or u.numel() != 1024
            or not torch.isfinite(u).all()
            or not torch.isclose(torch.linalg.vector_norm(u), u.new_tensor(1.), rtol=1e-5, atol=1e-6)):
        raise ProbeError("Direction must be a finite 1024D unit vector")
    if model.training:
        raise ProbeError("Intervention requires model.eval() for deterministic inference")

    def forward_logits():
        with torch.inference_mode():
            output = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                           use_cache=False)
            logits = output.logits
            if logits.ndim != 3 or logits.shape[0] != 1:
                raise ProbeError("Unexpected model logits")
            return logits[0, -1, :].detach().float()

    original = forward_logits()

    def project(_module, _args, output):
        if output.ndim != 3 or output.shape[-1] != 1024:
            raise ProbeError("Unexpected down_proj output shape")
        basis = u.to(dtype=output.dtype, device=output.device)
        updated = output.clone()
        last = updated[:, -1, :]
        updated[:, -1, :] = last - float(alpha) * (last @ basis).unsqueeze(-1) * basis
        return updated

    handle = down.register_forward_hook(project)
    try:
        modified = forward_logits()
    finally:
        handle.remove()
    if not torch.isfinite(original).all() or not torch.isfinite(modified).all():
        raise ProbeError("Non-finite next-token logits")
    logp = torch.log_softmax(original, dim=-1)
    logq = torch.log_softmax(modified, dim=-1)
    kl = float(torch.sum(torch.exp(logp) * (logp - logq)).item())
    return {
        "layer": layer_index, "alpha": float(alpha),
        "base_argmax_token": int(original.argmax()),
        "intervened_argmax_token": int(modified.argmax()),
        "next_token_kl_base_to_intervened": max(0., kl),
        "next_token_logit_l2": float(torch.linalg.vector_norm(modified-original).item()),
        "intervention": "last_token_down_proj_output_projection_ablation",
        "evidence_scope": "fixed_input_distribution_change_not_task_score",
    }
