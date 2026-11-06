"""Tiny PyTorch integration: checks hook capture, cleanup and causal response.

These synthetic model tests do NOT claim the actual Qwen checkpoint was run.
Install optional torch to enable them; otherwise this module is skipped.
"""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from wsp.read.activation_probe import collect, probe_one, save_probe_bundle  # noqa: E402
from wsp.read.causal_intervention import compare_next_token_logits  # noqa: E402


class TargetLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.down_proj = nn.Linear(3584, 1024, bias=False)
        with torch.no_grad():
            self.mlp.down_proj.weight.zero_()
            self.mlp.down_proj.weight[0, 0] = 1.

    def forward(self, x):
        return x + self.mlp.down_proj(F.pad(x, (0, 3584 - 1024)))


class ToyQwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(32, 1024)
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList(
            [TargetLayer() if i == 15 else nn.Identity() for i in range(24)])
        self.lm_head = nn.Linear(1024, 4, bias=False)
        with torch.no_grad():
            self.embed.weight.zero_()
            self.embed.weight[:, 0] = 1.
            self.lm_head.weight.zero_()
            self.lm_head.weight[0, 0] = 1.
            self.lm_head.weight[1, 0] = -1.

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        h = self.embed(input_ids)
        for layer in self.model.language_model.layers:
            h = layer(h)
        return SimpleNamespace(logits=self.lm_head(h))


class Tok:
    def encode(self, text, add_special_tokens=True):
        return [1, 2, 3, 4] if text.startswith("p") else [4, 3, 2, 1]


def model_and_pairs():
    m = ToyQwen().eval()
    pairs = [{"id": f"pair{i}", "productive": f"p{i}", "repetitive": f"r{i}"}
             for i in range(3)]
    return m, pairs


def test_hook_capture_shapes_values_and_cleanup():
    m, _ = model_and_pairs()
    target = m.model.language_model.layers[15]
    before = sum(len(x._forward_pre_hooks) for x in
                 (target, target.mlp.down_proj))
    h, a, n = probe_one(m, Tok(), "productive", layer_index=15,
                         window=2, max_tokens=4, device="cpu")
    assert h.shape == (1024,) and a.shape == (3584,) and n == 4
    np.testing.assert_allclose(h[0], 1)
    np.testing.assert_allclose(a[0], 1)
    after = sum(len(x._forward_pre_hooks) for x in
                (target, target.mlp.down_proj))
    assert before == after == 0


def test_collect_feeds_existing_r2a_contract(tmp_path):
    m, pairs = model_and_pairs()
    arrays, meta = collect(m, Tok(), pairs, layer_index=15,
                           window=2, max_tokens=4, device="cpu")
    assert arrays["productive_states"].shape == (3, 1024)
    assert arrays["repetitive_activations"].shape == (3, 3584)
    assert arrays["down_proj_weight"].shape == (1024, 3584)
    assert meta["pair_ids"] == ["pair0", "pair1", "pair2"]
    meta["model_revision"] = "a" * 40
    npz, manifest = save_probe_bundle(tmp_path / "new", arrays, meta)
    assert set(np.load(npz).files) == {
        "productive_states", "repetitive_states",
        "productive_activations", "repetitive_activations", "down_proj_weight"}
    assert json.loads(manifest.read_text())["weight_shape"] == [1024, 3584]
    with pytest.raises(FileExistsError):
        save_probe_bundle(tmp_path / "new", arrays, meta)


def test_causal_intervention_observes_logits_and_never_mutates_weights():
    m, _ = model_and_pairs()
    source = m.model.language_model.layers[15].mlp.down_proj.weight.clone()
    ids = torch.tensor([[1, 2, 3, 4]])
    direction = np.zeros(1024, dtype=np.float32)
    direction[0] = 1.
    zero = compare_next_token_logits(m, ids, layer_index=15,
                                     direction=direction, alpha=0)
    changed = compare_next_token_logits(m, ids, layer_index=15,
                                        direction=direction, alpha=1)
    assert zero["next_token_logit_l2"] == 0
    assert changed["next_token_logit_l2"] > 0
    assert changed["next_token_kl_base_to_intervened"] > 0
    assert torch.equal(source, m.model.language_model.layers[15].mlp.down_proj.weight)
    assert not m.model.language_model.layers[15].mlp.down_proj._forward_hooks


def test_intervention_bad_direction_fails_closed():
    m, _ = model_and_pairs()
    ids = torch.tensor([[1, 2]])
    with pytest.raises(ValueError, match="unit"):
        compare_next_token_logits(m, ids, layer_index=15,
                                  direction=np.zeros(1024), alpha=1)
