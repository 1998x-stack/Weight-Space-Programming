"""No model download, no PyTorch required: strict R2-B input tests."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from wsp.adapters.qwen import ProbeError, resolve_layer  # noqa: E402
from wsp.read.activation_probe import load_diagnostic_pairs, _tokens  # noqa: E402


def _write(tmp_path, obj):
    p = tmp_path / "pairs.json"
    p.write_text(json.dumps(obj))
    return p


def _good():
    return {"split": "diagnostic", "pairs": [
        {"id": str(i), "productive": f"progress {i}", "repetitive": f"repeat {i}"}
        for i in range(3)
    ]}


def test_diagnostic_json_valid(tmp_path):
    assert len(load_diagnostic_pairs(_write(tmp_path, _good()))) == 3


@pytest.mark.parametrize("key", ["answer", "solution", "label", "holdout"])
def test_reject_extra_answer_or_test_fields(tmp_path, key):
    d = _good()
    d["pairs"][0][key] = "not_allowed"
    with pytest.raises(ProbeError, match="only"):
        load_diagnostic_pairs(_write(tmp_path, d))


@pytest.mark.parametrize("split", ["holdout", "validation", "test", "train"])
def test_reject_non_diagnostic_split(tmp_path, split):
    d = _good()
    d["split"] = split
    with pytest.raises(ProbeError, match="diagnostic"):
        load_diagnostic_pairs(_write(tmp_path, d))


def test_duplicate_and_identical_prompts_rejected(tmp_path):
    d = _good()
    d["pairs"][1]["id"] = d["pairs"][0]["id"]
    with pytest.raises(ProbeError, match="duplicate"):
        load_diagnostic_pairs(_write(tmp_path, d))
    d = _good()
    d["pairs"][0]["repetitive"] = d["pairs"][0]["productive"]
    with pytest.raises(ProbeError, match="identical"):
        load_diagnostic_pairs(_write(tmp_path, d))


def test_token_length_fail_closed():
    class Tok:
        def encode(self, x, add_special_tokens=True):
            return [1, 2, 3]
    with pytest.raises(ProbeError, match="length"):
        _tokens(Tok(), "text", window=4, max_tokens=5)
    with pytest.raises(ProbeError, match="length"):
        _tokens(Tok(), "text", window=2, max_tokens=2)
    assert _tokens(Tok(), "text", window=2, max_tokens=3) == [1, 2, 3]


def test_invalid_layer_index_does_not_access_model():
    with pytest.raises(ProbeError, match="15 and 19"):
        resolve_layer(None, 3)


def test_official_preflight_rejects_missing_weights(tmp_path):
    from wsp.adapters.qwen import inspect_official_base
    (tmp_path / "config.json").write_text(json.dumps({
        "text_config": {"hidden_size": 1024,
                        "intermediate_size": 3584, "num_hidden_layers": 24}}))
    with pytest.raises(ProbeError, match="Missing model.safetensors"):
        inspect_official_base(tmp_path)


def test_official_preflight_rejects_wrong_topology(tmp_path):
    from wsp.adapters.qwen import inspect_official_base
    (tmp_path / "config.json").write_text(json.dumps({
        "text_config": {"hidden_size": 1025,
                        "intermediate_size": 3584, "num_hidden_layers": 24}}))
    with pytest.raises(ProbeError, match="Wrong pinned"):
        inspect_official_base(tmp_path)
