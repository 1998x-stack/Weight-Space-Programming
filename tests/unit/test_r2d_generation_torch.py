"""Tiny mocked inference semantics for R2-D; not a Qwen real-model score."""
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest

torch=pytest.importorskip("torch")
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from wsp.evaluate.trace_generation import run_greedy_pilot  # noqa: E402


class FakeTokenizer:
    eos_token_id=9
    pad_token_id=0
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize and add_generation_prompt and enable_thinking
        return [2, 3, 4]
    def decode(self, ids, skip_special_tokens, clean_up_tokenization_spaces):
        assert skip_special_tokens is False
        return " ".join(str(i) for i in ids)


class FakeGenerator:
    training=False
    generation_config=SimpleNamespace(eos_token_id=9)
    def generate(self, input_ids, attention_mask, *, max_new_tokens, do_sample,
                 use_cache, pad_token_id):
        assert not do_sample and use_cache and pad_token_id==9 or pad_token_id==0
        suffix=torch.tensor([[7,9]], dtype=input_ids.dtype,device=input_ids.device)
        return torch.cat([input_ids,suffix],dim=1)


def test_trace_pilot_emits_honest_early_eos():
    data=run_greedy_pilot(FakeGenerator(),FakeTokenizer(),
                          [{"id":"a","prompt":"2+2"},{"id":"b","prompt":"3+3"}],
                          device="cpu",max_new_tokens=8,max_input_tokens=8)
    assert data["summary"]["prompts"]==2
    assert data["summary"]["cut_count"]==0
    assert data["records"][0]["classification"]["finish_reason"]=="eos"
    assert data["summary"]["scientific_gate_r2_passed"] is False
    assert data["decoding"]["mode"]=="greedy"
