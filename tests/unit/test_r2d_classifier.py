"""R2-D token-level failure taxonomy and diagnostic input safety."""
from pathlib import Path
import json
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"src"))
from wsp.repair.failure_classifier import TraceError, repeated_token_span, classify_generation  # noqa: E402
from wsp.evaluate.trace_generation import load_public_diagnostic_prompts  # noqa: E402
from wsp.adapters.qwen import ProbeError  # noqa: E402


def test_exact_consecutive_motif_is_detected():
    t=[29,30] + [1,2,3,4]*4 + [99]
    span=repeated_token_span(t)
    assert span["start_token"]==2 and span["period_tokens"]==4
    assert span["copies"]==4 and span["span_tokens"]==16


def test_nonconsecutive_repetitions_not_a_loop_candidate():
    assert repeated_token_span([1,2,3,9,1,2,3,8,1,2,3]) is None


def test_cutoff_and_repetition_are_independent():
    seq=[1,2,3]*4
    r=classify_generation(seq,"some unfinished output",finish_reason="length",max_new_tokens=12)
    assert r["cut_at_limit"] and r["consecutive_repetition_candidate"]
    assert r["accuracy"] is None and r["mathematical_correctness"] is None
    assert r["loop_causal_supported"] is False
    s=classify_generation([1,2,3,4],"answer: \\\\boxed{3}",finish_reason="eos",max_new_tokens=12)
    assert s["boxed_marker_present"] and not s["cut_at_limit"]
    assert s["consecutive_repetition_candidate"] is False


@pytest.mark.parametrize("tokens,reason,budget",[
    ([1,2],"length",3),
    ([1,2],"unknown",3),
    ([1,2,3],"eos",2),
    ([-1],"eos",3),
    ([1,True,2],"eos",3),
    ([],"eos",3),
])
def test_fail_closed_invalid_traces(tokens, reason, budget):
    with pytest.raises(TraceError):
        classify_generation(tokens,"",finish_reason=reason,max_new_tokens=budget)


def test_diagnostic_prompts_valid_and_no_answers(tmp_path):
    p=tmp_path/"prompts.json"
    p.write_text(json.dumps({"split":"diagnostic","prompts":[
        {"id":"a","prompt":"Compute 2+2."},{"id":"b","prompt":"Compute 3+5."}
    ]}))
    assert len(load_public_diagnostic_prompts(p))==2
    p.write_text(json.dumps({"split":"holdout","prompts":[
        {"id":"a","prompt":"hi"},{"id":"b","prompt":"there"}
    ]}))
    with pytest.raises(ProbeError,match="diagnostic"):
        load_public_diagnostic_prompts(p)
    p.write_text(json.dumps({"split":"diagnostic","prompts":[
        {"id":"a","prompt":"hi","answer":"4"},{"id":"b","prompt":"there"}
    ]}))
    with pytest.raises(ProbeError,match="extra fields"):
        load_public_diagnostic_prompts(p)
