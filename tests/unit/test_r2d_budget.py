"""Budget-only comparison must verify token prefix and data contracts."""
import hashlib
import sys
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from wsp.repair.failure_classifier import classify_generation
from wsp.evaluate.budget_sensitivity import BudgetComparisonError,compare_budgets


def make(limit,tokens):
    text=" ".join(str(t) for t in tokens)
    row={"prompt_id":"a","prompt_sha256":"f"*64,"input_tokens":10,
         "generated_token_ids":tokens,"generated_text":text,
         "generated_text_sha256":hashlib.sha256(text.encode()).hexdigest(),
         "classification":classify_generation(tokens,text,
            finish_reason="length" if len(tokens)==limit else "eos",
            max_new_tokens=limit)}
    return {"status":"real_model_generated_diagnostic_traces_no_scientific_loop_claim",
            "decoding":{"mode":"greedy","thinking_template":True,"do_sample":False,
                        "max_input_tokens":100,"max_new_tokens":limit},
            "provenance":{"model_revision_declared":"a"*40,
                          "model_attestation_scope":"two_target_tensors_only_not_complete_model",
                          "base_target_bf16_sha256":{"layer15":"v"},
                          "model_config_sha256":"b"*64,"input_prompts_sha256":"c"*64,
                          "torch":"test","transformers":"test","device":"cpu"},
            "records":[row]}


def test_budget_comparison_counts_and_prefix():
    a=make(3,[1,2,3])
    b=make(6,[1,2,3,4,5,6])
    r=compare_budgets(a,b)
    assert r["problems"]==1
    assert r["all_short_prefixes_identical"] is True
    assert r["short_cut_count"]==1 and r["long_cut_count"]==1
    assert r["r2_scientific_gate_passed"] is False


def test_divergent_prefix_fails_closed():
    with pytest.raises(BudgetComparisonError,match="prefix"):
        compare_budgets(make(3,[1,2,3]),make(6,[1,3,3,4,5,6]))


def test_changed_model_provenance_refused():
    a=make(3,[1,2,3])
    b=make(6,[1,2,3,4,5,6])
    b["provenance"]["model_config_sha256"]="different"
    with pytest.raises(BudgetComparisonError,match="mismatch"):
        compare_budgets(a,b)


def test_altered_trace_hash_or_classification_refused():
    a=make(3,[1,2,3]);b=make(6,[1,2,3,4,5,6])
    b["records"][0]["generated_text_sha256"]="bad"
    with pytest.raises(BudgetComparisonError,match="SHA"):
        compare_budgets(a,b)
    b=make(6,[1,2,3,4,5,6])
    b["records"][0]["classification"]["cut_at_limit"]=False
    with pytest.raises(BudgetComparisonError,match="Classifications"):
        compare_budgets(a,b)


def test_nonascending_budgets_refused():
    with pytest.raises(BudgetComparisonError,match="increasing"):
        compare_budgets(make(6,[1,2,3]),make(3,[1,2,3]))
