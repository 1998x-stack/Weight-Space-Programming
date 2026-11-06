"""R2-D budget sensitivity analysis on two fully recorded diagnostic runs.

This supports honest comparison of censored traces without treating differences
as model improvement. Inputs and decoding protocol are fail-closed checked.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any

from wsp.repair.failure_classifier import classify_generation, TraceError


class BudgetComparisonError(ValueError):
    """Different runs cannot be interpreted as a token-cap-only contrast."""


def _validate_record(row: dict, limit: int) -> None:
    if not isinstance(row, dict) or not all(k in row for k in (
            "prompt_id", "prompt_sha256", "input_tokens",
            "generated_token_ids", "generated_text", "generated_text_sha256",
            "classification")):
        raise BudgetComparisonError("Missing original per-prompt records")
    if not isinstance(row["prompt_id"], str) or not row["prompt_id"]:
        raise BudgetComparisonError("Invalid prompt ID")
    if not isinstance(row["generated_text"], str):
        raise BudgetComparisonError("Invalid generated text")
    if hashlib.sha256(row["generated_text"].encode()).hexdigest() != row["generated_text_sha256"]:
        raise BudgetComparisonError("Generated text SHA mismatch")
    try:
        measured = classify_generation(
            row["generated_token_ids"], row["generated_text"],
            finish_reason=row["classification"]["finish_reason"], max_new_tokens=limit)
    except (TraceError, KeyError, TypeError) as exc:
        raise BudgetComparisonError("Invalid generation classification") from exc
    if measured != row["classification"]:
        raise BudgetComparisonError("Classifications do not match actual token IDs")


def compare_budgets(short: dict[str, Any], long: dict[str, Any]) -> dict:
    status = "real_model_generated_diagnostic_traces_no_scientific_loop_claim"
    if any(not isinstance(x, dict) or x.get("status") != status for x in (short, long)):
        raise BudgetComparisonError("Not a recognized R2-D original trace result")
    a, b = short.get("decoding", {}), long.get("decoding", {})
    if a.get("mode") != "greedy" or b.get("mode") != "greedy":
        raise BudgetComparisonError("Only comparable greedy pilots are supported")
    sa, sb = a.get("max_new_tokens"), b.get("max_new_tokens")
    if type(sa) is not int or type(sb) is not int or not 1 <= sa < sb <= 4096:
        raise BudgetComparisonError("Require strictly increasing token budgets")
    for key in ("mode", "thinking_template", "max_input_tokens", "do_sample"):
        if a.get(key) != b.get(key):
            raise BudgetComparisonError(f"Decoding mismatch: {key}")
    for key in ("model_revision_declared", "model_attestation_scope",
                "base_target_bf16_sha256", "model_config_sha256",
                "input_prompts_sha256", "torch", "transformers", "device"):
        if short.get("provenance", {}).get(key) is None or (
                short["provenance"].get(key) != long.get("provenance", {}).get(key)):
            raise BudgetComparisonError(f"Model/source/runtime mismatch: {key}")
    rshort, rlong = short.get("records"), long.get("records")
    if (not isinstance(rshort, list) or not isinstance(rlong, list)
            or not rshort or len(rshort) != len(rlong)):
        raise BudgetComparisonError("Missing or mismatched record lists")
    if (len({r.get("prompt_id") for r in rshort}) != len(rshort)
            or len({r.get("prompt_id") for r in rlong}) != len(rlong)):
        raise BudgetComparisonError("Duplicate prompt ID")
    b_by_id = {r["prompt_id"]: r for r in rlong}
    if {r["prompt_id"] for r in rshort} != set(b_by_id):
        raise BudgetComparisonError("Prompt IDs differ")
    rows=[]
    for x in rshort:
        y=b_by_id[x["prompt_id"]]
        if (x["prompt_sha256"] != y["prompt_sha256"]
                or x["input_tokens"] != y["input_tokens"]):
            raise BudgetComparisonError("Prompt metadata mismatch")
        _validate_record(x, sa)
        _validate_record(y, sb)
        p=x["generated_token_ids"]
        q=y["generated_token_ids"]
        if q[:len(p)] != p:
            raise BudgetComparisonError("Token prefix diverged: no length-only causal comparison")
        rows.append({
            "prompt_id": x["prompt_id"],
            "short_tokens": len(p),
            "long_tokens": len(q),
            "short_cut_at_limit": x["classification"]["cut_at_limit"],
            "long_cut_at_limit": y["classification"]["cut_at_limit"],
            "short_repetition_candidate": x["classification"]["consecutive_repetition_candidate"],
            "long_repetition_candidate": y["classification"]["consecutive_repetition_candidate"],
            "prefix_bitwise_identical": True,
        })
    return {
        "schema_version": "r2d-budget-sensitivity-v1",
        "status": "token_budget_sensitivity_only_not_scientific_replication",
        "short_cap": sa, "long_cap": sb, "problems": len(rows),
        "all_short_prefixes_identical": True,
        "short_cut_count": sum(x["short_cut_at_limit"] for x in rows),
        "long_cut_count": sum(x["long_cut_at_limit"] for x in rows),
        "short_repetition_candidate_count": sum(x["short_repetition_candidate"] for x in rows),
        "long_repetition_candidate_count": sum(x["long_repetition_candidate"] for x in rows),
        "rows": rows,
        "interpretation": "Prefix equivalence holds, but length censoring and unverified loop labels remain.",
        "r2_scientific_gate_passed": False,
    }


def compare_files(short_path: str | Path, long_path: str | Path) -> dict:
    return compare_budgets(json.loads(Path(short_path).read_text()),
                           json.loads(Path(long_path).read_text()))
