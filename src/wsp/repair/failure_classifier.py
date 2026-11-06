"""R2-D conservative, deterministic tokenizer-level generation diagnostics.

Failure labels are distinct. Hitting the sampling limit is not proof of a
repetition loop; repeated n-grams are only candidate loop evidence, not a
validated error or lack of math knowledge.
"""
from __future__ import annotations
from typing import Any


class TraceError(ValueError):
    """Generation trace cannot be classified under the frozen protocol."""


def repeated_token_span(token_ids: list[int], *, min_period: int = 3,
                        max_period: int = 32, min_copies: int = 3) -> dict[str, int] | None:
    """Detect a *consecutive* repeated token motif anywhere in one generation.

    Prefer the longest repeated run (tie-break: earlier start, smaller period).
    Nonconsecutive n-gram recurrence is not called a loop.
    """
    if (not isinstance(token_ids, list)
            or any(type(t) is not int or t < 0 for t in token_ids)):
        raise TraceError("Token IDs must be a list of nonnegative integers")
    if not (2 <= min_period <= max_period <= 64 and 2 <= min_copies <= 10):
        raise TraceError("Invalid repetition detection thresholds")
    n = len(token_ids)
    best = None
    for period in range(min_period, min(max_period, n // min_copies) + 1):
        for start in range(0, n - min_copies * period + 1):
            motif = token_ids[start:start + period]
            copies = 1
            while start + (copies + 1) * period <= n and token_ids[
                start + copies * period:start + (copies + 1) * period
            ] == motif:
                copies += 1
            if copies < min_copies:
                continue
            run = copies * period
            candidate = {"start_token": start, "period_tokens": period,
                         "copies": copies, "span_tokens": run}
            if (best is None or run > best["span_tokens"]
                    or (run == best["span_tokens"] and
                        (start, period) < (best["start_token"], best["period_tokens"]))):
                best = candidate
    return best


def classify_generation(token_ids: list[int], text: str, *,
                        finish_reason: str, max_new_tokens: int) -> dict[str, Any]:
    """Return independent observable flags and leave causal/accuracy claims unset."""
    if finish_reason not in ("eos", "length", "other_stop"):
        raise TraceError("Unknown finish reason")
    if (type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 4096
            or not isinstance(text, str) or len(text) > 200_000):
        raise TraceError("Invalid generation text or token budget")
    if not 1 <= len(token_ids) <= max_new_tokens:
        raise TraceError("Generation outside token budget")
    if finish_reason == "length" and len(token_ids) != max_new_tokens:
        raise TraceError("Length finish must exhaust token budget")
    repeat = repeated_token_span(token_ids)
    truncated = finish_reason == "length"
    return {
        "finish_reason": finish_reason,
        "generated_tokens": len(token_ids),
        "cut_at_limit": truncated,
        "consecutive_repetition_candidate": repeat is not None,
        "repetition_evidence": repeat,
        "boxed_marker_present": "\\boxed{" in text or "\\fbox{" in text,
        "accuracy": None,
        "mathematical_correctness": None,
        "loop_causal_supported": False,
        "failure_taxonomy_note": (
            "Length cut and consecutive token repetition are separate flags. "
            "Neither implies mathematically incorrect or verified loop."
        ),
    }
