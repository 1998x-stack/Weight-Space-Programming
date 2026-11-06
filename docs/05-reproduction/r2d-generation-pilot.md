# R2-D1 — Official-model generation trace pilot (not R2 rediscovery)

Status: code prepared; actual execution will be recorded only after CI results.

## Frozen design

- Official fixed-revision Qwen/Qwen3.5-0.8B (2fc06364715b967f1860aea9cf38778875588b17) with both original p50c target-tensor SHA256 values confirmed at preflight.
- Four manually written arithmetic/algebra prompts in experiments/exp08_qwen_repair/generation_pilot.json, all diagnostic; NOT MATH-500, not published repair mask, not holdout.
- Real model.generate autoregressive decoding; thinking-enabled official tokenizer chat template, greedy/no sampling, at most 96 new tokens and 160 prompt tokens. This is NOT the authors' temperature=0.6/presence-penalty benchmark protocol.
- Per-trace record: generated text, token IDs, token counts, EOS vs length vs other stop, SHA256 prompt/output fingerprints and pinned model/runtime context.
- Only public math prompts and their machine-generated responses are saved in the CI artifact. Do not use this workflow for private prompts, credentials or PII. 40-minute cap, 14-day artifact retention, no model checkpoint uploads.

## Conservative taxonomy

The classifier records *separate* binary properties:
(1) cut_at_limit means the run reached its maximum new token count. It is NOT evidence by itself of a looping error.
(2) consecutive_repetition_candidate flags three or more consecutive copies of a 3–32 token motif anywhere in the output; this is a heuristic, not proof of a real failure.
(3) boxed_marker_present records a literal LaTeX final-answer marker, not accuracy or completion proof.
(4) accuracy and mathematical_correctness remain null, never zero by default.

If no eligible real outputs are observed, productive/repetitive pairing is explicitly zero and no fabricated contrasts or R2 results are produced. Repetition across different sections or semantic cycling may be missed by this conservative detector.

## Command

    python -m pip install -e '.[dev,r2-probe]'
    hf download Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 --local-dir /models/qwen35
    python scripts/r2d_generate.py --model /models/qwen35 \
      --prompts experiments/exp08_qwen_repair/generation_pilot.json \
      --out /runs/r2d/traces.json --device cpu \
      --max-new-tokens 96 --max-input-tokens 160 --confirm-compute

Workflow: .github/workflows/r2d-real-generations.yml. It does not edit weights and refuses to overwrite outputs.

## Acceptance

R2-D1 engineering gate: authentic model loaded and produced four genuine token traces in a successful CI run, with checkable provenance and non-overwriting artifact. R2-D2 scientific next step: multiple random seeds, independently witnessed and human-reviewed successful/repetitive traces, matched prompt/token locations, 15/19-layer contrasted measurements, preregistered controls and split isolation. R2-D3: actual generation-level causal intervention plus untouched holdout and off-target regressions. R1 MATH-500 base/edited GPU benchmark remains open.

## Execution log

To be added from real GitHub Actions outputs only; synthetic tests do not meet the engineering gate.
