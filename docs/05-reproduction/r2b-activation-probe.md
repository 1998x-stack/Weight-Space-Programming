# R2-B: pinned Qwen activation probes and diagnostic causal ablations

Status: **tooling implementation, not an independent mechanistic rediscovery**. R1 full MATH-500 GPU evaluation is still pending. This phase adds actual PyTorch forward hooks, guarded model loading and fixed-input interventions; its default CI uses a tiny synthetic torch model, not the official 0.8B checkpoint.

## Concrete Qwen3.5 architecture

The original Qwen/Qwen3.5-0.8B repository at revision
\`2fc06364715b967f1860aea9cf38778875588b17\` has 24 text decoder layers, hidden width 1024 and MLP width 3584. In Hugging Face Transformers' Qwen3.5 implementation, the multimodal wrapper exposes:

- \`model.language_model.layers[L]\`: decoder block; a **pre-hook** observes the input residual state.
- \`model.language_model.layers[L].mlp.down_proj\`: a **pre-hook** observes the post-gated MLP activation (the vector multiplied by W); a **forward-hook** can ablate a direction in the output of that module without changing weights.

The \`down_proj.weight\` layout is [1024, 3584]. This version only allows layers 15 and 19, and checks shape/layer count. The corresponding official BF16 tensor hashes must match the original p50c public manifest *before* weights can be loaded. This attests **the two base target tensors only**, not all checkpoint files, tokenizer revision or upstream model provenance.

Source references: [Transformers Qwen3.5 model implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py), [pinned official config](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/config.json), [author Exp 8 protocol](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/README.md).

## Inputs and controls

The collector **must not** take any of the author's final p50c directions/masks, MATH-500 questions or test answers as a discovery input. Prepare a researcher-authored diagnostic file (not a benchmark):

\`\`\`json
{
  "split": "diagnostic",
  "pairs": [
    {"id": "diag-a", "productive": "An independently prepared productive reasoning context", "repetitive": "A matched context from a repetition diagnostic"},
    {"id": "diag-b", "productive": "A second context ...", "repetitive": "A second matched repetition ..."},
    {"id": "diag-c", "productive": "A third context ...", "repetitive": "A third matched repetition ..."}
  ]
}
\`\`\`

This JSON example describes the schema, not validated examples. Require 3–128 unique nonempty pairs; forbid labels, answers and extra fields. Texts are raw tokenizer inputs with special tokens, **not** the original author's hidden harness prompt template. For each text, pool the final k *input* token positions, after one forward-only pass with \`use_cache=False\`. Paired sequences can have different absolute lengths; store both lengths and explicitly report this confound. No ground-truth mathematical correctness is inferred from the human-assigned labels 'productive'/'repetitive'.

## Setup and reproducible commands

Use a compatible recent Transformers release with \`Qwen3_5ForConditionalGeneration\` support; the model loader is intentionally offline (\`local_files_only=True\`); model asset acquisition is separate.

\`\`\`bash
python -m pip install -e '.[dev,r2-probe]'
hf download Qwen/Qwen3.5-0.8B \
  --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --local-dir /models/qwen35-original

python scripts/r2_collect.py \
  --model /models/qwen35-original --pairs /data/diagnostic_pairs.json \
  --layer 15 --last-k 16 --max-input-tokens 512 --device cuda \
  --out /runs/r2b/layer15 --confirm-compute

python scripts/r2_offline.py \
  --probes /runs/r2b/layer15/diagnostic.npz \
  --manifest /runs/r2b/layer15/diagnostic.manifest.json \
  --top-k 64 --alpha 1 --out /runs/r2b/layer15/candidate.json

python scripts/r2_intervene.py \
  --model /models/qwen35-original --pairs /data/diagnostic_pairs.json \
  --candidate /runs/r2b/layer15/candidate.json \
  --pair-id diag-a --layer 15 --alpha 1 --device cuda \
  --out /runs/r2b/layer15/intervention.json --confirm-compute
\`\`\`

Device may be \`cpu\`, but CPU full-model inference may be slow. CUDA must exist when requested. All commands refuse to overwrite outputs; the collector stores **diagnostic arrays** and a manifest only (model weight copy is included in the NPZ as required for R2-A; do not upload NPZ into a public action artifact without separately reviewing its licensing). Model checkpoints are never changed.

## Interpretation and validation plan

- Collector produces the exact five NPZ fields consumed by R2-A, plus pre-hook provenance, pinned-base target SHA verification, model/package versions and token lengths. The copied FP32 weight SHA in the manifest is **not** the published BF16 tensor hash.
- Intervention projects out a unit direction from the selected layer's **last-token down_proj output** (alpha in [0,1]). It compares the same fixed prompt's next-token logits with and without that hook; reports KL/logit-L2, **not** a benchmark score, a decrease in loops, or a causal model-accuracy result.
- Verification gate B1: run on the authentic fixed-revision Qwen weights and show real hook shapes, manifest hashes, and outputs; keep raw prompt text private.
- Gate B2: four independent matched contrast groups, shuffle/random directions, position-length matching, robustness across seeds, and layer-15 vs layer-19 validation.
- Gate B3: independent actual generation interventions and held-out regressions. No discovery-stage reads from author final edit mask or test answer key.
- Gate R1: finish full MATH-500 baseline/edited GPU benchmark and paired confidence interval. It is independent of the R2-B tooling gate.

Tests: \`python -m pytest -q tests/unit\` (contract tests always; PyTorch synthetic tests only when torch installed), and \`python -m compileall -q src/wsp scripts\`. Neither proves a scientific mechanism.
