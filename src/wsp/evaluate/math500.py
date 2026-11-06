"""R1 MATH-500 vLLM runner for Qwen3.5 text-only checkpoints.

This is an independent implementation of the original protocol. It is not
labelled a faithful score until the original assets, versions and outputs are
actually validated on a GPU machine. Optional dependencies are lazy imports.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import re
from typing import Any

INSTRUCTION = 'Please reason step by step, and put your final answer within \\boxed{}.'
SAMPLING = dict(temperature=0.6, top_p=0.95, top_k=20,
                presence_penalty=1.5, max_tokens=32768, seed=0, n=4)
BOX_RE = re.compile(r'\\boxed|\\fbox')


def extract_final_box(text: str) -> str | None:
    if '</think>' in text:
        text = text.rsplit('</think>', 1)[-1].strip()
    hits = list(BOX_RE.finditer(text))
    if not hits:
        return None
    match = hits[-1]
    rest = text[match.end():]
    trim = rest.lstrip()
    if trim.startswith('{'):
        depth = 0
        for j, char in enumerate(trim):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    return trim[1:j].strip() or None
        return None
    if match.group() == '\\boxed' and rest.startswith(' '):
        return rest.strip().split('$')[0].split('\n')[0].strip() or None
    return None


def _correct(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    from math_verify import parse, verify
    try:
        g = parse('\\boxed{' + str(gold) + '}', parsing_timeout=5)
        p = parse('\\boxed{' + pred + '}', parsing_timeout=5)
        return bool(g and p and verify(g, p, timeout_seconds=5))
    except Exception:
        return False


def _load_dataset(path: Path) -> list[dict[str, str]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 500:
        raise ValueError('Faithful MATH-500 requires exactly 500 questions')
    ids = [str(r['id']) for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate problem ID')
    if any(not r.get('problem') or not r.get('answer') for r in rows):
        raise ValueError('Missing problem/answer')
    return rows


def run_math500(model: str | Path, dataset: str | Path, out: str | Path) -> dict[str, Any]:
    """Execute actual generation and scoring. Requires GPU/vLLM + math-verify 0.9.0."""
    model, dataset, out = Path(model).resolve(), Path(dataset).resolve(), Path(out).resolve()
    if out.exists():
        raise ValueError('Refusing to overwrite existing benchmark run')
    if not (model / 'config.json').exists():
        raise ValueError('Checkpoint config.json missing')
    rows = _load_dataset(dataset)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(str(model), trust_remote_code=False)
    prompts = [tok.apply_chat_template(
        [{'role': 'user', 'content': r['problem'] + '\n' + INSTRUCTION}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True
    ) for r in rows]
    llm = LLM(model=str(model), dtype='bfloat16', gpu_memory_utilization=0.85,
              max_model_len=SAMPLING['max_tokens'] + 2048, seed=0,
              trust_remote_code=False)
    outs = llm.generate(prompts, SamplingParams(**SAMPLING))
    if len(outs) != len(rows):
        raise RuntimeError('Incomplete model responses')
    for o in outs:
        if len(o.outputs) != 4 or any(x.finish_reason not in ('stop', 'length') for x in o.outputs):
            raise RuntimeError('Unexpected/incomplete output generation')
    out.mkdir(parents=True)
    per, cut, tokens_sum, total, correct = [], 0, 0, 0, 0
    with gzip.open(out / 'math500_generations.jsonl.gz', 'wt', encoding='utf8') as stream:
        for r,o in zip(rows,outs):
            flags = []
            for i,x in enumerate(o.outputs):
                pred = extract_final_box(x.text)
                good = int(_correct(pred, r['answer']))
                cut += int(x.finish_reason == 'length')
                tokens_sum += len(x.token_ids)
                correct += good
                total += 1
                flags.append(good)
                stream.write(json.dumps({'id':str(r['id']), 'sample_index':i,
                    'text':x.text, 'answer':pred, 'correct':good,
                    'finish_reason':x.finish_reason, 'generated_tokens':len(x.token_ids)}, ensure_ascii=False)+'\n')
            per.append({'id':str(r['id']), 'correct':flags,
                        'pred_first':extract_final_box(o.outputs[0].text)})
    (out/'math500_per_problem.json').write_text(json.dumps(per,indent=2,ensure_ascii=False)+'\n')
    data = {'benchmark':'math500','problems':len(rows),'samples_per_problem':4,
            'acc':correct/total,'cut_share':cut/total,'mean_tokens':tokens_sum/total,
            'dataset_sha256':hashlib.sha256(dataset.read_bytes()).hexdigest(),
            'protocol':'think35','sampling':SAMPLING,
            'model':str(model), 'warning':'Independent grader implementation; versions require external verification.'}
    (out/'scores.json').write_text(json.dumps(data,indent=2)+'\n')
    return data
