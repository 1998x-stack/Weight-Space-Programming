# Sources & provenance registry

## Primary study
- Takehi, Rikiya; Teoh, Jayden; Zhang, Alex; Khattab, Omar (2026-11-02). [Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights](https://rikiyat.github.io/blog/weight-space-programming/). **Blog, not confirmed peer-reviewed conference paper.**
- [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs). Primary run logs, experiment directories, eval artefacts.
- [Exp 8 README and metric-to-file mapping](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/README.md); [Gemma follow-up](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp8_math/gemma_followup).
- [Qwen3.5-0.8B official model card](https://huggingface.co/Qwen/Qwen3.5-0.8B), including recommended text Thinking decoding.

## Independent verification
- [JonathanColetti/Qwen3.5-0.8B-Astra-WeightEdit](https://huggingface.co/JonathanColetti/Qwen3.5-0.8B-Astra-WeightEdit) — independent edited checkpoint hash recreation and MATH-500 Thinking On/Off test. Its card explicitly states GSM8K/IFEval not independently reproduced by that author.

## Foundational publications
- Meng et al. (2022), [ROME](https://arxiv.org/abs/2202.05262).
- Meng et al. (2022 preprint; later ICLR 2023), [MEMIT](https://arxiv.org/abs/2210.07229).
- Zou et al. (2023), [Representation Engineering](https://arxiv.org/abs/2310.01405).
- Bricken et al. (2023), [Towards Monosemanticity](https://www.transformer-circuits.pub/2023/monosemantic-features/index.html).
- Park, Choe and Veitch (ICML 2024), [Linear Representation Hypothesis](https://proceedings.mlr.press/v235/park24c.html).

## Secondary commentary
- 用户提供的 2026-11-05 [知乎回答](https://www.zhihu.com/question/2100892435056259459/answer/2101507493373424130) 及同一摘录中的另一项技术性解读。二者是观点与术语核对的线索，非模型实验的一手来源。

## Evidence standard
每项进入文档的数值需保留 source_url + experiment/run + exact path/metric key + conditions + evidence_type。对非公开内容、当前无法独立校验的信息明确标记“待证”。Last verified (documentation only): 2026-11-06.
