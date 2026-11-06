# Exp 8 精确复现协议 v0.1（计划，不是已运行结果）

## 必须固定的对象
| 资源 | 作者记录/预期字段 |
|---|---|
| 模型 | Qwen/Qwen3.5-0.8B，text part，24 层 |
| 基座版本 | 2fc06364715b967f1860aea9cf38778875588b17 |
| 最终权重 | runs/p50c/edit/manifest.json、edit.npz；应用后每个 edited tensor 的 sha256 |
| 运行环境 | vLLM、bfloat16，版本需在首次实际复现时锁定 |
| 模板 | 官方 chat template，Thinking On，问题 + boxed answer 指示 |
| 解码 | temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5, max_new_tokens=32768 |
| 评测 | MATH-500×4 samples，GSM8K×2，其他见作者 README |
| 数学评分 | math-verify，提取最后一个 boxed 答案 |
| 原始结果 | eval/unedited/merged.json 与 eval/p50c/merged.json |

出处：[原作者 exp8 README](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/README.md)。

## 两个不同任务不可混淆
**A. Replay：** 不需要重新发现方向，只验证公开的 delta 在精确模型上构建出完全一致的 checkpoint，随后独立评分。  
**B. Rediscovery：** 不能在候选选择前读取作者的最终方向与掩码，需要自己从 train/diagnostic inputs 产生对照方向并验证。

## 数据分割与隔离
- practice：作者训练/练习题集合，不应与 holdout benchmark 混用。
- diagnostic：用于建立循环特征和可读激活对照。
- validation：选择 α、层和掩码等超参数。
- holdout：在所有选择冻结后只评估一次；严格控制评测集污染。
- negative-controls：随机方向、非重复失败样本、Gemma 对照。
每个样本须附 dataset_version、sample_id、prompt_hash、split、allowed_for_selection 的显式布尔标记。

## 生成记录必须包含
run_id、base_sha、patch_sha、model_revision、dtype、chat_template_hash、generation_engine/version、seed、sample_id、sample_index、prompt_hash、finish_reason、generated_tokens、cut_at_limit、raw_output_ref、extracted_answer、grade、grader_version 和 elapsed/cost。不要把空格连发等只按截断分类；另加 failure_taxonomy。

## 统计
按 **题目** 对 Edited−Base 的准确率差计算 paired bootstrap CI（建议冻结 bootstrap 种子与采样次数）；多采样在题目内聚合，不能把所有生成当作互相独立。另提供 McNemar/配对方向性统计及 Right→Wrong 样本审计。对多任务对照报告完整指标而非只报最佳。

## 失败响应
- Hash 不符：STOP，绝不继续标注为 faithful replay。
- 评分器不可用：使用替代评分器时另开 comparable-but-not-identical 轨道。
- 显存不足：记录 OOM 配置，不暗示运行成功。
- 复测不一致：报告配置差异、抽样误差与分析；不调整结果来迎合博客数值。
