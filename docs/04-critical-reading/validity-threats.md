# 有效性威胁与复现研究设计

## Internal validity / 内部效度
- **Temperature 混杂：** 作者使用 0.6；[Qwen 模型卡](https://huggingface.co/Qwen/Qwen3.5-0.8B) 对文本 Thinking 推荐 1.0。应正交控制编辑标志、温度、思考开关、惩罚和最大 token。
- **模型与提示精确性：** tokenizer/chat template、system prompt、boxed answer、dtype、推理引擎版本都可能影响结果。
- **选择偏差：** 多次候选编辑、不同 pass 预算中选择最佳，需要记录所有尝试、失败与最终冻结时间。
- **统计单位：** 同题多采样不能全部视为独立题目；建议按题目 cluster bootstrap 计算 95% CI。

## Construct validity / 测量是否正确
- “cut at max tokens” 不是所有情形的“逻辑循环”，需进一步结合重复片段密度、EOS、空白输出和终止原因分类。
- MATH-500 准确率提升可能来自能够给出最终 boxed 答案，不能直接解读为推理深度变化。
- IFEval 需要按官方 strict checker；不能把自由文本自评当成同指标。

## External validity / 外部效度
- Exp 8 是一个基座、一组目标故障和有限预算；Gemma 负例提示尚未证明普遍性。
- 需扩展非数学任务、非重复故障、不同网络结构和多个独立模型。
- 实验日志公开不代表 Agent 再次从零独立发现必然成功。

## Reproducibility / 可复现性
完整发布至少包括 model revision、dtype、template、config、数据列表哈希、随机种子、生成结果、score 与长度、补丁 manifest、前后 tensor 哈希、git SHA、硬件和成本。

## 实验拒收规则
没有冻结评测集、缺少原始 checkpoint hash、把自测数据用于选择补丁、改变 Prompt 却声称严格复制、缺少 Right-to-Wrong 统计、评测失败后仅挑最好一次，都应拒绝报告“复现通过”。
