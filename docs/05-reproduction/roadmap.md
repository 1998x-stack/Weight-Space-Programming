# 从证据核验到自主复现：R0–R4 路线

> 全部阶段当前为 **NOT STARTED**。本仓库只有接口/目录占位，不含可运行的实验。

## R0 — Source Lock & Audit（先完成）
输入：原始博客、作者 GitHub、原始模型卡、第三方复测。产物：source registry、commit/revision、实验指标对应文件与 license/数据权限清单。门禁：任何数字都有一手定位、同名指标的口径一致、标记缺失的 Harness 或 grader。这里不需 GPU。

## R1 — Deterministic Checkpoint & Benchmark Replay
1. 从经过许可的发布源下载精确 revision（原始实验记录 2fc06364715b967f1860aea9cf38778875588b17），校验文件与 tokenizer。
2. 在可信隔离环境运行作者的 apply_edit；验证输出张量哈希和只有允许的两个矩阵发生修改。
3. 锁定作者 vLLM、BF16、提示模板、生成参数、math-verify 与解码；在可得许可的数据集上分别跑 baseline/edited。保留 token-level 原始结果及错误类型。
4. 对 500 道题的配对变化按题目聚类推导 CI，公开完整 metric manifest。
**Gate R1：** 检查点哈希完全一致；可明确区分“相同评测配置”与“确实获得相同分数”；若数据或 grader 不可得，只能报告部分复现，不得宣称 R1 完成。

## R2 — Local Mechanism Rediscovery
1. 自行从固定训练/诊断样本构造匹配的 productive/repetitive contexts，隔离最终测试集。
2. Teacher forcing 提取同一 token 位置的隐藏状态与 MLP 激活。
3. 计算每层对照方向；依次评估层贡献、MLP 与 Attention 贡献、候选神经元评分。
4. 对比 Exp 8 层 15/19 及 Top-64，执行单层、双层、随机同范数方向、不同 α 的前向干预。
5. 保存从探针输入→输出→选择决策→权重差分的证据链。
**Gate R2：** 能在未看到作者最终 Top-64 掩码的发现阶段重新定位机制，并在独立集验证；否则仅为 formula replay，不是 rediscovery。

## R3 — Agent-Guided Search
将“实验提议→预算审批→只读探针→候选修改→统计评测→选择或 No Edit”组织成确定状态机。Agent 只产出结构化 plan，执行器负责权限、张量 allowlist 与审计。Agent 无权直接改基座 checkpoint、读测试答案或修改审核门禁。
**Gate R3：** 不同随机种子和独立运行均报告成功率、中位搜索成本、No-Edit 频率与副作用；不能只提供一个成功故事。

## R4 — Generalization and Alternatives
Gemma 270M、另一架构/另一故障模式、规则可编译的小网络作为研究目标；同时比较：
- 原始模型使用官方解码；
- 原始模型使用作者解码；
- Edited 模型在两种解码；
- Thinking Off；
- 小规模预算匹配的训练/适配路线（若可行）；
- 单纯应用层超时和重试控制（用作诊断基线，不替代推理机制）。

**Gate R4：** 预先登记目标/负例、回归门槛与统计方法，报告失败实验，量化单位收益成本。

## 执行序列
R0 → R1 (patch replay) → R2 (mechanism) → R3 (agent) → R4 (generalization)。各阶段成果分别封存，不得先读取 R4 最终评测结果再回头选择 R2 编辑。
