# 系统架构：从研究规范到可审计执行器

本文件定义**未来架构契约**，不是已部署系统；R1 已在 repair/replay.py、evaluate/paired_statistics.py 与 evaluate/math500.py 实现工具；其他核心系统仍以占位为主，未部署常驻服务或 Agent。

## 逻辑数据流

User Research Goal → Experiment Planner → Policy & Budget Gate → Probe Runner (read-only) → Evidence Store → Hypothesis/Attribution → Candidate Patch → Patch Validator → Independent Evaluation → Approve / No Edit → Immutable Report

## 模块边界
1. **core**：类型与规则、实验状态机、预算、模型元数据；不运行目标模型。
2. **adapters**：Qwen/HF Transformer，游戏 MLP，扩散模型；标准化模型参数清单和激活读取协议。
3. **read**：静态张量结构、前向探针、Teacher Forcing、对照方向、层级贡献、神经元评分。
4. **repair**：失败分类、候选生成、允许的编辑算子、范数/稀疏度约束、应用前后哈希验证。
5. **synthesize**：规则 IR、神经电路规划、参数布局与规则到权重的候选编译；没有执行器。
6. **evaluate**：评测适配、逐题配对比较、置信区间、非目标回归和统计汇总。
7. **agents**：提出受 Schema 约束的实验申请；不直接持有无限 Python 运行环境或权重写入权限。
8. **artifacts**：只追加的 manifest、模型版本锁、候选与最终补丁、实验决策记录。

## 应用架构原则
- **Immutable Base**：基座只读，补丁是版本化 overlay；允许 rollback。
- **Read / Propose / Apply / Evaluate 分权**：Agent 提交 ExperimentPlan；执行器审计和执行；保留 No-Edit。
- **Determinism**：随机种子、Prompt、tokenizer、dtype、归因样本和 grader 均入 manifest。
- **Leakage Defense**：Agent 可以看到练习集，不可读 holdout 标签；评测服务只返回事先定义的统计结果。
- **No Side-effect Claims**：测试失败、不可用和未运行与成功同等可报告。

## 状态机提案
CREATED → SOURCES_VERIFIED → BASELINE_MEASURED → DIAGNOSED → PATCH_PROPOSED → PATCH_VALIDATED → HOLDOUT_EVALUATED → {APPROVED, REJECTED, NO_EDIT}

任一状态允许进入 FAILED 并写入不可变错误事件。没有通过 hash / data-split / regression gate，禁止进入 APPROVED。

## 外部依赖
原作者 Harness 未公开，不能直接 import；参考数据来自原作者日志，未来需要本项目独立设计前向服务和隔离沙盒。详细字段：[Data Contracts](data-contracts.md)。
