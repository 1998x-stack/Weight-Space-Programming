# 文件级复现代码逻辑设计：模块输入/输出与调用次序

> 本文档定义 **proposed APIs**，不是已有 Python 实现。全部 Python 文件都是只含文档字符串的占位模块。未来实际 API 必须在实现 PR 中明确签名、错误模型、测试和资源成本。

## 1. 从静态阅读走向机制诊断
| 文件 | 未来输入 | 未来输出 | 约束 |
|---|---|---|---|
| core/registry.py | ModelSpec, source revision | read-only tensor inventory | revision/hash 必填 |
| adapters/qwen.py | model id, tokenizer config | Qwen tensor names / forward adapter | 不可修改基座 |
| read/weight_inspector.py | tensor inventory | structure/norm report | 不推测语义真实性 |
| read/activation_probe.py | ProbeRequest, diagnostic samples | ProbeRecord, activation refs | 只可读，token 位置一致 |
| read/contrastive_direction.py | paired productive/repetitive states | DirectionEvidence | 方向范数和样本 provenance |
| read/layer_attribution.py | per-layer residual / attention / mlp | layer writing scores | 需归因对照 |
| read/neuron_score.py | MLP activations, output weights, direction | ranked neuron scores | 维度/激活定义一致 |

## 2. 从诊断走向权重修复
| 文件 | 输入 | 输出 | 约束 |
|---|---|---|---|
| repair/failure_classifier.py | raw generations, answers | typed failures | loop vs cut vs blank 必须分开 |
| repair/sparse_edit.py | frozen tensor, selected columns, direction, alpha | PatchProposal | 仅合法 allowlist，无写基座 |
| repair/validator.py | PatchProposal, model hash, policy | PatchManifest or rejection | 维度、dtype、L0/L2、hash 逐项验证 |
| repair/no_edit.py | failures / statistical evidence | NoEditDecision | 找不到可靠机制可以不编辑 |
| artifacts/manifest.py | approved proposal, artifact store refs | immutable provenance manifest | 记录 sha256、无敏感内容 |
| artifacts/audit.py | state transitions and tool calls | append-only audit trail | 不修改既有事件 |

## 3. 评测 / 实验 Agent 的分权
| 文件 | 输入 | 输出 | 门禁 |
|---|---|---|---|
| evaluate/benchmark.py | model, benchmark config, frozen split | per-problem predictions | 最终答案提取规则固定 |
| evaluate/paired_statistics.py | paired per-problem scores | effect size & CI | 题目是统计采样单位 |
| evaluate/ablation.py | preregistered experiment matrix | control/ablation summaries | 参数网格预先登记 |
| evaluate/regression.py | target + off-target results | gate decision | Right-to-Wrong 不可忽略 |
| agents/planner.py | goal, evidence, budget | structured ExperimentPlan | 禁止访问 holdout labels |
| agents/runner.py | approved ExperimentPlan | ResearchTrace / final decision | 预算和权限由执行器强制 |
| agents/reviewer.py | artifact bundle and metrics | Approved / Rejected / No Edit | 不接受模型自评分 |
| core/budget.py | requested & used pass/token/time | remaining budget / stop | 失败也记账且可审计 |

## 4. 从规则合成网络：独立子路径
synthesize/circuit_ir.py（规则→可解释计算 IR）→ synthesize/compiler.py（IR→参数候选）→ synthesize/verification.py（候选网络与参考规则/基线比较）。这些与现有模型修复不同，不能共用其成功指标。游戏类输入通过 adapters/game.py；小型扩散网络通过 adapters/ddpm.py。

## 5. 五阶段完整调用序列（未来）
```text
registry -> adapter -> failure_classifier -> activation_probe
        -> contrastive_direction -> layer_attribution -> neuron_score
        -> planner -> budget/policy -> sparse_edit -> validator
        -> benchmark -> paired_statistics -> ablation -> regression
        -> reviewer -> manifest/audit -> APPROVE / REJECT / NO_EDIT
```
每个箭头传递必须是有 schema_version 的对象或不可变 artifact 引用。模型实际前向、下载、GPU 调度与持久化仍是计划中的接口，绝不在 v0.1 假定可用。

## 6. 建议契约（文字接口，无可执行定义）
- inspect_model(ModelSpec) -> TensorInventory
- probe(ProbeRequest) -> ProbeRecord
- estimate_direction(ContrastPairs, ProbeRecords) -> DirectionEvidence
- rank_neurons(DirectionEvidence, Activations, Weights) -> NeuronRanking
- propose_edit(NeuronRanking, EditPolicy) -> PatchProposal
- validate_patch(PatchProposal, BaseManifest) -> PatchManifest | Rejection
- score_paired(BaseRun, EditedRun, BenchmarkSpec) -> EvaluationReport
- review(EvaluationReport, GatePolicy) -> Approval | NoEditDecision

## 7. 失败模式
任何 hash 不符、holdout 访问、预算超限、Tensor shape 不兼容、非目标回归超阈值、统计不稳定必须终止并写审计事件。禁止将异常吞掉再报告成功。

参照：[data contracts](data-contracts.md)、[reproduction protocol](../05-reproduction/protocol.md)、[milestones](../07-roadmap/milestones.md)。
