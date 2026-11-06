# Category Taxonomy：研究能力与代码职责的一一映射

| Category | 核心任务 | 规划代码包 | 对应实验 | 预期输出 |
|---|---|---|---|---|
| C0 CORE | 模型/实验/预算/权限的不可变身份 | src/wsp/core/ | 全部 | ModelSpec, Budget, ExperimentPlan |
| C1 READ | 解释静态结构与动态激活 | src/wsp/read/ | 1/2/3/7 | ProbeRecord, DirectionEvidence |
| C2 REPAIR | 对局部故障提出有约束修改 | src/wsp/repair/ | 4/8 | PatchProposal, PatchManifest |
| C3 SYNTHESIZE | 将规则/程序转换为权重候选 | src/wsp/synthesize/ | 5/6 | CircuitIR, WeightProposal |
| C4 EVALUATE | 比较效果、统计与退化 | src/wsp/evaluate/ | 全部 | EvaluationReport |
| C5 AGENTS | 研究假设、预算与实验编排 | src/wsp/agents/ | 工程扩展 | ResearchTrace, NoEditReason |
| C6 ADAPTERS | HF/Qwen、game MLP、DDPM 统一访问 | src/wsp/adapters/ | 1–8 | TensorInventory, ForwardProbe |
| C7 ARTIFACTS | 哈希、数据血缘、补丁记录 | src/wsp/artifacts/ | 全部 | ArtifactManifest |

## 为什么这样划分？
- READ 的输出是证据，不是修改命令；避免直接把相关性当因果性。
- REPAIR 与 SYNTHESIZE 处理不同问题：前者以基座保持和局部修复为核心，后者允许从零描述参数。
- EVALUATE 独立于产生补丁的 Agent，确保 holdout 与工具权限不被绕过。
- CORE/ADAPTERS/ARTIFACTS 提供横切约束；不是独立训练算法。
- 后续如扩展到新模型，首先新增 adapter，而不是复制整个 Agent。

## 不实施任何隐含能力
当前所有子包仅供导入定位，不提供训练执行器、模型下载器、推理 API、GPU 调度和编辑操作。若 future plugin 增加，应另行做权限评审。
