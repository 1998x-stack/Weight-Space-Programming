# Weight-Space Programming

> **Research-first, evidence-driven, reproduction-ready.**  
> 权重空间编程（WSP）研究笔记、原始实验复现路线与模块化代码骨架。当前版本 **v0.1 — documentation & scaffolding only**；没有实现训练、推理、权重编辑，也没有声称复现成功。

[研究文档](docs/index.md) · [原始研究与证据](docs/02-original-study/source-audit.md) · [复现路线](docs/05-reproduction/roadmap.md) · [系统架构](docs/06-architecture/system.md) · [开发里程碑](docs/07-roadmap/milestones.md)

## 1. What is Weight-Space Programming?

Weight-Space Programming（WSP）指通用语言模型围绕自然语言目标，直接分析并可能修改已有神经网络的参数，或构造新网络权重。它不同于在目标网络上运行反向传播、优化器或常规 SFT；也不等同于无需测试就可以理解任意模型。核心研究包括：

| Category | 问题 | 对应原始实验 |
|---|---|---|
| **READ / Understand** | 从参数结构、激活与前向探针理解模型行为 | Exp 1 / 2 / 3 / 7 |
| **REPAIR / Edit** | 定位行为故障，直接修改部分参数并评估副作用 | Exp 4 / 8 |
| **SYNTHESIZE / Write** | 把规则或目标计算编译为神经网络权重 | Exp 5 / 6 |
| **EVALUATE / Verify** | 因果消融、配对比较、统计与跨任务回归 | 横跨全部实验 |
| **AGENT / Orchestrate** | 假设生成、预算控制、实验日志和可审计决策 | 研究扩展方向，非作者公布的通用软件 |

**源头说明：** 截至 2026-11-06，核心研究出处是 Rikiya Takehi、Jayden Teoh、Alex Zhang、Omar Khattab 的 2026-11-02 研究博客 [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) 及其[公开实验日志](https://github.com/RikiyaT/lm_understand-logs)。不能将其冒充已经同行评审的正式会议论文。本仓库是独立研究/复现规划，不隶属于原作者，也不是作者的原始代码镜像。

## 2. Key observation, not a universal capability

研究报告：在 Qwen3.5-0.8B 的 **Thinking On、固定解码配置** 下，Exp 8 在 50 次前向实验预算内找到两层 MLP 的局部权重补丁：每层 64 个输出列、共 131,072 个被触及的位置（占约 0.8B 参数的 0.016%）。MATH-500 从 **7.9% → 48.1%**，并伴随截断率从 **93.0% → 33.1%** 的下降。这里的“6 倍”主要说明缓解了模型的重复推理/无法结束问题，不证明凭空获得六倍数学知识。[来源及精确定义](docs/02-original-study/exp08-qwen.md)。

重要边界：

- 50 *passes* 并不是仅测试 50 道题。单个 generation pass 最多可包含 512 道练习题；teacher-forced read 最多 1,024 段文本。
- Exp 8 的 50-pass 运行还包含 **158 agent turns / 80 Python runs / 20.3 小时的墙钟时间**；不是瞬间修复。
- **Gemma 3 270M** 在同类探索中没有显示可靠的大幅数学收益。不能把局部故障修复外推成缺失知识的直接注入。
- 官方 Qwen 文本 Thinking 建议 temperature=1.0；研究中使用 0.6。要做 **解码与权重编辑的因子消融**。
- 独立复测与反例、无效尝试都应记录；结果显著并不等于普适、零副作用、生产就绪。

## 3. Documentation map

| 主题 | 内容 |
|---|---|
| [Start here](docs/index.md) | 阅读顺序、证据级别与研究边界 |
| [基础概念](docs/01-foundations/concepts.md) | 权重、激活、残差流、稀疏编辑、子空间 |
| [数学方法](docs/01-foundations/mathematics.md) | 对照方向、神经元评分、掩码投影、干预强度与局限 |
| [八组实验](docs/02-original-study/eight-experiments.md) | Exp 1–8 的输入、权限、任务与结论 |
| [证据审计](docs/02-original-study/source-audit.md) | 一手证据路径、可复现度、未开放组件 |
| [Qwen Exp 8](docs/02-original-study/exp08-qwen.md) | 数据、矩阵更新、协议及警告 |
| [负面结果](docs/02-original-study/negative-results.md) | Gemma、sycophancy 静态识别与游戏泛化 |
| [ROME / MEMIT](docs/03-literature/model-editing.md) | 模型编辑算法与 WSP 的关系 |
| [RepE / SAE / 几何](docs/03-literature/representations.md) | 表征与因果解释工具 |
| [论文对比](docs/03-literature/method-comparison.md) | 方法类别、假设、适用范围 |
| [知乎讨论核查](docs/04-critical-reading/zhihu-claims.md) | 主张逐条分级：支持、夸大、待证 |
| [研究有效性](docs/04-critical-reading/validity-threats.md) | 采样、污染、选择偏差和推断限制 |
| [复现路线](docs/05-reproduction/roadmap.md) | R0→R4 逐级验收 |
| [精确协议](docs/05-reproduction/protocol.md) | Checkpoint、数据切分、采样、评分、记录 |
| [实验矩阵](docs/05-reproduction/experiment-matrix.md) | 对照组、消融、种子与统计 |
| [环境与资源](docs/05-reproduction/environment.md) | 运行条件、显存、开销、许可 |
| [系统架构](docs/06-architecture/system.md) | 组件边界、输入输出与职责 |
| [模块分类](docs/06-architecture/category-taxonomy.md) | READ/REPAIR/SYNTHESIZE/EVALUATE/AGENT |
| [数据契约](docs/06-architecture/data-contracts.md) | Evidence/Probe/Patch/Evaluation/Run 等 schema |
| [安全与治理](docs/06-architecture/governance.md) | 沙盒、权限、回滚、禁止虚构结果 |
| [阶段里程碑](docs/07-roadmap/milestones.md) | 文件级任务清单与验收 |
| [原始链接](docs/references/sources.md) | Blog、代码、论文、模型卡和讨论 |

## 4. Planned repository layout (placeholders only)

```text
Weight-Space-Programming/
├── README.md
├── pyproject.toml                 # package metadata only; no runtime
├── CONTRIBUTING.md
├── .gitignore
├── docs/
│   ├── 01-foundations/           # 术语、数学、假设
│   ├── 02-original-study/        # 8 实验与证据审计
│   ├── 03-literature/            # ROME / MEMIT / RepE / SAE
│   ├── 04-critical-reading/      # 知乎观点与研究局限
│   ├── 05-reproduction/          # R0–R4、评测、环境
│   ├── 06-architecture/          # 架构、模块与契约
│   ├── 07-roadmap/              # 里程碑与验收标准
│   └── references/              # 一手文献及 BibTeX
├── configs/                      # 配置样例/占位
├── src/wsp/
│   ├── core/                     # 模型注册、预算、类型
│   ├── read/                     # 权重检查、层归因、表示读取
│   ├── repair/                   # 故障分类、候选补丁、约束
│   ├── synthesize/               # 规则→神经网络的编译设计
│   ├── evaluate/                 # 配对基线、统计与副作用
│   ├── agents/                   # 实验决策与可审计轨迹
│   ├── adapters/                 # Transformer/RL/扩散模型适配
│   └── artifacts/                # 补丁清单、不可变记录
├── experiments/                  # exp01–exp08 设计占位
├── schemas/                      # 契约规范占位
├── tests/                        # 单测/集成/回归/基准占位
└── scripts/                      # 未来复现命令占位
```

**当前仓库承诺：** 所有 `src/` Python 文件仅含 docstring / TODO 或占位声明，不加载真实模型、不执行编辑、不访问网络。 `experiments/` 是计划而非已运行结果；无宣称已通过的 CI/Benchmark。

## 5. Reproduction ladder

| Gate | 目标 | 完成定义 |
|---|---|---|
| **R0 Evidence** | 核对博客、日志、模型卡及文件哈希 | 数据源索引、证据登记和差异说明 |
| **R1 Replay** | 复用公开补丁与基线跑原始评测 | 精确权重/配置、逐题结果与置信区间 |
| **R2 Rediscover** | 重算重复方向、定位层和 Top-64 | 方向相似度、层归因、单层及随机消融 |
| **R3 Agent** | 受预算约束的自动发现系统 | 独立盲测、多次运行、失败可报告 |
| **R4 Generalize** | Gemma、其他模型和真实局部故障 | 与采样/LoRA/无编辑方案比较，跨任务回归 |

当前均为 **planned / not implemented / not evaluated**。详情：[复现路线](docs/05-reproduction/roadmap.md) 与 [验收标准](docs/07-roadmap/milestones.md)。

## 6. Evidence policy

每个数值结论都应标出：（a）原始出处；（b）属于研究报告、复测或假设；（c）运行条件；（d）是否能由公开产物复核。知乎解读是二手材料，不能代替原始实验和论文。实验不可用时写 **not verified**，禁止编造日志、测试通过状态或权重下载结果。

**推荐引用：**

- Takehi et al. (2026), [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) — 研究博客，**不是已确认的同行评审论文**。
- [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs) — 一手实验日志。
- Meng et al. (2022), [ROME](https://arxiv.org/abs/2202.05262)；Meng et al. (2023), [MEMIT](https://arxiv.org/abs/2210.07229)。
- Zou et al. (2023), [Representation Engineering](https://arxiv.org/abs/2310.01405)；Park et al. (2024), [Linear Representation Hypothesis](https://proceedings.mlr.press/v235/park24c.html)。
- Qwen 官方 [model card](https://huggingface.co/Qwen/Qwen3.5-0.8B)。

本仓库不提供官方模型权重，也不会在没有执行验证的情况下发布“复现成功”的说明。

## 7. How to contribute

当前阶段欢迎改进：一手资料对应的事实核对、公式维度检查、复现协议、架构模块契约、测试设计。请参考 [CONTRIBUTING.md](CONTRIBUTING.md)。实现代码应留给 R1 之后的独立 PR，不应把占位模块宣称为已实现功能。

---

**Status:** v0.1 specification / skeleton · **Last reviewed:** 2026-11-06 · **No training or inference implemented.**
