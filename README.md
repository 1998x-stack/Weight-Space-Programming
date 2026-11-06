# Weight-Space Programming

> **Research-first, evidence-driven, reproduction-ready.**  
> 权重空间编程（WSP）研究笔记、原始实验复现路线与模块化代码骨架。当前处于 **R1 Checkpoint Replay 验证成功、MATH-500 GPU 评测待完成；R2-A CPU 机制候选分析已实现；R2-B 工具已实现，R2-C 的官方 Qwen3.5 CPU forward smoke 已在 GitHub Actions 实际成功（Layer 15、四对手工诊断输入，非机制复现）**：BF16 稀疏重建、哈希校验、配对统计及 GPU 评测脚本已实现；GitHub Actions 已用官方 Qwen3.5-0.8B 指定 revision 和公开 p50c 补丁完成真实张量 SHA256 验证。但本仓库尚未生成独立 MATH-500 分数。

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

## 4. Repository layout (R1 utilities plus future placeholders)

```text
Weight-Space-Programming/
├── README.md
├── pyproject.toml                 # R1 CPU dependency + optional dev tooling
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
│   ├── repair/                   # replay.py 已实现 BF16 p50c 补丁；其他模块占位
│   ├── synthesize/               # 规则→神经网络的编译设计
│   ├── evaluate/                 # paired_statistics.py / math500.py 已实现
│   ├── agents/                   # 实验决策与可审计轨迹
│   ├── adapters/                 # Transformer/RL/扩散模型适配
│   └── artifacts/                # 补丁清单、不可变记录
├── experiments/                  # exp01–exp08 设计占位
├── schemas/                      # 契约规范占位
├── tests/                        # R1 CPU 单元测试 + 后续测试占位
└── scripts/                      # r1.py / r1_math500.py / r1_prepare_math500.py
```

**实现状态：** R2-A 增加了纯 CPU 的 contrastive direction、neuron ranking、observational alignment、in-memory sparse proposal 与严格诊断输入 CLI；详见 [R2-A 实施与局限](docs/05-reproduction/r2-offline-implementation.md)。R2-B 已添加真实 PyTorch Qwen teacher-forcing hooks、BF16 基座校验、严格诊断输入和非破坏性前向干预工具；**尚无官方 Qwen R2-B 完整实验运行或机制复现证据**。详见 [R2-B 运行手册](docs/05-reproduction/r2b-activation-probe.md)、[R2-C 执行说明](docs/05-reproduction/r2c-real-smoke.md)及[官方模型实际执行审计](docs/05-reproduction/r2c-run-20261106.md)。R1 CPU replay/paired statistics 已通过合成数据单元测试；真实 Qwen p50c 重建与张量哈希经 GitHub Actions 验证；GPU runner 已编写但尚未运行。除这三个 R1 模块之外，其他 src/ 子包仍是占位。没有训练、Agent、自主权重发现实现。

## R1 Quickstart (separate CPU replay and GPU benchmark)

Use the exact upstream manifest and model revision. This pipeline has passed synthetic CPU tests **and** a real official-checkpoint p50c SHA replay in GitHub Actions; the MATH-500 numerical reproduction still requires a GPU run. [Detailed R1 protocol](docs/05-reproduction/r1-implementation.md).

1. Install CPU requirements: `python -m pip install -e '.[dev]'`.
2. Obtain `Qwen/Qwen3.5-0.8B` revision `2fc06364715b967f1860aea9cf38778875588b17` and `RikiyaT/lm_understand-logs/exp8_math/runs/p50c/edit/` from their official sources.
3. Replay: `python scripts/r1.py replay --base /models/qwen35 --edit /sources/lm_understand-logs/exp8_math/runs/p50c/edit --out /models/qwen35-p50c`.
4. After preparing the pinned MATH-500 dataset and installing a compatible GPU environment, run `scripts/r1_math500.py` separately for base and edited checkpoints.
5. Compare: `python scripts/r1.py compare --base /runs/base/math500_per_problem.json --edited /runs/edited/math500_per_problem.json --samples 4`.

Actual official-weight SHA replay: [GitHub Actions run #42187045405](https://github.com/1998x-stack/Weight-Space-Programming/actions/runs/42187045405) (success; audit artifact only, no weight uploads). MATH-500 was not run.

Run only CPU unit tests with `python -m pytest -q tests/unit`. No GPU, weight download or model edit is performed by this test command.

## 5. Reproduction ladder

| Gate | 目标 | 完成定义 |
|---|---|---|
| **R0 Evidence** | 核对博客、日志、模型卡及文件哈希 | 数据源索引、证据登记和差异说明 |
| **R1 Replay** | 真实官方权重补丁 SHA 校验已成功；GPU MATH-500 仍待运行 | 精确权重/配置、逐题结果与置信区间 |
| **R2 Rediscover** | CPU 候选估计代码已实现；真实机制发现仍待验证 | 方向相似度、层归因、单层及随机消融 |
| **R3 Agent** | 受预算约束的自动发现系统 | 独立盲测、多次运行、失败可报告 |
| **R4 Generalize** | Gemma、其他模型和真实局部故障 | 与采样/LoRA/无编辑方案比较，跨任务回归 |

R0/R2/R3/R4 的通用工程目标仍未完成；**R1 真实补丁已重建，但基准评测门禁尚未通过**。详情：[复现路线](docs/05-reproduction/roadmap.md) 与 [验收标准](docs/07-roadmap/milestones.md)。

## 6. Evidence policy

每个数值结论都应标出：（a）原始出处；（b）属于研究报告、复测或假设；（c）运行条件；（d）是否能由公开产物复核。知乎解读是二手材料，不能代替原始实验和论文。实验不可用时写 **not verified**，禁止编造日志、测试通过状态或权重下载结果。

**推荐引用：**

- Takehi et al. (2026), [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) — 研究博客，**不是已确认的同行评审论文**。
- [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs) — 一手实验日志。
- Meng et al. (2022), [ROME](https://arxiv.org/abs/2202.05262)；Meng et al. (2023), [MEMIT](https://arxiv.org/abs/2210.07229)。
- Zou et al. (2023), [Representation Engineering](https://arxiv.org/abs/2310.01405)；Park et al. (2024), [Linear Representation Hypothesis](https://proceedings.mlr.press/v235/park24c.html)。
- Qwen 官方 [model card](https://huggingface.co/Qwen/Qwen3.5-0.8B)。

本仓库不提供官方模型权重，也不会在没有执行验证的情况下发布“复现成功”的说明。R1 的明确运行说明见 [R1 Implementation](docs/05-reproduction/r1-implementation.md)。

## 7. How to contribute

当前阶段欢迎改进：一手资料对应的事实核对、公式维度检查、复现协议、架构模块契约、测试设计。请参考 [CONTRIBUTING.md](CONTRIBUTING.md)。实现代码应留给 R1 之后的独立 PR，不应把占位模块宣称为已实现功能。

---

**Status:** R1 official tensor replay verified; R2-A/R2-B implemented; R2-C genuine official Qwen fixed-input Layer-15 forward smoke verified (Actions #42200064239); R1 GPU benchmark and R2 scientific rediscovery not completed · **Last reviewed:** 2026-11-06.
