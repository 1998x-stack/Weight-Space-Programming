# R1 实施记录：Checkpoint Replay & MATH-500

> 代码级进展：**CPU 重建器与实际官方 Qwen p50c 重建成功、评测脚本及配对统计已实现，但独立数学分数尚未产生**。本文件同时说明复现条件与缺失证据，不把作者的 7.9%/48.1% 计为本仓库已测结果。

## 1. 原始材料锁定

- [作者日志 exp8_math](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp8_math)
- [p50c patch manifest](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/runs/p50c/edit/manifest.json)
- [作者 apply_edit.py](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/code/apply_edit.py)
- 基座模型：Qwen/Qwen3.5-0.8B, revision 2fc06364715b967f1860aea9cf38778875588b17
- 补丁张量：layers.15 与 layers.19 MLP down_proj，1024×3584（每层 64 个输出列）。
- BF16 changed entries：65,260 + 64,928 = 130,188；对应张量前后 SHA256 已硬编码为严格模式的 allowlist，拒绝不一致的 manifest。

## 2. 已实现模块

| 模块 | 能力 | 在本次环境执行？ |
|---|---|---|
| src/wsp/repair/replay.py | 基座张量 BF16 原始位读取、作者 sparse NPZ 补丁应用、前后 SHA256、非破坏性目录输出、拒绝覆盖 | **已运行 6 个合成测试，并在真实官方模型上通过 BF16 张量 hash 校验** |
| src/wsp/evaluate/paired_statistics.py | 按问题 ID 对齐、按题采样 Bootstrap CI、Wrong→Right 等转移统计 | **已运行 4 个合成测试** |
| src/wsp/evaluate/math500.py | 以作者 think35 采样配置进行 MATH-500×4 推理和 math-verify 0.9.0 数学评分 | 代码已实现；未在 GPU 运行 |
| scripts/r1.py | replay / compare 命令入口 | 代码已实现 |
| scripts/r1_math500.py | 需显式确认 GPU 成本的评测入口 | 代码已实现；未运行 GPU |
| scripts/r1_prepare_math500.py | 指定修订版本、生成 500 条评测 JSONL 与源文件 hash | 代码已实现；数据未下载 |
| tests/unit/test_r1_*.py | 11 项 CPU 测试 | **11 passed（本地测试环境）** |

可复现性限制：测试采用人造小张量，证明**工具操作正确性**，不证明真实 Qwen 模型已被成功修改。未完整复现原始模型并执行评测前，不能宣告 R1 gate 通过。

## 3. 真实官方 Checkpoint Replay 执行证据（2026-11-06）

GitHub Actions 在公共 Ubuntu runner 上下载 Qwen/Qwen3.5-0.8B 固定 revision 及 RikiyaT/lm_understand-logs 的 p50c manifest/edit.npz，运行本仓库 scripts/r1.py replay（strict=True）。[完整工作流运行 https://github.com/1998x-stack/Weight-Space-Programming/actions/runs/42187045405](https://github.com/1998x-stack/Weight-Space-Programming/actions/runs/42187045405) 结论 **success**。

- GitHub Actions workflow: .github/workflows/r1-real-replay.yml
- Workflow run ID: 42187045405；Git commit: 86d16c2428575da5a78ef6718886cbb212a698e2
- 两层修改后的 SHA256 与原作者公开 manifest 的已固定值一致。
- changed_tensors=2；changed_entries=130188；重建脚本检查成功；原始基座未被覆盖。
- 上传的是 SHA 审计报告（GitHub Actions artifact ID 13488535024），**不是**模型权重。
- 没有执行 vLLM 数学推理；作者的 48.1% 不能写为这次测得结果。

## 4. 获取源资产（在有网络的机器上）

首先确认官方权重许可、数据许可与磁盘空间，创建独立工作目录。示例：

    python -m pip install -e '.[dev]'
    python -m pip install 'huggingface_hub[cli]'
    hf download Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 --local-dir /models/qwen35
    git clone https://github.com/RikiyaT/lm_understand-logs.git /sources/lm_understand-logs

注意：原作者用的是 Qwen3.5 的 text-only 导出；官方模型文件必须与原始 manifest 的两个张量 hash 匹配，才能进入下一步。如果不匹配，本地工具会 fail closed。不得尝试忽略失败哈希继续运行。

## 5. CPU 离线重建与 SHA 验证

    python scripts/r1.py replay \
      --base /models/qwen35 \
      --edit /sources/lm_understand-logs/exp8_math/runs/p50c/edit \
      --out /models/qwen35-p50c

上面的脚本不访问网络，不修改基座；严格要求 p50c 的两层、两个公开 SHA、对应 shape 和总 changed entries。输出目录中的 wsp_r1_audit.json 记录验证后的张量 hash。**注意**：该报告验证两个修改张量和基座目录中的其他文件被拷贝，而非自动验证整个模型目录与 Hugging Face revision 的完全一致；revision 必须通过下载工具固定。

## 6. MATH-500 评测执行（真实 GPU，未在本轮执行）

需要 vLLM、transformers、datasets、math-verify==0.9.0、足够 GPU 显存。请记录实际版本，原始 Harness 没有全部公开，因此这是独立实现的相近协议，不应未经核对就称为字节级、评分器完全一致。

先确定 MATH-500 数据集的具体 commit（不要用浮动 main），然后：

    python scripts/r1_prepare_math500.py --revision <MATH500_DATASET_COMMIT> --out /data/math500.jsonl
    python scripts/r1_math500.py --model /models/qwen35 --dataset /data/math500.jsonl --out /runs/base --confirm-gpu-cost
    python scripts/r1_math500.py --model /models/qwen35-p50c --dataset /data/math500.jsonl --out /runs/edited --confirm-gpu-cost
    python scripts/r1.py compare --base /runs/base/math500_per_problem.json --edited /runs/edited/math500_per_problem.json --samples 4

think35 的采样设定是 temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5, n=4, max_tokens=32768, seed=0，官方聊天模板 enable_thinking=True，最后一个 boxed 答案由 math-verify 判定。单题多采样不当作独立问题计算 CI；如模型输出无法完成或数据缺失，评测失败，不自动补造记录。

如果是从官方完整 multimodal 权重下载，需要确认 vLLM 是否能按原文的 text-only 模式加载；需要的转换尚未包含在本次 R1 实现范围内。

## 7. 与作者报告的对照（纯外部参考，不是本轮测得）

| Metric | 作者 base | 作者 p50c |
|---|---:|---:|
| MATH-500 accuracy | 7.9% | 48.1% |
| MATH-500 cut_share | 93.0% | 33.1% |
| GSM8K accuracy | 13.8% | 67.1% |
| IFEval accuracy | 21.0% | 62.3% |

来源：[原作者 merged.json 数据索引](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/README.md)。本次只开发 MATH-500 的 GPU 评测入口；GSM8K/IFEval 尚未提供等效执行器，不能声称其已复现。

## 8. 尚未满足的 R1 最终验收项

- [x] 真实原始日志、p50c manifest 和 SHA 常量人工核对。
- [x] CPU BF16 变更重放逻辑和输入校验。
- [x] 非破坏性写入、异常失败测试、合成数据回归。
- [x] MATH-500 独立评测代码、逐题结果 schema、配对统计。
- [x] 从官方发布源下载指定 revision 模型并用于本次重建；**未**对全部未编辑权重逐张量进行独立校验。
- [x] 在真实 Qwen 权重上重建公开 p50c 的两个修改张量，精确 SHA 校验通过。
- [ ] GPU 上分别执行 base / edited MATH-500。
- [ ] 获取真正 500×4 对照报告、配对置信区间及异常样本。
- [ ] 扩展 GSM8K/IFEval 等多任务回归并确认评分器兼容。

结论：**R1 Checkpoint Replay 的核心 SHA 门禁已真实通过，但 MATH-500/其他任务评测门禁尚未通过，R1 整体验收仍未完成**。这与论文结果被作者或第三方复测是不同的事实。
