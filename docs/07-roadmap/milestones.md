# 分阶段交付与文件级验收

当前完成：仅 v0.1 文档与占位文件结构。以下实质性功能 **均未实现或运行**。

## M00 — Evidence-First Documentation（本次）
**交付：** README、理论/实验/负面结果/文献/知乎核查、复现协议、架构、模块分类、配置与占位目录。  
**门禁：** 链接有效、Exp8 关键公式维度正确、数字有原始出处、目录文件与 README 一致、没有假装测试成功。状态：**documented**，不代表研究结果复现。

## M01 — R0: Artifact & Source Audit
文件：src/wsp/artifacts/manifest.py, src/wsp/core/registry.py, schemas/, configs/；测试：tests/unit/。  
任务：基座 revision 锁、原作者差分 sha 验证、评测数据和 license registry。  
门禁：hash mismatch 必须失败关闭；给出真实本地审计日志。

## M02 — R1: Faithful Replay

**进展（2026-11-06）：** 已实现 BF16 稀疏 patch 的独立重建器、p50c 固定哈希校验、MATH-500 GPU 入口、按题目配对统计及 CPU 单元测试（合成数据）。真实 Qwen 权重未在此执行环境下载/重建，完整 benchmark 未运行；**R1 尚未达到最终验收门禁**。
文件：src/wsp/adapters/qwen.py, src/wsp/evaluate/{benchmark,paired_statistics}.py, scripts/；测试：tests/integration/、tests/regression/。  
任务：原始/修改后 checkpoint 重建、采样复现、逐题评测、统计 CI、解码因子表。  
门禁：报告 config hash、per-problem paired changes、环境差异；不得仅写“接近作者”就宣布全量重现。

## M03 — R2: Mechanistic Rediscovery
文件：src/wsp/read/{activation_probe,contrastive_direction,layer_attribution,neuron_score}.py, src/wsp/repair/{sparse_edit,validator}.py。  
任务：4 组以上对照、层/MLP 归因、Top-K、α 扫描、随机方向对照、BF16 距离比较。  
门禁：候选选择不泄漏作者最终掩码和 holdout；控制组审计齐全。

## M04 — R3: Agent Orchestration
文件：src/wsp/agents/{planner,runner}.py, src/wsp/core/budget.py, src/wsp/artifacts/audit.py；测试：tests/security/。  
任务：预算中止、No-Edit、工具访问控制、故障恢复和候选决策轨迹。  
门禁：单次运行有完整可重放计划；越权/预算用尽被确定拒绝。

## M05 — R4: Cross-Model and Synthesis
文件：src/wsp/adapters/{game,ddpm}.py, src/wsp/synthesize/{circuit_ir,compiler}.py，experiments/exp01..exp08。  
任务：小网络静态读取、规则编译、Gemma 负例、复杂场景失效、不同模型对比。  
门禁：跨模型、多个种子、负例和成本对照均完整报告。

## M06 — Public Research Release
论文式技术报告、实验卡、精确复现命令、数据许可说明、结果机器可读表、图表和已知失败。  
门禁：独立第三方复查 + 不可重复结果诚实披露。

## PR 验收模板
目标 / 对应论文与实验 / 对照与假设 / 受影响文件 / 数据和 license / 运行命令和真实输出 / 消融 / 退化 / 成本 / 未解决风险 / 回滚方法。
