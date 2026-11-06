# R1 + R2-A scripts

R1 已实现的命令入口：

- r1.py replay：离线重放官方公开的 p50c BF16 稀疏补丁，严格校验两层模型张量前后 SHA256。
- r1.py compare：根据按题配对的正确性 JSON 计算 Accuracy、Bootstrap CI 与翻转统计。
- r1_prepare_math500.py：要求 MATH-500 数据集指定 commit，生成 JSONL/数据 hash。
- r1_math500.py：真实 vLLM/MATH-500 评测入口，必须显式添加 --confirm-gpu-cost。**尚未在 GPU 实测。**

详见 [R1 运行记录与命令](../docs/05-reproduction/r1-implementation.md)。真实 p50c 权重重放已在 [GitHub Actions](https://github.com/1998x-stack/Weight-Space-Programming/actions/runs/42187045405) 成功，CPU 合成测试 11 项通过。R2-A 新增 r2_offline.py：只接受诊断 NPZ/严格 provenance manifest，输出内存候选 JSON，不写模型权重或使用 holdout。请参阅 [R2-A 运行说明](../docs/05-reproduction/r2-offline-implementation.md)。真实模型 teacher-forcing、因果实验、R1 GPU benchmark 仍待完成。
