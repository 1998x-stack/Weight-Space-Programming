# Unit tests

R1 CPU 测试包括 test_r1_replay.py、test_r1_pairing.py、test_r1_math500.py；R2-A 合成测试为 test_r2_core.py。执行：

    python -m pip install -e '.[dev]'
    python -m pytest -q tests/unit

R2-A PR #3 的 GitHub Actions 实际运行 24 passed，且 compileall 通过（[CI 证据](https://github.com/1998x-stack/Weight-Space-Programming/actions/runs/42189205561)）。此前 R1 单独为 11 passed。另有真实官方权重 SHA 重放 [GitHub Actions run #42187045405](https://github.com/1998x-stack/Weight-Space-Programming/actions/runs/42187045405)。**测试不等于完整数学 Benchmark 或证明模型能力提升。** 下一阶段的其他单元/集成用例仍待实现。

R2-B: `test_r2b_contract.py` 是不需要 Torch 的输入/门禁测试；`test_r2b_torch.py` 使用 **合成 tiny PyTorch 模型**验证 hook 注册/解绑、shape、不可修改基座和 logit KL，默认无 torch 环境中会 skip。CI 的 synthetic-torch-hook-tests job 单独安装 CPU torch。实际 Qwen 模型前向实验仍待单独执行。

R2-C 新增 test_r2c_controls.py（稳定性、标签打乱方向、异常输入）。真实 Qwen workflow 和示意诊断库见 docs/05-reproduction/r2c-real-smoke.md；科学门禁另行验收。

R2-C 首次真实官方模型工程 Smoke [Actions #42200064239](https://github.com/1998x-stack/Weight-Space-Programming/actions/runs/42200064239) 已通过；合成代码检查 [#42200064091](https://github.com/1998x-stack/Weight-Space-Programming/actions/runs/42200064091) 为 41 passed、1 skipped + PyTorch 4 passed。具体证据详见 [R2-C 审计](../../docs/05-reproduction/r2c-run-20261106.md)。

R2-D1 添加 test_r2d_classifier.py（纯 CPU Token 截断/重复分类和诊断输入验证）、test_r2d_generation_torch.py（合成 Torch 推理接口）。不构成真实模型跑分。[R2-D1 协议](../../docs/05-reproduction/r2d-generation-pilot.md)。

R2-D1 新增 test_r2d_budget.py，检查两次真实生成轨迹的 prompt/model/decoding/分类/Token 前缀一致性和不允许悄然覆盖。真实 96/256 运行详情参阅 [R2-D1 审计](../../docs/05-reproduction/r2d-run-20261106.md)。
