# Unit tests

R1 CPU 测试包括 test_r1_replay.py、test_r1_pairing.py、test_r1_math500.py；R2-A 合成测试为 test_r2_core.py。执行：

    python -m pip install -e '.[dev]'
    python -m pytest -q tests/unit

R2-A PR #3 的 GitHub Actions 实际运行 24 passed，且 compileall 通过（[CI 证据](https://github.com/1998x-stack/Weight-Space-Programming/actions/runs/42189205561)）。此前 R1 单独为 11 passed。另有真实官方权重 SHA 重放 [GitHub Actions run #42187045405](https://github.com/1998x-stack/Weight-Space-Programming/actions/runs/42187045405)。**测试不等于完整数学 Benchmark 或证明模型能力提升。** 下一阶段的其他单元/集成用例仍待实现。
