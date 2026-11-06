# Unit tests

R1 CPU 测试已实现：test_r1_replay.py、test_r1_pairing.py、test_r1_math500.py。执行：

    python -m pip install -e '.[dev]'
    python -m pytest -q tests/unit

本地测试为 11 passed，GitHub Actions 中的 cpu-tests job 通过。另有真实官方权重 SHA 重放 [GitHub Actions run #42187045405](https://github.com/1998x-stack/Weight-Space-Programming/actions/runs/42187045405)。**测试不等于完整数学 Benchmark 或证明模型能力提升。** 下一阶段的其他单元/集成用例仍待实现。
