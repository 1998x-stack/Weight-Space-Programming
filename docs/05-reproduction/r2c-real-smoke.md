# R2-C: 官方 Qwen 真权重诊断 Smoke（不等于机制复现）

**交付范围：** 用真实官方 Qwen3.5-0.8B 固定版本执行前向探针；在小规模诊断文本上调用 R2-A，进行 alpha=0、观察方向、随机方向的固定输入干预比较，并生成只含哈希与聚合数值的审计报告。

## 证据范围和限制

- 模型固定版本：2fc06364715b967f1860aea9cf38778875588b17。预检查第 15/19 层 down_proj 的原始 BF16 SHA256，与公开 p50c manifest 相同。该证明**仅覆盖两个目标张量**，不是完整 checkpoint 的全局证明。
- 输入：experiments/exp08_qwen_repair/diagnostic_smoke.json 共四组**手工编写、示意性质**的数学推理对照。重复推理文本是人工构造的重复句子，**不是 Qwen 自己生成后观察到的循环**。
- 前向采集：对选中层 decoder input residual 和 down_proj input（门控后激活）注册 pre-hook；取最后 k 个输入 token 平均。不同输入的 token 绝对位置可能不同，是尚未排除的混杂。
- 消融：对**最后一个 token 的 down_proj 输出**做投影删除，只比较相同输入下的下个 token logits。并非模型权重修改、完整 generation，也非 MATH-500 评测。
- 负对照：一个确定性独立随机方向，以及一半匹配对的 productive/repetitive 标记互换方向。仅为描述性 sanity check，不是正式 permutation p 值。
- 保存：最终只上传 r2c-audit.json。中间 diagnostic NPZ 包含模型权重切片的 FP32 副本，仅在本地临时目录存在，不允许上传；没有公开诊断原文或模型权重。
- GitHub Action 35 分钟上限，使用公共 CI runner CPU。GPU 或完整数学 Benchmark 未运行。

## 本地执行

先根据许可和资源，获取官方指定 revision 至独立目录。安装 CPU/GPU 适配的 PyTorch 及仓库额外依赖（例如 python -m pip install -e '.[dev,r2-probe]'），然后：

    python scripts/r2c_real_smoke.py \
      --model /models/qwen35-original \
      --pairs experiments/exp08_qwen_repair/diagnostic_smoke.json \
      --layer 15 --last-k 4 --max-input-tokens 128 \
      --device cpu --confirm-compute \
      --out /runs/r2c/r2c-audit.json

也可用已授权的 CUDA 环境设 --device cuda。命令拒绝覆盖已有结果；不会写改 checkpoint。CI 在 .github/workflows/r2c-authentic-smoke.yml 下载并执行相同固定版本。

## 成功与失败定义

成功的 B1 工程 smoke 需要：下载的官方目标张量前哈希匹配；实载模型前向 Hook 各捕获一次；输出张量形状符合 1024 × 3584； R2-A 选出 64 个候选列；三个固定输入对照得到有限数值；alpha=0 的 logits 差分为零；报告未满足的科学门禁；GitHub Actions 成功且 JSON 审计可下载。

失败时保留 workflow 日志，不填造新的 SHA、实验指标和“复现成功”的记录。如果 GitHub runner 内存或依赖不兼容导致失败，应按日志具体错误修复或说明阻塞。

**R2 科学验收尚需：** 真实模型自行生成的 productive/repetitive 配对、至少四个独立诊断组、长度/位置严格匹配、独立随机/打乱干预、层 15/19 因果对照、完整生成和未触碰 holdout、跨任务回归以及 R1 的 MATH-500 GPU 对照。四条示意文本的任何数值都不构成模型“循环故障得到修复”的证据。

## 运行记录

工程代码与工作流位于 R2-C PR。仅在 Actions 真正完成之后填写 run ID、准确测试数量和具体结果。
