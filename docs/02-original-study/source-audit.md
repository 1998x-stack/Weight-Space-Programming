# 原始研究与可追溯性审计

## 什么算“一手”？
- **研究源：** [2026-11-02 博客](https://rikiyat.github.io/blog/weight-space-programming/)；作者为 Rikiya Takehi / Jayden Teoh / Alex Zhang / Omar Khattab；MIT CSAIL。
- **执行证据：** [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs)，按 exp1..exp8 分类；每组 README、prompt、output、评测与若干代码。
- **模型配置：** [Qwen3.5-0.8B model card](https://huggingface.co/Qwen/Qwen3.5-0.8B)。
- **复测证据：** [JonathanColetti 发布的独立权重重建和 MATH-500 复测](https://huggingface.co/JonathanColetti/Qwen3.5-0.8B-Astra-WeightEdit)。

截至本次写作，没有把研究博客确认为同行评审会议论文；如后续出现论文版本，需新增论文链接、版本对照及发布日期。

## Exp 8 追溯表
| 主张 | 作者日志证据路径 |
|---|---|
| 原模型 MATH-500 7.9，GSM8K 13.8，IFEval 21.0 | exp8_math/eval/unedited/merged.json |
| 50-pass 结果 48.1，67.1，62.3 | exp8_math/eval/p50c/merged.json |
| cut_share 93.0 → 33.1 | 上述两个 merged.json 的 math500.cut_share |
| 初始 96 道中 85 道撞长度墙 | exp8_math/runs/p50c/requests/fwd_1.done.json |
| 最终编辑层、列数、哈希 | exp8_math/runs/p50c/final_answer.md、edit/manifest.json |
| 50 passes，158 turns，80 Python，20.3 hours | exp8_math/README.md |
| 重建最终 BF16 修改 | exp8_math/code/apply_edit.py；基座 revision 2fc06364715b967f1860aea9cf38778875588b17 |

**更正容易被复制的数字：** 原始博客的 GPQA Diamond 基线 24.5 属于评测 149/198 道时的中间值；完整 merged.json 对应 22.7，不能把两个数混用。

## 缺失的公开部分
作者 README 明确说明：没有发布完整 Agent Harness；若干生成与激活大型中间文件缺失；部分官方判题器的内部依赖未一并发布；GPQA 某些完整生成文本未公开。依照这些证据，区分：
1. **Checkpoint replay**：可从 diff 重建并与哈希核对；
2. **Metric replay**：部分已有独立复测；
3. **Agent autonomous rediscovery**：不能因为已有日志而宣称已完全独立复现。

## 证据管理字段
任何新发现须写：source URL、发布日期、访问日期、commit/revision、证据等级、数值与采样条件、是否实测、潜在冲突、复核者。请不要导入不许可再分发的训练/评测数据，也不要推测未公开的内容。
