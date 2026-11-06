# 文档导航与证据原则

本系列立足 [Takehi 等（2026）研究博客](https://rikiyat.github.io/blog/weight-space-programming/) 与 [作者公开日志](https://github.com/RikiyaT/lm_understand-logs)，配合历史正式论文、Qwen 模型卡和用户提供的知乎讨论展开独立分析。**此研究首发载体是博客，不应称其为已审稿论文。**

## 建议阅读顺序
1. [核心概念](01-foundations/concepts.md) → [数学与张量维度](01-foundations/mathematics.md)。
2. [原始证据追溯](02-original-study/source-audit.md) → [八组实验](02-original-study/eight-experiments.md) → [Exp 8 Qwen](02-original-study/exp08-qwen.md) → [失败案例](02-original-study/negative-results.md)。
3. [ROME/MEMIT](03-literature/model-editing.md) → [表示工程、SAE 与几何](03-literature/representations.md) → [横向比较](03-literature/method-comparison.md)。
4. [知乎观点核查](04-critical-reading/zhihu-claims.md) → [有效性威胁](04-critical-reading/validity-threats.md)。
5. [分阶段复现](05-reproduction/roadmap.md) → [协议](05-reproduction/protocol.md) → [实验矩阵](05-reproduction/experiment-matrix.md) → [环境](05-reproduction/environment.md)。
6. [系统设计](06-architecture/system.md) → [模块类别](06-architecture/category-taxonomy.md) → [数据契约](06-architecture/data-contracts.md) → [治理](06-architecture/governance.md)。
7. [里程碑与验收](07-roadmap/milestones.md) → [来源清单](references/sources.md)。

## 明确区分的五种陈述
| 标签 | 意义 | 可写法 |
|---|---|---|
| PRIMARY | 原作者博客/原始日志确实如此记录 | “作者报告……” |
| REPLICATION | 有独立公开复测 | “复测者报告……” |
| PAPER | 有正式原论文/会议论文 | “ROME 等提出……” |
| INTERPRETATION | 知乎评论或本仓库推论 | “一种合理解释……” |
| PROPOSED | 尚未执行的实现与实验 | “建议验证……（未实现）” |

## 不可跨越的边界
没有将作者未公开的 Harness 代码包装成完整开源；不把“达到 token 上限”全部等同“数学不懂”；不把修复循环写成注入知识；不把低修改比例理解成低扰动；不将未运行的实验写成成功。阅读者应当能够追溯每个关键数字到源文件。

## 项目状态
R1 更新：公开 Qwen p50c 补丁已在 GitHub Actions 上对官方权重实际校验成功（[运行证据](https://github.com/1998x-stack/Weight-Space-Programming/actions/runs/42187045405)）；R1 MATH-500 GPU benchmark 尚未执行。R2-A 的 CPU 机制候选工具已实现并通过合成测试，但真实 Qwen activation probing、因果实验与 R2 研究门禁均未完成。详细请见 [R2-A 离线工具说明](05-reproduction/r2-offline-implementation.md)。其余模块仍保留占位。
