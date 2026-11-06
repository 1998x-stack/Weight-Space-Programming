# 实验矩阵、消融与统计设计

## E1 — 两因素（最小）
| Model | Thinking | 目的 |
|---|---|---|
| Base | On | 作者基线 |
| Edited | On | 补丁贡献 |
| Base | Off | 原模型非思考能力 |
| Edited | Off | 对原有能力的潜在副作用 |

独立复测者报告 Thinking On: 7.4→50.2，Thinking Off: 45.8→43.6（MATH-500）；这与作者的 7.9→48.1 采用不同测试跑次。切勿把独立复测的其他任务写成已验证。[公开复测](https://huggingface.co/JonathanColetti/Qwen3.5-0.8B-Astra-WeightEdit)。

## E2 — 解码因子消融
至少比较 temperature={0.6,1.0}，Thinking On/Off，Base/Edited，保留其他参数与数据版本。主指标是 Accuracy、Cut/Loop rate 和 Tokens-to-answer。后续再加入 max_new_tokens、presence_penalty 的网格；避免一上来组合爆炸。

## E3 — 机制消融
- 层 15 only、层 19 only、双层；
- top-k ∈ {16,32,64,128}（**仅提案**，需要重新选择/避免事后调整）；
- α ∈ {0,1,2,4,6}；
- random direction，等范数、同稀疏度；
- shuffle productive/repetitive label；
- 无编辑（绝对基线）。

## E4 — 泛化与负例
Gemma 270M（数学不足、非循环主导）、其他文本故障、结构不同的小网络，必须记录 No Edit。报告每个任务的成功数/失败数/未完成原因。

## 评分的层级
每次生成：extraction/correct/finish_reason/tokens；每道题：平均正确率、paired difference；每个任务：均值、95% CI、回归率；每次 Agent run：总 passes、代码调用数、wall time、GPU 秒、编辑数量。总计时不跨粒度盲目相加。

## 冻结机制
候选选择发生于 validation，评测在 holdout；一个实验矩阵条目结束后写入不可变 manifest，所有失败条目与成功条目同等保留。
