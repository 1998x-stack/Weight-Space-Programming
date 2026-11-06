# Exp 8 专题：Qwen3.5-0.8B 推理循环的定向权重修复

## 原始问题与计算预算
目标：在不能训练、不能反向传播的条件下提升 Qwen3.5-0.8B 数学表现。原始练习池：GSM8K train 7,469，MATH train 7,275，AIME 714（答案）；不能接触最终评测数据。GPU 前向工具允许 generation 或 teacher-forced observation，每一次算一 pass；有 10/25/50 pass 三组。50-pass agent 实际经历 158 turns / 80 Python runs / 20.3h。

初始 96 条练习生成里 85 条达到 32,768 token 上限。作者最终选择 15、19 层 down_proj 中各 64 列，α=6 的方向编辑。不得将“鉴别重复行为的方向”误写为“发现一个独立数学神经元”。

## 三个运行是三种不同编辑，不是同一补丁的迭代版本
| 组 | 编辑方法 | MATH-500 | GSM8K | IFEval | cut_share |
|---|---|---:|---:|---:|---:|
| Unedited | 无编辑 | 7.9 | 13.8 | 21.0 | 93.0 |
| 10-pass p10c | 8–22 层 30 个输出矩阵的方向编辑 | 12.3 | 25.7 | 49.0 | 87.2 |
| 25-pass p25c | 关闭 15 层 head 7 的部分输出 | 14.1 | 25.1 | 39.9 | 86.4 |
| 50-pass p50c | 两层 MLP 的 Top-64 列方向编辑 | 48.1 | 67.1 | 62.3 | 33.1 |

数值来源：[exp8_math/README.md](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/README.md) 中的 merged.json 映射。比例单位均为百分比；不是独立多次实验均值。

## 数学更新
设 W_l ∈ ℝ^(1024×3584)，u_l ∈ ℝ^1024 且范数为 1，m_l 是选中 64 个中间神经元的布尔掩码：

W'_l = W_l − 6 · u_l · [m_l ⊙ (u_lᵀ W_l)] ； l∈{15,19}。

被选中列的方向投影由 p 变为 −5p。两层触及 131,072 个 BF16 参数位点，舍入后实际 130,188 个权重值改变。**低 L0 比例不是低范数修改，也不是无副作用。**

## 关键统计判读
作者报告独立练习题 1,152 条中 579 条由错转对，5 条由对转错，579 条中 576 条原来受最大 token 长度截断影响。结论应限定为“在给定 Thinking On / decoding 下主要缓解循环导致的无法完成”，而非“凭空学会更多数学知识”。

复测者 [Jonathan Coletti](https://huggingface.co/JonathanColetti/Qwen3.5-0.8B-Astra-WeightEdit) 给出 MATH-500 Thinking On 原始 7.4、编辑 50.2；Thinking Off 原始 45.8、编辑 43.6。其卡片明确说明只独立复现 MATH-500，未独立复现 GSM8K/IFEval，不能误报为全面复现。

## 精确评测配置
基座 revision：2fc06364715b967f1860aea9cf38778875588b17；vLLM BF16；Thinking On；temperature 0.6、top_p 0.95、top_k 20、presence_penalty 1.5；最多 32,768 new tokens；Math 使用最终 boxed 答案和 math-verify。MATH-500 四次采样/题，GSM8K 两次；请按原日志锁定提示模板、数据版本与评分代码。精确协议见 [protocol](../05-reproduction/protocol.md)。

**关键批判：** 官方 Qwen 文本 Thinking 推荐温度 1.0，而此研究采用 0.6。相同设置控制了“编辑前后”变量，但不等于证明补丁优于合理解码调优。下一步必须跑 factorial ablation。

## 复现实验的否决条件
如无法校验基座 revision / edited tensor sha256 / 聊天模板 / 数据隔离 / 题目粒度，就不能发布“成功复现”；任何人工注释或重新执行的中间表征只能标记为独立推断，不能伪称作者原始 artifact。
