# WSP 基础概念与边界

## 概念
Weight-Space Programming：由通用 LLM 围绕自然语言目标读/写一个目标神经网络的权重，并使用其自身的分析、计算工具与（可选）环境反馈提出操作。研究里明确不允许对目标模型进行梯度反向传播和训练；不意味着系统里不存在任何计算工具。

- **Static weight inspection**：只读权重张量及网络结构，不能凭空保证恢复出真实语义。
- **Forward probe**：输入样本、记录输出、Hidden States、注意力和 MLP 激活；这是一种信息更丰富的机制诊断。
- **Teacher forcing**：将一段已知上下文送入模型，分析其 token-level log probability 与内部状态，无须自由生成同样文本。
- **Residual stream**：注意力与 MLP 等模块向隐藏状态写入更新的主干。
- **Activation/representation direction**：由对照样本构造的向量；语义稳定性与因果关系需干预证明。
- **Rank-one edit**：形如 u v^T 的矩阵增量；“更新秩一”不等于“只修改一个参数”。
- **Sparse column edit**：仅选择部分神经元输出列；可触及的元素很多，且选中列的变化幅度可以很大。
- **Ablation**：针对性改变模块、层、方向或强度检验结果，控制随机与范数匹配基线。
- **Checkpoint provenance**：准确记录基座 revision、张量哈希、dtype、变更索引和生成器版本。

## 三个研究分支
READ（行为解释）、REPAIR（局部故障修复）、SYNTHESIZE（规则编译到权重）；EVALUATE（证据门禁）与 AGENT（预算内实验设计）跨越三个分支。

## 两种本质不同的低分
- 能力存在但被异常循环、格式、终止机制阻碍：有机会用局部修复释放已有能力。
- 推理能力本身不存在：少量矩阵编辑不一定有效；需与训练或其他架构路线对照。

本项目拒绝两类夸大：一是说 WSP 普遍取代梯度下降；二是断言其理论上永远不可能写入任何复杂能力。现有证据只支持具体实验边界。

来源：[原始博客](https://rikiyat.github.io/blog/weight-space-programming/)、[实验日志](https://github.com/RikiyaT/lm_understand-logs)。
