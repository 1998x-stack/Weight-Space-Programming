# 表示工程、稀疏特征与表示几何

## Representation Engineering
Zou et al., [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405)。从群体表征读取和操控行为相关的高维信号，而不是从单个神经元推断概念。对照表示方向可作为轻量初始候选，但不能直接认定具有因果性。

## Sparse Autoencoder（SAE）
Bricken et al., [Towards Monosemanticity](https://www.transformer-circuits.pub/2023/monosemantic-features/index.html)。某些语义特征可能是多个神经元的组合；把高激活神经元当成完整语义单位存在局限。未来可比较对照均值差分与字典学习特征，但要控制训练 SAE 的额外预算与依赖。

## Linear Representation Hypothesis
Park, Choe, Veitch, [ICML 2024](https://proceedings.mlr.press/v235/park24c.html)。讨论线性概念表示和内积几何，指出投影/余弦的意义依赖所采用度量。**欧氏方向编辑有效一次，不证明欧氏度量总是最优。**

## 设计原则
- 分开测量 “Probe predicts behavior” 与 “Intervention changes behavior”。
- 控制输入主题、token 位置、长度、词频、Prompt 提示造成的混杂。
- 对随机方向和方向置乱做范数匹配。
- 探索 rank-k 子空间，不必在所有任务强制单方向。
- 所有方法的计算资源、数据集划分和副作用必须公平报告。
