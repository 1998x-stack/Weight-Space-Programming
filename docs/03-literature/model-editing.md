# ROME、MEMIT 与直接权重修复

## ROME（2022）
Meng et al., [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262)。研究使用 causal tracing 找到 GPT 中与事实关联相关的中间层 MLP 计算，使用 Rank-One Model Editing 修改指定事实；评估特异性、泛化与对其他事实的保留。

## MEMIT（2022 arXiv / 后续 ICLR 2023）
Meng et al., [Mass-Editing Memory in a Transformer](https://arxiv.org/abs/2210.07229)。研究把事实编辑扩展到多层，解决单个事实操作在批量场景的容量限制；提供多条记忆的修改与保持评估。

## 与 WSP 的同异
- 共性：可以直接改变权重；需要定位关键层、计算矩阵增量、做因果与保真度验证。
- 差异：ROME/MEMIT 是研究者设计并参数化的模型编辑算法，主要面向事实关联；WSP 2026 实验尝试由通用 LLM **自主选择问题诊断路径与编辑程序**，Exp 8 主要修复推理循环。
- 不应把“ROME 的秩一事实更新”与“Exp 8 的稀疏输出列方向投影”视为完全相同算法。
- 也不应据此宣称 WSP 是首创“直接权重编程”；贡献偏向通用 Agent 的主动假设和验证。

## 复现比较建议
统一条件：相同可访问基座、任务、验证集、推理预算、编辑强度；报告 success、specificity、side effects、edit size、repair compute。禁止把针对事实修改的方法直接当成循环故障修复基线后再称其不佳；需要任务对齐。
