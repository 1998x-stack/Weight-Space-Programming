# Contributing

本仓库已进入 R1 工具链实现阶段，接受来源审计、受约束权重重建、数学基准复现、数据契约、模块化设计及完整验证测试。**不要添加虚构模型输出、benchmark、训练结果或未授权权重。**

## Review checklist
- 一手来源是否有 URL、实验编号、精确路径、指标定义？
- 是否把实验数据与个人解释分开？如引用知乎观点，请明确其为二手来源。
- 是否以 model revision / dtype / prompt / sampling / grader / sample count 固定实验条件？
- 是否把训练集/validation/holdout 分离，并记录负面与 No-Edit 结果？
- 新模块是否与 docs/06-architecture/category-taxonomy.md 一一对应？
- 本 PR 若仍是占位模块，是否没有可运行的误导性接口或假称测试通过？
- 若确实进入实现阶段，请在 PR 中给出真实执行日志、依赖版本、单元测试与回归结果。

## Proposed future PR sequence
M01: 来源和差分哈希审计 → M02: checkpoint replay → M03: 机制重发现 → M04: budgeted Agent → M05: 跨模型与合成 → M06: 公开技术报告。各阶段不应混杂证据与结果口径。

## Licensing note
当前仓库尚未决定对本项目代码/文档应用何种许可；不自动继承原作者博客、公开权重或 Qwen 模型许可。未来新增外部素材必须核对其授权，不直接复制非开放资源。
