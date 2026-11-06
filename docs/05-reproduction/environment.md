# 环境、资源预算与许可证

## 阶段预算
| Stage | 必要资源 | 限制 |
|---|---|---|
| R0 | 浏览器/文档、git | 无 GPU 需求 |
| R1 patch reconstruction | CPU RAM、受许可模型文件、hash 实现 | 原作者脚本需要支持具体 tensors |
| R1 scoring | GPU 推理服务、vLLM、充足存储 | 32,768-token 生成可能昂贵，不应承诺低配 GPU 可一次运行完整设置 |
| R2 activation probes | 可读中间激活的推理后端、存储 | 显存/内存随 batch × tokens × layers 增长 |
| R3 agent | 预算执行器、独立评测服务 | 权重写入权限严格分离 |
| R4 generalization | 不同模型/评测资源 | 额外许可和成本评审 |

## 机器可行性
8GB 显存环境适合阅读文档、编写结构、小尺寸网络验证及部分受限推理；**不能保证**在 8GB 上复现原作者 vLLM BF16 + 32k 输出 + 多批次评测。更大的 GPU 也必须先审视 max_model_len、KV cache、batch、dtype 和端到端吞吐量，不能给出缺依据的“运行 N 分钟”预测。

## 版本锁
将来需要锁定 Python、torch、transformers、vLLM、safetensors、math-verify、数据集 hash、git SHA、模型 revision、设备类型、CUDA/驱动及模型服务端参数。本 v0.1 未测试兼容版本，故不提供假冒可运行的 requirements.lock。

## 许可与数据
不要在本仓库镜像 Hugging Face 完整权重、未许可数据集与论文受限素材。重放应使用官方分发源并遵循许可；独立复测发布的权重有自身许可声明，不能默认可商用或与原仓库同许可。GPQA 部分原始生成受源数据限制。

## 成本披露
实际执行后填写：云 GPU 型号、计费时长、前向调用次数、输入/输出 tokens、存储、失败重试、CI 开销。原作者“50 passes”不是“50 单条推理”。
