# 未来数据契约 v0.1（设计，未实现）

所有实体应有 schema_version、id、created_at、run_id、source_provenance。不要把仓库中的 YAML 示例误认为已验证的运行数据。

## ModelSpec
model_id, provider, exact_revision, architecture, tensor_inventory[{name,shape,dtype,sha256}], tokenizer_hash, template_hash, license, permitted_ops。禁止模糊“latest” revision。

## DatasetSpec
dataset_id, version, split(train/diagnostic/validation/holdout), sample_hashes, label_policy, permitted_use, license。holdout 不可用于补丁选择。

## ProbeRequest / ProbeRecord
request: model_ref, input_ids_hash, probe_type(generation/teacher_forcing/activations), requested_layers, max_new_tokens, decoding, seed, budget_units。  
record: request_id, runtime_id, sample_ids, tensor_metadata, output_refs, duration, error, finish_reason, actual_budget。大体积激活只登记 artifact SHA/URI，不直接塞入 Agent 上下文。

## DirectionEvidence
contrast_pair_ids, layer, token_span, direction_norm, direction_artifact_sha, aggregation_rule, control_results, stability_stats。无随机对照不得标记 causal_supported。

## PatchProposal / PatchManifest
proposal_id, base_model_sha, touched_tensors, allowed_indices, operator, alpha, delta_dtype, output_sha256, norm_ratio, sparse_count, experiment_refs, status(draft/validated/rejected)。基座和补丁之间需要 byte-level 检查。

## EvaluationReport
benchmark_spec, grader_version, prompt_hash, sampling, per_problem_predictions_ref, cut_share, loop_rate, accuracy, paired_ci, right_to_wrong_count, regression_tasks, limits, cost, pass_fail。数值为空必须用 null，而非填写虚假 0。

## ResearchTrace / AuditEvent
agent_request, goal, hypothesis, proposed_tool, policy_decision, previous_state, next_state, timestamps, error_code, evidence_links。私有密钥与敏感输出必须分离存储。

## 兼容原则
未来 schema 变更采用 version + migration，旧报告只读；任何计数单位附单位（problems / generations / passes / turns / tokens / gpu_seconds）。所有 JSON Schema 文件尚未实现；此文档即当前契约真源。
