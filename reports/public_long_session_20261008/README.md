# 长会话故障排查：公开评测证据

本目录仅包含冻结合成数据与自动评测证据，未包含真实用户对话、凭据、服务器原始记录、数据库或向量缓存。冻结数据、回答与上下文按字节复制；报告和说明按公开范围整理，当前文件哈希列于 manifest.json。

- [最终报告](FINAL_AUTOMATIC_REPORT.md)。
- dataset.json：70个独立合成会话及标准答案；freeze.json：冻结参数、模型与源码哈希。
- development/：70条开发回答；test/：56会话×5策略的280条主回答。
- supplement/：70条重复与28条消融。每个任务包含 result.json 和实际可见的 context.json。
- comparison.json 与 cost_and_path_audit.json：原始规则结果、配对区间及成本。
- regression.json：已有全量回归的摘要，源码与测试未改动；发布时仅更新文档并新增离线导出工具。

45/56为含降级回退的整体方案自动规则通过数，正常路径2/2、降级路径43/54；窗口15/56。回答未进行独立人工语义验收，不等于生产准确率。树结构未显示明确额外增益，不证明树无效。

在仓库根目录离线核验全部文件哈希并按冻结评分器复算448条回答（不发起模型请求）：

```bash
python deploy/export_public_session_evidence.py --verify reports/public_long_session_20261008
```

如需新实验，按 docs/LONG_SESSION_AGENT_EXPERIMENT_20261008.md 创建独立输出目录、配置自己的模型凭据和请求额度。新模型结果允许不同；禁止覆盖本冻结证据。报告中提及的原始 ledger、缓存和人工审阅页面不在公开包内，1273次请求总数属于归档统计，公开包不提供完整HTTP账本的独立复核。
