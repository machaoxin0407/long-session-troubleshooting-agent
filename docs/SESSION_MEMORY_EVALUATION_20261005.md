# 会话记忆联调与对照评测（2026-10-05）

## 本轮交付

- `session_embeddings.py`：已配置 embedding 服务的有界 HTTP 适配器；批次校验、固定维度、总超时、请求取消、并发上限、无重试；默认词项模式保持原有行为。
- `benchmark_session_memory.py`：合成对话、单层/所有层/逐层对照、回查消融、热缓存计时、SQLite 占用、并发本地检查、线上前置配置检查、外部请求上限及提供方 token 使用量记录。
- 树摘要截止时间改为直接传入构建 deadline，摘要适配器显式选择是否为回答预留时间，避免外部注入回调得到延后的 deadline。
- 单元测试覆盖 HTTP 数据、批次排序、失败与取消、计费请求上限、引用/事实检查和离线禁止调用模型。

## 合成数据和指标

共 7 类场景：设备更正、否定条件、未完成操作、早期编号、长消息中间信息、旧客服回答与用户陈述冲突、同义表达。每次运行保存完整合成样例及 SHA-256、代码哈希和独立 SQLite 数据库。不使用生产用户数据、手册或视频评测集。

| 指标 | 含义 | 不能据此推断 |
|---|---|---|
| memory-only coverage | 近期原文 + 摘要/树窗口中是否包含完整目标原话，不含回查 | 模型一定理解了这句话 |
| source-text coverage | 加入独立原文回查后是否包含完整目标原话 | 回答正确率 |
| answer checks | 线上模型回答的预设关键事实、禁止陈述、用户角色、可见引用和源轮次校验 | 通用语义正确性、真实客服质量 |
| first-context ms | 首次获得上下文的时间，可能包含构建或缓存读取 | 两种树模式的独立冷构建对比 |
| warm median ms | 后续缓存读取/检索的计时，中位数 | 真实模型/端到端回答延迟 |
| remote requests | 在 HTTP 请求发出前计数的实际尝试数，包含失败请求 | 仅成功调用的数量 |
| provider usage | 提供方返回的数值 token 字段，不含费用估算 | 固定货币成本；字段缺失表示未知 |

两个树模式共用同一棵树，`traversal` 的首次上下文通常已使用 `collapsed` 构建的缓存。原文回查在三种方案中采用相同策略。离线模式仅使用本地摘录，无模型回答，“answer_check_passed” 为 null，不应解释为 0% 回答正确率。

## 已完成的本地对照

以下运行均为词项向量 + 本地摘要摘录，外部请求数为 0，不能作为真实模型效果结论。

| 对话长度 | 单层、不含回查 | 所有层、不含回查 | 逐层、不含回查 | 加入回查后三种方案 |
|---|---:|---:|---:|---:|
| 80 轮 | 2/7 | 6/7 | 5/7 | 均 7/7 |
| 160 轮 | 2/7 | 7/7 | 6/7 | 均 7/7 |

80 轮：[报告](../reports/session_memory_eval_20261005/run_580df537728c483e9cd1e41d0fa1ad4c/evaluation.json)。160 轮：[报告](../reports/session_memory_eval_20261005/run_bdffe57a5c1041af8d4a9c33fd125d86/evaluation.json)。这些早期报告的未尝试回答计数为 0，需结合 `answer_quality_measured=false` 理解；当前工具已将未尝试值改为 null。

80 轮运行中热缓存中位时间约为单层 10.9 ms、所有层 40.4 ms、逐层 40.8 ms；160 轮约为 13.3/76.5/74.7 ms。当前树有更多结构和向量需要反序列化，读取成本高于单层。不能声称整体性能更快。

并发本地检查使用 12 个合成会话、每会话 40 轮、4 个 worker：上下文中位 79.3 ms，P95 344.6 ms，所有历史预算检查通过，数据库约 434 KB。[并发报告](../reports/session_memory_eval_20261005/run_580df537728c483e9cd1e41d0fa1ad4c/concurrency.json)。样本量较小，测量期间存在其他 worker 追加记录的竞争，不能替代生产负载验收。

这些结果支持保留原文回查。树检索能补充更多旧原话，但逐层筛选与窗口裁剪仍会遗漏内容；需要真实模型评测才能判断最终问答收益。

## 当前机器真实联调配置

配置应在当前机器的环境变量或仓库 `.env` 中完成；不要把凭据写进源码、报告或聊天。

回答模型至少需要以下变量（使用已有其他路由也可以）：

```dotenv
SILICONFLOW_BASE_URL=<已授权测试服务的 OpenAI-compatible v1 地址>
SILICONFLOW_API_KEY=<凭据>
SILICONFLOW_MODEL=<该服务实际支持的模型>
```

语义向量联调还需要：

```dotenv
EMBEDDING_BASE_URL=<已授权测试服务的 v1 地址>
EMBEDDING_API_KEY=<凭据>
EMBEDDING_MODEL=<实际 embedding 模型>
```

API 使用语义向量时另设 `CHAT_SESSION_TREE_EMBEDDING=semantic`。评测 CLI 用 `--embedding semantic` 独立选择，不修改服务部署配置。`--online` 才会发送外部请求；离线使用 semantic 会被拒绝，防止本地结果冒充语义向量结果。

本轮检查当前机器未发现 `.env`，回答路由与 embedding 凭据均未配置。已保存 [联调前置检查报告](../reports/session_memory_eval_20261005/run_4073852244614f77990121c0f7bcdde4/evaluation.json)，外部调用为 0。用户选择在当前机器配置后联调；真实模型摘要、向量和问答结果尚未取得。

## 可重复执行

离线对照及本地并发检查：

```powershell
python benchmark_session_memory.py --turns 80 --concurrency
python benchmark_session_memory.py --turns 160
```

配置完成后的首轮小规模真实联调：

```powershell
python benchmark_session_memory.py --online --embedding semantic --turns 40 --case-limit 1 --max-remote-calls 12
```

最多 12 次外部 HTTP 尝试，默认整轮时间预算 180 秒；摘要调用使用已有默认限制，QA 单次最多 10 秒/400 输出 token，embedding 一次批次调用总时间最多 4 秒。停止于调用预算、超时或配置缺失不会自动加额度或重试。模型路径只测历史事实问答，不触发原有 ReAct、手册或视频工具。

默认在 `reports/session_memory_eval_20261005/run_<随机ID>/` 新建目录。显式 `--output` 必须是尚不存在的目录，避免覆盖历史证据。配置缺失或在线问答请求不完整返回非零退出码；回答确实返回但事实/引用检查失败属于有效的失败评测结果，需查看 `answer_check`，不能只看进程退出码。

## 后续事项

1. 当前机器配置凭据后运行首轮真实探测，再决定是否扩大合成案例与调用上限。
2. 逐条人工复核模型摘要及回答，尤其是更正、否定和建议/已执行的区分；自动字符检查不能验证完整语义。
3. 真实 embedding 服务的吞吐和截止时间需要实测，失败回退可能消耗预算后才发生，不宜直接扩大建树预算。
4. 全模型请求的统一上下文预算、树局部增量更新、生产并发负载及更长保留期限尚未作为本轮交付完成。

## 本地回归证据

最终全量 pytest、JUnit、源码前后哈希与独立测试数据库位于 [本轮回归目录](../reports/session_memory_eval_20261005/run_regression_ec37432fdbd548dfb174c4db833e7058/)，最终结果以 `verification.json` 为准。该运行强制本地词项向量和禁用未模拟的摘要调用；语义 HTTP 路径、用量统计、总超时和取消由 MockTransport 单元测试覆盖。
