# 同一会话上下文压缩与持久化记忆

> 本文记录第一版规则式记忆实现及其历史验证。当前 API 已升级为模型摘要与 token 估算预算，
> 以 [压缩升级说明](SESSION_COMPACTION_20261001.md) 为准；下文原始实现与测试数字保留作历史记录。

## 范围

仅按 `session_id` 续聊，没有用户画像、跨会话共享或跨会话检索。客户端仅发送当前问题和会话 ID。
无 ID 时新建，切换 ID 时隔离。直接调用 `run_agent` 的离线入口仍不自动装载历史；记忆接在 API 层。
请求成功后保存用户问题与最终交付文本；超时、断开连接和生成失败不追加成功轮次。
没有把过去的图片、视频二进制或工具执行状态存入会话，追问涉及原附件时需要重新上传。

## 2026-10-01 官方资料比较与选择

| 项目 | 可借鉴的能力 | 本项目取舍 |
| --- | --- | --- |
| [Mastra Observational Memory](https://mastra.ai/docs/memory/observational-memory) | 近期消息、带时间与重要程度的观察记录、分层压缩；可显式设置 thread scope | 借鉴分层与来源标识，保留用户条件、更正、型号与错误码。没有引入 TypeScript 框架或 Observer/Reflector 模型调用。 |
| [Deep Agents](https://www.langchain.com/blog/context-management-for-deepagents) | 压缩工作上下文，同时保存原始对话供恢复 | 摘录与原文独立存储，按当前问题在同一会话内回查。 |
| [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence) | 按 thread_id 持久化；原子状态写入 | 用 Python 标准库 SQLite 实现会话隔离、原子写入和重启恢复，避免迁移现有 ReAct 流程。 |
| [Hindsight](https://github.com/vectorize-io/hindsight) | retain/recall/reflect 分离、来源追踪；[9月21日更新](https://hindsight.vectorize.io/blog/2026/09/21/hindsight-cloud-0-10-0) 增强附件来源 | 借鉴原文与观察分离及可追溯回查；本轮不迁移其存储服务，不增加附件记忆。 |
| [Mem0](https://github.com/mem0ai/mem0/blob/main/docs/core-concepts/memory-evaluation.mdx) | 分层提取、多信号与时间检索 | 保留原文、角色和时间，用户更正不被旧建议覆盖；没有引入其跨会话用户记忆。 |
| [Letta conversation compaction](https://docs.letta.com/api/resources/conversations/subresources/messages/methods/compact) | 可配置 sliding window 与摘要大小 | 近期窗口与旧摘录分配独立预算。 |
| [Microsoft LLMLingua](https://www.microsoft.com/en-us/research/project/llmlingua/) | 学习式 token 压缩 | 当前不引入额外模型与推理依赖；特别保留客服场景的否定、条件与来源完整性。 |

[Mastra 官方研究](https://mastra.ai/research/observational-memory) 报告 gpt-5-mini 的 LongMemEval 分数 94.87%，gpt-4o 为 84.23%。
这是项目方报告、特定模型和多会话基准下的成绩，不能断言它截至今天在所有场景最好，也不能当成本项目成绩。
本次选择依据是同一会话、已有 Python 架构与证据忠实性，而非不可比的跨项目总分。
检索时 Hindsight 的[当前结果页](https://benchmarks.hindsight.vectorize.io/) 显示 LongMemEvalS 94.6%；
Mem0 官方文档报告 LongMemEval 94.4%，同时说明是含专有优化的托管平台成绩，不能视为开源 SDK 复现保证。
这些数字的模型与评测条件不同，不按小数差异决定框架选择。

## 实现

`session_memory.py` 的 `SessionMemoryStore` 保存两层持久化数据：

1. 当前窗口与压缩观察：默认最近 6 条消息；老消息压成原文句子摘录，保留角色、轮次编号和 Unix 时间。
2. 原始成功对话归档：以 zlib 无损压缩 JSON 存储。摘要中间遗漏的内容仍可在归档保留范围内找回。

压缩是本地确定性摘录，优先完整的型号、错误码、否定条件、故障和更正句子，不生成未经来源支持的事实。
摘要达到预算后，优先淘汰旧客服回答，再淘汰较低优先级的用户内容。用户与客服来源、时间顺序保留，
不自动把不同条件合并，也不将重复型号视为唯一型号或将历史建议当作事实。
近期单条长消息也会压缩；构建旧观察时重新读取归档原文，避免长期丢掉长消息中间的关键条件。

有压缩历史时，当前问题触发会话内词项匹配（英文/型号词项、中文双字词项），最多选 3 个相关轮次，
并按发生顺序以有界摘录加入上下文。已完整显示的近期轮次不重复回查；被压缩的近期长消息也可回查。
匹配是词法回查，不是语义向量检索；同义改写或纯代词追问可能漏召回。
生成提示明确要求本轮证据核验、用户当前更正优先以及省略不代表不存在。
压缩、回查和原文归档都不调用模型，因此没有额外外部服务调用或依赖变更；不声称与 Mastra 的学习式压缩等效。

SQLite 事务原子更新归档和压缩状态，多进程写入不会覆盖丢轮次。
同一会话并发请求的推理仍可能读取同一份旧上下文，写入按完成顺序；需要因果顺序时客户端应串行发问。
Demo 会在请求期间禁用发问、新建、清空与令牌修改，保存标签页的 ID，刷新可续聊。

## 配置与保留

| 环境变量 | 默认值 | 含义 |
| --- | --- | --- |
| `CHAT_SESSION_DB_PATH` | 仓库 `data/runtime/chat_sessions.sqlite3` | SQLite 路径；支持绝对路径，容器需持久卷 |
| `CHAT_SESSION_HISTORY_LIMIT` | `6` | 近期消息条数，必须为不少于 2 的偶数 |
| `CHAT_SESSION_MAX_SESSIONS` | `500` | 持久化会话容量；按访问 LRU 淘汰 |
| `CHAT_SESSION_TTL_S` | `3600` | 距最后一次成功写入的过期秒数，可按业务延长 |
| `CHAT_SESSION_RECENT_CHARS` | `6000` | 近期消息正文总预算，按消息条数均分 |
| `CHAT_SESSION_SUMMARY_CHARS` | `4000` | 摘录预算，包含来源与角色标记 |
| `CHAT_SESSION_EXCERPT_CHARS` | `800` | 每条旧消息的摘录预算 |
| `CHAT_SESSION_RECALL_CHARS` | `2000` | 原文回查正文预算，至少 384 |
| `CHAT_SESSION_ARCHIVE_TURNS` | `200` | 每个会话原文最多保留的成功轮次 |
| `CHAT_SESSION_ARCHIVE_CHARS` | `2000000` | 每个会话原文字符预算，超限淘汰旧轮次 |

预算以字符计，不是模型 token 保证；近期角色和提示固定文本会额外占少量上下文。
极端超大单轮超过归档字符预算时，保存带省略标记的首尾摘录，不能宣称该轮原文完整。
正常归档范围内保留完整原文；摘要的省略与归档容量淘汰是不同概念。
SQLite 按访问惰性删除过期行，LRU 淘汰和清空会级联删除原文；删除为逻辑删除，不承诺文件立即缩小或法证级擦除。
默认路径包含私有对话，不纳入 Git 或交付包。默认保留 1 小时，因此重启恢复要求使用同一文件且尚未过期。

## API 与诊断

`POST /chat` 与 `POST /v2/chat` 的请求/响应形状不变，复用 `session_id` 即续聊。
新增同鉴权的 `DELETE /v2/chat/sessions/{session_id}`，204 幂等清空该会话。
存储不可读写时返回 503，不把持久化失败伪装成成功。`agent_trace` 增加 `session_memory_compressed` 和 `session_memory_recalled`，
`session_history_turns` 只统计近期真实问答轮数，不把摘要和回查算成新轮次。

## 验证范围

离线用例覆盖独立进程写入后恢复、跨会话隔离、并发进程不丢写入、摘要与字符预算、
长消息中间条件保留、原文无损回查、否定与更正、TTL/LRU、归档容量、清空级联、
API 成功/失败写入边界和不可用存储的 503。旧 HTTP、取消与配置回归同步检查。
不重跑历史外部服务，不把离线验证当作真实模型质量、正式性能或稳定性验收。

本轮全量初跑：3760 passed、1 skipped、83 subtests passed；7 项旧测试分别因 Windows 命名管道权限
和系统 Bash 启动器失败。仅为这些纯本地测试使用 Git Bash 与允许命名管道的环境后，7 项复测全部通过，
没有修改旧运行测试或旧运行模块。新增 Demo 行为测试单独通过。
全量之后补充了近期长消息回查和清空 API 鉴权用例，记忆与 Demo 专项 20 项通过；
最终记忆、Demo、受影响 HTTP/取消/配置回归共 98 项通过。新模块 Ruff 通过，已改旧文件相对 HEAD 没有新增 Ruff 告警，差异检查通过。
已有 Windows 临时目录清理权限警告保留，不作为会话功能故障。
