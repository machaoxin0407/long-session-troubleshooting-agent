# 长会话故障排查 Agent：实验设计与证据

## 当前完成状态

已完成70次开发回答、280次主评测、70次重复和28次消融，共448条真实模型回答；累计1273/2000次外部HTTP尝试。用户已核对70例标准答案，并选择以自动规则与成本结果收尾，实际回答未进行独立人工语义验收。下文的数据核对和人工审阅步骤保留为实验协议参考，其中回答人工复核目前为可选项，不将自动通过数视为人工语义正确数。

公开的冻结数据、逐条回答、可见上下文、规则结果及文件哈希见 [公开证据包](../reports/public_long_session_20261008/README.md)。该包不包含凭据、服务器原始记录、运行数据库或向量缓存；完整私有工作目录保持不变。

本轮只扩展评测工具。生产 API、记忆实现、服务器配置和正式实例不变。当前候选数据目录为 `reports/long_session_agent_20261008_r3/`；前两个目录保留为冻结前迭代记录，不能混合汇总。

## 数据和人工核对

共有七类、70 个不同会话案例：开发集 14 个，测试集 56 个；测试集每类含 2 个 40 轮、3 个 80 轮、3 个 160 轮案例，以及其中 1 个证据不足案例。开发与测试的模板家族、标识符和事实组合分离。它仍是同一程序生成的合成数据，不代表真实客服分布或外部泛化。

1. 用户先打开候选目录的 `DEVELOPMENT_GOLD_REVIEW.md`，核对预期事实、来源轮次及是否有用户确认。长消息和未知案例需结合 `histories/` 完整原文。
2. 直接在聊天中逐项/按范围报告核对结果，或填写 `gold_review.json` 的 reviewer 和三个 verified 字段。工具不会自动勾选。聊天确认须保留原始确认记录再登记。
3. 开发实验后，用户核对 `TEST_GOLD_REVIEW.md`；测试集全部确认后才能冻结。发现数据错误时保存旧候选，生成新版本并重新核对受影响内容，不改已冻结数据。
4. 测试完成后打开 `BLIND_ANSWERS.csv`，在 `human_review.json` 填写事实是否正确、证据是否支持、角色/执行状态是否错误及备注。不看 `review_key.json`；未知答案的 evidence_supports 指“依据缺失证据恰当地回答未知”。错误类型可填 correction_lost、negation_reversed、advice_as_executed、unsupported_citation、other；无错误填空数组。这些计数只来自人工标注，不能从规则失败直接推断。

来源规则检查要求引用存在于对应用户原文和实际可见上下文，命中目标轮次，覆盖完整目标原话或其中至少八个字符的连续片段。它不是蕴含判断；型号正确但否定反转、引文片段不足等仍需人工审核。规则与人工结论分别保留。

## 五策略与公平性

五策略为 window、rolling、structured_recall、collapsed、traversal。前两种不能回查原文；structured_recall 是现有结构化摘要与同会话词项回查方案，不能称为纯窗口基线。树形策略沿用生产建树和检索逻辑，语义模式下底层为本地摘录、根层尝试模型摘要，不能写成每层均由模型生成。

- 共用历史输入上限 5232 个 UTF-8 字节估算单位，源自 6000 减 768 预留。近期/摘要/回查分别沿用 2800/1400/800 配置；窗口可使用全部上限。未用满预算是策略行为的一部分，报告实际大小。
- 整个模型请求继续使用 32768 上限和 1024 安全余量。估算不是提供方 tokenizer。summary/QA 的 usage 原样记录；embedding 适配器未暴露 usage，明确缺失，不能填零或捏造费用。
- 记忆构建只接收 records。当前测试问题、gold 和评分标签不得用于生成摘要、聚类或建索引。查询阶段才使用问题。
- 滚动摘要每四十轮更新，最多四次，只读取上一摘要及新记录；验证引用确实出现在该次可见输入，不能用隐藏原文档案替它补漏。
- 两种树策略共享同一棵树及同案例、同模型的精确文本向量缓存。原文和向量缓存不跨案例。重复实验复用构建缓存，只重跑回答。
- 随机执行顺序由固定种子生成；冷构建成本持久化并归属每个逻辑策略，而物理 HTTP 额度只记实际请求。暖查询不应冒充冷请求性能。
- no_root 消融只替换被接受的模型根摘要及相应向量，保留树结构；若原根摘要已回退，记录 treatment 未生效。no_recall 只移除额外词项原文回查，树节点本身携带的来源原话仍保留。

## 命令与额度

以下命令从主仓库执行。prepare 必须使用全新目录；现有候选已经 prepare，不要重建覆盖。

```powershell
# 无外部请求：准备新候选或离线检查
python benchmark_session_memory.py --experiment --phase prepare --output reports/NEW_CAMPAIGN
python benchmark_session_memory.py --experiment --phase development --output reports/NEW_CAMPAIGN
python benchmark_session_memory.py --experiment --phase test --output reports/NEW_CAMPAIGN

# 用户核对开发 gold 后：凭据通过 SSH 只注入本机子进程
python deploy/run_long_session_experiment.py --phase development --output reports/NEW_CAMPAIGN

# 开发结束、用户核对全部测试 gold 后：冻结及主评测
python deploy/run_long_session_experiment.py --phase freeze --output reports/NEW_CAMPAIGN
python deploy/run_long_session_experiment.py --phase test --output reports/NEW_CAMPAIGN
python deploy/run_long_session_experiment.py --phase supplement --output reports/NEW_CAMPAIGN

# 生成/更新结果与匿名审阅表，无模型调用
python benchmark_session_memory.py --experiment --phase report --output reports/NEW_CAMPAIGN
```

固定五本阶段账本上限分别为 250/1400/250/50/50，总计 2000。不转移额度，不新建目录绕过限额。每次发送前原子记账；无自动重试。已有完成任务按输入哈希复用；开始后未落盘的任务会阻止恢复，须人工核对账本和证据，不能猜测远端是否执行而自动补发。

离线保守估算包含按 20 条批量的叶与各层向量、摘要、查询向量和回答；忽略未知父节点重复以保守计数。同案例相同问题只请求一次向量。当前数据开发 242/250、主评测 1213/1400，补充最多 112/250，实际以账本为准。超过估算或阶段额度即保留进度并报告未完成。

主评测 280 条；预先选择每类 test_00 和 test_05，各五策略重复一次，共 70 条；另两项消融 28 条。重复/消融单列，不混入主结果或扩大 bootstrap 独立样本量。

只比较独立会话的配对结果，10,000 次 bootstrap，报告 95% 区间。这些是探索性比较，没有多重比较后的全局优越性结论。完整测试与人工复核前不推荐策略；之后按复杂度顺序，仅当人工联合指标差异区间为正且角色/执行错误不增加时升级推荐。推荐只是实验结论，不自动改变生产默认。

本轮没有修改运行时代码，所以 API 验证额度保留，未重跑服务器接口。历史 API 联调、此次本地实验和正式产品验收分别记录。

## 已核验的历史检索证据

`python validate_paper_retrieval_release.py --replay` 于本轮离线通过；不加载模型、不访问视频服务。冻结发布包含 100 查询、3775 个已判定 query-scene 对、725 个场景，五模式 Top-20 联合标注池。

| 模式 | nDCG@10（100 查询） | MRR@10（75 个有正例查询） | 池内 Recall@10（75 查询） |
|---|---:|---:|---:|
| BM25 | 0.469299 | 0.547143 | 0.483943 |
| Dense | 0.592678 | 0.629497 | 0.634454 |
| Visual | 0.588337 | 0.594138 | 0.651680 |
| Hybrid | 0.575311 | 0.663222 | 0.613911 |
| Tri-Hybrid | 0.646976 | 0.722497 | 0.716181 |

最终引用数值以冻结 CSV 为准。Tri-Hybrid 相对 BM25 的 nDCG@10 配对差异约 0.178，95% 区间 [0.146, 0.210]；相对 Dense 约 0.054，[0.016, 0.093]。这些是已有候选池的排名回放，不证明当前在线服务成功率、当前手册检索质量、整库召回或独立盲测通过。原始媒体与索引覆盖限制见冻结包 REPRODUCE.md / LARGE_ARTIFACT_LOCATOR.json，不能把清单校验写成所有原始视频均重新生成。

## 能力表述核对

| 能力 | 代码事实与证据 | 可用表述与限制 |
|---|---|---|
| 用户图片输入 | API 校验最多三张 data URL，完整透传视觉消息 | 已实现图片输入通路；本轮未测故障图片诊断准确率 |
| 媒体关联返回 | 手册图片锚点、视频时间区间与鉴权媒体返回 | 支持图文/视频证据关联；不是统一向量空间的证明 |
| 视频预处理 | preprocess/scenes 抽取媒体和场景，ASR、OCR、VLM 各有模块与清单 | 已实现离线处理链路；本轮没有重新运行生成过程 |
| 视觉检索 | Qwen3-VL-Embedding 版本固定的加载器及 visual/tri_hybrid 路径；存在冻结五模式结果 | 有历史视觉检索实验；不宣称所有图片、手册文本、视频共享一个线上向量库 |
| 用户视频诊断 | 未配置视觉模型时返回 insufficient_evidence | 实验性能力；不能宣称当前已完成准确诊断验收 |

对应代码：api_server.py，video_rag/asr.py、ocr.py、vlm.py、visual_model.py、retrieval.py、diagnosis.py。以上为静态能力和历史证据审计，不改变整体质量验收记录。

## 实验结论使用边界

项目名使用“设备故障排查 AI Agent”。强调长会话中的事实更正、否定、执行状态和来源追踪。新实验未完成前不声称分层记忆提高准确率，也不把 21 条原评测写成整体 95.2% 准确率。
