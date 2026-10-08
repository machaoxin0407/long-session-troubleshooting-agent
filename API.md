# API 接口说明

## 基本信息

- 服务入口：`api_server.py`
- 启动脚本：`./run_api.sh`
- 健康检查：`GET /health`
- 对话接口：`POST /chat`
- 默认端口：`8000`
- 认证：`Authorization: Bearer $KAFU_API_TOKEN`

## 启动

```bash
pip install -r requirements.txt
./run_api.sh
```

如需自定义监听地址：

```bash
HOST=127.0.0.1 PORT=8000 ./run_api.sh
```

## GET /health

```bash
curl http://127.0.0.1:8000/health
```

返回示例：

```json
{
  "status": "ok",
  "engine_ready": true,
  "timeout_s": 50.0,
  "multimodal_timeout_s": 60.0,
  "auth_configured": true,
  "classifier_provider": "deepseek_binary_vote",
  "classifier_configured": true,
  "classifier_model": "deepseek-v4-flash"
}
```

## POST /chat

### 请求头

```text
Authorization: Bearer <your-token>
Content-Type: application/json
```

### 请求体

```json
{
  "question": "椅子的扶手使用一段时间后为什么会松动？",
  "images": [],
  "session_id": "demo",
  "stream": false
}
```

字段说明：

| 字段 | 类型 | 必选 | 说明 |
| --- | --- | --- | --- |
| `question` | string | 是 | 用户问题，不能为空。 |
| `images` | array[string] | 否 | Base64 data URL 图片，格式如 `data:image/png;base64,...`，最多 3 张。 |
| `session_id` | string | 否 | 会话 ID；复用该 ID 接续持久化上下文，不传则新建会话。 |
| `stream` | boolean | 否 | 兼容字段，当前统一同步返回完整答案。 |
| `memory_retrieval` | string | 否 | `collapsed` 同时检索所有层；`traversal` 从根逐层筛选；缺省使用服务配置，非法值返回 422。 |

### 调用示例

服务按 `session_id` 保存原文、多层摘要树缓存及近期对话，只读取当前会话。
短历史保留原文；超出记忆预算后，在下一次提问前对较早文本分块、反复聚类并逐层摘要，默认保留最近 3 轮原文。
树检索结果占摘要窗口；关键词原文回查继续占独立回查窗口。两种树检索方式共用持久化树，不因切换模式重建。
记忆总预算 6000、近期窗口 2800、摘要 1400、回查 800，单位为保守的 UTF-8 字节 token 估算，
不是任意上游模型的精确 tokenizer 计数。当前问题保持原样；超长历史消息移出上下文并可回查。
默认使用本地词项 TF-IDF 向量和有容量限制的球面 k-means 聚类，不新增依赖或额外 embedding 请求。
设置 `CHAT_SESSION_TREE_EMBEDDING=semantic` 可使用现有 `EMBEDDING_*` 服务；总时间限制、取消或服务失败时回退词项向量，内部 trace 记录实际向量方式。适配器默认 4 秒且不重试。
摘要使用已配置的首个回答模型路由，按层批量请求，单次最多 4 秒、不重试；默认最多 3 次，总推理预算 8 秒，并服从请求剩余时间。
未送入模型的组、模型失败或引用校验不通过的组使用本地摘录，依然形成完整树。总推理预算不包含本地聚类和数据库操作。
旧客服建议与用户陈述分开，摘要不会作为已核验知识。
默认最后一次成功写入后 1 小时过期；重启后复用同一 ID、同一数据库且未过期时可继续。
数据库默认位于 `data/runtime/chat_sessions.sqlite3`，容器应挂载持久卷。
网页 Demo 会保存当前标签页的会话 ID，并提供新建和清空按钮。
设置 `CHAT_SESSION_TREE_ENABLED=0` 可恢复单层结构化摘要路径；请求检索模式字段仅在树开启且历史超预算时生效。
详细配置、检索语义和限制见 [树形记忆说明](docs/SESSION_TREE_20261005.md)。旧单层实现见 [压缩升级说明](docs/SESSION_COMPACTION_20261001.md)。

### 清空当前会话

`DELETE /v2/chat/sessions/{session_id}`，使用相同的 Bearer 鉴权，返回 HTTP 204。
同时删除该会话的摘要、记忆树、近期消息和原文；其他会话不受影响。
没有历史的 ID 也返回 204。记忆读取、写入或清理失败返回 503，不静默退化为无上下文。

### 单轮请求示例

```bash
curl -sS -X POST http://127.0.0.1:8000/chat \
  -H "Authorization: Bearer $KAFU_API_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "椅子的扶手使用一段时间后为什么会松动？",
    "session_id": "demo"
  }'
```

### 成功响应

```json
{
  "code": 0,
  "msg": "success",
  "data": {
    "answer": "...",
    "session_id": "demo",
    "timestamp": 1780000000,
    "videos": [
      {
        "scene_id": "camera-001-scene-0001",
        "record_id": "camera-001",
        "product_class": "Camera",
        "start_seconds": 0.0,
        "end_seconds": 25.4,
        "clip_url": "/video-media/processed/camera-001/clips/camera-001-scene-0001.mp4",
        "thumbnail_url": "/video-media/keyframes/camera-001/frame_000000.jpg",
        "score": 6.2,
        "evidence_text": "ASR: ..."
      }
    ]
  }
}
```

其中 `data.answer` 与离线 CSV 提交 `ret` 字段同源：

- 客服题：纯文本客服回答。
- 技术题：正文 + `<PIC>` 锚点 + 末尾图片数组字符串。
- `videos`：技术题命中的 0–3 个视频场景；客服题或无匹配时为空数组。

`clip_url` 与 `thumbnail_url` 需要携带相同的 Bearer Token 请求。媒体端点只允许读取项目生成的 MP4 片段和 JPG 关键帧：

```bash
curl -H "Authorization: Bearer $KAFU_API_TOKEN" \
  http://127.0.0.1:8000/video-media/processed/camera-001/clips/camera-001-scene-0001.mp4 \
  -o scene.mp4
```

离线验证视频排名、响应序列化和媒体目录边界：

```bash
python validate_video_retrieval.py
```

## 错误响应

常见错误：

| HTTP 状态码 | 场景 |
| ---: | --- |
| 401 | 缺少 Bearer Token 或 Token 错误。 |
| 422 | 请求字段校验失败，如问题为空、图片格式错误、图片过大。 |
| 500 | Agent 或上游模型内部错误。 |
| 504 | Agent 超时。 |

## 配置覆盖

默认配置写在 `config_runtime.py`。如需覆盖，可在启动前设置同名环境变量，例如：

```bash
SILICONFLOW_BASE_URL="https://your-openai-compatible-endpoint/v1" \
SILICONFLOW_API_KEY="sk-..." \
SILICONFLOW_MODEL="gpt-5.5" \
./run_api.sh
```

默认 embedding / rerank 使用硅基流动远程服务，无需本地启动额外服务。
# Request context limits (2026-10-08)

`/chat` and `/v2/chat` return HTTP 413 with the existing `detail` error shape when
fixed question, instructions, attachments, tools, output reservation and safety
margin cannot fit the configured request budget. Failed requests are not appended
to session memory. Success responses and optional `memory_retrieval` are unchanged.

The default request window is 32768 estimated tokens, safety margin 1024, and
visual reservation 8192 per image. Text is measured using conservative UTF-8 bytes;
this is not the provider's exact tokenizer or a guarantee about visual tokens.
Images remain intact. Three images may exceed the request budget even though the
request schema permits three images. Server-only trace records estimates, removed
context and fallback reasons; the API does not expose credentials or prompts.
