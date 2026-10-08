"""安全的默认运行配置。

本文件只保存非敏感默认值。API Key、Bearer Token、邮箱密码等秘密必须通过
进程环境、项目根目录下未纳入 Git 的 ``.env``，或部署平台 Secret 注入。
``apply_default_env`` 不会覆盖调用方已经设置的环境变量。
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

# 说明：DEFAULT_ENV 的取值会**盖过**各模块里 os.getenv 的第二个参数（兜底值）。
# 因为 apply_default_env() 在 api_server / llm_router 的模块导入阶段最先执行，
# setdefault 写入后，后续 os.getenv(key, fallback) 拿到的就是这里的值，fallback 永远不生效。
# 因此下面每个值都必须与各调用点的语义一致；新增条目时若与代码兜底值不同，等于改行为。
# 已知的"代码兜底值已失效"清单（保留兜底只是历史遗留，勿据此判断实际生效值）：
#   CHAT_TIMEOUT_S              代码兜底 20，实际 50
#   CHAT_MULTIMODAL_TIMEOUT_S   代码兜底 30，实际 60
#   DEEPSEEK_BINARY_TIMEOUT     代码兜底 20，实际 3
#   LLM_TIMEOUT_SECONDS         代码兜底 90，实际 30
#   SILICONFLOW_MAX_CONCURRENCY 代码兜底 15，实际 3
DEFAULT_ENV = {
    # 请求限制。
    "CHAT_TIMEOUT_S": "50",
    "CHAT_MULTIMODAL_TIMEOUT_S": "60",
    "CHAT_MAX_IMAGES": "3",
    "CHAT_MAX_IMAGE_BYTES": str(5 * 1024 * 1024),

    # service / tech 二分类器。DEEPSEEK_API_KEY 必须外部注入。
    "DEEPSEEK_BASE_URL": "https://api.deepseek.com",
    "DEEPSEEK_BINARY_MODEL": "deepseek-v4-flash",
    "DEEPSEEK_BINARY_TIMEOUT": "3",
    "DEEPSEEK_BINARY_MAX_TOKENS": "4",

    # 主回答模型。BASE_URL、API_KEY 与实际模型名由部署环境配置。
    "SILICONFLOW_ONLY": "1",
    "SILICONFLOW_MAX_CONCURRENCY": "3",
    "SILICONFLOW_MODEL": "gpt-5.5-openai-compact",
    "AGENT_MAX_TOKENS": "8192",
    "LLM_CONTEXT_WINDOW_TOKENS": "32768",
    "LLM_CONTEXT_SAFETY_TOKENS": "1024",
    "LLM_IMAGE_TOKEN_RESERVE": "8192",
    "LLM_TIMEOUT_SECONDS": "30",
    "LLM_TRANSIENT_RETRY_ATTEMPTS": "3",

    # LLM 路由韧性（审计 B3 引入）。
    "LLM_ROUTE_DISABLE_TTL_S": "300",
    "LLM_ROUTE_ACQUIRE_TIMEOUT_S": "60",
    "LLM_THINKING_BUDGET_TOKENS": "4096",
    "HIGHSPEED_MAX_CONCURRENCY": "20",
    "TOKEN_PLAN_MAX_CONCURRENCY": "2",

    # 检索服务。EMBEDDING_API_KEY / RERANK_API_KEY 必须外部注入。
    "EMBEDDING_BASE_URL": "https://api.siliconflow.cn/v1",
    "EMBEDDING_MODEL": "Pro/BAAI/bge-m3",
    "EMBEDDING_MAX_CONCURRENCY": "4",
    "RERANK_BASE_URL": "https://api.siliconflow.cn/v1",
    "RERANK_MODEL_ALIAS": "BAAI/bge-reranker-v2-m3",
    "RERANK_ENABLED": "1",
    # rerank 重试预算（审计 B3 引入）：总预算 20s、单次 15s。
    "RERANK_TOTAL_BUDGET_S": "20",
    "RERANK_TIMEOUT_S": "15",
    # 关闭 dense 召回后仅走 BM25；默认开启。
    "MANUAL_DENSE_ENABLED": "1",
    # Acceptance profiles opt in: never replace failed manual services with BM25/original ordering.
    "MANUAL_REQUIRE_EXACT_MODE": "0",
    # Empty uses the repository's data/index; acceptance can point at a private rebuilt index.
    "MANUAL_INDEX_DIR": "",

    # chunk 命中后返回完整 parent section。
    "RETURN_PARENT_SECTION": "1",

    # agent 答案整形（SECTION_FULL_TOP_N 由审计 B4 恢复为 2，0 可回到旧行为）。
    "AGENT_SECTION_FULL_TOP_N": "2",
    "AGENT_PREREAD_AS_TEXT": "0",
    "AGENT_FINALIZE_MAX_TOKENS": "4096",
    "GROUNDING_REVISION_MAX_TOKENS": "1600",
    "GROUNDING_REVISION_TIMEOUT_SECONDS": "45",

    # /chat 会话历史与 agent 执行器（审计 B2/B3 引入）。
    "CHAT_SESSION_HISTORY_LIMIT": "6",
    "CHAT_SESSION_MAX_SESSIONS": "500",
    "CHAT_SESSION_TTL_S": "3600",
    "CHAT_SESSION_DB_PATH": str(Path(__file__).resolve().parent / "data/runtime/chat_sessions.sqlite3"),
    "CHAT_SESSION_RECENT_CHARS": "6000",
    "CHAT_SESSION_SUMMARY_CHARS": "4000",
    "CHAT_SESSION_EXCERPT_CHARS": "800",
    "CHAT_SESSION_RECALL_CHARS": "2000",
    "CHAT_SESSION_ARCHIVE_TURNS": "200",
    "CHAT_SESSION_ARCHIVE_CHARS": "2000000",
    "CHAT_SESSION_CONTEXT_TOKENS": "6000",
    "CHAT_SESSION_RECENT_TOKENS": "2800",
    "CHAT_SESSION_SUMMARY_TOKENS": "1400",
    "CHAT_SESSION_RECALL_TOKENS": "800",
    "CHAT_SESSION_SUMMARY_INPUT_TOKENS": "16000",
    "CHAT_SESSION_MODEL_SUMMARY": "1",
    "CHAT_SESSION_SUMMARY_MODEL": "",
    "CHAT_SESSION_SUMMARY_TIMEOUT_S": "4",
    "CHAT_SESSION_SUMMARY_MAX_TOKENS": "1200",
    "CHAT_SESSION_TREE_ENABLED": "1",
    "CHAT_SESSION_TREE_RETRIEVAL": "collapsed",
    "CHAT_SESSION_TREE_CHUNK_TOKENS": "800",
    "CHAT_SESSION_TREE_MAX_LEAVES": "512",
    "CHAT_SESSION_TREE_BRANCH_SIZE": "4",
    "CHAT_SESSION_TREE_MAX_LEVELS": "8",
    "CHAT_SESSION_TREE_MODEL_CALLS": "3",
    "CHAT_SESSION_TREE_BUILD_TIMEOUT_S": "8",
    "CHAT_SESSION_TREE_TOP_K": "3",
    "CHAT_SESSION_TREE_EMBEDDING": "lexical",
    "CHAT_SESSION_TREE_EMBEDDING_TIMEOUT_S": "4",
    "CHAT_AGENT_MAX_WORKERS": "8",
    "CHAT_AGENT_DEADLINE_ENABLED": "1",
    # 审计 B2 把硬编码的 /tmp 路径换成系统临时目录（Windows 下 /tmp 会落到 \tmp\）。
    "CHAT_API_TRACE_PATH": str(Path(tempfile.gettempdir()) / "kbrag_chat_api_server.trace.jsonl"),

    # 用户视频异步诊断：inline 与代码兜底一致；spool 需外部显式开启。
    "USER_VIDEO_ASYNC_MODE": "inline",

    # P1 video retrieval. Production deployment additionally sets exact mode to 1.
    "VIDEO_RETRIEVAL_MODE": "tri_hybrid",
    "VIDEO_REQUIRE_EXACT_MODE": "0",
    # 与 .env.example 的 production 不同，这里刻意保留 development：
    # production 会强制校验 tri_hybrid + exact mode，默认打开会让所有未显式配置的
    # 环境在 import api_server 阶段直接 RuntimeError。因此改为"漏配时显式告警"
    # （见 VIDEO_PROFILE_WARN_NON_PRODUCTION 与 /health 的 video_profile_warning）。
    "VIDEO_RUNTIME_PROFILE": "development",
    "VIDEO_PROFILE_WARN_NON_PRODUCTION": "1",
    "VIDEO_MIN_CONTENT_BM25": "2.0",
    "VIDEO_SCENE_CONTENT_GATE_RELATIVE": "0.10",
    "VIDEO_SCENE_CONTENT_GATE_ABSOLUTE": "0.25",
    "USER_VIDEO_MAX_BYTES": str(200 * 1024 * 1024),
    "USER_VIDEO_MAX_DURATION_S": "60",
    "USER_VIDEO_MAX_PIXELS": str(3840 * 2160),
}

# 运行期真实读取、但**不能**在 DEFAULT_ENV 里给默认值的变量。
# 三类原因：① 密钥/令牌，写进来等于把秘密提交进 Git；
#          ② 空值本身有意义（empty string = 该功能关闭），给默认值会把它打开；
#          ③ 由代码在运行期写入，不属配置。
# 登记在这里是为了让 missing_declared_env() 能报出"该配没配"，而不是让它们静默退化。
DOCUMENTED_ENV: dict[str, str] = {
    # ① 密钥/令牌
    "KAFU_API_TOKEN": "secret",
    "DEEPSEEK_API_KEY": "secret",
    "EMBEDDING_API_KEY": "secret",
    "CHAT_SESSION_TREE_EMBEDDING_API_KEY": "secret",
    "CHAT_SESSION_TREE_EMBEDDING_BASE_URL": "secret",
    "CHAT_SESSION_TREE_EMBEDDING_MODEL": "empty-disables",
    "MODEL_ATTEMPT_LEDGER": "empty-disables",
    "RERANK_API_KEY": "secret",
    "SILICONFLOW_API_KEY": "secret",
    "SILICONFLOW_BASE_URL": "secret",
    # ② 空值有意义：留空表示不启用该路能力，给默认值会误开启
    "VIDEO_DENSE_ENDPOINT": "empty-disables",
    "VIDEO_VISUAL_DENSE_ENDPOINT": "empty-disables",
    "USER_VIDEO_VLM_BASE_URL": "empty-disables",
    "USER_VIDEO_VLM_API_KEY": "empty-disables",
    "USER_VIDEO_VLM_MODEL": "empty-disables",
    "RERANK_FALLBACK_LOG": "empty-disables",
    "RERANK_TIMING_LOG": "empty-disables",
    "CHAT_API_RAW_PATH": "empty-disables",
    "DEBUG_ROUTE": "empty-disables",
    # ③ 运行期写入
    "CURRENT_QID": "runtime-written",
}


def apply_default_env() -> None:
    """Apply non-sensitive defaults without overriding caller-provided values."""
    for key, value in DEFAULT_ENV.items():
        os.environ.setdefault(key, value)


def missing_declared_env() -> list[str]:
    """返回已登记但当前进程里未设置的变量名（按名排序）。

    用于启动自检与 /health：这些变量配错了不会报错，只会静默退化成"少了某种能力"，
    是排查线上行为异常时最先要看的清单。
    """
    return sorted(name for name in DOCUMENTED_ENV if not os.environ.get(name))
