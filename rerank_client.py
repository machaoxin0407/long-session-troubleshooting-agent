"""
rerank 服务客户端。

当前主线默认对接硅基流动远端 `BAAI/bge-reranker-v2-m3`。
本客户端仍兼容本地 llama-server rerank 接口，但本地 8090 仅作为显式 fallback：
只有设置 `RERANK_BASE_URL=http://127.0.0.1:8090` 时才会使用。
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import requests
from dotenv import load_dotenv

load_dotenv()

try:
    from config_runtime import apply_default_env

    apply_default_env()
except Exception:
    pass


# 固定默认：硅基流动远端 BAAI/bge-reranker-v2-m3（2026-06-05 用户锁定，本地 8090 不再依赖）。
# 仍可用 RERANK_* 环境变量临时覆盖。
DEFAULT_RERANK_BASE_URL = os.getenv("RERANK_BASE_URL", "https://api.siliconflow.cn/v1")
DEFAULT_RERANK_API_KEY = os.getenv("RERANK_API_KEY", "")
DEFAULT_RERANK_MODEL = os.getenv("RERANK_MODEL_ALIAS", "BAAI/bge-reranker-v2-m3")

# rerank 总预算（秒）：从第一次请求起算，超时后停止重试直接报错。0=不限制（旧行为）。
RERANK_TOTAL_BUDGET_S = float(os.getenv("RERANK_TOTAL_BUDGET_S", "20"))
# 单次请求超时（秒）：从 120 降到 15，失败快速轮换而不是吊死在单个端点上。
RERANK_TIMEOUT_S = int(os.getenv("RERANK_TIMEOUT_S", "15"))
# 瞬时错误的最大尝试轮数
_MAX_RERANK_ATTEMPTS = 6


class RerankError(RuntimeError):
    """Raised when every rerank endpoint/payload attempt fails."""
    pass


@dataclass
class RerankResult:
    """One reranked document index and its relevance score."""
    index: int
    score: float


class RerankClient:
    """Small compatibility client for SiliconFlow and local rerank endpoints."""

    def __init__(
        self,
        base_url: str = DEFAULT_RERANK_BASE_URL,
        api_key: str = DEFAULT_RERANK_API_KEY,
        model: str = DEFAULT_RERANK_MODEL,
        timeout: int = RERANK_TIMEOUT_S,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        # 复用连接池：每次调用新建 Session 会浪费 TLS 握手并泄漏 fd
        self._session = requests.Session()
        self._session.trust_env = False

    def rerank(self, query: str, documents: list[str], top_n: int | None = None) -> list[RerankResult]:
        """Return documents sorted by rerank score, retrying transient endpoint failures.

        重试策略（审计 F3）：
        - 首个组合（/rerank + documents 字段）是主路径，每轮最先尝试；
        - 4xx 客户端错误（鉴权/参数问题）重试也不会成功，立即失败不重试；
        - 仅瞬时错误（网络/5xx/格式）重试，且受 RERANK_TOTAL_BUDGET_S 总预算约束；
        - 预算耗尽或轮数用尽后抛 RerankError，由上层降级到召回排序。
        """
        if not documents:
            return []

        top_n = top_n or len(documents)
        deadline = time.monotonic() + RERANK_TOTAL_BUDGET_S if RERANK_TOTAL_BUDGET_S > 0 else None

        combos = [
            (endpoint, body)
            for endpoint in self._candidate_endpoints()
            for body in self._candidate_payloads(query, documents, top_n)
        ]
        last_error: Exception | None = None

        strict = os.getenv("MANUAL_REQUIRE_EXACT_MODE", "0").lower() in {"1", "true", "yes", "on"}
        attempts = 1 if strict else _MAX_RERANK_ATTEMPTS
        if strict:
            combos = combos[:1]
        for attempt in range(attempts):
            for endpoint, body in combos:
                # 预算检查放在每个组合之前（首个组合总是放行，保证至少试一次主路径）
                if last_error is not None and deadline is not None and time.monotonic() >= deadline:
                    raise RerankError(
                        f"调用 rerank 服务失败（预算 {RERANK_TOTAL_BUDGET_S:g}s 耗尽）: {last_error}"
                    )
                try:
                    response_payload = self._post(endpoint, body)
                    return self._parse_response(response_payload, len(documents))
                except Exception as exc:
                    if self._is_permanent_error(exc):
                        raise RerankError(f"rerank 请求被拒绝（4xx 不重试）: {exc}") from exc
                    last_error = exc
            if attempt < attempts - 1:
                if deadline is not None and time.monotonic() >= deadline:
                    break
                time.sleep(min(8.0, 0.8 * (2 ** attempt)))

        raise RerankError(f"调用 rerank 服务失败: {last_error}")

    @staticmethod
    def _is_permanent_error(exc: Exception) -> bool:
        """4xx（鉴权失败/参数错误等）重试无意义，识别为永久错误。"""
        status = getattr(getattr(exc, "response", None), "status_code", None)
        return status is not None and 400 <= status < 500

    def _candidate_endpoints(self) -> list[str]:
        base = self.base_url.rstrip("/")
        candidates = [
            f"{base}/rerank",
            f"{base}/reranking",
        ]
        if not base.endswith("/v1"):
            candidates.extend(
                [
                    f"{base}/v1/rerank",
                    f"{base}/v1/reranking",
                ]
            )
        return list(dict.fromkeys(candidates))

    def _candidate_payloads(self, query: str, documents: list[str], top_n: int) -> list[dict]:
        return [
            {
                "model": self.model,
                "query": query,
                "documents": documents,
                "top_n": top_n,
                "return_documents": False,
            },
            {
                "model": self.model,
                "query": query,
                "texts": documents,
                "top_n": top_n,
                "return_documents": False,
            },
        ]

    def _post(self, endpoint: str, payload: dict) -> dict:
        response = self._session.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def _parse_response(self, payload: dict, total_documents: int) -> list[RerankResult]:
        strict = os.getenv("MANUAL_REQUIRE_EXACT_MODE", "0").lower() in {"1", "true", "yes", "on"}
        if strict:
            items = payload.get("results") if isinstance(payload, dict) else None
            if not isinstance(items, list) or not items:
                raise RerankError("Strict rerank response must contain a nonempty results list")
            seen = set()
            for item in items:
                if not isinstance(item, dict):
                    raise RerankError("Strict rerank result must be an object")
                index, score = item.get("index"), item.get("relevance_score")
                if (type(index) is not int or not 0 <= index < total_documents or index in seen
                        or type(score) not in (int, float) or not math.isfinite(score)):
                    raise RerankError("Strict rerank result has an invalid index or score")
                seen.add(index)
            return sorted([RerankResult(index=item["index"], score=float(item["relevance_score"]))
                           for item in items], key=lambda item: item.score, reverse=True)
        # 用 is None 逐个判断：or 链会让"存在但为空列表"的合法响应 falsy 穿透到下一个 key。
        items = payload.get("results")
        if items is None:
            items = payload.get("data")
        if items is None:
            items = payload.get("ranked_documents")
        if items is None:
            raise RerankError(f"未知 rerank 返回格式: {payload}")

        results: list[RerankResult] = []
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            index = item.get("index", idx)
            score = item.get("relevance_score", item.get("score", item.get("similarity", 0.0)))
            try:
                index = int(index)
            except (TypeError, ValueError):
                index = idx
            try:
                score = float(score)
            except (TypeError, ValueError):
                score = 0.0
            if 0 <= index < total_documents:
                results.append(RerankResult(index=index, score=score))

        if not results:
            raise RerankError(f"rerank 返回为空: {payload}")

        results.sort(key=lambda item: item.score, reverse=True)
        return results
