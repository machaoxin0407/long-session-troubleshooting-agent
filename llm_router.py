"""Anthropic 路由：默认走 highspeed，额度耗尽后自动切到 token plan。"""

from __future__ import annotations

import os
import json
import time
import threading
from dataclasses import dataclass
from types import SimpleNamespace

from dotenv import load_dotenv

from local_model_deadline import call_local_model_until
from local_model_stream import stream_local_model_until, uses_request_local_transport
from request_cancellation import acquire_request_slot, check_request_cancelled
from request_context import prepare_body
from model_attempts import record_attempt

load_dotenv()

try:
    from config_runtime import apply_default_env

    apply_default_env()
except Exception as exc:
    # 静默吞掉会让所有默认值失效且无从排查，至少留一条告警。
    import logging

    logging.getLogger(__name__).warning("config_runtime defaults not applied: %s", exc)


@dataclass(frozen=True)
class AnthropicRoute:
    """一个可用 LLM 上游路由。

    protocol 区分 Anthropic Messages API 与 OpenAI-compatible Chat Completions；其余调用方统一拿到 Anthropic 形状的 response，降低 agent 主循环复杂度。
    """
    name: str
    base_url: str
    api_key: str
    model: str
    protocol: str = "anthropic"


# 路由禁用记录：name -> 禁用时刻。保留时间戳是为了让禁用可自动过期，
# 否则一次误判会把路由永久钉死，进程内无法自愈。
_DISABLED_ROUTES: dict[str, float] = {}
_ROUTE_LOCK = threading.Lock()
_ROUTE_SEMAPHORES: dict[str, threading.BoundedSemaphore] = {}

_DEFAULT_ROUTE_CONCURRENCY = {
    "highspeed": int(os.getenv("HIGHSPEED_MAX_CONCURRENCY", "20")),
    "token-plan": int(os.getenv("TOKEN_PLAN_MAX_CONCURRENCY", "2")),
    "siliconflow": int(os.getenv("SILICONFLOW_MAX_CONCURRENCY", "15")),
}
_DEFAULT_LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT_SECONDS", "90"))
_TRANSIENT_RETRY_ATTEMPTS = max(1, int(os.getenv("LLM_TRANSIENT_RETRY_ATTEMPTS", "3")))

# 路由被判额度耗尽后的冷却时长（秒）。过期后自动重新纳入候选，避免永久禁用。
_ROUTE_DISABLE_TTL_S = float(os.getenv("LLM_ROUTE_DISABLE_TTL_S", "300"))

# 等待路由并发槽的超时（秒）。超时后不再无限阻塞，转下一条路由或直接报错。
# <=0 表示维持旧的无限等待行为。
_ROUTE_ACQUIRE_TIMEOUT_S = float(os.getenv("LLM_ROUTE_ACQUIRE_TIMEOUT_S", "60"))

# OpenAI 兼容客户端缓存：(base_url, api_key) -> client。
# 每次请求新建客户端会重复建连/TLS 握手，并泄漏 httpx 连接池。
_OPENAI_CLIENT_CACHE: dict[tuple[str, str], object] = {}
_OPENAI_CLIENT_LOCK = threading.Lock()


def _get_openai_client(base_url: str, api_key: str):
    """按 (base_url, api_key) 复用 OpenAI 兼容客户端实例。"""
    key = (base_url, api_key)
    with _OPENAI_CLIENT_LOCK:
        client = _OPENAI_CLIENT_CACHE.get(key)
        if client is None:
            from openai import OpenAI

            options = {'max_retries': 0} if os.getenv('MODEL_ATTEMPT_LEDGER') else {}
            client = OpenAI(base_url=base_url, api_key=api_key, timeout=_DEFAULT_LLM_TIMEOUT, **options)
            _OPENAI_CLIENT_CACHE[key] = client
        return client


def _norm(text: str | None) -> str:
    return (text or "").strip()


def _build_route(prefix: str, default_name: str, default_model: str) -> AnthropicRoute | None:
    default_base_url = ""
    default_api_key = ""
    if prefix == "SILICONFLOW":
        default_model = "gpt-5.5-openai-compact"
    base_url = _norm(os.getenv(f"{prefix}_BASE_URL", default_base_url))
    api_key = _norm(os.getenv(f"{prefix}_API_KEY", default_api_key))
    model = _norm(os.getenv(f"{prefix}_MODEL")) or default_model
    if not base_url or not api_key or not model:
        return None
    return AnthropicRoute(
        name=default_name,
        base_url=base_url,
        api_key=api_key,
        model=model,
    )


def _build_openai_route(prefix: str, default_name: str, default_model: str) -> AnthropicRoute | None:
    route = _build_route(prefix, default_name, default_model)
    if route is None:
        return None
    return AnthropicRoute(
        name=route.name,
        base_url=route.base_url,
        api_key=route.api_key,
        model=route.model,
        protocol="openai",
    )


def _env_flag(name: str) -> bool:
    return _norm(os.getenv(name)).lower() in {"1", "true", "yes", "on"}


def get_routes() -> list[AnthropicRoute]:
    """返回按优先级排序的可用路由。"""
    routes: list[AnthropicRoute] = []

    siliconflow = _build_openai_route(
        prefix="SILICONFLOW",
        default_name="siliconflow",
        default_model="Qwen/Qwen3.6-27B",
    )
    if siliconflow is not None and _env_flag("SILICONFLOW_ONLY"):
        return [siliconflow]
    if siliconflow is not None and not _env_flag("DISABLE_SILICONFLOW"):
        routes.append(siliconflow)

    if not _env_flag("DISABLE_HIGHSPEED"):
        highspeed = _build_route(
            prefix="HIGHSPEED",
            default_name="highspeed",
            default_model="MiniMax-M2.7-highspeed",
        )
        if highspeed is not None:
            routes.append(highspeed)

    token_plan = _build_route(
        prefix="ANTHROPIC",
        default_name="token-plan",
        default_model="MiniMax-M2.7",
    )
    if token_plan is not None:
        same_as_existing = any(
            r.base_url == token_plan.base_url
            and r.api_key == token_plan.api_key
            and r.model == token_plan.model
            for r in routes
        )
        if not same_as_existing:
            routes.append(token_plan)

    return routes


def get_default_model() -> str:
    routes = get_routes()
    if routes:
        return routes[0].model
    return "gpt-5.5"


def describe_routes() -> list[dict[str, str]]:
    """返回不含密钥的路由信息，方便调试。"""
    return [
        {
            "name": route.name,
            "base_url": route.base_url,
            "model": route.model,
            "protocol": route.protocol,
            "concurrency": str(get_route_concurrency(route.name)),
            "disabled": str(_route_disabled_at(route.name) is not None).lower(),
        }
        for route in get_routes()
    ]


# 额度/余额类错误的判据。注意不能用裸子串 "credit"：它会命中
# "invalid credentials"，把认证失败误判成额度耗尽，进而禁用路由。
_QUOTA_MARKERS = (
    "insufficient_quota",
    "quota",
    "余额",
    "额度",
    "余额不足",
    "配额",
    "402",
)
# 认证/鉴权类错误：即使文本里出现 quota/credit 等字样，也绝不视为额度问题。
_AUTH_MARKERS = (
    "credential",
    "unauthorized",
    "forbidden",
    "invalid api key",
    "authentication",
    "401",
    "403",
)


def _should_fallback(exc: Exception) -> bool:
    """判断是否应切换到下一条付费路由；只对额度/余额类错误 fallback。"""
    text = str(exc).lower()
    if any(marker in text for marker in _AUTH_MARKERS):
        return False
    return any(marker in text for marker in _QUOTA_MARKERS)


def _route_disabled_at(route_name: str) -> float | None:
    """返回路由的禁用时刻；超过冷却时长视为已恢复。

    TTL <= 0 表示不启用冷却：任何禁用记录立即过期。不能把 0 解释成"永不过期"，
    那正是本次要根治的永久禁用行为。
    """
    with _ROUTE_LOCK:
        disabled_at = _DISABLED_ROUTES.get(route_name)
        if disabled_at is None:
            return None
        if _ROUTE_DISABLE_TTL_S <= 0 or (time.monotonic() - disabled_at) >= _ROUTE_DISABLE_TTL_S:
            del _DISABLED_ROUTES[route_name]
            return None
        return disabled_at


def _disable_route(route_name: str) -> None:
    with _ROUTE_LOCK:
        _DISABLED_ROUTES[route_name] = time.monotonic()


def _should_retry_same_route(exc: Exception) -> bool:
    """判断是否在同一路由内重试；仅覆盖网络抖动、超时、连接断开等瞬时错误。"""
    text = str(exc).lower()
    markers = [
        "connection error",
        "apiconnectionerror",
        "apiconnectionerror",
        "readtimeout",
        "read timeout",
        "timeout",
        "timed out",
        "remoteprotocolerror",
        "protocol error",
        "max retries exceeded",
        "connection aborted",
        "connection reset",
        "server disconnected",
        "temporarily unavailable",
        "temporary failure",
    ]
    return any(marker in text for marker in markers)


def _active_routes() -> list[AnthropicRoute]:
    routes = get_routes()
    active = [r for r in routes if _route_disabled_at(r.name) is None]
    return active or routes


def get_route_concurrency(route_name: str) -> int:
    limit = _DEFAULT_ROUTE_CONCURRENCY.get(route_name, 1)
    return max(1, int(limit))


def _get_route_semaphore(route_name: str) -> threading.BoundedSemaphore:
    with _ROUTE_LOCK:
        sem = _ROUTE_SEMAPHORES.get(route_name)
        if sem is None:
            sem = threading.BoundedSemaphore(get_route_concurrency(route_name))
            _ROUTE_SEMAPHORES[route_name] = sem
        return sem


def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif item.get("type") == "tool_result":
                    parts.append(str(item.get("content", "")))
                elif item.get("type") in {"image_url", "image"}:
                    parts.append("[用户上传图片]")
            elif hasattr(item, "text"):
                parts.append(str(item.text))
        return "\n".join(p for p in parts if p)
    return str(content or "")


def _convert_multimodal_item_for_openai(item) -> dict | None:
    """把 Anthropic 风格 content block 转为 OpenAI-compatible 多模态 block。"""
    if not isinstance(item, dict):
        return {"type": "text", "text": _content_to_text(item)}
    item_type = item.get("type")
    if item_type == "text":
        return {"type": "text", "text": str(item.get("text", ""))}
    if item_type == "image_url":
        image_url = item.get("image_url")
        if isinstance(image_url, dict) and image_url.get("url"):
            return {"type": "image_url", "image_url": {"url": str(image_url["url"])}}
        if isinstance(image_url, str):
            return {"type": "image_url", "image_url": {"url": image_url}}
    return None


def _convert_messages_for_openai(system: str, messages: list) -> list[dict]:
    """把 agent 内部 Anthropic Messages 形状转换成 OpenAI Chat Completions 形状。

    需要同时处理三类差异：system 独立字段转 system message；assistant tool_use 转 tool_calls；user tool_result 转 tool 消息。若用户 content 含 image_url，则保留多模态数组而不是拍平成纯文本。
    """
    converted: list[dict] = [{"role": "system", "content": system}]
    pending_tool_ids: set[str] = set()

    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")

        if role == "assistant" and isinstance(content, list):
            text_parts: list[str] = []
            tool_calls: list[dict] = []
            for block in content:
                def field(name, default=None):
                    return block.get(name, default) if isinstance(block, dict) else getattr(block, name, default)
                block_type = field("type")
                if block_type == "text":
                    text_parts.append(field("text", "") or "")
                elif block_type == "tool_use":
                    tool_id = field("id", "") or f"call_{len(pending_tool_ids) + 1}"
                    pending_tool_ids.add(tool_id)
                    tool_calls.append({
                        "id": tool_id,
                        "type": "function",
                        "function": {
                            "name": field("name", ""),
                            "arguments": json.dumps(
                                field("input", {}) or {},
                                ensure_ascii=False,
                            ),
                        },
                    })
            converted_message: dict = {
                "role": "assistant",
                "content": "\n".join(p for p in text_parts if p) or None,
            }
            if tool_calls:
                converted_message["tool_calls"] = tool_calls
            converted.append(converted_message)
            continue

        if role == "user" and isinstance(content, list):
            user_parts: list[str] = []
            multimodal_parts: list[dict] = []
            has_multimodal_input = False
            for item in content:
                if isinstance(item, dict) and item.get("type") == "tool_result":
                    tool_id = str(item.get("tool_use_id", ""))
                    result_text = str(item.get("content", ""))
                    if tool_id in pending_tool_ids:
                        converted.append({
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": result_text,
                        })
                        pending_tool_ids.discard(tool_id)
                    else:
                        user_parts.append(result_text)
                        multimodal_parts.append({"type": "text", "text": result_text})
                else:
                    converted_item = _convert_multimodal_item_for_openai(item)
                    if converted_item is not None:
                        multimodal_parts.append(converted_item)
                        if converted_item.get("type") != "text":
                            has_multimodal_input = True
                    user_parts.append(_content_to_text(item))
            if multimodal_parts:
                if has_multimodal_input:
                    converted.append({"role": "user", "content": multimodal_parts})
                else:
                    converted.append({"role": "user", "content": "\n".join(p for p in user_parts if p)})
            continue

        converted.append({"role": role, "content": _content_to_text(content)})

    return converted


def _convert_tools_for_openai(tools: list | None) -> list[dict] | None:
    if tools is None:
        return None
    converted = []
    for tool in tools:
        converted.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return converted


def _openai_response_to_anthropic_shape(response) -> SimpleNamespace:
    """把 OpenAI response 包装成 response.content=[text/tool_use] 的 Anthropic 兼容形状。"""
    if not hasattr(response, 'choices'):
        print(f"[llm] API 返回非预期类型: type={type(response).__name__} repr={repr(response)[:300]}")
        raise AttributeError(f"API response has no 'choices' attribute, got {type(response).__name__}: {str(response)[:200]}")
    if not response.choices:
        raise ValueError(
            f"API response has no choices; got {type(response).__name__}: {str(response)[:200]}"
        )
    message = response.choices[0].message
    blocks = []
    content = getattr(message, "content", None)
    if content:
        blocks.append(SimpleNamespace(type="text", text=content))

    tool_calls = getattr(message, "tool_calls", None) or []
    for tool_call in tool_calls:
        function = tool_call.function
        raw_args = function.arguments or "{}"
        try:
            args = json.loads(raw_args)
        except Exception:
            args = {}
        blocks.append(SimpleNamespace(
            type="tool_use",
            id=tool_call.id,
            name=function.name,
            input=args,
            raw_arguments=function.arguments,
        ))
    return _response_with_finish_reason(
        blocks, getattr(response.choices[0], "finish_reason", None)
    )


def _response_with_finish_reason(blocks, finish_reason):
    """Preserve provider termination metadata without inferring completeness."""
    stop_reason = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
    }.get(finish_reason, finish_reason)
    return SimpleNamespace(
        content=blocks,
        finish_reason=finish_reason,
        stop_reason=stop_reason,
    )


def _thinking_extra_body(base_url: str | None) -> dict:
    """按端点选 thinking 开关写法。
    - DeepSeek（base_url 含 deepseek）：官方开关是 {"thinking": {"type": "enabled"/"disabled"}}，
      不吃 budget_tokens / enable_thinking；强度只接受 reasoning_effort=high/max（此处不强加）。
    - 其它 OpenAI 兼容端点：沿用旧逻辑（thinking={type,budget_tokens} 开 / enable_thinking=False 关）。
    由环境变量 LLM_ENABLE_THINKING 决定开关。"""
    if os.getenv("LLM_THINKING_WIRE_FORMAT") == "chat_template_kwargs":
        return {"chat_template_kwargs": {"enable_thinking": _env_flag("LLM_ENABLE_THINKING")}}
    is_deepseek = "deepseek" in (base_url or "").lower()
    if _env_flag("LLM_ENABLE_THINKING"):
        if is_deepseek:
            return {"thinking": {"type": "enabled"}}
        budget = int(os.getenv("LLM_THINKING_BUDGET_TOKENS", "4096"))
        return {"thinking": {"type": "enabled", "budget_tokens": budget}}
    if is_deepseek:
        return {"thinking": {"type": "disabled"}}
    if "0-0.pro" in (base_url or "").lower():
        return {"enable_thinking": False, "thinking": {"type": "disabled"}}
    return {"enable_thinking": False}


def create_completion_no_retry(*, route, timeout, arguments):
    """One bounded completion; local API requests share cancellation semantics."""
    check_request_cancelled()
    arguments = prepare_body(arguments)
    record_attempt('completion_no_retry')
    if uses_request_local_transport(route.base_url):
        return call_local_model_until(base_url=route.base_url, api_key=route.api_key,
                                      deadline_ts=time.time() + timeout, arguments=arguments)
    client = _get_openai_client(route.base_url, route.api_key).with_options(
        max_retries=0, timeout=timeout)
    return client.chat.completions.create(**arguments)


def _create_openai_message(
    *,
    route: AnthropicRoute,
    system: str,
    messages: list,
    max_tokens: int,
    tools: list | None,
    model: str | None,
    timeout: int | float | None,
):
    check_request_cancelled()
    kwargs = {
        "model": model or route.model,
        "messages": _convert_messages_for_openai(system, messages),
        "max_tokens": max_tokens,
        "temperature": 0,
        # timeout 按请求传入而非固定在客户端上，缓存客户端才能被不同 timeout 的调用共享
        "timeout": timeout if timeout is not None else _DEFAULT_LLM_TIMEOUT,
    }
    kwargs["extra_body"] = _thinking_extra_body(route.base_url)
    openai_tools = _convert_tools_for_openai(tools)
    if openai_tools is not None:
        kwargs["tools"] = openai_tools
        kwargs["tool_choice"] = "auto"
    kwargs = prepare_body(kwargs)
    record_attempt('answer')
    if uses_request_local_transport(route.base_url):
        budget = kwargs.pop("timeout")
        response = call_local_model_until(base_url=route.base_url, api_key=route.api_key,
            deadline_ts=time.time() + budget, arguments=kwargs)
    else:
        client = _get_openai_client(route.base_url, route.api_key)
        response = client.chat.completions.create(**kwargs)
    return _openai_response_to_anthropic_shape(response)


def create_message_streaming(
    *,
    system: str,
    messages: list,
    max_tokens: int,
    tools: list | None = None,
    model: str | None = None,
    timeout: int | float | None = None,
    on_delta=None,
):
    """流式调用，但把流拼回与 create_message_with_fallback 完全相同的 response 结构。

    返回 (response, route_name, ttft)：
    - response：SimpleNamespace(content=[text/tool_use blocks])，与非流式同构，调用方逻辑无需改
    - ttft：首个**文本** token 的耗时（秒）；纯工具调用轮没有文本则为 None
    用法：每轮主循环都用它，第一个增量是 content→本轮是最终回答(ttft 有值)；是 tool_calls→本轮调工具(ttft=None)。
    on_delta(text)：可选回调，拿到每个文本增量（供 HTTP 端流式转发，本测量场景可不传）。
    仅走 OpenAI 兼容协议主路由，不做跨路由 fallback。
    """
    active = _active_routes()
    if not active:
        raise RuntimeError(
            "未配置可用的 LLM 路由：请检查 SILICONFLOW_* / HIGHSPEED_* / TOKEN_PLAN_* 环境变量"
        )
    route = active[0]
    check_request_cancelled()
    kwargs = {
        "model": model or route.model,
        "messages": _convert_messages_for_openai(system, messages),
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "extra_body": _thinking_extra_body(route.base_url),
        "timeout": timeout if timeout is not None else _DEFAULT_LLM_TIMEOUT,
    }
    openai_tools = _convert_tools_for_openai(tools)
    if openai_tools is not None:
        kwargs['tools'] = openai_tools
        kwargs['tool_choice'] = 'auto'
    kwargs = prepare_body(kwargs)
    record_attempt('streaming_answer')

    text_parts: list[str] = []
    tool_acc: dict[int, dict] = {}
    finish_reason = None
    ttft = None
    t0 = time.time()
    def consume_chunk(chunk):
        nonlocal ttft, finish_reason
        check_request_cancelled()
        if not chunk.choices:
            return
        chunk_finish = getattr(chunk.choices[0], "finish_reason", None)
        if chunk_finish is not None:
            finish_reason = chunk_finish
        delta = chunk.choices[0].delta
        text = getattr(delta, "content", None)
        if text:
            if ttft is None:
                ttft = time.time() - t0
            text_parts.append(text)
            if on_delta is not None:
                on_delta(text)
        for tc in (getattr(delta, "tool_calls", None) or []):
            acc = tool_acc.setdefault(tc.index, {"id": None, "name": None, "args": ""})
            if tc.id:
                acc["id"] = tc.id
            fn = getattr(tc, "function", None)
            if fn is not None:
                if fn.name:
                    acc["name"] = fn.name
                if fn.arguments:
                    acc["args"] += fn.arguments

    if uses_request_local_transport(route.base_url):
        budget = kwargs.pop("timeout")
        stream_local_model_until(base_url=route.base_url, api_key=route.api_key,
            deadline_ts=time.time() + budget, arguments=kwargs, on_chunk=consume_chunk)
    else:
        client = _get_openai_client(route.base_url, route.api_key)
        stream = client.chat.completions.create(**kwargs)
        try:
            for chunk in stream:
                consume_chunk(chunk)
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()

    check_request_cancelled()
    blocks = []
    content = "".join(text_parts)
    if content:
        blocks.append(SimpleNamespace(type="text", text=content))
    for idx in sorted(tool_acc):
        acc = tool_acc[idx]
        try:
            args = json.loads(acc["args"] or "{}")
        except Exception:
            args = {}
        blocks.append(SimpleNamespace(
            type="tool_use",
            id=acc["id"] or f"call_{idx}",
            name=acc["name"],
            input=args,
            raw_arguments=acc["args"],
        ))
    return _response_with_finish_reason(blocks, finish_reason), route.name, ttft


def create_message_with_fallback(
    *,
    system: str,
    messages: list,
    max_tokens: int,
    tools: list | None = None,
    model: str | None = None,
    timeout: int | float | None = None,
):
    """调用 Anthropic Messages API，默认走 highspeed，额度耗尽后自动切到 token plan。"""
    routes = _active_routes()
    last_exc: Exception | None = None

    for idx, route in enumerate(routes):
        route_sem = _get_route_semaphore(route.name)
        # 有限等待：上游卡死时不让请求在这里无限排队，转下一条路由或直接报错。
        acquired = acquire_request_slot(route_sem, _ROUTE_ACQUIRE_TIMEOUT_S)
        if not acquired:
            last_exc = RuntimeError(
                f"{route.name} 并发槽等待超时（{_ROUTE_ACQUIRE_TIMEOUT_S:g}s），放弃该路由"
            )
            has_next = idx < len(routes) - 1
            if has_next:
                print(f"[llm] {route.name} 并发槽等待超时，切换到 {routes[idx + 1].name}")
                continue
            raise last_exc
        try:
            if route.protocol == "openai":
                for attempt in range(_TRANSIENT_RETRY_ATTEMPTS):
                    try:
                        return _create_openai_message(
                            route=route,
                            system=system,
                            messages=messages,
                            max_tokens=max_tokens,
                            tools=tools,
                            model=model,
                            timeout=timeout,
                        ), route
                    except Exception as exc:
                        check_request_cancelled()
                        if attempt < _TRANSIENT_RETRY_ATTEMPTS - 1 and _should_retry_same_route(exc):
                            print(f"[llm] {route.name} 短暂连接异常，重试 {attempt + 1}/{_TRANSIENT_RETRY_ATTEMPTS}")
                            continue
                        raise

            import anthropic

            client = anthropic.Anthropic(
                base_url=route.base_url,
                api_key=route.api_key,
                **({'max_retries': 0} if os.getenv('MODEL_ATTEMPT_LEDGER') else {}),
            )
            kwargs = {
                "model": model or route.model,
                "max_tokens": max_tokens,
                "system": system,
                "messages": messages,
            }
            if tools is not None:
                kwargs["tools"] = tools
            kwargs["timeout"] = timeout if timeout is not None else _DEFAULT_LLM_TIMEOUT
            # 默认关 thinking：多数模型走非 reasoning 模式更快；要开 thinking 设 LLM_ENABLE_THINKING=1
            if not _env_flag("LLM_ENABLE_THINKING"):
                kwargs["extra_body"] = {"enable_thinking": False}
            kwargs = prepare_body(kwargs)

            for attempt in range(_TRANSIENT_RETRY_ATTEMPTS):
                try:
                    record_attempt('anthropic_answer')
                    return client.messages.create(**kwargs), route
                except Exception as exc:
                    check_request_cancelled()
                    if attempt < _TRANSIENT_RETRY_ATTEMPTS - 1 and _should_retry_same_route(exc):
                        print(f"[llm] {route.name} 短暂连接异常，重试 {attempt + 1}/{_TRANSIENT_RETRY_ATTEMPTS}")
                        continue
                    raise
        except Exception as exc:
            check_request_cancelled()
            last_exc = exc
            has_next = idx < len(routes) - 1
            if has_next and _should_fallback(exc):
                # 记录禁用时刻而不是永久拉黑：冷却结束后自动重新纳入候选。
                _disable_route(route.name)
                print(
                    f"[llm] {route.name} 配额不可用，切换到 {routes[idx + 1].name}"
                    f"（{route.name} 冷却 {_ROUTE_DISABLE_TTL_S:g}s 后自动恢复）"
                )
                continue
            raise
        finally:
            route_sem.release()

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("未配置可用的 LLM 路由")
