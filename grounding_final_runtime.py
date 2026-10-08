"""Bounded local-only calls for the opt-in grounded final-tool agent path."""
import time
from urllib.parse import urlparse

import llm_router
from local_model_deadline import call_local_model_until
from request_cancellation import acquire_request_slot, check_request_cancelled
from response_integrity import IncompleteGenerationError


def call_final_tool_decision(*, system, messages, tools, deadline_ts, route_name=None):
    check_request_cancelled()
    routes = llm_router._active_routes()
    if len(routes) != 1:
        raise IncompleteGenerationError('最终答案候选路径要求单一本地模型路由。')
    route = routes[0]
    if (route.protocol != 'openai'
            or urlparse(route.base_url).hostname not in ('127.0.0.1', 'localhost', '::1')
            or (route_name is not None and route.name != route_name)):
        raise IncompleteGenerationError('最终答案候选路径禁止切换模型或使用外部端点。')
    if deadline_ts is None or deadline_ts - time.time() < 5:
        raise IncompleteGenerationError('最终答案候选路径剩余时间不足。')
    call_deadline = min(deadline_ts - 1, time.time() + 30)
    semaphore = llm_router._get_route_semaphore(route.name)
    if not acquire_request_slot(semaphore, max(0.001, min(1, call_deadline - time.time()))):
        raise IncompleteGenerationError('模型繁忙，未完成最终答案。')
    try:
        remaining = call_deadline - time.time()
        if remaining <= 0:
            raise IncompleteGenerationError('模型调用期限已过。')
        raw = call_local_model_until(
            base_url=route.base_url, api_key=route.api_key, deadline_ts=call_deadline,
            arguments={'model': route.model, 'max_tokens': 1800, 'temperature': 0,
                'messages': llm_router._convert_messages_for_openai(system, messages),
                'tools': llm_router._convert_tools_for_openai(tools),
                'tool_choice': 'required',
                'extra_body': llm_router._thinking_extra_body(route.base_url)})
        if time.time() >= call_deadline:
            raise IncompleteGenerationError('模型结果超过调用期限。')
        return llm_router._openai_response_to_anthropic_shape(raw), route
    except IncompleteGenerationError:
        raise
    except Exception as exc:
        raise IncompleteGenerationError('最终答案调用失败，未返回未经核验的回答。') from exc
    finally:
        semaphore.release()
