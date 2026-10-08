"""One same-route recovery of a truncated answer within the request deadline."""

import time

import llm_router
from request_cancellation import acquire_request_slot, check_request_cancelled
from response_integrity import IncompleteGenerationError, require_untruncated_response


def recover_text_once(*, route, system, messages, max_tokens, deadline_ts, model=None):
    """Regenerate from trusted history, never from the rejected partial output.

    Only the already selected OpenAI-compatible route is supported. A recovery
    has no tools, provider retries, or route fallback and reserves five seconds
    for subsequent application processing.
    """
    check_request_cancelled()
    if deadline_ts is None:
        raise IncompleteGenerationError("缺少请求截止时间，不能继续恢复截断回答。")
    remaining = deadline_ts - time.time() - 5
    if remaining < 5:
        raise IncompleteGenerationError("剩余时间不足，无法完整生成回答。")
    if isinstance(route, str):
        route = next((r for r in llm_router._active_routes() if r.name == route), None)
    if route is None or route.protocol != "openai":
        raise IncompleteGenerationError("当前模型路由不支持有界回答恢复。")
    from openai import APIError

    semaphore = llm_router._get_route_semaphore(route.name)
    if not acquire_request_slot(semaphore, min(1, remaining)):
        raise IncompleteGenerationError("模型繁忙，无法在剩余时间内恢复完整回答。")
    try:
        remaining = deadline_ts - time.time() - 5
        if remaining < 5:
            raise IncompleteGenerationError("剩余时间不足，无法完整生成回答。")
        history = list(messages) + [{
            "role": "user",
            "content": (
                "请现在只回答用户原始问题，不再调用工具。基于已经检索到的证据，"
                "重新生成一个完整、简洁的最终答案，保持问题的语言。"
                "只保留直接相关的步骤、判断条件、必要警告及其准确引用；"
                "不要翻译或复述无关手册段落，不要补写证据没有展示的操作。"
                "最多六个简短要点，目标为中文350字或英文180词以内。"
                "若证据不足以回答，应明确说明不能确认的具体内容。"
            ),
        }]
        raw = llm_router.create_completion_no_retry(route=route, timeout=min(30, remaining),
            arguments={'model': model or route.model,
                       'messages': llm_router._convert_messages_for_openai(system, history),
                       'max_tokens': min(max_tokens, 1024), 'temperature': 0,
                       'extra_body': llm_router._thinking_extra_body(route.base_url)})
        response = llm_router._openai_response_to_anthropic_shape(raw)
        require_untruncated_response(response)
        if (response.finish_reason != "stop"
                or any(getattr(b, "type", None) == "tool_use" for b in response.content)
                or not any(getattr(b, "text", "").strip() for b in response.content)):
            raise IncompleteGenerationError("恢复调用未返回完整文本回答。")
        if time.time() >= deadline_ts:
            raise IncompleteGenerationError("恢复回答超出请求截止时间。")
        return response
    except (APIError, OSError) as exc:
        raise IncompleteGenerationError("模型恢复调用失败，未能完整生成回答。") from exc
    finally:
        semaphore.release()
