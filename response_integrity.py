"""Reject explicitly truncated generations before text or tools are consumed."""


class IncompleteGenerationError(RuntimeError):
    """The provider reports that its output stopped at the length limit."""


def require_untruncated_response(response):
    if (getattr(response, "stop_reason", None) == "max_tokens"
            or getattr(response, "finish_reason", None) == "length"):
        raise IncompleteGenerationError(
            "模型回答达到长度上限，未能完整生成，请缩小问题范围后重试。"
        )
    return response
