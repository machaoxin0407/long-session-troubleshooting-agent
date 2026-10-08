"""A single evidence revision on the already selected route and deadline."""
import json
import os
import time
from dataclasses import replace

import llm_router
from grounding_contract import (
    REVISION_SYSTEM,
    GroundedRevision,
    GroundingContractError,
    parse_revision,
)
from request_cancellation import acquire_request_slot, check_request_cancelled
from response_integrity import IncompleteGenerationError, require_untruncated_response


def revise_with_evidence(*, question, draft, sources, route_name, deadline_ts):
    check_request_cancelled()
    if deadline_ts is None or deadline_ts - time.time() < 10:
        raise IncompleteGenerationError('剩余时间不足，无法完成证据核验。')
    if not sources:
        return GroundedRevision((), True)
    route = next((r for r in llm_router._active_routes() if r.name == route_name), None)
    if route is None:
        raise IncompleteGenerationError('无法确定原模型路由，证据核验未完成。')
    audited = os.getenv('GROUNDING_CLAIM_AUDIT', '0') == '1'
    selected = audited or os.getenv('GROUNDING_PASSAGE_SELECTION', '0') == '1'
    statements = None
    catalog, response_options, system = None, {}, REVISION_SYSTEM
    if selected:
        from grounding_selection import (
            SELECTION_SYSTEM,
            passage_catalog,
            selection_schema,
            source_condition_lines,
        )

        if route.protocol != 'openai':
            raise IncompleteGenerationError('当前路由不支持已配置的结构化证据选择。')
        try:
            catalog = passage_catalog(sources)
        except GroundingContractError as exc:
            raise IncompleteGenerationError('证据准备失败，未返回未经核验的回答。') from exc
        system = SELECTION_SYSTEM
        response_options['response_format'] = {'type': 'json_schema', 'json_schema': {
            'name': 'grounded_selection', 'strict': True, 'schema': selection_schema(catalog)}}
        if audited:
            from grounding_audit import AUDIT_SYSTEM, audit_schema, draft_statements

            try:
                statements = draft_statements(draft)
            except GroundingContractError as exc:
                raise IncompleteGenerationError('草稿无法逐句核验，未返回未经核验的回答。') from exc
            system = AUDIT_SYSTEM
            response_options['response_format']['json_schema'] = {
                'name': 'grounded_audit', 'strict': True, 'schema': audit_schema(statements, catalog)}
    messages = [{'role': 'user', 'content': json.dumps({
        # Rewriting never receives a draft as evidence. Optional audit mode gets
        # immutable candidate statements, explicitly separated from sources.
        'question': question,
        # Selected passages already contain the immutable source text. Sending
        # it twice enlarges the context without adding evidence. Source order
        # and IDs remain explicit; the original sources still validate quotes.
        'sources': [{'source_id': s.source_id, **({} if selected else {'text': s.text})}
                    for s in sources],
        **({'passages': catalog} if selected else {}),
        **({'source_condition_lines': source_condition_lines(catalog)} if selected and not audited else {}),
        **({'statements_to_audit': statements} if audited else {}),
    }, ensure_ascii=False)}]
    if selected and not audited:
        from grounding_examples import selection_example_messages

        # Examples are separate turns. The actual catalog and source validators
        # remain unchanged; example IDs can never become current evidence.
        messages = selection_example_messages() + messages
    semaphore = llm_router._get_route_semaphore(route.name)
    if not acquire_request_slot(semaphore, max(0.001, min(1, deadline_ts - time.time() - 5))):
        raise IncompleteGenerationError('模型繁忙，证据核验未完成。')
    try:
        remaining = deadline_ts - time.time() - 5
        if remaining < 5:
            raise IncompleteGenerationError('剩余时间不足，无法完成证据核验。')
        timeout = min(30, remaining)
        if route.protocol == 'openai':
            raw = llm_router.create_completion_no_retry(route=route, timeout=timeout,
                arguments={'model': route.model,
                           'messages': llm_router._convert_messages_for_openai(system, messages),
                           'max_tokens': 1800, 'temperature': 0,
                           'extra_body': llm_router._thinking_extra_body(route.base_url),
                           **response_options})
            response = llm_router._openai_response_to_anthropic_shape(raw)
            complete = response.finish_reason == 'stop'
        elif route.protocol == 'anthropic':
            import anthropic

            with anthropic.Anthropic(base_url=route.base_url, api_key=route.api_key,
                                     max_retries=0, timeout=timeout) as client:
                response = client.messages.create(model=route.model, system=REVISION_SYSTEM,
                                                  messages=messages, max_tokens=1800, temperature=0)
            complete = response.stop_reason == 'end_turn'
        else:
            raise GroundingContractError('Unsupported selected protocol')
        require_untruncated_response(response)
        if not complete or any(getattr(b, 'type', None) == 'tool_use' for b in response.content):
            raise GroundingContractError('Revision did not finish normally')
        raw_text = '\n'.join(getattr(b, 'text', '') for b in response.content)
        if audited:
            from grounding_audit import expand_audit

            raw_text = expand_audit(raw_text, statements, catalog, question=question)
        elif selected:
            from grounding_selection import expand_selection

            raw_text = expand_selection(raw_text, catalog)
        revision = parse_revision(raw_text, sources,
                                  require_assessment=True)
        if selected and not audited:
            from grounding_scope import scope_revision

            # Parse the original contract first. Scope filtering must not turn
            # malformed or contradictory output into a successful response.
            revision = scope_revision(question, revision)
        if selected:
            revision = replace(revision, review_limited=True)
        if time.time() >= deadline_ts - 1:
            raise IncompleteGenerationError('证据核验超过请求期限。')
        return revision
    except IncompleteGenerationError:
        raise
    except Exception as exc:
        # Do not expose provider error messages, endpoint configuration or draft.
        raise IncompleteGenerationError('证据核验失败，未返回未经核验的回答。') from exc
    finally:
        semaphore.release()
