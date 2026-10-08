"""Finite cross-claim omissions grounded in the current catalog, not model notes."""
import re

from grounding_inference_completion import inference_completion_rejections
from grounding_procedure_cautions import (
    KEYBOARD_REASON,
    procedure_caution_rejections,
    switch_removal_source,
)
from grounding_supported_limits import (
    BATTERY_LIMIT_REASON,
    battery_service_limit_source,
    supported_limit_rejections,
)
from grounding_task_completion import task_completion_rejections


def _positive_lid_action(text):
    """Bind negation to the lid action, not a separate water-splash warning."""
    action = r'(?:关闭|盖好|盖上)(?:洗涤|洗衣)桶盖|close (?:the )?wash tub lid'
    positive = False
    for clause in re.split(r'[。！？;；,，\n]|(?<=[.!?])\s+', text):
        for match in re.finditer(action, clause, re.IGNORECASE):
            if re.search(r'不必|不用|不要|无需|不得|不应|do not|need not|never|don.t',
                         clause[:match.end()], re.IGNORECASE):
                return False
            positive = True
    return positive


def _pockets_under_preoperation_heading(text):
    """Keep a pre-operation heading's scope across a bounded precaution list.

    The caller must bind this to the washing manual and a washing procedure.
    A later temporal/conditional scope or sentence boundary cannot inherit it.
    """
    heading = re.match(r'^操作前注意事项\s*[：:]\s*', text.strip())
    if heading is None:
        return False
    body = text.strip()[heading.end():]
    preceding = ''
    for clause in re.split(r'[；;]', body):
        scoped = preceding + clause.strip().removesuffix('。').removesuffix('.')
        if re.search(r'[。！？\n]|(?:洗涤|操作|启动|清洗|洗完)(?:后|结束)|'
                     r'之后|以后|完成后|脱水前|维修前|清洁前|仅当|只有|如果|若', scoped):
            return False
        if re.fullmatch(r'\s*(?:请|需|需要|应)?(?:清空|排空)(?:所有)?口袋'
                        r'(?:[，,]\s*(?:取出|清除)钉子或别针|'
                        r'[，,]?\s*(?:以防|以免)钉子或别针损坏(?:洗衣机|机器)(?:或|和)衣物)?'
                        r'\s*[。.]?\s*', clause):
            return True
        preceding += clause + '；'
    return False


def omission_passage_ids(catalog, reason):
    """Locate exact source anchors for known omissions; never invent support."""
    if reason == BATTERY_LIMIT_REASON:
        return [pid for pid, passage in catalog.items()
                if passage['source_id'].startswith('M')
                and battery_service_limit_source(passage['quote'])]
    if reason == KEYBOARD_REASON:
        return [pid for pid, passage in catalog.items()
                if passage['source_id'].startswith('M') and switch_removal_source(passage['quote'])]
    anchors = {
        'washing_pocket_prerequisite_missing_from_answer': 'Before washing, empty all pockets.',
        'washing_lid_precaution_missing_from_answer': 'To avoid water splashes, close the wash tub lid.',
        'washing_water_detergent_step_missing': 'Fill water in the washtub and add the detergent.',
        'washing_laundry_level_step_missing': "Load laundry in the wash tub and fill water to 'H' high water level.",
        'washing_timer_step_missing': 'Set the WASH TIMER 1-15 minutes.',
        'washing_temperature_precaution_missing': "Don't use excessively hot water. (50°C or more)",
        'washing_pressure_precaution_missing': 'Close the water tap a little if the water pressure is too high.',
    }
    anchor = anchors.get(reason)
    if anchor is None:
        return []
    material = reason in ('washing_water_detergent_step_missing', 'washing_laundry_level_step_missing',
                          'washing_timer_step_missing')
    return [pid for pid, passage in catalog.items()
            if passage['source_id'].startswith('M')
            and passage['quote'].startswith('Washing Machine /')
            and (not material or '/ To Wash\n' in passage['quote'])
            and anchor.casefold() in passage['quote'].casefold()]


def _readiness_assertion_text(text):
    """Exclude only bounded, explicit insufficiency sentences, not whole answers."""
    limitation = (
        r'(?:现有|这些|所提供的)?(?:视频|证据)(?:未|没有)(?:明确)?描述'
        r'[“\"]?(?:就绪|ready)[”\"]?状态时控制面板的(?:具体)?显示内容'
        r'|(?:现有|这些|所提供的)?(?:视频|证据)(?:不能|不足以|无法)证明打印机(?:已)?就绪'
        r'|(?:the |these )?(?:videos|evidence) (?:do not|does not|cannot) '
        r'(?:establish|prove|confirm) that the printer is ready'
    )
    return '。'.join(part for part in re.split(r'[。.!！?？;；\n]', text)
                    if not re.fullmatch(limitation, part.strip(), re.IGNORECASE))


def _video_scope_rejections(context, revision):
    # Metadata restricts applicability; it never proves a video event or repair.
    cartridge_sources = {sid for sid, key, value in context.source_context
                         if key == 'topic' and re.search(r'\bcartridge\b|墨盒', value, re.IGNORECASE)}
    rejected = []
    for index, claim in enumerate(revision.claims):
        assertion_text = _readiness_assertion_text(claim.text)
        relevant = any(sid in cartridge_sources and sid.startswith('V')
                       and re.search(r'ASR:.*LED.*falha.*parou de piscar', quote, re.IGNORECASE)
                       for sid, quote in claim.evidence)
        # A cartridge-related title restricts the source's applicability; it
        # cannot rename the transcript's printer-fault indicator. Restrict this
        # check to that exact narration and a single cited source, so additional
        # evidence identifying an indicator is not rejected by an absence rule.
        if (relevant and len(claim.evidence) == 1
                and re.search(r'LED indicador de falha da impressora', claim.evidence[0][1], re.IGNORECASE)):
            specialized = r'墨盒故障(?:的)?(?:LED)?(?:指示)?灯|\bcartridge[- ](?:fault|error) (?:LED|indicator|light)\b'
            for match in re.finditer(specialized, claim.text, re.IGNORECASE):
                prefix = claim.text[:match.start()]
                if not re.search(r'(?:不能称为|并非|不是|not (?:a |the )?)\s*$', prefix, re.IGNORECASE):
                    rejected.append({'claim_index': index, 'reason': 'fault_indicator_identity_specialized'})
                    break
        if (relevant and re.search(r'就绪|准备好(?:进行)?打印|可打印状态|ready|pronta', assertion_text, re.IGNORECASE)
                and not (re.search(r'墨盒|\bcartridge\b', claim.text, re.IGNORECASE)
                         and re.search(r'视频|旁白|\bvideo\b|\bnarrat', claim.text, re.IGNORECASE))):
            rejected.append({'claim_index': index, 'reason': 'cartridge_video_scope_missing'})
        if (relevant and re.search(r'就绪|准备好(?:进行)?打印|可打印状态|ready|pronta', assertion_text, re.IGNORECASE)
                and any(sid.startswith('V') and not re.search(
                    r'ASR:.*LED.*falha.*parou de piscar', quote, re.IGNORECASE)
                        for sid, quote in claim.evidence)):
            rejected.append({'claim_index': index, 'reason': 'cartridge_readiness_not_in_cited_video'})
    return rejected


def answer_level_rejections(context, revision):
    rejected = (_video_scope_rejections(context, revision) + task_completion_rejections(context, revision)
                + procedure_caution_rejections(revision) + inference_completion_rejections(context, revision)
                + supported_limit_rejections(context, revision))
    # Check only a delivered washing procedure tied to the manual's To Wash
    # section. A state question, empty answer, or another manual cannot trigger it.
    procedure_question = re.search(
        r'(?:怎样|如何|怎么)(?:进行)?洗涤[？?]|启动洗衣机|使用洗衣机|洗涤(?:步骤|流程|程序)|'
        r'\bhow (?:to|do I|should I) wash\b|\bwashing procedure\b',
        context.question, re.IGNORECASE)
    operating = any(
        (re.search(r'WASH TIMER', claim.text, re.IGNORECASE)
         or (procedure_question and re.search(r'WASH SELECTOR', claim.text, re.IGNORECASE)))
        and re.search(r'设|调|旋|\bset|\bturn', claim.text, re.IGNORECASE)
        and any(sid.startswith('M') and quote.startswith('Washing Machine /')
                and '/ To Wash\n' in quote for sid, quote in claim.evidence)
        for claim in revision.claims)
    if not operating:
        return rejected
    # These presence checks complement clause-level boundary/condition checks.
    # Assessment notes are not delivered precautions or supporting citations.
    water_precautions = [
        ("Don't use excessively hot water. (50°C or more)",
         (r'水|\bwater\b', r'(?<![\d.])50\s*(?:°\s*C|℃|摄氏度|degrees?\s*Celsius)',
          r'不要|不得|禁止|避免|低于|不能|不应|不使用|勿使用|do not|don.t|never|avoid|below|less than'),
         'washing_temperature_precaution_missing'),
        ('Close the water tap a little if the water pressure is too high.',
         (r'水压|(?:water\s+)?pressure', r'水龙头|\btap\b',
          r'关闭|关小|调小|稍关|\bclose\b|\bshut\b|turn down'),
         'washing_pressure_precaution_missing'),
    ]
    for anchor, patterns, reason in water_precautions:
        applicable_sources = {s.source_id for s in context.sources
                              if s.source_id.startswith('M') and s.text.startswith('Washing Machine /')
                              and anchor in s.text}
        if not applicable_sources:
            continue
        retained = any(
            any(all(re.search(pattern, sentence, re.IGNORECASE) for pattern in patterns)
                for sentence in re.split(r'[。！？;；\n]|(?<=[.!?])\s+', claim.text))
            and not re.search(r'未说明|未提供|无法确定|does not (?:state|specify|provide)|not specified',
                              claim.text, re.IGNORECASE)
            and any(sid in applicable_sources and anchor in quote for sid, quote in claim.evidence)
            for claim in revision.claims)
        if not retained:
            rejected.append({'reason': reason})
    material_steps = [
        ('Set the WASH TIMER 1-15 minutes.',
         (r'(?:WASH TIMER|洗涤定时器).{0,12}(?:设|调)|'
          r'(?:设置|设定|调整)\s*(?:WASH TIMER|洗涤定时器)|\bset (?:the )?WASH TIMER'),
         r'(?<![\d.])1\s*(?:-|–|—|至|到|to)\s*15\s*(?:分钟|minutes?\b)',
         'washing_timer_step_missing'),
        ('Fill water in the washtub and add the detergent.',
         r'(?:加水|注水|fill water)', r'(?:加入|加|add).{0,8}(?:洗涤剂|detergent)',
         'washing_water_detergent_step_missing'),
        ("Load laundry in the wash tub and fill water to 'H' high water level.",
         r'(?:放入|装入).{0,8}衣物|(?:将|把)衣物(?:放入|装入)(?:洗涤|洗衣)桶|load laundry', r'(?:加水|注水|将水加至|水位|fill water).{0,20}H',
         'washing_laundry_level_step_missing'),
    ]
    for source_step, action, companion, reason in material_steps:
        applicable_sources = {s.source_id for s in context.sources
                              if s.source_id.startswith('M') and s.text.startswith('Washing Machine /')
                              and '/ To Wash\n' in s.text and source_step in s.text}
        if not applicable_sources:
            continue
        cited_steps = [claim for claim in revision.claims
                      if any(sid in applicable_sources and source_step in quote
                             for sid, quote in claim.evidence)]
        retained = all(any(
            re.search(pattern, claim.text, re.IGNORECASE)
            and not re.search(r'无需|不必|不用|不要|do not|need not', claim.text, re.IGNORECASE)
            for claim in cited_steps) for pattern in (action, companion))
        if not retained:
            rejected.append({'reason': reason})
    lid_sources = {s.source_id for s in context.sources if s.source_id.startswith('M')
                   and s.text.startswith('Washing Machine /')
                   and 'To avoid water splashes, close the wash tub lid.' in s.text}
    if lid_sources:
        lid_retained = any(
            _positive_lid_action(claim.text)
            and any(sid in lid_sources and 'To avoid water splashes, close the wash tub lid.' in quote
                    for sid, quote in claim.evidence)
            for claim in revision.claims)
        if not lid_retained:
            rejected.append({'reason': 'washing_lid_precaution_missing_from_answer'})
    applicable = [s for s in context.sources if s.source_id.startswith('M')
                  and s.text.startswith('Washing Machine /')
                  and re.search(r'Before washing, empty all pockets\.', s.text, re.IGNORECASE)]
    if not applicable:
        return rejected
    # The object is the whole pocket, not selected items inside it.
    pocket_action = r'(?:清空|排空)(?:所有|全部)?口袋(?![的中里内])'
    retained = any(
        (re.search(rf'操作前(?:需|需要|应)?{pocket_action}|'
                  rf'洗涤前.{{0,8}}{pocket_action}|洗前.{{0,8}}{pocket_action}|'
                  rf'启动洗衣机前[^。！？\n]{{0,240}}{pocket_action}|'
                  rf'开始洗涤前(?:(?!洗涤后|洗涤结束|洗完)[^。！？\n]){{0,240}}{pocket_action}|'
                  rf'{pocket_action}.{{0,8}}(?:再|然后)洗涤|'
                  r'before washing.{0,8}empty all pockets|empty all pockets before washing|'
                  r'before starting(?: washing)?\s*,\s*empty all pockets',
                  claim.text, re.IGNORECASE)
         or _pockets_under_preoperation_heading(claim.text))
        and not re.search(r'无需|不必|不用|do not|need not', claim.text, re.IGNORECASE)
        and any(sid in {s.source_id for s in applicable}
                and re.search(r'Before washing, empty all pockets\.', quote, re.IGNORECASE)
                for sid, quote in claim.evidence)
        for claim in revision.claims)
    if retained:
        return rejected
    return rejected + [{'reason': 'washing_pocket_prerequisite_missing_from_answer'}]
