"""Finite task-coverage exclusions; passing is not proof of task completion."""
import re


def scoped_emptying_fact(text):
    """Recognize a bounded terminal-step description, never removal instructions.

    This only exempts the task-mismatch exclusion. Citation support and answer
    completeness must still be checked independently.
    """
    # Parse context separately from the one allowed action. Optional placement
    # of the documented purpose or terminal-step label must not change scope.
    purpose = r'（用于(?:长期(?:停用|不使用)|非使用期(?:间)?|不使用期间)、防冻(?:保护)?或维修前）'
    context = (r'在(?:执行)?(?:系统排空|系统清空|清空系统|排空系统|(?:排空|清空)(?=模式))'
               r'(?:模式(?:下)?|程序|流程)(?:[（(]Emptying Mode[）)])?(?:' + purpose + r')?')
    scoped = re.fullmatch(
        context + r'(?:(?:流程|程序)?(?:的)?(?:(?:最后(?:步骤|一步)|步骤\s*6)(?:中)?)?'
        r'|程序及清理(?:步骤中|阶段)|的清理步骤中)(?:（步骤\s*6）)?(?:' + purpose + r')?[，,]\s*(?P<action>.+)',
        text.strip())
    if scoped:
        return bool(re.fullmatch(
            r'(?:(?:需|需要|要求)|最后一步是|步骤\s*6\s*要求)清空(?:并(?:清洁|清洗))?'
            r'(?:用过的|已用|废(?:弃)?)胶囊容器和(?:滴水盘|接水盘)'
            r'(?:[（(](?:drip tray|Empty and clean the used capsule container and drip tray)[）)])?[。.]?',
            scoped['action']))
    return bool(re.fullmatch(
        r'In (?:the )?system emptying mode, (?:the )?last step is to empty and clean '
        r'the used capsule container and drip tray[.]?', text.strip(), re.IGNORECASE))

_TRAY_REMOVAL = re.compile(
    r'(?:取出|拆下|移除)(?:并清空)?(?:意式咖啡机|咖啡机)?(?:的)?接水盘|'
    r'(?:remove|removing|take out|pull out) (?:and empty )?(?:the )?'
    r'(?:espresso machine(?:\x27s)? )?drip tray|'
    r'(?:the )?drip tray (?:is |can be )?(?:removed|pulled out)', re.IGNORECASE)


def task_completion_rejections(context, revision):
    if not revision.assessment or revision.assessment['answerability'] != 'complete':
        return []
    oily_smoke = r'油烟|非黑烟|(?:oily|non.dark|ordinary|white) smoke'
    oily_smoke_requested = (re.search(oily_smoke, context.question, re.IGNORECASE)
                            and re.search(r'处理|怎么办|\b(?:handle|deal with)\b|what (?:should|do).{0,40}\bdo\b',
                                          context.question, re.IGNORECASE))
    air_fryer_warning = any(
        sid.startswith('M') and quote.startswith('Air Fryer /')
        and re.search(r'Immediately unplug.*dark smoke', quote, re.IGNORECASE)
        for claim in revision.claims for sid, quote in claim.evidence)
    if oily_smoke_requested and air_fryer_warning:
        handling = r'应|需要|无需|不必|请|关闭|停止|拔|清洁|清洗|\b(?:should|must|stop|unplug|clean)\b|no action'
        covered = any(
            re.search(oily_smoke, sentence, re.IGNORECASE)
            and re.search(handling, sentence, re.IGNORECASE)
            and not re.search(r'未说明|未提供|无法确定|does not (?:describe|specify)|not specified|unknown',
                              sentence, re.IGNORECASE)
            for claim in revision.claims
            if any(sid.startswith('M') and quote.startswith('Air Fryer /')
                   for sid, quote in claim.evidence)
            for sentence in re.split(r'[。！？;；\n]|(?<=[.!?])\s+', claim.text))
        if not covered:
            return [{'reason': 'complete_answer_omits_oily_smoke_handling'}]
    start_requested = re.search(
        r'启动(?:这台|该)?洗衣机|让洗衣机开始洗涤|'
        r'\bstart (?:the |a |this )?(?:washing machine|washer)\b', context.question, re.IGNORECASE)
    washing_settings = any(
        re.search(r'WASH TIMER|WASH SELECTOR', claim.text, re.IGNORECASE)
        and any(sid.startswith('M') and quote.startswith('Washing Machine /')
                and '/ To Wash\n' in quote for sid, quote in claim.evidence)
        for claim in revision.claims)
    if start_requested and washing_settings:
        start_described = any(
            re.search(r'(?:按下|按|点击).{0,12}(?:START|启动|开始)|'
                      r'\b(?:press|push|tap)\b.{0,20}\bSTART\b|'
                      r'自动(?:启动|开始)|automatically (?:starts?|begins?)', sentence, re.IGNORECASE)
            and not re.search(r'未说明|未提供|无法确定|不要|不得|无需|不能|'
                              r'not (?:specified|described)|does not|do not|cannot|unknown',
                              sentence, re.IGNORECASE)
            for claim in revision.claims
            if any(sid.startswith('M') and quote.startswith('Washing Machine /')
                   and '/ To Wash\n' in quote for sid, quote in claim.evidence)
            for sentence in re.split(r'[。！？;；\n]|(?<=[.!?])\s+', claim.text))
        if not start_described:
            return [{'reason': 'complete_answer_omits_requested_washer_start'}]
    if not _TRAY_REMOVAL.search(context.question):
        return []
    for claim in revision.claims:
        for sentence in re.split(r'[。！？;；\n]|(?<=[.!?])\s+', claim.text):
            if (_TRAY_REMOVAL.search(sentence)
                    and not re.search(r'不能|无法|未说明|未提供|不要|不得|不应|'
                                      r'cannot|does not|do not|not described|not provided',
                                      sentence, re.IGNORECASE)):
                return []
    return [{'reason': 'complete_answer_omits_requested_tray_removal'}]
