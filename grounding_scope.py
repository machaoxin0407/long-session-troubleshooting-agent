"""Narrow, explicit negative-evidence checks; not a general entailment model."""
import re
from dataclasses import replace

from grounding_component_health import component_health_errors
from grounding_contract import GroundedClaim
from grounding_cycle_conditions import low_battery_during_cleaning
from grounding_modal_strength import unnecessary_as_prohibited
from grounding_model_scope import model_scope_mismatch
from grounding_observation_actions import instruction_view
from grounding_operations import mentions_reinsertion, operation_mismatch
from grounding_procedure_checkpoint import procedure_checkpoint_mismatch
from grounding_purchase_service import purchase_service_mismatch
from grounding_qualifiers import qualifier_mismatch
from grounding_source_relations import source_relation_mismatch
from grounding_task_completion import scoped_emptying_fact


def is_instruction(text):
    # CLEAN is also a printed button name. A state such as "the CLEAN button
    # pulses" is not an instruction to clean anything; "press CLEAN" still is.
    text = re.sub(r'\bCLEAN\s+button\b', 'control button', text, flags=re.IGNORECASE)
    text = instruction_view(text)
    if mentions_reinsertion(text):
        return True
    return bool(re.search(r'\b(?:press(?:es|ed|ing)?|hold(?:s|ing)?|held|turn(?:s|ed|ing)?|'
                          r'remov(?:e[sd]?|ing)|(?:re)?insert(?:s|ed|ing)?|set(?:s|ting)?|'
                          r'(?:dis)?connect(?:s|ed|ing)?|unplug(?:s|ged|ging)?|clean(?:s|ed|ing)?|'
                          r'wip(?:e[sd]?|ing)|purg(?:e[sd]?|ing))\b|'
                          r'''按\s*["'“”‘’]*(?:确认|启动|OK\b)|'''
                          r'按下|按住|按压|将|取出|移除|拆下|装回|重新插入|设置|设定|设为|调至|接通|连接|断开电源|切断电源|拔掉|拔下|清洁|擦拭|清空|打开', text, re.IGNORECASE))


def describes_operation(question, text):
    # Describing a visible interaction is useful for observation questions.
    # The same phrasing in a how-to answer must not bypass procedure checks.
    procedure = re.search(r'\bhow\b|怎样|如何|步骤', question, re.IGNORECASE)
    for sentence in re.split(r'[。！？;；\n]|(?<=[.!?])\s+', text):
        observation = re.search(
            r'^(?:A |The )?(?:video|finger|hand)\b|^视频(?:显示|中)|'
            r'^(?:A |The )?visual summary (?:shows|depicts|describes)\b|^视觉摘要(?:显示|描述)|'
            r'^(?:A |The )?simulated image (?:shows|depicts)\b|^模拟(?:图片|图像)(?:显示|描绘)',
            sentence.strip(), re.IGNORECASE)
        result = re.search(r'\bto (?:initiate|start|print|activate)|用于|以便|'
                           r'\bthen (?:press|hold|turn|remove)|然后(?:按|将)|接着(?:按|将)|'
                           r'(?:[,，]|\band\b)\s*(?:please\s+)?(?:press|hold|turn|remove|insert|set)\b',
                           sentence, re.IGNORECASE)
        if is_instruction(sentence) and (not observation or procedure or result):
            return True
    return False


def has_b_type_qualification(text):
    """Require the subtype on each selector operation, not elsewhere in a claim."""
    subtype = r'B\s*(?:TYPE|型|类)'
    control = r'WATER\s+SELECTOR|水流选择|水选择'
    clauses = re.split(r'[。！？;；\n]|(?<=[.!?])\s+', text)
    # A literal standalone source qualifier can attach to the preceding step.
    standalone = rf'\s*[（(]?\s*(?:FOR\s+|仅限|适用于)?{subtype}(?:\s+ONLY|机型)?\s*[）)]?\s*'
    # Explicit anaphora qualifies the immediately preceding step. A repeated
    # subtype instruction ("B models also set it") does not have this meaning.
    attached = (rf'(?:对于{ subtype }机型[，,]\s*此步骤适用|'
                rf'此步骤(?:仅)?适用于{ subtype }机型|'
                rf'This step applies (?:only )?to { subtype }(?: models)?)')
    for index, clause in enumerate(clauses):
        if not re.search(control, clause, re.IGNORECASE) or not is_instruction(clause):
            continue
        if re.search(rf'(?:非|不属于|not\s+|non[- ]?){subtype}|'
                     r'所有机型|任何机型|不论机型|无论机型|all\s+(?:models|types)|'
                     r'regardless\s+of\s+(?:model|type)', clause, re.IGNORECASE):
            return False
        if re.search(subtype, clause, re.IGNORECASE):
            continue
        if index + 1 < len(clauses) and re.fullmatch(standalone, clauses[index + 1], re.IGNORECASE):
            continue
        if index + 1 < len(clauses) and re.fullmatch(
                rf'\s*{attached}[.]?\s*', clauses[index + 1], re.IGNORECASE):
            continue
        return False
    return True


def has_pump_qualification(text):
    """Recognize explicit branch qualifiers in the mode-setting sentence only."""
    no_pump = r'(?:无|没有|不带)(?:排水泵|泵排水)|(?:without\s+(?:a\s+)?(?:drain\s+)?|no\s+(?:drain\s+)?)pump'
    model_suffix = r'(?:时|(?:的)?机型|(?:的)?型号)?'
    conditional_prefix = r'(?:if\s+|若|如果)?'
    # Printed control names are not the standalone WASH mode value.
    wash_only = r"WASH\b(?!\s*-?\s*RINSE|\s+(?:SELECTOR|TIMER)(?![a-z0-9_]))"
    # A no-pump mention must qualify WASH-RINSE, never the exception WASH.
    # Stop at another mode instead of borrowing a qualifier across alternatives.
    wrong_prefix = rf'(?:{no_pump})(?P<link>(?:(?!WASH|[。！？;；\n]).){{0,80}}){wash_only}'
    for match in re.finditer(wrong_prefix, text, re.IGNORECASE):
        link = match.group('link')
        if (re.search(r'设|为|选择|\b(?:use|select|set)\b', link, re.IGNORECASE)
                and not re.search(r'不要|不应|不得|do not|don.t|never', link, re.IGNORECASE)):
            return False
    if re.search(rf"{wash_only}['\"’”\s]*[（(]\s*{conditional_prefix}(?:{no_pump}){model_suffix}\s*[）)]", text, re.IGNORECASE):
        return False
    mode = r'WASH\s*-?\s*RINSE|洗涤.?漂洗'
    # A correct positive branch cannot excuse denying the no-pump mode.
    if re.search(rf'(?:{no_pump})[^。！？;；,，\n]{{0,30}}'
                 rf'(?:不要|不应|不得|do not|never)[^。！？;；,，\n]{{0,20}}'
                 rf'(?:{mode})', text, re.IGNORECASE):
        return False
    wash = r'''[ '"“”‘’]*WASH\b(?!\s*-?\s*RINSE)'''
    positive_pump = (
        r'(?:若|如|如果|在|对于)?(?<![没无不未配否])(?:带|有|配有)排水泵'
        r'(?:时|的情况下|的情况|(?:的)?(?:机型|机器|型号|设备)(?:时)?)?|'
        r'(?:若|如|如果)为(?:有)?排水泵(?:的)?(?:机型|型号|机器|设备)|'
        r'(?:若|如|如果)为有泵排水(?:的)?(?:机型|型号|设备)|'
        r'(?:若|如果)(?:设备|机器|机型)?配备排水泵')
    exception = (
        rf'(?:{positive_pump})\s*[,，]?\s*(?:则|应|需要|需)?'
        rf'(?:将\s*CYCLE SELECTOR(?: KNOB)?\s*(?:旋钮)?\s*)?'
        rf'(?:设为|设置为|设定为|设定至|设置至|调至|调为|为|用|使用|选择){wash}|'
        rf'(?:if|with|in case of)\s+(?:a\s+)?(?:drain\s+)?pump\s*[,，]?\s*'
        rf'(?:(?:is\s+)?(?:fitted|present)\s*[,，]?\s*)?'
        rf'(?:use|select|set\s+(?:it\s+)?to){wash}|'
        rf'if\s+(?:the\s+)?(?:machine|washer|model)\s+has\s+(?:a\s+)?drain\s+pump\s*,?\s*'
        rf'(?:use|select|set\s+(?:it\s+)?to){wash}|'
        rf'{wash}[\s\'"’”]*\s*(?:only\s+)?if\s+(?:the\s+)?(?:machine|washer)\s+has\s+(?:a\s+)?drain\s+pump')
    # Bind the negative condition to a mode-setting phrase or an immediately
    # following parenthesis, not any no-pump mention elsewhere in a sentence.
    setting = (
        r'设置\s*(?:旋钮|模式|CYCLE\s+SELECTOR(?:\s+KNOB)?)\s*(?:为|到)\s*|'
        r'(?:将\s*(?:旋钮|模式|CYCLE\s+SELECTOR(?:\s+KNOB)?)\s*)?'
        r'(?:设置为|设定为|设为|选择|使用|用|为)\s*|'
        r'(?:use|select)\s+|set\s+(?:(?:the\s+)?(?:selector|mode|knob)|it)\s+to\s+')
    qualified_mode = (
        rf'(?:{no_pump})(?:时|的机型)?\s*[,，]?\s*(?:{setting})[\s\'"‘’“”]*(?:{mode})|'
        rf'(?:{mode})[\s\'"‘’“”]*[（(]\s*{conditional_prefix}(?:{no_pump}){model_suffix}\s*[）)]')
    sentences = re.split(r'[。！？\n]|(?<=[.!?])\s+', text)
    for index, sentence in enumerate(sentences):
        if not re.search(mode, sentence, re.IGNORECASE):
            continue
        if re.search(qualified_mode, sentence, re.IGNORECASE):
            return True
        if re.search(exception, sentence, re.IGNORECASE):
            return True
        # A directly following, complete conditional sentence may qualify the
        # named cycle selector. Do not borrow conditions past another action.
        if (re.search(r'(?<![a-z0-9_])CYCLE SELECTOR(?: KNOB)?(?![a-z0-9_])', sentence, re.IGNORECASE)
                and index + 1 < len(sentences)
                and re.fullmatch(rf'\s*(?:{exception})[\s\'"’”]*(?:only)?[.!?]?\s*',
                                 sentences[index + 1], re.IGNORECASE)):
            return True
    return False


def _without_explicit_normality_denials(text):
    # Remove only the negated phrase, so a separate affirmative assertion in
    # the same clause remains visible to both health and simulation checks.
    return re.sub(
        r'\b(?:not a (?:general )?list of|(?:do|does) not (?:explicitly )?list the) '
        r'(?:visible )?(?:signs|indicators) that '
        r'(?:confirm normal (?:operation|functioning)|(?:a |the )?(?:vacuum|device|machine|appliance) '
        r'is (?:operating|working|functioning) normally)\b|'
        r'\b(?:cannot|can not|does not|do not) (?:confirm|verify|represent) (?:general )?normal (?:operation|functioning)\b|'
        r'\bnot (?:general )?normal (?:operation|functioning)\b|\bdistinct from normal (?:operation|functioning) signals\b|'
        r"\bnot a direct observation of (?:a |the )?(?:specific )?device(?:'s|’s) normal (?:operation|functioning)\b",
        'limited observation', text, flags=re.IGNORECASE)


def asserts_normality(text):
    """Recognize explicit health assertions, without letting a later denial erase one."""
    for clause in re.split(r'[。！？;；\n]|(?<=[.!?])\s+|\bbut\b|\bhowever\b|但是|但|然而', _without_explicit_normality_denials(text), flags=re.IGNORECASE):
        if not re.search(r'\bnormal (?:operation|functioning)\b|(?:operating|working|functioning|works) normally|'
                         r'\b(?:vacuum|device|machine|printer|appliance)\s+(?:is |appears |remains )?(?:in )?(?:a )?normal state\b|'
                         r'(?:整机|设备|机器|打印机|吸尘器)(?:处于|处在)?正常状态|'
                         r'(?:operat(?:e|es|ed|ing)|work(?:s|ed|ing)?|function(?:s|ed|ing)?) (?:normally|properly|correctly)|fully operational|fault[- ]free|'
                         r'not in (?:a )?fault state|正常(?:运行|工作|运转|运作)|(?:运行|工作|运转)正常|'
                         r'整机正常|无故障|没有故障', clause, re.IGNORECASE):
            continue
        if re.search(r'(?:cannot|can not|does not|do not) (?:establish|prove|show|indicate|mean)|'
                     r'not (?:fully operational|(?:operating|working|functioning) (?:normally|properly|correctly))|'
                     r'(?:does|do|did) not (?:operate|work|function) (?:normally|properly|correctly)|'
                     r'(?:not|no) (?:a )?(?:signs?|evidence|proof) of|'
                     r'(?:不能|无法|不足以|不代表|不意味着|并非).*(?:正常|故障)', clause, re.IGNORECASE):
            continue
        return True
    return False


def asserts_actual_operation(text):
    """Explicit real-device operation, including a later assertion after denial."""
    for clause in re.split(r'[。！？;；\n]|(?<=[.!?])\s+|\bbut\b|\bhowever\b|但是|但|然而', text, flags=re.IGNORECASE):
        match = re.search(r'\b(?:actual|real)\b.{0,60}\b(?:running|operat\w*|working|works|functions?)\b|'
                          r'(?:真实|实际).{0,30}(?:运行|运转|工作)', clause, re.IGNORECASE)
        if match is None:
            continue
        # A later caveat about another feature does not withdraw the assertion.
        prefix = re.split(r'[,，]', clause[:match.start()])[-1]
        if (re.search(r'(?:cannot|can not|does not|do not).{0,35}(?:establish|prove|show|indicate|confirm)|'
                      r'不能|无法|不代表|不足以|没有证明|未证明', prefix, re.IGNORECASE)
                or re.search(r'\bnot\s+(?:running|operating|working)|未运行|没有运行|并非', match.group(), re.IGNORECASE)):
            continue
        return True
    return False


def asserts_spin_cycle(text):
    """Separate a cycle assertion from an explicit denial of that inference."""
    for clause in re.split(r'[。！？;；\n]|(?<=[.!?])\s+|\bbut\b|\bhowever\b|但是|但|然而', text, flags=re.IGNORECASE):
        if not re.search(r'脱水|甩干|\bspin[- ]cycle\b', clause, re.IGNORECASE):
            continue
        # A complete absence-of-evidence clause is not a positive cycle claim.
        # Anchor the whole clause so an appended affirmative conclusion survives.
        if re.fullmatch(r'\s*(?:未|没有)提供(?:文本|标签|文本或标签|文字或标签)'
                        r'(?:以|来)?确认(?:当前|目前)?(?:为|是|处于)(?:脱水|甩干)'
                        r'(?:阶段|状态)?\s*', clause):
            continue
        if re.fullmatch(r'\s*(?:未|没有)(?:明确)?(?:(?:标识|标注|标明)为|显示(?:是|为))'
                        r'(?:脱水|甩干)(?:阶段|状态|模式)?\s*', clause):
            continue
        if re.fullmatch(r'\s*(?:未|没有)(?:明确)?区分(?:是)?'
                        r'(?:洗涤(?:还是|与|和|或)脱水|脱水(?:还是|与|和|或)洗涤|脱水)(?:阶段)?\s*', clause):
            continue
        if re.search(r'(?:不能|无法|不足以|不代表|不意味着|并非|不是).*?(?:脱水|甩干)|'
                     r'(?:cannot|can not|does not|do not).{0,60}(?:establish|prove|show|indicate|mean|confirm|tell|determine)|'
                     r'cannot be (?:confirmed|determined)|\bnot (?:in )?(?:a |the )?spin[- ]cycle\b',
                     clause, re.IGNORECASE):
            continue
        return True
    return False


def _only_qualified_silent_return(text, *, require_before=True):
    """Do not apply the completion-tone rule to the separate silent return branch."""
    tones = list(re.finditer(r'\btones?\b|提示音|声音', text, re.IGNORECASE))
    if not tones:
        return False
    if not (re.search(r'battery.{0,20}low|low.{0,20}battery|低电量|电量不足', text, re.IGNORECASE)
            and (not require_before or low_battery_during_cleaning(text)
                 or re.search(r'before (?:finishing|completing)|完成.*之前|未完成', text, re.IGNORECASE))
            and re.search(r'return.*recharge|回充|返回.*充电|when (?:it |the vacuum )?docks|when docking', text, re.IGNORECASE)):
        return False
    # Every tone mention must itself be negated. A later affirmative tone
    # assertion must still pass the completion condition check.
    return all(re.search(r'(?:will |does )?not (?:play|emit|sound) (?:a |any )?$|'
                         r'without (?:playing|emitting|sounding) (?:a |any )?$|'
                         r'(?:不会|不)(?:播放|发出|响起)?$',
                         text[max(0, match.start() - 45):match.start()], re.IGNORECASE)
               or (re.search(r'\bno $', text[max(0, match.start() - 45):match.start()], re.IGNORECASE)
                   and re.match(r'\s+(?:is|will be) (?:played|emitted|sounded)\b', text[match.end():], re.IGNORECASE))
               for match in tones)


def independent_claim_errors(claim):
    """Collect independent finite checks without short-circuiting repair feedback."""
    return list(dict.fromkeys(component_health_errors(claim) + [reason for check in (
        source_relation_mismatch, procedure_checkpoint_mismatch,
        model_scope_mismatch, purchase_service_mismatch)
        if (reason := check(claim))]))


def exclusion_reason(question, claim):
    component_errors = independent_claim_errors(claim)
    if component_errors:
        return component_errors[0]
    text = claim['text']
    evidence = claim['evidence']
    sources = '\n'.join(item['quote'] for item in evidence)
    instruction = is_instruction(text)
    # Distinguish documented observations from adding a control tutorial when
    # the user only asks how to recognize a state. A cited operation can still
    # be irrelevant to that request. Do not strip it and retain an unsafe fragment.
    vacuum_manual = any(e['source_id'].startswith('M') and e['quote'].startswith('Vacuum /')
                        for e in evidence)
    status_question = re.search(
        r'what (?:visible )?(?:signs|indicators)|how can I tell|how do I know|'
        r'what does.{0,40}(?:indicator|light).{0,20}(?:show|mean)|'
        r'(?:怎样|如何).{0,12}(?:判断|看出)|什么(?:可见)?(?:信号|迹象)|指示灯.*(?:表示|代表)',
        question, re.IGNORECASE)
    requested_operation = re.search(
        r'how (?:do I|should I|to) (?:reboot|wake|start|pause|resume|end)|'
        r'(?:怎样|如何)(?:操作|重启|唤醒|启动|暂停|恢复|结束)', question, re.IGNORECASE)
    if vacuum_manual and status_question and not requested_operation:
        for sentence in re.split(r'[。！？;；\n]|(?<=[.!?])\s+', text):
            if re.search(
                    r'^\s*To (?:manually )?(?:reboot|wake|start|pause|resume|end)\b'
                    r'[^.!?。\n]{0,100}\b(?:press(?:es|ed|ing)?|hold(?:s|ing)?|held)\b|'
                    r'^\s*(?:When|While) (?:manually )?(?:rebooting|waking|starting|pausing|resuming|ending)'
                    r'[^,.!?。\n]{0,80},\s*(?:press|hold)\b|'
                    r'^\s*(?:Press|Hold)\b.{0,40}\bCLEAN\b|'
                    r'^\s*(?:Pressing(?: and holding)?|Holding) (?:the )?CLEAN(?: button)?'
                    r'(?: (?:once|again)| until (?:the )?indicators turn off)? '
                    r'(?:starts|pauses|resumes|ends) (?:a|the) cleaning cycle\b|'
                    r'^\s*(?:Pressing(?: and holding)?|Holding) (?:the )?CLEAN(?: button)?'
                    r'(?: until (?:the )?indicators turn off)? '
                    r'(?:puts|places|sends) (?:the )?vacuum (?:in|into) standby(?: mode)?\b|'
                    r'\bafter (?:pressing and )?holding (?:it|(?:the )?CLEAN(?: button)?) '
                    r'for \d+ seconds until all indicators illuminate\b|'
                    r'^\s*(?:若要|要|为了)(?:重启|唤醒|启动|开始|暂停|恢复|结束)'
                    r'.{0,80}(?:按下|按住|按压)', sentence, re.IGNORECASE):
                return 'vacuum_control_tutorial_does_not_answer_state_question'
    post_smoke_cleaning = (
        re.search(r'(?:冒烟|烟雾)(?:已经|已)?停止后|after (?:the )?smoke has stopped', question, re.IGNORECASE)
        and re.search(r'(?:怎样|如何|怎么)(?:安全地)?(?:清洁|清洗)|'
                      r'how (?:do I|should I|to) (?:safely )?clean', question, re.IGNORECASE)
        and not re.search(r'处理(?:冒烟|油烟|烟雾)|(?:冒烟|油烟|烟雾).{0,10}(?:怎么|如何)处理|'
                          r'(?:handle|deal with) (?:the )?smoke', question, re.IGNORECASE))
    smoke_response = (re.search(r'冒烟|油烟|烟雾|\bsmok(?:e|ing)\b', question, re.IGNORECASE)
                      and re.search(r'怎样|如何|怎么|处理|\bhow\b|what (?:should|do)', question, re.IGNORECASE)
                      and not post_smoke_cleaning)
    cleaning_action = instruction or re.search(
        r'倒掉|清除|冷却|清洗|\b(?:dispose|remove|cool|wash|brush|wipe)\b', text, re.IGNORECASE)
    if smoke_response and cleaning_action:
        for item in evidence:
            quote = item['quote']
            if (item['source_id'].startswith('M')
                    and quote.startswith('Air Fryer / Cleaning and Storage / Cleaning /')
                    and not re.search(r'\bsmok(?:e|ing)\b|冒烟|油烟|烟雾',
                                      quote.partition('\n')[2], re.IGNORECASE)):
                return 'routine_cleaning_not_smoke_response'
    # A manual setup statement is not a transcript of timer behavior or drum
    # motion. Only reject these explicit added observations in setup excerpts.
    for item in evidence:
        if (not item['source_id'].startswith('M')
                or not re.search(r'SPIN\s+DRY\s+TIMER', item['quote'], re.IGNORECASE)):
            continue
        for clause in re.split(r'[。！？;；\n]|(?<=[.!?])\s+', text):
            if re.search(r'不能|无法|未说明|不代表|cannot|does not|not established', clause, re.IGNORECASE):
                continue
            for pattern, reason in (
                (r'倒计时|count(?:s|ing)?\s*down|countdown', 'timer_setting_does_not_establish_countdown_observation'),
                (r'高速旋转|high[- ]speed\s+(?:spin|rotat)|spin\w*\s+rapidly', 'timer_setting_does_not_establish_high_speed_observation'),
            ):
                if (re.search(pattern, clause, re.IGNORECASE)
                        and not re.search(pattern, item['quote'], re.IGNORECASE)):
                    return reason
    # Motion/ordinary washing in a cited video is not evidence of a named
    # spin cycle. Check each cited video separately, not pooled topic words.
    if asserts_spin_cycle(text):
        for item in evidence:
            if (item['source_id'].startswith('V')
                    and re.search(r'tumbl|rotat|spinning|in motion|wash cycle|翻滚|转动|旋转|洗涤',
                                  item['quote'], re.IGNORECASE)
                    and not asserts_spin_cycle(item['quote'])):
                return 'motion_does_not_establish_spin_cycle'
    if (re.search(r'battery gets low before finishing a cleaning cycle', sources, re.IGNORECASE)
            and _only_qualified_silent_return(text, require_before=False)
            and not _only_qualified_silent_return(text)):
        return 'missing_low_battery_before_completion_condition'
    if (re.search(r'If .*returning to recharge after completing a cleaning cycle', sources, re.IGNORECASE)
            and re.search(r'\btones?\b|提示音|声音', text, re.IGNORECASE)
            and not _only_qualified_silent_return(text)
            and (not re.search(r'return.*recharge|回充|返回.*充电', text, re.IGNORECASE)
                 or not re.search(r'after (?:successfully )?(?:finishing|completing)|完成[^。！？;；]{0,20}(?:后|之后)|'
                                  r'清洁周期结束(?:后|之后)?返回充电|'
                                  r'\b(?:completes|finishes) (?:a |the )?cleaning cycle and returns to recharge\b', text, re.IGNORECASE)
                 or re.search(r'before (?:finishing|completing)|完成.*之前|未完成|未结束|尚未结束|结束(?:之)?前', text, re.IGNORECASE))):
        return 'completion_tone_requires_return_to_recharge_condition'
    # The source explicitly disclaims an actual observed state.
    if (evidence and all(re.search(r'image simulated|模拟图|模拟画面|模拟示意', item['quote'], re.IGNORECASE)
                         for item in evidence) and asserts_actual_operation(text)):
        return 'simulated_source_cannot_establish_actual_operation'
    if (re.search(r'simulated|illustrative|模拟|示意', sources, re.IGNORECASE)
            and re.search(r'normal(?:ly)?|正常', _without_explicit_normality_denials(text), re.IGNORECASE)
            and not re.search(r'(?:cannot|does not) (?:establish|prove|show|indicate).*normal|'
                              r'(?:不能|无法|不足以|不代表).*正常', text, re.IGNORECASE)):
        return 'simulated_source_cannot_establish_normality'
    # Local component behavior and disassembly descriptions are not a whole-
    # device health test. This is a narrow exclusion, not general entailment.
    if (asserts_normality(text)
            and re.search(r'partially disassembled|internal components|colorful wires|'
                          r'CLEAN button|battery indicator|troubleshooting indicator|distress sound|indicator lights?|\bLEDs?\b|screen|display|'
                          r'指示灯|显示屏|部分拆解|内部零件', sources, re.IGNORECASE)):
        return 'component_observation_cannot_establish_device_health'
    if (asserts_normality(text)
            and re.search(r'safety instructions|general safety|\bCAUTION\b|\bWARNING\b|安全说明|安全须知|安全警告',
                          sources, re.IGNORECASE)
            and not asserts_normality(sources)):
        return 'safety_instructions_do_not_establish_device_health'
    if (re.search(r'可安全|安全地|safe to|safely', text, re.IGNORECASE)
            and re.search(r'steady lights|blinking lights', sources, re.IGNORECASE)
            and not re.search(r'safe to|safely', sources, re.IGNORECASE)):
        return 'readiness_indicator_is_not_safety_guarantee'
    if (re.search(r'脱水|甩干|spin', question, re.IGNORECASE) and re.search(r'判断|正在|current|tell', question, re.IGNORECASE)
            and re.search(r'SPIN\s+DRY\s+TIMER|脱水定时器|甩干定时器|'
                          r'(?:设置|设定|设为)\s*(?:脱水|甩干)时间', text, re.IGNORECASE)
            and re.search(r'设置|设定|设为|\bset\b', text, re.IGNORECASE)):
        return 'timer_setup_is_not_current_cycle_observation'
    if (re.search(r'脱水|甩干|spin', question, re.IGNORECASE) and re.search(r'判断|正在|current|tell', question, re.IGNORECASE)
            and re.search(r'打开|open', text, re.IGNORECASE) and re.search(r'盖|lid', text, re.IGNORECASE)
            and re.search(r'停止|stop|brak', text, re.IGNORECASE)):
        return 'lid_interlock_is_not_current_cycle_observation'
    # A topic match at the end of a different procedure is not a task match.
    if (re.search(r'接水盘|滴水盘|drip\s*tray', question, re.IGNORECASE)
            and re.search(r'取出|拆|怎样|如何|how|remove', question, re.IGNORECASE)
            and re.search(r'Emptying the System|emptying mode', sources, re.IGNORECASE)
            and not scoped_emptying_fact(text)):
        return 'whole_system_emptying_is_not_tray_removal'
    if (describes_operation(question, text) and evidence and all(item['source_id'].startswith('V') for item in evidence)
            and not re.search(r'(?m)^ASR:\s*\S', sources)):
        return 'video_summary_or_ocr_is_not_instruction_transcript'
    # Explicit subtype prerequisites must survive into operating instructions.
    if (instruction and re.search(r'WATER\s+SELECTOR|水流选择|水选择', text, re.IGNORECASE)
            and re.search(r'FOR\s+B\s+TYPE', sources, re.IGNORECASE)
            and not has_b_type_qualification(text)):
        return 'missing_b_type_qualification'
    if (instruction and re.search(r'WASH\s*-?\s*RINSE|洗涤.?漂洗', text, re.IGNORECASE)
            and re.search(r'in case drain pump', sources, re.IGNORECASE)
            and not has_pump_qualification(text)):
        return 'missing_drain_pump_qualification'
    # Colour-preserving translations are valid; smoke density (浓烟) is not
    # equivalent to dark smoke. Keep the broader alternative exclusion.
    dark_smoke = r'(?:黑烟|(?:深色|黑色)烟雾|dark smoke)'
    broad_smoke = r'(?:油烟|浓烟|烟雾|smoke)'
    smoke_alternative = rf'(?:{broad_smoke}\s*(?:或|或者|or)\s*{dark_smoke}|{dark_smoke}\s*(?:或|或者|or)\s*{broad_smoke})'
    if (re.search(dark_smoke, sources, re.IGNORECASE)
            and re.search(r'拔|断开电源|切断电源|unplug|disconnect', text, re.IGNORECASE)
            and (not re.search(dark_smoke, text, re.IGNORECASE)
                 or re.search(smoke_alternative, text, re.IGNORECASE))):
        return 'smoke_condition_broadened'
    return (unnecessary_as_prohibited(claim)
            or qualifier_mismatch(question, claim) or operation_mismatch(claim))


def filter_scoped_claims(question, claims):
    kept, rejected = [], []
    for index, claim in enumerate(claims):
        text = claim['text']
        if re.search(r'[（(]\s*[EMV]\d+\s*[:：]\s*$', text):
            reason = 'unfinished_citation_scaffold'
        else:
            reason = exclusion_reason(question, claim)
        if reason:
            rejected.append({'claim_index': index, 'reason': reason})
        else:
            kept.append(claim)
    # Removing an operating prerequisite must not leave a fragmentary procedure.
    if any(is_instruction(claims[row['claim_index']]['text']) for row in rejected):
        for claim in kept:
            rejected.append({'claim_index': claims.index(claim), 'reason': 'procedure_requires_revalidation'})
        # Even declarative clauses can contain implied operating advice. The
        # surviving set has not been revalidated as a coherent procedure.
        kept = []
    return kept, rejected


def scope_revision(question, revision):
    """Apply the shared exclusions only after strict contract parsing."""
    scoped, rejected = filter_scoped_claims(question, [
        {'text': claim.text, 'evidence': [{'source_id': sid, 'quote': quote}
                                        for sid, quote in claim.evidence]}
        for claim in revision.claims])
    if rejected:
        assessment = dict(revision.assessment)
        assessment.update(answerability='partial' if scoped else 'none',
            reason='Scope checks removed statements: ' + ', '.join(
                sorted({row['reason'] for row in rejected})))
        revision = replace(revision, insufficient=True, assessment=assessment,
            claims=tuple(GroundedClaim(claim['text'], tuple(
                (ref['source_id'], ref['quote']) for ref in claim['evidence'])) for claim in scoped))
    return replace(revision, review_limited=True)
