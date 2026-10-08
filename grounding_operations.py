"""Conservative bilingual operation mismatch exclusions, not entailment scores.

An explicit operation in an answer needs explicit support in every cited manual
excerpt. Another retrieved source cannot repair the chosen citation. Matching
an operation is necessary here, never sufficient: object, negation, conditions,
counts and model identity still require semantic review.
"""
import re

from grounding_readiness_context import readiness_context_mismatch


def _reversed_inner_cover_relation(sentence):
    pattern = (
        r'(?:衣物|衣服)(?:需要|必须|应当|应该|需|应|要)?(?:覆盖|盖住)(?:在)?内盖|'
        r'内盖(?:需要|必须|应当|应该|需|应|要)?(?:被|由)(?:衣物|衣服)(?:覆盖|盖住)|'
        r'\bcover (?:the )?inner cover with (?:the )?(?:laundry|clothes)\b|'
        r'\b(?:laundry|clothes) (?:must |should |needs? to )?cover (?:the )?inner cover\b|'
        r'\binner cover (?:must |should )?be covered (?:by|with) (?:the )?(?:laundry|clothes)\b')
    for match in re.finditer(pattern, sentence, re.IGNORECASE):
        prefix = sentence[:match.start()]
        if re.search(r'(?:不要|不得|不能|切勿)(?:让|使)?\s*$|\b(?:not|never)\s+$', prefix, re.IGNORECASE):
            continue
        return True
    return False

# Only unambiguous operations are covered. Do not use generic "return" (which
# includes returning to a dock), "replace" (which may mean a new component), or
# "install" to justify putting an already removed component back.
_REINSERT = re.compile(
    r'\bre[- ]?insert(?:s|ed|ing)?\b|\bre[- ]?install(?:s|ed|ing)?\b|'
    r'\b(?:put|putting|fit|fitting|place|placing)\b[^.!?\n]{0,60}\bback\b|'
    r'重新插入|重新安装|装回|插回|放回|放回原位', re.IGNORECASE)

# A possessive device prefix does not change the targeted component.
_BASKET_CN = r'(?:(?:空气炸锅|炸锅|设备|机器)的)?(?:食物|炸锅|炸)?篮(?:子|筐)?'
_FILTER_DOOR_CLOSURE = re.compile(
    r"filter door.{0,40}(?:won't|will not|cannot|can't).{0,15}close|"
    r'滤网门.{0,15}(?:无法|不能|不会|不可)关闭', re.IGNORECASE)


def mentions_reinsertion(text):
    return bool(_REINSERT.search(text))


def _pan_has_basket_gloss(text):
    """A parenthesized translation must not relabel the cited pan as a basket."""
    for match in re.finditer(r'(?<![a-zA-Z])pan\s*[（(]([^）)\n]{1,60})[）)]', text, re.IGNORECASE):
        gloss = match[1]
        if (re.search(r'(?:炸|食物|炸锅)?篮(?:子|筐)?|\bbasket\b', gloss, re.IGNORECASE)
                and not re.search(r'不是|并非|不指|\bnot\b', gloss, re.IGNORECASE)):
            return True
    return False


def _binds_bin_reinsertion(text):
    for clause in re.split(r'[。！？\n]|(?<=[.!?])\s+', text):
        if re.search(r'(?:装回|插回|放回)(?:吸尘器)?(?:集尘盒|尘盒)|reinsert\w*\s+(?:the\s+)?(?:dust\s+)?bin', clause, re.IGNORECASE):
            return True
        if (re.search(r'\bbin\b.{0,90}\breinsert\w*\s+it\b', clause, re.IGNORECASE)
                and not re.search(r'\bfilter\b', clause, re.IGNORECASE)):
            return True
    return False


def _bin_resume_relation(text):
    """Finite explicit relation, not co-occurrence across separate sentences."""
    for sentence in re.split(r'[。！？;；\n]|(?<=[.!?])\s+', text):
        if not mentions_reinsertion(sentence):
            continue
        if not re.search(r'\bbin\b|集尘盒|尘盒', sentence, re.IGNORECASE):
            continue
        if re.search(r'并不|不能|不代表|不会|does not|do not|cannot|will not', sentence, re.IGNORECASE):
            continue
        if (re.search(r'automatic|自动', sentence, re.IGNORECASE)
                and re.search(r'resum|continu|恢复|继续', sentence, re.IGNORECASE)):
            return True
    return False


def _asserts_timer_start(text):
    for sentence in re.split(r'[。！？\n]|(?<=[.!?])\s+', text):
        if re.search(r'不代表|不能|不会|未证明|没有.{0,12}证据|does not|do not|not to|cannot|will not', sentence, re.IGNORECASE):
            continue
        setting = re.search(r'WASH TIMER|(?:完成|设置|设定).{0,8}(?:设置|设定|完成)后|'
                            r'\b(?:setting|settings|timer)\b', sentence, re.IGNORECASE)
        effect = re.search(r'(?:即|就|会|将)?开始洗涤|启动洗涤(?:程序)?|洗涤程序(?:即|就|会|将)?启动|洗衣机启动|'
                           r'\b(?:starts?|begins?) (?:the )?wash', sentence, re.IGNORECASE)
        if setting and effect:
            return True
    return False


def action_relation_mismatch(claim):
    """Reject observed object/effect substitutions only in explicit source contexts.

    No relation is inferred from an unrelated passage containing the same words.
    These exclusions neither translate an answer nor prove the remaining claims.
    """
    text = claim['text']
    for item in claim['evidence']:
        if not item['source_id'].startswith('M'):
            continue
        quote = item['quote']
        if (quote.startswith('Vacuum /')
                and 'area of high debris concentration' in quote
                and 'you will see the indicator illuminate' in quote
                and not re.search(r'Dirt Detect', quote, re.IGNORECASE)):
            for sentence in re.split(r'[。！？;；\n]|(?<=[.!?])\s+', text):
                named_light = re.search(
                    r'Dirt Detect(?:™)?(?: indicator|指示灯)?[^.!?。！？;；\n]{0,35}'
                    r'(?:illuminat\w*|lights? up|亮起|点亮|发亮)', sentence, re.IGNORECASE)
                limited = re.search(
                    r'不能|无法|未说明|未指明|不代表|does not|cannot|not (?:named|identified)|'
                    r'unknown|unclear|whether', sentence, re.IGNORECASE)
                if named_light and not limited:
                    return 'unnamed_indicator_does_not_identify_dirt_detect'
        if (quote.startswith('Washing Machine /')
                and 'Do not put your hands into the spin dryer basket during spinning.' in quote
                and 'spinning laundry' in quote
                and not re.search(r'\bdrum\b|滚筒', quote, re.IGNORECASE)):
            # The warning names a basket and moving laundry. It does not
            # identify a drum or transfer observations from another model.
            for match in re.finditer(
                    r'滚筒(?:会|正在|在|持续)?(?:旋转|转动)|'
                    r'\b(?:the )?drum (?:is |will |can )?(?:spinning|rotating|spins|rotates)\b',
                    text, re.IGNORECASE):
                prefix = text[:match.start()]
                if not re.search(r'(?:不能|无法|不足以)(?:证明|表明|确认)\s*$|'
                                 r'(?:does not|cannot) (?:prove|show|establish|confirm)(?: that)?\s*$',
                                 prefix, re.IGNORECASE):
                    return 'spin_basket_warning_does_not_identify_drum_motion'
        if ('/ First Use or After a Long Period of Non-Use /' in quote
                and 'Press the Lungo button to rinse the machine. Repeat 3 times.' in quote
                and not re.search(r'coffee will.{0,30}flow|ready to (?:brew|extract)', quote, re.IGNORECASE)):
            for sentence in re.split(r'[。！？;；\n]|(?<=[.!?])\s+', text):
                brewing = re.search(r'可以(?:开始|进行)?(?:咖啡)?萃取|已准备好(?:进行)?萃取|ready to (?:brew|extract)',
                                    sentence, re.IGNORECASE)
                negative = re.search(r'不能|不代表|无法|does not|do not|cannot|not ready',
                                     sentence, re.IGNORECASE)
                if brewing and not negative:
                    return 'first_use_rinsing_readiness_not_brewing_evidence'
        readiness_reason = readiness_context_mismatch(text, quote)
        if readiness_reason:
            return readiness_reason
        # A full-bin recovery instruction does not establish a general
        # prerequisite for every subsequent use. Keep this finite exclusion
        # bound to the observed source and positive general-use wording.
        if ('bin is full' in quote.lower()
                and 'In this case, remove and empty the bin, then reinsert it' in quote):
            for sentence in re.split(r'[。！？;；\n]|(?<=[.!?])\s+', text):
                general_use = re.search(
                    r'在需要重新使用吸尘器前|(?:在)?清空(?:集尘盒|尘盒)后|'
                    r'(?:在)?(?:需要)?(?:重新)?(?:开始|恢复|继续)(?:或(?:重新)?(?:开始|恢复|继续))?清洁(?:循环|周期)(?:之)?前|'
                    r'\bbefore (?:starting|restarting|resuming|continuing)(?: or (?:starting|resuming|continuing))? '
                    r'(?:a |the )?cleaning cycle\b|'
                    r'before (?:using|you use) (?:the )?vacuum again|after emptying (?:the )?(?:dust )?bin',
                    sentence, re.IGNORECASE)
                positive = re.search(r'需|必须|应|\bmust\b|\bneed to\b', sentence, re.IGNORECASE)
                limitation = general_use and re.search(
                    r'不代表|不能|未说明|does not|cannot', sentence[:general_use.start()], re.IGNORECASE)
                # Merely naming a full bin inside an example does not bind
                # the whole reinsertion requirement to that condition.
                full_bin = re.search(
                    r'^\s*(?:如果|若|当)(?:集尘盒|尘盒)(?:已)?满(?:时)?[，,]|'
                    r'^\s*(?:if|when) (?:the |its )?(?:dust )?bin is full,',
                    sentence, re.IGNORECASE)
                if (general_use and positive and mentions_reinsertion(sentence)
                        and not limitation and not full_bin):
                    return 'full_bin_reinsertion_not_general_use_requirement'
        if (re.search(r'without the drip tray and drip grid', quote, re.IGNORECASE)
                and re.search(
                    r'(?:滴水盘|接水盘|托盘)\s*[（(]\s*drip grid\s*[）)]|'
                    r'(?<![A-Za-z])drip grid\s*[（(]\s*(?:滴水盘|接水盘|托盘)\s*[）)]',
                    text, re.IGNORECASE)):
            return 'drip_grid_gloss_conflicts_with_tray'
        if not _FILTER_DOOR_CLOSURE.search(quote):
            for sentence in re.split(r'[。！？;；\n]|(?<=[.!?])\s+', text):
                if (re.search(r'未说明|无法确认|不能确认|does not establish|not specified', sentence, re.IGNORECASE)
                        and not re.search(r'但是|然而|\bbut\b|\bhowever\b', sentence, re.IGNORECASE)):
                    continue
                if _FILTER_DOOR_CLOSURE.search(sentence):
                    return 'filter_door_condition_not_in_selected_quote'
        if (quote.startswith('Vacuum /')
                and 'press and hold CLEAN for 10 seconds until all indicators illuminate' in quote):
            for sentence in re.split(r'[。！？;；\n]|(?<=[.!?])\s+', text):
                if not re.search(r'reboot|重启', sentence, re.IGNORECASE):
                    continue
                hold = re.search(r'(?:按住|长按|hold(?:ing)?)\s*(?:the )?CLEAN', sentence, re.IGNORECASE)
                duration = re.search(r'(?<![\d.])(?:10|ten|十)\s*(?:秒|seconds?\b)', sentence, re.IGNORECASE)
                if hold and not duration:
                    return 'reboot_hold_requires_ten_seconds'
                if (hold and re.search(r'\btone\b|提示音', sentence, re.IGNORECASE)
                        and 'When you release the CLEAN button' in quote):
                    release = re.search(r'\breleas(?:e|ed|ing)\b|松开|释放\s*CLEAN\s*(?:键|按钮)', sentence, re.IGNORECASE)
                    negated = re.search(r'(?:not|never|without|before)\s+releas|(?:不要|不得|无需|不必|不)(?:松开|释放)|'
                                        r'(?:松开|释放)\s*CLEAN\s*(?:键|按钮)?(?:之)?前',
                                        sentence, re.IGNORECASE)
                    if not release or negated:
                        return 'reboot_tone_requires_button_release'
        if (quote.startswith('Washing Machine /')
                and 'cover with the inner cover on laundry before spinning' in quote.lower()):
            for sentence in re.split(r'[。！？;；\n]|(?<=[.!?])\s+', text):
                if not re.search(r'内盖|\binner cover\b', sentence, re.IGNORECASE):
                    continue
                if re.search(r'衣物.{0,12}(?:放入|装入|塞入)内盖|'
                             r'(?:put|place|insert) (?:the )?laundry (?:in|into) (?:the )?inner cover',
                             sentence, re.IGNORECASE):
                    return 'inner_cover_is_not_laundry_container'
                if re.search(r'未说明|无法确定|does not establish|not specified', sentence, re.IGNORECASE):
                    continue
                if _reversed_inner_cover_relation(sentence):
                    return 'inner_cover_relation_reversed'
                action = re.search(r'盖上|盖好|覆盖|\bclose\b|\bcover (?:the|with)\b', sentence, re.IGNORECASE)
                spinning = re.search(r'脱水前|甩干前|before spinning|before (?:the )?spin cycle',
                                     sentence, re.IGNORECASE)
                wrong_scope = re.search(r'洗涤前|洗衣前|before washing|脱水后|after spinning',
                                        sentence, re.IGNORECASE)
                if action and (not spinning or wrong_scope):
                    return 'inner_cover_requires_spinning_scope'
        if (re.search(r'without the drip tray and drip grid', quote, re.IGNORECASE)
                and re.search(r'without|没有|缺少|无', text, re.IGNORECASE)
                and re.search(r'use|使用', text, re.IGNORECASE)
                and re.search(r'drip tray|接水盘|滴水盘', text, re.IGNORECASE)
                and not re.search(r'drip grid|drip grate|格栅|滴水格|接水格', text, re.IGNORECASE)):
            return 'tray_safeguard_requires_distinct_grid'
        if (re.search(r'pressing CLEAN once', quote, re.IGNORECASE)
                and re.search(r'to start a cleaning cycle, press CLEAN again', quote, re.IGNORECASE)):
            for sentence in re.split(r'[。！？\n]|(?<=[.!?])\s+', text):
                starts = list(re.finditer(r'start.{0,15}cleaning|开始清洁|启动清洁', sentence, re.IGNORECASE))
                # Exempt only start assertions directly tied to pressing again.
                # A separate earlier first-press assertion must remain rejected.
                second_press = bool(starts) and all(re.search(
                    r'\b(?:press|pressing) CLEAN again (?:to )?$',
                    sentence[:match.start()], re.IGNORECASE) for match in starts)
                if (starts and re.search(r'CLEAN', sentence, re.IGNORECASE)
                        and not second_press
                        and not re.search(r'wake|waking|唤醒|does not|不能|并不', sentence, re.IGNORECASE)
                        and (re.search(r'beep|鸣叫|蜂鸣|亮起|illuminat', sentence, re.IGNORECASE)
                             or re.search(r'\bonce\b|按.{0,10}一次', sentence, re.IGNORECASE))):
                    return 'waking_feedback_is_not_cleaning_start'
        if (re.search(r'Set the WASH TIMER.{0,15}1-15', quote, re.IGNORECASE)
                and not re.search(r'timer.{0,40}(?:starts|begins) (?:the )?wash|wash\w* (?:starts|begins)', quote, re.IGNORECASE)
                and _asserts_timer_start(text)):
            return 'timer_setting_does_not_establish_start_trigger'
        if (re.search(r'Set the WATER SELECTOR KNOB to WASH', quote, re.IGNORECASE)
                and (re.search(r'WATER SELECTOR(?: KNOB)?\s*[（(]水位选择(?:旋钮|器)[）)]', text, re.IGNORECASE)
                     or re.search(r'WATER SELECTOR(?: KNOB)?[^;；,，。.!?\n]{0,60}'
                                  r'(?<!不)(?<!不能)(?<!不可)(?<!并非)(?<!不是)'
                                  r'(?:以|用于|用来)(?:调节|调整|控制|设置)水位', text, re.IGNORECASE)
                     or re.search(r'(?:^|[。！？;；\n])\s*设置水位选择\s*[:：]'
                                  r'[^。！？;；\n]{0,60}\bWATER SELECTOR(?: KNOB)?\b',
                                  text, re.IGNORECASE))
                and not re.search(r'WATER SELECTOR.{0,40}water level', quote, re.IGNORECASE)):
            return 'water_selector_does_not_establish_level_control'
        if (re.search(r'(?:装回|插回|放回)(?:吸尘器)?(?:集尘盒|尘盒)|reinsert\s+(?:the\s+)?(?:dust\s+)?bin', text, re.IGNORECASE)
                and re.search(r'\bfilter\b', quote, re.IGNORECASE)
                and mentions_reinsertion(quote)
                and not _binds_bin_reinsertion(quote)):
            return 'filter_reinsertion_does_not_support_bin_refitting'
        # The smoke warning names the pan, not its separately removable basket.
        # Restrict this exclusion to that explicit relation in the selected
        # quote; do not infer that every device uses separate components.
        if (re.search(r'smoke.{0,80}before.{0,40}pull the pan out', quote, re.IGNORECASE)
                and not re.search(r'\bbasket\b', quote, re.IGNORECASE)
                and (_pan_has_basket_gloss(text) or re.search(r'(?:取出|移除|拿出|拉出|抽出)(?:来)?' + _BASKET_CN + r'|'
                              r'(?:将|把)' + _BASKET_CN + r'[^。！？;；\n]{0,30}(?:取出|移除|拿出|拉出|抽出)|'
                              r'(?:remov(?:e|ing)|pull(?:ing)? out)\s+(?:the\s+)?(?:food\s+)?basket|'
                              r'(?:pull(?:ing)?|tak(?:e|ing))\s+(?:the\s+)?(?:food\s+)?basket\s+out', text, re.IGNORECASE))):
            return 'pan_smoke_warning_does_not_support_basket_removal'
        if (_bin_resume_relation(text)
                and re.search(r'after.{0,40}(?:battery|recharg).{0,80}automatic', quote, re.IGNORECASE)
                and mentions_reinsertion(quote) and not _bin_resume_relation(quote)):
            return 'bin_reinsertion_does_not_imply_automatic_resumption'
        filter_door = r"filter door.{0,35}(?:won't|will not|cannot).{0,15}close"
        bin_door = r"bin door.{0,35}(?:won't|will not|cannot).{0,15}close|(?:集尘盒|尘盒)门.{0,12}(?:无法|不能|不会)关闭"
        if (re.search(filter_door, quote, re.IGNORECASE)
                and re.search(bin_door, text, re.IGNORECASE)
                and not re.search(bin_door, quote, re.IGNORECASE)):
            return 'filter_door_condition_is_not_bin_door_condition'
        if (re.search(r'load laundry.{0,50}fill water.{0,30}water level', quote, re.IGNORECASE)
                and re.search(r'(?:装入衣物|装衣|装入衣服)\s*(?:到|至)|load laundry\s+(?:up\s+)?to\b', text, re.IGNORECASE)
                and re.search(r'水位|water level', text, re.IGNORECASE)):
            return 'water_fill_level_is_not_laundry_load_limit'
    return None


def operation_mismatch(claim):
    for evidence in claim['evidence']:
        quote = evidence['quote']
        if (not evidence['source_id'].startswith('M')
                or 'clean the surface of the appliance' not in quote.lower()
                or re.search(r'clean (?:the )?drip tray.{0,60}(?:damp cloth|mild cleaning)|'
                             r'(?:damp cloth|mild cleaning)[^.!?\n]{0,80}clean (?:the )?drip tray',
                             quote, re.IGNORECASE)):
            continue
        for sentence in re.split(r'[。！？;；\n]|(?<=[.!?])\s+', claim['text']):
            if re.search(r'未说明|不能确定|无法确认|does not establish|not specified', sentence, re.IGNORECASE):
                continue
            if (re.search(r'接水盘|drip tray', sentence, re.IGNORECASE)
                    and re.search(r'清洁|擦拭|\bclean\b', sentence, re.IGNORECASE)
                    and re.search(r'湿布|温和清洁剂|damp cloth|mild cleaning', sentence, re.IGNORECASE)):
                return 'surface_cleaning_does_not_establish_tray_method'
    relation = action_relation_mismatch(claim)
    if relation:
        return relation
    if not mentions_reinsertion(claim['text']):
        return None
    for evidence in claim['evidence']:
        if (evidence['source_id'].startswith('M')
                and not _REINSERT.search(evidence['quote'])):
            return 'reinsertion_not_in_selected_manual_quote'
    return None
