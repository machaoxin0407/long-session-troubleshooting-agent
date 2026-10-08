"""Finite checks for observed missing qualifiers; not general entailment."""
import re

from grounding_cycle_conditions import low_battery_during_cleaning


def _has(pattern, text):
    return bool(re.search(pattern, text, re.IGNORECASE))


def _separate_brush_cleaning(text, source):
    """A charger-only prerequisite does not qualify explicit brush/handle rinsing."""
    if not _has(r'brush heads? and (?:the )?handle can be cleaned by rinsing', source):
        return False
    power_lines = [line for line in re.split(r'[.!?\n]', source) if _has(r'\bunplug\b', line)]
    accessory_rule = (r'\bunplug (?:the )?(?:charger and travel case|USB wall adapter and chargers?) '
                      r'before (?:you clean|cleaning) them\s*$')
    if not power_lines or not all(_has(accessory_rule, line) for line in power_lines):
        return False
    return (_has(r'brush heads?|\bhandle\b|rubber seal|刷头|手柄|橡胶密封圈', text)
            and not _has(r'charg|travel case|adapter|power|mains|appliance|device|product|'
                         r'充电|旅行盒|适配器|电源|整机|设备|产品', text))


def _postfixed_drain_branches(text):
    """Recognize a complete positive pair, without borrowing across clauses."""
    pattern = (
        r'根据(?:设备|机器|机型)?是否有排水泵[，,]\s*将排水(?:软)?管'
        r'向下放置至(?:水槽孔|排水孔|排水口|地漏|下水口)'
        r'[（(]无排水泵(?:机型)?[）)]或放入水槽[/／或]浴缸'
        r'[（(]有排水泵(?:机型)?[）)]')
    return any(re.fullmatch(pattern, clause.strip())
               for clause in re.split(r'[;；。！？\n]', text))


def _video_fatty_smoke_support(sentence, quote):
    """Finite relation check on transcript lines, never visual co-occurrence."""
    fatty = r'fatty foods?|高脂肪(?:食物)?|脂肪(?:类)?食物|脂肪含量高的食物'
    smoke = r'smoke|冒烟|产生烟雾'
    negative = r'\b(?:not|never|cannot)\b|won.t|doesn.t|can.t|不会|不(?:会)?(?:导致|产生)|不能'
    relation = (rf'(?:{fatty})[^.!?。！？\n]{{0,100}}(?:{smoke})|'
                rf'(?:{smoke})[^.!?。！？\n]{{0,100}}(?:{fatty})')
    wanted = re.search(relation, sentence, re.IGNORECASE)
    if wanted is None:
        return False
    wanted_negative = _has(negative, wanted.group())
    for line in quote.splitlines():
        if not re.match(r'^ASR:\s*\S', line):
            continue
        for match in re.finditer(relation, line, re.IGNORECASE):
            if _has(negative, match.group()) == wanted_negative:
                return True
    return False


def qualifier_mismatch(question, claim):
    text = claim['text']
    source = '\n'.join(item['quote'] for item in claim['evidence'])
    # A documented completion tone is auditory. Restrict this exclusion to
    # explicit positive descriptions of that tone, not combined light/sound
    # observations or denials of visual support.
    if any(item['source_id'].startswith('M')
           and item['quote'].startswith('Vacuum /')
           and 'play a series of tones to indicate successful completion' in item['quote']
           for item in claim['evidence']):
        for sentence in re.split(r'[。！？;；\n]|(?<=[.!?])\s+', text):
            if _has(r'\bindicator\b|\blights?\b|\bLEDs?\b|指示灯|灯光', sentence):
                continue
            if _has(r'\btones?\b|提示音', sentence) and _has(
                    r'\b(?:is|are) (?:a |an )?(?:visible|visual)(?: and audible)? (?:sign|signal|indicator)\b|'
                    r'提示音(?:是|属于)(?:可见|视觉)(?:和听觉)?(?:迹象|信号)', sentence):
                return 'completion_tone_is_not_visual_signal'
    # The source binds all indicators lighting to holding CLEAN BEFORE release;
    # only the subsequent tone signifies success. Do not move the light state
    # into a post-reboot success condition.
    reboot_sequence = any(
        item['source_id'].startswith('M') and item['quote'].startswith('Vacuum /')
        and _has(r'until all indicators illuminate, then release', item['quote'])
        and _has(r'tone signifying a successful reboot', item['quote'])
        for item in claim['evidence'])
    if reboot_sequence:
        for sentence in re.split(r'[。！？;；\n]|(?<=[.!?])\s+', text):
            if _has(
                    r'^\s*(?:(?:when|after) (?:a |the )?reboot (?:is successful|succeeds|has succeeded)|'
                    r'after (?:a |the )?successful reboot),?\s+'
                    r'all indicators (?:will )?(?:illuminate|light up|be illuminated|be lit)|'
                    r'^\s*(?:重启成功(?:之)?后|当重启成功时)[，,]?\s*所有指示灯(?:将|会)?(?:亮起|点亮)',
                    sentence):
                return 'reboot_hold_indicators_are_not_success_state'
            # This source puts the hold before success, not after it. Keep
            # negated/cautionary descriptions outside this positive pattern.
            if _has(
                    r'^\s*after (?:a |the )?successful reboot,?\s*'
                    r'(?:press(?:ing)? and )?hold(?:ing)? (?:the )?CLEAN\b|'
                    r'^\s*重启成功(?:之)?后[，,]?\s*(?:再|需|需要|应)?(?:按住|长按)\s*CLEAN',
                    sentence):
                return 'reboot_action_occurs_before_success'
    error_signal = any(item['source_id'].startswith('M')
                       and item['quote'].startswith('Vacuum /')
                       and 'two-tone distress sound followed by a spoken message' in item['quote']
                       for item in claim['evidence'])
    if error_signal:
        for sentence in re.split(r'[。！？;；\n]|(?<=[.!?])\s+', text):
            if _has(r'does not establish|cannot (?:prove|confirm)|不能证明|无法确定|不代表', sentence):
                continue
            absent = _has(r'(?:absence of|\bno\b|没有|未出现).{0,70}'
                          r'(?:distress sound|troubleshooting indicator|报警声|故障灯)', sentence)
            inferred = _has(r'not reporting an error|no (?:active |current |ongoing )?(?:error|fault)\b|not in an? error state|'
                            r'not experiencing (?:an? )?(?:error|fault)|fault.free|'
                            r'没有报错|(?:没有|不存在)(?:当前|现有|正在发生的)?故障|未发生故障', sentence)
            if absent and inferred:
                return 'absence_of_error_signal_not_established'
    for sentence in re.split(r'[。！？;；\n]|(?<=[.!?])\s+', text):
        if (not _has(r'smoke|冒烟|烟雾', sentence)
                or not _has(r'\bnormal\b|正常', sentence)
                or _has(r'not normal|cannot.{0,20}normal|不代表.{0,12}正常|不能.{0,12}正常|并非正常', sentence)):
            continue
        first_use = _has(r'first (?:time|use)|首次|初次|第一次', sentence)
        fatty_food = _has(r'fatty foods?|高脂肪|脂肪(?:类)?食物|脂肪含量高的食物|'
                         r'脂肪含量高|油脂.{0,6}食物', sentence)
        for evidence in claim['evidence']:
            quote = evidence['quote']
            if not evidence['source_id'].startswith('M') or not _has(r'Air Fryer /', quote):
                continue
            # A first-use exception cannot normalize smoke under a different
            # condition, and each selected passage needs the relevant relation.
            if first_use and not _has(r'smoke.{0,100}first time.{0,30}(?:this is )?normal', quote):
                return 'first_use_smoke_normality_not_in_citation'
            if fatty_food and not _has(r'fatty food.{0,100}smoke.{0,30}(?:this is )?normal', quote):
                return 'fatty_food_smoke_not_established_normal'
    for sentence in re.split(r'[。！？;；\n]|(?<=[.!?])\s+', text):
        if (not _has(r'smoke|冒烟|烟雾', sentence)
                or _has(r'不能根据|无法判断|未说明|does not establish|cannot determine', sentence)):
            continue
        fatty = _has(r'fatty foods?|高脂肪|脂肪(?:类)?食物|脂肪含量高|高脂肪含量|油脂.{0,6}食物', sentence)
        doneness = (_has(r'熟透|未熟|undercooked|cook.{0,12}fully', sentence)
                    and _has(r'导致|避免|cause|prevent', sentence))
        for evidence in claim['evidence']:
            quote = evidence['quote']
            if (evidence['source_id'].startswith('V') and fatty
                    and not _video_fatty_smoke_support(sentence, quote)):
                return 'fatty_food_smoke_not_in_cited_passage'
            if not evidence['source_id'].startswith('M') or not _has(r'^Air Fryer /', quote):
                continue
            if fatty and not _has(r'fatty foods?[^.!?\n]{0,100}smoke|'
                                  r'smoke[^.!?\n]{0,100}fatty foods?', quote):
                return 'fatty_food_smoke_not_in_cited_passage'
            if doneness and not _has(r'undercooked|cook.{0,12}fully|熟透|未熟', quote):
                return 'food_doneness_smoke_cause_not_in_cited_passage'
    if _has(r'25\s*(?:秒|seconds?\b)', text) and _has(r'预热|加热|heating|heat.up', text):
        for evidence in claim['evidence']:
            quote = evidence['quote']
            if (evidence['source_id'].startswith('M')
                    and _has(r'^Espresso Machine /', quote)
                    and not _has(r'25\s*(?:秒|seconds?\b)', quote)):
                return 'heating_duration_not_in_cited_passage'
    # This observed safety warning excludes equality too. Do not silently
    # rewrite the claim: a mismatch must use the existing bounded repair.
    if _has(r"don['’]t use excessively hot water\.\s*\(50\s*°C or more\)", source):
        for sentence in re.split(r'[。！？;；\n]|(?<=[.!?])\s+', text):
            if not _has(r'water|水温|温水|热水|的水', sentence):
                continue
            if _has(r'(?:不得|不能|不可|不要|不应)(?:使用)?超过\s*50\s*(?:°\s*C|℃)|'
                    r'(?:不超过|不高于|最高|至多|≤|<=)\s*50\s*(?:°\s*C|℃)|'
                    r'(?:must not exceed|not exceed|at most|no higher than|up to)\s*50\s*(?:°\s*C|℃)|'
                    r'(?:do not|never) use water (?:above|over)\s*50\s*(?:°\s*C|℃)', sentence):
                return 'hot_water_limit_excludes_equality'
    if _has(r'bin is full.{0,80}complete its cleaning cycle by default', source):
        for sentence in re.split(r'[。！？;；\n]|(?<=[.!?])\s+', text):
            full_bin = _has(r'\bbin\b.{0,24}\bfull\b|\bfull (?:dust )?bin\b|'
                           r'(?:集尘盒|尘盒|尘桶).{0,8}满', sentence)
            completion = _has(r'(?:complet|finish)\w*.{0,25}(?:cleaning )?cycle|完成.{0,12}清洁周期', sentence)
            limited = _has(r'does not (?:mean|prove|imply)|cannot (?:prove|establish)|不代表|不能证明', sentence)
            default = _has(r'by default|default settings?|默认', sentence)
            if full_bin and completion and not limited and (
                    not default or _has(r'not by default|非默认|并非默认|不是默认|regardless of|无论', sentence)):
                return 'full_bin_completion_requires_default_setting'
    # An explicit no-pump branch still needs its direction even without a
    # "depending on model" preface. Destination and direction are distinct.
    if (_has(r'lay the drain hose down toward a sinkhole', source)
            and _has(r'in case no drain pump', source)):
        # Preserve direction when the explicit no-pump condition follows the
        # action instead of preceding it. Do not borrow a later branch's down.
        for suffix_branch in re.finditer(
                r'\b(?:lay|place|put) (?:the )?drain hose '
                r'(?P<direction>[^;.!?()\n]{0,70}?)\s*\(if no drain pump\)',
                text, re.IGNORECASE):
            direction = suffix_branch.group('direction')
            if (_has(r'\bsinkhole\b', direction)
                    and (not _has(r'\bdown\b', direction) or _has(r'\bup\b|not.{0,20}down', direction))):
                return 'ambiguous_drain_hose_branch'
        for suffix_branch in re.finditer(
                r'排水(?:软)?管'
                r'(?:[^;；。！？（）()\n]{0,70}[（(]有(?:排水)?泵'
                r'(?:时|机型|的机型|型号|的型号)?[）)]\s*或)?'
                r'(?P<direction>[^;；。！？（）()\n]{0,70}?)'
                r'[（(](?:无|没有)(?:排水)?泵(?:时|机型|的机型|型号|的型号)?[）)]', text):
            direction = suffix_branch.group('direction')
            if (_has(r'地漏|下水(?:口|道(?:口)?)|水槽孔|排水孔|排水口|水槽|\bsinkhole\b', direction)
                    and _has(r'朝向|放置|放入|置于|向上|向下|放低', direction)
                    and (not _has(r'向下(?!水(?:口|道))|放低', direction)
                         or _has(r'(?:不要|不得|不)[^;；。！？]{0,12}向下|向上|不要放低|不得放低', direction))):
                return 'ambiguous_drain_hose_branch'
        branches = re.finditer(
            r'(?:无(?:排水)?泵|没有(?:排水)?泵|without (?:a )?(?:drain )?pump)'
            r'[^;；。！？）)\n]*', text, re.IGNORECASE)
        for match in branches:
            branch = re.split(r'(?<![无没])有(?:排水)?泵|\bwith a drain pump', match.group(), flags=re.IGNORECASE)[0]
            if (_has(r'排水(?:软)?管|\b(?:drain )?hose\b', branch)
                    and _has(r'向下(?!水(?:口|道))|放低|\bdown\b', branch)
                    and _has(r'(?:至|入|到|于|朝向)水槽(?!孔|口|的(?:孔|排水))|'
                             r'\b(?:into|in|toward) (?:a |the )?(?:sink|bath)\b', branch)
                    and not _has(r'不要|不得|不能|不应|无法|not|cannot', branch)):
                return 'no_pump_sinkhole_not_pump_sink_destination'
            # In a parenthesis the hose object may precede the condition:
            # "排水管放置妥当（无排水泵时朝向水槽，有排水泵...）".
            # Bind that ellipsis only to the immediately preceding hose phrase.
            hose_ellipsis = _has(
                r'排水(?:软)?管[^。！？;；（）()\n]{0,30}[（(]$', text[:match.start()])
            if (hose_ellipsis and _has(r'朝向|向上|向下', branch)
                    and _has(r'水槽|排水孔|排水口|地漏|下水(?:口|道(?:口)?)', branch)
                    and (not _has(r'向下(?!水(?:口|道))|放低', branch)
                         or _has(r'(?:不要|不得|不)[^;；。！？]{0,12}向下|向上', branch))):
                return 'ambiguous_drain_hose_branch'
            if (_has(r'放置|放入|置于|放向|朝向|放低|向下|\b(?:lay|place|put)\b', branch)
                    and _has(r'排水管|排水软管|水槽孔|排水孔|排水口|地漏|下水(?:口|道(?:口)?)|sinkhole|\bhose\b', branch)
                    and (not _has(r'向下(?!水(?:口|道))|放低|\bdown\b', branch)
                         or _has(r'(?:不要|不得|不)[^;；。！？]{0,12}向下|向上|not.{0,30}\bdown\b|\bup\b', branch))):
                return 'ambiguous_drain_hose_branch'
    if (_has(r'drain hose', source) and _has(r'in case no drain pump', source)
            and _has(r'sink or bath', source)
            and _has(r'排水管|drain hose', text) and _has(r'水槽|浴缸|sink|bath', text)
            and _has(r'放入|放置|\b(?:place|put)\b', text)
            and _has(r'根据(?:设备|机器|机型)?是否.{0,6}排水泵|depending on whether.{0,20}pump', text)
            and not _postfixed_drain_branches(text)
            and not (_has(r'无排水泵时(?:将排水管)?放低(?:排水管)?(?:朝向)?(?:排水口|水槽孔|排水孔)|'
                          r'无排水泵时(?:将排水(?:软)?管)?向下朝向(?:排水口|水槽孔|排水孔)|'
                          r'无排水泵时(?:将排水管)?向下(?:放入|放置于|放置到|放置至)(?:水槽孔|排水孔)|'
                          r'无排水泵时(?:将排水(?:软)?管)?向下(?:放置至|放置到|放入|通向)(?:地漏|下水(?:口|道(?:口)?))|'
                          r'without (?:a )?(?:drain )?pump[, ]+(?:lay|place|put) (?:the )?(?:drain )?hose down (?:toward |into )?(?:a |the )?sinkhole', text)
                     and _has(r'(?<![没无否])有排水泵[^（(，,。；;]{0,25}(?:水槽|浴缸)|'
                              r'with a drain pump.{0,35}(?:sink|bath)', text))):
        return 'ambiguous_drain_hose_branch'
    if _has(r'close the water tap a little if the water pressure is too high', source):
        for clause in re.split(r'[。！？;；\n]|(?<=[.!?])\s+', text):
            if not _has(r'(?:关闭|关掉|关上|关小|稍关).{0,8}水龙头|(?:close|shut|turn off).{0,15}(?:water )?tap', clause):
                continue
            partial = _has(r'稍|一点|小一点|关小|(?<!不要)(?<!无需)(?:微调|略微)关闭水龙头|a little|slightly|partially', clause)
            conditional = _has(r'(?:如果|若|当).{0,8}水压.{0,4}(?:过高|太高)|水压.{0,4}(?:过高|太高)时|'
                               r'(?:if|when)\s+(?:the\s+)?(?:water\s+)?pressure\s+(?:is\s+)?(?:too high|excessive)', clause)
            if not partial or not conditional:
                return 'partial_tap_closure_condition_lost'
    cleaning = _has(r'\b(?:clean(?:ing)?|wipe|wiping|purge|purging)\b|清洁|擦拭|冲洗|排汽', text)
    missing_procedure = re.fullmatch(
        r'(?:the )?(?:supplied|available) (?:material|evidence|sources?) (?:does|do) not '
        r'(?:document|establish) (?:an? )?(?:[\w-]+ ){0,8}(?:procedure|sequence)\.?',
        text.strip(), re.IGNORECASE)
    if missing_procedure:
        cleaning = False  # A standalone evidence limit is not a cleaning directive.
    if (cleaning and _has(r'steam\s*wand|蒸汽(?:棒|管)', question)
            and _has(r'coffee\s*outlet|咖啡出口|咖啡出水口', text)):
        return 'coffee_outlet_cleaning_is_not_steam_wand_procedure'
    # A directive must carry this explicit prerequisite in the same claim.
    # Other claims are not guaranteed to survive filtering or citation binding.
    if (cleaning and _has(r'unplug.{0,40}before cleaning|清洁前.{0,12}(?:拔|断电)', source)
            and not _separate_brush_cleaning(text, source)
            and (_has(r'(?:不需要|不必|不用|无需|无须|不要|不得|禁止|切勿)[^。；;!?，,\n]{0,8}(?:拔|断电|断开电源)|'
                      r'\b(?:not|never|without)\s+(?:unplug\w*|disconnect\w*)', text)
                 or not _has(r'(?:unplug|disconnect.{0,20}(?:power|mains)).{0,60}before.{0,20}(?:clean|wip)|'
                         r'before.{0,20}(?:clean|wip).{0,60}(?:unplug|disconnect)|'
                         r'清洁[^。；;!?，,\n]{0,30}前.{0,20}(?:拔|断电|断开电源)|先.{0,12}(?:拔|断电|断开电源).{0,30}(?:再|然后).{0,12}(?:清洁|擦拭)', text))):
        return 'missing_unplug_before_cleaning_prerequisite'
    low_battery = r'low.{0,12}battery|battery.{0,12}low|电量不足|低电量|电量低'
    recharge = r'return\w*.{0,25}recharg|返回充电|回充'
    before_completion = r'before.{0,25}(?:finish|complet)|未完成|完成.{0,12}之前'
    missing_recharge_condition = any(
        _has(low_battery, sentence) and _has(recharge, sentence)
        and not _has(r'does not|will not|cannot|不代表|不会|不能', sentence)
        and not (_has(before_completion, sentence) or low_battery_during_cleaning(sentence))
        for sentence in re.split(r'[。！？;；\n]|(?<=[.!?])\s+', text))
    pulse = (_has(r'CLEAN\s*button|CLEAN\s*按钮|清洁按钮|清扫按钮', text)
             and _has(r'puls|闪烁|闪动|脉冲', text))
    if (_has(r'battery gets low before finishing a cleaning cycle', source)
            and (missing_recharge_condition or (pulse and not (
                _has(low_battery, text) and (
                    _has(before_completion, text) or low_battery_during_cleaning(text)))))):
        return 'missing_low_battery_before_completion_condition'
    if (_has(r'image simulated|模拟图|模拟画面', source)
            and _has(r'display|screen|显示屏|屏幕', text)
            and not _has(r'simulat|illustrative|模拟|示意', text)):
        return 'missing_simulated_image_qualification'
    # A correctly phrased unplugging prerequisite does not make instructions
    # for one component applicable to another. Keep an explicit insufficiency
    # statement available; it must not also prescribe the borrowed procedure.
    wand = r'steam\s*wand|蒸汽(?:棒|管|喷嘴)'
    borrowed_action = r'(?:before|after).{0,35}(?:clean|wip).{0,25}steam\s*wand|(?:clean|wipe)\s+(?:the\s+)?steam\s*wand|清洁蒸汽|擦拭蒸汽|蒸汽(?:棒|管|喷嘴).{0,15}(?:前|后).{0,15}(?:拔|擦|清洁)'
    if cleaning and _has(wand, question) and _has(borrowed_action, text):
        for item in claim['evidence']:
            if (item['source_id'].startswith('M')
                    and _has(r'coffee\s*outlet|咖啡出口|咖啡出水口', item['quote'])
                    and not _has(wand, item['quote'])):
                return 'coffee_outlet_source_does_not_establish_steam_wand_procedure'
    return None
