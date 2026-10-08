"""Finite exclusion: a visual procedure checkpoint is not a success signal."""
import re

_CLAUSE = re.compile(r'[。！？;；，,.!\n]|\bbut\b|\bthen\b|但是|然后', re.IGNORECASE)
_LIGHT = re.compile(
    r'\ball (?:the )?(?:indicator(?:s| lights)?|lights|LEDs) '
    r'(?:are |have |will |will be )?(?:illuminat(?:e|es|ed|ing)|lit|light(?:s|ing)? up)|'
    r'(?:所有|全部)(?:指示灯|灯)(?:都|全部)?(?:亮起|点亮|亮着)|指示灯全亮', re.IGNORECASE)
_PROCEDURE = re.compile(r'\b(?:reboot\w*|restart\w*|reset\w*)\b|重启|重新启动|复位', re.IGNORECASE)
_SUCCESS = re.compile(
    r'\brespond(?:s|ing)? (?:normally|properly|correctly)\b|'
    r'\b(?:successful(?:ly)? (?:reboot\w*|restart\w*|reset\w*)|'
    r'(?:reboot\w*|restart\w*|reset\w*) (?:is |was |has been )?successful)\b|'
    r'正常(?:响应|回应)|(?:重启|重新启动|复位)成功|成功(?:重启|重新启动|复位)', re.IGNORECASE)
_DENIAL = re.compile(r'不能|无法|不代表|不意味着|未证明|是否|'
                     r'\b(?:cannot|can not|not|whether|unknown)\b', re.IGNORECASE)
_HOLD = re.compile(r'\b(?:press and hold|hold|keep pressing)\b|按住|长按', re.IGNORECASE)
_UNTIL = re.compile(r'\buntil\b|直到|直至', re.IGNORECASE)
_SOUND = re.compile(r'\b(?:tone|sound|beep)\b|提示音|声音|蜂鸣', re.IGNORECASE)

# Attribute only the matched success phrase to its explicit audible subject.
# Do not exempt a whole sentence merely because it also mentions a sound:
# another success phrase can still assign the unsupported meaning to the light.
_AUDIBLE_SUCCESS = re.compile(
    rf'(?:{_SUCCESS.pattern})\s+(?:is|was|will be)\s+'
    r'(?:signified|indicated|confirmed|signalled|signaled)\s+by\s+'
    r'(?:an?\s+)?(?:audible\s+)?(?:tone|sound|beep)\b|'
    r'\b(?:tone|sound|beep)\s+(?:signif(?:ies|ying)|indicat(?:es|ing)|'
    r'confirm(?:s|ing)|signal(?:s|ling|ing))\s+(?:a\s+)?'
    rf'(?:{_SUCCESS.pattern})|'
    rf'(?:提示音|声音|蜂鸣)(?:表示|表明|标志着|标志|证明)(?:{_SUCCESS.pattern})|'
    rf'(?:{_SUCCESS.pattern})(?:由|以)(?:提示音|声音|蜂鸣)(?:表示|表明|确认)',
    re.IGNORECASE)


def _light_success(text):
    for clause in _CLAUSE.split(text):
        if not _LIGHT.search(clause) or _DENIAL.search(clause):
            continue
        audible = [match.span() for match in _AUDIBLE_SUCCESS.finditer(clause)]
        for success in _SUCCESS.finditer(clause):
            if not any(start <= success.start() and success.end() <= end
                       for start, end in audible):
                return True
    return False


def procedure_checkpoint_mismatch(claim):
    if not _PROCEDURE.search(claim['text']) or not _light_success(claim['text']):
        return None
    for reference in claim['evidence']:
        quote = reference['quote']
        if _light_success(quote):
            continue
        # Do not borrow a success event from a separate source paragraph.
        for block in re.split(r'\n\s*\n', quote):
            if not _PROCEDURE.search(block):
                continue
            checkpoint = any(_HOLD.search(c) and _UNTIL.search(c) and _LIGHT.search(c)
                             for c in _CLAUSE.split(block))
            audible_success = any(_SOUND.search(c) and _SUCCESS.search(c)
                                  and not _DENIAL.search(c) for c in _CLAUSE.split(block))
            if checkpoint and audible_success:
                return 'procedure_checkpoint_does_not_establish_success'
    return None
