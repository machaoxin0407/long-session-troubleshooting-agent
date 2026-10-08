"""Choose the language of fixed UI notices from the user's asking sentence."""
import re


def question_is_chinese(question):
    # Quoted evidence, control labels and titles need not share the question's
    # language. This is a display heuristic, not a semantic quality judgment.
    prose = re.sub(r'`[^`]*`|"[^"\n]*"|“[^”]*”|「[^」]*」|《[^》]*》', ' ', question)
    if not re.search(r'[A-Za-z\u3400-\u9fff]', prose):
        prose = question
    choices = []
    patterns = [
        (r'(?:用|以)(中文|汉语|英文|英语)(?:来)?(?:回答|回复|说明|解释|作答|描述)', True),
        ((r'\b(?:answer|reply|respond|explain)(?:\s+(?:me|it|this|that|the|question|please))*'
          r'\s+in\s+(Chinese|Mandarin|English)\b'), False),
    ]
    for pattern, chinese_request in patterns:
        for match in re.finditer(pattern, prose, re.IGNORECASE):
            prefix = prose[max(0, match.start()-20):match.start()]
            if re.search(r'(?:\bnot|\bnever|\bdon.t|不要|不得|请勿)\s*$', prefix, re.IGNORECASE):
                continue
            language = match[1].lower()
            chinese = language in ('中文', '汉语') if chinese_request else language in ('chinese', 'mandarin')
            choices.append((match.start(), chinese))
    if choices:
        return max(choices)[1]
    cjk = re.findall(r'[\u3400-\u9fff]', prose)
    if not cjk:
        return False
    chinese_asking = re.search(r'如何|怎样|是否|为什么|多少|能否|能不能|什么|哪(?:个|些|种|里)|吗|请问', prose)
    english_frame = re.match(
        r'\s*(?:according to|based on|what|which|when|where|why|how|can|could|does|do|is|are|'
        r'will|would|should|explain|describe|compare|give|provide|tell|in|for)\b', prose, re.IGNORECASE)
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", prose)
    if english_frame and not chinese_asking and len(words) >= 3:
        return False
    return len(cjk) >= len(words)
