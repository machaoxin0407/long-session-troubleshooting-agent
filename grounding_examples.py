"""Synthetic demonstrations of bounded task coverage, never runtime evidence."""
import json

EXAMPLES = [
    {
        'question': 'How do I detach and refit the tray?',
        'passages': {
            'E900001': {'source_id':'M900001', 'quote':'To detach the tray, press its release latch and lift it out.'},
            'E900002': {'source_id':'M900002', 'quote':'After detaching the tray, wipe its contacts with a dry cloth.'},
        },
        'response': {
            'assessment': {'requested_task':'Detach and refit the tray',
                           'missing_parts': ['Refit the tray'], 'required_conditions': [], 'answerability':'partial',
                           'reason':'The detachment procedure is documented. Refitting is not documented; contact cleaning does not supply the missing procedure.'},
            'claims':[{'passage_ids':['E900001'], 'text':'To detach the tray, press its release latch and lift it out.'}],
            'insufficient':True,
        },
    },
    {
        'question':'型号 C 的选择开关应设在哪一档？',
        'passages': {
            'E900003': {'source_id':'M900003', 'quote':'For model C without the optional attachment, set the selector to LOW. With the attachment fitted, set it to AUTO.'},
        },
        'response': {
            'assessment': {'requested_task':'型号 C 的选择开关档位', 'missing_parts': [],
                           'required_conditions': ['仅适用于型号 C', '未装附件用 LOW；已装附件用 AUTO'], 'answerability':'complete',
                           'reason':'The source gives both attachment-dependent settings; the answer preserves both conditions rather than assuming which applies.'},
            'claims':[{'passage_ids':['E900003'], 'text':'型号 C 未安装选配附件时，选择开关设为 LOW；已安装该附件时，设为 AUTO。'}],
            'insufficient':False,
        },
    },
    {
        'question':'How do I start a calibration cycle?',
        'passages': {
            'E900004': {'source_id':'V900004', 'quote':'OCR: SERVICE; HOLD; NEXT. VLM: a hand touches a control panel. The order and purpose of the interaction are unknown.'},
        },
        'response': {
            'assessment': {'requested_task':'Start a calibration cycle',
                           'missing_parts': ['Ordered calibration procedure and identified model'],
                           'required_conditions': [], 'answerability':'none',
                           'reason':'The excerpt does not establish an ordered calibration procedure. Describing the touch as initiating calibration would invent its purpose.'},
            'claims':[], 'insufficient':True,
        },
    },
    {
        'question': '仪器 R 的指示灯怎样表示测量完成？',
        'passages': {
            'E900005': {'source_id': 'M900005', 'quote': 'Instrument R / Measurement: A flashing amber indicator means measuring. A steady blue indicator means measurement complete.'},
            'E900006': {'source_id': 'M900006', 'quote': 'Instrument R / Quick reference: A steady blue indicator means measurement complete.'},
        },
        'response': {
            'assessment': {'requested_task': '仪器 R 测量完成的灯态', 'missing_parts': [],
                           'required_conditions': ['仪器 R 测量场景；蓝灯常亮'], 'answerability': 'complete',
                           'reason': 'E900005 直接支持测量完成的灯态。E900006 重复同一事实，无需重复回答。'},
            'claims': [{'passage_ids': ['E900005'], 'text': '根据仪器 R 的测量说明，蓝色指示灯常亮表示测量完成。'}],
            'insufficient': False,
        },
    },
    {
        'question': '型号 D 怎样设置并开始通风？',
        'passages': {
            'E900007': {'source_id': 'M900007', 'quote': 'Model D / Ventilation: Before operation, close the protective cover. Without the optional duct, set MODE to FREE; with the duct fitted, set MODE to DUCT. Turn TIMER to the required duration between 2 and 8 minutes to begin ventilation.'},
            'E900008': {'source_id': 'V900008', 'quote': 'VLM: An unidentified touchscreen appliance shows an AUTO icon. A finger touches the icon. The operating sequence is unknown.'},
        },
        'response': {
            'assessment': {'requested_task': '型号 D 设置并开始通风', 'missing_parts': [],
                           'required_conditions': ['仅适用于型号 D', '操作前关闭防护盖',
                                                   '无风管用 FREE；有风管用 DUCT', 'TIMER 范围 2 至 8 分钟'],
                           'answerability': 'complete',
                           'reason': 'E900007 给出防护前提、两种配置及定时启动方式。E900008 的设备和操作顺序未知，不参与该程序。'},
            'claims': [{'passage_ids': ['E900007'], 'text': '型号 D 操作前先关闭防护盖。未安装选配风管时将 MODE 设为 FREE，已安装时设为 DUCT；再将 TIMER 转到所需的 2 至 8 分钟，以开始通风。'}],
            'insufficient': False,
        },
    },
]


def selection_example_messages():
    """Independent synthetic turns; never add demonstration IDs to the real catalog."""
    messages = []
    for example in EXAMPLES:
        payload = {'example_only': True, 'question': example['question'],
                   'sources': [{'source_id': p['source_id']} for p in example['passages'].values()],
                   'passages': example['passages']}
        messages.extend([
            {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)},
            {'role': 'assistant', 'content': json.dumps(example['response'], ensure_ascii=False)},
        ])
    return messages
