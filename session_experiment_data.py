"""Synthetic, versioned long-conversation fixtures and transparent rule scoring.

Gold data is kept outside the records passed to memory constructors. No LLM judge.
Rules establish a narrow contract; human entailment review remains separate.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from pathlib import Path

CATEGORIES = ('correction', 'negation', 'action_status', 'early_identifier',
              'middle_of_long_turn', 'assistant_trap', 'paraphrase')
STRATEGIES = ('window', 'rolling', 'structured_recall', 'collapsed', 'traversal')
SEED = 20261008
SCORER_VERSION = 'long-session-rules-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def make_dataset():
    """14 development + 56 test cases; templates and identifiers split by family."""
    cases = []
    for split, count in (('development', 2), ('test', 8)):
        for category_index, category in enumerate(CATEGORIES):
            for variant in range(count):
                turns = (40, 80)[variant] if split == 'development' else (40, 40, 80, 80, 80, 160, 160, 160)[variant]
                index = (1000 if split == 'development' else 5000) + category_index * 100 + variant
                rng = random.Random(SEED + index)
                positions = (3, turns // 2, turns - 9)
                position = positions[(variant + category_index) % 3]
                if category == 'early_identifier':
                    position = 3 + variant % 3
                unknown = split == 'test' and variant == 7
                positive = variant % 2 == 1
                identifier, old = f'DEV-{index}', f'DEV-{index + 10}'
                code, ticket, battery = f'ERR-{index}', f'CASE-{index}', f'PACK-{index}'
                if split == 'development':
                    prefix = f'补充第{variant + 1}份现场记录：'
                else:
                    prefix = (f'现场复核记录{variant + 1}：', f'请按本次实际情况登记{variant + 1}：',
                              f'我把上次不清楚的地方说明一下{variant + 1}：')[variant % 3]
                facts = {
                    'correction': (f'更正：实际设备型号为 {identifier}，不是 {old}。',
                                   '我后来确认的实际设备型号是什么？', [[identifier]], [old]),
                    'negation': ('我已经更换主板，连接线没有更换。' if positive else '我没有更换主板，只更换了连接线。',
                                 '我实际更换了哪个部件，另一个部件动过吗？',
                                 [['更换主板', '更换了主板'], ['连接线'], ['没有', '未']] if positive else
                                 [['连接线'], ['没有更换主板', '未更换主板', '主板没有更换', '主板未更换', '未动主板']],
                                 ['只更换了连接线'] if positive else ['已经更换主板', '已更换主板', '更换了主板']),
                    'action_status': ('我已经执行断电操作，重连后问题仍然存在。' if positive else
                                      '我尚未执行断电操作，只记录了建议。',
                                      '建议的断电操作我到底执行过了吗？',
                                      [['断电'], ['已经执行', '已执行', '执行过', '已完成']] if positive else
                                      [['断电'], ['未执行', '尚未', '没有执行', '未进行']],
                                      ['没有执行', '尚未执行', '未执行'] if positive else ['已经执行', '已执行', '已经完成']),
                    'early_identifier': (f'维修单编号 {ticket}，约定下周送检。', '之前登记的维修单编号是什么？', [[ticket]], []),
                    'middle_of_long_turn': (f'关键错误码 {code}，仅在启动时出现。',
                                            '仅在启动时出现的那个错误码是什么？', [[code]], []),
                    'assistant_trap': ('我确实已经恢复出厂设置，但故障没有消失。' if positive else
                                       '我没有恢复出厂设置，仅询问步骤。',
                                       '依据我自己的确认，我恢复过出厂设置吗？',
                                       [['恢复出厂设置'], ['已经', '确实', '执行过', '恢复过']] if positive else
                                       [['恢复出厂设置', '恢复出厂'], ['没有', '未执行', '未恢复', '并未']],
                                       ['没有恢复', '未恢复', '未执行'] if positive else ['已经恢复', '已恢复出厂', '确实恢复']),
                    'paraphrase': (f'电池已经无法蓄电，充电后很快耗尽，型号 {battery}。',
                                   '之前提过的续航很差的电池型号是什么？', [[battery]], []),
                }
                quote, question, required, forbidden = facts[category]
                records = []
                dev_fill = ['预约时间需要改到周末', '外壳有轻微划痕', '还在寻找购机凭证', '包装盒放在储物间']
                test_fill = ['收件地点暂定单位前台', '工作日晚上有人接电话', '备用设备在另一处房间',
                             '请把费用明细单独列出', '今天暂不安排上门', '说明书封面有磨损',
                             '需要先确认配件库存', '运输时会用泡沫保护边角']
                fills = dev_fill if split == 'development' else test_fill
                for n in range(1, turns + 1):
                    a, b = rng.sample(fills, 2)
                    records.append({'id': n, 'question': f'第{n}次补充，{a}。{b}。本条没有新增排障执行结果。',
                                    'answer': '补充信息已记录。请继续确认实际观察，建议需由您执行后才能记录结果。'})
                records[0]['question'] = f'最早填过的设备型号是 {old}，尚待核实。'
                records[max(0, position - 2)]['answer'] = (('您没有恢复出厂设置。' if positive else '您已经恢复出厂设置。')
                                                        if category == 'assistant_trap' else
                                                        '建议先断电重连，再决定是否更换部件。')
                records[min(turns - 1, position + 3)]['question'] = f'邻居设备曾出现 ERR-{index + 20}，不是我的故障记录。'
                if category == 'early_identifier':
                    records[min(turns - 1, position + 2)]['question'] = f'快递单号 LOG-{index}，不是维修单号。'
                if unknown:
                    records[position - 1]['question'] = prefix + '我还没有核实相关信息，请不要把建议或别人的情况当作我的确认。'
                elif category == 'middle_of_long_turn':
                    records[position - 1]['question'] = prefix + ('外包装和预约安排仍待确认。' * 85) + quote + ('配件库存和运输方式之后再说。' * 85)
                else:
                    records[position - 1]['question'] = prefix + quote
                gold = {'answerable': not unknown, 'expected_fact': quote if not unknown else '未知：没有用户确认的目标事实',
                        'sources': [] if unknown else [{'turn_id': position, 'role': 'user', 'quote': quote}],
                        'required_groups': required if not unknown else [['未知', '无法确定', '未提供', '没有足够', '未确认', '无法确认', '不确定']],
                        'forbidden_phrases': forbidden if not unknown else [identifier, code, ticket, battery]}
                cases.append({'id': f'{split}_{category}_{variant:02d}', 'category': category,
                              'split': split, 'template_family': f'{split}_family_{variant}',
                              'turns': turns, 'question': question, 'records': records, 'gold': gold})
    return {'schema': 'long-session-dataset-v1', 'seed': SEED, 'cases': cases}


def validate_dataset(dataset):
    cases = dataset['cases']
    if len(cases) != 70 or len({c['id'] for c in cases}) != 70:
        raise ValueError('dataset must contain 70 distinct cases')
    families = {split: {c['template_family'] for c in cases if c['split'] == split}
                for split in ('development', 'test')}
    if families['development'] & families['test']:
        raise ValueError('template family leakage')
    for category in CATEGORIES:
        dev = [c for c in cases if c['category'] == category and c['split'] == 'development']
        test = [c for c in cases if c['category'] == category and c['split'] == 'test']
        if len(dev) != 2 or len(test) != 8 or sum(not c['gold']['answerable'] for c in test) != 1:
            raise ValueError('incorrect stratification')
    for case in cases:
        if len(case['records']) != case['turns'] or [r['id'] for r in case['records']] != list(range(1, case['turns'] + 1)):
            raise ValueError('invalid chronological history')
        for source in case['gold']['sources']:
            if source['quote'] not in case['records'][source['turn_id'] - 1]['question']:
                raise ValueError('gold source absent')


def score(value, case, context):
    result = {'scorer_version': SCORER_VERSION, 'fact_check': False, 'citation_check': False,
              'passed': False, 'human_semantic_review': None}
    if not isinstance(value, dict) or set(value) != {'answer', 'sources'}:
        return result
    answer, citations = value['answer'], value['sources']
    if not isinstance(answer, str) or not answer.strip() or not isinstance(citations, list):
        return result
    answer = re.sub(r'\s+', '', answer).lower()
    gold = case['gold']
    def normalized(text):
        return re.sub(r'\s+', '', text).lower()
    forbidden_answer = answer
    if case['category'] == 'correction':
        # Correctly denying an obsolete identifier must not fail merely because
        # its literal spelling appears in the answer. Other uses still fail.
        for old in gold['forbidden_phrases']:
            forbidden_answer = re.sub(r'(?:不是|并非|非)' + re.escape(normalized(old)), '', forbidden_answer)
    fact = (all(any(normalized(term) in answer for term in group) for group in gold['required_groups'])
            and not any(normalized(term) in forbidden_answer for term in gold['forbidden_phrases']))
    valid = not citations if not gold['answerable'] else bool(citations)
    gold_hit = not gold['answerable']
    visible = '\n'.join(m['content'] for m in context)
    raw = {r['id']: r for r in case['records']}
    for citation in citations:
        if (not isinstance(citation, dict) or set(citation) != {'turn_id', 'role', 'quote'}
                or type(citation['turn_id']) is not int or citation['turn_id'] not in raw
                or citation['role'] != 'user' or not isinstance(citation['quote'], str)
                or not citation['quote'].strip() or citation['quote'] not in raw[citation['turn_id']]['question']
                or citation['quote'] not in visible):
            valid = False
            continue
        gold_hit |= any(citation['turn_id'] == g['turn_id'] and
                        (g['quote'] in citation['quote'] or
                         (len(citation['quote']) >= 8 and citation['quote'] in g['quote']))
                        for g in gold['sources'])
    result.update(fact_check=fact, citation_check=valid and gold_hit,
                  passed=fact and valid and gold_hit)
    return result


def prepare(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    dataset = make_dataset()
    validate_dataset(dataset)
    write_json(output / 'dataset.json', dataset)
    (output / 'histories').mkdir()
    for case in dataset['cases']:
        write_json(output / 'histories' / (case['id'] + '.json'),
                   {'case': case['id'], 'question': case['question'], 'records': case['records']})
    dataset_hash = digest(dataset)
    review = {'dataset_sha256': dataset_hash, 'reviewer': '', 'cases': [
        {'case_id': c['id'], 'gold_verified': False, 'source_verified': False,
         'answerability_verified': False, 'notes': ''} for c in dataset['cases']]}
    write_json(output / 'gold_review.json', review)
    lines = ['# 标准答案与来源审阅表', '', f'数据 SHA256：`{dataset_hash}`', '',
             '请核对目标事实、来源轮次与证据充分性，并在 gold_review.json 逐项记录真实核对结果和审阅者。',
             '未知案例需查看完整历史；下表的邻近片段不是全部证据。未核对前不可运行真实模型或冻结。', '']
    for c in dataset['cases']:
        lines += [f"## {c['id']}（{c['turns']} 轮）", '', f"问题：{c['question']}", '',
                  f"预期：{c['gold']['expected_fact']}", '', f"来源：{json.dumps(c['gold']['sources'], ensure_ascii=False)}", '',
                  f"[完整历史](histories/{c['id']}.json)", '']
        selected = {0, len(c['records']) - 1}
        for g in c['gold']['sources']:
            selected.update(range(max(0, g['turn_id'] - 3), min(len(c['records']), g['turn_id'] + 2)))
        for n in sorted(selected):
            r = c['records'][n]
            text = r['question'] if len(r['question']) < 600 else '[长消息，完整原文见链接；本页上方已列出目标原话]'
            lines += [f"- 第{r['id']}轮 用户：{text}", f"  客服：{r['answer']}"]
        lines += ['']
    (output / 'GOLD_REVIEW.md').write_text('\n'.join(lines), encoding='utf-8')
    sections = '\n'.join(lines).split('\n## ')
    (output / 'DEVELOPMENT_GOLD_REVIEW.md').write_text(sections[0] + ''.join(
        '\n## ' + part for part in sections[1:] if part.startswith('development_')), encoding='utf-8')
    (output / 'TEST_GOLD_REVIEW.md').write_text(sections[0] + ''.join(
        '\n## ' + part for part in sections[1:] if part.startswith('test_')), encoding='utf-8')
    # Budget files have fixed, non-transferable caps; their sum is the global hard cap.
    caps = {'development': 250, 'test': 1400, 'supplement': 250, 'api': 50, 'reserve': 50}
    for phase, cap in caps.items():
        write_json(output / f'ledger_{phase}.json', {'limit': cap, 'attempts': 0, 'kinds': {}})
    write_json(output / 'campaign.json', {'schema': 1, 'dataset_sha256': dataset_hash,
               'phase_caps': caps, 'total_limit': 2000, 'status': 'awaiting_gold_review'})
    with (output / 'gold_review.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['case_id', 'split', 'question', 'expected', 'gold_sources', 'answerable'])
        for c in dataset['cases']:
            writer.writerow([c['id'], c['split'], c['question'], c['gold']['expected_fact'],
                             json.dumps(c['gold']['sources'], ensure_ascii=False), c['gold']['answerable']])
    return output


def require_review(output, split=None):
    output = Path(output)
    dataset = json.loads((output / 'dataset.json').read_text(encoding='utf-8'))
    validate_dataset(dataset)
    review = json.loads((output / 'gold_review.json').read_text(encoding='utf-8'))
    if review['dataset_sha256'] != digest(dataset) or not review.get('reviewer', '').strip():
        raise ValueError('human gold review pending or dataset changed')
    rows = review['cases']
    if len(rows) != 70 or len({r['case_id'] for r in rows}) != 70:
        raise ValueError('incomplete gold review identifiers')
    by_id = {r['case_id']: r for r in rows}
    for c in dataset['cases']:
        if split is not None and c['split'] != split:
            continue
        row = by_id.get(c['id'], {})
        if not all(row.get(k) is True for k in ('gold_verified', 'source_verified', 'answerability_verified')):
            raise ValueError('human gold review pending: ' + c['id'])
    return dataset
