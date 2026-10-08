"""Reports separate rules, human judgments, source visibility and model fallbacks."""
from __future__ import annotations

import csv
import json
import random
import statistics
from pathlib import Path

from session_experiment_data import CATEGORIES, SEED, STRATEGIES, digest, write_json

ERROR_TYPES = ('correction_lost', 'negation_reversed', 'advice_as_executed',
               'unsupported_citation', 'other')


def percentile(values, fraction):
    values = sorted(v for v in values if v is not None)
    return values[max(0, min(len(values) - 1, int(len(values) * fraction + .999) - 1))] if values else None


def paired_interval(left, right, *, draws=10000):
    """Independent-case paired bootstrap; repeats never enter this computation."""
    common = sorted(set(left) & set(right))
    if not common:
        return {'n': 0, 'delta': None, 'ci95': None, 'supports_positive_gain': False}
    differences = [float(left[k]) - float(right[k]) for k in common]
    rng = random.Random(SEED)
    samples = sorted(statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(draws))
    interval = [samples[int(draws * .025)], samples[min(draws - 1, int(draws * .975))]]
    return {'n': len(common), 'delta': statistics.mean(differences), 'ci95': interval,
            'supports_positive_gain': interval[0] > 0}


def summary(rows):
    online = [r for r in rows if r['online']]
    checks = [(r.get('answer_check') or {}) for r in online]
    usages = []
    for row in rows:
        # Cost of the logical strategy includes its shared construction once,
        # irrespective of which tree mode physically triggered construction.
        observations = [r for r in row['observations'] if r['kind'] == 'qa']
        observations += (row.get('shared_build') or {}).get('observations', [])
        usages += [r.get('usage') for r in observations]
    return {'tasks': len(rows), 'answers': sum('answer' in r for r in online),
            'rule_passed': sum(bool(c.get('passed')) for c in checks) if online else None,
            'fact_rule_passed': sum(bool(c.get('fact_check')) for c in checks) if online else None,
            'source_rule_passed': sum(bool(c.get('citation_check')) for c in checks) if online else None,
            'unknown_cases': sum(not r['answerable'] for r in online),
            'unknown_rule_passed': sum(not r['answerable'] and bool((r.get('answer_check') or {}).get('passed')) for r in online),
            'fallback_tasks': sum(bool(r['fallbacks']) for r in rows),
            'semantic_tree_queries': sum(r.get('vector_method') == 'semantic' for r in rows),
            'lexical_tree_queries': sum(r.get('vector_method') == 'lexical_tfidf' for r in rows),
            'tree_queries_with_model_nodes': sum((r.get('tree_model_nodes') or 0) > 0 for r in rows),
            'normal_rule_passed': sum(not r['fallbacks'] and bool((r.get('answer_check') or {}).get('passed')) for r in online),
            'degraded_rule_passed': sum(bool(r['fallbacks']) and bool((r.get('answer_check') or {}).get('passed')) for r in online),
            'source_visible': sum(r['source_text_coverage'] is True for r in rows),
            'source_visibility_denominator': sum(r['answerable'] for r in rows),
            'history_budget_violations': sum(not r['history_budget_ok'] for r in rows),
            'query_p50_ms': percentile([r['query_ms'] for r in rows], .5),
            'query_p95_ms': percentile([r['query_ms'] for r in rows], .95),
            'context_answer_p50_ms': percentile([r['context_plus_answer_ms'] for r in rows], .5),
            'context_answer_p95_ms': percentile([r['context_plus_answer_ms'] for r in rows], .95),
            'cold_context_answer_p50_ms': percentile([r['cold_context_plus_answer_ms'] for r in rows], .5),
            'cold_context_answer_p95_ms': percentile([r['cold_context_plus_answer_ms'] for r in rows], .95),
            'provider_usage_missing_observations': sum(u is None for u in usages),
            'provider_prompt_tokens_observed': sum(u.get('prompt_tokens', 0) for u in usages if u),
            'provider_completion_tokens_observed': sum(u.get('completion_tokens', 0) for u in usages if u),
            'root_removal_applied_tasks': sum(r.get('root_removal_applied') is True for r in rows),
            'embedding_usage': 'not exposed by current embedding adapter; missing, not zero',
            'trimmed_history_bytes': sum(r['trimmed_history_bytes'] for r in rows)}


def review_exports(output, rows, dataset):
    """Stable blinded IDs. Regeneration never discards existing human decisions."""
    key_path = output / 'review_key.json'
    key = json.loads(key_path.read_text(encoding='utf-8')) if key_path.exists() else {}
    existing_path = output / 'human_review.json'
    existing = json.loads(existing_path.read_text(encoding='utf-8')) if existing_path.exists() else {'reviewer': '', 'rows': []}
    old = {r['review_id']: r for r in existing['rows']}
    cases = {c['id']: c for c in dataset['cases']}
    public, forms = [], []
    for row in rows:
        task = row['task_id']
        review_id = 'R-' + digest((SEED, task))[:14]
        key[review_id] = task
        case = cases[row['case']]
        public.append({'review_id': review_id, 'case': row['case'], 'question': case['question'],
                       'expected_fact': case['gold']['expected_fact'], 'expected_sources': case['gold']['sources'],
                       'history_file': f"histories/{row['case']}.json",
                       'answer': row.get('answer'), 'failure': row.get('answer_error_type')})
        forms.append({**{'review_id': review_id, 'fact_correct': None, 'evidence_supports': None,
                      'role_or_execution_error': None, 'error_types': [], 'notes': ''}, **old.get(review_id, {})})
    random.Random(SEED).shuffle(public)
    random.Random(SEED).shuffle(forms)
    write_json(key_path, key)
    write_json(output / 'BLIND_ANSWERS.json', public)
    write_json(existing_path, {'reviewer': existing['reviewer'], 'rows': forms})
    with (output / 'BLIND_ANSWERS.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['review_id', 'case', 'question', 'expected_fact', 'expected_sources', 'answer', 'failure'])
        for r in public:
            writer.writerow([r['review_id'], r['case'], r['question'], r['expected_fact'],
                             json.dumps(r['expected_sources'], ensure_ascii=False), json.dumps(r['answer'], ensure_ascii=False), r['failure']])
    decisions = {}
    if existing['reviewer'].strip():
        for r in forms:
            if (all(type(r[k]) is bool for k in ('fact_correct', 'evidence_supports', 'role_or_execution_error'))
                    and isinstance(r.get('error_types'), list) and all(t in ERROR_TYPES for t in r['error_types'])):
                decisions[key[r['review_id']]] = r
    return decisions


def select_strategy(main, decisions):
    """Evidence-based experimental recommendation; never changes production config."""
    if len(main) != 280 or len(decisions) != 280 or any('answer' not in r for r in main):
        return {'status': 'pending_human_and_complete_test', 'strategy': None}
    values = {s: {} for s in STRATEGIES}
    critical = {s: 0 for s in STRATEGIES}
    calls = {s: [] for s in STRATEGIES}
    for row in main:
        d = decisions[row['task_id']]
        values[row['strategy']][row['case']] = bool(d['fact_correct'] and d['evidence_supports'] and not d['role_or_execution_error'])
        critical[row['strategy']] += d['role_or_execution_error']
        # Logical cold cost: both tree modes require a question vector, even
        # when the second physical query reused the first mode's cached vector.
        calls[row['strategy']].append((row.get('shared_build') or {}).get('attempts', 0)
                                      + 1 + int(row['strategy'] in ('collapsed', 'traversal')))
    chosen, changes = 'window', []
    # Simpler strategies win when no positive paired evidence justifies more machinery.
    for candidate in STRATEGIES[1:]:
        comparison = paired_interval(values[candidate], values[chosen])
        if comparison['supports_positive_gain'] and critical[candidate] <= critical[chosen]:
            changes.append({'from': chosen, 'to': candidate, 'evidence': comparison})
            chosen = candidate
        elif (values[candidate] == values[chosen] and critical[candidate] == critical[chosen]
              and statistics.mean(calls[candidate]) < statistics.mean(calls[chosen])):
            changes.append({'from': chosen, 'to': candidate, 'reason': 'identical observed human outcomes, fewer cold HTTP attempts'})
            chosen = candidate
    return {'status': 'experimental_recommendation', 'strategy': chosen, 'changes': changes,
            'critical_errors': critical, 'production_config_changed': False,
            'logical_cold_attempts_mean': {s: statistics.mean(v) for s, v in calls.items()},
            'rule': 'positive human paired CI and no additional critical errors; identical outcomes prefer fewer calls, then simpler design',
            'caution': 'exploratory paired comparisons on synthetic cases, not production acceptance'}


def report(output):
    from session_experiment import verify_ledgers
    output = Path(output)
    dataset = json.loads((output / 'dataset.json').read_text(encoding='utf-8'))
    all_rows = []
    for phase in ('development', 'test', 'supplement', 'offline_development', 'offline_test', 'offline_supplement'):
        all_rows += [json.loads(p.read_text(encoding='utf-8')) for p in sorted((output / phase).glob('*/result.json'))]
    main = [r for r in all_rows if r['online'] and r['split'] == 'test' and r['variant'] == 'main']
    decisions = review_exports(output, main, dataset)
    complete = len(main) == 280 and all('answer' in r for r in main)
    human_complete = complete and len(decisions) == 280
    modes = {s: summary([r for r in main if r['strategy'] == s]) for s in STRATEGIES}
    by_category = {c: {s: summary([r for r in main if r['category'] == c and r['strategy'] == s])
                       for s in STRATEGIES} for c in CATEGORIES}
    by_length = {str(n): {s: summary([r for r in main if r['turns'] == n and r['strategy'] == s])
                         for s in STRATEGIES} for n in (40, 80, 160)}
    comparisons = []
    for metric in ('rule', 'human'):
        values = {s: {} for s in STRATEGIES}
        for r in main:
            if metric == 'rule':
                values[r['strategy']][r['case']] = bool((r.get('answer_check') or {}).get('passed'))
            elif r['task_id'] in decisions:
                d = decisions[r['task_id']]
                values[r['strategy']][r['case']] = d['fact_correct'] and d['evidence_supports'] and not d['role_or_execution_error']
        for reference in ('window', 'structured_recall'):
            for candidate in STRATEGIES:
                if candidate != reference:
                    comparisons.append({'metric': metric, 'candidate': candidate, 'reference': reference,
                                        **paired_interval(values[candidate], values[reference])})
    ablation_comparisons = []
    for variant in ('no_root', 'no_recall'):
        treatment = [r for r in all_rows if r['online'] and r['variant'] == variant]
        if variant == 'no_root':
            treatment = [r for r in treatment if r.get('root_removal_applied') is True]
        original = {r['case']: bool((r.get('answer_check') or {}).get('passed'))
                    for r in main if r['strategy'] == 'collapsed'}
        removed = {r['case']: bool((r.get('answer_check') or {}).get('passed')) for r in treatment}
        ablation_comparisons.append({'variant': variant, 'metric': 'rule_only_not_human_entailment',
                                     'delta_direction': 'original_minus_removed', **paired_interval(original, removed)})
    ledgers = {phase: json.loads((output / f'ledger_{phase}.json').read_text(encoding='utf-8')) for phase in ('development', 'test', 'supplement', 'api', 'reserve')}
    results = {'schema': 1, 'main_complete': complete, 'main_expected': 280, 'main_tasks': len(main),
               'human_review_complete': human_complete, 'human_reviews': len(decisions),
               'dataset_sha256': digest(dataset), 'total_attempts': verify_ledgers(output), 'total_limit': 2000,
               'ledgers': ledgers, 'by_strategy': modes, 'by_category': by_category, 'by_length': by_length,
               'paired_comparisons': comparisons,
               'ablation_comparisons': ablation_comparisons,
               'repeats': {s: summary([r for r in all_rows if r['online'] and r['variant'] == 'repeat' and r['strategy'] == s]) for s in STRATEGIES},
               'ablations': {v: summary([r for r in all_rows if r['online'] and r['variant'] == v]) for v in ('no_root', 'no_recall')},
               'selection_status': 'awaiting_complete_frozen_and_human_results' if not human_complete else 'compare_effect_cost_table',
               'recommendation': select_strategy(main, decisions),
               'human_by_strategy': {s: {
                   'reviewed': sum(r['strategy'] == s and r['task_id'] in decisions for r in main),
                   'fact_correct': sum(r['strategy'] == s and decisions.get(r['task_id'], {}).get('fact_correct') is True for r in main),
                   'evidence_supports': sum(r['strategy'] == s and decisions.get(r['task_id'], {}).get('evidence_supports') is True for r in main),
                   'role_or_execution_errors': sum(r['strategy'] == s and decisions.get(r['task_id'], {}).get('role_or_execution_error') is True for r in main),
                   'error_types': {tag: sum(r['strategy'] == s and tag in decisions.get(r['task_id'], {}).get('error_types', [])
                                             for r in main) for tag in ERROR_TYPES}
               } for s in STRATEGIES},
               'limitations': ['synthetic cases, not customer accuracy', 'rule source matching is not semantic entailment',
                               'local context+QA timing excludes Agent tools/API/network overhead outside the measured calls',
                               'within-case exact-text vector cache shared by tree modes; report cached timings explicitly',
                               'paired intervals are exploratory; no external generalization claim']}
    write_json(output / 'comparison.json', results)
    write_json(output / 'error_cases.json', [r for r in main if not (r.get('answer_check') or {}).get('passed') or r['fallbacks']])
    lines = ['# 长会话 Agent 设计收益实验', '',
             f"外部尝试：{results['total_attempts']}/2000；冻结主评测：{len(main)}/280；人工复核：{len(decisions)}/280。", '',
             '尚未完成的实验不填入准确率；来源可见性、规则评分与人工语义判定分别统计。', '',
             '| 策略 | 完整回答 | 规则通过 | 来源规则通过 | 降级任务 |', '|---|---:|---:|---:|---:|']
    for s, r in modes.items():
        lines.append(f"| {s} | {r['answers']} | {r['rule_passed']} | {r['source_rule_passed']} | {r['fallback_tasks']} |")
    lines += ['', '## 阅读与复现', '',
              '- GOLD_REVIEW.md / gold_review.json：先核对标准答案和来源，不能由助手冒充人工确认。',
              '- BLIND_ANSWERS.csv / human_review.json：隐藏策略名的回答审阅表；review_key.json 为恢复对应关系的内部文件，盲审时不要打开。',
              '- comparison.json：分场景、历史长度、规则/人工结果、配对区间及额度。',
              '- error_cases.json：错误和降级逐条保留；不能将回退算作模型摘要成功。',
              '- 各任务 context.json / result.json：实际输入、模型输出、时间、usage 和失败类型。', '',
              '当前不宣称树形记忆优于基线；推荐配置必须等待完整冻结测试和人工结果。', '']
    (output / 'EXPERIMENT_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    return {'main_complete': complete, 'main_tasks': len(main), 'human_reviews': len(decisions),
            'attempts': results['total_attempts'], 'report': str(output / 'EXPERIMENT_REPORT.md')}
