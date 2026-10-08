import json

import pytest

from session_experiment_data import (CATEGORIES, STRATEGIES, digest, make_dataset,
                                     prepare, require_review, score, validate_dataset, write_json)
from session_experiment import (CaseEmbedder, HISTORY_LIMIT, budget_preflight, evaluate_task,
                                fit_history, freeze, render_rolling, task_order, verify_ledgers, window)
from session_experiment_report import paired_interval, report
from session_context import _size


def test_fixture_split_lengths_gold_and_unknown_coverage():
    dataset = make_dataset()
    assert dataset == make_dataset()
    validate_dataset(dataset)
    assert len([c for c in dataset['cases'] if c['split'] == 'development']) == 14
    for category in CATEGORIES:
        test = [c for c in dataset['cases'] if c['split'] == 'test' and c['category'] == category]
        assert sorted(c['turns'] for c in test) == [40, 40, 80, 80, 80, 160, 160, 160]
        assert sum(not c['gold']['answerable'] for c in test) == 1
    assert len({digest(c['records']) for c in dataset['cases']}) == 70


def test_rule_scoring_requires_correct_visible_target_user_evidence():
    case = make_dataset()['cases'][0]
    source = case['gold']['sources'][0]
    correct = {'answer': case['gold']['expected_fact'], 'sources': [source]}
    context = [{'content': source['quote']}]
    assert score(correct, case, context)['passed']
    assert not score(correct, case, [])['citation_check']
    bad = {**correct, 'answer': '实际型号 DEV-9999'}
    assert not score(bad, case, context)['passed']
    unrelated = {'turn_id': 2, 'role': 'user', 'quote': case['records'][1]['question']}
    assert not score({**correct, 'sources': [unrelated]}, case, context + [{'content': unrelated['quote']}])['citation_check']
    assert not score({**correct, 'sources': [{**source, 'role': 'assistant'}]}, case, context)['citation_check']
    assert not score({**correct, 'sources': []}, case, context)['passed']


def test_negative_execution_and_unknown_contracts():
    data = make_dataset()['cases']
    case = next(c for c in data if c['category'] == 'action_status' and c['id'].endswith('_00'))
    source = case['gold']['sources'][0]
    assert score({'answer': '尚未执行断电', 'sources': [source]}, case, [{'content': source['quote']}])['passed']
    assert not score({'answer': '已经执行断电', 'sources': [source]}, case, [{'content': source['quote']}])['passed']
    unknown = next(c for c in data if not c['gold']['answerable'])
    assert score({'answer': '无法确定，用户尚未确认', 'sources': []}, unknown, [])['passed']
    assert not score({'answer': '无法确定', 'sources': [source]}, unknown, [{'content': source['quote']}])['passed']


def test_manual_gold_gate_and_no_silent_review(tmp_path):
    path = prepare(tmp_path / 'campaign')
    with pytest.raises(ValueError, match='human gold review'):
        require_review(path)
    review = json.loads((path / 'gold_review.json').read_text(encoding='utf-8'))
    review['reviewer'] = 'test-only fixture reviewer'
    for row in review['cases']:
        if row['case_id'].startswith('development_'):
            row.update(gold_verified=True, source_verified=True, answerability_verified=True)
    write_json(path / 'gold_review.json', review)
    assert len(require_review(path, 'development')['cases']) == 70
    with pytest.raises(ValueError, match='human gold review'):
        require_review(path)
    dataset = json.loads((path / 'dataset.json').read_text(encoding='utf-8'))
    dataset['seed'] += 1
    write_json(path / 'dataset.json', dataset)
    with pytest.raises(ValueError):
        require_review(path, 'development')
    assert verify_ledgers(path) == 0


def test_window_has_complete_pairs_and_shared_cap():
    records = make_dataset()['cases'][0]['records']
    context = window(records, HISTORY_LIMIT)
    assert _size(context) <= HISTORY_LIMIT
    for n in {m['turn_id'] for m in context}:
        assert {m['role'] for m in context if m['turn_id'] == n} == {'user', 'assistant'}
    oversized = [{'role': 'memory', 'content': '证据' * 10000}] + context
    fitted, trimmed = fit_history(oversized)
    assert _size(fitted) <= HISTORY_LIMIT and trimmed > 0


def test_preflight_bounds_and_fixed_task_counts():
    data = make_dataset()
    assert budget_preflight(data, 'development')['fits_cap']
    assert budget_preflight(data, 'test')['fits_cap']
    assert len(task_order(data, 'development')) == 70
    assert len(task_order(data, 'test')) == 280
    assert len(task_order(data, 'supplement')) == 98
    assert [(c['id'], s, v) for c, s, v in task_order(data, 'test')] == [(c['id'], s, v) for c, s, v in task_order(data, 'test')]


def test_case_local_embedding_cache_avoids_repeated_query_requests(tmp_path):
    class Provider:
        signature = 'fixture-model'
        calls = []
        def connection_scope(self):
            from contextlib import nullcontext
            return nullcontext()
        def __call__(self, texts, **kw):
            self.calls.append(texts)
            return [[1., float(len(t))] for t in texts]
    provider = Provider()
    cache = CaseEmbedder(tmp_path, provider)
    assert cache(['same', 'same', 'other']) == [[1., 4.], [1., 4.], [1., 5.]]
    assert cache(['same']) == [[1., 4.]]
    assert provider.calls == [['same', 'other']]
    assert CaseEmbedder(tmp_path, provider)(['same']) == [[1., 4.]]
    assert len(provider.calls) == 1


def test_rolling_cannot_recover_invisible_archive_sources():
    source = {'turn_id': 1, 'role': 'user', 'quote': '用户尚未执行断电操作。'}
    value = {'summary': '尚未执行断电', 'sources': [source]}
    assert source['quote'] in render_rolling(value, [source])
    with pytest.raises(ValueError, match='not visible'):
        render_rolling(value, [])
    with pytest.raises(ValueError):
        render_rolling({**value, 'summary': '很长' * 2000}, [source])


def test_offline_task_is_resumable_and_does_not_claim_answers(tmp_path, monkeypatch):
    import session_experiment as experiment
    path = prepare(tmp_path / 'campaign')
    case = make_dataset()['cases'][0]
    def forbidden(*args, **kwargs):
        raise AssertionError('offline must not instantiate providers')
    monkeypatch.setattr(experiment, 'ModelSessionSummarizer', forbidden)
    row = evaluate_task(path, case, 'window', 'main', phase='offline_development', online=False)
    assert row['answer_check'] is None and 'answer' not in row and row['history_budget_ok']
    assert evaluate_task(path, case, 'window', 'main', phase='offline_development', online=False) == row
    directory = path / 'offline_development' / f"{case['id']}__rolling__main"
    directory.mkdir()
    write_json(directory / 'started.json', {'started': True})
    with pytest.raises(ValueError, match='interrupted task'):
        evaluate_task(path, case, 'rolling', 'main', phase='offline_development', online=False)
    assert verify_ledgers(path) == 0


def test_ledger_cap_mutation_rejected(tmp_path):
    path = prepare(tmp_path / 'campaign')
    write_json(path / 'ledger_test.json', {'limit': 1401, 'attempts': 0, 'kinds': {}})
    with pytest.raises(ValueError, match='cap changed'):
        verify_ledgers(path)


def test_freeze_requires_real_development_and_review(tmp_path):
    path = prepare(tmp_path / 'campaign')
    with pytest.raises(ValueError):
        freeze(path)
    assert not (path / 'freeze.json').exists()


def test_bootstrap_pairs_only_independent_cases():
    assert paired_interval({'a': 1, 'b': 1}, {'a': 0, 'b': 0}, draws=100)['ci95'] == [1., 1.]
    assert paired_interval({'a': 1}, {'a': 1}, draws=100)['supports_positive_gain'] is False
    assert paired_interval({}, {})['n'] == 0


def test_empty_report_is_pending_and_preserves_zero_model_calls(tmp_path):
    path = prepare(tmp_path / 'campaign')
    status = report(path)
    assert status['main_tasks'] == 0 and status['attempts'] == 0
    comparison = json.loads((path / 'comparison.json').read_text(encoding='utf-8'))
    assert not comparison['human_review_complete'] and not comparison['main_complete']
    assert set(comparison['by_strategy']) == set(STRATEGIES)
    assert '95.2%' not in (path / 'EXPERIMENT_REPORT.md').read_text(encoding='utf-8')


@pytest.mark.parametrize('fail_query', [False, True])
def test_live_adapter_contract_with_fake_models_and_blind_build(tmp_path, monkeypatch, fail_query):
    """No network: verify construction boundary, shared tree and ledger behavior."""
    import session_experiment as experiment
    from model_attempts import record_attempt
    from session_tree import MemoryTree
    from session_context import FIELDS
    from contextlib import nullcontext
    path = prepare(tmp_path / 'campaign')
    monkeypatch.setenv('MODEL_ATTEMPT_LEDGER', str(path / 'ledger_development.json'))
    builds = []
    original = MemoryTree.build
    def build(cls, records, **kw):
        assert all(set(r) == {'id', 'created_at', 'question', 'answer'} for r in records)
        builds.append(records)
        return original(records, **kw)
    monkeypatch.setattr(MemoryTree, 'build', classmethod(build))
    class Model:
        def __init__(self, **kw):
            self.kw = kw
            self.signature = digest(kw.get('prompt', 'structured'))
        def __call__(self, payload, **kw):
            record_attempt('summary_or_qa')
            self.kw['on_usage']({'prompt_tokens': 12, 'completion_tokens': 6})
            value = json.loads(payload)
            if isinstance(value, dict) and 'groups' in value:
                return {'nodes': [{'id': g['id'], 'text': g['evidence'][0]['quote'][:35],
                                  'sources': [{'leaf_id': g['evidence'][0]['leaf_id'],
                                               'quote': g['evidence'][0]['quote']}]} for g in value['groups']]}
            if isinstance(value, dict) and 'history' in value:
                return {'answer': '无法确定', 'sources': []}
            return {k: [] for k in FIELDS}
    class Embedder:
        signature = 'fake'
        def __init__(self, **kw):
            pass
        def connection_scope(self):
            return nullcontext()
        def __call__(self, texts, **kw):
            record_attempt('embedding')
            if fail_query and texts == [make_dataset()['cases'][0]['question']]:
                raise RuntimeError('fake query service failure')
            return [[1., float(len(t)), float(int(digest(t)[:4], 16))] for t in texts]
    monkeypatch.setattr(experiment, 'ModelSessionSummarizer', Model)
    monkeypatch.setattr(experiment, 'SessionEmbedder', Embedder)
    case = make_dataset()['cases'][0]
    first = evaluate_task(path, case, 'collapsed', 'main', phase='development', online=True)
    second = evaluate_task(path, case, 'traversal', 'main', phase='development', online=True)
    assert len(builds) == 1
    assert first['shared_build'] == second['shared_build']
    expected = 'lexical_tfidf' if fail_query else 'semantic'
    assert first['vector_method'] == expected and second['vector_method'] == expected
    assert first['query_attempts'] == 1 and second['query_attempts'] == (1 if fail_query else 0)
    assert first['history_budget_ok'] and second['history_budget_ok']
    assert first['observations'][-1]['usage']['prompt_tokens'] == 12
    before = verify_ledgers(path)
    assert evaluate_task(path, case, 'collapsed', 'main', phase='development', online=True) == first
    assert verify_ledgers(path) == before
    modified = {**case, 'question': 'changed'}
    with pytest.raises(ValueError, match='input changed'):
        evaluate_task(path, modified, 'collapsed', 'main', phase='development', online=True)


def test_blinded_export_keeps_decisions_and_does_not_expose_strategy(tmp_path):
    from session_experiment_report import review_exports
    path = prepare(tmp_path / 'campaign')
    data = make_dataset()
    case = next(c for c in data['cases'] if c['split'] == 'test')
    row = {'task_id': 'fixture-collapsed-main', 'case': case['id'],
           'answer': {'answer': '未知', 'sources': []}, 'answer_error_type': None}
    assert review_exports(path, [row], data) == {}
    public = json.loads((path / 'BLIND_ANSWERS.json').read_text(encoding='utf-8'))
    assert 'collapsed' not in json.dumps(public)
    form = json.loads((path / 'human_review.json').read_text(encoding='utf-8'))
    form['reviewer'] = 'test fixture reviewer'
    form['rows'][0].update(fact_correct=False, evidence_supports=False, role_or_execution_error=False, notes='fixture')
    write_json(path / 'human_review.json', form)
    decisions = review_exports(path, [row], data)
    assert decisions[row['task_id']]['notes'] == 'fixture'


def test_recommendation_can_prefer_cheaper_structured_memory_over_rolling():
    from session_experiment_report import select_strategy
    rows, decisions = [], {}
    for strategy in STRATEGIES:
        for i in range(56):
            task = f'{strategy}_{i}'
            rows.append({'task_id': task, 'case': str(i), 'strategy': strategy, 'answer': {},
                         'shared_build': {'attempts': {'window': 0, 'rolling': 4, 'structured_recall': 1,
                                                        'collapsed': 10, 'traversal': 10}[strategy]}})
            decisions[task] = {'fact_correct': strategy != 'window', 'evidence_supports': strategy != 'window',
                               'role_or_execution_error': False}
    result = select_strategy(rows, decisions)
    assert result['strategy'] == 'structured_recall'
    assert not result['production_config_changed']
