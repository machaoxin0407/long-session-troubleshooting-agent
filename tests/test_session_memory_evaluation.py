import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from benchmark_session_memory import RemoteBudget, evaluate, main, make_cases, score_answer


def test_synthetic_cases_are_deterministic_and_cover_required_categories():
    assert make_cases(20) == make_cases(20)
    cases = make_cases(20)
    assert {'correction', 'negation', 'action_status', 'middle_of_long_turn', 'assistant_trap', 'paraphrase'} <= {c['id'] for c in cases}
    for case in cases:
        assert case['gold']['quote'] in case['records'][case['gold']['turn_id'] - 1]['question']


def test_answer_requires_correct_fact_and_visible_role_correct_source():
    case = make_cases(20)[0]
    context = [{'content': case['gold']['quote']}]
    value = {'answer': '型号是 ZX901。', 'sources': [case['gold']]}
    assert score_answer(value, case, context)['passed']
    assert not score_answer(value, case, [])['passed']
    bad = {'answer': '型号是 ZX900。', 'sources': [case['gold']]}
    assert not score_answer(bad, case, context)['passed']
    value['sources'][0] = {**case['gold'], 'role': 'assistant'}
    assert not score_answer(value, case, context)['citation_check']
    value['sources'] = []
    assert not score_answer(value, case, context)['passed']


def test_budget_is_atomic_and_denies_excess_requests():
    budget = RemoteBudget(3, float('inf'))
    def run(_):
        try:
            budget.consume('test')
            return True
        except TimeoutError:
            return False
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(run, range(20))) == 3
    assert len(budget.calls) == 3


def test_offline_reports_coverage_not_answer_quality_and_never_calls_models(tmp_path, monkeypatch):
    import benchmark_session_memory as benchmark
    def forbidden(**kw):
        raise AssertionError('offline must not instantiate live models')
    monkeypatch.setattr(benchmark, 'ModelSessionSummarizer', forbidden)
    report = evaluate(tmp_path / 'result', turns=20, case_limit=1, warm_repeats=1)
    assert report['status'] == 'completed' and report['remote_requests'] == 0
    assert not report['answer_quality_measured'] and not report['answer_quality_attempted']
    assert len(report['results']) == 3
    assert all(row['history_budget_ok'] for row in report['results'])
    assert all(row['source_text_coverage'] >= row['memory_only_source_text_coverage'] for row in report['results'])
    assert all(row['answer_check'] is None for row in report['results'])
    assert json.loads((tmp_path / 'result/evaluation.json').read_text(encoding='utf-8')) == report


def test_online_missing_credentials_records_preflight_without_network(tmp_path, monkeypatch):
    import benchmark_session_memory as benchmark
    monkeypatch.setattr(benchmark, 'preflight', lambda *_: {'ready': False, 'online': True, 'missing': ['configured_answer_route']})
    report = evaluate(tmp_path / 'result', turns=20, online=True, case_limit=1)
    assert report['status'] == 'configuration_missing'
    assert report['remote_requests'] == 0 and not report['results']


def test_single_case_selection_preserves_original_case_and_gold(tmp_path):
    report = evaluate(tmp_path / 'one', turns=20, case_id='negation', warm_repeats=0)
    assert {row['case'] for row in report['results']} == {'negation'}
    assert len(report['results']) == 3
    assert json.loads((tmp_path / 'one/synthetic_cases.json').read_text(encoding='utf-8')) == [make_cases(20)[1]]


def test_cli_refuses_fake_semantic_offline_results_and_online_concurrency(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(['--embedding', 'semantic', '--output', str(tmp_path / 'no')])
    assert exc.value.code == 2 and not (tmp_path / 'no').exists()
    with pytest.raises(SystemExit):
        main(['--online', '--concurrency'])


def test_online_transport_failure_is_incomplete_and_does_not_leak_private_error(tmp_path, monkeypatch):
    import benchmark_session_memory as benchmark
    monkeypatch.setattr(benchmark, 'preflight', lambda *_: {'ready': True, 'online': True, 'missing': []})
    class FailingModel:
        signature = 'fake'
        def __init__(self, **kw):
            self.request = kw['on_request']
        def __call__(self, payload, **kw):
            self.request()
            raise RuntimeError('private endpoint and key')
    monkeypatch.setattr(benchmark, 'ModelSessionSummarizer', FailingModel)
    report = evaluate(tmp_path / 'result', turns=20, online=True, case_limit=1, max_remote_calls=2)
    assert report['status'] == 'probes_incomplete' and report['remote_requests'] == 2
    assert not report['answer_quality_measured'] and report['answer_quality_attempted']
    assert 'private endpoint' not in json.dumps(report)
