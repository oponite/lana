"""The checked workshop reports exact answers and preserves reports on failure."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

repo = Path(__file__).resolve().parents[1]
lana = str(Path(sys.argv[1]).resolve())
spec = importlib.util.spec_from_file_location('brain_workshop', repo / 'examples/brain/workshop.py')
workshop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workshop)

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    report_path = root / 'report.json'
    result = subprocess.run([str(repo / 'examples/brain/run.sh'), str(root / 'brain'), '--report', str(report_path)],
                            env=dict(os.environ, LANA_BIN=lana), capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(report_path.read_bytes())
    assert report == json.loads(result.stdout)
    assert report['counts'] == dict(supported_correct=2, unsupported_correct=1, ambiguous_correct=1, wrong=0)
    assert report['memory_revision_before'] == report['memory_revision_after'] == '8'
    assert report['forecast_scores'] == [{'id': 'sunny-next', 'observed_label': 'sunny', 'brier_score': 0.125}]
    assert report['decision_cases'] == []
    assert report['brain_sha256'] == hashlib.sha256((root / 'brain').read_bytes()).hexdigest()
    assert report['fixture_sha256'] == hashlib.sha256((workshop.HERE / 'questions.jsonl').read_bytes()).hexdigest()
    paired = report['paired_observation']
    assert len(paired['cases']) == 4 and not paired['held_out'] and not paired['improvement_claim']
    assert paired['ordinary_parameter_sha256'] == paired['grounded_parameter_sha256'] == report['parameter_sha256']
    assert all(case['ordinary_elapsed_ns'] >= 0 and case['grounded_elapsed_ns'] >= 0 for case in paired['cases'])
    assert next(case for case in paired['cases'] if case['id'] == 'weather')['grounded_selected_records'] == 3
    heldout = workshop.workflow(lana, root / 'heldout-brain', workshop.HERE / 'heldout_questions.jsonl', held_out=True)
    observed = heldout['paired_observation']
    assert observed['held_out'] and observed['improvement_claim']
    assert heldout['fixture_sha256'] == workshop.HELDOUT_SHA256
    assert heldout['counts'] == dict(supported_correct=2, unsupported_correct=1, ambiguous_correct=1, wrong=0)
    assert all(case['grounded_correct'] and case['required_evidence_retained'] for case in observed['cases'])
    assert any(not case['ordinary_correct'] for case in observed['cases'])
    assert observed['ordinary_parameter_sha256'] == observed['grounded_parameter_sha256'] == heldout['parameter_sha256']
    assert observed['context_token_cap'] == 4096 and observed['selected_record_cap'] == 64
    decision = heldout['decision_cases'][0]
    assert decision['id'] == 'heldout-weather' and decision['baseline_action'] == 'carry'
    assert decision['recommended'] == ['forecast']
    assert [(item['observation'], item['net_value']) for item in decision['candidates']] == [
        ('forecast', 1.5), ('expensive', -1), ('unknown', None)]
    assert decision['ordinary_action'] == 'carry' and decision['ordinary_utility'] == 0
    assert decision['grounded_action'] == 'leave' and decision['grounded_utility'] == 4
    assert decision['utility_delta'] == 4 and decision['evaluation_elapsed_ns'] >= 0
    changed = root / 'changed-heldout.jsonl'
    changed.write_bytes((workshop.HERE / 'heldout_questions.jsonl').read_bytes() + b'\n')
    try:
        workshop.workflow(lana, root / 'changed-brain', changed, held_out=True)
    except ValueError as error:
        assert 'digest changed' in str(error), error
    else:
        raise AssertionError('changed held-out fixture accepted')
    before = report_path.read_bytes()
    cases = [json.loads(line) for line in (workshop.HERE / 'questions.jsonl').read_text().splitlines()]
    cases[0]['expected_answer'] = 'wrong answer'
    fixture = root / 'wrong.jsonl'
    fixture.write_text(''.join(json.dumps(case) + '\n' for case in cases))
    try:
        workshop.workflow(lana, root / 'bad-brain', fixture, report_path)
    except ValueError as error:
        assert 'wrong=1' in str(error), error
    else:
        raise AssertionError('wrong answer accepted')
    assert report_path.read_bytes() == before

print('BRAIN_WORKSHOP_PASS')
