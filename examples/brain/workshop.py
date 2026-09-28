"""Local Brain workshop with exact-answer checks and atomic report publication."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time

HERE = Path(__file__).resolve().parent
HELDOUT_SHA256 = '52186c30c85adf13f68e5d008f56354c049a06f8ccbc22e493c9398ad3b63bcb'
spec = importlib.util.spec_from_file_location('lana_hf', HERE.parents[1] / 'tools/lana-hf/lana_hf.py')
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


def decision_report(lana, decision, source):
    expected = {'states', 'observations', 'actions', 'utility', 'costs', 'answer_states',
                'observed_state', 'expected_recommended', 'expected_unscorable'}
    if not isinstance(decision, dict) or set(decision) != expected:
        raise ValueError('invalid decision fixture')
    values = [json.dumps(decision[key], ensure_ascii=False, allow_nan=False) for key in
              ('states', 'observations', 'actions', 'utility', 'costs')]
    source.write_text('import "std/decision" as decision;\n'
                      f'let result = decision.value_of_information({", ".join(values)});\n'
                      'print(array_length(result.baseline.actions));\n'
                      'print(result.baseline.actions[0]);\n'
                      'print(result.baseline.expected_utility);\n'
                      'print(array_length(result.recommended));\n'
                      'let index = 0;\n'
                      'while (index < array_length(result.recommended)) {\n'
                      '    print(result.recommended[index]);\n'
                      '    index = index + 1;\n'
                      '}\n'
                      'print(array_length(result.candidates));\n'
                      'index = 0;\n'
                      'while (index < array_length(result.candidates)) {\n'
                      '    print(result.candidates[index].observation);\n'
                      '    print(result.candidates[index].status);\n'
                      '    print(result.candidates[index].net_value);\n'
                      '    index = index + 1;\n'
                      '}\n')
    start = time.perf_counter_ns()
    environment = dict(os.environ, LANA_STDLIB_DIR=os.environ.get('LANA_STDLIB_DIR', str(HERE.parents[1] / 'stdlib')))
    result = subprocess.run([str(lana), 'run', str(source)], env=environment,
                            capture_output=True, text=True, timeout=120)
    elapsed_ns = time.perf_counter_ns() - start
    if result.returncode:
        raise ValueError(result.stderr.strip() or 'decision evaluation failed')
    lines = iter(result.stdout.splitlines())
    try:
        baseline_count = int(next(lines))
        baseline_action = next(lines)
        baseline_value = float(next(lines))
        recommended = [next(lines) for _ in range(int(next(lines)))]
        candidates = [dict(observation=next(lines), status=next(lines),
                           net_value=json.loads(next(lines))) for _ in range(int(next(lines)))]
    except (StopIteration, ValueError, json.JSONDecodeError) as error:
        raise ValueError('invalid decision evaluation output') from error
    if next(lines, None) is not None or baseline_count != 1 or not math.isfinite(baseline_value) or \
            recommended != decision['expected_recommended'] or \
            [candidate['observation'] for candidate in candidates if candidate['status'] != 'ranked'] != decision['expected_unscorable'] or \
            any(candidate['status'] == 'ranked' and
                (not isinstance(candidate['net_value'], (int, float)) or not math.isfinite(candidate['net_value']))
                for candidate in candidates):
        raise ValueError('wrong=1: decision recommendation mismatch')
    return dict(baseline_action=baseline_action, baseline_expected_utility=baseline_value,
                recommended=recommended, candidates=candidates, evaluation_elapsed_ns=elapsed_ns)


def decision_outcomes(decision, measured, ordinary, grounded):
    actions = decision['actions']
    observed_state = decision['observed_state']
    utility = {(row['action'], row['state']): row['value'] for row in decision['utility']}
    states = {row['state'] for row in decision['states']}
    if not isinstance(actions, list) or measured['baseline_action'] not in actions or \
            len(utility) != len(actions) * len(decision['states']) or \
            observed_state not in states or any((action, observed_state) not in utility for action in actions) or \
            any(row['state'] not in states or not isinstance(row['answer'], dict) for row in decision['answer_states']):
        raise ValueError('invalid decision utility fixture')

    def action_for(answer):
        state = next((row['state'] for row in decision['answer_states'] if row['answer'] == answer), None)
        if state is None:
            return measured['baseline_action']
        return max(actions, key=lambda action: utility[action, state])

    before_action = action_for(ordinary)
    after_action = action_for(grounded)
    before = utility[before_action, observed_state]
    after = utility[after_action, observed_state]
    return dict(**measured, observed_state=observed_state, ordinary_action=before_action,
                grounded_action=after_action, ordinary_utility=before,
                grounded_utility=after, utility_delta=after - before)


def workflow(lana, brain, fixture, report_path=None, held_out=False):
    environment = dict(os.environ, LANA_HF=os.environ.get('LANA_HF', str(Path(bridge.__file__))))
    tokenizer = HERE / 'tokenizer.json'
    fixture_bytes = bridge.read(fixture)
    fixture_sha256 = hashlib.sha256(fixture_bytes).hexdigest()
    if held_out and fixture_sha256 != HELDOUT_SHA256:
        raise ValueError('held-out fixture digest changed')
    fixtures = [bridge.parse_json(line) for line in fixture_bytes.splitlines()]
    fields = {'id', 'question', 'expected_resolution', 'expected_answer', 'expected_evidence_ids', 'expected_reason'}
    ids = set()
    forecast_ids = set()
    for case in fixtures:
        if not isinstance(case, dict) or not fields <= set(case) <= fields | {'forecast', 'decision'} or not isinstance(case['id'], str) or not case['id'] or case['id'] in ids:
            raise ValueError('invalid or duplicate fixture ID')
        ids.add(case['id'])
        if not isinstance(case['question'], str) or case['expected_resolution'] not in ('exact', 'unsupported', 'ambiguous'):
            raise ValueError('invalid fixture question or resolution')
        evidence = case['expected_evidence_ids']
        if not isinstance(evidence, list) or any(not isinstance(item, str) for item in evidence) or len(set(evidence)) != len(evidence):
            raise ValueError('invalid fixture evidence')
        exact = case['expected_resolution'] == 'exact'
        if (exact and (case['expected_answer'] is None or case['expected_reason'] is not None)) or (
                not exact and (case['expected_answer'] is not None or not isinstance(case['expected_reason'], str) or not case['expected_reason'])):
            raise ValueError('invalid fixture answer or reason')
        if 'forecast' in case:
            forecast = case['forecast']
            expected = {'id', 'target', 'created_at', 'horizon', 'labels', 'probabilities',
                        'evidence_refs', 'observed_label', 'observed_at', 'expected_brier'}
            if not isinstance(forecast, dict) or set(forecast) != expected or not isinstance(forecast['id'], str) or not forecast['id'] or forecast['id'] in forecast_ids:
                raise ValueError('invalid or duplicate forecast fixture')
            forecast_ids.add(forecast['id'])
            if not isinstance(forecast['expected_brier'], (int, float)) or not math.isfinite(forecast['expected_brier']):
                raise ValueError('invalid expected forecast score')
    if not fixtures:
        raise ValueError('empty fixture')
    if held_out:
        development_ids = {bridge.parse_json(line)['id'] for line in bridge.read(HERE / 'questions.jsonl').splitlines()}
        if ids & development_ids:
            raise ValueError('held-out fixture reuses a development ID')

    def call(*args):
        result = subprocess.run([str(lana), 'brain', *map(str, args)], env=environment,
                                capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise ValueError(result.stderr.strip() or 'Brain command failed')
        return bridge.parse_json(result.stdout.splitlines()[-1])

    call('new', brain, 3, 2, 2, 7)
    trained = call('train', brain, 2, 0.1, 0, 1)
    loss = trained['loss']
    if not isinstance(loss, (int, float)) or not math.isfinite(loss):
        raise ValueError('invalid next-token loss')
    call('remember', brain, 'name', 'Lana')
    with tempfile.TemporaryDirectory(prefix='lana-workshop-input-') as temporary:
        value = Path(temporary) / 'value.json'
        value.write_text('{"tag":"possibility","dependency_id":"1","support":[{"tag":"bool","value":false},{"tag":"bool","value":true}]}')
        call('memory', 'add', brain, 'weather', 'workshop observation', value)
        value.write_text('{"tag":"bool","value":true}')
        call('memory', 'observe', brain, 'weather', 'sunny', value)
    call('alias', brain, 'name', 'What is your name?')
    call('alias-root', brain, 'weather', 'Was it sunny?')
    call('alias', brain, 'name', 'Which record?')
    call('alias-root', brain, 'weather', 'Which record?')
    forecast_scores = []
    with tempfile.TemporaryDirectory(prefix='lana-workshop-forecast-') as temporary:
        declaration = Path(temporary) / 'forecast.json'
        for case in fixtures:
            if 'forecast' not in case:
                continue
            forecast = case['forecast']
            declaration.write_text(json.dumps({key: forecast[key] for key in
                                               ('created_at', 'labels', 'probabilities', 'evidence_refs')}))
            call('forecast', 'add', brain, forecast['id'], forecast['target'], forecast['horizon'], declaration)
            scored = call('forecast', 'score', brain, forecast['id'], forecast['observed_label'], forecast['observed_at'])
            if abs(scored['score'] - forecast['expected_brier']) > 1e-12:
                raise ValueError('wrong=1: forecast score mismatch')
            forecast_scores.append(dict(id=forecast['id'], observed_label=forecast['observed_label'],
                                        brier_score=scored['score']))
    before = call('inspect', brain)
    original = bridge.read(brain)
    call('save', brain, str(brain) + '.hf', tokenizer)
    call('load', str(brain) + '.hf', brain)
    after = call('inspect', brain)
    if original != bridge.read(brain) or before['typed_memory_revision'] != after['typed_memory_revision']:
        raise ValueError('wrong=1: reload mismatch')
    if call('recall', brain, 'name')['value'] != 'Lana':
        raise ValueError('wrong=1: recalled fact mismatch')
    counts = dict(supported_correct=0, unsupported_correct=0, ambiguous_correct=0, wrong=0)
    cases = []
    paired = []
    decision_cases = []
    with tempfile.TemporaryDirectory(prefix='lana-brain-pair-') as temporary:
        ordinary_brain = Path(temporary) / 'ordinary.lbrn'
        ordinary_brain.write_bytes(original)
        for case in fixtures:
            start = time.perf_counter_ns()
            ordinary = call('chat', ordinary_brain, tokenizer, case['question'])
            ordinary_elapsed_ns = time.perf_counter_ns() - start
            start = time.perf_counter_ns()
            actual = call('chat', brain, tokenizer, case['question'], '--grounded')
            grounded_elapsed_ns = time.perf_counter_ns() - start
            context_ids = {reference for entry in actual.get('selected_context', [])
                           for field in ('sources', 'observation_ids', 'derivation_refs', 'evidence_refs')
                           for reference in entry.get(field, [])}
            evidence_retained = case['expected_resolution'] != 'exact' or \
                all(reference in context_ids for reference in case['expected_evidence_ids'])
            correct = all(actual[key] == case[expected] for key, expected in (
                ('resolution', 'expected_resolution'), ('answer', 'expected_answer'),
                ('evidence_refs', 'expected_evidence_ids'), ('reason', 'expected_reason'))) and evidence_retained
            bucket = {'exact': 'supported_correct', 'unsupported': 'unsupported_correct', 'ambiguous': 'ambiguous_correct'}
            counts[bucket[actual['resolution']] if correct else 'wrong'] += 1
            cases.append(dict(id=case['id'], actual_resolution=actual['resolution'], actual_answer=actual['answer'],
                              actual_evidence_ids=actual['evidence_refs'], actual_reason=actual['reason'], correct=correct))
            paired.append(dict(id=case['id'], ordinary_response=ordinary['response'], ordinary_evidence_ids=[],
                               ordinary_elapsed_ns=ordinary_elapsed_ns, grounded_elapsed_ns=grounded_elapsed_ns,
                               grounded_selected_records=len(actual.get('selected_context', [])),
                               required_evidence_retained=evidence_retained,
                               ordinary_correct=(case['expected_resolution'] == 'exact' and
                                                 ordinary['response'] == case['expected_answer']),
                               grounded_correct=correct))
            if 'decision' in case:
                measured = decision_report(lana, case['decision'], Path(temporary) / 'decision.lana')
                outcome = decision_outcomes(case['decision'], measured, ordinary['response'], actual['answer'] if actual['resolution'] == 'exact' else None)
                decision_cases.append(dict(id=case['id'], **outcome))
                paired[-1]['decision'] = outcome
        ordinary_parameters = call('inspect', ordinary_brain)['parameter_sha256']
    if counts['wrong']:
        raise ValueError(f"wrong={counts['wrong']}: grounded answer mismatch")
    grounded_parameters = call('inspect', brain)['parameter_sha256']
    if ordinary_parameters != before['parameter_sha256'] or grounded_parameters != before['parameter_sha256']:
        raise ValueError('wrong=1: grounded answers changed parameters')
    improved = held_out and all(item['grounded_correct'] and item['required_evidence_retained'] for item in paired) and \
        all(item['grounded_selected_records'] <= 64 for item in paired) and \
        all(item['utility_delta'] >= 0 for item in decision_cases) and \
        (any(item['grounded_correct'] and not item['ordinary_correct'] for item in paired) or
         any(item['utility_delta'] > 0 for item in decision_cases))
    report = dict(schema_version=1, brain_sha256=hashlib.sha256(bridge.read(brain)).hexdigest(),
                  tokenizer_sha256=hashlib.sha256(bridge.read(tokenizer)).hexdigest(),
                  fixture_sha256=fixture_sha256, parameter_sha256=before['parameter_sha256'],
                  memory_revision_before=before['typed_memory_revision'], memory_revision_after=after['typed_memory_revision'],
                  cases=cases, counts=counts, next_token_loss=loss, forecast_scores=forecast_scores,
                  decision_cases=decision_cases,
                  paired_observation=dict(method='perf_counter_ns around separate CLI calls',
                                          starting_brain_sha256=hashlib.sha256(original).hexdigest(),
                                          ordinary_parameter_sha256=ordinary_parameters,
                                          grounded_parameter_sha256=grounded_parameters,
                                          context_token_cap=4096, selected_record_cap=64,
                                          held_out=held_out, improvement_claim=improved,
                                          scope=('unseen question text for fixed setup targets and one declared decision' if held_out else 'setup questions'),
                                          cases=paired))
    if report_path is not None:
        bridge.atomic_write(report_path, (json.dumps(report, ensure_ascii=False, sort_keys=True) + '\n').encode())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('brain', nargs='?', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--held-out', action='store_true')
    args = parser.parse_args()
    lana = os.environ.get('LANA_BIN', 'lana')
    try:
        if args.brain is None:
            with tempfile.TemporaryDirectory(prefix='lana-brain-') as directory:
                result = workflow(lana, Path(directory) / 'brain.lbrn',
                                  HERE / ('heldout_questions.jsonl' if args.held_out else 'questions.jsonl'),
                                  args.report, args.held_out)
        else:
            result = workflow(lana, args.brain,
                              HERE / ('heldout_questions.jsonl' if args.held_out else 'questions.jsonl'),
                              args.report, args.held_out)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise SystemExit(str(error))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
