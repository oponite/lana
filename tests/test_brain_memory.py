"""Typed Brain memory replays through Core across independent CLI processes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

lana = str(Path(sys.argv[1]).resolve())
repo = Path(__file__).resolve().parents[1]
environment = dict(os.environ, LANA_HF=str(repo / 'tools/lana-hf/lana_hf.py'))

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    model, value, evidence = root / 'brain', root / 'value.json', root / 'evidence.json'

    def call(*args, ok=True):
        result = subprocess.run([lana, 'brain', *map(str, args)], env=environment, capture_output=True, text=True, timeout=60)
        assert (result.returncode == 0) == ok, result.stdout + result.stderr
        return json.loads(result.stdout.splitlines()[-1]) if ok else result

    call('new', model, 3, 2, 4, 7)
    logits = call('evaluate', model, 0, 1)['logits']
    call('remember', model, 'name', 'Lana')
    law = {'tag': 'distribution', 'dependency_id': '7', 'rows': [
        [{'tag': 'bool', 'value': False}, 0.25], [{'tag': 'bool', 'value': True}, 0.75]]}
    value.write_text(json.dumps(law))
    assert call('memory', 'add', model, 'weather', 'sensor', value)['memory_revision'] == '1'
    assert model.read_bytes().startswith(b'LBRN2')
    assert call('evaluate', model, 0, 1)['logits'] == logits
    assert call('recall', model, 'name')['value'] == 'Lana'
    before = model.read_bytes()
    assert not call('memory', 'add', model, 'weather', 'sensor', value)['changed']
    assert model.read_bytes() == before
    inspected = call('memory', 'inspect', model, 'weather')
    assert len(inspected['current_information']['rows']) == 2
    evidence.write_text('{"tag":"bool","value":true}')
    assert call('memory', 'observe', model, 'weather', 'sunny', evidence)['memory_revision'] == '2'
    inspected = call('memory', 'inspect', model, 'weather')
    assert len(inspected['current_information']['rows']) == 1
    assert inspected['derivation_refs'] == ['brain/memory/1', 'brain/memory/2']
    before = model.read_bytes()
    assert not call('memory', 'observe', model, 'weather', 'sunny', evidence)['changed']
    assert model.read_bytes() == before
    evidence.write_text('{"tag":"bool","value":false}')
    call('memory', 'observe', model, 'weather', 'sunny', evidence, ok=False)
    call('memory', 'observe', model, 'weather', 'impossible', evidence, ok=False)
    assert model.read_bytes() == before
    joint = {'tag': 'finite_joint', 'relationship_id': '9', 'names': ['x', 'y'], 'domains': ['bool', 'bool'],
             'rows': [{'values': [{'tag': 'bool', 'value': b}, {'tag': 'bool', 'value': b}], 'weight': 0.5}
                      for b in (False, True)]}
    value.write_text(json.dumps(joint))
    call('memory', 'add', model, 'paired', 'declared joint', value)
    evidence.write_text('{"x":{"tag":"bool","value":true}}')
    call('memory', 'observe', model, 'paired', 'x_seen', evidence)
    inspected = call('memory', 'inspect', model, 'paired')
    assert len(inspected['current_information']['rows']) == 1
    assert inspected['current_information']['rows'][0]['values'][1]['value'] is True
    before = model.read_bytes()
    evidence.write_text('{"x":{"tag":"bool","value":true},"x":{"tag":"bool","value":false}}')
    call('memory', 'observe', model, 'paired', 'duplicate', evidence, ok=False)
    assert model.read_bytes() == before
    copy = root / 'copy'
    call('save', model, copy)
    assert copy.read_bytes() == before
    package = root / 'package'
    call('save', model, package, repo / 'tools/lana-hf/tests/wordlevel.json')
    call('load', package, copy)
    assert copy.read_bytes() == before
    # Repair the outer digest after tampering so rejection must come from replay.
    damaged = before[:-32].replace(b'brain/memory/1', b'brain/memory/9', 1)
    copy.write_bytes(damaged + hashlib.sha256(damaged).digest())
    call('inspect', copy, ok=False)
    assert model.read_bytes() == before

    call('alias', model, 'name', '  What\t is YOUR name?  ')
    saved = model.read_bytes()
    assert not call('alias', model, 'name', 'what is your name?')['changed']
    assert model.read_bytes() == saved
    tokenizer = repo / 'tools/lana-hf/tests/wordlevel.json'
    answer = call('chat', model, tokenizer, 'WHAT is your name?', '--grounded')
    assert answer['answer'] == 'Lana'
    assert answer['resolution'] == 'exact' and answer['evidence_refs'] == ['fact:name']
    assert not answer['unsupported'] and answer['assumptions'] == []
    saved = model.read_bytes()
    assert call('chat', model, tokenizer, 'unknown?', '--grounded')['reason'] == 'no_alias'
    assert model.read_bytes() == saved
    call('alias-root', model, 'paired', 'What is your name?')
    saved = model.read_bytes()
    ambiguous = call('chat', model, tokenizer, 'what is your name?', '--grounded')
    assert ambiguous['resolution'] == 'ambiguous'
    assert ambiguous['evidence_refs'] == ['fact:name', 'root:paired']
    assert ambiguous['answer'] is None and model.read_bytes() == saved
    call('alias-root', model, 'weather', 'Was it sunny?')
    exact = call('chat', model, tokenizer, 'Was it sunny?', '--grounded')
    assert exact['answer'] == {'tag': 'bool', 'value': True}
    assert exact['evidence_refs'] == ['brain/memory/2']
    assert [entry['target_id'] for entry in exact['selected_context']] == ['weather', 'weather']
    assert exact['selected_context'][1]['observation_ids'] == ['sunny']
    value.write_text(json.dumps(law))
    call('memory', 'add', model, 'uncertain', 'sensor', value)
    call('alias-root', model, 'uncertain', 'Tomorrow?')
    saved = model.read_bytes()
    unresolved = call('chat', model, tokenizer, 'Tomorrow?', '--grounded')
    assert unresolved['reason'] == 'unresolved_value'
    assert unresolved['evidence_refs'] and unresolved['selected_context'][0]['value']['tag'] == 'distribution'
    assert model.read_bytes() == saved
    call('chat', model, tokenizer, ' \t ', '--grounded', ok=False)
    assert model.read_bytes() == saved
    call('save', model, root / 'grounded-package', tokenizer)
    call('load', root / 'grounded-package', copy)
    assert copy.read_bytes() == saved

    # Equal marginals remain independent roots; only a declared joint propagates evidence.
    for root_id in ('left', 'right'):
        value.write_text(json.dumps({'tag': 'possibility', 'dependency_id': '7',
                                    'support': [{'tag': 'bool', 'value': b} for b in (False, True)]}))
        call('memory', 'add', model, root_id, 'separate source', value)
        call('alias-root', model, root_id, root_id + '?')
    right = call('memory', 'inspect', model, 'right')
    saved = model.read_bytes()
    unresolved = call('chat', model, tokenizer, 'right?', '--grounded')
    assert unresolved['resolution'] == 'unsupported' and unresolved['answer'] is None
    assert unresolved['selected_context'][0]['value']['tag'] == 'possibility'
    assert model.read_bytes() == saved
    evidence.write_text('{"tag":"bool","value":true}')
    call('memory', 'observe', model, 'left', 'left_seen', evidence)
    assert call('memory', 'inspect', model, 'right')['current_information'] == right['current_information']
    assert call('chat', model, tokenizer, 'left?', '--grounded')['resolution'] == 'exact'
    assert call('chat', model, tokenizer, 'right?', '--grounded')['reason'] == 'unresolved_value'
    call('alias-root', model, 'paired', 'paired?')
    assert call('chat', model, tokenizer, 'paired?', '--grounded')['resolution'] == 'exact'
    saved = model.read_bytes()
    evidence.write_text('{"tag":"bool","value":false}')
    call('memory', 'observe', model, 'left', 'contradiction', evidence, ok=False)
    assert model.read_bytes() == saved
    assert call('chat', model, tokenizer, 'missing?', '--grounded')['resolution'] == 'unsupported'
    assert model.read_bytes() == saved

    forecast = root / 'forecast.json'
    forecast.write_text(json.dumps({'created_at': 10, 'labels': ['yes', 'no'],
                                    'probabilities': [0.75, 0.25], 'evidence_refs': ['weather']}))
    previous_revision = int(call('inspect', model)['typed_memory_revision'])
    added = call('forecast', 'add', model, 'rain-1', 'rain', 20, forecast)
    assert added['changed'] and int(added['memory_revision']) == previous_revision + 1, added
    before = model.read_bytes()
    assert not call('forecast', 'add', model, 'rain-1', 'rain', 20, forecast)['changed']
    assert model.read_bytes() == before
    call('forecast', 'score', model, 'rain-1', 'yes', 19, ok=False)
    assert model.read_bytes() == before
    scored = call('forecast', 'score', model, 'rain-1', 'yes', 21)
    assert scored['changed'] and scored['score'] == 0.125, scored
    scored_bytes = model.read_bytes()
    assert not call('forecast', 'score', model, 'rain-1', 'yes', 21)['changed']
    call('forecast', 'score', model, 'rain-1', 'no', 21, ok=False)
    assert model.read_bytes() == scored_bytes
    call('save', model, copy)
    assert copy.read_bytes() == scored_bytes
    scored_package = root / 'scored-package'
    call('save', model, scored_package, tokenizer)
    call('load', scored_package, copy)
    assert copy.read_bytes() == scored_bytes
    forecast.write_text(json.dumps({'created_at': 10, 'labels': ['yes', 'no'],
                                    'probabilities': [0.75, 0.25], 'evidence_refs': ['missing']}))
    call('forecast', 'add', model, 'rain-2', 'rain', 20, forecast, ok=False)
    assert model.read_bytes() == scored_bytes
    for values in ([0.75, 0.20], [float('nan'), 0.25], [-0.1, 1.1]):
        forecast.write_text(json.dumps({'created_at': 10, 'labels': ['yes', 'no'],
                                        'probabilities': values, 'evidence_refs': ['weather']}))
        call('forecast', 'add', model, 'rain-2', 'rain', 20, forecast, ok=False)
        assert model.read_bytes() == scored_bytes

print('BRAIN_MEMORY_PASS')
