#!/usr/bin/env python3
"""Local WordLevel/SafeTensors bridge. No network or production dependencies."""
import json
import hashlib
import math
import os
import re
import struct
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MAX_BYTES = 256 * 1024 * 1024


class DurabilityUncertain(OSError):
    pass


def read(path):
    with Path(path).open('rb') as source:
        if os.fstat(source.fileno()).st_size > MAX_BYTES:
            raise ValueError('file exceeds 256 MiB')
        data = source.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError('file exceeds 256 MiB')
    return data


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def parse_json(data):
    return json.loads(data, object_pairs_hook=unique_object)


def atomic_write(path, data):
    path = Path(path)
    if len(data) > MAX_BYTES:
        raise ValueError('file exceeds 256 MiB')
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        os.replace(name, path)
        sync_directory(path.parent)
    finally:
        Path(name).unlink(missing_ok=True)


def sync_directory(path):
    if os.name == 'posix':
        try:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError as error:
            raise DurabilityUncertain('destination replaced; durability uncertain; inspect before retry') from error


def tokenizer(path):
    return parse_tokenizer(read(path))


def parse_tokenizer(raw):
    data = parse_json(raw)
    model = data['model']
    if model['type'] != 'WordLevel':
        raise ValueError('unsupported tokenizer model')
    if any(data.get(key) is not None for key in ('normalizer', 'post_processor', 'decoder', 'padding', 'truncation')) or data.get('added_tokens'):
        raise ValueError('unsupported tokenizer processing')
    pre = data.get('pre_tokenizer')
    if pre not in (None, {'type': 'WhitespaceSplit'}):
        raise ValueError('unsupported pre-tokenizer')
    vocab, unknown = model['vocab'], model.get('unk_token', '[UNK]')
    if not isinstance(vocab, dict) or not vocab or unknown not in vocab:
        raise ValueError('tokenizer needs a vocabulary and unknown token')
    if any(type(value) is not int or value < 0 for value in vocab.values()) or len(set(vocab.values())) != len(vocab):
        raise ValueError('token IDs must be distinct nonnegative integers')
    return vocab, unknown, pre is not None


def _product(shape):
    if not isinstance(shape, list) or not shape or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError('invalid parameter shape')
    count = math.prod(shape)
    if count * 4 > MAX_BYTES:
        raise ValueError('parameters exceed 256 MiB')
    return count


def shapes(values):
    vocab, width, hidden = values[:3]
    if len(values) == 6:
        result = [('embedding.weights', [vocab, width])]
        for index, layer in enumerate(values[5]):
            result.extend(((f'hidden.{index}.weights', [layer['width'], width]),
                           (f'hidden.{index}.bias', [layer['width']])))
            width = layer['width']
        return (*result, ('output.weights', [vocab, width]), ('output.bias', [vocab]))
    return (('embedding.weights', [vocab, width]), ('hidden.weights', [hidden, width]),
            ('hidden.bias', [hidden]), ('output.weights', [vocab, hidden]), ('output.bias', [vocab]))


def finite_floats(payload):
    if len(payload) % 4 or any(not math.isfinite(value) for (value,) in struct.iter_unpack('<f', payload)):
        raise ValueError('non-finite or truncated parameters/history')


def parse_brain(data):
    original = data
    layered = data[:5] == b'LBRN2'
    if data[:5] not in (b'LBRN1', b'LBRN2') or len(data) > MAX_BYTES:
        raise ValueError('invalid Lana brain file')
    if layered:
        if len(data) < 77 or hashlib.sha256(data[:-32]).digest() != data[-32:]:
            raise ValueError('invalid Brain digest')
        data = data[:-32]
    values = list(struct.unpack_from('<5Q', data, 5))
    offset = 45
    if layered:
        vocab, width, version, seed, count = values
        if not 1 <= count <= 16:
            raise ValueError('invalid hidden layer count')
        hidden = []
        for _ in range(count):
            size, activation = struct.unpack_from('<QB', data, offset)
            if not 1 <= size <= 4096 or activation not in (1, 2) or data[offset + 9:offset + 16] != bytes(7):
                raise ValueError('invalid hidden layer')
            hidden.append({'width': size, 'activation': 'relu' if activation == 1 else 'gelu'})
            offset += 16
        values = [vocab, width, hidden[-1]['width'], version, seed, hidden]
    expected = shapes(values)
    if sum(_product(shape) for _, shape in expected) * 4 > MAX_BYTES:
        raise ValueError('parameters exceed 256 MiB')
    groups = {}
    for name, shape in expected:
        count, = struct.unpack_from('<Q', data, offset)
        offset += 8
        if count != _product(shape) or count * 4 > len(data) - offset:
            raise ValueError('invalid parameter length or shape')
        payload = data[offset:offset + count * 4]
        finite_floats(payload)
        groups[name] = (shape, payload)
        offset += count * 4
    tail = data[offset:]
    if layered or offset < len(data):
        count, = struct.unpack_from('<Q', data, offset)
        offset += 8
        for _ in range(count):
            size, = struct.unpack_from('<Q', data, offset)
            offset += 8
            if size > len(data) - offset:
                raise ValueError('truncated memory')
            data[offset:offset + size].decode('utf-8')
            offset += size
    if layered or offset < len(data):
        count, = struct.unpack_from('<Q', data, offset)
        offset += 8
        if count * 4 > len(data) - offset:
            raise ValueError('truncated training history')
        finite_floats(data[offset:offset + count * 4])
        offset += count * 4
        struct.unpack_from('<Q', data, offset)
        offset += 8
    if layered:
        typed_length, = struct.unpack_from('<Q', data, offset)
        offset += 8
        if typed_length > len(data) - offset:
            raise ValueError('truncated typed memory')
        if typed_length:
            # Rust owns Core observation and full replay; do not duplicate that law here.
            executable = os.environ.get('LANA_CLI') or shutil.which('lana')
            if not executable:
                raise ValueError('typed memory requires Lana on PATH or LANA_CLI for replay validation')
            with tempfile.TemporaryDirectory(prefix='lana-brain-validation-') as directory:
                snapshot = Path(directory) / 'brain.lbrn'
                snapshot.write_bytes(original)
                checked = subprocess.run([executable, 'brain', 'inspect', str(snapshot)],
                                         capture_output=True, text=True, timeout=120)
                if checked.returncode != 0:
                    raise ValueError('Rust loader rejected typed Brain memory')
        offset += typed_length
    if offset != len(data):
        raise ValueError('unexpected brain trailing data')
    return values, groups, tail


def brain(path):
    return parse_brain(read(path))


def export_safetensors(brain_path, output):
    _, groups, _ = brain(brain_path)
    header, body = {}, bytearray()
    for name, (shape, payload) in groups.items():
        start = len(body)
        body.extend(payload)
        header[name] = {'dtype': 'F32', 'shape': shape, 'data_offsets': [start, len(body)]}
    encoded = json.dumps(header, separators=(',', ':')).encode()
    encoded += b' ' * (-len(encoded) % 8)
    atomic_write(output, struct.pack('<Q', len(encoded)) + encoded + body)


def imported_brain(raw, original):
    values, expected, tail = parse_brain(original)
    header_length, = struct.unpack_from('<Q', raw)
    if header_length > len(raw) - 8:
        raise ValueError('truncated safetensors header')
    header = parse_json(raw[8:8 + header_length])
    if not isinstance(header, dict) or set(header) != set(expected):
        raise ValueError('parameter names differ from the Lana brain')
    body = raw[8 + header_length:]
    groups, ranges = [], []
    for name, (shape, _) in expected.items():
        item = header[name]
        if not isinstance(item, dict) or item.get('dtype') != 'F32' or item.get('shape') != shape:
            raise ValueError('incompatible parameter dtype or shape')
        _product(item.get('shape'))
        start, end = item['data_offsets']
        if type(start) is not int or type(end) is not int or not 0 <= start <= end <= len(body) or end - start != _product(shape) * 4:
            raise ValueError('invalid tensor offsets')
        finite_floats(body[start:end])
        ranges.append((start, end))
        groups.append(body[start:end])
    end = 0
    for start, next_end in sorted(ranges):
        if start != end:
            raise ValueError('overlapping or unreferenced tensor data')
        end = next_end
    if end != len(body):
        raise ValueError('unreferenced tensor data')
    layered = original[:5] == b'LBRN2'
    output = bytearray(original[:45 + (16 * len(values[5]) if layered else 0)])
    for group in groups:
        output.extend(struct.pack('<Q', len(group) // 4))
        output.extend(group)
    output.extend(tail)
    if layered:
        output.extend(hashlib.sha256(output).digest())
    parse_brain(output)
    return output


def import_safetensors(source, brain_path):
    atomic_write(brain_path, imported_brain(read(source), read(brain_path)))


def package_config(values, original, weights, tokenizer_bytes):
    if original[:5] == b'LBRN1':
        return dict(zip(('vocab_size', 'embedding_width', 'hidden_width', 'version', 'seed'), values), model_type='lana-brain')
    return {'package_schema': 'lana-brain-hf-v2', 'brain_format': 'LBRN2',
            'vocab_size': values[0], 'embedding_width': values[1], 'hidden': values[5],
            'brain_version': values[3], 'seed': values[4],
            'brain_sha256': hashlib.sha256(original).hexdigest(),
            'weights_sha256': hashlib.sha256(weights).hexdigest(),
            'tokenizer_sha256': hashlib.sha256(tokenizer_bytes).hexdigest()}


def package(brain_path, folder, tokenizer_path):
    folder = Path(folder)
    if folder.exists() or folder.is_symlink():
        raise ValueError('package destination already exists')
    original = read(brain_path)
    values, _, _ = parse_brain(original)
    tokenizer_bytes = read(tokenizer_path)
    vocab, _, _ = parse_tokenizer(tokenizer_bytes)
    if max(vocab.values()) >= values[0]:
        raise ValueError('tokenizer IDs exceed brain vocabulary')
    folder.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + folder.name + '-', dir=folder.parent) as directory:
        temporary = Path(directory) / 'package'
        temporary.mkdir()
        atomic_write(temporary / 'brain.lbrn', original)
        export_safetensors(temporary / 'brain.lbrn', temporary / 'model.safetensors')
        config = package_config(values, original, read(temporary / 'model.safetensors'), tokenizer_bytes)
        atomic_write(temporary / 'config.json', (json.dumps(config) + '\n').encode())
        atomic_write(temporary / 'generation_config.json', b'{"do_sample":false}\n')
        atomic_write(temporary / 'README.md', b'# Lana Brain\n\nLocal package exported by lana-hf.\n')
        atomic_write(temporary / 'tokenizer.json', tokenizer_bytes)
        if folder.exists() or folder.is_symlink():
            raise ValueError('package destination already exists')
        # ponytail: one writer per destination; use a directory lock if concurrent publishers are required.
        temporary.rename(folder)
        sync_directory(folder.parent)


def unpackage(folder, brain_path):
    folder, brain_path = Path(folder), Path(brain_path)
    config = parse_json(read(folder / 'config.json'))
    if 'package_schema' in config or config.get('brain_format') == 'LBRN2':
        expected_files = {'brain.lbrn', 'model.safetensors', 'config.json',
                          'generation_config.json', 'tokenizer.json', 'README.md'}
        if set(path.name for path in folder.iterdir()) != expected_files or any(
                path.is_symlink() or not path.is_file() for path in folder.iterdir()):
            raise ValueError('invalid package files')
        original = read(folder / 'brain.lbrn')
        if original[:5] != b'LBRN2':
            raise ValueError('invalid package Brain format')
        values, _, _ = parse_brain(original)
        weights, tokenizer_bytes = read(folder / 'model.safetensors'), read(folder / 'tokenizer.json')
        expected = package_config(values, original, weights, tokenizer_bytes)
        if json.dumps(config, sort_keys=True) != json.dumps(expected, sort_keys=True):
            raise ValueError('package metadata or digest mismatch')
        vocab, _, _ = parse_tokenizer(tokenizer_bytes)
        if max(vocab.values()) >= values[0]:
            raise ValueError('tokenizer IDs exceed brain vocabulary')
        if parse_json(read(folder / 'generation_config.json')) != {'do_sample': False}:
            raise ValueError('unsupported generation configuration')
        read(folder / 'README.md').decode('utf-8')
        if imported_brain(weights, original) != original:
            raise ValueError('package weights differ from Brain snapshot')
        atomic_write(brain_path, original)
        return
    if config.get('model_type') != 'lana-brain':
        raise ValueError('unsupported model package')
    values = [config[key] for key in ('vocab_size', 'embedding_width', 'hidden_width', 'version', 'seed')]
    if any(type(value) is not int or not 0 <= value <= 2**64 - 1 for value in values):
        raise ValueError('invalid package dimensions or version')
    vocab, _, _ = tokenizer(folder / 'tokenizer.json')
    if max(vocab.values()) >= values[0]:
        raise ValueError('tokenizer IDs exceed brain vocabulary')
    if parse_json(read(folder / 'generation_config.json')) != {'do_sample': False}:
        raise ValueError('unsupported generation configuration')
    complete = folder / 'brain.lbrn'
    if complete.is_file():
        original = read(complete)
    elif brain_path.exists():
        original = read(brain_path)
    else:
        counts = [_product(shape) for _, shape in shapes(values)]
        if sum(counts) * 4 + 85 > MAX_BYTES:
            raise ValueError('brain exceeds 256 MiB')
        original = b'LBRN1' + struct.pack('<5Q', *values)
        for count in counts:
            original += struct.pack('<Q', count) + b'\0' * (count * 4)
    if parse_brain(original)[0] != values:
        raise ValueError('config and brain metadata differ')
    atomic_write(brain_path, imported_brain(read(folder / 'model.safetensors'), original))


def main(args):
    try:
        command = args[0] if args else ''
        operations = {'export': (export_safetensors, 3), 'import': (import_safetensors, 3),
                      'package': (package, 4), 'unpackage': (unpackage, 3)}
        if command in operations and len(args) == operations[command][1]:
            operations[command][0](*(Path(arg) for arg in args[1:]))
            result = {'status': 'ok', 'path': args[2]}
        elif command in ('tokenize', 'detokenize') and len(args) == 3:
            vocab, unknown, split = tokenizer(Path(args[1]))
            if command == 'tokenize':
                # Match Unicode White_Space; Python split also treats U+001C..001F as spaces.
                words = [word for word in re.split(r'[\t-\r \u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+', args[2]) if word] if split else ([args[2]] if args[2] else [])
                result = {'status': 'ok', 'token_ids': [vocab.get(word, vocab[unknown]) for word in words]}
            else:
                inverse = {value: key for key, value in vocab.items()}
                ids = parse_json(args[2])
                if not isinstance(ids, list) or any(type(value) is not int or value not in inverse for value in ids):
                    raise ValueError('token IDs must be known nonnegative integers')
                result = {'status': 'ok', 'text': ' '.join(inverse[value] for value in ids)}
        else:
            raise ValueError('usage: lana-hf tokenize|detokenize|export|import|package|unpackage ...')
        print(json.dumps(result))
    except (OSError, ValueError, TypeError, KeyError, AttributeError, struct.error, OverflowError, subprocess.SubprocessError) as error:
        result = {'status': 'unsupported', 'error': str(error)}
        if isinstance(error, DurabilityUncertain):
            result.update(status='error', durability='uncertain', path=args[2])
        raise SystemExit(json.dumps(result))


if __name__ == '__main__':
    main(sys.argv[1:])
