#!/usr/bin/env python3
"""Local Hugging Face WordLevel tokenizer bridge; never accesses the network."""

import json
import shutil
import struct
import sys
from pathlib import Path


def tokenizer(path: Path):
    try:
        data = json.loads(path.read_text())
        model = data["model"]
        if model["type"] != "WordLevel":
            raise ValueError("unsupported tokenizer model; install a dedicated bridge for this model")
        return model["vocab"], model.get("unk_token", "[UNK]")
    except (OSError, KeyError, json.JSONDecodeError, ValueError) as error:
        raise SystemExit(json.dumps({"status": "unsupported", "error": str(error)}))


def brain(path: Path):
    data = path.read_bytes()
    if data[:5] != b"LBRN1":
        raise ValueError("not a Lana brain file")
    offset = 5
    values = list(struct.unpack_from("<5Q", data, offset))
    offset += 40
    vocab, width, hidden, version, seed = values
    groups = {}
    for name, shape in (("embedding.weights", [vocab, width]), ("hidden.weights", [hidden, width]),
                        ("hidden.bias", [hidden]), ("output.weights", [vocab, hidden]), ("output.bias", [vocab])):
        count, = struct.unpack_from("<Q", data, offset)
        offset += 8
        if count != _product(shape):
            raise ValueError(f"brain group {name} has an invalid shape")
        groups[name] = (shape, data[offset:offset + count * 4])
        offset += count * 4
    return values, groups, data[offset:]


def _product(shape):
    result = 1
    for value in shape:
        result *= value
    return result


def export_safetensors(brain_path: Path, output: Path):
    _, groups, _ = brain(brain_path)
    offset = 0
    header = {}
    body = bytearray()
    for name, (shape, payload) in groups.items():
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [offset, offset + len(payload)]}
        offset += len(payload)
        body.extend(payload)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(struct.pack("<Q", len(encoded)) + encoded + body)
    temporary.replace(output)


def import_safetensors(source: Path, brain_path: Path):
    values, expected, tail = brain(brain_path)
    raw = source.read_bytes()
    if len(raw) < 8:
        raise ValueError("truncated safetensors file")
    header_length, = struct.unpack_from("<Q", raw)
    try:
        header = json.loads(raw[8:8 + header_length])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid safetensors header") from error
    body = raw[8 + header_length:]
    if set(header) != set(expected):
        raise ValueError("parameter names differ from the Lana brain")
    groups = []
    for name, (shape, _) in expected.items():
        item = header[name]
        if item.get("dtype") != "F32" or item.get("shape") != shape:
            raise ValueError(f"parameter {name} has incompatible dtype or shape")
        start, end = item.get("data_offsets", [None, None])
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end - start != _product(shape) * 4:
            raise ValueError(f"parameter {name} has invalid offsets")
        groups.append(body[start:end])
    if sum(len(group) for group in groups) != len(body):
        raise ValueError("safetensors has overlapping or unreferenced data")
    output = bytearray(b"LBRN1" + struct.pack("<5Q", *values))
    for group in groups:
        output.extend(struct.pack("<Q", len(group) // 4))
        output.extend(group)
    output.extend(tail)
    temporary = brain_path.with_suffix(brain_path.suffix + ".tmp")
    temporary.write_bytes(output)
    temporary.replace(brain_path)


def package(brain_path: Path, folder: Path, tokenizer_path: Path):
    values, groups, _ = brain(brain_path)
    if not tokenizer_path.is_file():
        raise ValueError("tokenizer.json is required")
    temporary = folder.with_name(folder.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    if folder.exists():
        raise ValueError("package destination already exists")
    temporary.mkdir(parents=True)
    export_safetensors(brain_path, temporary / "model.safetensors")
    config = {"model_type": "lana-brain", "vocab_size": groups["output.bias"][0][0],
              "embedding_width": groups["embedding.weights"][0][1], "hidden_width": groups["hidden.bias"][0][0],
              "version": values[3], "seed": values[4]}
    (temporary / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (temporary / "generation_config.json").write_text(json.dumps({"do_sample": False}, indent=2) + "\n")
    (temporary / "README.md").write_text("# Lana Brain\n\nLocal package exported by lana-hf.\n")
    shutil.copyfile(tokenizer_path, temporary / "tokenizer.json")
    shutil.copyfile(brain_path, temporary / "brain.lbrn")
    temporary.replace(folder)


def unpackage(folder: Path, brain_path: Path):
    config = json.loads((folder / "config.json").read_text())
    if config.get("model_type") != "lana-brain":
        raise ValueError("unsupported model package")
    temporary = brain_path.with_suffix(brain_path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    complete = folder / "brain.lbrn"
    if complete.is_file():
        shutil.copyfile(complete, temporary)
    elif not brain_path.exists():
        vocab, width, hidden = (config.get("vocab_size"), config.get("embedding_width"), config.get("hidden_width"))
        if not all(isinstance(value, int) and value > 0 for value in (vocab, width, hidden)):
            raise ValueError("invalid brain package dimensions")
        values = (vocab, width, hidden, config.get("version", 1), config.get("seed", 0))
        groups = (vocab * width, hidden * width, hidden, vocab * hidden, vocab)
        payload = bytearray(b"LBRN1" + struct.pack("<5Q", *values))
        for count in groups:
            payload.extend(struct.pack("<Q", count))
            payload.extend(b"\0" * (count * 4))
        temporary.write_bytes(payload)
    else:
        shutil.copyfile(brain_path, temporary)
    try:
        import_safetensors(folder / "model.safetensors", temporary)
        temporary.replace(brain_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(args):
    if args[:1] == ["export"] and len(args) == 3:
        try:
            export_safetensors(Path(args[1]), Path(args[2]))
            print(json.dumps({"status": "ok", "path": args[2]}))
        except (OSError, ValueError) as error:
            raise SystemExit(json.dumps({"status": "unsupported", "error": str(error)}))
        return
    if args[:1] == ["import"] and len(args) == 3:
        try:
            import_safetensors(Path(args[1]), Path(args[2]))
            print(json.dumps({"status": "ok", "path": args[2]}))
        except (OSError, ValueError, struct.error) as error:
            raise SystemExit(json.dumps({"status": "unsupported", "error": str(error)}))
        return
    if args[:1] == ["package"] and len(args) == 4:
        try:
            package(Path(args[1]), Path(args[2]), Path(args[3]))
            print(json.dumps({"status": "ok", "path": args[2]}))
        except (OSError, ValueError, struct.error) as error:
            raise SystemExit(json.dumps({"status": "unsupported", "error": str(error)}))
        return
    if args[:1] == ["unpackage"] and len(args) == 3:
        try:
            unpackage(Path(args[1]), Path(args[2]))
            print(json.dumps({"status": "ok", "path": args[2]}))
        except (OSError, ValueError, struct.error) as error:
            raise SystemExit(json.dumps({"status": "unsupported", "error": str(error)}))
        return
    if len(args) < 3 or args[0] not in {"tokenize", "detokenize"}:
        raise SystemExit("usage: lana-hf tokenize|detokenize tokenizer.json text|token-ids; lana-hf export|import brain safetensors")
    vocab, unknown = tokenizer(Path(args[1]))
    if args[0] == "tokenize":
        unknown_id = vocab.get(unknown)
        if unknown_id is None:
            raise SystemExit(json.dumps({"status": "unsupported", "error": "tokenizer has no unknown token"}))
        ids = [vocab.get(token, unknown_id) for token in args[2].split()]
        print(json.dumps({"status": "ok", "token_ids": ids}))
        return
    try:
        inverse = {value: key for key, value in vocab.items()}
        ids = json.loads(args[2])
        if not isinstance(ids, list) or any(not isinstance(value, int) for value in ids):
            raise ValueError("token IDs must be a JSON integer array")
        print(json.dumps({"status": "ok", "text": " ".join(inverse.get(value, unknown) for value in ids)}))
    except (json.JSONDecodeError, ValueError) as error:
        raise SystemExit(json.dumps({"status": "unsupported", "error": str(error)}))


if __name__ == "__main__":
    main(sys.argv[1:])
