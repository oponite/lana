import json
import hashlib
import importlib.util
import math
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "lana_hf.py"
TOKENIZER = Path(__file__).with_name("wordlevel.json")


SPEC = importlib.util.spec_from_file_location("lana_hf", SCRIPT)
BRIDGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BRIDGE)


def brain_bytes():
    data = b"LBRN1" + struct.pack("<5Q", 3, 1, 1, 1, 7)
    for values in ((1., 2., 3.), (3.,), (4.,), (5., 6., 7.), (7., 8., 9.)):
        data += struct.pack("<Q", len(values)) + struct.pack(f"<{len(values)}f", *values)
    return data


def layered_brain_bytes():
    data = b'LBRN2' + struct.pack('<5Q', 3, 1, 2, 7, 2)
    data += struct.pack('<QB7xQB7x', 2, 1, 1, 2)
    for values in ((1., 2., 3.), (0.2, 0.3), (0., 0.), (0.4, 0.5), (0.,), (1., 2., 3.), (0., 0., 0.)):
        data += struct.pack('<Q', len(values)) + struct.pack(f'<{len(values)}f', *values)
    data += struct.pack('<QQ', 1, 6) + b'memory'
    data += struct.pack('<QfQQ', 1, 0.5, 1, 0)
    return data + hashlib.sha256(data).digest()


class BridgeTest(unittest.TestCase):
    def test_cli_reports_uncertain_durability_after_complete_replacement(self):
        runner = '''
import os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import lana_hf as bridge
activated = [False]
sync = os.fsync
def fail_after_replace(fd):
    if activated[0]: raise OSError('injected directory sync failure')
    return sync(fd)
os.fsync = fail_after_replace
if sys.argv[2] == 'package':
    rename = Path.rename
    def replace_folder(path, target):
        result = rename(path, target)
        if path.name == 'package': activated[0] = True
        return result
    Path.rename = replace_folder
else:
    replace = os.replace
    def replace_file(source, target):
        result = replace(source, target)
        activated[0] = True
        return result
    os.replace = replace_file
bridge.main(sys.argv[2:])
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            brain = root / 'brain.lbrn'
            original = layered_brain_bytes()
            brain.write_bytes(original)
            package = root / 'package'
            result = subprocess.run([sys.executable, '-c', runner, str(ROOT), 'package',
                                     str(brain), str(package), str(TOKENIZER)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(json.loads(result.stderr), {'status': 'error',
                'error': 'destination replaced; durability uncertain; inspect before retry',
                'durability': 'uncertain', 'path': str(package)})
            self.assertTrue(package.is_dir())
            destination = root / 'restored.lbrn'
            for present in (False, True):
                destination.unlink(missing_ok=True)
                if present:
                    destination.write_bytes(brain_bytes())
                result = subprocess.run([sys.executable, '-c', runner, str(ROOT), 'unpackage',
                                         str(package), str(destination)],
                                        capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(json.loads(result.stderr)['durability'], 'uncertain')
                self.assertEqual(json.loads(result.stderr)['path'], str(destination))
                self.assertEqual(destination.read_bytes(), original)
                verified = subprocess.run([sys.executable, str(SCRIPT), 'unpackage',
                                           str(package), str(destination)],
                                          capture_output=True, text=True)
                self.assertEqual(verified.returncode, 0, verified.stderr)
                self.assertEqual(destination.read_bytes(), original)

    def test_package_process_exit_preserves_complete_publications(self):
        runner = '''
import os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import lana_hf as bridge
action, stage = sys.argv[2], int(sys.argv[3])
if action == 'package':
    original = Path.rename
    def rename(path, target):
        if path.name == 'package':
            if stage == 0: os._exit(73)
            result = original(path, target)
            os._exit(73)
        return original(path, target)
    Path.rename = rename
    bridge.package(sys.argv[4], sys.argv[5], sys.argv[6])
else:
    if stage == 0:
        os.fsync = lambda fd: os._exit(73)
    elif stage == 1:
        os.replace = lambda source, target: os._exit(73)
    else:
        original = os.replace
        def replace(source, target):
            original(source, target)
            os._exit(73)
        os.replace = replace
    bridge.unpackage(sys.argv[4], sys.argv[5])
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            brain = root / 'brain.lbrn'
            original = layered_brain_bytes()
            brain.write_bytes(original)
            for stage in (0, 1):
                package = root / 'package'
                result = subprocess.run([sys.executable, '-c', runner, str(ROOT), 'package', str(stage),
                                         str(brain), str(package), str(TOKENIZER)])
                self.assertEqual(result.returncode, 73)
                self.assertEqual(package.exists(), stage == 1)
                if package.exists():
                    restored = root / 'restored.lbrn'
                    BRIDGE.unpackage(package, restored)
                    self.assertEqual(restored.read_bytes(), original)
                    restored.unlink()
                    shutil.rmtree(package)
            BRIDGE.package(brain, root / 'package', TOKENIZER)
            for present in (False, True):
                for stage in (0, 1, 2):
                    destination = root / 'restored.lbrn'
                    destination.unlink(missing_ok=True)
                    if present:
                        destination.write_bytes(brain_bytes())
                    result = subprocess.run([sys.executable, '-c', runner, str(ROOT), 'unpackage', str(stage),
                                             str(root / 'package'), str(destination)])
                    self.assertEqual(result.returncode, 73)
                    expected = original if stage == 2 else brain_bytes() if present else None
                    self.assertEqual(destination.read_bytes() if destination.exists() else None, expected)
                    if expected is not None:
                        BRIDGE.parse_brain(destination.read_bytes())

    def test_atomic_save_reports_uncertain_only_after_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'brain'
            for present in (False, True):
                for after in (False, True):
                    destination.unlink(missing_ok=True)
                    if present:
                        destination.write_bytes(b'old')
                    calls = [None, OSError('sync failed')] if after else [OSError('sync failed')]
                    with patch.object(BRIDGE.os, 'fsync', side_effect=calls):
                        with self.assertRaises(OSError) as error:
                            BRIDGE.atomic_write(destination, layered_brain_bytes())
                    self.assertEqual(isinstance(error.exception, BRIDGE.DurabilityUncertain), after)
                    if after:
                        self.assertEqual(destination.read_bytes(), layered_brain_bytes())
                    elif present:
                        self.assertEqual(destination.read_bytes(), b'old')
                    else:
                        self.assertFalse(destination.exists())

    def test_layered_package_and_import_preserve_snapshot_and_reject_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            brain, tensor, package = root / 'brain', root / 'weights', root / 'package'
            original = layered_brain_bytes()
            brain.write_bytes(original)
            BRIDGE.export_safetensors(brain, tensor)
            BRIDGE.import_safetensors(tensor, brain)
            self.assertEqual(brain.read_bytes(), original)
            BRIDGE.package(brain, package, TOKENIZER)
            restored = root / 'restored'
            BRIDGE.unpackage(package, restored)
            self.assertEqual(restored.read_bytes(), original)
            saved = {path.name: path.read_bytes() for path in package.iterdir()}
            for mutation in ('activation', 'digest', 'snapshot', 'weights', 'offsets', 'tokenizer', 'duplicate', 'extra'):
                for name, data in saved.items():
                    (package / name).write_bytes(data)
                config = json.loads(saved['config.json'])
                if mutation == 'activation':
                    config['hidden'][0]['activation'] = 'gelu'
                elif mutation == 'digest':
                    config['brain_sha256'] = '0' * 64
                elif mutation == 'snapshot':
                    (package / 'brain.lbrn').unlink()
                elif mutation in ('weights', 'offsets'):
                    raw = saved['model.safetensors']
                    length, = struct.unpack_from('<Q', raw)
                    header, body = json.loads(raw[8:8 + length]), raw[8 + length:]
                    if mutation == 'weights':
                        body = struct.pack('<f', 9.0) + body[4:]
                    else:
                        header['hidden.0.bias']['data_offsets'] = [0, 8]
                    encoded = json.dumps(header).encode()
                    raw = struct.pack('<Q', len(encoded)) + encoded + body
                    (package / 'model.safetensors').write_bytes(raw)
                    config['weights_sha256'] = hashlib.sha256(raw).hexdigest()
                elif mutation == 'tokenizer':
                    token = json.loads(saved['tokenizer.json'])
                    token['model']['vocab']['hello'] = 50
                    raw = json.dumps(token).encode()
                    (package / 'tokenizer.json').write_bytes(raw)
                    config['tokenizer_sha256'] = hashlib.sha256(raw).hexdigest()
                elif mutation == 'extra':
                    (package / 'extra').write_bytes(b'')
                raw = json.dumps(config).encode()
                if mutation == 'duplicate':
                    raw = raw[:-1] + b',"seed":7}'
                (package / 'config.json').write_bytes(raw)
                with self.subTest(mutation=mutation), self.assertRaises((ValueError, OSError)):
                    BRIDGE.unpackage(package, restored)
                self.assertEqual(restored.read_bytes(), original)

    def test_layered_loader_checks_digest_reserved_bytes_and_shapes(self):
        original = layered_brain_bytes()
        for offset, replacement in ((45, struct.pack('<Q', 0)), (53, b'\x03'), (54, b'\x01'), (77, struct.pack('<Q', 100))):
            data = bytearray(original[:-32])
            data[offset:offset + len(replacement)] = replacement
            data += hashlib.sha256(data).digest()
            with self.assertRaises(ValueError):
                BRIDGE.parse_brain(data)
        with self.assertRaises(ValueError):
            BRIDGE.parse_brain(original[:-1] + bytes([original[-1] ^ 1]))

    def test_corrupt_tensor_ranges_and_values_preserve_brain(self):
        with tempfile.TemporaryDirectory() as directory:
            brain = Path(directory) / "brain.lbrn"
            tensor = Path(directory) / "model.safetensors"
            original = brain_bytes()
            brain.write_bytes(original)
            BRIDGE.export_safetensors(brain, tensor)
            valid = tensor.read_bytes()
            length, = struct.unpack_from("<Q", valid)
            for mutation in ("overlap", "bounds", "shape", "nonfinite", "duplicate"):
                header = json.loads(valid[8:8 + length])
                body = valid[8 + length:]
                if mutation == "overlap":
                    header["hidden.bias"]["data_offsets"] = header["hidden.weights"]["data_offsets"]
                elif mutation == "bounds":
                    header["hidden.bias"]["data_offsets"] = [len(body), len(body) + 4]
                elif mutation == "shape":
                    header["hidden.bias"]["shape"] = [True]
                elif mutation == "nonfinite":
                    body = struct.pack("<f", math.inf) + body[4:]
                encoded = json.dumps(header).encode()
                if mutation == "duplicate":
                    encoded = encoded[:-1] + b',"hidden.bias":{} }'
                tensor.write_bytes(struct.pack("<Q", len(encoded)) + encoded + body)
                with self.assertRaises((ValueError, TypeError)):
                    BRIDGE.import_safetensors(tensor, brain)
                self.assertEqual(brain.read_bytes(), original)
                self.assertEqual(set(Path(directory).iterdir()), {brain, tensor})

    def test_package_failure_does_not_delete_unrelated_temporary_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            brain = root / "brain.lbrn"
            brain.write_bytes(brain_bytes())
            destination = root / "package"
            destination.mkdir()
            sibling = root / "package.tmp"
            sibling.mkdir()
            marker = sibling / "keep"
            marker.write_text("unrelated")
            with self.assertRaises(ValueError):
                BRIDGE.package(brain, destination, TOKENIZER)
            self.assertEqual(marker.read_text(), "unrelated")

    def test_invalid_package_metadata_preserves_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            brain = root / "brain.lbrn"
            original = brain_bytes()
            brain.write_bytes(original)
            package = root / "package"
            BRIDGE.package(brain, package, TOKENIZER)
            config = json.loads((package / "config.json").read_text())
            config["hidden_width"] += 1
            (package / "config.json").write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                BRIDGE.unpackage(package, brain)
            self.assertEqual(brain.read_bytes(), original)
            self.assertEqual(set(root.iterdir()), {brain, package})

    def test_unsupported_tokenizer_processing_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            config = json.loads(TOKENIZER.read_text())
            config["normalizer"] = {"type": "Lowercase"}
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                BRIDGE.tokenizer(path)

    def test_reference_hugging_face_round_trip(self):
        try:
            from tokenizers import Tokenizer
            from safetensors import safe_open
        except ImportError:
            self.skipTest("reference libraries run in the isolated integration environment")
        reference = Tokenizer.from_file(str(TOKENIZER))
        for text in ("hello lana", "hello, lana", " hello\tlana\n", "hello\x1clana", "hello\u0085lana", "猫", ""):
            output = subprocess.check_output([sys.executable, str(SCRIPT), "tokenize", str(TOKENIZER), text], text=True)
            self.assertEqual(json.loads(output)["token_ids"], reference.encode(text).ids)
        with tempfile.TemporaryDirectory() as directory:
            brain, tensor = Path(directory) / "brain", Path(directory) / "model.safetensors"
            brain.write_bytes(brain_bytes())
            BRIDGE.export_safetensors(brain, tensor)
            with safe_open(tensor, framework="numpy") as model:
                self.assertEqual(set(model.keys()), set(BRIDGE.brain(brain)[1]))
                self.assertEqual(model.get_slice("embedding.weights").get_shape(), [3, 1])

    def test_local_wordlevel_round_trip(self):
        encoded = subprocess.run(
            ["python3", SCRIPT, "tokenize", TOKENIZER, "hello lana"],
            check=True, capture_output=True, text=True,
        )
        self.assertEqual(json.loads(encoded.stdout)["token_ids"], [1, 2])
        decoded = subprocess.run(
            ["python3", SCRIPT, "detokenize", TOKENIZER, "[1,2]"],
            check=True, capture_output=True, text=True,
        )
        self.assertEqual(json.loads(decoded.stdout)["text"], "hello lana")

    def test_safetensors_round_trip_preserves_named_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            brain = Path(directory) / "brain.lbrn"
            original = b"LBRN1" + struct.pack("<5Q", 2, 1, 1, 1, 7)
            for values in ((1.0, 2.0), (3.0,), (4.0,), (5.0, 6.0), (7.0, 8.0)):
                original += struct.pack("<Q", len(values)) + struct.pack(f"<{len(values)}f", *values)
            original += struct.pack("<Q", 1) + struct.pack("<Q", 6) + b"memory"
            brain.write_bytes(original)
            package = Path(directory) / "model.safetensors"
            subprocess.run(["python3", SCRIPT, "export", brain, package], check=True)
            subprocess.run(["python3", SCRIPT, "import", package, brain], check=True)
            self.assertEqual(brain.read_bytes(), original)

    def test_package_round_trip_preserves_brain(self):
        with tempfile.TemporaryDirectory() as directory:
            brain = Path(directory) / "brain.lbrn"
            original = b"LBRN1" + struct.pack("<5Q", 3, 1, 1, 1, 7)
            for values in ((1.0, 2.0, 3.0), (3.0,), (4.0,), (5.0, 6.0, 7.0), (7.0, 8.0, 9.0)):
                original += struct.pack("<Q", len(values)) + struct.pack(f"<{len(values)}f", *values)
            original += struct.pack("<Q", 1) + struct.pack("<Q", 6) + b"memory"
            original += struct.pack("<Q", 1) + struct.pack("<f", 0.5) + struct.pack("<Q", 1)
            brain.write_bytes(original)
            package = Path(directory) / "package"
            subprocess.run(["python3", SCRIPT, "package", brain, package, TOKENIZER], check=True)
            self.assertTrue((package / "config.json").is_file())
            self.assertTrue((package / "generation_config.json").is_file())
            self.assertTrue((package / "README.md").is_file())
            self.assertTrue((package / "tokenizer.json").is_file())
            restored = Path(directory) / "restored.lbrn"
            subprocess.run(["python3", SCRIPT, "unpackage", package, restored], check=True)
            self.assertEqual(restored.read_bytes(), original)

    def test_package_refuses_to_replace_existing_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            brain = Path(directory) / "brain.lbrn"
            brain.write_bytes(b"invalid")
            package = Path(directory) / "package"
            package.mkdir()
            result = subprocess.run(["python3", SCRIPT, "package", brain, package, TOKENIZER], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(package.is_dir())


if __name__ == "__main__":
    unittest.main()
