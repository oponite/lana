import json
import importlib.util
import math
import struct
import subprocess
import sys
import tempfile
import unittest
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


class BridgeTest(unittest.TestCase):
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
