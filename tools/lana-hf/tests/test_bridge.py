import json
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "lana_hf.py"
TOKENIZER = Path(__file__).with_name("wordlevel.json")


class BridgeTest(unittest.TestCase):
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
            original = b"LBRN1" + struct.pack("<5Q", 2, 1, 1, 1, 7)
            for values in ((1.0, 2.0), (3.0,), (4.0,), (5.0, 6.0), (7.0, 8.0)):
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
