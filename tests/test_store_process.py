"""Public store APIs across processes: locks, killed writers, and torn journals."""
import argparse
from contextlib import contextmanager, ExitStack
import itertools
import hashlib
import os
from pathlib import Path
import select
import signal
import struct
import subprocess
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Probe:
    def __init__(self, binary, path, timeout):
        self.finished = False
        self.started_at = time.monotonic()
        self.process = subprocess.Popen([str(binary), "--probe", str(path), str(timeout)],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, bufsize=0)
        self.expect("ATTEMPT")

    def line(self, timeout=5):
        deadline = time.monotonic() + timeout
        data = bytearray()
        while not data.endswith(b"\n"):
            ready, _, _ = select.select([self.process.stdout], [], [], max(0, deadline - time.monotonic()))
            if not ready:
                raise AssertionError(f"probe {self.process.pid} timed out: {data!r}")
            byte = os.read(self.process.stdout.fileno(), 1)
            if not byte:
                raise AssertionError(f"probe exited {self.process.poll()} before newline: {data!r}")
            data.extend(byte)
        return data.decode().rstrip("\n")

    def expect(self, expected):
        actual = self.line()
        assert actual == expected, (actual, expected)

    def command(self, command, expected):
        self.process.stdin.write((command + "\n").encode())
        self.expect(expected)

    def finish(self, code=0):
        if self.finished:
            return
        if self.process.poll() is None and code == 0:
            self.process.stdin.write(b"exit\n")
        stdout, stderr = self.process.communicate(timeout=5)
        assert (self.process.returncode, stdout, stderr) == (code, b"", b""), (self.process.returncode, stdout, stderr)
        self.finished = True


@contextmanager
def probe(binary, path, timeout=1000, opened=True):
    client = Probe(binary, path, timeout)
    try:
        if opened:
            client.expect("OPEN")
        yield client
        client.finish()
    finally:
        if client.process.poll() is None:
            client.process.kill()
        client.process.communicate(timeout=5)


class StoreProcesses(unittest.TestCase):
    binaries = []

    def test_valid_checksums_do_not_admit_impossible_counts_or_embedded_nuls(self):
        value = b"42\0"
        payloads = [(0xffffffff, b"D\0\0\0\x01k"),
                    (1, b"D\0\0\0\x03k\0x"),
                    (1, b"P\0\0\0\x03key" + hashlib.sha256(value).digest() + struct.pack(">Q", len(value)) + value)]
        for binary, (count, payload) in itertools.product(self.binaries, payloads):
            with self.subTest(binary=binary, count=count, payload=payload), tempfile.TemporaryDirectory(prefix="lana-malformed-store-") as directory:
                path = Path(directory) / "db"
                with probe(binary, path) as first:
                    first.command("put 41", "OK")
                    first.command("commit", "REV 1")
                digest = hashlib.sha256(struct.pack(">QQI", 1, 0, count) + payload).digest()
                journal = (b"LREV" + struct.pack(">QQIQ", 1, 0, count, len(payload)) + digest + payload
                           + b"LCMT" + struct.pack(">Q", 1) + digest)
                (path / "journal").write_bytes(journal)
                with probe(binary, path, opened=False) as rejected:
                    rejected.expect("ERR LANA_ERR_CORRUPTION")
                    rejected.finish(1)
                self.assertEqual((path / "journal").read_bytes(), journal)

    def test_lock_timeout_and_close_release_before_process_exit(self):
        for owner, waiter in itertools.product(self.binaries, repeat=2):
            with self.subTest(owner=owner, waiter=waiter), tempfile.TemporaryDirectory(prefix="lana-lock-") as directory:
                path = Path(directory) / "db"
                with probe(owner, path) as first:
                    first.command("put 41", "OK")
                    first.command("commit", "REV 1")
                    before = {p.name: p.read_bytes() for p in path.iterdir()}
                    with probe(waiter, path, 50, opened=False) as timed:
                        timed.expect("ERR LANA_ERR_TIMEOUT")
                        self.assertGreaterEqual(time.monotonic() - timed.started_at, 0.025)
                        timed.finish(1)
                    self.assertEqual(before, {p.name: p.read_bytes() for p in path.iterdir()})
                    with probe(waiter, path, 0, opened=False) as blocking:
                        self.assertEqual(select.select([blocking.process.stdout], [], [], 0.05)[0], [])
                        first.command("stage", "OK")
                        first.command("close", "OK")
                        self.assertIsNone(first.process.poll())
                        first.command("get", "ERR LANA_ERR_INVALID_STATE")
                        blocking.expect("OPEN")
                        blocking.command("get", "VALUE 41")
                        blocking.command("staged", "ERR LANA_ERR_NOT_FOUND")

    def test_killed_writer_preserves_acknowledged_data_not_staging(self):
        for owner, reader in itertools.product(self.binaries, repeat=2):
            with self.subTest(owner=owner, reader=reader), tempfile.TemporaryDirectory(prefix="lana-kill-") as directory:
                path = Path(directory) / "db"
                with probe(owner, path) as first:
                    first.command("put 41", "OK")
                    first.command("commit", "REV 1")
                    first.command("stage", "OK")
                    first.process.kill()
                    first.finish(-signal.SIGKILL)
                with probe(reader, path, 100) as recovered:
                    recovered.command("get", "VALUE 41")
                    recovered.command("staged", "ERR LANA_ERR_NOT_FOUND")
                    recovered.command("put 42", "OK")
                    recovered.command("commit", "REV 2")
                with probe(owner, path) as reopened:
                    reopened.command("get", "VALUE 42")

    def test_every_journal_cut_preserves_acknowledged_revision_floor(self):
        for writer in self.binaries:
            with tempfile.TemporaryDirectory(prefix="lana-cuts-") as directory:
                directory = Path(directory)
                path = directory / "original"
                with probe(writer, path) as original:
                    original.command("put 41", "OK")
                    original.command("commit", "REV 1")
                    first_manifest = (path / "manifest").read_bytes()
                    first_length = (path / "journal").stat().st_size
                    original.command("put 42", "OK")
                    original.command("commit", "REV 2")
                journal = (path / "journal").read_bytes()
                final_manifest = (path / "manifest").read_bytes()
                # Reopen a fresh public-API handle for each cut, without
                # restarting the sanitizer runtime thousands of times.
                with ExitStack() as stack:
                    clients = {binary: stack.enter_context(probe(binary, directory / f"empty-{i}"))
                               for i, binary in enumerate(self.binaries)}
                    for client in clients.values():
                        client.command("close", "OK")
                    for reader, cut in itertools.product(self.binaries, range(len(journal) + 1)):
                        with self.subTest(writer=writer, reader=reader, cut=cut):
                            for acknowledged in (False, True):
                                case = directory / f"cut-{self.binaries.index(reader)}-{cut}-{acknowledged}"
                                case.mkdir()
                                (case / "journal").write_bytes(journal[:cut])
                                (case / "manifest").write_bytes(final_manifest if acknowledged else first_manifest)
                                floor = len(journal) if acknowledged else first_length
                                recovered = clients[reader]
                                if cut < floor:
                                    recovered.command(f"open {case}", "ERR LANA_ERR_CORRUPTION")
                                    recovered.command("get", "ERR LANA_ERR_INVALID_STATE")
                                    self.assertEqual((case / "journal").read_bytes(), journal[:cut])
                                else:
                                    recovered.command(f"open {case}", "OPEN")
                                    complete = cut == len(journal)
                                    recovered.command("get", "VALUE 42" if complete else "VALUE 41")
                                    recovered.command("put 43", "OK")
                                    recovered.command("commit", "REV 3" if complete else "REV 2")
                                    recovered.command("close", "OK")
                                    check = clients[writer]
                                    check.command(f"open {case}", "OPEN")
                                    check.command("get", "VALUE 43")
                                    check.command("close", "OK")
                print(f"{writer.name}: {2 * len(self.binaries) * (len(journal) + 1)} journal-cut recoveries checked", flush=True)

    def test_manifest_publication_failure_and_compaction_recovery(self):
        for writer, reader in itertools.product(self.binaries, repeat=2):
            with self.subTest(writer=writer, reader=reader), tempfile.TemporaryDirectory(prefix="lana-publish-") as directory:
                path = Path(directory) / "db"
                with probe(writer, path) as first:
                    first.command("put 41", "OK")
                    first.command("commit", "REV 1")
                    previous = (path / "manifest").read_bytes()
                    (path / "manifest.tmp").mkdir()
                    first.command("put 42", "OK")
                    first.command("commit", "ERR LANA_ERR_IO")
                    self.assertEqual((path / "manifest").read_bytes(), previous)
                    (path / "manifest.tmp").rmdir()
                # A failed acknowledgement may still have a complete durable
                # journal record. Never silently discard that committed record.
                with probe(reader, path) as recovered:
                    recovered.command("get", "VALUE 42")
                    recovered.command("compact", "REV 2")
                    self.assertEqual((path / "journal").stat().st_size, 0)
                    recovered.command("put 43", "OK")
                    recovered.command("commit", "REV 3")
                with probe(writer, path) as reopened:
                    reopened.command("get", "VALUE 43")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c11", type=Path, default=ROOT / "build/lana_store_tests")
    parser.add_argument("--rust", type=Path)
    args, remaining = parser.parse_known_args()
    StoreProcesses.binaries = [args.c11.resolve()] + ([args.rust.resolve()] if args.rust else [])
    unittest.main(argv=[__file__, *remaining])
