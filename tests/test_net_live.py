"""Real loopback HTTP/TLS with ephemeral ports and independent server evidence."""
import argparse
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import shutil
import ssl
import subprocess
import tempfile
import threading

ROOT = Path(__file__).resolve().parents[1]


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        with self.server.changed:
            self.server.requests.append(self.path)
            self.server.changed.notify_all()
        body = b"hello from lana"
        self.send_response(200)
        if self.path == "/conflict":
            self.send_header("Content-Length", "1")
            self.send_header("Content-Length", "2")
        elif self.path == "/overflow":
            self.send_header("Content-Length", "18446744073709551615")
        elif self.path == "/invalid":
            self.send_header("Content-Length", "14junk")
        elif self.path == "/encoded":
            self.send_header("Transfer-Encoding", "chunked")
        else:
            self.send_header("Content-Length", str(len(body) + (1 if self.path == "/truncated" else 0)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


class Server(ThreadingHTTPServer):
    daemon_threads = False

    def __init__(self, context=None):
        super().__init__(("127.0.0.1", 0), Handler)
        self.context = context
        self.requests = []
        self.rejections = []
        self.changed = threading.Condition()

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(3)
        if self.context:
            try:
                connection = self.context.wrap_socket(connection, server_side=True)
            except ssl.SSLError as error:
                with self.changed:
                    self.rejections.append(error.reason)
                    self.changed.notify_all()
                connection.close()
                raise
        return connection, address


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lana", type=Path, default=os.environ.get("LANA", ROOT / "build/lana"))
    parser.add_argument("--c11", type=Path, default=os.environ.get("LANAVM", ROOT / "build/lanavm"))
    parser.add_argument("--rust", type=Path)
    args = parser.parse_args()
    if not shutil.which("openssl"):
        print("SKIP: openssl is required to generate the loopback test certificate")
        return 77
    cases = dict(http_pass="NET_HTTP_OK\n", http_framing="NET_HTTP_FRAMING_OK\n", tls_verify_default="NET_TLS_DEFAULT_REJECT\n",
                 tls_verify_false="NET_TLS_VERIFY_FALSE_OK\n", tls_trusted_host="NET_TLS_TRUSTED_HOST_OK\n")
    fixtures = ROOT / "tests/conformance/differential/net_live"
    assert {path.stem for path in fixtures.glob("*.lana")} == set(cases), "unregistered network fixture"
    binaries = [args.c11.resolve()] + ([args.rust.resolve()] if args.rust else [])
    environment = {**os.environ, "LANA_STDLIB_DIR": str(ROOT / "stdlib")}
    with tempfile.TemporaryDirectory(prefix="lana-net-") as directory, ExitStack() as stack:
        directory = Path(directory)
        cert, key = directory / "cert.pem", directory / "key.pem"
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-keyout", str(key),
                        "-out", str(cert), "-days", "2", "-nodes", "-subj", "/CN=wrong.example",
                        "-addext", "subjectAltName=IP:127.0.0.1"], check=True, capture_output=True, timeout=20)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert, key)
        servers = [Server(), Server(context)]
        for server in servers:
            thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
            thread.start()
            stack.callback(server.server_close)
            stack.callback(thread.join, 5)
            stack.callback(server.shutdown)
        for name, expected in cases.items():
            source = (fixtures / f"{name}.lana").read_text()
            source = source.replace(":18100/", f":{servers[0].server_port}/")
            source = source.replace(":18101/", f":{servers[1].server_port}/")
            temporary = directory / f"{name}.lana"
            temporary.write_text(source)
            bytecode = temporary.with_suffix(".labc")
            compiled = subprocess.run([str(args.lana.resolve()), "compile", str(temporary), "-o", str(bytecode)],
                                      cwd=directory, env=environment, capture_output=True, timeout=30)
            assert (compiled.returncode, compiled.stdout, compiled.stderr) == (0, b"", b""), compiled
            # OpenSSL supports a per-process trust file; Rust uses its compiled root store.
            for binary in (binaries[:1] if name == "tls_trusted_host" else binaries):
                counts = [(len(s.requests), len(s.rejections)) for s in servers]
                run_environment = {**environment, "SSL_CERT_FILE": str(cert)} if name == "tls_trusted_host" else environment
                result = subprocess.run([str(binary), "run", str(bytecode)], cwd=directory,
                                        env=run_environment, capture_output=True, timeout=15)
                assert (result.returncode, result.stdout, result.stderr) == (0, expected.encode(), b""), (name, binary, result)
                wanted = ([(1, 0), (0, 0)] if name == "http_pass" else
                          [(5, 0), (0, 0)] if name == "http_framing" else
                          [(0, 0), (1, 1)] if name == "tls_trusted_host" else
                          [(0, 0), (0, 1)] if name == "tls_verify_default" else [(0, 0), (1, 0)])
                for server, (requests, rejects), (new_requests, new_rejects) in zip(servers, counts, wanted):
                    with server.changed:
                        assert server.changed.wait_for(lambda:
                            len(server.requests) >= requests + new_requests and
                            len(server.rejections) >= rejects + new_rejects, timeout=3), name
                delta = [(len(s.requests) - requests, len(s.rejections) - rejects)
                         for s, (requests, rejects) in zip(servers, counts)]
                assert delta == wanted, (name, binary, delta)
                if name == "tls_verify_default":
                    assert servers[1].rejections[-1] in ("TLSV1_ALERT_UNKNOWN_CA", "SSLV3_ALERT_BAD_CERTIFICATE", "SSLV3_ALERT_CERTIFICATE_UNKNOWN"), servers[1].rejections
                print(f"PASS {binary.name}: {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
