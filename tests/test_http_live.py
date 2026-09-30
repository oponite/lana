"""Exercise source -> native compiler -> HTTP host against a local peer."""
from pathlib import Path
import socket
import subprocess
import sys
import threading


def main():
    errors = []
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(30)
        port = listener.getsockname()[1]

        def serve():
            try:
                with listener.accept()[0] as peer:
                    peer.settimeout(5)
                    request = b""
                    while not request.endswith(b"\r\n\r\nabc"):
                        byte = peer.recv(1)
                        assert byte, request
                        request += byte
                    assert b"x-test: sent\r\n" in request, request
                    assert f"Host: 127.0.0.1:{port}\r\n".encode() in request, request
                    for part in [b"HTTP/1.1 103 Early Hints\r\n\r\n", b"HTTP/1.1 200 OK\r\n",
                                 b"Set-Cookie: a\r\nset-cookie: b\r\nTransfer-Encoding: chunked\r\n\r\n",
                                 b'2;label="test"\r\nok\r\n0\r\nX-End: yes\r\n\r\n']:
                        peer.sendall(part)
                    assert peer.recv(1) == b"", "client did not finish on framing"
            except Exception as error:
                errors.append(error)

        server = threading.Thread(target=serve, daemon=True)
        server.start()
        source = Path(__file__).parent / "regression/http_headers_pass.lana"
        result = subprocess.run([sys.argv[1], "run", str(source), "--", f"http://127.0.0.1:{port}/"],
                                capture_output=True, text=True, timeout=30)
        server.join(1)
        assert result.returncode == 0, result.stdout + result.stderr
        assert not server.is_alive(), "HTTP peer remained blocked"
        assert not errors, errors
        assert "HTTP_HEADERS_PASS" in result.stdout, result.stdout
        print(result.stdout, end="")


if __name__ == "__main__":
    main()
