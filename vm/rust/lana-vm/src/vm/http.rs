//! Bounded HTTP/1.x response framing (RFC 9112 sections 5–7).
use super::{Arc, Buffer, Heap, LanaError, Map, ValueKind};

pub(super) const HEADER_LIMIT: usize = 65_536;
const CHUNK_LINE_LIMIT: usize = 8192;
pub(super) type Headers = Buffer<(Arc<str>, Arc<str>)>;

#[derive(Debug, PartialEq)]
pub(super) enum Error { Protocol, Resource(LanaError), Timeout, Io }
impl From<LanaError> for Error {
    fn from(error: LanaError) -> Self { Self::Resource(error) }
}

fn token(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&byte)
}

fn field_byte(byte: u8) -> bool { byte == b'\t' || (byte >= b' ' && byte != 127) }

fn transport_header(name: &str) -> bool {
    matches!(name, "host" | "content-length" | "transfer-encoding" | "connection"
        | "upgrade" | "trailer" | "te" | "proxy-connection")
}

/// Validate before opening a socket. The verification switch is never sent.
pub(super) fn request_headers(heap: &Heap, headers: &Map) -> Result<(bool, Buffer<u8>), Error> {
    let mut bytes = Buffer::new(heap, 0, 0)?;
    let mut verify = None;
    for entry in &headers.entries {
        let name = entry.key.to_ascii_lowercase();
        if name == "verify" {
            let ValueKind::Bool(value) = entry.value.kind else { return Err(LanaError::Type.into()); };
            if verify.replace(value).is_some() { return Err(Error::Protocol); }
            continue;
        }
        if name.is_empty() || !name.bytes().all(token) || transport_header(&name) || name == "expect" {
            return Err(Error::Protocol);
        }
        let ValueKind::String(value) = &entry.value.kind else { return Err(LanaError::Type.into()); };
        if !value.bytes().all(field_byte) { return Err(Error::Protocol); }
        if name.len().saturating_add(value.len()).saturating_add(bytes.len()).saturating_add(4) > HEADER_LIMIT {
            return Err(Error::Protocol);
        }
        bytes.extend_from_slice(name.as_bytes())?;
        bytes.extend_from_slice(b": ")?;
        bytes.extend_from_slice(value.as_bytes())?;
        bytes.extend_from_slice(b"\r\n")?;
    }
    Ok((verify.unwrap_or(true), bytes))
}

#[derive(Clone, Copy)]
enum State { Head, Fixed(usize), Eof, ChunkSize, Chunk(usize), ChunkEnd(u8), Trailers, Done }

pub(super) struct Response {
    pub status: u16,
    pub headers: Headers,
    pub trailers: Headers,
    pub body: Buffer<u8>,
    heap: Heap,
    line: Buffer<u8>,
    state: State,
    header_bytes: usize,
    interim: usize,
}

impl Response {
    pub fn new(heap: &Heap) -> Result<Self, Error> {
        Ok(Self { status: 0, headers: Buffer::new(heap, 0, 0)?,
            trailers: Buffer::new(heap, 0, 0)?, body: Buffer::new(heap, 0, 0)?,
            heap: heap.clone(), line: Buffer::new(heap, 0, 0)?, state: State::Head,
            header_bytes: 0, interim: 0 })
    }

    pub fn complete(&self) -> bool { matches!(self.state, State::Done) }

    pub fn eof(&mut self) -> Result<(), Error> {
        if !matches!(self.state, State::Eof | State::Done) { return Err(Error::Protocol); }
        self.state = State::Done;
        Ok(())
    }

    pub fn feed(&mut self, mut input: &[u8]) -> Result<(), Error> {
        while !input.is_empty() {
            match self.state {
                State::Done => return Ok(()), // Connection closes; never interpret surplus as another response.
                State::Fixed(left) | State::Chunk(left) => {
                    let count = input.len().min(left);
                    self.body.extend_from_slice(&input[..count])?;
                    input = &input[count..];
                    self.state = match self.state {
                        State::Fixed(_) if count == left => State::Done,
                        State::Fixed(_) => State::Fixed(left - count),
                        _ if count == left => State::ChunkEnd(0),
                        _ => State::Chunk(left - count),
                    };
                }
                State::Eof => { self.body.extend_from_slice(input)?; return Ok(()); }
                State::ChunkEnd(index) => {
                    if input[0] != b"\r\n"[index as usize] { return Err(Error::Protocol); }
                    input = &input[1..];
                    self.state = if index == 0 { State::ChunkEnd(1) } else { State::ChunkSize };
                }
                _ => {
                    let byte = input[0];
                    input = &input[1..];
                    if (byte == b'\n' && self.line.last() != Some(&b'\r'))
                        || (self.line.last() == Some(&b'\r') && byte != b'\n') {
                        return Err(Error::Protocol);
                    }
                    let limit = if matches!(self.state, State::ChunkSize) { CHUNK_LINE_LIMIT } else {
                        self.header_bytes += 1;
                        if self.header_bytes > HEADER_LIMIT { return Err(Error::Protocol); }
                        HEADER_LIMIT
                    };
                    if self.line.len() >= limit { return Err(Error::Protocol); }
                    self.line.push(byte)?;
                    if byte == b'\n' {
                        let mut line = std::mem::replace(&mut self.line, Buffer::new(&self.heap, 0, 0)?);
                        line.truncate(line.len() - 2);
                        self.line_complete(&line)?;
                        line.clear();
                        self.line = line;
                    }
                }
            }
        }
        Ok(())
    }

    fn line_complete(&mut self, line: &[u8]) -> Result<(), Error> {
        match self.state {
            State::Head if self.status == 0 => {
                if line.len() < 13 || !matches!(&line[..9], b"HTTP/1.0 " | b"HTTP/1.1 ")
                    || !line[9..12].iter().all(u8::is_ascii_digit) || line[12] != b' '
                    || !line[13..].iter().copied().all(field_byte) { return Err(Error::Protocol); }
                self.status = ((line[9] - b'0') as u16) * 100
                    + ((line[10] - b'0') as u16) * 10 + (line[11] - b'0') as u16;
                if !(100..=599).contains(&self.status) || self.status == 101 { return Err(Error::Protocol); }
            }
            State::Head if line.is_empty() => self.finish_head()?,
            State::Head | State::Trailers => {
                if line.is_empty() { self.state = State::Done; return Ok(()); }
                let colon = line.iter().position(|&byte| byte == b':').ok_or(Error::Protocol)?;
                if colon == 0 || !line[..colon].iter().copied().all(token)
                    || !line[colon + 1..].iter().copied().all(field_byte) { return Err(Error::Protocol); }
                let name = String::from_utf8(line[..colon].to_ascii_lowercase()).unwrap();
                let raw = &line[colon + 1..];
                let start = raw.iter().position(|byte| !matches!(byte, b' ' | b'\t')).unwrap_or(raw.len());
                let end = raw.iter().rposition(|byte| !matches!(byte, b' ' | b'\t')).map_or(start, |pos| pos + 1);
                // HTTP field octets map reversibly to Unicode U+0000..U+00FF.
                let value: String = raw[start..end].iter().map(|&byte| char::from(byte)).collect();
                let field = (self.heap.string(&name)?, self.heap.string(&value)?);
                if matches!(self.state, State::Trailers) {
                    if transport_header(&name) || matches!(name.as_str(), "authorization" | "proxy-authorization"
                        | "content-type" | "content-encoding" | "content-range") { return Err(Error::Protocol); }
                    self.trailers.push(field)?;
                } else { self.headers.push(field)?; }
            }
            State::ChunkSize => {
                let count = line.iter().take_while(|byte| byte.is_ascii_hexdigit()).count();
                if count == 0 || !extensions(&line[count..]) { return Err(Error::Protocol); }
                let mut size = 0usize;
                for byte in &line[..count] {
                    let digit = (*byte as char).to_digit(16).unwrap() as usize;
                    size = size.checked_mul(16).and_then(|size| size.checked_add(digit)).ok_or(Error::Protocol)?;
                }
                self.body.reserve(size)?;
                self.state = if size == 0 { State::Trailers } else { State::Chunk(size) };
            }
            _ => return Err(Error::Protocol),
        }
        Ok(())
    }

    fn finish_head(&mut self) -> Result<(), Error> {
        if self.status < 200 || self.status == 204 {
            if self.headers.iter().any(|(name, _)| matches!(name.as_ref(), "content-length" | "transfer-encoding")) {
                return Err(Error::Protocol);
            }
            if self.status == 204 { self.state = State::Done; } else {
                self.interim += 1;
                if self.interim > 16 { return Err(Error::Protocol); }
                self.status = 0;
                self.headers.clear();
            }
            return Ok(());
        }
        if self.status == 304 { self.state = State::Done; return Ok(()); }
        let mut length = None;
        let mut chunked = false;
        for (name, value) in &self.headers {
            if name.as_ref() == "transfer-encoding" {
                if chunked || !value.eq_ignore_ascii_case("chunked") { return Err(Error::Protocol); }
                chunked = true;
            }
            if name.as_ref() == "content-length" {
                for value in value.split(',').map(|value| value.trim_matches([' ', '\t'])) {
                    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) { return Err(Error::Protocol); }
                    let parsed = value.parse::<usize>().map_err(|_| Error::Protocol)?;
                    if length.is_some_and(|old| old != parsed) { return Err(Error::Protocol); }
                    length = Some(parsed);
                }
            }
        }
        if chunked && length.is_some() { return Err(Error::Protocol); }
        self.state = if chunked { State::ChunkSize } else if let Some(length) = length {
            self.body.reserve(length)?;
            if length == 0 { State::Done } else { State::Fixed(length) }
        } else { State::Eof };
        Ok(())
    }
}

fn extensions(mut text: &[u8]) -> bool {
    fn whitespace(text: &mut &[u8]) {
        while text.first().is_some_and(|byte| matches!(byte, b' ' | b'\t')) { *text = &text[1..]; }
    }
    while !text.is_empty() {
        whitespace(&mut text);
        if text.first() != Some(&b';') { return false; }
        text = &text[1..];
        whitespace(&mut text);
        let count = text.iter().take_while(|&&byte| token(byte)).count();
        if count == 0 { return false; }
        text = &text[count..];
        whitespace(&mut text);
        if text.first() == Some(&b'=') {
            text = &text[1..];
            whitespace(&mut text);
            if text.first() == Some(&b'"') {
                text = &text[1..];
                loop {
                    let Some(&byte) = text.first() else { return false; };
                    text = &text[1..];
                    if byte == b'"' { break; }
                    if !field_byte(byte) { return false; }
                    if byte == b'\\' {
                        if !text.first().is_some_and(|&byte| field_byte(byte)) { return false; }
                        text = &text[1..];
                    }
                }
            } else {
                let count = text.iter().take_while(|&&byte| token(byte)).count();
                if count == 0 { return false; }
                text = &text[count..];
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(bytes: &[u8], split: usize) -> Result<Response, Error> {
        let mut response = Response::new(&Heap::new(1 << 20))?;
        for part in bytes.chunks(split.max(1)) { response.feed(part)?; }
        response.eof()?;
        Ok(response)
    }

    #[test]
    fn http_fragmented_framing_and_headers() {
        let cases: &[(&[u8], &[u8])] = &[
            (b"HTTP/1.1 200 OK\r\nContent-Length: 4, 4\r\nContent-Length: 4\r\n\r\ntest", b"test"),
            (b"HTTP/1.0 200 OK\r\n\r\neof body", b"eof body"),
            (b"HTTP/1.1 100 Continue\r\n\r\nHTTP/1.1 103 Early Hints\r\nLink: x\r\n\r\nHTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n", b""),
            (b"HTTP/1.1 204 No Content\r\n\r\n", b""),
            (b"HTTP/1.1 304 Not Modified\r\nContent-Length: 999\r\n\r\n", b""),
            (b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nSet-Cookie: a\r\nset-cookie: b\r\nX-Octet: \xff\r\n\r\n2;tag=\"a;\\\"b\"\r\nte\r\n2; flag\r\nst\r\n0\r\nX-Sum: yes\r\n\r\n", b"test"),
        ];
        for &(wire, body) in cases {
            for split in 1..=wire.len() {
                let response = parse(wire, split).unwrap();
                assert_eq!(&*response.body, body);
                assert!(response.complete());
            }
        }
        let response = parse(cases.last().unwrap().0, 1).unwrap();
        assert_eq!(response.headers.iter().filter(|(key, _)| key.as_ref() == "set-cookie")
            .map(|(_, value)| value.as_ref()).collect::<Vec<_>>(), ["a", "b"]);
        assert_eq!(response.headers.last().unwrap().1.as_ref(), "ÿ");
        assert_eq!(response.trailers[0].0.as_ref(), "x-sum");
        assert_eq!(response.trailers[0].1.as_ref(), "yes");
    }

    #[test]
    fn http_rejects_ambiguous_malformed_and_truncated_input() {
        for head in ["HTTP/2.0 200 OK\r\n\r\n", "HTTP/1.1 099 Bad\r\n\r\n", "HTTP/1.1 101 Upgrade\r\n\r\n",
            "HTTP/1.1 200\r\n\r\n", "HTTP/1.1 200 OK\n\n", "HTTP/1.1 200 OK\rX"] {
            assert!(matches!(parse(head.as_bytes(), 1), Err(Error::Protocol)), "{head:?}");
        }
        for fields in ["Content-Length: -1", "Content-Length: +1", "Content-Length: 1, 2", "Content-Length: 1,",
            "Content-Length: 184467440737095516160", "Transfer-Encoding: gzip", "Transfer-Encoding: chunked, chunked",
            "Transfer-Encoding: chunked\r\nContent-Length: 1", "X : bad", " X: folded", "X: bad\u{7f}",
            "Transfer-Encoding: chunked\r\nTransfer-Encoding: chunked"] {
            let wire = format!("HTTP/1.1 200 OK\r\n{fields}\r\n\r\nx");
            assert!(matches!(parse(wire.as_bytes(), 1), Err(Error::Protocol)), "{fields:?}");
        }
        for chunk in ["g\r\n", "10000000000000000\r\n", "1;\r\n", "1;x=\"open\r\n", "1;x=\r\n",
            "1\r\nx!\r\n", "0\r\nContent-Length: 1\r\n\r\n"] {
            let wire = format!("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n{chunk}");
            assert!(matches!(parse(wire.as_bytes(), 1), Err(Error::Protocol)), "{chunk:?}");
        }
        for wire in [b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\n\r\ntest".as_slice(),
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n1\r\nx\r\n0\r\n\r\n"] {
            for end in 0..wire.len() { assert!(parse(&wire[..end], 1).is_err(), "prefix {end}"); }
        }
        let hints = "HTTP/1.1 100 Continue\r\n\r\n".repeat(17);
        assert!(matches!(parse(hints.as_bytes(), 1), Err(Error::Protocol)));
        let huge = format!("HTTP/1.1 200 OK\r\nX: {}\r\n\r\n", "x".repeat(HEADER_LIMIT));
        assert!(matches!(parse(huge.as_bytes(), 4096), Err(Error::Protocol)));
    }

    #[test]
    fn http_declared_bodies_respect_heap_before_reading() {
        for framing in ["Content-Length: 1000000\r\n\r\n", "Transfer-Encoding: chunked\r\n\r\n100000\r\n"] {
            let mut response = Response::new(&Heap::new(4096)).unwrap();
            assert_eq!(response.feed(format!("HTTP/1.1 200 OK\r\n{framing}").as_bytes()), Err(Error::Resource(LanaError::Oom)));
            assert!(!response.complete());
            assert!(response.body.is_empty());
        }
    }

    #[test]
    fn http_request_fields_are_validated_before_transport() {
        use super::super::Value;
        let heap = Heap::new(1 << 20);
        for name in ["host", "Content-Length", "Transfer-Encoding", "Connection", "Expect", "bad name", "X\r\nY"] {
            let mut map = Map::new(&heap, 1).unwrap();
            map.set(Arc::from(name), Value::string(Arc::from("value")), false).unwrap();
            assert!(matches!(request_headers(&heap, &map), Err(Error::Protocol)), "{name}");
        }
        for value in ["a\r\nb", "a\nb", "a\0b", "a\u{7f}b"] {
            let mut map = Map::new(&heap, 1).unwrap();
            map.set(Arc::from("x"), Value::string(Arc::from(value)), false).unwrap();
            assert!(matches!(request_headers(&heap, &map), Err(Error::Protocol)));
        }
        let mut map = Map::new(&heap, 2).unwrap();
        map.set(Arc::from("verify"), Value::boolean(false), false).unwrap();
        map.set(Arc::from("X-Test"), Value::string(Arc::from("yes")), false).unwrap();
        let (verify, wire) = request_headers(&heap, &map).unwrap();
        assert!(!verify);
        assert_eq!(&*wire, b"x-test: yes\r\n");
        map.set(Arc::from("Verify"), Value::boolean(true), false).unwrap();
        assert!(matches!(request_headers(&heap, &map), Err(Error::Protocol)));
    }

    #[test]
    fn http_urls_and_live_socket_deadlines() {
        use super::super::{net_parse_url, NetSocket, Vm};
        use std::io::{Read, Write};
        use std::net::{TcpListener, TcpStream};
        for url in ["http://x:0", "http://x:+80", "http://[::1]:+80", "http://a@b", "http://x/a b",
            "http://x/\r\n", "http://x/%a", "http://x/%zz", "ftp://x", "http://x\\y"] {
            assert!(net_parse_url(url).is_none(), "{url}");
        }
        assert_eq!(net_parse_url("https://[::1]:8443?q=x#fragment").unwrap(),
            ("https".into(), "::1".into(), 8443, "/?q=x".into()));
        let chunk = lana_bytecode::assembler::assemble("HALT\n").unwrap();
        for wire in [b"HTTP/1.1 204 No Content\r\n\r\n".as_slice(),
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n1\r\nx\r\n0\r\nX-End: yes\r\n\r\n"] {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let address = listener.local_addr().unwrap();
            let server = std::thread::spawn(move || {
                let (mut socket, _) = listener.accept().unwrap();
                for byte in wire { socket.write_all(&[*byte]).unwrap(); }
                // The client must finish on framing, while the server waits for EOF.
                socket.set_read_timeout(Some(std::time::Duration::from_secs(2))).unwrap();
                assert_eq!(socket.read(&mut [0u8; 1]).unwrap(), 0);
            });
            let mut socket = NetSocket::Plain(TcpStream::connect(address).unwrap());
            let mut vm = Vm::new(&chunk);
            let response = vm.net_receive_response(&mut socket, 1000).unwrap();
            assert!(response.complete());
            drop(socket);
            server.join().unwrap();
        }
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut socket = NetSocket::Plain(TcpStream::connect(listener.local_addr().unwrap()).unwrap());
        let (_peer, _) = listener.accept().unwrap();
        let mut vm = Vm::new(&chunk);
        assert!(matches!(vm.net_receive_response(&mut socket, 20), Err(Error::Timeout)));
    }

    #[test]
    fn http_live_request_and_response_mapping() {
        use super::super::{Vm, Value};
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            socket.set_read_timeout(Some(std::time::Duration::from_secs(2))).unwrap();
            let mut request = Vec::new();
            while !request.ends_with(b"\r\n\r\nabc") {
                let mut byte = [0u8; 1];
                assert_eq!(socket.read(&mut byte).unwrap(), 1);
                request.push(byte[0]);
            }
            assert_eq!(String::from_utf8(request).unwrap(), format!(
                "POST /?q=x HTTP/1.1\r\nHost: {address}\r\nConnection: close\r\nContent-Length: 3\r\nx-test: sent\r\n\r\nabc"));
            socket.write_all(b"HTTP/1.1 200 OK\r\nSet-Cookie: a\r\nSet-Cookie: b\r\nTransfer-Encoding: chunked\r\n\r\n2\r\nok\r\n0\r\nX-End: yes\r\n\r\n").unwrap();
        });
        let chunk = lana_bytecode::assembler::assemble("HALT\n").unwrap();
        let mut vm = Vm::new(&chunk);
        let mut out = Value::null();
        assert_eq!(vm.net_http_request("POST", &format!("http://{address}?q=x#ignored"),
            Some("abc"), 1000., true, b"x-test: sent\r\n", &mut out), LanaError::Ok);
        let ValueKind::Array(result) = &out.kind else { panic!("result"); };
        let result = result.lock().unwrap();
        assert!(matches!(result.items()[0].kind, ValueKind::Bool(true)));
        let response = vm.information_snapshot(&result.items()[1]).unwrap();
        let ValueKind::Map(response) = response.kind else { panic!("response"); };
        let response = response.lock().unwrap();
        assert_eq!(response.get("status").unwrap().as_number(), 200.);
        assert_eq!(response.get("body").unwrap().as_string().as_ref(), "ok");
        assert!(response.get("headers").unwrap().print().contains("[a, b]"));
        assert!(response.get("trailers").unwrap().print().contains("[yes]"));
        server.join().unwrap();
    }

    #[cfg(all(feature = "net-tls", not(target_arch = "wasm32")))]
    #[test]
    fn http_live_trusted_tls_and_default_rejection() {
        use super::super::{net_tls_connect, NetSocket, Vm};
        use std::io::{Read, Write};
        use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer, ServerName};
        let work = std::env::temp_dir().join(format!("lana-http-tls-{}", std::process::id()));
        std::fs::create_dir_all(&work).unwrap();
        let cert = work.join("cert.pem");
        let key = work.join("key.pem");
        let output = std::process::Command::new("openssl").args(["req", "-x509", "-newkey", "rsa:2048", "-nodes",
            "-days", "1", "-subj", "/CN=localhost", "-addext", "subjectAltName=DNS:localhost",
            "-addext", "basicConstraints=critical,CA:FALSE", "-keyout"]).arg(&key).arg("-out").arg(&cert).output().unwrap();
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
        let output = std::process::Command::new("openssl").args(["x509", "-in"]).arg(&cert).args(["-outform", "DER"]).output().unwrap();
        assert!(output.status.success());
        let cert = CertificateDer::from(output.stdout);
        let output = std::process::Command::new("openssl").args(["pkcs8", "-topk8", "-nocrypt", "-in"])
            .arg(&key).args(["-outform", "DER"]).output().unwrap();
        assert!(output.status.success());
        let key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(output.stdout));
        std::fs::remove_dir_all(work).unwrap();
        let server_config = Arc::new(rustls::ServerConfig::builder().with_no_client_auth()
            .with_single_cert(vec![cert.clone()], key).unwrap());
        let mut roots = rustls::RootCertStore::empty();
        roots.add(cert).unwrap();
        let trusted = Arc::new(rustls::ClientConfig::builder().with_root_certificates(roots).with_no_client_auth());
        let chunk = lana_bytecode::assembler::assemble("HALT\n").unwrap();
        for mode in 0..3 {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let address = listener.local_addr().unwrap();
            let config = server_config.clone();
            let server = std::thread::spawn(move || {
                let (socket, _) = listener.accept().unwrap();
                socket.set_read_timeout(Some(std::time::Duration::from_secs(2))).unwrap();
                let connection = rustls::ServerConnection::new(config).unwrap();
                let mut tls = rustls::StreamOwned::new(connection, socket);
                if mode == 0 { assert!(tls.read(&mut [0u8; 1]).is_err()); return; }
                tls.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok").unwrap();
                tls.flush().unwrap();
                // No close_notify: framed bodies must not wait for TLS EOF.
                let _ = tls.read(&mut [0u8; 1]);
            });
            let stream = std::net::TcpStream::connect(address).unwrap();
            stream.set_read_timeout(Some(std::time::Duration::from_secs(2))).unwrap();
            if mode == 0 { assert!(net_tls_connect(stream, "localhost", true).is_err()); }
            else {
                let mut socket = if mode == 1 {
                    let connection = rustls::ClientConnection::new(trusted.clone(), ServerName::try_from("localhost").unwrap()).unwrap();
                    NetSocket::Tls(Box::new(rustls::StreamOwned::new(connection, stream)))
                } else { net_tls_connect(stream, "localhost", false).unwrap() };
                let mut vm = Vm::new(&chunk);
                assert_eq!(&*vm.net_receive_response(&mut socket, 1000).unwrap().body, b"ok");
            }
            server.join().unwrap();
        }
    }

}
