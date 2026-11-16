//! Security hardening: HTTP authentication, CORS, bounded request parsing,
//! connection admission, refusal to serve an open API beyond loopback, and
//! authenticated node-to-node connections.

use std::io::{BufRead, BufReader, Cursor, Read, Write};
use std::net::TcpStream;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use tesseract_fabric::server::http::read_request;
use tesseract_fabric::wire::{auth, Endpoint, FrameType, TesseractFrame};

struct Server(Child, String);

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Start the real `tesseract-serve` (tiny synthetic model, one local node).
fn serve(extra: &[&str]) -> Server {
    let mut child = Command::new(env!("CARGO_BIN_EXE_tesseract-serve"))
        .args([
            "--http",
            "127.0.0.1:0",
            "--model",
            "tiny",
            "--local-nodes",
            "1",
        ])
        .args(extra)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut line = String::new();
    BufReader::new(child.stdout.take().unwrap())
        .read_line(&mut line)
        .unwrap();
    let addr = line
        .trim()
        .strip_prefix("listening on http://")
        .expect("server address line")
        .to_string();
    Server(child, addr)
}

/// One raw request; returns the whole response (head + body).
fn raw(addr: &str, req: &str) -> String {
    let mut s = TcpStream::connect(addr).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(30))).unwrap();
    s.write_all(req.as_bytes()).unwrap();
    let mut out = String::new();
    let _ = s.read_to_string(&mut out);
    out
}

fn get(addr: &str, path: &str, auth: Option<&str>) -> String {
    let a = auth
        .map(|k| format!("Authorization: Bearer {k}\r\n"))
        .unwrap_or_default();
    raw(addr, &format!("GET {path} HTTP/1.1\r\nHost: x\r\n{a}\r\n"))
}

#[test]
fn api_key_is_required_on_v1_endpoints_and_cors_is_explicit() {
    let srv = serve(&[
        "--api-key",
        "s3cret-key",
        "--cors-origin",
        "https://app.example",
    ]);
    let a = &srv.1;
    // no key, wrong key → 401 with a Bearer challenge
    for k in [None, Some("wrong"), Some("s3cret-key-but-longer")] {
        let r = get(a, "/v1/models", k);
        assert!(r.starts_with("HTTP/1.1 401"), "{k:?}: {r}");
        assert!(r.contains("WWW-Authenticate: Bearer"));
        assert!(r.contains("authentication_error"));
    }
    let r = get(a, "/v1/cluster/status", None);
    assert!(r.starts_with("HTTP/1.1 401"));
    let r = raw(
        a,
        "POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}",
    );
    assert!(r.starts_with("HTTP/1.1 401"), "chat without key: {r}");
    // right key → 200, CORS only for the configured origin
    let r = get(a, "/v1/models", Some("s3cret-key"));
    assert!(r.starts_with("HTTP/1.1 200"), "{r}");
    assert!(r.contains("Access-Control-Allow-Origin: https://app.example"));
    assert!(!r.contains("Access-Control-Allow-Origin: *"));
    // the dashboard page itself is static and public; its API calls need the key
    let r = get(a, "/", None);
    assert!(r.starts_with("HTTP/1.1 200") && r.contains("#key="));
    // CORS preflight carries no credentials and is answered
    let r = raw(a, "OPTIONS /v1/models HTTP/1.1\r\n\r\n");
    assert!(r.starts_with("HTTP/1.1 204"));
}

#[test]
fn without_cors_origin_no_cors_headers_are_sent() {
    let srv = serve(&[]);
    let r = get(&srv.1, "/v1/models", None);
    assert!(r.starts_with("HTTP/1.1 200"), "{r}");
    assert!(!r.contains("Access-Control-Allow-Origin"), "{r}");
}

#[test]
fn open_api_beyond_loopback_is_refused() {
    let out = Command::new(env!("CARGO_BIN_EXE_tesseract-serve"))
        .args([
            "--http",
            "0.0.0.0:0",
            "--model",
            "tiny",
            "--local-nodes",
            "1",
        ])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("refusing to serve"));
}

#[test]
fn connections_beyond_the_limit_get_503_and_slots_are_released() {
    let srv = serve(&["--max-connections", "2", "--max-per-client", "2"]);
    let a = &srv.1;
    // two idle connections take both slots
    let c1 = TcpStream::connect(a).unwrap();
    let c2 = TcpStream::connect(a).unwrap();
    std::thread::sleep(Duration::from_millis(200));
    let t = Instant::now();
    let r = get(a, "/v1/models", None);
    assert!(r.starts_with("HTTP/1.1 503"), "{r}");
    assert!(
        t.elapsed() < Duration::from_secs(2),
        "503 must be immediate"
    );
    // closing them frees the slots
    drop((c1, c2));
    let t = Instant::now();
    loop {
        let r = get(a, "/v1/models", None);
        if r.starts_with("HTTP/1.1 200") {
            break;
        }
        assert!(
            t.elapsed() < Duration::from_secs(10),
            "slots never released: {r}"
        );
        std::thread::sleep(Duration::from_millis(100));
    }
}

#[test]
fn oversized_request_heads_are_rejected_while_streaming() {
    let srv = serve(&[]);
    let a = &srv.1;
    // 1 MB without a newline: rejected once 64 KB have arrived
    let mut s = TcpStream::connect(a).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    let junk = vec![b'a'; 1 << 20];
    let _ = s.write_all(&junk); // the server may close before all of it is sent
    let mut out = String::new();
    let _ = s.read_to_string(&mut out);
    assert!(
        out.starts_with("HTTP/1.1 400") && out.contains("too large"),
        "{out}"
    );
    // and the server keeps serving
    assert!(get(a, "/v1/models", None).starts_with("HTTP/1.1 200"));
}

#[test]
fn request_parser_limits() {
    let parse = |b: &[u8]| read_request(&mut BufReader::new(Cursor::new(b.to_vec())));
    // a single head line longer than 64 KB
    let mut long = b"GET /".to_vec();
    long.extend(vec![b'a'; 70_000]);
    long.extend(b" HTTP/1.1\r\n\r\n");
    assert!(parse(&long).unwrap_err().contains("too large"));
    // too many header lines
    let mut many = b"GET / HTTP/1.1\r\n".to_vec();
    for i in 0..200 {
        many.extend(format!("X-{i}: y\r\n").as_bytes());
    }
    many.extend(b"\r\n");
    assert!(parse(&many).unwrap_err().contains("too many"));
    // chunked bodies, duplicate or bad lengths, oversized or short bodies
    assert!(parse(b"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n").is_err());
    assert!(parse(b"POST / HTTP/1.1\r\nContent-Length: 1\r\nContent-Length: 2\r\n\r\nab").is_err());
    assert!(parse(b"POST / HTTP/1.1\r\nContent-Length: x\r\n\r\n").is_err());
    assert!(parse(b"POST / HTTP/1.1\r\nContent-Length: 99999999999\r\n\r\n").is_err());
    assert!(parse(b"POST / HTTP/1.1\r\nContent-Length: 10\r\n\r\nshort").is_err());
    // a normal request still parses
    let r = parse(b"POST /v1/x HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}").unwrap();
    assert_eq!(
        (r.method.as_str(), r.path.as_str(), r.body.as_slice()),
        ("POST", "/v1/x", &b"{}"[..])
    );
}

#[test]
fn node_endpoints_drop_connections_that_skip_the_handshake() {
    let ep = Endpoint::bind(7, "127.0.0.1:0").unwrap();
    let before = auth::rejected_connections();
    // a well-formed frame sent without the cluster handshake
    let f = TesseractFrame::new(FrameType::Generate, 1, 7, vec![1, 2, 3]);
    let mut s = TcpStream::connect(ep.addr()).unwrap();
    s.write_all(&f.encode()).unwrap();
    assert!(
        ep.recv_timeout(Duration::from_millis(500)).is_none(),
        "frame accepted without handshake"
    );
    let t = Instant::now();
    while auth::rejected_connections() == before {
        assert!(t.elapsed() < Duration::from_secs(5));
        std::thread::sleep(Duration::from_millis(20));
    }
    // a peer that knows the secret still gets through
    let peer = Endpoint::bind(8, "127.0.0.1:0").unwrap();
    peer.shared.set_addr(7, ep.addr());
    peer.send(7, TesseractFrame::new(FrameType::Generate, 8, 7, vec![9]))
        .unwrap();
    let got = ep
        .recv_timeout(Duration::from_secs(5))
        .expect("authenticated frame");
    assert_eq!(got.payload, vec![9]);
}
