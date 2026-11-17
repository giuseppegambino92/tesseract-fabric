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
        .stderr(if std::env::var_os("CI").is_some() {
            Stdio::inherit()
        } else {
            Stdio::null()
        })
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

#[test]
fn requests_waiting_for_a_slot_are_served_in_arrival_order() {
    // one engine slot: requests queue for it and must be served FIFO
    // (~0.4 s per request on the tiny model, sent 30 ms apart: requests 1–5
    // all queue behind request 0, so the order is decided by the queue)
    let srv = serve(&["--slots", "1", "--max-tokens", "1024"]);
    let a = srv.1.clone();
    let done = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut hs = Vec::new();
    for i in 0..6usize {
        let (a, done) = (a.clone(), done.clone());
        hs.push(std::thread::spawn(move || {
            let body = format!(
                r#"{{"messages":[{{"role":"user","content":"request {i}"}}],"max_tokens":1024,"temperature":0}}"#
            );
            let r = raw(
                &a,
                &format!(
                    "POST /v1/chat/completions HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}",
                    body.len()
                ),
            );
            assert!(r.starts_with("HTTP/1.1 200"), "{r}");
            done.lock().unwrap().push(i);
        }));
        std::thread::sleep(Duration::from_millis(30));
    }
    for h in hs {
        h.join().unwrap();
    }
    let order = done.lock().unwrap().clone();
    assert_eq!(
        order,
        (0..6).collect::<Vec<_>>(),
        "not served in arrival order"
    );
}

/// A TCP relay between a dialer and `target` that records the bytes the
/// dialer sends and can flip one of them (a network eavesdropper / tamperer).
fn relay(
    target: std::net::SocketAddr,
    flip_at: Option<usize>,
) -> (
    std::net::SocketAddr,
    std::sync::Arc<std::sync::Mutex<Vec<u8>>>,
) {
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = l.local_addr().unwrap();
    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let seen2 = seen.clone();
    std::thread::spawn(move || {
        for c in l.incoming() {
            let Ok(mut down) = c else { return };
            let mut up = TcpStream::connect(target).unwrap();
            let (mut down2, mut up2) = (down.try_clone().unwrap(), up.try_clone().unwrap());
            std::thread::spawn(move || {
                let _ = std::io::copy(&mut up2, &mut down2);
            });
            let seen = seen2.clone();
            std::thread::spawn(move || {
                let mut buf = [0u8; 4096];
                loop {
                    let n = match down.read(&mut buf) {
                        Ok(0) | Err(_) => return,
                        Ok(n) => n,
                    };
                    let mut s = seen.lock().unwrap();
                    let start = s.len();
                    s.extend_from_slice(&buf[..n]);
                    if let Some(at) = flip_at {
                        if (start..start + n).contains(&at) {
                            buf[at - start] ^= 0x01;
                        }
                    }
                    drop(s);
                    if up.write_all(&buf[..n]).is_err() {
                        return;
                    }
                }
            });
        }
    });
    (addr, seen)
}

#[test]
fn node_traffic_is_encrypted_and_tamper_evident_with_a_cluster_secret() {
    let secret = "correct horse battery staple, 2026";
    let cfg = auth::WireConfig::new(secret, true);
    let marker: Vec<u8> = b"TOP-SECRET-ACTIVATIONS-".repeat(200);

    // 1. through an eavesdropping relay: delivered, never visible
    let rx = Endpoint::bind_with(20, "127.0.0.1:0", cfg.clone()).unwrap();
    let (raddr, seen) = relay(rx.addr(), None);
    let tx = Endpoint::bind_with(21, "127.0.0.1:0", cfg.clone()).unwrap();
    tx.shared.set_addr(20, raddr);
    tx.send(
        20,
        TesseractFrame::new(FrameType::Generate, 21, 20, marker.clone()),
    )
    .unwrap();
    let got = rx
        .recv_timeout(Duration::from_secs(5))
        .expect("frame delivered");
    assert_eq!(got.payload, marker);
    let wire = seen.lock().unwrap().clone();
    assert!(wire.len() > marker.len(), "the relay saw the traffic");
    assert!(
        !wire.windows(23).any(|w| w == &marker[..23]),
        "plaintext visible on the wire"
    );

    // 2. one flipped bit in the first record: the connection is dropped and
    //    nothing is delivered (the dialer's handshake is 53 + 32 bytes)
    let rx = Endpoint::bind_with(22, "127.0.0.1:0", cfg.clone()).unwrap();
    let (raddr, _) = relay(rx.addr(), Some(53 + 32 + 30));
    let tx = Endpoint::bind_with(23, "127.0.0.1:0", cfg.clone()).unwrap();
    tx.shared.set_addr(22, raddr);
    let _ = tx.send(
        22,
        TesseractFrame::new(FrameType::Generate, 23, 22, marker.clone()),
    );
    assert!(
        rx.recv_timeout(Duration::from_millis(800)).is_none(),
        "tampered traffic accepted"
    );

    // 2b. a man in the middle who swaps the dialer's ephemeral key (here: one
    //     flipped bit of it) without the secret breaks the handshake
    let rx = Endpoint::bind_with(26, "127.0.0.1:0", cfg.clone()).unwrap();
    let (raddr, _) = relay(rx.addr(), Some(21 + 9));
    let tx = Endpoint::bind_with(27, "127.0.0.1:0", cfg.clone()).unwrap();
    tx.shared.set_addr(26, raddr);
    assert!(
        tx.send(
            26,
            TesseractFrame::new(FrameType::Generate, 27, 26, marker.clone()),
        )
        .is_err(),
        "handshake with a substituted key-exchange value succeeded"
    );
    assert!(rx.recv_timeout(Duration::from_millis(300)).is_none());

    // 3. one side configured without encryption cannot talk to the other
    let rx = Endpoint::bind_with(24, "127.0.0.1:0", cfg.clone()).unwrap();
    let tx = Endpoint::bind_with(25, "127.0.0.1:0", auth::WireConfig::new(secret, false)).unwrap();
    tx.shared.set_addr(24, rx.addr());
    assert!(
        tx.send(
            24,
            TesseractFrame::new(FrameType::Generate, 25, 24, vec![1])
        )
        .is_err(),
        "mismatched encryption mode connected"
    );
    assert!(rx.recv_timeout(Duration::from_millis(300)).is_none());
}
