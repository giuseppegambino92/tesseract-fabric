//! Fault injection with real processes: a worker is killed (SIGKILL) in the
//! middle of a streaming generation. The request must end promptly with an
//! explicit error (no hang), the server must survive, and once the dead
//! worker is dropped and the cluster re-planned, requests must succeed on
//! the surviving workers. Node traffic is encrypted (custom cluster secret).

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpStream, UdpSocket};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use tesseract_fabric::server::json::Json;

const SECRET: &str = "fault-injection test secret, not for production";

struct Proc(Child);
impl Drop for Proc {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn log() -> Stdio {
    if std::env::var_os("CI").is_some() {
        Stdio::inherit()
    } else {
        Stdio::null()
    }
}

fn free_udp_port() -> u16 {
    UdpSocket::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn worker(id: u16, seed: &str) -> Proc {
    Proc(
        Command::new(env!("CARGO_BIN_EXE_tesseract-node"))
            .args([
                "--role",
                "worker",
                "--id",
                &id.to_string(),
                "--listen",
                "127.0.0.1:0",
            ])
            .args(["--discover", "--seeds", seed, "--cluster-secret", SECRET])
            .stdout(Stdio::null())
            .stderr(log())
            .spawn()
            .unwrap(),
    )
}

fn request(addr: &str, body: &str) -> TcpStream {
    let mut s = TcpStream::connect(addr).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(60))).unwrap();
    write!(
        s,
        "POST /v1/chat/completions HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    s
}

fn get(addr: &str, path: &str) -> String {
    let mut s = TcpStream::connect(addr).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(60))).unwrap();
    write!(s, "GET {path} HTTP/1.1\r\n\r\n").unwrap();
    let mut out = String::new();
    let _ = s.read_to_string(&mut out);
    out
}

#[test]
fn killing_a_worker_mid_generation_fails_the_request_cleanly_and_the_cluster_recovers() {
    let port = free_udp_port();
    let seed = format!("127.0.0.1:{port}");
    let mut server = Command::new(env!("CARGO_BIN_EXE_tesseract-serve"))
        .args([
            "--http",
            "127.0.0.1:0",
            "--model",
            "tiny",
            "--discover",
            "3",
        ])
        .args([
            "--discovery-bind",
            &seed,
            "--cluster-secret",
            SECRET,
            "--max-tokens",
            "4096",
        ])
        .stdout(Stdio::piped())
        .stderr(log())
        .spawn()
        .unwrap();
    let out = server.stdout.take();
    let server = Proc(server);
    let mut workers: Vec<Proc> = (1..=3).map(|id| worker(id, &seed)).collect();
    let mut lines = BufReader::new(out.unwrap()).lines();
    let http = loop {
        let l = lines.next().expect("server exited").unwrap();
        if let Some(a) = l.strip_prefix("listening on http://") {
            break a.to_string();
        }
    };
    // pick a worker the solver actually uses for compute (a slower machine
    // may get a pipeline over fewer nodes than it was offered)
    let status = get(&http, "/v1/cluster/status");
    assert!(status.starts_with("HTTP/1.1 200"), "{status}");
    let st = Json::parse(status.split_once("\r\n\r\n").unwrap().1).unwrap();
    let victim = st
        .get("nodes")
        .and_then(|n| n.as_array())
        .unwrap()
        .iter()
        .filter(|n| {
            n.get("roles").and_then(|r| r.as_array()).is_some_and(|r| {
                r.iter().filter_map(|x| x.as_str()).any(|x| {
                    x.contains("Shard") || x.contains("Stage") || x.contains("Coordinator")
                })
            })
        })
        .filter_map(|n| n.get("id").and_then(|i| i.as_f64()).map(|i| i as u16))
        .find(|id| (1..=3).contains(id))
        .expect("a worker with a compute role");

    // a long streaming request; kill worker 2 once tokens are flowing
    let body = r#"{"stream":true,"max_tokens":4096,"temperature":0,"messages":[{"role":"user","content":"keep going"}]}"#;
    let s = request(&http, body);
    let mut r = BufReader::new(s);
    let mut chunks = 0;
    let mut line = String::new();
    while chunks < 3 {
        line.clear();
        assert!(r.read_line(&mut line).unwrap() > 0, "stream ended early");
        if line.starts_with("data: {") {
            chunks += 1;
        }
    }
    let killed_at = Instant::now();
    workers.remove(victim as usize - 1); // SIGKILL it (Proc::drop)
    let mut ended_with_error = false;
    loop {
        line.clear();
        match r.read_line(&mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) => {
                if line.contains("cluster_error") {
                    ended_with_error = true;
                }
                if line.starts_with("data: [DONE]") {
                    break;
                }
            }
        }
    }
    let took = killed_at.elapsed();
    assert!(ended_with_error, "the stream did not report the failure");
    assert!(
        took < Duration::from_secs(15),
        "failure took {took:?} to surface"
    );

    // the server survives; after the dead worker is dropped (3 s without
    // beacons) the slot is re-planned on the two survivors
    let t = Instant::now();
    let ok = loop {
        let mut s = request(
            &http,
            r#"{"max_tokens":32,"temperature":0,"messages":[{"role":"user","content":"hello again"}]}"#,
        );
        let mut resp = String::new();
        let _ = s.read_to_string(&mut resp);
        if resp.starts_with("HTTP/1.1 200") {
            break resp;
        }
        assert!(
            resp.starts_with("HTTP/1.1 503") && resp.contains("cluster_error"),
            "unexpected response while recovering: {resp}"
        );
        assert!(
            t.elapsed() < Duration::from_secs(30),
            "cluster never recovered"
        );
        std::thread::sleep(Duration::from_millis(500));
    };
    assert!(ok.contains("chat.completion"));
    eprintln!(
        "worker {victim} killed: failure reported {:.2} s later; served again after {:.1} s",
        took.as_secs_f64(),
        killed_at.elapsed().as_secs_f64()
    );
    let status = get(&http, "/v1/cluster/status");
    assert!(
        !status.contains(&format!("\"id\":{victim},")),
        "the dead worker is still planned: {status}"
    );
    drop(server);
}
