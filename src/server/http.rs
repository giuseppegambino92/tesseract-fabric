//! OpenAI-compatible HTTP/1.1 + Server-Sent-Events server over a live
//! `tesseract-fabric` cluster (std `TcpListener`, one thread per connection).
//!
//! * `POST /v1/chat/completions` — JSON or `"stream": true` SSE; every
//!   speculative verification step is flushed to the socket as soon as the
//!   head node accepts it (1–N tokens per chunk), and the last chunk / the
//!   response carries a `tesseract_telemetry` object.
//! * `GET /v1/models`, `GET /v1/cluster/status`, `GET /` (dashboard + chat).
//!
//! Concurrency, two modes:
//! * **Continuous batching** (`--sessions N`, H-TEP engines): one engine
//!   serves up to N requests at once. The head interleaves them in shared
//!   forward passes (one batched GEMM per projection and one all-reduce per
//!   layer for every active session), each session with its own KV slot on
//!   every rank; requests join and leave between steps. Each KV slot keeps
//!   its committed tokens, and a new request lands on the slot sharing its
//!   longest prefix.
//! * **Engine slots** (`--slots S`): S complete cluster engines (own node
//!   endpoints, KV caches, drafter state) on the same machines; file-backed
//!   weights are shared through the OS page cache. Concurrent requests take
//!   different slots; when every slot is busy, a request waits.
//!
//! Security ([`HttpSecurity`]): with an API key every `/v1/*` endpoint
//! requires `Authorization: Bearer <key>` (constant-time comparison); CORS
//! is off unless one origin is configured; requests are parsed with hard
//! limits enforced while bytes arrive (64 KB head, 8 MB body, no chunked
//! bodies); and admission is capped globally and per client IP, so excess
//! connections get an immediate 503 instead of a thread. TLS belongs in a
//! reverse proxy in front of the server.
//!
//! Hot-join: with a [`HotJoin`] watcher, workers that start announcing (or
//! stop) are probed and the affected slot is re-planned between requests;
//! a request in flight always finishes on the plan it started with.

use super::json::Json;
use crate::cluster::batch::BatchClient;
use crate::cluster::discovery::{
    probe_all, replan, slot_of, DiscoveryListener, EngineChoice, ProbedPeer,
};
use crate::cluster::{
    Assignment, Driver, EngineKind, GenerateExt, GenerateRequest, GenerateResult, KvMode,
    NodeStats, TesseractOptions, TokenBurst, KV_SERVICE_BASE,
};
use crate::model::ModelConfig;
use crate::tokenizer::{stop_tokens, ChatFormat, SamplingParams, Tokenizer};
use std::collections::HashMap;
use std::io::Read;
use std::io::{BufRead, BufReader, Write};
use std::net::{IpAddr, SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

const UI: &str = include_str!("ui.html");
const MAX_HEAD: usize = 64 << 10;
const MAX_BODY: usize = 8 << 20;
const MAX_HEADERS: usize = 100;

/// Access control and resource limits of the HTTP server.
#[derive(Clone, Debug)]
pub struct HttpSecurity {
    /// Bearer token required on every `/v1/*` endpoint (`None`: open API).
    pub api_key: Option<String>,
    /// The one origin allowed by CORS (`None`: no CORS headers at all).
    pub cors_origin: Option<String>,
    /// Concurrent connections served at once; more get an immediate 503.
    pub max_connections: usize,
    /// Concurrent connections per client IP.
    pub max_per_client: usize,
}

impl Default for HttpSecurity {
    fn default() -> Self {
        HttpSecurity {
            api_key: None,
            cors_origin: None,
            max_connections: 64,
            max_per_client: 16,
        }
    }
}

thread_local! {
    /// CORS origin of the connection this thread is serving (each connection
    /// has its own thread, so `respond` can add the header without plumbing).
    static CORS: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

fn cors_headers() -> String {
    CORS.with(|c| match c.borrow().as_deref() {
        Some(o) => format!(
            "Access-Control-Allow-Origin: {o}\r\nVary: Origin\r\nAccess-Control-Allow-Headers: Content-Type, Authorization\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        ),
        None => String::new(),
    })
}

/// A measured link to a node (from discovery probes), for the status page.
#[derive(Clone, Debug, Default)]
pub struct NodeLink {
    pub rtt_us: f64,
    pub bandwidth_gbps: f64,
    pub platform: String,
    pub ram_mb: u32,
}

/// How the server reaches an engine's head.
pub enum EngineLink {
    /// One request at a time (the slot's lock is held while it runs).
    Direct(Driver),
    /// Continuous batching: concurrent requests multiplexed by a client.
    Batched(BatchClient),
}

/// One engine slot: a cluster engine and the probed peers it runs on.
pub struct Engine {
    pub link: EngineLink,
    pub assignment: Assignment,
    pub probed: Vec<ProbedPeer>,
}

impl Engine {
    /// A slot driven one request at a time.
    pub fn direct(driver: Driver, assignment: Assignment, probed: Vec<ProbedPeer>) -> Self {
        Engine {
            link: EngineLink::Direct(driver),
            assignment,
            probed,
        }
    }

    /// A continuously batched engine (the head runs the session scheduler:
    /// `TesseractOptions::sessions > 0`).
    pub fn batched(driver: Driver, assignment: Assignment, probed: Vec<ProbedPeer>) -> Self {
        let client = BatchClient::new(driver, assignment.head);
        Engine {
            link: EngineLink::Batched(client),
            assignment,
            probed,
        }
    }

    /// Run `f` with the engine's driver (a batched engine runs it once no
    /// request is in flight).
    pub fn with_driver<T: Send + 'static>(
        &self,
        f: impl FnOnce(&Driver) -> T + Send + 'static,
    ) -> T {
        match &self.link {
            EngineLink::Direct(d) => f(d),
            EngineLink::Batched(c) => c.exec(f),
        }
    }

    pub fn stats(&self, ids: Vec<u16>) -> Vec<NodeStats> {
        match &self.link {
            EngineLink::Direct(d) => d.stats(&ids),
            EngineLink::Batched(c) => c.exec_now(move |d| d.stats(&ids)),
        }
    }

    fn links(&self) -> Vec<(u16, NodeLink)> {
        self.probed
            .iter()
            .map(|p| {
                (
                    p.peer.beacon.node_id,
                    NodeLink {
                        rtt_us: p.link.rtt_us,
                        bandwidth_gbps: p.link.bandwidth_gbps,
                        platform: p.peer.beacon.platform.name().into(),
                        ram_mb: p.peer.beacon.free_ram_mb,
                    },
                )
            })
            .collect()
    }
}

/// Membership watcher configuration (hot-join / leave).
pub struct HotJoin {
    pub listener: Arc<DiscoveryListener>,
    pub tess: TesseractOptions,
    pub choice: EngineChoice,
    /// A worker silent for this long is treated as gone.
    pub leave_after: Duration,
}

pub struct ServerState {
    pub slots: Vec<Mutex<Engine>>,
    /// Last status of each slot's nodes (served while the slot is busy).
    cached: Vec<Mutex<Json>>,
    pub cfg: ModelConfig,
    pub tokenizer: Tokenizer,
    pub chat: ChatFormat,
    pub stops: Vec<u32>,
    pub model_id: String,
    /// Weight format / origin shown by `/v1/models`.
    pub model_desc: Json,
    pub default_max_tokens: u32,
    pub hot_join: Option<HotJoin>,
    pub replans: AtomicU64,
    pub security: HttpSecurity,
    queue: SlotQueue,
    started: Instant,
    served: Mutex<(u64, u64)>,
}

impl ServerState {
    pub fn new(
        engines: Vec<Engine>,
        tokenizer: Tokenizer,
        model_id: &str,
        model_desc: Json,
    ) -> Self {
        assert!(!engines.is_empty(), "at least one engine slot");
        let n = engines.len();
        ServerState {
            cfg: engines[0].assignment.model.clone(),
            slots: engines.into_iter().map(Mutex::new).collect(),
            cached: (0..n).map(|_| Mutex::new(Json::Arr(vec![]))).collect(),
            chat: ChatFormat::detect(&tokenizer),
            stops: stop_tokens(&tokenizer),
            tokenizer,
            model_id: model_id.to_string(),
            model_desc,
            default_max_tokens: 256,
            hot_join: None,
            replans: AtomicU64::new(0),
            security: HttpSecurity::default(),
            queue: SlotQueue::default(),
            started: Instant::now(),
            served: Mutex::new((0, 0)),
        }
    }

    /// A free engine slot. Requests are served strictly in arrival order
    /// (FIFO tickets); a waiting request sleeps on a condition variable that
    /// is signalled whenever a slot is released, instead of polling.
    fn acquire(&self) -> (usize, SlotGuard<'_>) {
        let q = &self.queue;
        let mut t = q.state.lock().unwrap_or_else(|e| e.into_inner());
        let ticket = t.0;
        t.0 += 1;
        loop {
            if t.1 == ticket {
                for (i, s) in self.slots.iter().enumerate() {
                    // a slot poisoned by an earlier panic is still usable
                    // (and is re-planned by the membership watcher)
                    let g = match s.try_lock() {
                        Ok(g) => g,
                        Err(std::sync::TryLockError::Poisoned(p)) => p.into_inner(),
                        Err(std::sync::TryLockError::WouldBlock) => continue,
                    };
                    t.1 += 1; // the next ticket may now look for a slot
                    q.cv.notify_all();
                    return (
                        i,
                        SlotGuard {
                            guard: Some(g),
                            queue: q,
                        },
                    );
                }
            }
            // woken when a slot is released or the queue advances; the
            // timeout only covers slots released without a signal (the
            // membership watcher's short re-plans)
            t =
                q.cv.wait_timeout(t, Duration::from_millis(50))
                    .unwrap_or_else(|e| e.into_inner())
                    .0;
        }
    }
}

/// FIFO admission to engine slots: (next ticket to hand out, ticket now
/// allowed to take a free slot).
#[derive(Default)]
struct SlotQueue {
    state: Mutex<(u64, u64)>,
    cv: Condvar,
}

/// An engine slot held by one request; releasing it wakes the queue.
pub struct SlotGuard<'a> {
    guard: Option<MutexGuard<'a, Engine>>,
    queue: &'a SlotQueue,
}

impl std::ops::Deref for SlotGuard<'_> {
    type Target = Engine;
    fn deref(&self) -> &Engine {
        self.guard.as_ref().unwrap()
    }
}

impl std::ops::DerefMut for SlotGuard<'_> {
    fn deref_mut(&mut self) -> &mut Engine {
        self.guard.as_mut().unwrap()
    }
}

impl Drop for SlotGuard<'_> {
    fn drop(&mut self) {
        drop(self.guard.take());
        // take the queue lock so a waiter cannot miss the signal between its
        // check and its wait
        let _q = self.queue.state.lock().unwrap_or_else(|e| e.into_inner());
        self.queue.cv.notify_all();
    }
}

/// Watch beacons and re-plan a slot whenever its set of live workers
/// changes (between requests: the slot's lock is taken first).
fn membership_loop(st: Arc<ServerState>) {
    let Some(hj) = st.hot_join.as_ref() else {
        return;
    };
    loop {
        std::thread::sleep(Duration::from_millis(300));
        let live = hj.listener.peers(hj.leave_after);
        for (si, slot) in st.slots.iter().enumerate() {
            let mut eng = match slot.try_lock() {
                Ok(g) => g,
                Err(std::sync::TryLockError::Poisoned(p)) => p.into_inner(),
                Err(std::sync::TryLockError::WouldBlock) => continue,
            };
            let mut want: Vec<_> = live
                .iter()
                .filter(|p| slot_of(p.beacon.node_id) == si)
                .cloned()
                .collect();
            want.sort_by_key(|p| p.beacon.node_id);
            let have: Vec<u16> = eng.assignment.nodes.iter().map(|n| n.id).collect();
            let want_ids: Vec<u16> = want.iter().map(|p| p.beacon.node_id).collect();
            if want.is_empty() || want_ids == have {
                continue;
            }
            // probe newcomers; keep measurements of known nodes
            let known: Vec<ProbedPeer> = eng
                .probed
                .iter()
                .filter(|p| want_ids.contains(&p.peer.beacon.node_id))
                .cloned()
                .collect();
            let fresh: Vec<_> = want
                .into_iter()
                .filter(|p| !have.contains(&p.beacon.node_id))
                .collect();
            let Ok(new) = eng.with_driver(move |d| probe_all(d, fresh)) else {
                continue;
            };
            let mut peers = known;
            peers.extend(new);
            peers.sort_by_key(|p| p.peer.beacon.node_id);
            let (old, cfg, pc, tess, choice) = (
                eng.assignment.clone(),
                st.cfg.clone(),
                peers.clone(),
                hj.tess.clone(),
                hj.choice,
            );
            let (plan, a) = eng.with_driver(move |d| replan(d, &old, &cfg, &pc, tess, choice));
            eprintln!(
                "slot {si}: membership {have:?} → {want_ids:?}; re-planned: {} stages {:?} ({})",
                a.engine.name(),
                a.stages,
                plan.notes.join("; ")
            );
            if let EngineLink::Batched(c) = &eng.link {
                c.set_head(a.head);
            }
            eng.assignment = a;
            eng.probed = peers;
            st.replans.fetch_add(1, Ordering::Relaxed);
        }
    }
}

pub struct Server {
    listener: TcpListener,
    pub state: Arc<ServerState>,
}

impl Server {
    pub fn bind(addr: &str, state: ServerState) -> std::io::Result<Self> {
        Ok(Server {
            listener: TcpListener::bind(addr)?,
            state: Arc::new(state),
        })
    }

    pub fn local_addr(&self) -> SocketAddr {
        self.listener.local_addr().unwrap()
    }

    /// Accept connections forever (one thread each).
    pub fn serve(self) {
        if self.state.hot_join.is_some() {
            let st = self.state.clone();
            let _ = std::thread::Builder::new()
                .name("tess-membership".into())
                .spawn(move || membership_loop(st));
        }
        let admission = Arc::new(Admission::default());
        for conn in self.listener.incoming() {
            let Ok(mut s) = conn else { continue };
            let ip = s.peer_addr().map(|a| a.ip()).ok();
            let sec = &self.state.security;
            let Some(ticket) = admission.admit(ip, sec.max_connections, sec.max_per_client) else {
                // over capacity: answer at once, never spawn a thread
                let body =
                    r#"{"error":{"message":"server busy","type":"server_overloaded_error"}}"#;
                let _ = s.set_write_timeout(Some(Duration::from_secs(1)));
                let _ = s.write_all(
                    format!(
                        "HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nRetry-After: 1\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    )
                    .as_bytes(),
                );
                continue;
            };
            let st = self.state.clone();
            let _ = std::thread::Builder::new()
                .name("tess-http".into())
                .spawn(move || {
                    let _ticket = ticket;
                    handle(s, &st)
                });
        }
    }

    /// Serve on a background thread.
    pub fn spawn(self) -> std::thread::JoinHandle<()> {
        std::thread::Builder::new()
            .name("tess-http-accept".into())
            .spawn(move || self.serve())
            .expect("spawn server")
    }
}

// --- admission -----------------------------------------------------------------------

/// Counts of connections being served, globally and per client IP.
#[derive(Default)]
struct Admission {
    total: AtomicUsize,
    per_ip: Mutex<HashMap<IpAddr, usize>>,
}

/// One admitted connection; releases its slot when dropped.
struct Ticket {
    adm: Arc<Admission>,
    ip: Option<IpAddr>,
}

impl Admission {
    fn admit(
        self: &Arc<Self>,
        ip: Option<IpAddr>,
        max: usize,
        per_client: usize,
    ) -> Option<Ticket> {
        let mut m = self.per_ip.lock().unwrap_or_else(|e| e.into_inner());
        if self.total.load(Ordering::Acquire) >= max {
            return None;
        }
        if let Some(ip) = ip {
            let n = m.entry(ip).or_insert(0);
            if *n >= per_client {
                return None;
            }
            *n += 1;
        }
        self.total.fetch_add(1, Ordering::AcqRel);
        Some(Ticket {
            adm: self.clone(),
            ip,
        })
    }
}

impl Drop for Ticket {
    fn drop(&mut self) {
        let mut m = self.adm.per_ip.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(ip) = self.ip {
            if let Some(n) = m.get_mut(&ip) {
                *n -= 1;
                if *n == 0 {
                    m.remove(&ip);
                }
            }
        }
        self.adm.total.fetch_sub(1, Ordering::AcqRel);
    }
}

// --- HTTP plumbing ------------------------------------------------------------------

#[derive(Debug)]
pub struct Request {
    pub method: String,
    pub path: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Request {
    pub fn header(&self, k: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(k))
            .map(|(_, v)| v.as_str())
    }
}

/// Append one line (through `\n`) to `out`, never letting `out` exceed
/// `limit` bytes: the limit is enforced while bytes arrive, so a client
/// streaming a line without a newline cannot make the server allocate past
/// it. Returns the bytes read (0 at end of stream).
fn read_line_bounded(
    r: &mut impl BufRead,
    out: &mut Vec<u8>,
    limit: usize,
) -> Result<usize, String> {
    let start = out.len();
    loop {
        let buf = r.fill_buf().map_err(|e| e.to_string())?;
        if buf.is_empty() {
            return Ok(out.len() - start);
        }
        let room = limit.saturating_sub(out.len());
        let (take, done) = match buf.iter().position(|&b| b == b'\n') {
            Some(i) => (i + 1, true),
            None => (buf.len(), false),
        };
        if take > room {
            return Err("request head too large".into());
        }
        out.extend_from_slice(&buf[..take]);
        r.consume(take);
        if done {
            return Ok(out.len() - start);
        }
    }
}

/// Parse one HTTP/1.1 request (Content-Length bodies), with the head capped
/// at 64 KB / 100 header lines and the body at 8 MB, all enforced as bytes
/// arrive. Chunked bodies are refused.
pub fn read_request(r: &mut impl BufRead) -> Result<Request, String> {
    let mut head = Vec::new();
    let mut lines = 0usize;
    loop {
        let n = read_line_bounded(r, &mut head, MAX_HEAD)?;
        if n == 0 {
            return Err("connection closed".into());
        }
        if head.ends_with(b"\r\n\r\n") || head.ends_with(b"\n\n") {
            break;
        }
        lines += 1;
        if lines > MAX_HEADERS + 1 {
            return Err("too many header lines".into());
        }
    }
    let text = String::from_utf8(head).map_err(|_| "non-UTF-8 request head")?;
    let mut lines = text.lines();
    let mut start = lines.next().ok_or("empty request")?.split_whitespace();
    let method = start.next().ok_or("no method")?.to_string();
    let path = start.next().ok_or("no path")?.to_string();
    let headers: Vec<(String, String)> = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
        .collect();
    if headers
        .iter()
        .any(|(k, _)| k.eq_ignore_ascii_case("transfer-encoding"))
    {
        return Err("Transfer-Encoding is not supported; send Content-Length".into());
    }
    let lens: Vec<&str> = headers
        .iter()
        .filter(|(k, _)| k.eq_ignore_ascii_case("content-length"))
        .map(|(_, v)| v.as_str())
        .collect();
    if lens.len() > 1 {
        return Err("duplicate Content-Length".into());
    }
    let len = match lens.first() {
        Some(v) => v.parse::<usize>().map_err(|_| "bad Content-Length")?,
        None => 0,
    };
    if len > MAX_BODY {
        return Err("body too large".into());
    }
    // grow with the bytes actually received, not with the declared length
    let mut body = Vec::with_capacity(len.min(64 << 10));
    r.take(len as u64)
        .read_to_end(&mut body)
        .map_err(|e| e.to_string())?;
    if body.len() != len {
        return Err("body shorter than Content-Length".into());
    }
    Ok(Request {
        method,
        path,
        headers,
        body,
    })
}

fn respond(s: &mut TcpStream, code: u16, ctype: &str, body: &[u8]) {
    let reason = match code {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        405 => "Method Not Allowed",
        503 => "Service Unavailable",
        _ => "Internal Server Error",
    };
    let auth = if code == 401 {
        "WWW-Authenticate: Bearer\r\n"
    } else {
        ""
    };
    let head = format!(
        "HTTP/1.1 {code} {reason}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\n{}{auth}Connection: close\r\n\r\n",
        body.len(),
        cors_headers()
    );
    let _ = s.write_all(head.as_bytes());
    let _ = s.write_all(body);
    let _ = s.flush();
}

/// A request that the cluster could not complete: a `503` before any byte
/// was sent, or an SSE error event closing a stream already under way.
fn cluster_error(s: &mut TcpStream, streaming: bool, msg: &str) {
    let err = Json::obj(vec![(
        "error",
        Json::obj(vec![
            ("message", Json::str(msg)),
            ("type", Json::str("cluster_error")),
        ]),
    )]);
    if streaming {
        let _ = s.write_all(format!("data: {}\n\ndata: [DONE]\n\n", err.dump()).as_bytes());
        let _ = s.flush();
    } else {
        respond(s, 503, "application/json", err.dump().as_bytes());
    }
}

fn json_error(s: &mut TcpStream, code: u16, msg: &str) {
    let j = Json::obj(vec![(
        "error",
        Json::obj(vec![
            ("message", Json::str(msg)),
            (
                "type",
                Json::str(if code == 401 {
                    "authentication_error"
                } else {
                    "invalid_request_error"
                }),
            ),
        ]),
    )]);
    respond(s, code, "application/json", j.dump().as_bytes());
}

/// Whether `req` carries the configured API key (always true without one).
fn authorized(req: &Request, sec: &HttpSecurity) -> bool {
    let Some(key) = sec.api_key.as_deref() else {
        return true;
    };
    let given = req
        .header("authorization")
        .and_then(|v| {
            v.strip_prefix("Bearer ")
                .or_else(|| v.strip_prefix("bearer "))
        })
        .unwrap_or("")
        .trim();
    crate::wire::auth::ct_eq(given.as_bytes(), key.as_bytes())
}

fn handle(mut s: TcpStream, st: &ServerState) {
    let _ = s.set_nodelay(true);
    let _ = s.set_read_timeout(Some(Duration::from_secs(30)));
    let _ = s.set_write_timeout(Some(Duration::from_secs(60)));
    CORS.with(|c| *c.borrow_mut() = st.security.cors_origin.clone());
    let req = {
        let mut r = BufReader::new(match s.try_clone() {
            Ok(c) => c,
            Err(_) => return,
        });
        match read_request(&mut r) {
            Ok(q) => q,
            Err(e) => return json_error(&mut s, 400, &e),
        }
    };
    let path = req.path.split('?').next().unwrap_or("/").to_string();
    if path.starts_with("/v1/") && req.method != "OPTIONS" && !authorized(&req, &st.security) {
        return json_error(
            &mut s,
            401,
            "missing or invalid API key (Authorization: Bearer <key>)",
        );
    }
    match (req.method.as_str(), path.as_str()) {
        ("OPTIONS", _) => respond(&mut s, 204, "text/plain", b""),
        ("GET", "/") | ("GET", "/index.html") => {
            respond(&mut s, 200, "text/html; charset=utf-8", UI.as_bytes())
        }
        ("GET", "/v1/models") => respond(
            &mut s,
            200,
            "application/json",
            models_json(st).dump().as_bytes(),
        ),
        ("GET", "/v1/cluster/status") => {
            let j = cluster_status(st);
            respond(&mut s, 200, "application/json", j.dump().as_bytes())
        }
        ("POST", "/v1/chat/completions") => chat(&mut s, st, &req),
        (_, "/v1/chat/completions") | (_, "/v1/models") | (_, "/v1/cluster/status") => {
            json_error(&mut s, 405, "method not allowed")
        }
        _ => json_error(&mut s, 404, "not found"),
    }
}

// --- endpoints ------------------------------------------------------------------------

fn models_json(st: &ServerState) -> Json {
    Json::obj(vec![
        ("object", Json::str("list")),
        (
            "data",
            Json::Arr(vec![Json::obj(vec![
                ("id", Json::str(st.model_id.clone())),
                ("object", Json::str("model")),
                ("created", Json::num(0.0)),
                ("owned_by", Json::str("tesseract-fabric")),
                ("tesseract", st.model_desc.clone()),
            ])]),
        ),
    ])
}

fn all_endpoint_ids(a: &Assignment) -> Vec<u16> {
    let mut v: Vec<u16> = a.nodes.iter().map(|n| n.id).collect();
    v.extend(a.nodes.iter().map(|n| KV_SERVICE_BASE + n.id));
    v
}

fn role_of(a: &Assignment, id: u16) -> Vec<String> {
    let mut roles = Vec::new();
    let world = a.stages.len();
    for (r, &(n, s, e)) in a.stages.iter().enumerate() {
        if n != id {
            continue;
        }
        roles.push(match a.engine {
            EngineKind::TesseractHtep => format!(
                "H-TEP Shard (rank {r}/{world}: replicated attention, 1/{world} of every expert)"
            ),
            _ => format!("Pipeline Stage (layers {}–{})", s, e.saturating_sub(1)),
        });
    }
    if a.head == id {
        roles.push("Coordinator (sampling + verification)".into());
    }
    if a.tess.drafter == Some(id) {
        roles.push("Edge Drafter".into());
    }
    if a.tess.kv_mode == KvMode::Disaggregated {
        roles.push("Disaggregated KV Partner".into());
    }
    if roles.is_empty() {
        roles.push("Idle".into());
    }
    roles
}

/// One slot's status (needs the slot's engine).
fn slot_status(si: usize, eng: &Engine) -> Json {
    let a = &eng.assignment;
    let stats = eng.stats(all_endpoint_ids(a));
    let by = |id: u16| stats.iter().find(|s| s.node == id);
    let links = eng.links();
    let mut nodes = Vec::new();
    let mut loads = Vec::new();
    for n in &a.nodes {
        let s = by(n.id);
        let kv = by(KV_SERVICE_BASE + n.id).map(|s| s.kv_bytes).unwrap_or(0)
            + s.map(|s| s.kv_bytes).unwrap_or(0);
        let w = s.map(|s| s.weight_bytes).unwrap_or(0);
        loads.push((w + kv) as f64);
        let link = links
            .iter()
            .find(|(i, _)| *i == n.id)
            .map(|(_, l)| l.clone());
        nodes.push(Json::obj(vec![
            ("id", Json::num(n.id as f64)),
            ("slot", Json::num(si as f64)),
            ("addr", Json::str(n.addr.to_string())),
            (
                "roles",
                Json::Arr(role_of(a, n.id).into_iter().map(Json::Str).collect()),
            ),
            ("weight_bytes", Json::num(w as f64)),
            ("kv_bytes", Json::num(kv as f64)),
            ("kv_capacity_bytes", Json::num(n.kv_capacity as f64)),
            (
                "bytes_sent",
                Json::num(s.map(|s| s.bytes_sent).unwrap_or(0) as f64),
            ),
            (
                "commits",
                Json::num(s.map(|s| s.commits).unwrap_or(0) as f64),
            ),
            (
                "busy_ms",
                Json::num(s.map(|s| s.busy_ns).unwrap_or(0) as f64 / 1e6),
            ),
            (
                "link",
                match link {
                    Some(l) => Json::obj(vec![
                        ("rtt_us", Json::num(l.rtt_us)),
                        ("bandwidth_gbps", Json::num(l.bandwidth_gbps)),
                        ("platform", Json::str(l.platform)),
                        ("ram_mb", Json::num(l.ram_mb as f64)),
                    ]),
                    None => Json::Null,
                },
            ),
        ]));
    }
    let mean = loads.iter().sum::<f64>() / loads.len().max(1) as f64;
    let max = loads.iter().cloned().fold(0.0, f64::max);
    Json::obj(vec![
        ("slot", Json::num(si as f64)),
        ("engine", Json::str(a.engine.name())),
        ("head", Json::num(a.head as f64)),
        (
            "stages",
            Json::Arr(
                a.stages
                    .iter()
                    .map(|&(n, s, e)| {
                        Json::obj(vec![
                            ("node", Json::num(n as f64)),
                            ("first_layer", Json::num(s as f64)),
                            ("end_layer", Json::num(e as f64)),
                        ])
                    })
                    .collect(),
            ),
        ),
        (
            "kv_mode",
            Json::str(if a.tess.kv_mode == KvMode::Disaggregated {
                "disaggregated"
            } else {
                "layer-coupled"
            }),
        ),
        (
            "memory_balance_max_over_mean",
            Json::num(if mean > 0.0 { max / mean } else { 1.0 }),
        ),
        ("nodes", Json::Arr(nodes)),
    ])
}

fn cluster_status(st: &ServerState) -> Json {
    let mut slots = Vec::new();
    for (si, slot) in st.slots.iter().enumerate() {
        let j = match slot.try_lock() {
            Ok(eng) => {
                let j = slot_status(si, &eng);
                *st.cached[si].lock().unwrap() = j.clone();
                j
            }
            // a request is running on this slot: report its last status
            Err(_) => st.cached[si].lock().unwrap().clone(),
        };
        slots.push(j);
    }
    let nodes: Vec<Json> = slots
        .iter()
        .flat_map(|j| {
            j.get("nodes")
                .and_then(|n| n.as_array())
                .unwrap_or(&[])
                .to_vec()
        })
        .collect();
    let first = slots.first().cloned().unwrap_or(Json::Null);
    let pick = |k: &str| first.get(k).cloned().unwrap_or(Json::Null);
    let served = *st.served.lock().unwrap();
    Json::obj(vec![
        ("engine", pick("engine")),
        ("model", Json::str(st.model_id.clone())),
        ("chat_format", Json::str(st.chat.name())),
        ("head", pick("head")),
        ("stages", pick("stages")),
        ("kv_mode", pick("kv_mode")),
        (
            "memory_balance_max_over_mean",
            pick("memory_balance_max_over_mean"),
        ),
        ("active_nodes", Json::num(nodes.len() as f64)),
        ("nodes", Json::Arr(nodes)),
        ("slots", Json::Arr(slots)),
        (
            "replans",
            Json::num(st.replans.load(Ordering::Relaxed) as f64),
        ),
        ("requests_served", Json::num(served.0 as f64)),
        ("tokens_generated", Json::num(served.1 as f64)),
        ("uptime_s", Json::num(st.started.elapsed().as_secs_f64())),
    ])
}

/// Prompt text for a chat (a plain role-tagged template; GGUF Jinja chat
/// templates are not interpreted).
pub fn chat_prompt(messages: &[(String, String)]) -> String {
    let mut p = String::new();
    for (role, content) in messages {
        p.push_str(&format!("<|{role}|>\n{content}\n"));
    }
    p.push_str("<|assistant|>\n");
    p
}

/// Parsed `/v1/chat/completions` body.
#[derive(Clone, Debug)]
pub struct ChatRequest {
    pub messages: Vec<(String, String)>,
    pub max_tokens: u32,
    pub stream: bool,
    pub sampling: SamplingParams,
}

pub fn parse_chat(j: &Json, default_max: u32) -> Result<ChatRequest, String> {
    let msgs = j
        .get("messages")
        .and_then(|m| m.as_array())
        .ok_or("messages must be an array")?;
    let mut messages = Vec::new();
    for m in msgs {
        let role = m
            .get("role")
            .and_then(|r| r.as_str())
            .ok_or("message without role")?;
        let content = match m.get("content") {
            Some(Json::Str(s)) => s.clone(),
            Some(Json::Arr(parts)) => parts
                .iter()
                .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join(""),
            _ => return Err("message content must be a string".into()),
        };
        messages.push((role.to_string(), content));
    }
    if messages.is_empty() {
        return Err("messages is empty".into());
    }
    let f = |k: &str| j.get(k).and_then(|v| v.as_f64());
    let max_tokens = f("max_tokens")
        .or_else(|| f("max_completion_tokens"))
        .map(|v| v.clamp(1.0, 4096.0) as u32)
        .unwrap_or(default_max);
    let sampling = SamplingParams {
        temperature: f("temperature").unwrap_or(0.7).clamp(0.0, 4.0) as f32,
        top_k: f("top_k").unwrap_or(0.0).max(0.0) as usize,
        top_p: f("top_p").unwrap_or(1.0).clamp(0.0, 1.0) as f32,
        min_p: f("min_p").unwrap_or(0.0).clamp(0.0, 1.0) as f32,
        repetition_penalty: f("repetition_penalty").unwrap_or(1.0).clamp(0.5, 4.0) as f32,
        repeat_last_n: 64,
        seed: f("seed").map(|s| s as u64).unwrap_or(0x5EED),
    };
    Ok(ChatRequest {
        messages,
        max_tokens,
        stream: j.get("stream").and_then(|v| v.as_bool()).unwrap_or(false),
        sampling,
    })
}

fn sum_stats(v: &[NodeStats]) -> (u64, u64) {
    (
        v.iter().map(|s| s.bytes_sent).sum(),
        v.iter().map(|s| s.act_bytes).sum(),
    )
}

/// The `tesseract_telemetry` object for one finished generation.
#[allow(clippy::too_many_arguments)]
fn telemetry(
    st: &ServerState,
    a: &Assignment,
    res: &GenerateResult,
    ttft_ms: f64,
    before: &[NodeStats],
    after: &[NodeStats],
    prompt_len: usize,
) -> Json {
    let n = res.tokens.len().max(1) as f64;
    let decode_s = res.decode_ns as f64 / 1e9;
    let drafted: u64 = res
        .budgets
        .iter()
        .map(|&b| b.saturating_sub(1) as u64)
        .sum();
    let accepted: u64 = res
        .per_epoch
        .iter()
        .map(|&e| e.saturating_sub(1) as u64)
        .sum();
    let (b0, a0) = sum_stats(before);
    let (b1, a1) = sum_stats(after);
    let hops = if a.engine == EngineKind::Tesseract {
        a.stages.len().saturating_sub(1)
    } else {
        0
    };
    let rows: u64 =
        prompt_len.saturating_sub(1) as u64 + res.budgets.iter().map(|&b| b as u64).sum::<u64>();
    let fp16 = rows * hops as u64 * st.cfg.d_model as u64 * 2;
    let kv_nodes = after
        .iter()
        .filter(|s| s.kv_bytes > 0)
        .map(|s| s.node % KV_SERVICE_BASE)
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    Json::obj(vec![
        (
            "tokens_per_sec",
            Json::num(if decode_s > 0.0 {
                res.tokens.len() as f64 / decode_s
            } else {
                0.0
            }),
        ),
        ("ttft_ms", Json::num(ttft_ms)),
        ("prefill_ms", Json::num(res.prefill_ns as f64 / 1e6)),
        (
            "speculative_acceptance_rate",
            Json::num(if drafted > 0 {
                accepted as f64 / drafted as f64
            } else {
                0.0
            }),
        ),
        (
            "accepted_tokens_per_step",
            Json::num(if res.per_epoch.is_empty() {
                0.0
            } else {
                res.per_epoch.iter().sum::<u32>() as f64 / res.per_epoch.len() as f64
            }),
        ),
        ("verification_steps", Json::num(res.per_epoch.len() as f64)),
        (
            "wire_bytes_per_token",
            Json::num(b1.saturating_sub(b0) as f64 / n),
        ),
        // inter-stage activation bytes vs raw FP16 rows (pipeline engines only)
        (
            "klt_compression_ratio",
            if fp16 > 0 {
                Json::num(a1.saturating_sub(a0) as f64 / fp16 as f64)
            } else {
                Json::Null
            },
        ),
        ("kv_fabric_nodes_used", Json::num(kv_nodes as f64)),
        ("engine", Json::str(a.engine.name())),
    ])
}

fn chunk_json(
    id: &str,
    created: u64,
    model: &str,
    delta: Json,
    finish: Option<&str>,
    extra: Vec<(&str, Json)>,
) -> String {
    let mut fields = vec![
        ("id", Json::str(id)),
        ("object", Json::str("chat.completion.chunk")),
        ("created", Json::num(created as f64)),
        ("model", Json::str(model)),
        (
            "choices",
            Json::Arr(vec![Json::obj(vec![
                ("index", Json::num(0.0)),
                ("delta", delta),
                ("finish_reason", finish.map(Json::str).unwrap_or(Json::Null)),
            ])]),
        ),
    ];
    fields.extend(extra);
    format!("data: {}\n\n", Json::obj(fields).dump())
}

fn chat(s: &mut TcpStream, st: &ServerState, req: &Request) {
    let body = match std::str::from_utf8(&req.body)
        .ok()
        .and_then(|t| Json::parse(t).ok())
    {
        Some(j) => j,
        None => return json_error(s, 400, "body must be JSON"),
    };
    let cr = match parse_chat(&body, st.default_max_tokens) {
        Ok(c) => c,
        Err(e) => return json_error(s, 400, &e),
    };
    let t_req = Instant::now();
    let tok = &st.tokenizer;
    let mut prompt: Vec<u32> = if st.chat == ChatFormat::Plain {
        tok.bos.into_iter().collect()
    } else {
        Vec::new()
    };
    prompt.extend(st.chat.encode(tok, &cr.messages));
    while prompt.len() < 2 {
        prompt.insert(0, tok.bos.unwrap_or(b'\n' as u32 % st.cfg.vocab as u32));
    }
    prompt.retain(|&t| (t as usize) < st.cfg.vocab);
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let id = format!("chatcmpl-tess-{created:x}{:04x}", prompt.len());
    let greq = GenerateRequest {
        prompt: prompt.clone(),
        max_new: cr.max_tokens,
        seed: cr.sampling.seed,
        temperature: cr.sampling.temperature,
        record_logits: false,
    };
    let ext = GenerateExt {
        stream: cr.stream,
        stop: st.stops.clone(),
        sampling: Some(cr.sampling.clone()),
    };
    // a batched engine takes the request alongside the ones in flight; a
    // direct slot is held for the whole request
    let (slot, mut guard, client, assignment) = {
        let (slot, eng) = st.acquire();
        match &eng.link {
            EngineLink::Batched(c) => {
                let (c, a) = (c.clone(), eng.assignment.clone());
                drop(eng);
                (slot, None, Some(c), a)
            }
            EngineLink::Direct(_) => {
                let a = eng.assignment.clone();
                (slot, Some(eng), None, a)
            }
        }
    };
    // a node of this slot already failed: answer now; the membership
    // watcher drops it and re-plans the slot within a few seconds
    if let Some(n) = guard.as_ref().and_then(|g| match &g.link {
        EngineLink::Direct(d) => d.lost_node(),
        EngineLink::Batched(_) => None,
    }) {
        drop(guard);
        return cluster_error(
            s,
            false,
            &format!(
                "cluster node {n} is unavailable; the cluster is being re-planned, retry shortly"
            ),
        );
    }
    let ids = all_endpoint_ids(&assignment);
    let before = match &guard {
        Some(eng) => eng.stats(ids.clone()),
        None => Vec::new(),
    };
    let mut dec = tok.stream();
    let mut text = String::new();
    let mut ttft: Option<f64> = None;
    let mut stream_ok = true;
    if cr.stream {
        let head = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n{}Connection: close\r\n\r\n",
            cors_headers()
        );
        stream_ok = s.write_all(head.as_bytes()).is_ok()
            && s.write_all(
                chunk_json(
                    &id,
                    created,
                    &st.model_id,
                    Json::obj(vec![
                        ("role", Json::str("assistant")),
                        ("content", Json::str("")),
                    ]),
                    None,
                    vec![],
                )
                .as_bytes(),
            )
            .is_ok();
        let _ = s.flush();
    }
    let res = {
        let mut on_burst = |b: &TokenBurst| {
            if ttft.is_none() {
                ttft = Some(t_req.elapsed().as_secs_f64() * 1e3);
            }
            let mut piece = String::new();
            for &t in &b.tokens {
                if !ext.stop.contains(&t) {
                    piece.push_str(&dec.push(t));
                }
            }
            text.push_str(&piece);
            if cr.stream && stream_ok {
                let step = Json::obj(vec![
                    ("tokens", Json::num(b.tokens.len() as f64)),
                    ("tree_size", Json::num(b.budget as f64)),
                ]);
                let c = chunk_json(
                    &id,
                    created,
                    &st.model_id,
                    Json::obj(vec![("content", Json::str(piece))]),
                    None,
                    vec![("tesseract_step", step)],
                );
                stream_ok = s.write_all(c.as_bytes()).is_ok() && s.flush().is_ok();
            }
        };
        // the slot guard lives outside this closure, so even an unexpected
        // panic cannot poison the slot: it becomes an error response
        let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            match (&mut guard, &client) {
                (Some(eng), _) => match &eng.link {
                    EngineLink::Direct(d) => {
                        d.try_generate_stream(assignment.head, &greq, &ext, &mut on_burst)
                    }
                    EngineLink::Batched(c) => Ok(c.generate_stream(&greq, &ext, &mut on_burst)),
                },
                (None, Some(c)) => Ok(c.generate_stream(&greq, &ext, &mut on_burst)),
                _ => unreachable!(),
            }
        }));
        match run {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => {
                drop(guard);
                return cluster_error(s, cr.stream, &format!("{e}; retry shortly"));
            }
            Err(_) => {
                drop(guard);
                return cluster_error(s, cr.stream, "internal error while generating");
            }
        }
    };
    let after = match &guard {
        Some(eng) => eng.stats(ids.clone()),
        None => Vec::new(),
    };
    let tail = dec.finish();
    text.push_str(&tail);
    if !cr.stream {
        let kept: Vec<u32> = res
            .tokens
            .iter()
            .copied()
            .filter(|t| !ext.stop.contains(t))
            .collect();
        text = tok.decode(&kept);
    }
    let finish = if res.tokens.last().is_some_and(|t| ext.stop.contains(t)) {
        "stop"
    } else {
        "length"
    };
    let ttft_ms = ttft.unwrap_or_else(|| {
        (res.prefill_ns + res.token_times_ns.first().copied().unwrap_or(0)) as f64 / 1e6
    });
    let mut tel = telemetry(
        st,
        &assignment,
        &res,
        ttft_ms,
        &before,
        &after,
        prompt.len(),
    );
    if let Json::Obj(v) = &mut tel {
        v.push(("slot".into(), Json::num(slot as f64)));
        v.push((
            "prefix_cached_tokens".into(),
            Json::num(res.prefix_reused as f64),
        ));
        v.push(("continuous_batching".into(), Json::Bool(client.is_some())));
    }
    drop(guard);
    {
        let mut g = st.served.lock().unwrap();
        g.0 += 1;
        g.1 += res.tokens.len() as u64;
    }
    let completion = res.tokens.iter().filter(|t| !ext.stop.contains(t)).count();
    let usage = Json::obj(vec![
        ("prompt_tokens", Json::num(prompt.len() as f64)),
        ("completion_tokens", Json::num(completion as f64)),
        (
            "total_tokens",
            Json::num((prompt.len() + completion) as f64),
        ),
    ]);
    if cr.stream {
        if stream_ok {
            let c = chunk_json(
                &id,
                created,
                &st.model_id,
                Json::obj(vec![("content", Json::str(tail))]),
                Some(finish),
                vec![("usage", usage), ("tesseract_telemetry", tel)],
            );
            let _ = s.write_all(c.as_bytes());
            let _ = s.write_all(b"data: [DONE]\n\n");
            let _ = s.flush();
        }
        return;
    }
    let j = Json::obj(vec![
        ("id", Json::str(id)),
        ("object", Json::str("chat.completion")),
        ("created", Json::num(created as f64)),
        ("model", Json::str(st.model_id.clone())),
        (
            "choices",
            Json::Arr(vec![Json::obj(vec![
                ("index", Json::num(0.0)),
                (
                    "message",
                    Json::obj(vec![
                        ("role", Json::str("assistant")),
                        ("content", Json::str(text)),
                    ]),
                ),
                ("finish_reason", Json::str(finish)),
            ])]),
        ),
        ("usage", usage),
        ("tesseract_telemetry", tel),
    ]);
    respond(s, 200, "application/json", j.dump().as_bytes());
}
