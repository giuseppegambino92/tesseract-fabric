//! Live TCP transport with physical link shaping.
//!
//! Every directed logical link is a real `TcpStream` owned by a writer thread.
//! Before a frame is written, the shaper models the physical link it stands
//! for: FIFO serialisation at the link bandwidth (`bytes·8/BW`), queueing
//! behind earlier frames, then one-way propagation latency (+ jitter). The
//! writer thread holds the frame until that delivery instant and then writes
//! it to the socket, so the receiver observes realistic arrival times while
//! the bytes themselves really cross the OS network stack.
//!
//! Large frames to peers with several links are striped by the
//! `MultipathScheduler`; receivers reassemble and acknowledge stripes so the
//! scheduler learns per-link latency/bandwidth online.

use super::multipath::{LinkProfile, MultipathScheduler};
use super::protocol::{flags, FrameType, TesseractFrame};
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::io::{BufReader, BufWriter, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

// --- clock ------------------------------------------------------------------

static CLOCK: OnceLock<(Instant, u64)> = OnceLock::new();

/// Monotonic nanoseconds anchored to wall-clock at process start, so values
/// are comparable across processes on the same host.
pub fn mono_ns() -> u64 {
    let (i, base) = CLOCK.get_or_init(|| {
        (
            Instant::now(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        )
    });
    base + i.elapsed().as_nanos() as u64
}

/// Precise wait: coarse sleep then yield-spin for the final stretch.
pub fn wait_until_ns(t: u64) {
    loop {
        let now = mono_ns();
        if now >= t {
            return;
        }
        let rem = t - now;
        if rem > 400_000 {
            thread::sleep(Duration::from_nanos(rem - 300_000));
        } else if rem > 30_000 {
            thread::yield_now();
        } else {
            std::hint::spin_loop();
        }
    }
}

/// Poll a channel for up to `SPIN_RECV_NS` before parking the thread. A
/// parked thread costs tens of µs to wake on a busy host — several times the
/// Thunderbolt one-way latency being modelled — and a round trip crosses six
/// such hand-offs (writer → socket reader → consumer, both directions).
pub const SPIN_RECV_NS: u64 = 50_000;

pub fn spin_recv<T>(rx: &Receiver<T>) -> Result<T, std::sync::mpsc::RecvError> {
    let t0 = mono_ns();
    loop {
        match rx.try_recv() {
            Ok(v) => return Ok(v),
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                return Err(std::sync::mpsc::RecvError)
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                if mono_ns() - t0 > SPIN_RECV_NS {
                    return rx.recv();
                }
                std::hint::spin_loop();
            }
        }
    }
}

// --- link shaper ---------------------------------------------------------------

struct ShaperState {
    profile: LinkProfile,
    free_at_ns: u64,
    jitter_seed: u64,
}

pub struct LinkShaper {
    state: Mutex<ShaperState>,
    tx: Mutex<Sender<(u64, Vec<u8>)>>,
}

impl LinkShaper {
    fn spawn(stream: TcpStream, profile: LinkProfile) -> Arc<Self> {
        let (tx, rx) = channel::<(u64, Vec<u8>)>();
        let mut w = BufWriter::with_capacity(1 << 16, stream);
        thread::Builder::new()
            .name("tess-link-writer".into())
            .spawn(move || {
                while let Ok((deliver_at, bytes)) = spin_recv(&rx) {
                    wait_until_ns(deliver_at);
                    if w.write_all(&bytes).is_err() {
                        break;
                    }
                    // Flush when no further frame is already due.
                    if w.flush().is_err() {
                        break;
                    }
                }
            })
            .expect("spawn writer");
        Arc::new(LinkShaper {
            state: Mutex::new(ShaperState {
                profile,
                free_at_ns: 0,
                jitter_seed: 0x1234_5678,
            }),
            tx: Mutex::new(tx),
        })
    }

    /// Enqueue an encoded frame; returns its modelled delivery time.
    fn send(&self, bytes: Vec<u8>) -> u64 {
        let now = mono_ns();
        let deliver = {
            let mut s = self.state.lock().unwrap();
            if s.profile.is_unshaped() {
                now
            } else {
                let ser = s.profile.serialization_ns(bytes.len()) as u64;
                let start = s.free_at_ns.max(now);
                s.free_at_ns = start + ser;
                s.jitter_seed = s
                    .jitter_seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let j = ((s.jitter_seed >> 33) as f64 / (1u64 << 31) as f64)
                    * s.profile.jitter_us
                    * 1000.0;
                s.free_at_ns + (s.profile.one_way_us * 1000.0) as u64 + j as u64
            }
        };
        let _ = self.tx.lock().unwrap().send((deliver, bytes));
        deliver
    }

    pub fn set_profile(&self, p: LinkProfile) {
        self.state.lock().unwrap().profile = p;
    }

    pub fn profile(&self) -> LinkProfile {
        self.state.lock().unwrap().profile
    }
}

/// All logical links to one peer plus the multipath scheduler.
pub struct PeerLinks {
    pub links: Vec<Arc<LinkShaper>>,
    pub scheduler: Mutex<MultipathScheduler>,
}

// --- endpoint ----------------------------------------------------------------

/// Counters cover inter-host traffic only (frames to peers marked local —
/// services on the same host, the control-plane driver — are not counted).
pub struct TrafficStats {
    pub bytes_sent: AtomicU64,
    pub frames_sent: AtomicU64,
    pub payload_bytes_sent: AtomicU64,
    pub by_type: [AtomicU64; 64],
}

impl Default for TrafficStats {
    fn default() -> Self {
        TrafficStats {
            bytes_sent: AtomicU64::new(0),
            frames_sent: AtomicU64::new(0),
            payload_bytes_sent: AtomicU64::new(0),
            by_type: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }
}

impl TrafficStats {
    fn add(&self, ft: FrameType, wire: usize, payload: usize) {
        self.bytes_sent.fetch_add(wire as u64, Ordering::Relaxed);
        self.frames_sent.fetch_add(1, Ordering::Relaxed);
        self.payload_bytes_sent
            .fetch_add(payload as u64, Ordering::Relaxed);
        self.by_type[(ft as usize) & 63].fetch_add(wire as u64, Ordering::Relaxed);
    }
    pub fn bytes_of(&self, ft: FrameType) -> u64 {
        self.by_type[(ft as usize) & 63].load(Ordering::Relaxed)
    }
}

/// Partially received striped frame: (stripes received, stripe payloads, first header).
type Reassembly = (u32, Vec<Option<Vec<u8>>>, TesseractFrame);

/// Thread-safe sending half of a node's endpoint.
pub struct EndpointShared {
    pub id: u16,
    pub listen_addr: SocketAddr,
    addr_book: Mutex<HashMap<u16, SocketAddr>>,
    link_profiles: Mutex<HashMap<u16, Vec<LinkProfile>>>,
    peers: Mutex<HashMap<u16, Arc<PeerLinks>>>,
    seq: AtomicU64,
    pub stats: TrafficStats,
    /// Frames larger than this are striped across multiple links.
    pub stripe_threshold: AtomicU64,
    reassembly: Mutex<HashMap<(u16, u64), Reassembly>>,
    closed: AtomicBool,
    local_peers: Mutex<std::collections::HashSet<u16>>,
}

impl EndpointShared {
    /// Mark a peer as same-host / control-plane: its traffic is not counted.
    pub fn set_local(&self, peer: u16) {
        self.local_peers.lock().unwrap().insert(peer);
    }

    pub fn set_addr(&self, node: u16, addr: SocketAddr) {
        self.addr_book.lock().unwrap().insert(node, addr);
    }

    /// Configure the physical links to `peer` (one entry per logical link).
    pub fn set_links(&self, peer: u16, profiles: Vec<LinkProfile>) {
        self.link_profiles.lock().unwrap().insert(peer, profiles);
    }

    pub fn links_to(&self, peer: u16) -> Vec<LinkProfile> {
        self.link_profiles
            .lock()
            .unwrap()
            .get(&peer)
            .cloned()
            .unwrap_or_else(|| vec![LinkProfile::unshaped()])
    }

    fn peer(&self, to: u16) -> std::io::Result<Arc<PeerLinks>> {
        if let Some(p) = self.peers.lock().unwrap().get(&to) {
            return Ok(p.clone());
        }
        let addr = *self.addr_book.lock().unwrap().get(&to).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("node {} has no address for peer {to}", self.id),
            )
        })?;
        let profiles = self.links_to(to);
        let mut links = Vec::new();
        for (li, p) in profiles.iter().enumerate() {
            let mut s = None;
            for attempt in 0..200 {
                match TcpStream::connect(addr) {
                    Ok(c) => {
                        s = Some(c);
                        break;
                    }
                    Err(e) if attempt == 199 => return Err(e),
                    Err(_) => thread::sleep(Duration::from_millis(10)),
                }
            }
            let mut s = s.unwrap();
            s.set_nodelay(true)?;
            // prove the cluster secret before sending any frame
            super::auth::client_handshake(&mut s, &super::auth::cluster_key())?;
            let mut hello = TesseractFrame::new(FrameType::Hello, self.id, to, vec![li as u8]);
            hello.header.sequence_id = li as u64;
            s.write_all(&hello.encode())?;
            links.push(LinkShaper::spawn(s, *p));
        }
        let pl = Arc::new(PeerLinks {
            scheduler: Mutex::new(MultipathScheduler::new(&profiles)),
            links,
        });
        let mut peers = self.peers.lock().unwrap();
        Ok(peers.entry(to).or_insert(pl).clone())
    }

    /// Change a live link's physical profile (brownout injection / recovery).
    pub fn reshape_link(&self, peer: u16, link: usize, p: LinkProfile) -> std::io::Result<()> {
        let pl = self.peer(peer)?;
        pl.links[link].set_profile(p);
        Ok(())
    }

    pub fn next_seq(&self) -> u64 {
        self.seq.fetch_add(1, Ordering::Relaxed)
    }

    /// Send a frame (header fields other than seq/len/crc/ts are preserved).
    pub fn send(&self, to: u16, mut frame: TesseractFrame) -> std::io::Result<()> {
        if frame.header.sequence_id == 0 {
            frame.header.sequence_id = self.next_seq();
        }
        frame.header.source = self.id;
        frame.header.target = to;
        let peer = self.peer(to)?;
        let total = frame.wire_len();
        let now = mono_ns() as f64;
        let threshold = self.stripe_threshold.load(Ordering::Relaxed) as usize;
        let plan = if peer.links.len() > 1 && total > threshold {
            peer.scheduler
                .lock()
                .unwrap()
                .plan(now, frame.payload.len())
        } else {
            vec![(
                peer.scheduler.lock().unwrap().best_link(now, total),
                frame.payload.len(),
            )]
        };
        let counted = !self.local_peers.lock().unwrap().contains(&to);
        let ft = frame.header.frame_type;
        if peer.links.len() > 1 {
            // latency probes on every link of a multi-link peer (acked by the receiver)
            let due = peer.scheduler.lock().unwrap().probes_due(now, 2_000_000.0);
            for link in due {
                let mut hb =
                    TesseractFrame::new(FrameType::HeartbeatTelemetry, self.id, to, vec![]);
                hb.header.sequence_id = self.next_seq();
                hb.header.send_ts_ns = mono_ns();
                peer.links[link].send(hb.encode());
            }
        }
        if plan.len() == 1 {
            let link = plan[0].0;
            frame.header.send_ts_ns = mono_ns();
            let enc = frame.encode();
            if counted {
                self.stats.add(ft, enc.len(), frame.payload.len());
            }
            peer.scheduler.lock().unwrap().on_send(link, now, enc.len());
            peer.links[link].send(enc);
            return Ok(());
        }
        let count = plan.len() as u32;
        let mut off = 0;
        for (si, &(link, n)) in plan.iter().enumerate() {
            let mut sub = TesseractFrame {
                header: frame.header,
                payload: frame.payload[off..off + n].to_vec(),
            };
            off += n;
            sub.header.flags |= flags::STRIPED;
            sub.header.stripe_seq = si as u32;
            sub.header.stripe_count = count;
            sub.header.send_ts_ns = mono_ns();
            let enc = sub.encode();
            if counted {
                self.stats.add(ft, enc.len(), n);
            }
            peer.scheduler.lock().unwrap().on_send(link, now, enc.len());
            peer.links[link].send(enc);
        }
        Ok(())
    }

    pub fn send_payload(
        &self,
        to: u16,
        ft: FrameType,
        epoch: u64,
        layer: u16,
        payload: Vec<u8>,
    ) -> std::io::Result<()> {
        let mut f = TesseractFrame::new(ft, self.id, to, payload);
        f.header.tree_epoch = epoch;
        f.header.layer = layer;
        self.send(to, f)
    }

    pub fn scheduler_snapshot(&self, peer: u16) -> Option<MultipathScheduler> {
        self.peers
            .lock()
            .unwrap()
            .get(&peer)
            .map(|p| p.scheduler.lock().unwrap().clone())
    }

    pub fn bytes_sent(&self) -> u64 {
        self.stats.bytes_sent.load(Ordering::Relaxed)
    }

    /// Stop accepting, drop all outbound links (their writer threads exit and
    /// sockets close, which ends the remote reader threads).
    pub fn close(&self) {
        self.closed.store(true, Ordering::Relaxed);
        self.peers.lock().unwrap().clear();
        let _ = TcpStream::connect(self.listen_addr); // unblock accept()
    }
}

/// A node's network endpoint: listener + inbox. Owned by the node's thread.
pub struct Endpoint {
    pub shared: Arc<EndpointShared>,
    rx: Receiver<TesseractFrame>,
    deferred: RefCell<VecDeque<TesseractFrame>>,
}

impl Endpoint {
    /// Bind a listener on `addr` (use port 0 for ephemeral).
    pub fn bind(id: u16, addr: &str) -> std::io::Result<Self> {
        let listener = TcpListener::bind(addr)?;
        let listen_addr = listener.local_addr()?;
        let (tx, rx) = channel();
        let shared = Arc::new(EndpointShared {
            id,
            listen_addr,
            addr_book: Mutex::new(HashMap::new()),
            link_profiles: Mutex::new(HashMap::new()),
            peers: Mutex::new(HashMap::new()),
            seq: AtomicU64::new(1),
            stats: TrafficStats::default(),
            stripe_threshold: AtomicU64::new(64 * 1024),
            reassembly: Mutex::new(HashMap::new()),
            closed: AtomicBool::new(false),
            local_peers: Mutex::new(std::collections::HashSet::new()),
        });
        let sh = shared.clone();
        let key = super::auth::cluster_key();
        thread::Builder::new()
            .name(format!("tess-accept-{id}"))
            .spawn(move || {
                for conn in listener.incoming() {
                    if sh.closed.load(Ordering::Relaxed) {
                        break;
                    }
                    let Ok(conn) = conn else { continue };
                    let _ = conn.set_nodelay(true);
                    let tx = tx.clone();
                    let sh2 = sh.clone();
                    thread::Builder::new()
                        .name(format!("tess-reader-{id}"))
                        .spawn(move || {
                            // no frame is read from a peer that cannot prove
                            // the cluster secret
                            let mut conn = conn;
                            if super::auth::server_handshake(&mut conn, &key).is_ok() {
                                reader_loop(conn, tx, sh2);
                            }
                        })
                        .ok();
                }
            })?;
        Ok(Endpoint {
            shared,
            rx,
            deferred: RefCell::new(VecDeque::new()),
        })
    }

    pub fn id(&self) -> u16 {
        self.shared.id
    }
    pub fn addr(&self) -> SocketAddr {
        self.shared.listen_addr
    }

    pub fn send(&self, to: u16, f: TesseractFrame) -> std::io::Result<()> {
        self.shared.send(to, f)
    }

    pub fn send_payload(
        &self,
        to: u16,
        ft: FrameType,
        epoch: u64,
        layer: u16,
        payload: Vec<u8>,
    ) -> std::io::Result<()> {
        self.shared.send_payload(to, ft, epoch, layer, payload)
    }

    /// Next frame (deferred frames first).
    pub fn recv(&self) -> TesseractFrame {
        if let Some(f) = self.deferred.borrow_mut().pop_front() {
            return f;
        }
        spin_recv(&self.rx).expect("endpoint inbox closed")
    }

    pub fn recv_timeout(&self, d: Duration) -> Option<TesseractFrame> {
        if let Some(f) = self.deferred.borrow_mut().pop_front() {
            return Some(f);
        }
        match self.rx.recv_timeout(d) {
            Ok(f) => Some(f),
            Err(RecvTimeoutError::Timeout) => None,
            Err(RecvTimeoutError::Disconnected) => None,
        }
    }

    /// Receive the first frame matching `pred`; others are deferred in order.
    pub fn recv_match(&self, pred: &dyn Fn(&TesseractFrame) -> bool) -> TesseractFrame {
        {
            let mut d = self.deferred.borrow_mut();
            if let Some(pos) = d.iter().position(pred) {
                return d.remove(pos).unwrap();
            }
        }
        loop {
            let f = spin_recv(&self.rx).expect("endpoint inbox closed");
            if pred(&f) {
                return f;
            }
            self.deferred.borrow_mut().push_back(f);
        }
    }

    pub fn recv_match_timeout(
        &self,
        pred: &dyn Fn(&TesseractFrame) -> bool,
        timeout: Duration,
    ) -> Option<TesseractFrame> {
        {
            let mut d = self.deferred.borrow_mut();
            if let Some(pos) = d.iter().position(pred) {
                return d.remove(pos);
            }
        }
        let deadline = Instant::now() + timeout;
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return None;
            }
            match self.rx.recv_timeout(left) {
                Ok(f) if pred(&f) => return Some(f),
                Ok(f) => self.deferred.borrow_mut().push_back(f),
                Err(_) => return None,
            }
        }
    }

    /// Non-blocking poll for a matching frame.
    pub fn try_recv_match(&self, pred: &dyn Fn(&TesseractFrame) -> bool) -> Option<TesseractFrame> {
        while let Ok(f) = self.rx.try_recv() {
            self.deferred.borrow_mut().push_back(f);
        }
        let mut d = self.deferred.borrow_mut();
        let pos = d.iter().position(pred)?;
        d.remove(pos)
    }
}

fn reader_loop(conn: TcpStream, tx: Sender<TesseractFrame>, sh: Arc<EndpointShared>) {
    let mut r = BufReader::with_capacity(1 << 16, conn);
    let mut link_idx = 0usize;
    loop {
        let f = match TesseractFrame::read_from(&mut r) {
            Ok(f) => f,
            Err(_) => return,
        };
        let recv_ns = mono_ns();
        match f.header.frame_type {
            FrameType::Hello => {
                link_idx = f.payload.first().copied().unwrap_or(0) as usize;
                continue;
            }
            FrameType::StripeAck => {
                // payload: link u16, bytes u32, sent u64, recv u64
                if let Some(p) = sh.peers.lock().unwrap().get(&f.header.source).cloned() {
                    let b = &f.payload;
                    let link = u16::from_le_bytes([b[0], b[1]]) as usize;
                    let bytes = u32::from_le_bytes(b[2..6].try_into().unwrap()) as usize;
                    let sent = u64::from_le_bytes(b[6..14].try_into().unwrap()) as f64;
                    let recv = u64::from_le_bytes(b[14..22].try_into().unwrap()) as f64;
                    let mut s = p.scheduler.lock().unwrap();
                    if link < s.n_links() {
                        s.on_ack(link, bytes, sent, recv);
                    }
                }
                continue;
            }
            _ => {}
        }
        let is_probe = f.header.frame_type == FrameType::HeartbeatTelemetry && f.payload.is_empty();
        let striped = f.header.flags & flags::STRIPED != 0;
        // bandwidth sample for large frames on multi-link peers (links are symmetric)
        let bw_sample = f.payload.len() >= 4096
            && sh
                .link_profiles
                .lock()
                .unwrap()
                .get(&f.header.source)
                .map(|v| v.len() > 1)
                .unwrap_or(false);
        if striped || is_probe || bw_sample {
            // Acknowledge for link telemetry (tiny frame on the reverse path).
            let mut ack = Vec::with_capacity(22);
            ack.extend_from_slice(&(link_idx as u16).to_le_bytes());
            ack.extend_from_slice(&((f.payload.len() + 64) as u32).to_le_bytes());
            ack.extend_from_slice(&f.header.send_ts_ns.to_le_bytes());
            ack.extend_from_slice(&recv_ns.to_le_bytes());
            let _ = sh.send(
                f.header.source,
                TesseractFrame::new(FrameType::StripeAck, sh.id, f.header.source, ack),
            );
            if is_probe {
                continue;
            }
        }
        if striped {
            let key = (f.header.source, f.header.sequence_id);
            let mut ra = sh.reassembly.lock().unwrap();
            let count = f.header.stripe_count as usize;
            let entry = ra
                .entry(key)
                .or_insert_with(|| (0, vec![None; count], f.clone()));
            let idx = f.header.stripe_seq as usize;
            if entry.1[idx].is_none() {
                entry.1[idx] = Some(f.payload.clone());
                entry.0 += 1;
            }
            if entry.0 as usize == count {
                let (_, parts, first) = ra.remove(&key).unwrap();
                let mut payload =
                    Vec::with_capacity(parts.iter().map(|p| p.as_ref().unwrap().len()).sum());
                for p in parts {
                    payload.extend_from_slice(&p.unwrap());
                }
                let mut whole = first;
                whole.header.flags &= !flags::STRIPED;
                whole.header.stripe_seq = 0;
                whole.header.stripe_count = 1;
                whole.header.payload_len = payload.len() as u32;
                whole.payload = payload;
                if tx.send(whole).is_err() {
                    return;
                }
            }
            continue;
        }
        if tx.send(f).is_err() {
            return;
        }
    }
}
