//! Zero-configuration cluster formation over LAN multicast / broadcast and
//! unicast seeds (e.g. Tailscale `100.x.y.z` peers).
//!
//! Every `tesseract-node` daemon periodically sends a 64-byte
//! [`DiscoveryBeacon`] over UDP (default port 49777): to the multicast group
//! `239.255.77.77`, to the LAN broadcast address, and to any unicast seeds.
//! Beacons carry the node's endpoint ports and a self-measured hardware
//! profile, and are authenticated with SipHash-2-4 under a key derived from
//! the cluster secret (nodes of other clusters, or forged/replayed beacons,
//! are ignored).
//!
//! The coordinator's [`DiscoveryListener`] collects beacons, then [`probe`]s
//! each new worker over its real `TesseractWire` endpoint — three echo pings
//! for RTT and a 64 KB burst for bandwidth — and feeds the measured profiles
//! to `HeteroClusterSolver`, whose plan becomes the cluster `Assignment`.

use super::{
    Assignment, Driver, EngineKind, NodeSetup, TesseractOptions, DRAFT_SERVICE_BASE, DRIVER_ID,
    KV_SERVICE_BASE,
};
use crate::model::ModelConfig;
use crate::topology::{
    footprint, ClusterPlan, HardwareNodeSpec, HeteroClusterSolver, LinkSpec, Platform, SolverConfig,
};
use crate::wire::multipath::{LinkKind, LinkProfile};
use crate::wire::protocol::FrameType;
use crate::wire::transport::{mono_ns, Endpoint};
use std::collections::BTreeMap;
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, UdpSocket};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub const DISCOVERY_PORT: u16 = 49777;
pub const MULTICAST_GROUP: Ipv4Addr = Ipv4Addr::new(239, 255, 77, 77);
pub const BEACON_LEN: usize = 64;
const MAGIC: &[u8; 4] = b"TDSC";
/// Beacons older than this (sender clock) are rejected as replays.
const MAX_AGE_MS: u64 = 30_000;

// --- SipHash-2-4 ------------------------------------------------------------------

fn sip_round(v: &mut [u64; 4]) {
    v[0] = v[0].wrapping_add(v[1]);
    v[1] = v[1].rotate_left(13) ^ v[0];
    v[0] = v[0].rotate_left(32);
    v[2] = v[2].wrapping_add(v[3]);
    v[3] = v[3].rotate_left(16) ^ v[2];
    v[0] = v[0].wrapping_add(v[3]);
    v[3] = v[3].rotate_left(21) ^ v[0];
    v[2] = v[2].wrapping_add(v[1]);
    v[1] = v[1].rotate_left(17) ^ v[2];
    v[2] = v[2].rotate_left(32);
}

/// SipHash-2-4 (64-bit tag) of `data` under key `(k0, k1)`.
pub fn siphash24(k0: u64, k1: u64, data: &[u8]) -> u64 {
    let mut v = [
        k0 ^ 0x736f_6d65_7073_6575,
        k1 ^ 0x646f_7261_6e64_6f6d,
        k0 ^ 0x6c79_6765_6e65_7261,
        k1 ^ 0x7465_6462_7974_6573,
    ];
    let (chunks, rem) = data.as_chunks::<8>();
    for c in chunks {
        let m = u64::from_le_bytes(*c);
        v[3] ^= m;
        sip_round(&mut v);
        sip_round(&mut v);
        v[0] ^= m;
    }
    let mut last = [0u8; 8];
    last[..rem.len()].copy_from_slice(rem);
    last[7] = data.len() as u8;
    let m = u64::from_le_bytes(last);
    v[3] ^= m;
    sip_round(&mut v);
    sip_round(&mut v);
    v[0] ^= m;
    v[2] ^= 0xff;
    for _ in 0..4 {
        sip_round(&mut v);
    }
    v[0] ^ v[1] ^ v[2] ^ v[3]
}

/// Keys and id derived from a cluster secret.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClusterKey {
    pub cluster_id: u64,
    k0: u64,
    k1: u64,
}

impl ClusterKey {
    pub fn from_secret(secret: &str) -> Self {
        let b = secret.as_bytes();
        ClusterKey {
            cluster_id: siphash24(0x7465_7373, 0x6964, b),
            k0: siphash24(0x7465_7373, 0x6b30, b),
            k1: siphash24(0x7465_7373, 0x6b31, b),
        }
    }
}

// --- beacon ----------------------------------------------------------------------

pub fn platform_code(p: Platform) -> u8 {
    match p {
        Platform::AppleSiliconUMA => 0,
        Platform::DiscreteGpuWorkstation => 1,
        Platform::MobileEdgeNpu => 2,
        Platform::CpuServer => 3,
    }
}

pub fn platform_from_code(c: u8) -> Platform {
    match c {
        0 => Platform::AppleSiliconUMA,
        1 => Platform::DiscreteGpuWorkstation,
        2 => Platform::MobileEdgeNpu,
        _ => Platform::CpuServer,
    }
}

/// 64-byte signed announcement of a worker node.
///
/// | bytes | field |
/// |---|---|
/// | 0..4 | magic `TDSC` |
/// | 4 | version (1) |
/// | 5 | platform kind |
/// | 6..8 | node endpoint TCP port |
/// | 8..16 | cluster id |
/// | 16..18 | node id |
/// | 18..20 | KV service port |
/// | 20..24 | compute TFLOPS (f32) |
/// | 24..28 | memory bandwidth GB/s (f32) |
/// | 28..32 | free RAM MB (u32) |
/// | 32..36 | active load (1-min loadavg, f32) |
/// | 36..38 | draft service port |
/// | 38..40 | reserved |
/// | 40..48 | sender time, ms since the Unix epoch (replay window) |
/// | 48..56 | sequence number |
/// | 56..64 | SipHash-2-4 tag over bytes 0..56 |
#[derive(Clone, Debug, PartialEq)]
pub struct DiscoveryBeacon {
    pub cluster_id: u64,
    pub node_id: u16,
    pub tcp_port: u16,
    pub kv_port: u16,
    pub draft_port: u16,
    pub platform: Platform,
    pub compute_tflops_fp16: f32,
    pub mem_bandwidth_gbps: f32,
    pub free_ram_mb: u32,
    pub active_load: f32,
    pub time_ms: u64,
    pub seq: u64,
}

fn unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

impl DiscoveryBeacon {
    pub fn encode(&self, key: &ClusterKey) -> [u8; BEACON_LEN] {
        let mut b = [0u8; BEACON_LEN];
        b[0..4].copy_from_slice(MAGIC);
        b[4] = 1;
        b[5] = platform_code(self.platform);
        b[6..8].copy_from_slice(&self.tcp_port.to_le_bytes());
        b[8..16].copy_from_slice(&self.cluster_id.to_le_bytes());
        b[16..18].copy_from_slice(&self.node_id.to_le_bytes());
        b[18..20].copy_from_slice(&self.kv_port.to_le_bytes());
        b[20..24].copy_from_slice(&self.compute_tflops_fp16.to_le_bytes());
        b[24..28].copy_from_slice(&self.mem_bandwidth_gbps.to_le_bytes());
        b[28..32].copy_from_slice(&self.free_ram_mb.to_le_bytes());
        b[32..36].copy_from_slice(&self.active_load.to_le_bytes());
        b[36..38].copy_from_slice(&self.draft_port.to_le_bytes());
        b[40..48].copy_from_slice(&self.time_ms.to_le_bytes());
        b[48..56].copy_from_slice(&self.seq.to_le_bytes());
        let tag = siphash24(key.k0, key.k1, &b[..56]);
        b[56..64].copy_from_slice(&tag.to_le_bytes());
        b
    }

    /// Verify magic, version, cluster and tag; `None` for anything else.
    pub fn decode(b: &[u8], key: &ClusterKey) -> Option<Self> {
        if b.len() != BEACON_LEN || &b[0..4] != MAGIC || b[4] != 1 {
            return None;
        }
        let tag = u64::from_le_bytes(b[56..64].try_into().unwrap());
        if siphash24(key.k0, key.k1, &b[..56]) != tag {
            return None;
        }
        let u16_at = |o: usize| u16::from_le_bytes([b[o], b[o + 1]]);
        let f32_at = |o: usize| f32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let bc = DiscoveryBeacon {
            cluster_id: u64::from_le_bytes(b[8..16].try_into().unwrap()),
            node_id: u16_at(16),
            tcp_port: u16_at(6),
            kv_port: u16_at(18),
            draft_port: u16_at(36),
            platform: platform_from_code(b[5]),
            compute_tflops_fp16: f32_at(20),
            mem_bandwidth_gbps: f32_at(24),
            free_ram_mb: u32::from_le_bytes(b[28..32].try_into().unwrap()),
            active_load: f32_at(32),
            time_ms: u64::from_le_bytes(b[40..48].try_into().unwrap()),
            seq: u64::from_le_bytes(b[48..56].try_into().unwrap()),
        };
        (bc.cluster_id == key.cluster_id).then_some(bc)
    }
}

// --- local hardware profile --------------------------------------------------------

/// The hardware a node announces: measured on this host (or overridden).
#[derive(Clone, Debug, PartialEq)]
pub struct LocalProfile {
    pub platform: Platform,
    pub compute_tflops_fp16: f32,
    pub mem_bandwidth_gbps: f32,
    pub free_ram_mb: u32,
}

#[cfg(unix)]
extern "C" {
    fn getloadavg(loadavg: *mut f64, nelem: i32) -> i32;
}
#[cfg(target_os = "macos")]
extern "C" {
    fn sysctlbyname(
        name: *const i8,
        oldp: *mut std::ffi::c_void,
        oldlenp: *mut usize,
        newp: *mut std::ffi::c_void,
        newlen: usize,
    ) -> i32;
}

/// 1-minute load average (0 where unavailable).
pub fn load_average() -> f32 {
    #[cfg(unix)]
    {
        let mut l = [0.0f64; 3];
        // SAFETY: writes at most 3 doubles into `l`.
        if unsafe { getloadavg(l.as_mut_ptr(), 3) } >= 1 {
            return l[0] as f32;
        }
    }
    0.0
}

/// Physical memory available to the engine, in MB.
pub fn available_ram_mb() -> u32 {
    #[cfg(target_os = "macos")]
    {
        let mut v = 0u64;
        let mut len = std::mem::size_of::<u64>();
        let name = b"hw.memsize\0";
        // SAFETY: NUL-terminated name, `v` is 8 writable bytes.
        let r = unsafe {
            sysctlbyname(
                name.as_ptr() as *const i8,
                &mut v as *mut u64 as *mut _,
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        if r == 0 {
            return (v >> 20) as u32;
        }
    }
    #[cfg(target_os = "linux")]
    if let Ok(s) = std::fs::read_to_string("/proc/meminfo") {
        if let Some(kb) = s
            .lines()
            .find(|l| l.starts_with("MemAvailable:"))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<u64>().ok())
        {
            return (kb >> 10) as u32;
        }
    }
    0
}

impl LocalProfile {
    /// Measure this host: single-core GEMM throughput, single-stream copy
    /// bandwidth and installed RAM (≈ 50 ms).
    pub fn detect() -> Self {
        let platform = if cfg!(target_os = "android") {
            Platform::MobileEdgeNpu
        } else if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            Platform::AppleSiliconUMA
        } else {
            Platform::CpuServer
        };
        // memory bandwidth: best of 3 copies of 64 MB
        let n = 16 << 20;
        let src = vec![1.0f32; n];
        let mut dst = vec![0.0f32; n];
        let mut best = f64::MAX;
        for _ in 0..3 {
            let t = Instant::now();
            dst.copy_from_slice(&src);
            std::hint::black_box(&dst);
            best = best.min(t.elapsed().as_secs_f64());
        }
        let bw = (2.0 * 4.0 * n as f64) / best / 1e9;
        // compute: best of 3 runs of a 256×512×512 GEMM on one core
        let mut rng = crate::tensor::Rng::new(1);
        let x = crate::tensor::Tensor2D::randn(256, 512, 1.0, &mut rng);
        let w = crate::tensor::Tensor2D::randn(512, 512, 1.0, &mut rng);
        let mut y = crate::tensor::Tensor2D::zeros(256, 512);
        let mut tb = f64::MAX;
        for _ in 0..3 {
            let t = Instant::now();
            crate::tensor::matmul_tiled_batch(&x, &w, &mut y);
            std::hint::black_box(&y);
            tb = tb.min(t.elapsed().as_secs_f64());
        }
        let tflops = 2.0 * 256.0 * 512.0 * 512.0 / tb / 1e12;
        LocalProfile {
            platform,
            compute_tflops_fp16: tflops as f32,
            mem_bandwidth_gbps: bw as f32,
            free_ram_mb: available_ram_mb(),
        }
    }
}

// --- announcer ---------------------------------------------------------------------

/// Where a node sends its beacons.
#[derive(Clone, Debug)]
pub struct AnnounceTargets {
    pub multicast: bool,
    pub broadcast: bool,
    pub port: u16,
    /// Unicast peers (`host:port`), e.g. a coordinator's Tailscale address.
    pub seeds: Vec<SocketAddr>,
}

/// Safe default: unicast seeds only. LAN multicast / broadcast must be
/// requested explicitly (`--lan-broadcast`).
impl Default for AnnounceTargets {
    fn default() -> Self {
        AnnounceTargets {
            multicast: false,
            broadcast: false,
            port: DISCOVERY_PORT,
            seeds: Vec::new(),
        }
    }
}

/// Background beacon sender; stops when dropped.
pub struct Announcer {
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Announcer {
    #[allow(clippy::too_many_arguments)]
    pub fn start(
        key: ClusterKey,
        node_id: u16,
        ports: (u16, u16, u16),
        profile: LocalProfile,
        targets: AnnounceTargets,
        interval: Duration,
    ) -> std::io::Result<Self> {
        let sock = UdpSocket::bind("0.0.0.0:0")?;
        let _ = sock.set_broadcast(true);
        let _ = sock.set_multicast_ttl_v4(1);
        let _ = sock.set_multicast_loop_v4(true);
        let stop = Arc::new(AtomicBool::new(false));
        let st = stop.clone();
        let handle = std::thread::Builder::new()
            .name(format!("tess-beacon-{node_id}"))
            .spawn(move || {
                let mut seq = 0u64;
                while !st.load(Ordering::Relaxed) {
                    seq += 1;
                    let b = DiscoveryBeacon {
                        cluster_id: key.cluster_id,
                        node_id,
                        tcp_port: ports.0,
                        kv_port: ports.1,
                        draft_port: ports.2,
                        platform: profile.platform,
                        compute_tflops_fp16: profile.compute_tflops_fp16,
                        mem_bandwidth_gbps: profile.mem_bandwidth_gbps,
                        free_ram_mb: profile.free_ram_mb,
                        active_load: load_average(),
                        time_ms: unix_ms(),
                        seq,
                    }
                    .encode(&key);
                    // delivery is best-effort on every path
                    if targets.multicast {
                        let _ = sock.send_to(&b, SocketAddrV4::new(MULTICAST_GROUP, targets.port));
                    }
                    if targets.broadcast {
                        let _ =
                            sock.send_to(&b, SocketAddrV4::new(Ipv4Addr::BROADCAST, targets.port));
                    }
                    for s in &targets.seeds {
                        let _ = sock.send_to(&b, s);
                    }
                    let t = Instant::now();
                    while t.elapsed() < interval && !st.load(Ordering::Relaxed) {
                        std::thread::sleep(Duration::from_millis(10).min(interval));
                    }
                }
            })?;
        Ok(Announcer {
            stop,
            handle: Some(handle),
        })
    }
}

impl Drop for Announcer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

// --- listener ----------------------------------------------------------------------

/// A discovered worker.
#[derive(Clone, Debug)]
pub struct Peer {
    pub beacon: DiscoveryBeacon,
    pub ip: std::net::IpAddr,
    pub last_seen: Instant,
}

impl Peer {
    pub fn node_addr(&self) -> SocketAddr {
        SocketAddr::new(self.ip, self.beacon.tcp_port)
    }
    pub fn kv_addr(&self) -> SocketAddr {
        SocketAddr::new(self.ip, self.beacon.kv_port)
    }
    pub fn draft_addr(&self) -> SocketAddr {
        SocketAddr::new(self.ip, self.beacon.draft_port)
    }
}

/// Collects authenticated beacons on a UDP port.
pub struct DiscoveryListener {
    pub addr: SocketAddr,
    peers: Arc<Mutex<BTreeMap<u16, Peer>>>,
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
    pub rejected: Arc<std::sync::atomic::AtomicU64>,
}

impl DiscoveryListener {
    /// Listen on `bind` (`0.0.0.0:49777` on a LAN; port 0 for an ephemeral
    /// port); `multicast` joins `239.255.77.77`.
    pub fn start(bind: &str, key: ClusterKey, multicast: bool) -> std::io::Result<Self> {
        let sock = UdpSocket::bind(bind)?;
        if multicast {
            let _ = sock.join_multicast_v4(&MULTICAST_GROUP, &Ipv4Addr::UNSPECIFIED);
        }
        sock.set_read_timeout(Some(Duration::from_millis(50)))?;
        let addr = sock.local_addr()?;
        let peers: Arc<Mutex<BTreeMap<u16, Peer>>> = Default::default();
        let stop = Arc::new(AtomicBool::new(false));
        let rejected = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let (p, st, rj) = (peers.clone(), stop.clone(), rejected.clone());
        let handle = std::thread::Builder::new()
            .name("tess-discovery".into())
            .spawn(move || {
                let mut buf = [0u8; 512];
                while !st.load(Ordering::Relaxed) {
                    let Ok((n, from)) = sock.recv_from(&mut buf) else {
                        continue;
                    };
                    let Some(b) = DiscoveryBeacon::decode(&buf[..n], &key) else {
                        rj.fetch_add(1, Ordering::Relaxed);
                        continue;
                    };
                    let now = unix_ms();
                    if b.time_ms + MAX_AGE_MS < now || b.time_ms > now + MAX_AGE_MS {
                        rj.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                    let mut g = p.lock().unwrap();
                    match g.get_mut(&b.node_id) {
                        Some(old) if old.beacon.seq >= b.seq && old.beacon.time_ms >= b.time_ms => {
                            rj.fetch_add(1, Ordering::Relaxed); // replay / duplicate
                        }
                        Some(old) => {
                            old.beacon = b;
                            old.ip = from.ip();
                            old.last_seen = Instant::now();
                        }
                        None => {
                            g.insert(
                                b.node_id,
                                Peer {
                                    beacon: b,
                                    ip: from.ip(),
                                    last_seen: Instant::now(),
                                },
                            );
                        }
                    }
                }
            })?;
        Ok(DiscoveryListener {
            addr,
            peers,
            stop,
            handle: Some(handle),
            rejected,
        })
    }

    /// Peers heard from within `fresh`.
    pub fn peers(&self, fresh: Duration) -> Vec<Peer> {
        self.peers
            .lock()
            .unwrap()
            .values()
            .filter(|p| p.last_seen.elapsed() <= fresh)
            .cloned()
            .collect()
    }

    /// Wait until at least `n` peers are known (or `timeout`).
    pub fn wait_for(&self, n: usize, timeout: Duration) -> Vec<Peer> {
        let t = Instant::now();
        loop {
            let p = self.peers(Duration::from_secs(10));
            if p.len() >= n || t.elapsed() >= timeout {
                return p;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }
}

impl Drop for DiscoveryListener {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

// --- link probe ----------------------------------------------------------------------

/// Measured link to one worker.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinkMeasurement {
    pub rtt_us: f64,
    pub bandwidth_gbps: f64,
}

impl LinkMeasurement {
    /// A solver link profile for this measurement (class from the numbers).
    pub fn profile(&self) -> LinkProfile {
        let kind = if self.rtt_us < 100.0 && self.bandwidth_gbps >= 20.0 {
            LinkKind::ThunderboltRdma
        } else if self.bandwidth_gbps >= 5.0 {
            LinkKind::Ethernet10G
        } else if self.rtt_us < 600.0 {
            LinkKind::Ethernet2_5G
        } else {
            LinkKind::WiFi6E
        };
        LinkProfile {
            kind,
            one_way_us: self.rtt_us / 2.0,
            bandwidth_gbps: self.bandwidth_gbps,
            jitter_us: 0.0,
        }
    }
}

pub const PROBE_PINGS: usize = 3;
pub const PROBE_BURST_BYTES: usize = 64 << 10;

/// Worker side: answer a `ProbePing` (learning the driver's address from it).
pub fn answer_probe(ep: &Endpoint, payload: &[u8]) {
    if payload.len() < 10 {
        return;
    }
    let seq = &payload[0..8];
    let n = u16::from_le_bytes([payload[8], payload[9]]) as usize;
    if let Some(addr) = payload
        .get(10..10 + n)
        .and_then(|s| std::str::from_utf8(s).ok())
        .and_then(|s| s.parse::<SocketAddr>().ok())
    {
        ep.shared.set_addr(DRIVER_ID, addr);
        ep.shared
            .set_links(DRIVER_ID, vec![LinkProfile::unshaped()]);
    }
    let _ = ep.send_payload(DRIVER_ID, FrameType::ProbePong, 0, 0, seq.to_vec());
}

/// Coordinator side: 3 echo pings (min RTT) + one 64 KB burst over the
/// worker's real endpoint.
pub fn probe(driver: &Driver, id: u16, addr: SocketAddr) -> Option<LinkMeasurement> {
    driver.ep.shared.set_addr(id, addr);
    driver
        .ep
        .shared
        .set_links(id, vec![LinkProfile::unshaped()]);
    let me = driver.addr().to_string();
    let ping = |seq: u64, pad: usize| -> Option<f64> {
        let mut p = seq.to_le_bytes().to_vec();
        p.extend_from_slice(&(me.len() as u16).to_le_bytes());
        p.extend_from_slice(me.as_bytes());
        p.resize(p.len().max(pad), 0xA5);
        let t = mono_ns();
        driver
            .ep
            .send_payload(id, FrameType::ProbePing, 0, 0, p)
            .ok()?;
        let tag = seq.to_le_bytes();
        driver.ep.recv_match_timeout(
            &|f| {
                f.header.frame_type == FrameType::ProbePong && f.payload.get(..8) == Some(&tag[..])
            },
            Duration::from_secs(5),
        )?;
        Some((mono_ns() - t) as f64 / 1e3)
    };
    let base = 0x5052_0000u64 + ((id as u64) << 8);
    // the first exchange also opens the TCP connection: not timed
    ping(base, 0)?;
    let rtt = (1..=PROBE_PINGS as u64)
        .map(|i| ping(base + i, 0))
        .collect::<Option<Vec<_>>>()?
        .into_iter()
        .fold(f64::MAX, f64::min);
    let burst = ping(base + 10, PROBE_BURST_BYTES)?;
    let xfer_us = (burst - rtt).max(0.5);
    let gbps = (PROBE_BURST_BYTES as f64 * 8.0 / (xfer_us * 1e3)).min(400.0);
    Some(LinkMeasurement {
        rtt_us: rtt,
        bandwidth_gbps: gbps,
    })
}

// --- plan → assignment -------------------------------------------------------------

/// A probed worker ready for the solver.
#[derive(Clone, Debug)]
pub struct ProbedPeer {
    pub peer: Peer,
    pub link: LinkMeasurement,
}

/// Hardware specs for the solver from probed peers. Worker↔worker links are
/// estimated from the coordinator's measurements (the slower of the two
/// legs), since workers do not probe each other.
pub fn solver_specs(peers: &[ProbedPeer]) -> Vec<HardwareNodeSpec> {
    peers
        .iter()
        .map(|p| HardwareNodeSpec {
            node_id: p.peer.beacon.node_id,
            name: format!("node {} @ {}", p.peer.beacon.node_id, p.peer.ip),
            platform: p.peer.beacon.platform,
            compute_tflops_fp16: p.peer.beacon.compute_tflops_fp16 as f64,
            memory_bandwidth_gbps: p.peer.beacon.mem_bandwidth_gbps as f64,
            available_ram_mb: p.peer.beacon.free_ram_mb as f64,
            links: peers
                .iter()
                .filter(|q| q.peer.beacon.node_id != p.peer.beacon.node_id)
                .map(|q| {
                    let m = LinkMeasurement {
                        rtt_us: p.link.rtt_us.max(q.link.rtt_us),
                        bandwidth_gbps: p.link.bandwidth_gbps.min(q.link.bandwidth_gbps),
                    };
                    LinkSpec {
                        peer: q.peer.beacon.node_id,
                        profiles: vec![m.profile()],
                    }
                })
                .collect(),
        })
        .collect()
}

/// Which engine a discovered cluster runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineChoice {
    /// The solver's plan (H-TEP for < 100 µs groups, else a pipeline).
    Auto,
    /// H-TEP over every discovered node.
    Htep,
    /// A layer pipeline over every discovered node (equal layer split).
    Pipeline,
}

impl EngineChoice {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "auto" => EngineChoice::Auto,
            "htep" => EngineChoice::Htep,
            "pipeline" => EngineChoice::Pipeline,
            _ => return None,
        })
    }
}

/// Engine-slot of a node id: daemons serving several independent sessions
/// expose slot `s` as node `id + SLOT_STRIDE·s`.
pub const SLOT_STRIDE: u16 = 1024;

pub fn slot_of(id: u16) -> usize {
    (id / SLOT_STRIDE) as usize
}

/// Solve the topology for `cfg` over the probed peers and build the
/// assignment (real links: no emulated shaping or compute slowdown).
pub fn plan_assignment(
    cfg: &ModelConfig,
    peers: &[ProbedPeer],
    driver: SocketAddr,
    tess: TesseractOptions,
) -> (ClusterPlan, Assignment) {
    plan_assignment_with(cfg, peers, driver, tess, EngineChoice::Auto)
}

/// [`plan_assignment`] with an engine override.
pub fn plan_assignment_with(
    cfg: &ModelConfig,
    peers: &[ProbedPeer],
    driver: SocketAddr,
    tess: TesseractOptions,
    choice: EngineChoice,
) -> (ClusterPlan, Assignment) {
    let specs = solver_specs(peers);
    let mut plan = HeteroClusterSolver::new(
        specs,
        footprint(cfg, tess.holo_bits),
        SolverConfig::default(),
    )
    .solve();
    let ids: Vec<u16> = peers.iter().map(|p| p.peer.beacon.node_id).collect();
    match choice {
        EngineChoice::Auto => {}
        EngineChoice::Htep => {
            plan.htep = Some(ids.clone());
            plan.notes
                .push("engine forced: H-TEP over every node".into());
        }
        EngineChoice::Pipeline => {
            plan.htep = None;
            let n = ids.len();
            plan.stages = ids
                .iter()
                .enumerate()
                .map(|(i, &node)| crate::topology::StageAssignment {
                    node,
                    layers: cfg.n_layers * i / n..cfg.n_layers * (i + 1) / n,
                })
                .filter(|s| !s.layers.is_empty())
                .collect();
            plan.coordinator = plan.stages.last().map(|s| s.node).unwrap_or(ids[0]);
            plan.notes
                .push("engine forced: layer pipeline over every node".into());
        }
    }
    let (engine, stages, head) = match &plan.htep {
        Some(ranks) => (
            EngineKind::TesseractHtep,
            ranks
                .iter()
                .map(|&r| (r, 0, cfg.n_layers))
                .collect::<Vec<_>>(),
            ranks[0],
        ),
        None => (
            EngineKind::Tesseract,
            plan.stages
                .iter()
                .map(|s| (s.node, s.layers.start, s.layers.end))
                .collect(),
            plan.coordinator,
        ),
    };
    let mut links = Vec::new();
    for (i, &a) in ids.iter().enumerate() {
        for &b in &ids[i + 1..] {
            links.push((a, b, vec![LinkProfile::unshaped()]));
        }
    }
    let nodes = peers
        .iter()
        .map(|p| NodeSetup {
            id: p.peer.beacon.node_id,
            addr: p.peer.node_addr(),
            slowdown: 1.0,
            kv_capacity: plan
                .kv_pools
                .iter()
                .find(|k| k.0 == p.peer.beacon.node_id)
                .map(|k| k.1 as u64)
                .unwrap_or(((p.peer.beacon.free_ram_mb as u64) << 20) / 8),
        })
        .collect();
    let a = Assignment {
        engine,
        model: cfg.clone(),
        nodes,
        links,
        stages,
        head,
        hot_capacity: 0,
        inject_context: 0,
        tess: {
            let mut t = tess;
            t.use_plan_drafter(plan.drafter, plan.coordinator);
            t
        },
        driver_addr: driver,
        service_addrs: peers
            .iter()
            .flat_map(|p| {
                let id = p.peer.beacon.node_id;
                [
                    (KV_SERVICE_BASE + id, p.peer.kv_addr()),
                    (DRAFT_SERVICE_BASE + id, p.peer.draft_addr()),
                ]
            })
            .collect(),
    };
    (plan, a)
}

/// Discover `n` workers, probe each, solve, and assign: the whole
/// zero-configuration bring-up. Returns the probed peers, plan and assignment.
pub fn auto_cluster(
    listener: &DiscoveryListener,
    driver: &Driver,
    cfg: &ModelConfig,
    n: usize,
    timeout: Duration,
    tess: TesseractOptions,
) -> Result<(Vec<ProbedPeer>, ClusterPlan, Assignment), String> {
    auto_cluster_with(
        listener,
        driver,
        cfg,
        n,
        timeout,
        tess,
        EngineChoice::Auto,
        &|_| true,
    )
}

/// [`auto_cluster`] over the peers accepted by `filter` (e.g. one engine
/// slot), with an engine override.
#[allow(clippy::too_many_arguments)]
pub fn auto_cluster_with(
    listener: &DiscoveryListener,
    driver: &Driver,
    cfg: &ModelConfig,
    n: usize,
    timeout: Duration,
    tess: TesseractOptions,
    choice: EngineChoice,
    filter: &dyn Fn(u16) -> bool,
) -> Result<(Vec<ProbedPeer>, ClusterPlan, Assignment), String> {
    let t = Instant::now();
    let found = loop {
        let f: Vec<Peer> = listener
            .peers(Duration::from_secs(10))
            .into_iter()
            .filter(|p| filter(p.beacon.node_id))
            .collect();
        if f.len() >= n || t.elapsed() >= timeout {
            break f;
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    if found.len() < n {
        return Err(format!(
            "discovered {} of {n} workers within {timeout:?}",
            found.len()
        ));
    }
    let probed = probe_all(driver, found.into_iter().take(n).collect())?;
    let (plan, a) = plan_assignment_with(cfg, &probed, driver.addr(), tess, choice);
    driver.assign(&a);
    Ok((probed, plan, a))
}

/// Probe every peer's link.
pub fn probe_all(driver: &Driver, peers: Vec<Peer>) -> Result<Vec<ProbedPeer>, String> {
    let mut probed = Vec::new();
    for p in peers {
        let link = probe(driver, p.beacon.node_id, p.node_addr()).ok_or_else(|| {
            format!(
                "probe of node {} at {} failed",
                p.beacon.node_id,
                p.node_addr()
            )
        })?;
        probed.push(ProbedPeer { peer: p, link });
    }
    Ok(probed)
}

/// Re-plan a running cluster over `peers` (hot-join / leave): end the old
/// assignment on every reachable node, then assign the new plan. Called
/// between requests, so no in-flight generation is dropped.
pub fn replan(
    driver: &Driver,
    old: &Assignment,
    cfg: &ModelConfig,
    peers: &[ProbedPeer],
    tess: TesseractOptions,
    choice: EngineChoice,
) -> (ClusterPlan, Assignment) {
    driver.reset(old);
    let (plan, a) = plan_assignment_with(cfg, peers, driver.addr(), tess, choice);
    driver.assign(&a);
    (plan, a)
}
