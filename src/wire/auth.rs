//! Authenticated, encrypted `TesseractWire` connections.
//!
//! Every connection between nodes (stages, KV and draft services, the
//! driver) starts with a mutual, authenticated key exchange before any frame
//! is accepted, and — with a cluster secret of your own — then carries its
//! frames in ChaCha20-Poly1305 records:
//!
//! ```text
//! dialer  → listener   "TFA3" ‖ mode ‖ n_c ‖ E_c                (nonce 16 B, X25519 key 32 B)
//! listener → dialer    n_s ‖ E_s ‖ HMAC(K_auth, "S" ‖ T)
//! dialer  → listener   HMAC(K_auth, "C" ‖ T)            T = mode ‖ n_c ‖ E_c ‖ n_s ‖ E_s
//! K_session = HKDF(salt K, ikm X25519(e_c, E_s) = X25519(e_s, E_c), info "session" ‖ T)
//! then, if mode = encrypted, every write the dialer makes is a record
//!     len(4) ‖ seg₀ ‖ tag₀ ‖ seg₁ ‖ tag₁ ‖ …     (segments of ≤ 64 KiB)
//! where segment i is ChaCha20-Poly1305(K_session, nonce = seq + i, aad = len)
//! and seq counts segments over the whole connection.
//! ```
//!
//! `K` = HKDF-Extract of the cluster secret (`--cluster-secret`, shared with
//! discovery). Both sides prove knowledge of `K` over a transcript holding
//! the other side's fresh nonce and both ephemeral keys, so a host without
//! the secret can neither open a connection, inject frames, nor substitute
//! its own key exchange, and a recorded handshake cannot be replayed.
//!
//! **Forward secrecy.** The session key also depends on an ephemeral X25519
//! exchange whose private keys exist only during the handshake. A recorded
//! session cannot be decrypted later even by someone who obtains the
//! cluster secret afterwards: they would also need one of the two
//! ephemeral private keys, which are erased once the handshake ends.
//!
//! Segment numbers are implicit, so a modified, dropped, replayed or
//! reordered segment fails authentication and closes the connection.
//! Connections carry frames one way (dialer → listener), so one key per
//! connection suffices. Large records are sealed and opened on several cores
//! at once, one segment per task ([`crypto_threads`]).
//!
//! Encryption is on whenever a secret of your own is set. With the public
//! default secret it is off (a key everyone knows would authenticate
//! nothing), which also keeps the single-host emulated benchmarks unchanged.
//! `--wire-plaintext` turns it off on purpose for trusted dedicated links;
//! both sides must agree, or the handshake fails.
//!
//! Limits: authentication is only as strong as the secret's entropy (use a
//! long random one: someone who knows it can impersonate a node while it is
//! in use); a live connection keeps its session key in memory until it
//! closes; traffic size and timing remain visible.

use super::crypto::{hkdf_expand, hkdf_extract, hmac_sha256, open, seal, x25519, X25519_BASE};
use std::io::{Error, ErrorKind, Read, Result, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;
use std::time::Duration;

/// The secret used when none is configured (public: protects nothing).
pub const DEFAULT_SECRET: &str = "tesseract-local";
const MAGIC: &[u8; 4] = b"TFA3";
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);
/// Largest record payload (bigger writes are split).
const MAX_RECORD: usize = 1 << 20;
/// Segment size inside a record: each segment has its own tag, so the
/// segments of a large record can be sealed and opened in parallel.
const SEGMENT: usize = 1 << 16;
/// Records at least this large are processed on several threads.
const PARALLEL_MIN: usize = 1 << 18;
const HELLO: usize = 4 + 1 + 16 + 32;
const REPLY: usize = 16 + 32 + 32;

/// Keys and mode of a process's node connections.
#[derive(Clone)]
pub struct WireConfig {
    prk: [u8; 32],
    auth: [u8; 32],
    /// Seal frames in ChaCha20-Poly1305 records after the handshake.
    pub encrypt: bool,
}

impl WireConfig {
    pub fn new(secret: &str, encrypt: bool) -> Self {
        let prk = hkdf_extract(b"tesseract-wire v3", secret.as_bytes());
        let mut auth = [0u8; 32];
        hkdf_expand(&prk, b"auth", &mut auth);
        WireConfig { prk, auth, encrypt }
    }

    fn mode(&self) -> u8 {
        self.encrypt as u8
    }

    fn mac(&self, label: u8, t: &Transcript) -> [u8; 32] {
        hmac_sha256(&self.auth, &[&[label, self.mode()], &t.bytes()])
    }

    fn session_key(&self, t: &Transcript, shared: &[u8; 32]) -> [u8; 32] {
        let prk = hkdf_extract(&self.prk, shared);
        let mut info = b"session".to_vec();
        info.push(self.mode());
        info.extend_from_slice(&t.bytes());
        let mut k = [0u8; 32];
        hkdf_expand(&prk, &info, &mut k);
        k
    }
}

/// The handshake values both MACs and the session key are bound to.
struct Transcript {
    nc: [u8; 16],
    ec: [u8; 32],
    ns: [u8; 16],
    es: [u8; 32],
}

impl Transcript {
    fn bytes(&self) -> [u8; 96] {
        let mut b = [0u8; 96];
        b[..16].copy_from_slice(&self.nc);
        b[16..48].copy_from_slice(&self.ec);
        b[48..64].copy_from_slice(&self.ns);
        b[64..].copy_from_slice(&self.es);
        b
    }
}

/// An ephemeral X25519 key pair; the private half is wiped on drop.
struct Ephemeral {
    secret: [u8; 32],
    public: [u8; 32],
}

impl Ephemeral {
    fn new() -> Result<Self> {
        let mut secret = [0u8; 32];
        random(&mut secret)?;
        let public = x25519(&secret, &X25519_BASE);
        Ok(Ephemeral { secret, public })
    }

    /// The shared secret with the peer's public key; refuses the all-zero
    /// result of a small-order point (RFC 7748 §6.1).
    fn agree(&self, peer: &[u8; 32]) -> Result<[u8; 32]> {
        let shared = x25519(&self.secret, peer);
        if ct_eq(&shared, &[0u8; 32]) {
            return Err(denied("peer sent an invalid key-exchange value"));
        }
        Ok(shared)
    }
}

impl Drop for Ephemeral {
    fn drop(&mut self) {
        wipe(&mut self.secret);
    }
}

/// Overwrite key material in a way the optimizer cannot drop.
fn wipe(b: &mut [u8]) {
    for x in b.iter_mut() {
        // SAFETY: a valid, aligned, exclusive reference to a u8.
        unsafe { std::ptr::write_volatile(x, 0) };
    }
    std::sync::atomic::compiler_fence(Ordering::SeqCst);
}

struct Global {
    cfg: WireConfig,
    default_secret: bool,
}

static GLOBAL: RwLock<Option<Global>> = RwLock::new(None);
static REJECTED: AtomicU64 = AtomicU64::new(0);

/// Set the process-wide cluster secret and mode used by every endpoint
/// bound afterwards (binaries do it from `--cluster-secret` and
/// `--wire-plaintext`). Encryption is on unless the secret is the public
/// default or `plaintext` is requested.
pub fn configure(secret: &str, plaintext: bool) {
    let default_secret = secret == DEFAULT_SECRET;
    *GLOBAL.write().unwrap_or_else(|e| e.into_inner()) = Some(Global {
        cfg: WireConfig::new(secret, !default_secret && !plaintext),
        default_secret,
    });
}

/// [`configure`] with encryption on (for a secret of your own).
pub fn set_cluster_secret(secret: &str) {
    configure(secret, false);
}

/// The process-wide configuration (the default secret's if none was set).
pub fn wire_config() -> WireConfig {
    GLOBAL
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map(|g| g.cfg.clone())
        .unwrap_or_else(|| WireConfig::new(DEFAULT_SECRET, false))
}

/// Whether the process still uses the public default secret.
pub fn using_default_secret() -> bool {
    GLOBAL
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map(|g| g.default_secret)
        .unwrap_or(true)
}

/// Connections rejected by this process's listeners (failed handshakes).
pub fn rejected_connections() -> u64 {
    REJECTED.load(Ordering::Relaxed)
}

/// Random bytes from the OS (`/dev/urandom`).
fn random(out: &mut [u8]) -> Result<()> {
    std::fs::File::open("/dev/urandom")?.read_exact(out)
}

fn nonce() -> Result<[u8; 16]> {
    let mut n = [0u8; 16];
    random(&mut n)?;
    Ok(n)
}

/// Constant-time equality of two tags.
pub fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |d, (x, y)| d | (x ^ y)) == 0
}

fn denied(msg: &str) -> Error {
    Error::new(ErrorKind::PermissionDenied, msg.to_string())
}

/// Dialer side: run the key exchange, prove the key, check the listener's
/// proof; returns the session key when the connection is encrypted.
pub fn client_handshake(s: &mut TcpStream, cfg: &WireConfig) -> Result<Option<[u8; 32]>> {
    s.set_read_timeout(Some(HANDSHAKE_TIMEOUT))?;
    let nc = nonce()?;
    let eph = Ephemeral::new()?;
    let mut hello = [0u8; HELLO];
    hello[..4].copy_from_slice(MAGIC);
    hello[4] = cfg.mode();
    hello[5..21].copy_from_slice(&nc);
    hello[21..].copy_from_slice(&eph.public);
    s.write_all(&hello)?;
    let mut reply = [0u8; REPLY];
    s.read_exact(&mut reply).map_err(|_| {
        denied("peer closed the cluster handshake (different --cluster-secret, --wire-plaintext setting or version?)")
    })?;
    let t = Transcript {
        nc,
        ec: eph.public,
        ns: reply[..16].try_into().unwrap(),
        es: reply[16..48].try_into().unwrap(),
    };
    if !ct_eq(&reply[48..], &cfg.mac(b'S', &t)) {
        return Err(denied(
            "peer failed cluster authentication (different --cluster-secret)",
        ));
    }
    let shared = eph.agree(&t.es)?;
    drop(eph);
    s.write_all(&cfg.mac(b'C', &t))?;
    s.set_read_timeout(None)?;
    Ok(cfg.encrypt.then(|| cfg.session_key(&t, &shared)))
}

/// Listener side: run the key exchange and check the dialer's proof before
/// any frame is read; returns the session key when the connection is
/// encrypted.
pub fn server_handshake(s: &mut TcpStream, cfg: &WireConfig) -> Result<Option<[u8; 32]>> {
    let r = (|| {
        s.set_read_timeout(Some(HANDSHAKE_TIMEOUT))?;
        let mut hello = [0u8; HELLO];
        s.read_exact(&mut hello)?;
        if &hello[..4] != MAGIC {
            return Err(denied("not a TesseractWire v3 handshake"));
        }
        if hello[4] != cfg.mode() {
            return Err(denied(
                "encryption mode differs (--wire-plaintext on one side only)",
            ));
        }
        let eph = Ephemeral::new()?;
        let t = Transcript {
            nc: hello[5..21].try_into().unwrap(),
            ec: hello[21..].try_into().unwrap(),
            ns: nonce()?,
            es: eph.public,
        };
        // a small-order key from the dialer is refused before replying
        let shared = eph.agree(&t.ec)?;
        drop(eph);
        let mut reply = [0u8; REPLY];
        reply[..16].copy_from_slice(&t.ns);
        reply[16..48].copy_from_slice(&t.es);
        reply[48..].copy_from_slice(&cfg.mac(b'S', &t));
        s.write_all(&reply)?;
        let mut tag = [0u8; 32];
        s.read_exact(&mut tag)?;
        if !ct_eq(&tag, &cfg.mac(b'C', &t)) {
            return Err(denied("dialer failed cluster authentication"));
        }
        s.set_read_timeout(None)?;
        Ok(cfg.encrypt.then(|| cfg.session_key(&t, &shared)))
    })();
    if r.is_err() {
        REJECTED.fetch_add(1, Ordering::Relaxed);
    }
    r
}

// --- record layer ------------------------------------------------------------------

fn record_nonce(seq: u64) -> [u8; 12] {
    let mut n = [0u8; 12];
    n[4..].copy_from_slice(&seq.to_le_bytes());
    n
}

/// Threads used to seal or open one large record (`TESSERACT_WIRE_THREADS`,
/// default: up to 4, at most half the cores).
pub fn crypto_threads() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("TESSERACT_WIRE_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| {
                let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
                (cores / 2).clamp(1, 4)
            })
            .max(1)
    })
}

/// Run `f(segment index, segment ‖ tag slot)` over the `SEGMENT + 16`-byte
/// pieces of `body`, on several threads when the record is large.
fn for_segments<F>(body: &mut [u8], payload: usize, f: F) -> bool
where
    F: Fn(u64, &mut [u8]) -> bool + Sync,
{
    let threads = crypto_threads();
    if payload < PARALLEL_MIN || threads == 1 {
        return body
            .chunks_mut(SEGMENT + 16)
            .enumerate()
            .all(|(i, c)| f(i as u64, c));
    }
    let segs: Vec<(u64, &mut [u8])> = body
        .chunks_mut(SEGMENT + 16)
        .enumerate()
        .map(|(i, c)| (i as u64, c))
        .collect();
    let per = segs.len().div_ceil(threads);
    let mut groups: Vec<Vec<(u64, &mut [u8])>> = Vec::new();
    let mut it = segs.into_iter().peekable();
    while it.peek().is_some() {
        groups.push(it.by_ref().take(per).collect());
    }
    std::thread::scope(|sc| {
        let f = &f;
        let mut handles = Vec::new();
        let mut groups = groups.into_iter();
        let mine = groups.next().unwrap_or_default();
        for g in groups {
            handles.push(sc.spawn(move || g.into_iter().all(|(i, c)| f(i, c))));
        }
        let ok = mine.into_iter().all(|(i, c)| f(i, c));
        handles
            .into_iter()
            .fold(ok, |acc, h| h.join().unwrap_or(false) && acc)
    })
}

/// This machine's record-layer throughput in GB/s (seal, 1 MiB records,
/// [`crypto_threads`] threads), measured on 16 MiB: for benchmark reports,
/// to tell whether encryption could limit a link.
pub fn measure_seal_gbps() -> f64 {
    let data = vec![0x5au8; 16 << 20];
    let mut sink = Vec::with_capacity(data.len() + (1 << 16));
    let mut best = f64::MAX;
    for _ in 0..3 {
        sink.clear();
        let t = std::time::Instant::now();
        SealedWriter::new(&mut sink, [1u8; 32])
            .write_all(&data)
            .expect("writing to memory");
        best = best.min(t.elapsed().as_secs_f64());
    }
    data.len() as f64 / best / 1e9
}

/// Seals everything written to it into ChaCha20-Poly1305 records.
pub struct SealedWriter<W: Write> {
    inner: W,
    key: [u8; 32],
    seq: u64,
    buf: Vec<u8>,
}

impl<W: Write> SealedWriter<W> {
    pub fn new(inner: W, key: [u8; 32]) -> Self {
        SealedWriter {
            inner,
            key,
            seq: 0,
            buf: Vec::new(),
        }
    }
}

impl<W: Write> Drop for SealedWriter<W> {
    fn drop(&mut self) {
        wipe(&mut self.key);
    }
}

impl<W: Write> Write for SealedWriter<W> {
    fn write(&mut self, data: &[u8]) -> Result<usize> {
        let n = data.len().min(MAX_RECORD);
        if n == 0 {
            return Ok(0);
        }
        let segs = n.div_ceil(SEGMENT);
        self.buf.clear();
        self.buf.extend_from_slice(&(n as u32).to_le_bytes());
        for seg in data[..n].chunks(SEGMENT) {
            self.buf.extend_from_slice(seg);
            self.buf.extend_from_slice(&[0u8; 16]);
        }
        let (len, body) = self.buf.split_at_mut(4);
        let (key, seq, len) = (&self.key, self.seq, &*len);
        for_segments(body, n, |i, c| {
            let (data, tag) = c.split_at_mut(c.len() - 16);
            tag.copy_from_slice(&seal(key, &record_nonce(seq + i), len, data));
            true
        });
        self.seq += segs as u64;
        self.inner.write_all(&self.buf)?;
        Ok(n)
    }

    fn flush(&mut self) -> Result<()> {
        self.inner.flush()
    }
}

/// Opens the records written by a [`SealedWriter`]; any segment that fails
/// authentication (modified, dropped, replayed, reordered) is an error.
pub struct OpenedReader<R: Read> {
    inner: R,
    key: [u8; 32],
    seq: u64,
    buf: Vec<u8>,
    pos: usize,
}

impl<R: Read> OpenedReader<R> {
    pub fn new(inner: R, key: [u8; 32]) -> Self {
        OpenedReader {
            inner,
            key,
            seq: 0,
            buf: Vec::new(),
            pos: 0,
        }
    }
}

impl<R: Read> Drop for OpenedReader<R> {
    fn drop(&mut self) {
        wipe(&mut self.key);
    }
}

impl<R: Read> Read for OpenedReader<R> {
    fn read(&mut self, out: &mut [u8]) -> Result<usize> {
        if out.is_empty() {
            return Ok(0);
        }
        if self.pos == self.buf.len() {
            let mut len = [0u8; 4];
            match self.inner.read_exact(&mut len) {
                Ok(()) => {}
                Err(e) if e.kind() == ErrorKind::UnexpectedEof => return Ok(0),
                Err(e) => return Err(e),
            }
            let n = u32::from_le_bytes(len) as usize;
            if n == 0 || n > MAX_RECORD {
                return Err(Error::new(ErrorKind::InvalidData, "bad record length"));
            }
            let segs = n.div_ceil(SEGMENT);
            self.buf.resize(n + 16 * segs, 0);
            self.inner.read_exact(&mut self.buf)?;
            let (key, seq) = (&self.key, self.seq);
            let ok = for_segments(&mut self.buf, n, |i, c| {
                let (data, tag) = c.split_at_mut(c.len() - 16);
                open(key, &record_nonce(seq + i), &len, data, tag)
            });
            if !ok {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "record authentication failed",
                ));
            }
            self.seq += segs as u64;
            // drop the tags: move each segment down over the previous tags
            for i in 1..segs {
                let from = i * (SEGMENT + 16);
                let end = (from + SEGMENT).min(self.buf.len() - 16);
                self.buf.copy_within(from..end, i * SEGMENT);
            }
            self.buf.truncate(n);
            self.pos = 0;
        }
        let k = out.len().min(self.buf.len() - self.pos);
        out[..k].copy_from_slice(&self.buf[self.pos..self.pos + k]);
        self.pos += k;
        Ok(k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    type HsResult = Result<Option<[u8; 32]>>;

    fn pair(server: WireConfig, client: WireConfig) -> (HsResult, HsResult) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let srv = std::thread::spawn(move || {
            let (mut s, _) = l.accept().unwrap();
            server_handshake(&mut s, &server)
        });
        let mut c = TcpStream::connect(addr).unwrap();
        let cr = client_handshake(&mut c, &client);
        drop(c);
        (srv.join().unwrap(), cr)
    }

    #[test]
    fn handshake_accepts_the_same_secret_and_mode_and_rejects_others() {
        let a = WireConfig::new("alpha", true);
        let (s, c) = pair(a.clone(), a.clone());
        let (s, c) = (s.unwrap(), c.unwrap());
        assert!(
            s.is_some() && s == c,
            "both sides derive the same session key"
        );
        // every connection gets a fresh session key
        let (s2, _) = pair(a.clone(), a.clone());
        assert_ne!(s, s2.unwrap());
        // another secret, or another mode, is refused
        let (_, c) = pair(a.clone(), WireConfig::new("beta", true));
        assert!(c.is_err());
        let (s, c) = pair(a.clone(), WireConfig::new("alpha", false));
        assert!(s.is_err() && c.is_err());
        // plaintext mode: authenticated, no session key
        let p = WireConfig::new("alpha", false);
        let (s, c) = pair(p.clone(), p);
        assert_eq!((s.unwrap(), c.unwrap()), (None, None));
    }

    #[test]
    fn listener_rejects_a_raw_frame_without_handshake() {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let cfg = WireConfig::new("alpha", true);
        let srv = std::thread::spawn(move || {
            let (mut s, _) = l.accept().unwrap();
            server_handshake(&mut s, &cfg)
        });
        let mut c = TcpStream::connect(addr).unwrap();
        c.write_all(&[0x54u8; 64]).unwrap();
        assert!(srv.join().unwrap().is_err());
    }

    #[test]
    fn records_round_trip_and_reject_tampering_replay_and_reordering() {
        let key = [7u8; 32];
        let msgs: Vec<Vec<u8>> = (0..5usize).map(|i| vec![i as u8; 1 + i * 40_000]).collect();
        let mut wire = Vec::new();
        {
            let mut w = SealedWriter::new(&mut wire, key);
            for m in &msgs {
                w.write_all(m).unwrap();
            }
        }
        let plain: Vec<u8> = msgs.concat();
        // the messages are runs of one repeated byte; ciphertext has none
        assert!(
            !wire.windows(32).any(|w| w.iter().all(|&b| b == w[0])),
            "plaintext visible"
        );
        let mut got = Vec::new();
        OpenedReader::new(&wire[..], key)
            .read_to_end(&mut got)
            .unwrap();
        assert_eq!(got, plain);
        // a flipped bit anywhere in a record is detected
        let mut bad = wire.clone();
        bad[10] ^= 1;
        assert!(OpenedReader::new(&bad[..], key)
            .read_to_end(&mut Vec::new())
            .is_err());
        // records out of order (second before first) are detected
        let first = 4 + msgs[0].len() + 16;
        let second = 4 + msgs[1].len() + 16;
        let mut swapped = wire[first..first + second].to_vec();
        swapped.extend_from_slice(&wire[..first]);
        assert!(OpenedReader::new(&swapped[..], key)
            .read_to_end(&mut Vec::new())
            .is_err());
        // a replayed record is detected
        let mut replay = wire[..first].to_vec();
        replay.extend_from_slice(&wire[..first]);
        assert!(OpenedReader::new(&replay[..], key)
            .read_to_end(&mut Vec::new())
            .is_err());
        // the wrong key opens nothing
        assert!(OpenedReader::new(&wire[..], [8u8; 32])
            .read_to_end(&mut Vec::new())
            .is_err());
    }

    #[test]
    fn handshake_refuses_a_small_order_key_exchange_value() {
        // a dialer that knows the secret but sends u = 0 (a small-order
        // point) would force an all-zero shared secret: refused
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let cfg = WireConfig::new("alpha", true);
        let srv = std::thread::spawn(move || {
            let (mut s, _) = l.accept().unwrap();
            server_handshake(&mut s, &cfg)
        });
        let mut c = TcpStream::connect(addr).unwrap();
        let mut hello = [0u8; HELLO];
        hello[..4].copy_from_slice(MAGIC);
        hello[4] = 1;
        c.write_all(&hello).unwrap();
        let err = srv.join().unwrap().unwrap_err();
        assert!(err.to_string().contains("invalid key-exchange"), "{err}");
    }

    #[test]
    fn session_key_depends_on_the_ephemeral_exchange_not_only_the_secret() {
        // what someone holding the secret and a recording of the handshake
        // can compute is not the session key (forward secrecy): the X25519
        // shared secret is missing. Recorded values are the transcript T.
        let a = WireConfig::new("alpha", true);
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let (sa, ca) = (a.clone(), a.clone());
        let srv = std::thread::spawn(move || {
            let (mut s, _) = l.accept().unwrap();
            // record what the dialer sends while handshaking
            let mut tap = [0u8; HELLO];
            s.peek(&mut tap).unwrap();
            while s.peek(&mut tap).unwrap() < HELLO {}
            (tap, server_handshake(&mut s, &sa).unwrap().unwrap())
        });
        let mut c = TcpStream::connect(addr).unwrap();
        let key = client_handshake(&mut c, &ca).unwrap().unwrap();
        let (hello, skey) = srv.join().unwrap();
        assert_eq!(key, skey);
        // the attacker knows K and E_c, n_c (and E_s, n_s from the reply);
        // public values in place of the shared secret give other keys
        let t = Transcript {
            nc: hello[5..21].try_into().unwrap(),
            ec: hello[21..].try_into().unwrap(),
            ns: [0; 16],
            es: [0; 32],
        };
        for guess in [[0u8; 32], t.ec, X25519_BASE] {
            assert_ne!(a.session_key(&t, &guess), key);
        }
    }

    #[test]
    fn large_records_use_parallel_segments_and_reject_segment_swaps() {
        let key = [9u8; 32];
        let msg: Vec<u8> = (0..(MAX_RECORD + 12_345))
            .map(|i| (i * 7 + i / 999) as u8)
            .collect();
        let mut wire = Vec::new();
        SealedWriter::new(&mut wire, key).write_all(&msg).unwrap();
        // two records: 1 MiB (16 segments) and the rest
        let segs = MAX_RECORD / SEGMENT + (msg.len() - MAX_RECORD).div_ceil(SEGMENT);
        assert_eq!(wire.len(), 8 + msg.len() + 16 * segs);
        let mut got = Vec::new();
        OpenedReader::new(&wire[..], key)
            .read_to_end(&mut got)
            .unwrap();
        assert!(got == msg, "large record round trip");
        // a flipped bit in a middle segment, or two segments swapped inside
        // one record, is detected
        let mut bad = wire.clone();
        bad[4 + 5 * (SEGMENT + 16) + 77] ^= 0x80;
        assert!(OpenedReader::new(&bad[..], key)
            .read_to_end(&mut Vec::new())
            .is_err());
        let mut swapped = wire.clone();
        let (s1, s2) = (4 + 2 * (SEGMENT + 16), 4 + 3 * (SEGMENT + 16));
        let seg1 = wire[s1..s1 + SEGMENT + 16].to_vec();
        swapped.copy_within(s2..s2 + SEGMENT + 16, s1);
        swapped[s2..s2 + SEGMENT + 16].copy_from_slice(&seg1);
        assert!(OpenedReader::new(&swapped[..], key)
            .read_to_end(&mut Vec::new())
            .is_err());
        // a truncated record is an error, not a short read
        assert!(OpenedReader::new(&wire[..wire.len() - 1], key)
            .read_to_end(&mut Vec::new())
            .is_err());
    }
}
