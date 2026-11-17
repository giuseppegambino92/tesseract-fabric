//! Authenticated, encrypted `TesseractWire` connections.
//!
//! Every connection between nodes (stages, KV and draft services, the
//! driver) starts with a mutual challenge–response before any frame is
//! accepted, and — with a cluster secret of your own — then carries its
//! frames in ChaCha20-Poly1305 records:
//!
//! ```text
//! dialer  → listener   "TFA2" ‖ mode ‖ n_c                 (16-byte nonce)
//! listener → dialer    n_s ‖ HMAC(K_auth, "S" ‖ mode ‖ n_c ‖ n_s)
//! dialer  → listener   HMAC(K_auth, "C" ‖ mode ‖ n_s ‖ n_c)
//! then, if mode = encrypted, every byte the dialer sends is a record
//!     len(4, authenticated) ‖ ChaCha20-Poly1305(K_session, seq, payload) ‖ tag(16)
//! with K_session = HKDF(K, "session" ‖ n_c ‖ n_s) and seq = 0, 1, 2, …
//! ```
//!
//! `K` = HKDF-Extract of the cluster secret (`--cluster-secret`, shared with
//! discovery). Both sides prove knowledge of `K` over the other side's fresh
//! nonce, so a host without the secret cannot open a connection or inject
//! frames, and a recorded handshake cannot be replayed. Each connection has
//! its own session key; record numbers are implicit, so a modified, dropped,
//! replayed or reordered record fails authentication and closes the
//! connection. Connections carry frames one way (dialer → listener), so one
//! key per connection suffices.
//!
//! Encryption is on whenever a secret of your own is set. With the public
//! default secret it is off (a key everyone knows would protect nothing),
//! which also keeps the single-host emulated benchmarks unchanged.
//! `--wire-plaintext` turns it off on purpose for trusted dedicated links;
//! both sides must agree, or the handshake fails.
//!
//! Limits: the secret is only as strong as its entropy (use a long random
//! one); there is no forward secrecy (a leaked secret plus recorded traffic
//! reveals that traffic); traffic size and timing remain visible.

use super::crypto::{hkdf_expand, hkdf_extract, hmac_sha256, open, seal};
use std::io::{Error, ErrorKind, Read, Result, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;
use std::time::Duration;

/// The secret used when none is configured (public: protects nothing).
pub const DEFAULT_SECRET: &str = "tesseract-local";
const MAGIC: &[u8; 4] = b"TFA2";
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);
/// Largest record payload (bigger writes are split).
const MAX_RECORD: usize = 1 << 20;

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
        let prk = hkdf_extract(b"tesseract-wire v2", secret.as_bytes());
        let mut auth = [0u8; 32];
        hkdf_expand(&prk, b"auth", &mut auth);
        WireConfig { prk, auth, encrypt }
    }

    fn mode(&self) -> u8 {
        self.encrypt as u8
    }

    fn mac(&self, label: u8, a: &[u8; 16], b: &[u8; 16]) -> [u8; 32] {
        hmac_sha256(&self.auth, &[&[label, self.mode()], a, b])
    }

    fn session_key(&self, nc: &[u8; 16], ns: &[u8; 16]) -> [u8; 32] {
        let mut info = b"session".to_vec();
        info.extend_from_slice(nc);
        info.extend_from_slice(ns);
        let mut k = [0u8; 32];
        hkdf_expand(&self.prk, &info, &mut k);
        k
    }
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

/// 16 random bytes from the OS (`/dev/urandom`).
fn nonce() -> Result<[u8; 16]> {
    let mut n = [0u8; 16];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut n)?;
    Ok(n)
}

/// Constant-time equality of two tags.
pub fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |d, (x, y)| d | (x ^ y)) == 0
}

fn denied(msg: &str) -> Error {
    Error::new(ErrorKind::PermissionDenied, msg.to_string())
}

/// Dialer side: prove the key, check the listener's proof; returns the
/// session key when the connection is encrypted.
pub fn client_handshake(s: &mut TcpStream, cfg: &WireConfig) -> Result<Option<[u8; 32]>> {
    s.set_read_timeout(Some(HANDSHAKE_TIMEOUT))?;
    let nc = nonce()?;
    let mut hello = [0u8; 21];
    hello[..4].copy_from_slice(MAGIC);
    hello[4] = cfg.mode();
    hello[5..].copy_from_slice(&nc);
    s.write_all(&hello)?;
    let mut reply = [0u8; 48];
    s.read_exact(&mut reply).map_err(|_| {
        denied("peer closed the cluster handshake (different --cluster-secret, --wire-plaintext setting or version?)")
    })?;
    let ns: [u8; 16] = reply[..16].try_into().unwrap();
    if !ct_eq(&reply[16..], &cfg.mac(b'S', &nc, &ns)) {
        return Err(denied(
            "peer failed cluster authentication (different --cluster-secret)",
        ));
    }
    s.write_all(&cfg.mac(b'C', &ns, &nc))?;
    s.set_read_timeout(None)?;
    Ok(cfg.encrypt.then(|| cfg.session_key(&nc, &ns)))
}

/// Listener side: check the dialer's proof before any frame is read;
/// returns the session key when the connection is encrypted.
pub fn server_handshake(s: &mut TcpStream, cfg: &WireConfig) -> Result<Option<[u8; 32]>> {
    let r = (|| {
        s.set_read_timeout(Some(HANDSHAKE_TIMEOUT))?;
        let mut hello = [0u8; 21];
        s.read_exact(&mut hello)?;
        if &hello[..4] != MAGIC {
            return Err(denied("not a TesseractWire v2 handshake"));
        }
        if hello[4] != cfg.mode() {
            return Err(denied(
                "encryption mode differs (--wire-plaintext on one side only)",
            ));
        }
        let nc: [u8; 16] = hello[5..].try_into().unwrap();
        let ns = nonce()?;
        let mut reply = [0u8; 48];
        reply[..16].copy_from_slice(&ns);
        reply[16..].copy_from_slice(&cfg.mac(b'S', &nc, &ns));
        s.write_all(&reply)?;
        let mut tag = [0u8; 32];
        s.read_exact(&mut tag)?;
        if !ct_eq(&tag, &cfg.mac(b'C', &ns, &nc)) {
            return Err(denied("dialer failed cluster authentication"));
        }
        s.set_read_timeout(None)?;
        Ok(cfg.encrypt.then(|| cfg.session_key(&nc, &ns)))
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

impl<W: Write> Write for SealedWriter<W> {
    fn write(&mut self, data: &[u8]) -> Result<usize> {
        let n = data.len().min(MAX_RECORD);
        if n == 0 {
            return Ok(0);
        }
        self.buf.clear();
        self.buf.extend_from_slice(&(n as u32).to_le_bytes());
        self.buf.extend_from_slice(&data[..n]);
        let (len, body) = self.buf.split_at_mut(4);
        let tag = seal(&self.key, &record_nonce(self.seq), len, body);
        self.seq += 1;
        self.buf.extend_from_slice(&tag);
        self.inner.write_all(&self.buf)?;
        Ok(n)
    }

    fn flush(&mut self) -> Result<()> {
        self.inner.flush()
    }
}

/// Opens the records written by a [`SealedWriter`]; any record that fails
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
            self.buf.resize(n + 16, 0);
            self.inner.read_exact(&mut self.buf)?;
            let (body, tag) = self.buf.split_at_mut(n);
            if !open(&self.key, &record_nonce(self.seq), &len, body, tag) {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "record authentication failed",
                ));
            }
            self.seq += 1;
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
}
