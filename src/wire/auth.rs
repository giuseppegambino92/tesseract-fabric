//! Mutual authentication of `TesseractWire` TCP connections.
//!
//! Every connection between nodes (stages, KV and draft services, the
//! driver) starts with a challenge–response before any frame is accepted:
//!
//! ```text
//! dialer  → listener   "TFA1" ‖ n_c                (16-byte random nonce)
//! listener → dialer    n_s ‖ MAC(K, "S" ‖ n_c ‖ n_s)
//! dialer  → listener   MAC(K, "C" ‖ n_s ‖ n_c)
//! ```
//!
//! `K` is derived from the cluster secret (`--cluster-secret`, shared with
//! discovery); `MAC` is a 128-bit SipHash-2-4 tag (two independent keyed
//! passes). Both sides prove knowledge of `K` over the other side's fresh
//! nonce, so a host without the secret cannot open a connection and inject
//! frames (assignments, generate requests, KV traffic), and a recorded
//! handshake cannot be replayed. The listener drops the socket on any
//! mismatch, before reading a single frame.
//!
//! Scope: this authenticates connections; it does **not** encrypt or
//! integrity-protect the frames that follow. On networks you do not trust,
//! run the cluster over WireGuard / Tailscale (discovery supports unicast
//! seeds for exactly that), which provides confidentiality and per-packet
//! integrity.

use crate::cluster::discovery::siphash24;
use std::io::{Error, ErrorKind, Read, Result, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;
use std::time::Duration;

/// The secret used when none is configured (public: protects nothing).
pub const DEFAULT_SECRET: &str = "tesseract-local";
const MAGIC: &[u8; 4] = b"TFA1";
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

/// Connection-authentication key derived from a cluster secret.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct WireKey {
    k0: u64,
    k1: u64,
}

impl WireKey {
    pub fn from_secret(secret: &str) -> Self {
        let b = secret.as_bytes();
        // domain-separated from the discovery beacon keys
        WireKey {
            k0: siphash24(0x7465_7373_7769_7265, 0x6b30, b),
            k1: siphash24(0x7465_7373_7769_7265, 0x6b31, b),
        }
    }

    fn mac(&self, label: u8, a: &[u8; 16], b: &[u8; 16]) -> [u8; 16] {
        let mut m = Vec::with_capacity(37);
        m.push(label);
        m.extend_from_slice(a);
        m.extend_from_slice(b);
        let mut out = [0u8; 16];
        out[..8].copy_from_slice(&siphash24(self.k0, self.k1, &m).to_le_bytes());
        m.push(0xa5);
        out[8..].copy_from_slice(&siphash24(self.k1 ^ 0x5bd1_e995, self.k0, &m).to_le_bytes());
        out
    }
}

static KEY: RwLock<Option<(WireKey, bool)>> = RwLock::new(None);
static REJECTED: AtomicU64 = AtomicU64::new(0);

/// Set the process-wide cluster secret used by every endpoint (call before
/// binding endpoints; binaries do it from `--cluster-secret`).
pub fn set_cluster_secret(secret: &str) {
    *KEY.write().unwrap_or_else(|e| e.into_inner()) =
        Some((WireKey::from_secret(secret), secret == DEFAULT_SECRET));
}

/// The process-wide key (the default secret's if none was set).
pub fn cluster_key() -> WireKey {
    KEY.read()
        .unwrap_or_else(|e| e.into_inner())
        .map(|k| k.0)
        .unwrap_or_else(|| WireKey::from_secret(DEFAULT_SECRET))
}

/// Whether the process still uses the public default secret.
pub fn using_default_secret() -> bool {
    KEY.read()
        .unwrap_or_else(|e| e.into_inner())
        .map(|k| k.1)
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

/// Dialer side: prove the key and check the listener's proof.
pub fn client_handshake(s: &mut TcpStream, key: &WireKey) -> Result<()> {
    s.set_read_timeout(Some(HANDSHAKE_TIMEOUT))?;
    let nc = nonce()?;
    let mut hello = [0u8; 20];
    hello[..4].copy_from_slice(MAGIC);
    hello[4..].copy_from_slice(&nc);
    s.write_all(&hello)?;
    let mut reply = [0u8; 32];
    s.read_exact(&mut reply)
        .map_err(|_| denied("peer closed the cluster handshake (wrong --cluster-secret?)"))?;
    let ns: [u8; 16] = reply[..16].try_into().unwrap();
    if !ct_eq(&reply[16..], &key.mac(b'S', &nc, &ns)) {
        return Err(denied(
            "peer failed cluster authentication (different --cluster-secret)",
        ));
    }
    s.write_all(&key.mac(b'C', &ns, &nc))?;
    s.set_read_timeout(None)?;
    Ok(())
}

/// Listener side: check the dialer's proof before any frame is read.
pub fn server_handshake(s: &mut TcpStream, key: &WireKey) -> Result<()> {
    let r = (|| {
        s.set_read_timeout(Some(HANDSHAKE_TIMEOUT))?;
        let mut hello = [0u8; 20];
        s.read_exact(&mut hello)?;
        if &hello[..4] != MAGIC {
            return Err(denied("not a TesseractWire handshake"));
        }
        let nc: [u8; 16] = hello[4..].try_into().unwrap();
        let ns = nonce()?;
        let mut reply = [0u8; 32];
        reply[..16].copy_from_slice(&ns);
        reply[16..].copy_from_slice(&key.mac(b'S', &nc, &ns));
        s.write_all(&reply)?;
        let mut tag = [0u8; 16];
        s.read_exact(&mut tag)?;
        if !ct_eq(&tag, &key.mac(b'C', &ns, &nc)) {
            return Err(denied("dialer failed cluster authentication"));
        }
        s.set_read_timeout(None)?;
        Ok(())
    })();
    if r.is_err() {
        REJECTED.fetch_add(1, Ordering::Relaxed);
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    fn pair(server_key: WireKey, client_key: WireKey) -> (Result<()>, Result<()>) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let srv = std::thread::spawn(move || {
            let (mut s, _) = l.accept().unwrap();
            server_handshake(&mut s, &server_key)
        });
        let mut c = TcpStream::connect(addr).unwrap();
        let cr = client_handshake(&mut c, &client_key);
        drop(c);
        (srv.join().unwrap(), cr)
    }

    #[test]
    fn handshake_accepts_the_same_secret_and_rejects_others() {
        let a = WireKey::from_secret("alpha");
        let b = WireKey::from_secret("beta");
        let (s, c) = pair(a, a);
        assert!(s.is_ok() && c.is_ok());
        // either side refuses a peer with another secret
        let (s, c) = pair(a, b);
        assert!(s.is_err() || c.is_err());
        assert!(
            c.is_err(),
            "the dialer must detect a listener with another key"
        );
        // the tag depends on both nonces: two handshakes never repeat
        let n1 = nonce().unwrap();
        let n2 = nonce().unwrap();
        assert_ne!(a.mac(b'C', &n1, &n2), a.mac(b'C', &n2, &n1));
        assert_ne!(a.mac(b'S', &n1, &n2), b.mac(b'S', &n1, &n2));
    }

    #[test]
    fn listener_rejects_a_raw_frame_without_handshake() {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let key = WireKey::from_secret("alpha");
        let srv = std::thread::spawn(move || {
            let (mut s, _) = l.accept().unwrap();
            server_handshake(&mut s, &key)
        });
        let mut c = TcpStream::connect(addr).unwrap();
        // a TesseractWire frame header instead of the handshake
        c.write_all(&[0x54u8; 64]).unwrap();
        assert!(srv.join().unwrap().is_err());
    }
}
