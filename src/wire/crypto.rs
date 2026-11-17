//! Cryptographic primitives for authenticated encryption of node traffic,
//! implemented in-tree (zero external crates) and checked against the
//! published test vectors (FIPS 180-4, RFC 4231, RFC 5869, RFC 7748, RFC 8439) and,
//! during development, against an independent implementation (OpenSSL via
//! Python `cryptography`):
//!
//! * SHA-256, HMAC-SHA256, HKDF-SHA256 — key derivation and handshake MACs;
//! * X25519 (RFC 7748) — the ephemeral key exchange that gives every
//!   connection forward secrecy;
//! * ChaCha20, Poly1305 and the ChaCha20-Poly1305 AEAD (RFC 8439) — the
//!   record layer of encrypted `TesseractWire` connections.
//!
//! Tag comparisons are constant-time. These are straightforward portable
//! implementations, not hardened against every side channel (e.g. cache
//! timing on exotic CPUs); ChaCha20, Poly1305 and X25519 use only
//! additions, rotations, XORs, masks and multiplications, with no
//! secret-dependent branches or memory indices, which is constant-time on
//! the targets this project runs on.

// --- SHA-256 ------------------------------------------------------------------

const K256: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// Incremental SHA-256.
#[derive(Clone)]
pub struct Sha256 {
    h: [u32; 8],
    buf: [u8; 64],
    len: usize,
    total: u64,
}

impl Default for Sha256 {
    fn default() -> Self {
        Sha256 {
            h: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            buf: [0; 64],
            len: 0,
            total: 0,
        }
    }
}

impl Sha256 {
    fn block(&mut self, b: &[u8; 64]) {
        let mut w = [0u32; 64];
        for (i, c) in b.as_chunks::<4>().0.iter().enumerate() {
            w[i] = u32::from_be_bytes(*c);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b_, mut c, mut d, mut e, mut f, mut g, mut h] = self.h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K256[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b_) ^ (a & c) ^ (b_ & c);
            let t2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b_;
            b_ = a;
            a = t1.wrapping_add(t2);
        }
        for (s, v) in self.h.iter_mut().zip([a, b_, c, d, e, f, g, h]) {
            *s = s.wrapping_add(v);
        }
    }

    pub fn update(&mut self, mut data: &[u8]) {
        self.total += data.len() as u64;
        if self.len > 0 {
            let take = (64 - self.len).min(data.len());
            self.buf[self.len..self.len + take].copy_from_slice(&data[..take]);
            self.len += take;
            data = &data[take..];
            if self.len < 64 {
                return; // still a partial block: nothing more to consume
            }
            let b = self.buf;
            self.block(&b);
            self.len = 0;
        }
        while data.len() >= 64 {
            self.block(data[..64].try_into().unwrap());
            data = &data[64..];
        }
        self.buf[..data.len()].copy_from_slice(data);
        self.len = data.len();
    }

    pub fn finish(mut self) -> [u8; 32] {
        let bits = self.total.wrapping_mul(8);
        self.update(&[0x80]);
        while self.len != 56 {
            self.update(&[0]);
        }
        self.update(&bits.to_be_bytes());
        let mut out = [0u8; 32];
        for (o, v) in out.as_chunks_mut::<4>().0.iter_mut().zip(self.h) {
            o.copy_from_slice(&v.to_be_bytes());
        }
        out
    }
}

pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut h = Sha256::default();
    h.update(data);
    h.finish()
}

/// HMAC-SHA256 over the concatenation of `parts`.
pub fn hmac_sha256(key: &[u8], parts: &[&[u8]]) -> [u8; 32] {
    let mut k = [0u8; 64];
    if key.len() > 64 {
        k[..32].copy_from_slice(&sha256(key));
    } else {
        k[..key.len()].copy_from_slice(key);
    }
    let mut inner = Sha256::default();
    inner.update(&k.map(|b| b ^ 0x36));
    for p in parts {
        inner.update(p);
    }
    let mut outer = Sha256::default();
    outer.update(&k.map(|b| b ^ 0x5c));
    outer.update(&inner.finish());
    outer.finish()
}

/// HKDF-SHA256 extract (RFC 5869).
pub fn hkdf_extract(salt: &[u8], ikm: &[u8]) -> [u8; 32] {
    hmac_sha256(salt, &[ikm])
}

/// HKDF-SHA256 expand (RFC 5869) into `out` (at most 255 × 32 bytes).
pub fn hkdf_expand(prk: &[u8; 32], info: &[u8], out: &mut [u8]) {
    assert!(out.len() <= 255 * 32);
    let mut t: Vec<u8> = Vec::new();
    for (i, chunk) in out.chunks_mut(32).enumerate() {
        let block = hmac_sha256(prk, &[&t, info, &[(i + 1) as u8]]);
        chunk.copy_from_slice(&block[..chunk.len()]);
        t = block.to_vec();
    }
}

// --- ChaCha20 -------------------------------------------------------------------

const SIGMA: [u32; 4] = [0x6170_7865, 0x3320_646e, 0x7962_2d32, 0x6b20_6574];

fn chacha_init(key: &[u8; 32], counter: u32, nonce: &[u8; 12]) -> [u32; 16] {
    let mut init = [0u32; 16];
    init[..4].copy_from_slice(&SIGMA);
    for i in 0..8 {
        init[4 + i] = u32::from_le_bytes(key[4 * i..4 * i + 4].try_into().unwrap());
    }
    init[12] = counter;
    for i in 0..3 {
        init[13 + i] = u32::from_le_bytes(nonce[4 * i..4 * i + 4].try_into().unwrap());
    }
    init
}

#[inline(always)]
fn qr(s: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
    s[a] = s[a].wrapping_add(s[b]);
    s[d] = (s[d] ^ s[a]).rotate_left(16);
    s[c] = s[c].wrapping_add(s[d]);
    s[b] = (s[b] ^ s[c]).rotate_left(12);
    s[a] = s[a].wrapping_add(s[b]);
    s[d] = (s[d] ^ s[a]).rotate_left(8);
    s[c] = s[c].wrapping_add(s[d]);
    s[b] = (s[b] ^ s[c]).rotate_left(7);
}

/// One 64-byte ChaCha20 keystream block (RFC 8439 §2.3).
pub fn chacha20_block(key: &[u8; 32], counter: u32, nonce: &[u8; 12]) -> [u8; 64] {
    let init = chacha_init(key, counter, nonce);
    let mut s = init;
    for _ in 0..10 {
        qr(&mut s, 0, 4, 8, 12);
        qr(&mut s, 1, 5, 9, 13);
        qr(&mut s, 2, 6, 10, 14);
        qr(&mut s, 3, 7, 11, 15);
        qr(&mut s, 0, 5, 10, 15);
        qr(&mut s, 1, 6, 11, 12);
        qr(&mut s, 2, 7, 8, 13);
        qr(&mut s, 3, 4, 9, 14);
    }
    let mut out = [0u8; 64];
    for i in 0..16 {
        out[4 * i..4 * i + 4].copy_from_slice(&s[i].wrapping_add(init[i]).to_le_bytes());
    }
    out
}

/// Portable multi-block path (x86_64 and other targets; aarch64 uses the
/// explicit NEON code below). `L` blocks side by side, one block per vector lane: word `i` of every
/// block lives in `s[i]`, so each quarter-round step is one SIMD operation
/// on NEON (`L` = 4 or 8) and AVX2 (`L` = 8); the compiler vectorizes the
/// lane loops.
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn qr_lanes<const L: usize>(s: &mut [[u32; L]; 16], a: usize, b: usize, c: usize, d: usize) {
    macro_rules! step {
        ($x:expr, $y:expr, $z:expr, $r:expr) => {
            for l in 0..L {
                s[$x][l] = s[$x][l].wrapping_add(s[$y][l]);
            }
            for l in 0..L {
                s[$z][l] = (s[$z][l] ^ s[$x][l]).rotate_left($r);
            }
        };
    }
    step!(a, b, d, 16);
    step!(c, d, b, 12);
    step!(a, b, d, 8);
    step!(c, d, b, 7);
}

/// XOR `L` consecutive keystream blocks into `data` (exactly `64 · L` bytes).
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn chacha_xor_lanes<const L: usize>(init: &[u32; 16], counter: u32, data: &mut [u8]) {
    let mut s = [[0u32; L]; 16];
    for i in 0..16 {
        s[i] = [init[i]; L];
    }
    for (l, c) in s[12].iter_mut().enumerate() {
        *c = counter.wrapping_add(l as u32);
    }
    let start = s;
    for _ in 0..10 {
        qr_lanes(&mut s, 0, 4, 8, 12);
        qr_lanes(&mut s, 1, 5, 9, 13);
        qr_lanes(&mut s, 2, 6, 10, 14);
        qr_lanes(&mut s, 3, 7, 11, 15);
        qr_lanes(&mut s, 0, 5, 10, 15);
        qr_lanes(&mut s, 1, 6, 11, 12);
        qr_lanes(&mut s, 2, 7, 8, 13);
        qr_lanes(&mut s, 3, 4, 9, 14);
    }
    for i in 0..16 {
        for l in 0..L {
            s[i][l] = s[i][l].wrapping_add(start[i][l]);
        }
    }
    for (l, block) in data.as_chunks_mut::<64>().0.iter_mut().enumerate() {
        for (i, w) in block.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            *w = (u32::from_le_bytes(*w) ^ s[i][l]).to_le_bytes();
        }
    }
}

/// XOR the keystream into `data`, a multiple of 256 bytes; returns the
/// next block counter.
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn chacha_xor_bulk(init: &[u32; 16], mut counter: u32, data: &mut [u8]) -> u32 {
    let (big, rest) = data.as_chunks_mut::<512>();
    for c in big {
        chacha_xor_lanes::<8>(init, counter, c);
        counter = counter.wrapping_add(8);
    }
    for c in rest.as_chunks_mut::<256>().0 {
        chacha_xor_lanes::<4>(init, counter, c);
        counter = counter.wrapping_add(4);
    }
    counter
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn chacha_xor_avx2(init: &[u32; 16], counter: u32, data: &mut [u8]) -> u32 {
    chacha_xor_bulk(init, counter, data)
}

/// Four ChaCha20 blocks in NEON registers (one block per lane); 16-bit and
/// 8-bit rotations are byte shuffles, the others shift-and-insert.
#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    #[inline(always)]
    unsafe fn rotl<const L: i32, const R: i32>(x: uint32x4_t) -> uint32x4_t {
        vsriq_n_u32::<R>(vshlq_n_u32::<L>(x), x)
    }

    #[inline(always)]
    unsafe fn rot16(x: uint32x4_t) -> uint32x4_t {
        vreinterpretq_u32_u16(vrev32q_u16(vreinterpretq_u16_u32(x)))
    }

    #[inline(always)]
    unsafe fn rot8(x: uint32x4_t, tbl: uint8x16_t) -> uint32x4_t {
        vreinterpretq_u32_u8(vqtbl1q_u8(vreinterpretq_u8_u32(x), tbl))
    }

    #[inline(always)]
    unsafe fn qr(s: &mut [uint32x4_t; 16], a: usize, b: usize, c: usize, d: usize, t8: uint8x16_t) {
        s[a] = vaddq_u32(s[a], s[b]);
        s[d] = rot16(veorq_u32(s[d], s[a]));
        s[c] = vaddq_u32(s[c], s[d]);
        s[b] = rotl::<12, 20>(veorq_u32(s[b], s[c]));
        s[a] = vaddq_u32(s[a], s[b]);
        s[d] = rot8(veorq_u32(s[d], s[a]), t8);
        s[c] = vaddq_u32(s[c], s[d]);
        s[b] = rotl::<7, 25>(veorq_u32(s[b], s[c]));
    }

    #[inline(always)]
    unsafe fn qr2(
        s: &mut [uint32x4_t; 16],
        u: &mut [uint32x4_t; 16],
        (a, b, c, d): (usize, usize, usize, usize),
        t8: uint8x16_t,
    ) {
        s[a] = vaddq_u32(s[a], s[b]);
        u[a] = vaddq_u32(u[a], u[b]);
        s[d] = rot16(veorq_u32(s[d], s[a]));
        u[d] = rot16(veorq_u32(u[d], u[a]));
        s[c] = vaddq_u32(s[c], s[d]);
        u[c] = vaddq_u32(u[c], u[d]);
        s[b] = rotl::<12, 20>(veorq_u32(s[b], s[c]));
        u[b] = rotl::<12, 20>(veorq_u32(u[b], u[c]));
        s[a] = vaddq_u32(s[a], s[b]);
        u[a] = vaddq_u32(u[a], u[b]);
        s[d] = rot8(veorq_u32(s[d], s[a]), t8);
        u[d] = rot8(veorq_u32(u[d], u[a]), t8);
        s[c] = vaddq_u32(s[c], s[d]);
        u[c] = vaddq_u32(u[c], u[d]);
        s[b] = rotl::<7, 25>(veorq_u32(s[b], s[c]));
        u[b] = rotl::<7, 25>(veorq_u32(u[b], u[c]));
    }

    /// XOR 8 keystream blocks into `data` (512 bytes): two interleaved
    /// groups of four, for more independent work per cycle.
    pub fn xor8(init: &[u32; 16], counter: u32, data: &mut [u8; 512]) {
        // SAFETY: as in `xor4`.
        unsafe {
            const ROT8: [u8; 16] = [3, 0, 1, 2, 7, 4, 5, 6, 11, 8, 9, 10, 15, 12, 13, 14];
            let t8 = vld1q_u8(ROT8.as_ptr());
            let mut s: [uint32x4_t; 16] = [vdupq_n_u32(0); 16];
            for i in 0..16 {
                s[i] = vdupq_n_u32(init[i]);
            }
            let mut u = s;
            let ctr: [u32; 8] = std::array::from_fn(|i| counter.wrapping_add(i as u32));
            s[12] = vld1q_u32(ctr.as_ptr());
            u[12] = vld1q_u32(ctr.as_ptr().add(4));
            for _ in 0..10 {
                qr2(&mut s, &mut u, (0, 4, 8, 12), t8);
                qr2(&mut s, &mut u, (1, 5, 9, 13), t8);
                qr2(&mut s, &mut u, (2, 6, 10, 14), t8);
                qr2(&mut s, &mut u, (3, 7, 11, 15), t8);
                qr2(&mut s, &mut u, (0, 5, 10, 15), t8);
                qr2(&mut s, &mut u, (1, 6, 11, 12), t8);
                qr2(&mut s, &mut u, (2, 7, 8, 13), t8);
                qr2(&mut s, &mut u, (3, 4, 9, 14), t8);
            }
            finish(&mut s, init, counter, data.as_mut_ptr());
            finish(
                &mut u,
                init,
                counter.wrapping_add(4),
                data.as_mut_ptr().add(256),
            );
        }
    }

    /// Add the input words back and XOR the 4 blocks into `p[..256]`.
    #[inline(always)]
    unsafe fn finish(s: &mut [uint32x4_t; 16], init: &[u32; 16], counter: u32, p: *mut u8) {
        for i in 0..16 {
            s[i] = vaddq_u32(s[i], vdupq_n_u32(init[i]));
        }
        let ctr = [0, 1, 2, 3u32].map(|k| counter.wrapping_add(k));
        s[12] = vaddq_u32(
            s[12],
            vsubq_u32(vld1q_u32(ctr.as_ptr()), vdupq_n_u32(init[12])),
        );
        store(s, p);
    }

    #[inline(always)]
    unsafe fn store(s: &[uint32x4_t; 16], p: *mut u8) {
        for g in 0..4 {
            let (a, b, c, d) = (s[4 * g], s[4 * g + 1], s[4 * g + 2], s[4 * g + 3]);
            let ab_lo = vreinterpretq_u64_u32(vzip1q_u32(a, b));
            let ab_hi = vreinterpretq_u64_u32(vzip2q_u32(a, b));
            let cd_lo = vreinterpretq_u64_u32(vzip1q_u32(c, d));
            let cd_hi = vreinterpretq_u64_u32(vzip2q_u32(c, d));
            let rows = [
                vzip1q_u64(ab_lo, cd_lo),
                vzip2q_u64(ab_lo, cd_lo),
                vzip1q_u64(ab_hi, cd_hi),
                vzip2q_u64(ab_hi, cd_hi),
            ];
            for (lane, row) in rows.into_iter().enumerate() {
                let q = p.add(64 * lane + 16 * g);
                vst1q_u8(q, veorq_u8(vld1q_u8(q), vreinterpretq_u8_u64(row)));
            }
        }
    }

    /// XOR 4 keystream blocks into `data` (256 bytes).
    pub fn xor4(init: &[u32; 16], counter: u32, data: &mut [u8; 256]) {
        // SAFETY: NEON is mandatory on aarch64; all loads and stores stay
        // within the arrays and the 256-byte block.
        unsafe {
            const ROT8: [u8; 16] = [3, 0, 1, 2, 7, 4, 5, 6, 11, 8, 9, 10, 15, 12, 13, 14];
            let t8 = vld1q_u8(ROT8.as_ptr());
            let mut s: [uint32x4_t; 16] = [vdupq_n_u32(0); 16];
            for i in 0..16 {
                s[i] = vdupq_n_u32(init[i]);
            }
            let ctr = [
                counter,
                counter.wrapping_add(1),
                counter.wrapping_add(2),
                counter.wrapping_add(3),
            ];
            s[12] = vld1q_u32(ctr.as_ptr());
            let start = s;
            for _ in 0..10 {
                qr(&mut s, 0, 4, 8, 12, t8);
                qr(&mut s, 1, 5, 9, 13, t8);
                qr(&mut s, 2, 6, 10, 14, t8);
                qr(&mut s, 3, 7, 11, 15, t8);
                qr(&mut s, 0, 5, 10, 15, t8);
                qr(&mut s, 1, 6, 11, 12, t8);
                qr(&mut s, 2, 7, 8, 13, t8);
                qr(&mut s, 3, 4, 9, 14, t8);
            }
            for i in 0..16 {
                s[i] = vaddq_u32(s[i], start[i]);
            }
            // transpose 4×4 word groups so each vector holds 16 bytes of one block
            let p = data.as_mut_ptr();
            for g in 0..4 {
                let (a, b, c, d) = (s[4 * g], s[4 * g + 1], s[4 * g + 2], s[4 * g + 3]);
                let ab_lo = vzip1q_u32(a, b);
                let ab_hi = vzip2q_u32(a, b);
                let cd_lo = vzip1q_u32(c, d);
                let cd_hi = vzip2q_u32(c, d);
                let rows = [
                    vreinterpretq_u32_u64(vzip1q_u64(
                        vreinterpretq_u64_u32(ab_lo),
                        vreinterpretq_u64_u32(cd_lo),
                    )),
                    vreinterpretq_u32_u64(vzip2q_u64(
                        vreinterpretq_u64_u32(ab_lo),
                        vreinterpretq_u64_u32(cd_lo),
                    )),
                    vreinterpretq_u32_u64(vzip1q_u64(
                        vreinterpretq_u64_u32(ab_hi),
                        vreinterpretq_u64_u32(cd_hi),
                    )),
                    vreinterpretq_u32_u64(vzip2q_u64(
                        vreinterpretq_u64_u32(ab_hi),
                        vreinterpretq_u64_u32(cd_hi),
                    )),
                ];
                for (lane, row) in rows.into_iter().enumerate() {
                    let q = p.add(64 * lane + 16 * g);
                    let v = veorq_u8(vld1q_u8(q), vreinterpretq_u8_u32(row));
                    vst1q_u8(q, v);
                }
            }
        }
    }
}

/// XOR `data` in place with the ChaCha20 keystream starting at block
/// `counter` (RFC 8439 §2.4).
pub fn chacha20_xor(key: &[u8; 32], counter: u32, nonce: &[u8; 12], data: &mut [u8]) {
    let init = chacha_init(key, counter, nonce);
    let bulk = data.len() / 256 * 256;
    let (head, tail) = data.split_at_mut(bulk);
    #[cfg(target_arch = "x86_64")]
    let next = if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 is present (checked at run time).
        unsafe { chacha_xor_avx2(&init, counter, head) }
    } else {
        chacha_xor_bulk(&init, counter, head)
    };
    #[cfg(target_arch = "aarch64")]
    let next = {
        let mut c = counter;
        let (big, rest) = head.as_chunks_mut::<512>();
        for chunk in big {
            neon::xor8(&init, c, chunk);
            c = c.wrapping_add(8);
        }
        for chunk in rest.as_chunks_mut::<256>().0 {
            neon::xor4(&init, c, chunk);
            c = c.wrapping_add(4);
        }
        c
    };
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let next = chacha_xor_bulk(&init, counter, head);
    let mut counter = next;
    for chunk in tail.chunks_mut(64) {
        let ks = chacha20_block(key, counter, nonce);
        for (d, k) in chunk.iter_mut().zip(ks.iter()) {
            *d ^= k;
        }
        counter = counter.wrapping_add(1);
    }
}

// --- Poly1305 -------------------------------------------------------------------

fn le64(b: &[u8]) -> u64 {
    u64::from_le_bytes(b[..8].try_into().unwrap())
}

/// Incremental Poly1305 (RFC 8439 §2.5) in base 2⁶⁴: the accumulator is
/// two 64-bit words plus a few top bits, so one block costs four 64×64-bit
/// products (the layout OpenSSL's portable code uses).
pub struct Poly1305 {
    r: [u64; 2],
    pad: [u64; 2],
    h: [u64; 3],
    buf: [u8; 16],
    len: usize,
}

impl Poly1305 {
    pub fn new(key: &[u8; 32]) -> Self {
        Poly1305 {
            r: [
                le64(&key[0..]) & 0x0fff_fffc_0fff_ffff,
                le64(&key[8..]) & 0x0fff_fffc_0fff_fffc,
            ],
            pad: [le64(&key[16..]), le64(&key[24..])],
            h: [0; 3],
            buf: [0; 16],
            len: 0,
        }
    }

    /// Absorb whole 16-byte blocks; `padbit` is the 2¹²⁸ bit of each block.
    #[inline(always)]
    fn blocks(&mut self, data: &[u8], padbit: u64) {
        let [r0, r1] = self.r;
        // r1 is a multiple of 4, so 2¹³⁰ ≡ 5 folds into s1 = 5·r1/4 exactly
        let s1 = r1 + (r1 >> 2);
        let [mut h0, mut h1, mut h2] = self.h;
        let m = |a: u64, b: u64| a as u128 * b as u128;
        for blk in data.as_chunks::<16>().0 {
            // h += m
            let d0 = h0 as u128 + le64(&blk[..8]) as u128;
            h0 = d0 as u64;
            let d1 = h1 as u128 + (d0 >> 64) + le64(&blk[8..]) as u128;
            h1 = d1 as u64;
            h2 += (d1 >> 64) as u64 + padbit;
            // h *= r, partially reduced mod 2¹³⁰ − 5
            let d0 = m(h0, r0) + m(h1, s1);
            let mut d1 = m(h0, r1) + m(h1, r0) + (h2 * s1) as u128;
            h2 *= r0;
            h0 = d0 as u64;
            d1 += d0 >> 64;
            h1 = d1 as u64;
            h2 += (d1 >> 64) as u64;
            // fold the bits above 2¹³⁰: (h2 >> 2) · 5
            let c = (h2 >> 2) + (h2 & !3);
            h2 &= 3;
            let (t0, c0) = h0.overflowing_add(c);
            h0 = t0;
            let (t1, c1) = h1.overflowing_add(c0 as u64);
            h1 = t1;
            h2 += c1 as u64;
        }
        self.h = [h0, h1, h2];
    }

    pub fn update(&mut self, mut data: &[u8]) {
        if self.len > 0 {
            let take = (16 - self.len).min(data.len());
            self.buf[self.len..self.len + take].copy_from_slice(&data[..take]);
            self.len += take;
            data = &data[take..];
            if self.len < 16 {
                return; // still a partial block: nothing more to consume
            }
            let b = self.buf;
            self.blocks(&b, 1);
            self.len = 0;
        }
        let whole = data.len() / 16 * 16;
        self.blocks(&data[..whole], 1);
        let rest = &data[whole..];
        self.buf[..rest.len()].copy_from_slice(rest);
        self.len = rest.len();
    }

    /// Pad the input to a 16-byte boundary with zeros (AEAD construction).
    fn pad16(&mut self) {
        if self.len > 0 {
            self.update(&[0u8; 16][..16 - self.len]);
        }
    }

    pub fn finish(mut self) -> [u8; 16] {
        if self.len > 0 {
            let mut b = [0u8; 16];
            b[..self.len].copy_from_slice(&self.buf[..self.len]);
            b[self.len] = 1;
            self.blocks(&b, 0);
        }
        let [h0, h1, h2] = self.h;
        // g = h + 5; if it reaches 2¹³⁰ then h ≥ p and h − p = g mod 2¹³⁰
        let t = h0 as u128 + 5;
        let g0 = t as u64;
        let t = h1 as u128 + (t >> 64);
        let g1 = t as u64;
        let g2 = h2 + (t >> 64) as u64;
        let use_g = 0u64.wrapping_sub(g2 >> 2);
        let h0 = (h0 & !use_g) | (g0 & use_g);
        let h1 = (h1 & !use_g) | (g1 & use_g);
        // tag = (h + s) mod 2¹²⁸
        let t = h0 as u128 + self.pad[0] as u128;
        let lo = t as u64;
        let hi = (h1 as u128 + (t >> 64) + self.pad[1] as u128) as u64;
        let mut out = [0u8; 16];
        out[..8].copy_from_slice(&lo.to_le_bytes());
        out[8..].copy_from_slice(&hi.to_le_bytes());
        out
    }
}

pub fn poly1305(key: &[u8; 32], msg: &[u8]) -> [u8; 16] {
    let mut p = Poly1305::new(key);
    p.update(msg);
    p.finish()
}

// --- X25519 (RFC 7748) -------------------------------------------------------------

/// A field element of GF(2²⁵⁵ − 19) in five 51-bit limbs.
type Fe = [u64; 5];
const M51: u64 = (1 << 51) - 1;

fn fe_load(b: &[u8; 32]) -> Fe {
    let w = |i: usize| le64(&b[i..]);
    [
        w(0) & M51,
        (w(6) >> 3) & M51,
        (w(12) >> 6) & M51,
        (w(19) >> 1) & M51,
        (w(24) >> 12) & M51, // also drops bit 255, as RFC 7748 requires
    ]
}

fn fe_reduce(t: &mut Fe) {
    t[1] += t[0] >> 51;
    t[0] &= M51;
    t[2] += t[1] >> 51;
    t[1] &= M51;
    t[3] += t[2] >> 51;
    t[2] &= M51;
    t[4] += t[3] >> 51;
    t[3] &= M51;
    t[0] += 19 * (t[4] >> 51);
    t[4] &= M51;
}

/// The canonical 32-byte encoding (fully reduced mod p).
fn fe_store(mut t: Fe) -> [u8; 32] {
    fe_reduce(&mut t);
    fe_reduce(&mut t);
    // now 0 ≤ t < 2²⁵⁵ with carried limbs; add 19 and fold: t ≥ p wraps
    t[0] += 19;
    fe_reduce(&mut t);
    // t + 19 (mod 2²⁵⁵); subtract 19 by adding 2²⁵⁵ − 19 and dropping 2²⁵⁵
    t[0] += (1 << 51) - 19;
    t[1] += (1 << 51) - 1;
    t[2] += (1 << 51) - 1;
    t[3] += (1 << 51) - 1;
    t[4] += (1 << 51) - 1;
    t[1] += t[0] >> 51;
    t[0] &= M51;
    t[2] += t[1] >> 51;
    t[1] &= M51;
    t[3] += t[2] >> 51;
    t[2] &= M51;
    t[4] += t[3] >> 51;
    t[3] &= M51;
    t[4] &= M51;
    let words = [
        t[0] | (t[1] << 51),
        (t[1] >> 13) | (t[2] << 38),
        (t[2] >> 26) | (t[3] << 25),
        (t[3] >> 39) | (t[4] << 12),
    ];
    let mut out = [0u8; 32];
    for (o, w) in out.as_chunks_mut::<8>().0.iter_mut().zip(words) {
        *o = w.to_le_bytes();
    }
    out
}

fn fe_add(a: &Fe, b: &Fe) -> Fe {
    std::array::from_fn(|i| a[i] + b[i])
}

/// a − b, computed as a + 4p − b so limbs never go negative.
fn fe_sub(a: &Fe, b: &Fe) -> Fe {
    const P4: Fe = [
        0x1f_ffff_ffff_ffb4,
        0x1f_ffff_ffff_fffc,
        0x1f_ffff_ffff_fffc,
        0x1f_ffff_ffff_fffc,
        0x1f_ffff_ffff_fffc,
    ];
    let mut t: Fe = std::array::from_fn(|i| a[i] + P4[i] - b[i]);
    fe_reduce(&mut t);
    t
}

fn fe_mul(a: &Fe, b: &Fe) -> Fe {
    let m = |x: u64, y: u64| x as u128 * y as u128;
    let [a0, a1, a2, a3, a4] = *a;
    let [b0, b1, b2, b3, b4] = *b;
    let (b1, b2, b3, b4, b1_19, b2_19, b3_19, b4_19) =
        (b1, b2, b3, b4, b1 * 19, b2 * 19, b3 * 19, b4 * 19);
    let t0 = m(a0, b0) + m(a1, b4_19) + m(a2, b3_19) + m(a3, b2_19) + m(a4, b1_19);
    let t1 = m(a0, b1) + m(a1, b0) + m(a2, b4_19) + m(a3, b3_19) + m(a4, b2_19);
    let t2 = m(a0, b2) + m(a1, b1) + m(a2, b0) + m(a3, b4_19) + m(a4, b3_19);
    let t3 = m(a0, b3) + m(a1, b2) + m(a2, b1) + m(a3, b0) + m(a4, b4_19);
    let t4 = m(a0, b4) + m(a1, b3) + m(a2, b2) + m(a3, b1) + m(a4, b0);
    let mut r = [0u64; 5];
    let mut c: u128;
    r[0] = t0 as u64 & M51;
    c = t0 >> 51;
    let t1 = t1 + c;
    r[1] = t1 as u64 & M51;
    c = t1 >> 51;
    let t2 = t2 + c;
    r[2] = t2 as u64 & M51;
    c = t2 >> 51;
    let t3 = t3 + c;
    r[3] = t3 as u64 & M51;
    c = t3 >> 51;
    let t4 = t4 + c;
    r[4] = t4 as u64 & M51;
    c = t4 >> 51;
    r[0] += c as u64 * 19;
    r[1] += r[0] >> 51;
    r[0] &= M51;
    r
}

fn fe_sq_n(a: &Fe, n: usize) -> Fe {
    let mut t = *a;
    for _ in 0..n {
        t = fe_mul(&t, &t);
    }
    t
}

/// z^(p − 2) = z⁻¹ (Fermat), the usual 254-squaring addition chain.
fn fe_invert(z: &Fe) -> Fe {
    let z2 = fe_sq_n(z, 1);
    let z9 = fe_mul(&fe_sq_n(&z2, 2), z);
    let z11 = fe_mul(&z9, &z2);
    let z_5_0 = fe_mul(&fe_sq_n(&z11, 1), &z9);
    let z_10_0 = fe_mul(&fe_sq_n(&z_5_0, 5), &z_5_0);
    let z_20_0 = fe_mul(&fe_sq_n(&z_10_0, 10), &z_10_0);
    let z_40_0 = fe_mul(&fe_sq_n(&z_20_0, 20), &z_20_0);
    let z_50_0 = fe_mul(&fe_sq_n(&z_40_0, 10), &z_10_0);
    let z_100_0 = fe_mul(&fe_sq_n(&z_50_0, 50), &z_50_0);
    let z_200_0 = fe_mul(&fe_sq_n(&z_100_0, 100), &z_100_0);
    let z_250_0 = fe_mul(&fe_sq_n(&z_200_0, 50), &z_50_0);
    fe_mul(&fe_sq_n(&z_250_0, 5), &z11)
}

/// Constant-time conditional swap (`swap` is 0 or 1).
fn fe_cswap(swap: u64, a: &mut Fe, b: &mut Fe) {
    let mask = swap.wrapping_neg();
    for i in 0..5 {
        let t = mask & (a[i] ^ b[i]);
        a[i] ^= t;
        b[i] ^= t;
    }
}

/// The X25519 function (RFC 7748 §5): `scalar` · `u`, as a u-coordinate.
/// The Montgomery ladder runs a fixed sequence of operations for every
/// scalar (constant time).
pub fn x25519(scalar: &[u8; 32], u: &[u8; 32]) -> [u8; 32] {
    let mut k = *scalar;
    k[0] &= 248;
    k[31] &= 127;
    k[31] |= 64;
    let x1 = fe_load(u);
    let (mut x2, mut z2, mut x3, mut z3): (Fe, Fe, Fe, Fe) =
        ([1, 0, 0, 0, 0], [0; 5], x1, [1, 0, 0, 0, 0]);
    let a24: Fe = [121_665, 0, 0, 0, 0];
    let mut swap = 0u64;
    for t in (0..255).rev() {
        let bit = ((k[t / 8] >> (t % 8)) & 1) as u64;
        swap ^= bit;
        fe_cswap(swap, &mut x2, &mut x3);
        fe_cswap(swap, &mut z2, &mut z3);
        swap = bit;
        let a = fe_add(&x2, &z2);
        let aa = fe_mul(&a, &a);
        let b = fe_sub(&x2, &z2);
        let bb = fe_mul(&b, &b);
        let e = fe_sub(&aa, &bb);
        let c = fe_add(&x3, &z3);
        let d = fe_sub(&x3, &z3);
        let da = fe_mul(&d, &a);
        let cb = fe_mul(&c, &b);
        let s = fe_add(&da, &cb);
        x3 = fe_mul(&s, &s);
        let dif = fe_sub(&da, &cb);
        z3 = fe_mul(&x1, &fe_mul(&dif, &dif));
        x2 = fe_mul(&aa, &bb);
        z2 = fe_mul(&e, &fe_add(&aa, &fe_mul(&a24, &e)));
    }
    fe_cswap(swap, &mut x2, &mut x3);
    fe_cswap(swap, &mut z2, &mut z3);
    k.fill(0);
    fe_store(fe_mul(&x2, &fe_invert(&z2)))
}

/// The X25519 base point (u = 9).
pub const X25519_BASE: [u8; 32] = {
    let mut b = [0u8; 32];
    b[0] = 9;
    b
};

// --- ChaCha20-Poly1305 AEAD (RFC 8439 §2.8) -----------------------------------------

fn aead_tag(key: &[u8; 32], nonce: &[u8; 12], aad: &[u8], ct: &[u8]) -> [u8; 16] {
    let otk: [u8; 32] = chacha20_block(key, 0, nonce)[..32].try_into().unwrap();
    let mut p = Poly1305::new(&otk);
    p.update(aad);
    p.pad16();
    p.update(ct);
    p.pad16();
    p.update(&(aad.len() as u64).to_le_bytes());
    p.update(&(ct.len() as u64).to_le_bytes());
    p.finish()
}

/// Encrypt `data` in place; returns the 16-byte tag.
pub fn seal(key: &[u8; 32], nonce: &[u8; 12], aad: &[u8], data: &mut [u8]) -> [u8; 16] {
    chacha20_xor(key, 1, nonce, data);
    aead_tag(key, nonce, aad, data)
}

/// Verify `tag` and decrypt `data` in place; on failure `data` is left
/// encrypted and `false` is returned.
pub fn open(key: &[u8; 32], nonce: &[u8; 12], aad: &[u8], data: &mut [u8], tag: &[u8]) -> bool {
    let want = aead_tag(key, nonce, aad, data);
    if !super::auth::ct_eq(&want, tag) {
        return false;
    }
    chacha20_xor(key, 1, nonce, data);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(s: &str) -> Vec<u8> {
        let s: String = s.chars().filter(|c| c.is_ascii_hexdigit()).collect();
        (0..s.len() / 2)
            .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
            .collect()
    }

    #[test]
    fn sha256_hmac_hkdf_match_the_standard_vectors() {
        // FIPS 180-4 examples
        assert_eq!(
            sha256(b"").to_vec(),
            hex("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
        );
        assert_eq!(
            sha256(b"abc").to_vec(),
            hex("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        );
        assert_eq!(
            sha256(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq").to_vec(),
            hex("248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1")
        );
        // incremental updates across block boundaries agree with one shot
        let msg: Vec<u8> = (0..1000u32).map(|i| (i * 7) as u8).collect();
        let mut inc = Sha256::default();
        for c in msg.chunks(37) {
            inc.update(c);
        }
        assert_eq!(inc.finish(), sha256(&msg));
        for step in [1usize, 5, 63, 65] {
            let mut inc = Sha256::default();
            for c in msg.chunks(step) {
                inc.update(c);
            }
            assert_eq!(inc.finish(), sha256(&msg), "step {step}");
        }
        // RFC 4231 test case 2
        assert_eq!(
            hmac_sha256(b"Jefe", &[b"what do ya want for nothing?"]).to_vec(),
            hex("5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843")
        );
        // RFC 5869 test case 1
        let prk = hkdf_extract(&hex("000102030405060708090a0b0c"), &[0x0b; 22]);
        assert_eq!(
            prk.to_vec(),
            hex("077709362c2e32df0ddc3f0dc47bba6390b6c73bb50f9c3122ec844ad7c2b3e5")
        );
        let mut okm = [0u8; 42];
        hkdf_expand(&prk, &hex("f0f1f2f3f4f5f6f7f8f9"), &mut okm);
        assert_eq!(
            okm.to_vec(),
            hex("3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865")
        );
    }

    #[test]
    fn chacha20_poly1305_match_rfc_8439() {
        let key: [u8; 32] = std::array::from_fn(|i| i as u8);
        // §2.3.2 block function
        let nonce: [u8; 12] = hex("000000090000004a00000000").try_into().unwrap();
        let b = chacha20_block(&key, 1, &nonce);
        assert_eq!(b[..16].to_vec(), hex("10f1e7e4d13b5915500fdd1fa32071c4"));
        assert_eq!(b[48..].to_vec(), hex("b5129cd1de164eb9cbd083e8a2503c4e"));
        // §2.4.2 encryption
        let nonce: [u8; 12] = hex("000000000000004a00000000").try_into().unwrap();
        let mut text = b"Ladies and Gentlemen of the class of '99: If I could offer you only one tip for the future, sunscreen would be it.".to_vec();
        chacha20_xor(&key, 1, &nonce, &mut text);
        assert_eq!(text[..16].to_vec(), hex("6e2e359a2568f98041ba0728dd0d6981"));
        assert_eq!(
            text[text.len() - 10..].to_vec(),
            hex("b40b8eedf2785e42874d")
        );
        // §2.5.2 Poly1305
        let pk: [u8; 32] = hex("85d6be7857556d337f4452fe42d506a80103808afb0db2fd4abff6af4149f51b")
            .try_into()
            .unwrap();
        assert_eq!(
            poly1305(&pk, b"Cryptographic Forum Research Group").to_vec(),
            hex("a8061dc1305136c6c22b8baf0c0127a9")
        );
        // incremental updates in uneven pieces (partial blocks) agree
        let msg: Vec<u8> = (0..300u32).map(|i| (i * 13) as u8).collect();
        for step in [1usize, 3, 7, 15, 17, 31] {
            let mut p = Poly1305::new(&pk);
            for c in msg.chunks(step) {
                p.update(c);
            }
            assert_eq!(p.finish(), poly1305(&pk, &msg), "step {step}");
        }
        // §2.8.2 AEAD
        let akey: [u8; 32] =
            hex("808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9f")
                .try_into()
                .unwrap();
        let anonce: [u8; 12] = hex("070000004041424344454647").try_into().unwrap();
        let aad = hex("50515253c0c1c2c3c4c5c6c7");
        let mut pt = b"Ladies and Gentlemen of the class of '99: If I could offer you only one tip for the future, sunscreen would be it.".to_vec();
        let orig = pt.clone();
        let tag = seal(&akey, &anonce, &aad, &mut pt);
        assert_eq!(pt[..16].to_vec(), hex("d31a8d34648e60db7b86afbc53ef7ec2"));
        assert_eq!(tag.to_vec(), hex("1ae10b594f09e26a7e902ecbd0600691"));
        // round trip, and any modification is rejected
        let mut ct = pt.clone();
        assert!(open(&akey, &anonce, &aad, &mut ct, &tag));
        assert_eq!(ct, orig);
        let mut bad = pt.clone();
        bad[5] ^= 1;
        assert!(!open(&akey, &anonce, &aad, &mut bad, &tag));
        let mut ct = pt.clone();
        assert!(!open(&akey, &anonce, b"other aad", &mut ct, &tag));
    }
    #[test]
    fn multi_block_chacha20_matches_the_one_block_reference() {
        // lengths across the 4- and 8-block paths, the single-block tail
        // and a counter that wraps inside a lane group
        let key: [u8; 32] = std::array::from_fn(|i| (i * 11 + 3) as u8);
        let nonce: [u8; 12] = std::array::from_fn(|i| (i * 5 + 1) as u8);
        for (len, counter) in [
            (255usize, 1u32),
            (256, 1),
            (511, 7),
            (512, 1),
            (777, 0),
            (1024, u32::MAX - 5),
            (4096 + 100, 3),
        ] {
            let msg: Vec<u8> = (0..len).map(|i| (i * 31) as u8).collect();
            let mut fast = msg.clone();
            chacha20_xor(&key, counter, &nonce, &mut fast);
            let mut slow = msg.clone();
            for (b, chunk) in slow.chunks_mut(64).enumerate() {
                let ks = chacha20_block(&key, counter.wrapping_add(b as u32), &nonce);
                for (d, k) in chunk.iter_mut().zip(ks) {
                    *d ^= k;
                }
            }
            assert_eq!(fast, slow, "len {len} counter {counter}");
        }
    }

    #[test]
    fn x25519_matches_rfc_7748() {
        let x = |k: &str, u: &str| {
            let k: [u8; 32] = hex(k).try_into().unwrap();
            let u: [u8; 32] = hex(u).try_into().unwrap();
            x25519(&k, &u).to_vec()
        };
        // §5.2 test vectors (the second has a u-coordinate with bit 255 set)
        assert_eq!(
            x(
                "a546e36bf0527c9d3b16154b82465edd62144c0ac1fc5a18506a2244ba449ac4",
                "e6db6867583030db3594c1a424b15f7c726624ec26b3353b10a903a6d0ab1c4c"
            ),
            hex("c3da55379de9c6908e94ea4df28d084f32eccf03491c71f754b4075577a28552")
        );
        assert_eq!(
            x(
                "4b66e9d4d1b4673c5ad22691957d6af5c11b6421e0ea01d42ca4169e7918ba0d",
                "e5210f12786811d3f4b7959d0538ae2c31dbe7106fc03c3efc4cd549c715a493"
            ),
            hex("95cbde9476e8907d7aade45cb4b873f88b595a68799fa152e6f8f7647aac7957")
        );
        // §5.2 iterated: k = u = 9, then k ← X25519(k, u), u ← old k
        let (mut k, mut u) = (X25519_BASE, X25519_BASE);
        for i in 1..=1000 {
            let next = x25519(&k, &u);
            u = k;
            k = next;
            if i == 1 {
                assert_eq!(
                    k.to_vec(),
                    hex("422c8e7a6227d7bca1350b3e2bb7279f7897b87bb6854b783c60e80311ae3079")
                );
            }
        }
        assert_eq!(
            k.to_vec(),
            hex("684cf59ba83309552800ef566f2f4d3c1c3887c49360e3875f2eb94d99532c51")
        );
        // §6.1 Diffie–Hellman
        let a: [u8; 32] = hex("77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a")
            .try_into()
            .unwrap();
        let b: [u8; 32] = hex("5dab087e624a8a4b79e17f8b83800ee66f3bb1292618b6fd1c2f8b27ff88e0eb")
            .try_into()
            .unwrap();
        let pa = x25519(&a, &X25519_BASE);
        let pb = x25519(&b, &X25519_BASE);
        assert_eq!(
            pa.to_vec(),
            hex("8520f0098930a754748b7ddcb43ef75a0dbf3a0d26381af4eba4a98eaa9b4e6a")
        );
        assert_eq!(
            pb.to_vec(),
            hex("de9edb7d7b7dc1b4d35b61c2ece435373f8343c85b78674dadfc7e146f882b4f")
        );
        let shared = hex("4a5d9d5ba4ce2de1728e3bf480350f25e07e21c947d19e3376f09b3c1e161742");
        assert_eq!(x25519(&a, &pb).to_vec(), shared);
        assert_eq!(x25519(&b, &pa).to_vec(), shared);
    }

    /// 400 deterministic cases (random and all-0xff keys, messages and
    /// u-coordinates; lengths across every block, lane and segment edge)
    /// for the AEAD, Poly1305 and X25519. Their output was checked case by
    /// case against OpenSSL (Python `cryptography`) with 0 mismatches; the
    /// digest freezes it, so every SIMD path on every CI platform must
    /// reproduce exactly the same bytes.
    #[test]
    fn outputs_match_the_openssl_verified_corpus() {
        fn hexs(b: &[u8]) -> String {
            b.iter().map(|x| format!("{x:02x}")).collect()
        }
        fn fill(rnd: &mut impl FnMut() -> u64, n: usize, edge: bool) -> Vec<u8> {
            (0..n)
                .map(|_| if edge { 0xff } else { rnd() as u8 })
                .collect()
        }
        let mut st = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = move || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            st
        };
        let lens = [
            0usize, 1, 15, 16, 17, 63, 64, 65, 255, 256, 257, 511, 512, 513, 767, 1023, 1024, 4095,
            65535, 65536, 65537, 200_000,
        ];
        let mut out = String::new();
        for case in 0..400 {
            let edge = case % 10 == 0;
            let key: [u8; 32] = fill(&mut rnd, 32, edge).try_into().unwrap();
            let nonce: [u8; 12] = fill(&mut rnd, 12, false).try_into().unwrap();
            let aad = fill(&mut rnd, case % 40, false);
            let n = if case < lens.len() * 2 {
                lens[case % lens.len()]
            } else {
                (rnd() % 3000) as usize
            };
            let msg = fill(&mut rnd, n, case % 7 == 0);
            let mut ct = msg.clone();
            let tag = seal(&key, &nonce, &aad, &mut ct);
            let ptag = poly1305(&key, &msg);
            let k2: [u8; 32] = fill(&mut rnd, 32, false).try_into().unwrap();
            let u: [u8; 32] = fill(&mut rnd, 32, edge).try_into().unwrap();
            let fields = [
                &key[..],
                &nonce,
                &aad,
                &msg,
                &ct,
                &tag,
                &ptag,
                &k2,
                &u,
                &x25519(&k2, &u),
            ];
            out += &fields.map(hexs).join(" ");
            out.push('\n');
        }
        assert_eq!(
            hexs(&sha256(out.as_bytes())),
            "585bae20329645c03618daff0e1e382717e0abcfca11b76a63af51013193f16f"
        );
    }
}
