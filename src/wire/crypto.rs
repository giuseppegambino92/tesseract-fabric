//! Cryptographic primitives for authenticated encryption of node traffic,
//! implemented in-tree (zero external crates) and checked against the
//! published test vectors (FIPS 180-4, RFC 4231, RFC 5869, RFC 8439) and,
//! during development, against an independent implementation (OpenSSL via
//! Python `cryptography`):
//!
//! * SHA-256, HMAC-SHA256, HKDF-SHA256 — key derivation and handshake MACs;
//! * ChaCha20, Poly1305 and the ChaCha20-Poly1305 AEAD (RFC 8439) — the
//!   record layer of encrypted `TesseractWire` connections.
//!
//! Tag comparisons are constant-time. These are straightforward portable
//! implementations, not hardened against every side channel (e.g. cache
//! timing on exotic CPUs); ChaCha20 and Poly1305 use only additions,
//! rotations, XORs and multiplications, which are constant-time on the
//! targets this project runs on.

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
    let mut init = [0u32; 16];
    init[..4].copy_from_slice(&[0x6170_7865, 0x3320_646e, 0x7962_2d32, 0x6b20_6574]);
    for i in 0..8 {
        init[4 + i] = u32::from_le_bytes(key[4 * i..4 * i + 4].try_into().unwrap());
    }
    init[12] = counter;
    for i in 0..3 {
        init[13 + i] = u32::from_le_bytes(nonce[4 * i..4 * i + 4].try_into().unwrap());
    }
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

/// XOR `data` in place with the ChaCha20 keystream starting at block
/// `counter` (RFC 8439 §2.4).
pub fn chacha20_xor(key: &[u8; 32], mut counter: u32, nonce: &[u8; 12], data: &mut [u8]) {
    for chunk in data.chunks_mut(64) {
        let ks = chacha20_block(key, counter, nonce);
        for (d, k) in chunk.iter_mut().zip(ks.iter()) {
            *d ^= k;
        }
        counter = counter.wrapping_add(1);
    }
}

// --- Poly1305 -------------------------------------------------------------------

/// Incremental Poly1305 (RFC 8439 §2.5), 26-bit limbs.
pub struct Poly1305 {
    r: [u64; 5],
    s: [u32; 4],
    h: [u64; 5],
    buf: [u8; 16],
    len: usize,
}

fn le32(b: &[u8]) -> u64 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u64
}

impl Poly1305 {
    pub fn new(key: &[u8; 32]) -> Self {
        let k = key;
        Poly1305 {
            r: [
                le32(&k[0..]) & 0x3ff_ffff,
                (le32(&k[3..]) >> 2) & 0x3ff_ff03,
                (le32(&k[6..]) >> 4) & 0x3ff_c0ff,
                (le32(&k[9..]) >> 6) & 0x3f0_3fff,
                (le32(&k[12..]) >> 8) & 0x00f_ffff,
            ],
            s: std::array::from_fn(|i| {
                u32::from_le_bytes(k[16 + 4 * i..20 + 4 * i].try_into().unwrap())
            }),
            h: [0; 5],
            buf: [0; 16],
            len: 0,
        }
    }

    fn block(&mut self, m: &[u8; 16], hibit: u64) {
        let [r0, r1, r2, r3, r4] = self.r;
        let (s1, s2, s3, s4) = (r1 * 5, r2 * 5, r3 * 5, r4 * 5);
        let h = &mut self.h;
        h[0] += le32(&m[0..]) & 0x3ff_ffff;
        h[1] += (le32(&m[3..]) >> 2) & 0x3ff_ffff;
        h[2] += (le32(&m[6..]) >> 4) & 0x3ff_ffff;
        h[3] += (le32(&m[9..]) >> 6) & 0x3ff_ffff;
        h[4] += (le32(&m[12..]) >> 8) | hibit;
        let d0 = h[0] * r0 + h[1] * s4 + h[2] * s3 + h[3] * s2 + h[4] * s1;
        let mut d1 = h[0] * r1 + h[1] * r0 + h[2] * s4 + h[3] * s3 + h[4] * s2;
        let mut d2 = h[0] * r2 + h[1] * r1 + h[2] * r0 + h[3] * s4 + h[4] * s3;
        let mut d3 = h[0] * r3 + h[1] * r2 + h[2] * r1 + h[3] * r0 + h[4] * s4;
        let mut d4 = h[0] * r4 + h[1] * r3 + h[2] * r2 + h[3] * r1 + h[4] * r0;
        let mut c = d0 >> 26;
        h[0] = d0 & 0x3ff_ffff;
        d1 += c;
        c = d1 >> 26;
        h[1] = d1 & 0x3ff_ffff;
        d2 += c;
        c = d2 >> 26;
        h[2] = d2 & 0x3ff_ffff;
        d3 += c;
        c = d3 >> 26;
        h[3] = d3 & 0x3ff_ffff;
        d4 += c;
        c = d4 >> 26;
        h[4] = d4 & 0x3ff_ffff;
        h[0] += c * 5;
        c = h[0] >> 26;
        h[0] &= 0x3ff_ffff;
        h[1] += c;
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
            self.block(&b, 1 << 24);
            self.len = 0;
        }
        while data.len() >= 16 {
            self.block(data[..16].try_into().unwrap(), 1 << 24);
            data = &data[16..];
        }
        self.buf[..data.len()].copy_from_slice(data);
        self.len = data.len();
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
            self.block(&b, 0);
        }
        let mask = 0x3ff_ffffu64;
        let mut h = self.h;
        let mut c = h[1] >> 26;
        h[1] &= mask;
        h[2] += c;
        c = h[2] >> 26;
        h[2] &= mask;
        h[3] += c;
        c = h[3] >> 26;
        h[3] &= mask;
        h[4] += c;
        c = h[4] >> 26;
        h[4] &= mask;
        h[0] += c * 5;
        c = h[0] >> 26;
        h[0] &= mask;
        h[1] += c;
        // g = h + 5 − 2¹³⁰; keep g if non-negative (h ≥ p), else h
        let mut g = [0u64; 5];
        g[0] = h[0] + 5;
        c = g[0] >> 26;
        g[0] &= mask;
        g[1] = h[1] + c;
        c = g[1] >> 26;
        g[1] &= mask;
        g[2] = h[2] + c;
        c = g[2] >> 26;
        g[2] &= mask;
        g[3] = h[3] + c;
        c = g[3] >> 26;
        g[3] &= mask;
        g[4] = h[4].wrapping_add(c).wrapping_sub(1 << 26);
        let keep_g = (g[4] >> 63).wrapping_sub(1);
        for i in 0..5 {
            h[i] = (h[i] & !keep_g) | (g[i] & keep_g);
        }
        let w = [
            (h[0] | (h[1] << 26)) & 0xffff_ffff,
            ((h[1] >> 6) | (h[2] << 20)) & 0xffff_ffff,
            ((h[2] >> 12) | (h[3] << 14)) & 0xffff_ffff,
            ((h[3] >> 18) | (h[4] << 8)) & 0xffff_ffff,
        ];
        let mut out = [0u8; 16];
        let mut f = 0u64;
        for i in 0..4 {
            f = w[i] + self.s[i] as u64 + (f >> 32);
            out[4 * i..4 * i + 4].copy_from_slice(&(f as u32).to_le_bytes());
        }
        out
    }
}

pub fn poly1305(key: &[u8; 32], msg: &[u8]) -> [u8; 16] {
    let mut p = Poly1305::new(key);
    p.update(msg);
    p.finish()
}

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
}
