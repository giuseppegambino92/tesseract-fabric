//! Holographic Residual Subspace Compression with Error-Feedback (HRQ-EF).
//!
//! Sender, per step t:
//!   Δh_t = h_t − h_prev                      (temporal / inter-shard delta)
//!   u_t  = Δh_t + γ·e_{t−1}                  (error-feedback injection)
//!   outliers: channels with |u_i| > τ·rms(u), at most ⌈k_frac·d⌉, sent exact
//!   v_t  = H·D·pad(u_t − outliers)           (randomised FWHT spreads energy)
//!   codes = per-64-block signed mid-rise lattice, b ∈ {2,3,4,8} bits, f16 scale
//!   û_t  = unpad(D·H·Dequant(codes)) + outliers
//!   e_t  = u_t − û_t ;  h_prev ← h_t
//! Receiver: ĥ_t = ĥ_{t−1} + û_t.
//!
//! With γ→1 the reconstruction telescopes: Σ û = Σ Δh − e_t, so the receiver's
//! drift stays bounded by one step's error instead of random-walking.

use super::fwht::HadamardPlan;
use crate::tensor::{f16_to_f32, f32_to_f16};
use crate::wire::protocol::{ByteReader, ByteWriter};

#[derive(Clone, Debug, PartialEq)]
pub struct HoloConfig {
    /// Bits per bulk coordinate (2, 3, 4 or 8).
    pub bits: u8,
    /// Coordinates per scale block.
    pub block: usize,
    /// Max fraction of channels escaped as exact outliers.
    pub outlier_frac: f32,
    /// Escape threshold in units of rms(u).
    pub outlier_sigma: f32,
    /// Error-feedback decay γ (0 disables error feedback).
    pub gamma: f32,
    /// Bits used for the first (key) frame of a stream.
    pub keyframe_bits: u8,
    /// Hadamard tile; `None` = full power-of-two transform.
    pub hadamard_block: Option<usize>,
    pub seed: u64,
}

impl Default for HoloConfig {
    fn default() -> Self {
        HoloConfig {
            bits: 2,
            block: 64,
            outlier_frac: 0.01,
            outlier_sigma: 4.0,
            gamma: 0.95,
            keyframe_bits: 8,
            hadamard_block: None,
            seed: 0x401_06AA,
        }
    }
}

impl HoloConfig {
    pub fn plan(&self, dim: usize) -> HadamardPlan {
        match self.hadamard_block {
            Some(b) => HadamardPlan::blocked(dim, b, self.seed),
            None => HadamardPlan::full(dim, self.seed),
        }
    }
}

/// MSE-optimal uniform step (in units of block RMS) for a Gaussian source.
pub(crate) fn step_factor(bits: u8) -> f32 {
    match bits {
        1 => 1.596,
        2 => 0.9957,
        3 => 0.5860,
        4 => 0.3352,
        5 => 0.1881,
        6 => 0.1041,
        _ => 0.0,
    }
}

/// One compressed vector on the wire.
#[derive(Clone, Debug, PartialEq)]
pub struct HolographicFrame {
    pub dim: u32,
    pub bits: u8,
    /// 0 = delta vs previous, 1 = key/intra frame.
    pub mode: u8,
    pub hadamard_log2: u8,
    pub outliers: Vec<(u16, u16)>,
    pub scales: Vec<u16>,
    pub codes: Vec<u8>,
}

impl HolographicFrame {
    pub fn encode(&self, w: &mut ByteWriter) {
        w.u32(self.dim);
        w.u8(self.bits);
        w.u8(self.mode);
        w.u8(self.hadamard_log2);
        w.u16(self.outliers.len() as u16);
        for &(i, v) in &self.outliers {
            w.u16(i);
            w.u16(v);
        }
        for &s in &self.scales {
            w.u16(s);
        }
        w.bytes(&self.codes);
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut w = ByteWriter::with_capacity(self.wire_bytes());
        self.encode(&mut w);
        w.into_inner()
    }

    /// Decode; `padded` and `block` must match the sender's plan.
    pub fn decode(r: &mut ByteReader, padded_of: &dyn Fn(usize) -> usize, block: usize) -> Self {
        let dim = r.u32();
        let bits = r.u8();
        let mode = r.u8();
        let hadamard_log2 = r.u8();
        let n_out = r.u16() as usize;
        let outliers = (0..n_out).map(|_| (r.u16(), r.u16())).collect();
        let padded = padded_of(dim as usize);
        let n_blocks = padded.div_ceil(block);
        let scales = (0..n_blocks).map(|_| r.u16()).collect();
        let n_code_bytes = (padded * bits as usize).div_ceil(8);
        let codes = r.bytes(n_code_bytes).to_vec();
        HolographicFrame {
            dim,
            bits,
            mode,
            hadamard_log2,
            outliers,
            scales,
            codes,
        }
    }

    pub fn wire_bytes(&self) -> usize {
        4 + 3 + 2 + self.outliers.len() * 4 + self.scales.len() * 2 + self.codes.len()
    }
}

// --- bit packing -------------------------------------------------------------

pub(crate) fn pack_codes(codes: &[u8], bits: u8) -> Vec<u8> {
    let mut out = vec![0u8; (codes.len() * bits as usize).div_ceil(8)];
    let mut bitpos = 0usize;
    for &c in codes {
        let mut v = c as u32;
        let mut remaining = bits as usize;
        while remaining > 0 {
            let byte = bitpos / 8;
            let off = bitpos % 8;
            let take = remaining.min(8 - off);
            out[byte] |= ((v & ((1 << take) - 1)) << off) as u8;
            v >>= take;
            remaining -= take;
            bitpos += take;
        }
    }
    out
}

pub(crate) fn unpack_codes(packed: &[u8], bits: u8, n: usize, out: &mut Vec<u8>) {
    out.clear();
    let mut bitpos = 0usize;
    for _ in 0..n {
        let mut v = 0u32;
        let mut got = 0usize;
        while got < bits as usize {
            let byte = bitpos / 8;
            let off = bitpos % 8;
            let take = (bits as usize - got).min(8 - off);
            let chunk = (packed[byte] as u32 >> off) & ((1 << take) - 1);
            v |= chunk << got;
            got += take;
            bitpos += take;
        }
        out.push(v as u8);
    }
}

// --- stateless vector codec ---------------------------------------------------

/// Stateless core: quantise `u` and return (frame, reconstruction û).
pub struct VectorCodec {
    pub plan: HadamardPlan,
    pub cfg: HoloConfig,
    rot: Vec<f32>,
    bulk: Vec<f32>,
    codes_scratch: Vec<u8>,
}

impl VectorCodec {
    pub fn new(dim: usize, cfg: HoloConfig) -> Self {
        let plan = cfg.plan(dim);
        let padded = plan.padded;
        VectorCodec {
            plan,
            cfg,
            rot: vec![0.0; padded],
            bulk: vec![0.0; dim],
            codes_scratch: Vec::new(),
        }
    }

    pub fn dim(&self) -> usize {
        self.plan.dim
    }

    /// Encode `u` at `bits`; writes the reconstruction into `u_hat`.
    pub fn encode(&mut self, u: &[f32], bits: u8, mode: u8, u_hat: &mut [f32]) -> HolographicFrame {
        let d = self.plan.dim;
        let block = self.cfg.block;
        // 1. outlier escape (channel domain, before rotation)
        let rms = (crate::tensor::dot(u, u) / d as f32).sqrt();
        let thr = self.cfg.outlier_sigma * rms;
        let kmax = ((self.cfg.outlier_frac * d as f32).ceil() as usize).min(u16::MAX as usize);
        let mut cand: Vec<(usize, f32)> = if thr > 0.0 && kmax > 0 {
            u.iter()
                .enumerate()
                .filter(|(_, v)| v.abs() > thr)
                .map(|(i, v)| (i, *v))
                .collect()
        } else {
            Vec::new()
        };
        if cand.len() > kmax {
            cand.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
            cand.truncate(kmax);
        }
        cand.sort_by_key(|c| c.0);
        self.bulk.copy_from_slice(u);
        let mut outliers = Vec::with_capacity(cand.len());
        for &(i, v) in &cand {
            self.bulk[i] = 0.0;
            outliers.push((i as u16, f32_to_f16(v)));
        }
        // 2. randomised Hadamard rotation
        self.plan.forward(&self.bulk, &mut self.rot);
        // 3. block lattice quantisation
        let padded = self.plan.padded;
        let levels = 1u32 << bits;
        let half = (levels / 2) as f32;
        let mut scales = Vec::with_capacity(padded.div_ceil(block));
        self.codes_scratch.clear();
        for blk in self.rot.chunks_mut(block) {
            let step = if bits <= 6 {
                let brms = (blk.iter().map(|v| v * v).sum::<f32>() / blk.len() as f32).sqrt();
                step_factor(bits) * brms
            } else {
                let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                amax / (half - 0.5)
            };
            let sh = f32_to_f16(step);
            let step_q = f16_to_f32(sh);
            scales.push(sh);
            let inv = if step_q > 0.0 { 1.0 / step_q } else { 0.0 };
            for v in blk.iter_mut() {
                let c = ((*v * inv).floor() + half).clamp(0.0, (levels - 1) as f32);
                self.codes_scratch.push(c as u8);
                *v = (c - half + 0.5) * step_q; // dequantised in place
            }
        }
        let codes = pack_codes(&self.codes_scratch, bits);
        // 4. reconstruction û = unpad(D·H·v̂) + outliers
        self.plan.inverse(&mut self.rot, u_hat);
        for &(i, v) in &outliers {
            u_hat[i as usize] = f16_to_f32(v);
        }
        HolographicFrame {
            dim: d as u32,
            bits,
            mode,
            hadamard_log2: self.plan.log2_block(),
            outliers,
            scales,
            codes,
        }
    }

    /// Decode a frame into û.
    pub fn decode(&mut self, f: &HolographicFrame, u_hat: &mut [f32]) {
        let padded = self.plan.padded;
        let block = self.cfg.block;
        unpack_codes(&f.codes, f.bits, padded, &mut self.codes_scratch);
        let half = ((1u32 << f.bits) / 2) as f32;
        for (b, blk) in self.rot.chunks_mut(block).enumerate() {
            let step = f16_to_f32(f.scales[b]);
            for (j, v) in blk.iter_mut().enumerate() {
                *v = (self.codes_scratch[b * block + j] as f32 - half + 0.5) * step;
            }
        }
        self.plan.inverse(&mut self.rot, u_hat);
        for &(i, v) in &f.outliers {
            u_hat[i as usize] = f16_to_f32(v);
        }
    }

    pub fn decode_bytes(&mut self, r: &mut ByteReader, u_hat: &mut [f32]) -> HolographicFrame {
        let padded = self.plan.padded;
        let f = HolographicFrame::decode(r, &|_| padded, self.cfg.block);
        self.decode(&f, u_hat);
        f
    }
}

// --- stateful temporal stream codec (sender + receiver) -----------------------

pub struct HolographicCompressor {
    pub codec: VectorCodec,
    pub h_prev: Vec<f32>,
    pub err: Vec<f32>,
    pub steps: u64,
    u: Vec<f32>,
    u_hat: Vec<f32>,
}

impl HolographicCompressor {
    pub fn new(dim: usize, cfg: HoloConfig) -> Self {
        HolographicCompressor {
            codec: VectorCodec::new(dim, cfg),
            h_prev: vec![0.0; dim],
            err: vec![0.0; dim],
            steps: 0,
            u: vec![0.0; dim],
            u_hat: vec![0.0; dim],
        }
    }

    pub fn compress(&mut self, h: &[f32]) -> HolographicFrame {
        let key = self.steps == 0;
        let gamma = self.codec.cfg.gamma;
        for i in 0..h.len() {
            self.u[i] = h[i] - self.h_prev[i] + gamma * self.err[i];
        }
        let bits = if key {
            self.codec.cfg.keyframe_bits
        } else {
            self.codec.cfg.bits
        };
        let u = std::mem::take(&mut self.u);
        let mut u_hat = std::mem::take(&mut self.u_hat);
        let f = self.codec.encode(&u, bits, key as u8, &mut u_hat);
        for i in 0..h.len() {
            self.err[i] = u[i] - u_hat[i];
        }
        self.u = u;
        self.u_hat = u_hat;
        self.h_prev.copy_from_slice(h);
        self.steps += 1;
        f
    }
}

pub struct HolographicDecompressor {
    pub codec: VectorCodec,
    pub h_hat: Vec<f32>,
    u_hat: Vec<f32>,
}

impl HolographicDecompressor {
    pub fn new(dim: usize, cfg: HoloConfig) -> Self {
        HolographicDecompressor {
            codec: VectorCodec::new(dim, cfg),
            h_hat: vec![0.0; dim],
            u_hat: vec![0.0; dim],
        }
    }

    pub fn decompress(&mut self, f: &HolographicFrame) -> &[f32] {
        let mut u_hat = std::mem::take(&mut self.u_hat);
        self.codec.decode(f, &mut u_hat);
        for (h, u) in self.h_hat.iter_mut().zip(&u_hat) {
            *h += u;
        }
        self.u_hat = u_hat;
        &self.h_hat
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Rng;

    #[test]
    fn pack_roundtrip() {
        let mut rng = Rng::new(1);
        for bits in [2u8, 3, 4, 8] {
            let codes: Vec<u8> = (0..301)
                .map(|_| (rng.next_u64() % (1 << bits)) as u8)
                .collect();
            let p = pack_codes(&codes, bits);
            let mut out = Vec::new();
            unpack_codes(&p, bits, codes.len(), &mut out);
            assert_eq!(out, codes);
        }
    }

    #[test]
    fn codec_frame_roundtrip_bytes() {
        let mut rng = Rng::new(5);
        let x: Vec<f32> = (0..512).map(|_| rng.normal()).collect();
        let mut c = VectorCodec::new(
            512,
            HoloConfig {
                bits: 3,
                ..Default::default()
            },
        );
        let mut uh = vec![0.0; 512];
        let f = c.encode(&x, 3, 1, &mut uh);
        let bytes = f.to_bytes();
        assert_eq!(bytes.len(), f.wire_bytes());
        let mut d = VectorCodec::new(
            512,
            HoloConfig {
                bits: 3,
                ..Default::default()
            },
        );
        let mut out = vec![0.0; 512];
        d.decode_bytes(&mut ByteReader::new(&bytes), &mut out);
        assert_eq!(out, uh);
        assert!(crate::tensor::cosine(&x, &out) > 0.97);
    }
}
