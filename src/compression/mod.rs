//! Activation compression for inter-shard hops.
//!
//! * `holographic` — HRQ-EF temporal stream codec (delta + error feedback +
//!   randomised FWHT + outlier escape + 2/3/4/8-bit block lattice).
//! * `TreeActivationCodec` — codes a whole speculative-tree activation matrix
//!   in one payload. Each row is coded either *intra* or *inter* (closed-loop
//!   delta against its parent row's reconstruction, which the receiver has
//!   already decoded because parents precede children), whichever has less
//!   energy — the same decision a video codec makes per macroblock.

pub mod fwht;
pub mod holographic;
pub mod klt;

pub use holographic::{
    HoloConfig, HolographicCompressor, HolographicDecompressor, HolographicFrame, VectorCodec,
};

use crate::tensor::Tensor2D;
use crate::wire::protocol::{ByteReader, ByteWriter};

pub struct TreeActivationCodec {
    pub codec: VectorCodec,
    pub inter_rows: u64,
    pub intra_rows: u64,
}

impl TreeActivationCodec {
    pub fn new(dim: usize, cfg: HoloConfig) -> Self {
        TreeActivationCodec {
            codec: VectorCodec::new(dim, cfg),
            inter_rows: 0,
            intra_rows: 0,
        }
    }

    /// Encode rows of `x`; `parents[i]` is the in-batch parent row (or -1).
    /// Returns the payload and the receiver-identical reconstruction.
    pub fn encode(&mut self, x: &Tensor2D, parents: &[i32], bits: u8) -> (Vec<u8>, Tensor2D) {
        let d = x.cols;
        let mut recon = Tensor2D::zeros(x.rows, d);
        let mut w = ByteWriter::with_capacity(x.rows * (d * bits as usize / 8 + d / 32 + 16));
        w.u32(x.rows as u32);
        w.u32(d as u32);
        let mut u = vec![0.0f32; d];
        let mut uh = vec![0.0f32; d];
        for i in 0..x.rows {
            let row = x.row(i);
            let p = parents.get(i).copied().unwrap_or(-1);
            let mut use_inter = false;
            if p >= 0 && (p as usize) < i {
                let pr = recon.row(p as usize);
                let mut e_delta = 0.0f32;
                for j in 0..d {
                    u[j] = row[j] - pr[j];
                    e_delta += u[j] * u[j];
                }
                let e_intra = crate::tensor::dot(row, row);
                use_inter = e_delta < 0.7 * e_intra;
            }
            if use_inter {
                self.inter_rows += 1;
                w.u32(p as u32);
                let f = self.codec.encode(&u, bits, 0, &mut uh);
                f.encode(&mut w);
                let pr = recon.row(p as usize).to_vec();
                for (j, r) in recon.row_mut(i).iter_mut().enumerate() {
                    *r = pr[j] + uh[j];
                }
            } else {
                self.intra_rows += 1;
                w.u32(u32::MAX);
                let f = self.codec.encode(row, bits, 1, &mut uh);
                f.encode(&mut w);
                recon.row_mut(i).copy_from_slice(&uh);
            }
        }
        (w.into_inner(), recon)
    }

    pub fn decode(&mut self, bytes: &[u8]) -> Tensor2D {
        let mut r = ByteReader::new(bytes);
        let rows = r.u32() as usize;
        let d = r.u32() as usize;
        let mut out = Tensor2D::zeros(rows, d);
        let mut uh = vec![0.0f32; d];
        for i in 0..rows {
            let p = r.u32();
            self.codec.decode_bytes(&mut r, &mut uh);
            if p == u32::MAX {
                out.row_mut(i).copy_from_slice(&uh);
            } else {
                let pr = out.row(p as usize).to_vec();
                for (j, o) in out.row_mut(i).iter_mut().enumerate() {
                    *o = pr[j] + uh[j];
                }
            }
        }
        out
    }
}

/// Uncompressed FP16 activation payload (what ring pipelines ship today).
pub fn encode_fp16_rows(x: &Tensor2D) -> Vec<u8> {
    let mut b = Vec::with_capacity(8 + x.data.len() * 2);
    b.extend_from_slice(&(x.rows as u32).to_le_bytes());
    b.extend_from_slice(&(x.cols as u32).to_le_bytes());
    crate::tensor::encode_f16_slice(&x.data, &mut b);
    b
}

pub fn decode_fp16_rows(b: &[u8]) -> Tensor2D {
    let rows = u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
    let cols = u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize;
    let mut t = Tensor2D::zeros(rows, cols);
    crate::tensor::decode_f16_slice(&b[8..], &mut t.data);
    t
}

/// HRQ-V2: trunk-residual coding of pipeline activations.
///
/// A shard boundary carries h = e(token) + Σ_{l<L} Δ_l. Every stage holds the
/// embedding table and receives the token ids with the strand, so the base
/// e(token) is reconstructed exactly and only the trunk residual
/// r = h − e(token) is transmitted:
///   1. sketch: in every 64-element block, the 2 largest-|r| channels are sent
///      exactly as (u8 offset, f16) and removed;
///   2. the remainder is rotated by a randomised full-width FWHT;
///   3. each 128-element block of the rotated residual is quantised to a signed
///      `bits`-bit mid-rise lattice with an MSE-optimal Gaussian step and an f16
///      scale.
///
/// Temporal error feedback is deliberately *not* applied across tree rows:
/// rows are decoded independently, so injecting a neighbour's quantisation
/// error can only add variance to each row (it helps only for a single
/// accumulated stream, which is what `HolographicCompressor` covers).
pub struct TrunkResidualCodec {
    pub plan: fwht::HadamardPlan,
    pub bits: u8,
    pub sketch_block: usize,
    pub sketch_k: usize,
    pub quant_block: usize,
    rot: Vec<f32>,
    codes: Vec<u8>,
}

impl TrunkResidualCodec {
    pub fn new(dim: usize, bits: u8, seed: u64) -> Self {
        let plan = fwht::HadamardPlan::full(dim, seed);
        let padded = plan.padded;
        TrunkResidualCodec {
            plan,
            bits,
            sketch_block: 64,
            sketch_k: 2,
            quant_block: 128,
            rot: vec![0.0; padded],
            codes: Vec::new(),
        }
    }

    /// Wire bytes per row (excluding the 10-byte frame header).
    pub fn row_bytes(&self) -> usize {
        let d = self.plan.dim;
        d.div_ceil(self.sketch_block) * self.sketch_k * 3
            + self.plan.padded.div_ceil(self.quant_block) * 2
            + (self.plan.padded * self.bits as usize).div_ceil(8)
    }

    /// Encode rows of `x` against `base`; returns payload + exact receiver reconstruction.
    pub fn encode(&mut self, x: &Tensor2D, base: &Tensor2D) -> (Vec<u8>, Tensor2D) {
        let (rows, d) = (x.rows, x.cols);
        let mut w = ByteWriter::with_capacity(10 + rows * self.row_bytes());
        w.u32(rows as u32);
        w.u32(d as u32);
        w.u8(self.bits);
        w.u8(self.sketch_k as u8);
        let mut recon = Tensor2D::zeros(rows, d);
        let mut r = vec![0.0f32; d];
        let mut rhat = vec![0.0f32; d];
        for i in 0..rows {
            for j in 0..d {
                r[j] = x.row(i)[j] - base.row(i)[j];
            }
            // 1. per-block outlier sketch
            let mut sketch: Vec<(usize, u16)> =
                Vec::with_capacity(d / self.sketch_block * self.sketch_k);
            for (b, blk) in r.chunks_mut(self.sketch_block).enumerate() {
                for _ in 0..self.sketch_k.min(blk.len()) {
                    let (mut bi, mut bv) = (0usize, -1.0f32);
                    for (t, v) in blk.iter().enumerate() {
                        if v.abs() > bv {
                            bv = v.abs();
                            bi = t;
                        }
                    }
                    let h = crate::tensor::f32_to_f16(blk[bi]);
                    sketch.push((b * self.sketch_block + bi, h));
                    w.u8(bi as u8);
                    w.u16(h);
                    blk[bi] = 0.0;
                }
            }
            // 2–3. rotate + lattice-quantise the smooth remainder
            self.quantise_into(&r, &mut w, &mut rhat);
            for &(j, h) in &sketch {
                rhat[j] = crate::tensor::f16_to_f32(h);
            }
            for j in 0..d {
                recon.row_mut(i)[j] = base.row(i)[j] + rhat[j];
            }
        }
        (w.into_inner(), recon)
    }

    fn quantise_into(&mut self, r: &[f32], w: &mut ByteWriter, rhat: &mut [f32]) {
        self.plan.forward(r, &mut self.rot);
        let levels = 1u32 << self.bits;
        let half = (levels / 2) as f32;
        self.codes.clear();
        let mut scales = Vec::with_capacity(self.rot.len().div_ceil(self.quant_block));
        for blk in self.rot.chunks_mut(self.quant_block) {
            let step = if self.bits <= 6 {
                let rms = (blk.iter().map(|v| v * v).sum::<f32>() / blk.len() as f32).sqrt();
                holographic::step_factor(self.bits) * rms
            } else {
                blk.iter().fold(0.0f32, |m, v| m.max(v.abs())) / (half - 0.5)
            };
            let sh = crate::tensor::f32_to_f16(step);
            let sq = crate::tensor::f16_to_f32(sh);
            scales.push(sh);
            let inv = if sq > 0.0 { 1.0 / sq } else { 0.0 };
            for v in blk.iter_mut() {
                let c = ((*v * inv).floor() + half).clamp(0.0, (levels - 1) as f32);
                self.codes.push(c as u8);
                *v = (c - half + 0.5) * sq;
            }
        }
        for s in scales {
            w.u16(s);
        }
        w.bytes(&holographic::pack_codes(&self.codes, self.bits));
        self.plan.inverse(&mut self.rot, rhat);
    }

    pub fn decode(&mut self, bytes: &[u8], base: &Tensor2D) -> Tensor2D {
        let mut rd = ByteReader::new(bytes);
        let rows = rd.u32() as usize;
        let d = rd.u32() as usize;
        let bits = rd.u8();
        let k = rd.u8() as usize;
        let padded = self.plan.padded;
        let nq = padded.div_ceil(self.quant_block);
        let mut out = Tensor2D::zeros(rows, d);
        let mut rhat = vec![0.0f32; d];
        let half = ((1u32 << bits) / 2) as f32;
        for i in 0..rows {
            let mut sketch = Vec::with_capacity(d.div_ceil(self.sketch_block) * k);
            for b in 0..d.div_ceil(self.sketch_block) {
                for _ in 0..k {
                    let off = rd.u8() as usize;
                    sketch.push((b * self.sketch_block + off, rd.u16()));
                }
            }
            let scales: Vec<f32> = (0..nq)
                .map(|_| crate::tensor::f16_to_f32(rd.u16()))
                .collect();
            let packed = rd.bytes((padded * bits as usize).div_ceil(8));
            holographic::unpack_codes(packed, bits, padded, &mut self.codes);
            for (j, v) in self.rot.iter_mut().enumerate() {
                *v = (self.codes[j] as f32 - half + 0.5) * scales[j / self.quant_block];
            }
            self.plan.inverse(&mut self.rot, &mut rhat);
            for &(j, h) in &sketch {
                rhat[j] = crate::tensor::f16_to_f32(h);
            }
            for j in 0..d {
                out.row_mut(i)[j] = base.row(i)[j] + rhat[j];
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// HRQ-V2.1: entropy-coded trunk-residual lattice
// ---------------------------------------------------------------------------

/// Standard normal CDF via erf (Abramowitz–Stegun 7.1.26, |err| < 1.5e-7).
fn norm_cdf(x: f64) -> f64 {
    let z = x.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.327_591_1 * z);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-z * z).exp();
    if x >= 0.0 {
        0.5 * (1.0 + y)
    } else {
        0.5 * (1.0 - y)
    }
}

const LUT_BITS: u32 = 10;

/// Canonical Huffman code over symbols 0..n designed for given probabilities.
#[derive(Clone, Debug)]
pub struct CanonicalHuffman {
    pub lengths: Vec<u8>,
    pub codes: Vec<u32>,
    /// 2^LUT_BITS-entry decode table: (symbol, code length) for every code of
    /// length ≤ LUT_BITS, keyed by the next LUT_BITS input bits; length 0
    /// marks a longer code (slow path).
    pub lut: Vec<(u16, u8)>,
    /// (length, first code, first symbol index into `sorted`) per length
    first: Vec<(u8, u32, usize, usize)>,
    sorted: Vec<u16>,
}

impl CanonicalHuffman {
    pub fn from_probs(p: &[f64]) -> Self {
        let n = p.len();
        // Huffman code lengths by repeated merging (n is small)
        let mut nodes: Vec<(f64, Vec<usize>)> = p
            .iter()
            .enumerate()
            .map(|(i, &q)| (q.max(1e-12), vec![i]))
            .collect();
        let mut lengths = vec![0u8; n];
        while nodes.len() > 1 {
            nodes.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            let a = nodes.pop().unwrap();
            let b = nodes.pop().unwrap();
            for &s in a.1.iter().chain(&b.1) {
                lengths[s] += 1;
            }
            let mut syms = a.1;
            syms.extend(b.1);
            nodes.push((a.0 + b.0, syms));
        }
        if n == 1 {
            lengths[0] = 1;
        }
        // canonical assignment: by (length, symbol)
        let mut sorted: Vec<u16> = (0..n as u16).collect();
        sorted.sort_by_key(|&s| (lengths[s as usize], s));
        let mut codes = vec![0u32; n];
        let mut code = 0u32;
        let mut prev_len = lengths[sorted[0] as usize];
        let mut first = Vec::new();
        let mut start = 0usize;
        for (k, &s) in sorted.iter().enumerate() {
            let l = lengths[s as usize];
            if k > 0 {
                code = (code + 1) << (l - prev_len);
            }
            if k == 0 || l != prev_len {
                if k > 0 {
                    let last = first.len() - 1;
                    let e: &mut (u8, u32, usize, usize) = &mut first[last];
                    e.3 = k - start;
                }
                first.push((l, code, k, 0));
                start = k;
            }
            codes[s as usize] = code;
            prev_len = l;
        }
        let last = first.len() - 1;
        first[last].3 = n - start;
        let mut lut = vec![(0u16, 0u8); 1 << LUT_BITS];
        for s in 0..n {
            let l = lengths[s] as u32;
            if l <= LUT_BITS {
                let base = (codes[s] << (LUT_BITS - l)) as usize;
                for e in lut.iter_mut().skip(base).take(1 << (LUT_BITS - l)) {
                    *e = (s as u16, l as u8);
                }
            }
        }
        CanonicalHuffman {
            lengths,
            codes,
            lut,
            first,
            sorted,
        }
    }

    pub fn expected_bits(&self, p: &[f64]) -> f64 {
        p.iter()
            .zip(&self.lengths)
            .map(|(q, &l)| q * l as f64)
            .sum()
    }
}

struct BitWriter {
    out: Vec<u8>,
    acc: u64,
    n: u32,
}

impl BitWriter {
    fn new() -> Self {
        BitWriter {
            out: Vec::new(),
            acc: 0,
            n: 0,
        }
    }
    fn put(&mut self, code: u32, len: u8) {
        // MSB-first
        self.acc = (self.acc << len) | code as u64;
        self.n += len as u32;
        while self.n >= 8 {
            self.n -= 8;
            self.out.push((self.acc >> self.n) as u8);
        }
    }
    fn finish(mut self) -> Vec<u8> {
        if self.n > 0 {
            self.out.push((self.acc << (8 - self.n)) as u8);
        }
        self.out
    }
}

struct BitReader<'a> {
    b: &'a [u8],
    pos: usize,
}

impl BitReader<'_> {
    /// Next `n` bits MSB-first without consuming them (zero-padded at the end).
    fn peek(&self, n: u32) -> u32 {
        let mut v = 0u32;
        for i in 0..n as usize {
            let p = self.pos + i;
            let bit = (self.b.get(p / 8).copied().unwrap_or(0) >> (7 - p % 8)) & 1;
            v = (v << 1) | bit as u32;
        }
        v
    }

    fn bit(&mut self) -> u32 {
        let v = (self.b.get(self.pos / 8).copied().unwrap_or(0) >> (7 - self.pos % 8)) & 1;
        self.pos += 1;
        v as u32
    }
}

/// HRQ-V2.1: trunk residual r = h − e(token) → randomised full-width FWHT →
/// uniform lattice with step Δ·rms per 128-coefficient block (f16 scale) →
/// canonical Huffman code designed for the Gaussian law of rotated
/// coefficients. A per-row rate controller widens Δ until the row fits
/// `max_row_bytes` (default 25% of the FP16 row), so the wire budget is a hard
/// guarantee. At the same bytes this has ≈⅓ the squared error of the 3-bit +
/// outlier-sketch V2 layout, because entropy coding spends bits only where the
/// Gaussian puts probability mass.
pub struct EntropyResidualCodec {
    /// AWQ-style static channel salience s_j (powers of two, derived from the
    /// receiving layer's weights identically on both ends: 0 wire bytes).
    /// The residual is coded as u_j·s_j and divided by s_j after decoding.
    pub salience: Option<Vec<f32>>,
    /// Sensitivity subspace (k × d, orthonormal rows): the residual's k
    /// coordinates in this basis are sent as BFP16; only the orthogonal
    /// remainder goes through the entropy-coded lattice.
    pub basis: Option<Tensor2D>,
    pub plan: fwht::HadamardPlan,
    pub step: f32,
    pub max_sym: i32,
    pub quant_block: usize,
    pub max_row_bytes: usize,
    pub huff: CanonicalHuffman,
    rot: Vec<f32>,
}

impl EntropyResidualCodec {
    /// `budget_frac` of the FP16 row size is the hard per-row cap.
    pub fn new(dim: usize, budget_frac: f64, seed: u64) -> Self {
        let plan = fwht::HadamardPlan::full(dim, seed);
        let padded = plan.padded;
        let max_sym = 15i32;
        let quant_block = 128;
        let max_row_bytes = ((dim * 2) as f64 * budget_frac) as usize;
        // choose the finest step whose expected Huffman rate fits the budget
        let header = 2 + padded.div_ceil(quant_block) * 2;
        let target_bits = ((max_row_bytes - header) * 8) as f64 / padded as f64 * 0.97;
        let mut step = 0.2f32;
        let mut huff;
        loop {
            let p = Self::symbol_probs(step, max_sym);
            huff = CanonicalHuffman::from_probs(&p);
            if huff.expected_bits(&p) <= target_bits || step > 3.0 {
                break;
            }
            step *= 1.02;
        }
        EntropyResidualCodec {
            salience: None,
            basis: None,
            plan,
            step,
            max_sym,
            quant_block,
            max_row_bytes,
            huff,
            rot: vec![0.0; padded],
        }
    }

    /// Protect a sensitivity subspace: `basis` rows are orthonormal (k × d).
    /// The lattice step is re-designed so lattice + subspace bytes still fit
    /// the same per-row cap.
    pub fn with_basis(mut self, basis: Tensor2D) -> Self {
        let k = basis.rows;
        let padded = self.plan.padded;
        let header = 2 + padded.div_ceil(self.quant_block) * 2 + 1 + 2 * k;
        let target_bits = ((self.max_row_bytes - header) * 8) as f64 / padded as f64 * 0.97;
        let mut step = 0.2f32;
        loop {
            let p = Self::symbol_probs(step, self.max_sym);
            let h = CanonicalHuffman::from_probs(&p);
            if h.expected_bits(&p) <= target_bits || step > 3.0 {
                self.huff = h;
                break;
            }
            step *= 1.02;
        }
        self.step = step;
        self.basis = Some(basis);
        self
    }

    /// P(symbol s) for a unit Gaussian quantised with mid-tread step Δ,
    /// tails folded into ±max_sym.
    fn symbol_probs(step: f32, max_sym: i32) -> Vec<f64> {
        let d = step as f64;
        (-max_sym..=max_sym)
            .map(|s| {
                let lo = if s == -max_sym {
                    f64::NEG_INFINITY
                } else {
                    (s as f64 - 0.5) * d
                };
                let hi = if s == max_sym {
                    f64::INFINITY
                } else {
                    (s as f64 + 0.5) * d
                };
                let c = |x: f64| {
                    if x.is_infinite() {
                        if x > 0.0 {
                            1.0
                        } else {
                            0.0
                        }
                    } else {
                        norm_cdf(x)
                    }
                };
                c(hi) - c(lo)
            })
            .collect()
    }

    fn encode_row(&mut self, r: &[f32], scale_mult: f32, rhat: &mut [f32]) -> Vec<u8> {
        let scaled: Vec<f32>;
        let r = match &self.salience {
            Some(sal) => {
                scaled = r.iter().zip(sal).map(|(a, b)| a * b).collect();
                &scaled[..]
            }
            None => r,
        };
        self.plan.forward(r, &mut self.rot);
        let mut bw = BitWriter::new();
        let mut scales = Vec::new();
        for blk in self.rot.chunks_mut(self.quant_block) {
            let rms = (blk.iter().map(|v| v * v).sum::<f32>() / blk.len() as f32).sqrt();
            let sh = crate::tensor::f32_to_f16(rms * self.step * scale_mult);
            let sq = crate::tensor::f16_to_f32(sh);
            scales.push(sh);
            let inv = if sq > 0.0 { 1.0 / sq } else { 0.0 };
            for v in blk.iter_mut() {
                let s = (*v * inv)
                    .round()
                    .clamp(-self.max_sym as f32, self.max_sym as f32)
                    as i32;
                let sym = (s + self.max_sym) as usize;
                bw.put(self.huff.codes[sym], self.huff.lengths[sym]);
                *v = s as f32 * sq;
            }
        }
        self.plan.inverse(&mut self.rot, rhat);
        if let Some(sal) = &self.salience {
            for (x, s) in rhat.iter_mut().zip(sal) {
                *x /= s;
            }
        }
        let bits = bw.finish();
        let mut out = Vec::with_capacity(2 + scales.len() * 2 + bits.len());
        out.extend_from_slice(&(bits.len() as u16).to_le_bytes());
        for s in scales {
            out.extend_from_slice(&s.to_le_bytes());
        }
        out.extend_from_slice(&bits);
        out
    }

    /// Encode rows of `x` against `base`; returns payload + exact receiver reconstruction.
    pub fn encode(&mut self, x: &Tensor2D, base: &Tensor2D) -> (Vec<u8>, Tensor2D) {
        let (rows, d) = (x.rows, x.cols);
        let mut w = ByteWriter::with_capacity(8 + rows * self.max_row_bytes);
        w.u32(rows as u32);
        w.u32(d as u32);
        let mut recon = Tensor2D::zeros(rows, d);
        let mut r = vec![0.0f32; d];
        let mut rhat = vec![0.0f32; d];
        for i in 0..rows {
            for j in 0..d {
                r[j] = x.row(i)[j] - base.row(i)[j];
            }
            // sensitivity subspace: exact-ish BFP16 coordinates, remove them
            let mut sketch_bytes = Vec::new();
            let mut sub = vec![0.0f32; d];
            if let Some(b) = &self.basis {
                let c: Vec<f32> = (0..b.rows)
                    .map(|k| crate::tensor::dot(b.row(k), &r))
                    .collect();
                let mut sw = ByteWriter::default();
                crate::kv_fabric::disaggregated_attention::bfp_put(&mut sw, &c, 16);
                sketch_bytes = sw.into_inner();
                let ch = crate::kv_fabric::disaggregated_attention::bfp_get(
                    &mut ByteReader::new(&sketch_bytes),
                    b.rows,
                    16,
                );
                for (k, ck) in ch.iter().enumerate() {
                    for (sv, bv) in sub.iter_mut().zip(b.row(k)) {
                        *sv += ck * bv;
                    }
                }
                for j in 0..d {
                    r[j] -= sub[j];
                }
            }
            let cap = self.max_row_bytes - sketch_bytes.len();
            // rate control: widen the step until the row fits its hard cap
            let mut mult = 1.0f32;
            let mut row = self.encode_row(&r, mult, &mut rhat);
            while row.len() > cap && mult < 8.0 {
                mult *= 1.12;
                row = self.encode_row(&r, mult, &mut rhat);
            }
            w.bytes(&sketch_bytes);
            w.bytes(&row);
            for j in 0..d {
                recon.row_mut(i)[j] = base.row(i)[j] + rhat[j] + sub[j];
            }
        }
        (w.into_inner(), recon)
    }

    pub fn decode(&mut self, bytes: &[u8], base: &Tensor2D) -> Tensor2D {
        let mut rd = ByteReader::new(bytes);
        let rows = rd.u32() as usize;
        let d = rd.u32() as usize;
        let padded = self.plan.padded;
        let nq = padded.div_ceil(self.quant_block);
        let mut out = Tensor2D::zeros(rows, d);
        let mut rhat = vec![0.0f32; d];
        for i in 0..rows {
            let mut sub = vec![0.0f32; d];
            if let Some(b) = &self.basis {
                let ch = crate::kv_fabric::disaggregated_attention::bfp_get(&mut rd, b.rows, 16);
                for (k, ck) in ch.iter().enumerate() {
                    for (sv, bv) in sub.iter_mut().zip(b.row(k)) {
                        *sv += ck * bv;
                    }
                }
            }
            let nbits = rd.u16() as usize;
            let scales: Vec<f32> = (0..nq)
                .map(|_| crate::tensor::f16_to_f32(rd.u16()))
                .collect();
            let mut br = BitReader {
                b: rd.bytes(nbits),
                pos: 0,
            };
            for j in 0..padded {
                // table-driven decode for codes ≤ LUT_BITS, canonical walk otherwise
                let (ls, ll) = self.huff.lut[br.peek(LUT_BITS) as usize];
                if ll > 0 {
                    br.pos += ll as usize;
                    self.rot[j] = (ls as i32 - self.max_sym) as f32 * scales[j / self.quant_block];
                    continue;
                }
                let mut code = 0u32;
                let mut len = 0u8;
                let mut sym = 0usize;
                'dec: loop {
                    code = (code << 1) | br.bit();
                    len += 1;
                    for &(l, first, idx, count) in &self.huff.first {
                        if l == len && code >= first && code < first + count as u32 {
                            sym = self.huff.sorted[idx + (code - first) as usize] as usize;
                            break 'dec;
                        }
                    }
                    if len > 31 {
                        break;
                    }
                }
                self.rot[j] = (sym as i32 - self.max_sym) as f32 * scales[j / self.quant_block];
            }
            self.plan.inverse(&mut self.rot, &mut rhat);
            if let Some(sal) = &self.salience {
                for (x, s) in rhat.iter_mut().zip(sal) {
                    *x /= s;
                }
            }
            for j in 0..d {
                out.row_mut(i)[j] = base.row(i)[j] + rhat[j] + sub[j];
            }
        }
        out
    }

    pub fn row_bytes_cap(&self) -> usize {
        self.max_row_bytes
    }
}

/// AWQ-style static channel salience for a layer boundary, from the receiving
/// layer's input matrices: s_j = clamp((‖W[:, j]‖ / mean_k ‖W[:, k]‖)^0.75,
/// 1/8, 16), with every input column scaled by the RMSNorm weight g_j that
/// multiplies channel j before it is read, rounded to a power of two so both
/// ends derive bit-identical constants.
pub fn channel_salience(mats: &[&crate::tensor::Linear], g: &[f32]) -> Vec<f32> {
    let d = g.len();
    let mut norms = vec![0.0f64; d];
    for m in mats {
        let w = m.dense();
        for r in 0..w.rows {
            for (j, v) in w.row(r).iter().enumerate() {
                norms[j] += (*v as f64 * g[j] as f64).powi(2);
            }
        }
    }
    let n: Vec<f64> = norms.iter().map(|x| x.sqrt()).collect();
    let mean = n.iter().sum::<f64>() / d as f64;
    n.iter()
        .map(|x| {
            let s = (x / mean.max(1e-30)).powf(0.75).clamp(0.125, 16.0);
            2f64.powi(s.log2().round() as i32) as f32
        })
        .collect()
}
