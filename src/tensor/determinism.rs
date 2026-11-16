//! Cross-silicon numerical determinism.
//!
//! Floating-point addition is not associative: NEON (4 lanes), AVX2 (8 lanes),
//! CUDA warp shuffles (32 lanes) and scalar loops all reduce `Σ xᵢ²` in
//! different orders, and FMA contraction changes rounding again. When an
//! activation crosses a shard boundary between different silicon, those tiny
//! differences in RMSNorm statistics compound into divergent logits.
//!
//! `CanonicalBoundaryAnchor` removes the order dependence:
//!   1. every block of `block` values is given a power-of-two scale `2^e` with
//!      `max|x| · 2^-e ∈ [0.5, 1)`;
//!   2. values are rounded (half-to-even) onto the Q8.24 grid `q · 2^(e-24)`,
//!      `q: i32` — the 8 integer bits are headroom for partial sums;
//!   3. reductions (Σ q²) are accumulated in exact `u128` integer arithmetic,
//!      which *is* associative, so any lane width / order gives the same bits;
//!   4. the single normalisation factor is produced by correctly-rounded IEEE
//!      f64 `sqrt`/`div` and each output by one correctly-rounded multiply chain
//!      (Rust never contracts to FMA implicitly).
//!
//! Result: bit-identical RMSNorm outputs for identical canonical inputs, no
//! matter how the host would have ordered its own float reductions.

pub const FRAC_BITS: i32 = 24;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AnchoredBlock {
    /// Block exponent `e` such that value = q · 2^(e − 24).
    pub exp: i8,
    pub q: Vec<i32>,
}

#[derive(Clone, Debug)]
pub struct CanonicalBoundaryAnchor {
    pub block: usize,
}

impl Default for CanonicalBoundaryAnchor {
    fn default() -> Self {
        CanonicalBoundaryAnchor { block: 64 }
    }
}

#[inline]
fn exp2i(e: i32) -> f64 {
    // Exact power of two for the exponent range we use.
    f64::from_bits(((e + 1023) as u64) << 52)
}

impl CanonicalBoundaryAnchor {
    pub fn new(block: usize) -> Self {
        assert!(block > 0);
        CanonicalBoundaryAnchor { block }
    }

    /// Block exponent: smallest e with max|x| < 2^e (so normalised max ∈ [0.5, 1)).
    fn block_exp(blk: &[f32]) -> i8 {
        let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        if amax == 0.0 || !amax.is_finite() {
            return 0;
        }
        // f32 exponent extraction is exact: amax = 1.m · 2^E  ⇒  amax < 2^(E+1)
        let e = ((amax.to_bits() >> 23) & 0xff) as i32 - 127 + 1;
        e.clamp(-100, 100) as i8
    }

    /// Encode into block-scaled Q8.24 fixed point.
    pub fn encode(&self, x: &[f32]) -> Vec<AnchoredBlock> {
        x.chunks(self.block)
            .map(|blk| {
                let e = Self::block_exp(blk);
                let s = exp2i(FRAC_BITS - e as i32);
                let q = blk
                    .iter()
                    .map(|&v| {
                        let r = (v as f64 * s).round_ties_even();
                        r.clamp(i32::MIN as f64, i32::MAX as f64) as i32
                    })
                    .collect();
                AnchoredBlock { exp: e, q }
            })
            .collect()
    }

    pub fn decode(&self, blocks: &[AnchoredBlock], out: &mut [f32]) {
        let mut o = 0;
        for b in blocks {
            let s = exp2i(b.exp as i32 - FRAC_BITS);
            for &q in &b.q {
                out[o] = (q as f64 * s) as f32;
                o += 1;
            }
        }
    }

    /// Snap a vector onto the canonical grid in place. Idempotent.
    pub fn anchor_inplace(&self, x: &mut [f32]) {
        let blocks = self.encode(x);
        self.decode(&blocks, x);
    }

    /// Exact, order-independent Σ x² of the canonical values, returned as an
    /// f64 (single correctly-rounded conversion). Accumulates in u128.
    pub fn exact_sum_squares(&self, blocks: &[AnchoredBlock]) -> f64 {
        let e_ref = blocks.iter().map(|b| b.exp as i32).max().unwrap_or(0);
        let mut acc: u128 = 0;
        for b in blocks {
            let shift = (2 * (e_ref - b.exp as i32)) as u32;
            for &q in &b.q {
                let sq = (q as i64 * q as i64) as u128; // < 2^62
                acc += if shift >= 127 { 0 } else { sq >> shift };
            }
        }
        // value = acc · 2^(2·(e_ref − 24))
        acc as f64 * exp2i(2 * (e_ref - FRAC_BITS))
    }

    /// Deterministic RMSNorm over canonical inputs. `x` is anchored first.
    pub fn rms_norm(&self, x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
        let blocks = self.encode(x);
        let ss = self.exact_sum_squares(&blocks);
        let mean = ss / x.len() as f64;
        let inv = 1.0 / (mean + eps as f64).sqrt();
        let mut o = 0;
        for b in &blocks {
            let s = exp2i(b.exp as i32 - FRAC_BITS);
            for &q in &b.q {
                let xv = q as f64 * s; // exact
                out[o] = ((xv * inv) * w[o] as f64) as f32;
                o += 1;
            }
        }
    }

    /// The `canonical_scale_q16` header field: the max block exponent packed
    /// with the block size so the receiver can validate the anchor grid.
    pub fn header_scale_q16(&self, blocks: &[AnchoredBlock]) -> u64 {
        let e_max = blocks.iter().map(|b| b.exp as i32).max().unwrap_or(0);
        (((e_max + 128) as u64) << 16) | (self.block as u64 & 0xffff)
    }
}

/// Non-deterministic reference reductions emulating different silicon, used by
/// tests and the benchmark to demonstrate the divergence the anchor removes.
pub mod emulate {
    /// Sequential scalar loop.
    pub fn sum_sq_scalar(x: &[f32]) -> f32 {
        let mut s = 0.0f32;
        for v in x {
            s += v * v;
        }
        s
    }
    /// `lanes`-wide SIMD-style reduction (NEON=4, AVX2=8, warp=32).
    pub fn sum_sq_lanes(x: &[f32], lanes: usize) -> f32 {
        let mut acc = vec![0.0f32; lanes];
        for (i, v) in x.iter().enumerate() {
            acc[i % lanes] += v * v;
        }
        // tree reduce lanes
        let mut n = lanes;
        while n > 1 {
            let h = n / 2;
            for i in 0..h {
                acc[i] += acc[i + h];
            }
            if n % 2 == 1 {
                acc[0] += acc[n - 1];
            }
            n = h;
        }
        acc[0]
    }
    /// Fused multiply-add contraction (as GPUs and `-ffp-contract=fast` do).
    pub fn sum_sq_fma(x: &[f32]) -> f32 {
        let mut s = 0.0f32;
        for v in x {
            s = v.mul_add(*v, s);
        }
        s
    }
    /// Reversed order.
    pub fn sum_sq_reversed(x: &[f32]) -> f32 {
        let mut s = 0.0f32;
        for v in x.iter().rev() {
            s += v * v;
        }
        s
    }
    /// RMSNorm using a given (non-canonical) sum-of-squares.
    pub fn rms_with(x: &[f32], w: &[f32], eps: f32, ss: f32, out: &mut [f32]) {
        let inv = 1.0 / (ss / x.len() as f32 + eps).sqrt();
        for i in 0..x.len() {
            out[i] = x[i] * inv * w[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Rng;

    #[test]
    fn encode_decode_idempotent() {
        let a = CanonicalBoundaryAnchor::new(64);
        let mut rng = Rng::new(9);
        let mut x: Vec<f32> = (0..300).map(|_| rng.normal() * 3.0).collect();
        a.anchor_inplace(&mut x);
        let y = x.clone();
        a.anchor_inplace(&mut x);
        assert_eq!(x, y);
    }
}
