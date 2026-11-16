//! K-quants and the remaining legacy GGML block formats.
//!
//! Byte layouts follow `ggml-common.h` exactly:
//!
//! | type | id | weights / block | bytes | layout |
//! |---|---:|---:|---:|---|
//! | Q4_1 | 3 | 32 | 20 | `d: f16, m: f16, qs[16]` — `w = d·q + m` |
//! | Q5_0 | 6 | 32 | 22 | `d: f16, qh: u32, qs[16]` — `w = d·(q − 16)` |
//! | Q5_1 | 7 | 32 | 24 | `d, m: f16, qh: u32, qs[16]` — `w = d·q + m` |
//! | Q4_K | 12 | 256 | 144 | `d, dmin: f16, scales[12], qs[128]` — 8 sub-blocks of 32, 6-bit scale / min each |
//! | Q5_K | 13 | 256 | 176 | `d, dmin: f16, scales[12], qh[32], qs[128]` |
//! | Q6_K | 14 | 256 | 210 | `ql[128], qh[64], scales: i8[16], d: f16` — 16 sub-blocks of 16 |
//!
//! Every format decodes a block into signed 8-bit integers plus per-16-element
//! scales and per-32-element additive terms ([`GroupBlock`]). One kernel then
//! takes the integer dot of those values with the Q8 activation blocks
//! (NEON `sdot`, AVX2 `maddubs`, or scalar) and applies the scales:
//! `Σ w·x = Σ_b d_x[b]·(s_lo·Σ_lo q·q_x + s_hi·Σ_hi q·q_x + m·Σ q_x)`.
//! Activations use the engine's 32-element Q8 blocks, which are finer than
//! llama.cpp's 256-element `Q8_K` activation blocks for K-quants.

use super::quant::{DotKernel, Q8Acts, WeightFormat, QK};
use super::{f16_to_f32, f32_to_f16};

pub const QK_K: usize = 256;

#[inline(always)]
fn h(b: &[u8], o: usize) -> f32 {
    super::quant::f16_fast(u16::from_le_bytes([b[o], b[o + 1]]))
}

/// One decoded weight block.
#[derive(Clone)]
pub struct GroupBlock {
    /// Integer weights in element order (32 or 256 valid).
    pub q: [i8; QK_K],
    /// Effective scale of each 16-element group.
    pub scale: [f32; 16],
    /// Additive term per 32-element group (`w = scale·q + add`).
    pub add: [f32; 8],
}

impl Default for GroupBlock {
    fn default() -> Self {
        GroupBlock {
            q: [0; QK_K],
            scale: [0.0; 16],
            add: [0.0; 8],
        }
    }
}

/// 6-bit scale / min `j` of a Q4_K / Q5_K block (`get_scale_min_k4`).
#[inline(always)]
pub fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// Bytes per block of `fmt`.
pub fn block_bytes(fmt: WeightFormat) -> usize {
    match fmt {
        WeightFormat::Q4_1 => 20,
        WeightFormat::Q5_0 => 22,
        WeightFormat::Q5_1 => 24,
        WeightFormat::Q4_K => 144,
        WeightFormat::Q5_K => 176,
        WeightFormat::Q6_K => 210,
        WeightFormat::Q8_0 => 34,
        WeightFormat::Q4_0 => 18,
        _ => unreachable!("not a block format"),
    }
}

/// Decode block `b` (raw bytes) of `fmt` into `g`.
pub fn decode_group(fmt: WeightFormat, b: &[u8], g: &mut GroupBlock) {
    match fmt {
        WeightFormat::Q4_1 | WeightFormat::Q5_0 | WeightFormat::Q5_1 => {
            let (d, m, qh, qs) = match fmt {
                WeightFormat::Q4_1 => (h(b, 0), h(b, 2), 0u32, &b[4..20]),
                WeightFormat::Q5_0 => (
                    h(b, 0),
                    0.0,
                    u32::from_le_bytes(b[2..6].try_into().unwrap()),
                    &b[6..22],
                ),
                _ => (
                    h(b, 0),
                    h(b, 2),
                    u32::from_le_bytes(b[4..8].try_into().unwrap()),
                    &b[8..24],
                ),
            };
            let off: i32 = if fmt == WeightFormat::Q5_0 { 16 } else { 0 };
            for j in 0..16 {
                let (mut lo, mut hi) = ((qs[j] & 0x0F) as i32, (qs[j] >> 4) as i32);
                if fmt != WeightFormat::Q4_1 {
                    lo |= (((qh >> j) << 4) & 0x10) as i32;
                    hi |= ((qh >> (j + 12)) & 0x10) as i32;
                }
                g.q[j] = (lo - off) as i8;
                g.q[j + 16] = (hi - off) as i8;
            }
            g.scale[0] = d;
            g.scale[1] = d;
            g.add[0] = m;
        }
        WeightFormat::Q4_K | WeightFormat::Q5_K => {
            let (d, dmin) = (h(b, 0), h(b, 2));
            let sc = &b[4..16];
            let (qh, ql) = if fmt == WeightFormat::Q5_K {
                (&b[16..48], &b[48..176])
            } else {
                (&[][..], &b[16..144])
            };
            for j in 0..4 {
                // sub-blocks 2j (low nibbles) and 2j+1 (high nibbles) of qs[32j..32j+32]
                let q = &ql[32 * j..32 * j + 32];
                for l in 0..32 {
                    let (mut lo, mut hi) = (q[l] & 0x0F, q[l] >> 4);
                    if fmt == WeightFormat::Q5_K {
                        lo |= ((qh[l] >> (2 * j)) & 1) << 4;
                        hi |= ((qh[l] >> (2 * j + 1)) & 1) << 4;
                    }
                    g.q[64 * j + l] = lo as i8;
                    g.q[64 * j + 32 + l] = hi as i8;
                }
                for s in 0..2 {
                    let (scv, mv) = scale_min_k4(2 * j + s, sc);
                    let k = 2 * j + s;
                    g.scale[2 * k] = d * scv as f32;
                    g.scale[2 * k + 1] = d * scv as f32;
                    g.add[k] = -dmin * mv as f32;
                }
            }
        }
        WeightFormat::Q6_K => {
            let (ql, qh, sc) = (&b[0..128], &b[128..192], &b[192..208]);
            let d = h(b, 208);
            for half in 0..2 {
                let (ql, qh) = (&ql[64 * half..], &qh[32 * half..]);
                let base = 128 * half;
                for l in 0..32 {
                    let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i8 - 32;
                    let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
                    let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
                    let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;
                    g.q[base + l] = q1;
                    g.q[base + 32 + l] = q2;
                    g.q[base + 64 + l] = q3;
                    g.q[base + 96 + l] = q4;
                }
            }
            for k in 0..16 {
                g.scale[k] = d * sc[k] as i8 as f32;
            }
            g.add = [0.0; 8];
        }
        _ => unreachable!("decode_group: {fmt:?}"),
    }
}

/// Dequantize `out.len()` elements.
pub fn decode_row(b: &[u8], fmt: WeightFormat, out: &mut [f32]) {
    let n = fmt.block_len();
    let bb = block_bytes(fmt);
    let mut g = GroupBlock::default();
    for (i, blk) in out.chunks_mut(n).enumerate() {
        decode_group(fmt, &b[i * bb..(i + 1) * bb], &mut g);
        for (e, y) in blk.iter_mut().enumerate() {
            *y = g.scale[e / 16] * g.q[e] as f32 + g.add[e / 32];
        }
    }
}

// --- quantizers (exporters) ---------------------------------------------------

#[inline]
fn put_h(out: &mut Vec<u8>, x: f32) {
    out.extend_from_slice(&f32_to_f16(x).to_le_bytes());
}

/// Q5_0 (llama.cpp reference rounding).
fn quantize_q5_0(x: &[f32], out: &mut Vec<u8>) {
    for blk in x.as_chunks::<QK>().0 {
        let (mut amax, mut max) = (0.0f32, 0.0f32);
        for &v in blk {
            if v.abs() > amax {
                amax = v.abs();
                max = v;
            }
        }
        let d = max / -16.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        put_h(out, d);
        let mut qh = 0u32;
        let mut qs = [0u8; 16];
        for j in 0..16 {
            let a = ((blk[j] * id + 16.5) as i32).clamp(0, 31) as u32;
            let b = ((blk[j + 16] * id + 16.5) as i32).clamp(0, 31) as u32;
            qs[j] = ((a & 0xF) | ((b & 0xF) << 4)) as u8;
            qh |= ((a & 0x10) >> 4) << j;
            qh |= ((b & 0x10) >> 4) << (j + 16);
        }
        out.extend_from_slice(&qh.to_le_bytes());
        out.extend_from_slice(&qs);
    }
}

/// Q6_K: per 16-element group absmax scale, 8-bit group scales under one
/// f16 super-scale, then 6-bit codes re-rounded with the stored scales.
fn quantize_q6_k(x: &[f32], out: &mut Vec<u8>) {
    for blk in x.as_chunks::<QK_K>().0 {
        let mut gs = [0.0f32; 16];
        for (k, g) in blk.as_chunks::<16>().0.iter().enumerate() {
            let (mut amax, mut max) = (0.0f32, 0.0f32);
            for &v in g {
                if v.abs() > amax {
                    amax = v.abs();
                    max = v;
                }
            }
            gs[k] = max / -32.0;
        }
        let amax_s = gs.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = f16_to_f32(f32_to_f16(amax_s / 127.0));
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let mut sc = [0i8; 16];
        let mut l = [0u8; QK_K];
        for k in 0..16 {
            sc[k] = (gs[k] * id).round().clamp(-128.0, 127.0) as i8;
            let s = d * sc[k] as f32;
            let is = if s != 0.0 { 1.0 / s } else { 0.0 };
            for t in 0..16 {
                l[16 * k + t] = ((blk[16 * k + t] * is).round() as i32 + 32).clamp(0, 63) as u8;
            }
        }
        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];
        for half in 0..2 {
            for t in 0..32 {
                let b = 128 * half;
                let (q1, q2, q3, q4) = (l[b + t], l[b + 32 + t], l[b + 64 + t], l[b + 96 + t]);
                ql[64 * half + t] = (q1 & 0xF) | ((q3 & 0xF) << 4);
                ql[64 * half + t + 32] = (q2 & 0xF) | ((q4 & 0xF) << 4);
                qh[32 * half + t] =
                    (q1 >> 4) | ((q2 >> 4) << 2) | ((q3 >> 4) << 4) | ((q4 >> 4) << 6);
            }
        }
        out.extend_from_slice(&ql);
        out.extend_from_slice(&qh);
        out.extend(sc.iter().map(|&s| s as u8));
        put_h(out, d);
    }
}

/// Q4_K: per 32-element group affine fit `w ≈ s·q − m` (q ∈ 0..15), 6-bit
/// scales and mins under f16 super-scales, codes re-rounded with them.
fn quantize_q4_k(x: &[f32], out: &mut Vec<u8>) {
    for blk in x.as_chunks::<QK_K>().0 {
        let mut s = [0.0f32; 8];
        let mut m = [0.0f32; 8];
        for (j, g) in blk.as_chunks::<32>().0.iter().enumerate() {
            let lo = g.iter().fold(f32::MAX, |a, &v| a.min(v)).min(0.0);
            let hi = g.iter().fold(f32::MIN, |a, &v| a.max(v));
            s[j] = (hi - lo) / 15.0;
            m[j] = -lo;
        }
        let d = f16_to_f32(f32_to_f16(s.iter().fold(0.0f32, |a, &v| a.max(v)) / 63.0));
        let dmin = f16_to_f32(f32_to_f16(m.iter().fold(0.0f32, |a, &v| a.max(v)) / 63.0));
        let mut ls = [0u8; 8];
        let mut lm = [0u8; 8];
        for j in 0..8 {
            ls[j] = if d > 0.0 {
                (s[j] / d).round().clamp(0.0, 63.0) as u8
            } else {
                0
            };
            lm[j] = if dmin > 0.0 {
                (m[j] / dmin).round().clamp(0.0, 63.0) as u8
            } else {
                0
            };
        }
        let mut sc = [0u8; 12];
        for j in 0..8 {
            if j < 4 {
                sc[j] = ls[j];
                sc[j + 4] = lm[j];
            } else {
                sc[j + 4] = (ls[j] & 0xF) | ((lm[j] & 0xF) << 4);
                sc[j - 4] |= (ls[j] >> 4) << 6;
                sc[j] |= (lm[j] >> 4) << 6;
            }
        }
        let mut qs = [0u8; 128];
        for j in 0..8 {
            let (sv, mv) = scale_min_k4(j, &sc);
            let (sj, mj) = (d * sv as f32, dmin * mv as f32);
            let is = if sj != 0.0 { 1.0 / sj } else { 0.0 };
            for t in 0..32 {
                let q = ((blk[32 * j + t] + mj) * is).round().clamp(0.0, 15.0) as u8;
                let byte = &mut qs[32 * (j / 2) + t];
                *byte |= if j % 2 == 0 { q } else { q << 4 };
            }
        }
        put_h(out, d);
        put_h(out, dmin);
        out.extend_from_slice(&sc);
        out.extend_from_slice(&qs);
    }
}

/// Encode one row in `fmt` (K-quant / legacy formats of this module).
pub fn quantize_row(fmt: WeightFormat, x: &[f32], out: &mut Vec<u8>) {
    match fmt {
        WeightFormat::Q5_0 => quantize_q5_0(x, out),
        WeightFormat::Q4_K => quantize_q4_k(x, out),
        WeightFormat::Q6_K => quantize_q6_k(x, out),
        f => panic!("no quantizer for {f:?} (decode-only format)"),
    }
}

// --- integer dot kernel ---------------------------------------------------------

/// (Σ q·x over elements 0..16, Σ over 16..32) of one 32-element sub-block.
#[inline(always)]
pub fn dot16x2(kernel: DotKernel, w: &[i8], x: &[i8]) -> (i32, i32) {
    match kernel {
        #[cfg(target_arch = "aarch64")]
        DotKernel::NeonDot => {
            // SAFETY: `dotprod` was detected at runtime; 32 elements in bounds.
            unsafe { neon::dot16x2_sdot(w.as_ptr(), x.as_ptr()) }
        }
        #[cfg(target_arch = "aarch64")]
        DotKernel::Neon => {
            // SAFETY: NEON is baseline on aarch64.
            unsafe { neon::dot16x2_mull(w.as_ptr(), x.as_ptr()) }
        }
        #[cfg(target_arch = "x86_64")]
        DotKernel::Avx2 => {
            // SAFETY: AVX2 was detected at runtime (or forced by a test).
            unsafe { avx2::dot16x2(w.as_ptr(), x.as_ptr()) }
        }
        _ => {
            let lo: i32 = (0..16).map(|t| w[t] as i32 * x[t] as i32).sum();
            let hi: i32 = (16..32).map(|t| w[t] as i32 * x[t] as i32).sum();
            (lo, hi)
        }
    }
}

/// Fused dot of one weight row in a group format against up to 4 Q8
/// activation rows. Q4_K / Q5_K / Q6_K use dedicated SIMD kernels (NEON
/// `sdot`, AVX2) that unpack nibbles and high bits in registers and keep the
/// integer sums in vector lanes; the other formats (and CPUs without those
/// extensions) decode each weight block once and reuse it for every row.
pub fn row_dot_group(
    kernel: DotKernel,
    fmt: WeightFormat,
    row: &[u8],
    acts: &Q8Acts,
    rows: &[usize],
    out: &mut [f32; 4],
) {
    match (kernel, fmt) {
        #[cfg(target_arch = "aarch64")]
        (DotKernel::NeonDot, WeightFormat::Q4_K) => {
            // SAFETY: `dotprod` was detected at runtime; loads stay in the row / acts.
            unsafe { kneon::q4k::<false, false>(row, acts, rows, out, &mut [0; 4]) };
            return;
        }
        #[cfg(target_arch = "aarch64")]
        (DotKernel::NeonDot, WeightFormat::Q5_K) => {
            // SAFETY: as above.
            unsafe { kneon::q4k::<false, true>(row, acts, rows, out, &mut [0; 4]) };
            return;
        }
        #[cfg(target_arch = "aarch64")]
        (DotKernel::NeonDot, WeightFormat::Q6_K) => {
            // SAFETY: as above.
            unsafe { kneon::q6k::<false>(row, acts, rows, out, &mut [0; 4]) };
            return;
        }
        #[cfg(target_arch = "x86_64")]
        (DotKernel::Avx2, WeightFormat::Q4_K) => {
            // SAFETY: AVX2 + FMA were detected at runtime (or forced by a test).
            unsafe { kavx2::q4k(row, acts, rows, out) };
            return;
        }
        #[cfg(target_arch = "x86_64")]
        (DotKernel::Avx2, WeightFormat::Q6_K) => {
            // SAFETY: as above.
            unsafe { kavx2::q6k(row, acts, rows, out) };
            return;
        }
        _ => {}
    }
    let n = fmt.block_len();
    let bb = block_bytes(fmt);
    let sub = n / QK;
    let nb = acts.cols / QK;
    let mut g = GroupBlock::default();
    *out = [0.0; 4];
    for blk in 0..acts.cols / n {
        decode_group(fmt, &row[blk * bb..(blk + 1) * bb], &mut g);
        for (r, &i) in rows.iter().enumerate() {
            let mut acc = 0.0f32;
            for s in 0..sub {
                let b = blk * sub + s;
                let xq = &acts.q[i * acts.cols + b * QK..i * acts.cols + (b + 1) * QK];
                let (lo, hi) = dot16x2(kernel, &g.q[s * QK..(s + 1) * QK], xq);
                let k = i * nb + b;
                acc += acts.d[k]
                    * (g.scale[2 * s] * lo as f32
                        + g.scale[2 * s + 1] * hi as f32
                        + g.add[s] * acts.s[k] as f32);
            }
            out[r] += acc;
        }
    }
}

// --- exact (fixed-point) accumulation -------------------------------------------

/// Fixed-point grid of exact accumulation: partial products are truncated to
/// multiples of 2^-32 and summed as `i64`, so a dot product's value no longer
/// depends on how its terms are grouped (SIMD lanes, 1- vs 4-row tiles, or
/// shards of a row-parallel split summed by an all-reduce).
pub const FIX_ONE: f64 = 4_294_967_296.0;

/// One canonical f32 partial product on the fixed-point grid.
#[inline(always)]
pub fn to_fix(p: f32) -> i64 {
    (p as f64 * FIX_ONE) as i64
}

/// A fixed-point sum back to f32 (one deterministic rounding).
#[inline(always)]
pub fn from_fix(a: i64) -> f32 {
    (a as f64 / FIX_ONE) as f32
}

/// Whether `fmt` has per-block additive minimums (`w = s·q + m`).
#[inline(always)]
pub fn has_mins(fmt: WeightFormat) -> bool {
    matches!(
        fmt,
        WeightFormat::Q4_1 | WeightFormat::Q5_1 | WeightFormat::Q4_K | WeightFormat::Q5_K
    )
}

/// Exact-mode dot of one group-format weight row against up to 4 rows.
///
/// Canonical partial product of 32-element sub-block `s` (every exact kernel,
/// SIMD or scalar, computes exactly these f32 operations):
/// * Q6_K (two 16-element scales): `d_x·(s_a·Σ_a + s_b·Σ_b)`;
/// * other formats: `d_x·(s·Σ)`, plus `+ m·(d_x·Σq_x)` with minimums.
pub fn row_dot_group_fixed(
    kernel: DotKernel,
    fmt: WeightFormat,
    row: &[u8],
    acts: &Q8Acts,
    rows: &[usize],
    out: &mut [i64; 4],
) {
    match (kernel, fmt) {
        #[cfg(target_arch = "aarch64")]
        (DotKernel::NeonDot, WeightFormat::Q4_K) => {
            // SAFETY: `dotprod` was detected at runtime.
            unsafe { kneon::q4k::<true, false>(row, acts, rows, &mut [0.0; 4], out) };
            return;
        }
        #[cfg(target_arch = "aarch64")]
        (DotKernel::NeonDot, WeightFormat::Q5_K) => {
            // SAFETY: as above.
            unsafe { kneon::q4k::<true, true>(row, acts, rows, &mut [0.0; 4], out) };
            return;
        }
        #[cfg(target_arch = "aarch64")]
        (DotKernel::NeonDot, WeightFormat::Q6_K) => {
            // SAFETY: as above.
            unsafe { kneon::q6k::<true>(row, acts, rows, &mut [0.0; 4], out) };
            return;
        }
        _ => {}
    }
    let n = fmt.block_len();
    let bb = block_bytes(fmt);
    let sub = n / QK;
    let nb = acts.cols / QK;
    let mins = has_mins(fmt);
    let mut g = GroupBlock::default();
    *out = [0; 4];
    for blk in 0..acts.cols / n {
        decode_group(fmt, &row[blk * bb..(blk + 1) * bb], &mut g);
        for (r, &i) in rows.iter().enumerate() {
            for s in 0..sub {
                let b = blk * sub + s;
                let xq = &acts.q[i * acts.cols + b * QK..i * acts.cols + (b + 1) * QK];
                let (lo, hi) = dot16x2(kernel, &g.q[s * QK..(s + 1) * QK], xq);
                let k = i * nb + b;
                let p = if fmt == WeightFormat::Q6_K {
                    acts.d[k] * (g.scale[2 * s] * lo as f32 + g.scale[2 * s + 1] * hi as f32)
                } else {
                    let t = acts.d[k] * (g.scale[2 * s] * (lo + hi) as f32);
                    if mins {
                        t + g.add[s] * acts.ds[k]
                    } else {
                        t
                    }
                };
                out[r] += to_fix(p);
            }
        }
    }
}

/// d·scale and dmin·min of the 8 sub-blocks of a Q4_K / Q5_K block (`b`
/// starts at the block's `d`).
#[inline(always)]
fn scales_k4(b: &[u8]) -> ([f32; 8], [f32; 8]) {
    let (d, dmin) = (h(b, 0), h(b, 2));
    let q = &b[4..16];
    let mut s = [0.0f32; 8];
    let mut m = [0.0f32; 8];
    for j in 0..8 {
        let (a, c) = scale_min_k4(j, q);
        s[j] = d * a as f32;
        m[j] = dmin * c as f32;
    }
    (s, m)
}

#[cfg(target_arch = "aarch64")]
mod kneon {
    use super::*;
    use std::arch::aarch64::*;

    /// Q4_K (`Q5 = false`) / Q5_K (`Q5 = true`) × up to 4 Q8 rows. Each
    /// 32-byte chunk of `qs` holds two sub-blocks (low / high nibbles); the
    /// Q5_K fifth bit comes from `qh` with a bit test. Float mode keeps two
    /// f32x4 accumulators per row and folds the minimums in once per
    /// sub-block (`m·d_x·Σq_x`, precomputed per activation block); exact mode
    /// emits the canonical per-sub-block products on the fixed-point grid.
    #[target_feature(enable = "dotprod")]
    pub unsafe fn q4k<const EXACT: bool, const Q5: bool>(
        row: &[u8],
        acts: &Q8Acts,
        rows: &[usize],
        outf: &mut [f32; 4],
        outi: &mut [i64; 4],
    ) {
        match rows.len() {
            1 => q4k_rows::<EXACT, Q5, 1>(row, acts, rows[0], outf, outi),
            2 => q4k_rows::<EXACT, Q5, 2>(row, acts, rows[0], outf, outi),
            3 => q4k_rows::<EXACT, Q5, 3>(row, acts, rows[0], outf, outi),
            _ => q4k_rows::<EXACT, Q5, 4>(row, acts, rows[0], outf, outi),
        }
    }

    #[target_feature(enable = "dotprod")]
    unsafe fn q4k_rows<const EXACT: bool, const Q5: bool, const R: usize>(
        row: &[u8],
        acts: &Q8Acts,
        i0: usize,
        outf: &mut [f32; 4],
        outi: &mut [i64; 4],
    ) {
        let bb = if Q5 { 176 } else { 144 };
        let nb = acts.cols / QK;
        let z = vdupq_n_s32(0);
        let m4 = vdupq_n_u8(0x0f);
        let s16 = vdupq_n_u8(0x10);
        let mut acc0 = [vdupq_n_f32(0.0); R];
        let mut acc1 = [vdupq_n_f32(0.0); R];
        let mut macc = [0.0f32; R];
        let mut fix = [0i64; R];
        for blk in 0..acts.cols / QK_K {
            let b = row.get_unchecked(blk * bb..(blk + 1) * bb);
            let (sc, mn) = scales_k4(b);
            let qs = b.as_ptr().add(if Q5 { 48 } else { 16 });
            let (qh0, qh1) = if Q5 {
                (vld1q_u8(b.as_ptr().add(16)), vld1q_u8(b.as_ptr().add(32)))
            } else {
                (vdupq_n_u8(0), vdupq_n_u8(0))
            };
            for j in 0..4 {
                let v0 = vld1q_u8(qs.add(32 * j));
                let v1 = vld1q_u8(qs.add(32 * j + 16));
                let (mut l0, mut l1) = (vandq_u8(v0, m4), vandq_u8(v1, m4));
                let (mut h0, mut h1) = (vshrq_n_u8(v0, 4), vshrq_n_u8(v1, 4));
                if Q5 {
                    let bl = vdupq_n_u8(1 << (2 * j));
                    let bh = vdupq_n_u8(2 << (2 * j));
                    l0 = vorrq_u8(l0, vandq_u8(vtstq_u8(qh0, bl), s16));
                    l1 = vorrq_u8(l1, vandq_u8(vtstq_u8(qh1, bl), s16));
                    h0 = vorrq_u8(h0, vandq_u8(vtstq_u8(qh0, bh), s16));
                    h1 = vorrq_u8(h1, vandq_u8(vtstq_u8(qh1, bh), s16));
                }
                let (l0, l1) = (vreinterpretq_s8_u8(l0), vreinterpretq_s8_u8(l1));
                let (h0, h1) = (vreinterpretq_s8_u8(h0), vreinterpretq_s8_u8(h1));
                let sb = blk * 8 + 2 * j;
                for r in 0..R {
                    let i = i0 + r;
                    let x = acts.q.as_ptr().add(i * acts.cols + sb * QK);
                    let il = vdotq_s32(vdotq_s32(z, l0, vld1q_s8(x)), l1, vld1q_s8(x.add(16)));
                    let ih = vdotq_s32(
                        vdotq_s32(z, h0, vld1q_s8(x.add(32))),
                        h1,
                        vld1q_s8(x.add(48)),
                    );
                    let k = i * nb + sb;
                    let (dx0, dx1) = (*acts.d.get_unchecked(k), *acts.d.get_unchecked(k + 1));
                    let (ds0, ds1) = (*acts.ds.get_unchecked(k), *acts.ds.get_unchecked(k + 1));
                    if EXACT {
                        let p0 = dx0 * (sc[2 * j] * vaddvq_s32(il) as f32) - mn[2 * j] * ds0;
                        let p1 =
                            dx1 * (sc[2 * j + 1] * vaddvq_s32(ih) as f32) - mn[2 * j + 1] * ds1;
                        fix[r] += to_fix(p0) + to_fix(p1);
                    } else {
                        acc0[r] = vfmaq_n_f32(acc0[r], vcvtq_f32_s32(il), sc[2 * j] * dx0);
                        acc1[r] = vfmaq_n_f32(acc1[r], vcvtq_f32_s32(ih), sc[2 * j + 1] * dx1);
                        macc[r] += mn[2 * j] * ds0 + mn[2 * j + 1] * ds1;
                    }
                }
            }
        }
        for r in 0..R {
            if EXACT {
                outi[r] = fix[r];
            } else {
                outf[r] = vaddvq_f32(vaddq_f32(acc0[r], acc1[r])) - macc[r];
            }
        }
    }

    /// Q6_K × up to 4 Q8 rows: the 6-bit codes of four sub-blocks are
    /// rebuilt from `ql` / `qh` with nibble masks and shifts, recentred to
    /// −32..31, and dotted per 16-element group (two `sdot` per sub-block);
    /// float mode applies the two group scales in integer arithmetic.
    #[target_feature(enable = "dotprod")]
    pub unsafe fn q6k<const EXACT: bool>(
        row: &[u8],
        acts: &Q8Acts,
        rows: &[usize],
        outf: &mut [f32; 4],
        outi: &mut [i64; 4],
    ) {
        match rows.len() {
            1 => q6k_rows::<EXACT, 1>(row, acts, rows[0], outf, outi),
            2 => q6k_rows::<EXACT, 2>(row, acts, rows[0], outf, outi),
            3 => q6k_rows::<EXACT, 3>(row, acts, rows[0], outf, outi),
            _ => q6k_rows::<EXACT, 4>(row, acts, rows[0], outf, outi),
        }
    }

    #[target_feature(enable = "dotprod")]
    unsafe fn q6k_rows<const EXACT: bool, const R: usize>(
        row: &[u8],
        acts: &Q8Acts,
        i0: usize,
        outf: &mut [f32; 4],
        outi: &mut [i64; 4],
    ) {
        let nb = acts.cols / QK;
        let z = vdupq_n_s32(0);
        let m4 = vdupq_n_u8(0x0f);
        let m3 = vdupq_n_u8(0x03);
        let off = vdupq_n_s8(32);
        let mut acc = [[vdupq_n_f32(0.0); R]; 2];
        let mut fix = [0i64; R];
        for blk in 0..acts.cols / QK_K {
            let b = row.get_unchecked(blk * 210..(blk + 1) * 210);
            let d = h(b, 208);
            let sc = b.get_unchecked(192..208);
            let mut scf = [0.0f32; 16];
            if EXACT {
                for k in 0..16 {
                    scf[k] = d * sc[k] as i8 as f32;
                }
            }
            for half in 0..2 {
                let ql = b.as_ptr().add(64 * half);
                let qh = b.as_ptr().add(128 + 32 * half);
                let (a0, a1, a2, a3) = (
                    vld1q_u8(ql),
                    vld1q_u8(ql.add(16)),
                    vld1q_u8(ql.add(32)),
                    vld1q_u8(ql.add(48)),
                );
                let (g0, g1) = (vld1q_u8(qh), vld1q_u8(qh.add(16)));
                let mk = |lo: uint8x16_t, hb: uint8x16_t| {
                    vsubq_s8(
                        vreinterpretq_s8_u8(vorrq_u8(lo, vshlq_n_u8(vandq_u8(hb, m3), 4))),
                        off,
                    )
                };
                let q = [
                    [mk(vandq_u8(a0, m4), g0), mk(vandq_u8(a1, m4), g1)],
                    [
                        mk(vandq_u8(a2, m4), vshrq_n_u8(g0, 2)),
                        mk(vandq_u8(a3, m4), vshrq_n_u8(g1, 2)),
                    ],
                    [
                        mk(vshrq_n_u8(a0, 4), vshrq_n_u8(g0, 4)),
                        mk(vshrq_n_u8(a1, 4), vshrq_n_u8(g1, 4)),
                    ],
                    [
                        mk(vshrq_n_u8(a2, 4), vshrq_n_u8(g0, 6)),
                        mk(vshrq_n_u8(a3, 4), vshrq_n_u8(g1, 6)),
                    ],
                ];
                for (t, qt) in q.iter().enumerate() {
                    let sbk = 4 * half + t;
                    let sb = blk * 8 + sbk;
                    let (ka, kb) = (2 * sbk, 2 * sbk + 1);
                    for r in 0..R {
                        let i = i0 + r;
                        let x = acts.q.as_ptr().add(i * acts.cols + sb * QK);
                        let ia = vdotq_s32(z, qt[0], vld1q_s8(x));
                        let ib = vdotq_s32(z, qt[1], vld1q_s8(x.add(16)));
                        let dx = *acts.d.get_unchecked(i * nb + sb);
                        if EXACT {
                            let p = dx
                                * (scf[ka] * vaddvq_s32(ia) as f32
                                    + scf[kb] * vaddvq_s32(ib) as f32);
                            fix[r] += to_fix(p);
                        } else {
                            let v = vmlaq_n_s32(
                                vmulq_n_s32(ia, sc[ka] as i8 as i32),
                                ib,
                                sc[kb] as i8 as i32,
                            );
                            acc[t & 1][r] = vfmaq_n_f32(acc[t & 1][r], vcvtq_f32_s32(v), d * dx);
                        }
                    }
                }
            }
        }
        for r in 0..R {
            if EXACT {
                outi[r] = fix[r];
            } else {
                outf[r] = vaddvq_f32(vaddq_f32(acc[0][r], acc[1][r]));
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod kavx2 {
    use super::*;
    use std::arch::x86_64::*;

    #[inline(always)]
    unsafe fn hsum(v: __m256) -> f32 {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        _mm_cvtss_f32(s)
    }

    /// Q4_K × up to 4 Q8 rows: unsigned nibbles go straight into `maddubs`.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn q4k(row: &[u8], acts: &Q8Acts, rows: &[usize], out: &mut [f32; 4]) {
        let nb = acts.cols / QK;
        let ones = _mm256_set1_epi16(1);
        let m4 = _mm256_set1_epi8(0x0f);
        let mut acc = [_mm256_setzero_ps(); 4];
        let mut macc = [0.0f32; 4];
        for blk in 0..acts.cols / QK_K {
            let b = &row[blk * 144..(blk + 1) * 144];
            let (sc, mn) = scales_k4(b);
            for j in 0..4 {
                let v = _mm256_loadu_si256(b.as_ptr().add(16 + 32 * j) as *const __m256i);
                let lo = _mm256_and_si256(v, m4);
                let hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), m4);
                let sb = blk * 8 + 2 * j;
                for (r, &i) in rows.iter().enumerate() {
                    let x = acts.q.as_ptr().add(i * acts.cols + sb * QK);
                    let xl = _mm256_loadu_si256(x as *const __m256i);
                    let xh = _mm256_loadu_si256(x.add(32) as *const __m256i);
                    let il = _mm256_madd_epi16(_mm256_maddubs_epi16(lo, xl), ones);
                    let ih = _mm256_madd_epi16(_mm256_maddubs_epi16(hi, xh), ones);
                    let k = i * nb + sb;
                    acc[r] = _mm256_fmadd_ps(
                        _mm256_set1_ps(sc[2 * j] * acts.d[k]),
                        _mm256_cvtepi32_ps(il),
                        acc[r],
                    );
                    acc[r] = _mm256_fmadd_ps(
                        _mm256_set1_ps(sc[2 * j + 1] * acts.d[k + 1]),
                        _mm256_cvtepi32_ps(ih),
                        acc[r],
                    );
                    macc[r] += mn[2 * j] * acts.ds[k] + mn[2 * j + 1] * acts.ds[k + 1];
                }
            }
        }
        for r in 0..rows.len() {
            out[r] = hsum(acc[r]) - macc[r];
        }
    }

    /// Q6_K × up to 4 Q8 rows: codes rebuilt with byte masks / 16-bit shifts
    /// (masked first, so no bits cross bytes), recentred, sign trick for
    /// `maddubs`, group scales applied per 128-bit half.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn q6k(row: &[u8], acts: &Q8Acts, rows: &[usize], out: &mut [f32; 4]) {
        let nb = acts.cols / QK;
        let ones = _mm256_set1_epi16(1);
        let m4 = _mm256_set1_epi8(0x0f);
        let off = _mm256_set1_epi8(32);
        let mut acc = [_mm256_setzero_ps(); 4];
        for blk in 0..acts.cols / QK_K {
            let b = &row[blk * 210..(blk + 1) * 210];
            let d = h(b, 208);
            let sc = &b[192..208];
            for half in 0..2 {
                let p = b.as_ptr();
                let a01 = _mm256_loadu_si256(p.add(64 * half) as *const __m256i);
                let a23 = _mm256_loadu_si256(p.add(64 * half + 32) as *const __m256i);
                let hq = _mm256_loadu_si256(p.add(128 + 32 * half) as *const __m256i);
                let q = [
                    _mm256_or_si256(
                        _mm256_and_si256(a01, m4),
                        _mm256_slli_epi16(_mm256_and_si256(hq, _mm256_set1_epi8(0x03)), 4),
                    ),
                    _mm256_or_si256(
                        _mm256_and_si256(a23, m4),
                        _mm256_slli_epi16(_mm256_and_si256(hq, _mm256_set1_epi8(0x0c)), 2),
                    ),
                    _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16(a01, 4), m4),
                        _mm256_and_si256(hq, _mm256_set1_epi8(0x30)),
                    ),
                    _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16(a23, 4), m4),
                        _mm256_srli_epi16(_mm256_and_si256(hq, _mm256_set1_epi8(0xc0u8 as i8)), 2),
                    ),
                ];
                for (t, qt) in q.iter().enumerate() {
                    let w = _mm256_sub_epi8(*qt, off);
                    let aw = _mm256_sign_epi8(w, w);
                    let sbk = 4 * half + t;
                    let sb = blk * 8 + sbk;
                    let scv = _mm256_setr_epi32(
                        sc[2 * sbk] as i8 as i32,
                        sc[2 * sbk] as i8 as i32,
                        sc[2 * sbk] as i8 as i32,
                        sc[2 * sbk] as i8 as i32,
                        sc[2 * sbk + 1] as i8 as i32,
                        sc[2 * sbk + 1] as i8 as i32,
                        sc[2 * sbk + 1] as i8 as i32,
                        sc[2 * sbk + 1] as i8 as i32,
                    );
                    for (r, &i) in rows.iter().enumerate() {
                        let x = _mm256_loadu_si256(
                            acts.q.as_ptr().add(i * acts.cols + sb * QK) as *const __m256i
                        );
                        let p32 = _mm256_madd_epi16(
                            _mm256_maddubs_epi16(aw, _mm256_sign_epi8(x, w)),
                            ones,
                        );
                        let v = _mm256_mullo_epi32(p32, scv);
                        acc[r] = _mm256_fmadd_ps(
                            _mm256_set1_ps(d * acts.d[i * nb + sb]),
                            _mm256_cvtepi32_ps(v),
                            acc[r],
                        );
                    }
                }
            }
        }
        for r in 0..rows.len() {
            out[r] = hsum(acc[r]);
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    #[target_feature(enable = "dotprod")]
    pub unsafe fn dot16x2_sdot(w: *const i8, x: *const i8) -> (i32, i32) {
        let z = vdupq_n_s32(0);
        let lo = vdotq_s32(z, vld1q_s8(w), vld1q_s8(x));
        let hi = vdotq_s32(z, vld1q_s8(w.add(16)), vld1q_s8(x.add(16)));
        (vaddvq_s32(lo), vaddvq_s32(hi))
    }

    pub unsafe fn dot16x2_mull(w: *const i8, x: *const i8) -> (i32, i32) {
        let f = |w: int8x16_t, x: int8x16_t| {
            let p = vpaddlq_s16(vmull_s8(vget_low_s8(w), vget_low_s8(x)));
            vaddvq_s32(vpadalq_s16(p, vmull_high_s8(w, x)))
        };
        (
            f(vld1q_s8(w), vld1q_s8(x)),
            f(vld1q_s8(w.add(16)), vld1q_s8(x.add(16))),
        )
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2")]
    pub unsafe fn dot16x2(w: *const i8, x: *const i8) -> (i32, i32) {
        let wv = _mm256_loadu_si256(w as *const __m256i);
        let xv = _mm256_loadu_si256(x as *const __m256i);
        // |w| · (x · sign w): maddubs needs an unsigned operand; |w| ≤ 128
        let p16 = _mm256_maddubs_epi16(_mm256_sign_epi8(wv, wv), _mm256_sign_epi8(xv, wv));
        let p32 = _mm256_madd_epi16(p16, _mm256_set1_epi16(1));
        let hsum = |v: __m128i| {
            let v = _mm_add_epi32(v, _mm_shuffle_epi32(v, 0b01_00_11_10));
            let v = _mm_add_epi32(v, _mm_shuffle_epi32(v, 0b10_11_00_01));
            _mm_cvtsi128_si32(v)
        };
        (
            hsum(_mm256_castsi256_si128(p32)),
            hsum(_mm256_extracti128_si256(p32, 1)),
        )
    }
}
