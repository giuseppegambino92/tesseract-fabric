//! Core tensor runtime: row-major 2-D tensors, SIMD-friendly dot/GEMM kernels,
//! RMSNorm, RoPE, SwiGLU, softmax, grouped-query attention (reference path),
//! int8 block-quantized GEMM, IEEE-754 binary16 codec and a deterministic RNG.
//!
//! Everything here is plain Rust; the hot loops are written with independent
//! accumulator lanes so LLVM auto-vectorises them to NEON / AVX2.

pub mod accel;
pub mod cuda;
pub mod determinism;
pub mod eigen;
pub mod kquant;
pub mod pool;
pub mod quant;

// ---------------------------------------------------------------------------
// Deterministic RNG (SplitMix64 seeding + xoshiro256**)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Rng {
            s: [next(), next(), next(), next()],
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform in [0, 1).
    #[inline]
    pub fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 * (1.0 / (1u64 << 24) as f32)
    }

    #[inline]
    pub fn uniform_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Standard normal via Box-Muller.
    pub fn normal(&mut self) -> f32 {
        let u1 = self.uniform_f64().max(1e-300);
        let u2 = self.uniform_f64();
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    pub fn fill_normal(&mut self, out: &mut [f32], std: f32) {
        for v in out.iter_mut() {
            *v = self.normal() * std;
        }
    }
}

// ---------------------------------------------------------------------------
// IEEE-754 binary16 codec (round-to-nearest-even), used on the wire.
// ---------------------------------------------------------------------------

pub fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;
    if exp == 0xff {
        // Inf / NaN
        return sign | 0x7c00 | if mant != 0 { 0x0200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = mant | 0x0080_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rem = m & ((1u32 << shift) - 1);
        let mut r = m >> shift;
        if rem > half || (rem == half && (r & 1) == 1) {
            r += 1;
        }
        return sign | r as u16;
    }
    let mut r = ((e as u32) << 10) | (mant >> 13);
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (r & 1) == 1) {
        r += 1; // may carry into exponent, which is the correct behaviour
    }
    sign | r as u16
}

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x03ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // subnormal: normalise
            let mut e = 127 - 15 + 1;
            let mut m = mant;
            while m & 0x0400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((m & 0x03ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

pub fn encode_f16_slice(xs: &[f32], out: &mut Vec<u8>) {
    out.reserve(xs.len() * 2);
    for &x in xs {
        out.extend_from_slice(&f32_to_f16(x).to_le_bytes());
    }
}

pub fn decode_f16_slice(bytes: &[u8], out: &mut [f32]) {
    for (i, o) in out.iter_mut().enumerate() {
        *o = f16_to_f32(u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]));
    }
}

// ---------------------------------------------------------------------------
// Vector kernels
// ---------------------------------------------------------------------------

/// Dot product. On aarch64: explicit NEON FMA with 4 independent vector
/// accumulators (16 lanes); elsewhere 8 scalar lanes that LLVM vectorises.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: both slices have the same length; all loads stay in bounds.
        unsafe { neon::dot(a, b) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if x86::has_avx2_fma() {
            // SAFETY: AVX2 + FMA checked at runtime; equal lengths.
            return unsafe { x86::dot(a, b) };
        }
        dot_portable(a, b)
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        dot_portable(a, b)
    }
}

/// Portable 8-lane dot product.
#[inline]
pub fn dot_portable(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let chunks = n / 8;
    let mut acc = [0.0f32; 8];
    for c in 0..chunks {
        let aa = &a[c * 8..c * 8 + 8];
        let bb = &b[c * 8..c * 8 + 8];
        for l in 0..8 {
            acc[l] += aa[l] * bb[l];
        }
    }
    let mut s = (acc[0] + acc[4]) + (acc[1] + acc[5]) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
    for i in chunks * 8..n {
        s += a[i] * b[i];
    }
    s
}

/// y += alpha * x
#[inline]
pub fn axpy(alpha: f32, x: &[f32], y: &mut [f32]) {
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi += alpha * xi;
    }
}

pub fn l2_norm(x: &[f32]) -> f32 {
    x.iter()
        .map(|v| (*v as f64) * (*v as f64))
        .sum::<f64>()
        .sqrt() as f32
}

pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let mut ab = 0.0f64;
    let mut aa = 0.0f64;
    let mut bb = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        ab += *x as f64 * *y as f64;
        aa += *x as f64 * *x as f64;
        bb += *y as f64 * *y as f64;
    }
    if aa == 0.0 || bb == 0.0 {
        return if aa == bb { 1.0 } else { 0.0 };
    }
    ab / (aa.sqrt() * bb.sqrt())
}

pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// In-place numerically stable softmax.
pub fn softmax_inplace(x: &mut [f32]) {
    let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - m).exp();
        s += *v;
    }
    let inv = 1.0 / s;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// RMSNorm: y = x / sqrt(mean(x^2) + eps) * w
pub fn rms_norm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let ss = dot(x, x) / x.len() as f32;
    let inv = 1.0 / (ss + eps).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * inv * w[i];
    }
}

/// Apply rotary position embedding (half-split / NeoX layout) to one head vector.
pub fn rope_inplace(v: &mut [f32], pos: usize, theta: f32) {
    let d = v.len();
    let half = d / 2;
    for i in 0..half {
        let freq = 1.0 / theta.powf((2 * i) as f32 / d as f32);
        let ang = pos as f32 * freq;
        let (s, c) = ang.sin_cos();
        let a = v[i];
        let b = v[i + half];
        v[i] = a * c - b * s;
        v[i + half] = a * s + b * c;
    }
}

/// RoPE frequency scaling read from a model file.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct RopeScaling {
    /// Per-frequency divisors (Llama 3 `rope_freqs.weight`); empty = none.
    pub freq_factors: Vec<f32>,
    /// Linear position interpolation factor (0 or 1 = none).
    pub linear: f32,
    /// YaRN: (factor, original context length); llama.cpp's defaults
    /// β_fast = 32, β_slow = 1.
    pub yarn: Option<(f32, u32)>,
}

impl RopeScaling {
    pub fn is_none(&self) -> bool {
        self.freq_factors.is_empty()
            && (self.linear == 0.0 || self.linear == 1.0)
            && self.yarn.is_none()
    }
}

/// Precomputed RoPE tables for a given head dimension (avoids powf in hot loop).
#[derive(Clone, Debug)]
pub struct RopeTable {
    pub inv_freq: Vec<f32>,
    /// Magnitude correction of cos / sin (YaRN; 1 otherwise).
    pub mscale: f32,
}

impl RopeTable {
    pub fn new(d_head: usize, theta: f32) -> Self {
        let half = d_head / 2;
        RopeTable {
            inv_freq: (0..half)
                .map(|i| 1.0 / theta.powf((2 * i) as f32 / d_head as f32))
                .collect(),
            mscale: 1.0,
        }
    }

    /// RoPE with frequency scaling (identical to [`RopeTable::new`] when
    /// `scaling` is empty). Follows ggml's `rope_yarn` / freq-factor rules.
    pub fn with_scaling(d_head: usize, theta: f32, scaling: &RopeScaling) -> Self {
        let mut t = Self::new(d_head, theta);
        for (f, ff) in t.inv_freq.iter_mut().zip(&scaling.freq_factors) {
            *f /= ff;
        }
        if scaling.linear > 0.0 && scaling.linear != 1.0 {
            for f in t.inv_freq.iter_mut() {
                *f /= scaling.linear;
            }
        }
        if let Some((factor, n_ctx_orig)) = scaling.yarn {
            let fs = 1.0 / factor;
            let corr = |beta: f32| {
                d_head as f32 * (n_ctx_orig as f32 / (beta * 2.0 * std::f32::consts::PI)).ln()
                    / (2.0 * theta.ln())
            };
            let lo = corr(32.0).floor().max(0.0);
            let hi = corr(1.0).ceil().min(d_head as f32 - 1.0);
            for (i, f) in t.inv_freq.iter_mut().enumerate() {
                let y = (i as f32 - lo) / (hi - lo).max(0.001);
                let ramp = 1.0 - y.clamp(0.0, 1.0);
                *f *= fs * (1.0 - ramp) + ramp;
            }
            t.mscale = 1.0 + 0.1 * (1.0 / fs).ln();
        }
        t
    }

    pub fn apply(&self, v: &mut [f32], pos: usize) {
        let half = self.inv_freq.len();
        let m = self.mscale;
        for i in 0..half {
            let (s, c) = (pos as f32 * self.inv_freq[i]).sin_cos();
            let (s, c) = if m == 1.0 { (s, c) } else { (s * m, c * m) };
            let a = v[i];
            let b = v[i + half];
            v[i] = a * c - b * s;
            v[i + half] = a * s + b * c;
        }
    }
}

// ---------------------------------------------------------------------------
// Tensor2D: row-major [rows, cols]
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor2D {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Tensor2D {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Tensor2D {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), rows * cols, "Tensor2D shape mismatch");
        Tensor2D { rows, cols, data }
    }
    pub fn randn(rows: usize, cols: usize, std: f32, rng: &mut Rng) -> Self {
        let mut t = Self::zeros(rows, cols);
        rng.fill_normal(&mut t.data, std);
        t
    }
    #[inline]
    pub fn row(&self, r: usize) -> &[f32] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }
    #[inline]
    pub fn row_mut(&mut self, r: usize) -> &mut [f32] {
        &mut self.data[r * self.cols..(r + 1) * self.cols]
    }
    pub fn bytes(&self) -> usize {
        self.data.len() * 4
    }
}

/// Tensor3D: [d0, d1, d2] row-major (e.g. tokens x heads x head_dim).
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor3D {
    pub d0: usize,
    pub d1: usize,
    pub d2: usize,
    pub data: Vec<f32>,
}

impl Tensor3D {
    pub fn zeros(d0: usize, d1: usize, d2: usize) -> Self {
        Tensor3D {
            d0,
            d1,
            d2,
            data: vec![0.0; d0 * d1 * d2],
        }
    }
    #[inline]
    pub fn at(&self, i: usize, j: usize) -> &[f32] {
        let o = (i * self.d1 + j) * self.d2;
        &self.data[o..o + self.d2]
    }
    #[inline]
    pub fn at_mut(&mut self, i: usize, j: usize) -> &mut [f32] {
        let o = (i * self.d1 + j) * self.d2;
        &mut self.data[o..o + self.d2]
    }
}

/// Linear layer weight stored as [out_features, in_features] so every output's
/// weights are contiguous (dot-product friendly, rows reused across a batch).
///
/// `q` holds an optional zero-copy quantized / half-precision view (GGUF or
/// SafeTensors weights): when present, `w` keeps only the shape (`data` is
/// empty) and every product runs on the compressed bytes.
#[derive(Clone, Debug)]
pub struct Linear {
    pub w: Tensor2D,
    pub q: Option<quant::QuantView>,
    /// Exact accumulation: every output is a fixed-point sum of canonical
    /// per-block products ([`quant::gemm_fixed`]), independent of tiling and
    /// of how a row-parallel split of the layer is summed. Set on the
    /// row-parallel projections (attention output, FFN down) of exact models.
    pub exact: bool,
}

impl Linear {
    pub fn new(w: Tensor2D) -> Self {
        Linear {
            w,
            q: None,
            exact: false,
        }
    }
    /// The same layer with exact accumulation on or off.
    pub fn with_exact(mut self, exact: bool) -> Self {
        self.exact = exact;
        self
    }
    /// A linear layer over a quantized / half-precision view.
    pub fn quantized(v: quant::QuantView) -> Self {
        Linear {
            w: Tensor2D {
                rows: v.rows,
                cols: v.cols,
                data: Vec::new(),
            },
            q: Some(v),
            exact: false,
        }
    }
    pub fn new_random(out_f: usize, in_f: usize, std: f32, rng: &mut Rng) -> Self {
        Linear::new(Tensor2D::randn(out_f, in_f, std, rng))
    }
    pub fn out_features(&self) -> usize {
        self.w.rows
    }
    pub fn in_features(&self) -> usize {
        self.w.cols
    }
    /// Storage format of the weights.
    pub fn format(&self) -> quant::WeightFormat {
        self.q
            .as_ref()
            .map(|v| v.fmt)
            .unwrap_or(quant::WeightFormat::F32)
    }
    /// Bytes of weight storage.
    pub fn bytes(&self) -> usize {
        match &self.q {
            Some(v) => v.bytes(),
            None => self.w.bytes(),
        }
    }
    /// The weights as a dense f32 matrix (dequantized copy for quantized
    /// layers; borrowed otherwise). Used by init-time analyses only.
    pub fn dense(&self) -> std::borrow::Cow<'_, Tensor2D> {
        match &self.q {
            Some(v) => std::borrow::Cow::Owned(v.dequantize()),
            None => std::borrow::Cow::Borrowed(&self.w),
        }
    }

    /// y[m, out] = x[m, in] · Wᵀ  — batched GEMM. Each weight row is loaded
    /// once per tile of input rows, so a tree batch of N candidates costs far
    /// less than N separate memory-bound matrix-vector passes.
    pub fn forward(&self, x: &Tensor2D) -> Tensor2D {
        let mut y = Tensor2D::zeros(x.rows, self.w.rows);
        self.forward_into(x, &mut y);
        y
    }

    pub fn forward_into(&self, x: &Tensor2D, y: &mut Tensor2D) {
        if self.exact {
            for (o, a) in y.data.iter_mut().zip(self.forward_fixed(x)) {
                *o = kquant::from_fix(a);
            }
            return;
        }
        match &self.q {
            Some(v) => quant::gemm_quant(x, v, y),
            None => gemm_xwt(x, &self.w, y),
        }
    }

    /// Exact-mode product as fixed-point sums (`x.rows × out` `i64`, grid
    /// 2^-32): the partial an all-reduce sums exactly before rounding.
    pub fn forward_fixed(&self, x: &Tensor2D) -> Vec<i64> {
        match &self.q {
            Some(v) => quant::gemm_fixed(x, Some(v), None),
            None => quant::gemm_fixed(x, None, Some(&self.w)),
        }
    }

    /// Single-vector forward.
    pub fn forward_vec(&self, x: &[f32], y: &mut [f32]) {
        if self.q.is_some() {
            let xt = Tensor2D::from_vec(1, x.len(), x.to_vec());
            y.copy_from_slice(&self.forward(&xt).data);
            return;
        }
        for (o, yo) in y.iter_mut().enumerate() {
            *yo = dot(self.w.row(o), x);
        }
    }

    /// y (m × (end − start)) for output rows [start, end) — the lane pool's
    /// per-stripe entry point (zero copy for both storage kinds).
    pub fn forward_rows(&self, x: &Tensor2D, start: usize, end: usize, y: &mut [f32]) {
        // Accelerate (AMX) for MoE expert stripes: dense f32 except 2–3 rows,
        // quantized stripes (dequantized tile + SGEMM) from
        // `ACCEL_QUANT_MIN_ROWS` rows
        if accel::accelerate_active() {
            let k = self.w.cols;
            match &self.q {
                // measured: AMX loses only at 2–3 rows (examples/accel_moe.rs)
                None if !(2..4).contains(&x.rows)
                    && accel::sgemm_xwt(
                        &x.data,
                        x.rows,
                        k,
                        &self.w.data[start * k..end * k],
                        end - start,
                        y,
                        end - start,
                    ) =>
                {
                    return
                }
                Some(v) if x.rows >= accel::ACCEL_QUANT_MIN_ROWS => {
                    let mut t = Tensor2D::zeros(x.rows, end - start);
                    accel::gemm_dequant_tiles(x, &v.slice_rows(start, end), &mut t);
                    y.copy_from_slice(&t.data);
                    return;
                }
                _ => {}
            }
        }
        match &self.q {
            Some(v) => {
                let mut t = Tensor2D::zeros(x.rows, end - start);
                quant::gemm_quant_with(
                    x,
                    &v.slice_rows(start, end),
                    &mut t,
                    quant::active_kernel(),
                );
                y.copy_from_slice(&t.data);
            }
            None => {
                let k = self.w.cols;
                matmul_rows(x, &self.w.data[start * k..end * k], end - start, y);
            }
        }
    }

    /// madvise(WILLNEED) the weight bytes of a file-backed layer; under the
    /// CUDA backend, also upload them to VRAM now (stage start-up) instead
    /// of on first use.
    pub fn prefetch(&self) {
        if let Some(v) = &self.q {
            v.prefetch();
            if accel::cuda_active() {
                cuda::preload(v);
            }
        }
    }

    /// Restrict to a contiguous range of output rows (column-parallel split).
    pub fn slice_rows(&self, start: usize, end: usize) -> Linear {
        if let Some(v) = &self.q {
            return Linear::quantized(v.slice_rows(start, end)).with_exact(self.exact);
        }
        let c = self.w.cols;
        Linear::new(Tensor2D::from_vec(
            end - start,
            c,
            self.w.data[start * c..end * c].to_vec(),
        ))
        .with_exact(self.exact)
    }

    /// Restrict to a contiguous range of input columns (row-parallel split).
    /// Quantized views stay zero-copy when the bounds are block aligned;
    /// otherwise the slice is materialised in f32.
    pub fn slice_cols(&self, start: usize, end: usize) -> Linear {
        if let Some(v) = &self.q {
            if v.cols_aligned(start, end) {
                return Linear::quantized(v.slice_cols(start, end)).with_exact(self.exact);
            }
            return Linear::new(self.dense().into_owned())
                .with_exact(self.exact)
                .slice_cols(start, end);
        }
        let mut w = Tensor2D::zeros(self.w.rows, end - start);
        for r in 0..self.w.rows {
            w.row_mut(r).copy_from_slice(&self.w.row(r)[start..end]);
        }
        Linear::new(w).with_exact(self.exact)
    }
}

/// A token-embedding table: a shared dense matrix (synthetic models) or a
/// zero-copy view into a model file whose rows are decoded on lookup.
#[derive(Clone, Debug)]
pub enum Embedding {
    Dense(std::sync::Arc<Tensor2D>),
    Quant(quant::QuantView),
}

impl From<Tensor2D> for Embedding {
    fn from(t: Tensor2D) -> Self {
        Embedding::Dense(std::sync::Arc::new(t))
    }
}

impl Embedding {
    pub fn rows(&self) -> usize {
        match self {
            Embedding::Dense(t) => t.rows,
            Embedding::Quant(v) => v.rows,
        }
    }
    pub fn cols(&self) -> usize {
        match self {
            Embedding::Dense(t) => t.cols,
            Embedding::Quant(v) => v.cols,
        }
    }
    /// Row `t` (borrowed when dense, decoded otherwise).
    pub fn row(&self, t: usize) -> std::borrow::Cow<'_, [f32]> {
        match self {
            Embedding::Dense(m) => std::borrow::Cow::Borrowed(m.row(t)),
            Embedding::Quant(v) => {
                let mut r = vec![0.0f32; v.cols];
                v.decode_row_into(t, &mut r);
                std::borrow::Cow::Owned(r)
            }
        }
    }
    /// Storage bytes.
    pub fn bytes(&self) -> usize {
        match self {
            Embedding::Dense(t) => t.bytes(),
            Embedding::Quant(v) => v.bytes(),
        }
    }
    /// The whole table as f32 (copy for file-backed tables).
    pub fn to_dense(&self) -> Tensor2D {
        match self {
            Embedding::Dense(t) => (**t).clone(),
            Embedding::Quant(v) => v.dequantize(),
        }
    }
}

/// Below this many multiply-accumulates a GEMM stays on one lane.
const LANE_MIN_MACS: usize = 1 << 17;

/// y = x · Wᵀ where x: [m, k], W: [n, k], y: [m, n]. On a node thread with a
/// lane pool (installed identically for every engine) large GEMMs stripe
/// their output features across the lanes; every output element is the same
/// tiled dot product either way.
pub fn gemm_xwt(x: &Tensor2D, w: &Tensor2D, y: &mut Tensor2D) {
    let (m, k, n) = (x.rows, x.cols, w.rows);
    if accel::accelerate_active() && accel::gemm_dense(x, w, y) {
        return;
    }
    if m * n * k >= LANE_MIN_MACS {
        if let Some(pool) = pool::current() {
            let lanes = pool.lanes();
            let yp = pool::SyncPtr(y.data.as_mut_ptr());
            pool.run(lanes, &|t| {
                let (j0, j1) = (n * t / lanes, n * (t + 1) / lanes);
                let nr = j1 - j0;
                if nr == 0 {
                    return;
                }
                let mut part = vec![0.0f32; m * nr];
                matmul_rows(x, &w.data[j0 * k..j1 * k], nr, &mut part);
                for i in 0..m {
                    for j in 0..nr {
                        // SAFETY: lanes write disjoint column stripes of y.
                        unsafe { *yp.ptr().add(i * n + j0 + j) = part[i * nr + j] };
                    }
                }
            });
            return;
        }
    }
    matmul_tiled_batch(x, w, y)
}

/// Weight-reuse register-tiled multi-token GEMM.
///
/// Rows of `x` (speculative tree tokens) are processed in register tiles of
/// R rows × C weight rows: every 4-wide chunk of a weight row is loaded once
/// and fused-multiply-added into all R token accumulators before moving on, so
/// a batch of B tokens streams the weight matrix ⌈B/R⌉ times instead of B
/// times. Tile shapes are matched to the batch (8×3, 6×4, 4×4, 2×8, 1×8 on
/// aarch64 — at most 24 live vector accumulators of the 32 NEON registers).
/// Leftover rows are zero-padded into a full tile (cheaper than narrow tiles).
pub fn matmul_tiled_batch(x: &Tensor2D, w: &Tensor2D, y: &mut Tensor2D) {
    let (m, k, n) = (x.rows, x.cols, w.rows);
    assert_eq!(w.cols, k, "gemm inner dim mismatch");
    assert_eq!(y.rows, m);
    assert_eq!(y.cols, n);
    #[cfg(target_arch = "aarch64")]
    {
        let mut i = 0;
        while i < m {
            let rem = m - i;
            i += match rem {
                r if r >= 8 => tile_block::<8, 3>(x, w, y, i),
                5..=7 => tile_block::<6, 4>(x, w, y, i),
                3..=4 => tile_block::<4, 4>(x, w, y, i),
                2 => tile_block::<2, 8>(x, w, y, i),
                _ => tile_block::<1, 8>(x, w, y, i),
            };
        }
    }
    #[cfg(target_arch = "x86_64")]
    if x86::has_avx2_fma() {
        let mut i = 0;
        while i < m {
            i += x86_tiles(x, &w.data, n, &mut y.data, i);
        }
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut i = 0;
        while i + 4 <= m {
            tile_rows_portable::<4>(x, w, y, i);
            i += 4;
        }
        if m - i >= 2 {
            let rem = m - i;
            let mut xp = Tensor2D::zeros(4, k);
            xp.data[..rem * k].copy_from_slice(&x.data[i * k..m * k]);
            let mut yp = Tensor2D::zeros(4, n);
            tile_rows_portable::<4>(&xp, w, &mut yp, 0);
            y.data[i * n..m * n].copy_from_slice(&yp.data[..rem * n]);
            i = m;
        }
        while i < m {
            tile_rows_portable::<1>(x, w, y, i);
            i += 1;
        }
    }
}

/// y (m × n_rows) = x · W[rows]ᵀ for a contiguous row range of a row-major
/// weight matrix, given as its raw slice (`n_rows × k`). Used by the lane
/// pool to stripe one GEMM's output features across cores without copying
/// weights.
pub fn matmul_rows(x: &Tensor2D, w_rows: &[f32], n_rows: usize, y: &mut [f32]) {
    let (m, k) = (x.rows, x.cols);
    assert_eq!(w_rows.len(), n_rows * k);
    assert_eq!(y.len(), m * n_rows);
    #[cfg(target_arch = "aarch64")]
    {
        let mut i = 0;
        while i < m {
            let rem = m - i;
            i += match rem {
                r if r >= 8 => tile_block_raw::<8, 3>(x, w_rows, n_rows, y, i),
                5..=7 => tile_block_raw::<6, 4>(x, w_rows, n_rows, y, i),
                3..=4 => tile_block_raw::<4, 4>(x, w_rows, n_rows, y, i),
                2 => tile_block_raw::<2, 8>(x, w_rows, n_rows, y, i),
                _ => tile_block_raw::<1, 8>(x, w_rows, n_rows, y, i),
            };
        }
    }
    #[cfg(target_arch = "x86_64")]
    if x86::has_avx2_fma() {
        let mut i = 0;
        while i < m {
            i += x86_tiles(x, w_rows, n_rows, y, i);
        }
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..m {
            for j in 0..n_rows {
                y[i * n_rows + j] = dot(x.row(i), &w_rows[j * k..(j + 1) * k]);
            }
        }
    }
}

/// One AVX2 tile step at row `i0`: picks the tile shape for the rows left
/// (16 ymm registers: R·C accumulators + C weight vectors + 1 input vector)
/// and returns the rows consumed.
#[cfg(target_arch = "x86_64")]
fn x86_tiles(x: &Tensor2D, w: &[f32], n: usize, y: &mut [f32], i0: usize) -> usize {
    match x.rows - i0 {
        r if r >= 7 => x86_tile_block::<4, 3>(x, w, n, y, i0),
        5..=6 => x86_tile_block::<6, 2>(x, w, n, y, i0),
        3..=4 => x86_tile_block::<4, 3>(x, w, n, y, i0),
        2 => x86_tile_block::<2, 4>(x, w, n, y, i0),
        _ => x86_tile_block::<1, 4>(x, w, n, y, i0),
    }
}

/// R token rows × C weight rows of `y = x · wᵀ` with AVX2 tiles (zero-pads
/// missing rows); returns the real rows consumed.
#[cfg(target_arch = "x86_64")]
fn x86_tile_block<const R: usize, const C: usize>(
    x: &Tensor2D,
    w: &[f32],
    n: usize,
    y: &mut [f32],
    i0: usize,
) -> usize {
    let (m, k) = (x.rows, x.cols);
    let r = (m - i0).min(R);
    let padded;
    let xs: &[f32] = if r == R {
        &x.data[i0 * k..(i0 + R) * k]
    } else {
        let mut p = vec![0.0f32; R * k];
        p[..r * k].copy_from_slice(&x.data[i0 * k..(i0 + r) * k]);
        padded = p;
        &padded
    };
    let xr: [*const f32; R] = std::array::from_fn(|a| xs[a * k..].as_ptr());
    let mut j = 0;
    while j + C <= n {
        let wr: [*const f32; C] = std::array::from_fn(|b| w[(j + b) * k..].as_ptr());
        // SAFETY: AVX2 + FMA were checked by the caller; every pointer
        // addresses a full row of k floats.
        let o = unsafe { x86::tile::<R, C>(xr, wr, k) };
        for a in 0..r {
            y[(i0 + a) * n + j..(i0 + a) * n + j + C].copy_from_slice(&o[a]);
        }
        j += C;
    }
    while j < n {
        let wrow = &w[j * k..(j + 1) * k];
        for a in 0..r {
            y[(i0 + a) * n + j] = dot(&xs[a * k..(a + 1) * k], wrow);
        }
        j += 1;
    }
    r
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    /// AVX2 and FMA available on this CPU (checked once).
    pub fn has_avx2_fma() -> bool {
        static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *F.get_or_init(|| is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"))
    }

    #[inline(always)]
    unsafe fn hsum(v: __m256) -> f32 {
        let q = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        let d = _mm_add_ps(q, _mm_movehl_ps(q, q));
        _mm_cvtss_f32(_mm_add_ss(d, _mm_shuffle_ps(d, d, 1)))
    }

    /// AVX2 dot product with four independent FMA chains.
    ///
    /// # Safety
    /// AVX2 + FMA must be available; `a` and `b` must have equal length.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        let k = a.len();
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut acc = [_mm256_setzero_ps(); 4];
        let mut c = 0;
        while c + 32 <= k {
            for (u, s) in acc.iter_mut().enumerate() {
                let o = c + 8 * u;
                *s = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(o)), _mm256_loadu_ps(pb.add(o)), *s);
            }
            c += 32;
        }
        while c + 8 <= k {
            acc[0] = _mm256_fmadd_ps(
                _mm256_loadu_ps(pa.add(c)),
                _mm256_loadu_ps(pb.add(c)),
                acc[0],
            );
            c += 8;
        }
        let mut s = hsum(_mm256_add_ps(
            _mm256_add_ps(acc[0], acc[1]),
            _mm256_add_ps(acc[2], acc[3]),
        ));
        while c < k {
            s += a[c] * b[c];
            c += 1;
        }
        s
    }

    /// R token rows × C weight rows register tile over the full k dimension.
    ///
    /// # Safety
    /// AVX2 + FMA must be available; every pointer must address at least `k`
    /// readable floats.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn tile<const R: usize, const C: usize>(
        x: [*const f32; R],
        w: [*const f32; C],
        k: usize,
    ) -> [[f32; C]; R] {
        let mut acc = [[_mm256_setzero_ps(); C]; R];
        let mut c = 0;
        while c + 8 <= k {
            let mut wv = [_mm256_setzero_ps(); C];
            for b in 0..C {
                wv[b] = _mm256_loadu_ps(w[b].add(c));
            }
            for a in 0..R {
                let xv = _mm256_loadu_ps(x[a].add(c));
                for b in 0..C {
                    acc[a][b] = _mm256_fmadd_ps(xv, wv[b], acc[a][b]);
                }
            }
            c += 8;
        }
        let mut o = [[0.0; C]; R];
        for a in 0..R {
            for b in 0..C {
                let mut s = hsum(acc[a][b]);
                let mut t = c;
                while t < k {
                    s += *x[a].add(t) * *w[b].add(t);
                    t += 1;
                }
                o[a][b] = s;
            }
        }
        o
    }
}

/// Run one R-row tile starting at row `i0` (zero-padding missing rows);
/// returns the number of real rows consumed.
#[cfg(target_arch = "aarch64")]
fn tile_block<const R: usize, const C: usize>(
    x: &Tensor2D,
    w: &Tensor2D,
    y: &mut Tensor2D,
    i0: usize,
) -> usize {
    tile_block_raw::<R, C>(x, &w.data, w.rows, &mut y.data, i0)
}

/// `tile_block` over a raw row-major weight slice (`n × k`) and output
/// (`m × n`).
#[cfg(target_arch = "aarch64")]
fn tile_block_raw<const R: usize, const C: usize>(
    x: &Tensor2D,
    w: &[f32],
    n: usize,
    y: &mut [f32],
    i0: usize,
) -> usize {
    let (m, k) = (x.rows, x.cols);
    let r = (m - i0).min(R);
    let padded;
    let xs: &[f32] = if r == R {
        &x.data[i0 * k..(i0 + R) * k]
    } else {
        let mut p = vec![0.0f32; R * k];
        p[..r * k].copy_from_slice(&x.data[i0 * k..(i0 + r) * k]);
        padded = p;
        &padded
    };
    let xr: [*const f32; R] = std::array::from_fn(|a| xs[a * k..].as_ptr());
    let mut j = 0;
    while j + C <= n {
        let wr: [*const f32; C] = std::array::from_fn(|b| w[(j + b) * k..].as_ptr());
        // SAFETY: every pointer addresses a full row of k floats.
        let o = unsafe { neon::tile::<R, C>(xr, wr, k) };
        for a in 0..r {
            y[(i0 + a) * n + j..(i0 + a) * n + j + C].copy_from_slice(&o[a]);
        }
        j += C;
    }
    while j < n {
        let wrow = &w[j * k..(j + 1) * k];
        for a in 0..r {
            y[(i0 + a) * n + j] = dot(&xs[a * k..(a + 1) * k], wrow);
        }
        j += 1;
    }
    r
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    /// NEON dot product with four independent FMA chains.
    ///
    /// # Safety
    /// `a` and `b` must have equal length.
    #[inline(always)]
    pub unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        let k = a.len();
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let (mut a0, mut a1, mut a2, mut a3) = (
            vdupq_n_f32(0.0),
            vdupq_n_f32(0.0),
            vdupq_n_f32(0.0),
            vdupq_n_f32(0.0),
        );
        let mut c = 0;
        while c + 16 <= k {
            a0 = vfmaq_f32(a0, vld1q_f32(pa.add(c)), vld1q_f32(pb.add(c)));
            a1 = vfmaq_f32(a1, vld1q_f32(pa.add(c + 4)), vld1q_f32(pb.add(c + 4)));
            a2 = vfmaq_f32(a2, vld1q_f32(pa.add(c + 8)), vld1q_f32(pb.add(c + 8)));
            a3 = vfmaq_f32(a3, vld1q_f32(pa.add(c + 12)), vld1q_f32(pb.add(c + 12)));
            c += 16;
        }
        while c + 4 <= k {
            a0 = vfmaq_f32(a0, vld1q_f32(pa.add(c)), vld1q_f32(pb.add(c)));
            c += 4;
        }
        let mut s = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
        while c < k {
            s += a[c] * b[c];
            c += 1;
        }
        s
    }

    /// R token rows × C weight rows register tile over the full k dimension.
    ///
    /// # Safety
    /// Every pointer must address at least `k` readable floats.
    #[inline(always)]
    pub unsafe fn tile<const R: usize, const C: usize>(
        x: [*const f32; R],
        w: [*const f32; C],
        k: usize,
    ) -> [[f32; C]; R] {
        let mut acc = [[vdupq_n_f32(0.0); C]; R];
        let mut c = 0;
        while c + 4 <= k {
            let mut wv = [vdupq_n_f32(0.0); C];
            for b in 0..C {
                wv[b] = vld1q_f32(w[b].add(c));
            }
            for a in 0..R {
                let xv = vld1q_f32(x[a].add(c));
                for b in 0..C {
                    acc[a][b] = vfmaq_f32(acc[a][b], xv, wv[b]);
                }
            }
            c += 4;
        }
        let mut o = [[0.0; C]; R];
        for a in 0..R {
            for b in 0..C {
                let mut s = vaddvq_f32(acc[a][b]);
                let mut t = c;
                while t < k {
                    s += *x[a].add(t) * *w[b].add(t);
                    t += 1;
                }
                o[a][b] = s;
            }
        }
        o
    }
}

/// Portable R-row × 4-weight-row tile (non-aarch64 targets).
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn tile_rows_portable<const R: usize>(x: &Tensor2D, w: &Tensor2D, y: &mut Tensor2D, i0: usize) {
    let (k, n) = (x.cols, w.rows);
    let xr: [&[f32]; R] = std::array::from_fn(|a| &x.data[(i0 + a) * k..(i0 + a + 1) * k]);
    if R == 1 {
        for j in 0..n {
            y.data[i0 * n + j] = dot(xr[0], &w.data[j * k..(j + 1) * k]);
        }
        return;
    }
    let k4 = k / 4 * 4;
    let mut j = 0;
    while j + 4 <= n {
        let wr: [&[f32]; 4] = std::array::from_fn(|b| &w.data[(j + b) * k..(j + b + 1) * k]);
        let mut acc = [[[0.0f32; 4]; 4]; R];
        let mut c = 0;
        while c < k4 {
            let wv: [[f32; 4]; 4] = std::array::from_fn(|b| wr[b][c..c + 4].try_into().unwrap());
            for a in 0..R {
                let xv: [f32; 4] = xr[a][c..c + 4].try_into().unwrap();
                for b in 0..4 {
                    for l in 0..4 {
                        acc[a][b][l] += xv[l] * wv[b][l];
                    }
                }
            }
            c += 4;
        }
        for a in 0..R {
            for b in 0..4 {
                let mut s = (acc[a][b][0] + acc[a][b][2]) + (acc[a][b][1] + acc[a][b][3]);
                for t in k4..k {
                    s += xr[a][t] * wr[b][t];
                }
                y.data[(i0 + a) * n + j + b] = s;
            }
        }
        j += 4;
    }
    while j < n {
        let wrow = &w.data[j * k..(j + 1) * k];
        for a in 0..R {
            y.data[(i0 + a) * n + j] = dot(xr[a], wrow);
        }
        j += 1;
    }
}

// ---------------------------------------------------------------------------
// Int8 block-quantized weights + GEMM (used for cold expert storage)
// ---------------------------------------------------------------------------

pub const Q8_BLOCK: usize = 32;

/// Row-major int8 matrix with one f32 scale per 32-element block of each row
/// (the final block of a row may be shorter).
#[derive(Clone, Debug)]
pub struct QMatrix {
    pub rows: usize,
    pub cols: usize,
    pub q: Vec<i8>,
    pub scales: Vec<f32>,
}

impl QMatrix {
    fn blocks_per_row(cols: usize) -> usize {
        cols.div_ceil(Q8_BLOCK)
    }

    pub fn quantize(w: &Tensor2D) -> Self {
        let bpr = Self::blocks_per_row(w.cols);
        let mut q = vec![0i8; w.rows * w.cols];
        let mut scales = vec![0.0f32; w.rows * bpr];
        for r in 0..w.rows {
            let row = w.row(r);
            for (b, blk) in row.chunks(Q8_BLOCK).enumerate() {
                let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let scale = if amax > 0.0 { amax / 127.0 } else { 1.0 };
                scales[r * bpr + b] = scale;
                let inv = 1.0 / scale;
                for (t, v) in blk.iter().enumerate() {
                    q[r * w.cols + b * Q8_BLOCK + t] = (v * inv).round().clamp(-127.0, 127.0) as i8;
                }
            }
        }
        QMatrix {
            rows: w.rows,
            cols: w.cols,
            q,
            scales,
        }
    }

    pub fn dequantize(&self) -> Tensor2D {
        let mut t = Tensor2D::zeros(self.rows, self.cols);
        self.dequantize_into(&mut t.data);
        t
    }

    pub fn dequantize_into(&self, out: &mut [f32]) {
        let bpr = Self::blocks_per_row(self.cols);
        for r in 0..self.rows {
            let o = r * self.cols;
            for (c, (dst, &qv)) in out[o..o + self.cols]
                .iter_mut()
                .zip(&self.q[o..o + self.cols])
                .enumerate()
            {
                *dst = qv as f32 * self.scales[r * bpr + c / Q8_BLOCK];
            }
        }
    }

    pub fn bytes(&self) -> usize {
        self.q.len() + self.scales.len() * 4
    }

    /// y[m, rows] = x[m, cols] · Qᵀ with on-the-fly block dequantisation.
    pub fn gemm(&self, x: &Tensor2D) -> Tensor2D {
        assert_eq!(x.cols, self.cols);
        let bpr = Self::blocks_per_row(self.cols);
        let mut y = Tensor2D::zeros(x.rows, self.rows);
        let mut wrow = vec![0.0f32; self.cols];
        for r in 0..self.rows {
            let o = r * self.cols;
            for (c, w) in wrow.iter_mut().enumerate() {
                *w = self.q[o + c] as f32 * self.scales[r * bpr + c / Q8_BLOCK];
            }
            for i in 0..x.rows {
                y.data[i * self.rows + r] = dot(x.row(i), &wrow);
            }
        }
        y
    }
}

// ---------------------------------------------------------------------------
// Reference grouped-query attention (monolithic, single buffer)
// ---------------------------------------------------------------------------

/// Monolithic scaled-dot-product attention for one query vector against
/// `t` keys/values laid out [t, d_head] (a single KV head). `mask[t] = false`
/// excludes a position. Computed in f64 for use as a ground-truth reference.
pub fn reference_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    d_head: usize,
    allowed: &dyn Fn(usize) -> bool,
    out: &mut [f32],
) {
    let t = k.len() / d_head;
    let scale = 1.0 / (d_head as f64).sqrt();
    let mut logits = Vec::with_capacity(t);
    let mut m = f64::NEG_INFINITY;
    for j in 0..t {
        if allowed(j) {
            let kj = &k[j * d_head..(j + 1) * d_head];
            let s: f64 = q
                .iter()
                .zip(kj)
                .map(|(a, b)| *a as f64 * *b as f64)
                .sum::<f64>()
                * scale;
            m = m.max(s);
            logits.push((j, s));
        }
    }
    let mut acc = vec![0.0f64; d_head];
    let mut l = 0.0f64;
    for (j, s) in logits {
        let p = (s - m).exp();
        l += p;
        let vj = &v[j * d_head..(j + 1) * d_head];
        for d in 0..d_head {
            acc[d] += p * vj[d] as f64;
        }
    }
    for d in 0..d_head {
        out[d] = (acc[d] / l) as f32;
    }
}

/// Simple wall-clock helper.
pub fn now_ns() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

/// Ask the OS to schedule the calling thread on performance cores.
/// On Apple Silicon, threads at default QoS may land on efficiency cores,
/// which run ~3× slower and make timings erratic.
pub fn prefer_performance_cores() {
    #[cfg(target_os = "macos")]
    {
        extern "C" {
            fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
        }
        const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;
        // SAFETY: plain libSystem call affecting only the current thread.
        unsafe {
            pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_roundtrip() {
        for &x in &[0.0f32, 1.0, -2.5, 65504.0, 1e-5, 1.2345, -0.000061035156] {
            let y = f16_to_f32(f32_to_f16(x));
            assert!((x - y).abs() <= x.abs() * 1e-3 + 1e-7, "{x} -> {y}");
        }
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(-2.0), 0xc000);
        assert!(f16_to_f32(f32_to_f16(1e6)).is_infinite());
    }

    #[test]
    fn gemm_matches_naive() {
        let mut rng = Rng::new(1);
        let x = Tensor2D::randn(7, 37, 1.0, &mut rng);
        let w = Tensor2D::randn(11, 37, 1.0, &mut rng);
        let mut y = Tensor2D::zeros(7, 11);
        gemm_xwt(&x, &w, &mut y);
        for i in 0..7 {
            for j in 0..11 {
                let r: f32 = (0..37).map(|t| x.row(i)[t] * w.row(j)[t]).sum();
                assert!((r - y.data[i * 11 + j]).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn q8_ragged_cols() {
        let mut rng = Rng::new(8);
        let w = Tensor2D::randn(5, 176, 0.1, &mut rng);
        let d = QMatrix::quantize(&w).dequantize();
        assert!(cosine(&w.data, &d.data) > 0.9999);
    }

    #[test]
    fn q8_gemm_close() {
        let mut rng = Rng::new(2);
        let w = Tensor2D::randn(16, 64, 0.1, &mut rng);
        let x = Tensor2D::randn(3, 64, 1.0, &mut rng);
        let q = QMatrix::quantize(&w);
        let y = q.gemm(&x);
        let r = Linear::new(w).forward(&x);
        assert!(cosine(&y.data, &r.data) > 0.9995);
    }
}
