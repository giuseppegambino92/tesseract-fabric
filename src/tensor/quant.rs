//! Block-quantized and half-precision weight formats (GGML-compatible byte
//! layouts) with fused kernels that compute directly on the compressed bytes.
//!
//! * `Q8_0`: 34 bytes per 32 weights — one IEEE f16 scale `d` + 32 `i8`
//!   (`w = q · d`).
//! * `Q4_0`: 18 bytes per 32 weights — one f16 scale + 16 bytes of nibbles;
//!   byte `j` holds element `j` (low nibble) and `j + 16` (high nibble),
//!   `w = (q − 8) · d`.
//! * `F16`, `BF16`: 2 bytes per weight, converted on the fly.
//! * `F32`: 4 bytes per weight.
//!
//! A [`QuantView`] is a zero-copy, row-strided window into any byte source
//! (an `mmap`ed model file or an owned buffer): row slices and block-aligned
//! column slices only move offsets. For `Q8_0` / `Q4_0` weights the activation
//! rows are quantized once per GEMM into `Q8_0` blocks and every block product
//! is an integer dot (`i8 · i8 → i32`) scaled by `d_w · d_x`, with up to four
//! activation rows (a speculative tree batch) sharing each decoded weight
//! block. NEON (`aarch64`, `sdot` when available) and AVX2 (`x86_64`, runtime
//! detected) kernels fall back to portable scalar code elsewhere.

use super::{f16_to_f32, f32_to_f16, Tensor2D};
use std::sync::Arc;

pub const QK: usize = 32;
pub const Q8_0_BLOCK_BYTES: usize = 34;
pub const Q4_0_BLOCK_BYTES: usize = 18;

/// Weight storage formats (values are the GGML type ids; names follow GGML).
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WeightFormat {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    BF16 = 30,
}

impl WeightFormat {
    pub fn from_ggml(t: u32) -> Option<Self> {
        Some(match t {
            0 => WeightFormat::F32,
            1 => WeightFormat::F16,
            2 => WeightFormat::Q4_0,
            3 => WeightFormat::Q4_1,
            6 => WeightFormat::Q5_0,
            7 => WeightFormat::Q5_1,
            8 => WeightFormat::Q8_0,
            12 => WeightFormat::Q4_K,
            13 => WeightFormat::Q5_K,
            14 => WeightFormat::Q6_K,
            30 => WeightFormat::BF16,
            _ => return None,
        })
    }
    pub fn ggml_id(self) -> u32 {
        self as u32
    }
    pub fn name(self) -> &'static str {
        match self {
            WeightFormat::F32 => "F32",
            WeightFormat::F16 => "F16",
            WeightFormat::BF16 => "BF16",
            WeightFormat::Q8_0 => "Q8_0",
            WeightFormat::Q4_0 => "Q4_0",
            WeightFormat::Q4_1 => "Q4_1",
            WeightFormat::Q5_0 => "Q5_0",
            WeightFormat::Q5_1 => "Q5_1",
            WeightFormat::Q4_K => "Q4_K",
            WeightFormat::Q5_K => "Q5_K",
            WeightFormat::Q6_K => "Q6_K",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s.to_ascii_lowercase().as_str() {
            "f32" | "fp32" => WeightFormat::F32,
            "f16" | "fp16" => WeightFormat::F16,
            "bf16" => WeightFormat::BF16,
            "q8_0" | "q8" => WeightFormat::Q8_0,
            "q4_0" | "q4" => WeightFormat::Q4_0,
            "q5_0" => WeightFormat::Q5_0,
            "q4_k" => WeightFormat::Q4_K,
            "q6_k" => WeightFormat::Q6_K,
            _ => return None,
        })
    }
    /// Elements per storage block (1 for element formats).
    pub fn block_len(self) -> usize {
        match self {
            WeightFormat::Q8_0
            | WeightFormat::Q4_0
            | WeightFormat::Q4_1
            | WeightFormat::Q5_0
            | WeightFormat::Q5_1 => QK,
            WeightFormat::Q4_K | WeightFormat::Q5_K | WeightFormat::Q6_K => super::kquant::QK_K,
            _ => 1,
        }
    }
    /// Bytes of `n` elements (`n` must be a multiple of `block_len`).
    pub fn row_bytes(self, n: usize) -> usize {
        match self {
            WeightFormat::F32 => 4 * n,
            WeightFormat::F16 | WeightFormat::BF16 => 2 * n,
            WeightFormat::Q8_0 => n / QK * Q8_0_BLOCK_BYTES,
            WeightFormat::Q4_0 => n / QK * Q4_0_BLOCK_BYTES,
            f => n / f.block_len() * super::kquant::block_bytes(f),
        }
    }
    pub fn is_block(self) -> bool {
        self.block_len() > 1
    }
    /// Whether this engine can also *write* the format (exporters).
    pub fn can_encode(self) -> bool {
        !matches!(
            self,
            WeightFormat::Q4_1 | WeightFormat::Q5_1 | WeightFormat::Q5_K
        )
    }
    /// Bits per weight as stored.
    pub fn bits_per_weight(self) -> f64 {
        self.row_bytes(self.block_len() * 8) as f64 * 8.0 / (self.block_len() * 8) as f64
    }
}

// ---------------------------------------------------------------------------
// Element conversions
// ---------------------------------------------------------------------------

/// bfloat16 → f32 (exact).
#[inline]
pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// f32 → bfloat16, round-to-nearest-even (NaN stays NaN).
#[inline]
pub fn f32_to_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return ((b >> 16) | 0x40) as u16;
    }
    let round = 0x7fff + ((b >> 16) & 1);
    (b.wrapping_add(round) >> 16) as u16
}

/// Exact branch-free f16 → f32 (vectorizes): the magnitude bits are placed
/// in an f32 and rescaled by 2¹¹², which handles normals and subnormals
/// alike; Inf / NaN get the f32 all-ones exponent.
#[inline(always)]
pub fn f16_fast(h: u16) -> f32 {
    let h = h as u32;
    let mag = (h & 0x7fff) << 13;
    let v = f32::from_bits(mag) * f32::from_bits(0x7780_0000);
    let v = if h & 0x7c00 == 0x7c00 {
        f32::from_bits(mag | 0x7f80_0000)
    } else {
        v
    };
    f32::from_bits(v.to_bits() | ((h & 0x8000) << 16))
}

#[inline]
fn rd_u16(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}

// ---------------------------------------------------------------------------
// Row quantizers (GGML reference semantics)
// ---------------------------------------------------------------------------

/// Append `x` (length multiple of 32) as Q8_0 blocks.
pub fn quantize_row_q8_0(x: &[f32], out: &mut Vec<u8>) {
    assert!(
        x.len().is_multiple_of(QK),
        "Q8_0 rows must be multiples of 32"
    );
    for blk in x.as_chunks::<QK>().0 {
        let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        out.extend(blk.iter().map(|v| (v * id).round() as i8 as u8));
    }
}

/// Append `x` (length multiple of 32) as Q4_0 blocks.
pub fn quantize_row_q4_0(x: &[f32], out: &mut Vec<u8>) {
    assert!(
        x.len().is_multiple_of(QK),
        "Q4_0 rows must be multiples of 32"
    );
    for blk in x.as_chunks::<QK>().0 {
        let (mut amax, mut max) = (0.0f32, 0.0f32);
        for &v in blk {
            if v.abs() > amax {
                amax = v.abs();
                max = v;
            }
        }
        let d = max / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        for j in 0..QK / 2 {
            let q0 = ((blk[j] * id + 8.5) as i8).clamp(0, 15) as u8;
            let q1 = ((blk[j + QK / 2] * id + 8.5) as i8).clamp(0, 15) as u8;
            out.push(q0 | (q1 << 4));
        }
    }
}

/// Encode a row-major matrix in `fmt`.
pub fn encode_matrix(w: &[f32], cols: usize, fmt: WeightFormat) -> Vec<u8> {
    let mut out = Vec::with_capacity(fmt.row_bytes(cols) * (w.len() / cols.max(1)));
    for row in w.chunks_exact(cols) {
        match fmt {
            WeightFormat::F32 => row
                .iter()
                .for_each(|v| out.extend_from_slice(&v.to_le_bytes())),
            WeightFormat::F16 => row
                .iter()
                .for_each(|v| out.extend_from_slice(&f32_to_f16(*v).to_le_bytes())),
            WeightFormat::BF16 => row
                .iter()
                .for_each(|v| out.extend_from_slice(&f32_to_bf16(*v).to_le_bytes())),
            WeightFormat::Q8_0 => quantize_row_q8_0(row, &mut out),
            WeightFormat::Q4_0 => quantize_row_q4_0(row, &mut out),
            f => super::kquant::quantize_row(f, row, &mut out),
        }
    }
    out
}

/// Decode `out.len()` elements of `fmt` from `b`.
pub fn decode_row(b: &[u8], fmt: WeightFormat, out: &mut [f32]) {
    match fmt {
        WeightFormat::F32 => {
            for (i, o) in out.iter_mut().enumerate() {
                *o = f32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap());
            }
        }
        WeightFormat::F16 => {
            for (o, c) in out.iter_mut().zip(b.as_chunks::<2>().0) {
                *o = f16_fast(u16::from_le_bytes(*c));
            }
        }
        WeightFormat::BF16 => {
            for (o, c) in out.iter_mut().zip(b.as_chunks::<2>().0) {
                *o = bf16_to_f32(u16::from_le_bytes(*c));
            }
        }
        WeightFormat::Q8_0 => {
            for (bi, blk) in out.as_chunks_mut::<QK>().0.iter_mut().enumerate() {
                let o = bi * Q8_0_BLOCK_BYTES;
                let d = f16_fast(rd_u16(b, o));
                for (j, v) in blk.iter_mut().enumerate() {
                    *v = b[o + 2 + j] as i8 as f32 * d;
                }
            }
        }
        WeightFormat::Q4_0 => {
            for (bi, blk) in out.as_chunks_mut::<QK>().0.iter_mut().enumerate() {
                let o = bi * Q4_0_BLOCK_BYTES;
                let d = f16_fast(rd_u16(b, o));
                for j in 0..QK / 2 {
                    let q = b[o + 2 + j];
                    blk[j] = ((q & 0x0f) as i32 - 8) as f32 * d;
                    blk[j + QK / 2] = ((q >> 4) as i32 - 8) as f32 * d;
                }
            }
        }
        f => super::kquant::decode_row(b, f, out),
    }
}

// ---------------------------------------------------------------------------
// Zero-copy matrix view
// ---------------------------------------------------------------------------

/// Anything that exposes a stable byte buffer (an `mmap`ed file, a `Vec`).
pub trait ByteSource: Send + Sync {
    fn bytes(&self) -> &[u8];
    /// Hint that `[offset, offset + len)` will be read soon (mmap: madvise).
    fn will_need(&self, _offset: usize, _len: usize) {}
    /// Whether the bytes are a zero-copy OS mapping.
    fn is_mapped(&self) -> bool {
        false
    }
}

impl ByteSource for Vec<u8> {
    fn bytes(&self) -> &[u8] {
        self
    }
}

/// A `rows × cols` weight matrix stored row-major in `fmt` inside a shared
/// byte source, starting at `offset` with `row_stride` bytes between rows.
#[derive(Clone)]
pub struct QuantView {
    pub fmt: WeightFormat,
    pub rows: usize,
    pub cols: usize,
    src: Arc<dyn ByteSource>,
    offset: usize,
    row_stride: usize,
}

impl std::fmt::Debug for QuantView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "QuantView({} {}x{} @{})",
            self.fmt.name(),
            self.rows,
            self.cols,
            self.offset
        )
    }
}

impl QuantView {
    /// View a contiguous row-major matrix at `offset` in `src`.
    pub fn new(
        src: Arc<dyn ByteSource>,
        offset: usize,
        fmt: WeightFormat,
        rows: usize,
        cols: usize,
    ) -> Self {
        assert!(
            cols.is_multiple_of(fmt.block_len()),
            "{} needs cols % {} == 0",
            fmt.name(),
            fmt.block_len()
        );
        let row_stride = fmt.row_bytes(cols);
        assert!(
            offset + row_stride * rows <= src.bytes().len(),
            "tensor exceeds its byte source"
        );
        QuantView {
            fmt,
            rows,
            cols,
            src,
            offset,
            row_stride,
        }
    }

    /// Encode an f32 matrix into an owned buffer (used by exporters/tests).
    pub fn from_f32(w: &Tensor2D, fmt: WeightFormat) -> Self {
        let b = encode_matrix(&w.data, w.cols, fmt);
        Self::new(Arc::new(b), 0, fmt, w.rows, w.cols)
    }

    /// Bytes this view addresses (its storage footprint).
    pub fn bytes(&self) -> usize {
        self.fmt.row_bytes(self.cols) * self.rows
    }

    /// The raw bytes of row `r`.
    #[inline]
    pub fn row_bytes(&self, r: usize) -> &[u8] {
        let o = self.offset + r * self.row_stride;
        &self.src.bytes()[o..o + self.fmt.row_bytes(self.cols)]
    }

    /// Rows [start, end) — zero copy.
    pub fn slice_rows(&self, start: usize, end: usize) -> Self {
        assert!(start <= end && end <= self.rows);
        QuantView {
            rows: end - start,
            offset: self.offset + start * self.row_stride,
            ..self.clone()
        }
    }

    /// Whether a column slice at these bounds stays zero-copy.
    pub fn cols_aligned(&self, start: usize, end: usize) -> bool {
        let b = self.fmt.block_len();
        start.is_multiple_of(b) && end.is_multiple_of(b)
    }

    /// Columns [start, end) — zero copy (bounds must be block aligned).
    pub fn slice_cols(&self, start: usize, end: usize) -> Self {
        assert!(self.cols_aligned(start, end) && end <= self.cols);
        QuantView {
            cols: end - start,
            offset: self.offset + self.fmt.row_bytes(start),
            ..self.clone()
        }
    }

    pub fn decode_row_into(&self, r: usize, out: &mut [f32]) {
        decode_row(self.row_bytes(r), self.fmt, &mut out[..self.cols]);
    }

    /// Dequantize the whole view into an f32 matrix.
    pub fn dequantize(&self) -> Tensor2D {
        let mut t = Tensor2D::zeros(self.rows, self.cols);
        for r in 0..self.rows {
            let c = self.cols;
            self.decode_row_into(r, &mut t.data[r * c..(r + 1) * c]);
        }
        t
    }

    /// Byte span `[start, end)` this view reads inside its source.
    pub fn span(&self) -> (usize, usize) {
        if self.rows == 0 {
            return (self.offset, self.offset);
        }
        let end = self.offset + (self.rows - 1) * self.row_stride + self.fmt.row_bytes(self.cols);
        (self.offset, end)
    }

    /// Ask the OS to start paging this view's bytes in.
    pub fn prefetch(&self) {
        let (s, e) = self.span();
        self.src.will_need(s, e - s);
    }

    /// True when the view reads an `mmap`ed file directly.
    pub fn is_mapped(&self) -> bool {
        self.src.is_mapped()
    }

    /// Identity of the underlying byte source (tests: zero-copy checks).
    pub fn source_ptr(&self) -> *const u8 {
        self.src.bytes().as_ptr()
    }
    pub fn byte_offset(&self) -> usize {
        self.offset
    }
    /// Bytes between the starts of consecutive rows (≥ the row's own bytes
    /// for a column slice).
    pub fn row_stride(&self) -> usize {
        self.row_stride
    }
    /// The whole underlying byte source (device uploads index it by `span`).
    pub fn source_bytes(&self) -> &[u8] {
        self.src.bytes()
    }
}

// ---------------------------------------------------------------------------
// Activation quantization
// ---------------------------------------------------------------------------

/// Activation rows quantized to Q8_0 blocks, split into contiguous `i8`
/// values and f32 scales (decoded once from the block's f16 scale).
pub struct Q8Acts {
    pub rows: usize,
    pub cols: usize,
    pub q: Vec<i8>,
    pub d: Vec<f32>,
    /// Σ q of every block (formats with additive mins need Σ x = d·Σ q).
    pub s: Vec<i32>,
    /// d·Σ q of every block (≈ Σ x), the factor of a block's minimum term.
    pub ds: Vec<f32>,
}

impl Q8Acts {
    pub fn quantize(x: &Tensor2D) -> Self {
        assert!(x.cols.is_multiple_of(QK));
        let nb = x.cols / QK;
        let mut q = vec![0i8; x.rows * x.cols];
        let mut d = vec![0.0f32; x.rows * nb];
        let mut sums = vec![0i32; x.rows * nb];
        for i in 0..x.rows {
            for (b, blk) in x.row(i).as_chunks::<QK>().0.iter().enumerate() {
                let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                // the block scale is stored as f16 (GGML BlockQ8_0)
                let dd = f16_to_f32(f32_to_f16(amax / 127.0));
                let id = if dd != 0.0 { 1.0 / dd } else { 0.0 };
                d[i * nb + b] = dd;
                let o = i * x.cols + b * QK;
                let mut sm = 0i32;
                for (t, v) in blk.iter().enumerate() {
                    let qv = (v * id).round().clamp(-127.0, 127.0) as i8;
                    q[o + t] = qv;
                    sm += qv as i32;
                }
                sums[i * nb + b] = sm;
            }
        }
        let ds = d.iter().zip(&sums).map(|(d, s)| d * *s as f32).collect();
        Q8Acts {
            rows: x.rows,
            cols: x.cols,
            q,
            d,
            s: sums,
            ds,
        }
    }
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

/// Which integer-dot implementation the block kernels use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DotKernel {
    Scalar,
    #[cfg(target_arch = "aarch64")]
    Neon,
    #[cfg(target_arch = "aarch64")]
    NeonDot,
    #[cfg(target_arch = "x86_64")]
    Avx2,
}

impl DotKernel {
    pub fn name(self) -> &'static str {
        match self {
            DotKernel::Scalar => "scalar",
            #[cfg(target_arch = "aarch64")]
            DotKernel::Neon => "NEON (vmull_s8)",
            #[cfg(target_arch = "aarch64")]
            DotKernel::NeonDot => "NEON (sdot)",
            #[cfg(target_arch = "x86_64")]
            DotKernel::Avx2 => "AVX2 + FMA",
        }
    }
}

/// Best kernel this CPU supports.
pub fn best_kernel() -> DotKernel {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return DotKernel::NeonDot;
        }
        DotKernel::Neon
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return DotKernel::Avx2;
        }
        DotKernel::Scalar
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        DotKernel::Scalar
    }
}

static KERNEL: std::sync::OnceLock<DotKernel> = std::sync::OnceLock::new();

pub fn active_kernel() -> DotKernel {
    *KERNEL.get_or_init(best_kernel)
}

/// Decode one weight block of a Q8_0 / Q4_0 row into 32 `i8` and its scale.
#[inline(always)]
fn decode_block_i8(b: &[u8], fmt: WeightFormat, blk: usize, out: &mut [i8; QK]) -> f32 {
    match fmt {
        WeightFormat::Q8_0 => {
            let o = blk * Q8_0_BLOCK_BYTES;
            for j in 0..QK {
                out[j] = b[o + 2 + j] as i8;
            }
            f16_fast(rd_u16(b, o))
        }
        WeightFormat::Q5_0 => {
            let o = blk * Q5_0_BLOCK_BYTES;
            let qh = u32::from_le_bytes(b[o + 2..o + 6].try_into().unwrap());
            for j in 0..QK / 2 {
                let q = b[o + 6 + j];
                out[j] = ((q & 0x0f) | ((((qh >> j) << 4) & 0x10) as u8)) as i8 - 16;
                out[j + QK / 2] = ((q >> 4) | (((qh >> (j + 12)) & 0x10) as u8)) as i8 - 16;
            }
            f16_fast(rd_u16(b, o))
        }
        _ => {
            let o = blk * Q4_0_BLOCK_BYTES;
            for j in 0..QK / 2 {
                let q = b[o + 2 + j];
                out[j] = (q & 0x0f) as i8 - 8;
                out[j + QK / 2] = (q >> 4) as i8 - 8;
            }
            f16_fast(rd_u16(b, o))
        }
    }
}

pub const Q5_0_BLOCK_BYTES: usize = 22;

/// Byte b → 8 bytes, byte i = 0x10 if bit i of b is set (Q5_0 high bits).
static HIGH_BITS: std::sync::OnceLock<[u64; 256]> = std::sync::OnceLock::new();

fn high_bits() -> &'static [u64; 256] {
    HIGH_BITS.get_or_init(|| {
        let mut t = [0u64; 256];
        for (b, e) in t.iter_mut().enumerate() {
            for i in 0..8 {
                if b >> i & 1 == 1 {
                    *e |= 0x10u64 << (8 * i);
                }
            }
        }
        t
    })
}

/// `y[i] = Σ_b d_w[b] · d_x[i,b] · (w_b · x_{i,b})` for up to 4 activation
/// rows against one quantized weight row (`row` = its raw bytes).
fn row_dot_q(
    kernel: DotKernel,
    fmt: WeightFormat,
    row: &[u8],
    acts: &Q8Acts,
    rows: &[usize],
    out: &mut [f32; 4],
) {
    match kernel {
        #[cfg(target_arch = "aarch64")]
        DotKernel::NeonDot => {
            // SAFETY: `dotprod` was detected at runtime; all loads stay
            // within the row / act buffers.
            unsafe { neon::row_dot_sdot::<false>(fmt, row, acts, rows, out, &mut [0; 4]) }
        }
        #[cfg(target_arch = "aarch64")]
        DotKernel::Neon => {
            // SAFETY: NEON is baseline on aarch64; loads stay in bounds.
            unsafe { neon::row_dot(false, fmt, row, acts, rows, out) }
        }
        #[cfg(target_arch = "x86_64")]
        DotKernel::Avx2 => {
            // SAFETY: AVX2 + FMA were detected at runtime.
            unsafe { avx2::row_dot(fmt, row, acts, rows, out) }
        }
        _ => row_dot_scalar(fmt, row, acts, rows, out),
    }
}

/// Exact-mode dot (fixed-point accumulation, see [`super::kquant::to_fix`])
/// of one weight row in any block format against up to 4 activation rows.
/// Canonical partial product of block `b`: `d_x · (d_w · Σ q_w·q_x)` for the
/// symmetric formats; group formats as in `kquant::row_dot_group_fixed`.
pub fn row_dot_fixed(
    kernel: DotKernel,
    fmt: WeightFormat,
    row: &[u8],
    acts: &Q8Acts,
    rows: &[usize],
    out: &mut [i64; 4],
) {
    match fmt {
        WeightFormat::Q8_0 | WeightFormat::Q4_0 | WeightFormat::Q5_0 => {
            #[cfg(target_arch = "aarch64")]
            if kernel == DotKernel::NeonDot {
                // SAFETY: `dotprod` was detected at runtime.
                unsafe { neon::row_dot_sdot::<true>(fmt, row, acts, rows, &mut [0.0; 4], out) };
                return;
            }
            let nb = acts.cols / QK;
            let mut w = [0i8; QK];
            *out = [0; 4];
            for b in 0..nb {
                let dw = decode_block_i8(row, fmt, b, &mut w);
                for (r, &i) in rows.iter().enumerate() {
                    let x = &acts.q[i * acts.cols + b * QK..i * acts.cols + (b + 1) * QK];
                    let (lo, hi) = super::kquant::dot16x2(kernel, &w, x);
                    out[r] += super::kquant::to_fix(acts.d[i * nb + b] * (dw * (lo + hi) as f32));
                }
            }
        }
        f => super::kquant::row_dot_group_fixed(kernel, f, row, acts, rows, out),
    }
}

fn row_dot_scalar(
    fmt: WeightFormat,
    row: &[u8],
    acts: &Q8Acts,
    rows: &[usize],
    out: &mut [f32; 4],
) {
    let nb = acts.cols / QK;
    let mut w = [0i8; QK];
    *out = [0.0; 4];
    for b in 0..nb {
        let dw = decode_block_i8(row, fmt, b, &mut w);
        for (r, &i) in rows.iter().enumerate() {
            let x = &acts.q[i * acts.cols + b * QK..i * acts.cols + (b + 1) * QK];
            let s: i32 = w.iter().zip(x).map(|(a, c)| *a as i32 * *c as i32).sum();
            out[r] += dw * acts.d[i * nb + b] * s as f32;
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use std::arch::aarch64::*;

    #[inline(always)]
    unsafe fn load_w(fmt: WeightFormat, row: &[u8], b: usize) -> (int8x16_t, int8x16_t, f32) {
        match fmt {
            WeightFormat::Q8_0 => {
                let o = b * Q8_0_BLOCK_BYTES;
                let p = row.as_ptr().add(o + 2) as *const i8;
                (vld1q_s8(p), vld1q_s8(p.add(16)), f16_fast(rd_u16(row, o)))
            }
            WeightFormat::Q5_0 => {
                let o = b * Q5_0_BLOCK_BYTES;
                let t = high_bits();
                let qh = &row[o + 2..o + 6];
                let v = vld1q_u8(row.as_ptr().add(o + 6));
                let m = vdupq_n_u8(0x0f);
                let h0 = vcombine_u8(vcreate_u8(t[qh[0] as usize]), vcreate_u8(t[qh[1] as usize]));
                let h1 = vcombine_u8(vcreate_u8(t[qh[2] as usize]), vcreate_u8(t[qh[3] as usize]));
                let lo = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(v, m), h0));
                let hi = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(v, 4), h1));
                let e = vdupq_n_s8(16);
                (vsubq_s8(lo, e), vsubq_s8(hi, e), f16_fast(rd_u16(row, o)))
            }
            _ => {
                let o = b * Q4_0_BLOCK_BYTES;
                let v = vld1q_u8(row.as_ptr().add(o + 2));
                let m = vdupq_n_u8(0x0f);
                let lo = vreinterpretq_s8_u8(vandq_u8(v, m));
                let hi = vreinterpretq_s8_u8(vshrq_n_u8(v, 4));
                let e = vdupq_n_s8(8);
                (vsubq_s8(lo, e), vsubq_s8(hi, e), f16_fast(rd_u16(row, o)))
            }
        }
    }

    #[inline(always)]
    unsafe fn dot32_mull(w0: int8x16_t, w1: int8x16_t, x: *const i8) -> int32x4_t {
        let x0 = vld1q_s8(x);
        let x1 = vld1q_s8(x.add(16));
        let p0 = vmull_s8(vget_low_s8(w0), vget_low_s8(x0));
        let p1 = vmull_high_s8(w0, x0);
        let p2 = vmull_s8(vget_low_s8(w1), vget_low_s8(x1));
        let p3 = vmull_high_s8(w1, x1);
        let s = vpaddlq_s16(p0);
        let s = vpadalq_s16(s, p1);
        let s = vpadalq_s16(s, p2);
        vpadalq_s16(s, p3)
    }

    #[target_feature(enable = "dotprod")]
    unsafe fn dot32_sdot(w0: int8x16_t, w1: int8x16_t, x: *const i8) -> int32x4_t {
        let x0 = vld1q_s8(x);
        let x1 = vld1q_s8(x.add(16));
        let s = vdotq_s32(vdupq_n_s32(0), w0, x0);
        vdotq_s32(s, w1, x1)
    }

    /// Q5_0 block with the fifth bits rebuilt in registers: the 4 `qh`
    /// bytes are spread to 8 lanes each (`tbl`) and bit-tested.
    #[inline(always)]
    unsafe fn load_q5_0(row: &[u8], b: usize) -> (int8x16_t, int8x16_t, f32) {
        const SPREAD: [u8; 16] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
        const BITS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
        let o = b * Q5_0_BLOCK_BYTES;
        let p = row.as_ptr().add(o);
        let qh = vreinterpretq_u8_u32(vdupq_n_u32(u32::from_le_bytes([
            *p.add(2),
            *p.add(3),
            *p.add(4),
            *p.add(5),
        ])));
        let idx = vld1q_u8(SPREAD.as_ptr());
        let bits = vld1q_u8(BITS.as_ptr());
        let s16 = vdupq_n_u8(0x10);
        let h0 = vandq_u8(vtstq_u8(vqtbl1q_u8(qh, idx), bits), s16);
        let h1 = vandq_u8(
            vtstq_u8(vqtbl1q_u8(qh, vaddq_u8(idx, vdupq_n_u8(2))), bits),
            s16,
        );
        let v = vld1q_u8(p.add(6));
        let m = vdupq_n_u8(0x0f);
        let lo = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(v, m), h0));
        let hi = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(v, 4), h1));
        let e = vdupq_n_s8(16);
        (vsubq_s8(lo, e), vsubq_s8(hi, e), f16_fast(rd_u16(row, o)))
    }

    /// Q8_0 / Q4_0 / Q5_0 with `sdot` inlined (the whole loop is compiled
    /// for `dotprod`), two blocks per iteration into separate accumulators;
    /// `EXACT` emits canonical fixed-point block products instead. The row
    /// count and format are compile-time constants of the inner kernel, so
    /// every accumulator lives in a register.
    #[target_feature(enable = "dotprod")]
    pub unsafe fn row_dot_sdot<const EXACT: bool>(
        fmt: WeightFormat,
        row: &[u8],
        acts: &Q8Acts,
        rows: &[usize],
        outf: &mut [f32; 4],
        outi: &mut [i64; 4],
    ) {
        let i0 = rows[0];
        macro_rules! go {
            ($f:expr) => {
                match rows.len() {
                    1 => sdot_rows::<EXACT, $f, 1>(row, acts, i0, outf, outi),
                    2 => sdot_rows::<EXACT, $f, 2>(row, acts, i0, outf, outi),
                    3 => sdot_rows::<EXACT, $f, 3>(row, acts, i0, outf, outi),
                    _ => sdot_rows::<EXACT, $f, 4>(row, acts, i0, outf, outi),
                }
            };
        }
        match fmt {
            WeightFormat::Q8_0 => go!(0),
            WeightFormat::Q5_0 => go!(1),
            _ => go!(2),
        }
    }

    #[inline(always)]
    unsafe fn load_f<const F: u8>(row: &[u8], b: usize) -> (int8x16_t, int8x16_t, f32) {
        match F {
            0 => load_w(WeightFormat::Q8_0, row, b),
            1 => load_q5_0(row, b),
            _ => load_w(WeightFormat::Q4_0, row, b),
        }
    }

    /// `R` consecutive activation rows from `i0` (1 ≤ R ≤ 4), format `F`
    /// (0 = Q8_0, 1 = Q5_0, 2 = Q4_0).
    #[target_feature(enable = "dotprod")]
    unsafe fn sdot_rows<const EXACT: bool, const F: u8, const R: usize>(
        row: &[u8],
        acts: &Q8Acts,
        i0: usize,
        outf: &mut [f32; 4],
        outi: &mut [i64; 4],
    ) {
        let nb = acts.cols / QK;
        let z = vdupq_n_s32(0);
        let mut acc0 = [vdupq_n_f32(0.0); R];
        let mut acc1 = [vdupq_n_f32(0.0); R];
        let mut fix = [0i64; R];
        let xq = acts.q.as_ptr();
        let xd = acts.d.as_ptr();
        let mut b = 0;
        while b + 2 <= nb {
            let (w0, w1, dw) = load_f::<F>(row, b);
            let (u0, u1, du) = load_f::<F>(row, b + 1);
            for r in 0..R {
                let i = i0 + r;
                let x = xq.add(i * acts.cols + b * QK);
                let s = vdotq_s32(vdotq_s32(z, w0, vld1q_s8(x)), w1, vld1q_s8(x.add(16)));
                let t = vdotq_s32(
                    vdotq_s32(z, u0, vld1q_s8(x.add(32))),
                    u1,
                    vld1q_s8(x.add(48)),
                );
                let dx = *xd.add(i * nb + b);
                let dy = *xd.add(i * nb + b + 1);
                if EXACT {
                    fix[r] += super::super::kquant::to_fix(dx * (dw * vaddvq_s32(s) as f32))
                        + super::super::kquant::to_fix(dy * (du * vaddvq_s32(t) as f32));
                } else {
                    acc0[r] = vfmaq_n_f32(acc0[r], vcvtq_f32_s32(s), dw * dx);
                    acc1[r] = vfmaq_n_f32(acc1[r], vcvtq_f32_s32(t), du * dy);
                }
            }
            b += 2;
        }
        if b < nb {
            let (w0, w1, dw) = load_f::<F>(row, b);
            for r in 0..R {
                let i = i0 + r;
                let x = xq.add(i * acts.cols + b * QK);
                let s = vdotq_s32(vdotq_s32(z, w0, vld1q_s8(x)), w1, vld1q_s8(x.add(16)));
                let dx = *xd.add(i * nb + b);
                if EXACT {
                    fix[r] += super::super::kquant::to_fix(dx * (dw * vaddvq_s32(s) as f32));
                } else {
                    acc0[r] = vfmaq_n_f32(acc0[r], vcvtq_f32_s32(s), dw * dx);
                }
            }
        }
        for r in 0..R {
            if EXACT {
                outi[r] = fix[r];
            } else {
                outf[r] = vaddvq_f32(vaddq_f32(acc0[r], acc1[r]));
            }
        }
    }

    pub unsafe fn row_dot(
        sdot: bool,
        fmt: WeightFormat,
        row: &[u8],
        acts: &Q8Acts,
        rows: &[usize],
        out: &mut [f32; 4],
    ) {
        let nb = acts.cols / QK;
        // two accumulator sets (even / odd blocks) break the FMA dependency
        // chain, which otherwise bounds single-row decode
        let mut acc = [[vdupq_n_f32(0.0); 4]; 2];
        for b in 0..nb {
            let (w0, w1, dw) = load_w(fmt, row, b);
            let a = &mut acc[b & 1];
            for (r, &i) in rows.iter().enumerate() {
                let x = acts.q.as_ptr().add(i * acts.cols + b * QK);
                let s = if sdot {
                    dot32_sdot(w0, w1, x)
                } else {
                    dot32_mull(w0, w1, x)
                };
                let scale = dw * *acts.d.get_unchecked(i * nb + b);
                a[r] = vfmaq_n_f32(a[r], vcvtq_f32_s32(s), scale);
            }
        }
        for r in 0..rows.len() {
            out[r] = vaddvq_f32(vaddq_f32(acc[0][r], acc[1][r]));
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::*;
    use std::arch::x86_64::*;

    #[inline(always)]
    unsafe fn load_w(fmt: WeightFormat, row: &[u8], b: usize) -> (__m256i, f32) {
        match fmt {
            WeightFormat::Q8_0 => {
                let o = b * Q8_0_BLOCK_BYTES;
                (
                    _mm256_loadu_si256(row.as_ptr().add(o + 2) as *const __m256i),
                    f16_fast(rd_u16(row, o)),
                )
            }
            WeightFormat::Q5_0 => {
                let o = b * Q5_0_BLOCK_BYTES;
                let tb = high_bits();
                let qh = &row[o + 2..o + 6];
                let t = _mm_loadu_si128(row.as_ptr().add(o + 6) as *const __m128i);
                let hi = _mm_srli_epi16(t, 4);
                let v = _mm256_and_si256(_mm256_set_m128i(hi, t), _mm256_set1_epi8(0x0f));
                let h = _mm256_set_epi64x(
                    tb[qh[3] as usize] as i64,
                    tb[qh[2] as usize] as i64,
                    tb[qh[1] as usize] as i64,
                    tb[qh[0] as usize] as i64,
                );
                (
                    _mm256_sub_epi8(_mm256_or_si256(v, h), _mm256_set1_epi8(16)),
                    f16_fast(rd_u16(row, o)),
                )
            }
            _ => {
                let o = b * Q4_0_BLOCK_BYTES;
                let t = _mm_loadu_si128(row.as_ptr().add(o + 2) as *const __m128i);
                let hi = _mm_srli_epi16(t, 4);
                let v = _mm256_and_si256(_mm256_set_m128i(hi, t), _mm256_set1_epi8(0x0f));
                (
                    _mm256_sub_epi8(v, _mm256_set1_epi8(8)),
                    f16_fast(rd_u16(row, o)),
                )
            }
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn row_dot(
        fmt: WeightFormat,
        row: &[u8],
        acts: &Q8Acts,
        rows: &[usize],
        out: &mut [f32; 4],
    ) {
        let nb = acts.cols / QK;
        let ones = _mm256_set1_epi16(1);
        let mut acc = [_mm256_setzero_ps(); 4];
        for b in 0..nb {
            let (w, dw) = load_w(fmt, row, b);
            // |w| and sign(x)·sign(w) trick: maddubs needs an unsigned operand
            let aw = _mm256_sign_epi8(w, w);
            for (r, &i) in rows.iter().enumerate() {
                let x = _mm256_loadu_si256(
                    acts.q.as_ptr().add(i * acts.cols + b * QK) as *const __m256i
                );
                let sx = _mm256_sign_epi8(x, w);
                let p16 = _mm256_maddubs_epi16(aw, sx);
                let p32 = _mm256_madd_epi16(p16, ones);
                let scale = _mm256_set1_ps(dw * *acts.d.get_unchecked(i * nb + b));
                acc[r] = _mm256_fmadd_ps(scale, _mm256_cvtepi32_ps(p32), acc[r]);
            }
        }
        for r in 0..rows.len() {
            let v = acc[r];
            let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
            let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
            let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
            out[r] = _mm_cvtss_f32(s);
        }
    }
}

/// `y[:, j0..j1] = x · W[j0..j1]ᵀ` for a quantized/half view, writing into
/// `y` laid out `x.rows × ld` at column offset `j0` (lanes write disjoint
/// column stripes). Activations for block formats come pre-quantized.
#[allow(clippy::too_many_arguments)]
fn gemm_range(
    x: &Tensor2D,
    acts: Option<&Q8Acts>,
    v: &QuantView,
    j0: usize,
    j1: usize,
    y: *mut f32,
    ld: usize,
    kernel: DotKernel,
) {
    let m = x.rows;
    if v.fmt.is_block() {
        let acts = acts.expect("block formats need quantized activations");
        let mut out = [0.0f32; 4];
        for j in j0..j1 {
            let row = v.row_bytes(j);
            let mut i = 0;
            while i < m {
                let n = (m - i).min(4);
                let idx = [i, i + 1, i + 2, i + 3];
                match v.fmt {
                    WeightFormat::Q8_0 | WeightFormat::Q4_0 | WeightFormat::Q5_0 => {
                        row_dot_q(kernel, v.fmt, row, acts, &idx[..n], &mut out)
                    }
                    f => super::kquant::row_dot_group(kernel, f, row, acts, &idx[..n], &mut out),
                }
                for (t, &val) in out[..n].iter().enumerate() {
                    // SAFETY: (i + t, j) is inside y, and only this call writes column j.
                    unsafe { *y.add((i + t) * ld + j) = val };
                }
                i += n;
            }
        }
    } else {
        // element formats: decode each weight row once, reuse for all rows
        let mut w = vec![0.0f32; v.cols];
        for j in j0..j1 {
            v.decode_row_into(j, &mut w);
            for i in 0..m {
                let val = super::dot(x.row(i), &w);
                // SAFETY: as above.
                unsafe { *y.add(i * ld + j) = val };
            }
        }
    }
}

/// Exact-mode `gemm_range`: fixed-point sums (`i64`, grid 2^-32) of
/// `x · W[j0..j1]ᵀ` written into `y` (`x.rows × ld`, column offset `j0`).
/// Element formats sum 32-element chunks in order, one canonical product each.
#[allow(clippy::too_many_arguments)]
fn gemm_range_fixed(
    x: &Tensor2D,
    acts: Option<&Q8Acts>,
    v: Option<&QuantView>,
    dense: Option<&Tensor2D>,
    j0: usize,
    j1: usize,
    y: *mut i64,
    ld: usize,
    kernel: DotKernel,
) {
    use super::kquant::to_fix;
    let m = x.rows;
    let put = |i: usize, j: usize, val: i64| {
        // SAFETY: (i, j) is inside y, and only this call writes column j.
        unsafe { *y.add(i * ld + j) = val };
    };
    match (v, acts) {
        (Some(v), Some(acts)) => {
            let mut out = [0i64; 4];
            for j in j0..j1 {
                let row = v.row_bytes(j);
                let mut i = 0;
                while i < m {
                    let n = (m - i).min(4);
                    let idx = [i, i + 1, i + 2, i + 3];
                    row_dot_fixed(kernel, v.fmt, row, acts, &idx[..n], &mut out);
                    for (t, &val) in out[..n].iter().enumerate() {
                        put(i + t, j, val);
                    }
                    i += n;
                }
            }
        }
        _ => {
            let k = x.cols;
            let mut w = vec![0.0f32; k];
            for j in j0..j1 {
                let wr: &[f32] = match (v, dense) {
                    (Some(v), _) => {
                        v.decode_row_into(j, &mut w);
                        &w
                    }
                    (None, Some(d)) => d.row(j),
                    _ => unreachable!(),
                };
                for i in 0..m {
                    let xr = x.row(i);
                    let mut a = 0i64;
                    for c in (0..k).step_by(QK) {
                        let e = (c + QK).min(k);
                        let mut p = 0.0f32;
                        for t in c..e {
                            p += xr[t] * wr[t];
                        }
                        a += to_fix(p);
                    }
                    put(i, j, a);
                }
            }
        }
    }
}

/// Exact-mode GEMM: `x[m, k] · Wᵀ` as fixed-point sums (`m × n` `i64`,
/// grid 2^-32; see [`super::kquant::to_fix`]). Every output element is the
/// exact integer sum of canonical per-block products, so it is identical
/// for any row tiling or lane striping, and the sum of the results over a
/// block-aligned column split of `W` equals the result for the whole `W`.
pub fn gemm_fixed(x: &Tensor2D, v: Option<&QuantView>, dense: Option<&Tensor2D>) -> Vec<i64> {
    let (m, k) = (x.rows, x.cols);
    let n = v
        .map(|v| v.rows)
        .or(dense.map(|d| d.rows))
        .expect("weights");
    if let Some(v) = v.filter(|_| super::accel::cuda_active()) {
        if let Some(y) = super::cuda::gemm_fixed(x, v) {
            return y;
        }
    }
    let acts = v.filter(|v| v.fmt.is_block()).map(|_| Q8Acts::quantize(x));
    let kernel = active_kernel();
    let mut y = vec![0i64; m * n];
    let yp = super::pool::SyncPtr(y.as_mut_ptr());
    if m * n * k >= LANE_MIN_MACS {
        if let Some(pool) = super::pool::current() {
            let lanes = pool.lanes();
            pool.run(lanes, &|t| {
                let (j0, j1) = (n * t / lanes, n * (t + 1) / lanes);
                gemm_range_fixed(x, acts.as_ref(), v, dense, j0, j1, yp.ptr(), n, kernel);
            });
            return y;
        }
    }
    gemm_range_fixed(x, acts.as_ref(), v, dense, 0, n, yp.ptr(), n, kernel);
    y
}

/// [`gemm_fixed`] with an explicit kernel, single lane (tests).
pub fn gemm_fixed_with(x: &Tensor2D, v: &QuantView, kernel: DotKernel) -> Vec<i64> {
    let acts = v.fmt.is_block().then(|| Q8Acts::quantize(x));
    let mut y = vec![0i64; x.rows * v.rows];
    let yp = super::pool::SyncPtr(y.as_mut_ptr());
    gemm_range_fixed(
        x,
        acts.as_ref(),
        Some(v),
        None,
        0,
        v.rows,
        yp.ptr(),
        v.rows,
        kernel,
    );
    y
}

/// Below this many multiply-accumulates a quantized GEMM stays on one lane.
const LANE_MIN_MACS: usize = 1 << 17;

/// y[m, n] = x[m, k] · Wᵀ for a quantized / half-precision weight view.
/// Large GEMMs stripe their output rows across the node's lane pool (the
/// same pool every engine installs); on macOS with the Accelerate backend,
/// prefill-sized batches dequantize weight tiles and run `cblas_sgemm`.
pub fn gemm_quant(x: &Tensor2D, v: &QuantView, y: &mut Tensor2D) {
    assert_eq!(x.cols, v.cols, "activation width ≠ weight cols");
    assert_eq!((y.rows, y.cols), (x.rows, v.rows));
    let (m, n, k) = (x.rows, v.rows, v.cols);
    if super::accel::cuda_active() && super::cuda::gemm(x, v, y) {
        return;
    }
    if super::accel::accelerate_active() && m >= super::accel::ACCEL_MIN_ROWS {
        super::accel::gemm_dequant_tiles(x, v, y);
        return;
    }
    let acts = v.fmt.is_block().then(|| Q8Acts::quantize(x));
    let kernel = active_kernel();
    let yp = super::pool::SyncPtr(y.data.as_mut_ptr());
    if m * n * k >= LANE_MIN_MACS {
        if let Some(pool) = super::pool::current() {
            let lanes = pool.lanes();
            pool.run(lanes, &|t| {
                let (j0, j1) = (n * t / lanes, n * (t + 1) / lanes);
                gemm_range(x, acts.as_ref(), v, j0, j1, yp.ptr(), n, kernel);
            });
            return;
        }
    }
    gemm_range(x, acts.as_ref(), v, 0, n, yp.ptr(), n, kernel);
}

/// Same as [`gemm_quant`] with an explicit kernel (benchmarks / tests).
pub fn gemm_quant_with(x: &Tensor2D, v: &QuantView, y: &mut Tensor2D, kernel: DotKernel) {
    let acts = v.fmt.is_block().then(|| Q8Acts::quantize(x));
    let yp = super::pool::SyncPtr(y.data.as_mut_ptr());
    gemm_range(x, acts.as_ref(), v, 0, v.rows, yp.ptr(), v.rows, kernel);
}

/// Fused Q4_0 × Q8_0 dot of one weight row and one activation vector.
pub fn vec_dot_q4_0_q8_0(w_row: &[u8], x: &[f32]) -> f32 {
    vec_dot_block(WeightFormat::Q4_0, w_row, x)
}

/// Fused Q4_K × Q8 dot of one weight row and one activation vector.
pub fn vec_dot_q4_k(w_row: &[u8], x: &[f32]) -> f32 {
    vec_dot_block(WeightFormat::Q4_K, w_row, x)
}

/// Fused Q6_K × Q8 dot of one weight row and one activation vector.
pub fn vec_dot_q6_k(w_row: &[u8], x: &[f32]) -> f32 {
    vec_dot_block(WeightFormat::Q6_K, w_row, x)
}

/// Fused Q8_0 × Q8_0 dot of one weight row and one activation vector.
pub fn vec_dot_q8_0_q8_0(w_row: &[u8], x: &[f32]) -> f32 {
    vec_dot_block(WeightFormat::Q8_0, w_row, x)
}

fn vec_dot_block(fmt: WeightFormat, w_row: &[u8], x: &[f32]) -> f32 {
    let xt = Tensor2D::from_vec(1, x.len(), x.to_vec());
    let acts = Q8Acts::quantize(&xt);
    let mut out = [0.0f32; 4];
    match fmt {
        WeightFormat::Q8_0 | WeightFormat::Q4_0 | WeightFormat::Q5_0 => {
            row_dot_q(active_kernel(), fmt, w_row, &acts, &[0], &mut out)
        }
        f => super::kquant::row_dot_group(active_kernel(), f, w_row, &acts, &[0], &mut out),
    }
    out[0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Rng;

    #[test]
    fn half_formats_are_exact_where_representable() {
        // every finite f16 survives f16 → f32 → f16; every bf16 likewise
        for h in 0..=u16::MAX {
            let f = f16_to_f32(h);
            if f.is_finite() {
                assert_eq!(f32_to_f16(f), h, "f16 {h:#06x}");
            }
            let g = bf16_to_f32(h);
            if g.is_finite() {
                assert_eq!(f32_to_bf16(g), h, "bf16 {h:#06x}");
            }
        }
        // the branch-free decoder is bit-exact against the reference codec
        for h in 0..=u16::MAX {
            let (a, b) = (f16_fast(h), f16_to_f32(h));
            assert!(
                a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
                "f16 {h:#06x}: {a} vs {b}"
            );
        }
        assert_eq!(f32_to_bf16(1.0 + 1.0 / 256.0), 0x3f80); // tie → even
        assert_eq!(f32_to_bf16(1.0 + 3.0 / 256.0), 0x3f82);
    }

    /// A view with random valid blocks of any block format (decode-only
    /// formats included): random code bytes, small positive f16 scales.
    fn random_view(fmt: WeightFormat, rows: usize, cols: usize, rng: &mut Rng) -> QuantView {
        let bb = super::super::kquant::block_bytes(fmt);
        let nblk = rows * cols / fmt.block_len();
        let mut b: Vec<u8> = (0..nblk * bb).map(|_| rng.below(256) as u8).collect();
        let scale_at: &[usize] = match fmt {
            WeightFormat::Q6_K => &[208],
            WeightFormat::Q4_1 | WeightFormat::Q5_1 | WeightFormat::Q4_K | WeightFormat::Q5_K => {
                &[0, 2]
            }
            _ => &[0],
        };
        for k in 0..nblk {
            for &o in scale_at {
                let v = f32_to_f16(0.001 + 0.01 * rng.uniform());
                b[k * bb + o..k * bb + o + 2].copy_from_slice(&v.to_le_bytes());
            }
        }
        QuantView::new(Arc::new(b), 0, fmt, rows, cols)
    }

    const ALL_BLOCK: [WeightFormat; 8] = [
        WeightFormat::Q8_0,
        WeightFormat::Q4_0,
        WeightFormat::Q4_1,
        WeightFormat::Q5_0,
        WeightFormat::Q5_1,
        WeightFormat::Q4_K,
        WeightFormat::Q5_K,
        WeightFormat::Q6_K,
    ];

    fn kernels() -> Vec<DotKernel> {
        let mut k = vec![DotKernel::Scalar];
        #[cfg(target_arch = "aarch64")]
        k.push(DotKernel::Neon);
        if best_kernel() != DotKernel::Scalar && !k.contains(&best_kernel()) {
            k.push(best_kernel());
        }
        k
    }

    #[test]
    fn every_kernel_matches_the_scalar_reference() {
        let mut rng = Rng::new(3);
        // 512 columns: two K-quant super-blocks, an even number of 32-blocks;
        // 7 activation rows: one 4-row tile plus a 3-row remainder
        let x = Tensor2D::randn(7, 512, 1.0, &mut rng);
        for fmt in ALL_BLOCK {
            let v = random_view(fmt, 19, 512, &mut rng);
            let deq = v.dequantize();
            let mut want = Tensor2D::zeros(7, 19);
            gemm_xwt_reference(&x, &deq, &mut want);
            let mut a = Tensor2D::zeros(7, 19);
            gemm_quant_with(&x, &v, &mut a, DotKernel::Scalar);
            for k in kernels() {
                let mut b = Tensor2D::zeros(7, 19);
                gemm_quant_with(&x, &v, &mut b, k);
                for (e, ((p, q), w)) in a.data.iter().zip(&b.data).zip(&want.data).enumerate() {
                    assert!(
                        (p - q).abs() <= 1e-4 * (1.0 + p.abs()),
                        "{fmt:?} {k:?}: {p} vs {q}"
                    );
                    // only the Q8 activation rounding (|δx| ≤ amax/254 per
                    // element) separates the kernel from the f32 product
                    let wn = crate::tensor::l2_norm(deq.row(e % 19));
                    assert!(
                        (p - w).abs() <= 0.03 * wn,
                        "{fmt:?}: {p} vs f32 {w} (‖w‖ {wn})"
                    );
                }
            }
        }
    }

    fn gemm_xwt_reference(x: &Tensor2D, w: &Tensor2D, y: &mut Tensor2D) {
        for i in 0..x.rows {
            for j in 0..w.rows {
                y.data[i * w.rows + j] = x
                    .row(i)
                    .iter()
                    .zip(w.row(j))
                    .map(|(a, b)| *a as f64 * *b as f64)
                    .sum::<f64>() as f32;
            }
        }
    }

    #[test]
    fn exact_accumulation_is_kernel_tiling_and_split_invariant() {
        use super::super::kquant::from_fix;
        let mut rng = Rng::new(9);
        let x = Tensor2D::randn(6, 1024, 1.0, &mut rng);
        for fmt in ALL_BLOCK {
            let v = random_view(fmt, 11, 1024, &mut rng);
            let full = gemm_fixed_with(&x, &v, DotKernel::Scalar);
            // every kernel, bit for bit
            for k in kernels() {
                assert_eq!(gemm_fixed_with(&x, &v, k), full, "{fmt:?} {k:?}");
            }
            // one row at a time (decode) == the 4-row tiles (verification)
            for i in 0..x.rows {
                let xi = Tensor2D::from_vec(1, x.cols, x.row(i).to_vec());
                assert_eq!(
                    gemm_fixed_with(&xi, &v, active_kernel()),
                    full[i * 11..(i + 1) * 11].to_vec(),
                    "{fmt:?} row {i}"
                );
            }
            // a row-parallel split (block aligned) summed as integers == whole
            let cut = 512;
            let xa = Tensor2D::from_vec(
                6,
                cut,
                (0..6).flat_map(|i| x.row(i)[..cut].to_vec()).collect(),
            );
            let xb = Tensor2D::from_vec(
                6,
                1024 - cut,
                (0..6).flat_map(|i| x.row(i)[cut..].to_vec()).collect(),
            );
            let pa = gemm_fixed_with(&xa, &v.slice_cols(0, cut), active_kernel());
            let pb = gemm_fixed_with(&xb, &v.slice_cols(cut, 1024), active_kernel());
            let sum: Vec<i64> = pa.iter().zip(&pb).map(|(a, b)| a + b).collect();
            assert_eq!(sum, full, "{fmt:?} split");
            // and the value is the float result up to accumulation order
            let mut f = Tensor2D::zeros(6, 11);
            gemm_quant_with(&x, &v, &mut f, active_kernel());
            for (a, b) in full.iter().zip(&f.data) {
                assert!(
                    (from_fix(*a) - b).abs() <= 1e-4 * (1.0 + b.abs()),
                    "{fmt:?}: {} vs {b}",
                    from_fix(*a)
                );
            }
        }
    }
}
