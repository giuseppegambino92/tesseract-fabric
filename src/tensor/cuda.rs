//! NVIDIA CUDA backend, loaded at run time.
//!
//! Like the Accelerate binding on macOS, this needs no crate and no build
//! step. On Linux the driver (`libcuda.so.1`), the run-time compiler
//! (`libnvrtc.so`) and, when present, cuBLAS (`libcublas.so`) are opened
//! with `dlopen` the first time the backend is asked for. A binary built on a
//! machine without NVIDIA software therefore builds and runs the same; the
//! backend simply reports itself unavailable.
//!
//! * **Kernels**: the CUDA C source below is compiled once per process by
//!   NVRTC for the GPU's own architecture (`sm_XY` CUBIN, PTX as fallback).
//!   `qmv` is the decode kernel: one warp per weight row, the row's GGUF
//!   blocks (Q4_0/1, Q5_0/1, Q8_0, Q4_K, Q5_K, Q6_K, F16, BF16, F32) are
//!   decoded in registers and dotted with up to 8 f32 activation rows, so
//!   the weights are read once per call.
//! * **cuBLAS**: batches of at least [`CUBLAS_MIN_ROWS`] rows (prefill)
//!   dequantize weight tiles to f32 in VRAM (`dequant`) and run
//!   `cublasSgemm`. Without cuBLAS, they loop the `qmv` kernel 8 rows at a
//!   time.
//! * **Weights stay in VRAM**: a matrix is uploaded the first time it is
//!   used (or by [`preload`]) and kept for the process lifetime. If VRAM runs
//!   out, the remaining matrices stay on the CPU kernels.
//!
//! Activations are f32 on the GPU (the CPU kernels quantize them to Q8
//! blocks), so results differ from the portable backend by rounding only.
//! Exact-accumulation layers (`Linear::exact`) always stay on the CPU.

use super::quant::QuantView;
use super::Tensor2D;

/// Batches with at least this many rows use dequantize + `cublasSgemm`.
pub const CUBLAS_MIN_ROWS: usize = 32;

/// Activation rows per `qmv` launch.
pub const QMV_ROWS: usize = 8;

/// Whether this build can try CUDA at all (Linux only).
pub const CUDA_SUPPORTED: bool = cfg!(target_os = "linux");

/// True when a CUDA GPU and the libraries were found and the kernels built.
pub fn available() -> bool {
    imp::available()
}

/// "NVIDIA GeForce RTX 5090 · sm_120 · 31.4 GiB" (None without a GPU).
pub fn device_name() -> Option<String> {
    imp::device_name()
}

/// Why the backend is unavailable (library / driver / compiler message).
pub fn init_error() -> Option<String> {
    imp::init_error()
}

/// `y = x · vᵀ` on the GPU. Returns false (caller falls back to the CPU
/// kernels) when CUDA is unavailable or the matrix did not fit in VRAM.
pub fn gemm(x: &Tensor2D, v: &QuantView, y: &mut Tensor2D) -> bool {
    if !trace::on() {
        return imp::gemm(x, v, y);
    }
    let t = std::time::Instant::now();
    let ok = imp::gemm(x, v, y);
    trace::record(ok, x.rows, t.elapsed());
    ok
}

/// `TESSERACT_CUDA_TRACE=1`: every 2000 products, print how many ran on the
/// GPU, their mean latency and the share of wall time they took.
mod trace {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    use std::sync::OnceLock;
    use std::time::{Duration, Instant};

    static ON: OnceLock<bool> = OnceLock::new();
    static START: OnceLock<Instant> = OnceLock::new();
    static CALLS: AtomicU64 = AtomicU64::new(0);
    static FALLBACK: AtomicU64 = AtomicU64::new(0);
    static ROWS: AtomicU64 = AtomicU64::new(0);
    static NS: AtomicU64 = AtomicU64::new(0);

    pub fn on() -> bool {
        *ON.get_or_init(|| std::env::var("TESSERACT_CUDA_TRACE").is_ok_and(|v| v == "1"))
    }

    pub fn record(ok: bool, rows: usize, d: Duration) {
        let start = *START.get_or_init(Instant::now);
        if !ok {
            FALLBACK.fetch_add(1, Relaxed);
        }
        ROWS.fetch_add(rows as u64, Relaxed);
        let ns = NS.fetch_add(d.as_nanos() as u64, Relaxed) + d.as_nanos() as u64;
        let n = CALLS.fetch_add(1, Relaxed) + 1;
        if n.is_multiple_of(2000) {
            let wall = start.elapsed().as_nanos() as f64;
            eprintln!(
                "[cuda trace] {n} products ({} on CPU), {:.1} rows avg, {:.1} µs avg, {:.0} % of wall time",
                FALLBACK.load(Relaxed),
                ROWS.load(Relaxed) as f64 / n as f64,
                ns as f64 / n as f64 / 1e3,
                100.0 * ns as f64 / wall
            );
        }
    }
}

/// Exact-mode product (`Linear::exact`): the fixed-point sums of
/// [`super::quant::gemm_fixed`], bit for bit. None when CUDA is unavailable,
/// the format is not a block format or the matrix did not fit in VRAM.
pub fn gemm_fixed(x: &Tensor2D, v: &QuantView) -> Option<Vec<i64>> {
    if !v.fmt.is_block() {
        return None;
    }
    if !trace::on() {
        return imp::gemm_fixed(x, v);
    }
    let t = std::time::Instant::now();
    let r = imp::gemm_fixed(x, v);
    trace::record(r.is_some(), x.rows, t.elapsed());
    r
}

/// Upload a weight matrix now instead of on first use.
pub fn preload(v: &QuantView) -> bool {
    imp::preload(v)
}

/// What a GPU node advertises to the planner, measured on the device.
#[derive(Clone, Copy, Debug)]
pub struct GpuProfile {
    /// Device-to-device copy bandwidth (read + write), GB/s.
    pub mem_bandwidth_gbps: f64,
    /// `cublasSgemm` throughput on 2048³, TFLOPS (0 without cuBLAS).
    pub tflops_fp32: f64,
    /// Free VRAM, MB.
    pub free_vram_mb: f64,
}

/// Measure the GPU (≈ 0.1 s). None when CUDA is unavailable.
pub fn measure() -> Option<GpuProfile> {
    imp::measure()
}

/// Bytes of weights resident in VRAM.
pub fn resident_bytes() -> usize {
    imp::resident_bytes()
}

/// Whether cuBLAS was loaded (prefill-sized batches use it).
pub fn cublas_loaded() -> bool {
    imp::cublas_loaded()
}

/// The kernels, compiled at run time by NVRTC.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
const KERNELS: &str = r#"
typedef unsigned char u8;
typedef unsigned int u32;
typedef long long i64;

__device__ __forceinline__ float h2f(const u8* p) {
    unsigned short h = (unsigned short)p[0] | ((unsigned short)p[1] << 8);
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

__device__ __forceinline__ void scale_min_k4(int j, const u8* q, int* sc, int* mn) {
    if (j < 4) {
        *sc = q[j] & 63;
        *mn = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *mn = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// GGML type ids
#define F_F32 0
#define F_F16 1
#define F_Q4_0 2
#define F_Q4_1 3
#define F_Q5_0 6
#define F_Q5_1 7
#define F_Q8_0 8
#define F_Q4_K 12
#define F_Q5_K 13
#define F_Q6_K 14
#define F_BF16 30

// The 32 weights of group g (elements 32g .. 32g+31) of one row of k
// elements. Element formats zero-fill past k.
__device__ __forceinline__ void decode32(int fmt, const u8* row, int g, int k, float* w) {
    switch (fmt) {
    case F_F32: {
        const float* p = (const float*)row + g * 32;
#pragma unroll
        for (int i = 0; i < 32; i++) w[i] = (g * 32 + i < k) ? p[i] : 0.f;
        break;
    }
    case F_F16: {
        const u8* p = row + g * 64;
#pragma unroll
        for (int i = 0; i < 32; i++) w[i] = (g * 32 + i < k) ? h2f(p + 2 * i) : 0.f;
        break;
    }
    case F_BF16: {
        const u8* p = row + g * 64;
#pragma unroll
        for (int i = 0; i < 32; i++) {
            u32 b = (u32)p[2 * i] | ((u32)p[2 * i + 1] << 8);
            w[i] = (g * 32 + i < k) ? __int_as_float((int)(b << 16)) : 0.f;
        }
        break;
    }
    case F_Q4_0: {
        const u8* b = row + g * 18;
        float d = h2f(b);
#pragma unroll
        for (int i = 0; i < 16; i++) {
            w[i] = d * (float)((b[2 + i] & 15) - 8);
            w[i + 16] = d * (float)((b[2 + i] >> 4) - 8);
        }
        break;
    }
    case F_Q4_1: {
        const u8* b = row + g * 20;
        float d = h2f(b), m = h2f(b + 2);
#pragma unroll
        for (int i = 0; i < 16; i++) {
            w[i] = d * (float)(b[4 + i] & 15) + m;
            w[i + 16] = d * (float)(b[4 + i] >> 4) + m;
        }
        break;
    }
    case F_Q5_0:
    case F_Q5_1: {
        bool one = fmt == F_Q5_1;
        const u8* b = row + g * (one ? 24 : 22);
        float d = h2f(b), m = one ? h2f(b + 2) : 0.f;
        const u8* h = b + (one ? 4 : 2);
        u32 qh = (u32)h[0] | ((u32)h[1] << 8) | ((u32)h[2] << 16) | ((u32)h[3] << 24);
        const u8* qs = h + 4;
        int off = one ? 0 : 16;
#pragma unroll
        for (int i = 0; i < 16; i++) {
            int lo = (qs[i] & 15) | (((qh >> i) & 1) << 4);
            int hi = (qs[i] >> 4) | (((qh >> (i + 16)) & 1) << 4);
            w[i] = d * (float)(lo - off) + m;
            w[i + 16] = d * (float)(hi - off) + m;
        }
        break;
    }
    case F_Q8_0: {
        const u8* b = row + g * 34;
        float d = h2f(b);
#pragma unroll
        for (int i = 0; i < 32; i++) w[i] = d * (float)(signed char)b[2 + i];
        break;
    }
    case F_Q4_K:
    case F_Q5_K: {
        bool five = fmt == F_Q5_K;
        int j = g & 7;
        const u8* b = row + (g >> 3) * (five ? 176 : 144);
        float d = h2f(b), dmin = h2f(b + 2);
        int sc, mn;
        scale_min_k4(j, b + 4, &sc, &mn);
        float ds = d * (float)sc, dm = dmin * (float)mn;
        const u8* ql = b + (five ? 48 : 16) + 32 * (j >> 1);
        const u8* qh = b + 16;
        int sh = (j & 1) * 4;
        if (!five && ((size_t)ql & 3) == 0) {
            // 144-byte blocks of 16-byte aligned rows: word loads
            const u32* q4 = (const u32*)ql;
#pragma unroll
            for (int t = 0; t < 8; t++) {
                u32 v = (q4[t] >> sh) & 0x0F0F0F0Fu;
#pragma unroll
                for (int c = 0; c < 4; c++) w[4 * t + c] = ds * (float)((v >> (8 * c)) & 15) - dm;
            }
            break;
        }
#pragma unroll
        for (int l = 0; l < 32; l++) {
            int q = (ql[l] >> sh) & 15;
            if (five) q |= ((qh[l] >> j) & 1) << 4;
            w[l] = ds * (float)q - dm;
        }
        break;
    }
    case F_Q6_K: {
        const u8* b = row + (g >> 3) * 210;
        float d = h2f(b + 208);
        int e0 = (g & 7) * 32;
        int half = e0 >> 7, quarter = (e0 & 127) >> 5;
        const u8* ql = b + 64 * half + (quarter & 1) * 32;
        const u8* qh = b + 128 + 32 * half;
        const signed char* sc = (const signed char*)(b + 192) + (e0 >> 4);
        int sl = (quarter >> 1) * 4, shh = quarter * 2;
        float d0 = d * (float)sc[0], d1 = d * (float)sc[1];
#pragma unroll
        for (int l = 0; l < 32; l++) {
            int q = ((ql[l] >> sl) & 15) | (((qh[l] >> shh) & 3) << 4);
            w[l] = (l < 16 ? d0 : d1) * (float)(q - 32);
        }
        break;
    }
    default:
#pragma unroll
        for (int i = 0; i < 32; i++) w[i] = 0.f;
    }
}

// Y[r, row] = sum_k W[row, k] * X[r, k] for r < m <= 8; one warp per row.
extern "C" __global__ void qmv(const u8* W, i64 row_bytes, int n, int k, int fmt,
                               const float* X, int m, float* Y, int ldy) {
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (row >= n) return;
    const u8* wr = W + (i64)row * row_bytes;
    int groups = (k + 31) >> 5;
    bool vec = (k & 3) == 0;
    float acc[8];
#pragma unroll
    for (int r = 0; r < 8; r++) acc[r] = 0.f;
    for (int g = lane; g < groups; g += 32) {
        float w[32];
        decode32(fmt, wr, g, k, w);
        bool full = vec && g * 32 + 32 <= k;
#pragma unroll
        for (int r = 0; r < 8; r++) {
            if (r < m) {
                const float* x = X + (i64)r * k + g * 32;
                float s = 0.f;
                if (full) {
                    const float4* x4 = (const float4*)x;
#pragma unroll
                    for (int i = 0; i < 8; i++) {
                        float4 v = x4[i];
                        s += w[4 * i] * v.x + w[4 * i + 1] * v.y + w[4 * i + 2] * v.z + w[4 * i + 3] * v.w;
                    }
                } else {
                    for (int i = 0; i < 32; i++)
                        if (g * 32 + i < k) s += w[i] * x[i];
                }
                acc[r] += s;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < 8; r++) {
        if (r < m) {
            float v = acc[r];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
            if (lane == 0) Y[(i64)r * ldy + row] = v;
        }
    }
}

// Exact mode (`Linear::exact`): the integer weights of group g plus the f32
// factors of the canonical partial product (see kquant::row_dot_group_fixed):
//   two  = 0: p = dx * (s0 * (lo + hi))  [+ add * ds when the format has mins]
//   two  = 1: p = dx * (s0 * lo + s1 * hi)                       (Q6_K)
__device__ __forceinline__ void decode32i(int fmt, const u8* row, int g, int* q,
                                          float* s0, float* s1, float* add, int* two) {
    *add = 0.f;
    *two = 0;
    switch (fmt) {
    case F_Q8_0: {
        const u8* b = row + g * 34;
        *s0 = h2f(b);
        for (int i = 0; i < 32; i++) q[i] = (signed char)b[2 + i];
        break;
    }
    case F_Q4_0: {
        const u8* b = row + g * 18;
        *s0 = h2f(b);
        for (int i = 0; i < 16; i++) {
            q[i] = (b[2 + i] & 15) - 8;
            q[i + 16] = (b[2 + i] >> 4) - 8;
        }
        break;
    }
    case F_Q4_1: {
        const u8* b = row + g * 20;
        *s0 = h2f(b);
        *add = h2f(b + 2);
        for (int i = 0; i < 16; i++) {
            q[i] = b[4 + i] & 15;
            q[i + 16] = b[4 + i] >> 4;
        }
        break;
    }
    case F_Q5_0:
    case F_Q5_1: {
        bool one = fmt == F_Q5_1;
        const u8* b = row + g * (one ? 24 : 22);
        *s0 = h2f(b);
        if (one) *add = h2f(b + 2);
        const u8* h = b + (one ? 4 : 2);
        u32 qh = (u32)h[0] | ((u32)h[1] << 8) | ((u32)h[2] << 16) | ((u32)h[3] << 24);
        const u8* qs = h + 4;
        int off = one ? 0 : 16;
        for (int i = 0; i < 16; i++) {
            q[i] = ((qs[i] & 15) | (((qh >> i) & 1) << 4)) - off;
            q[i + 16] = ((qs[i] >> 4) | (((qh >> (i + 16)) & 1) << 4)) - off;
        }
        break;
    }
    case F_Q4_K:
    case F_Q5_K: {
        bool five = fmt == F_Q5_K;
        int j = g & 7;
        const u8* b = row + (g >> 3) * (five ? 176 : 144);
        int sc, mn;
        scale_min_k4(j, b + 4, &sc, &mn);
        *s0 = __fmul_rn(h2f(b), (float)sc);
        *add = __fmul_rn(-h2f(b + 2), (float)mn);
        const u8* ql = b + (five ? 48 : 16) + 32 * (j >> 1);
        const u8* qh = b + 16;
        int sh = (j & 1) * 4;
        for (int l = 0; l < 32; l++) {
            int v = (ql[l] >> sh) & 15;
            if (five) v |= ((qh[l] >> j) & 1) << 4;
            q[l] = v;
        }
        break;
    }
    case F_Q6_K: {
        const u8* b = row + (g >> 3) * 210;
        float d = h2f(b + 208);
        int e0 = (g & 7) * 32;
        int half = e0 >> 7, quarter = (e0 & 127) >> 5;
        const u8* ql = b + 64 * half + (quarter & 1) * 32;
        const u8* qh = b + 128 + 32 * half;
        const signed char* sc = (const signed char*)(b + 192) + (e0 >> 4);
        int sl = (quarter >> 1) * 4, shh = quarter * 2;
        *s0 = __fmul_rn(d, (float)sc[0]);
        *s1 = __fmul_rn(d, (float)sc[1]);
        *two = 1;
        for (int l = 0; l < 32; l++)
            q[l] = (((ql[l] >> sl) & 15) | (((qh[l] >> shh) & 3) << 4)) - 32;
        break;
    }
    default:
        *s0 = 0.f;
        for (int i = 0; i < 32; i++) q[i] = 0;
    }
}

// Y[r, row] = sum over 32-element groups of to_fix(canonical partial product),
// with Q8 activations (XQ int8, XD scales, XDS = d * sum q per block). Integer
// sums are order independent, so the result is bit-identical to the CPU's.
extern "C" __global__ void qmv_fixed(const u8* W, i64 row_bytes, int n, int k, int fmt,
                                     const signed char* XQ, const float* XD, const float* XDS,
                                     int m, i64* Y, int ldy) {
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (row >= n) return;
    const u8* wr = W + (i64)row * row_bytes;
    int groups = k >> 5;
    i64 acc[8];
#pragma unroll
    for (int r = 0; r < 8; r++) acc[r] = 0;
    for (int g = lane; g < groups; g += 32) {
        int q[32];
        float s0, s1, add;
        int two;
        decode32i(fmt, wr, g, q, &s0, &s1, &add, &two);
        int wp[8];
#pragma unroll
        for (int i = 0; i < 8; i++)
            wp[i] = (q[4 * i] & 255) | ((q[4 * i + 1] & 255) << 8) | ((q[4 * i + 2] & 255) << 16) |
                    ((q[4 * i + 3] & 255) << 24);
#pragma unroll
        for (int r = 0; r < 8; r++) {
            if (r < m) {
                const int* xq = (const int*)(XQ + (i64)r * k + g * 32);
                int lo = 0, hi = 0;
#pragma unroll
                for (int i = 0; i < 4; i++) lo = __dp4a(wp[i], xq[i], lo);
#pragma unroll
                for (int i = 4; i < 8; i++) hi = __dp4a(wp[i], xq[i], hi);
                float dx = XD[(i64)r * groups + g];
                float p;
                if (two) {
                    p = __fmul_rn(dx, __fadd_rn(__fmul_rn(s0, (float)lo), __fmul_rn(s1, (float)hi)));
                } else {
                    p = __fmul_rn(dx, __fmul_rn(s0, (float)(lo + hi)));
                    if (fmt == F_Q4_1 || fmt == F_Q5_1 || fmt == F_Q4_K || fmt == F_Q5_K)
                        p = __fadd_rn(p, __fmul_rn(add, XDS[(i64)r * groups + g]));
                }
                acc[r] += (i64)((double)p * 4294967296.0);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < 8; r++) {
        if (r < m) {
            i64 v = acc[r];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
            if (lane == 0) Y[(i64)r * ldy + row] = v;
        }
    }
}

// out[row, :] = decoded W[row, :] (row-major f32, k per row).
extern "C" __global__ void dequant(const u8* W, i64 row_bytes, int n, int k, int fmt, float* out) {
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    int groups = (k + 31) >> 5;
    if (t >= (i64)n * groups) return;
    int row = (int)(t / groups), g = (int)(t % groups);
    float w[32];
    decode32(fmt, W + (i64)row * row_bytes, g, k, w);
    float* o = out + (i64)row * k + g * 32;
#pragma unroll
    for (int i = 0; i < 32; i++)
        if (g * 32 + i < k) o[i] = w[i];
}
"#;

#[cfg(target_os = "linux")]
mod imp {
    use super::super::quant::{Q8Acts, QuantView};
    use super::super::Tensor2D;
    use super::{CUBLAS_MIN_ROWS, KERNELS, QMV_ROWS};
    use std::collections::{HashMap, HashSet};
    use std::ffi::{c_char, c_int, c_void, CStr, CString};
    use std::sync::{Mutex, OnceLock};

    extern "C" {
        fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        fn dlerror() -> *mut c_char;
    }
    const RTLD_NOW: c_int = 2;

    type Ptr = *mut c_void;
    type DevPtr = u64;

    /// Open the first library of `names` that loads.
    fn open(names: &[&str]) -> Result<Ptr, String> {
        let mut last = String::new();
        for n in names {
            let c = CString::new(*n).unwrap();
            // SAFETY: dlopen with a valid C string; a null result is handled.
            let h = unsafe { dlopen(c.as_ptr(), RTLD_NOW) };
            if !h.is_null() {
                return Ok(h);
            }
            // SAFETY: dlerror returns a thread-local C string or null.
            let e = unsafe { dlerror() };
            if !e.is_null() {
                last = unsafe { CStr::from_ptr(e) }.to_string_lossy().into_owned();
            }
        }
        Err(last)
    }

    /// Resolve `name` in `lib` as a function pointer of type `T`.
    fn sym<T: Copy>(lib: Ptr, name: &str) -> Result<T, String> {
        let c = CString::new(name).unwrap();
        // SAFETY: dlsym on a live handle.
        let p = unsafe { dlsym(lib, c.as_ptr()) };
        if p.is_null() {
            return Err(format!("missing symbol {name}"));
        }
        assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<Ptr>());
        // SAFETY: T is a function-pointer type of the symbol's C signature.
        Ok(unsafe { std::mem::transmute_copy::<Ptr, T>(&p) })
    }

    #[allow(clippy::type_complexity)]
    struct Driver {
        get_error_name: unsafe extern "C" fn(c_int, *mut *const c_char) -> c_int,
        ctx_set_current: unsafe extern "C" fn(Ptr) -> c_int,
        mem_alloc: unsafe extern "C" fn(*mut DevPtr, usize) -> c_int,
        mem_free: unsafe extern "C" fn(DevPtr) -> c_int,
        mem_get_info: unsafe extern "C" fn(*mut usize, *mut usize) -> c_int,
        htod: unsafe extern "C" fn(DevPtr, *const c_void, usize) -> c_int,
        dtoh: unsafe extern "C" fn(*mut c_void, DevPtr, usize) -> c_int,
        dtod: unsafe extern "C" fn(DevPtr, DevPtr, usize) -> c_int,
        memset32: unsafe extern "C" fn(DevPtr, u32, usize) -> c_int,
        sync: unsafe extern "C" fn() -> c_int,
        launch: unsafe extern "C" fn(
            Ptr,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            Ptr,
            *mut Ptr,
            *mut Ptr,
        ) -> c_int,
    }

    #[allow(clippy::type_complexity)]
    struct Cublas {
        handle: Ptr,
        sgemm: unsafe extern "C" fn(
            Ptr,
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            *const f32,
            DevPtr,
            c_int,
            DevPtr,
            c_int,
            *const f32,
            DevPtr,
            c_int,
        ) -> c_int,
    }

    struct Weights {
        ptr: DevPtr,
        bytes: usize,
        row_bytes: usize,
        /// Keeps the byte source alive, so its address cannot be reused by
        /// another matrix while this entry exists.
        _view: QuantView,
    }

    /// A grow-only device buffer.
    #[derive(Default)]
    struct Scratch {
        ptr: DevPtr,
        bytes: usize,
    }

    struct State {
        drv: Driver,
        ctx: Ptr,
        qmv: Ptr,
        qmv_fixed: Ptr,
        dequant: Ptr,
        blas: Option<Cublas>,
        name: String,
        weights: HashMap<(usize, usize, usize, usize, u32, usize), Weights>,
        too_big: HashSet<(usize, usize, usize, usize, u32, usize)>,
        resident: usize,
        x: Scratch,
        y: Scratch,
        deq: Scratch,
        reported: bool,
    }

    // SAFETY: the raw CUDA handles are only used under the STATE mutex, after
    // making the context current on the calling thread.
    unsafe impl Send for State {}

    static STATE: OnceLock<Result<Mutex<State>, String>> = OnceLock::new();

    /// Leave this much VRAM free for cuBLAS workspaces and the driver.
    const VRAM_RESERVE: usize = 768 << 20;
    /// Largest f32 tile `dequant` writes for one SGEMM.
    const DEQUANT_TILE_BYTES: usize = 256 << 20;

    fn state() -> Option<&'static Mutex<State>> {
        STATE.get_or_init(|| init().map(Mutex::new)).as_ref().ok()
    }

    pub fn available() -> bool {
        state().is_some()
    }

    pub fn init_error() -> Option<String> {
        STATE
            .get_or_init(|| init().map(Mutex::new))
            .as_ref()
            .err()
            .cloned()
    }

    pub fn device_name() -> Option<String> {
        state().map(|s| s.lock().unwrap().name.clone())
    }

    pub fn resident_bytes() -> usize {
        state().map(|s| s.lock().unwrap().resident).unwrap_or(0)
    }

    pub fn cublas_loaded() -> bool {
        state().is_some_and(|s| s.lock().unwrap().blas.is_some())
    }

    fn check(drv: &Driver, r: c_int, what: &str) -> Result<(), String> {
        if r == 0 {
            return Ok(());
        }
        let mut p: *const c_char = std::ptr::null();
        // SAFETY: cuGetErrorName writes a static string pointer.
        unsafe { (drv.get_error_name)(r, &mut p) };
        let name = if p.is_null() {
            format!("error {r}")
        } else {
            unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
        };
        Err(format!("{what}: {name}"))
    }

    fn cuda_lib_dirs() -> Vec<String> {
        let mut v = Vec::new();
        for var in ["CUDA_HOME", "CUDA_PATH"] {
            if let Ok(p) = std::env::var(var) {
                v.push(format!("{p}/lib64/"));
            }
        }
        v.push("/usr/local/cuda/lib64/".into());
        v.push(String::new()); // the loader's own search path
        v
    }

    fn open_cuda_lib(stems: &[&str]) -> Result<Ptr, String> {
        let names: Vec<String> = cuda_lib_dirs()
            .iter()
            .flat_map(|d| stems.iter().map(move |s| format!("{d}{s}")))
            .collect();
        let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        open(&refs)
    }

    /// Compile the kernels for compute capability `cc` (major·10 + minor).
    fn compile(cc: i32) -> Result<Vec<u8>, String> {
        type Prog = Ptr;
        let lib = open_cuda_lib(&[
            "libnvrtc.so",
            "libnvrtc.so.13",
            "libnvrtc.so.12",
            "libnvrtc.so.11.2",
        ])
        .map_err(|e| format!("NVRTC not found ({e})"))?;
        let create: unsafe extern "C" fn(
            *mut Prog,
            *const c_char,
            *const c_char,
            c_int,
            *const *const c_char,
            *const *const c_char,
        ) -> c_int = sym(lib, "nvrtcCreateProgram")?;
        let compile: unsafe extern "C" fn(Prog, c_int, *const *const c_char) -> c_int =
            sym(lib, "nvrtcCompileProgram")?;
        let log_size: unsafe extern "C" fn(Prog, *mut usize) -> c_int =
            sym(lib, "nvrtcGetProgramLogSize")?;
        let log: unsafe extern "C" fn(Prog, *mut c_char) -> c_int = sym(lib, "nvrtcGetProgramLog")?;
        let destroy: unsafe extern "C" fn(*mut Prog) -> c_int = sym(lib, "nvrtcDestroyProgram")?;
        let src = CString::new(KERNELS).unwrap();
        let name = CString::new("tesseract_kernels.cu").unwrap();
        let mut last = String::new();
        // native CUBIN first; PTX for the same family if this NVRTC is older
        // than the GPU's real architecture
        for (arch, cubin) in [(format!("sm_{cc}"), true), (format!("compute_{cc}"), false)] {
            let mut prog: Prog = std::ptr::null_mut();
            // SAFETY: valid C strings; no headers.
            let r = unsafe {
                create(
                    &mut prog,
                    src.as_ptr(),
                    name.as_ptr(),
                    0,
                    std::ptr::null(),
                    std::ptr::null(),
                )
            };
            if r != 0 {
                return Err(format!("nvrtcCreateProgram failed ({r})"));
            }
            let opts = [
                CString::new(format!("--gpu-architecture={arch}")).unwrap(),
                CString::new("-default-device").unwrap(),
            ];
            let optp: Vec<*const c_char> = opts.iter().map(|o| o.as_ptr()).collect();
            // SAFETY: prog is live; options are valid C strings.
            let r = unsafe { compile(prog, optp.len() as c_int, optp.as_ptr()) };
            if r == 0 {
                let (size_fn, get_fn) = if cubin {
                    ("nvrtcGetCUBINSize", "nvrtcGetCUBIN")
                } else {
                    ("nvrtcGetPTXSize", "nvrtcGetPTX")
                };
                let size: unsafe extern "C" fn(Prog, *mut usize) -> c_int = sym(lib, size_fn)?;
                let get: unsafe extern "C" fn(Prog, *mut c_char) -> c_int = sym(lib, get_fn)?;
                let mut n = 0usize;
                // SAFETY: prog compiled; buffer sized by the size query.
                unsafe { size(prog, &mut n) };
                let mut out = vec![0u8; n];
                unsafe { get(prog, out.as_mut_ptr() as *mut c_char) };
                unsafe { destroy(&mut prog) };
                return Ok(out);
            }
            let mut n = 0usize;
            // SAFETY: as above.
            unsafe { log_size(prog, &mut n) };
            let mut buf = vec![0u8; n.max(1)];
            unsafe { log(prog, buf.as_mut_ptr() as *mut c_char) };
            unsafe { destroy(&mut prog) };
            last = format!(
                "NVRTC ({arch}): {}",
                String::from_utf8_lossy(&buf).trim_end_matches('\0').trim()
            );
        }
        Err(last)
    }

    fn init() -> Result<State, String> {
        let lib = open(&["libcuda.so.1", "libcuda.so"])
            .map_err(|e| format!("NVIDIA driver not found ({e})"))?;
        let cu_init: unsafe extern "C" fn(u32) -> c_int = sym(lib, "cuInit")?;
        let device_get: unsafe extern "C" fn(*mut c_int, c_int) -> c_int = sym(lib, "cuDeviceGet")?;
        let device_count: unsafe extern "C" fn(*mut c_int) -> c_int = sym(lib, "cuDeviceGetCount")?;
        let get_name: unsafe extern "C" fn(*mut c_char, c_int, c_int) -> c_int =
            sym(lib, "cuDeviceGetName")?;
        let get_attr: unsafe extern "C" fn(*mut c_int, c_int, c_int) -> c_int =
            sym(lib, "cuDeviceGetAttribute")?;
        let total_mem: unsafe extern "C" fn(*mut usize, c_int) -> c_int =
            sym(lib, "cuDeviceTotalMem_v2")?;
        let ctx_retain: unsafe extern "C" fn(*mut Ptr, c_int) -> c_int =
            sym(lib, "cuDevicePrimaryCtxRetain")?;
        let module_load: unsafe extern "C" fn(*mut Ptr, *const c_void) -> c_int =
            sym(lib, "cuModuleLoadData")?;
        let get_function: unsafe extern "C" fn(*mut Ptr, Ptr, *const c_char) -> c_int =
            sym(lib, "cuModuleGetFunction")?;
        let drv = Driver {
            get_error_name: sym(lib, "cuGetErrorName")?,
            ctx_set_current: sym(lib, "cuCtxSetCurrent")?,
            mem_alloc: sym(lib, "cuMemAlloc_v2")?,
            mem_free: sym(lib, "cuMemFree_v2")?,
            mem_get_info: sym(lib, "cuMemGetInfo_v2")?,
            htod: sym(lib, "cuMemcpyHtoD_v2")?,
            dtoh: sym(lib, "cuMemcpyDtoH_v2")?,
            dtod: sym(lib, "cuMemcpyDtoD_v2")?,
            memset32: sym(lib, "cuMemsetD32_v2")?,
            sync: sym(lib, "cuCtxSynchronize")?,
            launch: sym(lib, "cuLaunchKernel")?,
        };
        // SAFETY (this block): driver calls with valid out-pointers.
        unsafe {
            check(&drv, cu_init(0), "cuInit")?;
            let mut count = 0;
            check(&drv, device_count(&mut count), "cuDeviceGetCount")?;
            if count == 0 {
                return Err("no CUDA device".into());
            }
            let ordinal: c_int = std::env::var("TESSERACT_CUDA_DEVICE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            let mut dev = 0;
            check(&drv, device_get(&mut dev, ordinal), "cuDeviceGet")?;
            let mut buf = [0 as c_char; 256];
            check(
                &drv,
                get_name(buf.as_mut_ptr(), 256, dev),
                "cuDeviceGetName",
            )?;
            let gpu = CStr::from_ptr(buf.as_ptr()).to_string_lossy().into_owned();
            let (mut major, mut minor) = (0, 0);
            check(&drv, get_attr(&mut major, 75, dev), "compute capability")?;
            check(&drv, get_attr(&mut minor, 76, dev), "compute capability")?;
            let mut mem = 0usize;
            check(&drv, total_mem(&mut mem, dev), "cuDeviceTotalMem")?;
            let mut ctx: Ptr = std::ptr::null_mut();
            check(&drv, ctx_retain(&mut ctx, dev), "cuDevicePrimaryCtxRetain")?;
            check(&drv, (drv.ctx_set_current)(ctx), "cuCtxSetCurrent")?;
            let cc = major * 10 + minor;
            let image = compile(cc)?;
            let mut module: Ptr = std::ptr::null_mut();
            check(
                &drv,
                module_load(&mut module, image.as_ptr() as *const c_void),
                "cuModuleLoadData",
            )?;
            let mut qmv: Ptr = std::ptr::null_mut();
            let mut dequant: Ptr = std::ptr::null_mut();
            let mut qmv_fixed: Ptr = std::ptr::null_mut();
            check(
                &drv,
                get_function(&mut qmv_fixed, module, c"qmv_fixed".as_ptr()),
                "qmv_fixed kernel",
            )?;
            check(
                &drv,
                get_function(&mut qmv, module, c"qmv".as_ptr()),
                "qmv kernel",
            )?;
            check(
                &drv,
                get_function(&mut dequant, module, c"dequant".as_ptr()),
                "dequant kernel",
            )?;
            let blas = open_cuda_lib(&["libcublas.so", "libcublas.so.13", "libcublas.so.12"])
                .ok()
                .and_then(|l| {
                    let create: unsafe extern "C" fn(*mut Ptr) -> c_int =
                        sym(l, "cublasCreate_v2").ok()?;
                    let mut handle: Ptr = std::ptr::null_mut();
                    (create(&mut handle) == 0).then_some(())?;
                    Some(Cublas {
                        handle,
                        sgemm: sym(l, "cublasSgemm_v2").ok()?,
                    })
                });
            Ok(State {
                drv,
                ctx,
                qmv,
                qmv_fixed,
                dequant,
                blas,
                name: format!(
                    "{gpu} · sm_{cc} · {:.1} GiB",
                    mem as f64 / (1u64 << 30) as f64
                ),
                weights: HashMap::new(),
                too_big: HashSet::new(),
                resident: 0,
                x: Scratch::default(),
                y: Scratch::default(),
                deq: Scratch::default(),
                reported: false,
            })
        }
    }

    impl State {
        fn grow(&mut self, which: u8, bytes: usize) -> Result<DevPtr, String> {
            let drv = &self.drv;
            let s = match which {
                0 => &mut self.x,
                1 => &mut self.y,
                _ => &mut self.deq,
            };
            if s.bytes < bytes {
                // SAFETY: freeing our own allocation, then allocating anew.
                unsafe {
                    if s.ptr != 0 {
                        (drv.mem_free)(s.ptr);
                    }
                    s.ptr = 0;
                    s.bytes = 0;
                    let want = bytes.next_power_of_two().max(1 << 20);
                    check(drv, (drv.mem_alloc)(&mut s.ptr, want), "cuMemAlloc")?;
                    s.bytes = want;
                }
            }
            Ok(s.ptr)
        }

        /// Device copy of `v` (uploaded on first use); None if it does not fit.
        fn weights(&mut self, v: &QuantView) -> Option<(DevPtr, usize)> {
            let key = (
                v.source_ptr() as usize,
                v.byte_offset(),
                v.rows,
                v.cols,
                v.fmt.ggml_id(),
                v.row_stride(),
            );
            if let Some(w) = self.weights.get(&key) {
                return Some((w.ptr, w.row_bytes));
            }
            if self.too_big.contains(&key) {
                return None;
            }
            let row_bytes = v.fmt.row_bytes(v.cols);
            let bytes = row_bytes * v.rows;
            let drv = &self.drv;
            let (mut free, mut total) = (0usize, 0usize);
            // SAFETY: valid out-pointers.
            unsafe { (drv.mem_get_info)(&mut free, &mut total) };
            if bytes + VRAM_RESERVE > free {
                self.too_big.insert(key);
                return None;
            }
            let mut ptr: DevPtr = 0;
            // SAFETY: allocation + copy of `bytes` host bytes we own or borrow.
            unsafe {
                if (drv.mem_alloc)(&mut ptr, bytes) != 0 {
                    self.too_big.insert(key);
                    return None;
                }
                let r = if v.row_stride() == row_bytes {
                    let (s, e) = v.span();
                    let src = &v.source_bytes()[s..e];
                    (drv.htod)(ptr, src.as_ptr() as *const c_void, bytes)
                } else {
                    let mut packed = Vec::with_capacity(bytes);
                    for r in 0..v.rows {
                        packed.extend_from_slice(v.row_bytes(r));
                    }
                    (drv.htod)(ptr, packed.as_ptr() as *const c_void, bytes)
                };
                if r != 0 {
                    (drv.mem_free)(ptr);
                    self.too_big.insert(key);
                    return None;
                }
            }
            self.resident += bytes;
            self.weights.insert(
                key,
                Weights {
                    ptr,
                    bytes,
                    row_bytes,
                    _view: v.clone(),
                },
            );
            Some((ptr, row_bytes))
        }

        #[allow(clippy::too_many_arguments)]
        fn launch(&self, f: Ptr, grid: u32, block: u32, args: &mut [Ptr]) -> Result<(), String> {
            // SAFETY: args point at live locals matching the kernel signature.
            let r = unsafe {
                (self.drv.launch)(
                    f,
                    grid,
                    1,
                    1,
                    block,
                    1,
                    1,
                    0,
                    std::ptr::null_mut(),
                    args.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            };
            check(&self.drv, r, "cuLaunchKernel")
        }

        fn measure(&self) -> Result<super::GpuProfile, String> {
            let drv = &self.drv;
            let time = |f: &mut dyn FnMut() -> c_int| -> Result<f64, String> {
                let mut best = f64::MAX;
                for i in 0..4 {
                    let t = std::time::Instant::now();
                    check(drv, f(), "measure")?;
                    // SAFETY: synchronizes the current context.
                    check(drv, unsafe { (drv.sync)() }, "cuCtxSynchronize")?;
                    if i > 0 {
                        best = best.min(t.elapsed().as_secs_f64());
                    }
                }
                Ok(best)
            };
            let bytes = 256usize << 20;
            let (mut a, mut b) = (0 as DevPtr, 0 as DevPtr);
            // SAFETY (this block): our own allocations, freed below.
            unsafe {
                check(drv, (drv.mem_alloc)(&mut a, bytes), "cuMemAlloc")?;
                if let Err(e) = check(drv, (drv.mem_alloc)(&mut b, bytes), "cuMemAlloc") {
                    (drv.mem_free)(a);
                    return Err(e);
                }
                (drv.memset32)(a, 0, bytes / 4);
            }
            let copy = time(&mut || unsafe { (drv.dtod)(b, a, bytes) });
            let mut tflops = 0.0;
            if let (Ok(_), Some(blas)) = (&copy, &self.blas) {
                // 2048³ SGEMM on three 16 MB matrices carved out of a and b
                let n = 2048usize;
                let (x, w, y) = (a, a + (n * n * 4) as u64, b);
                let (one, zero) = (1.0f32, 0.0f32);
                let ni = n as c_int;
                if let Ok(t) = time(&mut || unsafe {
                    (blas.sgemm)(
                        blas.handle,
                        1,
                        0,
                        ni,
                        ni,
                        ni,
                        &one,
                        w,
                        ni,
                        x,
                        ni,
                        &zero,
                        y,
                        ni,
                    )
                }) {
                    tflops = 2.0 * (n * n * n) as f64 / t / 1e12;
                }
            }
            let (mut free, mut total) = (0usize, 0usize);
            // SAFETY: freeing our allocations; valid out-pointers.
            unsafe {
                (drv.mem_free)(a);
                (drv.mem_free)(b);
                (drv.mem_get_info)(&mut free, &mut total);
            }
            Ok(super::GpuProfile {
                mem_bandwidth_gbps: 2.0 * bytes as f64 / copy? / 1e9,
                tflops_fp32: tflops,
                free_vram_mb: free as f64 / 1e6,
            })
        }

        fn run_fixed(&mut self, x: &Tensor2D, v: &QuantView) -> Result<Option<Vec<i64>>, String> {
            let (m, n, k) = (x.rows, v.rows, v.cols);
            let Some((w, row_bytes)) = self.weights(v) else {
                return Ok(None);
            };
            let acts = Q8Acts::quantize(x);
            let nb = k / 32;
            // one upload: [q: m·k i8 | d: m·nb f32 | ds: m·nb f32]
            let qb = (m * k).next_multiple_of(256);
            let mut host = Vec::with_capacity(qb + 8 * m * nb);
            host.extend(acts.q.iter().map(|&q| q as u8));
            host.resize(qb, 0);
            for f in acts.d.iter().chain(&acts.ds) {
                host.extend_from_slice(&f.to_le_bytes());
            }
            let xp = self.grow(0, host.len())?;
            let yp = self.grow(1, m * n * 8)?;
            let drv = &self.drv;
            // SAFETY: the device buffer holds host.len() bytes.
            unsafe {
                check(
                    drv,
                    (drv.htod)(xp, host.as_ptr() as *const c_void, host.len()),
                    "copy to GPU",
                )?;
            }
            let mut wp = w;
            let mut rb = row_bytes as i64;
            let mut nn = n as c_int;
            let mut kk = k as c_int;
            let mut fmt = v.fmt.ggml_id() as c_int;
            let mut ldy = n as c_int;
            let mut r0 = 0;
            while r0 < m {
                let mut mm = (m - r0).min(QMV_ROWS) as c_int;
                let mut xq = xp + (r0 * k) as u64;
                let mut xd = xp + (qb + r0 * nb * 4) as u64;
                let mut xds = xp + (qb + m * nb * 4 + r0 * nb * 4) as u64;
                let mut yr = yp + (r0 * n * 8) as u64;
                self.launch(
                    self.qmv_fixed,
                    (n as u32).div_ceil(8),
                    256,
                    &mut [
                        &mut wp as *mut _ as Ptr,
                        &mut rb as *mut _ as Ptr,
                        &mut nn as *mut _ as Ptr,
                        &mut kk as *mut _ as Ptr,
                        &mut fmt as *mut _ as Ptr,
                        &mut xq as *mut _ as Ptr,
                        &mut xd as *mut _ as Ptr,
                        &mut xds as *mut _ as Ptr,
                        &mut mm as *mut _ as Ptr,
                        &mut yr as *mut _ as Ptr,
                        &mut ldy as *mut _ as Ptr,
                    ],
                )?;
                r0 += QMV_ROWS;
            }
            let mut y = vec![0i64; m * n];
            // SAFETY: synchronous copy of m·n i64.
            unsafe {
                check(
                    drv,
                    (drv.dtoh)(y.as_mut_ptr() as *mut c_void, yp, m * n * 8),
                    "copy from GPU",
                )?;
            }
            Ok(Some(y))
        }

        fn run(&mut self, x: &Tensor2D, v: &QuantView, y: &mut Tensor2D) -> Result<bool, String> {
            let (m, n, k) = (x.rows, v.rows, v.cols);
            let Some((w, row_bytes)) = self.weights(v) else {
                return Ok(false);
            };
            let xp = self.grow(0, m * k * 4)?;
            let yp = self.grow(1, m * n * 4)?;
            let use_blas = m >= CUBLAS_MIN_ROWS && self.blas.is_some();
            let tile = (DEQUANT_TILE_BYTES / (k * 4)).clamp(1, n);
            let dp = if use_blas {
                self.grow(2, tile * k * 4)?
            } else {
                0
            };
            let drv = &self.drv;
            // SAFETY: device buffers hold m·k / m·n f32.
            unsafe {
                check(
                    drv,
                    (drv.htod)(xp, x.data.as_ptr() as *const c_void, m * k * 4),
                    "copy to GPU",
                )?;
            }
            let mut fmt = v.fmt.ggml_id() as c_int;
            let mut rb = row_bytes as i64;
            let mut kk = k as c_int;
            if use_blas {
                let mut j0 = 0;
                while j0 < n {
                    let nr = tile.min(n - j0);
                    let mut wp = w + (j0 * row_bytes) as u64;
                    let mut nn = nr as c_int;
                    let mut out = dp;
                    let total = (nr * k.div_ceil(32)) as u32;
                    self.launch(
                        self.dequant,
                        total.div_ceil(256),
                        256,
                        &mut [
                            &mut wp as *mut _ as Ptr,
                            &mut rb as *mut _ as Ptr,
                            &mut nn as *mut _ as Ptr,
                            &mut kk as *mut _ as Ptr,
                            &mut fmt as *mut _ as Ptr,
                            &mut out as *mut _ as Ptr,
                        ],
                    )?;
                    let b = self.blas.as_ref().unwrap();
                    let (one, zero) = (1.0f32, 0.0f32);
                    // column-major: Yᵀ[n, m] = W[n, k] · Xᵀ[k, m]
                    // SAFETY: device pointers sized above.
                    let r = unsafe {
                        (b.sgemm)(
                            b.handle,
                            1,
                            0,
                            nr as c_int,
                            m as c_int,
                            k as c_int,
                            &one,
                            dp,
                            k as c_int,
                            xp,
                            k as c_int,
                            &zero,
                            yp + (j0 * 4) as u64,
                            n as c_int,
                        )
                    };
                    if r != 0 {
                        return Err(format!("cublasSgemm failed ({r})"));
                    }
                    j0 += nr;
                }
            } else {
                let mut wp = w;
                let mut nn = n as c_int;
                let mut ldy = n as c_int;
                let mut r0 = 0;
                while r0 < m {
                    let mut mm = (m - r0).min(QMV_ROWS) as c_int;
                    let mut xr = xp + (r0 * k * 4) as u64;
                    let mut yr = yp + (r0 * n * 4) as u64;
                    let warps_per_block = 8u32;
                    self.launch(
                        self.qmv,
                        (n as u32).div_ceil(warps_per_block),
                        32 * warps_per_block,
                        &mut [
                            &mut wp as *mut _ as Ptr,
                            &mut rb as *mut _ as Ptr,
                            &mut nn as *mut _ as Ptr,
                            &mut kk as *mut _ as Ptr,
                            &mut fmt as *mut _ as Ptr,
                            &mut xr as *mut _ as Ptr,
                            &mut mm as *mut _ as Ptr,
                            &mut yr as *mut _ as Ptr,
                            &mut ldy as *mut _ as Ptr,
                        ],
                    )?;
                    r0 += QMV_ROWS;
                }
            }
            // SAFETY: synchronous copy back (orders after the kernels).
            unsafe {
                check(
                    drv,
                    (drv.dtoh)(y.data.as_mut_ptr() as *mut c_void, yp, m * n * 4),
                    "copy from GPU",
                )?;
            }
            Ok(true)
        }
    }

    fn with_state<R>(f: impl FnOnce(&mut State) -> R) -> Option<R> {
        let s = state()?;
        let mut g = s.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: binding our retained primary context to this thread.
        unsafe { (g.drv.ctx_set_current)(g.ctx) };
        Some(f(&mut g))
    }

    pub fn gemm(x: &Tensor2D, v: &QuantView, y: &mut Tensor2D) -> bool {
        if x.rows == 0 || v.rows == 0 {
            return false;
        }
        with_state(|s| match s.run(x, v, y) {
            Ok(done) => done,
            Err(e) => {
                if !s.reported {
                    eprintln!("[cuda] {e}; this product falls back to the CPU kernels");
                    s.reported = true;
                }
                false
            }
        })
        .unwrap_or(false)
    }

    pub fn gemm_fixed(x: &Tensor2D, v: &QuantView) -> Option<Vec<i64>> {
        if x.rows == 0 || v.rows == 0 {
            return None;
        }
        with_state(|s| match s.run_fixed(x, v) {
            Ok(y) => y,
            Err(e) => {
                if !s.reported {
                    eprintln!("[cuda] {e}; this product falls back to the CPU kernels");
                    s.reported = true;
                }
                None
            }
        })
        .flatten()
    }

    pub fn preload(v: &QuantView) -> bool {
        with_state(|s| s.weights(v).is_some()).unwrap_or(false)
    }

    pub fn measure() -> Option<super::GpuProfile> {
        with_state(|s| s.measure().ok()).flatten()
    }

    impl Drop for State {
        fn drop(&mut self) {
            for w in self.weights.values() {
                // SAFETY: our allocations.
                unsafe { (self.drv.mem_free)(w.ptr) };
                let _ = w.bytes;
            }
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod imp {
    use super::super::quant::QuantView;
    use super::super::Tensor2D;

    pub fn available() -> bool {
        false
    }
    pub fn init_error() -> Option<String> {
        Some("the CUDA backend is built on Linux only".into())
    }
    pub fn device_name() -> Option<String> {
        None
    }
    pub fn gemm(_: &Tensor2D, _: &QuantView, _: &mut Tensor2D) -> bool {
        false
    }
    pub fn gemm_fixed(_: &Tensor2D, _: &QuantView) -> Option<Vec<i64>> {
        None
    }
    pub fn preload(_: &QuantView) -> bool {
        false
    }
    pub fn measure() -> Option<super::GpuProfile> {
        None
    }
    pub fn resident_bytes() -> usize {
        0
    }
    pub fn cublas_loaded() -> bool {
        false
    }
}
