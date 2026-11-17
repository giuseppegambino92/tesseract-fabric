//! Native acceleration dispatch.
//!
//! * **Accelerate (macOS)**: `cblas_sgemm` from Apple's `Accelerate.framework`
//!   through a thin `extern "C"` binding. On Apple Silicon Accelerate runs
//!   SGEMM on the AMX matrix coprocessor. Dense FP32 GEMMs go straight to it;
//!   quantized weights used with prefill-sized batches are dequantized in
//!   row tiles and multiplied there too (small decode batches keep the
//!   integer-dot kernels, which read 3.8–7.1× fewer weight bytes).
//! * **CUDA (Linux)**: NVIDIA GPUs through the driver, NVRTC and cuBLAS
//!   libraries opened with `dlopen` at run time ([`super::cuda`]). Every
//!   quantized / half-precision weight matrix lives in VRAM after its first
//!   use and each product runs there; exact-accumulation layers and dense
//!   f32 matrices stay on the CPU.
//! * **Portable** (default, every target): the in-tree register-tiled
//!   kernels plus NEON / AVX2 integer dots (`tensor::quant`).
//!
//! The backend is chosen once from
//! `TESSERACT_BACKEND=portable|accelerate|cuda|auto` (default `portable`, so
//! `tesseract-bench`'s engine comparisons keep strict kernel parity; `auto`
//! picks CUDA when a GPU is found) and can be overridden in-process with
//! [`set_backend`]. Accelerate and CUDA run outside the equal-lanes rule, so
//! they are never enabled implicitly inside the engine comparisons.

use super::quant::QuantView;
use super::Tensor2D;
use std::sync::atomic::{AtomicU8, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Portable = 1,
    Accelerate = 2,
    Cuda = 3,
}

impl Backend {
    pub fn name(self) -> &'static str {
        match self {
            Backend::Portable => "portable",
            Backend::Accelerate => "accelerate",
            Backend::Cuda => "cuda",
        }
    }
}

static BACKEND: AtomicU8 = AtomicU8::new(0);

/// Whether this build can call Accelerate.
pub const ACCELERATE_AVAILABLE: bool = cfg!(target_os = "macos");

/// Batches with at least this many rows use the dequantize + SGEMM path for
/// quantized weights under the Accelerate backend (whole-matrix GEMMs).
pub const ACCEL_MIN_ROWS: usize = 16;

/// The same threshold for MoE expert stripes (`Linear::forward_rows`),
/// measured with `examples/accel_crossover.rs` (see the README).
pub const ACCEL_QUANT_MIN_ROWS: usize = 16;

/// The active backend (`TESSERACT_BACKEND`, read once; default portable).
pub fn backend() -> Backend {
    match BACKEND.load(Ordering::Relaxed) {
        1 => Backend::Portable,
        2 => Backend::Accelerate,
        3 => Backend::Cuda,
        _ => {
            let b = match std::env::var("TESSERACT_BACKEND").as_deref() {
                Ok("accelerate") | Ok("amx") if ACCELERATE_AVAILABLE => Backend::Accelerate,
                Ok("cuda") | Ok("gpu") => {
                    if super::cuda::available() {
                        Backend::Cuda
                    } else {
                        eprintln!(
                            "[backend] TESSERACT_BACKEND=cuda but CUDA is unavailable ({}); using the portable kernels",
                            super::cuda::init_error().unwrap_or_default()
                        );
                        Backend::Portable
                    }
                }
                Ok("auto") if super::cuda::available() => Backend::Cuda,
                _ => Backend::Portable,
            };
            BACKEND.store(b as u8, Ordering::Relaxed);
            b
        }
    }
}

/// Override the backend for this process. Returns the backend actually set
/// (Accelerate / CUDA fall back to portable where they do not exist).
pub fn set_backend(b: Backend) -> Backend {
    let b = match b {
        Backend::Accelerate if !ACCELERATE_AVAILABLE => Backend::Portable,
        Backend::Cuda if !super::cuda::available() => Backend::Portable,
        b => b,
    };
    BACKEND.store(b as u8, Ordering::Relaxed);
    b
}

thread_local! {
    static FORCE_PORTABLE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Run `f` on this thread with the portable kernels whatever the process
/// backend is. Used where separate machines must derive bit-identical
/// results from the same seeds (codec calibration), which only the portable
/// kernels guarantee.
pub fn with_portable<R>(f: impl FnOnce() -> R) -> R {
    let prev = FORCE_PORTABLE.with(|c| c.replace(true));
    let r = f();
    FORCE_PORTABLE.with(|c| c.set(prev));
    r
}

fn forced_portable() -> bool {
    FORCE_PORTABLE.with(|c| c.get())
}

pub fn accelerate_active() -> bool {
    ACCELERATE_AVAILABLE && backend() == Backend::Accelerate && !forced_portable()
}

pub fn cuda_active() -> bool {
    super::cuda::CUDA_SUPPORTED && backend() == Backend::Cuda && !forced_portable()
}

/// One line for logs and reports: "cuda (NVIDIA GeForce RTX 5090 · …)".
pub fn describe() -> String {
    match backend() {
        Backend::Cuda => format!("cuda ({})", super::cuda::device_name().unwrap_or_default()),
        b => b.name().to_string(),
    }
}

#[cfg(target_os = "macos")]
mod ffi {
    pub const ROW_MAJOR: i32 = 101;
    pub const NO_TRANS: i32 = 111;
    pub const TRANS: i32 = 112;
    #[link(name = "Accelerate", kind = "framework")]
    extern "C" {
        #[allow(clippy::too_many_arguments)]
        pub fn cblas_sgemm(
            order: i32,
            trans_a: i32,
            trans_b: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );
    }
}

/// `y[m, n] = x[m, k] · w[n, k]ᵀ` with raw row-major slices and leading
/// dimension `ldy` for y (`w` may be a row range of a larger matrix).
/// Returns false when Accelerate is unavailable.
pub fn sgemm_xwt(
    x: &[f32],
    m: usize,
    k: usize,
    w: &[f32],
    n: usize,
    y: &mut [f32],
    ldy: usize,
) -> bool {
    #[cfg(target_os = "macos")]
    {
        assert!(x.len() >= m * k && w.len() >= n * k && y.len() >= (m.max(1) - 1) * ldy + n);
        // SAFETY: bounds asserted above; Accelerate reads x, w and writes y only.
        unsafe {
            ffi::cblas_sgemm(
                ffi::ROW_MAJOR,
                ffi::NO_TRANS,
                ffi::TRANS,
                m as i32,
                n as i32,
                k as i32,
                1.0,
                x.as_ptr(),
                k as i32,
                w.as_ptr(),
                k as i32,
                0.0,
                y.as_mut_ptr(),
                ldy as i32,
            );
        }
        true
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (x, m, k, w, n, y, ldy);
        false
    }
}

/// Weight matrices at least this large (`n · k` elements) go to Accelerate;
/// smaller GEMMs (attention's 32-token score / value blocks) stay on the
/// portable kernels, where Accelerate's per-call overhead would dominate.
pub const ACCEL_MIN_WEIGHT_ELEMS: usize = 1 << 16;

/// Dense GEMM through Accelerate (`gemm_xwt`'s accelerated path). Returns
/// false (caller falls back) for small matrices or without Accelerate.
pub fn gemm_dense(x: &Tensor2D, w: &Tensor2D, y: &mut Tensor2D) -> bool {
    if w.rows * w.cols < ACCEL_MIN_WEIGHT_ELEMS {
        return false;
    }
    sgemm_xwt(
        &x.data,
        x.rows,
        x.cols,
        &w.data,
        w.rows,
        &mut y.data,
        y.cols,
    )
}

/// Rows of a quantized weight matrix dequantized per SGEMM tile.
const DEQUANT_TILE_ROWS: usize = 256;

/// Quantized weights × large batch: dequantize `DEQUANT_TILE_ROWS` weight
/// rows at a time into a scratch tile and multiply it with Accelerate.
pub fn gemm_dequant_tiles(x: &Tensor2D, v: &QuantView, y: &mut Tensor2D) {
    let (m, k, n) = (x.rows, x.cols, v.rows);
    let mut tile = vec![0.0f32; DEQUANT_TILE_ROWS * k];
    let mut j0 = 0;
    while j0 < n {
        let nr = (n - j0).min(DEQUANT_TILE_ROWS);
        for r in 0..nr {
            v.decode_row_into(j0 + r, &mut tile[r * k..(r + 1) * k]);
        }
        let ok = sgemm_xwt(&x.data, m, k, &tile[..nr * k], nr, &mut y.data[j0..], n);
        assert!(ok, "Accelerate backend selected but unavailable");
        j0 += nr;
    }
}
