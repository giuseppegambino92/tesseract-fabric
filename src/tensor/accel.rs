//! Native acceleration dispatch.
//!
//! * **Accelerate (macOS)**: `cblas_sgemm` from Apple's `Accelerate.framework`
//!   through a thin `extern "C"` binding. On Apple Silicon Accelerate runs
//!   SGEMM on the AMX matrix coprocessor. Dense FP32 GEMMs go straight to it;
//!   quantized weights used with prefill-sized batches are dequantized in
//!   row tiles and multiplied there too (small decode batches keep the
//!   integer-dot kernels, which read 3.8–7.1× fewer weight bytes).
//! * **Portable** (default, every target): the in-tree register-tiled
//!   kernels plus NEON / AVX2 integer dots (`tensor::quant`).
//!
//! The backend is chosen once from `TESSERACT_BACKEND=accelerate|portable`
//! (default `portable`, so `tesseract-bench`'s engine comparisons keep strict
//! kernel parity) and can be overridden in-process with [`set_backend`].
//! Accelerate uses its own internal threads, so it is never enabled
//! implicitly inside the equal-lanes engine comparisons.

use super::quant::QuantView;
use super::Tensor2D;
use std::sync::atomic::{AtomicU8, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Portable = 1,
    Accelerate = 2,
}

impl Backend {
    pub fn name(self) -> &'static str {
        match self {
            Backend::Portable => "portable",
            Backend::Accelerate => "accelerate",
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
        _ => {
            let b = match std::env::var("TESSERACT_BACKEND").as_deref() {
                Ok("accelerate") | Ok("amx") if ACCELERATE_AVAILABLE => Backend::Accelerate,
                _ => Backend::Portable,
            };
            BACKEND.store(b as u8, Ordering::Relaxed);
            b
        }
    }
}

/// Override the backend for this process. Returns the backend actually set
/// (Accelerate falls back to portable where it does not exist).
pub fn set_backend(b: Backend) -> Backend {
    let b = if b == Backend::Accelerate && !ACCELERATE_AVAILABLE {
        Backend::Portable
    } else {
        b
    };
    BACKEND.store(b as u8, Ordering::Relaxed);
    b
}

pub fn accelerate_active() -> bool {
    ACCELERATE_AVAILABLE && backend() == Backend::Accelerate
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
