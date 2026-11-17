//! CUDA backend correctness (runs where a CUDA GPU is present; elsewhere it
//! reports why the backend is unavailable and passes).
//!
//! Every weight format, decode-sized and prefill-sized batches (the `qmv`
//! kernel and the dequantize + cuBLAS path), odd shapes and zero-copy column
//! slices are checked against an f32 reference computed from the same
//! dequantized weights.

use std::sync::Arc;
use tesseract_fabric::tensor::quant::{active_kernel, gemm_fixed_with, QuantView, WeightFormat};
use tesseract_fabric::tensor::{cuda, f32_to_f16, Rng, Tensor2D};

/// Random blocks of a decode-only format, with small finite f16 scales at
/// the offsets the format keeps them (`d`, then `m` / `dmin`).
fn random_blocks(fmt: WeightFormat, n: usize, k: usize, rng: &mut Rng) -> QuantView {
    let row = fmt.row_bytes(k);
    let mut b: Vec<u8> = (0..n * row).map(|_| rng.next_u64() as u8).collect();
    let block = row / (k / fmt.block_len());
    for blk in b.chunks_exact_mut(block) {
        for off in [0, 2] {
            let v = f32_to_f16(0.01 + 0.02 * rng.uniform());
            blk[off..off + 2].copy_from_slice(&v.to_le_bytes());
        }
    }
    QuantView::new(Arc::new(b), 0, fmt, n, k)
}

fn reference(x: &Tensor2D, v: &QuantView) -> Vec<f32> {
    let w = v.dequantize();
    let mut y = vec![0.0f32; x.rows * w.rows];
    for r in 0..x.rows {
        for j in 0..w.rows {
            let s: f64 = x
                .row(r)
                .iter()
                .zip(w.row(j))
                .map(|(a, b)| *a as f64 * *b as f64)
                .sum();
            y[r * w.rows + j] = s as f32;
        }
    }
    y
}

fn check(x: &Tensor2D, v: &QuantView, what: &str) {
    let want = reference(x, v);
    let mut y = Tensor2D::zeros(x.rows, v.rows);
    assert!(cuda::gemm(x, v, &mut y), "{what}: GPU product refused");
    let scale = want.iter().fold(0.0f32, |a, b| a.max(b.abs())).max(1e-6);
    let err = want
        .iter()
        .zip(&y.data)
        .fold(0.0f32, |a, (p, q)| a.max((p - q).abs()));
    assert!(
        err / scale < 2e-5,
        "{what}: max error {err} (scale {scale})"
    );
}

#[test]
fn cuda_products_match_the_f32_reference_for_every_format() {
    if !cuda::available() {
        eprintln!(
            "skipped: CUDA unavailable ({})",
            cuda::init_error().unwrap_or_default()
        );
        return;
    }
    eprintln!(
        "GPU: {} · cuBLAS {}",
        cuda::device_name().unwrap(),
        cuda::cublas_loaded()
    );
    let mut rng = Rng::new(11);
    let formats = [
        WeightFormat::F32,
        WeightFormat::F16,
        WeightFormat::BF16,
        WeightFormat::Q8_0,
        WeightFormat::Q4_0,
        WeightFormat::Q5_0,
        WeightFormat::Q4_K,
        WeightFormat::Q6_K,
    ];
    for fmt in formats {
        let (n, k) = (131, 768);
        let w = Tensor2D::randn(n, k, 0.05, &mut rng);
        let v = QuantView::from_f32(&w, fmt);
        for m in [1, 3, 8, 9, 40] {
            let x = Tensor2D::randn(m, k, 1.0, &mut rng);
            check(&x, &v, &format!("{} m={m}", fmt.name()));
        }
        // a zero-copy column slice (row stride > row bytes) and a row slice
        let cols = v.slice_cols(256, 768);
        let x = Tensor2D::randn(5, 512, 1.0, &mut rng);
        check(&x, &cols, &format!("{} column slice", fmt.name()));
        let rows = v.slice_rows(7, 100);
        let x = Tensor2D::randn(2, k, 1.0, &mut rng);
        check(&x, &rows, &format!("{} row slice", fmt.name()));
    }
    for fmt in [WeightFormat::Q4_1, WeightFormat::Q5_1, WeightFormat::Q5_K] {
        let v = random_blocks(fmt, 70, 512, &mut rng);
        for m in [1, 8, 33] {
            let x = Tensor2D::randn(m, 512, 1.0, &mut rng);
            check(&x, &v, &format!("{} m={m}", fmt.name()));
        }
    }
    // element formats need not be a multiple of 32 (or of 4) wide
    for fmt in [WeightFormat::F32, WeightFormat::F16] {
        let w = Tensor2D::randn(19, 45, 0.1, &mut rng);
        let v = QuantView::from_f32(&w, fmt);
        for m in [1, 33] {
            let x = Tensor2D::randn(m, 45, 1.0, &mut rng);
            check(&x, &v, &format!("{} k=45 m={m}", fmt.name()));
        }
    }
    assert!(cuda::resident_bytes() > 0);
    // what a CUDA worker advertises in its discovery beacons
    let g = cuda::measure().expect("GPU measurement");
    eprintln!(
        "measured: {:.0} GB/s, {:.1} TFLOPS SGEMM, {:.0} MB free VRAM",
        g.mem_bandwidth_gbps, g.tflops_fp32, g.free_vram_mb
    );
    assert!(g.mem_bandwidth_gbps > 10.0 && g.free_vram_mb > 0.0);
}

#[test]
fn cuda_exact_products_are_bit_identical_to_the_cpu() {
    if !cuda::available() {
        eprintln!("skipped: CUDA unavailable");
        return;
    }
    let mut rng = Rng::new(5);
    let k = 1024;
    let mut views = Vec::new();
    for fmt in [
        WeightFormat::Q8_0,
        WeightFormat::Q4_0,
        WeightFormat::Q5_0,
        WeightFormat::Q4_K,
        WeightFormat::Q6_K,
    ] {
        let w = Tensor2D::randn(97, k, 0.05, &mut rng);
        views.push(QuantView::from_f32(&w, fmt));
    }
    for fmt in [WeightFormat::Q4_1, WeightFormat::Q5_1, WeightFormat::Q5_K] {
        views.push(random_blocks(fmt, 97, k, &mut rng));
    }
    for v in &views {
        for m in [1, 4, 8, 11] {
            let x = Tensor2D::randn(m, k, 1.0, &mut rng);
            let cpu = gemm_fixed_with(&x, v, active_kernel());
            let gpu = cuda::gemm_fixed(&x, v).expect("GPU exact product");
            assert_eq!(cpu, gpu, "{} m={m}", v.fmt.name());
        }
        // a zero-copy column slice, as a row-parallel split uses
        let half = v.slice_cols(512, 1024);
        let x = Tensor2D::randn(3, 512, 1.0, &mut rng);
        assert_eq!(
            gemm_fixed_with(&x, &half, active_kernel()),
            cuda::gemm_fixed(&x, &half).unwrap(),
            "{} column slice",
            v.fmt.name()
        );
    }
}
