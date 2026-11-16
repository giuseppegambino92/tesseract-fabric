//! Single-row GEMV throughput per weight format, float and exact
//! (fixed-point) accumulation: `cargo run --release --example kquant_bench`.
use std::time::Instant;
use tesseract_fabric::tensor::quant::*;
use tesseract_fabric::tensor::{Rng, Tensor2D};
fn main() {
    let mut rng = Rng::new(1);
    let (n, k) = (2048usize, 2048usize);
    let w = Tensor2D::randn(n, k, 0.05, &mut rng);
    let x = Tensor2D::randn(1, k, 1.0, &mut rng);
    println!("format  float µs  GMAC/s  weights GB/s | exact µs  GMAC/s");
    for fmt in [
        WeightFormat::Q8_0,
        WeightFormat::Q4_0,
        WeightFormat::Q5_0,
        WeightFormat::Q4_K,
        WeightFormat::Q6_K,
        WeightFormat::F16,
    ] {
        let v = QuantView::from_f32(&w, fmt);
        let mut y = Tensor2D::zeros(1, n);
        let mut best = f64::MAX;
        let mut best_x = f64::MAX;
        for _ in 0..30 {
            let t = Instant::now();
            gemm_quant_with(&x, &v, &mut y, best_kernel());
            best = best.min(t.elapsed().as_secs_f64());
            let t = Instant::now();
            std::hint::black_box(gemm_fixed_with(&x, &v, best_kernel()));
            best_x = best_x.min(t.elapsed().as_secs_f64());
        }
        let g = |t: f64| (n * k) as f64 / t / 1e9;
        println!(
            "{:5}  {:8.1}  {:6.2}  {:12.1} | {:8.1}  {:6.2}",
            fmt.name(),
            best * 1e6,
            g(best),
            v.bytes() as f64 / best / 1e9,
            best_x * 1e6,
            g(best_x)
        );
    }
}
