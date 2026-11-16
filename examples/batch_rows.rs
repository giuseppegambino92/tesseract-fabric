//! GEMM throughput vs batch rows per weight format (one lane): how much a
//! batched step amortizes over single-row decode.
//! `cargo run --release --example batch_rows`
use std::time::Instant;
use tesseract_fabric::tensor::quant::*;
use tesseract_fabric::tensor::{Rng, Tensor2D};
fn main() {
    let mut rng = Rng::new(1);
    let (n, k) = (2048usize, 2048usize);
    let w = Tensor2D::randn(n, k, 0.05, &mut rng);
    print!("format ");
    let ms = [1usize, 2, 4, 8, 16];
    for m in ms {
        print!(" {m:>2} rows (float/exact)");
    }
    println!();
    for fmt in [
        WeightFormat::Q8_0,
        WeightFormat::Q4_0,
        WeightFormat::Q5_0,
        WeightFormat::Q4_K,
        WeightFormat::Q6_K,
    ] {
        let v = QuantView::from_f32(&w, fmt);
        print!("{:6}", fmt.name());
        for m in ms {
            let x = Tensor2D::randn(m, k, 1.0, &mut rng);
            let mut y = Tensor2D::zeros(m, n);
            let mut best = f64::MAX;
            let mut bx = f64::MAX;
            for _ in 0..15 {
                let t = Instant::now();
                gemm_quant_with(&x, &v, &mut y, best_kernel());
                best = best.min(t.elapsed().as_secs_f64());
                let t = Instant::now();
                std::hint::black_box(gemm_fixed_with(&x, &v, best_kernel()));
                bx = bx.min(t.elapsed().as_secs_f64());
            }
            print!(
                " {:>7.1} /{:>6.1}",
                (m * n * k) as f64 / best / 1e9,
                (m * n * k) as f64 / bx / 1e9
            );
        }
        println!();
    }
}
