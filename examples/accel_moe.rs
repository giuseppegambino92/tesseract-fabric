//! Expert-stripe GEMM (`Linear::forward_rows`, as in MoE layers): portable
//! integer / tiled kernels vs Accelerate (AMX), by batch size and format:
//! `cargo run --release --example accel_moe`.
use std::time::Instant;
use tesseract_fabric::tensor::accel::{self, Backend};
use tesseract_fabric::tensor::quant::{QuantView, WeightFormat};
use tesseract_fabric::tensor::{Linear, Rng, Tensor2D};
fn main() {
    let mut rng = Rng::new(7);
    let (n, k) = (704usize, 512usize); // one benchmark-model expert's gate/up
    let w = Tensor2D::randn(n, k, 0.05, &mut rng);
    for fmt in [WeightFormat::F32, WeightFormat::Q8_0, WeightFormat::Q4_K] {
        let lin = if fmt == WeightFormat::F32 {
            Linear::new(w.clone())
        } else {
            Linear::quantized(QuantView::from_f32(&w, fmt))
        };
        for m in [1usize, 2, 4, 8, 16, 64] {
            let x = Tensor2D::randn(m, k, 1.0, &mut rng);
            let mut y = vec![0.0f32; m * n];
            let mut t = [0.0f64; 2];
            for (i, b) in [Backend::Portable, Backend::Accelerate]
                .into_iter()
                .enumerate()
            {
                accel::set_backend(b);
                // force the accelerated quantized path at every batch size for the measurement
                let mut best = f64::MAX;
                for _ in 0..100 {
                    let s = Instant::now();
                    if b == Backend::Accelerate && fmt != WeightFormat::F32 {
                        let mut tt = Tensor2D::zeros(m, n);
                        accel::gemm_dequant_tiles(&x, lin.q.as_ref().unwrap(), &mut tt);
                    } else {
                        lin.forward_rows(&x, 0, n, &mut y);
                    }
                    best = best.min(s.elapsed().as_secs_f64());
                }
                t[i] = best * 1e6;
            }
            println!(
                "{:5} m {m:2}: portable {:7.1} µs  accelerate {:7.1} µs  ({:.2}×)",
                fmt.name(),
                t[0],
                t[1],
                t[0] / t[1]
            );
        }
    }
    accel::set_backend(Backend::Portable);
}
