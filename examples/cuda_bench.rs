//! Per-call latency of the CUDA backend against the CPU kernels for the
//! weight shapes of Llama-3.2-1B (decode: 1 row; verification: 8 rows;
//! prefill: 64 rows).
//!
//! TESSERACT_BACKEND=cuda cargo run --release --example cuda_bench
use std::time::Instant;
use tesseract_fabric::tensor::accel::{self, Backend};
use tesseract_fabric::tensor::quant::{gemm_quant, QuantView, WeightFormat};
use tesseract_fabric::tensor::{cuda, Rng, Tensor2D};

fn time(f: &mut dyn FnMut()) -> f64 {
    f();
    let mut best = f64::MAX;
    for _ in 0..20 {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64());
    }
    best * 1e6
}

fn main() {
    if !cuda::available() {
        eprintln!(
            "CUDA unavailable: {}",
            cuda::init_error().unwrap_or_default()
        );
        return;
    }
    println!("{}", cuda::device_name().unwrap());
    if let Some(g) = cuda::measure() {
        println!(
            "measured: {:.0} GB/s copy bandwidth · {:.1} TFLOPS SGEMM · {:.0} MB free VRAM",
            g.mem_bandwidth_gbps, g.tflops_fp32, g.free_vram_mb
        );
    }
    accel::set_backend(Backend::Cuda);
    let p = tesseract_fabric::cluster::discovery::LocalProfile::detect();
    println!(
        "advertised in discovery beacons: {} · {:.1} TFLOPS · {:.0} GB/s · {} MB",
        p.platform.name(),
        p.compute_tflops_fp16,
        p.mem_bandwidth_gbps,
        p.free_ram_mb
    );
    let mut rng = Rng::new(1);
    let shapes = [
        ("q/o proj 2048x2048 Q4_K", 2048, 2048, WeightFormat::Q4_K),
        ("k/v proj 512x2048 Q4_K", 512, 2048, WeightFormat::Q4_K),
        ("gate/up 8192x2048 Q4_K", 8192, 2048, WeightFormat::Q4_K),
        ("down 2048x8192 Q6_K", 2048, 8192, WeightFormat::Q6_K),
        ("lm_head 128256x2048 Q6_K", 128256, 2048, WeightFormat::Q6_K),
    ];
    println!(
        "{:<28} {:>5} {:>10} {:>10} {:>8}",
        "matrix", "rows", "CPU µs", "GPU µs", "GPU GB/s"
    );
    for (name, n, k, fmt) in shapes {
        let w = Tensor2D::randn(n, k, 0.02, &mut rng);
        let v = QuantView::from_f32(&w, fmt);
        cuda::preload(&v);
        for m in [1usize, 8, 64] {
            let x = Tensor2D::randn(m, k, 1.0, &mut rng);
            let mut y = Tensor2D::zeros(m, n);
            accel::set_backend(Backend::Portable);
            let cpu = time(&mut || gemm_quant(&x, &v, &mut y));
            accel::set_backend(Backend::Cuda);
            let gpu = time(&mut || gemm_quant(&x, &v, &mut y));
            println!(
                "{name:<28} {m:>5} {cpu:>10.1} {gpu:>10.1} {:>8.0}",
                v.bytes() as f64 / gpu / 1e3
            );
        }
    }
}
