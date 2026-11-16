//! Run with `cargo run --release --target x86_64-apple-darwin --example avx2_check`
//! (under Rosetta 2 on Apple Silicon, or on any AVX2 machine).
// x86_64 only: run the AVX2 kernels directly (bypassing CPUID detection) and
// compare with the scalar reference — float mode within rounding, exact
// (fixed-point) mode bit for bit.
#[cfg(target_arch = "x86_64")]
fn main() {
    use tesseract_fabric::tensor::quant::*;
    use tesseract_fabric::tensor::{Rng, Tensor2D};
    let mut rng = Rng::new(3);
    let w = Tensor2D::randn(64, 512, 0.2, &mut rng);
    let x = Tensor2D::randn(5, 512, 1.0, &mut rng);
    let mut ok = true;
    for fmt in [
        WeightFormat::Q8_0,
        WeightFormat::Q4_0,
        WeightFormat::Q5_0,
        WeightFormat::Q4_K,
        WeightFormat::Q6_K,
    ] {
        let v = QuantView::from_f32(&w, fmt);
        let mut a = Tensor2D::zeros(5, 64);
        let mut b = Tensor2D::zeros(5, 64);
        gemm_quant_with(&x, &v, &mut a, DotKernel::Scalar);
        gemm_quant_with(&x, &v, &mut b, DotKernel::Avx2);
        let m = a
            .data
            .iter()
            .zip(&b.data)
            .map(|(p, q)| (p - q).abs() / (1.0 + p.abs()))
            .fold(0.0f32, f32::max);
        let exact =
            gemm_fixed_with(&x, &v, DotKernel::Scalar) == gemm_fixed_with(&x, &v, DotKernel::Avx2);
        ok &= m < 1e-4 && exact;
        println!(
            "{}: AVX2 vs scalar max rel diff {m:.2e}, exact mode identical: {exact}",
            fmt.name()
        );
    }
    assert!(ok);
}
#[cfg(not(target_arch = "x86_64"))]
fn main() {}
