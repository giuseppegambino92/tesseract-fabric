//! Fast Walsh–Hadamard Transform.
//!
//! `fwht_inplace` is an allocation-free O(d log d) butterfly normalised by
//! 1/√d, which makes the transform orthonormal *and* self-inverse
//! (H·H = I). `HadamardPlan` adds a fixed pseudo-random ±1 diagonal D
//! (randomised Hadamard, H·D) so adversarially aligned outliers are spread
//! too, and handles arbitrary dimensions via zero padding to a power of two
//! or via independent block-Hadamard tiles (64/128/256).

use crate::tensor::Rng;

/// In-place normalised FWHT. `x.len()` must be a power of two.
pub fn fwht_inplace(x: &mut [f32]) {
    let n = x.len();
    assert!(
        n.is_power_of_two(),
        "FWHT length must be a power of two, got {n}"
    );
    let mut h = 1;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..i + h {
                let a = x[j];
                let b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
            i += h * 2;
        }
        h *= 2;
    }
    let s = 1.0 / (n as f32).sqrt();
    for v in x.iter_mut() {
        *v *= s;
    }
}

#[derive(Clone, Debug)]
pub struct HadamardPlan {
    pub dim: usize,
    pub padded: usize,
    /// Tile size; equals `padded` for a full-length transform.
    pub block: usize,
    pub signs: Vec<f32>,
}

impl HadamardPlan {
    /// Full transform over the next power of two ≥ dim.
    pub fn full(dim: usize, seed: u64) -> Self {
        let padded = dim.next_power_of_two();
        Self::build(dim, padded, padded, seed)
    }

    /// Block-diagonal Hadamard with power-of-two tiles of `block` elements.
    pub fn blocked(dim: usize, block: usize, seed: u64) -> Self {
        assert!(block.is_power_of_two());
        let padded = dim.div_ceil(block) * block;
        Self::build(dim, padded, block, seed)
    }

    fn build(dim: usize, padded: usize, block: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed ^ 0xA5A5_5A5A_0F0F_F0F0);
        let signs = (0..padded)
            .map(|_| if rng.next_u64() & 1 == 0 { 1.0 } else { -1.0 })
            .collect();
        HadamardPlan {
            dim,
            padded,
            block,
            signs,
        }
    }

    pub fn log2_block(&self) -> u8 {
        self.block.trailing_zeros() as u8
    }

    /// out[padded] = H · D · pad(x)
    pub fn forward(&self, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!(x.len(), self.dim);
        debug_assert_eq!(out.len(), self.padded);
        for i in 0..self.dim {
            out[i] = x[i] * self.signs[i];
        }
        out[self.dim..].iter_mut().for_each(|v| *v = 0.0);
        for tile in out.chunks_mut(self.block) {
            fwht_inplace(tile);
        }
    }

    /// x[dim] = unpad(D · H · y)   (exact inverse of `forward`)
    pub fn inverse(&self, y: &mut [f32], x: &mut [f32]) {
        for tile in y.chunks_mut(self.block) {
            fwht_inplace(tile);
        }
        for i in 0..self.dim {
            x[i] = y[i] * self.signs[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn self_inverse() {
        let mut rng = Rng::new(3);
        let x: Vec<f32> = (0..256).map(|_| rng.normal()).collect();
        let mut y = x.clone();
        fwht_inplace(&mut y);
        fwht_inplace(&mut y);
        assert!(crate::tensor::max_abs_diff(&x, &y) < 1e-5);
    }
}
