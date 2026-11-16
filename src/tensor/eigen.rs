//! Dense symmetric eigendecomposition (f64): Householder reduction to
//! tridiagonal form followed by the implicit QL algorithm with Wilkinson-style
//! shifts (the classic EISPACK `tred2` / `tql2` pair). O(n³), deterministic.

/// Eigen-decompose the symmetric `n × n` matrix `a` (row-major). Returns
/// eigenvalues in descending order and the matching orthonormal eigenvectors
/// as the rows of an `n × n` row-major matrix.
pub fn symmetric_eigen(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(a.len(), n * n);
    // z holds the accumulated transformation (column j = eigenvector j)
    let mut z = a.to_vec();
    let mut d = vec![0.0f64; n];
    let mut e = vec![0.0f64; n];
    tred2(&mut z, n, &mut d, &mut e);
    tql2(&mut z, n, &mut d, &mut e);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| d[j].total_cmp(&d[i]));
    let vals = order.iter().map(|&i| d[i]).collect();
    let mut vecs = vec![0.0f64; n * n];
    for (r, &c) in order.iter().enumerate() {
        for k in 0..n {
            vecs[r * n + k] = z[k * n + c];
        }
    }
    (vals, vecs)
}

fn tred2(z: &mut [f64], n: usize, d: &mut [f64], e: &mut [f64]) {
    for j in 0..n {
        d[j] = z[(n - 1) * n + j];
    }
    for i in (1..n).rev() {
        let mut scale = 0.0;
        let mut h = 0.0;
        for k in 0..i {
            scale += d[k].abs();
        }
        if scale == 0.0 {
            e[i] = d[i - 1];
            for j in 0..i {
                d[j] = z[(i - 1) * n + j];
                z[i * n + j] = 0.0;
                z[j * n + i] = 0.0;
            }
        } else {
            for k in 0..i {
                d[k] /= scale;
                h += d[k] * d[k];
            }
            let mut f = d[i - 1];
            let mut g = h.sqrt();
            if f > 0.0 {
                g = -g;
            }
            e[i] = scale * g;
            h -= f * g;
            d[i - 1] = f - g;
            for item in e.iter_mut().take(i) {
                *item = 0.0;
            }
            for j in 0..i {
                f = d[j];
                z[j * n + i] = f;
                g = e[j] + z[j * n + j] * f;
                for k in j + 1..i {
                    g += z[k * n + j] * d[k];
                    e[k] += z[k * n + j] * f;
                }
                e[j] = g;
            }
            f = 0.0;
            for j in 0..i {
                e[j] /= h;
                f += e[j] * d[j];
            }
            let hh = f / (h + h);
            for j in 0..i {
                e[j] -= hh * d[j];
            }
            for j in 0..i {
                f = d[j];
                g = e[j];
                for k in j..i {
                    z[k * n + j] -= f * e[k] + g * d[k];
                }
                d[j] = z[(i - 1) * n + j];
                z[i * n + j] = 0.0;
            }
        }
        d[i] = h;
    }
    // accumulate transformations
    for i in 0..n.saturating_sub(1) {
        z[(n - 1) * n + i] = z[i * n + i];
        z[i * n + i] = 1.0;
        let h = d[i + 1];
        if h != 0.0 {
            for k in 0..=i {
                d[k] = z[k * n + i + 1] / h;
            }
            for j in 0..=i {
                let mut g = 0.0;
                for k in 0..=i {
                    g += z[k * n + i + 1] * z[k * n + j];
                }
                for k in 0..=i {
                    z[k * n + j] -= g * d[k];
                }
            }
        }
        for k in 0..=i {
            z[k * n + i + 1] = 0.0;
        }
    }
    for j in 0..n {
        d[j] = z[(n - 1) * n + j];
        z[(n - 1) * n + j] = 0.0;
    }
    z[(n - 1) * n + n - 1] = 1.0;
    e[0] = 0.0;
}

fn tql2(z: &mut [f64], n: usize, d: &mut [f64], e: &mut [f64]) {
    for i in 1..n {
        e[i - 1] = e[i];
    }
    e[n - 1] = 0.0;
    let mut f = 0.0f64;
    let mut tst1 = 0.0f64;
    let eps = f64::EPSILON;
    for l in 0..n {
        tst1 = tst1.max(d[l].abs() + e[l].abs());
        let mut m = l;
        while m < n {
            if e[m].abs() <= eps * tst1 {
                break;
            }
            m += 1;
        }
        if m > l {
            loop {
                let mut g = d[l];
                let mut p = (d[l + 1] - g) / (2.0 * e[l]);
                let mut r = p.hypot(1.0);
                if p < 0.0 {
                    r = -r;
                }
                d[l] = e[l] / (p + r);
                d[l + 1] = e[l] * (p + r);
                let dl1 = d[l + 1];
                let mut h = g - d[l];
                for item in d.iter_mut().take(n).skip(l + 2) {
                    *item -= h;
                }
                f += h;
                p = d[m];
                let mut c = 1.0;
                let mut c2 = c;
                let mut c3 = c;
                let el1 = e[l + 1];
                let mut s = 0.0;
                let mut s2 = 0.0;
                for i in (l..m).rev() {
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    g = c * e[i];
                    h = c * p;
                    r = p.hypot(e[i]);
                    e[i + 1] = s * r;
                    s = e[i] / r;
                    c = p / r;
                    p = c * d[i] - s * g;
                    d[i + 1] = h + s * (c * g + s * d[i]);
                    for k in 0..n {
                        h = z[k * n + i + 1];
                        z[k * n + i + 1] = s * z[k * n + i] + c * h;
                        z[k * n + i] = c * z[k * n + i] - s * h;
                    }
                }
                p = -s * s2 * c3 * el1 * e[l] / dl1;
                e[l] = s * p;
                d[l] = c * p;
                if e[l].abs() <= eps * tst1 {
                    break;
                }
            }
        }
        d[l] += f;
        e[l] = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Rng;

    #[test]
    fn eigen_reconstructs_random_symmetric_matrix() {
        let n = 48;
        let mut rng = Rng::new(5);
        let b: Vec<f64> = (0..n * n).map(|_| rng.normal() as f64).collect();
        // a = bᵀb (PSD, distinct eigenvalues with probability 1)
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = (0..n).map(|k| b[k * n + i] * b[k * n + j]).sum();
            }
        }
        let (vals, vecs) = symmetric_eigen(&a, n);
        assert!(vals.windows(2).all(|w| w[0] >= w[1]));
        for i in 0..n {
            for j in 0..n {
                let dot: f64 = (0..n).map(|k| vecs[i * n + k] * vecs[j * n + k]).sum();
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((dot - want).abs() < 1e-9, "orthonormality {i},{j}: {dot}");
            }
            // A v = λ v
            for r in 0..n {
                let av: f64 = (0..n).map(|k| a[r * n + k] * vecs[i * n + k]).sum();
                assert!((av - vals[i] * vecs[i * n + r]).abs() < 1e-7 * vals[0]);
            }
        }
    }
}
