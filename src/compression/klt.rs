//! HRQ-V5: calibrated Karhunen–Loève transform coding of the trunk residual.
//!
//! The residual r = h − e(token) crossing a stage boundary is far from white:
//! on the benchmark model the top 64 of 512 eigen-directions of its covariance
//! carry ≈ 98% of its variance, and it has a large mean. A randomised FWHT
//! (HRQ-V2/V2.1) spreads that energy evenly, which is optimal only for white
//! sources. For a Gaussian source with covariance eigenvalues λᵢ the
//! rate–distortion optimum is *reverse water-filling* in the eigenbasis:
//! every component with λᵢ > θ is coded with distortion θ, the rest are
//! dropped (reconstructed at the mean).
//!
//! Both ends derive the same calibration (mean μ, eigenvectors vᵢ, λᵢ) from
//! the model's own layers on a fixed calibration token set — 0 wire bytes,
//! the same practice as AWQ/GPTQ calibration. Per row:
//!
//!   yᵢ = vᵢ·(r − μ),  sᵢ = round(yᵢ / (Δ·m)),  i < K (active components)
//!
//! with one common step Δ chosen at design time so the expected coded size of
//! a row is ~97% of the hard cap (components with σᵢ/Δ < 0.12 carry almost no
//! information and are dropped), and a per-row multiplier m = 1.05^j (signed
//! byte): each row is coded with the finest step that still fits the cap. Symbols are range-coded with one
//! static model per component: the exact quantised-Gaussian law of sᵢ at the
//! calibrated σᵢ/Δ (large symbols split into a modelled high part and raw low
//! bits), built identically on both ends. The decoder rebuilds
//! r̂ = μ + Σ sᵢΔm·vᵢ; the sender returns exactly that reconstruction.

use crate::tensor::eigen::symmetric_eigen;
use crate::tensor::{dot, Tensor2D};
use crate::wire::protocol::{ByteReader, ByteWriter};

/// Mean, eigenbasis (rows, descending variance) and eigenvalues of the
/// boundary residual, estimated from calibration rows.
#[derive(Clone, Debug)]
pub struct KltCalibration {
    pub mean: Vec<f32>,
    pub vecs: Tensor2D,
    pub var: Vec<f64>,
    pub samples: usize,
    /// Conditional coding: linear predictor of the residual from receiver
    /// side information f = [e(token); e(previous token); 1], stored as
    /// d × (2d + 1) so that prediction = f · Wᵀ. The KLT above is then the
    /// KLT of the *prediction* residual.
    pub pred: Option<Tensor2D>,
    /// Held-out fraction of residual variance explained by the predictor.
    pub explained: f64,
}

/// Feature rows f = [e(token); e(previous token); 1].
pub fn side_features(cur: &Tensor2D, prev: &Tensor2D) -> Tensor2D {
    let (n, d) = (cur.rows, cur.cols);
    let mut f = Tensor2D::zeros(n, 2 * d + 1);
    for i in 0..n {
        let r = f.row_mut(i);
        r[..d].copy_from_slice(cur.row(i));
        r[d..2 * d].copy_from_slice(prev.row(i));
        r[2 * d] = 1.0;
    }
    f
}

/// Solve (FᵀF + λI) W = FᵀR for W (p × d) by Cholesky (f64).
fn ridge(ftf: &[f64], ftr: &[f64], p: usize, d: usize, lam: f64) -> Vec<f64> {
    let mut l = vec![0.0f64; p * p];
    for i in 0..p {
        for j in 0..=i {
            let mut s = ftf[i * p + j] + if i == j { lam } else { 0.0 };
            let (li, lj) = (&l[i * p..i * p + j], &l[j * p..j * p + j]);
            s -= li.iter().zip(lj).map(|(a, b)| a * b).sum::<f64>();
            l[i * p + j] = if i == j {
                s.max(1e-12).sqrt()
            } else {
                s / l[j * p + j]
            };
        }
    }
    let mut w = ftr.to_vec();
    // forward: L y = b (all d right-hand sides at once, row-major p × d)
    for i in 0..p {
        for j in 0..i {
            let lij = l[i * p + j];
            if lij != 0.0 {
                for k in 0..d {
                    w[i * d + k] -= lij * w[j * d + k];
                }
            }
        }
        let inv = 1.0 / l[i * p + i];
        for k in 0..d {
            w[i * d + k] *= inv;
        }
    }
    // backward: Lᵀ x = y
    for i in (0..p).rev() {
        for j in i + 1..p {
            let lji = l[j * p + i];
            if lji != 0.0 {
                for k in 0..d {
                    w[i * d + k] -= lji * w[j * d + k];
                }
            }
        }
        let inv = 1.0 / l[i * p + i];
        for k in 0..d {
            w[i * d + k] *= inv;
        }
    }
    w
}

fn transpose(x: &Tensor2D) -> Tensor2D {
    let mut t = Tensor2D::zeros(x.cols, x.rows);
    for i in 0..x.rows {
        for (j, v) in x.row(i).iter().enumerate() {
            t.data[j * x.rows + i] = *v;
        }
    }
    t
}

impl KltCalibration {
    /// Conditional calibration: ridge-regress the residual rows on the side
    /// features (λ chosen on a held-out fifth of the rows, then refit on all
    /// of them), and take the KLT of what the predictor leaves.
    pub fn conditional(rows: &Tensor2D, cur: &Tensor2D, prev: &Tensor2D) -> Self {
        use crate::tensor::gemm_xwt;
        let (n, d) = (rows.rows, rows.cols);
        let f = side_features(cur, prev);
        let p = f.cols;
        let gram = |sel: &dyn Fn(usize) -> bool| -> (Vec<f64>, Vec<f64>) {
            let idx: Vec<usize> = (0..n).filter(|&i| sel(i)).collect();
            let mut fs = Tensor2D::zeros(idx.len(), p);
            let mut rs = Tensor2D::zeros(idx.len(), d);
            for (k, &i) in idx.iter().enumerate() {
                fs.row_mut(k).copy_from_slice(f.row(i));
                rs.row_mut(k).copy_from_slice(rows.row(i));
            }
            let (ft, rt) = (transpose(&fs), transpose(&rs));
            let mut ftf = Tensor2D::zeros(p, p);
            gemm_xwt(&ft, &ft, &mut ftf);
            let mut ftr = Tensor2D::zeros(p, d);
            gemm_xwt(&ft, &rt, &mut ftr);
            (
                ftf.data.iter().map(|v| *v as f64).collect(),
                ftr.data.iter().map(|v| *v as f64).collect(),
            )
        };
        let predict = |w: &[f64], fi: &[f32], out: &mut [f64]| {
            out.iter_mut().for_each(|o| *o = 0.0);
            for (a, &fa) in fi.iter().enumerate() {
                if fa != 0.0 {
                    let wr = &w[a * d..(a + 1) * d];
                    for (o, wv) in out.iter_mut().zip(wr) {
                        *o += fa as f64 * wv;
                    }
                }
            }
        };
        let (ftf_fit, ftr_fit) = gram(&|i| i % 5 != 0);
        let scale = (0..p).map(|i| ftf_fit[i * p + i]).sum::<f64>() / p as f64;
        let mut best = (f64::MAX, 0.0, 0.0);
        let mut tot = 0.0;
        for c in [0.003f64, 0.01, 0.03, 0.1, 0.3] {
            let lam = c * scale;
            let w = ridge(&ftf_fit, &ftr_fit, p, d, lam);
            let (mut sse, mut ss) = (0.0f64, 0.0f64);
            let mut pr = vec![0.0f64; d];
            for i in (0..n).step_by(5) {
                predict(&w, f.row(i), &mut pr);
                for (j, &rv) in rows.row(i).iter().enumerate() {
                    sse += (rv as f64 - pr[j]).powi(2);
                    ss += (rv as f64).powi(2);
                }
            }
            if sse < best.0 {
                best = (sse, lam, 1.0 - sse / ss.max(1e-30));
            }
            tot = ss;
        }
        let _ = tot;
        let (ftf, ftr) = gram(&|_| true);
        let w = ridge(&ftf, &ftr, p, d, best.1);
        let mut z = Tensor2D::zeros(n, d);
        let mut pr = vec![0.0f64; d];
        for i in 0..n {
            predict(&w, f.row(i), &mut pr);
            for (j, zv) in z.row_mut(i).iter_mut().enumerate() {
                *zv = rows.row(i)[j] - pr[j] as f32;
            }
        }
        let mut c = KltCalibration::from_rows(&z);
        // store Wᵀ (d × p) for batched prediction f · Wᵀ
        let mut wt = Tensor2D::zeros(d, p);
        for a in 0..p {
            for k in 0..d {
                wt.data[k * p + a] = w[a * d + k] as f32;
            }
        }
        c.pred = Some(wt);
        c.explained = best.2;
        c
    }

    /// Estimate from residual rows (n × d).
    pub fn from_rows(rows: &Tensor2D) -> Self {
        let (n, d) = (rows.rows, rows.cols);
        let mut mean = vec![0.0f64; d];
        for i in 0..n {
            for (m, x) in mean.iter_mut().zip(rows.row(i)) {
                *m += *x as f64;
            }
        }
        for m in mean.iter_mut() {
            *m /= n.max(1) as f64;
        }
        let mut cov = vec![0.0f64; d * d];
        let mut c = vec![0.0f64; d];
        for i in 0..n {
            for (j, x) in rows.row(i).iter().enumerate() {
                c[j] = *x as f64 - mean[j];
            }
            for a in 0..d {
                let ca = c[a];
                let row = &mut cov[a * d..(a + 1) * d];
                for b in a..d {
                    row[b] += ca * c[b];
                }
            }
        }
        let inv = 1.0 / n.max(1) as f64;
        for a in 0..d {
            for b in a..d {
                let v = cov[a * d + b] * inv;
                cov[a * d + b] = v;
                cov[b * d + a] = v;
            }
        }
        let (vals, vecs) = symmetric_eigen(&cov, d);
        KltCalibration {
            mean: mean.iter().map(|m| *m as f32).collect(),
            vecs: Tensor2D::from_vec(d, d, vecs.iter().map(|v| *v as f32).collect()),
            var: vals.iter().map(|v| v.max(0.0)).collect(),
            samples: n,
            pred: None,
            explained: 0.0,
        }
    }

    /// Fraction of the residual variance in the top `k` directions.
    pub fn energy(&self, k: usize) -> f64 {
        let t: f64 = self.var.iter().sum();
        self.var[..k.min(self.var.len())].iter().sum::<f64>() / t.max(1e-30)
    }
}

const HEADER_BYTES: usize = 3;
/// Per-row step multiplier granularity (the row header carries j: m = 1.05^j).
const ROW_STEP: f32 = 1.05;
const TOT_BITS: u32 = 16;
const TOTAL: u32 = 1 << TOT_BITS;
/// Components whose σ/Δ is below this carry < 0.01 bit and are not coded.
const MIN_RHO: f64 = 0.12;
/// Per-component models cover |s_hi| ≤ 6ρ′ + 2 with ρ′ = ρ/2^k ≤ 24.
const MAX_RHO_HI: f64 = 24.0;

fn phi(x: f64) -> f64 {
    // Φ via erfc with a rational Chebyshev approximation (|ε| < 1.2e-7)
    let z = x.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.5 * z);
    let r = t
        * (-z * z - 1.265_512_23
            + t * (1.000_023_68
                + t * (0.374_091_96
                    + t * (0.096_784_18
                        + t * (-0.186_288_06
                            + t * (0.278_868_07
                                + t * (-1.135_203_98
                                    + t * (1.488_515_87
                                        + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
            .exp();
    if x >= 0.0 {
        1.0 - 0.5 * r
    } else {
        0.5 * r
    }
}

fn pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Static model of one component: the quantised symbol s = round(y/Δ) is
/// split as s = s_hi·2^k + lo; `lo` (k bits) is sent uniformly and s_hi with
/// integer frequencies of its exact quantised-Gaussian law. Index 2·smax+1 is
/// the escape for |s_hi| > smax (followed by 32 raw bits).
#[derive(Clone, Debug)]
struct SymModel {
    k: u32,
    smax: i32,
    cum: Vec<u32>,
}

impl SymModel {
    fn new(rho: f64) -> (Self, f64) {
        let k = if rho > MAX_RHO_HI {
            (rho / MAX_RHO_HI).log2().ceil() as u32
        } else {
            0
        };
        let w = (1u64 << k) as f64;
        let rho_hi = rho / w;
        let smax = (6.0 * rho_hi + 2.0).ceil() as i32;
        let n = (2 * smax + 2) as usize;
        let mut p = Vec::with_capacity(n);
        for sh in -smax..=smax {
            let lo = if sh == -smax {
                f64::NEG_INFINITY
            } else {
                (sh as f64 * w - 0.5) / rho
            };
            let hi = if sh == smax {
                f64::INFINITY
            } else {
                ((sh + 1) as f64 * w - 0.5) / rho
            };
            let c = |x: f64| {
                if x == f64::INFINITY {
                    1.0
                } else if x == f64::NEG_INFINITY {
                    0.0
                } else {
                    phi(x)
                }
            };
            p.push((c(hi) - c(lo)).max(0.0));
        }
        p.push(1e-7); // escape (never needed by the clamped tails above)
        let spare = TOTAL as f64 - n as f64;
        let mut f: Vec<u32> = p.iter().map(|&q| 1 + (q * spare).floor() as u32).collect();
        let sum: u32 = f.iter().sum();
        let imax = (0..n).max_by(|&a, &b| p[a].total_cmp(&p[b])).unwrap();
        f[imax] += TOTAL - sum;
        let mut cum = Vec::with_capacity(n + 1);
        let mut acc = 0u32;
        cum.push(0);
        for &x in &f {
            acc += x;
            cum.push(acc);
        }
        // expected code length of this model under the true law
        let bits: f64 = p
            .iter()
            .zip(&f)
            .map(|(&q, &fr)| {
                if q > 0.0 {
                    -q * (fr as f64 / TOTAL as f64).log2()
                } else {
                    0.0
                }
            })
            .sum::<f64>()
            + k as f64;
        (SymModel { k, smax, cum }, bits)
    }
}

/// Exact MSE (in units of σ²) of rounding a N(0, σ²) value to the lattice ΔZ,
/// with ρ = σ/Δ.
fn quant_mse(rho: f64) -> f64 {
    if rho > 6.0 {
        return 1.0 / (12.0 * rho * rho);
    }
    let smax = (8.0 * rho + 2.0).ceil() as i32;
    let mut m = 0.0;
    for s in -smax..=smax {
        let a = (s as f64 - 0.5) / rho;
        let b = (s as f64 + 0.5) / rho;
        let c = s as f64 / rho;
        let m0 = phi(b) - phi(a);
        let m1 = pdf(a) - pdf(b);
        let m2 = m0 + a * pdf(a) - b * pdf(b);
        m += m2 - 2.0 * c * m1 + c * c * m0;
    }
    m.max(0.0)
}

/// LZMA-style multi-symbol range encoder (32-bit range, carry via cache).
struct RangeEnc {
    low: u64,
    range: u32,
    cache: u8,
    cache_size: u64,
    out: Vec<u8>,
}

impl RangeEnc {
    fn new(cap: usize) -> Self {
        RangeEnc {
            low: 0,
            range: u32::MAX,
            cache: 0,
            cache_size: 1,
            out: Vec::with_capacity(cap),
        }
    }
    fn shift_low(&mut self) {
        if (self.low as u32) < 0xFF00_0000 || (self.low >> 32) != 0 {
            let carry = (self.low >> 32) as u8;
            let mut temp = self.cache;
            loop {
                self.out.push(temp.wrapping_add(carry));
                temp = 0xFF;
                self.cache_size -= 1;
                if self.cache_size == 0 {
                    break;
                }
            }
            self.cache = ((self.low >> 24) & 0xFF) as u8;
        }
        self.cache_size += 1;
        self.low = (self.low & 0x00FF_FFFF) << 8;
    }
    fn encode(&mut self, cum: u32, freq: u32) {
        let r = self.range >> TOT_BITS;
        self.low += r as u64 * cum as u64;
        self.range = r * freq;
        while self.range < (1 << 24) {
            self.range <<= 8;
            self.shift_low();
        }
    }
    fn raw16(&mut self, v: u32) {
        self.encode(v & 0xFFFF, 1);
    }
    fn raw_bits(&mut self, v: u32, k: u32) {
        // k ≤ 16 bits as one uniform symbol of 2^k (scaled into the total)
        if k > 0 {
            let sh = TOT_BITS - k;
            self.encode(v << sh, 1 << sh);
        }
    }
    /// Flush; the always-zero first byte is dropped (the decoder re-inserts it).
    fn finish(mut self) -> Vec<u8> {
        for _ in 0..5 {
            self.shift_low();
        }
        self.out.remove(0);
        self.out
    }
}

struct RangeDec<'a> {
    code: u32,
    range: u32,
    b: &'a [u8],
    pos: usize,
}

impl<'a> RangeDec<'a> {
    fn new(b: &'a [u8]) -> Self {
        let mut d = RangeDec {
            code: 0,
            range: u32::MAX,
            b,
            pos: 0,
        };
        for _ in 0..4 {
            d.code = (d.code << 8) | d.next() as u32;
        }
        d
    }
    fn next(&mut self) -> u8 {
        let v = self.b.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        v
    }
    fn target(&mut self) -> (u32, u32) {
        let r = self.range >> TOT_BITS;
        ((self.code / r).min(TOTAL - 1), r)
    }
    fn consume(&mut self, r: u32, cum: u32, freq: u32) {
        self.code -= r * cum;
        self.range = r * freq;
        while self.range < (1 << 24) {
            self.code = (self.code << 8) | self.next() as u32;
            self.range <<= 8;
        }
    }
    fn raw16(&mut self) -> u32 {
        let (v, r) = self.target();
        self.consume(r, v, 1);
        v
    }
    fn raw_bits(&mut self, k: u32) -> u32 {
        if k == 0 {
            return 0;
        }
        let sh = TOT_BITS - k;
        let (v, r) = self.target();
        let x = v >> sh;
        self.consume(r, x << sh, 1 << sh);
        x
    }
}

pub struct KltResidualCodec {
    cal: std::sync::Arc<KltCalibration>,
    pub active: usize,
    pub step: f32,
    models: Vec<SymModel>,
    pub max_row_bytes: usize,
    /// Expected relative error √(D / Σλ) of the design at m = 1.
    pub design_rel_err: f64,
    pub rows_widened: u64,
}

impl KltResidualCodec {
    /// Design for a per-row hard cap of `budget_frac` × the FP16 row.
    pub fn new(cal: std::sync::Arc<KltCalibration>, budget_frac: f64) -> Self {
        let d = cal.mean.len();
        let max_row_bytes = ((d * 2) as f64 * budget_frac) as usize;
        // header + range-coder flush overhead (4 bytes)
        let target = ((max_row_bytes - HEADER_BYTES - 4) * 8) as f64 * 0.97;
        let total: f64 = cal.var.iter().sum();
        let design = |delta: f64| -> (usize, f64, f64) {
            let mut bits = 0.0;
            let mut dist = 0.0;
            let mut active = 0;
            for &l in &cal.var {
                let rho = l.sqrt() / delta;
                if rho < MIN_RHO {
                    dist += l;
                    continue;
                }
                active += 1;
                bits += SymModel::new(rho).1;
                dist += l * quant_mse(rho);
            }
            (active, bits, dist)
        };
        let (mut lo, mut hi) = (1e-6f64, 1e3f64);
        for _ in 0..40 {
            let mid = (lo * hi).sqrt();
            if design(mid).1 > target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let (active, _, dist) = design(hi);
        let models = cal.var[..active]
            .iter()
            .map(|&l| SymModel::new(l.sqrt() / hi).0)
            .collect();
        KltResidualCodec {
            cal,
            active,
            step: hi as f32,
            models,
            max_row_bytes,
            design_rel_err: (dist / total.max(1e-30)).sqrt(),
            rows_widened: 0,
        }
    }

    fn encode_row(&self, y: &[f32], mult: f32) -> (Vec<u8>, Vec<i32>) {
        let inv = 1.0 / (self.step * mult);
        let mut rc = RangeEnc::new(self.max_row_bytes);
        let mut syms = Vec::with_capacity(self.active);
        for (m, &yi) in self.models.iter().zip(y) {
            let s = (yi * inv).round() as i32;
            syms.push(s);
            let hi = s >> m.k;
            let lo = (s - (hi << m.k)) as u32;
            if hi.abs() <= m.smax {
                let idx = (hi + m.smax) as usize;
                rc.encode(m.cum[idx], m.cum[idx + 1] - m.cum[idx]);
                rc.raw_bits(lo, m.k);
            } else {
                let idx = (2 * m.smax + 1) as usize;
                rc.encode(m.cum[idx], m.cum[idx + 1] - m.cum[idx]);
                let u = s as u32;
                rc.raw16(u & 0xFFFF);
                rc.raw16(u >> 16);
            }
        }
        (rc.finish(), syms)
    }

    /// Receiver-side prediction of every row: e(token) + W·f (or e(token)).
    fn prediction(&self, base: &Tensor2D, prev: Option<&Tensor2D>) -> Tensor2D {
        match (&self.cal.pred, prev) {
            (Some(wt), Some(pv)) => {
                let f = side_features(base, pv);
                let mut y = Tensor2D::zeros(base.rows, base.cols);
                crate::tensor::gemm_xwt(&f, wt, &mut y);
                for (a, b) in y.data.iter_mut().zip(&base.data) {
                    *a += b;
                }
                y
            }
            (Some(_), None) => panic!("conditional KLT codec needs previous-token embeddings"),
            (None, _) => base.clone(),
        }
    }

    /// Does this codec need the previous token of every row?
    pub fn conditional(&self) -> bool {
        self.cal.pred.is_some()
    }

    /// Encode rows of `x` given the receiver's side information (`base` =
    /// e(token) per row, `prev` = e(previous token) per row for conditional
    /// calibrations); returns payload + exact receiver reconstruction.
    pub fn encode(
        &mut self,
        x: &Tensor2D,
        base: &Tensor2D,
        prev: Option<&Tensor2D>,
    ) -> (Vec<u8>, Tensor2D) {
        let (rows, d) = (x.rows, x.cols);
        let base = &self.prediction(base, prev);
        let mut w = ByteWriter::with_capacity(8 + rows * self.max_row_bytes);
        w.u32(rows as u32);
        w.u32(d as u32);
        let mut recon = Tensor2D::zeros(rows, d);
        let mut c = vec![0.0f32; d];
        let mut y = vec![0.0f32; self.active];
        let cap = self.max_row_bytes - HEADER_BYTES;
        for i in 0..rows {
            for j in 0..d {
                c[j] = x.row(i)[j] - base.row(i)[j] - self.cal.mean[j];
            }
            for (k, yk) in y.iter_mut().enumerate() {
                *yk = dot(self.cal.vecs.row(k), &c);
            }
            // two-sided rate control: the finest step 1.05^j·Δ that fits the cap
            let mut j = 0i32;
            let (mut bytes, mut syms) = self.encode_row(&y, 1.0);
            if bytes.len() > cap {
                while bytes.len() > cap && j < 127 {
                    j += 1;
                    (bytes, syms) = self.encode_row(&y, ROW_STEP.powi(j));
                }
                self.rows_widened += 1;
            } else {
                while j > -127 {
                    let (b2, s2) = self.encode_row(&y, ROW_STEP.powi(j - 1));
                    if b2.len() > cap {
                        break;
                    }
                    j -= 1;
                    (bytes, syms) = (b2, s2);
                }
            }
            w.u8(j as i8 as u8);
            w.u16(bytes.len() as u16);
            w.bytes(&bytes);
            let r = recon.row_mut(i);
            self.reconstruct(&syms, ROW_STEP.powi(j), r);
            for (rv, bv) in r.iter_mut().zip(base.row(i)) {
                *rv += bv;
            }
        }
        (w.into_inner(), recon)
    }

    fn reconstruct(&self, syms: &[i32], mult: f32, out: &mut [f32]) {
        out.copy_from_slice(&self.cal.mean);
        let st = self.step * mult;
        for (k, &s) in syms.iter().enumerate() {
            if s == 0 {
                continue;
            }
            let a = s as f32 * st;
            for (o, v) in out.iter_mut().zip(self.cal.vecs.row(k)) {
                *o += a * v;
            }
        }
    }

    pub fn decode(&mut self, bytes: &[u8], base: &Tensor2D, prev: Option<&Tensor2D>) -> Tensor2D {
        let base = &self.prediction(base, prev);
        let mut rd = ByteReader::new(bytes);
        let rows = rd.u32() as usize;
        let d = rd.u32() as usize;
        let mut out = Tensor2D::zeros(rows, d);
        let mut syms = vec![0i32; self.active];
        for i in 0..rows {
            let j = rd.u8() as i8 as i32;
            let nb = rd.u16() as usize;
            let mut buf = Vec::with_capacity(nb + 1);
            buf.push(0u8);
            buf.extend_from_slice(rd.bytes(nb));
            let mut dec = RangeDec::new(&buf[1..]);
            // the encoder's dropped first byte is always 0: code starts from
            // the next four bytes, exactly as LZMA's decoder reads 5 with the
            // leading zero ignored
            for (m, s) in self.models.iter().zip(syms.iter_mut()) {
                let (v, r) = dec.target();
                // binary search the cumulative table
                let (mut a, mut b) = (0usize, m.cum.len() - 1);
                while b - a > 1 {
                    let mid = (a + b) / 2;
                    if m.cum[mid] <= v {
                        a = mid;
                    } else {
                        b = mid;
                    }
                }
                dec.consume(r, m.cum[a], m.cum[a + 1] - m.cum[a]);
                if a as i32 == 2 * m.smax + 1 {
                    let lo = dec.raw16();
                    let hi = dec.raw16();
                    *s = (lo | (hi << 16)) as i32;
                } else {
                    let hi = a as i32 - m.smax;
                    let lo = dec.raw_bits(m.k) as i32;
                    *s = (hi << m.k) + lo;
                }
            }
            let r = out.row_mut(i);
            self.reconstruct(&syms, ROW_STEP.powi(j), r);
            for (rv, bv) in r.iter_mut().zip(base.row(i)) {
                *rv += bv;
            }
        }
        out
    }

    pub fn row_bytes_cap(&self) -> usize {
        self.max_row_bytes
    }
}

/// Number of calibration sequences × tokens per sequence.
pub const CALIBRATION_SEQS: usize = 128;
pub const CALIBRATION_LEN: usize = 144;
/// Calibration sequences for file-backed (real) models.
pub const CALIBRATION_SEQS_FILE: usize = 32;
/// Every n-th calibration sequence is a T = 1 sample, the rest greedy.
pub const CALIBRATION_SAMPLED_EVERY: usize = 2;

/// Calibration text: the model's own continuations (half greedy, half T = 1
/// samples) of uniformly random 16-token prompts, from a seed independent of every benchmark and test
/// prompt (self-generated calibration data, as in data-free post-training
/// quantisation). Cached per model per process.
pub fn calibration_tokens(cfg: &crate::model::ModelConfig) -> std::sync::Arc<Vec<Vec<u32>>> {
    use crate::kv_fabric::{BatchMeta, LocalKvCache};
    use crate::model::{probs_from_logits, sample, HeadWeights, NoObserver, ShardModel};
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};
    type TokenCache = Mutex<HashMap<(u64, usize, usize), Arc<Vec<Vec<u32>>>>>;
    static CACHE: OnceLock<TokenCache> = OnceLock::new();
    let key = (cfg.seed ^ cfg.source.key(), cfg.d_model, cfg.n_layers);
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut g = cache.lock().unwrap();
    if let Some(c) = g.get(&key) {
        return c.clone();
    }
    let head = HeadWeights::generate(cfg);
    let model = ShardModel::generate(cfg, 0..cfg.n_layers, None);
    let prompt_len = 16.min(CALIBRATION_LEN);
    // even sequences: greedy continuations; odd: T = 1 samples
    let seqs: Vec<Vec<u32>> = std::thread::scope(|sc| {
        // real checkpoints (deep, wide) calibrate on fewer sequences; the
        // synthetic benchmark models keep the full set
        let n_seqs = if cfg.source.is_file() {
            CALIBRATION_SEQS_FILE
        } else {
            CALIBRATION_SEQS
        };
        let hs: Vec<_> = (0..n_seqs)
            .map(|s| {
                let (head, model) = (&head, &model);
                sc.spawn(move || {
                    crate::tensor::accel::with_portable(|| {
                        crate::tensor::prefer_performance_cores();
                        let mut rng =
                            crate::tensor::Rng::new(0xCA11_B8A7 ^ cfg.seed ^ ((s as u64) << 32));
                        let mut toks: Vec<u32> = (0..prompt_len)
                            .map(|_| rng.below(cfg.vocab) as u32)
                            .collect();
                        let mut kv = LocalKvCache::new(cfg.n_heads, cfg.n_kv_heads, cfg.d_head);
                        let meta = BatchMeta::causal_prefill(0, prompt_len - 1, 1);
                        let mut x = head.embed_tokens(&toks[..prompt_len - 1]);
                        model.forward(&mut x, &meta, &mut kv, &mut NoObserver);
                        kv.commit(&meta.keys, &|_| false);
                        while toks.len() < CALIBRATION_LEN {
                            let pos = toks.len() - 1;
                            let meta = BatchMeta::causal_prefill(pos, 1, 2 + pos as u64);
                            let mut x = head.embed_tokens(&toks[pos..]);
                            model.forward(&mut x, &meta, &mut kv, &mut NoObserver);
                            kv.commit(&meta.keys, &|_| false);
                            let l = head.logits(&x, cfg.rms_eps);
                            toks.push(
                                if s % CALIBRATION_SAMPLED_EVERY != CALIBRATION_SAMPLED_EVERY - 1 {
                                    crate::model::argmax(l.row(0)) as u32
                                } else {
                                    sample(&probs_from_logits(l.row(0), 1.0), &mut rng)
                                },
                            );
                        }
                        toks
                    })
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let c = Arc::new(seqs);
    g.insert(key, c.clone());
    c
}

/// Calibration for the boundary after layer `boundary − 1`: the model's own
/// layers `0..boundary` over the calibration text, one batched causal prefill
/// per sequence. Computed once per process per (model, boundary) and shared
/// by every node of that process; separate processes derive identical
/// results from the same seeds.
pub fn klt_calibration(
    cfg: &crate::model::ModelConfig,
    boundary: usize,
) -> std::sync::Arc<KltCalibration> {
    // bit-identical on every machine: portable kernels only
    crate::tensor::accel::with_portable(|| klt_calibration_portable(cfg, boundary))
}

fn klt_calibration_portable(
    cfg: &crate::model::ModelConfig,
    boundary: usize,
) -> std::sync::Arc<KltCalibration> {
    use crate::kv_fabric::{BatchMeta, LocalKvCache};
    use crate::model::{HeadWeights, NoObserver, ShardModel};
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};
    type Key = (u64, usize, usize, usize);
    static CACHE: OnceLock<Mutex<HashMap<Key, Arc<KltCalibration>>>> = OnceLock::new();
    let key = (
        cfg.seed ^ cfg.source.key(),
        cfg.d_model,
        cfg.n_layers,
        boundary,
    );
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(c) = cache.lock().unwrap().get(&key) {
        return c.clone();
    }
    let seqs = calibration_tokens(cfg);
    let mut g = cache.lock().unwrap();
    if let Some(c) = g.get(&key) {
        return c.clone();
    }
    let head = HeadWeights::generate(cfg);
    let shard = ShardModel::generate(cfg, 0..boundary.min(cfg.n_layers), None);
    let d = cfg.d_model;
    let n: usize = seqs.iter().map(|s| s.len()).sum();
    let mut rows = Tensor2D::zeros(n, d);
    let mut all_cur = Vec::with_capacity(n);
    let mut all_prev = Vec::with_capacity(n);
    let mut at = 0;
    for toks in seqs.iter() {
        let meta = BatchMeta::causal_prefill(0, toks.len(), 0);
        let mut kv = LocalKvCache::new(cfg.n_heads, cfg.n_kv_heads, cfg.d_head);
        let mut x = head.embed_tokens(toks);
        shard.forward(&mut x, &meta, &mut kv, &mut NoObserver);
        let e = head.embed_tokens(toks);
        for t in 0..toks.len() {
            let r = rows.row_mut(at);
            for j in 0..d {
                r[j] = x.row(t)[j] - e.row(t)[j];
            }
            all_cur.push(toks[t]);
            // position 0 has no previous token: by convention it is itself
            all_prev.push(toks[t.saturating_sub(1)]);
            at += 1;
        }
    }
    let c = Arc::new(KltCalibration::conditional(
        &rows,
        &head.embed_tokens(&all_cur),
        &head.embed_tokens(&all_prev),
    ));
    g.insert(key, c.clone());
    c
}
