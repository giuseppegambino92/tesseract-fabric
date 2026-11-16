//! Mathematical verification of the core transforms: FWHT, HRQ-EF,
//! exact LSE merge, 2-D tree masks / tree attention, lossless tree
//! verification, Q8.24 determinism and KV commit compaction.

use tesseract_fabric::compression::fwht::{fwht_inplace, HadamardPlan};
use tesseract_fabric::compression::{
    HoloConfig, HolographicCompressor, HolographicDecompressor, TreeActivationCodec,
};
use tesseract_fabric::kv_fabric::disaggregated_attention::fabric_with_capacities;
use tesseract_fabric::kv_fabric::WirePrecision;
use tesseract_fabric::kv_fabric::{lse_merge, node_key, partial_entries, BatchMeta, LocalKvCache};
use tesseract_fabric::model::{HeadWeights, ModelConfig, NoObserver, ShardModel};
use tesseract_fabric::speculative::tree_braid::{
    build_tree_mask, verify_tree, MergedTree, ProposalKind, SpeculativeTree,
};
use tesseract_fabric::tensor::determinism::{emulate, CanonicalBoundaryAnchor};
use tesseract_fabric::tensor::{cosine, l2_norm, reference_attention, Rng, Tensor2D};

/// Timing-sensitive tests and the multi-second compute tests take this lock,
/// so a timing assertion never runs beside a calibration or model sweep.
static EXCLUSIVE: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn exclusive() -> std::sync::MutexGuard<'static, ()> {
    EXCLUSIVE.lock().unwrap_or_else(|e| e.into_inner())
}

// ---------------------------------------------------------------- FWHT

#[test]
fn fwht_matches_explicit_hadamard_and_is_orthonormal() {
    let n = 16;
    // Sylvester construction
    let mut h = vec![vec![1.0f64]];
    while h.len() < n {
        let m = h.len();
        let mut nh = vec![vec![0.0; 2 * m]; 2 * m];
        for i in 0..m {
            for j in 0..m {
                nh[i][j] = h[i][j];
                nh[i][j + m] = h[i][j];
                nh[i + m][j] = h[i][j];
                nh[i + m][j + m] = -h[i][j];
            }
        }
        h = nh;
    }
    let mut rng = Rng::new(11);
    let x: Vec<f32> = (0..n).map(|_| rng.normal()).collect();
    let mut y = x.clone();
    fwht_inplace(&mut y);
    for i in 0..n {
        let r: f64 = (0..n).map(|j| h[i][j] * x[j] as f64).sum::<f64>() / (n as f64).sqrt();
        assert!((r - y[i] as f64).abs() < 1e-5);
    }
    // norm preservation + self-inverse at a large size
    let x: Vec<f32> = (0..4096).map(|_| rng.normal()).collect();
    let mut y = x.clone();
    fwht_inplace(&mut y);
    assert!((l2_norm(&x) - l2_norm(&y)).abs() / l2_norm(&x) < 1e-5);
    fwht_inplace(&mut y);
    assert!(x.iter().zip(&y).all(|(a, b)| (a - b).abs() < 1e-4));
}

#[test]
fn randomized_hadamard_spreads_outliers_and_inverts_for_any_dim() {
    for &(dim, block) in &[
        (4096usize, None),
        (5000, None),
        (5000, Some(256usize)),
        (768, Some(64)),
    ] {
        let plan = match block {
            None => HadamardPlan::full(dim, 3),
            Some(b) => HadamardPlan::blocked(dim, b, 3),
        };
        let mut rng = Rng::new(dim as u64);
        let mut x: Vec<f32> = (0..dim).map(|_| rng.normal() * 0.1).collect();
        x[7] = 300.0; // massive activation outlier
        x[dim / 2] = -250.0;
        let mut y = vec![0.0; plan.padded];
        plan.forward(&x, &mut y);
        let peak = |v: &[f32]| {
            v.iter().fold(0.0f32, |m, a| m.max(a.abs()))
                / (v.iter().map(|a| a * a).sum::<f32>() / v.len() as f32).sqrt()
        };
        // a spike spreads over its whole Hadamard tile: crest drops ~√tile
        let tile = block.unwrap_or(plan.padded) as f32;
        eprintln!(
            "dim {dim} tile {tile}: crest factor {:.1} -> {:.2}",
            peak(&x),
            peak(&y)
        );
        assert!(
            peak(&y) < peak(&x) / 4.0,
            "dim {dim}: crest factor {} -> {}",
            peak(&x),
            peak(&y)
        );
        let mut back = vec![0.0; dim];
        plan.inverse(&mut y, &mut back);
        let err = x
            .iter()
            .zip(&back)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(err < 1e-3, "dim {dim} inverse err {err}");
    }
}

// ---------------------------------------------------------------- HRQ-EF

/// LLM-like residual trajectory: static base with a few massive-activation
/// channels, a slowly evolving low-rank component, and small i.i.d. noise.
fn residual_trajectory(d: usize, steps: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Rng::new(seed);
    let mut base: Vec<f32> = (0..d).map(|_| rng.normal()).collect();
    for c in [3usize, 97, 1024, 2900] {
        if c < d {
            base[c] = 40.0 * if rng.uniform() < 0.5 { -1.0 } else { 1.0 };
        }
    }
    let r = 16;
    let u: Vec<Vec<f32>> = (0..r)
        .map(|_| (0..d).map(|_| rng.normal()).collect())
        .collect();
    let freq: Vec<f32> = (0..r).map(|_| 0.02 + 0.2 * rng.uniform()).collect();
    let phase: Vec<f32> = (0..r)
        .map(|_| std::f32::consts::TAU * rng.uniform())
        .collect();
    (0..steps)
        .map(|t| {
            let mut h = base.clone();
            for k in 0..r {
                let a = 0.12 * (freq[k] * t as f32 + phase[k]).sin();
                for i in 0..d {
                    h[i] += a * u[k][i];
                }
            }
            for v in h.iter_mut() {
                *v += 0.03 * rng.normal();
            }
            h
        })
        .collect()
}

struct StreamResult {
    cum_drift: f64,
    min_cos: f64,
    bytes: usize,
}

fn run_stream(traj: &[Vec<f32>], cfg: HoloConfig) -> StreamResult {
    let d = traj[0].len();
    let mut tx = HolographicCompressor::new(d, cfg.clone());
    let mut rx = HolographicDecompressor::new(d, cfg);
    let mut cum = 0.0;
    let mut min_cos = 1.0f64;
    let mut bytes = 0;
    for (t, h) in traj.iter().enumerate() {
        let f = tx.compress(h);
        let wire = f.to_bytes();
        if t > 0 {
            bytes += wire.len();
        }
        let hh = rx.decompress(&f).to_vec();
        let err: f64 = h
            .iter()
            .zip(&hh)
            .map(|(a, b)| ((a - b) as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        cum += err;
        min_cos = min_cos.min(cosine(h, &hh));
    }
    StreamResult {
        cum_drift: cum,
        min_cos,
        bytes,
    }
}

#[test]
fn hrq_error_feedback_beats_stateless_quantization() {
    let d = 4096;
    let steps = 128;
    let traj = residual_trajectory(d, steps, 77);
    for bits in [2u8, 3] {
        let ef = run_stream(
            &traj,
            HoloConfig {
                bits,
                gamma: 0.95,
                ..Default::default()
            },
        );
        let stateless = run_stream(
            &traj,
            HoloConfig {
                bits,
                gamma: 0.0,
                ..Default::default()
            },
        );
        let fp16_bytes = (steps - 1) * d * 2;
        let reduction = 1.0 - ef.bytes as f64 / fp16_bytes as f64;
        let ratio = stateless.cum_drift / ef.cum_drift;
        eprintln!(
            "{bits}-bit: EF drift {:.2} vs stateless {:.2} (×{ratio:.2}); min cos EF {:.5} stateless {:.5}; payload reduction vs FP16 {:.1}%",
            ef.cum_drift,
            stateless.cum_drift,
            ef.min_cos,
            stateless.min_cos,
            reduction * 100.0
        );
        assert!(ratio > 5.0, "{bits}-bit drift reduction only ×{ratio:.2}");
        assert!(ef.min_cos > 0.995, "{bits}-bit min cosine {}", ef.min_cos);
        if bits == 2 {
            assert!(reduction > 0.85, "2-bit payload reduction {:.3}", reduction);
        } else {
            assert!(reduction > 0.79, "3-bit payload reduction {:.3}", reduction);
        }
    }
}

#[test]
fn tree_activation_codec_roundtrip_is_bit_exact_with_sender_reconstruction() {
    let mut rng = Rng::new(5);
    let x = Tensor2D::randn(15, 512, 1.0, &mut rng);
    let parents: Vec<i32> = (0..15)
        .map(|i: i32| if i == 0 { -1 } else { (i - 1) / 2 })
        .collect();
    let mut enc = TreeActivationCodec::new(512, HoloConfig::default());
    let mut dec = TreeActivationCodec::new(512, HoloConfig::default());
    let (bytes, recon) = enc.encode(&x, &parents, 4);
    let out = dec.decode(&bytes);
    assert_eq!(out.data, recon.data);
    assert!(cosine(&out.data, &x.data) > 0.99);
    assert!(
        bytes.len() < x.data.len() * 2 * 32 / 100,
        "4-bit tree payload {} B",
        bytes.len()
    );
}

// ---------------------------------------------------------------- LSE merge

fn lse_case(t_ctx: usize, shard_caps: &[usize], seed: u64, min_decode_reduction: f64) {
    let (nh, nkv, dh) = (8, 2, 64);
    let kd = nkv * dh;
    let mut rng = Rng::new(seed);
    let k: Vec<f32> = (0..t_ctx * kd).map(|_| rng.normal()).collect();
    let v: Vec<f32> = (0..t_ctx * kd).map(|_| rng.normal()).collect();
    let caps: Vec<(u16, usize)> = shard_caps
        .iter()
        .enumerate()
        .map(|(i, c)| (i as u16, *c))
        .collect();
    let mut fabric = fabric_with_capacities(&caps, 256, nh, nkv, dh);
    for t in 0..t_ctx {
        fabric
            .append(0, t, &k[t * kd..(t + 1) * kd], &v[t * kd..(t + 1) * kd])
            .unwrap();
    }
    // uneven placement check
    let pages = fabric.table.pages_per_node();
    assert!(pages.iter().all(|p| p.1 > 0), "{pages:?}");

    // (a) causal queries inside the context
    let rows = 3;
    let qpos = [t_ctx / 3, t_ctx - 1, 77usize];
    let q: Vec<f32> = (0..rows * nh * dh).map(|_| rng.normal()).collect();
    let (out, stats) = fabric.attend_and_merge(0, &q, rows, &qpos);
    let mut max_err = 0.0f32;
    for r in 0..rows {
        for h in 0..nh {
            let g = h / (nh / nkv);
            let kh: Vec<f32> = (0..t_ctx)
                .flat_map(|t| k[t * kd + g * dh..t * kd + (g + 1) * dh].to_vec())
                .collect();
            let vh: Vec<f32> = (0..t_ctx)
                .flat_map(|t| v[t * kd + g * dh..t * kd + (g + 1) * dh].to_vec())
                .collect();
            let mut refo = vec![0.0; dh];
            let lim = qpos[r];
            reference_attention(
                &q[(r * nh + h) * dh..(r * nh + h + 1) * dh],
                &kh,
                &vh,
                dh,
                &|j| j <= lim,
                &mut refo,
            );
            let got = &out[(r * nh + h) * dh..(r * nh + h + 1) * dh];
            for i in 0..dh {
                max_err = max_err.max((got[i] - refo[i]).abs());
            }
        }
    }
    let _ = stats;
    eprintln!(
        "T={t_ctx} shards={}: causal max|err|={max_err:.2e}",
        shard_caps.len()
    );
    assert!(max_err < 1e-5, "causal LSE merge error {max_err}");

    // Decode-step traffic: one query row (all heads) per layer. Traffic is
    // O(shards · heads · d_h) — independent of context length — while
    // migrating remote pages is O(T).
    let q1: Vec<f32> = (0..nh * dh).map(|_| rng.normal()).collect();
    let (_, s1) = fabric.attend_and_merge(0, &q1, 1, &[usize::MAX]);
    let reduction = 1.0 - s1.network_bytes as f64 / s1.migration_bytes as f64;
    eprintln!(
        "T={t_ctx} shards={}: decode attention traffic {} B vs FP16 KV-page migration {} B → −{:.2}%",
        shard_caps.len(),
        s1.network_bytes,
        s1.migration_bytes,
        reduction * 100.0
    );
    assert!(
        reduction > min_decode_reduction,
        "traffic reduction {reduction}"
    );

    // (b) speculative tree queries after the context: remote partials over the
    // context merged with a local partial over the node's tree ancestors.
    let parents = [-1i32, 0, 0, 1, 1, 2, 3];
    let n = parents.len();
    let mask = build_tree_mask(&parents);
    let tk: Vec<f32> = (0..n * kd).map(|_| rng.normal()).collect();
    let tv: Vec<f32> = (0..n * kd).map(|_| rng.normal()).collect();
    let tq: Vec<f32> = (0..n * nh * dh).map(|_| rng.normal()).collect();
    let mut max_err = 0.0f32;
    for i in 0..n {
        for h in 0..nh {
            let g = h / (nh / nkv);
            let qh = &tq[(i * nh + h) * dh..(i * nh + h + 1) * dh];
            // distributed: per-shard partials + local tree partial, LSE-merged
            let mut parts = Vec::new();
            for s in fabric.shards.iter_mut() {
                let mut qfull = vec![0.0; nh * dh];
                qfull[h * dh..(h + 1) * dh].copy_from_slice(qh);
                parts.push(s.partials(0, &qfull, 1, &[usize::MAX]).swap_remove(h));
            }
            let anc: Vec<(&[f32], &[f32])> = (0..n)
                .filter(|&j| mask[i * n + j] == 0.0)
                .map(|j| {
                    (
                        &tk[j * kd + g * dh..j * kd + (g + 1) * dh],
                        &tv[j * kd + g * dh..j * kd + (g + 1) * dh],
                    )
                })
                .collect();
            parts.push(partial_entries(qh, anc.into_iter(), dh));
            let refs: Vec<_> = parts.iter().collect();
            let mut got = vec![0.0; dh];
            lse_merge(&refs, &mut got);
            // monolithic: one buffer [context ++ tree] with causal-tree mask
            let mut kh: Vec<f32> = (0..t_ctx)
                .flat_map(|t| k[t * kd + g * dh..t * kd + (g + 1) * dh].to_vec())
                .collect();
            let mut vh: Vec<f32> = (0..t_ctx)
                .flat_map(|t| v[t * kd + g * dh..t * kd + (g + 1) * dh].to_vec())
                .collect();
            for j in 0..n {
                kh.extend_from_slice(&tk[j * kd + g * dh..j * kd + (g + 1) * dh]);
                vh.extend_from_slice(&tv[j * kd + g * dh..j * kd + (g + 1) * dh]);
            }
            let mut refo = vec![0.0; dh];
            reference_attention(
                qh,
                &kh,
                &vh,
                dh,
                &|j| j < t_ctx || mask[i * n + (j - t_ctx)] == 0.0,
                &mut refo,
            );
            for x in 0..dh {
                max_err = max_err.max((got[x] - refo[x]).abs());
            }
        }
    }
    eprintln!(
        "T={t_ctx} shards={}: tree-masked max|err|={max_err:.2e}",
        shard_caps.len()
    );
    assert!(max_err < 1e-5, "tree LSE merge error {max_err}");
}

#[test]
fn disaggregated_lse_merge_matches_monolithic_4_shards() {
    lse_case(4096, &[512 << 20, 256 << 20, 128 << 20, 64 << 20], 1, 0.98);
}

#[test]
fn disaggregated_lse_merge_matches_monolithic_8_uneven_shards_16k() {
    lse_case(
        16384,
        &[
            900 << 20,
            600 << 20,
            300 << 20,
            300 << 20,
            200 << 20,
            120 << 20,
            80 << 20,
            40 << 20,
        ],
        2,
        0.99,
    );
}

// ---------------------------------------------------------------- tree attention

#[test]
fn tree_batched_forward_equals_sequential_path_forward() {
    let cfg = ModelConfig::tiny();
    let shard = ShardModel::generate(&cfg, 0..cfg.n_layers, None);
    let head = HeadWeights::generate(&cfg);
    let prompt: Vec<u32> = vec![5, 17, 99, 3, 250, 8];
    // committed prefix
    let mut kv = LocalKvCache::new(cfg.n_heads, cfg.n_kv_heads, cfg.d_head);
    let mut x = head.embed_tokens(&prompt);
    let meta = BatchMeta::causal_prefill(0, prompt.len(), 0);
    shard.forward(&mut x, &meta, &mut kv, &mut NoObserver);
    let keys: Vec<u64> = meta.keys.clone();
    kv.commit(&keys, &|_| false);
    // a speculative tree of 9 nodes
    let parents = vec![-1i32, 0, 0, 1, 1, 2, 3, 6, 5];
    let tokens: Vec<u32> = vec![42, 7, 9, 100, 101, 33, 64, 12, 200];
    let depth = tesseract_fabric::speculative::tree_depths(&parents);
    let tree = SpeculativeTree {
        epoch: 1,
        tokens: tokens.clone(),
        parents: parents.clone(),
        depth: depth.clone(),
        child_q: vec![vec![]; 9],
        path_q: vec![1.0; 9],
        proposal: ProposalKind::Deterministic,
    };
    let merged = MergedTree::split(tree, 32, 1, 1);
    let tmeta = merged.strand_meta(0, prompt.len());
    let mut tkv = kv.clone();
    let mut tx = head.embed_tokens(&tokens);
    shard.forward(&mut tx, &tmeta, &mut tkv, &mut NoObserver);
    // sequential reference for each node: run its root→node path causally
    for node in 0..tokens.len() {
        let path = merged.tree.path_to(node);
        let ptoks: Vec<u32> = path.iter().map(|&i| tokens[i]).collect();
        let mut skv = kv.clone();
        let mut sx = head.embed_tokens(&ptoks);
        let smeta = BatchMeta::causal_prefill(prompt.len(), ptoks.len(), 9);
        shard.forward(&mut sx, &smeta, &mut skv, &mut NoObserver);
        let a = tx.row(node);
        let b = sx.row(ptoks.len() - 1);
        let err = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(err < 1e-4, "node {node}: tree vs sequential err {err}");
    }
    // commit a path and verify compaction keeps exactly those entries
    let path = merged.tree.path_to(7);
    let acc: Vec<u64> = path.iter().map(|&i| merged.key(i)).collect();
    tkv.commit(&acc, &|_| false);
    assert_eq!(tkv.committed_len(0), prompt.len() + path.len());
    assert_eq!(tkv.pending_len(), 0);
}

// ---------------------------------------------------------------- lossless verification

fn random_dist(v: usize, rng: &mut Rng, sharp: f32) -> Vec<f32> {
    let mut p: Vec<f32> = (0..v).map(|_| (rng.normal() * sharp).exp()).collect();
    let s: f32 = p.iter().sum();
    p.iter_mut().for_each(|x| *x /= s);
    p
}

#[test]
fn tree_rejection_sampling_is_lossless() {
    let v = 6;
    let mut rng = Rng::new(123);
    // target: first-token dist p0 and conditional p1[x1]
    let p0 = random_dist(v, &mut rng, 1.2);
    let p1: Vec<Vec<f32>> = (0..v).map(|_| random_dist(v, &mut rng, 1.2)).collect();
    for kind in [ProposalKind::Deterministic, ProposalKind::SampledIid] {
        let trials = 120_000;
        let mut counts = vec![0f64; v * v];
        let mut drafter_rng = Rng::new(9);
        let q0 = random_dist(v, &mut drafter_rng, 0.8);
        for _ in 0..trials {
            // build a depth-2 tree: root → 3 candidates for x1; first cand → 2 for x2
            let pick = |q: &[f32], r: &mut Rng| -> u32 {
                let u = r.uniform();
                let mut c = 0.0;
                for (i, &x) in q.iter().enumerate() {
                    c += x;
                    if u < c {
                        return i as u32;
                    }
                }
                (q.len() - 1) as u32
            };
            let (c1, c2): (Vec<u32>, Vec<u32>) = match kind {
                ProposalKind::Deterministic => (vec![2, 0, 4], vec![1, 3]),
                ProposalKind::SampledIid => (
                    (0..3).map(|_| pick(&q0, &mut drafter_rng)).collect(),
                    (0..2).map(|_| pick(&q0, &mut drafter_rng)).collect(),
                ),
            };
            let qs: Vec<(u32, f32)> = q0.iter().enumerate().map(|(i, &x)| (i as u32, x)).collect();
            let mut tokens = vec![0u32];
            let mut parents = vec![-1i32];
            for &c in &c1 {
                tokens.push(c);
                parents.push(0);
            }
            for &c in &c2 {
                tokens.push(c);
                parents.push(1);
            }
            let n = tokens.len();
            let depth = tesseract_fabric::speculative::tree_depths(&parents);
            let tree = SpeculativeTree {
                epoch: 0,
                tokens: tokens.clone(),
                parents,
                depth,
                child_q: vec![qs.clone(); n],
                path_q: vec![1.0; n],
                proposal: kind,
            };
            let out = verify_tree(
                &tree,
                &mut |node| {
                    if node == 0 {
                        p0.clone()
                    } else {
                        p1[tokens[node] as usize].clone()
                    }
                },
                &mut rng,
            );
            let em = out.emitted(&tree);
            // first two emitted tokens (if only one emitted, sample the second from target)
            let x1 = em[0] as usize;
            let x2 = if em.len() > 1 {
                em[1] as usize
            } else {
                let p = &p1[x1];
                let u = rng.uniform();
                let mut c = 0.0;
                let mut s = v - 1;
                for (i, &x) in p.iter().enumerate() {
                    c += x;
                    if u < c {
                        s = i;
                        break;
                    }
                }
                s
            };
            counts[x1 * v + x2] += 1.0;
        }
        let mut tv = 0.0;
        for a in 0..v {
            for b in 0..v {
                let expect = p0[a] as f64 * p1[a][b] as f64;
                tv += (counts[a * v + b] / trials as f64 - expect).abs();
            }
        }
        tv *= 0.5;
        eprintln!("{kind:?}: total-variation distance of joint (x1,x2) vs target = {tv:.4}");
        assert!(tv < 0.012, "{kind:?} not lossless: TV {tv}");
    }
}

// ---------------------------------------------------------------- determinism

#[test]
fn q824_anchor_gives_bit_identical_rmsnorm_across_reduction_orders() {
    let a = CanonicalBoundaryAnchor::new(64);
    let mut rng = Rng::new(31);
    let mut divergent_naive = 0;
    for trial in 0..200 {
        let d = 4096;
        let x: Vec<f32> = (0..d)
            .map(|_| rng.normal() * (1.0 + 30.0 * (trial % 3) as f32))
            .collect();
        let w: Vec<f32> = (0..d).map(|_| 1.0 + 0.1 * rng.normal()).collect();
        // naive: different silicon reductions
        let sums = [
            emulate::sum_sq_scalar(&x),
            emulate::sum_sq_lanes(&x, 4),
            emulate::sum_sq_lanes(&x, 8),
            emulate::sum_sq_lanes(&x, 32),
            emulate::sum_sq_fma(&x),
            emulate::sum_sq_reversed(&x),
        ];
        let outs: Vec<Vec<f32>> = sums
            .iter()
            .map(|&s| {
                let mut o = vec![0.0; d];
                emulate::rms_with(&x, &w, 1e-5, s, &mut o);
                o
            })
            .collect();
        if outs.iter().any(|o| {
            o.iter()
                .zip(&outs[0])
                .any(|(p, q)| p.to_bits() != q.to_bits())
        }) {
            divergent_naive += 1;
        }
        // anchored: permute the input order of accumulation by shuffling blocks
        // (integer sums are associative) and require identical bits.
        let mut ref_out = vec![0.0; d];
        a.rms_norm(&x, &w, 1e-5, &mut ref_out);
        let blocks = a.encode(&x);
        let ss1 = a.exact_sum_squares(&blocks);
        let mut rev = blocks.clone();
        rev.reverse();
        let ss2 = a.exact_sum_squares(&rev);
        assert_eq!(ss1.to_bits(), ss2.to_bits());
        let mut again = vec![0.0; d];
        a.rms_norm(&x, &w, 1e-5, &mut again);
        assert!(ref_out
            .iter()
            .zip(&again)
            .all(|(p, q)| p.to_bits() == q.to_bits()));
        // and stays numerically faithful
        assert!(cosine(&ref_out, &outs[0]) > 0.999_999);
    }
    eprintln!(
        "naive float reductions diverged bitwise in {divergent_naive}/200 trials; anchored: 0/200"
    );
    assert!(divergent_naive > 100);
}

#[test]
fn kv_commit_compaction_is_microseconds() {
    let _guard = exclusive();
    let (nh, nkv, dh) = (8, 4, 64);
    let mut kv = LocalKvCache::new(nh, nkv, dh);
    let kd = nkv * dh;
    let mut rng = Rng::new(4);
    let mut total_ns = 0u128;
    let iters = 200;
    for it in 0..iters {
        let m = 16;
        let k = Tensor2D::randn(m, kd, 1.0, &mut rng);
        let v = Tensor2D::randn(m, kd, 1.0, &mut rng);
        let meta = BatchMeta {
            positions: (0..m).collect(),
            keys: (0..m).map(|i| node_key(it as u64, i)).collect(),
            visible: vec![vec![]; m],
        };
        kv.push_pending(0, &k, &v, &meta);
        let acc: Vec<u64> = [0usize, 1, 4, 9]
            .iter()
            .map(|&i| node_key(it as u64, i))
            .collect();
        let t0 = std::time::Instant::now();
        kv.commit(&acc, &|_| false);
        total_ns += t0.elapsed().as_nanos();
    }
    let per = total_ns as f64 / iters as f64 / 1000.0;
    eprintln!("KV commit compaction (16 pending → 4 accepted, one layer): {per:.2} µs");
    assert!(per < 5.0, "compaction {per} µs");
}

// ---------------------------------------------------------------- V2

fn min_time(reps: usize, f: &mut dyn FnMut()) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..reps {
        let t = std::time::Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64());
    }
    best
}

#[test]
fn tiled_batch_gemm_amortizes_weight_loads() {
    let _guard = exclusive();
    use tesseract_fabric::tensor::{matmul_tiled_batch, Linear};
    tesseract_fabric::tensor::prefer_performance_cores();
    let mut rng = Rng::new(17);
    let w = Tensor2D::randn(2048, 512, 0.05, &mut rng);
    // correctness against a naive f64 reference for every tile shape
    for m in [1usize, 2, 3, 5, 6, 7, 8, 11, 16] {
        let x = Tensor2D::randn(m, 512, 1.0, &mut rng);
        let mut y = Tensor2D::zeros(m, 2048);
        matmul_tiled_batch(&x, &w, &mut y);
        for i in 0..m {
            for j in (0..2048).step_by(97) {
                let r: f64 = (0..512)
                    .map(|t| x.row(i)[t] as f64 * w.row(j)[t] as f64)
                    .sum();
                assert!((r - y.data[i * 2048 + j] as f64).abs() < 1e-3, "m {m}");
            }
        }
    }
    let lin = Linear::new(w);
    // interleave the three batch sizes (min of each over 300 rounds) so host
    // load drift during the measurement hits all of them alike
    let xs: Vec<Tensor2D> = [1usize, 6, 8]
        .iter()
        .map(|&m| Tensor2D::randn(m, 512, 1.0, &mut rng))
        .collect();
    let mut best = [f64::MAX; 3];
    for _ in 0..300 {
        for (b, x) in best.iter_mut().zip(&xs) {
            *b = b.min(min_time(1, &mut || {
                std::hint::black_box(lin.forward(std::hint::black_box(x)));
            }));
        }
    }
    let (t1, t6, t8) = (best[0], best[1], best[2]);
    eprintln!("tiled batch GEMM 2048×512: B=6 costs {:.2}× B=1, B=8 costs {:.2}× B=1 (naive would be 6× / 8×)", t6 / t1, t8 / t1);
    assert!(t6 / t1 < 3.0, "B=6 ratio {}", t6 / t1);
    assert!(t8 / t1 < 4.0, "B=8 ratio {}", t8 / t1);
}

#[test]
fn compact_lse_wire_saves_over_99_2_percent_within_1e4() {
    let (nh, nkv, dh) = (8, 2, 64);
    let kd = nkv * dh;
    for (ctx, min_red) in [(4096usize, 0.992), (65536, 0.999)] {
        let mut rng = Rng::new(ctx as u64);
        let k: Vec<f32> = (0..ctx * kd).map(|_| rng.normal()).collect();
        let v: Vec<f32> = (0..ctx * kd).map(|_| rng.normal()).collect();
        let mut fab = fabric_with_capacities(
            &[
                (0, 400 << 20),
                (1, 200 << 20),
                (2, 150 << 20),
                (3, 100 << 20),
            ],
            256,
            nh,
            nkv,
            dh,
        );
        fab.precision = WirePrecision::F16;
        for t in 0..ctx {
            fab.append(0, t, &k[t * kd..(t + 1) * kd], &v[t * kd..(t + 1) * kd])
                .unwrap();
        }
        let q: Vec<f32> = (0..nh * dh).map(|_| rng.normal()).collect();
        let (out, st) = fab.attend_and_merge(0, &q, 1, &[usize::MAX]);
        let mut err = 0.0f32;
        for h in 0..nh {
            let g = h / (nh / nkv);
            let kh: Vec<f32> = (0..ctx)
                .flat_map(|t| k[t * kd + g * dh..t * kd + (g + 1) * dh].to_vec())
                .collect();
            let vh: Vec<f32> = (0..ctx)
                .flat_map(|t| v[t * kd + g * dh..t * kd + (g + 1) * dh].to_vec())
                .collect();
            let mut r = vec![0.0; dh];
            reference_attention(&q[h * dh..(h + 1) * dh], &kh, &vh, dh, &|_| true, &mut r);
            for d in 0..dh {
                err = err.max((r[d] - out[h * dh + d]).abs());
            }
        }
        let red = 1.0 - st.network_bytes as f64 / st.migration_bytes as f64;
        eprintln!("compact wire T={ctx}: {} B incl. headers vs {} B migration (−{:.3}%), max err {err:.2e}", st.network_bytes, st.migration_bytes, red * 100.0);
        assert!(red > min_red, "reduction {red}");
        assert!(err < 1e-4, "error {err}");
    }
}

#[test]
fn slab_commit_is_lock_free_and_fast_under_concurrent_load() {
    let _guard = exclusive();
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    let (nh, nkv, dh, layers) = (8, 4, 64, 16);
    let kd = nkv * dh;
    let stop = Arc::new(AtomicBool::new(false));
    // background load: busy GEMM threads competing for cores and memory bandwidth
    let load: Vec<_> = (0..4)
        .map(|i| {
            let stop = stop.clone();
            std::thread::spawn(move || {
                let mut rng = Rng::new(i);
                let w = Tensor2D::randn(512, 512, 0.1, &mut rng);
                let x = Tensor2D::randn(8, 512, 1.0, &mut rng);
                let lin = tesseract_fabric::tensor::Linear::new(w);
                while !stop.load(Ordering::Relaxed) {
                    std::hint::black_box(lin.forward(&x));
                }
            })
        })
        .collect();
    let mut kv = LocalKvCache::new(nh, nkv, dh);
    let mut rng = Rng::new(4);
    let mut times = Vec::new();
    for it in 0..400u64 {
        let m = 16;
        let k = Tensor2D::randn(m, kd, 1.0, &mut rng);
        let v = Tensor2D::randn(m, kd, 1.0, &mut rng);
        let meta = BatchMeta {
            positions: (0..m).collect(),
            keys: (0..m).map(|i| node_key(it, i)).collect(),
            visible: vec![vec![]; m],
        };
        for l in 0..layers {
            kv.push_pending(l, &k, &v, &meta);
        }
        let acc: Vec<u64> = [0usize, 1, 4, 9].iter().map(|&i| node_key(it, i)).collect();
        let t0 = std::time::Instant::now();
        kv.commit(&acc, &|_| false);
        times.push(t0.elapsed().as_nanos() as f64 / 1e3);
    }
    stop.store(true, Ordering::Relaxed);
    for h in load {
        h.join().unwrap();
    }
    assert_eq!(kv.committed_len(0), 400 * 4);
    assert_eq!(kv.pending_len(), 0);
    // rejected slots are recycled: the slab holds committed rows + one tree
    assert!(kv.slots.next_slot as usize <= 400 * 4 + 16);
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[times.len() / 2];
    eprintln!("slab commit (16 layers, 16 pending → 4 accepted) under 4 busy threads: median {median:.2} µs, p90 {:.2} µs", times[times.len() * 9 / 10]);
    assert!(median < 2.0, "median commit {median} µs");
}

// ---------------------------------------------------------------- V3

#[test]
fn contextual_model_defeats_zero_layer_drafter_but_not_early_exit() {
    let _guard = exclusive();
    use tesseract_fabric::model::{argmax, probs_from_logits, sample, ModelConfig, ReferenceModel};
    let cfg = ModelConfig::bench_moe();
    let m = ReferenceModel::new(&cfg);
    let mut rng = Rng::new(7);
    let mut toks: Vec<u32> = (0..16).map(|_| rng.below(cfg.vocab) as u32).collect();
    // sample a continuation from the target itself
    let mut kv = LocalKvCache::new(cfg.n_heads, cfg.n_kv_heads, cfg.d_head);
    let mut x = m.head.embed_tokens(&toks[..15]);
    let meta = BatchMeta::causal_prefill(0, 15, 0);
    m.shard.forward(&mut x, &meta, &mut kv, &mut NoObserver);
    kv.commit(&meta.keys, &|_| false);
    for s in 0..96u64 {
        let last = *toks.last().unwrap();
        let mut x = m.head.embed_tokens(&[last]);
        let meta = BatchMeta::causal_prefill(toks.len() - 1, 1, 100 + s);
        m.shard.forward(&mut x, &meta, &mut kv, &mut NoObserver);
        kv.commit(&meta.keys, &|_| false);
        toks.push(sample(
            &probs_from_logits(m.head.logits(&x, cfg.rms_eps).row(0), 1.0),
            &mut rng,
        ));
    }
    let target = m.forward_sequence(&toks);
    let early = ShardModel::generate(&cfg, 0..2, None);
    let mut kv2 = LocalKvCache::new(cfg.n_heads, cfg.n_kv_heads, cfg.d_head);
    let mut x2 = m.head.embed_tokens(&toks);
    early.forward(
        &mut x2,
        &BatchMeta::causal_prefill(0, toks.len(), 0),
        &mut kv2,
        &mut NoObserver,
    );
    let l2 = m.head.logits(&x2, cfg.rms_eps);
    let l0 = m.head.logits(&m.head.embed_tokens(&toks), cfg.rms_eps);
    let (mut a0, mut a2, mut n) = (0usize, 0usize, 0usize);
    for t in 15..toks.len() - 1 {
        let ta = argmax(target.row(t));
        a0 += (argmax(l0.row(t)) == ta) as usize;
        a2 += (argmax(l2.row(t)) == ta) as usize;
        n += 1;
    }
    let (r0, r2) = (a0 as f64 / n as f64, a2 as f64 / n as f64);
    eprintln!(
        "contextual model: zero-layer agreement {:.1}%, 2-layer early-exit agreement {:.1}%",
        r0 * 100.0,
        r2 * 100.0
    );
    assert!(r0 <= 0.55, "zero-layer drafter too good: {r0}");
    assert!(r2 >= 0.78, "early-exit drafter too weak: {r2}");
}

#[test]
fn hrq_v21_entropy_codec_is_exact_capped_and_beats_v2() {
    let _guard = exclusive();
    use tesseract_fabric::compression::{EntropyResidualCodec, TrunkResidualCodec};
    let mut rng = Rng::new(21);
    let d = 512;
    let x = Tensor2D::randn(48, d, 1.0, &mut rng);
    let base = Tensor2D::randn(48, d, 0.5, &mut rng);
    let mut enc = EntropyResidualCodec::new(d, 0.25, 9);
    let (bytes, recon) = enc.encode(&x, &base);
    let dec = EntropyResidualCodec::new(d, 0.25, 9).decode(&bytes, &base);
    assert_eq!(
        dec.data, recon.data,
        "receiver must reproduce the sender's reconstruction bit for bit"
    );
    let per_row = (bytes.len() - 8) as f64 / 48.0;
    assert!(per_row <= (d * 2) as f64 * 0.25, "row bytes {per_row}");
    let (b2, r2) = TrunkResidualCodec::new(d, 3, 9).encode(&x, &base);
    let mse = |r: &Tensor2D| {
        x.data
            .iter()
            .zip(&r.data)
            .map(|(a, b)| ((a - b) as f64).powi(2))
            .sum::<f64>()
    };
    eprintln!(
        "V2.1: {per_row:.1} B/row, V2: {:.1} B/row, MSE ratio V2.1/V2 = {:.3}",
        (b2.len() - 10) as f64 / 48.0,
        mse(&recon) / mse(&r2)
    );
    assert!(mse(&recon) < 0.5 * mse(&r2));
}

#[test]
fn coalesced_routing_keeps_anchor_exact_and_bounds_experts() {
    use tesseract_fabric::model::{
        route_coalesced, route_topk, FfnWeights, LayerWeights, ModelConfig,
    };
    let cfg = ModelConfig::bench_moe();
    let layer = LayerWeights::generate(&cfg, 3, None);
    let FfnWeights::Moe { router, top_k, .. } = &layer.ffn else {
        panic!("MoE expected")
    };
    let mut rng = Rng::new(3);
    for _ in 0..20 {
        let x = Tensor2D::randn(5, cfg.d_model, 1.0, &mut rng);
        let exact = route_topk(router, &x, *top_k);
        let co = route_coalesced(router, &x, *top_k);
        assert_eq!(exact[0], co[0], "anchor row must keep its exact routing");
        let distinct: std::collections::BTreeSet<usize> =
            co.iter().flat_map(|r| r.0.clone()).collect();
        assert!(distinct.len() <= top_k + 1);
        for r in &co {
            assert!((r.1.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        }
    }
}

#[test]
fn calibrated_roofline_explores_then_follows_its_model() {
    use tesseract_fabric::speculative::RooflineController;
    let mut c = RooflineController::new(vec![1, 2, 3, 4, 6]);
    // a model where 4 rows cost 1.5× one row
    c.set_time_model(vec![10.0, 11.0, 13.0, 15.0, 22.0]);
    assert_eq!(
        c.choose(),
        2,
        "must probe a tree before any acceptance evidence"
    );
    let q = vec![1.0, 0.95, 0.9, 0.85, 0.8, 0.75];
    for _ in 0..6 {
        c.record(4, &q, 4, 15, 14);
    }
    assert!(
        c.choose() >= 3,
        "high acceptance should pick a tree, chose {}",
        c.choose()
    );
    let mut d = RooflineController::new(vec![1, 2, 3, 4, 6]);
    d.set_time_model(vec![10.0, 18.0, 26.0, 34.0, 50.0]);
    for _ in 0..6 {
        d.record(4, &q, 1, 34, 33);
    }
    assert_eq!(
        d.choose(),
        1,
        "poor acceptance on expensive rows must fall back to N = 1"
    );
}

// ---------------------------------------------------------------- V4

#[test]
fn bfp_roundtrip_error_is_within_half_a_quantum() {
    use tesseract_fabric::kv_fabric::disaggregated_attention::{bfp_get, bfp_put};
    use tesseract_fabric::wire::protocol::{ByteReader, ByteWriter};
    let mut rng = Rng::new(0xBF9);
    for bits in [8u32, 12, 16] {
        for scale in [1e-3f32, 1.0, 37.0] {
            let v: Vec<f32> = (0..64).map(|_| rng.normal() * scale).collect();
            let mut w = ByteWriter::with_capacity(256);
            bfp_put(&mut w, &v, bits);
            let b = w.into_inner();
            assert_eq!(b.len(), 1 + (64 * bits as usize).div_ceil(8));
            let d = bfp_get(&mut ByteReader::new(&b), 64, bits);
            let amax = v.iter().fold(0.0f32, |m, x| m.max(x.abs()));
            // shared exponent e = ceil(log2 amax): quantum 2^e / qmax ≤ 2·amax / qmax
            let qmax = ((1u32 << (bits - 1)) - 1) as f32;
            let quantum = 2f32.powi(amax.log2().ceil() as i32) / qmax;
            for (a, r) in v.iter().zip(&d) {
                assert!(
                    (a - r).abs() <= 0.5 * quantum * (1.0 + 1e-5),
                    "bits {bits} scale {scale}: {a} vs {r}"
                );
            }
        }
    }
}

#[test]
fn weight_bound_is_rigorous_and_skipped_partials_are_negligible() {
    use tesseract_fabric::kv_fabric::disaggregated_attention::{
        decode_partials, encode_partials, weight_bound,
    };
    use tesseract_fabric::kv_fabric::AttnPartial;
    let dh = 32;
    let mut rng = Rng::new(0x5A1E);
    let mut skipped = 0;
    for trial in 0..400 {
        // one coordinator-held partial (the hint) and one remote partial with
        // a logit gap that sweeps from comparable to vanishing
        let mk = |rng: &mut Rng, m: f32| AttnPartial {
            m,
            l: 1.0 + rng.uniform() * 50.0,
            o: (0..dh).map(|_| rng.normal()).collect(),
        };
        let gap = trial as f32 * 0.06;
        let hint = mk(&mut rng, 4.0);
        let remote = mk(&mut rng, 4.0 - gap);
        // the true global mass is ≥ the hint's mass, so the true merge weight
        // of `remote` is ≤ the bound computed from the hint alone
        let m_other = 3.0 + rng.uniform();
        let other = mk(&mut rng, m_other);
        let bound = weight_bound(&remote, (hint.m, hint.l));
        let mg = hint.m.max(remote.m).max(other.m);
        let mass = |p: &AttnPartial| p.l * (p.m - mg).exp();
        let true_w = mass(&remote) / (mass(&hint) + mass(&remote) + mass(&other));
        assert!(
            true_w <= bound * (1.0 + 1e-5),
            "weight {true_w} > bound {bound}"
        );
        let wire = encode_partials(
            std::slice::from_ref(&remote),
            WirePrecision::Bfp,
            &[false],
            &[Some((hint.m, hint.l))],
        );
        let dec = decode_partials(&wire);
        let mut exact = vec![0.0; dh];
        let mut got = vec![0.0; dh];
        lse_merge(&[&hint, &remote, &other], &mut exact);
        lse_merge(&[&hint, &dec[0], &other], &mut got);
        let err = exact
            .iter()
            .zip(&got)
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        // the merge error is the partial's true weight times its O error, so it
        // is bounded by weight_bound × (half a BFP quantum, or |O|∞ if skipped)
        let amax = remote.o.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        let half_quantum = |bits: u32| {
            0.5 * 2f32.powi(amax.log2().ceil() as i32) / ((1u32 << (bits - 1)) - 1) as f32
        };
        let allowed = if bound < 1e-7 {
            skipped += 1;
            // code 3: no output vector on the wire
            assert!(wire.len() <= 9 + 9, "skipped partial still carries O");
            bound * amax
        } else if bound < 0.25 {
            bound * half_quantum(12)
        } else {
            bound * half_quantum(16)
        };
        assert!(
            err <= allowed + 1e-6,
            "trial {trial}: merge error {err} > {allowed} (bound {bound})"
        );
    }
    assert!(skipped > 0, "sweep never reached the skip threshold");
}

#[test]
fn bfp_lse_wire_is_smaller_than_f16_and_within_8e5() {
    let (nh, nkv, dh) = (8, 2, 64);
    let kd = nkv * dh;
    for ctx in [4096usize, 16384] {
        let mut rng = Rng::new(ctx as u64 + 1);
        let k: Vec<f32> = (0..ctx * kd).map(|_| rng.normal()).collect();
        let v: Vec<f32> = (0..ctx * kd).map(|_| rng.normal()).collect();
        let q: Vec<f32> = (0..nh * dh).map(|_| rng.normal()).collect();
        let run = |prec: WirePrecision| {
            let mut fab = fabric_with_capacities(
                &[
                    (0, 400 << 20),
                    (1, 200 << 20),
                    (2, 150 << 20),
                    (3, 100 << 20),
                ],
                256,
                nh,
                nkv,
                dh,
            );
            fab.precision = prec;
            for t in 0..ctx {
                fab.append(0, t, &k[t * kd..(t + 1) * kd], &v[t * kd..(t + 1) * kd])
                    .unwrap();
            }
            fab.attend_and_merge(0, &q, 1, &[usize::MAX])
        };
        let (out_f16, st_f16) = run(WirePrecision::F16);
        let (out, st) = run(WirePrecision::Bfp);
        let mut err = 0.0f32;
        let mut err_f16 = 0.0f32;
        for h in 0..nh {
            let g = h / (nh / nkv);
            let kh: Vec<f32> = (0..ctx)
                .flat_map(|t| k[t * kd + g * dh..t * kd + (g + 1) * dh].to_vec())
                .collect();
            let vh: Vec<f32> = (0..ctx)
                .flat_map(|t| v[t * kd + g * dh..t * kd + (g + 1) * dh].to_vec())
                .collect();
            let mut r = vec![0.0; dh];
            reference_attention(&q[h * dh..(h + 1) * dh], &kh, &vh, dh, &|_| true, &mut r);
            for d in 0..dh {
                err = err.max((r[d] - out[h * dh + d]).abs());
                err_f16 = err_f16.max((r[d] - out_f16[h * dh + d]).abs());
            }
        }
        eprintln!(
            "BFP wire T={ctx}: {} B (f16 {} B), max err {err:.2e} (f16 {err_f16:.2e})",
            st.network_bytes, st_f16.network_bytes
        );
        assert!(err < 8e-5, "BFP error {err} at {ctx}");
        assert!(err <= err_f16, "BFP {err} worse than f16 {err_f16}");
        assert!(st.network_bytes < st_f16.network_bytes);
    }
}

#[test]
fn attention_only_shortlist_drafter_passes_the_honesty_test_under_2_5_percent() {
    let _guard = exclusive();
    use tesseract_fabric::model::{argmax, probs_from_logits, sample, ReferenceModel};
    let cfg = ModelConfig::bench_moe();
    let m = ReferenceModel::new(&cfg);
    let mut rng = Rng::new(7);
    let mut toks: Vec<u32> = (0..16).map(|_| rng.below(cfg.vocab) as u32).collect();
    let mut kv = LocalKvCache::new(cfg.n_heads, cfg.n_kv_heads, cfg.d_head);
    let mut x = m.head.embed_tokens(&toks[..15]);
    let meta = BatchMeta::causal_prefill(0, 15, 0);
    m.shard.forward(&mut x, &meta, &mut kv, &mut NoObserver);
    kv.commit(&meta.keys, &|_| false);
    for s in 0..256u64 {
        let last = *toks.last().unwrap();
        let mut x = m.head.embed_tokens(&[last]);
        let meta = BatchMeta::causal_prefill(toks.len() - 1, 1, 100 + s);
        m.shard.forward(&mut x, &meta, &mut kv, &mut NoObserver);
        kv.commit(&meta.keys, &|_| false);
        toks.push(sample(
            &probs_from_logits(m.head.logits(&x, cfg.rms_eps).row(0), 1.0),
            &mut rng,
        ));
    }
    let target = m.forward_sequence(&toks);
    // the target's own layer 0, FFN skipped: no trained or tuned weights
    let mut attn = ShardModel::generate(&cfg, 0..1, None);
    attn.skip_ffn = true;
    let mut kv1 = LocalKvCache::new(cfg.n_heads, cfg.n_kv_heads, cfg.d_head);
    let mut x1 = m.head.embed_tokens(&toks);
    attn.forward(
        &mut x1,
        &BatchMeta::causal_prefill(0, toks.len(), 0),
        &mut kv1,
        &mut NoObserver,
    );
    // the deployed drafter's head: rank-r filter + exact rescoring of a shortlist
    let (sr, sk) = tesseract_fabric::speculative::drafter::shortlist_params(&cfg);
    let mut sh =
        tesseract_fabric::speculative::drafter::ShortlistHead::new(&m.head, sr, sk, 8, cfg.rms_eps);
    sh.certify = false;
    let l0 = m.head.logits(&m.head.embed_tokens(&toks), cfg.rms_eps);
    let (mut a0, mut a1, mut n) = (0usize, 0usize, 0usize);
    for t in 15..toks.len() - 1 {
        let ta = argmax(target.row(t));
        a0 += (argmax(l0.row(t)) == ta) as usize;
        a1 += (argmax(&sh.logits_row(x1.row(t))) == ta) as usize;
        n += 1;
    }
    // weight MACs per drafted node vs one target step
    let d = cfg.d_model as f64;
    let attn = 2.0 * d * (cfg.n_heads * cfg.d_head + cfg.n_kv_heads * cfg.d_head) as f64;
    let moe = cfg.moe.as_ref().unwrap();
    let step = cfg.n_layers as f64
        * (attn + d * moe.n_experts as f64 + (moe.top_k * 3 * moe.d_expert) as f64 * d)
        + d * cfg.vocab as f64;
    let cost = (attn + sh.mean_macs()) / step;
    eprintln!("drafter cost {:.2}% of a target step", cost * 100.0);
    assert!(cost < 0.025, "drafter costs {cost} of a step");
    let (r0, r1) = (a0 as f64 / n as f64, a1 as f64 / n as f64);
    eprintln!(
        "attention-only drafter: zero-layer {:.1}%, 1 attention-only layer + shortlist head {:.1}% ({n} tokens)",
        r0 * 100.0,
        r1 * 100.0
    );
    assert!(r0 <= 0.55, "zero-layer drafter too good: {r0}");
    assert!(r1 >= 0.78, "attention-only drafter too weak: {r1}");
}

// ---------------------------------------------------------------- V5

#[test]
fn shortlist_lm_head_certified_mode_is_exact_and_drafting_mode_keeps_argmax() {
    let _guard = exclusive();
    use tesseract_fabric::model::{argmax, ReferenceModel};
    use tesseract_fabric::speculative::drafter::{shortlist_params, ShortlistHead};
    let cfg = ModelConfig::bench_moe();
    let m = ReferenceModel::new(&cfg);
    let mut rng = Rng::new(0x5E1);
    let toks: Vec<u32> = (0..160).map(|_| rng.below(cfg.vocab) as u32).collect();
    // real drafter hidden states: layer 0 attention-only over a token stream
    let mut attn = ShardModel::generate(&cfg, 0..1, None);
    attn.skip_ffn = true;
    let mut kv = LocalKvCache::new(cfg.n_heads, cfg.n_kv_heads, cfg.d_head);
    let mut x = m.head.embed_tokens(&toks);
    attn.forward(
        &mut x,
        &BatchMeta::causal_prefill(0, toks.len(), 0),
        &mut kv,
        &mut NoObserver,
    );
    let full = m.head.logits(&x, cfg.rms_eps);
    // certified mode: the top-8 tokens and their logits equal the full head's
    let mut exact = ShortlistHead::new(&m.head, 32, 32, 8, cfg.rms_eps);
    for i in 0..x.rows {
        let l = exact.logits_row(x.row(i));
        let f = full.row(i);
        let mut fi: Vec<usize> = (0..f.len()).collect();
        fi.sort_by(|a, b| f[*b].total_cmp(&f[*a]));
        for &t in &fi[..8] {
            assert!(
                (l[t] - f[t]).abs() < 1e-3,
                "row {i} tok {t}: {} vs {}",
                l[t],
                f[t]
            );
        }
        assert_eq!(argmax(&l), argmax(f));
    }
    // drafting mode (no certificate): the default rank/shortlist keeps the
    // full head's argmax on ≥ 95% of drafter states at < 35% of its MACs
    let (r, k) = shortlist_params(&cfg);
    let mut sh = ShortlistHead::new(&m.head, r, k, 8, cfg.rms_eps);
    sh.certify = false;
    let hits = (0..x.rows)
        .filter(|&i| argmax(&sh.logits_row(x.row(i))) == argmax(full.row(i)))
        .count();
    let frac = sh.mean_macs() / sh.full_macs();
    eprintln!(
        "shortlist head: certified mode evaluates {:.0} rows/row (exact), drafting mode argmax recall {hits}/{} at {:.1}% of the full head",
        exact.exact_rows as f64 / exact.calls as f64,
        x.rows,
        100.0 * frac
    );
    assert!(hits * 100 >= 95 * x.rows, "argmax recall {hits}/{}", x.rows);
    assert!(frac < 0.35, "shortlist head costs {frac} of the full head");
}

#[test]
fn klt_codec_is_bit_exact_hard_capped_and_beats_v21_at_equal_bytes() {
    let _guard = exclusive();
    use tesseract_fabric::compression::klt::{klt_calibration, KltResidualCodec};
    use tesseract_fabric::compression::EntropyResidualCodec;
    use tesseract_fabric::model::ReferenceModel;
    let cfg = ModelConfig::tiny();
    let m = ReferenceModel::new(&cfg);
    let boundary = 3;
    let cal = klt_calibration(&cfg, boundary);
    assert!(cal.pred.is_some(), "conditional calibration expected");
    let mut klt = KltResidualCodec::new(cal.clone(), 0.25);
    let mut v21 = EntropyResidualCodec::new(cfg.d_model, 0.25, 7);
    let lo = ShardModel::generate(&cfg, 0..boundary, None);
    // held-out text: prompts from a seed the calibration never used
    let mut rng = Rng::new(0x7E57);
    let (mut e_klt, mut e_v21, mut energy) = (0.0f64, 0.0f64, 0.0f64);
    for _ in 0..4 {
        let toks: Vec<u32> = (0..96).map(|_| rng.below(cfg.vocab) as u32).collect();
        let meta = BatchMeta::causal_prefill(0, toks.len(), 0);
        let mut kv = LocalKvCache::new(cfg.n_heads, cfg.n_kv_heads, cfg.d_head);
        let mut x = m.head.embed_tokens(&toks);
        lo.forward(&mut x, &meta, &mut kv, &mut NoObserver);
        let base = m.head.embed_tokens(&toks);
        let prevt: Vec<u32> = (0..toks.len()).map(|t| toks[t.saturating_sub(1)]).collect();
        let prev = m.head.embed_tokens(&prevt);
        let (wire, rec) = klt.encode(&x, &base, Some(&prev));
        let dec = klt.decode(&wire, &base, Some(&prev));
        assert_eq!(
            dec.data, rec.data,
            "receiver must rebuild the sender's reconstruction"
        );
        // every row within its hard cap (payload after the 8-byte frame header)
        let mut r = tesseract_fabric::wire::protocol::ByteReader::new(&wire[8..]);
        for _ in 0..x.rows {
            let _j = r.u8();
            let nb = r.u16() as usize;
            assert!(3 + nb <= klt.row_bytes_cap(), "row of {} B > cap", 3 + nb);
            let _ = r.bytes(nb);
        }
        let (_, rec21) = v21.encode(&x, &base);
        for i in 0..x.rows {
            for j in 0..x.cols {
                let t = x.row(i)[j] - base.row(i)[j];
                energy += (t as f64).powi(2);
                e_klt += ((x.row(i)[j] - rec.row(i)[j]) as f64).powi(2);
                e_v21 += ((x.row(i)[j] - rec21.row(i)[j]) as f64).powi(2);
            }
        }
    }
    let (rk, rv) = ((e_klt / energy).sqrt(), (e_v21 / energy).sqrt());
    eprintln!(
        "HRQ-V5 conditional KLT: relative error {rk:.4} vs V2.1 {rv:.4} at the same 25% cap (predictor explains {:.0}% held out)",
        100.0 * cal.explained
    );
    assert!(rk < 0.75 * rv, "KLT {rk} not clearly better than V2.1 {rv}");
}

#[test]
fn paired_kv_placement_contacts_one_remote_shard_within_8e5() {
    use tesseract_fabric::kv_fabric::disaggregated_attention::DisaggregatedKVFabric;
    use tesseract_fabric::kv_fabric::page_table::pair_partners;
    use tesseract_fabric::kv_fabric::{KvPageTable, PlacementPolicy};
    let (nh, nkv, dh) = (8, 2, 64);
    let kd = nkv * dh;
    let caps = [
        (0u16, 400usize << 20),
        (1, 200 << 20),
        (2, 150 << 20),
        (3, 100 << 20),
    ];
    for ctx in [4096usize, 16384] {
        let mut rng = Rng::new(ctx as u64 + 3);
        let k: Vec<f32> = (0..ctx * kd).map(|_| rng.normal()).collect();
        let v: Vec<f32> = (0..ctx * kd).map(|_| rng.normal()).collect();
        let q: Vec<f32> = (0..nh * dh).map(|_| rng.normal()).collect();
        let mk = |policy: PlacementPolicy| {
            let mut fab =
                DisaggregatedKVFabric::new(KvPageTable::new(256, &caps, policy), nh, nkv, dh, 0);
            fab.precision = WirePrecision::Bfp;
            for t in 0..ctx {
                fab.append(0, t, &k[t * kd..(t + 1) * kd], &v[t * kd..(t + 1) * kd])
                    .unwrap();
            }
            fab
        };
        let owners = vec![(0..1, 0u16)];
        let mut paired = mk(PlacementPolicy::Paired {
            partners: pair_partners(&owners, &caps, 1),
            owners,
            local_pages: 12,
            remote_bias: 0,
        });
        let holders = paired.table.nodes_for_layer(0);
        assert_eq!(
            holders.len(),
            2,
            "paired layer must live on 2 nodes: {holders:?}"
        );
        let pages = paired.table.ranges_for_layer(0);
        let (a, b) = (pages[0].1.len() as i64, pages[1].1.len() as i64);
        let total = (ctx / 256) as i64;
        if total >= 24 {
            // past twice tier 0 the two holders split the layer evenly
            assert!((a - b).abs() <= 1, "pages not split evenly: {a} vs {b}");
        } else {
            // tier 0 (12 pages) stays with the computing node
            assert_eq!(a, 12, "tier 0 must stay local");
            assert_eq!(a + b, total);
        }
        let (out, st) = paired.attend_and_merge(0, &q, 1, &[usize::MAX]);
        let mut striped = mk(PlacementPolicy::ProportionalFreeRam);
        let (_, st4) = striped.attend_and_merge(0, &q, 1, &[usize::MAX]);
        let mut err = 0.0f32;
        for h in 0..nh {
            let g = h / (nh / nkv);
            let kh: Vec<f32> = (0..ctx)
                .flat_map(|t| k[t * kd + g * dh..t * kd + (g + 1) * dh].to_vec())
                .collect();
            let vh: Vec<f32> = (0..ctx)
                .flat_map(|t| v[t * kd + g * dh..t * kd + (g + 1) * dh].to_vec())
                .collect();
            let mut r = vec![0.0; dh];
            reference_attention(&q[h * dh..(h + 1) * dh], &kh, &vh, dh, &|_| true, &mut r);
            for d in 0..dh {
                err = err.max((r[d] - out[h * dh + d]).abs());
            }
        }
        eprintln!(
            "paired KV T={ctx}: {} B per layer exchange (striped over 4: {} B), max err {err:.2e}",
            st.network_bytes, st4.network_bytes
        );
        assert!(err < 8e-5, "paired error {err}");
        assert!(st.network_bytes * 2 < st4.network_bytes);
        // 8 layers of exchanges stay far below 34 KB per decode step
        assert!(8 * st.network_bytes < 24 * 1024);
    }
}

// ---------------------------------------------------------------- V6

#[test]
fn grouped_multi_query_kernel_matches_per_query_reference() {
    use tesseract_fabric::kv_fabric::disaggregated_attention::KvShard;
    use tesseract_fabric::kv_fabric::partial_strided;
    let (nh, nkv, dh) = (8, 2, 64);
    let kd = nkv * dh;
    let ctx = 1000; // 4 pages of 256, last one partial
    let mut rng = Rng::new(0x6A);
    let k: Vec<f32> = (0..ctx * kd).map(|_| rng.normal()).collect();
    let v: Vec<f32> = (0..ctx * kd).map(|_| rng.normal()).collect();
    let mut shard = KvShard::new(0, nh, nkv, dh, 256);
    shard.append_block(0, 0, &k, &v);
    // 3 rows with different causal limits (two share one), 8 heads each
    let rows = 3;
    let q: Vec<f32> = (0..rows * nh * dh).map(|_| rng.normal()).collect();
    let pos = [999usize, 300, 999];
    let got = shard.partials(0, &q, rows, &pos);
    let mut scratch = Vec::new();
    let mut worst = 0.0f32;
    for i in 0..rows {
        for h in 0..nh {
            let off = (h / (nh / nkv)) * dh;
            let want = partial_strided(
                &q[(i * nh + h) * dh..(i * nh + h + 1) * dh],
                &k,
                &v,
                kd,
                off,
                dh,
                0..pos[i] + 1,
                &mut scratch,
            );
            let g = &got[i * nh + h];
            // compare the normalised outputs and the log-mass m + ln l
            let lm = |p: &tesseract_fabric::kv_fabric::AttnPartial| p.m + p.l.ln();
            assert!((lm(g) - lm(&want)).abs() < 1e-4, "row {i} head {h} mass");
            for (a, b) in g.o.iter().zip(&want.o) {
                worst = worst.max((a - b).abs());
            }
        }
    }
    eprintln!("grouped kernel vs per-query reference: max |Δo| {worst:.2e}");
    assert!(worst < 1e-5, "grouped kernel deviates: {worst}");
}

#[test]
fn lane_parallel_moe_is_exact_for_one_row_and_trees() {
    use tesseract_fabric::model::{
        moe_combine, moe_combine_lanes, route_topk, FfnWeights, LayerWeights,
    };
    use tesseract_fabric::tensor::pool::LanePool;
    let cfg = ModelConfig::tiny();
    let layer = LayerWeights::generate(&cfg, 1, None);
    let FfnWeights::Moe {
        router,
        bank,
        top_k,
    } = &layer.ffn
    else {
        panic!("tiny model is MoE");
    };
    let mut rng = Rng::new(0x1A4E);
    for lanes in [1usize, 2, 3, 4] {
        let pool = LanePool::new(lanes);
        for rows in [1usize, 2, 4, 7] {
            let x = Tensor2D::randn(rows, cfg.d_model, 1.0, &mut rng);
            let r = route_topk(router, &x, *top_k);
            let a = moe_combine(bank, &x, &r);
            let b = moe_combine_lanes(bank, &x, &r, &pool);
            let d = a
                .data
                .iter()
                .zip(&b.data)
                .fold(0.0f32, |m, (p, q)| m.max((p - q).abs()));
            assert!(d < 1e-5, "lanes {lanes} rows {rows}: {d}");
        }
    }
}

#[test]
fn paired_placement_keeps_the_partner_bias_pages_behind() {
    use tesseract_fabric::kv_fabric::page_table::pair_partners;
    use tesseract_fabric::kv_fabric::{KvPageTable, PlacementPolicy};
    let caps = [
        (0u16, 1usize << 30),
        (1, 1 << 30),
        (2, 1 << 30),
        (3, 1 << 30),
    ];
    let owners = vec![(0..1, 0u16)];
    for bias in [0usize, 3, 5] {
        let mut t = KvPageTable::new(
            256,
            &caps,
            PlacementPolicy::Paired {
                partners: pair_partners(&owners, &caps, 1),
                owners: owners.clone(),
                local_pages: 4,
                remote_bias: bias,
            },
        );
        for p in 0..40 {
            t.locate_or_allocate(0, p, 1024).unwrap();
        }
        let r = t.ranges_for_layer(0);
        assert_eq!(r.len(), 2);
        let local = r.iter().find(|x| x.0 == 0).unwrap().1.len() as i64;
        let remote = 40 - local;
        assert!(
            (local - remote - bias as i64).abs() <= 1,
            "bias {bias}: local {local} remote {remote}"
        );
    }
}

#[test]
fn roofline_uses_measured_acceptance_when_large_trees_saturate() {
    use tesseract_fabric::speculative::RooflineController;
    // the drafter's path probabilities promise ever more tokens for bigger
    // trees, but measured acceptance saturates at ~2: cost grows with N, so
    // the controller must settle on a small tree
    let q: Vec<f32> = (0..16).map(|i| 0.95f32.powi(i)).collect();
    let cost = |n: usize| 50_000_000u64 + 12_000_000 * (n as u64 - 1);
    let acc = |n: usize| if n == 1 { 1 } else { 2 };
    let mut c = RooflineController::new(vec![1, 2, 3, 4, 6, 8, 16]);
    let mut picks = Vec::new();
    for _ in 0..120 {
        let n = c.choose();
        picks.push(n);
        c.record(n, &q, acc(n), cost(n), cost(n));
    }
    let small = picks[60..].iter().filter(|&&n| n == 2).count();
    assert!(small >= 45, "N = 2 chosen {small}/60: {picks:?}");
}

// ---------------------------------------------------------------- V6.2

#[test]
fn head_group_striped_kv_matches_monolithic_and_fits_the_byte_budget() {
    use tesseract_fabric::kv_fabric::disaggregated_attention::DisaggregatedKVFabric;
    use tesseract_fabric::kv_fabric::page_table::group_holders;
    use tesseract_fabric::kv_fabric::{KvPageTable, PlacementPolicy};
    let (nh, nkv, dh) = (8usize, 2usize, 64usize);
    let (kd, gh) = (nkv * dh, nh / nkv);
    let caps = [
        (0u16, 400usize << 20),
        (1, 200 << 20),
        (2, 150 << 20),
        (3, 100 << 20),
    ];
    for ctx in [8192usize, 16384] {
        let mut rng = Rng::new(ctx as u64 + 62);
        let k: Vec<f32> = (0..ctx * kd).map(|_| rng.normal()).collect();
        let v: Vec<f32> = (0..ctx * kd).map(|_| rng.normal()).collect();
        let q: Vec<f32> = (0..nh * dh).map(|_| rng.normal()).collect();
        for prec in [WirePrecision::F32, WirePrecision::Bfp] {
            let policy = PlacementPolicy::Striped {
                holders: group_holders(&[(0..1, 0u16)], &caps, 1, nkv, 0),
                owners: vec![(0..nkv, 0u16)],
                local_pages: 12,
                mirror: false,
            };
            let mut fab =
                DisaggregatedKVFabric::new(KvPageTable::new(256, &caps, policy), gh, 1, dh, 0);
            fab.precision = prec;
            for t in 0..ctx {
                for g in 0..nkv {
                    let o = t * kd + g * dh;
                    fab.append(g, t, &k[o..o + dh], &v[o..o + dh]).unwrap();
                }
            }
            // every node scans part of the layer's KV
            let mut holders = std::collections::BTreeSet::new();
            for g in 0..nkv {
                holders.extend(fab.table.nodes_for_layer(g));
            }
            assert_eq!(holders.len(), 4, "all 4 nodes hold KV: {holders:?}");
            let mut out = vec![0.0f32; nh * dh];
            let mut bytes = 0usize;
            for g in 0..nkv {
                let (o, st) =
                    fab.attend_and_merge(g, &q[g * gh * dh..(g + 1) * gh * dh], 1, &[usize::MAX]);
                out[g * gh * dh..(g + 1) * gh * dh].copy_from_slice(&o);
                bytes += st.network_bytes;
            }
            let mut err = 0.0f32;
            for h in 0..nh {
                let g = h / gh;
                let kh: Vec<f32> = (0..ctx)
                    .flat_map(|t| k[t * kd + g * dh..t * kd + (g + 1) * dh].to_vec())
                    .collect();
                let vh: Vec<f32> = (0..ctx)
                    .flat_map(|t| v[t * kd + g * dh..t * kd + (g + 1) * dh].to_vec())
                    .collect();
                let mut r = vec![0.0; dh];
                reference_attention(&q[h * dh..(h + 1) * dh], &kh, &vh, dh, &|_| true, &mut r);
                for d in 0..dh {
                    err = err.max((r[d] - out[h * dh + d]).abs());
                }
            }
            eprintln!("head-group striped T={ctx} {prec:?}: {bytes} B per layer exchange, max err {err:.2e}");
            match prec {
                WirePrecision::F32 => assert!(err < 1e-5, "f32 wire error {err}"),
                _ => {
                    assert!(err < 8e-5, "BFP wire error {err}");
                    // 8 layers of exchanges leave room for the K/V appends
                    assert!(8 * bytes < 32 * 1024, "{bytes} B per layer");
                }
            }
        }
    }
}

// ---------------------------------------------------------------- V6.3

#[test]
fn mirrored_prefix_delegation_spreads_8k_attention_over_all_four_nodes() {
    use tesseract_fabric::kv_fabric::page_table::group_holders;
    use tesseract_fabric::kv_fabric::{KvPageTable, PlacementPolicy};
    // Scenario 3 shape: one layer on node 0, 2 KV-head groups, 4 nodes,
    // 8K context = 32 pages of 256 tokens per group, 12-page tier 0
    let caps = [
        (0u16, 400usize << 20),
        (1, 200 << 20),
        (2, 150 << 20),
        (3, 100 << 20),
    ];
    let scan_units = |mirror: bool| {
        let mut t = KvPageTable::new(
            256,
            &caps,
            PlacementPolicy::Striped {
                holders: group_holders(&[(0..1, 0u16)], &caps, 1, 2, 0),
                owners: vec![(0..2, 0u16)],
                local_pages: 12,
                mirror,
            },
        );
        for p in 0..32 {
            for vl in 0..2 {
                t.locate_or_allocate(vl, p, 256 * 64 * 8).unwrap();
            }
        }
        let committed = 32 * 256;
        let mut units = std::collections::BTreeMap::<u16, usize>::new();
        for vl in 0..2 {
            for p in 0..32 {
                let node = match t.mirror_of(vl, p) {
                    Some(m) if t.delegated(committed) => m,
                    _ => t.lookup(vl, p).unwrap(),
                };
                *units.entry(node).or_default() += 1;
            }
        }
        // short contexts never delegate: 2K stays fully local
        assert!(!t.delegated(8 * 256));
        units
    };
    let without = scan_units(false);
    let with = scan_units(true);
    eprintln!("8K page-units scanned per node: no mirror {without:?}, mirrored prefix {with:?}");
    assert_eq!(
        without[&0], 28,
        "V6.2 layout: computing node scans 28 of 64"
    );
    assert!(
        with[&0] <= 16,
        "computing node still scans {} of 64",
        with[&0]
    );
    assert!(with.values().all(|&u| u >= 12), "uneven split {with:?}");
    assert_eq!(
        with.values().sum::<usize>(),
        64,
        "every page scanned exactly once"
    );
}

// ---------------------------------------------------------------- V6.8

#[test]
fn projected_step_bytes_match_the_live_scenario_3_measurements() {
    use tesseract_fabric::kv_fabric::page_table::project_step_bytes;
    use tesseract_fabric::model::ModelConfig;
    // Scenario 3: 8 layers split [4, 2, 1, 1] over 4 nodes; free KV RAM after
    // weights (MB) as the benchmark computes it
    let cfg = ModelConfig::long_context();
    let ram_mb = [512.0f64, 256.0, 192.0, 128.0];
    let counts = [4usize, 2, 1, 1];
    let layer_mb = cfg.layer_params() as f64 * 4.0 / 1e6;
    let caps: Vec<(u16, usize)> = (0..4)
        .map(|i| {
            (
                i as u16,
                ((ram_mb[i] - counts[i] as f64 * layer_mb) * 1e6) as usize,
            )
        })
        .collect();
    let owners = vec![(0..4, 0u16), (4..6, 1), (6..7, 2), (7..8, 3)];
    let kb = |bias: usize, ctx: usize| {
        project_step_bytes(
            &owners,
            &caps,
            cfg.n_layers,
            cfg.n_heads,
            cfg.n_kv_heads,
            cfg.d_head,
            256,
            12,
            bias,
            ctx,
        ) as f64
            / 1024.0
    };
    for bias in [0usize, 5, 6, 7, 13] {
        eprintln!(
            "bias {bias}: projected {:.1} / {:.1} / {:.1} KB per step at 8K / 32K / 64K",
            kb(bias, 8192),
            kb(bias, 32768),
            kb(bias, 65536)
        );
    }
    // live V6.7 run 1 (bias 13): 32.1 KB at 8K, 34.4 KB at 32K and 64K;
    // the projection sizes partials at BFP16, an upper bound on the wire
    assert!(
        kb(13, 8192) < kb(13, 32768),
        "bias 13 must cost more at 32K"
    );
    assert!(kb(13, 32768) >= 34.0);
    assert!(kb(6, 32768) < 34.0 && kb(6, 65536) < 34.0);
}
