//! Edge micro-drafter: suffix n-gram table + 2-layer low-rank neural draft
//! head (d_model → r → V), producing best-first speculative token trees.
//!
//! The neural head is warm-started from the target model's own output
//! projection — A spans the top-r right-singular subspace of
//! `lm_head · diag(final_norm)` (found by subspace iteration) and B = W·Aᵀ — so
//! before any training it is the target's "zero-layer" path restricted to r
//! dimensions. It is then distilled online from the target distributions the
//! verifier computes for every tree node (sparse top-k targets).

use super::tree_braid::{ProposalKind, SpeculativeTree};
use crate::kv_fabric::{node_key, BatchMeta, LocalKvCache, NodeKey};
use crate::model::{HeadWeights, ModelConfig, NoObserver, ShardModel};
use crate::tensor::{dot, rms_norm, softmax_inplace, Rng, Tensor2D};
use std::collections::{BinaryHeap, HashMap};

/// Sparse distribution: (token, probability), descending.
pub type SparseDist = Vec<(u32, f32)>;

/// Suffix n-gram statistics over the committed token stream.
#[derive(Clone, Debug, Default)]
pub struct NgramTable {
    pub max_n: usize,
    /// context (up to max_n−1 tokens, packed) → next-token counts
    table: HashMap<(u8, u64), HashMap<u32, u32>>,
}

fn pack_ctx(ctx: &[u32]) -> u64 {
    ctx.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &t| {
        (h ^ t as u64).wrapping_mul(0x0100_0000_01b3)
    })
}

impl NgramTable {
    pub fn new(max_n: usize) -> Self {
        NgramTable {
            max_n,
            table: HashMap::new(),
        }
    }

    /// Observe that `next` followed `history` (history = all tokens before it).
    pub fn observe(&mut self, history: &[u32], next: u32) {
        for n in 1..self.max_n {
            if history.len() < n {
                break;
            }
            let ctx = &history[history.len() - n..];
            *self
                .table
                .entry((n as u8, pack_ctx(ctx)))
                .or_default()
                .entry(next)
                .or_insert(0) += 1;
        }
    }

    pub fn observe_sequence(&mut self, seq: &[u32]) {
        for i in 1..seq.len() {
            self.observe(&seq[..i], seq[i]);
        }
    }

    /// Longest-suffix-match distribution: (sparse probs, total count).
    pub fn predict(&self, history: &[u32]) -> Option<(Vec<(u32, f32)>, u32)> {
        for n in (1..self.max_n).rev() {
            if history.len() < n {
                continue;
            }
            if let Some(c) = self
                .table
                .get(&(n as u8, pack_ctx(&history[history.len() - n..])))
            {
                let total: u32 = c.values().sum();
                let mut v: Vec<(u32, f32)> = c
                    .iter()
                    .map(|(&t, &k)| (t, k as f32 / total as f32))
                    .collect();
                v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
                return Some((v, total));
            }
        }
        None
    }
}

/// logits = B · (A · x),  A: r×d, B: V×r.
#[derive(Clone, Debug)]
pub struct LowRankDraftHead {
    pub a: Tensor2D,
    pub b: Tensor2D,
    pub norm_w: Vec<f32>,
    pub anchor_mix: f32,
    pub lr: f32,
    pub updates: u64,
}

/// LM heads up to this many elements are factorised exactly (4 Mi: the
/// synthetic models are ≤ 1 Mi).
pub const EXACT_FACTOR_ELEMS: usize = 1 << 22;

/// Vocabulary size above which the exact two-stage shortlist head is not
/// built (its dense copy and residual norms scale with V·d).
pub const SHORTLIST_MAX_VOCAB: usize = 16384;

impl LowRankDraftHead {
    /// Warm start from any (possibly quantized) LM head. Heads up to
    /// `EXACT_FACTOR_ELEMS` elements (every synthetic benchmark model) use
    /// the exact factorisation of [`LowRankDraftHead::from_target`]; larger
    /// real-checkpoint heads (e.g. 128K × 2048) use a randomized range
    /// finder on a strided sample of rows, rank ≤ 64, so start-up stays in
    /// seconds — the drafter only proposes, verification stays exact.
    pub fn from_linear(
        lm_head: &crate::tensor::Linear,
        final_norm: &[f32],
        rank: usize,
        seed: u64,
    ) -> Self {
        let (v, d) = (lm_head.out_features(), lm_head.in_features());
        if v * d <= EXACT_FACTOR_ELEMS {
            return Self::from_target(&lm_head.dense(), final_norm, rank, seed);
        }
        let rank = rank.min(64).min(d);
        let row = |r: usize, out: &mut [f32]| {
            match &lm_head.q {
                Some(q) => q.decode_row_into(r, out),
                None => out.copy_from_slice(lm_head.w.row(r)),
            }
            for (x, g) in out.iter_mut().zip(final_norm) {
                *x *= g;
            }
        };
        let stride = (v / 8192).max(1);
        let sample: Vec<usize> = (0..v).step_by(stride).collect();
        let mut ws = Tensor2D::zeros(sample.len(), d);
        for (i, &r) in sample.iter().enumerate() {
            row(r, ws.row_mut(i));
        }
        // subspace iteration on WᵀW without forming it: Q ← orth(Wᵀ(W Q))
        let mut rng = Rng::new(seed);
        let mut q: Vec<Vec<f32>> = (0..rank)
            .map(|_| (0..d).map(|_| rng.normal()).collect())
            .collect();
        for _ in 0..4 {
            let mut z: Vec<Vec<f32>> = q
                .iter()
                .map(|col| {
                    let wq: Vec<f32> = (0..ws.rows).map(|i| dot(ws.row(i), col)).collect();
                    let mut out = vec![0.0f32; d];
                    for (i, &c) in wq.iter().enumerate() {
                        crate::tensor::axpy(c, ws.row(i), &mut out);
                    }
                    out
                })
                .collect();
            for k in 0..rank {
                for p in 0..k {
                    let proj = dot(&z[k], &z[p]);
                    let zp = z[p].clone();
                    crate::tensor::axpy(-proj, &zp, &mut z[k]);
                }
                let n = dot(&z[k], &z[k]).sqrt().max(1e-30);
                z[k].iter_mut().for_each(|x| *x /= n);
            }
            q = z;
        }
        let a = Tensor2D::from_vec(rank, d, q.into_iter().flatten().collect());
        let mut b = Tensor2D::zeros(v, rank);
        let mut buf = vec![0.0f32; d];
        for r in 0..v {
            row(r, &mut buf);
            for k in 0..rank {
                b.data[r * rank + k] = dot(&buf, a.row(k));
            }
        }
        LowRankDraftHead {
            a,
            b,
            norm_w: vec![1.0; d],
            anchor_mix: 0.0,
            lr: 0.02,
            updates: 0,
        }
    }

    /// Warm start from the target head: W = lm_head·diag(final_norm) (V×d).
    pub fn from_target(lm_head: &Tensor2D, final_norm: &[f32], rank: usize, seed: u64) -> Self {
        let (v, d) = (lm_head.rows, lm_head.cols);
        let mut w = lm_head.clone();
        for r in 0..v {
            for (x, g) in w.row_mut(r).iter_mut().zip(final_norm) {
                *x *= g;
            }
        }
        // Gram matrix G = WᵀW (d×d)
        let mut g = vec![0.0f64; d * d];
        for r in 0..v {
            let row = w.row(r);
            for i in 0..d {
                let ri = row[i] as f64;
                for j in i..d {
                    g[i * d + j] += ri * row[j] as f64;
                }
            }
        }
        for i in 0..d {
            for j in 0..i {
                g[i * d + j] = g[j * d + i];
            }
        }
        // subspace iteration for top-`rank` eigenvectors of G
        let mut rng = Rng::new(seed);
        let mut q: Vec<Vec<f64>> = (0..rank)
            .map(|_| (0..d).map(|_| rng.normal() as f64).collect())
            .collect();
        for _ in 0..12 {
            let mut z: Vec<Vec<f64>> = q
                .iter()
                .map(|col| {
                    (0..d)
                        .map(|i| (0..d).map(|j| g[i * d + j] * col[j]).sum())
                        .collect()
                })
                .collect();
            // modified Gram-Schmidt
            for k in 0..rank {
                for p in 0..k {
                    let proj: f64 = (0..d).map(|i| z[k][i] * z[p][i]).sum();
                    for i in 0..d {
                        z[k][i] -= proj * z[p][i];
                    }
                }
                let n: f64 = z[k].iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-30);
                for x in z[k].iter_mut() {
                    *x /= n;
                }
            }
            q = z;
        }
        let a = Tensor2D::from_vec(
            rank,
            d,
            q.iter().flat_map(|c| c.iter().map(|x| *x as f32)).collect(),
        );
        // B = W · Aᵀ
        let mut b = Tensor2D::zeros(v, rank);
        for r in 0..v {
            for k in 0..rank {
                b.data[r * rank + k] = dot(w.row(r), a.row(k));
            }
        }
        LowRankDraftHead {
            a,
            b,
            norm_w: vec![1.0; d],
            anchor_mix: 0.0,
            lr: 0.02,
            updates: 0,
        }
    }

    pub fn rank(&self) -> usize {
        self.a.rows
    }

    /// Input feature: rmsnorm(emb(token) + mix · h_anchor).
    pub fn feature(&self, emb: &[f32], h_anchor: &[f32], out: &mut [f32]) {
        let mut x: Vec<f32> = emb.to_vec();
        if !h_anchor.is_empty() {
            for (xi, hi) in x.iter_mut().zip(h_anchor) {
                *xi += self.anchor_mix * hi;
            }
        }
        rms_norm(&x, &self.norm_w, 1e-5, out);
    }

    pub fn logits(&self, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let z: Vec<f32> = (0..self.rank()).map(|k| dot(self.a.row(k), x)).collect();
        let l: Vec<f32> = (0..self.b.rows).map(|r| dot(self.b.row(r), &z)).collect();
        (z, l)
    }

    /// One SGD step of cross-entropy toward a sparse target distribution.
    /// Returns the KL-free cross-entropy before the step.
    pub fn distill_step(&mut self, x: &[f32], target: &[(u32, f32)]) -> f32 {
        let (z, mut q) = self.logits(x);
        softmax_inplace(&mut q);
        let tsum: f32 = target.iter().map(|t| t.1).sum::<f32>().max(1e-9);
        let ce: f32 = -target
            .iter()
            .map(|&(t, p)| p / tsum * q[t as usize].max(1e-9).ln())
            .sum::<f32>();
        // g = q − p
        let mut g = q;
        for &(t, p) in target {
            g[t as usize] -= p / tsum;
        }
        let r = self.rank();
        // dz = Bᵀ g
        let mut dz = vec![0.0f32; r];
        for (row, &gv) in g.iter().enumerate() {
            if gv.abs() < 1e-7 {
                continue;
            }
            let br = &mut self.b.data[row * r..(row + 1) * r];
            for k in 0..r {
                dz[k] += br[k] * gv;
                br[k] -= self.lr * gv * z[k];
            }
        }
        for k in 0..r {
            let ar = self.a.row_mut(k);
            for (j, xv) in x.iter().enumerate() {
                ar[j] -= self.lr * 0.1 * dz[k] * xv;
            }
        }
        self.updates += 1;
        ce
    }
}

#[derive(Clone, Debug)]
pub struct DraftConfig {
    /// Total nodes to draft (including the root).
    pub budget: usize,
    pub max_depth: usize,
    pub max_children: usize,
    pub temperature: f32,
    pub proposal: ProposalKind,
    /// Sparse top-k kept from each draft distribution.
    pub top_k: usize,
}

impl Default for DraftConfig {
    fn default() -> Self {
        DraftConfig {
            budget: 16,
            max_depth: 6,
            max_children: 4,
            temperature: 1.0,
            proposal: ProposalKind::Deterministic,
            top_k: 8,
        }
    }
}

/// N-gram + neural tree drafter.
pub struct TreeDrafter {
    pub cfg: DraftConfig,
    pub ngram: NgramTable,
    pub head: LowRankDraftHead,
    pub embed: crate::tensor::Embedding,
    pub history: Vec<u32>,
    pub h_anchor: Vec<f32>,
    rng: Rng,
    pub trees_drafted: u64,
    /// Self-speculative layer-skipping mode: when set, node distributions come
    /// from the target's own first k layers + final norm + LM head.
    pub early: Option<EarlyExitDrafter>,
    /// Prompt-lookup mode: longest n-gram (≤ this length) to match.
    pub lookup: Option<usize>,
}

/// Prompt-lookup draft: find the most recent earlier occurrence of the
/// context's last n tokens (n = `max_n` … 1) and propose the tokens that
/// followed it, up to `len` of them.
pub fn prompt_lookup(history: &[u32], max_n: usize, len: usize) -> Vec<u32> {
    let h = history.len();
    for n in (1..=max_n.min(h.saturating_sub(1))).rev() {
        let pat = &history[h - n..];
        // latest match that leaves room for a continuation
        for start in (0..h - n).rev() {
            if &history[start..start + n] == pat {
                let from = start + n;
                return history[from..(from + len).min(h)].to_vec();
            }
        }
    }
    Vec::new()
}

/// Default (rank, shortlist) of the drafter's two-stage LM head: rank 96 and
/// 96 rescored rows on the benchmark model (d = 512, V = 2048), which keeps
/// the full head's argmax on ≈ 98% of drafter states at 28% of its MACs.
pub fn shortlist_params(cfg: &ModelConfig) -> (usize, usize) {
    ((cfg.d_model / 2).min(96), (cfg.vocab / 4).min(96))
}

/// Two-stage LM head for drafting (exact when `certify` is set).
///
/// Stage A scores every vocabulary row in the top-`r` right-singular subspace
/// of W = lm_head·diag(final_norm): c_v = B_v·(A x̂), with x̂ = rmsnorm(x).
/// Because A has orthonormal rows, the exact score satisfies
/// |W_v·x̂ − c_v| ≤ ρ_v·‖x̂⊥‖ (Cauchy–Schwarz), where ρ_v = ‖W_v(I − AᵀA)‖ is
/// precomputed and ‖x̂⊥‖² = ‖x̂‖² − ‖A x̂‖². Stage B evaluates the exact
/// full-dimension rows of the best `shortlist` coarse candidates, then keeps
/// evaluating every row whose upper bound c_v + ρ_v‖x̂⊥‖ still exceeds the
/// current `exact_top`-th best exact score. On return the `exact_top` largest
/// logits are therefore *exactly* those of the full LM head (same tokens, same
/// values up to f32 summation order); every other entry holds its coarse
/// score, which is provably below them.
#[derive(Clone, Debug)]
pub struct ShortlistHead {
    w: Tensor2D,
    a: Tensor2D,
    b: Tensor2D,
    resid: Vec<f32>,
    pub shortlist: usize,
    pub exact_top: usize,
    /// Expand until the top-`exact_top` is certified (exact mode). When false,
    /// only the coarse shortlist is rescored exactly: cheaper, approximate.
    pub certify: bool,
    eps: f32,
    /// Rows scored, full-dimension rows evaluated, rows whose first
    /// shortlist was already certified (no expansion needed).
    pub calls: u64,
    pub exact_rows: u64,
    pub certified: u64,
}

impl ShortlistHead {
    pub fn new(
        head: &HeadWeights,
        rank: usize,
        shortlist: usize,
        exact_top: usize,
        eps: f32,
    ) -> Self {
        let mut w = head.lm_head.dense().into_owned();
        let lr = LowRankDraftHead::from_target(&w, &head.final_norm, rank, 0x5157);
        for r in 0..w.rows {
            for (x, g) in w.row_mut(r).iter_mut().zip(&head.final_norm) {
                *x *= g;
            }
        }
        let resid = (0..w.rows)
            .map(|v| {
                let row = w.row(v);
                let n2: f64 = row.iter().map(|x| (*x as f64).powi(2)).sum();
                let p2: f64 = lr.b.row(v).iter().map(|x| (*x as f64).powi(2)).sum();
                // a hair of slack covers f32 rounding in A, B and the dots
                ((n2 - p2).max(0.0).sqrt() * 1.0001 + 1e-6) as f32
            })
            .collect();
        ShortlistHead {
            w,
            a: lr.a,
            b: lr.b,
            resid,
            shortlist,
            exact_top,
            certify: true,
            eps,
            calls: 0,
            exact_rows: 0,
            certified: 0,
        }
    }

    /// Multiply–accumulates per scored row on average (for cost accounting).
    pub fn mean_macs(&self) -> f64 {
        let d = self.w.cols as f64;
        let fixed = (self.a.rows as f64) * d + (self.b.rows * self.b.cols) as f64;
        fixed + d * self.exact_rows as f64 / self.calls.max(1) as f64
    }

    pub fn full_macs(&self) -> f64 {
        (self.w.rows * self.w.cols) as f64
    }

    pub fn logits_row(&mut self, x: &[f32]) -> Vec<f32> {
        let d = x.len();
        let ones = vec![1.0f32; d];
        let mut xn = vec![0.0f32; d];
        rms_norm(x, &ones, self.eps, &mut xn);
        let z: Vec<f32> = (0..self.a.rows).map(|k| dot(self.a.row(k), &xn)).collect();
        let perp = (dot(&xn, &xn) - dot(&z, &z)).max(0.0).sqrt();
        let v = self.w.rows;
        let mut out: Vec<f32> = (0..v).map(|t| dot(self.b.row(t), &z)).collect();
        let mut exact = vec![false; v];
        let mut idx: Vec<u32> = (0..v as u32).collect();
        let k = self.shortlist.min(v);
        idx.select_nth_unstable_by(k - 1, |a, b| out[*b as usize].total_cmp(&out[*a as usize]));
        let mut batch: Vec<usize> = idx[..k].iter().map(|&t| t as usize).collect();
        let mut first = true;
        let m = self.exact_top.min(v).max(1);
        let mut top: Vec<f32> = Vec::new();
        loop {
            for &t in &batch {
                out[t] = dot(self.w.row(t), &xn);
                exact[t] = true;
                top.push(out[t]);
            }
            self.exact_rows += batch.len() as u64;
            top.sort_by(|a, b| b.total_cmp(a));
            top.truncate(m);
            let tau = if top.len() >= m {
                top[m - 1]
            } else {
                f32::NEG_INFINITY
            };
            batch = (0..v)
                .filter(|&t| !exact[t] && out[t] + self.resid[t] * perp > tau)
                .collect();
            if batch.is_empty() {
                break;
            }
            first = false;
            if !self.certify {
                break;
            }
        }
        self.calls += 1;
        self.certified += first as u64;
        out
    }
}

/// Self-speculative early-exit drafter: the target model truncated after its
/// first `k` layers, followed by the target's final RMSNorm and LM head. It
/// keeps its own KV cache for those k layers over the committed context and
/// drafts tree nodes with the same tree-masked attention as the verifier.
pub struct EarlyExitDrafter {
    pub layers: ShardModel,
    pub head: HeadWeights,
    /// Exact two-stage LM head (top-k identical to `head`), when enabled.
    pub short: Option<ShortlistHead>,
    pub kv: LocalKvCache,
    pub committed: usize,
    pub rows_forwarded: u64,
    eps: f32,
    next_epoch: u64,
}

impl EarlyExitDrafter {
    pub fn new(cfg: &ModelConfig, k: usize) -> Self {
        EarlyExitDrafter {
            layers: ShardModel::generate(cfg, 0..k.min(cfg.n_layers), None),
            head: HeadWeights::generate(cfg),
            short: None,
            kv: LocalKvCache::new(cfg.n_heads, cfg.n_kv_heads, cfg.d_head),
            committed: 0,
            rows_forwarded: 0,
            eps: cfg.rms_eps,
            next_epoch: 1 << 40,
        }
    }

    fn fresh_epoch(&mut self) -> u64 {
        self.next_epoch += 1;
        self.next_epoch
    }

    /// Use the exact two-stage shortlist LM head (rank `r` filter, `k`
    /// shortlisted rows, exact top-`top`).
    pub fn with_shortlist(mut self, r: usize, k: usize, top: usize) -> Self {
        // large real vocabularies draft with the full (quantized) LM head:
        // the shortlist's dense copy and residual norms would cost V·d
        if self.head.lm_head.out_features() <= SHORTLIST_MAX_VOCAB {
            self.short = Some(ShortlistHead::new(&self.head, r, k, top, self.eps));
        }
        self
    }

    /// Draft logits for every row of `x`.
    fn draft_logits(&mut self, x: &Tensor2D) -> Tensor2D {
        match self.short.as_mut() {
            Some(sh) => {
                let mut l = Tensor2D::zeros(x.rows, sh.w.rows);
                for i in 0..x.rows {
                    let r = sh.logits_row(x.row(i));
                    l.row_mut(i).copy_from_slice(&r);
                }
                l
            }
            None => self.head.logits(x, self.eps),
        }
    }

    /// Extend the committed context (causal batch through the k layers).
    pub fn commit(&mut self, tokens: &[u32]) {
        if tokens.is_empty() {
            return;
        }
        let e = self.fresh_epoch();
        let meta = BatchMeta::causal_prefill(self.committed, tokens.len(), e);
        let mut x = self.head.embed_tokens(tokens);
        self.layers
            .forward(&mut x, &meta, &mut self.kv, &mut NoObserver);
        self.kv.commit(&meta.keys, &|_| false);
        self.committed += tokens.len();
        self.rows_forwarded += tokens.len() as u64;
    }

    /// Forward one speculative row; it attends to the committed context and
    /// the pending rows in `visible` (its ancestors). Returns (key, logits).
    pub fn forward_node(
        &mut self,
        token: u32,
        pos: usize,
        mut visible: Vec<NodeKey>,
    ) -> (NodeKey, Vec<f32>) {
        let key = node_key(self.fresh_epoch(), 0);
        visible.push(key);
        let meta = BatchMeta {
            positions: vec![pos],
            keys: vec![key],
            visible: vec![visible],
        };
        let mut x = self.head.embed_tokens(&[token]);
        self.layers
            .forward(&mut x, &meta, &mut self.kv, &mut NoObserver);
        self.rows_forwarded += 1;
        (key, self.draft_logits(&x).data)
    }

    /// Forward several speculative rows in one batch (one tree level).
    pub fn forward_nodes(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
        visibles: Vec<Vec<NodeKey>>,
    ) -> Vec<(NodeKey, Vec<f32>)> {
        let keys: Vec<NodeKey> = tokens
            .iter()
            .map(|_| node_key(self.fresh_epoch(), 0))
            .collect();
        let visible: Vec<Vec<NodeKey>> = visibles
            .into_iter()
            .zip(&keys)
            .map(|(mut v, k)| {
                v.push(*k);
                v
            })
            .collect();
        let meta = BatchMeta {
            positions: positions.to_vec(),
            keys: keys.clone(),
            visible,
        };
        let mut x = self.head.embed_tokens(tokens);
        self.layers
            .forward(&mut x, &meta, &mut self.kv, &mut NoObserver);
        self.rows_forwarded += tokens.len() as u64;
        let l = self.draft_logits(&x);
        keys.into_iter()
            .enumerate()
            .map(|(i, k)| (k, l.row(i).to_vec()))
            .collect()
    }

    /// Forward a causal chain of rows (each sees all earlier rows of the chain)
    /// as one batch; returns (key, logits) per row. Rows stay pending.
    pub fn forward_chain(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
    ) -> Vec<(NodeKey, Vec<f32>)> {
        let keys: Vec<NodeKey> = tokens
            .iter()
            .map(|_| node_key(self.fresh_epoch(), 0))
            .collect();
        let visible: Vec<Vec<NodeKey>> = (0..tokens.len()).map(|i| keys[..=i].to_vec()).collect();
        let meta = BatchMeta {
            positions: positions.to_vec(),
            keys: keys.clone(),
            visible,
        };
        let mut x = self.head.embed_tokens(tokens);
        self.layers
            .forward(&mut x, &meta, &mut self.kv, &mut NoObserver);
        self.rows_forwarded += tokens.len() as u64;
        let l = self.draft_logits(&x);
        keys.into_iter()
            .enumerate()
            .map(|(i, k)| (k, l.row(i).to_vec()))
            .collect()
    }

    /// Drop every pending (speculative) row.
    pub fn clear_pending(&mut self) {
        self.kv.commit(&[], &|_| false);
    }
}

#[derive(PartialEq)]
struct Frontier {
    score: f32,
    node: usize,
    cand: usize,
}
impl Eq for Frontier {}
impl PartialOrd for Frontier {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Frontier {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.score
            .partial_cmp(&o.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(o.node.cmp(&self.node))
    }
}

impl TreeDrafter {
    pub fn new(
        cfg: DraftConfig,
        head: LowRankDraftHead,
        embed: crate::tensor::Embedding,
        seed: u64,
    ) -> Self {
        TreeDrafter {
            cfg,
            ngram: NgramTable::new(4),
            head,
            embed,
            history: Vec::new(),
            h_anchor: Vec::new(),
            rng: Rng::new(seed),
            trees_drafted: 0,
            early: None,
            lookup: None,
        }
    }

    /// Switch to self-speculative layer skipping (first `k` target layers).
    pub fn with_early_exit(mut self, model: &ModelConfig, k: usize) -> Self {
        self.early = Some(EarlyExitDrafter::new(model, k));
        self
    }

    /// Attention-only self-speculative drafter: the target's first `k` layers
    /// with their FFNs skipped, then the target's final norm + LM head. On the
    /// contextual V3 model 1 attention-only layer agrees with the full model
    /// ≈ 84% of the time at ≈ 1/25 of a model step per node.
    pub fn with_attention_exit(mut self, model: &ModelConfig, k: usize) -> Self {
        let mut e = EarlyExitDrafter::new(model, k);
        e.layers.skip_ffn = true;
        let (r, sl) = shortlist_params(model);
        let mut e = e.with_shortlist(r, sl, 8);
        if let Some(sh) = e.short.as_mut() {
            // drafting needs no certificate: verification is exact regardless
            sh.certify = false;
        }
        self.early = Some(e);
        self
    }

    /// Feed newly committed tokens (and optionally the anchor hidden state).
    pub fn commit(&mut self, tokens: &[u32], h_anchor: Option<&[f32]>) {
        for &t in tokens {
            if !self.history.is_empty() {
                self.ngram.observe(&self.history, t);
            }
            self.history.push(t);
        }
        if let Some(e) = self.early.as_mut() {
            e.commit(tokens);
        }
        if let Some(h) = h_anchor {
            self.h_anchor = h.to_vec();
        }
    }

    /// Distillation pair: (parent anchor hidden, token) → target sparse dist.
    /// Only the low-rank head is trained; the early-exit path *is* the target.
    pub fn train(&mut self, h_parent: &[f32], token: u32, target: &[(u32, f32)]) -> f32 {
        if self.early.is_some() {
            return 0.0;
        }
        let mut x = vec![0.0; self.embed.cols()];
        self.head
            .feature(&self.embed.row(token as usize), h_parent, &mut x);
        self.head.distill_step(&x, target)
    }

    /// Low-rank-head logits for the token following `token`.
    pub fn head_logits(&self, token: u32) -> Vec<f32> {
        let mut x = vec![0.0; self.embed.cols()];
        self.head
            .feature(&self.embed.row(token as usize), &self.h_anchor, &mut x);
        self.head.logits(&x).1
    }

    /// Mixed draft distribution for the token following `path` (sparse top-k).
    pub fn draft_dist(&self, history: &[u32], token: u32) -> Vec<(u32, f32)> {
        self.mix(history, self.head_logits(token))
    }

    /// Temperature + n-gram mixture + sparse top-k of a logit vector.
    pub fn mix(&self, history: &[u32], mut l: Vec<f32>) -> Vec<(u32, f32)> {
        if self.cfg.temperature > 0.0 {
            for v in l.iter_mut() {
                *v /= self.cfg.temperature;
            }
        }
        softmax_inplace(&mut l);
        let (lam, ng) = match self.ngram.predict(history) {
            Some((d, n)) => (n as f32 / (n as f32 + 2.0), d),
            None => (0.0, Vec::new()),
        };
        if lam > 0.0 {
            for v in l.iter_mut() {
                *v *= 1.0 - lam;
            }
            for (t, p) in ng {
                l[t as usize] += lam * p;
            }
        }
        let mut idx: Vec<u32> = (0..l.len() as u32).collect();
        let k = self.cfg.top_k.min(l.len());
        idx.select_nth_unstable_by(k - 1, |a, b| {
            l[*b as usize].partial_cmp(&l[*a as usize]).unwrap()
        });
        let mut top: Vec<(u32, f32)> = idx[..k].iter().map(|&t| (t, l[t as usize])).collect();
        top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
        let s: f32 = top.iter().map(|t| t.1).sum();
        top.iter_mut().for_each(|t| t.1 /= s);
        top
    }

    /// Forget the context (a new request): history, anchor and the early-exit
    /// drafter's KV. Learned n-gram statistics are kept.
    pub fn reset_context(&mut self) {
        self.history.clear();
        self.h_anchor.clear();
        if let Some(e) = self.early.as_mut() {
            e.kv.reset();
            e.committed = 0;
        }
    }

    /// Prompt-lookup drafting (no draft model): chains copied from the
    /// context. Exact verification keeps the output distribution unchanged.
    pub fn with_prompt_lookup(mut self, max_n: usize) -> Self {
        self.lookup = Some(max_n);
        self
    }

    fn lookup_tree(&mut self, epoch: u64, root: u32, budget: usize) -> SpeculativeTree {
        let mut h = self.history.clone();
        h.push(root);
        let chain = prompt_lookup(&h, self.lookup.unwrap_or(3), budget.saturating_sub(1));
        self.trees_drafted += 1;
        SpeculativeTree::chain(epoch, root, &chain)
    }

    /// Best-first tree of `budget` nodes rooted at `root` (the last verified
    /// token).
    pub fn draft(&mut self, epoch: u64, root: u32, budget: usize) -> SpeculativeTree {
        if self.lookup.is_some() {
            return self.lookup_tree(epoch, root, budget);
        }
        self.draft_with_prefix(epoch, &[], root, budget)
    }

    /// Best-first tree rooted at `root`, assuming `prefix` follows the
    /// committed history (draft-ahead for a predicted verification outcome):
    /// repeatedly expand the frontier candidate with the highest cumulative
    /// draft probability.
    pub fn draft_with_prefix(
        &mut self,
        epoch: u64,
        prefix: &[u32],
        root: u32,
        budget: usize,
    ) -> SpeculativeTree {
        let cfg = self.cfg.clone();
        let mut t = SpeculativeTree::single(epoch, root);
        t.proposal = cfg.proposal;
        // early-exit: the assumed prefix becomes a pending causal chain
        let mut prefix_keys: Vec<NodeKey> = Vec::new();
        let base_pos = self.history.len();
        if let Some(e) = self.early.as_mut() {
            for (i, &tok) in prefix.iter().enumerate() {
                let (k, _) = e.forward_node(tok, base_pos + i, prefix_keys.clone());
                prefix_keys.push(k);
            }
        }
        let root_pos = base_pos + prefix.len();
        let mut hist0 = self.history.clone();
        hist0.extend_from_slice(prefix);
        hist0.push(root);
        let mut paths: Vec<Vec<u32>> = vec![hist0];
        let mut keys: Vec<Vec<NodeKey>> = Vec::new(); // visible chain (incl. self) per node
        let mut heap = BinaryHeap::new();
        let mut cands: Vec<SparseDist> = Vec::new();
        let (c0, q0, k0) =
            self.node_candidates(&paths[0], root, root_pos, prefix_keys.clone(), &cfg);
        t.child_q[0] = q0;
        keys.push(k0);
        cands.push(c0);
        if !cands[0].is_empty() {
            heap.push(Frontier {
                score: cands[0][0].1,
                node: 0,
                cand: 0,
            });
        }
        while t.len() < budget {
            let Some(f) = heap.pop() else { break };
            let (tok, q) = cands[f.node][f.cand];
            let parent_q = t.path_q[f.node];
            let id = t.len();
            t.tokens.push(tok);
            t.parents.push(f.node as i32);
            t.depth.push(t.depth[f.node] + 1);
            t.path_q.push(parent_q * q);
            let mut p = paths[f.node].clone();
            p.push(tok);
            // sibling continues
            if f.cand + 1 < cands[f.node].len() && f.cand + 1 < cfg.max_children {
                heap.push(Frontier {
                    score: parent_q * cands[f.node][f.cand + 1].1,
                    node: f.node,
                    cand: f.cand + 1,
                });
            }
            if t.depth[id] < cfg.max_depth {
                let (c, q_full, k) = self.node_candidates(
                    &p,
                    tok,
                    root_pos + t.depth[id],
                    keys[f.node].clone(),
                    &cfg,
                );
                t.child_q.push(q_full);
                if !c.is_empty() {
                    heap.push(Frontier {
                        score: parent_q * q * c[0].1,
                        node: id,
                        cand: 0,
                    });
                }
                cands.push(c);
                keys.push(k);
            } else {
                t.child_q.push(Vec::new());
                cands.push(Vec::new());
                keys.push(Vec::new());
            }
            paths.push(p);
        }
        if let Some(e) = self.early.as_mut() {
            e.clear_pending();
        }
        self.trees_drafted += 1;
        t
    }

    /// Level-synchronous drafting: all frontier nodes of one depth are
    /// forwarded through the early-exit layers as a single batch, then the
    /// globally best (cumulative draft probability) candidates become the next
    /// level. Same tree semantics as best-first, ≈depth batched passes instead
    /// of one pass per node. Requires an early-exit drafter.
    pub fn draft_bfs(&mut self, epoch: u64, root: u32, budget: usize) -> SpeculativeTree {
        self.commit_and_draft(&[], epoch, root, budget)
    }

    /// Commit `tokens` (the previous epoch's accepted path) and draft the next
    /// tree; the committed rows and the root share the first batched pass.
    pub fn commit_and_draft(
        &mut self,
        tokens: &[u32],
        epoch: u64,
        root: u32,
        budget: usize,
    ) -> SpeculativeTree {
        for &t in tokens {
            if !self.history.is_empty() {
                self.ngram.observe(&self.history, t);
            }
            self.history.push(t);
        }
        if self.lookup.is_some() {
            return self.lookup_tree(epoch, root, budget);
        }
        let cfg = self.cfg.clone();
        let base = self.history.len();
        let e = self
            .early
            .as_mut()
            .expect("draft_bfs needs an early-exit drafter");
        let c0 = e.committed;
        let mut toks = tokens.to_vec();
        toks.push(root);
        let pos: Vec<usize> = (0..toks.len()).map(|i| c0 + i).collect();
        let outs = e.forward_chain(&toks, &pos);
        let n_c = tokens.len();
        let commit_keys: Vec<NodeKey> = outs[..n_c].iter().map(|o| o.0).collect();
        let root_key = outs[n_c].0;
        e.kv.commit(&commit_keys, &|k| k == root_key);
        e.committed += n_c;
        let (root_key, root_logits) = outs[n_c].clone();
        let tree = self.grow_tree(
            SpeculativeTree::single(epoch, root),
            base,
            root_key,
            root_logits,
            budget,
            &cfg,
        );
        if let Some(e) = self.early.as_mut() {
            e.clear_pending();
        }
        self.trees_drafted += 1;
        tree
    }

    /// Grow a tree level by level from a root whose early-exit row and logits
    /// are already computed.
    fn grow_tree(
        &mut self,
        mut t: SpeculativeTree,
        base: usize,
        root_key: NodeKey,
        root_logits: Vec<f32>,
        budget: usize,
        cfg: &DraftConfig,
    ) -> SpeculativeTree {
        let root = t.tokens[0];
        let mut hist: Vec<Vec<u32>> = vec![{
            let mut h = self.history.clone();
            h.push(root);
            h
        }];
        let mut chain: Vec<Vec<NodeKey>> = vec![Vec::new()];
        let mut frontier = vec![0usize];
        let mut first = Some(vec![(root_key, root_logits)]);
        while !frontier.is_empty() {
            let outs = match first.take() {
                Some(o) => o,
                None => {
                    let toks: Vec<u32> = frontier.iter().map(|&i| t.tokens[i]).collect();
                    let pos: Vec<usize> = frontier.iter().map(|&i| base + t.depth[i]).collect();
                    let vis: Vec<Vec<NodeKey>> =
                        frontier.iter().map(|&i| chain[i].clone()).collect();
                    self.early.as_mut().unwrap().forward_nodes(&toks, &pos, vis)
                }
            };
            let mut cands: Vec<(f32, usize, u32)> = Vec::new();
            for (fi, &node) in frontier.iter().enumerate() {
                let (key, logits) = &outs[fi];
                chain[node].push(*key);
                let q = self.mix(&hist[node], logits.clone());
                if t.depth[node] < cfg.max_depth {
                    for &(tok, p) in q.iter().take(cfg.max_children) {
                        cands.push((t.path_q[node] * p, node, tok));
                    }
                }
                t.child_q[node] = q;
            }
            if t.len() >= budget {
                break;
            }
            cands.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            let mut next = Vec::new();
            for (score, parent, tok) in cands.into_iter().take(budget - t.len()) {
                let id = t.len();
                t.tokens.push(tok);
                t.parents.push(parent as i32);
                t.depth.push(t.depth[parent] + 1);
                t.path_q.push(score);
                t.child_q.push(Vec::new());
                let mut h = hist[parent].clone();
                h.push(tok);
                hist.push(h);
                chain.push(chain[parent].clone());
                next.push(id);
            }
            // leaves at the budget limit need no forward pass of their own
            if t.len() >= budget {
                break;
            }
            frontier = next;
        }
        t
    }

    /// Candidate children for a node: deterministic top-k, or i.i.d. samples
    /// from the draft distribution. Returns (candidates, proposal dist q, the
    /// node's visible pending chain for its own children).
    fn node_candidates(
        &mut self,
        history: &[u32],
        token: u32,
        pos: usize,
        visible: Vec<NodeKey>,
        cfg: &DraftConfig,
    ) -> (SparseDist, SparseDist, Vec<NodeKey>) {
        let (logits, chain) = match self.early.as_mut() {
            Some(e) => {
                let (k, l) = e.forward_node(token, pos, visible.clone());
                let mut chain = visible;
                chain.push(k);
                (l, chain)
            }
            None => (self.head_logits(token), Vec::new()),
        };
        let q = self.mix(history, logits);
        let cands = match cfg.proposal {
            ProposalKind::Deterministic => q.iter().take(cfg.max_children).cloned().collect(),
            ProposalKind::SampledIid => {
                let mut c = Vec::new();
                for _ in 0..cfg.max_children {
                    let r = self.rng.uniform();
                    let mut acc = 0.0;
                    let mut pick = q[q.len() - 1];
                    for &e in &q {
                        acc += e.1;
                        if r < acc {
                            pick = e;
                            break;
                        }
                    }
                    c.push(pick);
                }
                // Keep sampling order: MSS exactness needs i.i.d. candidates
                // examined in draw order.
                c
            }
        };
        (cands, q, chain)
    }
}

// ---------------------------------------------------------------------------
// EAGLE-style feature-conditioned drafter
// ---------------------------------------------------------------------------

/// Feature-conditioned draft head (EAGLE-style). It extrapolates the target's
/// own final hidden state one step: f̂ₜ₊₁ = e(xₜ₊₁) + W·fₜ, then reads it with
/// the target's final norm and LM head. `W` is fitted online by ridge
/// regression on (fₜ, xₜ₊₁) → fₜ₊₁ pairs that verification already produced;
/// the d×d solve runs on a background thread so drafting never waits for it.
/// A draft node costs one d×d product plus the LM head.
pub struct FeatureDrafter {
    pub d: usize,
    pub head: HeadWeights,
    pub w: Tensor2D,
    pub ngram: NgramTable,
    pub history: Vec<u32>,
    pub cfg: DraftConfig,
    sxx: Vec<f64>,
    syx: Vec<f64>,
    pairs: usize,
    pairs_at_solve: usize,
    pub solves: u64,
    pending: Option<std::sync::mpsc::Receiver<Tensor2D>>,
    pub lambda: f64,
}

impl FeatureDrafter {
    pub fn new(model: &ModelConfig, cfg: DraftConfig) -> Self {
        let d = model.d_model;
        FeatureDrafter {
            d,
            head: HeadWeights::generate(model),
            w: Tensor2D::zeros(d, d),
            ngram: NgramTable::new(4),
            history: Vec::new(),
            cfg,
            sxx: vec![0.0; d * d],
            syx: vec![0.0; d * d],
            pairs: 0,
            pairs_at_solve: 0,
            solves: 0,
            pending: None,
            lambda: 1.0,
        }
    }

    pub fn commit(&mut self, tokens: &[u32]) {
        for &t in tokens {
            if !self.history.is_empty() {
                self.ngram.observe(&self.history, t);
            }
            self.history.push(t);
        }
    }

    /// One training pair: feature f_t, next token x_{t+1}, its feature f_{t+1}.
    pub fn observe(&mut self, f_t: &[f32], next: u32, f_next: &[f32]) {
        let d = self.d;
        let e = self.head.embed.row(next as usize);
        for i in 0..d {
            let yi = (f_next[i] - e[i]) as f64;
            let row_y = &mut self.syx[i * d..(i + 1) * d];
            for (j, fj) in f_t.iter().enumerate() {
                row_y[j] += yi * *fj as f64;
            }
        }
        for i in 0..d {
            let fi = f_t[i] as f64;
            let row = &mut self.sxx[i * d..(i + 1) * d];
            for (j, fj) in f_t.iter().enumerate() {
                row[j] += fi * *fj as f64;
            }
        }
        self.pairs += 1;
        self.maybe_resolve();
    }

    /// Adopt a finished background solve; launch a new one once enough new
    /// pairs have accumulated (doubling schedule, then every 64 pairs).
    fn maybe_resolve(&mut self) {
        if let Some(rx) = &self.pending {
            if let Ok(w) = rx.try_recv() {
                self.w = w;
                self.solves += 1;
                self.pending = None;
            }
        }
        let due = self.pairs >= 16
            && (self.pairs >= 2 * self.pairs_at_solve || self.pairs >= self.pairs_at_solve + 64);
        if self.pending.is_none() && due {
            self.pairs_at_solve = self.pairs;
            let (d, lambda) = (self.d, self.lambda);
            let mut a = self.sxx.clone();
            let b = self.syx.clone();
            let (tx, rx) = std::sync::mpsc::channel();
            std::thread::spawn(move || {
                crate::tensor::prefer_performance_cores();
                for i in 0..d {
                    a[i * d + i] += lambda;
                }
                // Cholesky A = L Lᵀ (A symmetric positive definite)
                for j in 0..d {
                    let mut s = a[j * d + j];
                    for k in 0..j {
                        s -= a[j * d + k] * a[j * d + k];
                    }
                    let ljj = s.max(1e-12).sqrt();
                    a[j * d + j] = ljj;
                    for i in j + 1..d {
                        let mut s = a[i * d + j];
                        for k in 0..j {
                            s -= a[i * d + k] * a[j * d + k];
                        }
                        a[i * d + j] = s / ljj;
                    }
                }
                // W = B A⁻¹ : solve L Lᵀ wᵢ = bᵢ for every output row i
                let mut w = Tensor2D::zeros(d, d);
                let mut z = vec![0.0f64; d];
                for i in 0..d {
                    let bi = &b[i * d..(i + 1) * d];
                    for r in 0..d {
                        let mut s = bi[r];
                        for k in 0..r {
                            s -= a[r * d + k] * z[k];
                        }
                        z[r] = s / a[r * d + r];
                    }
                    for r in (0..d).rev() {
                        let mut s = z[r];
                        for k in r + 1..d {
                            s -= a[k * d + r] * z[k];
                        }
                        z[r] = s / a[r * d + r];
                    }
                    for (o, v) in w.row_mut(i).iter_mut().zip(&z) {
                        *o = *v as f32;
                    }
                }
                let _ = tx.send(w);
            });
            self.pending = Some(rx);
        }
    }

    /// Extrapolated feature for `token` following a node with feature `f`.
    pub fn next_feature(&self, f: &[f32], token: u32) -> Vec<f32> {
        let e = self.head.embed.row(token as usize);
        (0..self.d)
            .map(|i| e[i] + crate::tensor::dot(self.w.row(i), f))
            .collect()
    }

    fn dist(&self, history: &[u32], feat: &[f32]) -> SparseDist {
        let x = Tensor2D::from_vec(1, self.d, feat.to_vec());
        let logits = self.head.logits(&x, 1e-5).data;
        // reuse TreeDrafter-style mixing (temperature, n-gram, top-k)
        let mut l = logits;
        if self.cfg.temperature > 0.0 {
            for v in l.iter_mut() {
                *v /= self.cfg.temperature;
            }
        }
        softmax_inplace(&mut l);
        if let Some((ng, n)) = self.ngram.predict(history) {
            let lam = n as f32 / (n as f32 + 2.0);
            for v in l.iter_mut() {
                *v *= 1.0 - lam;
            }
            for (t, p) in ng {
                l[t as usize] += lam * p;
            }
        }
        let k = self.cfg.top_k.min(l.len());
        let mut idx: Vec<u32> = (0..l.len() as u32).collect();
        idx.select_nth_unstable_by(k - 1, |a, b| {
            l[*b as usize].partial_cmp(&l[*a as usize]).unwrap()
        });
        let mut top: Vec<(u32, f32)> = idx[..k].iter().map(|&t| (t, l[t as usize])).collect();
        top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
        let s: f32 = top.iter().map(|t| t.1).sum();
        top.iter_mut().for_each(|t| t.1 /= s);
        top
    }

    /// Best-first tree of `budget` nodes rooted at `root`, whose predecessor
    /// (the last verified node) has target feature `f_last`.
    pub fn draft(
        &mut self,
        epoch: u64,
        root: u32,
        f_last: &[f32],
        budget: usize,
    ) -> SpeculativeTree {
        let cfg = self.cfg.clone();
        let mut t = SpeculativeTree::single(epoch, root);
        let mut feats = vec![self.next_feature(f_last, root)];
        let mut hist = vec![{
            let mut h = self.history.clone();
            h.push(root);
            h
        }];
        let q0 = self.dist(&hist[0], &feats[0]);
        t.child_q[0] = q0.clone();
        let mut heap = BinaryHeap::new();
        let mut cands: Vec<SparseDist> = vec![q0.iter().take(cfg.max_children).cloned().collect()];
        if !cands[0].is_empty() {
            heap.push(Frontier {
                score: cands[0][0].1,
                node: 0,
                cand: 0,
            });
        }
        while t.len() < budget {
            let Some(f) = heap.pop() else { break };
            let (tok, q) = cands[f.node][f.cand];
            let pq = t.path_q[f.node];
            let id = t.len();
            t.tokens.push(tok);
            t.parents.push(f.node as i32);
            t.depth.push(t.depth[f.node] + 1);
            t.path_q.push(pq * q);
            if f.cand + 1 < cands[f.node].len() {
                heap.push(Frontier {
                    score: pq * cands[f.node][f.cand + 1].1,
                    node: f.node,
                    cand: f.cand + 1,
                });
            }
            let mut h = hist[f.node].clone();
            h.push(tok);
            let feat = self.next_feature(&feats[f.node], tok);
            if t.depth[id] < cfg.max_depth && t.len() < budget {
                let qd = self.dist(&h, &feat);
                let c: SparseDist = qd.iter().take(cfg.max_children).cloned().collect();
                if !c.is_empty() {
                    heap.push(Frontier {
                        score: pq * q * c[0].1,
                        node: id,
                        cand: 0,
                    });
                }
                t.child_q.push(qd);
                cands.push(c);
            } else {
                t.child_q.push(Vec::new());
                cands.push(Vec::new());
            }
            feats.push(feat);
            hist.push(h);
        }
        t
    }
}
