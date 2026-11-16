//! The Tesseract-Fabric engine: pipeline stages that execute speculative tree
//! strands, a coordinator (last stage) that verifies trees and broadcasts
//! 32-byte commits, a paged KV service per node, and an asynchronous draft
//! service that pre-drafts continuation trees while verification runs.
//!
//! Per cycle (one speculative epoch):
//!   coordinator ──SpeculativeTreeBatch(strand 0..S)──▶ stage 0
//!   stage i ──ActivationDelta(holographic, 1 hop per strand)──▶ stage i+1
//!   coordinator: logits → exact tree verification → TreeCommit ──▶ all stages
//!   meanwhile: coordinator ──DraftRequest(outcomes)──▶ draft service ──DraftTree──▶ coordinator
//! Out of band: stage(L) ──PreWakeSignal(L+ΔL experts)──▶ stage(L+ΔL)

use super::{
    ack_assign, reply_stats, Assignment, ComputeShaper, GenerateExt, GenerateRequest,
    GenerateResult, KvMode, NodeStats, DRAFT_SERVICE_BASE, DRIVER_ID, KV_SERVICE_BASE,
};
use crate::compression::klt::{klt_calibration, KltResidualCodec};
use crate::compression::{
    EntropyResidualCodec, HoloConfig, TreeActivationCodec, TrunkResidualCodec,
};
use crate::kv_fabric::disaggregated_attention::{
    decode_partials, decode_query, encode_partials, encode_query, QueryHeadMeta,
};
use crate::kv_fabric::{
    lse_merge, AttentionBackend, AttnPartial, BatchMeta, KvPageTable, KvShard, LocalKvCache,
    NodeKey, PlacementPolicy, WirePrecision,
};
use crate::model::{
    generate_router, ExpertBank, FfnWeights, HeadWeights, LayerObserver, LayerWeights, ModelConfig,
    ShardModel,
};
use crate::routing::{decode_prewake, encode_prewake, ExpertStore, LookaheadPredictor, Stager};
use crate::speculative::{
    DraftConfig, LowRankDraftHead, MergedTree, RooflineController, SpeculativeTree,
    TreeCommitFrame, TreeDrafter,
};
use crate::tensor::{rms_norm, Rng, RopeTable, Tensor2D};
use crate::wire::protocol::{flags, ByteReader, ByteWriter, FrameType, TesseractFrame};
use crate::wire::transport::{mono_ns, Endpoint};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// shared helpers
// ---------------------------------------------------------------------------

/// Deterministic synthetic context token at absolute position `pos`.
pub fn context_token(seed: u64, pos: usize, vocab: usize) -> u32 {
    let mut r = Rng::new(seed ^ (pos as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    r.below(vocab) as u32
}

/// Approximate-prefill K/V for injected context: the layer's real K/V
/// projections (with RoPE) of the context tokens' normalised embeddings.
/// Skips cross-token mixing of earlier layers so 64K contexts stay tractable.
pub fn context_kv(
    cfg: &ModelConfig,
    layer: &LayerWeights,
    rope: &RopeTable,
    embed: &crate::tensor::Embedding,
    positions: std::ops::Range<usize>,
) -> (Tensor2D, Tensor2D) {
    let n = positions.len();
    let mut xn = Tensor2D::zeros(n, cfg.d_model);
    for (i, p) in positions.clone().enumerate() {
        let t = context_token(cfg.seed, p, cfg.vocab) as usize;
        rms_norm(&embed.row(t), &layer.attn_norm, cfg.rms_eps, xn.row_mut(i));
    }
    let mut k = layer.wk.forward(&xn);
    let v = layer.wv.forward(&xn);
    for (i, p) in positions.enumerate() {
        for h in 0..cfg.n_kv_heads {
            rope.apply(&mut k.row_mut(i)[h * cfg.d_head..(h + 1) * cfg.d_head], p);
        }
    }
    (k, v)
}

/// Layer with only the attention projections needed for context KV.
fn kv_projection_layer(cfg: &ModelConfig, l: usize) -> LayerWeights {
    let mut c = cfg.clone();
    c.moe = None;
    c.d_ff = 32; // FFN is irrelevant for K/V; keep generation cheap
    LayerWeights::generate(&c, l, None)
}

/// A speculative row stage 0 computed ahead of its strand.
struct AheadRow {
    key: NodeKey,
    token: u32,
    out: Tensor2D,
}

/// Share of the per-token critical path (layers × slowdown) carried by the
/// slowest stage.
fn bottleneck_share(a: &Assignment) -> f64 {
    let cost: Vec<f64> = a
        .stages
        .iter()
        .map(|&(n, s, e)| (e - s) as f64 * a.node(n).slowdown)
        .collect();
    let total: f64 = cost.iter().sum();
    cost.iter().cloned().fold(0.0, f64::max) / total.max(1e-9)
}

/// Braid strands only when the slowest stage carries at most this share of
/// the per-token critical path (layers × slowdown, summed over stages).
const BRAID_MAX_BOTTLENECK_SHARE: f64 = 0.6;

/// KV-head groups striped as separate virtual layers `layer · G + g`
/// (1 = the whole layer is one unit).
pub fn kv_groups(a: &Assignment) -> usize {
    if a.tess.kv_group_striped && a.tess.kv_mode == KvMode::Disaggregated {
        a.model.n_kv_heads
    } else {
        1
    }
}

/// Columns of KV-head group `g` (of `groups`) from rows of width `kd`.
fn group_cols(m: &[f32], kd: usize, groups: usize, g: usize) -> Vec<f32> {
    let w = kd / groups;
    m.chunks(kd)
        .flat_map(|row| row[g * w..(g + 1) * w].iter().copied())
        .collect()
}

/// Page table shared (deterministically) by every node of a disaggregated run.
pub fn build_page_table(a: &Assignment) -> KvPageTable {
    let caps: Vec<(u16, usize)> = a
        .nodes
        .iter()
        .map(|n| (n.id, n.kv_capacity as usize))
        .collect();
    let g_n = kv_groups(a);
    let owners: Vec<(std::ops::Range<usize>, u16)> =
        a.stages.iter().map(|&(n, s, e)| (s..e, n)).collect();
    let policy = if g_n > 1 && caps.len() > 1 {
        PlacementPolicy::Striped {
            holders: crate::kv_fabric::page_table::group_holders(
                &owners,
                &caps,
                a.model.n_layers,
                g_n,
                a.tess.kv_remote_bias_pages as usize,
            ),
            owners: a
                .stages
                .iter()
                .map(|&(n, s, e)| (s * g_n..e * g_n, n))
                .collect(),
            local_pages: a.tess.tier0_pages as usize,
            mirror: a.tess.kv_prefix_mirror,
        }
    } else if a.tess.kv_paired && caps.len() > 1 {
        PlacementPolicy::Paired {
            partners: crate::kv_fabric::page_table::pair_partners(&owners, &caps, a.model.n_layers),
            owners,
            local_pages: a.tess.tier0_pages as usize,
            remote_bias: a.tess.kv_remote_bias_pages as usize,
        }
    } else if a.tess.tier0_pages > 0 {
        PlacementPolicy::LocalityFirst {
            owners,
            local_pages: a.tess.tier0_pages as usize,
        }
    } else {
        PlacementPolicy::ProportionalFreeRam
    };
    let mut t = KvPageTable::new(a.tess.page_tokens as usize, &caps, policy);
    let page_bytes = a.tess.page_tokens as usize * a.model.kv_dim() / g_n * 2 * 4;
    // pre-place pages for the injected context so all parties agree
    let pages = (a.inject_context as usize).div_ceil(a.tess.page_tokens as usize);
    for p in 0..pages {
        for vl in 0..a.model.n_layers * g_n {
            t.locate_or_allocate(vl, p, page_bytes)
                .expect("KV fabric capacity exceeded");
        }
    }
    t
}

fn sparse_top(p: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut idx: Vec<u32> = (0..p.len() as u32).collect();
    let k = k.min(p.len());
    idx.select_nth_unstable_by(k - 1, |a, b| {
        p[*b as usize].partial_cmp(&p[*a as usize]).unwrap()
    });
    let mut v: Vec<(u32, f32)> = idx[..k].iter().map(|&t| (t, p[t as usize])).collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    v
}

pub fn outcome_key(gen: u64, end_node: usize, bonus: u32) -> u64 {
    gen.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ ((end_node as u64) << 40) ^ bonus as u64
}

/// Attach the static channel salience of the layer that *receives* this hop
/// (`layer`) to an entropy codec, when enabled.
fn with_salience(mut c: ActCodec, a: &Assignment, layer: Option<usize>) -> ActCodec {
    if a.tess.codec == 4 {
        // HRQ-V5: conditional KLT calibrated for this boundary (both ends
        // derive the identical calibration; a stage without that hop keeps
        // the placeholder)
        if let Some(l) = layer.filter(|&l| l > 0) {
            return ActCodec::Klt(KltResidualCodec::new(klt_calibration(&a.model, l), 0.25));
        }
        return c;
    }
    if let (ActCodec::Entropy(e), Some(l), true) = (&mut c, layer, a.tess.salience) {
        let lw = LayerWeights::generate(&a.model, l, None);
        let mut mats: Vec<&crate::tensor::Linear> = vec![&lw.wq, &lw.wk, &lw.wv];
        let router;
        if let FfnWeights::Moe { router: r, .. } = &lw.ffn {
            router = r.clone();
            mats.push(&router);
        }
        e.salience = Some(crate::compression::channel_salience(&mats, &lw.attn_norm));
    }
    if let (ActCodec::Entropy(e), true) = (&mut c, a.tess.sens_rank > 0) {
        // top-k right-singular subspace of the readout lm_head·diag(final_norm):
        // the directions logits are most sensitive to (identical on both ends)
        let h = HeadWeights::generate(&a.model);
        let basis = LowRankDraftHead::from_target(
            &h.lm_head.dense(),
            &h.final_norm,
            a.tess.sens_rank as usize,
            0x5E45,
        )
        .a;
        let dim = e.plan.dim;
        let old = std::mem::replace(e, EntropyResidualCodec::new(dim, 0.25, 0));
        *e = old.with_basis(basis);
    }
    c
}

/// Inter-stage activation codec.
pub enum ActCodec {
    /// V1: per-row intra / closed-loop-inter holographic coding.
    Holographic(TreeActivationCodec, u8),
    /// V2: trunk residual vs e(token) + outlier sketch + FWHT lattice.
    TrunkResidual(TrunkResidualCodec),
    /// V2.1: trunk residual + FWHT + entropy-coded lattice (hard 25% cap).
    Entropy(EntropyResidualCodec),
    /// V5: conditional KLT transform coding (calibrated, range-coded, 25% cap).
    Klt(KltResidualCodec),
}

impl ActCodec {
    pub fn new(kind: u8, dim: usize, cfg: HoloConfig) -> Self {
        match kind {
            1 => ActCodec::Holographic(TreeActivationCodec::new(dim, cfg.clone()), cfg.bits),
            2 => ActCodec::TrunkResidual(TrunkResidualCodec::new(dim, cfg.bits, cfg.seed)),
            _ => ActCodec::Entropy(EntropyResidualCodec::new(dim, 0.25, cfg.seed)),
        }
    }
    pub fn encode(
        &mut self,
        x: &Tensor2D,
        base: &Tensor2D,
        prev: Option<&Tensor2D>,
        parents: &[i32],
    ) -> Vec<u8> {
        match self {
            ActCodec::Holographic(c, bits) => c.encode(x, parents, *bits).0,
            ActCodec::TrunkResidual(c) => c.encode(x, base).0,
            ActCodec::Entropy(c) => c.encode(x, base).0,
            ActCodec::Klt(c) => c.encode(x, base, prev).0,
        }
    }
    pub fn decode(&mut self, bytes: &[u8], base: &Tensor2D, prev: Option<&Tensor2D>) -> Tensor2D {
        match self {
            ActCodec::Holographic(c, _) => c.decode(bytes),
            ActCodec::TrunkResidual(c) => c.decode(bytes, base),
            ActCodec::Entropy(c) => c.decode(bytes, base),
            ActCodec::Klt(c) => c.decode(bytes, base, prev),
        }
    }
}

// ---------------------------------------------------------------------------
// strand payload codec
// ---------------------------------------------------------------------------

pub struct StrandPayload {
    pub commit_gen: u64,
    pub prefill: bool,
    pub meta: BatchMeta,
    pub parents: Vec<i32>,
    /// Token ids of every row (all hops: HRQ-V2 rebuilds e(token) from them).
    pub tokens: Vec<u32>,
    /// Previous token of every row (HRQ-V5 side information; empty otherwise).
    pub prev: Vec<u32>,
    /// Coded activations (empty on the hop into stage 0).
    pub act: Vec<u8>,
}

impl StrandPayload {
    pub fn encode(&self) -> Vec<u8> {
        let mut w = ByteWriter::default();
        w.u64(self.commit_gen);
        w.u8(self.prefill as u8);
        let mut m = Vec::new();
        self.meta.encode(&mut m);
        w.u32(m.len() as u32);
        w.bytes(&m);
        w.u16(self.parents.len() as u16);
        for &p in &self.parents {
            w.u32(p as u32);
        }
        for &x in &self.tokens {
            w.u32(x);
        }
        w.u8(!self.prev.is_empty() as u8);
        for &x in &self.prev {
            w.u32(x);
        }
        w.bytes(&self.act);
        w.into_inner()
    }

    pub fn decode(b: &[u8]) -> Self {
        let mut r = ByteReader::new(b);
        let commit_gen = r.u64();
        let prefill = r.u8() == 1;
        let ml = r.u32() as usize;
        let (meta, _) = BatchMeta::decode(r.bytes(ml));
        let n = r.u16() as usize;
        let parents = (0..n).map(|_| r.u32() as i32).collect();
        let tokens = (0..n).map(|_| r.u32()).collect();
        let prev = if r.u8() == 1 {
            (0..n).map(|_| r.u32()).collect()
        } else {
            Vec::new()
        };
        StrandPayload {
            commit_gen,
            prefill,
            meta,
            parents,
            tokens,
            prev,
            act: r.rest().to_vec(),
        }
    }
}

// ---------------------------------------------------------------------------
// disaggregated attention backend (stage side)
// ---------------------------------------------------------------------------

struct FabricBackend<'a> {
    kv: &'a mut LocalKvCache,
    ep: &'a Endpoint,
    table: &'a KvPageTable,
    committed_len: usize,
    seq: &'a mut u64,
    /// Decode-phase inter-host attention bytes (None while prefilling).
    traffic: Option<&'a mut u64>,
    /// [local scan ns, local tokens, remote wait ns, remote exchanges,
    /// min remote wait ns, min local scan ns per 256 tokens]
    timing: &'a mut [u64; 6],
    precision: WirePrecision,
    /// This node's own pages (tier 0 + its stripes of its own layers):
    /// attended in-process, no socket.
    local: &'a mut KvShard,
    /// Per layer, heads whose outputs must cross the wire in f32.
    wide_out: &'a HashMap<usize, Vec<bool>>,
    /// KV-head groups striped as separate virtual layers (1 = whole layer).
    groups: usize,
}

impl AttentionBackend for FabricBackend<'_> {
    fn attend(
        &mut self,
        layer: usize,
        q: &Tensor2D,
        k: &Tensor2D,
        v: &Tensor2D,
        meta: &BatchMeta,
    ) -> Tensor2D {
        self.kv.push_pending(layer, k, v, meta);
        let (nh, dh) = (self.kv.n_heads, self.kv.d_head);
        let g_n = self.groups;
        let gh = nh / g_n;
        let me = self.ep.id();
        *self.seq += 1;
        let req = *self.seq;
        // pending (speculative tree) partials first: cheap, and a rigorous
        // lower bound on each query's global softmax mass for the shards
        let index = self.kv.pending_index(layer);
        let mut parts: Vec<Vec<AttnPartial>> = vec![Vec::with_capacity(4); q.rows * nh];
        for i in 0..q.rows {
            for h in 0..nh {
                let qh = &q.row(i)[h * dh..(h + 1) * dh];
                parts[i * nh + h].push(self.kv.pending_partial(
                    layer,
                    qh,
                    h,
                    &meta.visible[i],
                    &index,
                ));
            }
        }
        // per KV-head group (virtual layer): send only that group's query
        // heads to the remote holders of its pages
        let mut expected = 0usize;
        let mut group_q: Vec<Vec<f32>> = Vec::with_capacity(g_n);
        for g in 0..g_n {
            let vl = layer * g_n + g;
            let mut qg = Vec::with_capacity(q.rows * gh * dh);
            for i in 0..q.rows {
                qg.extend_from_slice(&q.row(i)[g * gh * dh..(g + 1) * gh * dh]);
            }
            let owners: Vec<u16> = if self.committed_len > 0 {
                self.table
                    .scanners_for_layer(vl, self.committed_len)
                    .into_iter()
                    .filter(|&o| o != me)
                    .collect()
            } else {
                vec![]
            };
            if !owners.is_empty() {
                let wide = self.wide_out.get(&layer);
                let qmeta: Vec<QueryHeadMeta> = (0..q.rows * gh)
                    .map(|j| {
                        let h = g * gh + j % gh;
                        let p = &parts[(j / gh) * nh + h][0];
                        QueryHeadMeta {
                            wide_out: wide.map(|w| w[h]).unwrap_or(false),
                            hint: if p.l > 0.0 { Some((p.m, p.l)) } else { None },
                        }
                    })
                    .collect();
                let mut payload = encode_query(
                    vl,
                    &qg,
                    q.rows,
                    gh,
                    dh,
                    &vec![usize::MAX; q.rows],
                    self.precision,
                    &qmeta,
                );
                payload.extend_from_slice(&(self.committed_len as u64).to_le_bytes());
                for &o in &owners {
                    let mut f =
                        TesseractFrame::new(FrameType::KvQueryBroadcast, 0, 0, payload.clone());
                    f.header.layer = vl as u16;
                    f.header.tree_epoch = req;
                    if let Some(t) = self.traffic.as_deref_mut() {
                        *t += f.wire_len() as u64;
                    }
                    let _ = self.ep.send(KV_SERVICE_BASE + o, f);
                    expected += 1;
                }
            }
            group_q.push(qg);
        }
        // local pages of every group while remote shards work
        let t_l0 = mono_ns();
        let mut local_tokens = 0usize;
        // the computing node holds the primary prefix copies: it skips them
        // once they are delegated to their mirrors
        let skip_prefix = if self.table.delegated(self.committed_len) {
            self.table.mirror_pages
        } else {
            0
        };
        let keep = |p: usize| p >= skip_prefix;
        for (g, qg) in group_q.iter().enumerate() {
            let vl = layer * g_n + g;
            let toks = self.local.tokens_where(vl, &keep);
            if toks > 0 {
                local_tokens += toks;
                for (idx, p) in self
                    .local
                    .partials_where(vl, qg, q.rows, &vec![usize::MAX; q.rows], &keep)
                    .into_iter()
                    .enumerate()
                {
                    let (i, hh) = (idx / gh, idx % gh);
                    parts[i * nh + g * gh + hh].push(p);
                }
            }
        }
        let t_l1 = mono_ns();
        for _ in 0..expected {
            // a genuine wait on the KV shards: counted in full by the shaper
            let f = crate::tensor::pool::count_wall(|| {
                self.ep.recv_match(&|f| {
                    f.header.frame_type == FrameType::KvLsePartial && f.header.tree_epoch == req
                })
            });
            if Assignment::host_of(f.header.source) != me {
                if let Some(t) = self.traffic.as_deref_mut() {
                    *t += f.wire_len() as u64;
                }
            }
            let g = f.header.layer as usize % g_n;
            for (idx, p) in decode_partials(&f.payload).into_iter().enumerate() {
                let (i, hh) = (idx / gh, idx % gh);
                parts[i * nh + g * gh + hh].push(p);
            }
        }
        if self.traffic.is_some() {
            let t = &mut *self.timing;
            let scan = t_l1 - t_l0;
            t[0] += scan;
            t[1] += local_tokens as u64;
            if local_tokens >= 256 {
                let per_page = scan * 256 / local_tokens as u64;
                t[5] = if t[5] == 0 {
                    per_page
                } else {
                    t[5].min(per_page)
                };
            }
            if expected > 0 {
                let w = mono_ns() - t_l1;
                t[2] += w;
                t[3] += 1;
                t[4] = if t[4] == 0 { w } else { t[4].min(w) };
            }
        }
        let mut out = Tensor2D::zeros(q.rows, nh * dh);
        for i in 0..q.rows {
            for h in 0..nh {
                let refs: Vec<&AttnPartial> = parts[i * nh + h].iter().collect();
                lse_merge(&refs, &mut out.row_mut(i)[h * dh..(h + 1) * dh]);
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// lookahead observer
// ---------------------------------------------------------------------------

#[derive(Default)]
struct LookState {
    predictors: HashMap<usize, LookaheadPredictor>,
    /// (epoch, target layer) → (inputs at source layer, predicted set)
    pending: HashMap<(u64, usize), (Tensor2D, Vec<usize>)>,
    prewakes_sent: u64,
}

struct LookObserver<'a> {
    st: &'a mut LookState,
    ep: &'a Endpoint,
    stager: Option<&'a Stager>,
    a: &'a Assignment,
    my_layers: std::ops::Range<usize>,
    epoch: u64,
    horizon: usize,
}

fn owner_of(a: &Assignment, layer: usize) -> Option<u16> {
    a.stages
        .iter()
        .find(|(_, s, e)| (*s..*e).contains(&layer))
        .map(|x| x.0)
}

impl LayerObserver for LookObserver<'_> {
    fn on_layer_input(&mut self, layer: usize, h: &Tensor2D) {
        let Some(pred) = self.st.predictors.get_mut(&layer) else {
            return;
        };
        let target = layer + self.horizon;
        let experts = pred.predict(h);
        if self.my_layers.contains(&target) {
            if let Some(s) = self.stager {
                s.request(target, experts.clone());
            }
        } else if let Some(owner) = owner_of(self.a, target) {
            let mut f = TesseractFrame::new(
                FrameType::PreWakeSignal,
                0,
                0,
                encode_prewake(target, &experts),
            );
            f.header.flags |= flags::OUT_OF_BAND;
            f.header.tree_epoch = self.epoch;
            f.header.layer = layer as u16;
            let _ = self.ep.send(owner, f);
        }
        self.st.prewakes_sent += 1;
        self.st
            .pending
            .insert((self.epoch, target), (h.clone(), experts));
    }

    fn on_routing(&mut self, layer: usize, selected: &[Vec<usize>]) {
        if layer < self.horizon {
            return;
        }
        let source = layer - self.horizon;
        if self.my_layers.contains(&source) {
            if let Some((h, pred)) = self.st.pending.remove(&(self.epoch, layer)) {
                if let Some(p) = self.st.predictors.get_mut(&source) {
                    p.calibrate(&h, &pred, selected);
                }
            }
        } else if let Some(owner) = owner_of(self.a, source) {
            let mut w = ByteWriter::default();
            w.u16(layer as u16);
            w.u16(selected.len() as u16);
            for s in selected {
                w.u8(s.len() as u8);
                for &e in s {
                    w.u16(e as u16);
                }
            }
            let mut f = TesseractFrame::new(FrameType::RouteFeedback, 0, 0, w.into_inner());
            f.header.tree_epoch = self.epoch;
            f.header.flags |= flags::OUT_OF_BAND;
            let _ = self.ep.send(owner, f);
        }
    }
}

// ---------------------------------------------------------------------------
// Tesseract node (pipeline stage; the last stage is the coordinator)
// ---------------------------------------------------------------------------

struct CycleState {
    tree: MergedTree,
    logits: HashMap<usize, Tensor2D>,
    hidden_rows: HashMap<usize, Tensor2D>,
}

pub struct TesseractNode {
    a: Assignment,
    id: u16,
    stage_idx: usize,
    shard: ShardModel,
    head: Option<HeadWeights>,
    kv: LocalKvCache,
    shaper: ComputeShaper,
    tx_codec: ActCodec,
    rx_codec: ActCodec,
    /// Ship this node's outgoing hop as plain FP16: the codec's CPU time
    /// (measured at startup, both ends' slowdowns) exceeds the transfer time
    /// it saves on this link.
    tx_fp16: bool,
    embed: HeadWeights,
    applied_commits: u64,
    /// Tokens behind the committed KV (coordinator: prefix cache across
    /// requests).
    seq: Vec<u32>,
    committed_len: usize,
    deferred: Vec<TesseractFrame>,
    local_queue: VecDeque<TesseractFrame>,
    look: LookState,
    stager: Option<Stager>,
    stores: Vec<Arc<ExpertStore>>,
    table: Option<KvPageTable>,
    local_shard: Option<KvShard>,
    /// Copies of the prefix pages handed to mirrors (restored when a new,
    /// shorter request drops back below the delegation threshold).
    migrated: Vec<crate::kv_fabric::disaggregated_attention::KvPage>,
    /// Sensitivity-aware wire precision: heads with W_o gain > 4× the layer
    /// median return their attention outputs in f32.
    wide_out: HashMap<usize, Vec<bool>>,
    kv_seq: u64,
    kv_timing: [u64; 6],
    /// Stage-0 cross-step lookahead row (computed under next epoch's key).
    ahead: Option<AheadRow>,
    stats: NodeStats,
    // coordinator-only
    cycle: Option<CycleState>,
    temperature: f32,
    fallback: Option<TreeDrafter>,
    /// Speculation planner kept across requests (learned costs carry over).
    planner: Option<RooflineController>,
    draft_cache: HashMap<u64, SpeculativeTree>,
    last_outcome_key: Option<u64>,
    prefilling: bool,
}

impl TesseractNode {
    pub fn new(ep: &Endpoint, a: &Assignment) -> Self {
        crate::tensor::prefer_performance_cores();
        crate::tensor::pool::install(a.tess.node_lanes as usize);
        let id = ep.id();
        let stage_idx = a
            .stages
            .iter()
            .position(|s| s.0 == id)
            .expect("node has no stage");
        let (_, ls, le) = a.stages[stage_idx];
        let cfg = &a.model;
        let hot = if a.hot_capacity > 0 {
            Some(a.hot_capacity as usize)
        } else {
            None
        };
        let mut shard = ShardModel::generate(cfg, ls..le, hot);
        shard.prefetch();
        // lossless transport (codec 0) keeps every stage's f32 arithmetic
        // identical to a single node: the Q8.24 boundary anchor is lossy
        shard.deterministic_boundary =
            a.tess.deterministic_boundary && stage_idx > 0 && a.tess.codec != 0;
        shard.coalesce_experts = a.tess.coalesce_experts;
        let is_coord = id == a.head;
        let head = if stage_idx == 0 || is_coord {
            Some(HeadWeights::generate(cfg))
        } else {
            None
        };
        let mut kv =
            LocalKvCache::new(cfg.n_heads, cfg.n_kv_heads, cfg.d_head).with_exact(cfg.exact);
        let table = if a.tess.kv_mode == KvMode::Disaggregated {
            kv.hold_committed = false;
            Some(build_page_table(a))
        } else {
            None
        };
        let wide_out: HashMap<usize, Vec<bool>> = shard
            .layers
            .iter()
            .map(|l| {
                let dh = cfg.d_head;
                let wo = l.wo.dense();
                let gains: Vec<f32> = (0..cfg.n_heads)
                    .map(|h| {
                        (0..cfg.d_model)
                            .map(|i| {
                                wo.row(i)[h * dh..(h + 1) * dh]
                                    .iter()
                                    .map(|v| v * v)
                                    .sum::<f32>()
                            })
                            .sum::<f32>()
                            .sqrt()
                    })
                    .collect();
                let mut sorted = gains.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let med = sorted[sorted.len() / 2];
                (l.index, gains.iter().map(|g| *g > 4.0 * med).collect())
            })
            .collect();
        // this stage's own pages (tier 0 + own stripes of its layers): in-process
        let g_n = kv_groups(a);
        let local_shard = table.as_ref().map(|t| {
            let pt = a.tess.page_tokens as usize;
            let mut sh = KvShard::new(id, cfg.n_heads / g_n, cfg.n_kv_heads / g_n, cfg.d_head, pt);
            if a.inject_context > 0 {
                let embed = HeadWeights::generate(cfg).embed;
                let kd = cfg.kv_dim();
                // a context injected past the delegation threshold starts in
                // the migrated state: its mirrored prefix lives only on mirrors
                let moved = t.delegated(a.inject_context as usize);
                for layer in &shard.layers {
                    for p in 0..(a.inject_context as usize).div_ceil(pt) {
                        let mine: Vec<usize> = (0..g_n)
                            .filter(|&g| {
                                let vl = layer.index * g_n + g;
                                t.lookup(vl, p) == Some(id)
                                    && !(moved && t.mirror_of(vl, p).is_some())
                            })
                            .collect();
                        if mine.is_empty() {
                            continue;
                        }
                        let e = ((p + 1) * pt).min(a.inject_context as usize);
                        let (k, v) = context_kv(cfg, layer, &shard.rope, &embed, p * pt..e);
                        for g in mine {
                            sh.append_block(
                                layer.index * g_n + g,
                                p * pt,
                                &group_cols(&k.data, kd, g_n, g),
                                &group_cols(&v.data, kd, g_n, g),
                            );
                        }
                    }
                }
            }
            sh
        });
        // context injection for layer-coupled KV
        if a.inject_context > 0 && a.tess.kv_mode == KvMode::LayerCoupled {
            let embed = HeadWeights::generate(cfg).embed;
            for layer in &shard.layers {
                let (k, v) = context_kv(
                    cfg,
                    layer,
                    &shard.rope,
                    &embed,
                    0..a.inject_context as usize,
                );
                kv.inject_committed(layer.index, &k.data, &v.data, a.inject_context as usize);
            }
        }
        // expert stores + lookahead predictors
        let mut stores = Vec::new();
        let mut store_map = HashMap::new();
        for l in &shard.layers {
            if let FfnWeights::Moe {
                bank: ExpertBank::Staged(s),
                ..
            } = &l.ffn
            {
                stores.push(s.clone());
                store_map.insert(l.index, s.clone());
            }
        }
        let stager = if !store_map.is_empty() && a.tess.lookahead {
            Some(Stager::spawn(store_map))
        } else {
            None
        };
        let mut look = LookState::default();
        if a.tess.lookahead {
            if let Some(m) = &cfg.moe {
                let hz = a.tess.lookahead_horizon as usize;
                for src in ls..le {
                    let tgt = src + hz;
                    if tgt < cfg.n_layers {
                        let router = generate_router(cfg, tgt).unwrap();
                        let ffn_norm = crate::model::layer_ffn_norm(cfg, tgt);
                        look.predictors.insert(
                            src,
                            LookaheadPredictor::from_router(
                                src,
                                tgt,
                                &router,
                                &ffn_norm,
                                m.top_k,
                                a.tess.lookahead_margin as usize,
                            ),
                        );
                    }
                }
            }
        }
        let holo = HoloConfig {
            bits: a.tess.holo_bits,
            ..Default::default()
        };
        let fallback = if is_coord {
            let h = head.as_ref().unwrap();
            let rank = if a.tess.draft_rank == 0 {
                cfg.d_model
            } else {
                a.tess.draft_rank as usize
            };
            let d = TreeDrafter::new(
                DraftConfig::default(),
                LowRankDraftHead::from_linear(&h.lm_head, &h.final_norm, rank, 3),
                h.embed.clone(),
                17,
            );
            // local self-speculative drafter (attention-only early exit) unless disabled
            Some(if a.tess.draft_lookup {
                d.with_prompt_lookup(4)
            } else if a.tess.draft_layers > 0 && a.tess.budget != 1 {
                if a.tess.draft_attn_only {
                    d.with_attention_exit(cfg, a.tess.draft_layers as usize)
                } else {
                    d.with_early_exit(cfg, a.tess.draft_layers as usize)
                }
            } else {
                d
            })
        } else {
            None
        };
        let embed = head.clone().unwrap_or_else(|| HeadWeights::generate(cfg));
        let weight_bytes = shard
            .layers
            .iter()
            .map(|l| {
                if cfg.source.is_file() {
                    l.bytes()
                } else {
                    cfg.layer_params() * 4
                }
            })
            .sum::<usize>() as u64;
        let mut node = TesseractNode {
            a: a.clone(),
            id,
            stage_idx,
            shard,
            head,
            kv,
            shaper: ComputeShaper::new(a.node(id).slowdown),
            tx_codec: with_salience(
                ActCodec::new(a.tess.codec, cfg.d_model, holo.clone()),
                a,
                a.stages.get(stage_idx + 1).map(|s| s.1),
            ),
            tx_fp16: false,
            rx_codec: with_salience(
                ActCodec::new(a.tess.codec, cfg.d_model, holo),
                a,
                if stage_idx > 0 { Some(ls) } else { None },
            ),
            embed,
            applied_commits: 0,
            seq: Vec::new(),
            committed_len: a.inject_context as usize,
            deferred: Vec::new(),
            local_queue: VecDeque::new(),
            look,
            stager,
            stores,
            table,
            local_shard,
            migrated: Vec::new(),
            wide_out,
            kv_seq: 0,
            kv_timing: [0; 6],
            ahead: None,
            stats: NodeStats {
                weight_bytes,
                ..Default::default()
            },
            cycle: None,
            temperature: 1.0,
            fallback,
            planner: None,
            draft_cache: HashMap::new(),
            last_outcome_key: None,
            prefilling: false,
        };
        node.stats.kv_bytes = node.kv.bytes() as u64;
        node.tx_fp16 = node.fp16_pays_on_next_hop();
        ack_assign(ep);
        node
    }

    /// Per-hop codec choice (sender side; every payload carries a 1-byte tag,
    /// so the receiver never has to guess). The codec is worth its CPU time
    /// only if encode + decode (measured here on 4 rows, min of 3, stretched
    /// by both nodes' slowdowns) costs less than the transfer time its smaller
    /// rows save at the link's bandwidth.
    fn fp16_pays_on_next_hop(&mut self) -> bool {
        if !self.a.tess.codec_auto || self.a.tess.codec <= 1 {
            return false;
        }
        let Some(&(next, _, _)) = self.a.stages.get(self.stage_idx + 1) else {
            return false;
        };
        let bw: f64 = self
            .a
            .links
            .iter()
            .filter(|(x, y, _)| (*x, *y) == (self.id, next) || (*y, *x) == (self.id, next))
            .flat_map(|(_, _, ps)| ps.iter().map(|p| p.bandwidth_gbps))
            .sum();
        if bw <= 0.0 {
            return false;
        }
        let d = self.a.model.d_model;
        let toks = [1u32, 2, 3, 4];
        let base = self.embed.embed_tokens(&toks);
        let prev = self.embed.embed_tokens(&[0, 1, 2, 3]);
        let mut x = base.clone();
        for (i, v) in x.data.iter_mut().enumerate() {
            *v += 0.5 * ((i as f32) * 0.37).sin();
        }
        let mut best = u64::MAX;
        let mut bytes = 0usize;
        for _ in 0..3 {
            let t = mono_ns();
            let b = self.tx_codec.encode(&x, &base, Some(&prev), &[-1; 4]);
            best = best.min(mono_ns() - t);
            bytes = b.len();
        }
        let rows = toks.len() as f64;
        let slow = self.a.node(self.id).slowdown + self.a.node(next).slowdown;
        let codec_ns = best as f64 / rows * slow;
        let saved_ns = ((2 * d) as f64 - bytes as f64 / rows).max(0.0) * 8.0 / bw;
        codec_ns > saved_ns
    }

    /// Does any layer have KV scanned on another node at the current
    /// committed length? (Only with `spec_local_kv_only` and a fabric.)
    fn remote_kv_present(&self) -> bool {
        let Some(t) = self
            .table
            .as_ref()
            .filter(|_| self.a.tess.spec_local_kv_only)
        else {
            return false;
        };
        let g_n = kv_groups(&self.a);
        self.a.stages.iter().any(|&(owner, s, e)| {
            (s * g_n..e * g_n).any(|vl| {
                t.scanners_for_layer(vl, self.committed_len)
                    .iter()
                    .any(|&n| n != owner)
            })
        })
    }

    /// Cross-step lookahead applies on stage 0 of a pipeline whose stage 0
    /// carries most of the critical path, with layer-coupled KV. It is a
    /// speculative mechanism of its own: the speculation-off ablation turns
    /// `stage_lookahead` off.
    fn ahead_enabled(&self) -> bool {
        self.a.tess.stage_lookahead
            && self.stage_idx == 0
            && self.a.stages.len() > 1
            && self.table.is_none()
            && bottleneck_share(&self.a) > BRAID_MAX_BOTTLENECK_SHARE
    }

    /// While the later stages finish the current single-row step, compute
    /// this stage's layers for the predicted next token (an early exit of
    /// this stage's own output through the LM head) under the key the next
    /// epoch's root will carry. Between layers, the commit of the current
    /// step is checked: a different bonus token aborts the row at once, so a
    /// misprediction costs at most one layer.
    fn run_ahead(
        &mut self,
        ep: &Endpoint,
        out: &Tensor2D,
        cur_key: NodeKey,
        pos: usize,
        epoch: u64,
    ) {
        let eps = self.a.model.rms_eps;
        let pred = {
            let head = &self.embed;
            let l = self.shaper.run(|| head.logits(out, eps));
            crate::model::argmax(l.row(0)) as u32
        };
        // single-strand epochs advance the epoch counter by 2
        let key = crate::kv_fabric::node_key(epoch + 2, 0);
        let meta = BatchMeta {
            positions: vec![pos + 1],
            keys: vec![key],
            visible: vec![vec![cur_key, key]],
        };
        let mut x = self.embed.embed_tokens(&[pred]);
        let mut confirmed = false;
        for li in 0..self.shard.layers.len() {
            if !confirmed {
                if let Some(cf) =
                    ep.try_recv_match(&|f| f.header.frame_type == FrameType::TreeCommit)
                {
                    let c = TreeCommitFrame::decode(&cf.payload);
                    // processed in order right after this
                    self.local_queue.push_front(cf);
                    if c.bonus_token != pred || c.accepted_len != 1 {
                        self.stats.ahead_aborts += 1;
                        self.kv.commit(&[], &|k| k != key);
                        return;
                    }
                    confirmed = true;
                }
            }
            let det = self.shard.deterministic_boundary && li == 0;
            let (shard, kv, shaper) = (&self.shard, &mut self.kv, &mut self.shaper);
            shaper.run(|| {
                shard.layer_forward(
                    &shard.layers[li],
                    &mut x,
                    &meta,
                    kv,
                    &mut crate::model::NoObserver,
                    det,
                )
            });
        }
        self.ahead = Some(AheadRow {
            key,
            token: pred,
            out: x,
        });
    }

    fn is_coordinator(&self) -> bool {
        self.id == self.a.head
    }

    fn send_to(&mut self, ep: &Endpoint, to: u16, f: TesseractFrame) {
        if to == self.id {
            self.local_queue.push_back(f);
        } else {
            let _ = ep.send(to, f);
        }
    }

    fn next_frame(&mut self, ep: &Endpoint) -> TesseractFrame {
        if let Some(f) = self.local_queue.pop_front() {
            return f;
        }
        ep.recv()
    }

    pub fn run(&mut self, ep: &Endpoint) {
        loop {
            let f = self.next_frame(ep);
            if !self.handle(ep, f) {
                return;
            }
        }
    }

    /// Handle one frame. Returns false on Reset/Shutdown.
    fn handle(&mut self, ep: &Endpoint, f: TesseractFrame) -> bool {
        match f.header.frame_type {
            FrameType::SpeculativeTreeBatch | FrameType::ActivationDelta => self.on_strand(ep, f),
            FrameType::TreeCommit => {
                let c = TreeCommitFrame::decode(&f.payload);
                self.apply_commit(ep, &c);
            }
            FrameType::PreWakeSignal => {
                let (layer, experts) = decode_prewake(&f.payload);
                if let Some(s) = &self.stager {
                    s.request(layer, experts);
                }
            }
            FrameType::RouteFeedback => {
                let mut r = ByteReader::new(&f.payload);
                let layer = r.u16() as usize;
                let rows = r.u16() as usize;
                let sel: Vec<Vec<usize>> = (0..rows)
                    .map(|_| {
                        let k = r.u8() as usize;
                        (0..k).map(|_| r.u16() as usize).collect()
                    })
                    .collect();
                let hz = self.a.tess.lookahead_horizon as usize;
                if let Some((h, pred)) = self.look.pending.remove(&(f.header.tree_epoch, layer)) {
                    if let Some(p) = self.look.predictors.get_mut(&(layer - hz)) {
                        p.calibrate(&h, &pred, &sel);
                    }
                }
            }
            FrameType::DraftTree => {
                let key = u64::from_le_bytes(f.payload[0..8].try_into().unwrap());
                let (t, _) = SpeculativeTree::decode(&f.payload[8..]);
                self.draft_cache.insert(key, t);
            }
            FrameType::Generate => {
                let (req, ext) = GenerateRequest::decode_ext(&f.payload);
                let res = self.generate(ep, &req, &ext);
                let _ = ep.send_payload(DRIVER_ID, FrameType::GenerateResult, 0, 0, res.encode());
            }
            FrameType::StatsRequest => {
                let mut s = self.stats.clone();
                s.busy_ns = self.shaper.busy_ns;
                for st in &self.stores {
                    let snap = st.stats.snapshot();
                    for i in 0..6 {
                        s.staging[i] += snap[i];
                    }
                }
                for p in self.look.predictors.values() {
                    s.lookahead_needed += p.needed;
                    s.lookahead_covered += p.covered;
                }
                s.prewakes_sent = self.look.prewakes_sent;
                [
                    s.kv_local_ns,
                    s.kv_local_tokens,
                    s.kv_wait_ns,
                    s.kv_exchanges,
                    s.kv_wait_min_ns,
                    s.kv_page_min_ns,
                ] = self.kv_timing;
                s.ahead_hits = self.stats.ahead_hits;
                s.ahead_misses = self.stats.ahead_misses;
                s.ahead_aborts = self.stats.ahead_aborts;
                s.kv_bytes = s.kv_bytes.max(
                    self.kv.bytes() as u64
                        + self
                            .local_shard
                            .as_ref()
                            .map(|l| l.bytes() as u64)
                            .unwrap_or(0),
                );
                reply_stats(ep, s);
            }
            FrameType::Reset | FrameType::Shutdown => return false,
            _ => {}
        }
        true
    }

    /// Execute this stage's layers on a strand; forward or (coordinator) keep logits.
    fn on_strand(&mut self, ep: &Endpoint, f: TesseractFrame) {
        let sp = StrandPayload::decode(&f.payload);
        if sp.commit_gen < self.applied_commits {
            return; // stale strand of a cancelled braid epoch
        }
        if sp.commit_gen > self.applied_commits {
            self.deferred.push(f);
            return;
        }
        if sp.prefill {
            // a new request: keep only the KV of the prefix it shares
            if let Some(r) = self.ahead.take() {
                self.kv.commit(&[], &|k| k != r.key);
            }
            let p = sp.meta.positions.first().copied().unwrap_or(0);
            self.kv.truncate(p);
            // disaggregated KV: the remote holders drop the old request's
            // tail too, ahead of this request's appends and queries (same
            // ordered connection), and placement restarts at the prefix
            if let Some(t) = self.table.as_ref() {
                if p < self.committed_len {
                    // only this stage's (virtual) layers
                    let g_n = kv_groups(&self.a);
                    let (s, e) = self
                        .a
                        .stages
                        .iter()
                        .find(|st| st.0 == self.id)
                        .map(|st| (st.1, st.2))
                        .unwrap_or((0, 0));
                    let mut payload = u32::MAX.to_le_bytes().to_vec();
                    payload.extend_from_slice(&(p as u32).to_le_bytes());
                    payload.extend_from_slice(&((s * g_n) as u32).to_le_bytes());
                    payload.extend_from_slice(&((e * g_n) as u32).to_le_bytes());
                    for n in &t.nodes {
                        if n.node_id != self.id {
                            let _ = ep.send_payload(
                                KV_SERVICE_BASE + n.node_id,
                                FrameType::KvAppend,
                                0,
                                0,
                                payload.clone(),
                            );
                        }
                    }
                    let undelegate = t.delegated(self.committed_len) && !t.delegated(p);
                    if let Some(ls) = self.local_shard.as_mut() {
                        ls.truncate_layers(p, s * g_n..e * g_n);
                        // the prefix pages went to mirrors: scan them here
                        // again below the delegation threshold
                        if undelegate {
                            for pg in self.migrated.drain(..) {
                                let n = pg.len.min(p.saturating_sub(pg.start_pos));
                                if n > 0 {
                                    let kd = pg.k.len() / pg.len.max(1);
                                    ls.append_block(
                                        pg.layer,
                                        pg.start_pos,
                                        &pg.k[..n * kd],
                                        &pg.v[..n * kd],
                                    );
                                }
                            }
                        }
                    }
                    self.committed_len = p;
                }
            }
        }
        let epoch = f.header.tree_epoch;
        let cfg = self.a.model.clone();
        let rows = sp.meta.rows();
        let base = self.embed.embed_tokens(&sp.tokens);
        let prev_base = (!sp.prev.is_empty()).then(|| self.embed.embed_tokens(&sp.prev));
        // cross-step lookahead: reuse the row computed ahead of this strand
        // when it is exactly that row; otherwise free it
        let reused = match self.ahead.take() {
            // a tree's root is exactly the row computed ahead (same token,
            // key, position and visible set): reuse it for any tree size
            Some(r) if !sp.prefill && sp.meta.keys[0] == r.key && sp.tokens[0] == r.token => {
                self.stats.ahead_hits += 1;
                Some(r.out)
            }
            Some(r) => {
                self.stats.ahead_misses += 1;
                self.kv.commit(&[], &|k| k != r.key);
                None
            }
            None => None,
        };
        let mut x = if sp.act.is_empty() {
            base.clone()
        } else {
            let shaper = &mut self.shaper;
            let rx = &mut self.rx_codec;
            shaper.run(|| match sp.act[0] {
                1 => crate::compression::decode_fp16_rows(&sp.act[1..]),
                2 => {
                    let n = u32::from_le_bytes(sp.act[1..5].try_into().unwrap()) as usize;
                    let v: Vec<f32> = sp.act[5..]
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|b| f32::from_le_bytes(*b))
                        .collect();
                    Tensor2D::from_vec(n, v.len() / n.max(1), v)
                }
                _ => rx.decode(&sp.act[1..], &base, prev_base.as_ref()),
            })
        };
        debug_assert_eq!(x.rows, rows);
        // with a reused root, only the tree's other rows (which attend to the
        // root's pending KV) still go through this stage's layers
        let meta_fwd = if reused.is_some() {
            BatchMeta {
                positions: sp.meta.positions[1..].to_vec(),
                keys: sp.meta.keys[1..].to_vec(),
                visible: sp.meta.visible[1..].to_vec(),
            }
        } else {
            sp.meta.clone()
        };
        if reused.is_some() {
            let d = x.cols;
            x = Tensor2D::from_vec(rows - 1, d, x.data[d..].to_vec());
        }
        // forward through owned layers
        if meta_fwd.rows() > 0 {
            let my_layers = self.shard.layer_range();
            let horizon = self.a.tess.lookahead_horizon as usize;
            let mut obs = LookObserver {
                st: &mut self.look,
                ep,
                stager: self.stager.as_ref(),
                a: &self.a,
                my_layers,
                epoch,
                horizon,
            };
            let shard = &self.shard;
            let shaper = &mut self.shaper;
            match &self.table {
                None => shaper.run(|| shard.forward(&mut x, &meta_fwd, &mut self.kv, &mut obs)),
                Some(table) => {
                    let traffic = if sp.prefill {
                        None
                    } else {
                        Some(&mut self.stats.attn_bytes)
                    };
                    let mut be = FabricBackend {
                        wide_out: &self.wide_out,
                        local: self
                            .local_shard
                            .as_mut()
                            .expect("disaggregated stage has a local shard"),
                        kv: &mut self.kv,
                        ep,
                        table,
                        committed_len: self.committed_len,
                        seq: &mut self.kv_seq,
                        traffic,
                        timing: &mut self.kv_timing,
                        precision: WirePrecision::from_u8(self.a.tess.kv_wire),
                        groups: kv_groups(&self.a),
                    };
                    shaper.run(|| shard.forward(&mut x, &meta_fwd, &mut be, &mut obs))
                }
            }
        }
        if let Some(root) = reused {
            let d = root.cols;
            let mut full = root.data;
            full.extend_from_slice(&x.data[..(rows - 1) * d]);
            x = Tensor2D::from_vec(rows, d, full);
        }
        if sp.prefill {
            // prompt prefill: commit immediately on every stage
            let keys = sp.meta.keys.clone();
            self.prefilling = true;
            self.commit_keys(ep, &keys);
            self.prefilling = false;
            self.applied_commits += 1;
            self.drain_deferred(ep);
        }
        let last = self.stage_idx + 1 == self.a.stages.len();
        let (row_key, row_pos) = (sp.meta.keys[0], sp.meta.positions[0]);
        if !last {
            let next = self.a.stages[self.stage_idx + 1].0;
            let act = {
                let shaper = &mut self.shaper;
                let tx = &mut self.tx_codec;
                let fp16 = self.tx_fp16;
                let lossless = self.a.tess.codec == 0;
                shaper.run(|| {
                    if lossless {
                        // raw f32 rows: every stage then runs bit-for-bit the
                        // arithmetic of a single node
                        let mut out = vec![2u8];
                        out.extend((x.rows as u32).to_le_bytes());
                        out.extend(x.data.iter().flat_map(|v| v.to_le_bytes()));
                        return out;
                    }
                    let mut out = vec![fp16 as u8];
                    if fp16 {
                        out.extend(crate::compression::encode_fp16_rows(&x));
                    } else {
                        out.extend(tx.encode(&x, &base, prev_base.as_ref(), &sp.parents));
                    }
                    out
                })
            };
            let out = StrandPayload {
                commit_gen: sp.commit_gen,
                prefill: sp.prefill,
                meta: sp.meta,
                parents: sp.parents,
                tokens: sp.tokens,
                prev: sp.prev,
                act,
            };
            let mut nf = TesseractFrame::new(FrameType::ActivationDelta, 0, 0, out.encode());
            nf.header.tree_epoch = epoch;
            nf.header.flags = flags::COMPRESSED_HOLOGRAPHIC
                | if self.a.tess.deterministic_boundary && self.a.tess.codec != 0 {
                    flags::DETERMINISTIC_ANCHOR
                } else {
                    0
                };
            self.send_to(ep, next, nf);
            if rows == 1 && !sp.prefill && self.ahead_enabled() {
                self.run_ahead(ep, &x, row_key, row_pos, epoch);
            }
        } else if !sp.prefill {
            // coordinator: logits for this strand
            let head = self.head.as_ref().unwrap();
            let shaper = &mut self.shaper;
            let logits = shaper.run(|| head.logits(&x, cfg.rms_eps));
            if let Some(c) = self.cycle.as_mut() {
                if let Some(s) = c.tree.strands.iter().position(|s| s.epoch == epoch) {
                    c.logits.insert(s, logits);
                    c.hidden_rows.insert(s, x);
                }
            }
        } else if self.is_coordinator() {
            // prefill reached the coordinator
            if let Some(c) = self.cycle.as_mut() {
                c.logits.insert(usize::MAX, Tensor2D::zeros(0, 0));
            }
        }
    }

    fn drain_deferred(&mut self, ep: &Endpoint) {
        let d = std::mem::take(&mut self.deferred);
        for f in d {
            self.local_queue.push_front(f);
        }
        let _ = ep;
    }

    /// Commit keys locally (and append committed rows to their KV services).
    fn commit_keys(&mut self, ep: &Endpoint, keys: &[NodeKey]) {
        // (epoch, index) order is topological, hence the accepted path order
        let mut sorted = keys.to_vec();
        sorted.sort();
        let t0 = mono_ns();
        let ahead_key = self.ahead.as_ref().map(|r| r.key);
        let rows = self.kv.commit(&sorted, &|k| Some(k) == ahead_key);
        self.stats.commit_ns += mono_ns() - t0;
        self.stats.commits += 1;
        let g_n = kv_groups(&self.a);
        if let Some(table) = self.table.as_mut() {
            let kd_full = self.a.model.kv_dim();
            let kd = kd_full / g_n;
            let pt = table.page_tokens;
            // split each layer's committed rows into its KV-head groups
            let rows: Vec<(usize, Vec<f32>, Vec<f32>)> = rows
                .into_iter()
                .flat_map(|(layer, rk, rv)| {
                    (0..g_n)
                        .map(|g| {
                            (
                                layer * g_n + g,
                                group_cols(&rk, kd_full, g_n, g),
                                group_cols(&rv, kd_full, g_n, g),
                            )
                        })
                        .collect::<Vec<_>>()
                })
                .collect();
            // one KvAppend frame per destination node per commit, carrying
            // every (virtual layer, page run) record bound for that node
            let mut batches: std::collections::BTreeMap<u16, (u32, ByteWriter)> =
                Default::default();
            for (layer, rk, rv) in rows {
                let n = rk.len() / kd;
                let mut i = 0;
                while i < n {
                    let pos = self.committed_len + i;
                    // deterministic placement: every node computes the same table
                    let owner = table
                        .locate_or_allocate(layer, pos / pt, pt * kd * 8)
                        .expect("KV fabric full");
                    let run = (n - i).min(pt - pos % pt);
                    let mut w = ByteWriter::default();
                    w.u32(layer as u32);
                    w.u32(pos as u32);
                    w.u32(run as u32);
                    // BFP wire: per-row BFP16 (|err| ≤ row max · 2⁻¹⁶); else f32
                    let bfp = self.a.tess.kv_wire == 2;
                    w.u8(bfp as u8);
                    for src in [&rk[i * kd..(i + run) * kd], &rv[i * kd..(i + run) * kd]] {
                        if bfp {
                            for row in src.chunks(kd) {
                                crate::kv_fabric::disaggregated_attention::bfp_put(&mut w, row, 16);
                            }
                        } else {
                            w.f32_slice(src);
                        }
                    }
                    let rec = w.into_inner();
                    if owner == self.id {
                        if let Some(ls) = self.local_shard.as_mut() {
                            ls.append_block(
                                layer,
                                pos,
                                &rk[i * kd..(i + run) * kd],
                                &rv[i * kd..(i + run) * kd],
                            );
                        }
                        i += run;
                        continue;
                    }
                    let b = batches
                        .entry(owner)
                        .or_insert_with(|| (0, ByteWriter::default()));
                    b.0 += 1;
                    b.1.bytes(&rec);
                    i += run;
                }
            }
            for (owner, (count, w)) in batches {
                let mut payload = count.to_le_bytes().to_vec();
                payload.extend_from_slice(&w.into_inner());
                if !self.prefilling {
                    self.stats.attn_bytes +=
                        (payload.len() + crate::wire::protocol::HEADER_LEN) as u64;
                }
                let _ =
                    ep.send_payload(KV_SERVICE_BASE + owner, FrameType::KvAppend, 0, 0, payload);
            }
        }
        let was = self.table.as_ref().map(|t| t.delegated(self.committed_len));
        self.committed_len += sorted.len();
        if was == Some(false) && self.table.as_ref().unwrap().delegated(self.committed_len) {
            self.migrate_prefix(ep);
        }
    }

    /// The committed context just crossed the delegation threshold: move
    /// (not copy) every mirrored prefix page of this stage's layers to its
    /// mirror, then drop the local copy, so each page exists exactly once.
    /// The pages travel as ordinary batched KvAppend records ahead of the
    /// first delegated query on the same link.
    fn migrate_prefix(&mut self, ep: &Endpoint) {
        let (Some(table), Some(ls)) = (self.table.as_ref(), self.local_shard.as_mut()) else {
            return;
        };
        let g_n = kv_groups(&self.a);
        let mp = table.mirror_pages;
        let bfp = self.a.tess.kv_wire == 2;
        let mut batches: std::collections::BTreeMap<u16, (u32, ByteWriter)> = Default::default();
        let my_layers = self.shard.layer_range();
        for vl in my_layers.start * g_n..my_layers.end * g_n {
            for page in ls.take_prefix(vl, mp) {
                self.migrated.push(page.clone());
                let Some(m) = table.mirror_of(vl, page.page_idx) else {
                    // no mirror (out of room): keep scanning it locally
                    ls.append_block(vl, page.start_pos, &page.k, &page.v);
                    continue;
                };
                let kd = page.k.len() / page.len.max(1);
                let mut w = ByteWriter::default();
                w.u32(vl as u32);
                w.u32(page.start_pos as u32);
                w.u32(page.len as u32);
                w.u8(bfp as u8);
                for src in [&page.k, &page.v] {
                    if bfp {
                        for row in src.chunks(kd) {
                            crate::kv_fabric::disaggregated_attention::bfp_put(&mut w, row, 16);
                        }
                    } else {
                        w.f32_slice(src);
                    }
                }
                let b = batches
                    .entry(m)
                    .or_insert_with(|| (0, ByteWriter::default()));
                b.0 += 1;
                b.1.bytes(&w.into_inner());
            }
        }
        for (m, (count, w)) in batches {
            let mut payload = count.to_le_bytes().to_vec();
            payload.extend_from_slice(&w.into_inner());
            self.stats.kv_migration_bytes +=
                (payload.len() + crate::wire::protocol::HEADER_LEN) as u64;
            let _ = ep.send_payload(KV_SERVICE_BASE + m, FrameType::KvAppend, 0, 0, payload);
        }
    }

    fn apply_commit(&mut self, ep: &Endpoint, c: &TreeCommitFrame) {
        if c.commit_gen != self.applied_commits {
            return;
        }
        let keys = c.accepted_keys();
        self.commit_keys(ep, &keys);
        self.applied_commits += 1;
        self.drain_deferred(ep);
    }

    // ------------------------------------------------------------------ coordinator

    fn launch_strands(
        &mut self,
        ep: &Endpoint,
        merged: &MergedTree,
        base_pos: usize,
        prev_root: u32,
    ) {
        let need_prev = self.a.tess.codec == 4;
        let stage0 = self.a.stages[0].0;
        for s in 0..merged.strands.len() {
            let meta = merged.strand_meta(s, base_pos);
            let nodes = &merged.strands[s].nodes;
            // in-strand parent index for the activation codec
            let parents: Vec<i32> = nodes
                .iter()
                .map(|&i| {
                    let p = merged.tree.parents[i];
                    if p >= 0 && merged.strand_of[p as usize] == s {
                        merged.local_index[p as usize] as i32
                    } else {
                        -1
                    }
                })
                .collect();
            // previous token of each row: its tree parent's token, or the
            // token committed just before the tree root
            let prev = if need_prev {
                nodes
                    .iter()
                    .map(|&i| match merged.tree.parents[i] {
                        p if p >= 0 => merged.tree.tokens[p as usize],
                        _ => prev_root,
                    })
                    .collect()
            } else {
                Vec::new()
            };
            let sp = StrandPayload {
                commit_gen: self.applied_commits,
                prefill: false,
                meta,
                parents,
                tokens: merged.strand_tokens(s),
                prev,
                act: vec![],
            };
            let mut f = TesseractFrame::new(FrameType::SpeculativeTreeBatch, 0, 0, sp.encode());
            f.header.tree_epoch = merged.strands[s].epoch;
            self.send_to(ep, stage0, f);
        }
    }

    /// Block (processing other frames) until strand `s` logits are available.
    fn wait_strand(&mut self, ep: &Endpoint, s: usize) {
        while !self.cycle.as_ref().unwrap().logits.contains_key(&s) {
            let f = self.next_frame(ep);
            self.handle(ep, f);
        }
    }

    fn generate(
        &mut self,
        ep: &Endpoint,
        req: &GenerateRequest,
        ext: &GenerateExt,
    ) -> GenerateResult {
        self.temperature = req.temperature;
        let mut rng = Rng::new(req.seed);
        let mut res = GenerateResult::default();
        let n = req.prompt.len();
        assert!(n >= 2, "prompt needs ≥ 2 tokens");
        let base0 = self.a.inject_context as usize;
        // ---- prefill prompt[p..n-1], reusing the KV of the prefix shared
        // with the previous request (prefix cache)
        let t_pref = mono_ns();
        let p = super::shared_prefix(&self.seq, &req.prompt, n);
        res.prefix_reused = p as u32;
        self.seq.truncate(p);
        let meta = BatchMeta::causal_prefill(base0 + p, n - 1 - p, 1);
        let dummy = MergedTree::split(SpeculativeTree::single(1, 0), 1, 1, 1);
        self.cycle = Some(CycleState {
            tree: dummy,
            logits: HashMap::new(),
            hidden_rows: HashMap::new(),
        });
        let sp = StrandPayload {
            commit_gen: self.applied_commits,
            prefill: true,
            meta,
            parents: vec![-1; n - 1 - p],
            tokens: req.prompt[p..n - 1].to_vec(),
            // position 0's predecessor is unknown: by convention itself
            prev: if self.a.tess.codec == 4 {
                (p..n - 1)
                    .map(|i| req.prompt[i.saturating_sub(1)])
                    .collect()
            } else {
                Vec::new()
            },
            act: vec![],
        };
        let mut f = TesseractFrame::new(FrameType::SpeculativeTreeBatch, 0, 0, sp.encode());
        f.header.tree_epoch = 1;
        self.send_to(ep, self.a.stages[0].0, f);
        while !self
            .cycle
            .as_ref()
            .unwrap()
            .logits
            .contains_key(&usize::MAX)
        {
            let f = self.next_frame(ep);
            self.handle(ep, f);
        }
        res.prefill_ns = mono_ns() - t_pref;
        self.seq.extend_from_slice(&req.prompt[p..n - 1]);
        if let Some(fb) = self.fallback.as_mut() {
            fb.reset_context();
            if fb.early.is_none() {
                fb.commit(&req.prompt[..n - 1], None);
            }
        }
        // the local early-exit drafter syncs lazily: committed tokens are
        // pushed through its layers in the same batched pass as the next draft
        let mut drafter_pending: Vec<u32> = req.prompt[..n - 1].to_vec();
        // ---- decode cycles
        // braiding overlaps strand s+1 on stage i with strand s on stage i+1;
        // it only pays when no single stage dominates the critical path —
        // otherwise a stale second strand just occupies the bottleneck stage
        // that the next epoch has to wait for
        let stage_cost: Vec<f64> = self
            .a
            .stages
            .iter()
            .map(|&(n, s, e)| (e - s) as f64 * self.a.node(n).slowdown)
            .collect();
        let total: f64 = stage_cost.iter().sum();
        let bottleneck = stage_cost.iter().cloned().fold(0.0, f64::max) / total.max(1e-9);
        let strands = if bottleneck > BRAID_MAX_BOTTLENECK_SHARE {
            1
        } else {
            self.a.tess.braid_strands.clamp(1, 2) as usize
        };
        let cands = if self.a.tess.budget == 0 {
            vec![1, 2, 3, 4, 6, 8, 16]
        } else {
            vec![self.a.tess.budget as usize]
        };
        let key = planner_key(&self.a, self.id);
        let mut ctl = match self.planner.take().or_else(|| {
            planner_cache()
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get(&key)
                .cloned()
        }) {
            Some(c) if c.candidates == cands => c,
            _ => RooflineController::new(cands),
        };
        ctl.begin_request();
        ctl.charge_successor = self.a.tess.stage_lookahead && self.a.stages.len() > 1;
        let mut root = req.prompt[n - 1];
        let mut prev_root = req.prompt[n - 2];
        let mut base_pos = base0 + n - 1;
        let mut epoch_base = 2u64;
        let t0 = mono_ns();
        let mut pending_append: Vec<u32> = req.prompt[..n - 1].to_vec();
        let mut pending_train: Vec<(u32, Vec<(u32, f32)>)> = Vec::new();
        let draft_host = self.a.tess.drafter.unwrap_or(self.id);
        let mut budget = if self.remote_kv_present() {
            1
        } else {
            ctl.choose() as u16
        };
        let mut busy0 = self.shaper.busy_ns;
        while (res.tokens.len() as u32) < req.max_new {
            let tc = mono_ns();
            let gen = self.applied_commits;
            // 1. obtain the tree: pre-drafted continuation or local fallback
            while let Some(df) = ep.try_recv_match(&|f| f.header.frame_type == FrameType::DraftTree)
            {
                let key = u64::from_le_bytes(df.payload[0..8].try_into().unwrap());
                let (t, _) = SpeculativeTree::decode(&df.payload[8..]);
                self.draft_cache.insert(key, t);
            }
            let mut tree = match self
                .draft_cache
                .remove(&self.last_outcome_key.unwrap_or(u64::MAX))
            {
                Some(t) if t.tokens[0] == root && t.len() == budget as usize => {
                    res.draft_hits += 1;
                    t
                }
                _ => {
                    if budget <= 1 {
                        // speculation off / single-node epoch: no drafting work
                        SpeculativeTree::single(0, root)
                    } else {
                        if gen > 1 {
                            res.draft_misses += 1;
                        }
                        let fb = self.fallback.as_mut().unwrap();
                        if fb.early.is_some() {
                            let pending = std::mem::take(&mut drafter_pending);
                            self.shaper
                                .run(|| fb.commit_and_draft(&pending, 0, root, budget as usize))
                        } else {
                            self.shaper.run(|| fb.draft(0, root, budget as usize))
                        }
                    }
                }
            };
            self.draft_cache.clear();
            tree.epoch = epoch_base;
            let s_eff = strands.min(tree.len());
            let strand_nodes = tree.len().div_ceil(s_eff);
            let merged = MergedTree::split(tree, strand_nodes, epoch_base, s_eff);
            self.cycle = Some(CycleState {
                tree: merged.clone(),
                logits: HashMap::new(),
                hidden_rows: HashMap::new(),
            });
            self.launch_strands(ep, &merged, base_pos, prev_root);
            // 2. ask the draft service for continuations of the likely outcomes
            // network-budget-aware speculation: every tree row must send its
            // query to every remote KV holder, so trees run only while the
            // whole context is local (zero attention bytes); with remote KV
            // the step is a single exact row
            let next_budget = if self.remote_kv_present() {
                1
            } else {
                ctl.choose() as u16
            };
            self.request_drafts(
                ep,
                draft_host,
                &merged,
                gen + 1,
                root,
                &pending_append,
                &pending_train,
                next_budget,
            );
            pending_append.clear();
            pending_train.clear();
            // 3. lazy exact verification over braid strands
            let temp = self.temperature;
            let mut probs_cache: HashMap<usize, Vec<f32>> = HashMap::new();
            // a repetition penalty conditions each tree node on its own path
            let history: Vec<u32> = if ext
                .sampling
                .as_ref()
                .is_some_and(|p| p.repetition_penalty != 1.0)
            {
                req.prompt.iter().chain(&res.tokens).copied().collect()
            } else {
                Vec::new()
            };
            let outcome = {
                let tree_c = merged.tree.clone();
                let mut target = |i: usize| -> Vec<f32> {
                    let s = merged.strand_of[i];
                    self.wait_strand(ep, s);
                    let c = self.cycle.as_ref().unwrap();
                    let row = c.logits[&s].row(merged.local_index[i]);
                    let p = if history.is_empty() {
                        ext.probs(row, temp, &[])
                    } else {
                        let mut h = history.clone();
                        h.extend(super::tree_path_tokens(&merged.tree, i));
                        ext.probs(row, temp, &h)
                    };
                    probs_cache.insert(i, p.clone());
                    p
                };
                crate::speculative::verify_tree(&tree_c, &mut target, &mut rng)
            };
            // 4. commit broadcast + local compaction
            let commit = TreeCommitFrame::from_outcome(&merged, &outcome, gen);
            for &(node, _, _) in &self.a.stages.clone() {
                if node != self.id {
                    let mut f =
                        TesseractFrame::new(FrameType::TreeCommit, 0, 0, commit.encode().to_vec());
                    f.header.tree_epoch = commit.epoch;
                    let _ = ep.send(node, f);
                }
            }
            self.apply_commit(ep, &commit);
            // 5. emit tokens, fidelity logits, draft training data
            let emitted = outcome.emitted(&merged.tree);
            let c = self.cycle.take().unwrap();
            if req.record_logits {
                for &pn in &outcome.path {
                    let s = merged.strand_of[pn];
                    res.logits
                        .push(c.logits[&s].row(merged.local_index[pn]).to_vec());
                }
            }
            if self.a.tess.draft_ahead > 0 {
                for (&i, p) in &probs_cache {
                    pending_train.push((merged.tree.tokens[i], sparse_top(p, 8)));
                }
            }
            let mut committed_tokens = vec![root];
            committed_tokens.extend(outcome.path[1..].iter().map(|&i| merged.tree.tokens[i]));
            pending_append.extend_from_slice(&committed_tokens);
            self.seq.extend_from_slice(&committed_tokens);
            if let Some(fb) = self.fallback.as_mut().filter(|_| self.a.tess.budget != 1) {
                if fb.early.is_some() {
                    drafter_pending.extend_from_slice(&committed_tokens);
                } else {
                    fb.commit(&committed_tokens, None);
                }
            }
            let end = *outcome.path.last().unwrap();
            self.last_outcome_key = Some(outcome_key(gen + 1, end, outcome.bonus));
            let now = mono_ns();
            let n_emitted = emitted.len();
            let mut emitted = emitted;
            let room = req.max_new as usize - res.tokens.len();
            let done = super::emit_step(
                ep,
                ext,
                &mut emitted,
                budget,
                now - t0,
                res.prefill_ns,
                room,
            );
            for &t in &emitted {
                res.tokens.push(t);
                res.token_times_ns.push(now - t0);
            }
            res.per_epoch.push(n_emitted as u32);
            res.budgets.push(budget);
            ctl.record(
                budget as usize,
                &merged.tree.path_q,
                n_emitted,
                now - tc,
                self.shaper.busy_ns - busy0,
            );
            busy0 = self.shaper.busy_ns;
            budget = next_budget;
            prev_root = merged.tree.tokens[*outcome.path.last().unwrap()];
            root = outcome.bonus;
            base_pos += outcome.path.len();
            epoch_base += 2;
            if done {
                break;
            }
        }
        res.logits.truncate(res.tokens.len());
        res.decode_ns = mono_ns() - t0;
        if std::env::var_os("TESSERACT_PLANNER_TRACE").is_some() {
            let row: Vec<String> = ctl
                .candidates
                .iter()
                .map(|&c| {
                    format!(
                        "N{c}: T {:.2} ms α {:.2}",
                        ctl.time(c).unwrap_or(0.0) / 1e6,
                        ctl.alpha(c)
                    )
                })
                .collect();
            eprintln!("planner: {} | picks {:?}", row.join(", "), res.budgets);
        }
        planner_cache()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(key, ctl.clone());
        self.planner = Some(ctl);
        res
    }

    #[allow(clippy::too_many_arguments)]
    fn request_drafts(
        &mut self,
        ep: &Endpoint,
        host: u16,
        merged: &MergedTree,
        next_gen: u64,
        root: u32,
        append: &[u32],
        train: &[(u32, Vec<(u32, f32)>)],
        budget: u16,
    ) {
        let k = self.a.tess.draft_ahead as usize;
        if k == 0 {
            return;
        }
        let t = &merged.tree;
        let outs: Vec<(u64, Vec<u32>, u32)> = predicted_outcomes(t, k)
            .into_iter()
            .map(|(nidx, b)| {
                let mut assume = vec![root];
                assume.extend(t.path_to(nidx)[1..].iter().map(|&i| t.tokens[i]));
                (outcome_key(next_gen, nidx, b), assume, b)
            })
            .collect();
        let w_payload = encode_draft_request(append, train, budget, &outs);
        let _ = ep.send_payload(
            DRAFT_SERVICE_BASE + host,
            FrameType::DraftRequest,
            next_gen,
            0,
            w_payload,
        );
    }
}

/// Likely verification outcomes of a tree: path end `n` with probability
/// ∝ path_q(n)·(1 − Σ q(children of n)); bonus guess = the best draft token at
/// `n` that is not already a child. Returns the top `k` as (end node, bonus).
pub fn predicted_outcomes(t: &SpeculativeTree, k: usize) -> Vec<(usize, u32)> {
    let mut outs: Vec<(f32, usize, u32)> = Vec::new();
    for nidx in 0..t.len() {
        let kids: Vec<u32> = t.children(nidx).iter().map(|&c| t.tokens[c]).collect();
        let covered: f32 = t.child_q[nidx]
            .iter()
            .filter(|(tok, _)| kids.contains(tok))
            .map(|x| x.1)
            .sum();
        if let Some(&(b, qb)) = t.child_q[nidx].iter().find(|(tok, _)| !kids.contains(tok)) {
            outs.push((t.path_q[nidx] * (1.0 - covered).max(0.0) * qb, nidx, b));
        }
    }
    outs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    outs.truncate(k);
    outs.into_iter().map(|(_, n, b)| (n, b)).collect()
}

/// DraftRequest payload: newly committed tokens, distillation pairs, tree
/// budget, and the outcomes to draft for (key, assumed prefix, root token).
pub fn encode_draft_request(
    append: &[u32],
    train: &[(u32, Vec<(u32, f32)>)],
    budget: u16,
    outcomes: &[(u64, Vec<u32>, u32)],
) -> Vec<u8> {
    let mut w = ByteWriter::default();
    w.u32(append.len() as u32);
    for &x in append {
        w.u32(x);
    }
    w.u32(train.len() as u32);
    for (tok, tgt) in train {
        w.u32(*tok);
        w.u8(tgt.len() as u8);
        for &(tt, p) in tgt {
            w.u32(tt);
            w.f32(p);
        }
    }
    w.u16(budget);
    w.u8(outcomes.len() as u8);
    for (key, assume, b) in outcomes {
        w.u64(*key);
        w.u16(assume.len() as u16);
        for &x in assume {
            w.u32(x);
        }
        w.u32(*b);
    }
    w.into_inner()
}

// ---------------------------------------------------------------------------
// KV service: serves LSE partials over locally stored pages
// ---------------------------------------------------------------------------

pub fn run_kv_service(ep: Endpoint) {
    crate::tensor::prefer_performance_cores();
    loop {
        let f = ep.recv();
        match f.header.frame_type {
            FrameType::Assign => {
                let a = Assignment::decode(&f.payload);
                a.configure(&ep);
                if a.engine == super::EngineKind::Tesseract
                    && a.tess.kv_mode == KvMode::Disaggregated
                {
                    serve_kv(&ep, &a);
                } else {
                    ack_assign(&ep);
                    wait_reset(&ep);
                }
            }
            FrameType::Shutdown => {
                ep.shared.close();
                return;
            }
            _ => {}
        }
    }
}

/// Node endpoint of a host that is not a pipeline stage.
pub fn run_auxiliary(ep: &Endpoint) {
    ack_assign(ep);
    wait_reset(ep);
}

fn wait_reset(ep: &Endpoint) {
    loop {
        let f = ep.recv();
        match f.header.frame_type {
            FrameType::Reset | FrameType::Shutdown => return,
            FrameType::StatsRequest => reply_stats(ep, NodeStats::default()),
            _ => {}
        }
    }
}

fn serve_kv(ep: &Endpoint, a: &Assignment) {
    let host = Assignment::host_of(ep.id());
    let cfg = &a.model;
    let mut table = build_page_table(a);
    let pt = a.tess.page_tokens as usize;
    let g_n = kv_groups(a);
    let mut shard = KvShard::new(
        host,
        cfg.n_heads / g_n,
        cfg.n_kv_heads / g_n,
        cfg.d_head,
        pt,
    );
    let slowdown = a.node(host).slowdown;
    let mut shaper = ComputeShaper::new(slowdown);
    if a.inject_context > 0 {
        let embed = HeadWeights::generate(cfg).embed;
        let rope = cfg.rope_table();
        let pages = (a.inject_context as usize).div_ceil(pt);
        // pages of the host's own stage layers live in that stage's local shard
        let own: Vec<usize> = a
            .stages
            .iter()
            .filter(|s| s.0 == host)
            .flat_map(|s| s.1..s.2)
            .collect();
        for l in 0..cfg.n_layers {
            if own.contains(&l) {
                continue;
            }
            let layer = kv_projection_layer(cfg, l);
            let kd = cfg.kv_dim();
            for p in 0..pages {
                // primary pages and mirrored prefix copies held by this host
                let moved = table.delegated(a.inject_context as usize);
                let mine: Vec<usize> = (0..g_n)
                    .filter(|&g| {
                        let vl = l * g_n + g;
                        table.lookup(vl, p) == Some(host)
                            || (moved && table.mirror_of(vl, p) == Some(host))
                    })
                    .collect();
                if mine.is_empty() {
                    continue;
                }
                let s = p * pt;
                let e = ((p + 1) * pt).min(a.inject_context as usize);
                let (k, v) = context_kv(cfg, &layer, &rope, &embed, s..e);
                for g in mine {
                    shard.append_block(
                        l * g_n + g,
                        s,
                        &group_cols(&k.data, kd, g_n, g),
                        &group_cols(&v.data, kd, g_n, g),
                    );
                }
            }
        }
    }
    let mut served = 0u64;
    let mut peak_bytes = shard.bytes() as u64;
    let mut deferred: VecDeque<TesseractFrame> = VecDeque::new();
    ack_assign(ep);
    loop {
        let f = match deferred.pop_front() {
            Some(f) => f,
            None => ep.recv(),
        };
        match f.header.frame_type {
            FrameType::KvAppend => {
                let mut r = ByteReader::new(&f.payload);
                let count = r.u32();
                if count == u32::MAX {
                    // truncate record: a new request keeps positions < p of
                    // the sender's virtual layers [l0, l1)
                    let p = r.u32() as usize;
                    let (l0, l1) = (r.u32() as usize, r.u32() as usize);
                    shard.truncate_layers(p, l0..l1);
                    continue;
                }
                let count = count as usize;
                for _ in 0..count {
                    let layer = r.u32() as usize;
                    let pos = r.u32() as usize;
                    let n = r.u32() as usize;
                    let kd = cfg.kv_dim() / g_n;
                    let bfp = r.u8() == 1;
                    let read = |r: &mut ByteReader| -> Vec<f32> {
                        if bfp {
                            (0..n)
                                .flat_map(|_| {
                                    crate::kv_fabric::disaggregated_attention::bfp_get(r, kd, 16)
                                })
                                .collect()
                        } else {
                            r.f32_vec(n * kd)
                        }
                    };
                    let k = read(&mut r);
                    let v = read(&mut r);
                    let _ = table.locate_or_allocate(layer, pos / pt, pt * kd * 8);
                    shard.append_block(layer, pos, &k, &v);
                }
                peak_bytes = peak_bytes.max(shard.bytes() as u64);
            }
            FrameType::KvQueryBroadcast => {
                let qlen = f.payload.len() - 8;
                let committed = u64::from_le_bytes(f.payload[qlen..].try_into().unwrap()) as usize;
                let dq = decode_query(&f.payload[..qlen]);
                // expected local tokens for this layer given the committed length
                let expect: usize = table
                    .ranges_for_layer(dq.layer)
                    .into_iter()
                    .filter(|(n, _)| *n == host)
                    .flat_map(|(_, ps)| ps)
                    .map(|p| committed.saturating_sub(p * pt).min(pt))
                    .sum();
                if shard.tokens(dq.layer) < expect {
                    // wait for in-flight appends (ordering guard)
                    let g = ep.recv();
                    deferred.push_back(g);
                    deferred.push_back(f);
                    continue;
                }
                // a service holds prefix pages only as mirrors: it scans them
                // exactly when the computing node has delegated them
                let delegated = table.delegated(committed);
                let mp = table.mirror_pages;
                let keep = |p: usize| p >= mp || delegated;
                let parts = super::ComputePermits::global().run(|| {
                    shaper.run(|| {
                        shard.partials_where(dq.layer, &dq.q, dq.rows, &dq.query_pos, &keep)
                    })
                });
                let mut rf = TesseractFrame::new(
                    FrameType::KvLsePartial,
                    0,
                    0,
                    encode_partials(&parts, dq.precision, &dq.wide_out, &dq.hints),
                );
                rf.header.tree_epoch = f.header.tree_epoch;
                rf.header.layer = dq.layer as u16;
                let _ = ep.send(f.header.source, rf);
                served += 1;
            }
            FrameType::StatsRequest => {
                let s = NodeStats {
                    busy_ns: shaper.busy_ns,
                    kv_bytes: peak_bytes,
                    kv_queries_served: served,
                    ..Default::default()
                };
                reply_stats(ep, s);
            }
            FrameType::Reset | FrameType::Shutdown => return,
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// draft service: asynchronous continuation drafting + online distillation
// ---------------------------------------------------------------------------

pub fn run_draft_service(ep: Endpoint) {
    crate::tensor::prefer_performance_cores();
    loop {
        let f = ep.recv();
        match f.header.frame_type {
            FrameType::Assign => {
                let a = Assignment::decode(&f.payload);
                a.configure(&ep);
                let host = Assignment::host_of(ep.id());
                let active = matches!(
                    a.engine,
                    super::EngineKind::Tesseract | super::EngineKind::TesseractHtep
                ) && a.tess.drafter.unwrap_or(a.head) == host;
                if active {
                    serve_drafts(&ep, &a);
                } else {
                    ack_assign(&ep);
                    wait_reset(&ep);
                }
            }
            FrameType::Shutdown => {
                ep.shared.close();
                return;
            }
            _ => {}
        }
    }
}

fn serve_drafts(ep: &Endpoint, a: &Assignment) {
    let host = Assignment::host_of(ep.id());
    let cfg = &a.model;
    let h = HeadWeights::generate(cfg);
    let rank = if a.tess.draft_rank == 0 {
        cfg.d_model
    } else {
        a.tess.draft_rank as usize
    };
    let mut drafter = TreeDrafter::new(
        DraftConfig::default(),
        LowRankDraftHead::from_linear(&h.lm_head, &h.final_norm, rank, 3),
        h.embed.clone(),
        23,
    );
    if a.tess.draft_lookup {
        drafter = drafter.with_prompt_lookup(4);
    } else if a.tess.draft_layers > 0 {
        drafter = if a.tess.draft_attn_only {
            drafter.with_attention_exit(cfg, a.tess.draft_layers as usize)
        } else {
            drafter.with_early_exit(cfg, a.tess.draft_layers as usize)
        };
    }
    let mut shaper = ComputeShaper::new(a.node(host).slowdown);
    ack_assign(ep);
    loop {
        let mut f = ep.recv();
        // skip superseded requests: only the newest generation matters
        while f.header.frame_type == FrameType::DraftRequest {
            let Some(newer) =
                ep.try_recv_match(&|g| g.header.frame_type == FrameType::DraftRequest)
            else {
                break;
            };
            // still apply its history/training before dropping the older one
            apply_draft_history(&mut drafter, &f.payload, &mut shaper);
            f = newer;
        }
        match f.header.frame_type {
            FrameType::DraftRequest => {
                let mut r = ByteReader::new(&f.payload);
                let (budget, outcomes) = {
                    let na = r.u32() as usize;
                    let append: Vec<u32> = (0..na).map(|_| r.u32()).collect();
                    let nt = r.u32() as usize;
                    let mut train = Vec::with_capacity(nt);
                    for _ in 0..nt {
                        let tok = r.u32();
                        let k = r.u8() as usize;
                        train.push((tok, (0..k).map(|_| (r.u32(), r.f32())).collect::<Vec<_>>()));
                    }
                    shaper.run(|| {
                        drafter.commit(&append, None);
                        for (tok, tgt) in &train {
                            drafter.train(&[], *tok, tgt);
                        }
                    });
                    let budget = r.u16() as usize;
                    let no = r.u8() as usize;
                    let outs: Vec<(u64, Vec<u32>, u32)> = (0..no)
                        .map(|_| {
                            let key = r.u64();
                            let n = r.u16() as usize;
                            let assume = (0..n).map(|_| r.u32()).collect();
                            (key, assume, r.u32())
                        })
                        .collect();
                    (budget, outs)
                };
                for (key, assume, b) in outcomes {
                    let tree = shaper.run(|| drafter.draft_with_prefix(0, &assume, b, budget));
                    let mut payload = key.to_le_bytes().to_vec();
                    payload.extend(tree.encode());
                    let _ = ep.send_payload(
                        f.header.source,
                        FrameType::DraftTree,
                        f.header.tree_epoch,
                        0,
                        payload,
                    );
                }
            }
            FrameType::StatsRequest => reply_stats(
                ep,
                NodeStats {
                    busy_ns: shaper.busy_ns,
                    ..Default::default()
                },
            ),
            FrameType::Reset | FrameType::Shutdown => return,
            _ => {}
        }
    }
}

/// Speculation planners learned by this worker process, by assignment: a
/// worker re-assigned the same plan (same model, stages, options) resumes
/// with its learned tree costs and acceptance instead of re-exploring.
fn planner_cache() -> &'static Mutex<HashMap<u64, RooflineController>> {
    static C: std::sync::OnceLock<Mutex<HashMap<u64, RooflineController>>> =
        std::sync::OnceLock::new();
    C.get_or_init(|| Mutex::new(HashMap::new()))
}

fn planner_key(a: &Assignment, id: u16) -> u64 {
    use std::hash::{Hash, Hasher};
    // the plan, not where it runs: endpoints change on every re-assignment
    let unspecified = std::net::SocketAddr::from(([0, 0, 0, 0], 0));
    let mut plan = a.clone();
    for n in &mut plan.nodes {
        n.addr = unspecified;
    }
    plan.driver_addr = unspecified;
    plan.service_addrs.clear();
    let mut h = std::collections::hash_map::DefaultHasher::new();
    plan.encode().hash(&mut h);
    id.hash(&mut h);
    h.finish()
}

fn apply_draft_history(drafter: &mut TreeDrafter, payload: &[u8], shaper: &mut ComputeShaper) {
    let mut r = ByteReader::new(payload);
    let na = r.u32() as usize;
    let append: Vec<u32> = (0..na).map(|_| r.u32()).collect();
    let nt = r.u32() as usize;
    let mut train = Vec::with_capacity(nt);
    for _ in 0..nt {
        let tok = r.u32();
        let k = r.u8() as usize;
        train.push((tok, (0..k).map(|_| (r.u32(), r.f32())).collect::<Vec<_>>()));
    }
    shaper.run(|| {
        drafter.commit(&append, None);
        for (tok, tgt) in &train {
            drafter.train(&[], *tok, tgt);
        }
    });
}
