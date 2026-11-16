//! H-TEP: speculative tensor-expert parallelism for low-latency groups.
//!
//! Every rank of the group shards each layer like tensor parallelism —
//! attention heads (or a replicated attention stack, `htep_attn_split =
//! false`) and the hidden units of every expert — so all ranks compute on
//! every layer. Unlike the TP baseline, a whole *speculative tree* of N rows
//! goes through the layer stack together: the per-layer collectives are paid
//! once per epoch instead of once per token, and the N-row GEMMs run at the
//! tiled-batch kernel's 2–2.4× better per-row efficiency. Partials are summed
//! in rank order after the same encode/decode on every rank, so ranks stay
//! bit-identical.
//!
//! Rank 0 is the coordinator: it drafts locally with the self-speculative
//! early-exit drafter (one batched pass per tree level), verifies exactly, and
//! broadcasts 32-byte commits. Tree size comes from a roofline controller whose
//! epoch-time model is calibrated once at startup (no wall-clock feedback).

use super::batch::{
    plan_chains, BatchCommit, BatchStep, DraftModel, DraftOffer, DraftSession, Segment,
    SpecController, StepCost, SuffixIndex,
};
use super::node::StrandPayload;
use super::{
    ack_assign, reply_stats, Assignment, ComputeShaper, GenerateExt, GenerateRequest,
    GenerateResult, NodeStats, DRIVER_ID,
};
use crate::compression::{decode_fp16_rows, encode_fp16_rows, TrunkResidualCodec};
use crate::kv_fabric::{AttentionBackend, BatchMeta, LocalKvCache, NodeKey};
use crate::model::{
    moe_combine, route_coalesced, route_topk, Expert, ExpertBank, FfnWeights, HeadWeights,
    LayerWeights,
};
use crate::routing::{ExpertStore, LookaheadPredictor, Stager};
use crate::speculative::{
    DraftConfig, LowRankDraftHead, MergedTree, RooflineController, SpeculativeTree,
    TreeCommitFrame, TreeDrafter,
};
use crate::tensor::{rms_norm, Linear, Rng, RopeTable, Tensor2D};
use crate::wire::protocol::{ByteReader, ByteWriter, FrameType, TesseractFrame};
use crate::wire::transport::{mono_ns, Endpoint};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;

/// Epoch space of batched (session) steps, disjoint from the classic path's.
const BATCH_EPOCH: u64 = 1 << 40;
/// Prompt rows a session prefills per batched step (chunked prefill keeps
/// the other sessions' decode steps flowing).
const PREFILL_CHUNK: usize = 192;

/// A request being served by the session scheduler (head only).
struct Session {
    id: u64,
    slot: usize,
    req: GenerateRequest,
    ext: GenerateExt,
    rng: Rng,
    res: GenerateResult,
    /// Prompt + every emitted token (ends with the current root).
    sam: SuffixIndex,
    draft: DraftSession,
    ctl: SpecController,
    /// Next prompt row to prefill (prefill ends at prompt.len() − 1).
    prefill_at: usize,
    reset_sent: bool,
    root: u32,
    base_pos: usize,
    t_admit: u64,
    t_decode: u64,
}

/// What a session put into the current step.
enum StepPlan {
    Prefill(Vec<u32>),
    Decode {
        merged: Box<MergedTree>,
        drafted: usize,
        from_model: bool,
        offset: usize,
    },
}

struct HtepLayer {
    index: usize,
    extras: crate::model::AttnExtras,
    attn_norm: Vec<f32>,
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    ffn_norm: Vec<f32>,
    router: Option<Linear>,
    top_k: usize,
    /// every expert, hidden units sliced to this rank (TP-style)
    bank: Option<ExpertBank>,
    dense: Option<Expert>,
}

/// All-reduce payload coding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceCoding {
    F16 = 0,
    Hrq = 1,
    F32 = 2,
    /// Exact layers' fixed-point partial sums (`i64`, grid 2^-32): summed as
    /// integers, so the result is independent of the reduction order and
    /// identical to the unsplit layer (selected automatically).
    Fixed = 3,
}

/// One rank's share of a row-parallel layer output.
enum Partial {
    F32(Tensor2D),
    /// Fixed-point sums of an exact layer (`rows × cols`).
    Fixed {
        rows: usize,
        cols: usize,
        v: Vec<i64>,
    },
}

impl Partial {
    fn rows(&self) -> usize {
        match self {
            Partial::F32(t) => t.rows,
            Partial::Fixed { rows, .. } => *rows,
        }
    }
    fn cols(&self) -> usize {
        match self {
            Partial::F32(t) => t.cols,
            Partial::Fixed { cols, .. } => *cols,
        }
    }
    /// The rounded output (no reduction needed).
    fn into_f32(self) -> Tensor2D {
        match self {
            Partial::F32(t) => t,
            Partial::Fixed { rows, cols, v } => Tensor2D::from_vec(
                rows,
                cols,
                v.into_iter().map(crate::tensor::kquant::from_fix).collect(),
            ),
        }
    }
}

/// `x · Wᵀ` of a row-parallel layer as a partial: fixed-point for exact
/// layers, f32 otherwise.
fn row_parallel(w: &Linear, x: &Tensor2D) -> Partial {
    if w.exact {
        Partial::Fixed {
            rows: x.rows,
            cols: w.out_features(),
            v: w.forward_fixed(x),
        }
    } else {
        Partial::F32(w.forward(x))
    }
}

pub struct HtepNode {
    a: Assignment,
    id: u16,
    rank: usize,
    world: usize,
    ranks: Vec<u16>,
    layers: Vec<HtepLayer>,
    head: HeadWeights,
    rope: RopeTable,
    kv: LocalKvCache,
    shaper: ComputeShaper,
    stats: NodeStats,
    coding: ReduceCoding,
    hrq: TrunkResidualCodec,
    applied_commits: u64,
    /// Tokens behind this rank's committed KV (head: reused across requests
    /// as a prefix cache).
    seq: Vec<u32>,
    deferred: VecDeque<TesseractFrame>,
    drafter: Option<TreeDrafter>,
    attn_split: bool,
    pub barriers: u64,
    reduce_bytes: u64,
    stores: Vec<Arc<ExpertStore>>,
    /// Lookahead expert pre-staging (layer L predicts layer L+ΔL's experts;
    /// every rank holds the full residual after each collective, so all ranks
    /// make the same prediction for their own expert slices).
    stager: Option<Stager>,
    predictors: HashMap<usize, LookaheadPredictor>,
    look_pending: HashMap<usize, (Tensor2D, Vec<usize>)>,
    /// Per-session KV slots (continuous batching), lazily created.
    skv: Vec<LocalKvCache>,
    /// Head: tokens behind each slot's committed KV (prefix cache) and when
    /// the slot was last released (LRU tie-break).
    slot_seq: Vec<Vec<u32>>,
    slot_stamp: Vec<u64>,
    /// Batched steps issued (head) / applied in order (ranks).
    batch_seq: u64,
    batch_wait: BTreeMap<(u64, u8), TesseractFrame>,
    batch_next: (u64, u8),
    draft_model: Option<Arc<DraftModel>>,
    draft_tried: bool,
    step_cost: StepCost,
}

impl HtepNode {
    pub fn new(ep: &Endpoint, a: &Assignment) -> Self {
        crate::tensor::prefer_performance_cores();
        crate::tensor::pool::install(a.tess.node_lanes as usize);
        let cfg = &a.model;
        let ranks: Vec<u16> = a.stages.iter().map(|s| s.0).collect();
        let world = ranks.len();
        let rank = ranks
            .iter()
            .position(|&r| r == ep.id())
            .expect("node not an H-TEP rank");
        // split attention only when every rank's slice of the output
        // projection stays block aligned (a quantized file's zero-copy view;
        // a misaligned slice would be dequantized and lose exactness)
        let wo_block = if cfg.source.is_file() {
            LayerWeights::generate(cfg, 0, None).wo.format().block_len()
        } else {
            1
        };
        let attn_split = a.tess.htep_attn_split
            && cfg.n_heads.is_multiple_of(world)
            && cfg.n_kv_heads.is_multiple_of(world)
            && (cfg.n_heads / world * cfg.d_head).is_multiple_of(wo_block);
        let (hl, kl) = if attn_split {
            (cfg.n_heads / world, cfg.n_kv_heads / world)
        } else {
            (cfg.n_heads, cfg.n_kv_heads)
        };
        let dh = cfg.d_head;
        let hot = a.hot_capacity as usize;
        let mut stores = Vec::new();
        let mut weight_bytes = 0usize;
        let layers: Vec<HtepLayer> = (0..cfg.n_layers)
            .map(|l| {
                let full = LayerWeights::generate(cfg, l, None);
                let (wq, wk, wv, wo) = if attn_split {
                    (
                        full.wq.slice_rows(rank * hl * dh, (rank + 1) * hl * dh),
                        full.wk.slice_rows(rank * kl * dh, (rank + 1) * kl * dh),
                        full.wv.slice_rows(rank * kl * dh, (rank + 1) * kl * dh),
                        full.wo.slice_cols(rank * hl * dh, (rank + 1) * hl * dh),
                    )
                } else {
                    (full.wq, full.wk, full.wv, full.wo)
                };
                weight_bytes += wq.bytes() + wk.bytes() + wv.bytes() + wo.bytes();
                for w in [&wq, &wk, &wv, &wo] {
                    w.prefetch();
                }
                let (router, top_k, bank, dense) = match full.ffn {
                    FfnWeights::Moe {
                        router,
                        bank,
                        top_k,
                    } => {
                        let m = cfg.moe.as_ref().unwrap();
                        let sliced: Vec<Expert> = (0..m.n_experts)
                            .map(|e| {
                                let ex = bank.acquire(e);
                                let (h0, h1) = ex.hidden_range(rank, world);
                                ex.slice_hidden(h0, h1)
                            })
                            .collect();
                        weight_bytes += sliced.iter().map(|e| e.bytes()).sum::<usize>();
                        // page in only this rank's hidden slices of a mapped model
                        for e in &sliced {
                            e.w_gate.prefetch();
                            e.w_up.prefetch();
                            e.w_down.prefetch();
                        }
                        let bank = if hot > 0 && hot < m.n_experts && !cfg.source.is_file() {
                            let s = Arc::new(ExpertStore::new(l, &sliced, hot));
                            stores.push(s.clone());
                            ExpertBank::Staged(s)
                        } else {
                            ExpertBank::Resident(sliced.into_iter().map(Arc::new).collect())
                        };
                        (Some(router), top_k, Some(bank), None)
                    }
                    FfnWeights::Dense(e) => {
                        let (h0, h1) = e.hidden_range(rank, world);
                        let s = e.slice_hidden(h0, h1);
                        weight_bytes += s.bytes();
                        (None, 0, None, Some(s))
                    }
                };
                HtepLayer {
                    index: l,
                    extras: if attn_split {
                        full.extras.slice_heads(
                            rank * hl,
                            (rank + 1) * hl,
                            rank * kl,
                            (rank + 1) * kl,
                            dh,
                        )
                    } else {
                        full.extras.clone()
                    },
                    attn_norm: full.attn_norm,
                    wq,
                    wk,
                    wv,
                    wo,
                    ffn_norm: full.ffn_norm,
                    router,
                    top_k,
                    bank,
                    dense,
                }
            })
            .collect();
        let head = HeadWeights::generate(cfg);
        let drafter = if rank == 0 {
            let d = TreeDrafter::new(
                DraftConfig::default(),
                LowRankDraftHead::from_linear(&head.lm_head, &head.final_norm, cfg.d_model, 3),
                head.embed.clone(),
                17,
            );
            Some(if a.tess.draft_lookup {
                d.with_prompt_lookup(4)
            } else if a.tess.draft_attn_only {
                d.with_attention_exit(cfg, a.tess.draft_layers.max(1) as usize)
            } else {
                d.with_early_exit(cfg, a.tess.draft_layers.max(1) as usize)
            })
        } else {
            None
        };
        // lookahead pre-staging over this rank's staged expert slices
        let store_map: HashMap<usize, Arc<ExpertStore>> =
            stores.iter().map(|s| (s.layer, s.clone())).collect();
        let stager = if a.tess.lookahead && !store_map.is_empty() {
            Some(Stager::spawn(store_map))
        } else {
            None
        };
        let mut predictors = HashMap::new();
        if a.tess.lookahead {
            if let Some(m) = &cfg.moe {
                let hz = a.tess.lookahead_horizon as usize;
                for src in 0..cfg.n_layers.saturating_sub(hz) {
                    let tgt = src + hz;
                    let router = crate::model::generate_router(cfg, tgt).unwrap();
                    let ffn_norm = crate::model::layer_ffn_norm(cfg, tgt);
                    predictors.insert(
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
        let coding = if a.tess.htep_reduce_f32 {
            ReduceCoding::F32
        } else if a.tess.htep_hrq {
            ReduceCoding::Hrq
        } else {
            ReduceCoding::F16
        };
        let node = HtepNode {
            a: a.clone(),
            id: ep.id(),
            rank,
            world,
            ranks,
            layers,
            rope: cfg.rope_table(),
            kv: LocalKvCache::new(hl, kl, dh).with_exact(cfg.exact),
            shaper: ComputeShaper::new(a.node(ep.id()).slowdown),
            stats: NodeStats {
                weight_bytes: weight_bytes as u64,
                ..Default::default()
            },
            coding,
            hrq: TrunkResidualCodec::new(cfg.d_model, a.tess.holo_bits.max(2), 0x4E7),
            applied_commits: 0,
            seq: Vec::new(),
            deferred: VecDeque::new(),
            drafter,
            attn_split,
            head,
            barriers: 0,
            reduce_bytes: 0,
            stager,
            predictors,
            look_pending: HashMap::new(),
            stores,
            skv: Vec::new(),
            slot_seq: Vec::new(),
            slot_stamp: Vec::new(),
            batch_seq: 0,
            batch_wait: BTreeMap::new(),
            batch_next: (1, 0),
            draft_model: None,
            draft_tried: false,
            step_cost: StepCost::default(),
        };
        ack_assign(ep);
        node
    }

    fn next(&mut self, ep: &Endpoint) -> TesseractFrame {
        self.deferred.pop_front().unwrap_or_else(|| {
            ep.recv_match(&|f| f.header.frame_type != FrameType::AllReducePartial)
        })
    }

    pub fn run(&mut self, ep: &Endpoint) {
        loop {
            let f = self.next(ep);
            match f.header.frame_type {
                FrameType::SpeculativeTreeBatch => {
                    let sp = StrandPayload::decode(&f.payload);
                    if sp.commit_gen > self.applied_commits {
                        self.deferred.push_back(f);
                        continue;
                    }
                    if sp.prefill {
                        // a new request: keep only the prefix it shares
                        self.kv
                            .truncate(sp.meta.positions.first().copied().unwrap_or(0));
                    }
                    self.forward(ep, &sp.tokens, &sp.meta, f.header.tree_epoch);
                    if sp.prefill {
                        self.commit(&sp.meta.keys);
                    }
                }
                FrameType::TreeCommit => {
                    let c = TreeCommitFrame::decode(&f.payload);
                    if c.commit_gen == self.applied_commits {
                        self.commit(&c.accepted_keys());
                        // re-examine trees that were waiting for this commit
                        let d: Vec<TesseractFrame> = self.deferred.drain(..).collect();
                        self.deferred.extend(d);
                    }
                }
                FrameType::HeartbeatTelemetry => {
                    // calibration echo
                    let _ = ep.send_payload(
                        f.header.source,
                        FrameType::HeartbeatTelemetry,
                        f.header.tree_epoch,
                        1,
                        f.payload.clone(),
                    );
                }
                FrameType::BatchStep | FrameType::BatchCommit => self.batch_frame(ep, f),
                FrameType::Generate if self.a.tess.sessions > 0 => self.serve(ep, f),
                FrameType::Generate => {
                    let (req, ext) = GenerateRequest::decode_ext(&f.payload);
                    let r = self.generate(ep, &req, &ext);
                    let _ = ep.send_payload(DRIVER_ID, FrameType::GenerateResult, 0, 0, r.encode());
                }
                FrameType::StatsRequest => self.reply_stats_now(ep),
                FrameType::Reset | FrameType::Shutdown => return,
                _ => {}
            }
        }
    }

    fn reply_stats_now(&self, ep: &Endpoint) {
        let mut s = self.stats.clone();
        s.busy_ns = self.shaper.busy_ns;
        s.kv_bytes = (self.kv.bytes() + self.skv.iter().map(|k| k.bytes()).sum::<usize>()) as u64;
        s.act_bytes += self.reduce_bytes;
        for p in self.predictors.values() {
            s.lookahead_needed += p.needed;
            s.lookahead_covered += p.covered;
        }
        for st in &self.stores {
            let snap = st.stats.snapshot();
            for i in 0..6 {
                s.staging[i] += snap[i];
            }
        }
        reply_stats(ep, s);
    }

    // --- continuous batching (session slots) --------------------------------------

    fn slot_kv(&mut self, slot: usize) -> &mut LocalKvCache {
        while self.skv.len() <= slot {
            self.skv.push(
                LocalKvCache::new(self.kv.n_heads, self.kv.n_kv, self.a.model.d_head)
                    .with_exact(self.a.model.exact),
            );
        }
        &mut self.skv[slot]
    }

    /// Ranks: apply batched steps and commits strictly in issue order
    /// (step e, commit e, step e+1, …) whatever order they arrive in.
    fn batch_frame(&mut self, ep: &Endpoint, f: TesseractFrame) {
        let seq = f.header.tree_epoch - BATCH_EPOCH;
        let kind = (f.header.frame_type == FrameType::BatchCommit) as u8;
        self.batch_wait.insert((seq, kind), f);
        while let Some(f) = self.batch_wait.remove(&self.batch_next) {
            if self.batch_next.1 == 0 {
                let st = BatchStep::decode(&f.payload);
                self.batch_step(ep, &st, f.header.tree_epoch);
                self.batch_next.1 = 1;
            } else {
                self.batch_commit(&BatchCommit::decode(&f.payload));
                self.batch_next = (self.batch_next.0 + 1, 0);
            }
        }
    }

    /// One batched pass: slot resets, the forward over every segment, and
    /// the prefill segments' commits. Returns the final hidden rows.
    fn batch_step(&mut self, ep: &Endpoint, st: &BatchStep, epoch: u64) -> Tensor2D {
        for s in &st.segments {
            let kv = self.slot_kv(s.slot as usize);
            if let Some(r) = s.reset_to {
                kv.truncate(r as usize);
            }
        }
        let tokens = st.tokens();
        let positions = st.positions();
        let mut x = self.head.embed_tokens(&tokens);
        let all: Vec<usize> = (0..x.rows).collect();
        for li in 0..self.layers.len() {
            self.prewake(li, &x);
            let o = self.attention_segments(li, &x, &positions, st);
            let o = if self.attn_split {
                self.all_reduce(ep, &o, &all, epoch, 2 * li)
            } else {
                o.into_f32()
            };
            for (a, b) in x.data.iter_mut().zip(&o.data) {
                *a += b;
            }
            let f = self.ffn(li, &x);
            let f = self.all_reduce(ep, &f, &all, epoch, 2 * li + 1);
            for (a, b) in x.data.iter_mut().zip(&f.data) {
                *a += b;
            }
        }
        for s in &st.segments {
            if s.prefill {
                let mut k = s.meta.keys.clone();
                k.sort();
                self.slot_kv(s.slot as usize).commit(&k, &|_| false);
            }
        }
        x
    }

    fn batch_commit(&mut self, c: &BatchCommit) {
        for (slot, keys) in &c.slots {
            let mut k = keys.clone();
            k.sort();
            self.slot_kv(*slot as usize).commit(&k, &|_| false);
        }
        self.stats.commits += 1;
    }

    /// Attention over a batch of session segments: the projections run on
    /// all rows at once, attention per segment against its slot's KV.
    fn attention_segments(
        &mut self,
        li: usize,
        x: &Tensor2D,
        positions: &[usize],
        st: &BatchStep,
    ) -> Partial {
        let cfg = self.a.model.clone();
        let (m, dh) = (x.rows, cfg.d_head);
        let layer = &self.layers[li];
        let skv = &mut self.skv;
        let rope = &self.rope;
        self.shaper.run(|| {
            let mut xn = Tensor2D::zeros(m, cfg.d_model);
            for i in 0..m {
                rms_norm(x.row(i), &layer.attn_norm, cfg.rms_eps, xn.row_mut(i));
            }
            let (q, k, v) = crate::model::project_qkv(
                &layer.wq,
                &layer.wk,
                &layer.wv,
                &layer.extras,
                &xn,
                rope,
                positions,
                dh,
                cfg.rms_eps,
            );
            let mut att = Tensor2D::zeros(m, q.cols);
            let mut r0 = 0;
            for s in &st.segments {
                let n = s.tokens.len();
                if n == 0 {
                    continue;
                }
                let sub = |t: &Tensor2D| {
                    Tensor2D::from_vec(n, t.cols, t.data[r0 * t.cols..(r0 + n) * t.cols].to_vec())
                };
                let a =
                    skv[s.slot as usize].attend(layer.index, &sub(&q), &sub(&k), &sub(&v), &s.meta);
                att.data[r0 * a.cols..(r0 + n) * a.cols].copy_from_slice(&a.data);
                r0 += n;
            }
            row_parallel(&layer.wo, &att)
        })
    }

    fn broadcast_frame(&self, ep: &Endpoint, ft: FrameType, epoch: u64, payload: Vec<u8>) {
        for (r, &node) in self.ranks.iter().enumerate() {
            if r != self.rank {
                let _ = ep.send_payload(node, ft, epoch, 0, payload.clone());
            }
        }
    }

    /// Head: the continuous-batching scheduler. Serves `first` and every
    /// request that arrives meanwhile, up to `sessions` at a time, each step
    /// one batched forward pass over all active sessions (chunked prompt
    /// prefills and draft chains to verify). Returns when idle.
    fn serve(&mut self, ep: &Endpoint, first: TesseractFrame) {
        let cfg = self.a.model.clone();
        let n_slots = self.a.tess.sessions.max(1) as usize;
        if self.slot_seq.len() < n_slots {
            self.slot_seq.resize(n_slots, Vec::new());
            self.slot_stamp.resize(n_slots, 0);
        }
        if !self.draft_tried && !self.a.tess.draft_model.is_empty() {
            self.draft_tried = true;
            match DraftModel::load(&self.a.tess.draft_model, &cfg) {
                Ok(d) => self.draft_model = Some(Arc::new(d)),
                Err(e) => eprintln!("{e}; drafting from the context only"),
            }
        }
        let k_max = self.a.tess.spec_k_max.max(1) as usize;
        let spec = self.a.tess.budget != 1;
        let mut queue: VecDeque<(u64, GenerateRequest, GenerateExt)> = VecDeque::new();
        let take = |f: &TesseractFrame, q: &mut VecDeque<(u64, GenerateRequest, GenerateExt)>| {
            let (req, ext) = GenerateRequest::decode_ext(&f.payload);
            q.push_back((f.header.tree_epoch, req, ext));
        };
        take(&first, &mut queue);
        let mut active: Vec<Session> = Vec::new();
        let trace = std::env::var("TESSERACT_SCHED_TRACE").is_ok();
        // [steps, rows, decode rows, forward ns, logits ns, draft ns, verify+commit ns]
        let mut tr = [0u64; 9];
        loop {
            while let Some(f) = ep.try_recv_match(&|f| f.header.frame_type == FrameType::Generate) {
                take(&f, &mut queue);
            }
            // status requests are answered between steps
            while ep
                .try_recv_match(&|f| f.header.frame_type == FrameType::StatsRequest)
                .is_some()
            {
                self.reply_stats_now(ep);
            }
            // admit queued requests into free slots, preferring the slot
            // whose committed KV shares the longest prefix with the prompt
            while !queue.is_empty() {
                let free: Vec<usize> = (0..n_slots)
                    .filter(|s| !active.iter().any(|a| a.slot == *s))
                    .collect();
                if free.is_empty() {
                    break;
                }
                let (id, req, ext) = queue.pop_front().unwrap();
                let n = req.prompt.len();
                let slot = *free
                    .iter()
                    .max_by_key(|&&s| {
                        (
                            super::shared_prefix(&self.slot_seq[s], &req.prompt, n),
                            std::cmp::Reverse(self.slot_stamp[s]),
                        )
                    })
                    .unwrap();
                let p = super::shared_prefix(&self.slot_seq[slot], &req.prompt, n);
                self.slot_seq[slot].truncate(p);
                let mut sam = SuffixIndex::new();
                sam.extend(&req.prompt);
                active.push(Session {
                    id,
                    slot,
                    rng: Rng::new(req.seed),
                    res: GenerateResult {
                        prefix_reused: p as u32,
                        ..Default::default()
                    },
                    sam,
                    draft: DraftSession::default(),
                    ctl: SpecController::default(),
                    prefill_at: p,
                    reset_sent: false,
                    root: req.prompt[n - 1],
                    base_pos: n - 1,
                    t_admit: mono_ns(),
                    t_decode: 0,
                    req,
                    ext,
                });
            }
            if active.is_empty() {
                if trace && tr[0] > 0 {
                    eprintln!(
                        "sched: {} steps · {:.1} rows/step ({:.1} decode) · forward {:.2} ms · logits {:.2} ms · draft {:.2} ms · verify {:.2} ms per step · drafted {} accepted {} · fit {:?} t_draft {:.2} ms",
                        tr[0],
                        tr[1] as f64 / tr[0] as f64,
                        tr[2] as f64 / tr[0] as f64,
                        tr[3] as f64 / tr[0] as f64 / 1e6,
                        tr[4] as f64 / tr[0] as f64 / 1e6,
                        tr[5] as f64 / tr[0] as f64 / 1e6,
                        tr[6] as f64 / tr[0] as f64 / 1e6,
                        tr[7],
                        tr[8],
                        (1..=8).map(|r| self.step_cost.estimate(r).map(|t| (t / 1e6 * 10.0).round() / 10.0)).collect::<Vec<_>>(),
                        self.step_cost.t_draft_ns / 1e6
                    );
                }
                return;
            }
            // build the step
            self.batch_seq += 1;
            let epoch = BATCH_EPOCH + self.batch_seq;
            let t_step = mono_ns();
            let mut st = BatchStep::default();
            let mut plans: Vec<Option<StepPlan>> = Vec::with_capacity(active.len());
            // prefill chunks first, then one draft offer per decoding session
            let mut prefill_rows = 0;
            let mut offers = Vec::new();
            let mut offer_of = Vec::new();
            for (si, s) in active.iter_mut().enumerate() {
                let n = s.req.prompt.len();
                if s.prefill_at < n - 1 {
                    let reset_to = (!s.reset_sent).then_some(s.prefill_at as u32);
                    s.reset_sent = true;
                    let end = (s.prefill_at + PREFILL_CHUNK).min(n - 1);
                    let toks = s.req.prompt[s.prefill_at..end].to_vec();
                    prefill_rows += toks.len();
                    st.segments.push(Segment {
                        slot: s.slot as u16,
                        reset_to,
                        prefill: true,
                        meta: BatchMeta::causal_prefill(s.prefill_at, end - s.prefill_at, epoch),
                        tokens: toks.clone(),
                    });
                    s.prefill_at = end;
                    plans.push(Some(StepPlan::Prefill(toks)));
                    continue;
                }
                if s.t_decode == 0 {
                    s.t_decode = mono_ns();
                    s.res.prefill_ns = s.t_decode - s.t_admit;
                }
                plans.push(None);
                let room = (s.req.max_new as usize).saturating_sub(s.res.tokens.len());
                let cap = if spec {
                    k_max.min(room.saturating_sub(1))
                } else {
                    0
                };
                let (mlen, look) = s.sam.lookup(cap);
                // a context match of ≥ 3 tokens is a free, usually good draft;
                // otherwise prefer the companion model when there is one
                let model = self.draft_model.is_some();
                let lookup = if mlen >= 3 || !model {
                    look
                } else {
                    Vec::new()
                };
                offers.push(DraftOffer {
                    lookup,
                    alpha_lookup: s.ctl.alpha_lookup,
                    model,
                    alpha_model: s.ctl.alpha_model,
                    cap,
                });
                offer_of.push(si);
            }
            // a fixed tree budget (`budget > 1`) pins the chain length;
            // otherwise the cost model plans it
            let chains = if self.a.tess.budget > 1 {
                offers
                    .iter()
                    .map(|o| {
                        let k = (self.a.tess.budget as usize - 1).min(o.cap);
                        if o.lookup.len() >= k || !o.model {
                            (k.min(o.lookup.len()), false)
                        } else {
                            (k, true)
                        }
                    })
                    .collect()
            } else {
                plan_chains(&offers, &self.step_cost, prefill_rows)
            };
            let t_draft = mono_ns();
            let mut decode_rows = 0usize;
            for ((o, &(k, from_model)), &si) in offers.iter().zip(&chains).zip(&offer_of) {
                let s = &mut active[si];
                let drafts = if k == 0 {
                    Vec::new()
                } else if from_model {
                    let dm = self.draft_model.as_ref().unwrap();
                    let t0 = mono_ns();
                    let (d, fed) = dm.draft_counted(&mut s.draft, s.sam.tokens(), k);
                    // per drafted token: the chain's rows plus about one
                    // catch-up row per step (the accepted tokens arrive
                    // together, as one batched row block)
                    let per_row = (mono_ns() - t0) as f64 / fed.max(1) as f64;
                    if fed <= k + 8 {
                        self.step_cost.observe_draft(per_row);
                    }
                    d
                } else {
                    o.lookup[..k].to_vec()
                };
                let reset_to = (!s.reset_sent).then_some(s.prefill_at as u32);
                s.reset_sent = true;
                let tree = SpeculativeTree::chain(epoch, s.root, &drafts);
                let len = tree.len();
                let merged = MergedTree::split(tree, len, epoch, 1);
                st.segments.push(Segment {
                    slot: s.slot as u16,
                    reset_to,
                    prefill: false,
                    meta: merged.strand_meta(0, s.base_pos),
                    tokens: merged.strand_tokens(0),
                });
                plans[si] = Some(StepPlan::Decode {
                    merged: Box::new(merged),
                    drafted: drafts.len(),
                    from_model,
                    offset: decode_rows,
                });
                decode_rows += len;
            }
            // segments must follow the session order of `plans`
            let order: Vec<usize> = {
                let mut v: Vec<usize> = (0..active.len())
                    .filter(|&i| matches!(plans[i], Some(StepPlan::Prefill(_))))
                    .collect();
                v.extend(offer_of.iter().copied());
                v
            };
            let mut segs: Vec<Option<Segment>> = st.segments.drain(..).map(Some).collect();
            let mut by_session: Vec<Option<Segment>> = (0..active.len()).map(|_| None).collect();
            for (k, &si) in order.iter().enumerate() {
                by_session[si] = segs[k].take();
            }
            st.segments = by_session
                .into_iter()
                .map(|s| s.expect("one segment per session"))
                .collect();
            // decode offsets in final row order
            let mut off = 0;
            for p in plans.iter_mut() {
                if let Some(StepPlan::Decode { merged, offset, .. }) = p {
                    *offset = off;
                    off += merged.tree.len();
                }
            }
            let plans: Vec<StepPlan> = plans.into_iter().map(|p| p.unwrap()).collect();
            let _ = t_draft;
            let t_fwd = mono_ns();
            self.broadcast_frame(ep, FrameType::BatchStep, epoch, st.encode());
            let h = self.batch_step(ep, &st, epoch);
            let t_lg = mono_ns();
            // logits of every decode row in one batched LM-head pass
            let mut sel = Tensor2D::zeros(decode_rows, cfg.d_model);
            let mut r0 = 0;
            let mut o = 0;
            for seg in &st.segments {
                let n = seg.tokens.len();
                if !seg.prefill {
                    sel.data[o * cfg.d_model..(o + n) * cfg.d_model]
                        .copy_from_slice(&h.data[r0 * cfg.d_model..(r0 + n) * cfg.d_model]);
                    o += n;
                }
                r0 += n;
            }
            let head = &self.head;
            let logits = if decode_rows > 0 {
                self.shaper.run(|| head.logits(&sel, cfg.rms_eps))
            } else {
                Tensor2D::zeros(0, 0)
            };
            let t_ver = mono_ns();
            let mut commit = BatchCommit::default();
            let mut outcomes = Vec::with_capacity(active.len());
            for (s, plan) in active.iter_mut().zip(&plans) {
                match plan {
                    StepPlan::Prefill(toks) => {
                        self.slot_seq[s.slot].extend_from_slice(toks);
                        outcomes.push(None);
                    }
                    StepPlan::Decode { merged, offset, .. } => {
                        let temp = s.req.temperature;
                        let history: Vec<u32> = if s
                            .ext
                            .sampling
                            .as_ref()
                            .is_some_and(|p| p.repetition_penalty != 1.0)
                        {
                            s.sam.tokens().to_vec()
                        } else {
                            Vec::new()
                        };
                        let ext = &s.ext;
                        let out = crate::speculative::verify_tree(
                            &merged.tree,
                            &mut |i| {
                                if history.is_empty() {
                                    ext.probs(logits.row(offset + i), temp, &[])
                                } else {
                                    let mut hh = history.clone();
                                    hh.extend(super::tree_path_tokens(&merged.tree, i));
                                    ext.probs(logits.row(offset + i), temp, &hh)
                                }
                            },
                            &mut s.rng,
                        );
                        commit.slots.push((
                            s.slot as u16,
                            TreeCommitFrame::from_outcome(merged, &out, 0).accepted_keys(),
                        ));
                        outcomes.push(Some(out));
                    }
                }
            }
            self.broadcast_frame(ep, FrameType::BatchCommit, epoch, commit.encode());
            self.batch_commit(&commit);
            let rows = st.rows();
            // the forward + LM head + verification share of the step
            self.step_cost.observe(rows, (mono_ns() - t_fwd) as f64);
            tr[0] += 1;
            tr[1] += rows as u64;
            tr[2] += decode_rows as u64;
            tr[3] += t_lg - t_fwd;
            tr[4] += t_ver - t_lg;
            tr[5] += t_fwd - t_step;
            tr[6] += mono_ns() - t_ver;
            let now = mono_ns();
            let mut finished = Vec::new();
            for (i, (s, (plan, out))) in active
                .iter_mut()
                .zip(plans.iter().zip(outcomes))
                .enumerate()
            {
                let (
                    StepPlan::Decode {
                        merged,
                        drafted,
                        from_model,
                        offset,
                    },
                    Some(out),
                ) = (plan, out)
                else {
                    continue;
                };
                let mut committed = vec![s.root];
                committed.extend(out.path[1..].iter().map(|&j| merged.tree.tokens[j]));
                self.slot_seq[s.slot].extend_from_slice(&committed);
                let mut emitted = out.emitted(&merged.tree);
                s.sam.extend(&emitted);
                let n_emitted = emitted.len();
                let room = s.req.max_new as usize - s.res.tokens.len();
                let done = super::emit_step_for(
                    ep,
                    s.id,
                    &s.ext,
                    &mut emitted,
                    merged.tree.len() as u16,
                    now - s.t_decode,
                    s.res.prefill_ns,
                    room,
                );
                if s.req.record_logits {
                    for &pn in &out.path {
                        s.res.logits.push(logits.row(offset + pn).to_vec());
                    }
                }
                for &t in &emitted {
                    s.res.tokens.push(t);
                    s.res.token_times_ns.push(now - s.t_decode);
                }
                s.res.per_epoch.push(n_emitted as u32);
                s.res.budgets.push(merged.tree.len() as u16);
                s.ctl.record(*from_model, *drafted, out.path.len() - 1);
                tr[7] += *drafted as u64;
                tr[8] += (out.path.len() - 1) as u64;
                s.root = out.bonus;
                s.base_pos += out.path.len();
                if done || s.res.tokens.len() >= s.req.max_new as usize {
                    finished.push(i);
                }
            }
            for i in finished.into_iter().rev() {
                let mut s = active.swap_remove(i);
                s.res.logits.truncate(s.res.tokens.len());
                s.res.decode_ns = mono_ns() - s.t_decode;
                self.slot_stamp[s.slot] = self.batch_seq;
                let _ = ep.send_payload(
                    DRIVER_ID,
                    FrameType::GenerateResult,
                    s.id,
                    0,
                    s.res.encode(),
                );
            }
        }
    }

    fn commit(&mut self, keys: &[NodeKey]) {
        let mut k = keys.to_vec();
        k.sort();
        let t0 = mono_ns();
        self.kv.commit(&k, &|_| false);
        self.stats.commit_ns += mono_ns() - t0;
        self.stats.commits += 1;
        self.applied_commits += 1;
    }

    /// Sparse all-reduce of one layer's FFN partial: only rows this rank
    /// produced are shipped; everyone sums decoded partials in rank order.
    fn all_reduce(
        &mut self,
        ep: &Endpoint,
        partial: &Partial,
        rows: &[usize],
        epoch: u64,
        layer: usize,
    ) -> Tensor2D {
        let d = partial.cols();
        let mut w = ByteWriter::default();
        w.u16(rows.len() as u16);
        for &r in rows {
            w.u16(r as u16);
        }
        let (coding, coded) = match partial {
            Partial::Fixed { v, .. } => (
                ReduceCoding::Fixed,
                rows.iter()
                    .flat_map(|&r| v[r * d..(r + 1) * d].iter().flat_map(|x| x.to_le_bytes()))
                    .collect::<Vec<u8>>(),
            ),
            Partial::F32(t) => {
                let mut sub = Tensor2D::zeros(rows.len(), d);
                for (i, &r) in rows.iter().enumerate() {
                    sub.row_mut(i).copy_from_slice(t.row(r));
                }
                let c = match self.coding {
                    ReduceCoding::F16 => encode_fp16_rows(&sub),
                    ReduceCoding::Hrq => self.hrq.encode(&sub, &Tensor2D::zeros(sub.rows, d)).0,
                    _ => sub.data.iter().flat_map(|v| v.to_le_bytes()).collect(),
                };
                (
                    if self.coding == ReduceCoding::Hrq || self.coding == ReduceCoding::F16 {
                        self.coding
                    } else {
                        ReduceCoding::F32
                    },
                    c,
                )
            }
        };
        w.u8(coding as u8);
        w.bytes(&coded);
        let payload = w.into_inner();
        for (r, &node) in self.ranks.iter().enumerate() {
            if r != self.rank {
                let mut f = TesseractFrame::new(FrameType::AllReducePartial, 0, 0, payload.clone());
                f.header.tree_epoch = epoch;
                f.header.layer = layer as u16;
                f.header.head_group = self.rank as u16;
                self.reduce_bytes += f.wire_len() as u64;
                let _ = ep.send(node, f);
            }
        }
        let mut parts: Vec<Option<Vec<u8>>> = vec![None; self.world];
        parts[self.rank] = Some(payload);
        for _ in 0..self.world - 1 {
            let f = ep.recv_match(&|f| {
                f.header.frame_type == FrameType::AllReducePartial
                    && f.header.tree_epoch == epoch
                    && f.header.layer == layer as u16
            });
            parts[f.header.head_group as usize] = Some(f.payload);
        }
        self.barriers += 1;
        let m = partial.rows();
        let hrq = &mut self.hrq;
        self.shaper.run(|| {
            if coding == ReduceCoding::Fixed {
                // integer sums: any order gives the same bits
                let mut acc = vec![0i64; m * d];
                for p in parts.into_iter().flatten() {
                    let mut r = ByteReader::new(&p);
                    let n = r.u16() as usize;
                    let idx: Vec<usize> = (0..n).map(|_| r.u16() as usize).collect();
                    let _ = r.u8();
                    for (row, b) in idx.iter().zip(r.rest().chunks_exact(8 * d)) {
                        for (a, x) in acc[row * d..(row + 1) * d]
                            .iter_mut()
                            .zip(b.as_chunks::<8>().0)
                        {
                            *a += i64::from_le_bytes(*x);
                        }
                    }
                }
                return Tensor2D::from_vec(
                    m,
                    d,
                    acc.into_iter()
                        .map(crate::tensor::kquant::from_fix)
                        .collect(),
                );
            }
            let mut total = Tensor2D::zeros(m, d);
            for p in parts.into_iter().flatten() {
                let mut r = ByteReader::new(&p);
                let n = r.u16() as usize;
                let idx: Vec<usize> = (0..n).map(|_| r.u16() as usize).collect();
                let coding = r.u8();
                if n == 0 {
                    continue;
                }
                let dec = match coding {
                    0 => decode_fp16_rows(r.rest()),
                    2 => Tensor2D::from_vec(
                        n,
                        d,
                        r.rest()
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .map(|b| f32::from_le_bytes(*b))
                            .collect(),
                    ),
                    _ => hrq.decode(r.rest(), &Tensor2D::zeros(n, d)),
                };
                for (i, &row) in idx.iter().enumerate() {
                    for (t, v) in total.row_mut(row).iter_mut().zip(dec.row(i)) {
                        *t += v;
                    }
                }
            }
            total
        })
    }

    /// Attention for one layer on this rank (head slice or full).
    fn attention(&mut self, li: usize, x: &Tensor2D, meta: &BatchMeta) -> Partial {
        let cfg = self.a.model.clone();
        let (m, dh) = (x.rows, cfg.d_head);
        let layer = &self.layers[li];
        let kv = &mut self.kv;
        let rope = &self.rope;
        self.shaper.run(|| {
            let mut xn = Tensor2D::zeros(m, cfg.d_model);
            for i in 0..m {
                rms_norm(x.row(i), &layer.attn_norm, cfg.rms_eps, xn.row_mut(i));
            }
            let (q, k, v) = crate::model::project_qkv(
                &layer.wq,
                &layer.wk,
                &layer.wv,
                &layer.extras,
                &xn,
                rope,
                &meta.positions,
                dh,
                cfg.rms_eps,
            );
            let att = kv.attend(layer.index, &q, &k, &v, meta);
            row_parallel(&layer.wo, &att)
        })
    }

    /// FFN / MoE partial for one layer on this rank (expert hidden slices).
    fn ffn(&mut self, li: usize, x: &Tensor2D) -> Partial {
        let cfg = self.a.model.clone();
        let coalesce = self.a.tess.coalesce_experts;
        let layer = &self.layers[li];
        let mut used: Vec<Vec<usize>> = Vec::new();
        let used_ref = &mut used;
        let out = self.shaper.run(|| {
            let mut xn = Tensor2D::zeros(x.rows, cfg.d_model);
            for i in 0..x.rows {
                rms_norm(x.row(i), &layer.ffn_norm, cfg.rms_eps, xn.row_mut(i));
            }
            match (&layer.router, &layer.bank, &layer.dense) {
                (Some(router), Some(bank), _) => {
                    let routes = if coalesce && xn.rows > 1 {
                        route_coalesced(router, &xn, layer.top_k)
                    } else {
                        route_topk(router, &xn, layer.top_k)
                    };
                    *used_ref = routes.iter().map(|r| r.0.clone()).collect();
                    Partial::F32(moe_combine(bank, &xn, &routes))
                }
                (_, _, Some(e)) => row_parallel(&e.w_down, &e.hidden(&xn)),
                _ => Partial::F32(Tensor2D::zeros(x.rows, cfg.d_model)),
            }
        });
        // online calibration of the predictor that targeted this layer
        if let Some((h, pred)) = self.look_pending.remove(&li) {
            let hz = self.a.tess.lookahead_horizon as usize;
            if let Some(p) = self.predictors.get_mut(&(li - hz)) {
                p.calibrate(&h, &pred, &used);
            }
        }
        out
    }

    /// Lookahead: from the residual entering layer `li`, pre-stage the
    /// experts predicted for layer `li + ΔL` on the background stager.
    fn prewake(&mut self, li: usize, x: &Tensor2D) {
        let hz = self.a.tess.lookahead_horizon as usize;
        if let (Some(stager), Some(p)) = (self.stager.as_ref(), self.predictors.get_mut(&li)) {
            let experts = p.predict(x);
            stager.request(li + hz, experts.clone());
            self.stats.prewakes_sent += 1;
            self.look_pending.insert(li + hz, (x.clone(), experts));
        }
    }

    /// Run the full layer stack over a batch (every rank, in lock step):
    /// split attention → all-reduce (skipped when attention is replicated) →
    /// sliced FFN → all-reduce.
    fn forward(&mut self, ep: &Endpoint, tokens: &[u32], meta: &BatchMeta, epoch: u64) -> Tensor2D {
        let mut x = self.head.embed_tokens(tokens);
        let all: Vec<usize> = (0..x.rows).collect();
        for li in 0..self.layers.len() {
            self.prewake(li, &x);
            let o = self.attention(li, &x, meta);
            let o = if self.attn_split {
                self.all_reduce(ep, &o, &all, epoch, 2 * li)
            } else {
                o.into_f32()
            };
            for (a, b) in x.data.iter_mut().zip(&o.data) {
                *a += b;
            }
            let f = self.ffn(li, &x);
            let f = self.all_reduce(ep, &f, &all, epoch, 2 * li + 1);
            for (a, b) in x.data.iter_mut().zip(&f.data) {
                *a += b;
            }
        }
        x
    }

    /// Analytical epoch-time model T(N), calibrated once: rank 0's own
    /// per-epoch compute for N scratch rows (min of repeats, this rank's slice
    /// of every layer), the drafter's batched drafting time for N nodes, the
    /// measured echo RTT for every collective, and the coordinator's logits.
    fn calibrate(&mut self, ep: &Endpoint, candidates: &[usize], root: u32) -> Vec<f64> {
        let cfg = self.a.model.clone();
        let mut rtt = 0.0f64;
        for (r, &node) in self.ranks.iter().enumerate() {
            if r == self.rank {
                continue;
            }
            let mut best = f64::MAX;
            for i in 0..5u64 {
                let t0 = mono_ns();
                let _ = ep.send_payload(
                    node,
                    FrameType::HeartbeatTelemetry,
                    900_000 + i,
                    0,
                    vec![1; 8],
                );
                let _ = ep.recv_match(&|f| {
                    f.header.frame_type == FrameType::HeartbeatTelemetry
                        && f.header.tree_epoch == 900_000 + i
                });
                best = best.min((mono_ns() - t0) as f64);
            }
            rtt = rtt.max(best);
        }
        let collectives = cfg.n_layers * if self.attn_split { 2 } else { 1 };
        let slow = self.a.node(self.id).slowdown;
        let mut model = Vec::new();
        for &n in candidates {
            let toks: Vec<u32> = (0..n as u32).collect();
            let mut best = f64::MAX;
            for rep in 0..3u64 {
                let mut scratch = LocalKvCache::new(self.kv.n_heads, self.kv.n_kv, cfg.d_head);
                std::mem::swap(&mut scratch, &mut self.kv);
                let meta = BatchMeta::causal_prefill(0, n, 7_000_000 + 16 * n as u64 + rep);
                let busy0 = self.shaper.busy_ns;
                let t0 = mono_ns();
                let mut x = self.head.embed_tokens(&toks);
                for li in 0..self.layers.len() {
                    let o = self.attention(li, &x, &meta).into_f32();
                    for (a, b) in x.data.iter_mut().zip(&o.data) {
                        *a += b;
                    }
                    let f = self.ffn(li, &x).into_f32();
                    for (a, b) in x.data.iter_mut().zip(&f.data) {
                        *a += b;
                    }
                }
                std::hint::black_box(self.head.logits(&x, cfg.rms_eps));
                best = best.min((mono_ns() - t0) as f64);
                self.shaper.busy_ns = busy0;
                std::mem::swap(&mut scratch, &mut self.kv);
            }
            let draft = if n > 1 {
                let d = self.drafter.as_mut().unwrap();
                let mut b = f64::MAX;
                for _ in 0..2 {
                    let t0 = mono_ns();
                    std::hint::black_box(d.draft_bfs(0, root, n));
                    b = b.min((mono_ns() - t0) as f64);
                }
                b
            } else {
                0.0
            };
            // compute is stretched by this rank's slowdown (ranks are peers)
            model.push(best * slow + draft * slow + collectives as f64 * rtt / 2.0 + rtt);
        }
        model
    }

    fn broadcast(&self, ep: &Endpoint, ft: FrameType, epoch: u64, payload: &[u8]) {
        for (r, &node) in self.ranks.iter().enumerate() {
            if r != self.rank {
                let mut f = TesseractFrame::new(ft, 0, 0, payload.to_vec());
                f.header.tree_epoch = epoch;
                let _ = ep.send(node, f);
            }
        }
    }

    fn generate(
        &mut self,
        ep: &Endpoint,
        req: &GenerateRequest,
        ext: &GenerateExt,
    ) -> GenerateResult {
        let cfg = self.a.model.clone();
        let mut rng = Rng::new(req.seed);
        let mut res = GenerateResult::default();
        let n = req.prompt.len();
        let candidates: Vec<usize> = if self.a.tess.budget == 0 {
            vec![1, 2, 3, 4, 6]
        } else {
            vec![self.a.tess.budget as usize]
        };
        let mut ctl = RooflineController::new(candidates.clone());
        // prefill prompt[p..n-1], reusing the committed KV of the longest
        // prefix this prompt shares with the previous request (prefix cache)
        let t_pref = mono_ns();
        let p = super::shared_prefix(&self.seq, &req.prompt, n);
        res.prefix_reused = p as u32;
        self.kv.truncate(p);
        self.seq.truncate(p);
        let meta = BatchMeta::causal_prefill(p, n - 1 - p, 1);
        let sp = StrandPayload {
            commit_gen: 0,
            prefill: true,
            meta: meta.clone(),
            parents: vec![-1; n - 1 - p],
            tokens: req.prompt[p..n - 1].to_vec(),
            prev: Vec::new(),
            act: vec![],
        };
        self.broadcast(ep, FrameType::SpeculativeTreeBatch, 1, &sp.encode());
        self.forward(ep, &req.prompt[p..n - 1], &meta, 1);
        self.commit(&meta.keys);
        self.seq.extend_from_slice(&req.prompt[p..n - 1]);
        if let Some(d) = self.drafter.as_mut() {
            if self.a.tess.budget != 1 {
                d.reset_context();
                d.commit(&req.prompt[..n - 1], None);
            }
        }
        let mut drafter_pending: Vec<u32> = Vec::new();
        let mut root = req.prompt[n - 1];
        if candidates.len() > 1 {
            let model = self.calibrate(ep, &candidates, root);
            ctl.set_time_model(model);
        }
        res.prefill_ns = mono_ns() - t_pref;
        let mut base_pos = n - 1;
        let mut epoch = 2u64;
        let t0 = mono_ns();
        while (res.tokens.len() as u32) < req.max_new {
            let tc = mono_ns();
            let busy0 = self.shaper.busy_ns;
            let gen = self.applied_commits;
            let budget = ctl.choose();
            let spec = self.a.tess.budget != 1;
            let tree = if spec && budget > 1 {
                // lazy drafter sync: all tokens committed since the last tree go
                // through the early-exit layers in the same batched pass
                let d = self.drafter.as_mut().unwrap();
                let pending = std::mem::take(&mut drafter_pending);
                self.shaper
                    .run(|| d.commit_and_draft(&pending, epoch, root, budget))
            } else {
                SpeculativeTree::single(epoch, root)
            };
            let keep = tree.len();
            let merged = MergedTree::split(tree, keep, epoch, 1);
            let meta = merged.strand_meta(0, base_pos);
            let tokens = merged.strand_tokens(0);
            let sp = StrandPayload {
                commit_gen: gen,
                prefill: false,
                meta: meta.clone(),
                parents: vec![-1; tokens.len()],
                tokens: tokens.clone(),
                prev: Vec::new(),
                act: vec![],
            };
            self.broadcast(ep, FrameType::SpeculativeTreeBatch, epoch, &sp.encode());
            let h = self.forward(ep, &tokens, &meta, epoch);
            let head = &self.head;
            let logits = self.shaper.run(|| head.logits(&h, cfg.rms_eps));
            let temp = req.temperature;
            let history: Vec<u32> = if ext
                .sampling
                .as_ref()
                .is_some_and(|p| p.repetition_penalty != 1.0)
            {
                req.prompt.iter().chain(&res.tokens).copied().collect()
            } else {
                Vec::new()
            };
            let out = crate::speculative::verify_tree(
                &merged.tree,
                &mut |i| {
                    if history.is_empty() {
                        ext.probs(logits.row(i), temp, &[])
                    } else {
                        let mut h = history.clone();
                        h.extend(super::tree_path_tokens(&merged.tree, i));
                        ext.probs(logits.row(i), temp, &h)
                    }
                },
                &mut rng,
            );
            let commit = TreeCommitFrame::from_outcome(&merged, &out, gen);
            self.broadcast(ep, FrameType::TreeCommit, epoch, &commit.encode());
            self.commit(&commit.accepted_keys());
            let emitted = out.emitted(&merged.tree);
            if req.record_logits {
                for &pn in &out.path {
                    res.logits.push(logits.row(pn).to_vec());
                }
            }
            let mut committed = vec![root];
            committed.extend(out.path[1..].iter().map(|&i| merged.tree.tokens[i]));
            drafter_pending.extend_from_slice(&committed);
            self.seq.extend_from_slice(&committed);
            let now = mono_ns();
            let n_emitted = emitted.len();
            let mut emitted = emitted;
            let room = req.max_new as usize - res.tokens.len();
            let done = super::emit_step(
                ep,
                ext,
                &mut emitted,
                keep as u16,
                now - t0,
                res.prefill_ns,
                room,
            );
            for &t in &emitted {
                res.tokens.push(t);
                res.token_times_ns.push(now - t0);
            }
            res.per_epoch.push(n_emitted as u32);
            res.budgets.push(keep as u16);
            ctl.record(
                keep,
                &merged.tree.path_q,
                n_emitted,
                now - tc,
                self.shaper.busy_ns - busy0,
            );
            root = out.bonus;
            base_pos += out.path.len();
            epoch += 1;
            if done {
                break;
            }
        }
        res.logits.truncate(res.tokens.len());
        res.decode_ns = mono_ns() - t0;
        res
    }
}
