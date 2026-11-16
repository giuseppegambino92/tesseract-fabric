//! Faithful baseline engines running on the same runtime, model, transport
//! and link shapers as Tesseract-Fabric.
//!
//! * **Exo-Ring-Pipeline** — layer-sharded ring pipeline, one token per pass.
//!   Layers are partitioned proportional to node memory (exo's default
//!   memory-weighted ring partitioning); uncompressed FP16 activations hop
//!   stage→stage; the last stage samples and sends the token back to stage 0.
//!   KV is coupled to the layer's stage.
//! * **RDMA-Tensor-Parallel** — Megatron-style tensor parallelism on every
//!   node: column-parallel Q/K/V and expert up/gate, row-parallel O and expert
//!   down, i.e. two synchronous FP16 AllReduce barriers per layer per token
//!   (latency-optimal one-shot all-to-all reduce, summed in rank order so all
//!   ranks stay bit-identical), plus a token broadcast from rank 0.

use super::{
    ack_assign, reply_stats, Assignment, ComputeShaper, GenerateRequest, GenerateResult, NodeStats,
    DRIVER_ID,
};
use crate::compression::{decode_fp16_rows, encode_fp16_rows};
use crate::kv_fabric::{AttentionBackend, BatchMeta, LocalKvCache};
use crate::model::{
    moe_combine, probs_from_logits, route_topk, sample, Expert, ExpertBank, FfnWeights,
    HeadWeights, LayerWeights, NoObserver, ShardModel,
};
use crate::routing::ExpertStore;
use crate::tensor::{rms_norm, Linear, Rng, RopeTable, Tensor2D};
use crate::wire::protocol::{ByteReader, ByteWriter, FrameType, TesseractFrame};
use crate::wire::transport::{mono_ns, Endpoint};
use std::collections::VecDeque;
use std::sync::Arc;

fn staged_stores(shard: &ShardModel) -> Vec<Arc<ExpertStore>> {
    shard
        .layers
        .iter()
        .filter_map(|l| match &l.ffn {
            FfnWeights::Moe {
                bank: ExpertBank::Staged(s),
                ..
            } => Some(s.clone()),
            _ => None,
        })
        .collect()
}

fn add_staging(s: &mut NodeStats, stores: &[Arc<ExpertStore>]) {
    for st in stores {
        let snap = st.stats.snapshot();
        for i in 0..6 {
            s.staging[i] += snap[i];
        }
    }
}

/// Context injection for a layer-coupled KV cache (same approximate prefill
/// as the Tesseract stages and KV services).
fn inject(kv: &mut LocalKvCache, a: &Assignment, shard: &ShardModel) {
    if a.inject_context == 0 {
        return;
    }
    let embed = HeadWeights::generate(&a.model).embed;
    for layer in &shard.layers {
        let (k, v) = super::node::context_kv(
            &a.model,
            layer,
            &shard.rope,
            &embed,
            0..a.inject_context as usize,
        );
        kv.inject_committed(layer.index, &k.data, &v.data, a.inject_context as usize);
    }
}

// ---------------------------------------------------------------------------
// Baseline A: Exo-Ring-Pipeline
// ---------------------------------------------------------------------------

pub struct ExoRingNode {
    a: Assignment,
    id: u16,
    idx: usize,
    shard: ShardModel,
    head: Option<HeadWeights>,
    kv: LocalKvCache,
    shaper: ComputeShaper,
    stores: Vec<Arc<ExpertStore>>,
    local: VecDeque<TesseractFrame>,
    last_logits: Option<Vec<f32>>,
    stats: NodeStats,
    epoch: u64,
}

impl ExoRingNode {
    pub fn new(ep: &Endpoint, a: &Assignment) -> Self {
        crate::tensor::prefer_performance_cores();
        crate::tensor::pool::install(a.tess.node_lanes as usize);
        let id = ep.id();
        let idx = a
            .stages
            .iter()
            .position(|s| s.0 == id)
            .expect("node not in ring");
        let (_, s, e) = a.stages[idx];
        let hot = if a.hot_capacity > 0 {
            Some(a.hot_capacity as usize)
        } else {
            None
        };
        let shard = ShardModel::generate(&a.model, s..e, hot);
        let head = if idx == 0 || id == a.head {
            Some(HeadWeights::generate(&a.model))
        } else {
            None
        };
        let mut kv = LocalKvCache::new(a.model.n_heads, a.model.n_kv_heads, a.model.d_head);
        inject(&mut kv, a, &shard);
        let stores = staged_stores(&shard);
        let weight_bytes = ((e - s) * a.model.layer_params() * 4) as u64;
        ack_assign(ep);
        ExoRingNode {
            a: a.clone(),
            id,
            idx,
            shard,
            head,
            kv,
            shaper: ComputeShaper::new(a.node(id).slowdown),
            stores,
            local: VecDeque::new(),
            last_logits: None,
            stats: NodeStats {
                weight_bytes,
                ..Default::default()
            },
            epoch: 0,
        }
    }

    fn send_to(&mut self, ep: &Endpoint, to: u16, f: TesseractFrame) {
        if to == self.id {
            self.local.push_back(f);
        } else {
            let _ = ep.send(to, f);
        }
    }

    fn next(&mut self, ep: &Endpoint) -> TesseractFrame {
        self.local.pop_front().unwrap_or_else(|| ep.recv())
    }

    pub fn run(&mut self, ep: &Endpoint) {
        loop {
            let f = self.next(ep);
            if !self.handle(ep, f) {
                return;
            }
        }
    }

    fn handle(&mut self, ep: &Endpoint, f: TesseractFrame) -> bool {
        match f.header.frame_type {
            FrameType::TokenBroadcast | FrameType::RawActivation => self.on_activation(ep, f),
            FrameType::Generate => {
                let req = GenerateRequest::decode(&f.payload);
                let r = self.generate(ep, &req);
                let _ = ep.send_payload(DRIVER_ID, FrameType::GenerateResult, 0, 0, r.encode());
            }
            FrameType::StatsRequest => {
                let mut s = self.stats.clone();
                s.busy_ns = self.shaper.busy_ns;
                s.kv_bytes = self.kv.bytes() as u64;
                add_staging(&mut s, &self.stores);
                reply_stats(ep, s);
            }
            FrameType::Reset | FrameType::Shutdown => return false,
            _ => {}
        }
        true
    }

    fn on_activation(&mut self, ep: &Endpoint, f: TesseractFrame) {
        let mut r = ByteReader::new(&f.payload);
        let ml = r.u32() as usize;
        let (meta, _) = BatchMeta::decode(r.bytes(ml));
        let mut x = if f.header.frame_type == FrameType::TokenBroadcast {
            let n = r.u32() as usize;
            let toks: Vec<u32> = (0..n).map(|_| r.u32()).collect();
            self.head.as_ref().unwrap().embed_tokens(&toks)
        } else {
            decode_fp16_rows(r.rest())
        };
        let shard = &self.shard;
        let kv = &mut self.kv;
        self.shaper
            .run(|| shard.forward(&mut x, &meta, kv, &mut NoObserver));
        let t0 = mono_ns();
        self.kv.commit(&meta.keys, &|_| false);
        self.stats.commit_ns += mono_ns() - t0;
        self.stats.commits += 1;
        if self.idx + 1 < self.a.stages.len() {
            let next = self.a.stages[self.idx + 1].0;
            let mut w = ByteWriter::default();
            let mut m = Vec::new();
            meta.encode(&mut m);
            w.u32(m.len() as u32);
            w.bytes(&m);
            w.bytes(&encode_fp16_rows(&x));
            let mut nf = TesseractFrame::new(FrameType::RawActivation, 0, 0, w.into_inner());
            nf.header.tree_epoch = f.header.tree_epoch;
            nf.header.flags = crate::wire::protocol::flags::FP16_PAYLOAD;
            self.send_to(ep, next, nf);
        } else {
            // last stage: logits for the final row
            let last = Tensor2D::from_vec(1, x.cols, x.row(x.rows - 1).to_vec());
            let head = self.head.as_ref().unwrap();
            let eps = self.a.model.rms_eps;
            let l = self.shaper.run(|| head.logits(&last, eps));
            self.last_logits = Some(l.data);
        }
    }

    fn token_frame(&mut self, tokens: &[u32], meta: &BatchMeta) -> TesseractFrame {
        let mut w = ByteWriter::default();
        let mut m = Vec::new();
        meta.encode(&mut m);
        w.u32(m.len() as u32);
        w.bytes(&m);
        w.u32(tokens.len() as u32);
        for &t in tokens {
            w.u32(t);
        }
        self.epoch += 1;
        let mut f = TesseractFrame::new(FrameType::TokenBroadcast, 0, 0, w.into_inner());
        f.header.tree_epoch = self.epoch;
        f
    }

    fn wait_logits(&mut self, ep: &Endpoint) -> Vec<f32> {
        while self.last_logits.is_none() {
            let f = self.next(ep);
            self.handle(ep, f);
        }
        self.last_logits.take().unwrap()
    }

    fn generate(&mut self, ep: &Endpoint, req: &GenerateRequest) -> GenerateResult {
        let mut rng = Rng::new(req.seed);
        let mut res = GenerateResult::default();
        let stage0 = self.a.stages[0].0;
        let base = self.a.inject_context as usize;
        let n = req.prompt.len();
        let t_pref = mono_ns();
        let meta = BatchMeta::causal_prefill(base, n, 1);
        let f = self.token_frame(&req.prompt, &meta);
        self.send_to(ep, stage0, f);
        let mut logits = self.wait_logits(ep);
        res.prefill_ns = mono_ns() - t_pref;
        let t0 = mono_ns();
        let mut pos = base + n;
        while (res.tokens.len() as u32) < req.max_new {
            let p = probs_from_logits(&logits, req.temperature);
            let tok = sample(&p, &mut rng);
            if req.record_logits {
                res.logits.push(logits.clone());
            }
            res.tokens.push(tok);
            res.token_times_ns.push(mono_ns() - t0);
            res.per_epoch.push(1);
            if res.tokens.len() as u32 == req.max_new {
                break;
            }
            let meta = BatchMeta::causal_prefill(pos, 1, 2 + pos as u64);
            let f = self.token_frame(&[tok], &meta);
            self.send_to(ep, stage0, f);
            logits = self.wait_logits(ep);
            pos += 1;
        }
        res.decode_ns = mono_ns() - t0;
        res
    }
}

// ---------------------------------------------------------------------------
// Baseline B: RDMA-Tensor-Parallel
// ---------------------------------------------------------------------------

enum TpFfn {
    Dense(Expert),
    Moe {
        router: Linear,
        bank: ExpertBank,
        top_k: usize,
    },
}

struct TpLayer {
    index: usize,
    extras: crate::model::AttnExtras,
    attn_norm: Vec<f32>,
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    ffn_norm: Vec<f32>,
    ffn: TpFfn,
}

pub struct TensorParallelNode {
    a: Assignment,
    rank: usize,
    world: usize,
    ranks: Vec<u16>,
    layers: Vec<TpLayer>,
    head: HeadWeights,
    rope: RopeTable,
    kv: LocalKvCache,
    shaper: ComputeShaper,
    stores: Vec<Arc<ExpertStore>>,
    stats: NodeStats,
    step: u64,
    barriers: u64,
}

impl TensorParallelNode {
    pub fn new(ep: &Endpoint, a: &Assignment) -> Self {
        crate::tensor::prefer_performance_cores();
        crate::tensor::pool::install(a.tess.node_lanes as usize);
        let cfg = &a.model;
        let ranks: Vec<u16> = a.stages.iter().map(|s| s.0).collect();
        let world = ranks.len();
        let rank = ranks
            .iter()
            .position(|&r| r == ep.id())
            .expect("node not a TP rank");
        assert!(
            cfg.n_heads.is_multiple_of(world) && cfg.n_kv_heads.is_multiple_of(world),
            "heads must divide world size"
        );
        let hl = cfg.n_heads / world;
        let kl = cfg.n_kv_heads / world;
        let dh = cfg.d_head;
        let hot = a.hot_capacity as usize;
        let mut stores = Vec::new();
        let mut weight_bytes = 0usize;
        let layers: Vec<TpLayer> = (0..cfg.n_layers)
            .map(|l| {
                let full = LayerWeights::generate(cfg, l, None);
                let ffn = match full.ffn {
                    FfnWeights::Dense(e) => {
                        let (h0, h1) = e.hidden_range(rank, world);
                        TpFfn::Dense(e.slice_hidden(h0, h1))
                    }
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
                        let bank = if hot > 0 && hot < m.n_experts && !cfg.source.is_file() {
                            let s = Arc::new(ExpertStore::new(l, &sliced, hot));
                            stores.push(s.clone());
                            ExpertBank::Staged(s)
                        } else {
                            ExpertBank::Resident(sliced.into_iter().map(Arc::new).collect())
                        };
                        TpFfn::Moe {
                            router,
                            bank,
                            top_k,
                        }
                    }
                };
                weight_bytes += cfg.layer_params() * 4 / world;
                TpLayer {
                    index: l,
                    extras: full.extras.slice_heads(
                        rank * hl,
                        (rank + 1) * hl,
                        rank * kl,
                        (rank + 1) * kl,
                        dh,
                    ),
                    attn_norm: full.attn_norm,
                    wq: full.wq.slice_rows(rank * hl * dh, (rank + 1) * hl * dh),
                    wk: full.wk.slice_rows(rank * kl * dh, (rank + 1) * kl * dh),
                    wv: full.wv.slice_rows(rank * kl * dh, (rank + 1) * kl * dh),
                    wo: full.wo.slice_cols(rank * hl * dh, (rank + 1) * hl * dh),
                    ffn_norm: full.ffn_norm,
                    ffn,
                }
            })
            .collect();
        let mut kv = LocalKvCache::new(hl, kl, dh);
        if a.inject_context > 0 {
            let embed = HeadWeights::generate(cfg).embed;
            let rope = cfg.rope_table();
            for l in 0..cfg.n_layers {
                let full = LayerWeights::generate(
                    &crate::model::ModelConfig {
                        moe: None,
                        d_ff: 32,
                        ..cfg.clone()
                    },
                    l,
                    None,
                );
                let (k, v) = super::node::context_kv(
                    cfg,
                    &full,
                    &rope,
                    &embed,
                    0..a.inject_context as usize,
                );
                let (kk, vv) = (
                    Linear::new(k).slice_cols(rank * kl * dh, (rank + 1) * kl * dh),
                    Linear::new(v).slice_cols(rank * kl * dh, (rank + 1) * kl * dh),
                );
                kv.inject_committed(l, &kk.w.data, &vv.w.data, a.inject_context as usize);
            }
        }
        ack_assign(ep);
        TensorParallelNode {
            a: a.clone(),
            rank,
            world,
            ranks,
            layers,
            head: HeadWeights::generate(cfg),
            rope: cfg.rope_table(),
            kv,
            shaper: ComputeShaper::new(a.node(ep.id()).slowdown),
            stores,
            stats: NodeStats {
                weight_bytes: weight_bytes as u64,
                ..Default::default()
            },
            step: 0,
            barriers: 0,
        }
    }

    /// One-shot AllReduce: send FP16 partial to every rank, sum in rank order.
    fn all_reduce(&mut self, ep: &Endpoint, x: &mut Tensor2D, tag: u16) {
        let payload = encode_fp16_rows(x);
        for (r, &node) in self.ranks.iter().enumerate() {
            if r != self.rank {
                let mut f = TesseractFrame::new(FrameType::AllReducePartial, 0, 0, payload.clone());
                f.header.tree_epoch = self.step;
                f.header.layer = tag;
                f.header.head_group = self.rank as u16;
                let _ = ep.send(node, f);
            }
        }
        let mut parts: Vec<Option<Tensor2D>> = (0..self.world).map(|_| None).collect();
        parts[self.rank] = Some(decode_fp16_rows(&payload));
        for _ in 0..self.world - 1 {
            let step = self.step;
            let f = ep.recv_match(&|f| {
                f.header.frame_type == FrameType::AllReducePartial
                    && f.header.tree_epoch == step
                    && f.header.layer == tag
            });
            parts[f.header.head_group as usize] = Some(decode_fp16_rows(&f.payload));
        }
        self.barriers += 1;
        let shaper = &mut self.shaper;
        shaper.run(|| {
            x.data.iter_mut().for_each(|v| *v = 0.0);
            for p in parts.into_iter().flatten() {
                for (a, b) in x.data.iter_mut().zip(&p.data) {
                    *a += b;
                }
            }
        });
    }

    fn forward(&mut self, ep: &Endpoint, tokens: &[u32], meta: &BatchMeta) -> Tensor2D {
        let cfg = self.a.model.clone();
        let mut x = self.head.embed_tokens(tokens);
        let m = x.rows;
        let dh = cfg.d_head;
        for li in 0..self.layers.len() {
            // attention (column-parallel QKV, row-parallel O)
            let mut partial = {
                let layer = &self.layers[li];
                let kv = &mut self.kv;
                let rope = &self.rope;
                let xr = &x;
                self.shaper.run(|| {
                    let mut xn = Tensor2D::zeros(m, cfg.d_model);
                    for i in 0..m {
                        rms_norm(xr.row(i), &layer.attn_norm, cfg.rms_eps, xn.row_mut(i));
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
                    let a = kv.attend(layer.index, &q, &k, &v, meta);
                    layer.wo.forward(&a)
                })
            };
            self.all_reduce(ep, &mut partial, (li * 2) as u16);
            for (a, b) in x.data.iter_mut().zip(&partial.data) {
                *a += b;
            }
            // FFN (column-parallel gate/up, row-parallel down)
            let mut partial = {
                let layer = &self.layers[li];
                let xr = &x;
                self.shaper.run(|| {
                    let mut xn = Tensor2D::zeros(m, cfg.d_model);
                    for i in 0..m {
                        rms_norm(xr.row(i), &layer.ffn_norm, cfg.rms_eps, xn.row_mut(i));
                    }
                    match &layer.ffn {
                        TpFfn::Dense(e) => e.forward(&xn),
                        TpFfn::Moe {
                            router,
                            bank,
                            top_k,
                        } => {
                            let routes = route_topk(router, &xn, *top_k);
                            moe_combine(bank, &xn, &routes)
                        }
                    }
                })
            };
            self.all_reduce(ep, &mut partial, (li * 2 + 1) as u16);
            for (a, b) in x.data.iter_mut().zip(&partial.data) {
                *a += b;
            }
        }
        let t0 = mono_ns();
        self.kv.commit(&meta.keys, &|_| false);
        self.stats.commit_ns += mono_ns() - t0;
        self.stats.commits += 1;
        x
    }

    fn broadcast_tokens(&mut self, ep: &Endpoint, tokens: &[u32], meta: &BatchMeta) {
        let mut w = ByteWriter::default();
        let mut mb = Vec::new();
        meta.encode(&mut mb);
        w.u32(mb.len() as u32);
        w.bytes(&mb);
        w.u32(tokens.len() as u32);
        for &t in tokens {
            w.u32(t);
        }
        let payload = w.into_inner();
        for (r, &node) in self.ranks.iter().enumerate() {
            if r != self.rank {
                let mut f = TesseractFrame::new(FrameType::TokenBroadcast, 0, 0, payload.clone());
                f.header.tree_epoch = self.step;
                let _ = ep.send(node, f);
            }
        }
    }

    pub fn run(&mut self, ep: &Endpoint) {
        loop {
            // Only consume control frames here: AllReduce partials of a step
            // this rank has not started yet must stay queued for all_reduce().
            let f = ep.recv_match(&|f| f.header.frame_type != FrameType::AllReducePartial);
            match f.header.frame_type {
                FrameType::TokenBroadcast => {
                    let mut r = ByteReader::new(&f.payload);
                    let ml = r.u32() as usize;
                    let (meta, _) = BatchMeta::decode(r.bytes(ml));
                    let n = r.u32() as usize;
                    let toks: Vec<u32> = (0..n).map(|_| r.u32()).collect();
                    self.step = f.header.tree_epoch;
                    self.forward(ep, &toks, &meta);
                }
                FrameType::Generate => {
                    let req = GenerateRequest::decode(&f.payload);
                    let r = self.generate(ep, &req);
                    let _ = ep.send_payload(DRIVER_ID, FrameType::GenerateResult, 0, 0, r.encode());
                }
                FrameType::StatsRequest => {
                    let mut s = self.stats.clone();
                    s.busy_ns = self.shaper.busy_ns;
                    s.kv_bytes = self.kv.bytes() as u64;
                    add_staging(&mut s, &self.stores);
                    reply_stats(ep, s);
                }
                FrameType::Reset | FrameType::Shutdown => return,
                _ => {}
            }
        }
    }

    fn generate(&mut self, ep: &Endpoint, req: &GenerateRequest) -> GenerateResult {
        let mut rng = Rng::new(req.seed);
        let mut res = GenerateResult::default();
        let base = self.a.inject_context as usize;
        let n = req.prompt.len();
        let t_pref = mono_ns();
        self.step += 1;
        let meta = BatchMeta::causal_prefill(base, n, 1);
        self.broadcast_tokens(ep, &req.prompt, &meta);
        let x = self.forward(ep, &req.prompt, &meta);
        let eps = self.a.model.rms_eps;
        let head = &self.head;
        let mut logits = self
            .shaper
            .run(|| head.logits(&Tensor2D::from_vec(1, x.cols, x.row(n - 1).to_vec()), eps))
            .data;
        res.prefill_ns = mono_ns() - t_pref;
        let t0 = mono_ns();
        let mut pos = base + n;
        while (res.tokens.len() as u32) < req.max_new {
            let p = probs_from_logits(&logits, req.temperature);
            let tok = sample(&p, &mut rng);
            if req.record_logits {
                res.logits.push(logits.clone());
            }
            res.tokens.push(tok);
            res.token_times_ns.push(mono_ns() - t0);
            res.per_epoch.push(1);
            if res.tokens.len() as u32 == req.max_new {
                break;
            }
            self.step += 1;
            let meta = BatchMeta::causal_prefill(pos, 1, 2 + pos as u64);
            self.broadcast_tokens(ep, &[tok], &meta);
            let x = self.forward(ep, &[tok], &meta);
            let head = &self.head;
            logits = self.shaper.run(|| head.logits(&x, eps)).data;
            pos += 1;
        }
        res.decode_ns = mono_ns() - t0;
        res
    }
}
