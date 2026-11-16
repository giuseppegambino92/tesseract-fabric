//! Live multi-node cluster runtime.
//!
//! A *driver* endpoint assigns roles to worker nodes with an `Assign` frame,
//! asks the head node to `Generate`, and finally collects per-node `Stats`.
//! Workers are ordinary TCP endpoints — threads in the benchmark, separate
//! processes (or machines) with `tesseract-node`. All engine traffic is
//! `TesseractWire` frames over link-shaped sockets.

pub mod baselines;
pub mod batch;
pub mod discovery;
pub mod htep;
pub mod node;

use crate::model::ModelConfig;
use crate::wire::multipath::LinkProfile;
use crate::wire::protocol::{ByteReader, ByteWriter, FrameType};
use crate::wire::transport::{mono_ns, wait_until_ns, Endpoint};
use std::net::SocketAddr;
use std::time::Duration;

pub const DRIVER_ID: u16 = 0xFFF0;
/// Endpoint id offsets for per-node auxiliary services.
pub const KV_SERVICE_BASE: u16 = 0x4000;
pub const DRAFT_SERVICE_BASE: u16 = 0x2000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineKind {
    ExoRing = 1,
    TensorParallel = 2,
    Tesseract = 3,
    /// Tesseract with hybrid tensor-expert parallelism over a low-latency group.
    TesseractHtep = 4,
}

impl EngineKind {
    pub fn name(&self) -> &'static str {
        match self {
            EngineKind::ExoRing => "Exo-Ring-Pipeline",
            EngineKind::TensorParallel => "RDMA-Tensor-Parallel",
            EngineKind::Tesseract => "Tesseract-Fabric",
            EngineKind::TesseractHtep => "Tesseract-Fabric (H-TEP)",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvMode {
    /// Committed KV lives with the stage that computes the layer.
    LayerCoupled = 0,
    /// Committed KV is paged across KV services on all nodes (LSE merge).
    Disaggregated = 1,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TesseractOptions {
    /// Inter-stage activation codec: 0 = lossless raw f32 (bit-identical to
    /// one node), 1 = V1 holographic, 2 = HRQ-V2 trunk residual, 3 = HRQ-V2.1
    /// entropy-coded lattice, 4 = HRQ-V5 conditional KLT.
    pub codec: u8,
    /// Per-hop codec choice: ship FP16 on links where the codec's CPU time
    /// exceeds the transfer time it saves (false = always use `codec`).
    pub codec_auto: bool,
    pub holo_bits: u8,
    /// Tree budget (nodes incl. root); 0 = autotune from telemetry.
    pub budget: u16,
    pub braid_strands: u8,
    pub lookahead: bool,
    pub lookahead_horizon: u8,
    pub lookahead_margin: u8,
    pub deterministic_boundary: bool,
    pub kv_mode: KvMode,
    pub page_tokens: u32,
    /// Node hosting the asynchronous draft service (None = coordinator-local thread).
    pub drafter: Option<u16>,
    pub draft_rank: u16,
    /// Pre-drafted continuation outcomes requested per cycle.
    pub draft_ahead: u8,
    /// Draft service model: 0 = low-rank zero-layer head, k = self-speculative
    /// early exit after the target's first k layers.
    pub draft_layers: u8,
    /// Disaggregated-attention wire format: 0 = f32, 1 = V3 f16 compact,
    /// 2 = V4 block floating point with rigorous merge-weight bounds.
    pub kv_wire: u8,
    /// H-TEP all-reduce coding: false = FP16, true = HRQ trunk codec.
    pub htep_hrq: bool,
    /// H-TEP attention: true = head-sharded (2 collectives/layer), false =
    /// replicated (1 collective/layer, more compute per rank).
    pub htep_attn_split: bool,
    /// Disaggregated KV tier 0: pages per layer kept with the computing stage
    /// before striping across nodes (0 = stripe everything).
    pub tier0_pages: u16,
    /// Disaggregated KV beyond tier 0: pair each layer with one partner node
    /// (one remote KV service per layer and step) instead of striping
    /// across every node.
    pub kv_paired: bool,
    /// Paired placement: the partner holds this many fewer pages of a layer
    /// than the computing node, so both finish together despite the remote
    /// round trip (calibrated from `kv_wait_ns` / per-page scan cost).
    pub kv_remote_bias_pages: u16,
    /// Stripe disaggregated KV by KV-head group (virtual layer per group):
    /// each remote holder receives only its group's query heads.
    pub kv_group_striped: bool,
    /// Mirror tier-0 prefix pages on remote holders and delegate them in
    /// the long-context regime (head-group striping only).
    pub kv_prefix_mirror: bool,
    /// Speculate only while all KV is local: with remote KV every tree row
    /// multiplies query traffic to every holder, so steps stay single-row.
    pub spec_local_kv_only: bool,
    /// Lanes per node for the MoE expert path, installed identically on the
    /// node threads of every engine (Exo, TP, H-TEP, pipeline stages).
    pub node_lanes: u8,
    /// Cross-step stage-0 lookahead (speculation on, stage 0 the pipeline
    /// bottleneck): stage 0 precomputes its layers for the predicted next
    /// token while the later stages finish the current one.
    pub stage_lookahead: bool,
    /// H-TEP all-reduce in lossless f32 (overrides `htep_hrq`): ranks then
    /// differ from a single node only in f32 summation order. Off by default
    /// (the V7 benchmark rows use FP16); real-checkpoint serving turns it on.
    pub htep_reduce_f32: bool,
    /// Prompt-lookup drafting (n-gram continuation from the context) instead
    /// of the early-exit drafter: real checkpoints' early layers do not
    /// predict their output without trained exit heads.
    pub draft_lookup: bool,
    /// APPROXIMATE anchor-consensus expert coalescing in tree verification.
    pub coalesce_experts: bool,
    /// Draft with attention-only early-exit layers (FFN skipped).
    pub draft_attn_only: bool,
    /// AWQ-style static channel salience pre-scaling in the HRQ codec.
    pub salience: bool,
    /// HRQ sensitivity-subspace rank (0 = off): top-k readout directions sent as BFP16.
    pub sens_rank: u16,
    /// H-TEP session slots for continuous batching (0 = the classic
    /// single-request coordinator). With N > 0 the head interleaves up to N
    /// concurrent requests in shared forward passes, each with its own KV.
    pub sessions: u8,
    /// Companion draft checkpoint (same vocabulary) loaded by the H-TEP head
    /// for speculative chains; empty = suffix-automaton drafting only.
    pub draft_model: String,
    /// Longest draft chain of the session scheduler.
    pub spec_k_max: u8,
}

impl TesseractOptions {
    /// Adopt a solver plan's drafter. When it runs on another host than the
    /// coordinator, enable draft-ahead (2 predicted outcomes) so it drafts
    /// the next tree while the current one is verified, off the critical
    /// path; on the coordinator itself drafting stays local and lazy.
    pub fn use_plan_drafter(&mut self, drafter: Option<u16>, coordinator: u16) {
        self.drafter = drafter;
        if drafter.is_some_and(|d| d != coordinator) && self.budget != 1 && self.draft_ahead == 0 {
            self.draft_ahead = 2;
        }
    }
}

impl Default for TesseractOptions {
    fn default() -> Self {
        TesseractOptions {
            codec: 4,
            codec_auto: true,
            holo_bits: 3,
            budget: 0,
            braid_strands: 2,
            lookahead: true,
            lookahead_horizon: 3,
            lookahead_margin: 1,
            deterministic_boundary: true,
            kv_mode: KvMode::LayerCoupled,
            page_tokens: 256,
            drafter: None,
            draft_rank: 0,
            draft_ahead: 0,
            draft_layers: 1,
            kv_wire: 2,
            htep_hrq: false,
            htep_attn_split: true,
            tier0_pages: 12,
            kv_paired: true,
            kv_remote_bias_pages: 0,
            kv_group_striped: true,
            kv_prefix_mirror: true,
            spec_local_kv_only: true,
            node_lanes: 2,
            stage_lookahead: true,
            htep_reduce_f32: false,
            draft_lookup: false,
            coalesce_experts: false,
            draft_attn_only: true,
            salience: true,
            sens_rank: 0,
            sessions: 0,
            draft_model: String::new(),
            spec_k_max: 6,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NodeSetup {
    pub id: u16,
    pub addr: SocketAddr,
    /// Emulated compute slowdown relative to the fastest node (≥ 1).
    pub slowdown: f64,
    /// Bytes this node may devote to KV pages.
    pub kv_capacity: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Assignment {
    pub engine: EngineKind,
    pub model: ModelConfig,
    pub nodes: Vec<NodeSetup>,
    /// Symmetric physical links between node pairs.
    pub links: Vec<(u16, u16, Vec<LinkProfile>)>,
    /// Pipeline / ring order with layer ranges (TP: rank order, ranges unused).
    pub stages: Vec<(u16, usize, usize)>,
    pub head: u16,
    /// Hot experts per layer (0 = all resident).
    pub hot_capacity: u16,
    /// Synthetic context tokens injected into KV before the prompt.
    pub inject_context: u32,
    pub tess: TesseractOptions,
    pub driver_addr: SocketAddr,
    /// Addresses of auxiliary services (KV / draft) per node.
    pub service_addrs: Vec<(u16, SocketAddr)>,
}

fn w_addr(w: &mut ByteWriter, a: &SocketAddr) {
    let s = a.to_string();
    w.u16(s.len() as u16);
    w.bytes(s.as_bytes());
}
fn r_addr(r: &mut ByteReader) -> SocketAddr {
    let n = r.u16() as usize;
    std::str::from_utf8(r.bytes(n)).unwrap().parse().unwrap()
}

impl Assignment {
    pub fn encode(&self) -> Vec<u8> {
        let mut w = ByteWriter::default();
        w.u8(self.engine as u8);
        let m = self.model.encode();
        w.u32(m.len() as u32);
        w.bytes(&m);
        w.u16(self.nodes.len() as u16);
        for n in &self.nodes {
            w.u16(n.id);
            w_addr(&mut w, &n.addr);
            w.u64(n.slowdown.to_bits());
            w.u64(n.kv_capacity);
        }
        w.u16(self.links.len() as u16);
        for (a, b, ps) in &self.links {
            w.u16(*a);
            w.u16(*b);
            w.u8(ps.len() as u8);
            for p in ps {
                w.bytes(&p.encode());
            }
        }
        w.u16(self.stages.len() as u16);
        for (n, s, e) in &self.stages {
            w.u16(*n);
            w.u32(*s as u32);
            w.u32(*e as u32);
        }
        w.u16(self.head);
        w.u16(self.hot_capacity);
        w.u32(self.inject_context);
        let t = &self.tess;
        w.u8(t.codec);
        w.u8(t.codec_auto as u8);
        w.u8(t.holo_bits);
        w.u16(t.budget);
        w.u8(t.braid_strands);
        w.u8(t.lookahead as u8);
        w.u8(t.lookahead_horizon);
        w.u8(t.lookahead_margin);
        w.u8(t.deterministic_boundary as u8);
        w.u8(t.kv_mode as u8);
        w.u32(t.page_tokens);
        w.u16(t.drafter.unwrap_or(u16::MAX));
        w.u16(t.draft_rank);
        w.u8(t.draft_ahead);
        w.u8(t.draft_layers);
        w.u8(t.kv_wire);
        w.u8(t.htep_hrq as u8);
        w.u8(t.htep_attn_split as u8);
        w.u16(t.tier0_pages);
        w.u8(t.kv_paired as u8);
        w.u16(t.kv_remote_bias_pages);
        w.u8(t.kv_group_striped as u8);
        w.u8(t.kv_prefix_mirror as u8);
        w.u8(t.spec_local_kv_only as u8);
        w.u8(t.node_lanes);
        w.u8(t.stage_lookahead as u8);
        w.u8(t.htep_reduce_f32 as u8);
        w.u8(t.draft_lookup as u8);
        w.u8(t.coalesce_experts as u8);
        w.u8(t.draft_attn_only as u8);
        w.u8(t.salience as u8);
        w.u16(t.sens_rank);
        w.u8(t.sessions);
        w.u16(t.draft_model.len() as u16);
        w.bytes(t.draft_model.as_bytes());
        w.u8(t.spec_k_max);
        w_addr(&mut w, &self.driver_addr);
        w.u16(self.service_addrs.len() as u16);
        for (id, a) in &self.service_addrs {
            w.u16(*id);
            w_addr(&mut w, a);
        }
        w.into_inner()
    }

    pub fn decode(b: &[u8]) -> Self {
        let mut r = ByteReader::new(b);
        let engine = match r.u8() {
            1 => EngineKind::ExoRing,
            2 => EngineKind::TensorParallel,
            4 => EngineKind::TesseractHtep,
            _ => EngineKind::Tesseract,
        };
        let ml = r.u32() as usize;
        let (model, _) = ModelConfig::decode(r.bytes(ml));
        let nn = r.u16() as usize;
        let nodes = (0..nn)
            .map(|_| NodeSetup {
                id: r.u16(),
                addr: r_addr(&mut r),
                slowdown: f64::from_bits(r.u64()),
                kv_capacity: r.u64(),
            })
            .collect();
        let nl = r.u16() as usize;
        let links = (0..nl)
            .map(|_| {
                let a = r.u16();
                let b2 = r.u16();
                let k = r.u8() as usize;
                (
                    a,
                    b2,
                    (0..k).map(|_| LinkProfile::decode(r.bytes(25))).collect(),
                )
            })
            .collect();
        let ns = r.u16() as usize;
        let stages = (0..ns)
            .map(|_| (r.u16(), r.u32() as usize, r.u32() as usize))
            .collect();
        let head = r.u16();
        let hot_capacity = r.u16();
        let inject_context = r.u32();
        let tess = TesseractOptions {
            codec: r.u8(),
            codec_auto: r.u8() == 1,
            holo_bits: r.u8(),
            budget: r.u16(),
            braid_strands: r.u8(),
            lookahead: r.u8() == 1,
            lookahead_horizon: r.u8(),
            lookahead_margin: r.u8(),
            deterministic_boundary: r.u8() == 1,
            kv_mode: if r.u8() == 1 {
                KvMode::Disaggregated
            } else {
                KvMode::LayerCoupled
            },
            page_tokens: r.u32(),
            drafter: match r.u16() {
                u16::MAX => None,
                x => Some(x),
            },
            draft_rank: r.u16(),
            draft_ahead: r.u8(),
            draft_layers: r.u8(),
            kv_wire: r.u8(),
            htep_hrq: r.u8() == 1,
            htep_attn_split: r.u8() == 1,
            tier0_pages: r.u16(),
            kv_paired: r.u8() == 1,
            kv_remote_bias_pages: r.u16(),
            kv_group_striped: r.u8() == 1,
            kv_prefix_mirror: r.u8() == 1,
            spec_local_kv_only: r.u8() == 1,
            node_lanes: r.u8(),
            stage_lookahead: r.u8() == 1,
            htep_reduce_f32: r.u8() == 1,
            draft_lookup: r.u8() == 1,
            coalesce_experts: r.u8() == 1,
            draft_attn_only: r.u8() == 1,
            salience: r.u8() == 1,
            sens_rank: r.u16(),
            sessions: r.u8(),
            draft_model: {
                let n = r.u16() as usize;
                String::from_utf8_lossy(r.bytes(n)).into_owned()
            },
            spec_k_max: r.u8(),
        };
        let driver_addr = r_addr(&mut r);
        let nsv = r.u16() as usize;
        let service_addrs = (0..nsv).map(|_| (r.u16(), r_addr(&mut r))).collect();
        Assignment {
            engine,
            model,
            nodes,
            links,
            stages,
            head,
            hot_capacity,
            inject_context,
            tess,
            driver_addr,
            service_addrs,
        }
    }

    pub fn node(&self, id: u16) -> &NodeSetup {
        self.nodes
            .iter()
            .find(|n| n.id == id)
            .expect("node not in assignment")
    }

    pub fn links_between(&self, a: u16, b: u16) -> Vec<LinkProfile> {
        if a == b {
            return vec![LinkProfile::unshaped()];
        }
        self.links
            .iter()
            .find(|(x, y, _)| (*x == a && *y == b) || (*x == b && *y == a))
            .map(|l| l.2.clone())
            .unwrap_or_else(|| vec![LinkProfile::unshaped()])
    }

    /// Host node of an endpoint id (services live on their node).
    pub fn host_of(id: u16) -> u16 {
        if id == DRIVER_ID {
            id
        } else if id >= KV_SERVICE_BASE {
            id - KV_SERVICE_BASE
        } else if id >= DRAFT_SERVICE_BASE {
            id - DRAFT_SERVICE_BASE
        } else {
            id
        }
    }

    /// Configure an endpoint (node or service) with addresses and physical links.
    pub fn configure(&self, ep: &Endpoint) {
        let me = Self::host_of(ep.id());
        ep.shared.set_addr(DRIVER_ID, self.driver_addr);
        ep.shared
            .set_links(DRIVER_ID, vec![LinkProfile::unshaped()]);
        ep.shared.set_local(DRIVER_ID);
        let mut all: Vec<(u16, SocketAddr)> = self.nodes.iter().map(|n| (n.id, n.addr)).collect();
        all.extend(self.service_addrs.iter().cloned());
        for (id, addr) in all {
            ep.shared.set_addr(id, addr);
            ep.shared
                .set_links(id, self.links_between(me, Self::host_of(id)));
            if Self::host_of(id) == me {
                ep.shared.set_local(id);
            }
        }
    }
}

// --- generation request / result ---------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct GenerateRequest {
    pub prompt: Vec<u32>,
    pub max_new: u32,
    pub seed: u64,
    pub temperature: f32,
    pub record_logits: bool,
}

impl GenerateRequest {
    pub fn encode(&self) -> Vec<u8> {
        let mut w = ByteWriter::default();
        w.u32(self.prompt.len() as u32);
        for &t in &self.prompt {
            w.u32(t);
        }
        w.u32(self.max_new);
        w.u64(self.seed);
        w.f32(self.temperature);
        w.u8(self.record_logits as u8);
        w.into_inner()
    }
    pub fn decode(b: &[u8]) -> Self {
        let mut r = ByteReader::new(b);
        let n = r.u32() as usize;
        let prompt = (0..n).map(|_| r.u32()).collect();
        GenerateRequest {
            prompt,
            max_new: r.u32(),
            seed: r.u64(),
            temperature: r.f32(),
            record_logits: r.u8() == 1,
        }
    }
}

/// Optional generation controls, appended after a `GenerateRequest`'s
/// encoding (older requests simply end before it).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GenerateExt {
    /// Send a `TokenStream` frame to the driver after every verification step.
    pub stream: bool,
    /// Stop after emitting any of these tokens (e.g. EOS).
    pub stop: Vec<u32>,
    /// Full sampler (top-k / top-p / min-p / repetition penalty); `None`
    /// keeps plain temperature sampling.
    pub sampling: Option<crate::tokenizer::SamplingParams>,
}

impl GenerateExt {
    pub fn encode(&self) -> Vec<u8> {
        let mut w = ByteWriter::default();
        w.u8(self.stream as u8);
        w.u32(self.stop.len() as u32);
        for &t in &self.stop {
            w.u32(t);
        }
        match &self.sampling {
            None => w.u8(0),
            Some(p) => {
                w.u8(1);
                w.f32(p.temperature);
                w.u32(p.top_k as u32);
                w.f32(p.top_p);
                w.f32(p.min_p);
                w.f32(p.repetition_penalty);
                w.u32(p.repeat_last_n as u32);
                w.u64(p.seed);
            }
        }
        w.into_inner()
    }
    pub fn decode(b: &[u8]) -> Self {
        if b.is_empty() {
            return Self::default();
        }
        let mut r = ByteReader::new(b);
        let stream = r.u8() == 1;
        let n = r.u32() as usize;
        let stop = (0..n).map(|_| r.u32()).collect();
        let sampling = (r.u8() == 1).then(|| crate::tokenizer::SamplingParams {
            temperature: r.f32(),
            top_k: r.u32() as usize,
            top_p: r.f32(),
            min_p: r.f32(),
            repetition_penalty: r.f32(),
            repeat_last_n: r.u32() as usize,
            seed: r.u64(),
        });
        GenerateExt {
            stream,
            stop,
            sampling,
        }
    }

    /// Target distribution for `logits` after `history` (sampler or plain
    /// temperature).
    pub fn probs(&self, logits: &[f32], temperature: f32, history: &[u32]) -> Vec<f32> {
        match &self.sampling {
            Some(p) => p.probs(logits, history),
            None => crate::model::probs_from_logits(logits, temperature),
        }
    }
}

impl GenerateRequest {
    /// Decode a request and its optional [`GenerateExt`] trailer.
    pub fn decode_ext(b: &[u8]) -> (Self, GenerateExt) {
        let req = Self::decode(b);
        let n = req.encode().len();
        (req, GenerateExt::decode(b.get(n..).unwrap_or(&[])))
    }
}

/// Tokens of one verification step, streamed to the driver.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TokenBurst {
    pub tokens: Vec<u32>,
    /// Tree size verified in this step.
    pub budget: u16,
    /// ns since the head started decoding (after prefill).
    pub decode_ns: u64,
    pub prefill_ns: u64,
}

impl TokenBurst {
    pub fn encode(&self) -> Vec<u8> {
        let mut w = ByteWriter::default();
        w.u32(self.tokens.len() as u32);
        for &t in &self.tokens {
            w.u32(t);
        }
        w.u16(self.budget);
        w.u64(self.decode_ns);
        w.u64(self.prefill_ns);
        w.into_inner()
    }
    pub fn decode(b: &[u8]) -> Self {
        let mut r = ByteReader::new(b);
        let n = r.u32() as usize;
        TokenBurst {
            tokens: (0..n).map(|_| r.u32()).collect(),
            budget: r.u16(),
            decode_ns: r.u64(),
            prefill_ns: r.u64(),
        }
    }
}

/// Length of the prefix of `prompt[..n-1]` whose KV is already committed
/// (`seq`), capped so at least one prompt row is prefilled.
pub fn shared_prefix(seq: &[u32], prompt: &[u32], n: usize) -> usize {
    let lim = n.saturating_sub(2);
    seq.iter()
        .zip(&prompt[..n - 1])
        .take(lim)
        .take_while(|(a, b)| a == b)
        .count()
}

/// Tokens of a tree path: the history a tree node's prediction conditions
/// on beyond the committed context (root excluded).
pub fn tree_path_tokens(tree: &crate::speculative::SpeculativeTree, node: usize) -> Vec<u32> {
    let mut v = Vec::new();
    let mut i = node;
    while i != 0 {
        v.push(tree.tokens[i]);
        i = tree.parents[i] as usize;
    }
    v.reverse();
    v
}

/// Per-step bookkeeping shared by the Tesseract coordinators: streams the
/// step's tokens and applies stop tokens. Returns true when generation must
/// end after this step (`emitted` is truncated after the stop token).
pub fn emit_step(
    ep: &Endpoint,
    ext: &GenerateExt,
    emitted: &mut Vec<u32>,
    budget: u16,
    decode_ns: u64,
    prefill_ns: u64,
    room: usize,
) -> bool {
    emit_step_for(ep, 0, ext, emitted, budget, decode_ns, prefill_ns, room)
}

/// [`emit_step`] for request `id` (frame header `tree_epoch`; batched
/// sessions multiplex several requests over one driver connection).
#[allow(clippy::too_many_arguments)]
pub fn emit_step_for(
    ep: &Endpoint,
    id: u64,
    ext: &GenerateExt,
    emitted: &mut Vec<u32>,
    budget: u16,
    decode_ns: u64,
    prefill_ns: u64,
    room: usize,
) -> bool {
    emitted.truncate(room);
    let mut stop = emitted.len() >= room;
    if let Some(i) = emitted.iter().position(|t| ext.stop.contains(t)) {
        emitted.truncate(i + 1);
        stop = true;
    }
    if ext.stream {
        let b = TokenBurst {
            tokens: emitted.clone(),
            budget,
            decode_ns,
            prefill_ns,
        };
        let _ = ep.send_payload(DRIVER_ID, FrameType::TokenStream, id, 0, b.encode());
    }
    stop
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct GenerateResult {
    pub tokens: Vec<u32>,
    pub prefill_ns: u64,
    pub decode_ns: u64,
    /// Emission time of every generated token (ns since decode start).
    pub token_times_ns: Vec<u64>,
    /// Tokens emitted per speculative epoch (1 per step for baselines).
    pub per_epoch: Vec<u32>,
    pub budgets: Vec<u16>,
    /// Draft service cache hits / misses (Tesseract).
    pub draft_hits: u32,
    pub draft_misses: u32,
    pub logits: Vec<Vec<f32>>,
    /// Prompt tokens whose KV was reused from the previous request.
    pub prefix_reused: u32,
}

impl GenerateResult {
    pub fn encode(&self) -> Vec<u8> {
        let mut w = ByteWriter::default();
        w.u32(self.tokens.len() as u32);
        for &t in &self.tokens {
            w.u32(t);
        }
        w.u64(self.prefill_ns);
        w.u64(self.decode_ns);
        w.u32(self.token_times_ns.len() as u32);
        for &t in &self.token_times_ns {
            w.u64(t);
        }
        w.u32(self.per_epoch.len() as u32);
        for &t in &self.per_epoch {
            w.u32(t);
        }
        w.u32(self.budgets.len() as u32);
        for &t in &self.budgets {
            w.u16(t);
        }
        w.u32(self.draft_hits);
        w.u32(self.draft_misses);
        w.u32(self.prefix_reused);
        w.u32(self.logits.len() as u32);
        for l in &self.logits {
            w.u32(l.len() as u32);
            w.f32_slice(l);
        }
        w.into_inner()
    }
    pub fn decode(b: &[u8]) -> Self {
        let mut r = ByteReader::new(b);
        let n = r.u32() as usize;
        let tokens = (0..n).map(|_| r.u32()).collect();
        let prefill_ns = r.u64();
        let decode_ns = r.u64();
        let n = r.u32() as usize;
        let token_times_ns = (0..n).map(|_| r.u64()).collect();
        let n = r.u32() as usize;
        let per_epoch = (0..n).map(|_| r.u32()).collect();
        let n = r.u32() as usize;
        let budgets = (0..n).map(|_| r.u16()).collect();
        let draft_hits = r.u32();
        let draft_misses = r.u32();
        let prefix_reused = r.u32();
        let n = r.u32() as usize;
        let logits = (0..n)
            .map(|_| {
                let k = r.u32() as usize;
                r.f32_vec(k)
            })
            .collect();
        GenerateResult {
            tokens,
            prefill_ns,
            decode_ns,
            token_times_ns,
            per_epoch,
            budgets,
            draft_hits,
            draft_misses,
            logits,
            prefix_reused,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct NodeStats {
    pub node: u16,
    /// Compute-busy time (real compute × emulated slowdown).
    pub busy_ns: u64,
    pub bytes_sent: u64,
    pub frames_sent: u64,
    /// hits, blocking misses, blocking ns, prestaged, prestage ns, waited-inflight
    pub staging: [u64; 6],
    pub kv_bytes: u64,
    pub weight_bytes: u64,
    pub commits: u64,
    pub commit_ns: u64,
    pub lookahead_needed: u64,
    pub lookahead_covered: u64,
    pub prewakes_sent: u64,
    pub kv_queries_served: u64,
    /// Inter-host attention traffic during decode (KV query + LSE partials +
    /// K/V appends), measured by the compute stage.
    pub attn_bytes: u64,
    /// Inter-host bytes of out-of-band PreWake / route feedback.
    pub oob_bytes: u64,
    /// Inter-host bytes of inter-stage activation frames (compressed or FP16).
    pub act_bytes: u64,
    /// Disaggregated KV telemetry (decode): local scan time, local tokens
    /// scanned, time spent waiting for remote partials after the local scan,
    /// and remote exchanges. Calibrates the paired placement's page bias.
    pub kv_local_ns: u64,
    pub kv_local_tokens: u64,
    pub kv_wait_ns: u64,
    pub kv_exchanges: u64,
    /// Lower envelopes (robust to load spikes): minimum remote wait of one
    /// exchange and minimum local scan time per 256 tokens.
    pub kv_wait_min_ns: u64,
    pub kv_page_min_ns: u64,
    /// One-off bytes of prefix pages moved to their mirrors at switch-over.
    pub kv_migration_bytes: u64,
    /// Stage-0 cross-step lookahead rows: reused, discarded, aborted early.
    pub ahead_hits: u64,
    pub ahead_misses: u64,
    pub ahead_aborts: u64,
}

impl NodeStats {
    pub fn encode(&self) -> Vec<u8> {
        let mut w = ByteWriter::default();
        w.u16(self.node);
        for v in [self.busy_ns, self.bytes_sent, self.frames_sent] {
            w.u64(v);
        }
        for v in self.staging {
            w.u64(v);
        }
        for v in [
            self.kv_bytes,
            self.weight_bytes,
            self.commits,
            self.commit_ns,
            self.lookahead_needed,
            self.lookahead_covered,
            self.prewakes_sent,
            self.kv_queries_served,
            self.attn_bytes,
            self.oob_bytes,
            self.act_bytes,
            self.kv_local_ns,
            self.kv_local_tokens,
            self.kv_wait_ns,
            self.kv_exchanges,
            self.kv_wait_min_ns,
            self.kv_page_min_ns,
            self.kv_migration_bytes,
            self.ahead_hits,
            self.ahead_misses,
            self.ahead_aborts,
        ] {
            w.u64(v);
        }
        w.into_inner()
    }
    pub fn decode(b: &[u8]) -> Self {
        let mut r = ByteReader::new(b);
        let node = r.u16();
        let (busy_ns, bytes_sent, frames_sent) = (r.u64(), r.u64(), r.u64());
        let mut staging = [0u64; 6];
        for s in staging.iter_mut() {
            *s = r.u64();
        }
        NodeStats {
            node,
            busy_ns,
            bytes_sent,
            frames_sent,
            staging,
            kv_bytes: r.u64(),
            weight_bytes: r.u64(),
            commits: r.u64(),
            commit_ns: r.u64(),
            lookahead_needed: r.u64(),
            lookahead_covered: r.u64(),
            prewakes_sent: r.u64(),
            kv_queries_served: r.u64(),
            attn_bytes: r.u64(),
            oob_bytes: r.u64(),
            act_bytes: r.u64(),
            kv_local_ns: r.u64(),
            kv_local_tokens: r.u64(),
            kv_wait_ns: r.u64(),
            kv_exchanges: r.u64(),
            kv_wait_min_ns: r.u64(),
            kv_page_min_ns: r.u64(),
            kv_migration_bytes: r.u64(),
            ahead_hits: r.u64(),
            ahead_misses: r.u64(),
            ahead_aborts: r.u64(),
        }
    }

    pub fn merge(&mut self, o: &NodeStats) {
        self.busy_ns += o.busy_ns;
        self.bytes_sent += o.bytes_sent;
        self.frames_sent += o.frames_sent;
        for i in 0..6 {
            self.staging[i] += o.staging[i];
        }
        self.kv_bytes += o.kv_bytes;
        self.weight_bytes += o.weight_bytes;
        self.commits += o.commits;
        self.commit_ns += o.commit_ns;
        self.lookahead_needed += o.lookahead_needed;
        self.lookahead_covered += o.lookahead_covered;
        self.prewakes_sent += o.prewakes_sent;
        self.kv_queries_served += o.kv_queries_served;
        self.attn_bytes += o.attn_bytes;
        self.oob_bytes += o.oob_bytes;
        self.act_bytes += o.act_bytes;
        self.kv_local_ns += o.kv_local_ns;
        self.kv_local_tokens += o.kv_local_tokens;
        self.kv_wait_ns += o.kv_wait_ns;
        self.kv_exchanges += o.kv_exchanges;
        self.kv_wait_min_ns = match (self.kv_wait_min_ns, o.kv_wait_min_ns) {
            (0, b) => b,
            (a, 0) => a,
            (a, b) => a.min(b),
        };
        self.kv_migration_bytes += o.kv_migration_bytes;
        self.ahead_hits += o.ahead_hits;
        self.ahead_misses += o.ahead_misses;
        self.ahead_aborts += o.ahead_aborts;
        self.kv_page_min_ns = match (self.kv_page_min_ns, o.kv_page_min_ns) {
            (0, b) => b,
            (a, 0) => a,
            (a, b) => a.min(b),
        };
    }
}

// --- compute shaper ------------------------------------------------------------

/// Emulates a slower device: real compute runs at host speed, then the node
/// stays busy for `(slowdown − 1) ×` the compute that ran. All math is still
/// performed for real; only the *extra* time of a slower chip is modelled.
///
/// The stretched quantity is the closure's critical-path compute time
/// ([`crate::tensor::pool::critical_path`]: thread CPU time, parallel regions
/// counted by their busiest lane), not its wall time. Wall time also contains
/// every moment the host scheduler gave these threads' cores to other work —
/// a 2 ms daemon wake-up would become a 46 ms phantom stall on a 23× node,
/// which a real phone (with its own cores) never sees. Genuine waits inside
/// shaped work (remote KV partials, expert staging) are marked with
/// [`crate::tensor::pool::count_wall`] and still count in full.
#[derive(Clone, Debug)]
pub struct ComputeShaper {
    pub slowdown: f64,
    pub busy_ns: u64,
    /// Host time excluded from stretching (wall − critical path), summed.
    pub jitter_ns: u64,
}

/// Shaper model: `TESSERACT_SHAPER=wall` restores the ≤ V10.0 model
/// (stretch wall time) for comparisons; the default (1) stretches
/// critical-path CPU time.
fn shaper_mode() -> u8 {
    static M: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *M.get_or_init(|| match std::env::var("TESSERACT_SHAPER").as_deref() {
        Ok("wall") => 0,
        _ => 1,
    })
}

/// `TESSERACT_SHAPER_TRACE=1`: print every shaped call's wall and
/// critical-path time to stderr (diagnosing emulation noise).
fn shaper_trace() -> bool {
    static T: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *T.get_or_init(|| std::env::var_os("TESSERACT_SHAPER_TRACE").is_some())
}

impl ComputeShaper {
    pub fn new(slowdown: f64) -> Self {
        ComputeShaper {
            slowdown: slowdown.max(1.0),
            busy_ns: 0,
            jitter_ns: 0,
        }
    }
    pub fn run<T>(&mut self, f: impl FnOnce() -> T) -> T {
        if self.slowdown <= 1.0 {
            let t0 = mono_ns();
            let out = f();
            self.busy_ns += mono_ns() - t0;
            return out;
        }
        let (out, dt, compute) = if shaper_mode() == 0 {
            let t0 = mono_ns();
            let out = f();
            let dt = mono_ns() - t0;
            (out, dt, dt)
        } else {
            crate::tensor::pool::critical_path(f)
        };
        if shaper_trace() {
            eprintln!(
                "shaper x{:.1} wall_us {} crit_us {}",
                self.slowdown,
                dt / 1000,
                compute / 1000
            );
        }
        let extra = (compute as f64 * (self.slowdown - 1.0)) as u64;
        wait_until_ns(mono_ns() + extra);
        self.busy_ns += dt + extra;
        self.jitter_ns += dt - compute;
        out
    }
}

// --- bounded compute pool ----------------------------------------------------

/// Process-wide counting semaphore bounding concurrent heavy kernels (KV-shard
/// LSE partials). With every node of an emulated cluster in one process, the
/// KV services of all shards would otherwise run their attention partials
/// simultaneously on top of the stage threads and oversubscribe the cores.
pub struct ComputePermits {
    free: std::sync::Mutex<usize>,
    cv: std::sync::Condvar,
    pub capacity: usize,
}

impl ComputePermits {
    pub fn global() -> &'static ComputePermits {
        static P: std::sync::OnceLock<ComputePermits> = std::sync::OnceLock::new();
        P.get_or_init(|| {
            // leave headroom for the stage threads that issue the queries
            let cores = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4);
            let capacity = (cores / 2).max(1);
            ComputePermits {
                free: std::sync::Mutex::new(capacity),
                cv: std::sync::Condvar::new(),
                capacity,
            }
        })
    }

    /// Run `f` while holding one permit.
    pub fn run<T>(&self, f: impl FnOnce() -> T) -> T {
        {
            let mut n = self.free.lock().unwrap();
            while *n == 0 {
                n = self.cv.wait(n).unwrap();
            }
            *n -= 1;
        }
        let out = f();
        *self.free.lock().unwrap() += 1;
        self.cv.notify_one();
        out
    }
}

// --- worker / driver ------------------------------------------------------------

/// Run a worker node: serve assignments until the driver disconnects.
pub fn run_worker(ep: Endpoint) {
    crate::tensor::prefer_performance_cores();
    loop {
        let f = ep.recv();
        match f.header.frame_type {
            FrameType::Assign => {
                let a = Assignment::decode(&f.payload);
                a.configure(&ep);
                match a.engine {
                    EngineKind::ExoRing if a.stages.iter().any(|s| s.0 == ep.id()) => {
                        baselines::ExoRingNode::new(&ep, &a).run(&ep)
                    }
                    // Not part of the ring: idle host.
                    EngineKind::ExoRing => node::run_auxiliary(&ep),
                    EngineKind::TensorParallel => {
                        baselines::TensorParallelNode::new(&ep, &a).run(&ep)
                    }
                    EngineKind::Tesseract if a.stages.iter().any(|s| s.0 == ep.id()) => {
                        node::TesseractNode::new(&ep, &a).run(&ep)
                    }
                    // Off-pipeline host (drafter / KV pool): its services do the work.
                    EngineKind::Tesseract => node::run_auxiliary(&ep),
                    EngineKind::TesseractHtep if a.stages.iter().any(|s| s.0 == ep.id()) => {
                        htep::HtepNode::new(&ep, &a).run(&ep)
                    }
                    EngineKind::TesseractHtep => node::run_auxiliary(&ep),
                }
            }
            FrameType::Shutdown => {
                ep.shared.close();
                return;
            }
            FrameType::ProbePing => discovery::answer_probe(&ep, &f.payload),
            _ => {}
        }
    }
}

/// Send the `AssignAck` once a node is configured.
pub fn ack_assign(ep: &Endpoint) {
    let _ = ep.send_payload(
        DRIVER_ID,
        FrameType::AssignAck,
        0,
        0,
        ep.id().to_le_bytes().to_vec(),
    );
}

/// Common stats reply.
pub fn reply_stats(ep: &Endpoint, mut s: NodeStats) {
    s.node = ep.id();
    s.bytes_sent += ep.shared.bytes_sent();
    s.frames_sent += ep
        .shared
        .stats
        .frames_sent
        .load(std::sync::atomic::Ordering::Relaxed);
    let t = &ep.shared.stats;
    s.oob_bytes += t.bytes_of(FrameType::PreWakeSignal) + t.bytes_of(FrameType::RouteFeedback);
    s.act_bytes += t.bytes_of(FrameType::ActivationDelta) + t.bytes_of(FrameType::RawActivation);
    let _ = ep.send_payload(DRIVER_ID, FrameType::Stats, 0, 0, s.encode());
}

pub struct Driver {
    pub ep: Endpoint,
}

impl Driver {
    pub fn bind(addr: &str) -> std::io::Result<Self> {
        Ok(Driver {
            ep: Endpoint::bind(DRIVER_ID, addr)?,
        })
    }

    pub fn addr(&self) -> SocketAddr {
        self.ep.addr()
    }

    /// Assign roles to every node and wait until all are ready.
    pub fn assign(&self, a: &Assignment) {
        let targets = assignment_targets(a);
        for (id, addr) in &targets {
            self.ep.shared.set_addr(*id, *addr);
            self.ep.shared.set_links(*id, vec![LinkProfile::unshaped()]);
        }
        let payload = a.encode();
        for (id, _) in &targets {
            self.ep
                .send_payload(*id, FrameType::Assign, 0, 0, payload.clone())
                .expect("assign send");
        }
        let mut acked = 0;
        while acked < targets.len() {
            self.ep
                .recv_match_timeout(
                    &|f| f.header.frame_type == FrameType::AssignAck,
                    Duration::from_secs(600),
                )
                .expect("node failed to acknowledge assignment");
            acked += 1;
        }
    }

    pub fn generate(&self, head: u16, req: &GenerateRequest) -> GenerateResult {
        self.ep
            .send_payload(head, FrameType::Generate, 0, 0, req.encode())
            .expect("generate send");
        let f = self
            .ep
            .recv_match_timeout(
                &|f| f.header.frame_type == FrameType::GenerateResult,
                Duration::from_secs(900),
            )
            .expect(
                "head node did not return a result (a worker failed — see its panic message above)",
            );
        GenerateResult::decode(&f.payload)
    }

    /// Generate with controls; `on_burst` sees every verification step's
    /// tokens as they are accepted (when `ext.stream`).
    pub fn generate_stream(
        &self,
        head: u16,
        req: &GenerateRequest,
        ext: &GenerateExt,
        on_burst: &mut dyn FnMut(&TokenBurst),
    ) -> GenerateResult {
        let mut payload = req.encode();
        payload.extend(ext.encode());
        self.ep
            .send_payload(head, FrameType::Generate, 0, 0, payload)
            .expect("generate send");
        loop {
            let f = self
                .ep
                .recv_match_timeout(
                    &|f| matches!(f.header.frame_type, FrameType::GenerateResult | FrameType::TokenStream),
                    Duration::from_secs(900),
                )
                .expect("head node did not return a result (a worker failed — see its panic message above)");
            match f.header.frame_type {
                FrameType::TokenStream => on_burst(&TokenBurst::decode(&f.payload)),
                _ => return GenerateResult::decode(&f.payload),
            }
        }
    }

    pub fn stats(&self, nodes: &[u16]) -> Vec<NodeStats> {
        for &n in nodes {
            let _ = self
                .ep
                .send_payload(n, FrameType::StatsRequest, 0, 0, vec![]);
        }
        let mut out = Vec::new();
        for _ in nodes {
            match self.ep.recv_match_timeout(
                &|f| f.header.frame_type == FrameType::Stats,
                Duration::from_secs(30),
            ) {
                Some(f) => out.push(NodeStats::decode(&f.payload)),
                None => break,
            }
        }
        out.sort_by_key(|s| s.node);
        out
    }

    /// End the current engine on every node and service (workers return to idle).
    pub fn reset(&self, a: &Assignment) {
        for (id, _) in assignment_targets(a) {
            let _ = self.ep.send_payload(id, FrameType::Reset, 0, 0, vec![]);
        }
    }

    pub fn shutdown(&self, a: &Assignment) {
        for (id, _) in assignment_targets(a) {
            let _ = self.ep.send_payload(id, FrameType::Shutdown, 0, 0, vec![]);
        }
    }
}

/// A worker node plus its KV and draft services, all on loopback threads.
pub struct LocalWorker {
    pub id: u16,
    pub addr: SocketAddr,
    pub kv_addr: SocketAddr,
    pub draft_addr: SocketAddr,
    pub handles: Vec<std::thread::JoinHandle<()>>,
}

impl LocalWorker {
    pub fn service_addrs(&self) -> Vec<(u16, SocketAddr)> {
        vec![
            (KV_SERVICE_BASE + self.id, self.kv_addr),
            (DRAFT_SERVICE_BASE + self.id, self.draft_addr),
        ]
    }
}

fn spawn_thread(name: String, f: impl FnOnce() + Send + 'static) -> std::thread::JoinHandle<()> {
    std::thread::Builder::new()
        .name(name)
        .stack_size(16 << 20)
        .spawn(f)
        .expect("spawn thread")
}

/// Bind a node endpoint and its services on `ip` (ports: node, +1 KV, +2 draft
/// when `base_port` is non-zero; ephemeral otherwise) and start their threads.
pub fn start_worker(id: u16, ip: &str, base_port: u16) -> std::io::Result<LocalWorker> {
    let port = |off: u16| if base_port == 0 { 0 } else { base_port + off };
    let node = Endpoint::bind(id, &format!("{ip}:{}", port(0)))?;
    let kv = Endpoint::bind(KV_SERVICE_BASE + id, &format!("{ip}:{}", port(1)))?;
    let draft = Endpoint::bind(DRAFT_SERVICE_BASE + id, &format!("{ip}:{}", port(2)))?;
    let (addr, kv_addr, draft_addr) = (node.addr(), kv.addr(), draft.addr());
    let handles = vec![
        spawn_thread(format!("tess-node-{id}"), move || run_worker(node)),
        spawn_thread(format!("tess-kv-{id}"), move || node::run_kv_service(kv)),
        spawn_thread(format!("tess-draft-{id}"), move || {
            node::run_draft_service(draft)
        }),
    ];
    Ok(LocalWorker {
        id,
        addr,
        kv_addr,
        draft_addr,
        handles,
    })
}

/// Spawn workers (with services) on loopback for the given node ids.
pub fn spawn_local_workers(ids: &[u16]) -> Vec<LocalWorker> {
    ids.iter()
        .map(|&id| start_worker(id, "127.0.0.1", 0).expect("start worker"))
        .collect()
}

/// All endpoint ids (nodes + services) that receive an assignment.
pub fn assignment_targets(a: &Assignment) -> Vec<(u16, SocketAddr)> {
    let mut v: Vec<(u16, SocketAddr)> = a.nodes.iter().map(|n| (n.id, n.addr)).collect();
    v.extend(a.service_addrs.iter().cloned());
    v
}

#[cfg(test)]
mod shaper_tests {
    use super::*;

    #[test]
    fn shaper_stretches_compute_but_not_host_idle_time() {
        if crate::tensor::pool::thread_cpu_ns().is_none() {
            return;
        }
        let ms = 1_000_000u64;
        let mut s = ComputeShaper::new(10.0);
        // 15 ms off-CPU inside a 10× node: V10.0 stretched this to 150 ms.
        // Checked on the shaper's own accounting, not on wall time (a busy
        // CI VM can oversleep by tens of ms):
        // busy = wall_inside + 9·compute and jitter = wall_inside − compute,
        // so busy − jitter = 10·compute exactly.
        s.run(|| std::thread::sleep(std::time::Duration::from_millis(15)));
        let compute = (s.busy_ns - s.jitter_ns) / 10;
        assert!(
            s.jitter_ns >= 10 * ms,
            "off-CPU time not recognised: {}",
            s.jitter_ns
        );
        assert!(
            compute < 2 * ms,
            "idle time counted as compute (and stretched 9×): {compute}"
        );
        // 4 ms of real compute is still stretched 10×
        let mut s = ComputeShaper::new(10.0);
        let t0 = mono_ns();
        s.run(|| {
            let c0 = crate::tensor::pool::thread_cpu_ns().unwrap();
            while crate::tensor::pool::thread_cpu_ns().unwrap() - c0 < 4 * ms {
                std::hint::black_box(0u64);
            }
        });
        assert!(mono_ns() - t0 >= 40 * ms);
    }
}
