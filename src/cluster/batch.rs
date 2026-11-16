//! Continuous batching, multi-session KV and real-checkpoint drafting.
//!
//! * **Sessions.** Every rank keeps one KV cache per *session slot*. A
//!   [`BatchStep`] carries one [`Segment`] per active slot: its rows (a
//!   prompt chunk being prefilled, or a draft chain being verified), their
//!   positions and tree keys. All segments share the batched GEMMs and the
//!   per-layer all-reduces of one forward pass; attention runs per segment
//!   against that slot's KV. A [`BatchCommit`] then commits each slot's
//!   accepted path. Each slot remembers the tokens behind its KV, so a new
//!   request lands on the free slot sharing its longest prefix (prefix cache
//!   across requests and sessions).
//! * **Drafting.** [`SuffixIndex`] is an online suffix automaton over a
//!   session's prompt + output: the longest suffix of the context that
//!   occurred before, and what followed it, in O(1) amortized per token.
//!   [`DraftModel`] is a small companion checkpoint with the target's
//!   vocabulary (`--draft-model`), run greedily on the head with its own KV
//!   per session. [`SpecController`] picks the chain length from measured
//!   acceptance and timings. Verification is exact, so speculation never
//!   changes the output.
//! * **Client.** [`BatchClient`] multiplexes concurrent requests over one
//!   driver endpoint (request id in the frame header).

use super::{Driver, GenerateExt, GenerateRequest, GenerateResult, TokenBurst};
use crate::kv_fabric::{BatchMeta, LocalKvCache, NodeKey};
use crate::model::{argmax, HeadWeights, ModelConfig, NoObserver, ShardModel};
use crate::wire::protocol::{ByteReader, ByteWriter, FrameType};
use std::collections::HashMap;
use std::sync::mpsc;
use std::time::Duration;

// --- wire payloads ------------------------------------------------------------------

/// One session's rows in a batched forward pass.
#[derive(Clone, Debug)]
pub struct Segment {
    pub slot: u16,
    /// Truncate this slot's KV to this many committed tokens first (a new
    /// request keeping the prefix it shares with the slot's previous one).
    pub reset_to: Option<u32>,
    /// Commit every row right after the pass (prompt prefill).
    pub prefill: bool,
    pub meta: BatchMeta,
    pub tokens: Vec<u32>,
}

/// A batched forward pass over several sessions' segments.
#[derive(Clone, Debug, Default)]
pub struct BatchStep {
    pub segments: Vec<Segment>,
}

impl BatchStep {
    pub fn rows(&self) -> usize {
        self.segments.iter().map(|s| s.tokens.len()).sum()
    }

    pub fn tokens(&self) -> Vec<u32> {
        self.segments
            .iter()
            .flat_map(|s| s.tokens.iter().copied())
            .collect()
    }

    pub fn positions(&self) -> Vec<usize> {
        self.segments
            .iter()
            .flat_map(|s| s.meta.positions.iter().copied())
            .collect()
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut w = ByteWriter::default();
        w.u16(self.segments.len() as u16);
        for s in &self.segments {
            w.u16(s.slot);
            w.u32(s.reset_to.map(|r| r + 1).unwrap_or(0));
            w.u8(s.prefill as u8);
            let mut m = Vec::new();
            s.meta.encode(&mut m);
            w.u32(m.len() as u32);
            w.bytes(&m);
            w.u32(s.tokens.len() as u32);
            for &t in &s.tokens {
                w.u32(t);
            }
        }
        w.into_inner()
    }

    pub fn decode(b: &[u8]) -> Self {
        let mut r = ByteReader::new(b);
        let n = r.u16() as usize;
        let segments = (0..n)
            .map(|_| {
                let slot = r.u16();
                let reset_to = r.u32().checked_sub(1);
                let prefill = r.u8() == 1;
                let ml = r.u32() as usize;
                let (meta, _) = BatchMeta::decode(r.bytes(ml));
                let nt = r.u32() as usize;
                Segment {
                    slot,
                    reset_to,
                    prefill,
                    meta,
                    tokens: (0..nt).map(|_| r.u32()).collect(),
                }
            })
            .collect();
        BatchStep { segments }
    }
}

/// Accepted KV keys per slot after a batched verification step.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BatchCommit {
    pub slots: Vec<(u16, Vec<NodeKey>)>,
}

impl BatchCommit {
    pub fn encode(&self) -> Vec<u8> {
        let mut w = ByteWriter::default();
        w.u16(self.slots.len() as u16);
        for (s, keys) in &self.slots {
            w.u16(*s);
            w.u32(keys.len() as u32);
            for &k in keys {
                w.u64(k);
            }
        }
        w.into_inner()
    }

    pub fn decode(b: &[u8]) -> Self {
        let mut r = ByteReader::new(b);
        let n = r.u16() as usize;
        BatchCommit {
            slots: (0..n)
                .map(|_| {
                    let s = r.u16();
                    let k = r.u32() as usize;
                    (s, (0..k).map(|_| r.u64()).collect())
                })
                .collect(),
        }
    }
}

// --- suffix automaton drafting ------------------------------------------------------

/// Online suffix automaton over a token sequence.
///
/// After each [`push`](Self::push) the automaton knows the longest suffix of
/// the sequence that also occurs earlier (the suffix link of the new state)
/// and the end of an earlier occurrence of it — the most recent one for that
/// state, kept up to date in O(1) per token. [`lookup`](Self::lookup) returns
/// what followed that occurrence: a draft that copies the context.
#[derive(Clone, Debug)]
pub struct SuffixIndex {
    tokens: Vec<u32>,
    len: Vec<u32>,
    link: Vec<i32>,
    next: Vec<HashMap<u32, u32>>,
    /// An end position of the state's strings (the latest recorded).
    end: Vec<u32>,
    last: usize,
    /// (length, end position) of the latest earlier occurrence of the
    /// longest repeated suffix, as of the last push.
    matched: (usize, Option<usize>),
}

impl Default for SuffixIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl SuffixIndex {
    pub fn new() -> Self {
        SuffixIndex {
            tokens: Vec::new(),
            len: vec![0],
            link: vec![-1],
            next: vec![HashMap::new()],
            end: vec![0],
            last: 0,
            matched: (0, None),
        }
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    fn state(&mut self, len: u32, link: i32, next: HashMap<u32, u32>, end: u32) -> usize {
        self.len.push(len);
        self.link.push(link);
        self.next.push(next);
        self.end.push(end);
        self.len.len() - 1
    }

    /// Append one token (standard suffix-automaton extension).
    pub fn push(&mut self, t: u32) {
        let i = self.tokens.len() as u32;
        self.tokens.push(t);
        let cur = self.state(self.len[self.last] + 1, -1, HashMap::new(), i);
        let mut p = self.last as i32;
        while p >= 0 && !self.next[p as usize].contains_key(&t) {
            self.next[p as usize].insert(t, cur as u32);
            p = self.link[p as usize];
        }
        let s = if p < 0 {
            0
        } else {
            let p = p as usize;
            let q = self.next[p][&t] as usize;
            if self.len[p] + 1 == self.len[q] {
                q
            } else {
                let clone = self.state(
                    self.len[p] + 1,
                    self.link[q],
                    self.next[q].clone(),
                    self.end[q],
                );
                let mut pp = p as i32;
                while pp >= 0 && self.next[pp as usize].get(&t) == Some(&(q as u32)) {
                    self.next[pp as usize].insert(t, clone as u32);
                    pp = self.link[pp as usize];
                }
                self.link[q] = clone as i32;
                clone
            }
        };
        self.link[cur] = s as i32;
        // s's strings end at i now; remember the previous end before
        // recording this one
        self.matched = if s == 0 {
            (0, None)
        } else {
            (self.len[s] as usize, Some(self.end[s] as usize))
        };
        self.end[s] = i;
        self.last = cur;
    }

    pub fn extend(&mut self, ts: &[u32]) {
        for &t in ts {
            self.push(t);
        }
    }

    /// (match length, up to `k` tokens that followed the latest earlier
    /// occurrence of the longest repeated suffix).
    pub fn lookup(&self, k: usize) -> (usize, Vec<u32>) {
        match self.matched {
            (n, Some(e)) if n > 0 => {
                let from = e + 1;
                (
                    n,
                    self.tokens[from..(from + k).min(self.tokens.len())].to_vec(),
                )
            }
            _ => (0, Vec::new()),
        }
    }
}

// --- companion draft model --------------------------------------------------------------

/// A small checkpoint sharing the target's vocabulary, run greedily on the
/// coordinator to propose draft chains.
pub struct DraftModel {
    pub cfg: ModelConfig,
    pub head: HeadWeights,
    pub model: ShardModel,
}

/// One session's state in the draft model: its KV and the tokens behind it.
#[derive(Default)]
pub struct DraftSession {
    kv: Option<LocalKvCache>,
    seq: Vec<u32>,
    epoch: u64,
}

impl DraftModel {
    /// Load `path` and check that its tokenizer is the target's (same ids
    /// for every token the draft model can emit).
    pub fn load(path: &str, target: &ModelConfig) -> Result<Self, String> {
        let cfg = crate::loader::sharded_loader::config_from_file(path, None)
            .map_err(|e| format!("draft model {path}: {e}"))?;
        let dt = crate::loader::sharded_loader::tokenizer_for(&cfg);
        let tt = crate::loader::sharded_loader::tokenizer_for(target);
        let n = dt.tokens.len().min(tt.tokens.len());
        if cfg.vocab > target.vocab || dt.tokens[..n] != tt.tokens[..n] {
            return Err(format!(
                "draft model {path}: vocabulary differs from the target's"
            ));
        }
        Ok(DraftModel {
            head: HeadWeights::generate(&cfg),
            model: ShardModel::generate(&cfg, 0..cfg.n_layers, None),
            cfg,
        })
    }

    fn forward(&self, s: &mut DraftSession, tokens: &[u32], pos: usize) -> u32 {
        let kv = s.kv.get_or_insert_with(|| {
            LocalKvCache::new(self.cfg.n_heads, self.cfg.n_kv_heads, self.cfg.d_head)
        });
        s.epoch += 1;
        let meta = BatchMeta::causal_prefill(pos, tokens.len(), s.epoch);
        let mut x = self.head.embed_tokens(tokens);
        self.model.forward(&mut x, &meta, kv, &mut NoObserver);
        kv.commit(&meta.keys, &|_| false);
        let last = crate::tensor::Tensor2D::from_vec(1, x.cols, x.row(x.rows - 1).to_vec());
        argmax(self.head.logits(&last, self.cfg.rms_eps).row(0)) as u32
    }

    /// Up to `k` greedy draft tokens continuing `ctx` (which ends with the
    /// root). Re-uses the session's KV for the prefix it shares with `ctx`
    /// (rejected draft tokens are dropped by truncation).
    pub fn draft(&self, s: &mut DraftSession, ctx: &[u32], k: usize) -> Vec<u32> {
        self.draft_counted(s, ctx, k).0
    }

    /// [`draft`](Self::draft) plus the number of rows the draft model ran
    /// (context catch-up + chain).
    pub fn draft_counted(&self, s: &mut DraftSession, ctx: &[u32], k: usize) -> (Vec<u32>, usize) {
        if k == 0 || ctx.is_empty() {
            return (Vec::new(), 0);
        }
        let p = s
            .seq
            .iter()
            .zip(ctx)
            .take_while(|(a, b)| a == b)
            .count()
            .min(ctx.len() - 1);
        if let Some(kv) = s.kv.as_mut() {
            kv.truncate(p);
        }
        s.seq.truncate(p);
        let mut t = self.forward(s, &ctx[p..], p);
        let mut rows = ctx.len() - p;
        s.seq.extend_from_slice(&ctx[p..]);
        let mut out = vec![t];
        while out.len() < k {
            let pos = s.seq.len();
            s.seq.push(t);
            t = self.forward(s, &[t], pos);
            rows += 1;
            out.push(t);
        }
        (out, rows)
    }
}

/// Per-session draft acceptance (EMA of the per-token acceptance rate),
/// tracked separately for context-copy and draft-model chains.
#[derive(Clone, Debug)]
pub struct SpecController {
    pub alpha_lookup: f64,
    pub alpha_model: f64,
}

impl Default for SpecController {
    fn default() -> Self {
        SpecController {
            alpha_lookup: 0.5,
            alpha_model: 0.6,
        }
    }
}

impl SpecController {
    /// Record a step: `drafted` tokens proposed, `accepted` of them kept.
    pub fn record(&mut self, from_model: bool, drafted: usize, accepted: usize) {
        if drafted == 0 {
            return;
        }
        // tokens tried: the accepted prefix plus the first rejection
        let tried = (accepted + 1).min(drafted) as f64;
        let rate = accepted as f64 / tried;
        let a = if from_model {
            &mut self.alpha_model
        } else {
            &mut self.alpha_lookup
        };
        *a = 0.8 * *a + 0.2 * rate;
    }
}

/// Step-time model of the batched forward pass: an EMA of the measured
/// time per batch size (rows), interpolated between observed sizes. The cost
/// curve is concave on CPU kernels that amortize weight decoding over row
/// tiles, so a single linear fit would misprice draft rows. Also tracks the
/// draft model's time per row it processes.
#[derive(Clone, Debug, Default)]
pub struct StepCost {
    t: Vec<f64>,
    pub t_draft_ns: f64,
}

impl StepCost {
    const MAX_ROWS: usize = 512;

    pub fn observe(&mut self, rows: usize, ns: f64) {
        if rows == 0 || rows >= Self::MAX_ROWS {
            return;
        }
        if self.t.len() <= rows {
            self.t.resize(rows + 1, 0.0);
        }
        let v = &mut self.t[rows];
        *v = if *v <= 0.0 { ns } else { 0.7 * *v + 0.3 * ns };
    }

    pub fn observe_draft(&mut self, ns_per_row: f64) {
        self.t_draft_ns = if self.t_draft_ns <= 0.0 {
            ns_per_row
        } else {
            0.8 * self.t_draft_ns + 0.2 * ns_per_row
        };
    }

    /// Estimated step time for `rows` (None before any observation).
    pub fn estimate(&self, rows: usize) -> Option<f64> {
        let known: Vec<(usize, f64)> = self
            .t
            .iter()
            .enumerate()
            .filter(|(_, v)| **v > 0.0)
            .map(|(r, v)| (r, *v))
            .collect();
        let first = *known.first()?;
        if let Some(&(_, v)) = known.iter().find(|(r, _)| *r == rows) {
            return Some(v);
        }
        let lo = known.iter().rev().find(|(r, _)| *r < rows).copied();
        let hi = known.iter().find(|(r, _)| *r > rows).copied();
        Some(match (lo, hi) {
            (Some((r0, t0)), Some((r1, t1))) => {
                t0 + (t1 - t0) * (rows - r0) as f64 / (r1 - r0) as f64
            }
            (Some((r0, t0)), None) => {
                // extrapolate with the last observed slope, or assume an
                // extra row costs a quarter of a one-row step (exploration)
                let prev = known.iter().rev().find(|(r, _)| *r < r0);
                let slope = match prev {
                    Some(&(r, t)) => ((t0 - t) / (r0 - r) as f64).max(0.0),
                    None => 0.25 * t0 / r0 as f64,
                };
                t0 + slope * (rows - r0) as f64
            }
            (None, Some((r1, t1))) => t1 * (0.75 + 0.25 * rows as f64 / r1 as f64),
            (None, None) => first.1,
        })
    }
}

/// One session's drafting options for the next step.
pub struct DraftOffer {
    /// Context-copy continuation (free) and its estimated acceptance.
    pub lookup: Vec<u32>,
    pub alpha_lookup: f64,
    /// Draft-model chains available (cost `t_draft` per token).
    pub model: bool,
    pub alpha_model: f64,
    /// Longest chain allowed (remaining output room, `k_max`).
    pub cap: usize,
}

/// Global chain lengths for one batched step: starting from one row per
/// session (plus `fixed_rows` of prefill), add the draft token with the
/// highest acceptance probability α^j while it raises expected tokens per
/// unit time: `α^j·T > E·(b + t_draft)` with T = a + b·rows and E the
/// expected tokens so far. Returns (k, from_model) per session.
pub fn plan_chains(
    offers: &[DraftOffer],
    cost: &StepCost,
    fixed_rows: usize,
) -> Vec<(usize, bool)> {
    let mut out: Vec<(usize, bool)> = offers
        .iter()
        .map(|o| (0, o.lookup.is_empty() && o.model))
        .collect();
    let base = fixed_rows + offers.len();
    if cost.estimate(base).is_none() {
        // no timings yet: a short chain each
        return offers
            .iter()
            .map(|o| {
                if !o.lookup.is_empty() {
                    (o.lookup.len().min(o.cap).min(4), false)
                } else if o.model {
                    (o.cap.min(4), true)
                } else {
                    (0, false)
                }
            })
            .collect();
    }
    let t = |r: usize| cost.estimate(r).unwrap();
    let mut rows = base;
    let mut e = offers.len() as f64;
    // acceptance probability of each session's next draft token
    let mut next_p: Vec<f64> = offers
        .iter()
        .map(|o| {
            if !o.lookup.is_empty() {
                o.alpha_lookup
            } else if o.model {
                o.alpha_model
            } else {
                0.0
            }
        })
        .collect();
    let mut draft_ns = 0.0;
    loop {
        let mut best: Option<(usize, f64)> = None;
        for (i, o) in offers.iter().enumerate() {
            let (k, model) = out[i];
            let avail = if model {
                o.cap
            } else {
                o.lookup.len().min(o.cap)
            };
            if k < avail && best.is_none_or(|(_, p)| next_p[i] > p) {
                best = Some((i, next_p[i]));
            }
        }
        let Some((i, p)) = best else { break };
        let dt = if out[i].1 { cost.t_draft_ns } else { 0.0 };
        // tokens per ns now vs with this draft row added
        let now = e / (t(rows) + draft_ns);
        let with = (e + p) / (t(rows + 1) + draft_ns + dt);
        if with <= now {
            break;
        }
        rows += 1;
        e += p;
        draft_ns += dt;
        out[i].0 += 1;
        let al = if out[i].1 {
            offers[i].alpha_model
        } else {
            offers[i].alpha_lookup
        };
        next_p[i] *= al;
    }
    out
}

// --- concurrent client ------------------------------------------------------------

/// Events of one request multiplexed by [`BatchClient`].
pub enum ClientEvent {
    Burst(TokenBurst),
    Done(GenerateResult),
}

/// A closure run on the dispatcher thread with the driver.
type DriverFn = Box<dyn FnOnce(&Driver) + Send>;

enum Job {
    Generate(GenerateRequest, GenerateExt, mpsc::Sender<ClientEvent>),
    /// Run on the dispatcher thread with the driver: immediately (`false`:
    /// status queries, which the head answers between steps) or once no
    /// request is in flight (`true`: probing, re-planning).
    Exec(DriverFn, bool),
}

/// Concurrent requests over one driver endpoint: a dispatcher thread owns
/// the driver, tags every request with an id (frame header `tree_epoch`) and
/// routes streamed bursts and results back to the requesting thread. The
/// head's scheduler batches whatever is in flight.
#[derive(Clone)]
pub struct BatchClient {
    tx: mpsc::Sender<Job>,
    head: std::sync::Arc<std::sync::atomic::AtomicU16>,
}

impl BatchClient {
    pub fn new(driver: Driver, head: u16) -> Self {
        let (tx, rx) = mpsc::channel::<Job>();
        let head = std::sync::Arc::new(std::sync::atomic::AtomicU16::new(head));
        let head_t = head.clone();
        std::thread::Builder::new()
            .name("batch-client".into())
            .spawn(move || {
                let mut next_id = 1u64;
                let mut waiting: HashMap<u64, mpsc::Sender<ClientEvent>> = HashMap::new();
                let mut execs: std::collections::VecDeque<DriverFn> = Default::default();
                loop {
                    // admit new jobs (block only when idle)
                    loop {
                        let idle = waiting.is_empty() && execs.is_empty();
                        let job = if idle {
                            match rx.recv() {
                                Ok(j) => Some(j),
                                Err(_) => return,
                            }
                        } else {
                            rx.try_recv().ok()
                        };
                        match job {
                            None => break,
                            Some(Job::Exec(f, false)) => f(&driver),
                            Some(Job::Exec(f, true)) => execs.push_back(f),
                            Some(Job::Generate(req, ext, reply)) => {
                                let id = next_id;
                                next_id += 1;
                                let mut payload = req.encode();
                                payload.extend(ext.encode());
                                let h = head_t.load(std::sync::atomic::Ordering::Relaxed);
                                let _ =
                                    driver
                                        .ep
                                        .send_payload(h, FrameType::Generate, id, 0, payload);
                                waiting.insert(id, reply);
                            }
                        }
                    }
                    if waiting.is_empty() {
                        if let Some(f) = execs.pop_front() {
                            f(&driver);
                        }
                        continue;
                    }
                    if let Some(f) = driver.ep.recv_match_timeout(
                        &|f| {
                            matches!(
                                f.header.frame_type,
                                FrameType::GenerateResult | FrameType::TokenStream
                            )
                        },
                        Duration::from_millis(2),
                    ) {
                        let id = f.header.tree_epoch;
                        match f.header.frame_type {
                            FrameType::TokenStream => {
                                if let Some(r) = waiting.get(&id) {
                                    let _ =
                                        r.send(ClientEvent::Burst(TokenBurst::decode(&f.payload)));
                                }
                            }
                            _ => {
                                if let Some(r) = waiting.remove(&id) {
                                    let _ = r.send(ClientEvent::Done(GenerateResult::decode(
                                        &f.payload,
                                    )));
                                }
                            }
                        }
                    }
                }
            })
            .expect("spawn batch client");
        BatchClient { tx, head }
    }

    /// The head node new requests go to (after a re-plan).
    pub fn set_head(&self, head: u16) {
        self.head.store(head, std::sync::atomic::Ordering::Relaxed);
    }

    /// Submit a request; events arrive on the returned receiver.
    pub fn submit(&self, req: GenerateRequest, ext: GenerateExt) -> mpsc::Receiver<ClientEvent> {
        let (tx, rx) = mpsc::channel();
        self.tx
            .send(Job::Generate(req, ext, tx))
            .expect("batch client stopped");
        rx
    }

    /// Run `f` with the driver once no request is in flight; blocks for
    /// its result.
    pub fn exec<T: Send + 'static>(&self, f: impl FnOnce(&Driver) -> T + Send + 'static) -> T {
        self.exec_with(f, true)
    }

    /// Run `f` with the driver right away, between in-flight requests'
    /// frames (for queries the head answers while it serves).
    pub fn exec_now<T: Send + 'static>(&self, f: impl FnOnce(&Driver) -> T + Send + 'static) -> T {
        self.exec_with(f, false)
    }

    fn exec_with<T: Send + 'static>(
        &self,
        f: impl FnOnce(&Driver) -> T + Send + 'static,
        idle: bool,
    ) -> T {
        let (tx, rx) = mpsc::channel();
        self.tx
            .send(Job::Exec(
                Box::new(move |d| {
                    let _ = tx.send(f(d));
                }),
                idle,
            ))
            .expect("batch client stopped");
        rx.recv().expect("batch client stopped")
    }

    /// Generate (blocking), calling `on_burst` for streamed steps.
    pub fn generate_stream(
        &self,
        req: &GenerateRequest,
        ext: &GenerateExt,
        on_burst: &mut dyn FnMut(&TokenBurst),
    ) -> GenerateResult {
        let rx = self.submit(req.clone(), ext.clone());
        loop {
            match rx
                .recv_timeout(Duration::from_secs(900))
                .expect("head did not answer")
            {
                ClientEvent::Burst(b) => on_burst(&b),
                ClientEvent::Done(r) => return r,
            }
        }
    }

    pub fn generate(&self, req: &GenerateRequest) -> GenerateResult {
        self.generate_stream(req, &GenerateExt::default(), &mut |_| {})
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suffix_index_finds_the_longest_repeated_suffix() {
        let mut s = SuffixIndex::new();
        // a b c d x a b c → longest earlier suffix "a b c", followed by "d x"
        s.extend(&[1, 2, 3, 4, 9, 1, 2, 3]);
        assert_eq!(s.lookup(3), (3, vec![4, 9, 1]));
        // the latest occurrence wins: … a b c d x a b c y a b c → "y"
        s.extend(&[7, 1, 2, 3]);
        assert_eq!(s.lookup(2), (3, vec![7, 1]));
        // brute force agreement on a random-ish sequence
        let seq: Vec<u32> = (0..400u32).map(|i| (i * 7 + i / 13) % 11).collect();
        let mut s = SuffixIndex::new();
        for (i, &t) in seq.iter().enumerate() {
            s.push(t);
            let h = &seq[..=i];
            let best = (1..=i)
                .rev()
                .find(|&n| (0..=i - n).any(|st| h[st..st + n] == h[i + 1 - n..]))
                .unwrap_or(0);
            assert_eq!(s.lookup(1).0, best, "at {i}");
            if best > 0 {
                let (_, c) = s.lookup(4);
                // the continuation follows an earlier occurrence of the suffix
                let e = (0..=i - best)
                    .rev()
                    .find(|&st| h[st..st + best] == h[i + 1 - best..])
                    .unwrap();
                assert!(c.len() <= 4 && !c.is_empty());
                let _ = e;
            }
        }
    }

    #[test]
    fn batch_payloads_round_trip() {
        let st = BatchStep {
            segments: vec![
                Segment {
                    slot: 2,
                    reset_to: Some(5),
                    prefill: true,
                    meta: BatchMeta::causal_prefill(5, 3, 9),
                    tokens: vec![1, 2, 3],
                },
                Segment {
                    slot: 0,
                    reset_to: None,
                    prefill: false,
                    meta: BatchMeta::causal_prefill(40, 1, 10),
                    tokens: vec![7],
                },
            ],
        };
        let d = BatchStep::decode(&st.encode());
        assert_eq!(d.segments.len(), 2);
        assert_eq!(d.segments[0].reset_to, Some(5));
        assert_eq!(d.segments[1].tokens, vec![7]);
        assert_eq!(d.positions(), vec![5, 6, 7, 40]);
        let c = BatchCommit {
            slots: vec![(1, vec![3, 4]), (0, vec![])],
        };
        assert_eq!(BatchCommit::decode(&c.encode()), c);
    }
}
