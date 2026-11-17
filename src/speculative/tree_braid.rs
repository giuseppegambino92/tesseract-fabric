//! Speculative token trees: 2-D causal tree masks, RoPE depth positions,
//! exact (lossless) multi-branch tree rejection sampling, braid merging of
//! consecutive epochs, and the 32-byte `TreeCommitFrame`.

use crate::kv_fabric::{node_key, BatchMeta, NodeKey};
use crate::tensor::Rng;

/// How a node's siblings were proposed (determines the exact acceptance rule).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProposalKind {
    /// Deterministic distinct candidates (top-k / best-first expansion):
    /// one-hot proposals ⇒ sequential residual sampling without replacement.
    Deterministic,
    /// Children drawn i.i.d. from the draft distribution q (SpecInfer MSS).
    SampledIid,
}

/// A speculative prefix tree. Node 0 is the root: the most recent *verified*
/// token, whose KV has not been computed yet. Parents precede children.
#[derive(Clone, Debug, PartialEq)]
pub struct SpeculativeTree {
    pub epoch: u64,
    pub tokens: Vec<u32>,
    pub parents: Vec<i32>,
    pub depth: Vec<usize>,
    /// Draft distribution (sparse: token, prob) each node used to propose its
    /// children. Needed for `SampledIid` verification.
    pub child_q: Vec<Vec<(u32, f32)>>,
    /// Cumulative draft path probability (best-first priority).
    pub path_q: Vec<f32>,
    pub proposal: ProposalKind,
}

impl SpeculativeTree {
    /// A linear chain root → t₁ → t₂ → … (deterministic proposals).
    pub fn chain(epoch: u64, root: u32, rest: &[u32]) -> Self {
        let n = rest.len() + 1;
        let mut child_q = vec![vec![]; n];
        for i in 0..rest.len() {
            child_q[i] = vec![(rest[i], 1.0)];
        }
        SpeculativeTree {
            epoch,
            tokens: std::iter::once(root).chain(rest.iter().copied()).collect(),
            parents: (0..n).map(|i| i as i32 - 1).collect(),
            depth: (0..n).collect(),
            child_q,
            path_q: vec![1.0; n],
            proposal: ProposalKind::Deterministic,
        }
    }

    pub fn single(epoch: u64, root: u32) -> Self {
        SpeculativeTree {
            epoch,
            tokens: vec![root],
            parents: vec![-1],
            depth: vec![0],
            child_q: vec![vec![]],
            path_q: vec![1.0],
            proposal: ProposalKind::Deterministic,
        }
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn children(&self, i: usize) -> Vec<usize> {
        (0..self.len())
            .filter(|&j| self.parents[j] == i as i32)
            .collect()
    }

    pub fn max_depth(&self) -> usize {
        self.depth.iter().copied().max().unwrap_or(0)
    }

    /// Ancestors of `i` (root first), including `i`.
    pub fn path_to(&self, i: usize) -> Vec<usize> {
        let mut p = vec![i];
        let mut c = i as i32;
        while self.parents[c as usize] >= 0 {
            c = self.parents[c as usize];
            p.push(c as usize);
        }
        p.reverse();
        p
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(16 + self.len() * 12);
        b.extend_from_slice(&self.epoch.to_le_bytes());
        b.push(match self.proposal {
            ProposalKind::Deterministic => 0,
            ProposalKind::SampledIid => 1,
        });
        b.extend_from_slice(&(self.len() as u16).to_le_bytes());
        for i in 0..self.len() {
            b.extend_from_slice(&self.tokens[i].to_le_bytes());
            b.extend_from_slice(&(self.parents[i] as i16).to_le_bytes());
            b.extend_from_slice(&self.path_q[i].to_le_bytes());
            b.extend_from_slice(&(self.child_q[i].len() as u16).to_le_bytes());
            for &(t, q) in &self.child_q[i] {
                b.extend_from_slice(&t.to_le_bytes());
                b.extend_from_slice(&q.to_le_bytes());
            }
        }
        b
    }

    pub fn decode(b: &[u8]) -> (Self, usize) {
        let epoch = u64::from_le_bytes(b[0..8].try_into().unwrap());
        let proposal = if b[8] == 1 {
            ProposalKind::SampledIid
        } else {
            ProposalKind::Deterministic
        };
        let n = u16::from_le_bytes([b[9], b[10]]) as usize;
        let mut o = 11;
        let mut t = SpeculativeTree {
            epoch,
            tokens: vec![],
            parents: vec![],
            depth: vec![],
            child_q: vec![],
            path_q: vec![],
            proposal,
        };
        for _ in 0..n {
            t.tokens
                .push(u32::from_le_bytes(b[o..o + 4].try_into().unwrap()));
            let p = i16::from_le_bytes([b[o + 4], b[o + 5]]) as i32;
            t.parents.push(p);
            t.depth
                .push(if p < 0 { 0 } else { t.depth[p as usize] + 1 });
            t.path_q
                .push(f32::from_le_bytes(b[o + 6..o + 10].try_into().unwrap()));
            let nq = u16::from_le_bytes([b[o + 10], b[o + 11]]) as usize;
            o += 12;
            let mut q = Vec::with_capacity(nq);
            for _ in 0..nq {
                q.push((
                    u32::from_le_bytes(b[o..o + 4].try_into().unwrap()),
                    f32::from_le_bytes(b[o + 4..o + 8].try_into().unwrap()),
                ));
                o += 8;
            }
            t.child_q.push(q);
        }
        (t, o)
    }
}

/// Exact 2-D causal tree mask: M[i][j] = 0 iff j is an ancestor of i or j = i,
/// else −∞. Row-major N×N.
pub fn build_tree_mask(parents: &[i32]) -> Vec<f32> {
    let n = parents.len();
    let mut m = vec![f32::NEG_INFINITY; n * n];
    for i in 0..n {
        let mut c = i as i32;
        while c >= 0 {
            m[i * n + c as usize] = 0.0;
            c = parents[c as usize];
        }
    }
    m
}

/// Tree depth of every node (RoPE position offset from the root).
pub fn tree_depths(parents: &[i32]) -> Vec<usize> {
    let mut d = vec![0usize; parents.len()];
    for i in 0..parents.len() {
        if parents[i] >= 0 {
            assert!((parents[i] as usize) < i, "parents must precede children");
            d[i] = d[parents[i] as usize] + 1;
        }
    }
    d
}

/// A set of tree nodes computed together (one braid strand), possibly
/// hanging below nodes of an earlier in-flight epoch.
#[derive(Clone, Debug)]
pub struct BraidStrand {
    pub epoch: u64,
    /// Indices into the merged tree.
    pub nodes: Vec<usize>,
}

/// A merged tree spanning one or two braid epochs. Node `i` belongs to
/// `strand_of[i]`; its KV key is `(epoch, local index)`.
#[derive(Clone, Debug)]
pub struct MergedTree {
    pub tree: SpeculativeTree,
    pub strands: Vec<BraidStrand>,
    pub strand_of: Vec<usize>,
    pub local_index: Vec<usize>,
}

impl MergedTree {
    /// Split a best-first tree into strands of at most `strand_nodes` nodes.
    /// Best-first order guarantees each strand only depends on earlier ones.
    pub fn split(
        tree: SpeculativeTree,
        strand_nodes: usize,
        base_epoch: u64,
        max_strands: usize,
    ) -> Self {
        let n = tree.len();
        let mut strands = Vec::new();
        let mut strand_of = vec![0; n];
        let mut local_index = vec![0; n];
        for (s, chunk) in (0..n)
            .collect::<Vec<_>>()
            .chunks(strand_nodes.max(1))
            .enumerate()
            .take(max_strands)
        {
            for (li, &i) in chunk.iter().enumerate() {
                strand_of[i] = s;
                local_index[i] = li;
            }
            strands.push(BraidStrand {
                epoch: base_epoch + s as u64,
                nodes: chunk.to_vec(),
            });
        }
        let kept: usize = strands.iter().map(|s| s.nodes.len()).sum();
        let mut tree = tree;
        tree.tokens.truncate(kept);
        tree.parents.truncate(kept);
        tree.depth.truncate(kept);
        tree.child_q.truncate(kept);
        tree.path_q.truncate(kept);
        strand_of.truncate(kept);
        local_index.truncate(kept);
        MergedTree {
            tree,
            strands,
            strand_of,
            local_index,
        }
    }

    pub fn key(&self, i: usize) -> NodeKey {
        node_key(self.strands[self.strand_of[i]].epoch, self.local_index[i])
    }

    /// Batch metadata for strand `s`: positions = base_pos + depth; visible =
    /// all tree ancestors (possibly in earlier strands) and self.
    pub fn strand_meta(&self, s: usize, base_pos: usize) -> BatchMeta {
        let mut m = BatchMeta::default();
        for &i in &self.strands[s].nodes {
            m.positions.push(base_pos + self.tree.depth[i]);
            m.keys.push(self.key(i));
            m.visible.push(
                self.tree
                    .path_to(i)
                    .into_iter()
                    .map(|a| self.key(a))
                    .collect(),
            );
        }
        m
    }

    pub fn strand_tokens(&self, s: usize) -> Vec<u32> {
        self.strands[s]
            .nodes
            .iter()
            .map(|&i| self.tree.tokens[i])
            .collect()
    }
}

/// Result of verifying a tree against the target model.
#[derive(Clone, Debug, PartialEq)]
pub struct VerifyOutcome {
    /// Accepted nodes from the root (inclusive) along one path.
    pub path: Vec<usize>,
    /// Token sampled from the (residual) target distribution at the path end.
    pub bonus: u32,
}

impl VerifyOutcome {
    /// Tokens emitted this step: accepted non-root tokens + bonus.
    pub fn emitted(&self, tree: &SpeculativeTree) -> Vec<u32> {
        let mut v: Vec<u32> = self.path[1..].iter().map(|&i| tree.tokens[i]).collect();
        v.push(self.bonus);
        v
    }
}

fn sample_dense(p: &[f32], rng: &mut Rng) -> u32 {
    let total: f64 = p.iter().map(|v| *v as f64).sum();
    let r = rng.uniform_f64() * total;
    let mut c = 0.0f64;
    let mut last_nonzero = 0;
    for (i, &v) in p.iter().enumerate() {
        if v > 0.0 {
            last_nonzero = i;
        }
        c += v as f64;
        if r < c {
            return i as u32;
        }
    }
    last_nonzero as u32
}

/// Exact tree speculative sampling. `target(i)` returns the target model's
/// next-token distribution at node `i` (may block until that node's logits
/// arrive — this is how braid strands are verified lazily).
///
/// Deterministic proposals: at a node with residual p, try children in order;
/// accept child c with probability p(c)/Σp, else zero p(c) and renormalise.
/// This is sampling from p without replacement, hence exactly p-distributed.
/// I.i.d. proposals (SpecInfer): accept with min(1, p(c)/q(c)); on rejection
/// p ← norm(max(p − q, 0)).
pub fn verify_tree(
    tree: &SpeculativeTree,
    target: &mut dyn FnMut(usize) -> Vec<f32>,
    rng: &mut Rng,
) -> VerifyOutcome {
    let mut node = 0usize;
    let mut path = vec![0usize];
    loop {
        let mut p = target(node);
        let kids = tree.children(node);
        let mut accepted = None;
        for &c in &kids {
            let tok = tree.tokens[c] as usize;
            match tree.proposal {
                ProposalKind::Deterministic => {
                    let mass: f64 = p.iter().map(|v| *v as f64).sum();
                    if mass <= 0.0 {
                        break;
                    }
                    let a = p[tok] as f64 / mass;
                    if rng.uniform_f64() < a {
                        accepted = Some(c);
                        break;
                    }
                    p[tok] = 0.0;
                }
                ProposalKind::SampledIid => {
                    let q = &tree.child_q[node];
                    let qtok = q
                        .iter()
                        .find(|(t, _)| *t as usize == tok)
                        .map(|x| x.1)
                        .unwrap_or(0.0);
                    let mass: f64 = p.iter().map(|v| *v as f64).sum();
                    let ptok = p[tok] as f64 / mass;
                    if qtok > 0.0 && rng.uniform_f64() < (ptok / qtok as f64).min(1.0) {
                        accepted = Some(c);
                        break;
                    }
                    // residual: max(p_norm − q, 0)
                    let inv = 1.0 / mass as f32;
                    for v in p.iter_mut() {
                        *v *= inv;
                    }
                    for &(t, qv) in q {
                        let x = &mut p[t as usize];
                        *x = (*x - qv).max(0.0);
                    }
                }
            }
        }
        match accepted {
            Some(c) => {
                path.push(c);
                node = c;
            }
            None => {
                let bonus = sample_dense(&p, rng);
                return VerifyOutcome { path, bonus };
            }
        }
    }
}

/// Roofline dynamic speculation controller.
///
/// Chooses the tree size N maximising predicted throughput α̂(N) / T̂(N):
/// * α̂(N) = 1 + c · Σ_{i=1}^{N-1} q_i — the drafter's own best-first path
///   probabilities q_i (so N need not have been tried), calibrated by
///   c = EMA(accepted − 1) / EMA(predicted − 1) from verified epochs;
/// * T̂(N) = a + b·N — least-squares fit over the per-size epoch wall time of
///   every size used so far (compute ∝ rows plus a fixed network/merge share
///   a). Each size's time is the lower median of its last 8 epochs: on a
///   shared, contended host it ignores load spikes (the mean would follow
///   other tenants' load) without favouring the noisier tree sizes (a
///   minimum would). Tree sizes are stored relative to T̂(1), β(N) =
///   T̂(N)/T̂(1), and priced with the current T̂(1), which runs nearly every
///   epoch and so tracks the host's clock regime.
/// * A tree epoch is also charged what the following 1-row epoch loses when
///   the tree disables a cross-step optimisation (`charge_successor`).
/// * The controller outlives requests (a worker caches it per assignment),
///   so learned costs and acceptance carry over instead of re-exploring.
#[derive(Clone, Debug)]
pub struct RooflineController {
    pub candidates: Vec<usize>,
    pub ema: f64,
    wall: Vec<Option<f64>>,
    recent: Vec<std::collections::VecDeque<f64>>,
    model: Option<Vec<f64>>,
    acc_ema: f64,
    pred_ema: f64,
    last_q: Vec<f32>,
    epochs: u64,
    /// Epoch index at which each candidate was last measured.
    last_used: Vec<u64>,
    /// Per-size measured tokens per epoch (EMA) and sample count.
    acc_by_size: Vec<(f64, u32)>,
    pub net_share_ema: f64,
    /// Epochs recorded in the current request (the first pays warm-up).
    req_epochs: u64,
    /// A tree epoch disables a cross-step optimisation of the epoch after
    /// it (the pipeline's stage-0 lookahead runs only after single-row
    /// epochs): charge that epoch's excess over the 1-row cost to the tree.
    pub charge_successor: bool,
    /// Tree epoch whose cost waits for its successor: (size index, wall ns).
    deferred: Option<(usize, f64)>,
    /// Epoch of the next periodic probe / neighbour refresh, and the current
    /// interval: it starts at [`PROBE_EVERY`] and doubles (up to
    /// [`PROBE_EVERY_MAX`]) each time a probe tree is not accepted at all, so
    /// a drafter that never helps (a target far faster than its drafter)
    /// stops costing a tree epoch every 24.
    next_probe: u64,
    probe_every: u64,
}

/// Initial interval of the periodic re-probe, in epochs.
pub const PROBE_EVERY: u64 = 24;
/// Longest interval after repeated unaccepted probes.
pub const PROBE_EVERY_MAX: u64 = 384;

impl RooflineController {
    pub fn new(candidates: Vec<usize>) -> Self {
        let n = candidates.len();
        RooflineController {
            candidates,
            ema: 0.25,
            wall: vec![None; n],
            recent: vec![Default::default(); n],
            model: None,
            acc_ema: 0.0,
            pred_ema: 0.0,
            last_q: Vec::new(),
            epochs: 0,
            last_used: vec![0; n],
            acc_by_size: vec![(0.0, 0); n],
            net_share_ema: 0.0,
            req_epochs: 0,
            charge_successor: false,
            deferred: None,
            next_probe: PROBE_EVERY - 1,
            probe_every: PROBE_EVERY,
        }
    }

    /// Index of the 1-row candidate, if any.
    fn one(&self) -> Option<usize> {
        self.candidates.iter().position(|&c| c == 1)
    }

    /// Costs of tree sizes are kept *relative to the 1-row cost* (β(N),
    /// sampled when N runs), and turned into time with the current 1-row
    /// cost. The 1-row size runs nearly every epoch, so it follows the
    /// host's clock regime by itself; every size's absolute cost follows it
    /// without re-measuring, and without any regime threshold.
    fn relative(&self, i: usize) -> bool {
        self.one().is_some_and(|o| o != i)
    }

    /// Stored cost of candidate `i` in ns (lower median of recent samples).
    fn cost(&self, i: usize) -> Option<f64> {
        let v = self.wall[i]?;
        if self.relative(i) {
            Some(v * self.wall[self.one()?]?)
        } else {
            Some(v)
        }
    }

    /// Add one cost sample for candidate `i` (its cost = lower median of
    /// the last 8).
    fn push_time(&mut self, i: usize, wall: f64) {
        let wall = if self.relative(i) {
            match self.one().and_then(|o| self.wall[o]) {
                // N rows never cost less than one: a ratio below 1 only
                // means the 1-row reference was load-inflated
                Some(t1) => (wall / t1).max(1.0),
                // no 1-row reference yet (warm-up measures it first)
                None => return,
            }
        } else {
            wall
        };
        let r = &mut self.recent[i];
        r.push_back(wall);
        if r.len() > 8 {
            r.pop_front();
        }
        // lower median: robust to load spikes like a minimum, but not
        // biased toward each size's luckiest sample (tree epochs vary more
        // than 1-row epochs, so a minimum makes trees look cheaper)
        let mut v: Vec<f64> = r.iter().cloned().collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        self.wall[i] = Some(v[(v.len() - 1) / 2]);
    }

    /// Start a new request on a controller kept from earlier ones: learned
    /// acceptance and per-size costs carry over (no re-exploration), and
    /// tree costs are relative to the 1-row cost, so they hold in whatever
    /// clock regime the host is in now.
    pub fn begin_request(&mut self) {
        if let Some((j, w)) = self.deferred.take() {
            self.push_time(j, w);
        }
        self.req_epochs = 0;
    }

    fn predicted(q: &[f32], n: usize) -> f64 {
        let mut a = 1.0f64;
        for i in 1..n {
            a += match q.get(i) {
                Some(v) => *v as f64,
                // beyond the drafted tree: extend with its last marginal gain
                None => q.last().copied().unwrap_or(0.0) as f64,
            };
        }
        a
    }

    /// Expected tokens per epoch at tree size `n`: the size's own measured
    /// acceptance once it has been verified a few times (acceptance saturates
    /// with N, which one global calibration factor would overstate), the
    /// calibrated drafter model otherwise.
    pub fn alpha(&self, n: usize) -> f64 {
        if let Some(i) = self.candidates.iter().position(|&c| c == n) {
            let (a, k) = self.acc_by_size[i];
            if k >= 2 {
                return a;
            }
        }
        let c = if self.pred_ema > 1e-9 {
            (self.acc_ema / self.pred_ema).clamp(0.0, 2.0)
        } else {
            0.5
        };
        1.0 + c * (Self::predicted(&self.last_q, n) - 1.0)
    }

    /// Install a deterministic analytical epoch-time model (ns per candidate,
    /// from a startup calibration). Wall-clock samples are then ignored for
    /// cost, so host-load noise cannot steer the tree size.
    pub fn set_time_model(&mut self, model: Vec<f64>) {
        assert_eq!(model.len(), self.candidates.len());
        self.model = Some(model);
    }

    /// Epoch time (ns) at tree size `n`: the size's own measurement when it
    /// has one (MoE tree cost is concave in N — large trees saturate the
    /// experts — so a straight-line fit would overstate small trees), the
    /// linear fit otherwise.
    pub fn time(&self, n: usize) -> Option<f64> {
        if let Some(m) = &self.model {
            if let Some(i) = self.candidates.iter().position(|&c| c == n) {
                return Some(m[i]);
            }
        }
        if let Some(i) = self.candidates.iter().position(|&c| c == n) {
            if let Some(w) = self.cost(i) {
                return Some(w);
            }
        }
        let pts: Vec<(f64, f64)> = (0..self.candidates.len())
            .filter_map(|i| self.cost(i).map(|w| (self.candidates[i] as f64, w)))
            .collect();
        match pts.len() {
            0 => None,
            1 => Some(pts[0].1 * (0.75 + 0.25 * n as f64 / pts[0].0)),
            _ => {
                let k = pts.len() as f64;
                let (sx, sy) = pts.iter().fold((0.0, 0.0), |a, p| (a.0 + p.0, a.1 + p.1));
                let (mx, my) = (sx / k, sy / k);
                let sxx: f64 = pts.iter().map(|p| (p.0 - mx).powi(2)).sum();
                let sxy: f64 = pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
                let b = if sxx > 0.0 { (sxy / sxx).max(0.0) } else { 0.0 };
                Some((my + b * (n as f64 - mx)).max(1.0))
            }
        }
    }

    /// β(N): fitted epoch cost of N rows relative to 1 row.
    pub fn beta(&self, n: usize) -> Option<f64> {
        Some(self.time(n)? / self.time(1)?)
    }

    pub fn choose(&self) -> usize {
        if self.model.is_some() {
            // exploration: acceptance is unknown until a tree has been verified,
            // and is re-probed periodically while the controller sits at N = 1
            let probe = self.candidates.iter().copied().filter(|&c| c > 1).min();
            if let Some(p) = probe {
                if self.last_q.is_empty() || (self.probe_due() && self.epochs > 0) {
                    return p;
                }
            }
            let mut best = self.candidates[0];
            let mut bv = f64::MIN;
            for &n in &self.candidates {
                let v = self.alpha(n) / self.time(n).unwrap();
                if v > bv {
                    bv = v;
                    best = n;
                }
            }
            return best;
        }
        // warm-up: measure the two ends of the useful range first
        let unmeasured = |c: usize| {
            self.candidates
                .iter()
                .position(|&x| x == c)
                .map(|i| self.wall[i].is_none())
                .unwrap_or(false)
        };
        // warm-up: a single row (the reference every tree cost is relative
        // to), then the smallest tree; larger sizes are reached rung by rung
        // by the local search below
        let warm: Vec<usize> = self.candidates.iter().copied().take(2).collect();
        for c in warm {
            if unmeasured(c) {
                return c;
            }
        }
        // local search over the candidate ladder: every measured size, plus
        // the one unmeasured size just above the largest measured one (its
        // cost comes from the fit), so trees grow one rung at a time instead
        // of jumping to a large, possibly very expensive, size
        let n_c = self.candidates.len();
        let frontier = (0..n_c)
            .rev()
            .find(|&i| self.wall[i].is_some())
            .map(|i| i + 1);
        let allowed = |i: usize| self.wall[i].is_some() || Some(i) == frontier;
        let mut bi = 0;
        let mut bv = f64::MIN;
        for i in (0..n_c).filter(|&i| allowed(i)) {
            let n = self.candidates[i];
            if let Some(t) = self.time(n) {
                let v = self.alpha(n) / t;
                if v > bv {
                    bv = v;
                    bi = i;
                }
            }
        }
        // each size's cost is a min over its own recent samples, so a size
        // that stopped being chosen would keep whatever (possibly
        // load-inflated) samples it had: periodically refresh the staler of
        // the best size's two neighbours on the ladder
        // keep the 1-row reference fresh: every tree cost is relative to it
        if let Some(o) = self.one().filter(|&o| self.wall[o].is_some()) {
            if self.epochs >= self.last_used[o] + 8 {
                return 1;
            }
        }
        // (tree costs are relative to the always-fresh 1-row cost, so they
        // stay valid across clock regimes and need only rare refreshing)
        if self.probe_due() {
            let nb = [bi.checked_sub(1), (bi + 1 < n_c).then_some(bi + 1)];
            if let Some(i) = nb
                .into_iter()
                .flatten()
                .filter(|&i| allowed(i))
                .min_by_key(|&i| {
                    if self.wall[i].is_some() {
                        self.last_used[i]
                    } else {
                        0
                    }
                })
            {
                return self.candidates[i];
            }
        }
        self.candidates[bi]
    }

    /// Record a verified epoch: tree size used, the drafted tree's best-first
    /// path probabilities, tokens emitted, wall and compute time.
    fn probe_due(&self) -> bool {
        self.epochs >= self.next_probe
    }

    pub fn record(
        &mut self,
        n: usize,
        path_q: &[f32],
        emitted: usize,
        wall_ns: u64,
        compute_ns: u64,
    ) {
        if self.probe_due() && n > 1 {
            // a probe tree that emitted only the bonus token: back off
            self.probe_every = if emitted <= 1 {
                (self.probe_every * 2).min(PROBE_EVERY_MAX)
            } else {
                PROBE_EVERY
            };
            self.next_probe = self.epochs + self.probe_every;
        }
        let a = self.ema;
        // a single-node epoch carries no acceptance information
        if path_q.len() > 1 {
            let pred = Self::predicted(path_q, n) - 1.0;
            let acc = emitted as f64 - 1.0;
            if self.last_q.is_empty() {
                self.acc_ema = acc;
                self.pred_ema = pred;
            } else {
                self.acc_ema = (1.0 - a) * self.acc_ema + a * acc;
                self.pred_ema = (1.0 - a) * self.pred_ema + a * pred;
            }
            self.last_q = path_q.to_vec();
        }
        let idx = self.candidates.iter().position(|&c| c == n);
        if let Some(i) = idx {
            self.last_used[i] = self.epochs;
            let (a, k) = &mut self.acc_by_size[i];
            *a = if *k == 0 {
                emitted as f64
            } else {
                (1.0 - self.ema) * *a + self.ema * emitted as f64
            };
            *k += 1;
        }
        // settle the previous tree epoch: add what this 1-row epoch lost
        // (its excess over the stored 1-row cost) to the tree's cost
        if let Some((j, w)) = self.deferred.take() {
            let one = self.candidates.iter().position(|&c| c == 1);
            let extra = match (n, one.and_then(|o| self.wall[o])) {
                (1, Some(t1)) => (wall_ns as f64 - t1).max(0.0),
                _ => 0.0,
            };
            self.push_time(j, w + extra);
        }
        // the first epoch pays one-off warm-up (cold caches, first staging)
        if let Some(i) =
            idx.filter(|_| (self.epochs > 0 && self.req_epochs > 0) || self.candidates.len() == 1)
        {
            if self.charge_successor && n > 1 {
                self.deferred = Some((i, wall_ns as f64));
            } else {
                self.push_time(i, wall_ns as f64);
            }
        }
        let share = 1.0 - (compute_ns as f64 / wall_ns.max(1) as f64).min(1.0);
        self.net_share_ema = if self.epochs == 0 {
            share
        } else {
            (1.0 - a) * self.net_share_ema + a * share
        };
        self.epochs += 1;
        self.req_epochs += 1;
    }
}

/// 32-byte commit broadcast: all shards compact their KV to the accepted path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TreeCommitFrame {
    pub epoch: u64,
    /// Commits applied before this one (ordering guard for shards).
    pub commit_gen: u64,
    /// Bits 0..31: accepted nodes of strand `epoch`; bits 32..63: of `epoch+1`.
    pub accepted_mask: u64,
    pub bonus_token: u32,
    pub accepted_len: u8,
    pub strands: u8,
    pub reserved: u16,
}

impl TreeCommitFrame {
    pub const BYTES: usize = 32;

    pub fn from_outcome(m: &MergedTree, out: &VerifyOutcome, commit_gen: u64) -> Self {
        let mut mask = 0u64;
        for &i in &out.path {
            let s = m.strand_of[i];
            assert!(
                s < 2 && m.local_index[i] < 32,
                "commit mask covers two strands of ≤32 nodes"
            );
            mask |= 1u64 << (s * 32 + m.local_index[i]);
        }
        TreeCommitFrame {
            epoch: m.strands[0].epoch,
            commit_gen,
            accepted_mask: mask,
            bonus_token: out.bonus,
            accepted_len: out.path.len() as u8,
            strands: m.strands.len() as u8,
            reserved: 0,
        }
    }

    /// Accepted KV keys, in no particular order (shards sort by position).
    pub fn accepted_keys(&self) -> Vec<NodeKey> {
        (0..64)
            .filter(|b| self.accepted_mask >> b & 1 == 1)
            .map(|b| node_key(self.epoch + (b / 32) as u64, (b % 32) as usize))
            .collect()
    }

    pub fn encode(&self) -> [u8; 32] {
        let mut b = [0u8; 32];
        b[0..8].copy_from_slice(&self.epoch.to_le_bytes());
        b[8..16].copy_from_slice(&self.commit_gen.to_le_bytes());
        b[16..24].copy_from_slice(&self.accepted_mask.to_le_bytes());
        b[24..28].copy_from_slice(&self.bonus_token.to_le_bytes());
        b[28] = self.accepted_len;
        b[29] = self.strands;
        b[30..32].copy_from_slice(&self.reserved.to_le_bytes());
        b
    }

    pub fn decode(b: &[u8]) -> Self {
        TreeCommitFrame {
            epoch: u64::from_le_bytes(b[0..8].try_into().unwrap()),
            commit_gen: u64::from_le_bytes(b[8..16].try_into().unwrap()),
            accepted_mask: u64::from_le_bytes(b[16..24].try_into().unwrap()),
            bonus_token: u32::from_le_bytes(b[24..28].try_into().unwrap()),
            accepted_len: b[28],
            strands: b[29],
            reserved: u16::from_le_bytes([b[30], b[31]]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tree_is_charged_for_the_lookahead_its_successor_loses() {
        let ms = 1_000_000u64;
        let mut c = RooflineController::new(vec![1, 2]);
        c.charge_successor = true;
        c.record(1, &[1.0], 1, 10 * ms, 0); // warm-up epoch
        for _ in 0..3 {
            c.record(1, &[1.0], 1, 10 * ms, 0); // 1-row epochs with lookahead
        }
        // a 2-row epoch (14 ms) leaves the next 1-row epoch without its
        // lookahead (18 ms instead of 10): the tree costs 14 + 8 ms
        c.record(2, &[1.0, 0.6], 2, 14 * ms, 0);
        assert!(
            c.time(2).unwrap() < 14.0 * ms as f64,
            "deferred until its successor"
        );
        c.record(1, &[1.0], 1, 18 * ms, 0);
        assert_eq!(c.time(2).map(|t| t as u64), Some(22 * ms));
        assert_eq!(c.time(1).map(|t| t as u64), Some(10 * ms));
        // 2 tokens / 22 ms < 1 token / 10 ms: stay at one row
        assert_eq!(c.choose(), 1);
    }

    #[test]
    fn planner_costs_follow_the_clock_regime_without_re_measuring_trees() {
        let ms = 1_000_000u64;
        let mut c = RooflineController::new(vec![1, 2]);
        // request 1 (fast regime): 1 row 10 ms, 2 rows 12 ms
        for k in 0..12 {
            let (n, t) = if k % 2 == 0 {
                (1, 10 * ms)
            } else {
                (2, 12 * ms)
            };
            let q: Vec<f32> = if n == 2 { vec![1.0, 0.6] } else { vec![1.0] };
            c.record(n, &q, if n == 2 { 2 } else { 1 }, t, t / 2);
        }
        let t2 = c.time(2).unwrap();
        assert!((t2 / ms as f64 - 12.0).abs() < 0.01, "{t2}");
        // request 2 on the same controller: the host is now 1.8× slower and
        // only single rows run; the 2-row cost follows the 1-row cost
        c.begin_request();
        for _ in 0..10 {
            c.record(1, &[1.0], 1, 18 * ms, 9 * ms);
        }
        let r = c.time(2).unwrap() / t2;
        assert!((1.75..1.85).contains(&r), "2-row cost scaled by {r}");
    }

    #[test]
    fn roofline_prefers_trees_when_acceptance_beats_cost() {
        let mut c = RooflineController::new(vec![1, 4, 6, 8, 16]);
        let q = vec![1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3];
        // cheap rows (epoch time dominated by a fixed network share)
        for _ in 0..5 {
            c.record(4, &q, 4, 10_000_000, 2_000_000);
            c.record(1, &q, 1, 9_000_000, 1_000_000);
        }
        assert!(c.choose() >= 6, "chose {}", c.choose());
        // expensive rows, poor acceptance
        let mut d = RooflineController::new(vec![1, 4, 6, 8, 16]);
        for _ in 0..5 {
            d.record(4, &q, 1, 40_000_000, 39_000_000);
            d.record(1, &q, 1, 10_000_000, 9_000_000);
        }
        assert_eq!(d.choose(), 1);
    }

    #[test]
    fn roofline_recovers_from_a_load_inflated_single_row_sample() {
        // true cost: 10 ms + 6 ms per extra row; acceptance saturates at 2.5
        // tokens, so N = 1 (0.100 tok/ms) beats every tree (≤ 0.088 tok/ms)
        let cost = |n: usize| 10_000_000u64 + 6_000_000 * (n as u64 - 1);
        let acc = |n: usize| if n == 1 { 1 } else { 2 + (n >= 8) as usize };
        let q = vec![1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05];
        let mut c = RooflineController::new(vec![1, 4, 6, 8, 16]);
        c.record(4, &q, 2, cost(4) * 3, cost(4));
        // the only early N = 1 sample is inflated 4× by a transient load spike
        c.record(1, &q, 1, cost(1) * 4, cost(1));
        let mut picks = Vec::new();
        for _ in 0..90 {
            let n = c.choose();
            picks.push(n);
            c.record(n, &q, acc(n), cost(n), cost(n));
        }
        let ones = picks[30..].iter().filter(|&&n| n == 1).count();
        assert!(
            ones * 10 >= 60 * 7,
            "N = 1 chosen {ones}/60 times: {picks:?}"
        );
    }

    #[test]
    fn roofline_uses_measured_cost_for_concave_tree_cost() {
        // concave cost: 64 ms at N = 1, 200 ms at 4, 350 ms at 16; acceptance
        // 1 / 2.2 / 3.0 tokens → N = 1 maximises tokens per second, but a
        // straight line through these points overstates T(1) (≈ 98 ms)
        let cost = |n: usize| -> u64 {
            match n {
                1 => 64_000_000,
                4 => 200_000_000,
                6 => 240_000_000,
                8 => 280_000_000,
                _ => 350_000_000,
            }
        };
        let acc = |n: usize| match n {
            1 => 1,
            4 => 2,
            _ => 3,
        };
        let q = vec![1.0, 0.8, 0.6, 0.45, 0.35, 0.3, 0.25, 0.2];
        let mut c = RooflineController::new(vec![1, 4, 6, 8, 16]);
        for n in [4usize, 1, 4, 1, 16, 1, 8, 6, 1, 16] {
            c.record(n, &q, acc(n), cost(n), cost(n));
        }
        let picks: Vec<usize> = (0..12).map(|_| c.choose()).collect();
        assert_eq!(c.time(1), Some(64_000_000.0));
        assert_eq!(picks[0], 1, "picks {picks:?}");
    }

    #[test]
    fn mask_matches_ancestry() {
        // 0 -> 1 -> 3 ; 0 -> 2
        let parents = [-1, 0, 0, 1];
        let m = build_tree_mask(&parents);
        let z = |i: usize, j: usize| m[i * 4 + j] == 0.0;
        assert!(z(3, 0) && z(3, 1) && z(3, 3) && !z(3, 2));
        assert!(z(2, 0) && !z(2, 1));
        assert_eq!(tree_depths(&parents), vec![0, 1, 1, 2]);
    }

    #[test]
    fn commit_frame_is_32_bytes() {
        let f = TreeCommitFrame {
            epoch: 7,
            commit_gen: 3,
            accepted_mask: 0b1011 | (1 << 33),
            bonus_token: 42,
            accepted_len: 4,
            strands: 2,
            reserved: 0,
        };
        let e = f.encode();
        assert_eq!(e.len(), TreeCommitFrame::BYTES);
        assert_eq!(TreeCommitFrame::decode(&e), f);
        assert_eq!(f.accepted_keys().len(), 4);
    }
}
