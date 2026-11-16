//! KV cache management.
//!
//! * `LocalKvCache` — per-layer committed KV + a *pending* set of speculative
//!   tree-node entries keyed by `NodeKey`. Tree attention reads the committed
//!   prefix plus exactly the pending ancestors listed in `BatchMeta::visible`.
//!   On commit the accepted path is compacted in place onto the committed
//!   prefix and the rejected branches are dropped.
//! * `page_table` — distributed paged KV-MMU (placement across nodes by RAM).
//! * `disaggregated_attention` — zero-migration attention with exact
//!   associative log-sum-exp merging of per-shard partials.

pub mod disaggregated_attention;
pub mod page_table;

pub use disaggregated_attention::{
    lse_merge, partial_entries, partial_strided, AttnPartial, DisaggregatedKVFabric, KvShard,
    QueryHeadMeta, WirePrecision,
};
pub use page_table::{KvPageTable, PlacementPolicy};

use crate::tensor::Tensor2D;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Identifies a KV entry produced by a speculative tree node: (epoch, node).
pub type NodeKey = u64;

#[inline]
pub fn node_key(epoch: u64, node: usize) -> NodeKey {
    (epoch << 16) | node as u64
}
#[inline]
pub fn key_epoch(k: NodeKey) -> u64 {
    k >> 16
}

/// Per-batch attention metadata.
#[derive(Clone, Debug, Default)]
pub struct BatchMeta {
    /// RoPE position of each row.
    pub positions: Vec<usize>,
    /// Key under which each row's new K/V is stored (pending).
    pub keys: Vec<NodeKey>,
    /// Pending entries each row may attend to (its tree ancestors, itself, and
    /// any anchor-path entries from an in-flight earlier epoch).
    pub visible: Vec<Vec<NodeKey>>,
}

impl BatchMeta {
    /// Plain causal batch of `n` consecutive tokens starting at `start_pos`.
    pub fn causal_prefill(start_pos: usize, n: usize, epoch: u64) -> Self {
        let keys: Vec<NodeKey> = (0..n).map(|i| node_key(epoch, i)).collect();
        BatchMeta {
            positions: (start_pos..start_pos + n).collect(),
            visible: (0..n).map(|i| keys[..=i].to_vec()).collect(),
            keys,
        }
    }

    pub fn rows(&self) -> usize {
        self.positions.len()
    }

    pub fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.rows() as u32).to_le_bytes());
        for i in 0..self.rows() {
            out.extend_from_slice(&(self.positions[i] as u32).to_le_bytes());
            out.extend_from_slice(&self.keys[i].to_le_bytes());
            out.extend_from_slice(&(self.visible[i].len() as u16).to_le_bytes());
            for k in &self.visible[i] {
                out.extend_from_slice(&k.to_le_bytes());
            }
        }
    }

    pub fn decode(b: &[u8]) -> (Self, usize) {
        let n = u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
        let mut o = 4;
        let mut m = BatchMeta::default();
        for _ in 0..n {
            m.positions
                .push(u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as usize);
            m.keys
                .push(u64::from_le_bytes(b[o + 4..o + 12].try_into().unwrap()));
            let nv = u16::from_le_bytes(b[o + 12..o + 14].try_into().unwrap()) as usize;
            o += 14;
            let mut v = Vec::with_capacity(nv);
            for _ in 0..nv {
                v.push(u64::from_le_bytes(b[o..o + 8].try_into().unwrap()));
                o += 8;
            }
            m.visible.push(v);
        }
        (m, o)
    }
}

/// Attention provider used by `ShardModel::forward`.
pub trait AttentionBackend {
    /// `q`: [m, n_heads·d_head] (RoPE applied); `k`,`v`: [m, n_kv·d_head] for
    /// the batch rows (RoPE applied to k). Stores the new rows as pending and
    /// returns the attention output [m, n_heads·d_head].
    fn attend(
        &mut self,
        layer: usize,
        q: &Tensor2D,
        k: &Tensor2D,
        v: &Tensor2D,
        meta: &BatchMeta,
    ) -> Tensor2D;
}

/// One layer's K/V slab: row `slot` of every layer belongs to the same token,
/// so slot ids are allocated once per node and shared by all layers.
#[derive(Clone, Debug, Default)]
pub struct LayerSlab {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
}

/// Committed rows returned to the caller when committed KV lives remotely:
/// (layer, k rows, v rows) — rows are concatenated `kv_dim` vectors.
pub type CommittedRows = Vec<(usize, Vec<f32>, Vec<f32>)>;

/// Node-wide speculative slot state, shared by every layer of a shard.
///
/// `live` is a lock-free bitset of `AtomicU64` words over the pending entries
/// (a speculative tree has ≤64 rows, i.e. one word; long causal prefills use
/// more). Committing a verified tree path is one atomic store of the surviving
/// bits plus appending the accepted slot ids to the committed order —
/// independent of the number of layers and with no K/V copying (accepted rows
/// stay where they were written). Rejected slots are recycled for later trees.
#[derive(Debug, Default)]
pub struct SlotState {
    pub live: Vec<AtomicU64>,
    /// Pending entry i → (node key, slot).
    pub entries: Vec<(NodeKey, u32)>,
    /// Committed token order: position i of the context lives in slot order[i].
    pub order: Vec<u32>,
    pub free: Vec<u32>,
    pub next_slot: u32,
    /// True while order[i] == i for all i (contiguous fast path).
    pub contiguous: bool,
}

impl Clone for SlotState {
    fn clone(&self) -> Self {
        SlotState {
            live: self
                .live
                .iter()
                .map(|w| AtomicU64::new(w.load(Ordering::Acquire)))
                .collect(),
            entries: self.entries.clone(),
            order: self.order.clone(),
            free: self.free.clone(),
            next_slot: self.next_slot,
            contiguous: self.contiguous,
        }
    }
}

impl SlotState {
    fn alloc(&mut self) -> u32 {
        self.free.pop().unwrap_or_else(|| {
            self.next_slot += 1;
            self.next_slot - 1
        })
    }

    #[inline]
    pub fn is_live(&self, i: usize) -> bool {
        self.live
            .get(i / 64)
            .map(|w| w.load(Ordering::Acquire) >> (i % 64) & 1 == 1)
            .unwrap_or(false)
    }

    pub fn live_count(&self) -> usize {
        self.live
            .iter()
            .map(|w| w.load(Ordering::Acquire).count_ones() as usize)
            .sum()
    }

    /// Slot of a live pending key.
    pub fn slot_of(&self, key: NodeKey) -> Option<u32> {
        self.entries
            .iter()
            .enumerate()
            .find(|(i, e)| self.is_live(*i) && e.0 == key)
            .map(|(_, e)| e.1)
    }
}

/// KV cache coupled to the shard that computes the layer: per-layer slabs
/// plus one node-wide `SlotState`.
#[derive(Clone, Debug)]
pub struct LocalKvCache {
    pub n_heads: usize,
    pub n_kv: usize,
    pub d_head: usize,
    pub layers: HashMap<usize, LayerSlab>,
    pub slots: SlotState,
    /// When false, committed KV lives elsewhere (disaggregated fabric): the
    /// commit returns the rows and frees their slots.
    pub hold_committed: bool,
    /// Exact attention: every (row, head) is one two-pass softmax over its
    /// keys in context order — committed, then visible pending — so the
    /// result does not depend on which keys happen to be committed yet
    /// (prefix reuse, chunked prefill, batched sessions and multi-row draft
    /// verification all reproduce one-token-at-a-time decoding bit for bit).
    pub exact: bool,
    scratch: Vec<f32>,
}

impl LocalKvCache {
    pub fn new(n_heads: usize, n_kv: usize, d_head: usize) -> Self {
        LocalKvCache {
            n_heads,
            n_kv,
            d_head,
            layers: HashMap::new(),
            slots: SlotState {
                contiguous: true,
                ..Default::default()
            },
            hold_committed: true,
            exact: false,
            scratch: Vec::new(),
        }
    }

    /// The same cache with exact (split-invariant) attention on or off.
    pub fn with_exact(mut self, exact: bool) -> Self {
        self.exact = exact;
        self
    }

    pub fn kv_dim(&self) -> usize {
        self.n_kv * self.d_head
    }

    /// Keep only the first `n` committed tokens (a reusable prompt prefix)
    /// and drop every pending row; slab memory is kept for reuse.
    pub fn truncate(&mut self, n: usize) {
        let st = &mut self.slots;
        let n = n.min(st.order.len());
        st.order.truncate(n);
        st.entries.clear();
        st.live.clear();
        st.contiguous = st.order.iter().enumerate().all(|(i, &s)| s as usize == i);
        if st.contiguous {
            st.next_slot = n as u32;
            st.free.clear();
        } else {
            let keep: std::collections::HashSet<u32> = st.order.iter().copied().collect();
            st.free = (0..st.next_slot)
                .rev()
                .filter(|s| !keep.contains(s))
                .collect();
        }
    }

    /// Forget the whole context (a new, unrelated request).
    pub fn reset(&mut self) {
        self.truncate(0);
    }

    /// Committed context length (identical for every layer of the shard).
    pub fn committed_len(&self, _layer: usize) -> usize {
        self.slots.order.len()
    }

    /// Live pending (speculative) rows.
    pub fn pending_len(&self) -> usize {
        self.slots.live_count()
    }

    pub fn bytes(&self) -> usize {
        self.layers
            .values()
            .map(|l| (l.k.len() + l.v.len()) * 4)
            .sum()
    }

    fn write_row(&mut self, layer: usize, slot: u32, k: &[f32], v: &[f32]) {
        let kd = self.kv_dim();
        let sl = self.layers.entry(layer).or_default();
        let need = (slot as usize + 1) * kd;
        if sl.k.len() < need {
            let cap = need.max(sl.k.len() * 3 / 2);
            sl.k.resize(cap, 0.0);
            sl.v.resize(cap, 0.0);
        }
        let o = slot as usize * kd;
        sl.k[o..o + kd].copy_from_slice(k);
        sl.v[o..o + kd].copy_from_slice(v);
    }

    /// Install `n` committed context rows for a layer (context injection).
    pub fn inject_committed(&mut self, layer: usize, k: &[f32], v: &[f32], n: usize) {
        let kd = self.kv_dim();
        while self.slots.order.len() < n {
            let s = self.slots.alloc();
            self.slots.contiguous &= s as usize == self.slots.order.len();
            self.slots.order.push(s);
        }
        for i in 0..n {
            let s = self.slots.order[i];
            self.write_row(layer, s, &k[i * kd..(i + 1) * kd], &v[i * kd..(i + 1) * kd]);
        }
    }

    /// Store new pending rows for a layer (slots are shared across layers).
    pub fn push_pending(&mut self, layer: usize, k: &Tensor2D, v: &Tensor2D, meta: &BatchMeta) {
        for i in 0..k.rows {
            let key = meta.keys[i];
            let slot = match self.slots.slot_of(key) {
                Some(s) => s,
                None => {
                    let s = self.slots.alloc();
                    // reuse a dead entry index, else append one
                    let idx = (0..self.slots.entries.len())
                        .find(|&j| !self.slots.is_live(j))
                        .unwrap_or_else(|| {
                            self.slots.entries.push((0, 0));
                            self.slots.entries.len() - 1
                        });
                    while self.slots.live.len() <= idx / 64 {
                        self.slots.live.push(AtomicU64::new(0));
                    }
                    self.slots.entries[idx] = (key, s);
                    self.slots.live[idx / 64].fetch_or(1u64 << (idx % 64), Ordering::AcqRel);
                    s
                }
            };
            self.write_row(layer, slot, k.row(i), v.row(i));
        }
    }

    /// Attention partial over the pending rows visible to a query, head `h`.
    pub fn pending_partial(
        &self,
        layer: usize,
        q: &[f32],
        h: usize,
        visible: &[NodeKey],
        index: &HashMap<NodeKey, usize>,
    ) -> AttnPartial {
        let g = h / (self.n_heads / self.n_kv);
        let kd = self.kv_dim();
        let off = g * self.d_head;
        let sl = &self.layers[&layer];
        let entries = visible
            .iter()
            .filter_map(|key| index.get(key))
            .map(|&slot| {
                let o = slot * kd + off;
                (&sl.k[o..o + self.d_head], &sl.v[o..o + self.d_head])
            });
        partial_entries(q, entries, self.d_head)
    }

    /// Live pending key → slot map (index for `pending_partial`).
    pub fn pending_index(&self, _layer: usize) -> HashMap<NodeKey, usize> {
        self.slots
            .entries
            .iter()
            .enumerate()
            .filter(|(i, _)| self.slots.is_live(*i))
            .map(|(_, e)| (e.0, e.1 as usize))
            .collect()
    }

    /// Commit accepted pending rows (in the given order) onto the committed
    /// context; every other pending row survives only if `retain` says so.
    /// Node-wide: one atomic mask store + O(accepted) slot bookkeeping, no
    /// per-layer K/V movement. When committed KV is remote
    /// (`hold_committed == false`) the accepted rows are returned per layer and
    /// their slots recycled.
    pub fn commit(
        &mut self,
        accepted: &[NodeKey],
        retain: &dyn Fn(NodeKey) -> bool,
    ) -> CommittedRows {
        let words = self.slots.live.len();
        let mut keep = vec![0u64; words];
        let mut accepted_slots = Vec::with_capacity(accepted.len());
        for key in accepted {
            if let Some(s) = self.slots.slot_of(*key) {
                accepted_slots.push(s);
            }
        }
        for i in 0..self.slots.entries.len() {
            let e = self.slots.entries[i];
            if self.slots.is_live(i) && !accepted.contains(&e.0) {
                if retain(e.0) {
                    keep[i / 64] |= 1 << (i % 64);
                } else {
                    self.slots.free.push(e.1);
                }
            }
        }
        // the lock-free commit point: one atomic store per 64 pending rows
        // (exactly one for a speculative tree)
        for (w, k) in self.slots.live.iter().zip(keep) {
            w.store(k, Ordering::Release);
        }
        if self.slots.live_count() == 0 {
            self.slots.entries.clear();
        }
        let mut out = Vec::new();
        if self.hold_committed {
            for &s in &accepted_slots {
                self.slots.contiguous &= s as usize == self.slots.order.len();
                self.slots.order.push(s);
            }
        } else {
            let kd = self.kv_dim();
            for (&layer, sl) in &self.layers {
                let mut rk = Vec::with_capacity(accepted_slots.len() * kd);
                let mut rv = Vec::with_capacity(accepted_slots.len() * kd);
                for &s in &accepted_slots {
                    let o = s as usize * kd;
                    rk.extend_from_slice(&sl.k[o..o + kd]);
                    rv.extend_from_slice(&sl.v[o..o + kd]);
                }
                out.push((layer, rk, rv));
            }
            out.sort_by_key(|(l, _, _)| *l);
            self.slots
                .order
                .extend(std::iter::repeat_n(u32::MAX, accepted_slots.len()));
            self.slots.contiguous = false;
            self.slots.free.extend_from_slice(&accepted_slots);
        }
        out
    }

    /// Drop all pending rows of one epoch (cancelled braid strand).
    pub fn discard_epoch(&mut self, epoch: u64) {
        let accepted: [NodeKey; 0] = [];
        self.commit(&accepted, &|k| key_epoch(k) != epoch);
    }

    /// Attention partial over the committed context of a layer for head `h`.
    /// Committed-context partials of every (row, head) of `q`, indexed
    /// [row·n_heads + head]. All rows see the whole committed context, so one
    /// grouped sweep per KV head serves every row and every query head of the
    /// group (each key/value row is read once per block, not once per query).
    pub fn committed_partials(&mut self, layer: usize, q: &Tensor2D) -> Vec<AttnPartial> {
        let (nh, dh, kd) = (self.n_heads, self.d_head, self.kv_dim());
        let n = self.slots.order.len();
        let mut out = vec![AttnPartial::empty(dh); q.rows * nh];
        if !self.hold_committed || n == 0 {
            return out;
        }
        let group = nh / self.n_kv;
        let mut scores = std::mem::take(&mut self.scratch);
        let sl = &self.layers[&layer];
        for g in 0..self.n_kv {
            let ids: Vec<(usize, usize)> = (0..q.rows)
                .flat_map(|i| (g * group..(g + 1) * group).map(move |h| (i, h)))
                .collect();
            let qs: Vec<&[f32]> = ids
                .iter()
                .map(|&(i, h)| &q.row(i)[h * dh..(h + 1) * dh])
                .collect();
            let mut st = vec![disaggregated_attention::OnlineQuery::new(dh); qs.len()];
            let off = g * dh;
            if self.slots.contiguous {
                let kv = (0..n).map(|t| {
                    let o = t * kd + off;
                    (&sl.k[o..o + dh], &sl.v[o..o + dh])
                });
                disaggregated_attention::grouped_fold(&qs, kv, &mut st, &mut scores);
            } else {
                let kv = self.slots.order.iter().map(|&s| {
                    let o = s as usize * kd + off;
                    (&sl.k[o..o + dh], &sl.v[o..o + dh])
                });
                disaggregated_attention::grouped_fold(&qs, kv, &mut st, &mut scores);
            }
            for (&(i, h), s) in ids.iter().zip(st) {
                out[i * nh + h] = s.into_partial();
            }
        }
        self.scratch = scores;
        out
    }
}

impl AttentionBackend for LocalKvCache {
    fn attend(
        &mut self,
        layer: usize,
        q: &Tensor2D,
        k: &Tensor2D,
        v: &Tensor2D,
        meta: &BatchMeta,
    ) -> Tensor2D {
        self.push_pending(layer, k, v, meta);
        let index = self.pending_index(layer);
        let (nh, dh) = (self.n_heads, self.d_head);
        if self.exact && self.hold_committed {
            let kd = self.kv_dim();
            let group = nh / self.n_kv;
            let sl = &self.layers[&layer];
            let mut out = Tensor2D::zeros(q.rows, nh * dh);
            for i in 0..q.rows {
                let slots: Vec<usize> = self
                    .slots
                    .order
                    .iter()
                    .map(|&s| s as usize)
                    .chain(
                        meta.visible[i]
                            .iter()
                            .filter_map(|key| index.get(key).copied()),
                    )
                    .collect();
                for h in 0..nh {
                    let off = (h / group) * dh;
                    let qh = &q.row(i)[h * dh..(h + 1) * dh];
                    let entries = slots.iter().map(|&s| {
                        let o = s * kd + off;
                        (&sl.k[o..o + dh], &sl.v[o..o + dh])
                    });
                    let p = disaggregated_attention::partial_entries(qh, entries, dh);
                    out.row_mut(i)[h * dh..(h + 1) * dh].copy_from_slice(&p.o);
                }
            }
            return out;
        }
        let committed = self.committed_partials(layer, q);
        let mut out = Tensor2D::zeros(q.rows, nh * dh);
        for i in 0..q.rows {
            for h in 0..nh {
                let qh = &q.row(i)[h * dh..(h + 1) * dh];
                let pend = self.pending_partial(layer, qh, h, &meta.visible[i], &index);
                let com = &committed[i * nh + h];
                lse_merge(&[com, &pend], &mut out.row_mut(i)[h * dh..(h + 1) * dh]);
            }
        }
        out
    }
}
