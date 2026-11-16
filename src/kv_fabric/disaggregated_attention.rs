//! Zero-migration distributed attention with exact associative LSE merging.
//!
//! Each KV shard computes, for every (query row, head), a partial
//!   m_i = max_t s_t,   l_i = Σ_t exp(s_t − m_i),   O_i = (1/l_i) Σ_t exp(s_t − m_i) v_t
//! over *its own* pages only, and returns just (O_i, m_i, l_i) — d_h + 2
//! floats. The coordinator fuses them with
//!   m = max m_i,  l = Σ l_i e^{m_i − m},  O = Σ (l_i e^{m_i − m} / l) O_i
//! which is exactly softmax attention over the union of all shards.
//! Historical K/V never leave the node that stores them.

use super::page_table::{KvPageTable, PlacementPolicy};
use crate::wire::protocol::{ByteReader, ByteWriter};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq)]
pub struct AttnPartial {
    pub m: f32,
    pub l: f32,
    /// Locally normalised output (Σ p v / l). Zero vector when `l == 0`.
    pub o: Vec<f32>,
}

impl AttnPartial {
    pub fn empty(d_head: usize) -> Self {
        AttnPartial {
            m: f32::NEG_INFINITY,
            l: 0.0,
            o: vec![0.0; d_head],
        }
    }
    pub fn wire_bytes(&self) -> usize {
        (self.o.len() + 2) * 4
    }
}

/// Partial over an arbitrary set of (k, v) head vectors (pending tree nodes).
pub fn partial_entries<'a>(
    q: &[f32],
    entries: impl Iterator<Item = (&'a [f32], &'a [f32])>,
    d_head: usize,
) -> AttnPartial {
    let scale = 1.0 / (d_head as f32).sqrt();
    let items: Vec<(&[f32], &[f32])> = entries.collect();
    if items.is_empty() {
        return AttnPartial::empty(d_head);
    }
    let s: Vec<f32> = items
        .iter()
        .map(|(k, _)| crate::tensor::dot(q, k) * scale)
        .collect();
    let m = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut l = 0.0f32;
    let mut o = vec![0.0f32; d_head];
    for (si, (_, v)) in s.iter().zip(&items) {
        let p = (si - m).exp();
        l += p;
        crate::tensor::axpy(p, v, &mut o);
    }
    let inv = 1.0 / l;
    for x in o.iter_mut() {
        *x *= inv;
    }
    AttnPartial { m, l, o }
}

/// Two-pass partial over a contiguous strided KV buffer: token `t`'s head
/// vector lives at `[t·stride + offset .. + d_head]`. `range` selects tokens.
#[allow(clippy::too_many_arguments)]
pub fn partial_strided(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    stride: usize,
    offset: usize,
    d_head: usize,
    range: std::ops::Range<usize>,
    scratch: &mut Vec<f32>,
) -> AttnPartial {
    if range.is_empty() {
        return AttnPartial::empty(d_head);
    }
    let scale = 1.0 / (d_head as f32).sqrt();
    let n = range.len();
    scratch.clear();
    scratch.reserve(n);
    let mut m = f32::NEG_INFINITY;
    for t in range.clone() {
        let o = t * stride + offset;
        let s = crate::tensor::dot(q, &k[o..o + d_head]) * scale;
        m = m.max(s);
        scratch.push(s);
    }
    let mut l = 0.0f32;
    let mut acc = vec![0.0f32; d_head];
    for (j, t) in range.enumerate() {
        let p = (scratch[j] - m).exp();
        l += p;
        let o = t * stride + offset;
        let vv = &v[o..o + d_head];
        for d in 0..d_head {
            acc[d] += p * vv[d];
        }
    }
    let inv = 1.0 / l;
    for x in acc.iter_mut() {
        *x *= inv;
    }
    AttnPartial { m, l, o: acc }
}

/// Exact associative LSE merge of partials into `out`.
pub fn lse_merge(parts: &[&AttnPartial], out: &mut [f32]) {
    let m = parts
        .iter()
        .filter(|p| p.l > 0.0)
        .map(|p| p.m)
        .fold(f32::NEG_INFINITY, f32::max);
    out.iter_mut().for_each(|x| *x = 0.0);
    if m == f32::NEG_INFINITY {
        return;
    }
    let weights: Vec<f32> = parts
        .iter()
        .map(|p| {
            if p.l > 0.0 {
                p.l * (p.m - m).exp()
            } else {
                0.0
            }
        })
        .collect();
    let l: f32 = weights.iter().sum();
    for (p, w) in parts.iter().zip(&weights) {
        if *w > 0.0 {
            crate::tensor::axpy(w / l, &p.o, out);
        }
    }
}

/// Merge two partials into one (tree-reduction building block).
pub fn lse_combine(a: &AttnPartial, b: &AttnPartial) -> AttnPartial {
    if a.l == 0.0 {
        return b.clone();
    }
    if b.l == 0.0 {
        return a.clone();
    }
    let m = a.m.max(b.m);
    let wa = a.l * (a.m - m).exp();
    let wb = b.l * (b.m - m).exp();
    let l = wa + wb;
    let o =
        a.o.iter()
            .zip(&b.o)
            .map(|(x, y)| (wa * x + wb * y) / l)
            .collect();
    AttnPartial { m, l, o }
}

/// One KV page: `page_tokens` positions of one layer.
/// Online-softmax state of one query vector (running max, mass, unnormalised
/// output).
#[derive(Clone, Debug)]
pub struct OnlineQuery {
    pub m: f32,
    pub l: f32,
    pub acc: Vec<f32>,
}

impl OnlineQuery {
    pub fn new(d_head: usize) -> Self {
        OnlineQuery {
            m: f32::NEG_INFINITY,
            l: 0.0,
            acc: vec![0.0; d_head],
        }
    }

    pub fn into_partial(self) -> AttnPartial {
        if self.l <= 0.0 {
            return AttnPartial::empty(self.acc.len());
        }
        let inv = 1.0 / self.l;
        AttnPartial {
            m: self.m,
            l: self.l,
            o: self.acc.iter().map(|x| x * inv).collect(),
        }
    }
}

/// Tokens per block of the grouped kernel (scores for the whole query group
/// over one block stay in L1).
const GROUP_BLOCK: usize = 32;

/// Multi-query attention sweep: fold a stream of (key, value) rows into the
/// online-softmax state of every query in `qs` — all queries that read the
/// same KV head (the GQA heads of a group × the batch rows that see these
/// tokens). Each key and value row is loaded once per block and used by every
/// query, instead of once per query. Mathematically identical to running
/// `partial_strided` per query (up to f32 summation order).
pub fn grouped_fold<'a>(
    qs: &[&[f32]],
    kv: impl Iterator<Item = (&'a [f32], &'a [f32])>,
    st: &mut [OnlineQuery],
    scores: &mut Vec<f32>,
) {
    use crate::tensor::{gemm_xwt, Tensor2D};
    let g = qs.len();
    if g == 0 {
        return;
    }
    let dh = qs[0].len();
    let scale = 1.0 / (dh as f32).sqrt();
    // the query group as one (g × d_head) matrix, pre-scaled
    let mut qm = Tensor2D::zeros(g, dh);
    for (i, q) in qs.iter().enumerate() {
        for (d, x) in qm.row_mut(i).iter_mut().zip(q.iter()) {
            *d = x * scale;
        }
    }
    let mut kb = Tensor2D::zeros(GROUP_BLOCK, dh);
    let mut vt = Tensor2D::zeros(dh, GROUP_BLOCK);
    let mut sc = Tensor2D::zeros(g, GROUP_BLOCK);
    let mut pv = Tensor2D::zeros(g, dh);
    let _ = scores;
    let mut nb = 0usize;
    let mut flush = |nb: usize, kb: &mut Tensor2D, vt: &mut Tensor2D, sc: &mut Tensor2D| {
        if nb == 0 {
            return;
        }
        // scores = Q · K_blockᵀ and partial outputs = P · V_block, both on the
        // weight-reuse tiled GEMM (the block is the "weight" matrix)
        kb.rows = nb;
        kb.data.truncate(nb * dh);
        if sc.cols != nb {
            *sc = Tensor2D::zeros(g, nb);
        }
        gemm_xwt(&qm, kb, sc);
        for (qi, s) in st.iter_mut().enumerate() {
            let row = sc.row_mut(qi);
            let bm = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            if bm > s.m {
                let c = (s.m - bm).exp();
                s.l *= c;
                s.acc.iter_mut().for_each(|a| *a *= c);
                s.m = bm;
            }
            for x in row.iter_mut() {
                *x = (*x - s.m).exp();
                s.l += *x;
            }
        }
        if vt.cols != nb {
            // keep only the first nb columns of every Vᵀ row
            let mut t = Tensor2D::zeros(dh, nb);
            for d in 0..dh {
                t.row_mut(d).copy_from_slice(&vt.row(d)[..nb]);
            }
            *vt = t;
        }
        gemm_xwt(sc, vt, &mut pv);
        for (qi, s) in st.iter_mut().enumerate() {
            for (a, b) in s.acc.iter_mut().zip(pv.row(qi)) {
                *a += b;
            }
        }
    };
    for (k, v) in kv {
        kb.row_mut(nb).copy_from_slice(k);
        for (d, &x) in v.iter().enumerate() {
            vt.data[d * GROUP_BLOCK + nb] = x;
        }
        nb += 1;
        if nb == GROUP_BLOCK {
            flush(nb, &mut kb, &mut vt, &mut sc);
            nb = 0;
        }
    }
    if nb > 0 {
        flush(nb, &mut kb, &mut vt, &mut sc);
    }
}

#[derive(Clone, Debug)]
pub struct KvPage {
    pub layer: usize,
    pub page_idx: usize,
    pub start_pos: usize,
    pub len: usize,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
}

/// A node's local pool of KV pages (any node, including weight-less ones).
#[derive(Clone, Debug)]
pub struct KvShard {
    pub node_id: u16,
    pub n_heads: usize,
    pub n_kv: usize,
    pub d_head: usize,
    pub page_tokens: usize,
    pub pages: BTreeMap<(usize, usize), KvPage>,
    scratch: Vec<f32>,
}

impl KvShard {
    pub fn new(
        node_id: u16,
        n_heads: usize,
        n_kv: usize,
        d_head: usize,
        page_tokens: usize,
    ) -> Self {
        KvShard {
            node_id,
            n_heads,
            n_kv,
            d_head,
            page_tokens,
            pages: BTreeMap::new(),
            scratch: Vec::new(),
        }
    }

    pub fn kv_dim(&self) -> usize {
        self.n_kv * self.d_head
    }

    pub fn bytes(&self) -> usize {
        self.pages
            .values()
            .map(|p| (p.k.len() + p.v.len()) * 4)
            .sum()
    }

    pub fn tokens(&self, layer: usize) -> usize {
        self.pages
            .values()
            .filter(|p| p.layer == layer)
            .map(|p| p.len)
            .sum()
    }

    /// Drop every token at position ≥ `pos` (all layers): a new request
    /// keeps only the context prefix it shares with the previous one.
    pub fn truncate(&mut self, pos: usize) {
        self.truncate_layers(pos, 0..usize::MAX);
    }

    /// [`truncate`](Self::truncate) restricted to `layers` (one owner's
    /// layers: pipeline stages reset their own KV at different times).
    pub fn truncate_layers(&mut self, pos: usize, layers: std::ops::Range<usize>) {
        let kd = self.kv_dim();
        self.pages
            .retain(|_, p| p.start_pos < pos || !layers.contains(&p.layer));
        for p in self
            .pages
            .values_mut()
            .filter(|p| layers.contains(&p.layer))
        {
            if p.start_pos + p.len > pos {
                p.len = pos - p.start_pos;
                p.k.truncate(p.len * kd);
                p.v.truncate(p.len * kd);
            }
        }
    }

    /// Append one token's K/V for `layer` at absolute position `pos`.
    /// A position already held is overwritten in place (re-sent mirror
    /// pages); stale tails are dropped by explicit [`truncate`](Self::truncate).
    pub fn append(&mut self, layer: usize, pos: usize, k: &[f32], v: &[f32]) {
        let pt = self.page_tokens;
        let kd = self.kv_dim();
        let idx = pos / pt;
        if let Some(p) = self.pages.get_mut(&(layer, idx)) {
            if pos < p.start_pos + p.len {
                let o = (pos - p.start_pos) * kd;
                p.k[o..o + kd].copy_from_slice(k);
                p.v[o..o + kd].copy_from_slice(v);
                return;
            }
        }
        let page = self.pages.entry((layer, idx)).or_insert_with(|| KvPage {
            layer,
            page_idx: idx,
            start_pos: idx * pt,
            len: 0,
            k: Vec::with_capacity(pt * kd),
            v: Vec::with_capacity(pt * kd),
        });
        assert_eq!(
            page.start_pos + page.len,
            pos,
            "KV pages must be appended in position order"
        );
        page.k.extend_from_slice(k);
        page.v.extend_from_slice(v);
        page.len += 1;
    }

    /// Append a contiguous block of tokens [pos, pos+n) for a layer (bulk prefill).
    pub fn append_block(&mut self, layer: usize, pos: usize, k: &[f32], v: &[f32]) {
        let kd = self.kv_dim();
        for i in 0..k.len() / kd {
            self.append(
                layer,
                pos + i,
                &k[i * kd..(i + 1) * kd],
                &v[i * kd..(i + 1) * kd],
            );
        }
    }

    /// Local partials for query rows `q` ([m][n_heads·d_head]) with causal
    /// limit `query_pos[i]` (inclusive; `usize::MAX` for no causal mask).
    /// Returns partials indexed [row·n_heads + head].
    pub fn partials(
        &mut self,
        layer: usize,
        q: &[f32],
        rows: usize,
        query_pos: &[usize],
    ) -> Vec<AttnPartial> {
        self.partials_where(layer, q, rows, query_pos, &|_| true)
    }

    /// Remove and return the pages of `layer` with `page_idx < before`, in
    /// position order (prefix migration).
    pub fn take_prefix(&mut self, layer: usize, before: usize) -> Vec<KvPage> {
        let keys: Vec<(usize, usize)> = self
            .pages
            .range((layer, 0)..(layer, before))
            .map(|(k, _)| *k)
            .collect();
        keys.into_iter()
            .filter_map(|k| self.pages.remove(&k))
            .collect()
    }

    /// Tokens of the pages of `layer` that satisfy `keep(page_idx)`.
    pub fn tokens_where(&self, layer: usize, keep: &dyn Fn(usize) -> bool) -> usize {
        self.pages
            .range((layer, 0)..(layer + 1, 0))
            .filter(|(_, p)| keep(p.page_idx))
            .map(|(_, p)| p.len)
            .sum()
    }

    /// `partials` restricted to the pages with `keep(page_idx)`.
    pub fn partials_where(
        &mut self,
        layer: usize,
        q: &[f32],
        rows: usize,
        query_pos: &[usize],
        keep: &dyn Fn(usize) -> bool,
    ) -> Vec<AttnPartial> {
        let (nh, dh, kd) = (self.n_heads, self.d_head, self.kv_dim());
        let group = nh / self.n_kv;
        let mut out = vec![AttnPartial::empty(dh); rows * nh];
        let mut scores = std::mem::take(&mut self.scratch);
        // rows with the same causal limit see the same tokens: one grouped
        // sweep per (KV head, causal limit) covers all of them
        let mut limits: Vec<usize> = query_pos[..rows].to_vec();
        limits.sort_unstable();
        limits.dedup();
        for &lim in &limits {
            let rs: Vec<usize> = (0..rows).filter(|&i| query_pos[i] == lim).collect();
            for g in 0..self.n_kv {
                let ids: Vec<(usize, usize)> = rs
                    .iter()
                    .flat_map(|&i| (g * group..(g + 1) * group).map(move |h| (i, h)))
                    .collect();
                let qs: Vec<&[f32]> = ids
                    .iter()
                    .map(|&(i, h)| &q[(i * nh + h) * dh..(i * nh + h + 1) * dh])
                    .collect();
                let mut st = vec![OnlineQuery::new(dh); qs.len()];
                let off = g * dh;
                // one sweep across every visible page of the layer
                let kv = self
                    .pages
                    .range((layer, 0)..(layer + 1, 0))
                    .filter(|(_, page)| page.start_pos <= lim && keep(page.page_idx))
                    .flat_map(|(_, page)| {
                        let end = page
                            .len
                            .min(lim.saturating_sub(page.start_pos).saturating_add(1));
                        (0..end).map(move |t| {
                            let o = t * kd + off;
                            (&page.k[o..o + dh], &page.v[o..o + dh])
                        })
                    });
                grouped_fold(&qs, kv, &mut st, &mut scores);
                for (&(i, h), s) in ids.iter().zip(st) {
                    out[i * nh + h] = s.into_partial();
                }
            }
        }
        self.scratch = scores;
        out
    }
}

// --- wire codecs ------------------------------------------------------------

/// Number format of query / partial payloads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WirePrecision {
    /// Exact f32 everywhere.
    F32 = 0,
    /// V3 compact: f16 queries/outputs, f32 for high-norm queries and for
    /// flagged high-gain heads' outputs.
    F16 = 1,
    /// V4 block floating point: BFP14 queries (shared exponent per head, f32
    /// for high-norm heads); outputs in BFP16, BFP12 or skipped according to a
    /// *rigorous* upper bound on the shard's merge weight; f32 for flagged
    /// high-gain heads.
    Bfp = 2,
}

impl WirePrecision {
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => WirePrecision::F16,
            2 => WirePrecision::Bfp,
            _ => WirePrecision::F32,
        }
    }
}

/// Query heads with an L2 norm above this travel in f32 on compact wires.
pub const F16_QUERY_NORM_LIMIT: f32 = 16.0;
/// Merge-weight bound below which a shard skips its output vector.
pub const SKIP_WEIGHT_BOUND: f32 = 1e-7;
/// Merge-weight bound below which a shard's output goes as BFP12.
pub const COARSE_WEIGHT_BOUND: f32 = 0.25;

// --- block floating point -------------------------------------------------------

/// Shared-exponent block floating point: one i8 exponent e (2^e ≥ max|x|) and
/// signed `bits`-bit mantissas q = round(x · (2^(bits−1) − 1) / 2^e).
/// Absolute error ≤ 2^e / (2^bits − 2) ≈ max|x|·2^−bits.
pub fn bfp_put(w: &mut ByteWriter, v: &[f32], bits: u32) {
    let amax = v.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    let e: i32 = if amax > 0.0 {
        amax.log2().ceil() as i32
    } else {
        0
    };
    let e = e.clamp(-126, 127);
    w.u8(e as i8 as u8);
    let qmax = ((1u32 << (bits - 1)) - 1) as f32;
    let s = qmax / 2f32.powi(e);
    let mut acc: u64 = 0;
    let mut n = 0u32;
    let mut out = Vec::with_capacity((v.len() * bits as usize).div_ceil(8));
    for &x in v {
        let q = (x * s).round().clamp(-qmax, qmax) as i32 + (1 << (bits - 1));
        acc |= (q as u64) << n;
        n += bits;
        while n >= 8 {
            out.push(acc as u8);
            acc >>= 8;
            n -= 8;
        }
    }
    if n > 0 {
        out.push(acc as u8);
    }
    w.bytes(&out);
}

pub fn bfp_get(r: &mut ByteReader, len: usize, bits: u32) -> Vec<f32> {
    let e = r.u8() as i8 as i32;
    let qmax = ((1u32 << (bits - 1)) - 1) as f32;
    let inv = 2f32.powi(e) / qmax;
    let bytes = r.bytes((len * bits as usize).div_ceil(8));
    let mut v = Vec::with_capacity(len);
    let (mut acc, mut n, mut bi) = (0u64, 0u32, 0usize);
    for _ in 0..len {
        while n < bits {
            acc |= (bytes[bi] as u64) << n;
            bi += 1;
            n += 8;
        }
        let q = (acc & ((1u64 << bits) - 1)) as i32 - (1 << (bits - 1));
        acc >>= bits;
        n -= bits;
        v.push(q as f32 * inv);
    }
    v
}

fn put_f16(w: &mut ByteWriter, v: &[f32]) {
    let mut b = Vec::with_capacity(v.len() * 2);
    crate::tensor::encode_f16_slice(v, &mut b);
    w.bytes(&b);
}

fn get_f16(r: &mut ByteReader, n: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; n];
    crate::tensor::decode_f16_slice(r.bytes(n * 2), &mut v);
    v
}

/// Per (row, head) query metadata.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct QueryHeadMeta {
    /// Return this head's output vector in f32 (high-gain head).
    pub wide_out: bool,
    /// Rigorous lower bound on the global softmax mass for this query, as a
    /// partial (m, l) the stage already holds (its pending/tree rows).
    pub hint: Option<(f32, f32)>,
}

/// Bits per element of BFP-coded queries.
pub const QUERY_BFP_BITS: u32 = 16;

/// KvQueryBroadcast payload: layer, rows, n_heads, d_head, precision,
/// positions, then per (row, head) the coded query (+ metadata on compact wires).
#[allow(clippy::too_many_arguments)]
pub fn encode_query(
    layer: usize,
    q: &[f32],
    rows: usize,
    n_heads: usize,
    d_head: usize,
    query_pos: &[usize],
    prec: WirePrecision,
    meta: &[QueryHeadMeta],
) -> Vec<u8> {
    let mut w = ByteWriter::with_capacity(17 + rows * 4 + q.len() * 4);
    w.u32(layer as u32);
    w.u32(rows as u32);
    w.u16(n_heads as u16);
    w.u16(d_head as u16);
    w.u8(prec as u8);
    for &p in query_pos {
        w.u32(p.min(u32::MAX as usize) as u32);
    }
    if prec == WirePrecision::F32 {
        w.f32_slice(q);
        return w.into_inner();
    }
    for (j, h) in q.chunks(d_head).enumerate() {
        let m = meta.get(j).copied().unwrap_or_default();
        let wide = crate::tensor::l2_norm(h) > F16_QUERY_NORM_LIMIT;
        let has_hint = prec == WirePrecision::Bfp && m.hint.is_some();
        w.u8(wide as u8 | (m.wide_out as u8) << 1 | (has_hint as u8) << 2);
        if let (true, Some((hm, hl))) = (has_hint, m.hint) {
            w.f32(hm);
            w.f32(hl);
        }
        match (wide, prec) {
            (true, _) => w.f32_slice(h),
            (false, WirePrecision::F16) => put_f16(&mut w, h),
            _ => bfp_put(&mut w, h, QUERY_BFP_BITS),
        }
    }
    w.into_inner()
}

pub struct DecodedQuery {
    pub layer: usize,
    pub rows: usize,
    pub n_heads: usize,
    pub d_head: usize,
    pub precision: WirePrecision,
    pub query_pos: Vec<usize>,
    pub q: Vec<f32>,
    /// Per (row, head): return this partial's output in f32.
    pub wide_out: Vec<bool>,
    /// Per (row, head): rigorous lower bound on the global softmax mass.
    pub hints: Vec<Option<(f32, f32)>>,
}

pub fn decode_query(b: &[u8]) -> DecodedQuery {
    let mut r = ByteReader::new(b);
    let layer = r.u32() as usize;
    let rows = r.u32() as usize;
    let n_heads = r.u16() as usize;
    let d_head = r.u16() as usize;
    let precision = WirePrecision::from_u8(r.u8());
    let query_pos = (0..rows)
        .map(|_| {
            let p = r.u32();
            if p == u32::MAX {
                usize::MAX
            } else {
                p as usize
            }
        })
        .collect();
    let n = rows * n_heads;
    let mut wide_out = vec![false; n];
    let mut hints = vec![None; n];
    let q = if precision == WirePrecision::F32 {
        r.f32_vec(n * d_head)
    } else {
        let mut q = Vec::with_capacity(n * d_head);
        for j in 0..n {
            let f = r.u8();
            wide_out[j] = f & 2 != 0;
            if f & 4 != 0 {
                hints[j] = Some((r.f32(), r.f32()));
            }
            q.extend(if f & 1 != 0 {
                r.f32_vec(d_head)
            } else if precision == WirePrecision::F16 {
                get_f16(&mut r, d_head)
            } else {
                bfp_get(&mut r, d_head, QUERY_BFP_BITS)
            });
        }
        q
    };
    DecodedQuery {
        layer,
        rows,
        n_heads,
        d_head,
        precision,
        query_pos,
        q,
        wide_out,
        hints,
    }
}

/// Upper bound on a shard partial's merge weight given a rigorous lower
/// bound (m_h, l_h) on the rest of the softmax mass: X / (X + P).
pub fn weight_bound(p: &AttnPartial, hint: (f32, f32)) -> f32 {
    if p.l <= 0.0 {
        return 0.0;
    }
    let m = p.m.max(hint.0);
    let x = p.l * (p.m - m).exp();
    let h = hint.1 * (hint.0 - m).exp();
    x / (x + h)
}

/// KvLsePartial payload: count, d_head, precision, then (m, l, code, O) per
/// partial. Codes: 0 = BFP16, 1 = BFP12, 2 = f32, 3 = skipped, 4 = f16.
pub fn encode_partials(
    parts: &[AttnPartial],
    prec: WirePrecision,
    wide_out: &[bool],
    hints: &[Option<(f32, f32)>],
) -> Vec<u8> {
    let dh = parts.first().map(|p| p.o.len()).unwrap_or(0);
    let mut w = ByteWriter::with_capacity(9 + parts.len() * (dh + 3) * 4);
    w.u32(parts.len() as u32);
    w.u32(dh as u32);
    w.u8(prec as u8);
    for (i, p) in parts.iter().enumerate() {
        w.f32(p.m);
        w.f32(p.l);
        let wide = wide_out.get(i).copied().unwrap_or(false);
        let code = match prec {
            WirePrecision::F32 => 2,
            _ if wide => 2,
            WirePrecision::F16 => 4,
            WirePrecision::Bfp => match hints.get(i).copied().flatten() {
                Some(h) => {
                    let wb = weight_bound(p, h);
                    if wb < SKIP_WEIGHT_BOUND {
                        3
                    } else if wb < COARSE_WEIGHT_BOUND {
                        1
                    } else {
                        0
                    }
                }
                None => 0,
            },
        };
        w.u8(code);
        match code {
            0 => bfp_put(&mut w, &p.o, 16),
            1 => bfp_put(&mut w, &p.o, 12),
            2 => w.f32_slice(&p.o),
            4 => put_f16(&mut w, &p.o),
            _ => {}
        }
    }
    w.into_inner()
}

pub fn decode_partials(b: &[u8]) -> Vec<AttnPartial> {
    let mut r = ByteReader::new(b);
    let n = r.u32() as usize;
    let dh = r.u32() as usize;
    let _prec = r.u8();
    (0..n)
        .map(|_| {
            let m = r.f32();
            let l = r.f32();
            let o = match r.u8() {
                0 => bfp_get(&mut r, dh, 16),
                1 => bfp_get(&mut r, dh, 12),
                2 => r.f32_vec(dh),
                4 => get_f16(&mut r, dh),
                _ => vec![0.0; dh],
            };
            AttnPartial { m, l, o }
        })
        .collect()
}

/// In-process disaggregated fabric: a page table plus the shards it maps to.
/// Network bytes are accounted from the *encoded wire payloads*.
pub struct DisaggregatedKVFabric {
    pub table: KvPageTable,
    pub shards: Vec<KvShard>,
    pub coordinator: u16,
    pub precision: WirePrecision,
}

#[derive(Clone, Debug, Default)]
pub struct AttendStats {
    /// Bytes moved (incl. 64-byte frame headers): query broadcast to remote
    /// shards + partials returned.
    pub network_bytes: usize,
    /// Bytes that a KV-migrating design would ship (remote K/V of this layer).
    pub migration_bytes: usize,
}

impl DisaggregatedKVFabric {
    pub fn new(
        table: KvPageTable,
        n_heads: usize,
        n_kv: usize,
        d_head: usize,
        coordinator: u16,
    ) -> Self {
        let shards = table
            .nodes
            .iter()
            .map(|n| KvShard::new(n.node_id, n_heads, n_kv, d_head, table.page_tokens))
            .collect();
        DisaggregatedKVFabric {
            table,
            shards,
            coordinator,
            precision: WirePrecision::F32,
        }
    }

    fn shard_mut(&mut self, node: u16) -> &mut KvShard {
        self.shards
            .iter_mut()
            .find(|s| s.node_id == node)
            .expect("unknown KV node")
    }

    /// Append one token's K/V; the page table decides which node stores it.
    pub fn append(
        &mut self,
        layer: usize,
        pos: usize,
        k: &[f32],
        v: &[f32],
    ) -> Result<u16, String> {
        let kd = k.len();
        let page_bytes = self.table.page_tokens * kd * 2 * 4;
        let node =
            self.table
                .locate_or_allocate(layer, pos / self.table.page_tokens, page_bytes)?;
        self.shard_mut(node).append(layer, pos, k, v);
        Ok(node)
    }

    /// Distributed attention for `rows` query rows of one layer. Returns the
    /// merged output [rows][n_heads·d_head] and traffic statistics.
    pub fn attend_and_merge(
        &mut self,
        layer: usize,
        q: &[f32],
        rows: usize,
        query_pos: &[usize],
    ) -> (Vec<f32>, AttendStats) {
        let (nh, dh) = (self.shards[0].n_heads, self.shards[0].d_head);
        let mut stats = AttendStats::default();
        let prec = self.precision;
        let hdr = crate::wire::protocol::HEADER_LEN;
        let coord = self.coordinator;
        // the coordinator's own shard first: its partial is a rigorous lower
        // bound on each query's softmax mass, sent to the remote shards
        let local: Vec<AttnPartial> = match self.shards.iter_mut().find(|s| s.node_id == coord) {
            Some(sh) => sh.partials(layer, q, rows, query_pos),
            None => vec![AttnPartial::empty(dh); rows * nh],
        };
        let meta: Vec<QueryHeadMeta> = local
            .iter()
            .map(|p| QueryHeadMeta {
                wide_out: false,
                hint: if p.l > 0.0 { Some((p.m, p.l)) } else { None },
            })
            .collect();
        let query_wire = encode_query(layer, q, rows, nh, dh, query_pos, prec, &meta);
        let mut all: Vec<Vec<AttnPartial>> = vec![local];
        for shard in self.shards.iter_mut() {
            // like the live backend, only shards holding pages of this layer
            // are contacted
            if shard.node_id == coord || !shard.pages.values().any(|p| p.layer == layer) {
                continue;
            }
            // exactly what a remote shard would see and send back
            stats.network_bytes += hdr + query_wire.len();
            let dq = decode_query(&query_wire);
            let parts = shard.partials(dq.layer, &dq.q, dq.rows, &dq.query_pos);
            let wire = encode_partials(&parts, dq.precision, &dq.wide_out, &dq.hints);
            stats.network_bytes += hdr + wire.len();
            stats.migration_bytes += shard
                .pages
                .values()
                .filter(|p| p.layer == layer)
                .map(|p| p.len * shard.kv_dim() * 2 * 2)
                .sum::<usize>();
            all.push(decode_partials(&wire));
        }
        let mut out = vec![0.0f32; rows * nh * dh];
        for idx in 0..rows * nh {
            let refs: Vec<&AttnPartial> = all.iter().map(|v| &v[idx]).collect();
            lse_merge(&refs, &mut out[idx * dh..(idx + 1) * dh]);
        }
        (out, stats)
    }

    pub fn per_node_bytes(&self) -> Vec<(u16, usize)> {
        self.shards.iter().map(|s| (s.node_id, s.bytes())).collect()
    }
}

/// Convenience constructor for tests/benchmarks: proportional placement over
/// nodes with the given capacities (bytes).
pub fn fabric_with_capacities(
    caps: &[(u16, usize)],
    page_tokens: usize,
    n_heads: usize,
    n_kv: usize,
    d_head: usize,
) -> DisaggregatedKVFabric {
    let table = KvPageTable::new(page_tokens, caps, PlacementPolicy::ProportionalFreeRam);
    DisaggregatedKVFabric::new(table, n_heads, n_kv, d_head, caps[0].0)
}
