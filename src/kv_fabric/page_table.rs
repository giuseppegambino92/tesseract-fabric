//! Distributed paged KV-MMU.
//!
//! Context is split into fixed pages (`page_tokens` positions of one layer).
//! The table maps `(layer, page_idx)` → node and enforces per-node capacity, so
//! KV can live on *any* node with free RAM — including nodes that hold no
//! weights at all. Pages are never moved once placed (Zero KV-Migration Law).

use std::collections::HashMap;
use std::ops::Range;

#[derive(Clone, Debug)]
pub struct KvNodeState {
    pub node_id: u16,
    pub capacity_bytes: usize,
    pub used_bytes: usize,
    /// Smooth weighted round-robin credit.
    credit: i64,
}

impl KvNodeState {
    pub fn free(&self) -> usize {
        self.capacity_bytes.saturating_sub(self.used_bytes)
    }
    pub fn utilization(&self) -> f64 {
        if self.capacity_bytes == 0 {
            1.0
        } else {
            self.used_bytes as f64 / self.capacity_bytes as f64
        }
    }
}

#[derive(Clone, Debug)]
pub enum PlacementPolicy {
    /// Token-range sharding: page index `p` is assigned to nodes by smooth
    /// weighted round-robin over capacity, so each node stores a share of every
    /// layer's context proportional to its free RAM.
    ProportionalFreeRam,
    /// Keep pages on `node` until it reaches `high_water` utilisation, then
    /// overflow proportionally onto the other nodes.
    LocalFirst { node: u16, high_water: f64 },
    /// Traditional layer-coupled KV: a layer's pages live with its weights.
    LayerCoupled { owners: Vec<(Range<usize>, u16)> },
    /// Tiered: the first `local_pages` pages of every layer stay with the
    /// node that computes the layer (tier 0, zero network); later pages are
    /// striped across all nodes proportional to free RAM (tier 1, parallel).
    LocalityFirst {
        owners: Vec<(Range<usize>, u16)>,
        local_pages: usize,
    },
    /// Paired: tier 0 as in `LocalityFirst`; every later page of a layer goes
    /// to whichever of {computing node, the layer's partner} holds fewer of
    /// that layer's pages, so the two split the layer's attention work evenly
    /// while each decode step contacts a single remote KV service per layer.
    Paired {
        owners: Vec<(Range<usize>, u16)>,
        partners: Vec<(usize, u16)>,
        local_pages: usize,
        /// The partner is kept this many pages behind the computing node.
        remote_bias: usize,
    },
    /// Head-group striping over *virtual layers* (layer × KV-head group):
    /// tier 0 stays with the computing node; every later page of a virtual
    /// layer goes to the holder with the fewest pages of it plus that
    /// holder's offset (remote holders carry the calibrated bias), so the
    /// KV heads of one layer are scanned by several nodes at once while each
    /// node only receives the query heads of its own group.
    Striped {
        owners: Vec<(Range<usize>, u16)>,
        holders: Vec<(usize, Vec<(u16, usize)>)>,
        local_pages: usize,
        /// Mirror the tier-0 prefix pages onto the virtual layer's remote
        /// holders: short contexts scan them locally (zero traffic), long
        /// contexts delegate them to the mirrors (see `delegated`).
        mirror: bool,
    },
}

/// Holders per virtual layer `vl = layer · n_kv + g` for `Striped`: group 0
/// stays with the computing node plus remote help, the other groups live on
/// remote nodes only. Remote nodes (all other nodes, by capacity) are dealt
/// round-robin to groups 1, 2, …, n_kv − 1, 0, 1, … so every node scans a
/// similar share. Remote holders carry `bias` pages of offset.
pub fn group_holders(
    owners: &[(Range<usize>, u16)],
    capacities: &[(u16, usize)],
    n_layers: usize,
    n_kv: usize,
    bias: usize,
) -> Vec<(usize, Vec<(u16, usize)>)> {
    let owner_of = |l: usize| owners.iter().find(|(r, _)| r.contains(&l)).map(|o| o.1);
    let mut out = Vec::new();
    for l in 0..n_layers {
        let o = owner_of(l);
        let mut remotes: Vec<(u16, usize)> = capacities
            .iter()
            .filter(|c| Some(c.0) != o)
            .map(|c| (c.0, c.1))
            .collect();
        remotes.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let mut sets: Vec<Vec<(u16, usize)>> = vec![Vec::new(); n_kv];
        if let Some(o) = o {
            sets[0].push((o, 0));
        }
        let order: Vec<usize> = (1..n_kv).chain(std::iter::once(0)).collect();
        for (i, (n, _)) in remotes.iter().enumerate() {
            sets[order[i % order.len()]].push((*n, bias));
        }
        for (g, set) in sets.into_iter().enumerate() {
            out.push((l * n_kv + g, set));
        }
    }
    out
}

/// Partner per layer for `Paired`: greedily the other node whose KV
/// utilisation (own layers' halves + partnered halves, relative to capacity)
/// stays lowest. Deterministic for a given assignment.
pub fn pair_partners(
    owners: &[(Range<usize>, u16)],
    capacities: &[(u16, usize)],
    n_layers: usize,
) -> Vec<(usize, u16)> {
    let cap = |n: u16| {
        capacities
            .iter()
            .find(|c| c.0 == n)
            .map(|c| c.1.max(1) as f64)
            .unwrap_or(1.0)
    };
    let owner_of = |l: usize| owners.iter().find(|(r, _)| r.contains(&l)).map(|o| o.1);
    let mut load: HashMap<u16, f64> = capacities.iter().map(|c| (c.0, 0.0)).collect();
    for l in 0..n_layers {
        if let Some(o) = owner_of(l) {
            *load.entry(o).or_default() += 0.5;
        }
    }
    let mut out = Vec::new();
    for l in 0..n_layers {
        let o = owner_of(l);
        let best = capacities
            .iter()
            .map(|c| c.0)
            .filter(|&n| Some(n) != o)
            .min_by(|&a, &b| {
                let ua = (load[&a] + 0.5) / cap(a);
                let ub = (load[&b] + 0.5) / cap(b);
                ua.total_cmp(&ub).then(a.cmp(&b))
            });
        if let Some(p) = best {
            *load.get_mut(&p).unwrap() += 0.5;
            out.push((l, p));
        }
    }
    out
}

#[derive(Clone, Debug)]
pub struct KvPageTable {
    pub page_tokens: usize,
    pub nodes: Vec<KvNodeState>,
    pub policy: PlacementPolicy,
    map: HashMap<(usize, usize), u16>,
    page_owner: HashMap<usize, u16>,
    layer_pages: HashMap<(usize, u16), usize>,
    /// Mirror node of mirrored prefix pages, and the per-node page counts
    /// of the delegated (long-context) regime used for balancing.
    mirrors: HashMap<(usize, usize), u16>,
    scan_pages: HashMap<(usize, u16), usize>,
    /// Pages per virtual layer that are mirrored (0 = no mirroring).
    pub mirror_pages: usize,
}

impl KvPageTable {
    pub fn new(page_tokens: usize, capacities: &[(u16, usize)], policy: PlacementPolicy) -> Self {
        let mirror_pages = match &policy {
            PlacementPolicy::Striped {
                local_pages,
                mirror: true,
                ..
            } => *local_pages,
            _ => 0,
        };
        KvPageTable {
            page_tokens,
            nodes: capacities
                .iter()
                .map(|&(node_id, capacity_bytes)| KvNodeState {
                    node_id,
                    capacity_bytes,
                    used_bytes: 0,
                    credit: 0,
                })
                .collect(),
            policy,
            map: HashMap::new(),
            page_owner: HashMap::new(),
            layer_pages: HashMap::new(),
            mirrors: HashMap::new(),
            scan_pages: HashMap::new(),
            mirror_pages,
        }
    }

    /// Mirror node of a mirrored prefix page.
    pub fn mirror_of(&self, layer: usize, page_idx: usize) -> Option<u16> {
        self.mirrors.get(&(layer, page_idx)).copied()
    }

    /// Long-context regime: once the committed context reaches twice the
    /// mirrored prefix, the prefix pages are scanned by their mirrors and
    /// the computing node skips them. Every party evaluates this on the
    /// committed length carried by each query, so each page is scanned
    /// exactly once.
    pub fn delegated(&self, committed: usize) -> bool {
        self.mirror_pages > 0 && committed >= 2 * self.mirror_pages * self.page_tokens
    }

    /// Nodes that scan pages of `layer` at this committed length.
    pub fn scanners_for_layer(&self, layer: usize, committed: usize) -> Vec<u16> {
        let mut v = self.nodes_for_layer(layer);
        if self.delegated(committed) {
            for (&(l, _), &m) in &self.mirrors {
                if l == layer && !v.contains(&m) {
                    v.push(m);
                }
            }
            v.sort_unstable();
        }
        v
    }

    pub fn lookup(&self, layer: usize, page_idx: usize) -> Option<u16> {
        self.map.get(&(layer, page_idx)).copied()
    }

    fn node_mut(&mut self, id: u16) -> &mut KvNodeState {
        self.nodes
            .iter_mut()
            .find(|n| n.node_id == id)
            .expect("unknown node")
    }

    /// Find the node holding `(layer, page_idx)` or allocate it.
    pub fn locate_or_allocate(
        &mut self,
        layer: usize,
        page_idx: usize,
        page_bytes: usize,
    ) -> Result<u16, String> {
        if let Some(n) = self.lookup(layer, page_idx) {
            return Ok(n);
        }
        let chosen = match self.policy.clone() {
            PlacementPolicy::LayerCoupled { owners } => {
                let owner = owners
                    .iter()
                    .find(|(r, _)| r.contains(&layer))
                    .map(|(_, n)| *n)
                    .ok_or_else(|| format!("no owner for layer {layer}"))?;
                if self.node_mut(owner).free() < page_bytes {
                    return Err(format!(
                        "OOM: node {owner} cannot hold KV page (layer {layer}, page {page_idx})"
                    ));
                }
                owner
            }
            PlacementPolicy::ProportionalFreeRam => {
                // All layers of a page index go to the same node when possible.
                match self.page_owner.get(&page_idx).copied() {
                    Some(n) if self.node_mut(n).free() >= page_bytes => n,
                    _ => {
                        let n = self.pick_weighted(page_bytes).ok_or_else(|| {
                            format!("OOM: cluster KV fabric full at page {page_idx}")
                        })?;
                        self.page_owner.insert(page_idx, n);
                        n
                    }
                }
            }
            PlacementPolicy::LocalityFirst {
                owners,
                local_pages,
            } => {
                let owner = owners
                    .iter()
                    .find(|(r, _)| r.contains(&layer))
                    .map(|(_, n)| *n);
                match owner {
                    Some(o) if page_idx < local_pages && self.node_mut(o).free() >= page_bytes => o,
                    _ => match self.page_owner.get(&page_idx).copied() {
                        Some(n) if self.node_mut(n).free() >= page_bytes => n,
                        _ => {
                            let n = self.pick_weighted(page_bytes).ok_or_else(|| {
                                format!("OOM: cluster KV fabric full at page {page_idx}")
                            })?;
                            self.page_owner.insert(page_idx, n);
                            n
                        }
                    },
                }
            }
            PlacementPolicy::Paired {
                owners,
                partners,
                local_pages,
                remote_bias,
            } => {
                let owner = owners
                    .iter()
                    .find(|(r, _)| r.contains(&layer))
                    .map(|(_, n)| *n);
                let partner = partners.iter().find(|p| p.0 == layer).map(|p| p.1);
                let has_room = |t: &mut Self, n: u16| t.node_mut(n).free() >= page_bytes;
                match (owner, partner) {
                    (Some(o), _) if page_idx < local_pages && has_room(self, o) => o,
                    (Some(o), Some(q)) => {
                        let co = self.layer_pages.get(&(layer, o)).copied().unwrap_or(0);
                        let cq = self.layer_pages.get(&(layer, q)).copied().unwrap_or(0);
                        let (first, second) = if cq + remote_bias <= co {
                            (q, o)
                        } else {
                            (o, q)
                        };
                        if has_room(self, first) {
                            first
                        } else if has_room(self, second) {
                            second
                        } else {
                            self.pick_weighted(page_bytes).ok_or_else(|| {
                                format!("OOM: cluster KV fabric full at page {page_idx}")
                            })?
                        }
                    }
                    _ => self
                        .pick_weighted(page_bytes)
                        .ok_or_else(|| format!("OOM: cluster KV fabric full at page {page_idx}"))?,
                }
            }
            PlacementPolicy::Striped {
                owners,
                holders,
                local_pages,
                ..
            } => {
                let owner = owners
                    .iter()
                    .find(|(r, _)| r.contains(&layer))
                    .map(|(_, n)| *n);
                let set = holders
                    .iter()
                    .find(|h| h.0 == layer)
                    .map(|h| h.1.clone())
                    .unwrap_or_default();
                let mirror_on = self.mirror_pages > 0;
                match owner {
                    Some(o) if page_idx < local_pages && self.node_mut(o).free() >= page_bytes => {
                        if mirror_on {
                            // mirror the prefix page on a remote holder of
                            // this virtual layer (round-robin by page)
                            let remotes: Vec<u16> =
                                set.iter().map(|h| h.0).filter(|&n| n != o).collect();
                            if !remotes.is_empty() {
                                let m = remotes[page_idx % remotes.len()];
                                if self.node_mut(m).free() >= page_bytes {
                                    self.node_mut(m).used_bytes += page_bytes;
                                    self.mirrors.insert((layer, page_idx), m);
                                    *self.scan_pages.entry((layer, m)).or_default() += 1;
                                }
                            }
                        }
                        o
                    }
                    _ => {
                        let mut best: Option<(usize, u16)> = None;
                        for &(n, off) in &set {
                            if self.node_mut(n).free() < page_bytes {
                                continue;
                            }
                            // balance for the long-context regime: mirrored
                            // prefix pages count for their mirror
                            let counts = if mirror_on {
                                &self.scan_pages
                            } else {
                                &self.layer_pages
                            };
                            let c = counts.get(&(layer, n)).copied().unwrap_or(0) + off;
                            if best.map(|b| c < b.0).unwrap_or(true) {
                                best = Some((c, n));
                            }
                        }
                        match best {
                            Some((_, n)) => n,
                            None => self.pick_weighted(page_bytes).ok_or_else(|| {
                                format!("OOM: cluster KV fabric full at page {page_idx}")
                            })?,
                        }
                    }
                }
            }
            PlacementPolicy::LocalFirst { node, high_water } => {
                let st = self.node_mut(node);
                let after = (st.used_bytes + page_bytes) as f64 / st.capacity_bytes.max(1) as f64;
                if after <= high_water {
                    node
                } else {
                    match self.page_owner.get(&page_idx).copied() {
                        Some(n) if n != node && self.node_mut(n).free() >= page_bytes => n,
                        _ => {
                            let n = self
                                .swrr_pick_excluding(Some(node), page_bytes)
                                .ok_or_else(|| {
                                    format!("OOM: KV overflow pool full at page {page_idx}")
                                })?;
                            self.page_owner.insert(page_idx, n);
                            n
                        }
                    }
                }
            }
        };
        self.node_mut(chosen).used_bytes += page_bytes;
        self.map.insert((layer, page_idx), chosen);
        *self.layer_pages.entry((layer, chosen)).or_default() += 1;
        if !self.mirrors.contains_key(&(layer, page_idx)) {
            *self.scan_pages.entry((layer, chosen)).or_default() += 1;
        }
        Ok(chosen)
    }

    /// Smooth weighted round robin (nginx) over nodes with room for `need`.
    fn pick_weighted(&mut self, need: usize) -> Option<u16> {
        self.swrr_pick_excluding(None, need)
    }

    fn swrr_pick_excluding(&mut self, exclude: Option<u16>, need: usize) -> Option<u16> {
        let eligible: Vec<usize> = (0..self.nodes.len())
            .filter(|&i| Some(self.nodes[i].node_id) != exclude && self.nodes[i].free() >= need)
            .collect();
        if eligible.is_empty() {
            return None;
        }
        let total: i64 = eligible
            .iter()
            .map(|&i| (self.nodes[i].capacity_bytes / 4096) as i64)
            .sum();
        let mut best = eligible[0];
        for &i in &eligible {
            self.nodes[i].credit += (self.nodes[i].capacity_bytes / 4096) as i64;
            if self.nodes[i].credit > self.nodes[best].credit {
                best = i;
            }
        }
        self.nodes[best].credit -= total;
        Some(self.nodes[best].node_id)
    }

    /// Pages per node.
    pub fn pages_per_node(&self) -> Vec<(u16, usize)> {
        self.nodes
            .iter()
            .map(|n| {
                (
                    n.node_id,
                    self.map.values().filter(|&&m| m == n.node_id).count(),
                )
            })
            .collect()
    }

    /// Token ranges of `layer` held by each node.
    pub fn ranges_for_layer(&self, layer: usize) -> Vec<(u16, Vec<usize>)> {
        let mut by: HashMap<u16, Vec<usize>> = HashMap::new();
        for (&(l, p), &n) in &self.map {
            if l == layer {
                by.entry(n).or_default().push(p);
            }
        }
        let mut v: Vec<(u16, Vec<usize>)> = by
            .into_iter()
            .map(|(n, mut ps)| {
                ps.sort();
                (n, ps)
            })
            .collect();
        v.sort_by_key(|x| x.0);
        v
    }

    /// Nodes that hold at least one page of `layer`.
    pub fn nodes_for_layer(&self, layer: usize) -> Vec<u16> {
        self.ranges_for_layer(layer)
            .into_iter()
            .map(|(n, _)| n)
            .collect()
    }

    /// max utilisation / mean utilisation (1.0 = perfectly balanced).
    pub fn imbalance(&self) -> f64 {
        let u: Vec<f64> = self.nodes.iter().map(|n| n.utilization()).collect();
        let mean = u.iter().sum::<f64>() / u.len() as f64;
        let max = u.iter().cloned().fold(0.0, f64::max);
        if mean == 0.0 {
            1.0
        } else {
            max / mean
        }
    }
}

/// Dry-run projection of the disaggregated attention bytes of one decode
/// step (one query row) for a head-group striped placement: the page table
/// is laid out exactly as the engine does for `ctx` injected tokens, and every
/// frame is sized with the real wire encoders. Per virtual layer, each remote
/// scanner receives one query frame (the group's heads, BFP, with hints and
/// the committed length) and returns one partial frame (sized at BFP16, the
/// widest code), and the committed row is appended to the page that receives
/// position `ctx` (batched into one frame per destination node).
#[allow(clippy::too_many_arguments)]
pub fn project_step_bytes(
    owners: &[(Range<usize>, u16)],
    capacities: &[(u16, usize)],
    n_layers: usize,
    n_heads: usize,
    n_kv: usize,
    d_head: usize,
    page_tokens: usize,
    local_pages: usize,
    bias: usize,
    ctx: usize,
) -> usize {
    use super::disaggregated_attention::{
        bfp_put, encode_partials, encode_query, AttnPartial, QueryHeadMeta, WirePrecision,
    };
    use crate::wire::protocol::{ByteWriter, HEADER_LEN};
    let g_n = n_kv;
    let gh = n_heads / g_n;
    let vl_owners: Vec<(Range<usize>, u16)> = owners
        .iter()
        .map(|(r, n)| (r.start * g_n..r.end * g_n, *n))
        .collect();
    let mut t = KvPageTable::new(
        page_tokens,
        capacities,
        PlacementPolicy::Striped {
            holders: group_holders(owners, capacities, n_layers, g_n, bias),
            owners: vl_owners.clone(),
            local_pages,
            mirror: true,
        },
    );
    let page_bytes = page_tokens * d_head * 2 * 4;
    for p in 0..ctx.div_ceil(page_tokens) {
        for vl in 0..n_layers * g_n {
            let _ = t.locate_or_allocate(vl, p, page_bytes);
        }
    }
    let q = vec![0.5f32; gh * d_head];
    let meta = vec![
        QueryHeadMeta {
            wide_out: false,
            hint: Some((1.0, 1.0)),
        };
        gh
    ];
    let query = HEADER_LEN
        + encode_query(
            0,
            &q,
            1,
            gh,
            d_head,
            &[usize::MAX],
            WirePrecision::Bfp,
            &meta,
        )
        .len()
        + 8;
    let parts = vec![
        AttnPartial {
            m: 0.0,
            l: 1.0,
            o: vec![0.5; d_head],
        };
        gh
    ];
    let partial = HEADER_LEN
        + encode_partials(
            &parts,
            WirePrecision::Bfp,
            &vec![false; gh],
            &vec![None; gh],
        )
        .len();
    let record = {
        let mut w = ByteWriter::default();
        w.u32(0);
        w.u32(0);
        w.u32(1);
        w.u8(1);
        bfp_put(&mut w, &vec![0.5; d_head], 16);
        bfp_put(&mut w, &vec![0.5; d_head], 16);
        w.into_inner().len()
    };
    let mut bytes = 0usize;
    // each stage batches its own commit's records per destination node
    let mut appends: HashMap<(u16, u16), usize> = HashMap::new();
    for (r, owner) in owners {
        for l in r.clone() {
            for g in 0..g_n {
                let vl = l * g_n + g;
                let remote = t
                    .scanners_for_layer(vl, ctx)
                    .into_iter()
                    .filter(|&n| n != *owner)
                    .count();
                bytes += remote * (query + partial);
                if let Ok(dest) = t.locate_or_allocate(vl, ctx / page_tokens, page_bytes) {
                    if dest != *owner {
                        *appends.entry((*owner, dest)).or_default() += record;
                    }
                }
            }
        }
    }
    bytes + appends.values().map(|r| HEADER_LEN + 4 + r).sum::<usize>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proportional_placement_tracks_capacity() {
        let mut t = KvPageTable::new(
            256,
            &[
                (0, 400 << 20),
                (1, 200 << 20),
                (2, 100 << 20),
                (3, 100 << 20),
            ],
            PlacementPolicy::ProportionalFreeRam,
        );
        for p in 0..80 {
            t.locate_or_allocate(0, p, 1 << 20).unwrap();
        }
        let per: Vec<usize> = t.pages_per_node().into_iter().map(|x| x.1).collect();
        assert_eq!(per, vec![40, 20, 10, 10]);
    }

    #[test]
    fn layer_coupled_ooms() {
        let mut t = KvPageTable::new(
            256,
            &[(0, 3 << 20), (1, 100 << 20)],
            PlacementPolicy::LayerCoupled {
                owners: vec![(0..1, 0), (1..2, 1)],
            },
        );
        for p in 0..3 {
            t.locate_or_allocate(0, p, 1 << 20).unwrap();
        }
        assert!(t.locate_or_allocate(0, 3, 1 << 20).is_err());
    }
}
