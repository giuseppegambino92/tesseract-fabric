//! Heterogeneous cluster role & shard optimiser.
//!
//! Single-stream decode is a latency problem: every speculative epoch flows
//! through the weight pipeline once. `HeteroClusterSolver` enumerates ordered
//! subsets of capable nodes as pipeline stages (the last stage is the
//! coordinator, which owns the LM head and verifies trees), splits layers to
//! minimise the epoch critical path under RAM constraints, prices every hop
//! with the link ETTC of the compressed tree activation, and keeps slow/edge
//! links off the critical path. Leftover edge nodes become asynchronous tree
//! drafters and overflow KV page pools.

use crate::wire::multipath::LinkProfile;
use std::ops::Range;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Platform {
    AppleSiliconUMA,
    DiscreteGpuWorkstation,
    MobileEdgeNpu,
    CpuServer,
}

impl Platform {
    pub fn name(&self) -> &'static str {
        match self {
            Platform::AppleSiliconUMA => "AppleSiliconUMA",
            Platform::DiscreteGpuWorkstation => "DiscreteGpuWorkstation",
            Platform::MobileEdgeNpu => "MobileEdgeNpu",
            Platform::CpuServer => "CpuServer",
        }
    }
}

#[derive(Clone, Debug)]
pub struct LinkSpec {
    pub peer: u16,
    pub profiles: Vec<LinkProfile>,
}

#[derive(Clone, Debug)]
pub struct HardwareNodeSpec {
    pub node_id: u16,
    pub name: String,
    pub platform: Platform,
    pub compute_tflops_fp16: f64,
    pub memory_bandwidth_gbps: f64,
    pub available_ram_mb: f64,
    pub links: Vec<LinkSpec>,
}

impl HardwareNodeSpec {
    pub fn links_to(&self, peer: u16) -> Option<&Vec<LinkProfile>> {
        self.links
            .iter()
            .find(|l| l.peer == peer)
            .map(|l| &l.profiles)
    }

    /// Best single-link one-way ETTC to `peer` for `bytes` (ns).
    pub fn ettc_to(&self, peer: u16, bytes: usize) -> f64 {
        if peer == self.node_id {
            return 0.0;
        }
        self.links_to(peer)
            .map(|ps| {
                ps.iter()
                    .map(|p| p.ettc_ns(bytes))
                    .fold(f64::INFINITY, f64::min)
            })
            .unwrap_or(f64::INFINITY)
    }

    pub fn min_one_way_us(&self, peer: u16) -> f64 {
        self.links_to(peer)
            .map(|ps| {
                ps.iter()
                    .map(|p| p.one_way_us)
                    .fold(f64::INFINITY, f64::min)
            })
            .unwrap_or(f64::INFINITY)
    }
}

/// What the solver needs to know about the model being served.
#[derive(Clone, Debug)]
pub struct ModelFootprint {
    pub n_layers: usize,
    /// Resident bytes per layer (all experts, cold+hot accounting).
    pub layer_bytes: f64,
    /// Bytes streamed from memory per layer per epoch (active weights).
    pub layer_active_bytes: f64,
    /// FLOPs per row per layer.
    pub layer_flops_per_row: f64,
    /// Embedding + LM head bytes (coordinator / first stage).
    pub head_bytes: f64,
    /// Wire bytes per tree row after compression.
    pub act_bytes_per_row: f64,
    pub kv_bytes_per_token_layer: f64,
    /// Whether the FFN is mixture-of-experts (H-TEP needs experts to split).
    pub moe: bool,
    /// Attention weight bytes per layer (replicated under H-TEP).
    pub attn_bytes: f64,
    /// All-expert FFN bytes per layer (split across H-TEP ranks).
    pub expert_bytes: f64,
}

#[derive(Clone, Debug)]
pub struct SolverConfig {
    /// Rows per speculative epoch (tree size).
    pub tree_rows: usize,
    /// Expected tokens emitted per epoch (refined from telemetry).
    pub expected_accept: f64,
    /// Never put a link slower than this (one-way µs) on the critical path
    /// unless no feasible plan avoids it.
    pub max_critical_hop_us: f64,
    /// Fraction of each node's RAM reserved for its local KV pool.
    pub kv_reserve_frac: f64,
    pub max_stages: usize,
    /// Target layers the early-exit drafter runs per drafted row (prices
    /// the drafter's compute when choosing where it runs).
    pub draft_layers: usize,
}

impl Default for SolverConfig {
    fn default() -> Self {
        SolverConfig {
            tree_rows: 16,
            expected_accept: 2.0,
            max_critical_hop_us: 1000.0,
            kv_reserve_frac: 0.15,
            max_stages: 4,
            draft_layers: 1,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct StageAssignment {
    pub node: u16,
    pub layers: Range<usize>,
}

#[derive(Clone, Debug)]
pub struct ClusterPlan {
    pub stages: Vec<StageAssignment>,
    pub coordinator: u16,
    pub drafter: Option<u16>,
    /// (node, KV capacity bytes) for every node with surplus RAM.
    pub kv_pools: Vec<(u16, usize)>,
    pub est_epoch_ms: f64,
    pub est_tok_s: f64,
    pub notes: Vec<String>,
    /// Hybrid tensor-expert parallel group (ranks), chosen when capable nodes
    /// are joined by links with RTT < 100 µs and the group holds the model
    /// with replicated attention + experts split across ranks.
    pub htep: Option<Vec<u16>>,
}

impl ClusterPlan {
    pub fn stage_of_layer(&self, layer: usize) -> Option<u16> {
        self.stages
            .iter()
            .find(|s| s.layers.contains(&layer))
            .map(|s| s.node)
    }
}

pub struct HeteroClusterSolver {
    pub nodes: Vec<HardwareNodeSpec>,
    pub model: ModelFootprint,
    pub cfg: SolverConfig,
}

impl HeteroClusterSolver {
    pub fn new(nodes: Vec<HardwareNodeSpec>, model: ModelFootprint, cfg: SolverConfig) -> Self {
        HeteroClusterSolver { nodes, model, cfg }
    }

    fn node(&self, id: u16) -> &HardwareNodeSpec {
        self.nodes.iter().find(|n| n.node_id == id).unwrap()
    }

    /// Seconds for one layer on `node` for `rows` tree rows (roofline).
    pub fn layer_time_s(&self, node: &HardwareNodeSpec, rows: usize) -> f64 {
        let mem = self.model.layer_active_bytes / (node.memory_bandwidth_gbps * 1e9);
        let flops =
            self.model.layer_flops_per_row * rows as f64 / (node.compute_tflops_fp16 * 1e12);
        mem.max(flops)
    }

    fn weight_capacity_layers(&self, node: &HardwareNodeSpec, is_coord: bool) -> usize {
        let usable = node.available_ram_mb * 1e6 * (1.0 - self.cfg.kv_reserve_frac)
            - if is_coord { self.model.head_bytes } else { 0.0 };
        if usable <= 0.0 {
            0
        } else {
            (usable / self.model.layer_bytes).floor() as usize
        }
    }

    /// Optimal layer split for an ordered stage list: greedily fill the
    /// fastest stages first (minimises Σ nᵢ·cᵢ under capacity), keeping
    /// pipeline order. Returns None if the model does not fit.
    fn split_layers(&self, order: &[u16]) -> Option<Vec<usize>> {
        let rows = self.cfg.tree_rows;
        let caps: Vec<usize> = order
            .iter()
            .enumerate()
            .map(|(i, &n)| self.weight_capacity_layers(self.node(n), i + 1 == order.len()))
            .collect();
        if caps.iter().sum::<usize>() < self.model.n_layers || caps.contains(&0) {
            return None;
        }
        let mut by_speed: Vec<usize> = (0..order.len()).collect();
        by_speed.sort_by(|&a, &b| {
            self.layer_time_s(self.node(order[a]), rows)
                .partial_cmp(&self.layer_time_s(self.node(order[b]), rows))
                .unwrap()
        });
        let mut counts = vec![0usize; order.len()];
        let mut left = self.model.n_layers;
        // every stage gets ≥1 layer (otherwise it should not be a stage)
        for c in counts.iter_mut() {
            *c = 1;
            left -= 1;
        }
        for &i in &by_speed {
            let add = left.min(caps[i] - counts[i]);
            counts[i] += add;
            left -= add;
        }
        if left > 0 {
            None
        } else {
            Some(counts)
        }
    }

    /// Critical-path seconds of one epoch for an ordered stage list.
    fn epoch_time(&self, order: &[u16], counts: &[usize]) -> (f64, f64) {
        let rows = self.cfg.tree_rows;
        let mut t = 0.0;
        let mut worst_hop_us: f64 = 0.0;
        let coord = *order.last().unwrap();
        // coordinator → first stage: token ids + tree meta (tiny)
        if order[0] != coord {
            t += self.node(coord).ettc_to(order[0], 64 + rows * 64) * 1e-9;
            worst_hop_us = worst_hop_us.max(self.node(coord).min_one_way_us(order[0]));
        }
        for (i, &n) in order.iter().enumerate() {
            t += counts[i] as f64 * self.layer_time_s(self.node(n), rows);
            if i + 1 < order.len() {
                let bytes = (64.0 + self.model.act_bytes_per_row * rows as f64) as usize;
                t += self.node(n).ettc_to(order[i + 1], bytes) * 1e-9;
                worst_hop_us = worst_hop_us.max(self.node(n).min_one_way_us(order[i + 1]));
            }
        }
        // commit broadcast back (32 B) to the first stage
        if order.len() > 1 {
            t += self.node(coord).ettc_to(order[0], 96) * 1e-9;
        }
        (t, worst_hop_us)
    }

    pub fn solve(&self) -> ClusterPlan {
        let capable: Vec<u16> = self
            .nodes
            .iter()
            .filter(|n| n.platform != Platform::MobileEdgeNpu)
            .map(|n| n.node_id)
            .collect();
        let mut best: Option<(f64, Vec<u16>, Vec<usize>, bool)> = None;
        let mut notes = Vec::new();
        let n = capable.len();
        for mask in 1u32..(1 << n) {
            let subset: Vec<u16> = (0..n)
                .filter(|i| mask >> i & 1 == 1)
                .map(|i| capable[i])
                .collect();
            if subset.len() > self.cfg.max_stages {
                continue;
            }
            for order in permutations(&subset) {
                let Some(counts) = self.split_layers(&order) else {
                    continue;
                };
                let (t, worst) = self.epoch_time(&order, &counts);
                let ok_links = worst <= self.cfg.max_critical_hop_us;
                let better = match &best {
                    None => true,
                    Some((bt, _, _, bok)) => (ok_links && !bok) || (ok_links == *bok && t < *bt),
                };
                if better {
                    best = Some((t, order.clone(), counts, ok_links));
                }
            }
        }
        let (t, order, counts, ok) =
            best.expect("no feasible pipeline: model does not fit cluster RAM");
        if !ok {
            notes.push("no plan avoids slow links on the critical path".into());
        }
        let mut stages = Vec::new();
        let mut l = 0;
        for (i, &node) in order.iter().enumerate() {
            stages.push(StageAssignment {
                node,
                layers: l..l + counts[i],
            });
            l += counts[i];
        }
        let coordinator = *order.last().unwrap();
        // Drafter: the node that returns a drafted tree soonest — its compute
        // for the drafter's layers at one row plus the round trip to the
        // coordinator. Any node but the coordinator (which verifies) may
        // draft: a pipeline stage drafts on the cores its stage leaves idle
        // while later stages and verification run. Ties go to off-pipeline
        // nodes. (Before V10.1 an edge NPU was always preferred; a 23× phone
        // on Wi-Fi then delivered no tree in time.)
        let draft_cost = |x: &HardwareNodeSpec| {
            self.layer_time_s(x, 1) * self.cfg.draft_layers.max(1) as f64
                + 2e-6 * x.min_one_way_us(coordinator)
        };
        let mut cands: Vec<&HardwareNodeSpec> = self
            .nodes
            .iter()
            .filter(|x| x.node_id != coordinator)
            .collect();
        cands.sort_by(|a, b| {
            draft_cost(a)
                .partial_cmp(&draft_cost(b))
                .unwrap()
                .then((order.contains(&a.node_id)).cmp(&order.contains(&b.node_id)))
        });
        let drafter = cands.first().map(|x| x.node_id);
        for x in &cands {
            if Some(x.node_id) == drafter {
                notes.push(format!(
                    "{} → asynchronous tree drafter{} (est. {:.0} µs per draft incl. round trip)",
                    x.name,
                    if order.contains(&x.node_id) {
                        " on cores its pipeline stage leaves idle"
                    } else {
                        ""
                    },
                    draft_cost(x) * 1e6
                ));
            } else if !order.contains(&x.node_id) {
                notes.push(format!("{} → overflow KV page pool", x.name));
            }
        }
        // KV pools: free RAM after weights on every node.
        let mut kv_pools = Vec::new();
        for x in &self.nodes {
            let weights = stages
                .iter()
                .find(|s| s.node == x.node_id)
                .map(|s| s.layers.len() as f64 * self.model.layer_bytes)
                .unwrap_or(0.0)
                + if x.node_id == coordinator {
                    self.model.head_bytes
                } else {
                    0.0
                };
            let draft_reserve = if Some(x.node_id) == drafter {
                0.25
            } else {
                0.0
            };
            let free = (x.available_ram_mb * 1e6 - weights) * (1.0 - draft_reserve);
            if free > 0.0 {
                kv_pools.push((x.node_id, free as usize));
            }
        }
        let est_tok_s = self.cfg.expected_accept / t;
        let htep = self.htep_group();
        if let Some(g) = &htep {
            notes.push(format!("low-latency group {g:?} (RTT < 100 µs) → H-TEP: replicated attention, experts split across ranks, 1 sparse all-reduce per layer"));
        }
        let (stages, coordinator, drafter) = match &htep {
            Some(g) => (
                g.iter()
                    .map(|&n| StageAssignment {
                        node: n,
                        layers: 0..self.model.n_layers,
                    })
                    .collect(),
                g[0],
                drafter.filter(|d| !g.contains(d)).or(Some(g[0])),
            ),
            None => (stages, coordinator, drafter),
        };
        ClusterPlan {
            stages,
            coordinator,
            drafter,
            kv_pools,
            est_epoch_ms: t * 1e3,
            est_tok_s,
            notes,
            htep,
        }
    }
}

impl HeteroClusterSolver {
    /// Largest set of capable nodes whose pairwise best one-way latency is
    /// < 50 µs (RTT < 100 µs) and whose RAM fits replicated attention + 1/W of
    /// the experts per rank. `None` when no such group of ≥ 2 nodes exists.
    pub fn htep_group(&self) -> Option<Vec<u16>> {
        if !self.model.moe {
            return None;
        }
        let capable: Vec<&HardwareNodeSpec> = self
            .nodes
            .iter()
            .filter(|n| n.platform != Platform::MobileEdgeNpu)
            .collect();
        let mut best: Option<Vec<u16>> = None;
        for seed in &capable {
            let mut g = vec![seed.node_id];
            for c in &capable {
                if g.contains(&c.node_id) {
                    continue;
                }
                let ok = g
                    .iter()
                    .all(|&m| self.node(m).min_one_way_us(c.node_id) < 50.0);
                if ok {
                    g.push(c.node_id);
                }
            }
            let w = g.len() as f64;
            let per_rank = self.model.n_layers as f64
                * (self.model.attn_bytes + self.model.expert_bytes / w)
                + self.model.head_bytes;
            let fits = g.iter().all(|&n| {
                self.node(n).available_ram_mb * 1e6 * (1.0 - self.cfg.kv_reserve_frac) >= per_rank
            });
            if g.len() >= 2 && fits && best.as_ref().map(|b| g.len() > b.len()).unwrap_or(true) {
                g.sort();
                best = Some(g);
            }
        }
        best
    }
}

fn permutations(v: &[u16]) -> Vec<Vec<u16>> {
    if v.len() <= 1 {
        return vec![v.to_vec()];
    }
    let mut out = Vec::new();
    for i in 0..v.len() {
        let mut rest = v.to_vec();
        let x = rest.remove(i);
        for mut p in permutations(&rest) {
            p.insert(0, x);
            out.push(p);
        }
    }
    out
}
