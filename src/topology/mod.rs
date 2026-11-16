//! Cluster topology description, device presets and the heterogeneous
//! role/shard solver.

pub mod profiles;
pub mod solver;

pub use profiles::{preset, DevicePreset, PRESETS, RAM_SCALE};
pub use solver::{
    ClusterPlan, HardwareNodeSpec, HeteroClusterSolver, LinkSpec, ModelFootprint, Platform,
    SolverConfig, StageAssignment,
};

use crate::model::ModelConfig;

/// Solver footprint of a model at `holo_bits` activation compression.
pub fn footprint(cfg: &ModelConfig, holo_bits: u8) -> ModelFootprint {
    ModelFootprint {
        n_layers: cfg.n_layers,
        layer_bytes: (cfg.layer_params() * 4) as f64,
        layer_active_bytes: (cfg.layer_active_params() * 4) as f64,
        layer_flops_per_row: 2.0 * cfg.layer_active_params() as f64,
        head_bytes: (2 * cfg.vocab * cfg.d_model * 4) as f64,
        act_bytes_per_row: cfg.d_model as f64 * holo_bits as f64 / 8.0
            + cfg.d_model as f64 / 32.0
            + 16.0,
        kv_bytes_per_token_layer: cfg.kv_bytes_per_token_layer() as f64,
        moe: cfg.moe.is_some(),
        attn_bytes: (cfg.d_model * (2 * cfg.q_dim() + 2 * cfg.kv_dim()) * 4) as f64,
        expert_bytes: cfg
            .moe
            .as_ref()
            .map(|m| (m.n_experts * 3 * cfg.d_model * m.d_expert * 4) as f64)
            .unwrap_or(0.0),
    }
}

/// Exo's memory-weighted ring partition: layers ∝ node RAM (largest remainder).
pub fn memory_weighted_partition(n_layers: usize, ram: &[f64]) -> Vec<usize> {
    let total: f64 = ram.iter().sum();
    let raw: Vec<f64> = ram.iter().map(|r| r / total * n_layers as f64).collect();
    let mut counts: Vec<usize> = raw.iter().map(|x| x.floor() as usize).collect();
    let mut left = n_layers - counts.iter().sum::<usize>();
    let mut order: Vec<usize> = (0..ram.len()).collect();
    order.sort_by(|&a, &b| {
        (raw[b] - raw[b].floor())
            .partial_cmp(&(raw[a] - raw[a].floor()))
            .unwrap()
    });
    for &i in &order {
        if left == 0 {
            break;
        }
        counts[i] += 1;
        left -= 1;
    }
    // every ring member holds at least one layer
    for i in 0..counts.len() {
        if counts[i] == 0 {
            let j = (0..counts.len()).max_by_key(|&j| counts[j]).unwrap();
            counts[j] -= 1;
            counts[i] = 1;
        }
    }
    counts
}
