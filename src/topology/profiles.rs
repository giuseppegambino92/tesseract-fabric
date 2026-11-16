//! Device presets used by the benchmark and the `tesseract-node` CLI.
//!
//! Bandwidth / TFLOPS figures are public spec-sheet class numbers; they drive
//! (a) the solver's cost model and (b) the emulated compute slowdown of each
//! node relative to the fastest node in a cluster (decode is memory-bandwidth
//! bound, so slowdown = max_bw / node_bw). `ram_mb` is the device RAM; the
//! benchmark scales it by `RAM_SCALE` so the ~150M-parameter benchmark model
//! has to be split the way a 70B-class model would be on the real devices.

use super::solver::{HardwareNodeSpec, LinkSpec, Platform};
use crate::wire::multipath::LinkProfile;

/// Emulated capacity = device RAM × RAM_SCALE (MB → MB).
pub const RAM_SCALE: f64 = 1.0 / 200.0;

#[derive(Clone, Debug)]
pub struct DevicePreset {
    pub key: &'static str,
    pub name: &'static str,
    pub platform: Platform,
    pub tflops_fp16: f64,
    pub mem_bw_gbps: f64,
    pub ram_mb: f64,
}

pub const PRESETS: &[DevicePreset] = &[
    DevicePreset {
        key: "studio-m5max",
        name: "Mac Studio M5 Max",
        platform: Platform::AppleSiliconUMA,
        tflops_fp16: 34.0,
        mem_bw_gbps: 546.0,
        ram_mb: 128_000.0,
    },
    DevicePreset {
        key: "rtx5090",
        name: "Linux RTX 5090 Workstation",
        platform: Platform::DiscreteGpuWorkstation,
        tflops_fp16: 105.0,
        mem_bw_gbps: 1792.0,
        ram_mb: 32_000.0,
    },
    DevicePreset {
        key: "macbook-pro",
        name: "MacBook Pro (M-Pro)",
        platform: Platform::AppleSiliconUMA,
        tflops_fp16: 9.0,
        mem_bw_gbps: 273.0,
        ram_mb: 48_000.0,
    },
    DevicePreset {
        key: "android-snapdragon",
        name: "Android Snapdragon Edge",
        platform: Platform::MobileEdgeNpu,
        tflops_fp16: 4.5,
        mem_bw_gbps: 77.0,
        ram_mb: 16_000.0,
    },
    DevicePreset {
        key: "cpu-server",
        name: "x86_64 CPU Server",
        platform: Platform::CpuServer,
        tflops_fp16: 6.0,
        mem_bw_gbps: 300.0,
        ram_mb: 256_000.0,
    },
];

pub fn preset(key: &str) -> Option<&'static DevicePreset> {
    PRESETS.iter().find(|p| p.key == key)
}

impl DevicePreset {
    pub fn spec(&self, node_id: u16, links: Vec<LinkSpec>) -> HardwareNodeSpec {
        HardwareNodeSpec {
            node_id,
            name: self.name.to_string(),
            platform: self.platform,
            compute_tflops_fp16: self.tflops_fp16,
            memory_bandwidth_gbps: self.mem_bw_gbps,
            available_ram_mb: self.ram_mb * RAM_SCALE,
            links,
        }
    }
}

/// Physical links between two presets (what both ends physically share).
pub fn links_between(a: &DevicePreset, b: &DevicePreset, studio_tb5: bool) -> Vec<LinkProfile> {
    let is_desk = |p: &DevicePreset| matches!(p.key, "studio-m5max" | "rtx5090" | "cpu-server");
    if studio_tb5 && a.key == "studio-m5max" && b.key == "studio-m5max" {
        return vec![LinkProfile::thunderbolt5()];
    }
    if is_desk(a) && is_desk(b) {
        return vec![LinkProfile::ethernet_2_5g(), LinkProfile::wifi6e()];
    }
    vec![LinkProfile::wifi6e()]
}

/// Slowdown of each node relative to the fastest (by memory bandwidth).
pub fn slowdowns(specs: &[HardwareNodeSpec]) -> Vec<f64> {
    let best = specs
        .iter()
        .map(|s| s.memory_bandwidth_gbps)
        .fold(0.0, f64::max);
    specs
        .iter()
        .map(|s| best / s.memory_bandwidth_gbps)
        .collect()
}
