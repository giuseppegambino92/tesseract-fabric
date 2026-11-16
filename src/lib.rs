//! # tesseract-fabric
//!
//! A self-contained distributed transformer inference engine in pure Rust
//! (no external crates). Five cooperating subsystems:
//!
//! 1. [`compression`] — holographic residual compression (FWHT + outlier
//!    escape + sub-byte lattice + error feedback) for inter-shard activations.
//! 2. [`speculative`] — best-first speculative token trees, exact tree
//!    rejection sampling, braid strands and 32-byte tree commits.
//! 3. [`routing`] — lookahead MoE expert prediction and hot-buffer pre-staging.
//! 4. [`kv_fabric`] — paged, disaggregated KV with exact LSE-merged attention.
//! 5. [`wire`] / [`topology`] — the `TesseractWire` protocol, link-shaped
//!    multipath TCP transport, cross-silicon determinism, and the
//!    heterogeneous cluster solver.
//!
//! [`cluster`] wires everything into live multi-node engines (plus faithful
//! ring-pipeline and tensor-parallel baselines) that talk over real sockets.
//!
//! V8.0 production runtime: [`loader`] (zero-copy GGUF / SafeTensors),
//! quantized kernels and native acceleration in [`tensor::quant`] /
//! [`tensor::accel`], [`tokenizer`] (BPE + sampler), LAN / Tailscale
//! discovery in [`cluster::discovery`], and the OpenAI-compatible HTTP / SSE
//! server in [`server`].

// Dense numeric kernels index several parallel arrays by position.
#![allow(clippy::needless_range_loop)]

pub mod cluster;
pub mod compression;
pub mod kv_fabric;
pub mod loader;
pub mod model;
pub mod routing;
pub mod server;
pub mod speculative;
pub mod tensor;
pub mod tokenizer;
pub mod topology;
pub mod wire;
