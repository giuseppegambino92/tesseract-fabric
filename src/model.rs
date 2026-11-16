//! Synthetic decoder-only transformer (Llama/Qwen-style): RMSNorm → GQA
//! attention with RoPE → RMSNorm → SwiGLU FFN or top-k Mixture-of-Experts.
//!
//! Weights are generated deterministically from `(seed, layer, tensor)` so every
//! node in a cluster can materialise exactly the layers it owns without
//! shipping weights over the network. Output projections use GPT-2 style
//! `1/sqrt(2·n_layers)` residual scaling so the residual stream keeps token
//! identity (as trained LLMs do) and logits are scaled to give peaked,
//! LLM-like next-token distributions.

use crate::kv_fabric::{AttentionBackend, BatchMeta};
use crate::routing::ExpertStore;
use crate::tensor::determinism::CanonicalBoundaryAnchor;
use crate::tensor::{rms_norm, silu, Linear, Rng, RopeTable, Tensor2D};
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq)]
pub struct MoeConfig {
    pub n_experts: usize,
    pub top_k: usize,
    pub d_expert: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelConfig {
    pub vocab: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub d_head: usize,
    pub d_ff: usize,
    pub moe: Option<MoeConfig>,
    pub rope_theta: f32,
    pub rms_eps: f32,
    /// Standard deviation of the LM head's unstructured logit component.
    pub logit_std: f32,
    /// Logit boost (in nats) of a token's most likely successor under the
    /// token-transition prior embedded in the LM head (0 = unstructured, V1).
    pub transition_prior: f32,
    /// Power-law exponent p of the singular-value decay σ_i ∝ i^-p of layer
    /// weights (0 = isotropic Gaussian, V1).
    pub spectral_decay: f32,
    /// Logit boost (nats) of the successors of the *previous* token. Non-zero
    /// enables the V3 contextual construction: a layer-0 attention head that
    /// attends to position m−1 copies the previous token into the residual
    /// stream, and the LM head reads it. The next token then depends on two
    /// tokens, one of which is only reachable through attention.
    pub context_prior: f32,
    /// Shared embedding component β (in units of √d) that gives the layer-0
    /// positional head a content-independent query/key.
    pub context_bias: f32,
    pub seed: u64,
    /// Where the weights come from: generated from `seed` (synthetic) or
    /// memory-mapped from a GGUF / SafeTensors model file.
    pub source: WeightSource,
    /// RoPE frequency scaling (Llama 3 freq factors, linear, YaRN).
    pub rope_scaling: crate::tensor::RopeScaling,
    /// Exact accumulation in the row-parallel projections (attention output,
    /// dense FFN down): their outputs no longer depend on how a tensor-
    /// parallel split is reduced, so H-TEP reproduces a single node bit for
    /// bit. On by default for model files.
    pub exact: bool,
}

/// Origin of a model's weights.
#[derive(Clone, Debug, PartialEq, Default)]
pub enum WeightSource {
    /// Deterministic synthetic weights from `ModelConfig::seed`.
    #[default]
    Synthetic,
    /// A `.gguf` file, or a SafeTensors model (`.safetensors` file with a
    /// `config.json` beside it, or a directory). `requant` re-quantizes every
    /// matrix at load time (heap copy) instead of using the stored format.
    File {
        path: String,
        requant: Option<crate::tensor::quant::WeightFormat>,
    },
}

impl WeightSource {
    /// Stable 64-bit identity (cache keys).
    pub fn key(&self) -> u64 {
        match self {
            WeightSource::Synthetic => 0,
            WeightSource::File { path, requant } => {
                // FNV-1a over the path and requant id
                let mut h = 0xcbf2_9ce4_8422_2325u64;
                for b in path
                    .bytes()
                    .chain([requant.map(|f| f.ggml_id() as u8 + 1).unwrap_or(0)])
                {
                    h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3);
                }
                h
            }
        }
    }
    pub fn is_file(&self) -> bool {
        matches!(self, WeightSource::File { .. })
    }
}

/// Structured-initialisation defaults (see `ModelConfig::transition_prior`).
pub const STRUCTURED_LOGIT_STD: f32 = 1.5;
pub const STRUCTURED_PRIOR: f32 = 12.0;
pub const STRUCTURED_DECAY: f32 = 0.5;

/// V3 contextual-model defaults (see `ModelConfig::context_prior`).
pub const CONTEXT_PRIOR: f32 = 8.0;
/// Current-token prior used together with the context prior (equal weight:
/// the next token depends on both of the last two tokens).
pub const CONTEXT_TRANSITION_PRIOR: f32 = 8.0;
pub const CONTEXT_BIAS: f32 = 0.5;
/// Amplitude of the layer-0 previous-token head's positional query/key.
pub const PREV_HEAD_AMPLITUDE: f32 = 9.0;
/// RoPE frequency pairs used by the previous-token head (the fastest ones,
/// which discriminate neighbouring offsets).
pub const PREV_HEAD_PAIRS: usize = 12;
/// Norm of the copied previous-token vector relative to an embedding (κ).
pub const PREV_COPY_GAIN: f32 = 1.0;

/// RMS of the final residual stream in the contextual model (unit-variance
/// embedding + shared positional channel β² + previous-token copy κ² + the
/// ≈0.5 variance added by 1/√(2L)-scaled layers): priors are calibrated
/// against it so a prior's boost is ≈ its nominal value in nats.
pub fn final_rms(cfg: &ModelConfig) -> f32 {
    if cfg.context_prior > 0.0 {
        (1.0 + cfg.context_bias * cfg.context_bias + PREV_COPY_GAIN * PREV_COPY_GAIN + 0.5).sqrt()
    } else {
        1.0
    }
}

/// Relative weights of a token's successors under the transition prior.
pub const SUCCESSOR_WEIGHTS: [f32; 3] = [1.0, 0.8, 0.65];

impl ModelConfig {
    /// The V1 benchmark model: identical shapes, unstructured random weights.
    pub fn bench_moe_v1() -> Self {
        ModelConfig {
            logit_std: 6.0,
            transition_prior: 0.0,
            spectral_decay: 0.0,
            context_prior: 0.0,
            context_bias: 0.0,
            ..Self::bench_moe()
        }
    }

    /// The V2 benchmark model (structured, first-order transition prior only).
    pub fn bench_moe_v2() -> Self {
        ModelConfig {
            transition_prior: STRUCTURED_PRIOR,
            context_prior: 0.0,
            context_bias: 0.0,
            ..Self::bench_moe()
        }
    }

    /// MoE model used by the cluster benchmarks (≈150M params, 2-of-8 experts).
    pub fn bench_moe() -> Self {
        ModelConfig {
            vocab: 2048,
            d_model: 512,
            n_layers: 16,
            n_heads: 8,
            n_kv_heads: 4,
            d_head: 64,
            d_ff: 1408,
            moe: Some(MoeConfig {
                n_experts: 8,
                top_k: 2,
                d_expert: 704,
            }),
            rope_theta: 10_000.0,
            rms_eps: 1e-5,
            logit_std: STRUCTURED_LOGIT_STD,
            transition_prior: CONTEXT_TRANSITION_PRIOR,
            spectral_decay: STRUCTURED_DECAY,
            context_prior: CONTEXT_PRIOR,
            context_bias: CONTEXT_BIAS,
            seed: 0x7E55_E2AC,
            source: WeightSource::Synthetic,
            rope_scaling: Default::default(),
            exact: false,
        }
    }

    /// Dense model for the long-context stress test.
    pub fn long_context() -> Self {
        ModelConfig {
            vocab: 2048,
            d_model: 512,
            n_layers: 8,
            n_heads: 8,
            n_kv_heads: 2,
            d_head: 64,
            d_ff: 1024,
            moe: None,
            rope_theta: 500_000.0,
            rms_eps: 1e-5,
            logit_std: STRUCTURED_LOGIT_STD,
            transition_prior: STRUCTURED_PRIOR,
            spectral_decay: STRUCTURED_DECAY,
            context_prior: 0.0,
            context_bias: 0.0,
            seed: 0x10C0_47E7,
            source: WeightSource::Synthetic,
            rope_scaling: Default::default(),
            exact: false,
        }
    }

    /// Small config for unit/integration tests.
    pub fn tiny() -> Self {
        ModelConfig {
            vocab: 256,
            d_model: 128,
            n_layers: 6,
            n_heads: 4,
            n_kv_heads: 2,
            d_head: 32,
            d_ff: 256,
            moe: Some(MoeConfig {
                n_experts: 4,
                top_k: 2,
                d_expert: 128,
            }),
            rope_theta: 10_000.0,
            rms_eps: 1e-5,
            logit_std: STRUCTURED_LOGIT_STD,
            transition_prior: CONTEXT_TRANSITION_PRIOR,
            spectral_decay: STRUCTURED_DECAY,
            context_prior: CONTEXT_PRIOR,
            context_bias: CONTEXT_BIAS,
            seed: 42,
            source: WeightSource::Synthetic,
            rope_scaling: Default::default(),
            exact: false,
        }
    }

    pub fn q_dim(&self) -> usize {
        self.n_heads * self.d_head
    }
    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.d_head
    }

    /// Parameter count of one transformer layer (all experts).
    pub fn layer_params(&self) -> usize {
        let d = self.d_model;
        let attn = d * self.q_dim() * 2 + d * self.kv_dim() * 2 + 2 * d;
        let ffn = match &self.moe {
            Some(m) => m.n_experts * 3 * d * m.d_expert + m.n_experts * d,
            None => 3 * d * self.d_ff,
        };
        attn + ffn
    }

    /// Parameters touched per token in one layer (active experts only).
    pub fn layer_active_params(&self) -> usize {
        let d = self.d_model;
        let attn = d * self.q_dim() * 2 + d * self.kv_dim() * 2;
        let ffn = match &self.moe {
            Some(m) => m.top_k * 3 * d * m.d_expert + m.n_experts * d,
            None => 3 * d * self.d_ff,
        };
        attn + ffn
    }

    pub fn total_params(&self) -> usize {
        self.layer_params() * self.n_layers + 2 * self.vocab * self.d_model
    }

    /// KV bytes per token per layer at f32.
    pub fn kv_bytes_per_token_layer(&self) -> usize {
        2 * self.kv_dim() * 4
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut b = Vec::new();
        for v in [
            self.vocab,
            self.d_model,
            self.n_layers,
            self.n_heads,
            self.n_kv_heads,
            self.d_head,
            self.d_ff,
        ] {
            b.extend_from_slice(&(v as u32).to_le_bytes());
        }
        match &self.moe {
            Some(m) => {
                b.push(1);
                for v in [m.n_experts, m.top_k, m.d_expert] {
                    b.extend_from_slice(&(v as u32).to_le_bytes());
                }
            }
            None => b.push(0),
        }
        for v in [
            self.rope_theta,
            self.rms_eps,
            self.logit_std,
            self.transition_prior,
            self.spectral_decay,
            self.context_prior,
            self.context_bias,
        ] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b.extend_from_slice(&self.seed.to_le_bytes());
        match &self.source {
            WeightSource::Synthetic => b.push(0),
            WeightSource::File { path, requant } => {
                b.push(1);
                b.extend_from_slice(&(path.len() as u16).to_le_bytes());
                b.extend_from_slice(path.as_bytes());
                b.push(requant.map(|f| f.ggml_id() as u8 + 1).unwrap_or(0));
            }
        }
        let rs = &self.rope_scaling;
        b.extend_from_slice(&(rs.freq_factors.len() as u32).to_le_bytes());
        for f in &rs.freq_factors {
            b.extend_from_slice(&f.to_le_bytes());
        }
        b.extend_from_slice(&rs.linear.to_le_bytes());
        let (yf, yc) = rs.yarn.unwrap_or((0.0, 0));
        b.push(rs.yarn.is_some() as u8);
        b.extend_from_slice(&yf.to_le_bytes());
        b.extend_from_slice(&yc.to_le_bytes());
        b.push(self.exact as u8);
        b
    }

    /// The RoPE table of this model (with any frequency scaling).
    pub fn rope_table(&self) -> RopeTable {
        RopeTable::with_scaling(self.d_head, self.rope_theta, &self.rope_scaling)
    }

    pub fn decode(b: &[u8]) -> (Self, usize) {
        let u = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as usize;
        let f = |o: usize| f32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let mut o = 28;
        let moe = if b[o] == 1 {
            let m = MoeConfig {
                n_experts: u(o + 1),
                top_k: u(o + 5),
                d_expert: u(o + 9),
            };
            o += 13;
            Some(m)
        } else {
            o += 1;
            None
        };
        let cfg = ModelConfig {
            vocab: u(0),
            d_model: u(4),
            n_layers: u(8),
            n_heads: u(12),
            n_kv_heads: u(16),
            d_head: u(20),
            d_ff: u(24),
            moe,
            rope_theta: f(o),
            rms_eps: f(o + 4),
            logit_std: f(o + 8),
            transition_prior: f(o + 12),
            spectral_decay: f(o + 16),
            context_prior: f(o + 20),
            context_bias: f(o + 24),
            seed: u64::from_le_bytes(b[o + 28..o + 36].try_into().unwrap()),
            source: WeightSource::Synthetic,
            rope_scaling: Default::default(),
            exact: false,
        };
        let mut o = o + 36;
        let mut cfg = cfg;
        if b[o] == 1 {
            let n = u16::from_le_bytes([b[o + 1], b[o + 2]]) as usize;
            let path = String::from_utf8(b[o + 3..o + 3 + n].to_vec()).unwrap();
            let r = b[o + 3 + n];
            cfg.source = WeightSource::File {
                path,
                requant: (r > 0)
                    .then(|| crate::tensor::quant::WeightFormat::from_ggml(r as u32 - 1))
                    .flatten(),
            };
            o += 4 + n;
        } else {
            o += 1;
        }
        let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let nf = u32_at(o) as usize;
        cfg.rope_scaling.freq_factors = (0..nf)
            .map(|i| f32::from_bits(u32_at(o + 4 + 4 * i)))
            .collect();
        o += 4 + 4 * nf;
        cfg.rope_scaling.linear = f32::from_bits(u32_at(o));
        if b[o + 4] == 1 {
            cfg.rope_scaling.yarn = Some((f32::from_bits(u32_at(o + 5)), u32_at(o + 9)));
        }
        o += 13;
        cfg.exact = b[o] == 1;
        o += 1;
        (cfg, o)
    }
}

fn tensor_rng(seed: u64, layer: usize, tensor: u64) -> Rng {
    Rng::new(
        seed ^ ((layer as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15))
            ^ tensor.wrapping_mul(0xD1B5_4A32_D192_ED03),
    )
}

/// Weight matrix [out, in] with element std `std`. With `spectral_decay = p`
/// the singular values follow σ_i ∝ i^-p: Gaussian rows get power-law column
/// scales (normalised to unit mean square, so the overall scale is unchanged)
/// and are then rotated by a randomised block-Hadamard transform so the
/// spectrum is not axis-aligned.
pub fn weight(cfg: &ModelConfig, out_f: usize, in_f: usize, std: f32, rng: &mut Rng) -> Linear {
    let mut l = Linear::new_random(out_f, in_f, std, rng);
    let p = cfg.spectral_decay;
    if p > 0.0 {
        let raw: Vec<f32> = (0..in_f).map(|c| ((c + 1) as f32).powf(-p)).collect();
        let ms = (raw.iter().map(|v| v * v).sum::<f32>() / in_f as f32).sqrt();
        let scale: Vec<f32> = raw.iter().map(|v| v / ms).collect();
        let block = 1usize << in_f.trailing_zeros().min(10);
        let plan = crate::compression::fwht::HadamardPlan::blocked(in_f, block, rng.next_u64());
        let mut tmp = vec![0.0f32; plan.padded];
        for r in 0..out_f {
            let row = l.w.row_mut(r);
            for (v, s) in row.iter_mut().zip(&scale) {
                *v *= s;
            }
            plan.forward(row, &mut tmp);
            row.copy_from_slice(&tmp[..in_f]);
        }
    }
    l
}

/// The k-th most likely successor of token `t` under the transition prior.
pub fn successor(cfg: &ModelConfig, t: u32, k: usize) -> u32 {
    let mut r = Rng::new(cfg.seed ^ 0x5ACC_E550 ^ ((t as u64) << 8) ^ k as u64);
    r.below(cfg.vocab) as u32
}

/// Successor of token `t` under the *context* (previous-token) prior.
pub fn context_successor(cfg: &ModelConfig, t: u32, k: usize) -> u32 {
    let mut r = Rng::new(cfg.seed ^ 0xC0_7E47 ^ ((t as u64) << 9) ^ (k as u64 + 77));
    r.below(cfg.vocab) as u32
}

/// Remove the input direction `dir ⊙ g` (g = the RMSNorm weights applied
/// before this matrix) from every row: the matrix becomes blind to it.
pub fn project_out(lin: &mut Linear, dir: &[f32], g: &[f32]) {
    let ug: Vec<f32> = dir.iter().zip(g).map(|(a, b)| a * b).collect();
    let uu = crate::tensor::dot(&ug, &ug).max(1e-12);
    for r in 0..lin.w.rows {
        let row = lin.w.row_mut(r);
        let c = crate::tensor::dot(row, &ug) / uu;
        for (w, uv) in row.iter_mut().zip(&ug) {
            *w -= c * uv;
        }
    }
}

/// Unit direction û shared by all embeddings in the contextual model.
pub fn context_direction(cfg: &ModelConfig) -> Vec<f32> {
    let mut r = tensor_rng(cfg.seed, usize::MAX - 3, 1);
    let v: Vec<f32> = (0..cfg.d_model).map(|_| r.normal()).collect();
    let n = crate::tensor::l2_norm(&v);
    v.into_iter().map(|x| x / n).collect()
}

/// Positional query/key of the layer-0 previous-token head: with RoPE, the
/// score between positions m and n is Σ_i A² cos((m − n − 1)θ_i), maximal at
/// n = m − 1, because k̃_i = q̃_i·e^{iθ_i} in every complex pair.
pub fn prev_head_qk(cfg: &ModelConfig) -> (Vec<f32>, Vec<f32>) {
    let dh = cfg.d_head;
    let half = dh / 2;
    let rope = RopeTable::new(dh, cfg.rope_theta);
    let mut r = tensor_rng(cfg.seed, usize::MAX - 3, 2);
    let (mut q, mut k) = (vec![0.0; dh], vec![0.0; dh]);
    for i in 0..half.min(PREV_HEAD_PAIRS) {
        let phi = 6.283_185_5 * r.uniform();
        let th = rope.inv_freq[i];
        q[i] = PREV_HEAD_AMPLITUDE * phi.cos();
        q[i + half] = PREV_HEAD_AMPLITUDE * phi.sin();
        k[i] = PREV_HEAD_AMPLITUDE * (phi + th).cos();
        k[i + half] = PREV_HEAD_AMPLITUDE * (phi + th).sin();
    }
    (q, k)
}

/// Balanced split of `d` units into `world` contiguous ranges, cut on
/// multiples of `align` when there are enough aligned blocks.
pub fn hidden_split(d: usize, rank: usize, world: usize, align: usize) -> (usize, usize) {
    if align > 1 && d.is_multiple_of(align) && d / align >= world {
        let b = d / align;
        return (b * rank / world * align, b * (rank + 1) / world * align);
    }
    (d * rank / world, d * (rank + 1) / world)
}

/// SwiGLU FFN / single expert: down(silu(gate(x)) ⊙ up(x)).
#[derive(Clone, Debug)]
pub struct Expert {
    pub w_gate: Linear,
    pub w_up: Linear,
    pub w_down: Linear,
}

impl Expert {
    pub fn generate(cfg: &ModelConfig, layer: usize, id: u64, d_hidden: usize) -> Self {
        let d = cfg.d_model;
        let out_scale = 1.0 / (2.0 * cfg.n_layers as f32).sqrt();
        Expert {
            w_gate: weight(
                cfg,
                d_hidden,
                d,
                1.0 / (d as f32).sqrt(),
                &mut tensor_rng(cfg.seed, layer, 100 + 3 * id),
            ),
            w_up: weight(
                cfg,
                d_hidden,
                d,
                1.0 / (d as f32).sqrt(),
                &mut tensor_rng(cfg.seed, layer, 101 + 3 * id),
            ),
            w_down: weight(
                cfg,
                d,
                d_hidden,
                out_scale / (d_hidden as f32).sqrt(),
                &mut tensor_rng(cfg.seed, layer, 102 + 3 * id),
            ),
        }
    }

    pub fn forward(&self, x: &Tensor2D) -> Tensor2D {
        self.w_down.forward(&self.hidden(x))
    }

    /// silu(x·W_gateᵀ) ⊙ (x·W_upᵀ): the input of the down projection.
    pub fn hidden(&self, x: &Tensor2D) -> Tensor2D {
        let mut g = self.w_gate.forward(x);
        let u = self.w_up.forward(x);
        for (gi, ui) in g.data.iter_mut().zip(&u.data) {
            *gi = silu(*gi) * ui;
        }
        g
    }

    /// Rank `rank`'s share of this expert's hidden units out of `world`:
    /// a balanced split that covers every unit (any world size), on 32-unit
    /// block boundaries when the weights are block-quantized so the slices
    /// stay zero-copy views. Identical to `d / world` stripes whenever
    /// `world` divides the width (f32 weights).
    pub fn hidden_range(&self, rank: usize, world: usize) -> (usize, usize) {
        let d = self.w_gate.out_features();
        let align = self.w_down.format().block_len();
        hidden_split(d, rank, world, align)
    }

    /// Keep only hidden units [start, end) (tensor-parallel column split).
    pub fn slice_hidden(&self, start: usize, end: usize) -> Expert {
        Expert {
            w_gate: self.w_gate.slice_rows(start, end),
            w_up: self.w_up.slice_rows(start, end),
            w_down: self.w_down.slice_cols(start, end),
        }
    }

    pub fn bytes(&self) -> usize {
        self.w_gate.bytes() + self.w_up.bytes() + self.w_down.bytes()
    }
}

/// Where a layer's experts live.
#[derive(Clone)]
pub enum ExpertBank {
    /// All experts resident in fast memory.
    Resident(Vec<Arc<Expert>>),
    /// Cold int8 storage + bounded hot buffer (see `routing::ExpertStore`).
    Staged(Arc<ExpertStore>),
}

impl ExpertBank {
    pub fn acquire(&self, e: usize) -> Arc<Expert> {
        match self {
            ExpertBank::Resident(v) => v[e].clone(),
            ExpertBank::Staged(s) => s.acquire(e),
        }
    }
}

#[derive(Clone)]
pub enum FfnWeights {
    Dense(Arc<Expert>),
    Moe {
        router: Linear,
        bank: ExpertBank,
        top_k: usize,
    },
}

/// Optional per-architecture attention terms.
#[derive(Clone, Debug, Default)]
pub struct AttnExtras {
    /// Q / K / V projection biases (Qwen2).
    pub bq: Option<Vec<f32>>,
    pub bk: Option<Vec<f32>>,
    pub bv: Option<Vec<f32>>,
    /// Per-head RMSNorm weights applied to q / k before RoPE (Qwen3).
    pub q_norm: Option<Vec<f32>>,
    pub k_norm: Option<Vec<f32>>,
}

impl AttnExtras {
    /// Restrict the biases to heads [h0, h1) (q) and [g0, g1) (kv).
    pub fn slice_heads(&self, h0: usize, h1: usize, g0: usize, g1: usize, dh: usize) -> Self {
        let s = |b: &Option<Vec<f32>>, a: usize, z: usize| {
            b.as_ref().map(|v| v[a * dh..z * dh].to_vec())
        };
        AttnExtras {
            bq: s(&self.bq, h0, h1),
            bk: s(&self.bk, g0, g1),
            bv: s(&self.bv, g0, g1),
            q_norm: self.q_norm.clone(),
            k_norm: self.k_norm.clone(),
        }
    }
}

/// q, k, v for a batch of normalised rows: projections, optional biases,
/// optional per-head QK RMSNorm, then RoPE at each row's position. Shared by
/// every engine so all of them run the same attention math.
#[allow(clippy::too_many_arguments)]
pub fn project_qkv(
    wq: &Linear,
    wk: &Linear,
    wv: &Linear,
    ex: &AttnExtras,
    xn: &Tensor2D,
    rope: &RopeTable,
    positions: &[usize],
    d_head: usize,
    eps: f32,
) -> (Tensor2D, Tensor2D, Tensor2D) {
    let mut q = wq.forward(xn);
    let mut k = wk.forward(xn);
    let mut v = wv.forward(xn);
    let add = |t: &mut Tensor2D, b: &Option<Vec<f32>>| {
        if let Some(b) = b {
            for i in 0..t.rows {
                for (x, y) in t.row_mut(i).iter_mut().zip(b) {
                    *x += y;
                }
            }
        }
    };
    add(&mut q, &ex.bq);
    add(&mut k, &ex.bk);
    add(&mut v, &ex.bv);
    let mut buf = vec![0.0f32; d_head];
    for i in 0..xn.rows {
        let pos = positions[i];
        for (t, norm) in [(&mut q, &ex.q_norm), (&mut k, &ex.k_norm)] {
            for h in 0..t.cols / d_head {
                let head = &mut t.row_mut(i)[h * d_head..(h + 1) * d_head];
                if let Some(nw) = norm {
                    rms_norm(head, nw, eps, &mut buf);
                    head.copy_from_slice(&buf);
                }
                rope.apply(head, pos);
            }
        }
    }
    (q, k, v)
}

#[derive(Clone)]
pub struct LayerWeights {
    pub index: usize,
    pub extras: AttnExtras,
    pub attn_norm: Vec<f32>,
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub ffn_norm: Vec<f32>,
    pub ffn: FfnWeights,
}

fn norm_weights(cfg: &ModelConfig, layer: usize, tensor: u64) -> Vec<f32> {
    let mut rng = tensor_rng(cfg.seed, layer, tensor);
    (0..cfg.d_model).map(|_| 1.0 + 0.1 * rng.normal()).collect()
}

/// The pre-FFN RMSNorm weights of `layer` (lookahead predictors).
pub fn layer_ffn_norm(cfg: &ModelConfig, layer: usize) -> Vec<f32> {
    if cfg.source.is_file() {
        return crate::loader::sharded_loader::load_layer(cfg, layer).ffn_norm;
    }
    norm_weights(cfg, layer, 6)
}

/// Router weights for a layer (also used to warm-start lookahead predictors).
pub fn generate_router(cfg: &ModelConfig, layer: usize) -> Option<Linear> {
    if cfg.source.is_file() {
        return crate::loader::sharded_loader::load_router(cfg, layer);
    }
    cfg.moe.as_ref().map(|m| {
        let mut r = Linear::new_random(
            m.n_experts,
            cfg.d_model,
            1.5 / (cfg.d_model as f32).sqrt(),
            &mut tensor_rng(cfg.seed, layer, 50),
        );
        if cfg.context_prior > 0.0 {
            project_out(
                &mut r,
                &context_direction(cfg),
                &norm_weights(cfg, layer, 6),
            );
        }
        r
    })
}

impl LayerWeights {
    /// Weight storage bytes of this layer (all experts, as stored).
    pub fn bytes(&self) -> usize {
        let attn = self.wq.bytes() + self.wk.bytes() + self.wv.bytes() + self.wo.bytes();
        let norms = (self.attn_norm.len() + self.ffn_norm.len()) * 4;
        let ffn = match &self.ffn {
            FfnWeights::Dense(e) => e.bytes(),
            FfnWeights::Moe { router, bank, .. } => {
                router.bytes()
                    + match bank {
                        ExpertBank::Resident(v) => v.iter().map(|e| e.bytes()).sum(),
                        ExpertBank::Staged(s) => s.cold_bytes(),
                    }
            }
        };
        attn + norms + ffn
    }

    /// madvise(WILLNEED) every file-backed matrix of this layer.
    pub fn prefetch(&self) {
        for l in [&self.wq, &self.wk, &self.wv, &self.wo] {
            l.prefetch();
        }
        let ex = |e: &Expert| {
            e.w_gate.prefetch();
            e.w_up.prefetch();
            e.w_down.prefetch();
        };
        match &self.ffn {
            FfnWeights::Dense(e) => ex(e),
            FfnWeights::Moe {
                bank: ExpertBank::Resident(v),
                ..
            } => v.iter().for_each(|e| ex(e)),
            _ => {}
        }
    }

    /// Generate layer `index`. `staged` selects cold/hot expert staging with
    /// the given hot capacity (experts per layer), otherwise all resident.
    pub fn generate(cfg: &ModelConfig, index: usize, staged_capacity: Option<usize>) -> Self {
        if cfg.source.is_file() {
            // zero-copy views into the mapped file; the OS pages experts in
            // on demand, so cold/hot staging does not apply
            return crate::loader::sharded_loader::load_layer(cfg, index);
        }
        let d = cfg.d_model;
        let s_in = 1.0 / (d as f32).sqrt();
        let out_scale = 1.0 / (2.0 * cfg.n_layers as f32).sqrt();
        let ffn = match &cfg.moe {
            None => FfnWeights::Dense(Arc::new({
                let mut e = Expert::generate(cfg, index, 0, cfg.d_ff);
                e.w_down.exact = cfg.exact;
                if cfg.context_prior > 0.0 {
                    let u = context_direction(cfg);
                    let g = norm_weights(cfg, index, 6);
                    project_out(&mut e.w_gate, &u, &g);
                    project_out(&mut e.w_up, &u, &g);
                }
                e
            })),
            Some(m) => {
                let mut experts: Vec<Expert> = (0..m.n_experts)
                    .map(|e| Expert::generate(cfg, index, e as u64 + 1, m.d_expert))
                    .collect();
                if cfg.context_prior > 0.0 {
                    let u = context_direction(cfg);
                    let g = norm_weights(cfg, index, 6);
                    for e in experts.iter_mut() {
                        project_out(&mut e.w_gate, &u, &g);
                        project_out(&mut e.w_up, &u, &g);
                    }
                }
                let bank = match staged_capacity {
                    Some(cap) if cap < m.n_experts => {
                        ExpertBank::Staged(Arc::new(ExpertStore::new(index, &experts, cap)))
                    }
                    _ => ExpertBank::Resident(experts.into_iter().map(Arc::new).collect()),
                };
                FfnWeights::Moe {
                    router: generate_router(cfg, index).unwrap(),
                    bank,
                    top_k: m.top_k,
                }
            }
        };
        let mut lw = LayerWeights {
            index,
            extras: AttnExtras::default(),
            attn_norm: norm_weights(cfg, index, 1),
            wq: weight(
                cfg,
                cfg.q_dim(),
                d,
                s_in,
                &mut tensor_rng(cfg.seed, index, 2),
            ),
            wk: weight(
                cfg,
                cfg.kv_dim(),
                d,
                s_in,
                &mut tensor_rng(cfg.seed, index, 3),
            ),
            wv: weight(
                cfg,
                cfg.kv_dim(),
                d,
                s_in,
                &mut tensor_rng(cfg.seed, index, 4),
            ),
            wo: weight(
                cfg,
                d,
                cfg.q_dim(),
                out_scale / (cfg.q_dim() as f32).sqrt(),
                &mut tensor_rng(cfg.seed, index, 5),
            ),
            ffn_norm: norm_weights(cfg, index, 6),
            ffn,
        };
        if cfg.context_prior > 0.0 {
            // û is a dedicated positional channel: every residual-stream
            // reader except the layer-0 previous-token head ignores it.
            let u = context_direction(cfg);
            if index == 0 {
                lw.attn_norm = vec![1.0; d];
            }
            let g = lw.attn_norm.clone();
            project_out(&mut lw.wq, &u, &g);
            project_out(&mut lw.wk, &u, &g);
            project_out(&mut lw.wv, &u, &g);
        }
        if index == 0 && cfg.context_prior > 0.0 {
            // Previous-token head: query head 0 / kv head 0 read only the
            // shared embedding direction û, so their q/k are positional.
            let u = context_direction(cfg);
            let b = cfg.context_bias;
            let proj = b * (d as f32).sqrt() / (1.0 + b * b).sqrt();
            let (q0, k0) = prev_head_qk(cfg);
            lw.attn_norm = vec![1.0; d];
            // make the copied vector c_p = W_o,h0 W_v,g0 x̂_p as large as an
            // embedding (κ·√d), so it is readable above the residual noise
            // that other layers write into the same 64-d subspace
            let dh = cfg.d_head;
            let mut probe = Rng::new(cfg.seed ^ 0xC0FF);
            let mut norm_sum = 0.0f32;
            let trials = 32;
            for _ in 0..trials {
                let x: Vec<f32> = (0..d).map(|_| probe.normal()).collect();
                let mut xn = vec![0.0; d];
                rms_norm(&x, &lw.attn_norm, cfg.rms_eps, &mut xn);
                let v: Vec<f32> = (0..dh)
                    .map(|r| crate::tensor::dot(lw.wv.w.row(r), &xn))
                    .collect();
                let c: Vec<f32> = (0..d)
                    .map(|i| crate::tensor::dot(&lw.wo.w.row(i)[..dh], &v))
                    .collect();
                norm_sum += crate::tensor::l2_norm(&c);
            }
            let f = PREV_COPY_GAIN * (d as f32).sqrt() / (norm_sum / trials as f32);
            for i in 0..d {
                for w in lw.wo.w.row_mut(i)[..dh].iter_mut() {
                    *w *= f;
                }
            }
            for r in 0..cfg.d_head {
                for (w, uv) in lw.wq.w.row_mut(r).iter_mut().zip(&u) {
                    *w = q0[r] * uv / proj;
                }
                for (w, uv) in lw.wk.w.row_mut(r).iter_mut().zip(&u) {
                    *w = k0[r] * uv / proj;
                }
            }
        }
        lw.wo.exact = cfg.exact;
        lw
    }
}

/// Embedding table, final norm, LM head.
#[derive(Clone)]
pub struct HeadWeights {
    pub embed: crate::tensor::Embedding,
    pub final_norm: Vec<f32>,
    pub lm_head: Linear,
}

impl HeadWeights {
    pub fn generate(cfg: &ModelConfig) -> Self {
        if cfg.source.is_file() {
            return crate::loader::sharded_loader::load_head(cfg);
        }
        let mut r = tensor_rng(cfg.seed, usize::MAX - 1, 7);
        let mut embed = Tensor2D::randn(cfg.vocab, cfg.d_model, 1.0, &mut r);
        let d = cfg.d_model;
        if cfg.context_prior > 0.0 {
            let u = context_direction(cfg);
            let s = cfg.context_bias * (d as f32).sqrt();
            for t in 0..cfg.vocab {
                for (e, uv) in embed.row_mut(t).iter_mut().zip(&u) {
                    *e += s * uv;
                }
            }
        }
        let mut lm_head = Linear::new_random(
            cfg.vocab,
            d,
            cfg.logit_std / (d as f32).sqrt(),
            &mut tensor_rng(cfg.seed, usize::MAX - 1, 8),
        );
        if cfg.transition_prior > 0.0 {
            // Token-transition prior: successor j of token t gets row
            // += boost·ê_t/d, so logit_j(rmsnorm(e_t)) rises by ≈ boost nats.
            // (use the token-specific part only: the shared û direction of
            // the contextual model must not be boosted for everyone)
            let u = if cfg.context_prior > 0.0 {
                context_direction(cfg)
            } else {
                vec![0.0; d]
            };
            let su = cfg.context_bias * (d as f32).sqrt();
            let rms_e = final_rms(cfg);
            for t in 0..cfg.vocab {
                let e: Vec<f32> = embed
                    .row(t)
                    .iter()
                    .zip(&u)
                    .map(|(x, uv)| x - su * uv)
                    .collect();
                let e = &e[..];
                let inv = rms_e / (crate::tensor::dot(e, e) / d as f32);
                for (k, wk) in SUCCESSOR_WEIGHTS.iter().enumerate() {
                    let j = successor(cfg, t as u32, k) as usize;
                    let a = cfg.transition_prior * wk * inv / d as f32;
                    for (w, ev) in lm_head.w.row_mut(j).iter_mut().zip(e) {
                        *w += a * ev;
                    }
                }
            }
        }
        if cfg.context_prior > 0.0 {
            // Context prior: the previous token p is present in the final
            // residual stream as c_p = W_o,h0 · W_v,g0 · rmsnorm(e_p) (copied by
            // the layer-0 previous-token head); successors of p get lm_head
            // rows aligned with c_p.
            let l0 = LayerWeights::generate(
                &ModelConfig {
                    moe: None,
                    d_ff: 32,
                    ..cfg.clone()
                },
                0,
                None,
            );
            let dh = cfg.d_head;
            let rms_h = final_rms(cfg);
            let mut xn = vec![0.0f32; d];
            let mut v = vec![0.0f32; dh];
            for p in 0..cfg.vocab {
                crate::tensor::rms_norm(embed.row(p), &l0.attn_norm, cfg.rms_eps, &mut xn);
                for (r, vr) in v.iter_mut().enumerate() {
                    *vr = crate::tensor::dot(l0.wv.w.row(r), &xn);
                }
                let c: Vec<f32> = (0..d)
                    .map(|i| crate::tensor::dot(&l0.wo.w.row(i)[..dh], &v))
                    .collect();
                let cc = crate::tensor::dot(&c, &c).max(1e-12);
                for (k, wk) in SUCCESSOR_WEIGHTS.iter().enumerate() {
                    let j = context_successor(cfg, p as u32, k) as usize;
                    let a = cfg.context_prior * wk * rms_h / cc;
                    for (w, cv) in lm_head.w.row_mut(j).iter_mut().zip(&c) {
                        *w += a * cv;
                    }
                }
            }
        }
        let final_norm = norm_weights(cfg, usize::MAX - 1, 9);
        if cfg.context_prior > 0.0 {
            // The shared direction û carries position for attention only: make
            // the LM head blind to it (orthogonal to û ⊙ final_norm), otherwise
            // every row picks up a large constant, context-independent logit.
            let u = context_direction(cfg);
            let ug: Vec<f32> = u.iter().zip(&final_norm).map(|(a, g)| a * g).collect();
            let uu = crate::tensor::dot(&ug, &ug);
            for j in 0..cfg.vocab {
                let row = lm_head.w.row_mut(j);
                let c = crate::tensor::dot(row, &ug) / uu;
                for (w, uv) in row.iter_mut().zip(&ug) {
                    *w -= c * uv;
                }
            }
        }
        HeadWeights {
            embed: embed.into(),
            final_norm,
            lm_head,
        }
    }

    pub fn embed_tokens(&self, tokens: &[u32]) -> Tensor2D {
        let d = self.embed.cols();
        let mut x = Tensor2D::zeros(tokens.len(), d);
        for (i, &t) in tokens.iter().enumerate() {
            x.row_mut(i).copy_from_slice(&self.embed.row(t as usize));
        }
        x
    }

    /// Final RMSNorm + LM head → logits [m, vocab].
    pub fn logits(&self, h: &Tensor2D, eps: f32) -> Tensor2D {
        let mut n = Tensor2D::zeros(h.rows, h.cols);
        for i in 0..h.rows {
            rms_norm(h.row(i), &self.final_norm, eps, n.row_mut(i));
        }
        self.lm_head.forward(&n)
    }
}

/// Observes layer outputs / routing decisions (lookahead predictors, PreWake).
pub trait LayerObserver {
    fn on_layer_input(&mut self, _layer: usize, _h: &Tensor2D) {}
    fn on_routing(&mut self, _layer: usize, _selected: &[Vec<usize>]) {}
}

pub struct NoObserver;
impl LayerObserver for NoObserver {}

/// A contiguous range of transformer layers executed by one shard.
pub struct ShardModel {
    pub cfg: ModelConfig,
    pub layers: Vec<LayerWeights>,
    pub rope: RopeTable,
    /// Use `CanonicalBoundaryAnchor` for the first RMSNorm on this shard.
    pub deterministic_boundary: bool,
    pub anchor: CanonicalBoundaryAnchor,
    /// Approximate anchor-consensus expert coalescing for multi-row batches.
    pub coalesce_experts: bool,
    /// Attention-only layers (FFN skipped): the cheap self-speculative drafter.
    pub skip_ffn: bool,
}

impl ShardModel {
    pub fn generate(
        cfg: &ModelConfig,
        layers: std::ops::Range<usize>,
        staged_capacity: Option<usize>,
    ) -> Self {
        ShardModel {
            cfg: cfg.clone(),
            layers: layers
                .map(|l| LayerWeights::generate(cfg, l, staged_capacity))
                .collect(),
            rope: cfg.rope_table(),
            deterministic_boundary: false,
            anchor: CanonicalBoundaryAnchor::default(),
            coalesce_experts: false,
            skip_ffn: false,
        }
    }

    /// Hint the OS to page in this shard's file-backed weights (no-op for
    /// synthetic weights).
    pub fn prefetch(&self) {
        for l in &self.layers {
            l.prefetch();
        }
    }

    pub fn layer_range(&self) -> std::ops::Range<usize> {
        match (self.layers.first(), self.layers.last()) {
            (Some(a), Some(b)) => a.index..b.index + 1,
            _ => 0..0,
        }
    }

    /// Run all owned layers over a batch of rows (tree nodes or a single token).
    pub fn forward(
        &self,
        x: &mut Tensor2D,
        meta: &BatchMeta,
        attn: &mut dyn AttentionBackend,
        obs: &mut dyn LayerObserver,
    ) {
        for (li, layer) in self.layers.iter().enumerate() {
            obs.on_layer_input(layer.index, x);
            let det = self.deterministic_boundary && li == 0;
            self.layer_forward(layer, x, meta, attn, obs, det);
        }
    }

    pub fn layer_forward(
        &self,
        layer: &LayerWeights,
        x: &mut Tensor2D,
        meta: &BatchMeta,
        attn: &mut dyn AttentionBackend,
        obs: &mut dyn LayerObserver,
        deterministic: bool,
    ) {
        let cfg = &self.cfg;
        let m = x.rows;
        let mut xn = Tensor2D::zeros(m, cfg.d_model);
        for i in 0..m {
            if deterministic {
                self.anchor
                    .rms_norm(x.row(i), &layer.attn_norm, cfg.rms_eps, xn.row_mut(i));
            } else {
                rms_norm(x.row(i), &layer.attn_norm, cfg.rms_eps, xn.row_mut(i));
            }
        }
        let (q, k, v) = project_qkv(
            &layer.wq,
            &layer.wk,
            &layer.wv,
            &layer.extras,
            &xn,
            &self.rope,
            &meta.positions,
            cfg.d_head,
            cfg.rms_eps,
        );
        let a = attn.attend(layer.index, &q, &k, &v, meta);
        let o = layer.wo.forward(&a);
        for (xi, oi) in x.data.iter_mut().zip(&o.data) {
            *xi += oi;
        }
        if self.skip_ffn {
            return;
        }
        // FFN
        for i in 0..m {
            rms_norm(x.row(i), &layer.ffn_norm, cfg.rms_eps, xn.row_mut(i));
        }
        let f = if self.coalesce_experts && xn.rows > 1 {
            match &layer.ffn {
                FfnWeights::Moe {
                    router,
                    bank,
                    top_k,
                } => {
                    let routes = route_coalesced(router, &xn, *top_k);
                    obs.on_routing(
                        layer.index,
                        &routes.iter().map(|r| r.0.clone()).collect::<Vec<_>>(),
                    );
                    moe_combine(bank, &xn, &routes)
                }
                _ => ffn_forward(&layer.ffn, &xn, layer.index, obs),
            }
        } else {
            ffn_forward(&layer.ffn, &xn, layer.index, obs)
        };
        for (xi, fi) in x.data.iter_mut().zip(&f.data) {
            *xi += fi;
        }
    }
}

/// Top-k routing: returns per-row (expert ids, renormalised gate weights).
pub fn route_topk(router: &Linear, xn: &Tensor2D, top_k: usize) -> Vec<(Vec<usize>, Vec<f32>)> {
    let logits = router.forward(xn);
    (0..xn.rows)
        .map(|i| {
            let row = logits.row(i);
            let mut idx: Vec<usize> = (0..row.len()).collect();
            idx.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap().then(a.cmp(&b)));
            idx.truncate(top_k);
            let m = row[idx[0]];
            let w: Vec<f32> = idx.iter().map(|&e| (row[e] - m).exp()).collect();
            let s: f32 = w.iter().sum();
            (idx, w.into_iter().map(|v| v / s).collect())
        })
        .collect()
}

/// FFN / MoE forward over a batch. MoE rows are grouped per expert so each
/// expert runs one batched GEMM.
pub fn ffn_forward(
    ffn: &FfnWeights,
    xn: &Tensor2D,
    layer: usize,
    obs: &mut dyn LayerObserver,
) -> Tensor2D {
    match ffn {
        FfnWeights::Dense(e) => e.forward(xn),
        FfnWeights::Moe {
            router,
            bank,
            top_k,
        } => {
            let routes = route_topk(router, xn, *top_k);
            let selected: Vec<Vec<usize>> = routes.iter().map(|r| r.0.clone()).collect();
            obs.on_routing(layer, &selected);
            moe_combine(bank, xn, &routes)
        }
    }
}

/// Anchor-consensus expert coalescing (APPROXIMATE). Row 0 (the verified
/// root/anchor) keeps its exact top-k experts; every other row is restricted
/// to the anchor's experts plus at most one consensus expert (the most common
/// top-1 choice among the other rows), with gates re-normalised over its top-k
/// within that set. All rows then hit the same ≤ k+1 experts, so each expert
/// runs one batched GEMM. This changes the candidate rows' FFN outputs, so a
/// tree verified this way is NOT exactly distributed as the target model; the
/// deviation is measured by the benchmark and tests.
pub fn route_coalesced(
    router: &Linear,
    xn: &Tensor2D,
    top_k: usize,
) -> Vec<(Vec<usize>, Vec<f32>)> {
    let exact = route_topk(router, xn, top_k);
    if xn.rows <= 1 {
        return exact;
    }
    let anchor = exact[0].0.clone();
    let mut votes: std::collections::BTreeMap<usize, usize> = Default::default();
    for r in &exact[1..] {
        if !anchor.contains(&r.0[0]) {
            *votes.entry(r.0[0]).or_insert(0) += 1;
        }
    }
    let mut allowed = anchor.clone();
    if let Some((&e, _)) = votes
        .iter()
        .max_by_key(|(e, c)| (**c, std::cmp::Reverse(**e)))
    {
        allowed.push(e);
    }
    let logits = router.forward(xn);
    let mut out = vec![exact[0].clone()];
    for i in 1..xn.rows {
        let row = logits.row(i);
        let mut idx: Vec<usize> = allowed.clone();
        idx.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap().then(a.cmp(&b)));
        idx.truncate(top_k);
        let m = row[idx[0]];
        let w: Vec<f32> = idx.iter().map(|&e| (row[e] - m).exp()).collect();
        let sum: f32 = w.iter().sum();
        out.push((idx, w.into_iter().map(|v| v / sum).collect()));
    }
    out
}

pub fn moe_combine(
    bank: &ExpertBank,
    xn: &Tensor2D,
    routes: &[(Vec<usize>, Vec<f32>)],
) -> Tensor2D {
    // node threads with a lane pool (installed identically for every engine)
    // dispatch the active experts' stripes on it
    if let Some(pool) = crate::tensor::pool::current() {
        return moe_combine_lanes(bank, xn, routes, &pool);
    }
    let d = xn.cols;
    let mut out = Tensor2D::zeros(xn.rows, d);
    let mut per_expert: std::collections::BTreeMap<usize, Vec<(usize, f32)>> = Default::default();
    for (i, (ids, ws)) in routes.iter().enumerate() {
        for (e, w) in ids.iter().zip(ws) {
            per_expert.entry(*e).or_default().push((i, *w));
        }
    }
    for (e, rows) in per_expert {
        let expert = bank.acquire(e);
        let mut xb = Tensor2D::zeros(rows.len(), d);
        for (j, (i, _)) in rows.iter().enumerate() {
            xb.row_mut(j).copy_from_slice(xn.row(*i));
        }
        let yb = expert.forward(&xb);
        for (j, (i, w)) in rows.iter().enumerate() {
            let dst = out.row_mut(*i);
            for (o, y) in dst.iter_mut().zip(yb.row(j)) {
                *o += w * y;
            }
        }
    }
    out
}

/// `moe_combine` on an intra-node lane pool. Every active expert is split
/// into ⌈lanes / active experts⌉ stripes — of hidden units for the gate/up
/// projections, then (after a barrier) of output features for the down
/// projection — so a single-row step and a multi-row tree use the node's
/// lanes by the *same* rule: a 1-row step's 2 experts are striped just as
/// finely as a 4-row tree's 5–6. Each output element is the same dot product
/// as in `moe_combine`, so results agree to f32 rounding.
pub fn moe_combine_lanes(
    bank: &ExpertBank,
    xn: &Tensor2D,
    routes: &[(Vec<usize>, Vec<f32>)],
    pool: &crate::tensor::pool::LanePool,
) -> Tensor2D {
    use crate::tensor::pool::SyncPtr;
    let d = xn.cols;
    let mut per_expert: std::collections::BTreeMap<usize, Vec<(usize, f32)>> = Default::default();
    for (i, (ids, ws)) in routes.iter().enumerate() {
        for (e, w) in ids.iter().zip(ws) {
            per_expert.entry(*e).or_default().push((i, *w));
        }
    }
    // (expert, its (row, gate weight) pairs, its gathered input rows)
    type Group = (Arc<Expert>, Vec<(usize, f32)>, Tensor2D);
    let groups: Vec<Group> = per_expert
        .into_iter()
        .map(|(e, rows)| {
            let mut xb = Tensor2D::zeros(rows.len(), d);
            for (j, (i, _)) in rows.iter().enumerate() {
                xb.row_mut(j).copy_from_slice(xn.row(*i));
            }
            (bank.acquire(e), rows, xb)
        })
        .collect();
    let ne = groups.len();
    let mut out = Tensor2D::zeros(xn.rows, d);
    if ne == 0 {
        return out;
    }
    let s = pool.lanes().div_ceil(ne).max(1);
    let stripe = |len: usize, st: usize| (len * st / s, len * (st + 1) / s);
    // phase 1: hidden activations silu(x·Wgᵀ) ⊙ (x·Wuᵀ), striped over hidden units
    let mut hs: Vec<Tensor2D> = groups
        .iter()
        .map(|(ex, rows, _)| Tensor2D::zeros(rows.len(), ex.w_gate.w.rows))
        .collect();
    let hp: Vec<SyncPtr> = hs
        .iter_mut()
        .map(|h| SyncPtr(h.data.as_mut_ptr()))
        .collect();
    pool.run(ne * s, &|t| {
        let (e, st) = (t / s, t % s);
        let (ex, rows, xb) = &groups[e];
        let dh = ex.w_gate.w.rows;
        let (h0, h1) = stripe(dh, st);
        let nr = h1 - h0;
        if nr == 0 {
            return;
        }
        let mut g = vec![0.0f32; rows.len() * nr];
        let mut u = vec![0.0f32; rows.len() * nr];
        ex.w_gate.forward_rows(xb, h0, h1, &mut g);
        ex.w_up.forward_rows(xb, h0, h1, &mut u);
        for r in 0..rows.len() {
            for j in 0..nr {
                let v = silu(g[r * nr + j]) * u[r * nr + j];
                // SAFETY: stripes of different tasks cover disjoint columns.
                unsafe { *hp[e].0.add(r * dh + h0 + j) = v };
            }
        }
    });
    // phase 2: down projection, striped over output features
    let mut ys: Vec<Tensor2D> = groups
        .iter()
        .map(|(_, rows, _)| Tensor2D::zeros(rows.len(), d))
        .collect();
    let yp: Vec<SyncPtr> = ys
        .iter_mut()
        .map(|y| SyncPtr(y.data.as_mut_ptr()))
        .collect();
    pool.run(ne * s, &|t| {
        let (e, st) = (t / s, t % s);
        let (ex, rows, _) = &groups[e];
        let (o0, o1) = stripe(d, st);
        let nr = o1 - o0;
        if nr == 0 {
            return;
        }
        let mut y = vec![0.0f32; rows.len() * nr];
        ex.w_down.forward_rows(&hs[e], o0, o1, &mut y);
        for r in 0..rows.len() {
            for j in 0..nr {
                // SAFETY: stripes of different tasks cover disjoint columns.
                unsafe { *yp[e].0.add(r * d + o0 + j) = y[r * nr + j] };
            }
        }
    });
    for ((_, rows, _), yb) in groups.iter().zip(&ys) {
        for (j, (i, w)) in rows.iter().enumerate() {
            for (o, y) in out.row_mut(*i).iter_mut().zip(yb.row(j)) {
                *o += w * y;
            }
        }
    }
    out
}

/// Full monolithic FP32 reference model (all layers, all experts resident).
pub struct ReferenceModel {
    pub head: HeadWeights,
    pub shard: ShardModel,
}

impl ReferenceModel {
    pub fn new(cfg: &ModelConfig) -> Self {
        ReferenceModel {
            head: HeadWeights::generate(cfg),
            shard: ShardModel::generate(cfg, 0..cfg.n_layers, None),
        }
    }

    /// Teacher-forced forward over a full token sequence; returns logits for
    /// every position [T, vocab].
    pub fn forward_sequence(&self, tokens: &[u32]) -> Tensor2D {
        let cfg = &self.shard.cfg;
        let mut kv = crate::kv_fabric::LocalKvCache::new(cfg.n_heads, cfg.n_kv_heads, cfg.d_head)
            .with_exact(cfg.exact);
        let mut x = self.head.embed_tokens(tokens);
        let meta = BatchMeta::causal_prefill(0, tokens.len(), 0);
        self.shard.forward(&mut x, &meta, &mut kv, &mut NoObserver);
        self.head.logits(&x, cfg.rms_eps)
    }
}

/// Sample from logits with temperature (0 ⇒ greedy). Returns token and probs.
pub fn probs_from_logits(logits: &[f32], temperature: f32) -> Vec<f32> {
    let mut p: Vec<f32> = if temperature <= 0.0 {
        let mut best = 0;
        for i in 1..logits.len() {
            if logits[i] > logits[best] {
                best = i;
            }
        }
        let mut v = vec![0.0; logits.len()];
        v[best] = 1.0;
        return v;
    } else {
        logits.iter().map(|l| l / temperature).collect()
    };
    crate::tensor::softmax_inplace(&mut p);
    p
}

pub fn sample(p: &[f32], rng: &mut Rng) -> u32 {
    let r = rng.uniform() * p.iter().sum::<f32>();
    let mut c = 0.0;
    for (i, &v) in p.iter().enumerate() {
        c += v;
        if r < c {
            return i as u32;
        }
    }
    (p.len() - 1) as u32
}

/// Centered cosine between two logit vectors (removes the shared offset).
pub fn logit_fidelity(a: &[f32], b: &[f32]) -> f64 {
    let ma = a.iter().sum::<f32>() / a.len() as f32;
    let mb = b.iter().sum::<f32>() / b.len() as f32;
    let ca: Vec<f32> = a.iter().map(|v| v - ma).collect();
    let cb: Vec<f32> = b.iter().map(|v| v - mb).collect();
    crate::tensor::cosine(&ca, &cb)
}

pub fn argmax(x: &[f32]) -> usize {
    let mut b = 0;
    for i in 1..x.len() {
        if x[i] > x[b] {
            b = i;
        }
    }
    b
}

#[cfg(test)]
mod tests {
    use super::hidden_split;

    #[test]
    fn hidden_split_covers_every_unit_for_any_world() {
        for d in [96usize, 128, 176, 704] {
            for world in 1..=7 {
                for align in [1usize, 32] {
                    let mut next = 0;
                    for r in 0..world {
                        let (s, e) = hidden_split(d, r, world, align);
                        assert_eq!(s, next, "gap or overlap: d {d} world {world} align {align}");
                        assert!(e >= s);
                        if align > 1 && d % align == 0 && d / align >= world {
                            assert!(s % align == 0 && e % align == 0);
                        }
                        next = e;
                    }
                    assert_eq!(next, d, "units dropped: d {d} world {world} align {align}");
                }
            }
        }
        // f32 weights keep the old d / world stripes when world divides d
        assert_eq!(hidden_split(704, 2, 4, 1), (352, 528));
    }
}
