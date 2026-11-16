//! Shard-selective model loading from GGUF / SafeTensors files.
//!
//! A node asks only for the layers (and, under H-TEP, the expert hidden
//! slices) it owns: every matrix comes back as a zero-copy [`QuantView`] into
//! the shared mapping, so nothing is read from disk until a kernel touches
//! it, and a node never faults in pages of layers it does not run. Small
//! tensors (norm vectors, MoE routers, the embedding table) are decoded to
//! f32; F32 matrices are copied into the engine's dense tiled layout; Q/K of
//! GGUF files are re-ordered once (llama.cpp stores them permuted for its
//! interleaved RoPE, the engine uses the Hugging Face half-split layout).
//!
//! Supported layouts: GGUF `general.architecture = "llama"` (dense Llama and
//! Mixtral-style MoE, stacked `ffn_*_exps` or per-expert tensors) and
//! Hugging Face Llama / Mixtral SafeTensors (`config.json`, single file or
//! `model.safetensors.index.json` shards). GGML types: F32, F16, BF16, Q8_0,
//! Q4_0 (K-quants are rejected with an error).

use super::gguf::{GgufFile, GgufValue, GgufWriter};
use super::safetensors::{Dtype, SafeTensorsFile, SafeTensorsWriter};
use crate::model::{
    Expert, ExpertBank, FfnWeights, HeadWeights, LayerWeights, ModelConfig, MoeConfig, WeightSource,
};
use crate::server::json::Json;
use crate::tensor::quant::{encode_matrix, QuantView, WeightFormat};
use crate::tensor::{Linear, Tensor2D};
use crate::tokenizer::Tokenizer;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, Weak};

#[derive(Debug)]
pub struct LoadError(pub String);
impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "model load: {}", self.0)
    }
}
impl std::error::Error for LoadError {}

fn lerr<E: std::fmt::Display>(e: E) -> LoadError {
    LoadError(e.to_string())
}

/// Hugging Face SafeTensors model (one or more shard files + config.json).
pub struct StModel {
    pub dir: PathBuf,
    pub files: Vec<SafeTensorsFile>,
    index: HashMap<String, usize>,
    pub config: Json,
}

pub enum ModelFile {
    Gguf(GgufFile),
    SafeTensors(StModel),
}

/// Logical tensors of a decoder layer / the model head.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum T {
    Q,
    K,
    V,
    O,
    Gate,
    Up,
    Down,
    AttnNorm,
    FfnNorm,
    Router,
}

fn gguf_name(t: T, l: usize) -> String {
    let n = match t {
        T::Q => "attn_q",
        T::K => "attn_k",
        T::V => "attn_v",
        T::O => "attn_output",
        T::Gate => "ffn_gate",
        T::Up => "ffn_up",
        T::Down => "ffn_down",
        T::AttnNorm => "attn_norm",
        T::FfnNorm => "ffn_norm",
        T::Router => "ffn_gate_inp",
    };
    format!("blk.{l}.{n}.weight")
}

fn hf_name(t: T, l: usize, expert: Option<usize>) -> String {
    let p = format!("model.layers.{l}");
    match (t, expert) {
        (T::Q, _) => format!("{p}.self_attn.q_proj.weight"),
        (T::K, _) => format!("{p}.self_attn.k_proj.weight"),
        (T::V, _) => format!("{p}.self_attn.v_proj.weight"),
        (T::O, _) => format!("{p}.self_attn.o_proj.weight"),
        (T::AttnNorm, _) => format!("{p}.input_layernorm.weight"),
        (T::FfnNorm, _) => format!("{p}.post_attention_layernorm.weight"),
        (T::Router, _) => format!("{p}.block_sparse_moe.gate.weight"),
        (T::Gate, Some(e)) => format!("{p}.block_sparse_moe.experts.{e}.w1.weight"),
        (T::Down, Some(e)) => format!("{p}.block_sparse_moe.experts.{e}.w2.weight"),
        (T::Up, Some(e)) => format!("{p}.block_sparse_moe.experts.{e}.w3.weight"),
        (T::Gate, None) => format!("{p}.mlp.gate_proj.weight"),
        (T::Up, None) => format!("{p}.mlp.up_proj.weight"),
        (T::Down, None) => format!("{p}.mlp.down_proj.weight"),
    }
}

impl ModelFile {
    pub fn open(path: &str) -> Result<Self, LoadError> {
        let p = Path::new(path);
        let is_gguf = p.is_file() && {
            let mut m = [0u8; 4];
            std::io::Read::read_exact(&mut std::fs::File::open(p).map_err(lerr)?, &mut m).is_ok()
                && &m == b"GGUF"
        };
        if is_gguf {
            return Ok(ModelFile::Gguf(GgufFile::open(p).map_err(lerr)?));
        }
        let (dir, files) = if p.is_dir() {
            let idx = p.join("model.safetensors.index.json");
            let files = if idx.exists() {
                let j = Json::parse(&std::fs::read_to_string(&idx).map_err(lerr)?).map_err(lerr)?;
                let mut names: Vec<String> = j
                    .get("weight_map")
                    .and_then(|w| w.as_object())
                    .unwrap_or(&[])
                    .iter()
                    .filter_map(|(_, f)| f.as_str().map(String::from))
                    .collect();
                names.sort();
                names.dedup();
                names.into_iter().map(|n| p.join(n)).collect()
            } else {
                vec![p.join("model.safetensors")]
            };
            (p.to_path_buf(), files)
        } else {
            (
                p.parent().unwrap_or(Path::new(".")).to_path_buf(),
                vec![p.to_path_buf()],
            )
        };
        let files: Vec<SafeTensorsFile> = files
            .iter()
            .map(SafeTensorsFile::open)
            .collect::<Result<_, _>>()
            .map_err(lerr)?;
        let mut index = HashMap::new();
        for (i, f) in files.iter().enumerate() {
            for t in &f.tensors {
                index.insert(t.name.clone(), i);
            }
        }
        let cfg_path = dir.join("config.json");
        let config = Json::parse(
            &std::fs::read_to_string(&cfg_path)
                .map_err(|e| LoadError(format!("{}: {e}", cfg_path.display())))?,
        )
        .map_err(lerr)?;
        Ok(ModelFile::SafeTensors(StModel {
            dir,
            files,
            index,
            config,
        }))
    }

    fn st_view(m: &StModel, name: &str) -> Result<QuantView, LoadError> {
        let i = *m
            .index
            .get(name)
            .ok_or_else(|| LoadError(format!("missing tensor {name}")))?;
        m.files[i].view(name, None).map_err(lerr)
    }

    /// llama.cpp permutes Q/K rows of llama-family checkpoints for its
    /// interleaved RoPE; Qwen GGUFs keep the Hugging Face (half-split) order.
    fn permuted_qk(&self) -> bool {
        match self {
            ModelFile::Gguf(g) => matches!(g.architecture(), Some("llama") | Some("mistral")),
            ModelFile::SafeTensors(_) => false,
        }
    }

    /// Any float tensor by name, decoded to f32 (small tensors).
    fn named_vector(&self, name: &str) -> Result<Vec<f32>, LoadError> {
        match self {
            ModelFile::Gguf(g) => g.vector_f32(name).map_err(lerr),
            ModelFile::SafeTensors(m) => Ok(Self::st_view(m, name)?.dequantize().data),
        }
    }

    fn has(&self, name: &str) -> bool {
        match self {
            ModelFile::Gguf(g) => g.tensor(name).is_some(),
            ModelFile::SafeTensors(m) => m.index.contains_key(name),
        }
    }

    /// Zero-copy view of layer tensor `t` (expert `e` for MoE FFN weights).
    fn layer_view(&self, t: T, l: usize, e: Option<usize>) -> Result<QuantView, LoadError> {
        match self {
            ModelFile::Gguf(g) => match e {
                Some(e) => {
                    let stacked = gguf_name(t, l).replace(".weight", "_exps.weight");
                    if g.tensor(&stacked).is_some() {
                        g.view(&stacked, Some(e)).map_err(lerr)
                    } else {
                        let legacy = gguf_name(t, l).replace(".weight", &format!(".{e}.weight"));
                        g.view(&legacy, None).map_err(lerr)
                    }
                }
                None => g.view(&gguf_name(t, l), None).map_err(lerr),
            },
            ModelFile::SafeTensors(m) => Self::st_view(m, &hf_name(t, l, e)),
        }
    }

    fn head_view(&self, which: &str) -> Result<QuantView, LoadError> {
        match self {
            ModelFile::Gguf(g) => match which {
                "embed" => g.view("token_embd.weight", None),
                "norm" => g.view("output_norm.weight", None),
                _ => {
                    if g.tensor("output.weight").is_some() {
                        g.view("output.weight", None)
                    } else {
                        g.view("token_embd.weight", None)
                    }
                }
            }
            .map_err(lerr),
            ModelFile::SafeTensors(m) => match which {
                "embed" => Self::st_view(m, "model.embed_tokens.weight"),
                "norm" => Self::st_view(m, "model.norm.weight"),
                _ => {
                    let tied = m
                        .config
                        .get("tie_word_embeddings")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    if !tied && m.index.contains_key("lm_head.weight") {
                        Self::st_view(m, "lm_head.weight")
                    } else {
                        Self::st_view(m, "model.embed_tokens.weight")
                    }
                }
            },
        }
    }

    /// The model's hyper-parameters as an engine config.
    pub fn config(
        &self,
        path: &str,
        requant: Option<WeightFormat>,
    ) -> Result<ModelConfig, LoadError> {
        let src = WeightSource::File {
            path: path.to_string(),
            requant,
        };
        let base = ModelConfig {
            logit_std: 0.0,
            transition_prior: 0.0,
            spectral_decay: 0.0,
            context_prior: 0.0,
            context_bias: 0.0,
            seed: 0,
            source: src,
            exact: true,
            ..ModelConfig::tiny()
        };
        match self {
            ModelFile::Gguf(g) => {
                let arch = g.architecture().unwrap_or("?");
                if !SUPPORTED_ARCHS.contains(&arch) {
                    return Err(LoadError(format!(
                        "GGUF architecture {arch:?} is not supported (supported: {})",
                        SUPPORTED_ARCHS.join(", ")
                    )));
                }
                let need = |k: &str| {
                    g.arch_u64(k)
                        .ok_or_else(|| LoadError(format!("GGUF lacks {arch}.{k}")))
                };
                let d = need("embedding_length")? as usize;
                let n_heads = need("attention.head_count")? as usize;
                let n_kv = g
                    .arch_u64("attention.head_count_kv")
                    .unwrap_or(n_heads as u64) as usize;
                let d_head = g
                    .arch_u64("attention.key_length")
                    .map(|v| v as usize)
                    .unwrap_or(d / n_heads);
                let ff = need("feed_forward_length")? as usize;
                let n_exp = g.arch_u64("expert_count").unwrap_or(0) as usize;
                let vocab = g
                    .arch_u64("vocab_size")
                    .or_else(|| {
                        g.get("tokenizer.ggml.tokens")
                            .and_then(|v| v.as_array())
                            .map(|a| a.len() as u64)
                    })
                    .or_else(|| g.tensor("token_embd.weight").map(|t| t.dims[1]))
                    .ok_or_else(|| LoadError("cannot determine vocabulary size".into()))?
                    as usize;
                Ok(ModelConfig {
                    vocab,
                    d_model: d,
                    n_layers: need("block_count")? as usize,
                    n_heads,
                    n_kv_heads: n_kv,
                    d_head,
                    d_ff: ff,
                    moe: (n_exp > 0).then(|| MoeConfig {
                        n_experts: n_exp,
                        top_k: g.arch_u64("expert_used_count").unwrap_or(2) as usize,
                        d_expert: g
                            .arch_u64("expert_feed_forward_length")
                            .unwrap_or(ff as u64) as usize,
                    }),
                    rope_theta: g.arch_f64("rope.freq_base").unwrap_or(10_000.0) as f32,
                    rms_eps: g
                        .arch_f64("attention.layer_norm_rms_epsilon")
                        .unwrap_or(1e-5) as f32,
                    seed: g.get_u64("tesseract.synthetic_seed").unwrap_or(0),
                    rope_scaling: crate::tensor::RopeScaling {
                        freq_factors: if g.tensor("rope_freqs.weight").is_some() {
                            g.vector_f32("rope_freqs.weight").map_err(lerr)?
                        } else {
                            Vec::new()
                        },
                        linear: match g.get_str(&format!("{arch}.rope.scaling.type")) {
                            Some("linear") => {
                                g.arch_f64("rope.scaling.factor").unwrap_or(1.0) as f32
                            }
                            _ => 0.0,
                        },
                        yarn: match g.get_str(&format!("{arch}.rope.scaling.type")) {
                            Some("yarn") => Some((
                                g.arch_f64("rope.scaling.factor").unwrap_or(1.0) as f32,
                                g.arch_u64("rope.scaling.original_context_length")
                                    .unwrap_or(4096) as u32,
                            )),
                            _ => None,
                        },
                    },
                    ..base
                })
            }
            ModelFile::SafeTensors(m) => {
                let c = &m.config;
                let u = |k: &str| c.get(k).and_then(|v| v.as_u64()).map(|v| v as usize);
                let need =
                    |k: &str| u(k).ok_or_else(|| LoadError(format!("config.json lacks {k}")));
                let d = need("hidden_size")?;
                let n_heads = need("num_attention_heads")?;
                let n_exp = u("num_local_experts").unwrap_or(0);
                let ff = need("intermediate_size")?;
                Ok(ModelConfig {
                    vocab: need("vocab_size")?,
                    d_model: d,
                    n_layers: need("num_hidden_layers")?,
                    n_heads,
                    n_kv_heads: u("num_key_value_heads").unwrap_or(n_heads),
                    d_head: u("head_dim").unwrap_or(d / n_heads),
                    d_ff: ff,
                    moe: (n_exp > 0).then(|| MoeConfig {
                        n_experts: n_exp,
                        top_k: u("num_experts_per_tok").unwrap_or(2),
                        d_expert: ff,
                    }),
                    rope_theta: c
                        .get("rope_theta")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(10_000.0) as f32,
                    rms_eps: c
                        .get("rms_norm_eps")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(1e-5) as f32,
                    seed: u("tesseract_synthetic_seed").unwrap_or(0) as u64,
                    rope_scaling: hf_rope_scaling(
                        c.get("rope_scaling"),
                        u("head_dim").unwrap_or(d / n_heads),
                        c.get("rope_theta")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(10_000.0) as f32,
                    ),
                    ..base
                })
            }
        }
    }

    /// The tokenizer stored with the model: `Ok(None)` when the file has
    /// none, `Err` when it has one this engine cannot read.
    pub fn tokenizer(&self) -> Result<Option<Tokenizer>, LoadError> {
        match self {
            ModelFile::Gguf(g) => {
                if g.get("tokenizer.ggml.tokens").is_none() {
                    return Ok(None);
                }
                Tokenizer::from_gguf(g).map(Some).map_err(lerr)
            }
            ModelFile::SafeTensors(m) => {
                let Ok(s) = std::fs::read_to_string(m.dir.join("tokenizer.json")) else {
                    return Ok(None);
                };
                Tokenizer::from_hf_json(&Json::parse(&s).map_err(lerr)?)
                    .map(Some)
                    .map_err(lerr)
            }
        }
    }

    pub fn file_bytes(&self) -> u64 {
        match self {
            ModelFile::Gguf(g) => g.map.len() as u64,
            ModelFile::SafeTensors(m) => m.files.iter().map(|f| f.map.len() as u64).sum(),
        }
    }

    pub fn format_name(&self) -> &'static str {
        match self {
            ModelFile::Gguf(_) => "GGUF",
            ModelFile::SafeTensors(_) => "SafeTensors",
        }
    }
}

/// `general.architecture` values the loader maps onto the engine's decoder.
pub const SUPPORTED_ARCHS: [&str; 4] = ["llama", "mistral", "qwen2", "qwen3"];

/// Hugging Face `rope_scaling` → engine RoPE scaling (llama3 factors are
/// expanded into per-frequency divisors exactly as `transformers` does).
fn hf_rope_scaling(rs: Option<&Json>, d_head: usize, theta: f32) -> crate::tensor::RopeScaling {
    let mut out = crate::tensor::RopeScaling::default();
    let Some(rs) = rs.filter(|r| r.as_object().is_some()) else {
        return out;
    };
    let f = |k: &str, dflt: f64| rs.get(k).and_then(|v| v.as_f64()).unwrap_or(dflt) as f32;
    let kind = rs
        .get("rope_type")
        .or_else(|| rs.get("type"))
        .and_then(|t| t.as_str())
        .unwrap_or("default");
    match kind {
        "linear" => out.linear = f("factor", 1.0),
        "yarn" => {
            out.yarn = Some((
                f("factor", 1.0),
                f("original_max_position_embeddings", 4096.0) as u32,
            ))
        }
        "llama3" => {
            let (factor, lo, hi, orig) = (
                f("factor", 8.0),
                f("low_freq_factor", 1.0),
                f("high_freq_factor", 4.0),
                f("original_max_position_embeddings", 8192.0),
            );
            out.freq_factors = (0..d_head / 2)
                .map(|i| {
                    let freq = 1.0 / theta.powf((2 * i) as f32 / d_head as f32);
                    let wavelen = 2.0 * std::f32::consts::PI / freq;
                    if wavelen < orig / hi {
                        1.0
                    } else if wavelen > orig / lo {
                        factor
                    } else {
                        let smooth = (orig / wavelen - lo) / (hi - lo);
                        freq / ((1.0 - smooth) * freq / factor + smooth * freq)
                    }
                })
                .collect();
        }
        _ => {}
    }
    out
}

// --- process-wide file cache ----------------------------------------------------

type Cache = Mutex<HashMap<String, Weak<ModelFile>>>;
static CACHE: OnceLock<Cache> = OnceLock::new();
static PATH_OVERRIDE: OnceLock<Mutex<Option<String>>> = OnceLock::new();

/// Make every file-backed assignment on this process read `path` instead of
/// the coordinator's path (a node daemon's local copy of the model).
pub fn set_local_model_path(path: Option<String>) {
    *PATH_OVERRIDE
        .get_or_init(|| Mutex::new(None))
        .lock()
        .unwrap() = path;
}

fn resolve(path: &str) -> String {
    PATH_OVERRIDE
        .get()
        .and_then(|m| m.lock().unwrap().clone())
        .unwrap_or_else(|| path.to_string())
}

/// Open (or reuse) the mapped model file at `path`.
pub fn open(path: &str) -> Result<Arc<ModelFile>, LoadError> {
    let path = resolve(path);
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut g = cache.lock().unwrap();
    if let Some(f) = g.get(&path).and_then(|w| w.upgrade()) {
        return Ok(f);
    }
    let f = Arc::new(ModelFile::open(&path)?);
    g.insert(path, Arc::downgrade(&f));
    Ok(f)
}

/// Engine config for the model file at `path`.
pub fn config_from_file(
    path: &str,
    requant: Option<WeightFormat>,
) -> Result<ModelConfig, LoadError> {
    open(path)?.config(path, requant)
}

fn file_of(cfg: &ModelConfig) -> (Arc<ModelFile>, Option<WeightFormat>) {
    match &cfg.source {
        WeightSource::File { path, requant } => {
            (open(path).unwrap_or_else(|e| panic!("{e}")), *requant)
        }
        WeightSource::Synthetic => panic!("synthetic model has no file"),
    }
}

/// Wrap a view as an engine matrix: quantized / half formats stay zero-copy
/// views; F32 is copied into the dense tiled layout; `requant` re-encodes.
fn linear(v: QuantView, requant: Option<WeightFormat>) -> Linear {
    match requant {
        Some(WeightFormat::F32) => Linear::new(v.dequantize()),
        Some(f) if f != v.fmt => {
            let dense = v.dequantize();
            if f.is_block() && !dense.cols.is_multiple_of(f.block_len()) {
                return Linear::new(dense);
            }
            Linear::quantized(QuantView::from_f32(&dense, f))
        }
        _ if v.fmt == WeightFormat::F32 => Linear::new(v.dequantize()),
        _ => Linear::quantized(v),
    }
}

/// Re-order llama.cpp-permuted Q/K rows into the half-split RoPE layout.
fn unpermute_rows(v: &QuantView, n_head: usize) -> QuantView {
    let dh = v.rows / n_head;
    let half = dh / 2;
    let mut out = Vec::with_capacity(v.bytes());
    for h in 0..n_head {
        for a in 0..2 {
            for b in 0..half {
                out.extend_from_slice(v.row_bytes(h * dh + 2 * b + a));
            }
        }
    }
    QuantView::new(Arc::new(out), 0, v.fmt, v.rows, v.cols)
}

fn vec_f32(v: &QuantView) -> Vec<f32> {
    v.dequantize().data
}

/// Build decoder layer `l` from the model file (zero-copy matrix views).
pub fn load_layer(cfg: &ModelConfig, l: usize) -> LayerWeights {
    let (f, rq) = file_of(cfg);
    let get = |t: T, e: Option<usize>| f.layer_view(t, l, e).unwrap_or_else(|err| panic!("{err}"));
    let (q, k) = if f.permuted_qk() {
        (
            unpermute_rows(&get(T::Q, None), cfg.n_heads),
            unpermute_rows(&get(T::K, None), cfg.n_kv_heads),
        )
    } else {
        (get(T::Q, None), get(T::K, None))
    };
    let expert = |e: Option<usize>| Expert {
        w_gate: linear(get(T::Gate, e), rq),
        w_up: linear(get(T::Up, e), rq),
        // exact accumulation only pays off for dense FFNs (a MoE combine
        // re-weights expert outputs in f32 afterwards)
        w_down: linear(get(T::Down, e), rq).with_exact(cfg.exact && e.is_none()),
    };
    let router_name = match *f {
        ModelFile::Gguf(_) => gguf_name(T::Router, l),
        ModelFile::SafeTensors(_) => hf_name(T::Router, l, None),
    };
    let ffn = match &cfg.moe {
        Some(m) if f.has(&router_name) => FfnWeights::Moe {
            router: Linear::new(get(T::Router, None).dequantize()),
            bank: ExpertBank::Resident(
                (0..m.n_experts)
                    .map(|e| Arc::new(expert(Some(e))))
                    .collect(),
            ),
            top_k: m.top_k,
        },
        _ => FfnWeights::Dense(Arc::new(expert(None))),
    };
    let opt = |name: (&str, &str)| -> Option<Vec<f32>> {
        let n = match *f {
            ModelFile::Gguf(_) => format!("blk.{l}.{}", name.0),
            ModelFile::SafeTensors(_) => format!("model.layers.{l}.self_attn.{}", name.1),
        };
        f.has(&n)
            .then(|| f.named_vector(&n).unwrap_or_else(|e| panic!("{e}")))
    };
    LayerWeights {
        index: l,
        extras: crate::model::AttnExtras {
            bq: opt(("attn_q.bias", "q_proj.bias")),
            bk: opt(("attn_k.bias", "k_proj.bias")),
            bv: opt(("attn_v.bias", "v_proj.bias")),
            q_norm: opt(("attn_q_norm.weight", "q_norm.weight")),
            k_norm: opt(("attn_k_norm.weight", "k_norm.weight")),
        },
        attn_norm: vec_f32(&get(T::AttnNorm, None)),
        wq: linear(q, rq),
        wk: linear(k, rq),
        wv: linear(get(T::V, None), rq),
        wo: linear(get(T::O, None), rq).with_exact(cfg.exact),
        ffn_norm: vec_f32(&get(T::FfnNorm, None)),
        ffn,
    }
}

/// Embedding (decoded to f32), final norm and LM head (zero-copy view).
pub fn load_head(cfg: &ModelConfig) -> HeadWeights {
    let (f, rq) = file_of(cfg);
    let get = |w: &str| f.head_view(w).unwrap_or_else(|err| panic!("{err}"));
    HeadWeights {
        embed: crate::tensor::Embedding::Quant(get("embed")),
        final_norm: vec_f32(&get("norm")),
        lm_head: linear(get("output"), rq),
    }
}

/// MoE router of layer `l` (f32), if the model has experts.
pub fn load_router(cfg: &ModelConfig, l: usize) -> Option<Linear> {
    cfg.moe.as_ref()?;
    let (f, _) = file_of(cfg);
    f.layer_view(T::Router, l, None)
        .ok()
        .map(|v| Linear::new(v.dequantize()))
}

/// The tokenizer for any model config (file-backed: stored vocabulary;
/// synthetic: the engine's trained byte-level BPE of the right size).
/// Panics when a model file's own tokenizer cannot be read: silently
/// substituting another vocabulary would feed the model wrong token ids.
pub fn tokenizer_for(cfg: &ModelConfig) -> Tokenizer {
    if cfg.source.is_file() {
        match file_of(cfg).0.tokenizer() {
            Ok(Some(t)) => return t,
            Ok(None) => {}
            Err(e) => panic!("{e}"),
        }
    }
    crate::tokenizer::synthetic_tokenizer(cfg.vocab)
}

/// Weight bytes a node addresses (and would page in) for `layers`, with
/// expert / dense-FFN hidden units restricted to `hidden = (rank, world)`
/// as under H-TEP, plus the head when `with_head`.
pub fn shard_bytes(
    cfg: &ModelConfig,
    layers: std::ops::Range<usize>,
    hidden: Option<(usize, usize)>,
    with_head: bool,
) -> u64 {
    let mut total = 0u64;
    let slice = |e: &Expert| -> u64 {
        match hidden {
            Some((r, w)) => {
                let (h0, h1) = e.hidden_range(r, w);
                e.slice_hidden(h0, h1).bytes() as u64
            }
            None => e.bytes() as u64,
        }
    };
    for l in layers {
        let lw = LayerWeights::generate(cfg, l, None);
        total += (lw.wq.bytes() + lw.wk.bytes() + lw.wv.bytes() + lw.wo.bytes()) as u64;
        total += ((lw.attn_norm.len() + lw.ffn_norm.len()) * 4) as u64;
        match &lw.ffn {
            FfnWeights::Dense(e) => total += slice(e),
            FfnWeights::Moe { router, bank, .. } => {
                total += router.bytes() as u64;
                for e in 0..cfg.moe.as_ref().unwrap().n_experts {
                    total += slice(&bank.acquire(e));
                }
            }
        }
    }
    if with_head {
        let h = HeadWeights::generate(cfg);
        total += (h.lm_head.bytes() + h.embed.bytes() + h.final_norm.len() * 4) as u64;
    }
    total
}

// --- exporters ------------------------------------------------------------------

/// GGUF `general.file_type` for a uniform weight format.
fn file_type(fmt: WeightFormat) -> u32 {
    match fmt {
        WeightFormat::F32 => 0,
        WeightFormat::F16 => 1,
        WeightFormat::Q4_0 => 2,
        WeightFormat::Q8_0 => 7,
        WeightFormat::BF16 => 32,
        WeightFormat::Q4_1 => 3,
        WeightFormat::Q5_0 => 8,
        WeightFormat::Q5_1 => 9,
        WeightFormat::Q4_K => 15,
        WeightFormat::Q5_K => 17,
        WeightFormat::Q6_K => 18,
    }
}

/// Row permutation llama.cpp applies to Q/K (half-split → interleaved RoPE).
fn permute_rows(w: &Tensor2D, n_head: usize) -> Tensor2D {
    let dh = w.rows / n_head;
    let half = dh / 2;
    let mut out = Tensor2D::zeros(w.rows, w.cols);
    for h in 0..n_head {
        for j in 0..dh {
            let (b, a) = (j / 2, j % 2);
            out.row_mut(h * dh + j)
                .copy_from_slice(w.row(h * dh + a * half + b));
        }
    }
    out
}

/// Export a synthetic model as a GGUF v3 file (llama.cpp tensor names and
/// metadata, Mixtral-style stacked experts) with every matrix in `fmt`
/// (norms and routers stay F32). Returns the file size.
pub fn export_gguf(
    cfg: &ModelConfig,
    fmt: WeightFormat,
    tok: Option<&Tokenizer>,
    path: impl AsRef<Path>,
) -> std::io::Result<u64> {
    export_gguf_with(cfg, fmt, head_format(fmt), tok, path)
}

/// Format of `output.weight` in a GGUF export of format `fmt`: like
/// llama.cpp, Q4_0 / Q4_K files keep the LM head at Q6_K (the K-quant
/// fallback rule turns that into Q8_0 when the row width is not a multiple
/// of 256); every other format stores it as `fmt`.
pub fn head_format(fmt: WeightFormat) -> WeightFormat {
    match fmt {
        WeightFormat::Q4_0 | WeightFormat::Q4_K => WeightFormat::Q6_K,
        f => f,
    }
}

/// llama.cpp's substitute when a row width is not a multiple of a K-quant
/// super-block (256): Q4_K → Q5_0, Q6_K → Q8_0.
pub fn block_fallback(f: WeightFormat, cols: usize) -> WeightFormat {
    if cols.is_multiple_of(f.block_len()) {
        return f;
    }
    match f {
        WeightFormat::Q4_K => WeightFormat::Q5_0,
        WeightFormat::Q6_K | WeightFormat::Q5_K => WeightFormat::Q8_0,
        _ => WeightFormat::F16,
    }
}

/// Tensor roles of an export, for per-tensor format choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExportRole {
    Embed,
    Output,
    Attn,
    AttnV,
    FfnGateUp,
    FfnDown,
}

/// Quantization recipes for GGUF exports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantMix {
    /// Every matrix in one format (LM head per [`head_format`]).
    Uniform(WeightFormat),
    /// llama.cpp `Q4_K_M`: Q4_K everywhere; Q6_K for the LM head and for
    /// `attn_v` / `ffn_down` of the "more bits" layers (first and last eighth,
    /// every third in between).
    Q4KM,
}

impl QuantMix {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "q4_k_m" | "q4km" => Some(QuantMix::Q4KM),
            f => WeightFormat::parse(f).map(QuantMix::Uniform),
        }
    }
    pub fn name(&self) -> String {
        match self {
            QuantMix::Uniform(f) => f.name().to_string(),
            QuantMix::Q4KM => "Q4_K_M".to_string(),
        }
    }
    /// Format of one tensor (before the K-quant width fallback).
    pub fn format_for(&self, role: ExportRole, layer: usize, n_layers: usize) -> WeightFormat {
        match self {
            QuantMix::Uniform(f) => {
                if role == ExportRole::Output {
                    head_format(*f)
                } else {
                    *f
                }
            }
            QuantMix::Q4KM => {
                let more_bits = layer < n_layers / 8
                    || layer >= 7 * n_layers / 8
                    || (layer >= n_layers / 8 && (layer - n_layers / 8) % 3 == 2);
                match role {
                    ExportRole::Output => WeightFormat::Q6_K,
                    ExportRole::AttnV | ExportRole::FfnDown if more_bits => WeightFormat::Q6_K,
                    _ => WeightFormat::Q4_K,
                }
            }
        }
    }
    fn file_type(&self) -> u32 {
        match self {
            QuantMix::Uniform(f) => file_type(*f),
            QuantMix::Q4KM => 15,
        }
    }
}

/// Export with a quantization recipe (e.g. llama.cpp's `Q4_K_M`).
pub fn export_gguf_mix(
    cfg: &ModelConfig,
    mix: QuantMix,
    tok: Option<&Tokenizer>,
    path: impl AsRef<Path>,
) -> std::io::Result<u64> {
    export_core(
        cfg,
        &|role, l| mix.format_for(role, l, cfg.n_layers),
        mix.file_type(),
        tok,
        path,
    )
}

/// [`export_gguf`] with an explicit format for the LM head (`output.weight`).
pub fn export_gguf_with(
    cfg: &ModelConfig,
    fmt: WeightFormat,
    head_fmt: WeightFormat,
    tok: Option<&Tokenizer>,
    path: impl AsRef<Path>,
) -> std::io::Result<u64> {
    export_core(
        cfg,
        &|role, _| {
            if role == ExportRole::Output {
                head_fmt
            } else {
                fmt
            }
        },
        file_type(fmt),
        tok,
        path,
    )
}

fn export_core(
    cfg: &ModelConfig,
    fmt_of: &dyn Fn(ExportRole, usize) -> WeightFormat,
    ftype: u32,
    tok: Option<&Tokenizer>,
    path: impl AsRef<Path>,
) -> std::io::Result<u64> {
    let src = ModelConfig {
        source: WeightSource::Synthetic,
        ..cfg.clone()
    };
    let mut w = GgufWriter::new();
    let a = "llama";
    let u32v = |x: usize| GgufValue::U32(x as u32);
    w.add_kv("general.architecture", GgufValue::Str(a.into()));
    w.add_kv(
        "general.name",
        GgufValue::Str(format!(
            "tesseract-synthetic-{}L-{}d",
            cfg.n_layers, cfg.d_model
        )),
    );
    w.add_kv("general.file_type", GgufValue::U32(ftype));
    w.add_kv(&format!("{a}.context_length"), u32v(65_536));
    w.add_kv(&format!("{a}.embedding_length"), u32v(cfg.d_model));
    w.add_kv(&format!("{a}.block_count"), u32v(cfg.n_layers));
    let ff = cfg.moe.as_ref().map(|m| m.d_expert).unwrap_or(cfg.d_ff);
    w.add_kv(&format!("{a}.feed_forward_length"), u32v(ff));
    w.add_kv(&format!("{a}.attention.head_count"), u32v(cfg.n_heads));
    w.add_kv(
        &format!("{a}.attention.head_count_kv"),
        u32v(cfg.n_kv_heads),
    );
    w.add_kv(&format!("{a}.attention.key_length"), u32v(cfg.d_head));
    w.add_kv(&format!("{a}.attention.value_length"), u32v(cfg.d_head));
    w.add_kv(&format!("{a}.rope.dimension_count"), u32v(cfg.d_head));
    w.add_kv(
        &format!("{a}.rope.freq_base"),
        GgufValue::F32(cfg.rope_theta),
    );
    w.add_kv(
        &format!("{a}.attention.layer_norm_rms_epsilon"),
        GgufValue::F32(cfg.rms_eps),
    );
    w.add_kv(&format!("{a}.vocab_size"), u32v(cfg.vocab));
    if let Some(m) = &cfg.moe {
        w.add_kv(&format!("{a}.expert_count"), u32v(m.n_experts));
        w.add_kv(&format!("{a}.expert_used_count"), u32v(m.top_k));
    }
    w.add_kv("tesseract.synthetic_seed", GgufValue::U64(cfg.seed));
    if let Some(t) = tok {
        for (k, v) in t.gguf_metadata() {
            w.add_kv(&k, v);
        }
    }
    let pick = |role: ExportRole, l: usize, cols: usize| block_fallback(fmt_of(role, l), cols);
    let mat = |t: &Tensor2D, f: WeightFormat| {
        (
            vec![t.cols as u64, t.rows as u64],
            encode_matrix(&t.data, t.cols, f),
        )
    };
    let head = HeadWeights::generate(&src);
    let embed = head.embed.to_dense();
    let f = pick(ExportRole::Embed, 0, embed.cols);
    let (dims, b) = mat(&embed, f);
    w.add_tensor("token_embd.weight", &dims, f, b);
    w.add_tensor(
        "output_norm.weight",
        &[cfg.d_model as u64],
        WeightFormat::F32,
        encode_matrix(&head.final_norm, cfg.d_model, WeightFormat::F32),
    );
    let f = pick(ExportRole::Output, 0, head.lm_head.w.cols);
    let (dims, b) = mat(&head.lm_head.w, f);
    w.add_tensor("output.weight", &dims, f, b);
    for l in 0..cfg.n_layers {
        let lw = LayerWeights::generate(&src, l, None);
        let d = cfg.d_model as u64;
        w.add_tensor(
            &gguf_name(T::AttnNorm, l),
            &[d],
            WeightFormat::F32,
            encode_matrix(&lw.attn_norm, cfg.d_model, WeightFormat::F32),
        );
        w.add_tensor(
            &gguf_name(T::FfnNorm, l),
            &[d],
            WeightFormat::F32,
            encode_matrix(&lw.ffn_norm, cfg.d_model, WeightFormat::F32),
        );
        for (t, m, role) in [
            (T::Q, permute_rows(&lw.wq.w, cfg.n_heads), ExportRole::Attn),
            (
                T::K,
                permute_rows(&lw.wk.w, cfg.n_kv_heads),
                ExportRole::Attn,
            ),
            (T::V, lw.wv.w.clone(), ExportRole::AttnV),
            (T::O, lw.wo.w.clone(), ExportRole::Attn),
        ] {
            let f = pick(role, l, m.cols);
            let (dims, b) = mat(&m, f);
            w.add_tensor(&gguf_name(t, l), &dims, f, b);
        }
        match &lw.ffn {
            FfnWeights::Dense(e) => {
                for (t, m, role) in [
                    (T::Gate, &e.w_gate.w, ExportRole::FfnGateUp),
                    (T::Up, &e.w_up.w, ExportRole::FfnGateUp),
                    (T::Down, &e.w_down.w, ExportRole::FfnDown),
                ] {
                    let f = pick(role, l, m.cols);
                    let (dims, b) = mat(m, f);
                    w.add_tensor(&gguf_name(t, l), &dims, f, b);
                }
            }
            FfnWeights::Moe { router, bank, .. } => {
                let (dims, b) = mat(&router.w, WeightFormat::F32);
                w.add_tensor(&gguf_name(T::Router, l), &dims, WeightFormat::F32, b);
                let n = cfg.moe.as_ref().unwrap().n_experts;
                let experts: Vec<Arc<Expert>> = (0..n).map(|e| bank.acquire(e)).collect();
                for t in [T::Gate, T::Up, T::Down] {
                    let mut bytes = Vec::new();
                    let mut shape = (0, 0);
                    let role = if t == T::Down {
                        ExportRole::FfnDown
                    } else {
                        ExportRole::FfnGateUp
                    };
                    let f = pick(
                        role,
                        l,
                        if t == T::Down {
                            experts[0].w_down.w.cols
                        } else {
                            experts[0].w_gate.w.cols
                        },
                    );
                    for e in &experts {
                        let m = match t {
                            T::Gate => &e.w_gate.w,
                            T::Up => &e.w_up.w,
                            _ => &e.w_down.w,
                        };
                        shape = (m.cols as u64, m.rows as u64);
                        bytes.extend(encode_matrix(&m.data, m.cols, f));
                    }
                    let name = gguf_name(t, l).replace(".weight", "_exps.weight");
                    w.add_tensor(&name, &[shape.0, shape.1, n as u64], f, bytes);
                }
            }
        }
    }
    w.write(path)
}

/// Export a synthetic model as a Hugging Face Llama / Mixtral SafeTensors
/// model directory (`model.safetensors`, `config.json`, and
/// `tokenizer.json` when given). Returns the weight file size.
pub fn export_safetensors(
    cfg: &ModelConfig,
    dtype: Dtype,
    tok: Option<&Tokenizer>,
    dir: impl AsRef<Path>,
) -> std::io::Result<u64> {
    let dir = dir.as_ref();
    std::fs::create_dir_all(dir)?;
    let src = ModelConfig {
        source: WeightSource::Synthetic,
        ..cfg.clone()
    };
    let mut w = SafeTensorsWriter::new();
    w.add_metadata("format", "pt");
    let add = |w: &mut SafeTensorsWriter, name: &str, t: &Tensor2D, dt: Dtype| {
        w.add_f32(name, dt, &[t.rows, t.cols], &t.data);
    };
    let head = HeadWeights::generate(&src);
    add(
        &mut w,
        "model.embed_tokens.weight",
        &head.embed.to_dense(),
        dtype,
    );
    w.add_f32(
        "model.norm.weight",
        Dtype::F32,
        &[cfg.d_model],
        &head.final_norm,
    );
    add(&mut w, "lm_head.weight", &head.lm_head.w, dtype);
    for l in 0..cfg.n_layers {
        let lw = LayerWeights::generate(&src, l, None);
        w.add_f32(
            &hf_name(T::AttnNorm, l, None),
            Dtype::F32,
            &[cfg.d_model],
            &lw.attn_norm,
        );
        w.add_f32(
            &hf_name(T::FfnNorm, l, None),
            Dtype::F32,
            &[cfg.d_model],
            &lw.ffn_norm,
        );
        add(&mut w, &hf_name(T::Q, l, None), &lw.wq.w, dtype);
        add(&mut w, &hf_name(T::K, l, None), &lw.wk.w, dtype);
        add(&mut w, &hf_name(T::V, l, None), &lw.wv.w, dtype);
        add(&mut w, &hf_name(T::O, l, None), &lw.wo.w, dtype);
        match &lw.ffn {
            FfnWeights::Dense(e) => {
                add(&mut w, &hf_name(T::Gate, l, None), &e.w_gate.w, dtype);
                add(&mut w, &hf_name(T::Up, l, None), &e.w_up.w, dtype);
                add(&mut w, &hf_name(T::Down, l, None), &e.w_down.w, dtype);
            }
            FfnWeights::Moe { router, bank, .. } => {
                add(&mut w, &hf_name(T::Router, l, None), &router.w, Dtype::F32);
                for e in 0..cfg.moe.as_ref().unwrap().n_experts {
                    let ex = bank.acquire(e);
                    add(&mut w, &hf_name(T::Gate, l, Some(e)), &ex.w_gate.w, dtype);
                    add(&mut w, &hf_name(T::Up, l, Some(e)), &ex.w_up.w, dtype);
                    add(&mut w, &hf_name(T::Down, l, Some(e)), &ex.w_down.w, dtype);
                }
            }
        }
    }
    let size = w.write(dir.join("model.safetensors"))?;
    let mut c = vec![
        (
            "architectures",
            Json::Arr(vec![Json::str(if cfg.moe.is_some() {
                "MixtralForCausalLM"
            } else {
                "LlamaForCausalLM"
            })]),
        ),
        (
            "model_type",
            Json::str(if cfg.moe.is_some() {
                "mixtral"
            } else {
                "llama"
            }),
        ),
        ("vocab_size", Json::num(cfg.vocab as f64)),
        ("hidden_size", Json::num(cfg.d_model as f64)),
        ("num_hidden_layers", Json::num(cfg.n_layers as f64)),
        ("num_attention_heads", Json::num(cfg.n_heads as f64)),
        ("num_key_value_heads", Json::num(cfg.n_kv_heads as f64)),
        ("head_dim", Json::num(cfg.d_head as f64)),
        (
            "intermediate_size",
            Json::num(cfg.moe.as_ref().map(|m| m.d_expert).unwrap_or(cfg.d_ff) as f64),
        ),
        ("rope_theta", Json::num(cfg.rope_theta as f64)),
        ("rms_norm_eps", Json::num(cfg.rms_eps as f64)),
        ("tie_word_embeddings", Json::Bool(false)),
        (
            "torch_dtype",
            Json::str(match dtype {
                Dtype::F16 => "float16",
                Dtype::BF16 => "bfloat16",
                _ => "float32",
            }),
        ),
        ("tesseract_synthetic_seed", Json::num(cfg.seed as f64)),
    ];
    if let Some(m) = &cfg.moe {
        c.push(("num_local_experts", Json::num(m.n_experts as f64)));
        c.push(("num_experts_per_tok", Json::num(m.top_k as f64)));
    }
    std::fs::write(dir.join("config.json"), Json::obj(c).dump())?;
    if let Some(t) = tok {
        std::fs::write(dir.join("tokenizer.json"), t.to_hf_json().dump())?;
    }
    Ok(size)
}

/// Resolve a model spec: `tiny` / `bench` / `long` (synthetic) or a path to
/// a `.gguf` file / SafeTensors model, optionally re-quantized at load.
pub fn resolve_model(spec: &str, requant: Option<WeightFormat>) -> Result<ModelConfig, LoadError> {
    match spec {
        "tiny" => Ok(ModelConfig::tiny()),
        "bench" => Ok(ModelConfig::bench_moe()),
        "long" => Ok(ModelConfig::long_context()),
        p => config_from_file(p, requant),
    }
}

/// Short description of a model's weights for `/v1/models` and logs.
pub fn describe(cfg: &ModelConfig) -> Json {
    let (origin, path, fmt) = match &cfg.source {
        WeightSource::Synthetic => ("synthetic".to_string(), Json::Null, "F32".to_string()),
        WeightSource::File { path, requant } => {
            let f = open(path).ok();
            let fmt = match (requant, &f) {
                (Some(r), _) => format!("{} (re-quantized at load)", r.name()),
                (None, Some(f)) => match &**f {
                    ModelFile::Gguf(g) => g
                        .tensor("blk.0.attn_q.weight")
                        .and_then(|t| t.format())
                        .map(|f| f.name().to_string())
                        .unwrap_or_else(|| "?".into()),
                    ModelFile::SafeTensors(m) => m
                        .files
                        .first()
                        .and_then(|s| {
                            s.tensors
                                .iter()
                                .find(|t| t.name.contains("q_proj"))
                                .map(|t| t.dtype.name().to_string())
                        })
                        .unwrap_or_else(|| "?".into()),
                },
                _ => "?".into(),
            };
            (
                f.map(|f| f.format_name().to_string())
                    .unwrap_or_else(|| "file".into()),
                Json::str(path.clone()),
                fmt,
            )
        }
    };
    Json::obj(vec![
        ("origin", Json::str(origin)),
        ("path", path),
        ("weight_format", Json::str(fmt)),
        ("parameters", Json::num(cfg.total_params() as f64)),
        ("layers", Json::num(cfg.n_layers as f64)),
        ("d_model", Json::num(cfg.d_model as f64)),
        ("vocab", Json::num(cfg.vocab as f64)),
        (
            "experts",
            cfg.moe
                .as_ref()
                .map(|m| Json::str(format!("{} (top-{})", m.n_experts, m.top_k)))
                .unwrap_or(Json::Null),
        ),
    ])
}
