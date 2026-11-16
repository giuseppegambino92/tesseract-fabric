//! V9.0: real third-party GGUF checkpoints (downloaded once into `models/`),
//! K-quants, multi-architecture support, chat templates, distributed
//! exactness, prefix caching, hot-join and concurrent serving.
//!
//! The first run needs network access to huggingface.co (≈ 1.85 GB in total);
//! later runs use the cached files.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tesseract_fabric::cluster::discovery::*;
use tesseract_fabric::cluster::*;
use tesseract_fabric::kv_fabric::{BatchMeta, LocalKvCache};
use tesseract_fabric::loader::sharded_loader::{config_from_file, tokenizer_for};
use tesseract_fabric::loader::GgufFile;
use tesseract_fabric::model::*;
use tesseract_fabric::server::json::Json;
use tesseract_fabric::tensor::kquant::{decode_group, GroupBlock};
use tesseract_fabric::tensor::quant::*;
use tesseract_fabric::tensor::{f32_to_f16, Rng, Tensor2D};
use tesseract_fabric::tokenizer::{stop_tokens, ChatFormat, PreTokenizer, Tokenizer};
use tesseract_fabric::wire::multipath::LinkProfile;

// --- real checkpoints -------------------------------------------------------------

const SMOL_Q8: &str = "SmolLM2-135M-Instruct-Q8_0.gguf";
const SMOL_Q4KM: &str = "SmolLM2-135M-Instruct-Q4_K_M.gguf";
const QWEN25: &str = "qwen2.5-0.5b-instruct-q4_k_m.gguf";
const QWEN3: &str = "Qwen3-0.6B-Q4_K_M.gguf";
const LLAMA32: &str = "Llama-3.2-1B-Instruct-Q4_K_M.gguf";

fn repo(file: &str) -> &'static str {
    match file {
        SMOL_Q8 | SMOL_Q4KM => "bartowski/SmolLM2-135M-Instruct-GGUF",
        QWEN25 => "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        QWEN3 => "unsloth/Qwen3-0.6B-GGUF",
        LLAMA32 => "bartowski/Llama-3.2-1B-Instruct-GGUF",
        _ => panic!("unknown model {file}"),
    }
}

static DOWNLOAD: Mutex<()> = Mutex::new(());

/// Path of a cached checkpoint, downloading it from Hugging Face once.
fn model(file: &str) -> String {
    let _g = DOWNLOAD.lock().unwrap_or_else(|e| e.into_inner());
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models");
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join(file);
    if !p.exists() {
        let part = dir.join(format!("{file}.part"));
        let url = format!("https://huggingface.co/{}/resolve/main/{file}", repo(file));
        let ok = std::process::Command::new("curl")
            .args(["-sSfL", "--retry", "3", "-o"])
            .arg(&part)
            .arg(&url)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        assert!(
            ok,
            "download of {url} failed (network needed on the first run)"
        );
        std::fs::rename(&part, &p).unwrap();
    }
    p.to_string_lossy().into_owned()
}

/// Greedy single-node generation (one process, no cluster).
fn single_node(cfg: &ModelConfig, prompt: &[u32], n: usize, stops: &[u32]) -> Vec<u32> {
    let head = HeadWeights::generate(cfg);
    let model = ShardModel::generate(cfg, 0..cfg.n_layers, None);
    // V10: the model's exact (split-invariant) attention, as every engine uses
    let mut kv = LocalKvCache::new(cfg.n_heads, cfg.n_kv_heads, cfg.d_head).with_exact(cfg.exact);
    let mut toks = prompt.to_vec();
    let m = toks.len();
    let meta = BatchMeta::causal_prefill(0, m - 1, 1);
    let mut x = head.embed_tokens(&toks[..m - 1]);
    model.forward(&mut x, &meta, &mut kv, &mut NoObserver);
    kv.commit(&meta.keys, &|_| false);
    for _ in 0..n {
        let pos = toks.len() - 1;
        let meta = BatchMeta::causal_prefill(pos, 1, 2 + pos as u64);
        let mut x = head.embed_tokens(&toks[pos..]);
        model.forward(&mut x, &meta, &mut kv, &mut NoObserver);
        kv.commit(&meta.keys, &|_| false);
        let t = argmax(head.logits(&x, cfg.rms_eps).row(0)) as u32;
        toks.push(t);
        if stops.contains(&t) {
            break;
        }
    }
    toks[m..].to_vec()
}

/// Perplexity of `ids` under the model (teacher forced).
fn perplexity(cfg: &ModelConfig, ids: &[u32]) -> f64 {
    let l = ReferenceModel::new(cfg).forward_sequence(ids);
    let mut nll = 0.0f64;
    for i in 0..ids.len() - 1 {
        let row = l.row(i);
        let m = row.iter().fold(f32::MIN, |a, &b| a.max(b)) as f64;
        let lse = m + row.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln();
        nll += lse - row[ids[i + 1] as usize] as f64;
    }
    (nll / (ids.len() - 1) as f64).exp()
}

/// Real words, no replacement characters, a sentence's worth of them.
fn coherent(text: &str) -> bool {
    let words: Vec<&str> = text.split_whitespace().collect();
    let wordy = words
        .iter()
        .filter(|w| {
            w.chars()
                .all(|c| c.is_alphabetic() || ".,;:!?'’\"()-–—*0123456789".contains(c))
        })
        .count();
    !text.contains('\u{FFFD}') && words.len() >= 6 && wordy * 10 >= words.len() * 9
}

const ENGLISH: &str = "The quick brown fox jumps over the lazy dog. Paris is the capital of France, and Rome is the capital of Italy. Water boils at one hundred degrees Celsius at sea level, and it freezes at zero degrees.";

#[test]
fn real_gguf_checkpoints_generate_coherent_text() {
    for (file, prompt, must) in [
        (
            SMOL_Q8,
            "Explain in one sentence what distributed AI inference is.",
            "distribut",
        ),
        (
            SMOL_Q4KM,
            "Explain in one sentence what distributed AI inference is.",
            "distribut",
        ),
        (
            QWEN25,
            "Spiega in una frase che cos'è l'inferenza distribuita.",
            "distribu",
        ),
        (
            QWEN3,
            "Explain in one sentence what distributed AI inference is.",
            "distribut",
        ),
        (
            LLAMA32,
            "Explain in one sentence what distributed AI inference is.",
            "distribut",
        ),
    ] {
        let path = model(file);
        let cfg = config_from_file(&path, None).unwrap();
        let tok = tokenizer_for(&cfg);
        let fmt = ChatFormat::detect(&tok);
        let ids = fmt.encode(&tok, &[("user".into(), prompt.into())]);
        let t = Instant::now();
        let out = single_node(&cfg, &ids, 40, &stop_tokens(&tok));
        let text = tok.decode(&out);
        let english = tok.encode(ENGLISH);
        let ppl = perplexity(&cfg, &english);
        // the same tokens in a scrambled order: a model that read the text
        // must find it far less likely
        let mut shuffled = english.clone();
        let mut rng = Rng::new(9);
        for i in (1..shuffled.len()).rev() {
            shuffled.swap(i, rng.below(i + 1));
        }
        let ppl_shuffled = perplexity(&cfg, &shuffled);
        eprintln!(
            "{file} [{}, {}]: {:.1}s · perplexity {ppl:.1} (shuffled {ppl_shuffled:.0}) · {text:?}",
            fmt.name(),
            cfg.n_layers,
            t.elapsed().as_secs_f64()
        );
        assert!(coherent(&text), "{file}: incoherent output {text:?}");
        assert!(
            text.to_lowercase().contains(must),
            "{file}: off-topic output {text:?}"
        );
        assert!(ppl < 60.0, "{file}: perplexity {ppl} on plain English");
        assert!(
            ppl_shuffled > 4.0 * ppl,
            "{file}: shuffled text not much less likely ({ppl_shuffled} vs {ppl})"
        );
    }
}

#[test]
fn architectures_quants_and_rope_scaling_are_read_from_the_files() {
    // Qwen2.5: q/k/v biases, untied output; Qwen3: per-head QK-norm and
    // head_dim 128 ≠ d_model / heads; Llama 3.2: RoPE frequency factors
    let q25 = config_from_file(&model(QWEN25), None).unwrap();
    let l = LayerWeights::generate(&q25, 0, None);
    assert!(
        l.extras.bq.is_some()
            && l.extras.bk.is_some()
            && l.extras.bv.is_some()
            && l.extras.q_norm.is_none()
    );
    assert_eq!(
        (q25.d_model, q25.n_heads, q25.n_kv_heads, q25.d_head),
        (896, 14, 2, 64)
    );
    let q3 = config_from_file(&model(QWEN3), None).unwrap();
    let l = LayerWeights::generate(&q3, 0, None);
    assert!(l.extras.q_norm.as_ref().is_some_and(|n| n.len() == 128) && l.extras.bq.is_none());
    assert_eq!((q3.d_model, q3.d_head, q3.n_heads), (1024, 128, 16));
    assert_eq!(
        l.wq.out_features(),
        2048,
        "q projection is heads × head_dim"
    );
    let ll = config_from_file(&model(LLAMA32), None).unwrap();
    assert_eq!(ll.rope_scaling.freq_factors.len(), 32);
    assert!(ll.rope_scaling.freq_factors.iter().any(|&f| f > 1.0));
    assert_eq!(ll.rope_theta, 500_000.0);
    // the K-quant mix of a real Q4_K_M file is fully supported
    let g = GgufFile::open(model(LLAMA32)).unwrap();
    let mut seen = std::collections::BTreeSet::new();
    for t in &g.tensors {
        seen.insert(t.format().map(|f| f.name()).unwrap_or("?"));
    }
    assert!(
        seen.contains("Q4_K") && seen.contains("Q6_K") && !seen.contains("?"),
        "{seen:?}"
    );
}

#[test]
fn chat_templates_and_pretokenizers_match_the_models() {
    let smol = tokenizer_for(&config_from_file(&model(SMOL_Q8), None).unwrap());
    let qwen = tokenizer_for(&config_from_file(&model(QWEN25), None).unwrap());
    let qwen3 = tokenizer_for(&config_from_file(&model(QWEN3), None).unwrap());
    let llama = tokenizer_for(&config_from_file(&model(LLAMA32), None).unwrap());
    assert_eq!(
        (smol.pre, qwen.pre, llama.pre),
        (
            PreTokenizer::Smollm,
            PreTokenizer::Qwen2,
            PreTokenizer::Llama3
        )
    );
    // ids as in the models' own vocabularies
    assert_eq!(
        llama.encode("The capital of France is"),
        vec![791, 6864, 315, 9822, 374]
    );
    assert_eq!(
        qwen.encode("The capital of France is"),
        vec![785, 6722, 315, 9625, 374]
    );
    assert_eq!(
        smol.encode("The capital of France is"),
        vec![504, 3575, 282, 4649, 314]
    );
    // pre-tokenizer rules: digits (smollm / qwen: single, llama3: ≤ 3), newline runs
    let pieces = |t: &Tokenizer, s: &str| {
        t.encode(s)
            .iter()
            .map(|&i| t.tokens[i as usize].clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(pieces(&llama, " 2024"), vec!["Ġ", "202", "4"]);
    assert_eq!(pieces(&qwen, " 2024"), vec!["Ġ", "2", "0", "2", "4"]);
    assert_eq!(pieces(&smol, " 2024"), vec!["Ġ", "2", "0", "2", "4"]);
    let msgs = vec![("user".to_string(), "Hi!".to_string())];
    let r = |t: &Tokenizer| ChatFormat::detect(t).render(t, &msgs);
    assert_eq!(r(&smol), "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n<|im_start|>user\nHi!<|im_end|>\n<|im_start|>assistant\n");
    assert_eq!(r(&qwen), "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nHi!<|im_end|>\n<|im_start|>assistant\n");
    assert_eq!(
        r(&qwen3),
        "<|im_start|>user\nHi!<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
    assert_eq!(r(&llama), "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHi!<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");
    // special tokens become single ids; stop tokens are the turn terminators
    let ids = ChatFormat::detect(&qwen).encode(&qwen, &msgs);
    assert_eq!(ids[0], qwen.token_id("<|im_start|>").unwrap());
    assert!(stop_tokens(&qwen).contains(&qwen.token_id("<|im_end|>").unwrap()));
    assert!(stop_tokens(&llama).contains(&llama.token_id("<|eot_id|>").unwrap()));
}

#[test]
fn k_quant_layouts_kernels_and_encoders() {
    // golden Q4_K block: d = 1, dmin = 0.5, sub-block j has scale j+1 and
    // min 2j, every nibble = (index mod 16)
    let mut b = Vec::new();
    b.extend(f32_to_f16(1.0).to_le_bytes());
    b.extend(f32_to_f16(0.5).to_le_bytes());
    let (sc, mn): (Vec<u8>, Vec<u8>) = ((1..=8).collect(), (0..8).map(|j| 2 * j).collect());
    let mut s12 = [0u8; 12];
    for j in 0..8 {
        if j < 4 {
            s12[j] = sc[j];
            s12[j + 4] = mn[j];
        } else {
            s12[j + 4] = (sc[j] & 0xF) | ((mn[j] & 0xF) << 4);
            s12[j - 4] |= (sc[j] >> 4) << 6;
            s12[j] |= (mn[j] >> 4) << 6;
        }
    }
    b.extend(s12);
    b.extend((0..128u32).map(|i| ((i % 16) | ((15 - i % 16) << 4)) as u8));
    let mut g = GroupBlock::default();
    decode_group(WeightFormat::Q4_K, &b, &mut g);
    for e in 0..256 {
        let j = e / 32;
        let (chunk, l) = (e / 64, e % 32);
        let q = if (e / 32) % 2 == 0 {
            (chunk * 32 + l) % 16
        } else {
            15 - (chunk * 32 + l) % 16
        };
        let want = (j + 1) as f32 * q as f32 - 0.5 * (2 * j) as f32;
        let got = g.scale[e / 16] * g.q[e] as f32 + g.add[e / 32];
        assert_eq!(got, want, "Q4_K element {e}");
    }
    // encoders + fused kernels vs dequantized math
    let mut rng = Rng::new(31);
    let w = Tensor2D::randn(64, 1024, 0.05, &mut rng);
    let x = Tensor2D::randn(4, 1024, 1.0, &mut rng);
    for (fmt, max_w_err) in [
        (WeightFormat::Q6_K, 0.02),
        (WeightFormat::Q4_K, 0.12),
        (WeightFormat::Q5_0, 0.06),
    ] {
        let v = QuantView::from_f32(&w, fmt);
        let deq = v.dequantize();
        let werr = (deq
            .data
            .iter()
            .zip(&w.data)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / w.data.iter().map(|a| a * a).sum::<f32>())
        .sqrt();
        let mut y = Tensor2D::zeros(4, 64);
        gemm_quant_with(&x, &v, &mut y, best_kernel());
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for i in 0..4 {
            for j in 0..64 {
                let r: f64 = (0..1024)
                    .map(|t| x.row(i)[t] as f64 * deq.row(j)[t] as f64)
                    .sum();
                num += (y.data[i * 64 + j] as f64 - r).powi(2);
                den += r * r;
            }
        }
        let rel = (num / den).sqrt();
        eprintln!("{}: {:.2} bits/weight · weight rounding {werr:.3} · fused kernel vs dequantized math {rel:.2e}", fmt.name(), fmt.bits_per_weight());
        assert!(werr < max_w_err, "{}: weight error {werr}", fmt.name());
        assert!(rel < 1e-2, "{}: kernel error {rel}", fmt.name());
    }
    assert_eq!(WeightFormat::Q4_K.row_bytes(256), 144);
    assert_eq!(WeightFormat::Q6_K.row_bytes(256), 210);
    assert_eq!(WeightFormat::Q5_K.row_bytes(256), 176);
    // a real Q6_K tensor decodes to sane weights
    let gf = GgufFile::open(model(LLAMA32)).unwrap();
    let t = gf
        .tensors
        .iter()
        .find(|t| t.format() == Some(WeightFormat::Q6_K) && t.dims.len() == 2)
        .unwrap();
    let v = gf.view(&t.name, None).unwrap();
    let mut row = vec![0.0f32; v.cols];
    v.decode_row_into(0, &mut row);
    let rms = (row.iter().map(|a| a * a).sum::<f32>() / row.len() as f32).sqrt();
    assert!(
        row.iter().all(|a| a.is_finite()) && rms > 1e-4 && rms < 1.0,
        "{}: rms {rms}",
        t.name
    );
}

/// A cluster on `n` loopback workers; returns the driver and assignment.
fn cluster(
    cfg: &ModelConfig,
    n: u16,
    engine: EngineKind,
    tess: TesseractOptions,
) -> (Driver, Assignment, Vec<LocalWorker>) {
    let ws = spawn_local_workers(&(0..n).collect::<Vec<_>>());
    let driver = Driver::bind("127.0.0.1:0").unwrap();
    let mut links = vec![];
    for a in 0..n {
        for b in a + 1..n {
            links.push((a, b, vec![LinkProfile::thunderbolt5()]));
        }
    }
    let stages: Vec<(u16, usize, usize)> = match engine {
        EngineKind::TesseractHtep => (0..n).map(|i| (i, 0, cfg.n_layers)).collect(),
        _ => (0..n as usize)
            .map(|i| {
                (
                    i as u16,
                    cfg.n_layers * i / n as usize,
                    cfg.n_layers * (i + 1) / n as usize,
                )
            })
            .collect(),
    };
    let head = if engine == EngineKind::TesseractHtep {
        0
    } else {
        n - 1
    };
    let a = Assignment {
        engine,
        model: cfg.clone(),
        nodes: ws
            .iter()
            .map(|w| NodeSetup {
                id: w.id,
                addr: w.addr,
                slowdown: 1.0,
                kv_capacity: 1 << 30,
            })
            .collect(),
        links,
        stages,
        head,
        hot_capacity: 0,
        inject_context: 0,
        tess,
        driver_addr: driver.addr(),
        service_addrs: ws.iter().flat_map(|w| w.service_addrs()).collect(),
    };
    driver.assign(&a);
    (driver, a, ws)
}

fn greedy(prompt: &[u32], n: u32) -> GenerateRequest {
    GenerateRequest {
        prompt: prompt.to_vec(),
        max_new: n,
        seed: 1,
        temperature: 0.0,
        record_logits: false,
    }
}

fn real_opts() -> TesseractOptions {
    TesseractOptions {
        codec: 0,
        htep_reduce_f32: true,
        draft_lookup: true,
        ..Default::default()
    }
}

#[test]
fn real_checkpoint_distributed_matches_single_node() {
    for file in [SMOL_Q8, QWEN25] {
        let cfg = config_from_file(&model(file), None).unwrap();
        let tok = tokenizer_for(&cfg);
        let ids = ChatFormat::detect(&tok).encode(
            &tok,
            &[(
                "user".into(),
                "List the capitals of France, Italy and Spain, one per line.".into(),
            )],
        );
        let want = single_node(&cfg, &ids, 32, &[]);
        // 4-stage pipeline, lossless activations, prompt-lookup speculation:
        // every stage runs exactly a single node's arithmetic
        let (d, a, _w) = cluster(&cfg, 4, EngineKind::Tesseract, real_opts());
        let p = d.generate(a.head, &greedy(&ids, 32));
        d.reset(&a);
        d.shutdown(&a);
        // 4-rank H-TEP: f32 all-reduce; the split reduction reorders f32 sums,
        // so tokens agree until the first near-tie, and the text stays coherent
        let (d, a, _w) = cluster(&cfg, 4, EngineKind::TesseractHtep, real_opts());
        let h = d.generate(a.head, &greedy(&ids, 32));
        d.reset(&a);
        d.shutdown(&a);
        let agree = h
            .tokens
            .iter()
            .zip(&want)
            .take_while(|(a, b)| a == b)
            .count();
        eprintln!(
            "{file}: single {:?}\n  pipeline×4 {:.2} tok/step identical={} · H-TEP×4 first {agree} tokens identical: {:?}",
            tok.decode(&want),
            p.tokens.len() as f64 / p.per_epoch.len() as f64,
            p.tokens == want,
            tok.decode(&h.tokens)
        );
        assert_eq!(
            p.tokens, want,
            "{file}: lossless pipeline diverged from the single node"
        );
        assert!(
            agree >= 8 && coherent(&tok.decode(&h.tokens)),
            "{file}: H-TEP agreed for {agree} tokens"
        );
    }
    // KLT-compressed activations (25% of FP16) over a 4-stage pipeline: lossy
    // by design, still coherent text
    let cfg = config_from_file(&model(SMOL_Q8), None).unwrap();
    let tok = tokenizer_for(&cfg);
    let ids = ChatFormat::detect(&tok).encode(
        &tok,
        &[(
            "user".into(),
            "Explain in one sentence what distributed AI inference is.".into(),
        )],
    );
    let (d, a, _w) = cluster(
        &cfg,
        4,
        EngineKind::Tesseract,
        TesseractOptions {
            codec: 4,
            draft_lookup: true,
            ..Default::default()
        },
    );
    let k = d.generate(a.head, &greedy(&ids, 32));
    d.reset(&a);
    d.shutdown(&a);
    eprintln!("KLT pipeline×4: {:?}", tok.decode(&k.tokens));
    assert!(coherent(&tok.decode(&k.tokens)));
}

#[test]
fn successive_requests_reuse_the_prefix_and_never_see_stale_kv() {
    let cfg = config_from_file(&model(SMOL_Q8), None).unwrap();
    let tok = tokenizer_for(&cfg);
    let fmt = ChatFormat::detect(&tok);
    // unrelated prompts (different first token): the second request must be
    // bit-identical to the same request on a fresh engine. V8.0 failed this:
    // its engines kept the previous request's KV visible.
    let raw_a = tok.encode("Name three colours of the rainbow:");
    let raw_b = tok.encode("Fruits that grow on trees include");
    let chat_a = fmt.encode(&tok, &[("user".into(), "Name three colours.".into())]);
    let chat_b = fmt.encode(&tok, &[("user".into(), "Name three fruits.".into())]);
    for engine in [EngineKind::Tesseract, EngineKind::TesseractHtep] {
        let run_fresh = |p: &[u32]| {
            let (d, a, _w) = cluster(&cfg, 2, engine, real_opts());
            let r = d.generate(a.head, &greedy(p, 24));
            d.reset(&a);
            d.shutdown(&a);
            r
        };
        let fresh_raw_b = run_fresh(&raw_b);
        let fresh_chat_b = run_fresh(&chat_b);
        let (d, a, _w) = cluster(&cfg, 2, engine, real_opts());
        d.generate(a.head, &greedy(&raw_a, 24));
        let rb = d.generate(a.head, &greedy(&raw_b, 24));
        d.generate(a.head, &greedy(&chat_a, 24));
        let cb = d.generate(a.head, &greedy(&chat_b, 24));
        let cb2 = d.generate(a.head, &greedy(&chat_b, 24));
        d.reset(&a);
        d.shutdown(&a);
        let shared = chat_a
            .iter()
            .zip(&chat_b)
            .take_while(|(x, y)| x == y)
            .count();
        let agree = cb
            .tokens
            .iter()
            .zip(&fresh_chat_b.tokens)
            .take_while(|(x, y)| x == y)
            .count();
        eprintln!(
            "{}: unrelated request after another reused {} tokens, identical to fresh = {} · chat B reused {} of {} prompt tokens, first {agree} tokens as fresh: {:?} · repeated B reused {}",
            engine.name(),
            rb.prefix_reused,
            rb.tokens == fresh_raw_b.tokens,
            cb.prefix_reused,
            chat_b.len(),
            tok.decode(&cb.tokens),
            cb2.prefix_reused
        );
        assert_eq!(rb.prefix_reused, 0);
        assert_eq!(
            rb.tokens,
            fresh_raw_b.tokens,
            "{}: the previous request's KV leaked into the next",
            engine.name()
        );
        // prefix cache: exactly the shared system prompt + chat header is reused
        assert_eq!(cb.prefix_reused as usize, shared);
        assert!(cb2.prefix_reused as usize >= chat_b.len() - 2);
        // reused KV is mathematically the same context; f32 summation order
        // differs from an all-pending prefill, so only near-ties may flip
        assert!(
            agree >= 8 && coherent(&tok.decode(&cb.tokens)),
            "{}: {agree} tokens agree",
            engine.name()
        );
    }
}

// --- server tests ------------------------------------------------------------------

struct Child(std::process::Child);
impl Drop for Child {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Start `tesseract-serve`; returns (child, http addr, discovery addr).
fn serve(args: &[&str]) -> (Child, String, std::net::SocketAddr) {
    let mut c = std::process::Command::new(env!("CARGO_BIN_EXE_tesseract-serve"))
        .args(["--http", "127.0.0.1:0"])
        .args(args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    let mut out = BufReader::new(c.stdout.take().unwrap());
    let (mut l1, mut l2) = (String::new(), String::new());
    out.read_line(&mut l1).unwrap();
    out.read_line(&mut l2).unwrap();
    let http = l1
        .trim()
        .strip_prefix("listening on http://")
        .unwrap()
        .to_string();
    let disc = l2
        .trim()
        .strip_prefix("discovery on udp ")
        .unwrap()
        .split(' ')
        .next()
        .unwrap()
        .parse()
        .unwrap();
    (Child(c), http, disc)
}

fn get(addr: &str, path: &str) -> Json {
    let mut s = TcpStream::connect(addr).unwrap();
    write!(s, "GET {path} HTTP/1.1\r\nHost: x\r\n\r\n").unwrap();
    let mut out = String::new();
    s.read_to_string(&mut out).unwrap();
    Json::parse(out.split_once("\r\n\r\n").unwrap().1).unwrap()
}

/// Streams a chat request; returns (text, chunk arrival instants, final chunk).
fn stream_chat(addr: &str, body: &str) -> (String, Vec<Instant>, Json) {
    let mut s = TcpStream::connect(addr).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(300))).unwrap();
    write!(
        s,
        "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let r = BufReader::new(s);
    let (mut text, mut times, mut last) = (String::new(), Vec::new(), Json::Null);
    for l in r.lines() {
        let l = l.unwrap();
        let Some(d) = l.strip_prefix("data: ") else {
            continue;
        };
        if d == "[DONE]" {
            break;
        }
        let c = Json::parse(d).unwrap();
        if let Some(t) = c
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|c| c[0].get("delta"))
            .and_then(|d| d.get("content"))
            .and_then(|t| t.as_str())
        {
            text.push_str(t);
        }
        times.push(Instant::now());
        last = c;
    }
    (text, times, last)
}

#[test]
fn dynamic_worker_hot_join_mid_session() {
    let (_srv, http, disc) = serve(&["--model", "tiny", "--local-nodes", "2", "--engine", "htep"]);
    let body = r#"{"stream":true,"max_tokens":16,"temperature":0,"messages":[{"role":"user","content":"hello"}]}"#;
    let (_, t1, last1) = stream_chat(&http, body);
    assert!(!t1.is_empty() && last1.get("tesseract_telemetry").is_some());
    let st = get(&http, "/v1/cluster/status");
    assert_eq!(st.get("active_nodes").unwrap().as_f64(), Some(2.0));
    // a third worker appears and announces itself (as a remote daemon would)
    let key = ClusterKey::from_secret("tesseract-local");
    let w = start_worker(2, "127.0.0.1", 0).unwrap();
    let targets = AnnounceTargets {
        multicast: false,
        broadcast: false,
        port: disc.port(),
        seeds: vec![disc],
    };
    let beacon = Announcer::start(
        key,
        2,
        (w.addr.port(), w.kv_addr.port(), w.draft_addr.port()),
        LocalProfile::detect(),
        targets,
        Duration::from_millis(100),
    )
    .unwrap();
    let t = Instant::now();
    let st = loop {
        let st = get(&http, "/v1/cluster/status");
        if st.get("active_nodes").and_then(|v| v.as_f64()) == Some(3.0)
            || t.elapsed() > Duration::from_secs(30)
        {
            break st;
        }
        std::thread::sleep(Duration::from_millis(200));
    };
    assert_eq!(
        st.get("active_nodes").unwrap().as_f64(),
        Some(3.0),
        "worker never joined"
    );
    // request #2 runs on all three nodes
    let (_, t2, last2) = stream_chat(&http, body);
    assert!(!t2.is_empty() && last2.get("tesseract_telemetry").is_some());
    let st = get(&http, "/v1/cluster/status");
    let nodes = st.get("nodes").unwrap().as_array().unwrap().to_vec();
    eprintln!("after join: {}", st.get("slots").unwrap().dump());
    assert_eq!(nodes.len(), 3);
    for n in &nodes {
        let roles = n.get("roles").unwrap().as_array().unwrap();
        assert!(
            roles
                .iter()
                .any(|r| r.as_str().unwrap().starts_with("H-TEP Shard")),
            "{}",
            n.dump()
        );
        assert!(
            n.get("weight_bytes").unwrap().as_f64().unwrap() > 0.0,
            "node {} holds no weights",
            n.get("id").unwrap().dump()
        );
        assert!(
            n.get("commits").unwrap().as_f64().unwrap() > 0.0,
            "node {} did not take part",
            n.get("id").unwrap().dump()
        );
    }
    assert!(st.get("replans").unwrap().as_f64().unwrap() >= 1.0);
    // the worker leaves (beacons stop): the slot shrinks back and keeps serving
    drop(beacon);
    let t = Instant::now();
    while get(&http, "/v1/cluster/status")
        .get("active_nodes")
        .and_then(|v| v.as_f64())
        != Some(2.0)
        && t.elapsed() < Duration::from_secs(30)
    {
        std::thread::sleep(Duration::from_millis(200));
    }
    assert_eq!(
        get(&http, "/v1/cluster/status")
            .get("active_nodes")
            .unwrap()
            .as_f64(),
        Some(2.0),
        "departed worker kept"
    );
    let (_, t3, _) = stream_chat(&http, body);
    assert!(!t3.is_empty());
}

#[test]
fn concurrent_sse_streams_progress_in_parallel_with_isolated_kv() {
    let path = model(SMOL_Q8);
    let (_srv, http, _) = serve(&["--model", &path, "--local-nodes", "2", "--slots", "2"]);
    let body = r#"{"stream":true,"max_tokens":48,"temperature":0,"messages":[{"role":"user","content":"Describe the ocean in two sentences."}]}"#;
    let h1 = {
        let h = http.clone();
        std::thread::spawn(move || stream_chat(&h, body))
    };
    let h2 = {
        let h = http.clone();
        std::thread::spawn(move || stream_chat(&h, body))
    };
    let (a, ta, la) = h1.join().unwrap();
    let (b, tb, lb) = h2.join().unwrap();
    let slot = |l: &Json| {
        l.get("tesseract_telemetry")
            .unwrap()
            .get("slot")
            .unwrap()
            .as_f64()
            .unwrap()
    };
    eprintln!(
        "A (slot {}) {:?}\nB (slot {}) {:?}",
        slot(&la),
        a,
        slot(&lb),
        b
    );
    assert_ne!(slot(&la), slot(&lb), "both streams ran on one slot");
    // the two streams overlap in time: each started before the other ended
    assert!(
        ta[0] < *tb.last().unwrap() && tb[0] < *ta.last().unwrap(),
        "streams were serialised"
    );
    // isolated KV: identical greedy prompts give identical, coherent replies
    assert_eq!(a, b);
    assert!(coherent(&a));
    // a later request on a warm slot reuses its prompt prefix and changes nothing
    let (c, _, lc) = stream_chat(&http, body);
    let cached = lc
        .get("tesseract_telemetry")
        .unwrap()
        .get("prefix_cached_tokens")
        .unwrap()
        .as_f64()
        .unwrap();
    assert!(cached > 10.0, "prefix cache unused ({cached})");
    assert_eq!(c, a);
}

#[cfg(target_os = "macos")]
#[test]
fn accelerate_expert_offload_matches_portable_kernels() {
    use tesseract_fabric::tensor::accel::{self, Backend};
    use tesseract_fabric::tensor::Linear;
    let mut rng = Rng::new(5);
    let w = Tensor2D::randn(704, 512, 0.05, &mut rng);
    for (lin, rows) in [
        (Linear::new(w.clone()), [1usize, 8, 64]),
        (
            Linear::quantized(QuantView::from_f32(&w, WeightFormat::Q8_0)),
            [16, 32, 64],
        ),
        (
            Linear::quantized(QuantView::from_f32(&w, WeightFormat::Q4_K)),
            [16, 32, 64],
        ),
    ] {
        for m in rows {
            let x = Tensor2D::randn(m, 512, 1.0, &mut rng);
            let mut a = vec![0.0f32; m * 352];
            let mut b = vec![0.0f32; m * 352];
            accel::set_backend(Backend::Portable);
            lin.forward_rows(&x, 100, 452, &mut a);
            accel::set_backend(Backend::Accelerate);
            lin.forward_rows(&x, 100, 452, &mut b);
            accel::set_backend(Backend::Portable);
            let num: f32 = a.iter().zip(&b).map(|(p, q)| (p - q).powi(2)).sum();
            let den: f32 = a.iter().map(|p| p * p).sum();
            assert!(
                (num / den).sqrt() < 1e-2,
                "{:?} m {m}: {}",
                lin.format(),
                (num / den).sqrt()
            );
        }
    }
}
