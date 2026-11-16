//! V10.0: exact (fixed-point) row-parallel reductions, companion draft
//! models, continuous batching, multi-session KV, fast K-quant kernels.
//!
//! Uses the real checkpoints cached in `models/` by the V9 tests (downloaded
//! on the first run; network needed once).

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tesseract_fabric::cluster::*;
use tesseract_fabric::kv_fabric::{BatchMeta, LocalKvCache};
use tesseract_fabric::loader::sharded_loader::{config_from_file, tokenizer_for};
use tesseract_fabric::model::*;
use tesseract_fabric::server::json::Json;
use tesseract_fabric::tokenizer::ChatFormat;
use tesseract_fabric::wire::multipath::LinkProfile;

const SMOL_Q8: &str = "SmolLM2-135M-Instruct-Q8_0.gguf";
const SMOL_Q4KM: &str = "SmolLM2-135M-Instruct-Q4_K_M.gguf";
const QWEN25: &str = "qwen2.5-0.5b-instruct-q4_k_m.gguf";
const QWEN3: &str = "Qwen3-0.6B-Q4_K_M.gguf";
const LLAMA32: &str = "Llama-3.2-1B-Instruct-Q4_K_M.gguf";
/// Draft-model pair: SmolLM2-1.7B (target) with SmolLM2-135M (drafter),
/// same tokenizer.
const SMOL_17B: &str = "SmolLM2-1.7B-Instruct-Q4_K_M.gguf";

fn repo(file: &str) -> &'static str {
    match file {
        SMOL_Q8 | SMOL_Q4KM => "bartowski/SmolLM2-135M-Instruct-GGUF",
        SMOL_17B => "bartowski/SmolLM2-1.7B-Instruct-GGUF",
        QWEN25 => "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        QWEN3 => "unsloth/Qwen3-0.6B-GGUF",
        LLAMA32 => "bartowski/Llama-3.2-1B-Instruct-GGUF",
        _ => panic!("unknown model {file}"),
    }
}

static DOWNLOAD: Mutex<()> = Mutex::new(());

/// Speed thresholds are asserted unless `TESSERACT_SKIP_PERF_ASSERTS` is
/// set (CI does: shared runners measure the VM, not the code). The measured
/// values are printed either way.
fn perf_asserts() -> bool {
    std::env::var_os("TESSERACT_SKIP_PERF_ASSERTS").is_none()
}

/// The tests here measure throughput: run them one at a time.
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

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

/// Greedy single-node generation, one token per step (one process, no
/// cluster): the tokens and the logits that chose them.
fn single_node(cfg: &ModelConfig, prompt: &[u32], n: usize) -> (Vec<u32>, Vec<Vec<f32>>) {
    let head = HeadWeights::generate(cfg);
    let model = ShardModel::generate(cfg, 0..cfg.n_layers, None);
    let mut kv = LocalKvCache::new(cfg.n_heads, cfg.n_kv_heads, cfg.d_head).with_exact(cfg.exact);
    let mut toks = prompt.to_vec();
    let m = toks.len();
    let meta = BatchMeta::causal_prefill(0, m - 1, 1);
    let mut x = head.embed_tokens(&toks[..m - 1]);
    model.forward(&mut x, &meta, &mut kv, &mut NoObserver);
    kv.commit(&meta.keys, &|_| false);
    let mut logits = Vec::new();
    for _ in 0..n {
        let pos = toks.len() - 1;
        let meta = BatchMeta::causal_prefill(pos, 1, 2 + pos as u64);
        let mut x = head.embed_tokens(&toks[pos..]);
        model.forward(&mut x, &meta, &mut kv, &mut NoObserver);
        kv.commit(&meta.keys, &|_| false);
        let l = head.logits(&x, cfg.rms_eps).row(0).to_vec();
        toks.push(argmax(&l) as u32);
        logits.push(l);
    }
    (toks[m..].to_vec(), logits)
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
        record_logits: true,
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

fn chat(
    file: &str,
    text: &str,
) -> (
    ModelConfig,
    tesseract_fabric::tokenizer::Tokenizer,
    Vec<u32>,
) {
    let cfg = config_from_file(&model(file), None).unwrap();
    let tok = tokenizer_for(&cfg);
    let ids = ChatFormat::detect(&tok).encode(&tok, &[("user".into(), text.into())]);
    (cfg, tok, ids)
}

#[test]
fn htep_reproduces_the_single_node_bit_for_bit() {
    let _serial = serial();
    for file in [SMOL_Q8, SMOL_Q4KM, QWEN25, QWEN3, LLAMA32] {
        let (cfg, tok, ids) = chat(file, "List the capitals of France, Italy and Spain, one per line, then explain why Rome is famous.");
        assert!(
            cfg.exact,
            "model files default to exact row-parallel accumulation"
        );
        let n = 40;
        let (want, want_logits) = single_node(&cfg, &ids, n);
        let (d, a, _w) = cluster(&cfg, 4, EngineKind::TesseractHtep, real_opts());
        let h = d.generate(a.head, &greedy(&ids, n as u32));
        d.reset(&a);
        d.shutdown(&a);
        let same_logits = h
            .logits
            .iter()
            .zip(&want_logits)
            .take_while(|(a, b)| a == b)
            .count();
        eprintln!(
            "{file}: H-TEP×4 {}/{} tokens identical, {same_logits} logit rows bit-identical, {:.2} tok/step: {:?}",
            h.tokens.iter().zip(&want).take_while(|(a, b)| a == b).count(),
            want.len(),
            h.tokens.len() as f64 / h.per_epoch.len() as f64,
            tok.decode(&h.tokens)
        );
        assert_eq!(
            h.tokens, want,
            "{file}: H-TEP diverged from the single node"
        );
        assert_eq!(same_logits, want.len(), "{file}: logits differ");
    }
}

fn session_opts(sessions: u8, draft: Option<&str>) -> TesseractOptions {
    TesseractOptions {
        sessions,
        draft_model: draft.map(model).unwrap_or_default(),
        ..real_opts()
    }
}

#[test]
fn draft_model_speculation_is_lossless_and_accepts_long_chains() {
    let _serial = serial();
    // SmolLM2-135M drafts for SmolLM2-1.7B (same vocabulary, 12.6× fewer
    // parameters): greedy chains verified exactly by the target
    let (cfg, tok, ids) = chat(
        SMOL_17B,
        "Write a short paragraph about the history of the Roman Empire.",
    );
    let n = 64;
    let (want, _) = single_node(&cfg, &ids, n);
    let mut speed = Vec::new();
    // (ranks, draft model, fixed tree budget: 0 = cost-model planning)
    for (world, draft, budget) in [
        (1u16, None, 0u16),
        (1, Some(SMOL_Q8), 5),
        (1, Some(SMOL_Q8), 0),
        (4, None, 0),
        (4, Some(SMOL_Q8), 0),
    ] {
        let tess = TesseractOptions {
            budget,
            ..session_opts(1, draft)
        };
        let (d, a, _w) = cluster(&cfg, world, EngineKind::TesseractHtep, tess);
        let r = d.generate(a.head, &greedy(&ids, n as u32));
        d.reset(&a);
        d.shutdown(&a);
        let tps = r.tokens.len() as f64 / r.per_epoch.len() as f64;
        let rate = r.tokens.len() as f64 / (r.decode_ns as f64 / 1e9);
        eprintln!("H-TEP×{world} draft {draft:?} budget {budget}: {tps:.2} tokens/step, {rate:.1} tok/s, identical={}", r.tokens == want);
        assert_eq!(r.tokens, want, "speculation changed the output");
        if budget == 5 {
            assert!(tps >= 1.8, "{tps:.2} tokens per verification step");
        }
        speed.push(rate);
    }
    eprintln!("{:?}", tok.decode(&want));
    eprintln!(
        "1 node: planned drafting {:.2}× · H-TEP×4 {:.2}× · H-TEP×4 + planned drafting {:.2}× (vs 1 node without)",
        speed[2] / speed[0],
        speed[3] / speed[0],
        speed[4] / speed[0]
    );
}

#[test]
fn continuous_batching_serves_concurrent_requests_exactly() {
    let _serial = serial();
    use tesseract_fabric::cluster::batch::BatchClient;
    let file = QWEN25;
    let prompts = [
        "Name three rivers in Europe.",
        "What is the boiling point of water at sea level?",
        "Scrivi una frase sul mare.",
        "List the first five prime numbers.",
    ];
    let cfg = config_from_file(&model(file), None).unwrap();
    let tok = tokenizer_for(&cfg);
    let fmt = ChatFormat::detect(&tok);
    let ids: Vec<Vec<u32>> = prompts
        .iter()
        .map(|p| fmt.encode(&tok, &[("user".into(), p.to_string())]))
        .collect();
    let n = 32;
    let want: Vec<Vec<u32>> = ids.iter().map(|p| single_node(&cfg, p, n).0).collect();
    let (d, a, _w) = cluster(&cfg, 2, EngineKind::TesseractHtep, session_opts(4, None));
    let head = a.head;
    let client = BatchClient::new(d, head);
    // sequential baseline on the same engine
    let t0 = std::time::Instant::now();
    for (i, (p, w)) in ids.iter().zip(&want).enumerate() {
        let g = client.generate(&greedy(p, n as u32));
        let agree = g.tokens.iter().zip(w).take_while(|(a, b)| a == b).count();
        assert_eq!(
            &g.tokens,
            w,
            "sequential request {i}: agrees for {agree} tokens, prefix reused {}, {:?} vs {:?}",
            g.prefix_reused,
            tok.decode(&g.tokens),
            tok.decode(w)
        );
    }
    let seq = t0.elapsed().as_secs_f64();
    // four at once: one batched forward pass per step for all of them
    let t0 = std::time::Instant::now();
    let rx: Vec<_> = ids
        .iter()
        .map(|p| client.submit(greedy(p, n as u32), GenerateExt::default()))
        .collect();
    let got: Vec<GenerateResult> = rx
        .into_iter()
        .map(|r| loop {
            if let tesseract_fabric::cluster::batch::ClientEvent::Done(g) = r.recv().unwrap() {
                break g;
            }
        })
        .collect();
    let conc = t0.elapsed().as_secs_f64();
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        eprintln!(
            "[{i}] prefix reused {} · {:?}",
            g.prefix_reused,
            tok.decode(&g.tokens)
        );
        assert_eq!(
            &g.tokens, w,
            "request {i} differs from its single-node output"
        );
    }
    eprintln!("4 requests × {n} tokens: sequential {seq:.2} s, batched {conc:.2} s ({:.2}× aggregate throughput)", seq / conc);
    if perf_asserts() {
        assert!(seq / conc > 1.5, "batching gave {:.2}×", seq / conc);
    }
}

/// A 2-stage pipeline whose committed KV lives on remote holders (a third,
/// KV-only node and the stages' own fabric shards).
fn disaggregated(cfg: &ModelConfig) -> (Driver, Assignment, Vec<LocalWorker>) {
    let ws = spawn_local_workers(&[0, 1, 2]);
    let driver = Driver::bind("127.0.0.1:0").unwrap();
    let links = vec![
        (0, 1, vec![LinkProfile::thunderbolt5()]),
        (0, 2, vec![LinkProfile::thunderbolt5()]),
        (1, 2, vec![LinkProfile::thunderbolt5()]),
    ];
    let l = cfg.n_layers;
    let a = Assignment {
        engine: EngineKind::Tesseract,
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
        stages: vec![(0, 0, l / 2), (1, l / 2, l)],
        head: 1,
        hot_capacity: 0,
        inject_context: 0,
        tess: TesseractOptions {
            kv_mode: KvMode::Disaggregated,
            page_tokens: 16,
            tier0_pages: 1,
            braid_strands: 1,
            draft_ahead: 0,
            lookahead: false,
            ..real_opts()
        },
        driver_addr: driver.addr(),
        service_addrs: ws.iter().flat_map(|w| w.service_addrs()).collect(),
    };
    driver.assign(&a);
    (driver, a, ws)
}

#[test]
fn disaggregated_kv_serves_successive_requests() {
    let _serial = serial();
    let cfg = config_from_file(&model(SMOL_Q8), None).unwrap();
    let tok = tokenizer_for(&cfg);
    let enc = |s: &str| tok.encode(s);
    let a_prompt = enc("The history of the Roman Empire begins with the founding of the city, and over many centuries");
    let c_prompt = enc("Photosynthesis is the process by which green plants turn sunlight into");
    assert_ne!(
        a_prompt[0], c_prompt[0],
        "the second request must share no prefix"
    );
    let req = |p: &[u32]| GenerateRequest {
        record_logits: false,
        ..greedy(p, 32)
    };
    // fresh cluster: C alone
    let (d, a, _w) = disaggregated(&cfg);
    let fresh = d.generate(a.head, &req(&c_prompt));
    d.reset(&a);
    d.shutdown(&a);
    // A then C on one cluster: C's remote KV must not see A's tokens
    let (d, a, _w) = disaggregated(&cfg);
    let first = d.generate(a.head, &req(&a_prompt));
    let second = d.generate(a.head, &req(&c_prompt));
    // and A again (its prompt shares nothing with C either)
    let third = d.generate(a.head, &req(&a_prompt));
    d.reset(&a);
    d.shutdown(&a);
    eprintln!(
        "A: {:?}\nC after A: {:?}\nC fresh: {:?}",
        tok.decode(&first.tokens),
        tok.decode(&second.tokens),
        tok.decode(&fresh.tokens)
    );
    assert_eq!(
        second.tokens, fresh.tokens,
        "stale remote KV leaked into the next request"
    );
    assert_eq!(
        third.tokens, first.tokens,
        "the same request on a reused fabric differs"
    );
}

// --- server ------------------------------------------------------------------------

struct Child(std::process::Child);
impl Drop for Child {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Start `tesseract-serve`; returns (child, http addr).
fn serve(args: &[&str]) -> (Child, String) {
    let mut c = std::process::Command::new(env!("CARGO_BIN_EXE_tesseract-serve"))
        .args(["--http", "127.0.0.1:0"])
        .args(args)
        .stdout(std::process::Stdio::piped())
        .stderr(if std::env::var_os("CI").is_some() {
            // on CI the server's own log goes to the job log (crash diagnosis)
            std::process::Stdio::inherit()
        } else {
            std::process::Stdio::null()
        })
        .spawn()
        .unwrap();
    let mut out = BufReader::new(c.stdout.take().unwrap());
    let mut l1 = String::new();
    out.read_line(&mut l1).unwrap();
    let http = l1
        .trim()
        .strip_prefix("listening on http://")
        .unwrap()
        .to_string();
    std::thread::spawn(move || {
        let mut sink = String::new();
        let _ = out.read_to_string(&mut sink);
    });
    (Child(c), http)
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
fn server_continuous_batching_streams_concurrent_chats() {
    let _serial = serial();
    let path = model(QWEN25);
    let (_srv, http) = serve(&["--model", &path, "--local-nodes", "2", "--sessions", "4"]);
    let questions = [
        "Describe the ocean in two sentences.",
        "Name three planets.",
        "Che cos'è la pizza?",
        "Why is the sky blue?",
    ];
    let body = |q: &str| {
        format!(
            r#"{{"stream":true,"max_tokens":40,"temperature":0,"messages":[{{"role":"user","content":"{q}"}}]}}"#
        )
    };
    // each question alone first (the same engine, one session at a time)
    let alone: Vec<String> = questions
        .iter()
        .map(|q| stream_chat(&http, &body(q)).0)
        .collect();
    // then all four at once; the status endpoint answers mid-stream
    let hs: Vec<_> = questions
        .iter()
        .map(|q| {
            let (h, b) = (http.clone(), body(q));
            std::thread::spawn(move || stream_chat(&h, &b))
        })
        .collect();
    std::thread::sleep(Duration::from_millis(300));
    let t0 = Instant::now();
    let status = get(&http, "/v1/cluster/status");
    let status_ms = t0.elapsed().as_secs_f64() * 1e3;
    let got: Vec<(String, Vec<Instant>, Json)> =
        hs.into_iter().map(|h| h.join().unwrap()).collect();
    for (i, ((text, _, last), want)) in got.iter().zip(&alone).enumerate() {
        let tel = last.get("tesseract_telemetry").unwrap();
        eprintln!(
            "[{i}] batched={:?} {:?}",
            tel.get("continuous_batching"),
            text
        );
        assert_eq!(
            tel.get("continuous_batching").and_then(|b| b.as_bool()),
            Some(true)
        );
        assert_eq!(
            text, want,
            "request {i}: batched output differs from serving it alone"
        );
    }
    // the four streams overlap in time
    let first = got.iter().map(|g| g.1[0]).max().unwrap();
    let last = got.iter().map(|g| *g.1.last().unwrap()).min().unwrap();
    assert!(first < last, "streams were serialised");
    eprintln!(
        "status mid-stream in {status_ms:.0} ms: {} slot(s)",
        status
            .get("slots")
            .and_then(|s| s.as_array())
            .map(|s| s.len())
            .unwrap_or(0)
    );
    assert!(status_ms < 5000.0, "status blocked behind the streams");
}

#[test]
fn k_quant_kernels_reach_the_q8_class() {
    use tesseract_fabric::tensor::quant::*;
    use tesseract_fabric::tensor::{Rng, Tensor2D};
    let _serial = serial();
    let mut rng = Rng::new(1);
    let (n, k) = (1024usize, 2048usize);
    let w = Tensor2D::randn(n, k, 0.05, &mut rng);
    let x = Tensor2D::randn(1, k, 1.0, &mut rng);
    let rate = |fmt: WeightFormat| {
        let v = QuantView::from_f32(&w, fmt);
        let mut y = Tensor2D::zeros(1, n);
        let mut best = f64::MAX;
        for _ in 0..40 {
            let t = Instant::now();
            gemm_quant_with(&x, &v, &mut y, best_kernel());
            best = best.min(t.elapsed().as_secs_f64());
        }
        (n * k) as f64 / best / 1e9
    };
    let q8 = rate(WeightFormat::Q8_0);
    for fmt in [WeightFormat::Q4_K, WeightFormat::Q6_K, WeightFormat::Q5_0] {
        let r = rate(fmt);
        eprintln!("{}: {r:.1} GMAC/s at one row (Q8_0 {q8:.1})", fmt.name());
        if best_kernel() != DotKernel::Scalar && perf_asserts() {
            // K-quants stay in the Q8_0 class (within ~25% or better) on
            // every SIMD target; the absolute V10 target (≥ 13.5 GMAC/s on
            // one thread) was set on Apple Silicon and is checked there only,
            // since shared x86 CI runners reach ≈ 9 GMAC/s even for Q8_0
            assert!(
                r >= 0.75 * q8,
                "{}: {r:.1} GMAC/s vs Q8_0 {q8:.1}",
                fmt.name()
            );
            if cfg!(target_arch = "aarch64") {
                assert!(r >= 13.5, "{}: {r:.1} GMAC/s", fmt.name());
            }
        }
    }
}
