//! V8.0 production runtime: real model files over zero-copy `mmap`,
//! quantized kernels, BPE tokenization, UDP discovery and the OpenAI-compatible
//! HTTP / SSE server — all against real files and real sockets.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tesseract_fabric::cluster::discovery::*;
use tesseract_fabric::cluster::*;
use tesseract_fabric::loader::gguf::GgufFile;
use tesseract_fabric::loader::safetensors::{Dtype, SafeTensorsFile, SafeTensorsWriter};
use tesseract_fabric::loader::sharded_loader::*;
use tesseract_fabric::model::{argmax, logit_fidelity, ModelConfig, ReferenceModel, WeightSource};
use tesseract_fabric::server::json::Json;
use tesseract_fabric::tensor::quant::*;
use tesseract_fabric::tensor::{f16_to_f32, f32_to_f16, Rng, Tensor2D};
use tesseract_fabric::tokenizer::{synthetic_tokenizer, Tokenizer};
use tesseract_fabric::wire::multipath::LinkProfile;

/// A per-test scratch directory, removed on drop.
struct TempDir(PathBuf);
impl TempDir {
    fn new(name: &str) -> Self {
        let p = std::env::temp_dir().join(format!("tesseract-v8-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        TempDir(p)
    }
    fn path(&self, f: &str) -> String {
        self.0.join(f).to_string_lossy().into_owned()
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct Run {
    res: GenerateResult,
    stats: Vec<NodeStats>,
}

/// Run `cfg` on `n` loopback socket workers with the given engine layout.
fn run(
    cfg: &ModelConfig,
    n: u16,
    engine: EngineKind,
    stages: Vec<(u16, usize, usize)>,
    head: u16,
    tess: TesseractOptions,
    req: GenerateRequest,
) -> Run {
    let ids: Vec<u16> = (0..n).collect();
    let ws = spawn_local_workers(&ids);
    let driver = Driver::bind("127.0.0.1:0").unwrap();
    let mut links = Vec::new();
    for a in 0..n {
        for b in a + 1..n {
            links.push((a, b, vec![LinkProfile::thunderbolt5()]));
        }
    }
    let a = Assignment {
        engine,
        model: cfg.clone(),
        nodes: ws
            .iter()
            .map(|w| NodeSetup {
                id: w.id,
                addr: w.addr,
                slowdown: 1.0,
                kv_capacity: 256 << 20,
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
    let res = driver.generate(head, &req);
    std::thread::sleep(Duration::from_millis(50));
    let stats = driver.stats(&ids);
    driver.reset(&a);
    driver.shutdown(&a);
    Run { res, stats }
}

/// Teacher-forced fidelity of an engine's logits against the in-memory FP32
/// model: (mean centred cosine, top-1 agreement).
fn fidelity(reference: &ReferenceModel, prompt: &[u32], r: &GenerateResult) -> (f64, f64) {
    let mut seq = prompt.to_vec();
    seq.extend(&r.tokens);
    let rl = reference.forward_sequence(&seq);
    let (mut cos, mut agree) = (0.0, 0usize);
    for (i, l) in r.logits.iter().enumerate() {
        let row = rl.row(prompt.len() - 1 + i);
        cos += logit_fidelity(l, row);
        agree += (argmax(l) == argmax(row)) as usize;
    }
    (
        cos / r.logits.len() as f64,
        agree as f64 / r.logits.len() as f64,
    )
}

fn prompt(vocab: usize) -> Vec<u32> {
    (0..12).map(|i| (i * 53 % vocab) as u32).collect()
}

#[test]
fn gguf_roundtrip_mmap_sharded_inference() {
    let dir = TempDir::new("gguf");
    let cfg = ModelConfig::tiny();
    let tok = synthetic_tokenizer(cfg.vocab);
    let reference = ReferenceModel::new(&cfg);
    let p = prompt(cfg.vocab);
    let world = 4u16;
    for fmt in [WeightFormat::Q8_0, WeightFormat::Q4_0] {
        let path = dir.path(&format!("tiny-{}.gguf", fmt.name()));
        let size = export_gguf(&cfg, fmt, Some(&tok), &path).unwrap();
        // --- the file parses back with every hyper-parameter and the vocabulary
        let g = GgufFile::open(&path).unwrap();
        assert_eq!(g.version, 3);
        assert_eq!(g.map.len() as u64, size);
        assert!(g.map.is_mapped(), "GGUF must be memory-mapped, not read");
        assert_eq!(g.architecture(), Some("llama"));
        assert_eq!(g.arch_u64("block_count"), Some(cfg.n_layers as u64));
        assert_eq!(g.arch_u64("expert_count"), Some(4));
        assert_eq!(Tokenizer::from_gguf(&g).unwrap().tokens, tok.tokens);
        let v = g.view("blk.2.ffn_up_exps.weight", Some(3)).unwrap();
        assert_eq!((v.fmt, v.rows, v.cols), (fmt, 128, 128));
        let base = g.map.as_slice().as_ptr() as usize;
        assert!(
            v.is_mapped() && (v.source_ptr() as usize) == base,
            "expert view must alias the mapping"
        );
        let loaded = config_from_file(&path, None).unwrap();
        assert!(loaded.exact, "model files default to exact accumulation");
        // (generation-only synthetic fields and the dense-FFN width, unused by
        // MoE layers, are not model-file hyper-parameters)
        assert_eq!(
            ModelConfig {
                source: WeightSource::Synthetic,
                d_ff: cfg.d_ff,
                logit_std: cfg.logit_std,
                transition_prior: cfg.transition_prior,
                spectral_decay: cfg.spectral_decay,
                context_prior: cfg.context_prior,
                context_bias: cfg.context_bias,
                // V10: model files load with exact row-parallel accumulation
                exact: cfg.exact,
                ..loaded.clone()
            },
            cfg,
            "GGUF metadata must reproduce the model config"
        );
        // --- shard selectivity: an H-TEP rank addresses attention + 1/4 of experts
        let whole = shard_bytes(&loaded, 0..cfg.n_layers, None, false);
        let rank = shard_bytes(&loaded, 0..cfg.n_layers, Some((1, world as usize)), false);
        let layer0 = shard_bytes(&loaded, 0..1, None, false);
        assert!(
            rank * 2 < whole,
            "rank addresses {rank} of {whole} layer bytes"
        );
        assert!(layer0 * (cfg.n_layers as u64 - 1) < whole);
        // --- 4-node H-TEP over sockets with speculative decoding, from the file
        let r = run(
            &loaded,
            world,
            EngineKind::TesseractHtep,
            (0..world).map(|i| (i, 0, cfg.n_layers)).collect(),
            0,
            TesseractOptions {
                budget: 3,
                ..Default::default()
            },
            GenerateRequest {
                prompt: p.clone(),
                max_new: 32,
                seed: 5,
                temperature: 0.0,
                record_logits: true,
            },
        );
        assert_eq!(r.res.tokens.len(), 32);
        let (cos, top1) = fidelity(&reference, &p, &r.res);
        let weights: Vec<u64> = r.stats.iter().map(|s| s.weight_bytes).collect();
        eprintln!(
            "{}: file {:.2} MB · per-rank weights {:?} B · fidelity vs in-memory FP32: cosine {cos:.5}, top-1 {:.1}% · tokens/step {:.2}",
            fmt.name(),
            size as f64 / 1e6,
            weights,
            top1 * 100.0,
            32.0 / r.res.per_epoch.len() as f64
        );
        let commits: Vec<u64> = r.stats.iter().map(|s| s.commits).collect();
        assert!(
            commits.windows(2).all(|w| w[0] == w[1]),
            "ranks diverged: {commits:?}"
        );
        assert!(
            r.res.per_epoch.iter().any(|&k| k > 1),
            "speculation never accepted a draft"
        );
        // each rank holds a quarter of the expert bytes, not the model
        assert!(
            weights.iter().all(|&w| w > 0 && w * 2 < whole),
            "{weights:?} vs {whole}"
        );
        let (min_cos, min_top1) = match fmt {
            WeightFormat::Q8_0 => (0.995, 0.85),
            _ => (0.93, 0.55),
        };
        assert!(
            cos > min_cos && top1 >= min_top1,
            "{}: cosine {cos}, top-1 {top1}",
            fmt.name()
        );
    }
}

#[test]
fn safetensors_roundtrip_mmap() {
    let dir = TempDir::new("st");
    let mut rng = Rng::new(0x57);
    let a = Tensor2D::randn(48, 64, 1.0, &mut rng);
    let b = Tensor2D::randn(3 * 16, 32, 0.5, &mut rng);
    let c: Vec<f32> = (0..96).map(|i| (i as f32 - 40.0) * 0.37).collect();
    let raw: Vec<u8> = (0..=255u8).collect();
    let path = dir.path("t.safetensors");
    let mut w = SafeTensorsWriter::new();
    w.add_metadata("format", "pt");
    w.add_f32("a.f32", Dtype::F32, &[48, 64], &a.data);
    w.add_f32("b.f16", Dtype::F16, &[3, 16, 32], &b.data);
    w.add_f32("c.bf16", Dtype::BF16, &[96], &c);
    w.add("d.u8", Dtype::U8, &[256], raw.clone());
    let size = w.write(&path).unwrap();
    let f = SafeTensorsFile::open(&path).unwrap();
    assert!(f.map.is_mapped());
    assert_eq!(f.map.len() as u64, size);
    assert_eq!(f.metadata, vec![("format".to_string(), "pt".to_string())]);
    // header length keeps the data section 8-byte aligned
    let hdr = u64::from_le_bytes(f.map.as_slice()[..8].try_into().unwrap()) as usize;
    assert_eq!((8 + hdr) % 8, 0);
    // F32: bit-exact
    assert_eq!(f.to_f32("a.f32").unwrap(), a.data);
    // F16 / BF16: exactly the round-to-nearest-even value of every element
    let want16: Vec<f32> = b.data.iter().map(|&x| f16_to_f32(f32_to_f16(x))).collect();
    let got16: Vec<f32> = (0..3)
        .flat_map(|e| f.view("b.f16", Some(e)).unwrap().dequantize().data)
        .collect();
    assert_eq!(got16, want16);
    let wantbf: Vec<f32> = c.iter().map(|&x| bf16_to_f32(f32_to_bf16(x))).collect();
    assert_eq!(f.to_f32("c.bf16").unwrap(), wantbf);
    assert_eq!(f.bytes("d.u8").unwrap(), &raw[..]);
    // views alias the mapping (zero copy)
    let v = f.view("a.f32", None).unwrap();
    let base = f.map.as_slice().as_ptr() as usize;
    assert!(v.is_mapped() && v.source_ptr() as usize == base);
    let t = f.tensor("a.f32").unwrap();
    assert_eq!(v.byte_offset(), t.start);
    // corrupt headers are rejected, not trusted
    let mut bad = std::fs::read(&path).unwrap();
    bad[0..8].copy_from_slice(&(u64::MAX / 2).to_le_bytes());
    std::fs::write(dir.path("bad.safetensors"), &bad).unwrap();
    assert!(SafeTensorsFile::open(dir.path("bad.safetensors")).is_err());
    // a whole Hugging Face-style model directory (BF16) runs from the mapping
    let cfg = ModelConfig::tiny();
    let tok = synthetic_tokenizer(cfg.vocab);
    let mdir = dir.path("tiny-model");
    export_safetensors(&cfg, Dtype::BF16, Some(&tok), &mdir).unwrap();
    let loaded = config_from_file(&mdir, None).unwrap();
    assert_eq!(tokenizer_for(&loaded).tokens, tok.tokens);
    let seq = prompt(cfg.vocab);
    let l0 = ReferenceModel::new(&cfg).forward_sequence(&seq);
    let l1 = ReferenceModel::new(&loaded).forward_sequence(&seq);
    let cos: f64 = (0..seq.len())
        .map(|i| logit_fidelity(l1.row(i), l0.row(i)))
        .sum::<f64>()
        / seq.len() as f64;
    eprintln!("SafeTensors BF16 tiny model: cosine {cos:.6} vs FP32");
    assert!(cos > 0.9999, "BF16 SafeTensors model cosine {cos}");
}

#[test]
fn quantized_q4_0_and_q8_0_simd_accuracy_and_speed() {
    let mut rng = Rng::new(0x51);
    let (n, k) = (512usize, 1024usize);
    let w = Tensor2D::randn(n, k, 0.05, &mut rng);
    let x = Tensor2D::randn(4, k, 1.0, &mut rng);
    let kernel = best_kernel();
    eprintln!("integer-dot kernel: {}", kernel.name());
    for (fmt, max_rel, min_ratio) in [
        (WeightFormat::Q8_0, 1e-2, 3.7),
        (WeightFormat::Q4_0, 5e-2, 7.1),
    ] {
        let v = QuantView::from_f32(&w, fmt);
        // memory footprint vs FP32
        let ratio = (n * k * 4) as f64 / v.bytes() as f64;
        assert!(ratio >= min_ratio, "{}: {ratio:.2}× smaller", fmt.name());
        // fused 4-row GEMM vs FP32 math on the dequantized weights (f64)
        let deq = v.dequantize();
        let mut y = Tensor2D::zeros(4, n);
        gemm_quant_with(&x, &v, &mut y, kernel);
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for i in 0..4 {
            for j in 0..n {
                let r: f64 = (0..k)
                    .map(|t| x.row(i)[t] as f64 * deq.row(j)[t] as f64)
                    .sum();
                num += (y.data[i * n + j] as f64 - r).powi(2);
                den += r * r;
            }
        }
        let rel = (num / den).sqrt();
        // and vs the original FP32 weights (reported: includes weight rounding)
        let (mut num0, mut den0) = (0.0f64, 0.0f64);
        for i in 0..4 {
            for j in 0..n {
                let r: f64 = (0..k)
                    .map(|t| x.row(i)[t] as f64 * w.row(j)[t] as f64)
                    .sum();
                num0 += (y.data[i * n + j] as f64 - r).powi(2);
                den0 += r * r;
            }
        }
        // single-vector fused dots agree with the batched kernel
        let row = v.row_bytes(7);
        let d = match fmt {
            WeightFormat::Q8_0 => vec_dot_q8_0_q8_0(row, x.row(2)),
            _ => vec_dot_q4_0_q8_0(row, x.row(2)),
        };
        assert!((d - y.data[2 * n + 7]).abs() <= 1e-4 * (1.0 + d.abs()));
        // speed: SIMD vs scalar, and one 4-row pass vs four 1-row passes
        // the three variants are interleaved in every iteration so host
        // contention hits them alike; each keeps its best time
        let mut y4 = Tensor2D::zeros(4, n);
        let rows: Vec<Tensor2D> = (0..4)
            .map(|i| Tensor2D::from_vec(1, k, x.row(i).to_vec()))
            .collect();
        let mut y1 = Tensor2D::zeros(1, n);
        let (mut t_simd, mut t_scalar, mut t_four) = (f64::MAX, f64::MAX, f64::MAX);
        for _ in 0..60 {
            let t = Instant::now();
            gemm_quant_with(&x, &v, &mut y4, kernel);
            t_simd = t_simd.min(t.elapsed().as_secs_f64());
            let t = Instant::now();
            gemm_quant_with(&x, &v, &mut y4, DotKernel::Scalar);
            t_scalar = t_scalar.min(t.elapsed().as_secs_f64());
            let t = Instant::now();
            for r in &rows {
                gemm_quant_with(r, &v, &mut y1, kernel)
            }
            t_four = t_four.min(t.elapsed().as_secs_f64());
        }
        eprintln!(
            "{}: {ratio:.2}× smaller than FP32 · rel. error vs dequantized FP32 {rel:.2e} (vs original FP32 {:.2e}) · 4-row GEMM {:.0} µs ({}), scalar {:.0} µs, 4 × 1-row {:.0} µs",
            fmt.name(),
            (num0 / den0).sqrt(),
            t_simd * 1e6,
            kernel.name(),
            t_scalar * 1e6,
            t_four * 1e6
        );
        assert!(rel < max_rel, "{}: relative error {rel}", fmt.name());
        if kernel != DotKernel::Scalar {
            assert!(t_simd < t_scalar, "SIMD kernel slower than scalar");
        }
        assert!(
            t_simd < t_four,
            "4-row tile does not reuse decoded weight blocks"
        );
    }
}

#[test]
fn bpe_tokenizer_utf8_roundtrip() {
    let tok = synthetic_tokenizer(2048);
    assert_eq!(tok.vocab_size(), 2048);
    let texts = [
        "The quick brown fox jumps over the lazy dog, and the engine streams tokens.",
        "Perché è così? Città, caffè, più, già — l'università è là giù.",
        "fn main() {\n    let v: Vec<u32> = (0..10).map(|i| i * 2).collect();\n\tprintln!(\"{v:?}\");\n}\n",
        "Emoji: 🚀✨🤖 ❤️ 👩‍💻 🇮🇹 — CJK: 分布式推理 — math: ∑ x² ≤ π",
        "  leading spaces,   runs of   spaces\n\n\nand 2024-11-16 12:30:00 numbers 3.14159",
        "",
    ];
    for t in texts {
        let ids = tok.encode(t);
        assert_eq!(tok.decode(&ids), t, "round trip");
        // streaming: token by token, holding back incomplete UTF-8
        let mut s = tok.stream();
        let mut out = String::new();
        for &id in &ids {
            out.push_str(&s.push(id));
        }
        out.push_str(&s.finish());
        assert_eq!(out, t, "streaming round trip");
    }
    // multi-byte characters really do straddle token boundaries
    let rocket = tok.encode("🚀");
    assert!(rocket.len() > 1, "emoji is split into byte tokens");
    let mut s = tok.stream();
    assert_eq!(s.push(rocket[0]), "", "partial UTF-8 must be held back");
    // merges compress English and Italian below one token per byte
    let en = texts[0];
    assert!(
        tok.encode(en).len() * 2 < en.len(),
        "BPE merges barely compress"
    );
    // the vocabulary survives GGUF metadata and tokenizer.json
    let mut w = tesseract_fabric::loader::GgufWriter::new();
    for (k, v) in tok.gguf_metadata() {
        w.add_kv(&k, v);
    }
    let dir = TempDir::new("tok");
    let p = dir.path("tok.gguf");
    w.write(&p).unwrap();
    let g = Tokenizer::from_gguf(&GgufFile::open(&p).unwrap()).unwrap();
    let j = Tokenizer::from_hf_json(&Json::parse(&tok.to_hf_json().dump()).unwrap()).unwrap();
    for t in texts {
        assert_eq!(g.encode(t), tok.encode(t));
        assert_eq!(j.encode(t), tok.encode(t));
    }
    // SentencePiece-style (score merges + <0xNN> byte fallback) also round-trips
    let spm = tok.to_sentencepiece();
    for t in texts {
        assert_eq!(spm.decode(&spm.encode(t)), t, "SentencePiece round trip");
    }
}

#[test]
fn udp_lan_auto_discovery_and_probe() {
    let key = ClusterKey::from_secret("v8-test-cluster");
    // beacons are authenticated: tampering or another cluster's key fails
    let b = DiscoveryBeacon {
        cluster_id: key.cluster_id,
        node_id: 9,
        tcp_port: 7000,
        kv_port: 7001,
        draft_port: 7002,
        platform: tesseract_fabric::topology::Platform::DiscreteGpuWorkstation,
        compute_tflops_fp16: 104.8,
        mem_bandwidth_gbps: 1792.0,
        free_ram_mb: 32768,
        active_load: 0.5,
        time_ms: 1,
        seq: 1,
    };
    let enc = b.encode(&key);
    assert_eq!(enc.len(), 64);
    assert_eq!(DiscoveryBeacon::decode(&enc, &key), Some(b.clone()));
    let mut t = enc;
    t[20] ^= 1;
    assert_eq!(DiscoveryBeacon::decode(&t, &key), None);
    assert_eq!(
        DiscoveryBeacon::decode(&enc, &ClusterKey::from_secret("other")),
        None
    );

    let listener = DiscoveryListener::start("127.0.0.1:0", key, false).unwrap();
    let ids: Vec<u16> = vec![3, 5, 8];
    let ws: Vec<LocalWorker> = ids
        .iter()
        .map(|&i| start_worker(i, "127.0.0.1", 0).unwrap())
        .collect();
    let targets = AnnounceTargets {
        multicast: false,
        broadcast: false,
        port: listener.addr.port(),
        seeds: vec![listener.addr],
    };
    let profile = LocalProfile::detect();
    let _beacons: Vec<Announcer> = ws
        .iter()
        .map(|w| {
            Announcer::start(
                key,
                w.id,
                (w.addr.port(), w.kv_addr.port(), w.draft_addr.port()),
                profile.clone(),
                targets.clone(),
                Duration::from_millis(50),
            )
            .unwrap()
        })
        .collect();
    // an impostor with the wrong secret announces too, and must be ignored
    let _rogue = Announcer::start(
        ClusterKey::from_secret("intruder"),
        99,
        (1, 2, 3),
        profile.clone(),
        targets.clone(),
        Duration::from_millis(50),
    )
    .unwrap();
    let cfg = ModelConfig::tiny();
    let driver = Driver::bind("127.0.0.1:0").unwrap();
    let (probed, plan, a) = auto_cluster(
        &listener,
        &driver,
        &cfg,
        3,
        Duration::from_secs(20),
        TesseractOptions::default(),
    )
    .expect("bring-up");
    let found: Vec<u16> = probed.iter().map(|p| p.peer.beacon.node_id).collect();
    assert_eq!(found, ids, "discovered exactly the three workers");
    for p in &probed {
        eprintln!(
            "node {} @ {}: RTT {:.0} µs, {:.2} Gb/s ({}), {:.3} TFLOPS, {:.0} GB/s",
            p.peer.beacon.node_id,
            p.peer.node_addr(),
            p.link.rtt_us,
            p.link.bandwidth_gbps,
            p.link.profile().name(),
            p.peer.beacon.compute_tflops_fp16,
            p.peer.beacon.mem_bandwidth_gbps
        );
        assert_eq!(
            p.peer.node_addr(),
            ws.iter()
                .find(|w| w.id == p.peer.beacon.node_id)
                .unwrap()
                .addr
        );
        assert!(p.link.rtt_us > 0.0 && p.link.rtt_us < 100_000.0);
        assert!(p.link.bandwidth_gbps > 0.0);
    }
    assert!(
        listener.rejected.load(std::sync::atomic::Ordering::Relaxed) > 0,
        "rogue beacons were not rejected"
    );
    eprintln!(
        "solver: {} · stages {:?} · {}",
        a.engine.name(),
        a.stages,
        plan.notes.join("; ")
    );
    assert!(a.stages.iter().all(|s| ids.contains(&s.0)));
    // the auto-assigned cluster runs a distributed forward pass
    let p = prompt(cfg.vocab);
    let res = driver.generate(
        a.head,
        &GenerateRequest {
            prompt: p.clone(),
            max_new: 32,
            seed: 3,
            temperature: 0.0,
            record_logits: true,
        },
    );
    driver.reset(&a);
    driver.shutdown(&a);
    assert_eq!(res.tokens.len(), 32);
    let (cos, top1) = fidelity(&ReferenceModel::new(&cfg), &p, &res);
    eprintln!(
        "auto-assigned cluster fidelity vs FP32: cosine {cos:.5}, top-1 {:.1}%",
        top1 * 100.0
    );
    // the engine and rank count follow the measured RTTs (e.g. a 3-rank
    // H-TEP group, whose expert split must cover every hidden unit)
    assert!(
        cos > 0.999 && top1 >= 0.95,
        "auto cluster fidelity {cos} / {top1}"
    );
}

/// Kills the server child process when the test ends (or panics).
struct Child(std::process::Child);
impl Drop for Child {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn http(addr: &str, method: &str, path: &str, body: &str) -> (String, String) {
    let mut s = TcpStream::connect(addr).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(120))).unwrap();
    write!(s, "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len()).unwrap();
    let mut out = String::new();
    s.read_to_string(&mut out).unwrap();
    let (head, body) = out.split_once("\r\n\r\n").unwrap();
    (head.to_string(), body.to_string())
}

#[test]
fn openai_http_sse_streaming_e2e() {
    // a real Q8_0 GGUF model file served by the real tesseract-serve binary
    let dir = TempDir::new("serve");
    let cfg = ModelConfig::bench_moe_v2();
    let cfg = ModelConfig { n_layers: 4, ..cfg };
    let tok = synthetic_tokenizer(cfg.vocab);
    let path = dir.path("serve-q8.gguf");
    export_gguf(&cfg, WeightFormat::Q8_0, Some(&tok), &path).unwrap();
    let child = std::process::Command::new(env!("CARGO_BIN_EXE_tesseract-serve"))
        .args([
            "--http",
            "127.0.0.1:0",
            "--model",
            &path,
            "--local-nodes",
            "4",
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(if std::env::var_os("CI").is_some() {
            // on CI the server's own log goes to the job log (crash diagnosis)
            std::process::Stdio::inherit()
        } else {
            std::process::Stdio::null()
        })
        .spawn()
        .unwrap();
    let mut child = Child(child);
    let mut line = String::new();
    BufReader::new(child.0.stdout.take().unwrap())
        .read_line(&mut line)
        .unwrap();
    let addr = line
        .trim()
        .strip_prefix("listening on http://")
        .expect("server address line")
        .to_string();

    let (head, body) = http(&addr, "GET", "/v1/models", "");
    assert!(head.starts_with("HTTP/1.1 200"));
    let m = Json::parse(&body).unwrap();
    let desc = m.get("data").unwrap().as_array().unwrap()[0]
        .get("tesseract")
        .unwrap()
        .clone();
    assert_eq!(desc.get("origin").unwrap().as_str(), Some("GGUF"));
    assert_eq!(desc.get("weight_format").unwrap().as_str(), Some("Q8_0"));

    let req = r#"{"model":"x","stream":true,"max_tokens":40,"temperature":0.8,"top_p":0.95,"seed":11,
                  "messages":[{"role":"system","content":"You are concise."},{"role":"user","content":"Ciao! Spiegami l'inferenza distribuita 🚀"}]}"#;
    let mut s = TcpStream::connect(&addr).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(120))).unwrap();
    write!(s, "POST /v1/chat/completions HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{req}", req.len()).unwrap();
    let mut r = BufReader::new(s);
    let mut status = String::new();
    r.read_line(&mut status).unwrap();
    assert!(status.starts_with("HTTP/1.1 200"), "{status}");
    let mut ctype = String::new();
    loop {
        let mut h = String::new();
        r.read_line(&mut h).unwrap();
        if h.trim().is_empty() {
            break;
        }
        if h.to_ascii_lowercase().starts_with("content-type:") {
            ctype = h;
        }
    }
    assert!(ctype.contains("text/event-stream"));
    let (mut chunks, mut steps, mut streamed_tokens, mut done) = (0usize, 0usize, 0usize, false);
    let mut text = String::new();
    let mut last: Option<Json> = None;
    let t0 = Instant::now();
    let mut first_step_at = None;
    loop {
        let mut l = String::new();
        if r.read_line(&mut l).unwrap() == 0 {
            break;
        }
        let Some(data) = l.trim_end().strip_prefix("data: ") else {
            continue;
        };
        if data == "[DONE]" {
            done = true;
            break;
        }
        let c = Json::parse(data).expect("chunk is JSON");
        assert_eq!(
            c.get("object").unwrap().as_str(),
            Some("chat.completion.chunk")
        );
        chunks += 1;
        let ch = &c.get("choices").unwrap().as_array().unwrap()[0];
        if let Some(t) = ch
            .get("delta")
            .and_then(|d| d.get("content"))
            .and_then(|d| d.as_str())
        {
            text.push_str(t);
        }
        if let Some(st) = c.get("tesseract_step") {
            steps += 1;
            first_step_at.get_or_insert(t0.elapsed());
            let n = st.get("tokens").unwrap().as_f64().unwrap() as usize;
            assert!((1..=17).contains(&n), "a step accepts 1..=tree tokens");
            streamed_tokens += n;
        }
        last = Some(c);
    }
    assert!(done, "stream did not end with data: [DONE]");
    let last = last.unwrap();
    let tel = last
        .get("tesseract_telemetry")
        .expect("final chunk carries telemetry");
    let num = |k: &str| tel.get(k).and_then(|v| v.as_f64()).unwrap_or(-1.0);
    let usage = last.get("usage").unwrap();
    let completion = usage.get("completion_tokens").unwrap().as_f64().unwrap() as usize;
    eprintln!(
        "SSE: {chunks} chunks, {steps} verification steps, {streamed_tokens} tokens · {:.1} tok/s · TTFT {:.1} ms · {:.2} tokens/step · acceptance {:.2} · {:.0} wire B/token · act vs FP16 {:?} · KV nodes {} · engine {}",
        num("tokens_per_sec"),
        num("ttft_ms"),
        num("accepted_tokens_per_step"),
        num("speculative_acceptance_rate"),
        num("wire_bytes_per_token"),
        tel.get("klt_compression_ratio").and_then(|v| v.as_f64()),
        num("kv_fabric_nodes_used"),
        tel.get("engine").and_then(|v| v.as_str()).unwrap_or("?")
    );
    assert_eq!(
        steps as f64,
        num("verification_steps"),
        "one SSE chunk per verification step"
    );
    // EOS may end a sampled reply early; it is streamed but not counted
    let finish = last.get("choices").unwrap().as_array().unwrap()[0]
        .get("finish_reason")
        .and_then(|f| f.as_str())
        .unwrap()
        .to_string();
    assert!(streamed_tokens >= 1 && completion <= streamed_tokens && streamed_tokens <= 40);
    match finish.as_str() {
        "length" => assert_eq!(completion, 40),
        "stop" => assert_eq!(completion + 1, streamed_tokens),
        f => panic!("finish_reason {f}"),
    }
    assert!(
        num("tokens_per_sec") > 0.0 && num("ttft_ms") > 0.0 && num("wire_bytes_per_token") >= 0.0,
        "{}",
        tel.dump()
    );
    assert!(num("accepted_tokens_per_step") >= 1.0);
    assert!((0.0..=1.0).contains(&num("speculative_acceptance_rate")));
    assert!(num("kv_fabric_nodes_used") >= 1.0);
    eprintln!("streamed text ({finish}): {text:?}");
    // non-streaming request: a plain chat.completion object
    let (head, body) = http(
        &addr,
        "POST",
        "/v1/chat/completions",
        r#"{"messages":[{"role":"user","content":"hello"}],"max_tokens":12,"temperature":0}"#,
    );
    assert!(head.starts_with("HTTP/1.1 200"));
    let j = Json::parse(&body).unwrap();
    assert_eq!(j.get("object").unwrap().as_str(), Some("chat.completion"));
    assert!(j.get("tesseract_telemetry").is_some());
    // live cluster status: every discovered node, roles, links and KV pages
    let (head, body) = http(&addr, "GET", "/v1/cluster/status", "");
    assert!(head.starts_with("HTTP/1.1 200"));
    let st = Json::parse(&body).unwrap();
    let nodes = st.get("nodes").unwrap().as_array().unwrap();
    assert_eq!(nodes.len(), 4);
    assert!(nodes.iter().all(|n| n
        .get("link")
        .and_then(|l| l.get("rtt_us"))
        .and_then(|v| v.as_f64())
        .is_some_and(|r| r > 0.0)));
    let roles: Vec<String> = nodes
        .iter()
        .flat_map(|n| {
            n.get("roles")
                .unwrap()
                .as_array()
                .unwrap()
                .iter()
                .map(|r| r.as_str().unwrap().to_string())
        })
        .collect();
    assert!(
        roles
            .iter()
            .any(|r| r.starts_with("H-TEP Shard") || r.starts_with("Pipeline Stage")),
        "{roles:?}"
    );
    assert!(st.get("requests_served").unwrap().as_f64().unwrap() >= 2.0);
    // the solver may place a model this small on one node (no wire traffic);
    // any multi-node plan must show inter-node bytes
    let busy_nodes = nodes
        .iter()
        .filter(|n| {
            n.get("weight_bytes")
                .and_then(|w| w.as_f64())
                .unwrap_or(0.0)
                > 0.0
        })
        .count();
    if busy_nodes > 1 {
        assert!(
            num("wire_bytes_per_token") > 0.0,
            "{busy_nodes} weight-holding nodes but no wire traffic"
        );
    }
    // errors are JSON, not panics
    let (head, _) = http(&addr, "POST", "/v1/chat/completions", "{not json");
    assert!(head.starts_with("HTTP/1.1 400"));
    let (head, body) = http(&addr, "GET", "/", "");
    assert!(head.starts_with("HTTP/1.1 200") && body.contains("Tesseract Fabric"));
}
