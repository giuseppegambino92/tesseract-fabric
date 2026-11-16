//! `tesseract-serve` — OpenAI-compatible HTTP / SSE server over a live
//! tesseract-fabric cluster, plus a web dashboard and chat playground.
//!
//!   tesseract-serve [--http 127.0.0.1:8080] [--model tiny|bench|<file.gguf>|<safetensors dir>]
//!                   [--quant f16|bf16|q8_0|q4_0|q4_k|q6_k] [--local-nodes 4] [--slots 1]
//!                   [--discover N] [--cluster-secret S] [--discovery-bind ip:port] [--lan-broadcast]
//!                   [--engine auto|htep|pipeline] [--codec lossless|klt|fp16] [--max-tokens 256]
//!                   [--sessions N] [--draft-model <small.gguf>]
//!
//! `--sessions N` turns on continuous batching: one H-TEP engine serves up to
//! N concurrent requests in shared forward passes, each with its own KV slot
//! (prefix-cached across requests). `--draft-model` loads a small checkpoint
//! with the same vocabulary on the head to draft speculative chains (exact
//! verification: outputs are unchanged).
//!
//! With `--local-nodes N` (default 4) the server starts N worker nodes per
//! engine slot in this process on loopback; they announce themselves over
//! UDP (unicast to the server's listener) exactly as remote `tesseract-node
//! --discover` daemons do. With `--discover N` it waits for N remote workers
//! per slot; the listener joins the LAN multicast group only with
//! `--lan-broadcast`. Every worker is probed (RTT + 64 KB burst), the
//! topology is solved, and roles are assigned automatically. Workers that
//! start announcing later (or stop) are hot-joined (or dropped) between
//! requests. The bound addresses are printed on stdout:
//! `listening on http://…` and `discovery on udp …`.

use std::time::Duration;
use tesseract_fabric::cluster::discovery::{
    auto_cluster_with, slot_of, AnnounceTargets, Announcer, ClusterKey, DiscoveryListener,
    EngineChoice, LocalProfile, DISCOVERY_PORT, SLOT_STRIDE,
};
use tesseract_fabric::cluster::{start_worker, Driver, TesseractOptions};
use tesseract_fabric::loader::sharded_loader::{describe, resolve_model, tokenizer_for};
use tesseract_fabric::server::http::{Engine, HotJoin, Server, ServerState};
use tesseract_fabric::tensor::quant::WeightFormat;

fn arg(args: &[String], k: &str) -> Option<String> {
    args.iter()
        .position(|a| a == k)
        .and_then(|i| args.get(i + 1).cloned())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        eprintln!("usage: tesseract-serve [--http ip:port] [--model tiny|bench|PATH] [--quant FMT] [--local-nodes N | --discover N] [--slots S | --sessions N] [--draft-model PATH] [--engine auto|htep|pipeline] [--codec lossless|klt|fp16] [--cluster-secret S] [--discovery-bind ip:port] [--lan-broadcast] [--max-tokens N]");
        return;
    }
    tesseract_fabric::tensor::prefer_performance_cores();
    let http = arg(&args, "--http").unwrap_or_else(|| "127.0.0.1:8080".into());
    let spec = arg(&args, "--model")
        .or_else(|| arg(&args, "--gguf"))
        .or_else(|| arg(&args, "--safetensors"))
        .unwrap_or_else(|| "tiny".into());
    let requant = arg(&args, "--quant")
        .map(|q| WeightFormat::parse(&q).unwrap_or_else(|| panic!("unknown --quant {q}")));
    let cfg = resolve_model(&spec, requant).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1)
    });
    let sessions: u8 = arg(&args, "--sessions")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
        .min(64);
    let mut choice = arg(&args, "--engine")
        .map(|e| EngineChoice::parse(&e).unwrap_or_else(|| panic!("unknown --engine {e}")))
        .unwrap_or(EngineChoice::Auto);
    if sessions > 0 {
        // continuous batching runs on the H-TEP coordinator
        choice = EngineChoice::Htep;
    }
    // real checkpoints default to exact settings: lossless activations
    // between stages, f32 H-TEP reductions, prompt-lookup drafting
    let mut tess = TesseractOptions::default();
    if cfg.source.is_file() {
        tess.codec = 0;
        tess.htep_reduce_f32 = true;
        tess.draft_lookup = true;
    }
    tess.sessions = sessions;
    if let Some(d) = arg(&args, "--draft-model") {
        let p = std::fs::canonicalize(&d).unwrap_or_else(|e| {
            eprintln!("--draft-model {d}: {e}");
            std::process::exit(1)
        });
        tess.draft_model = p.to_string_lossy().into_owned();
        if sessions == 0 {
            tess.sessions = 1;
            choice = EngineChoice::Htep;
        }
    }
    match arg(&args, "--codec").as_deref() {
        Some("lossless") => tess.codec = 0,
        Some("klt") => tess.codec = 4,
        Some("fp16") => {
            tess.codec = 4;
            tess.codec_auto = true;
        }
        Some(c) => panic!("unknown --codec {c}"),
        None => {}
    }
    let secret = arg(&args, "--cluster-secret").unwrap_or_else(|| "tesseract-local".into());
    let key = ClusterKey::from_secret(&secret);
    let remote: Option<usize> = arg(&args, "--discover").and_then(|s| s.parse().ok());
    let n = remote.unwrap_or_else(|| {
        arg(&args, "--local-nodes")
            .and_then(|s| s.parse().ok())
            .unwrap_or(4)
    });
    let slots: usize = if tess.sessions > 0 {
        1
    } else {
        arg(&args, "--slots")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1)
            .clamp(1, 7)
    };
    let lan = args.iter().any(|a| a == "--lan-broadcast");
    let bind = arg(&args, "--discovery-bind").unwrap_or_else(|| {
        if remote.is_some() {
            format!("0.0.0.0:{DISCOVERY_PORT}")
        } else {
            "127.0.0.1:0".into()
        }
    });
    let listener = std::sync::Arc::new(
        DiscoveryListener::start(&bind, key, lan).expect("bind discovery listener"),
    );
    eprintln!(
        "discovery: listening for {n} workers × {slots} slot(s) on udp {}",
        listener.addr
    );
    // in-process workers announce to our listener like remote daemons would
    let mut _workers = Vec::new();
    let mut _beacons = Vec::new();
    if remote.is_none() {
        let profile = LocalProfile::detect();
        for s in 0..slots as u16 {
            for i in 0..n as u16 {
                let id = s * SLOT_STRIDE + i;
                let w = start_worker(id, "127.0.0.1", 0).expect("start worker");
                let targets = AnnounceTargets {
                    multicast: false,
                    broadcast: false,
                    port: listener.addr.port(),
                    seeds: vec![listener.addr],
                };
                _beacons.push(
                    Announcer::start(
                        key,
                        id,
                        (w.addr.port(), w.kv_addr.port(), w.draft_addr.port()),
                        profile.clone(),
                        targets,
                        Duration::from_millis(200),
                    )
                    .expect("start beacon"),
                );
                _workers.push(w);
            }
        }
    }
    let mut engines = Vec::new();
    for s in 0..slots {
        let driver = Driver::bind("127.0.0.1:0").expect("bind driver");
        let timeout = Duration::from_secs(if remote.is_some() { 600 } else { 60 });
        let (probed, plan, a) = auto_cluster_with(
            &listener,
            &driver,
            &cfg,
            n,
            timeout,
            tess.clone(),
            choice,
            &|id| slot_of(id) == s,
        )
        .unwrap_or_else(|e| {
            eprintln!("slot {s}: cluster bring-up failed: {e}");
            std::process::exit(1)
        });
        for p in &probed {
            eprintln!(
                "slot {s} · node {} @ {}: {} · {:.2} TFLOPS · {:.0} GB/s · {} MB · RTT {:.0} µs · {:.1} Gb/s",
                p.peer.beacon.node_id,
                p.peer.node_addr(),
                p.peer.beacon.platform.name(),
                p.peer.beacon.compute_tflops_fp16,
                p.peer.beacon.mem_bandwidth_gbps,
                p.peer.beacon.free_ram_mb,
                p.link.rtt_us,
                p.link.bandwidth_gbps
            );
        }
        eprintln!(
            "slot {s} · engine: {} · stages {:?} · head {} · {}",
            a.engine.name(),
            a.stages,
            a.head,
            plan.notes.join("; ")
        );
        engines.push(if tess.sessions > 0 {
            Engine::batched(driver, a, probed)
        } else {
            Engine::direct(driver, a, probed)
        });
    }
    let tok = tokenizer_for(&cfg);
    let model_id = match &cfg.source {
        tesseract_fabric::model::WeightSource::Synthetic => format!("tesseract-synthetic-{spec}"),
        tesseract_fabric::model::WeightSource::File { path, .. } => std::path::Path::new(path)
            .file_name()
            .map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.clone()),
    };
    let mut state = ServerState::new(engines, tok, &model_id, describe(&cfg));
    eprintln!("chat template: {}", state.chat.name());
    if tess.sessions > 0 {
        eprintln!(
            "continuous batching: up to {} sessions{}",
            tess.sessions,
            if tess.draft_model.is_empty() {
                String::new()
            } else {
                format!(" · draft model {}", tess.draft_model)
            }
        );
    }
    if let Some(m) = arg(&args, "--max-tokens").and_then(|s| s.parse().ok()) {
        state.default_max_tokens = m;
    }
    state.hot_join = Some(HotJoin {
        listener: listener.clone(),
        tess,
        choice,
        leave_after: Duration::from_secs(3),
    });
    let server = Server::bind(&http, state).expect("bind http");
    println!("listening on http://{}", server.local_addr());
    if secret == "tesseract-local" {
        println!(
            "discovery on udp {} (default cluster secret {:?})",
            listener.addr, secret
        );
    } else {
        println!("discovery on udp {} (custom cluster secret)", listener.addr);
    }
    use std::io::Write;
    let _ = std::io::stdout().flush();
    server.serve();
}
