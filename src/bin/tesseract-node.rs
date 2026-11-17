//! `tesseract-node` — run a live worker, or coordinate a cluster of workers.
//!
//! Worker (one per device; also binds its KV service on port+1 and its draft
//! service on port+2):
//!   tesseract-node --role worker --id 0 --listen 0.0.0.0:7000
//!
//! Coordinator (assigns roles, runs generation, prints telemetry):
//!   tesseract-node --role coordinator --listen 10.0.0.1:6999 \
//!       --peers 10.0.0.2:7000,10.0.0.3:7000 --profile studio-m5max,rtx5090 \
//!       [--engine tesseract|exo|tp|all] [--model bench|tiny|long] [--tokens 64]
//!       [--temperature 1.0] [--emulate-links] [--holo-bits 8] [--budget 0]
//!
//! Peer i gets node id i. Without `--emulate-links` links are left unshaped,
//! i.e. the real network is the link model.
//!
//! V8.0 additions:
//!   * real weights: `--model <file.gguf | safetensors dir>` (or `--gguf` /
//!     `--safetensors`) on the coordinator, optional `--quant q8_0|q4_0|f16|bf16`
//!     re-quantization at load; a worker's own `--gguf/--safetensors <path>`
//!     overrides the coordinator's path with its local copy;
//!   * zero-config discovery: `tesseract-node --role worker --id 3 --discover
//!     [--seeds 100.64.0.1:49777,...] [--cluster-secret S]` announces the node
//!     over UDP multicast / broadcast / unicast seeds; use `tesseract-serve
//!     --discover N` (or `--role coordinator --discover N`) to form the cluster;
//!   * export: `tesseract-node --role export --model tiny|bench --format
//!     gguf|safetensors --quant q8_0 --out PATH` writes a synthetic model as a
//!     real GGUF file or Hugging Face SafeTensors directory.

use std::net::SocketAddr;
use std::time::Duration;
use tesseract_fabric::cluster::discovery::{
    AnnounceTargets, Announcer, ClusterKey, DiscoveryListener, LocalProfile, DISCOVERY_PORT,
};
use tesseract_fabric::cluster::{
    start_worker, Assignment, Driver, EngineKind, GenerateRequest, NodeSetup, TesseractOptions,
    DRAFT_SERVICE_BASE, KV_SERVICE_BASE,
};
use tesseract_fabric::loader::sharded_loader::{
    export_gguf, export_safetensors, resolve_model, set_local_model_path,
};
use tesseract_fabric::model::ModelConfig;
use tesseract_fabric::tensor::quant::WeightFormat;
use tesseract_fabric::topology::profiles::{links_between, slowdowns};
use tesseract_fabric::topology::{
    footprint, memory_weighted_partition, preset, HeteroClusterSolver, LinkSpec, SolverConfig,
    PRESETS,
};
use tesseract_fabric::wire::multipath::LinkProfile;

fn arg(args: &[String], k: &str) -> Option<String> {
    args.iter()
        .position(|a| a == k)
        .and_then(|i| args.get(i + 1).cloned())
}

fn usage() -> ! {
    eprintln!("usage:\n  tesseract-node --role worker --id <n> --listen <ip:port> [--discover [--seeds a,b] [--cluster-secret S]] [--gguf PATH | --safetensors PATH]\n  tesseract-node --role coordinator --discover <N> [--model tiny|bench|PATH] [--quant q8_0|q4_0|f16|bf16]\n  tesseract-node --role export --model tiny|bench --format gguf|safetensors [--quant q8_0|q4_0|q4_k|q6_k|f16|bf16] --out PATH\n  tesseract-node --role chat --model FILE --prompt TEXT [--system TEXT] [--local-nodes 4] [--engine auto|htep|pipeline] [--draft-model SMALL.gguf] [--tokens 128] [--temperature 0]\n  tesseract-node --role coordinator --listen <ip:port> --peers <a,b,..> --profile <p,q,..> [--engine tesseract|exo|tp|all] [--model bench|tiny|long] [--tokens N] [--temperature T] [--emulate-links] [--holo-bits B] [--budget N]");
    eprintln!(
        "profiles: {}",
        PRESETS.iter().map(|p| p.key).collect::<Vec<_>>().join(", ")
    );
    std::process::exit(2)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    // the cluster secret authenticates every node-to-node TCP connection
    // (`wire::auth`), not only discovery beacons
    let secret = arg(&args, "--cluster-secret");
    let plaintext = args.iter().any(|a| a == "--wire-plaintext");
    if let Some(s) = secret.as_deref() {
        // authenticates node connections and, with a secret of our own,
        // encrypts them (X25519 + ChaCha20-Poly1305) unless --wire-plaintext
        tesseract_fabric::wire::auth::configure(s, plaintext);
        if matches!(
            arg(&args, "--role").as_deref(),
            Some("worker") | Some("coordinator")
        ) {
            eprintln!(
                "node traffic: authenticated{}",
                if plaintext || s == tesseract_fabric::wire::auth::DEFAULT_SECRET {
                    ", not encrypted"
                } else {
                    " and encrypted (X25519 forward secrecy + ChaCha20-Poly1305)"
                }
            );
        }
    }
    let exposed = arg(&args, "--listen")
        .and_then(|l| l.parse::<std::net::SocketAddr>().ok())
        .is_some_and(|a| !a.ip().is_loopback());
    if exposed && secret.is_none() {
        eprintln!(
            "warning: listening beyond loopback with the public default cluster secret; \
             any host that can reach this port can join the cluster. Pass the same \
             --cluster-secret to every node (it also turns on encryption of node traffic)."
        );
    }
    match arg(&args, "--role").as_deref() {
        Some("worker") => worker(&args),
        Some("coordinator") if args.iter().any(|a| a == "--discover") => {
            discover_coordinator(&args)
        }
        Some("coordinator") => coordinator(&args),
        Some("export") => export(&args),
        Some("chat") => chat(&args),
        _ => usage(),
    }
}

/// One chat turn on N local nodes: the prompt is formatted with the model's
/// chat template and the reply streams as each verification step lands.
fn chat(args: &[String]) {
    use std::io::Write;
    use tesseract_fabric::cluster::discovery::{
        plan_assignment_with, probe_all, DiscoveryBeacon, EngineChoice, LocalProfile as LP, Peer,
    };
    use tesseract_fabric::cluster::GenerateExt;
    use tesseract_fabric::tokenizer::{stop_tokens, ChatFormat, SamplingParams};
    let cfg = load_model(args);
    let prompt = arg(args, "--prompt").unwrap_or_else(|| usage());
    let n: u16 = arg(args, "--local-nodes")
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let draft = arg(args, "--draft-model");
    // draft-model chains run on the H-TEP coordinator's session scheduler
    let choice = if draft.is_some() {
        EngineChoice::Htep
    } else {
        arg(args, "--engine")
            .and_then(|e| EngineChoice::parse(&e))
            .unwrap_or(EngineChoice::Auto)
    };
    let tok = tesseract_fabric::loader::sharded_loader::tokenizer_for(&cfg);
    let fmt = ChatFormat::detect(&tok);
    let mut msgs = Vec::new();
    if let Some(sys) = arg(args, "--system") {
        msgs.push(("system".to_string(), sys));
    }
    msgs.push(("user".to_string(), prompt));
    let ids = fmt.encode(&tok, &msgs);
    // local workers, probed and planned like discovered ones
    let profile = LP::detect();
    let ws: Vec<_> = (0..n)
        .map(|i| start_worker(i, "127.0.0.1", 0).expect("worker"))
        .collect();
    let driver = Driver::bind("127.0.0.1:0").expect("driver");
    let peers: Vec<Peer> = ws
        .iter()
        .map(|w| Peer {
            beacon: DiscoveryBeacon {
                cluster_id: 0,
                node_id: w.id,
                tcp_port: w.addr.port(),
                kv_port: w.kv_addr.port(),
                draft_port: w.draft_addr.port(),
                platform: profile.platform,
                compute_tflops_fp16: profile.compute_tflops_fp16,
                mem_bandwidth_gbps: profile.mem_bandwidth_gbps,
                free_ram_mb: profile.free_ram_mb,
                active_load: 0.0,
                time_ms: 0,
                seq: 0,
            },
            ip: w.addr.ip(),
            last_seen: std::time::Instant::now(),
        })
        .collect();
    let probed = probe_all(&driver, peers).expect("probe");
    let mut tess = TesseractOptions::default();
    if cfg.source.is_file() {
        tess.codec = 0;
        tess.htep_reduce_f32 = true;
        tess.draft_lookup = true;
    }
    if let Some(d) = draft {
        tess.sessions = 1;
        tess.draft_model = std::fs::canonicalize(&d)
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or(d);
    }
    let (_, a) = plan_assignment_with(&cfg, &probed, driver.addr(), tess, choice);
    driver.assign(&a);
    eprintln!(
        "[{} · {} · stages {:?} · chat template {}]",
        a.engine.name(),
        cfg.n_layers,
        a.stages,
        fmt.name()
    );
    let ext = GenerateExt {
        stream: true,
        stop: stop_tokens(&tok),
        sampling: Some(SamplingParams {
            temperature: arg(args, "--temperature")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0),
            top_p: arg(args, "--top-p")
                .and_then(|s| s.parse().ok())
                .unwrap_or(1.0),
            seed: 7,
            ..Default::default()
        }),
    };
    let mut dec = tok.stream();
    let r = driver.generate_stream(
        a.head,
        &GenerateRequest {
            prompt: ids,
            max_new: arg(args, "--tokens")
                .and_then(|s| s.parse().ok())
                .unwrap_or(128),
            seed: 7,
            temperature: 0.0,
            record_logits: false,
        },
        &ext,
        &mut |b| {
            for &t in &b.tokens {
                if !ext.stop.contains(&t) {
                    print!("{}", dec.push(t));
                }
            }
            let _ = std::io::stdout().flush();
        },
    );
    println!("{}", dec.finish());
    let s = r.decode_ns as f64 * 1e-9;
    eprintln!(
        "[{} tokens · {:.1} tok/s · {:.2} tokens/step]",
        r.tokens.len(),
        r.tokens.len() as f64 / s,
        r.tokens.len() as f64 / r.per_epoch.len().max(1) as f64
    );
    driver.reset(&a);
    driver.shutdown(&a);
}

fn model_spec(args: &[String]) -> String {
    arg(args, "--model")
        .or_else(|| arg(args, "--gguf"))
        .or_else(|| arg(args, "--safetensors"))
        .unwrap_or_else(|| "bench".into())
}

fn quant(args: &[String]) -> Option<WeightFormat> {
    arg(args, "--quant")
        .filter(|q| !q.eq_ignore_ascii_case("q4_k_m"))
        .map(|q| WeightFormat::parse(&q).unwrap_or_else(|| usage()))
}

fn load_model(args: &[String]) -> ModelConfig {
    resolve_model(&model_spec(args), quant(args)).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1)
    })
}

fn export(args: &[String]) {
    let spec = arg(args, "--model").unwrap_or_else(|| "tiny".into());
    let cfg = resolve_model(&spec, None).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1)
    });
    let out = arg(args, "--out").unwrap_or_else(|| usage());
    let tok = tesseract_fabric::loader::sharded_loader::tokenizer_for(&cfg);
    let fmt = quant(args).unwrap_or(WeightFormat::F32);
    let t = std::time::Instant::now();
    let bytes = match arg(args, "--format").as_deref() {
        Some("safetensors") => {
            use tesseract_fabric::loader::safetensors::Dtype;
            let dt = match fmt {
                WeightFormat::F16 => Dtype::F16,
                WeightFormat::BF16 => Dtype::BF16,
                WeightFormat::F32 => Dtype::F32,
                f => {
                    eprintln!("SafeTensors stores F32 / F16 / BF16, not {}", f.name());
                    std::process::exit(2)
                }
            };
            export_safetensors(&cfg, dt, Some(&tok), &out)
        }
        _ => match arg(args, "--quant")
            .as_deref()
            .and_then(tesseract_fabric::loader::sharded_loader::QuantMix::parse)
        {
            Some(mix) => tesseract_fabric::loader::sharded_loader::export_gguf_mix(
                &cfg,
                mix,
                Some(&tok),
                &out,
            ),
            None => export_gguf(&cfg, fmt, Some(&tok), &out),
        },
    }
    .expect("export");
    let label = arg(args, "--quant")
        .map(|q| q.to_uppercase())
        .unwrap_or_else(|| fmt.name().to_string());
    println!(
        "wrote {out}: {:.1} MB in {:.2} s ({label} weights, {} params)",
        bytes as f64 / 1e6,
        t.elapsed().as_secs_f64(),
        cfg.total_params()
    );
}

/// Coordinator that forms its cluster from UDP beacons.
fn discover_coordinator(args: &[String]) {
    let cfg = load_model(args);
    let n: usize = arg(args, "--discover")
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| usage());
    let key = ClusterKey::from_secret(
        &arg(args, "--cluster-secret").unwrap_or_else(|| "tesseract-local".into()),
    );
    let bind = arg(args, "--discovery-bind").unwrap_or_else(|| format!("0.0.0.0:{DISCOVERY_PORT}"));
    let listener =
        DiscoveryListener::start(&bind, key, args.iter().any(|a| a == "--lan-broadcast"))
            .expect("bind discovery");
    let driver = Driver::bind(&arg(args, "--listen").unwrap_or_else(|| "0.0.0.0:0".into()))
        .expect("bind coordinator");
    eprintln!("waiting for {n} workers on udp {} …", listener.addr);
    let (probed, plan, a) = tesseract_fabric::cluster::discovery::auto_cluster(
        &listener,
        &driver,
        &cfg,
        n,
        Duration::from_secs(600),
        TesseractOptions::default(),
    )
    .unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1)
    });
    for p in &probed {
        println!(
            "node {} @ {}: RTT {:.0} µs, {:.1} Gb/s, {}",
            p.peer.beacon.node_id,
            p.peer.node_addr(),
            p.link.rtt_us,
            p.link.bandwidth_gbps,
            p.peer.beacon.platform.name()
        );
    }
    println!("plan: {} | {}", a.engine.name(), plan.notes.join("; "));
    let tokens: u32 = arg(args, "--tokens")
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let prompt: Vec<u32> = (0..16).map(|i| (i * 977 % cfg.vocab) as u32).collect();
    let r = driver.generate(
        a.head,
        &GenerateRequest {
            prompt,
            max_new: tokens,
            seed: 7,
            temperature: arg(args, "--temperature")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0),
            record_logits: false,
        },
    );
    let s = r.decode_ns as f64 * 1e-9;
    println!(
        "{} tokens in {:.3} s: {:.1} tok/s, {:.2} tokens / step",
        r.tokens.len(),
        s,
        r.tokens.len() as f64 / s,
        r.tokens.len() as f64 / r.per_epoch.len().max(1) as f64
    );
    driver.reset(&a);
}

fn worker(args: &[String]) {
    tesseract_fabric::tensor::prefer_performance_cores();
    let id: u16 = arg(args, "--id")
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| usage());
    let listen: SocketAddr = arg(args, "--listen")
        .unwrap_or_else(|| "0.0.0.0:7000".into())
        .parse()
        .expect("--listen ip:port");
    if let Some(p) = arg(args, "--gguf").or_else(|| arg(args, "--safetensors")) {
        set_local_model_path(Some(p));
    }
    let w = start_worker(id, &listen.ip().to_string(), listen.port()).expect("bind worker");
    eprintln!(
        "tesseract-node worker {id}: node {} · KV service {} · draft service {}",
        w.addr, w.kv_addr, w.draft_addr
    );
    let _beacon = args.iter().any(|a| a == "--discover").then(|| {
        let key = ClusterKey::from_secret(
            &arg(args, "--cluster-secret").unwrap_or_else(|| "tesseract-local".into()),
        );
        let port: u16 = arg(args, "--discovery-port")
            .and_then(|s| s.parse().ok())
            .unwrap_or(DISCOVERY_PORT);
        let seeds = arg(args, "--seeds")
            .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
            .unwrap_or_default();
        let profile = LocalProfile::detect();
        let lan = args.iter().any(|a| a == "--lan-broadcast");
        eprintln!(
            "announcing on udp {port} ({}{} seeds): {} · {:.2} TFLOPS · {:.0} GB/s · {} MB",
            if lan {
                "LAN multicast + broadcast + "
            } else {
                ""
            },
            arg(args, "--seeds")
                .map(|s| s.split(',').count())
                .unwrap_or(0),
            profile.platform.name(),
            profile.compute_tflops_fp16,
            profile.mem_bandwidth_gbps,
            profile.free_ram_mb
        );
        Announcer::start(
            key,
            id,
            (w.addr.port(), w.kv_addr.port(), w.draft_addr.port()),
            profile,
            AnnounceTargets {
                port,
                seeds,
                multicast: lan,
                broadcast: lan,
            },
            Duration::from_secs(1),
        )
        .expect("start beacon")
    });
    for h in w.handles {
        let _ = h.join();
    }
}

fn coordinator(args: &[String]) {
    let listen = arg(args, "--listen").unwrap_or_else(|| "127.0.0.1:0".into());
    let peers: Vec<SocketAddr> = arg(args, "--peers")
        .unwrap_or_else(|| usage())
        .split(',')
        .map(|s| s.trim().parse().expect("peer ip:port"))
        .collect();
    let profiles: Vec<&'static tesseract_fabric::topology::DevicePreset> = arg(args, "--profile")
        .unwrap_or_else(|| usage())
        .split(',')
        .map(|k| preset(k.trim()).unwrap_or_else(|| usage()))
        .collect();
    assert_eq!(peers.len(), profiles.len(), "one --profile per peer");
    let cfg = load_model(args);
    let tokens: u32 = arg(args, "--tokens")
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let temperature: f32 = arg(args, "--temperature")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    let emulate = args.iter().any(|a| a == "--emulate-links");
    let n = peers.len() as u16;
    let ids: Vec<u16> = (0..n).collect();
    // links: real network unless emulation requested
    let mut links = Vec::new();
    for a in 0..n {
        for b in a + 1..n {
            let ps = if emulate {
                links_between(profiles[a as usize], profiles[b as usize], false)
            } else {
                vec![LinkProfile::unshaped()]
            };
            links.push((a, b, ps));
        }
    }
    let specs: Vec<_> = ids
        .iter()
        .map(|&i| {
            let ls = ids
                .iter()
                .filter(|&&j| j != i)
                .map(|&j| LinkSpec {
                    peer: j,
                    profiles: links_between(profiles[i as usize], profiles[j as usize], false),
                })
                .collect();
            let mut spec = profiles[i as usize].spec(i, ls);
            if !emulate {
                // real machines: plan against the device's RAM, not the
                // emulated benchmark's scaled-down capacity
                spec.available_ram_mb = profiles[i as usize].ram_mb;
            }
            spec
        })
        .collect();
    let slow = if emulate {
        slowdowns(&specs)
    } else {
        vec![1.0; peers.len()]
    };
    let mut tess = TesseractOptions::default();
    if let Some(b) = arg(args, "--holo-bits").and_then(|s| s.parse().ok()) {
        tess.holo_bits = b;
    }
    if let Some(b) = arg(args, "--budget").and_then(|s| s.parse().ok()) {
        tess.budget = b;
    }
    let plan = HeteroClusterSolver::new(
        specs.clone(),
        footprint(&cfg, tess.holo_bits),
        SolverConfig::default(),
    )
    .solve();
    tess.use_plan_drafter(plan.drafter, plan.coordinator);
    let kv_caps: Vec<u64> = ids
        .iter()
        .map(|id| {
            plan.kv_pools
                .iter()
                .find(|p| p.0 == *id)
                .map(|p| p.1 as u64)
                .unwrap_or(64 << 20)
        })
        .collect();
    let driver = Driver::bind(&listen).expect("bind coordinator");
    let service_addrs: Vec<(u16, SocketAddr)> = peers
        .iter()
        .enumerate()
        .flat_map(|(i, a)| {
            [
                (
                    KV_SERVICE_BASE + i as u16,
                    SocketAddr::new(a.ip(), a.port() + 1),
                ),
                (
                    DRAFT_SERVICE_BASE + i as u16,
                    SocketAddr::new(a.ip(), a.port() + 2),
                ),
            ]
        })
        .collect();
    let engines: Vec<EngineKind> = match arg(args, "--engine").as_deref() {
        Some("exo") => vec![EngineKind::ExoRing],
        Some("tp") => vec![EngineKind::TensorParallel],
        Some("all") => vec![
            EngineKind::ExoRing,
            EngineKind::TensorParallel,
            EngineKind::Tesseract,
        ],
        _ => vec![EngineKind::Tesseract],
    };
    let tess_engine = if plan.htep.is_some() {
        EngineKind::TesseractHtep
    } else {
        EngineKind::Tesseract
    };
    let engines: Vec<EngineKind> = engines
        .into_iter()
        .map(|e| {
            if e == EngineKind::Tesseract {
                tess_engine
            } else {
                e
            }
        })
        .collect();
    let ram: Vec<f64> = specs.iter().map(|s| s.available_ram_mb).collect();
    println!(
        "cluster: {}",
        specs
            .iter()
            .zip(&peers)
            .map(|(s, a)| format!("[{} {} @ {a}]", s.node_id, s.name))
            .collect::<Vec<_>>()
            .join(" ")
    );
    println!(
        "solver plan: {} | coordinator {} | drafter {:?}",
        plan.stages
            .iter()
            .map(|s| format!("node {} L{}–{}", s.node, s.layers.start, s.layers.end - 1))
            .collect::<Vec<_>>()
            .join(" → "),
        plan.coordinator,
        plan.drafter
    );
    let rounds: usize = arg(args, "--rounds")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
        .max(1);
    // measured links (RTT pings + 64 KB burst over the real endpoints)
    let measured: Vec<Option<tesseract_fabric::cluster::discovery::LinkMeasurement>> = ids
        .iter()
        .map(|&i| tesseract_fabric::cluster::discovery::probe(&driver, i, peers[i as usize]))
        .collect();
    for (i, m) in measured.iter().enumerate() {
        match m {
            Some(m) => println!(
                "link to node {i}: RTT {:.0} µs · {:.2} Gb/s (measured)",
                m.rtt_us, m.bandwidth_gbps
            ),
            None => println!("link to node {i}: probe failed"),
        }
    }
    let prompt: Vec<u32> = (0..16).map(|i| (i * 977 % cfg.vocab) as u32).collect();
    // one assignment per engine (or why it cannot run)
    let mut runs: Vec<(EngineKind, Result<Assignment, String>)> = Vec::new();
    for engine in engines {
        if engine == EngineKind::TensorParallel
            && (!cfg.n_heads.is_multiple_of(peers.len())
                || !cfg.n_kv_heads.is_multiple_of(peers.len()))
        {
            runs.push((
                engine,
                Err(format!(
                    "{} q / {} kv heads do not divide across {} ranks",
                    cfg.n_heads,
                    cfg.n_kv_heads,
                    peers.len()
                )),
            ));
            continue;
        }
        let (stages, head) = match engine {
            EngineKind::ExoRing => {
                let counts = memory_weighted_partition(cfg.n_layers, &ram);
                let mut l = 0;
                let st: Vec<(u16, usize, usize)> = ids
                    .iter()
                    .zip(&counts)
                    .map(|(&i, &c)| {
                        let s = (i, l, l + c);
                        l += c;
                        s
                    })
                    .collect();
                let h = st.last().unwrap().0;
                (st, h)
            }
            EngineKind::TensorParallel => (ids.iter().map(|&i| (i, 0, 0)).collect(), 0),
            EngineKind::Tesseract | EngineKind::TesseractHtep => (
                plan.stages
                    .iter()
                    .map(|s| (s.node, s.layers.start, s.layers.end))
                    .collect(),
                plan.coordinator,
            ),
        };
        runs.push((
            engine,
            Ok(Assignment {
                engine,
                model: cfg.clone(),
                nodes: ids
                    .iter()
                    .map(|&i| NodeSetup {
                        id: i,
                        addr: peers[i as usize],
                        slowdown: slow[i as usize],
                        kv_capacity: kv_caps[i as usize],
                    })
                    .collect(),
                links: links.clone(),
                stages,
                head,
                hot_capacity: cfg
                    .moe
                    .as_ref()
                    .map(|m| (m.n_experts * 3 / 4) as u16)
                    .unwrap_or(0),
                inject_context: 0,
                tess: tess.clone(),
                driver_addr: driver.addr(),
                service_addrs: service_addrs.clone(),
            }),
        ));
    }
    // interleaved rounds: every engine once per round, so drifting load on
    // the machines or the network hits all engines alike
    // per engine: (tok/s, ms/token, tokens/epoch, network KB/token) per round
    let mut results: Vec<Vec<[f64; 4]>> = vec![Vec::new(); runs.len()];
    let mut loads = Vec::new();
    for round in 0..rounds {
        loads.push(tesseract_fabric::cluster::discovery::load_average());
        for (k, (engine, a)) in runs.iter().enumerate() {
            let Ok(a) = a else { continue };
            driver.assign(a);
            // node counters are cumulative: measure this run's traffic only
            let sent = |stats: Vec<tesseract_fabric::cluster::NodeStats>| {
                stats.iter().map(|x| x.bytes_sent).sum::<u64>()
            };
            let before = sent(driver.stats(&ids));
            let r = driver.generate(
                a.head,
                &GenerateRequest {
                    prompt: prompt.clone(),
                    max_new: tokens,
                    seed: 7,
                    temperature,
                    record_logits: false,
                },
            );
            let bytes = sent(driver.stats(&ids)).saturating_sub(before);
            driver.reset(a);
            let n = r.tokens.len().max(1) as f64;
            let s = r.decode_ns as f64 * 1e-9;
            let row = [
                n / s,
                s * 1e3 / n,
                n / r.per_epoch.len().max(1) as f64,
                bytes as f64 / 1024.0 / n,
            ];
            eprintln!(
                "round {}/{rounds} {}: {:.1} tok/s",
                round + 1,
                engine.name(),
                row[0]
            );
            results[k].push(row);
        }
    }
    let median = |v: &[f64]| {
        let mut v = v.to_vec();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let mut table = format!(
        "| Engine | tok/s, median of {rounds} | min–max | ms/token | tokens/epoch | network KB/token |\n|---|---:|---:|---:|---:|---:|\n"
    );
    for (k, (engine, a)) in runs.iter().enumerate() {
        match a {
            Err(why) => table.push_str(&format!("| {} | n/a — {why} | | | | |\n", engine.name())),
            Ok(_) => {
                let col = |c: usize| results[k].iter().map(|r| r[c]).collect::<Vec<_>>();
                let tps = col(0);
                let (lo, hi) = tps
                    .iter()
                    .fold((f64::MAX, 0.0f64), |(l, h), &x| (l.min(x), h.max(x)));
                table.push_str(&format!(
                    "| {} | {:.1} | {lo:.1}–{hi:.1} | {:.2} | {:.2} | {:.2} |\n",
                    engine.name(),
                    median(&tps),
                    median(&col(1)),
                    median(&col(2)),
                    median(&col(3))
                ));
            }
        }
    }
    println!("\n{table}");
    if let Some(path) = arg(args, "--report") {
        // never write secrets into a report meant to be shared
        let mut shown: Vec<String> = Vec::new();
        let mut hide = false;
        for a in args {
            shown.push(if hide { "<redacted>".into() } else { a.clone() });
            hide = matches!(a.as_str(), "--cluster-secret" | "--api-key");
        }
        let mut md = String::from("# tesseract-fabric physical cluster benchmark\n\n");
        md.push_str(&format!(
            "* Command: `{}`\n* Coordinator: {} / {} · tesseract-fabric {}\n* Model: {} layers, d_model {}, vocab {} · {tokens} new tokens · temperature {temperature} · {rounds} interleaved rounds\n* Links: {}\n* Node traffic: {}\n* Coordinator 1-minute load at each round: {}\n\n",
            shown.join(" "),
            std::env::consts::OS,
            std::env::consts::ARCH,
            env!("CARGO_PKG_VERSION"),
            cfg.n_layers,
            cfg.d_model,
            cfg.vocab,
            if emulate { "emulated from the device profiles" } else { "the real network (no shaping, no compute slowdown)" },
            traffic_line(),
            loads.iter().map(|l| format!("{l:.2}")).collect::<Vec<_>>().join(", ")
        ));
        md.push_str(&format!(
            "## Nodes\n\nLinks measured from the coordinator before the runs (3 echo pings for RTT, one 64 KB burst for bandwidth) over {} connections.\n\n| Node | Address | Declared profile | Measured RTT (µs) | Measured bandwidth (Gb/s) |\n|---|---|---|---:|---:|\n",
            if tesseract_fabric::wire::auth::wire_config().encrypt { "encrypted" } else { "plaintext" }
        ));
        for (i, m) in measured.iter().enumerate() {
            md.push_str(&format!(
                "| {i} | {} | {} | {} | {} |\n",
                peers[i],
                specs[i].name,
                m.as_ref()
                    .map(|m| format!("{:.0}", m.rtt_us))
                    .unwrap_or("—".into()),
                m.as_ref()
                    .map(|m| format!("{:.2}", m.bandwidth_gbps))
                    .unwrap_or("—".into())
            ));
        }
        md.push_str(&format!("\n## Results\n\n{table}\n## Per-round tok/s\n\n"));
        for (k, (engine, a)) in runs.iter().enumerate() {
            if a.is_err() {
                continue;
            }
            md.push_str(&format!(
                "* {}: {}\n",
                engine.name(),
                results[k]
                    .iter()
                    .map(|r| format!("{:.1}", r[0]))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        std::fs::write(&path, md).expect("write --report");
        println!("report written to {path}");
    }
}

/// The report's node-traffic line: mode, and for encrypted traffic this
/// machine's record-layer throughput, to compare with the measured links.
fn traffic_line() -> String {
    use tesseract_fabric::wire::auth;
    if !auth::wire_config().encrypt {
        return "plaintext (authenticated handshake only)".into();
    }
    let gbps = auth::measure_seal_gbps();
    format!(
        "encrypted (X25519 + ChaCha20-Poly1305); the coordinator seals {:.2} GB/s = {:.1} Gb/s with {} thread(s) per connection",
        gbps,
        gbps * 8.0,
        auth::crypto_threads()
    )
}
