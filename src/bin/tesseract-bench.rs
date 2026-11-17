//! `tesseract-bench` — empirical comparison of Exo-Ring-Pipeline,
//! RDMA-Tensor-Parallel and Tesseract-Fabric on live, link-shaped loopback
//! TCP clusters running real tensor compute.
//!
//! Usage: cargo run --release --bin tesseract-bench [-- --quick] [--out FILE]
//!        [--scenarios 1,2,3,micro,v8,v9]

use std::fmt::Write as _;
use std::time::Instant;
use tesseract_fabric::cluster::{
    node::context_kv, spawn_local_workers, Assignment, Driver, EngineKind, GenerateExt,
    GenerateRequest, GenerateResult, KvMode, NodeSetup, NodeStats, TesseractOptions,
    DRAFT_SERVICE_BASE, KV_SERVICE_BASE,
};
use tesseract_fabric::compression::{HoloConfig, HolographicCompressor, HolographicDecompressor};
use tesseract_fabric::kv_fabric::{
    DisaggregatedKVFabric, KvPageTable, PlacementPolicy, WirePrecision,
};
use tesseract_fabric::model::{
    argmax, logit_fidelity, HeadWeights, LayerWeights, ModelConfig, ReferenceModel,
};
use tesseract_fabric::speculative::drafter::{shortlist_params, ShortlistHead};
use tesseract_fabric::tensor::{reference_attention, Rng, RopeTable, Tensor2D};
use tesseract_fabric::topology::profiles::{links_between, slowdowns};
use tesseract_fabric::topology::{
    footprint, memory_weighted_partition, preset, DevicePreset, HardwareNodeSpec,
    HeteroClusterSolver, LinkSpec, SolverConfig,
};
use tesseract_fabric::wire::multipath::LinkProfile;
use tesseract_fabric::wire::protocol::FrameType;
use tesseract_fabric::wire::transport::{mono_ns, Endpoint};

// ---------------------------------------------------------------------------
// cluster definitions
// ---------------------------------------------------------------------------

struct ClusterDef {
    nodes: Vec<(u16, &'static DevicePreset)>,
    studio_tb5: bool,
}

impl ClusterDef {
    fn links(&self) -> Vec<(u16, u16, Vec<LinkProfile>)> {
        let mut v = Vec::new();
        for (i, &(a, pa)) in self.nodes.iter().enumerate() {
            for &(b, pb) in &self.nodes[i + 1..] {
                v.push((a, b, links_between(pa, pb, self.studio_tb5)));
            }
        }
        v
    }
    fn specs(&self) -> Vec<HardwareNodeSpec> {
        let links = self.links();
        self.nodes
            .iter()
            .map(|&(id, p)| {
                let ls = links
                    .iter()
                    .filter_map(|(a, b, ps)| {
                        if *a == id {
                            Some(LinkSpec {
                                peer: *b,
                                profiles: ps.clone(),
                            })
                        } else if *b == id {
                            Some(LinkSpec {
                                peer: *a,
                                profiles: ps.clone(),
                            })
                        } else {
                            None
                        }
                    })
                    .collect();
                p.spec(id, ls)
            })
            .collect()
    }
    fn ids(&self) -> Vec<u16> {
        self.nodes.iter().map(|n| n.0).collect()
    }
}

#[derive(Clone)]
struct RunCfg {
    label: String,
    engine: EngineKind,
    stages: Vec<(u16, usize, usize)>,
    head: u16,
    tess: TesseractOptions,
    kv_caps: Vec<u64>,
    inject: u32,
    note: String,
}

#[derive(Clone)]
struct Gen {
    prompt: Vec<u32>,
    max_new: u32,
    seed: u64,
    temperature: f32,
}

struct Measured {
    label: String,
    res: GenerateResult,
    stats: Vec<NodeStats>,
    tok_s: f64,
    ms_per_tok: f64,
    net_kb_per_tok: f64,
    util: f64,
    fidelity: Option<(f64, f64)>,
    tok_per_epoch: f64,
    note: String,
}

fn run_engine(
    cluster: &ClusterDef,
    cfg: &ModelConfig,
    rc: &RunCfg,
    g: &Gen,
    reference: Option<&ReferenceModel>,
) -> Measured {
    let ids = cluster.ids();
    let ws = spawn_local_workers(&ids);
    let driver = Driver::bind("127.0.0.1:0").expect("driver");
    let specs = cluster.specs();
    let slow = slowdowns(&specs);
    let nodes: Vec<NodeSetup> = ws
        .iter()
        .enumerate()
        .map(|(i, w)| NodeSetup {
            id: w.id,
            addr: w.addr,
            slowdown: slow[i],
            kv_capacity: rc.kv_caps[i],
        })
        .collect();
    let a = Assignment {
        engine: rc.engine,
        model: cfg.clone(),
        nodes,
        links: cluster.links(),
        stages: rc.stages.clone(),
        head: rc.head,
        hot_capacity: cfg
            .moe
            .as_ref()
            .map(|m| (m.n_experts * 3 / 4) as u16)
            .unwrap_or(0),
        inject_context: rc.inject,
        tess: rc.tess.clone(),
        driver_addr: driver.addr(),
        service_addrs: ws.iter().flat_map(|w| w.service_addrs()).collect(),
    };
    driver.assign(&a);
    let req = GenerateRequest {
        prompt: g.prompt.clone(),
        max_new: g.max_new,
        seed: g.seed,
        temperature: g.temperature,
        record_logits: reference.is_some(),
    };
    let res = driver.generate(rc.head, &req);
    let mut all_ids = ids.clone();
    all_ids.extend(ids.iter().map(|i| KV_SERVICE_BASE + i));
    all_ids.extend(ids.iter().map(|i| DRAFT_SERVICE_BASE + i));
    let stats = driver.stats(&all_ids);
    driver.reset(&a);
    driver.shutdown(&a);
    let n = res.tokens.len().max(1) as f64;
    let decode_s = res.decode_ns as f64 * 1e-9;
    let wall_s = (res.decode_ns + res.prefill_ns) as f64 * 1e-9;
    let bytes: u64 = stats.iter().map(|s| s.bytes_sent).sum();
    // utilisation: per host, busy time of node + its services, capped at wall
    let mut busy = 0.0;
    for &h in &ids {
        let b: u64 = stats
            .iter()
            .filter(|s| Assignment::host_of(s.node) == h)
            .map(|s| s.busy_ns)
            .sum();
        busy += (b as f64 * 1e-9).min(wall_s);
    }
    let fidelity = reference.map(|m| {
        let mut seq = g.prompt.clone();
        seq.extend(&res.tokens);
        let rl = m.forward_sequence(&seq);
        let mut cos = 0.0;
        let mut agree = 0usize;
        for (i, l) in res.logits.iter().enumerate() {
            let row = rl.row(g.prompt.len() - 1 + i);
            cos += logit_fidelity(l, row);
            agree += (argmax(l) == argmax(row)) as usize;
        }
        (
            cos / res.logits.len().max(1) as f64,
            agree as f64 / res.logits.len().max(1) as f64,
        )
    });
    Measured {
        label: rc.label.clone(),
        tok_s: n / decode_s,
        ms_per_tok: decode_s * 1e3 / n,
        net_kb_per_tok: bytes as f64 / 1024.0 / n,
        util: busy / (ids.len() as f64 * wall_s) * 100.0,
        fidelity,
        tok_per_epoch: n / res.per_epoch.len().max(1) as f64,
        note: rc.note.clone(),
        res,
        stats,
    }
}

fn ring_stages(order: &[u16], counts: &[usize]) -> Vec<(u16, usize, usize)> {
    let mut l = 0;
    order
        .iter()
        .zip(counts)
        .map(|(&n, &c)| {
            let s = (n, l, l + c);
            l += c;
            s
        })
        .collect()
}

fn fmt_fid(f: Option<(f64, f64)>) -> String {
    match f {
        Some((c, a)) => format!("{c:.5} ({:.0}% top-1)", a * 100.0),
        None => "n/a".into(),
    }
}

// ---------------------------------------------------------------------------
// scenarios 1 & 2
// ---------------------------------------------------------------------------

fn scenario_pipeline(
    title: &str,
    cluster: &ClusterDef,
    cfg: &ModelConfig,
    g: &Gen,
    reference: &ReferenceModel,
    rounds: usize,
    out: &mut String,
) -> Vec<Measured> {
    let specs = cluster.specs();
    let ids = cluster.ids();
    let ram: Vec<f64> = specs.iter().map(|s| s.available_ram_mb).collect();
    let layer_mb = cfg.layer_params() as f64 * 4.0 / 1e6;
    // Baseline A: exo memory-weighted ring over all nodes; head = last stage
    let counts = memory_weighted_partition(cfg.n_layers, &ram);
    let ring = ring_stages(&ids, &counts);
    let ring_fits = ring.iter().all(|&(n, s, e)| {
        (e - s) as f64 * layer_mb <= ram[ids.iter().position(|&x| x == n).unwrap()]
    });
    // Baseline B: TP across all nodes; head = rank 0
    let tp_mb = cfg.n_layers as f64 * layer_mb / ids.len() as f64;
    let tp_oom: Vec<String> = specs
        .iter()
        .filter(|s| s.available_ram_mb < tp_mb)
        .map(|s| s.name.clone())
        .collect();
    // Frontier: solver plan
    let tess_opts = TesseractOptions::default();
    let solver = HeteroClusterSolver::new(
        specs.clone(),
        footprint(cfg, tess_opts.holo_bits),
        SolverConfig::default(),
    );
    let plan = solver.solve();
    let tess_engine = if plan.htep.is_some() {
        EngineKind::TesseractHtep
    } else {
        EngineKind::Tesseract
    };
    let tstages: Vec<(u16, usize, usize)> = plan
        .stages
        .iter()
        .map(|s| (s.node, s.layers.start, s.layers.end))
        .collect();
    let kv_caps: Vec<u64> = ids
        .iter()
        .map(|id| {
            plan.kv_pools
                .iter()
                .find(|p| p.0 == *id)
                .map(|p| p.1 as u64)
                .unwrap_or(0)
        })
        .collect();
    let mut tess_full = tess_opts.clone();
    tess_full.use_plan_drafter(plan.drafter, plan.coordinator);
    let mut tess_nospec = tess_full.clone();
    tess_nospec.budget = 1;
    tess_nospec.braid_strands = 1;
    tess_nospec.draft_ahead = 0;
    let caps_default: Vec<u64> = ram.iter().map(|r| (*r * 1e6 * 0.15) as u64).collect();

    writeln!(out, "\n## {title}\n").unwrap();
    writeln!(out, "| Node | Device profile | Platform | Mem BW (GB/s) | Emulated RAM (MB) | Compute slowdown |").unwrap();
    writeln!(out, "|---|---|---|---:|---:|---:|").unwrap();
    let sl = slowdowns(&specs);
    for (i, s) in specs.iter().enumerate() {
        writeln!(
            out,
            "| {} | {} | {} | {:.0} | {:.0} | {:.2}× |",
            s.node_id,
            s.name,
            s.platform.name(),
            s.memory_bandwidth_gbps,
            s.available_ram_mb,
            sl[i]
        )
        .unwrap();
    }
    writeln!(
        out,
        "\nLinks: {}",
        cluster
            .links()
            .iter()
            .map(|(a, b, ps)| format!(
                "{a}↔{b}: {}",
                ps.iter().map(|p| p.name()).collect::<Vec<_>>().join("+")
            ))
            .collect::<Vec<_>>()
            .join(", ")
    )
    .unwrap();
    writeln!(
        out,
        "\n* Exo ring partition (layers ∝ RAM): {}",
        ring.iter()
            .map(|(n, s, e)| format!("node {n}: L{s}–{}", e - 1))
            .collect::<Vec<_>>()
            .join(" → ")
    )
    .unwrap();
    writeln!(out, "* HeteroClusterSolver plan: {} | coordinator = node {} | drafter = {} | est. epoch {:.2} ms", plan.stages.iter().map(|s| format!("node {}: L{}–{}", s.node, s.layers.start, s.layers.end - 1)).collect::<Vec<_>>().join(" → "), plan.coordinator, plan.drafter.map(|d| format!("node {d}")).unwrap_or("coordinator-local thread".into()), plan.est_epoch_ms).unwrap();
    for n in &plan.notes {
        writeln!(out, "  * {n}").unwrap();
    }

    let runs = [
        RunCfg {
            label: "Exo-Ring-Pipeline".into(),
            engine: EngineKind::ExoRing,
            stages: ring.clone(),
            head: ring.last().unwrap().0,
            tess: TesseractOptions::default(),
            kv_caps: caps_default.clone(),
            inject: 0,
            note: if ring_fits {
                String::new()
            } else {
                "ring exceeds node RAM".into()
            },
        },
        RunCfg {
            label: "RDMA-Tensor-Parallel".into(),
            engine: EngineKind::TensorParallel,
            stages: ids.iter().map(|&i| (i, 0, 0)).collect(),
            head: ids[0],
            tess: TesseractOptions::default(),
            kv_caps: caps_default.clone(),
            inject: 0,
            note: if tp_oom.is_empty() {
                String::new()
            } else {
                format!("1/{} shard ({tp_mb:.0} MB) exceeds RAM of {} — would OOM; run for latency only", ids.len(), tp_oom.join(", "))
            },
        },
        RunCfg {
            label: if plan.htep.is_some() {
                "Tesseract-Fabric (H-TEP)".into()
            } else {
                "Tesseract-Fabric".into()
            },
            engine: tess_engine,
            stages: tstages.clone(),
            head: plan.coordinator,
            tess: tess_full,
            kv_caps: kv_caps.clone(),
            inject: 0,
            note: String::new(),
        },
        RunCfg {
            label: "Tesseract-Fabric (speculation off, ablation)".into(),
            engine: tess_engine,
            stages: tstages,
            head: plan.coordinator,
            tess: tess_nospec,
            kv_caps,
            inject: 0,
            note: String::new(),
        },
    ];
    // Interleaved paired rounds: every round runs all engines back to back, so
    // a transient host-load spike hits every engine of that round; each
    // engine reports its median round (by throughput).
    let mut per: Vec<Vec<Measured>> = runs.iter().map(|_| Vec::new()).collect();
    for round in 0..rounds {
        for (i, rc) in runs.iter().enumerate() {
            eprint!("  round {}: {} … ", round + 1, rc.label);
            let t = Instant::now();
            let m = run_engine(cluster, cfg, rc, g, Some(reference));
            eprintln!("{:.1} tok/s ({:.1}s)", m.tok_s, t.elapsed().as_secs_f64());
            per[i].push(m);
        }
    }
    // paired estimator: every round runs all engines back to back, so a
    // round's ratio compares runs that saw the same host load
    let kept = kept_rounds(&per);
    let paired = |a: usize, b: usize| -> (f64, f64, f64) {
        let mut r: Vec<f64> = kept
            .iter()
            .map(|&k| per[a][k].tok_s / per[b][k].tok_s.max(1e-9))
            .collect();
        r.sort_by(|x, y| x.partial_cmp(y).unwrap());
        (r[r.len() / 2], r[0], r[r.len() - 1])
    };
    let env: Vec<f64> = per.iter().map(|v| envelope(v)).collect();
    let paired_line = if per.len() >= 4 {
        let (a, alo, ahi) = paired(2, 3);
        let (b, blo, bhi) = paired(3, 1);
        let (c, clo, chi) = paired(1, 0);
        format!("\nPaired per-round ratios (median of the {} of {rounds} rounds kept by the symmetric load-spike filter, min–max): spec-on / spec-off **{a:.2}×** ({alo:.2}–{ahi:.2}), spec-off / TP **{b:.2}×** ({blo:.2}–{bhi:.2}), TP / Exo **{c:.2}×** ({clo:.2}–{chi:.2}).\n\nFastest-quarter envelope ratios (mean of each engine's fastest ⌈{rounds}/4⌉ rounds): spec-on / spec-off **{:.2}×**, spec-off / TP **{:.2}×**, TP / Exo **{:.2}×**.", kept.len(), env[2] / env[3], env[3] / env[1], env[1] / env[0])
    } else {
        String::new()
    };
    let mut results = Vec::new();
    let mut spreads = Vec::new();
    for mut v in per {
        v.sort_by(|a, b| a.tok_s.partial_cmp(&b.tok_s).unwrap());
        spreads.push((v[0].tok_s, v[v.len() - 1].tok_s));
        let mid = v.len() / 2;
        results.push(v.swap_remove(mid));
    }
    let base = results[0].tok_s;
    writeln!(out, "\n| Engine | Throughput, median of {rounds} interleaved rounds (tok/s) | Fastest-quarter envelope (tok/s) | min–max | vs Exo | Per-token latency (ms) | Network per token (KB) | Active HW utilization | Cosine fidelity vs FP32 (top-1 agree) | Tokens / epoch |").unwrap();
    writeln!(out, "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|").unwrap();
    for ((m, sp), e) in results.iter().zip(&spreads).zip(&env) {
        writeln!(
            out,
            "| {}{} | {:.1} | {e:.1} | {:.1}–{:.1} | {:.2}× | {:.2} | {:.2} | {:.1}% | {} | {:.2} |",
            m.label,
            if m.note.is_empty() {
                String::new()
            } else {
                " †".into()
            },
            m.tok_s,
            sp.0,
            sp.1,
            m.tok_s / base,
            m.ms_per_tok,
            m.net_kb_per_tok,
            m.util,
            fmt_fid(m.fidelity),
            m.tok_per_epoch
        )
        .unwrap();
    }
    if !paired_line.is_empty() {
        writeln!(out, "{paired_line}").unwrap();
    }
    for m in &results {
        if !m.note.is_empty() {
            writeln!(out, "\n† {}: {}", m.label, m.note).unwrap();
        }
    }
    // pillar telemetry for the frontier run
    let t = &results[2];
    let mut agg = NodeStats::default();
    for s in &t.stats {
        agg.merge(s);
    }
    let base_agg = results[0]
        .stats
        .iter()
        .fold(NodeStats::default(), |mut a, s| {
            a.merge(s);
            a
        });
    let budgets: Vec<String> = {
        let mut h = std::collections::BTreeMap::new();
        for b in &t.res.budgets {
            *h.entry(*b).or_insert(0) += 1;
        }
        h.into_iter().map(|(b, c)| format!("{b}:{c}")).collect()
    };
    writeln!(out, "\nTesseract telemetry: decode {:.2}s over {} epochs; tree budget picks (budget:count) {}; draft-ahead hits/misses {}/{}; KV commits {} (mean {:.2} µs); expert staging — blocking misses {} (Exo: {}), pre-staged {}, lookahead recall {:.1}%, PreWake frames {} ({:.1} KB out-of-band); stage-0 cross-step lookahead rows reused / discarded / aborted early {} / {} / {}.",
        t.res.decode_ns as f64 * 1e-9, t.res.per_epoch.len(), budgets.join(" "), t.res.draft_hits, t.res.draft_misses, agg.commits, agg.commit_ns as f64 / agg.commits.max(1) as f64 / 1e3,
        agg.staging[1], base_agg.staging[1], agg.staging[3], 100.0 * agg.lookahead_covered as f64 / agg.lookahead_needed.max(1) as f64, agg.prewakes_sent, agg.oob_bytes as f64 / 1024.0, agg.ahead_hits, agg.ahead_misses, agg.ahead_aborts).unwrap();
    results
}

fn bits_ablation(
    cluster: &ClusterDef,
    cfg: &ModelConfig,
    g: &Gen,
    reference: &ReferenceModel,
    out: &mut String,
) {
    let specs = cluster.specs();
    // a fixed 2-stage pipeline (8 + 8 layers, one compressed hop): the codec
    // comparison must not depend on which parallelism the solver chose
    let ids = cluster.ids();
    let half = cfg.n_layers / 2;
    let stages: Vec<(u16, usize, usize)> = vec![(ids[0], 0, half), (ids[1], half, cfg.n_layers)];
    let coordinator = ids[1];
    let caps: Vec<u64> = specs
        .iter()
        .map(|s| (s.available_ram_mb * 1e6 * 0.15) as u64)
        .collect();
    let mut prng = Rng::new(0xC0DE);
    let prompts: Vec<Vec<u32>> = (0..3)
        .map(|i| {
            if i == 0 {
                g.prompt.clone()
            } else {
                (0..g.prompt.len())
                    .map(|_| prng.below(cfg.vocab) as u32)
                    .collect()
            }
        })
        .collect();
    writeln!(out, "\n### Activation codec frontier (Scenario 1 hardware, fixed 8+8-layer 2-stage pipeline, speculation off, {} prompts × {} tokens, temperature {})\n\nBytes are whole inter-stage frames (64-byte header + strand metadata + token ids + payload). Exo ships the hidden state as raw FP16 (2 B/element). Fidelity aggregates every generated position of all prompts, so a 0.1-point top-1 difference is about one token.\n\n| Activation coding | Inter-stage bytes per token | vs FP16 frame | Codec payload per row vs FP16 | Cosine fidelity vs FP32 | Top-1 agreement |\n|---|---:|---:|---:|---:|---:|", prompts.len(), g.max_new, g.temperature).unwrap();
    let mut fp16_bytes = None;
    let variants: [(&str, EngineKind, u8, u8); 7] = [
        (
            "FP16 (Exo ring, same 2-stage split)",
            EngineKind::ExoRing,
            0,
            0,
        ),
        ("V1 holographic 8-bit", EngineKind::Tesseract, 1, 8),
        ("V1 holographic 4-bit", EngineKind::Tesseract, 1, 4),
        (
            "V2 HRQ trunk-residual 3-bit + sketch",
            EngineKind::Tesseract,
            2,
            3,
        ),
        (
            "V2 HRQ trunk-residual 2-bit + sketch",
            EngineKind::Tesseract,
            2,
            2,
        ),
        (
            "V2.1 entropy-coded trunk residual, ≤25% cap",
            EngineKind::Tesseract,
            3,
            3,
        ),
        (
            "**V5 conditional-KLT transform coding, ≤25% cap (default)**",
            EngineKind::Tesseract,
            4,
            3,
        ),
    ];
    for (label, engine, codec, bits) in variants {
        let label = label.to_string();
        let tess = TesseractOptions {
            codec,
            // the frontier measures each codec, so no per-hop FP16 fallback
            codec_auto: false,
            holo_bits: bits,
            budget: 1,
            stage_lookahead: false,
            braid_strands: 1,
            draft_ahead: 0,
            ..Default::default()
        };
        let row_payload = match codec {
            0 => cfg.d_model * 2,
            1 => {
                9 + 4
                    + cfg.d_model.next_power_of_two() / 64 * 2
                    + (cfg.d_model.next_power_of_two() * bits as usize).div_ceil(8)
            }
            // V2.1 and V5 rows are hard-capped at 25% of the FP16 row
            3 | 4 => cfg.d_model * 2 / 4,
            _ => tesseract_fabric::compression::TrunkResidualCodec::new(cfg.d_model, bits, 0)
                .row_bytes(),
        };
        let rc = RunCfg {
            label: label.clone(),
            engine,
            stages: stages.clone(),
            head: coordinator,
            tess,
            kv_caps: caps.clone(),
            inject: 0,
            note: String::new(),
        };
        eprint!("  ablation {label} … ");
        let (mut act, mut toks, mut pos, mut cos, mut agree) = (0u64, 0usize, 0usize, 0.0, 0.0);
        for (pi, prompt) in prompts.iter().enumerate() {
            let gp = Gen {
                prompt: prompt.clone(),
                seed: g.seed + pi as u64,
                ..g.clone()
            };
            let m = run_engine(cluster, cfg, &rc, &gp, Some(reference));
            act += m.stats.iter().map(|s| s.act_bytes).sum::<u64>();
            toks += m.res.tokens.len();
            let n = m.res.logits.len();
            if let Some((c, a)) = m.fidelity {
                cos += c * n as f64;
                agree += a * n as f64;
            }
            pos += n;
        }
        eprintln!("done");
        let per_tok = act as f64 / toks.max(1) as f64;
        let base = *fp16_bytes.get_or_insert(per_tok);
        let pos_f = pos.max(1) as f64;
        writeln!(
            out,
            "| {label} | {per_tok:.0} B | {:.0}% | {:.1}% | {:.5} | {:.2}% ({}/{pos}) |",
            100.0 * per_tok / base,
            100.0 * row_payload as f64 / (cfg.d_model * 2) as f64,
            cos / pos_f,
            100.0 * agree / pos_f,
            agree.round() as usize
        )
        .unwrap();
    }
}

/// Symmetric lower-envelope throughput: the mean of an engine's fastest
/// quarter of rounds (host load only ever slows a round down).
fn envelope(v: &[Measured]) -> f64 {
    let mut t: Vec<f64> = v.iter().map(|m| m.tok_s).collect();
    t.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let q = t.len().div_ceil(4);
    t[..q].iter().sum::<f64>() / q as f64
}

/// Symmetric preemption filter for paired rounds: a round is dropped when any
/// engine in it ran below its own median throughput ÷ 1.5 (a load spike),
/// so every engine's rounds are filtered by the same rule. Never drops more
/// than half of the rounds.
fn kept_rounds(per: &[Vec<Measured>]) -> Vec<usize> {
    let n = per.first().map(|v| v.len()).unwrap_or(0);
    let med: Vec<f64> = per
        .iter()
        .map(|v| {
            let mut t: Vec<f64> = v.iter().map(|m| m.tok_s).collect();
            t.sort_by(|a, b| a.partial_cmp(b).unwrap());
            t[t.len() / 2]
        })
        .collect();
    let keep: Vec<usize> = (0..n)
        .filter(|&k| per.iter().zip(&med).all(|(v, m)| v[k].tok_s >= m / 1.5))
        .collect();
    if keep.len() * 2 >= n {
        keep
    } else {
        (0..n).collect()
    }
}

/// Row 6 as the deployed cluster pays it: H-TEP epochs with a fixed 1-row
/// step vs a fixed 4-row tree (lossless verification), interleaved rounds,
/// per-epoch decode time; reported next to the single-node MoE figures.
fn htep_step_ratio(
    cluster: &ClusterDef,
    cfg: &ModelConfig,
    g: &Gen,
    reference: &ReferenceModel,
    rounds: usize,
    out: &mut String,
) {
    let specs = cluster.specs();
    let plan =
        HeteroClusterSolver::new(specs.clone(), footprint(cfg, 3), SolverConfig::default()).solve();
    if plan.htep.is_none() {
        return;
    }
    let stages: Vec<(u16, usize, usize)> = plan
        .stages
        .iter()
        .map(|s| (s.node, s.layers.start, s.layers.end))
        .collect();
    let caps: Vec<u64> = specs
        .iter()
        .map(|s| (s.available_ram_mb * 1e6 * 0.15) as u64)
        .collect();
    let mk = |budget: u16| RunCfg {
        label: format!("{budget}-row"),
        engine: EngineKind::TesseractHtep,
        stages: stages.clone(),
        head: plan.coordinator,
        tess: TesseractOptions {
            budget,
            drafter: plan.drafter,
            ..Default::default()
        },
        kv_caps: caps.clone(),
        inject: 0,
        note: String::new(),
    };
    let runs = [mk(1), mk(4)];
    let mut ratios = Vec::new();
    let mut ep = [Vec::new(), Vec::new()];
    let mut tpe = Vec::new();
    for round in 0..rounds {
        let mut t = [0.0f64; 2];
        for (i, rc) in runs.iter().enumerate() {
            eprint!("  H-TEP step ratio round {}: {} … ", round + 1, rc.label);
            let m = run_engine(cluster, cfg, rc, g, Some(reference));
            let epochs = m.res.per_epoch.len().max(1) as f64;
            t[i] = m.res.decode_ns as f64 / epochs;
            if i == 1 {
                tpe.push(m.tok_per_epoch);
            }
            eprintln!("{:.2} ms/epoch", t[i] / 1e6);
            ep[i].push(t[i]);
        }
        ratios.push(t[1] / t[0]);
    }
    let med = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (v[v.len() / 2], v[0], v[v.len() - 1])
    };
    let (r, rlo, rhi) = med(ratios);
    let (e1, _, _) = med(ep[0].clone());
    let (e4, _, _) = med(ep[1].clone());
    let (a4, _, _) = med(tpe);
    // symmetric lower envelope: mean of the fastest quartile of each side
    let quart = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = v.len().div_ceil(4);
        v[..q].iter().sum::<f64>() / q as f64
    };
    let (q1, q4) = (quart(ep[0].clone()), quart(ep[1].clone()));
    writeln!(out, "\n#### 4-token verification in the deployed cluster (H-TEP, this hardware, median of {rounds} interleaved rounds)\n\nEvery rank holds a quarter of every expert's hidden units, so a 4-row tree still touches ≈ 5.4 expert slices per rank; the per-layer collectives and the replicated attention are paid once per epoch for any tree size. Verification is exact. Targets: the original ≤ 1.35× assumed dense-model cost; exact top-2 routing makes a 4-row tree read ≈ 2.7× the expert weights of one row, and V7.0 revised the target to ≤ 1.65× *after* measuring ≈ 1.6× (both thresholds are reported).\n\n| Epoch | Decode time per epoch | Tokens per epoch |\n|---|---:|---:|\n| 1-row step | {:.2} ms | 1.00 |\n| 4-row tree | {:.2} ms | {a4:.2} |\n| 4-row / 1-row, paired per-round median | {r:.2}× (rounds {rlo:.2}–{rhi:.2}) | |\n| **4-row / 1-row, lower envelope** (mean of the fastest quartile of each side: {:.2} vs {:.2} ms) | **{:.2}×** | |", e1 / 1e6, e4 / 1e6, q1 / 1e6, q4 / 1e6, q4 / q1).unwrap();
}

/// Scenario 1 at a second temperature (the headline runs greedy, T = 0; this
/// ablation runs sampled, T = 1), where speculative acceptance is capped by the
/// target's entropy rather than by drafter agreement.
fn temperature_ablation(
    cluster: &ClusterDef,
    cfg: &ModelConfig,
    g: &Gen,
    reference: &ReferenceModel,
    rounds: usize,
    temperature: f32,
    out: &mut String,
) {
    let specs = cluster.specs();
    let ids = cluster.ids();
    let plan =
        HeteroClusterSolver::new(specs.clone(), footprint(cfg, 3), SolverConfig::default()).solve();
    let engine = if plan.htep.is_some() {
        EngineKind::TesseractHtep
    } else {
        EngineKind::Tesseract
    };
    let stages: Vec<(u16, usize, usize)> = plan
        .stages
        .iter()
        .map(|s| (s.node, s.layers.start, s.layers.end))
        .collect();
    let caps: Vec<u64> = specs
        .iter()
        .map(|s| (s.available_ram_mb * 1e6 * 0.15) as u64)
        .collect();
    let gg = Gen {
        temperature,
        ..g.clone()
    };
    let runs = [
        RunCfg {
            label: "RDMA-Tensor-Parallel".into(),
            engine: EngineKind::TensorParallel,
            stages: ids.iter().map(|&i| (i, 0, 0)).collect(),
            head: ids[0],
            tess: TesseractOptions::default(),
            kv_caps: caps.clone(),
            inject: 0,
            note: String::new(),
        },
        RunCfg {
            label: "Tesseract-Fabric, speculation off".into(),
            engine,
            stages: stages.clone(),
            head: plan.coordinator,
            tess: TesseractOptions {
                budget: 1,
                ..Default::default()
            },
            kv_caps: caps.clone(),
            inject: 0,
            note: String::new(),
        },
        RunCfg {
            label: "Tesseract-Fabric, speculation on".into(),
            engine,
            stages,
            head: plan.coordinator,
            tess: TesseractOptions {
                drafter: plan.drafter,
                ..Default::default()
            },
            kv_caps: caps,
            inject: 0,
            note: String::new(),
        },
    ];
    let mut per: Vec<Vec<Measured>> = runs.iter().map(|_| Vec::new()).collect();
    for round in 0..rounds {
        for (i, rc) in runs.iter().enumerate() {
            eprint!("  T={temperature} round {}: {} … ", round + 1, rc.label);
            let m = run_engine(cluster, cfg, rc, &gg, Some(reference));
            eprintln!("{:.1} tok/s", m.tok_s);
            per[i].push(m);
        }
    }
    writeln!(out, "\n#### Sampled decoding (T = {temperature}) ablation, same cluster, median of {rounds} interleaved rounds\n\n| Engine | Throughput (tok/s) | min–max | Tokens / epoch | Cosine fidelity vs FP32 (top-1 agree) |\n|---|---:|---:|---:|---|").unwrap();
    for mut v in per {
        v.sort_by(|a, b| a.tok_s.partial_cmp(&b.tok_s).unwrap());
        let (lo, hi) = (v[0].tok_s, v[v.len() - 1].tok_s);
        let m = &v[v.len() / 2];
        writeln!(
            out,
            "| {} | {:.1} | {lo:.1}–{hi:.1} | {:.2} | {} |",
            m.label,
            m.tok_s,
            m.tok_per_epoch,
            fmt_fid(m.fidelity)
        )
        .unwrap();
    }
}

// ---------------------------------------------------------------------------
// scenario 3: long context
// ---------------------------------------------------------------------------

/// One layer of disaggregated attention over `ctx` context tokens: max abs
/// error vs monolithic attention, network bytes (incl. headers) and the FP16
/// page-migration equivalent. With `groups > 1` the policy places virtual
/// layers `g` (KV-head group g of layer 0) and every group is attended on
/// its own holders with only its query heads on the wire.
fn lse_precision(
    prec: WirePrecision,
    cfg: &ModelConfig,
    caps: &[(u16, usize)],
    ctx: usize,
    page_tokens: usize,
    policy: PlacementPolicy,
    groups: usize,
) -> (f32, usize, usize) {
    let embed = HeadWeights::generate(cfg).embed;
    let rope = RopeTable::new(cfg.d_head, cfg.rope_theta);
    let layer = LayerWeights::generate(
        &ModelConfig {
            moe: None,
            d_ff: 32,
            ..cfg.clone()
        },
        0,
        None,
    );
    let (k, v) = context_kv(cfg, &layer, &rope, &embed, 0..ctx);
    let table = KvPageTable::new(page_tokens, caps, policy);
    let (nh, dh, kd) = (cfg.n_heads, cfg.d_head, cfg.kv_dim());
    let (gh, kdg) = (nh / groups, kd / groups);
    let mut fab = DisaggregatedKVFabric::new(table, gh, cfg.n_kv_heads / groups, dh, caps[0].0);
    fab.precision = prec;
    for t in 0..ctx {
        for g in 0..groups {
            let o = t * kd + g * kdg;
            fab.append(g, t, &k.data[o..o + kdg], &v.data[o..o + kdg])
                .unwrap();
        }
    }
    let mut rng = Rng::new(ctx as u64);
    let q: Vec<f32> = (0..nh * dh).map(|_| rng.normal()).collect();
    let mut out = vec![0.0f32; nh * dh];
    let (mut net, mut mig) = (0usize, 0usize);
    for g in 0..groups {
        let qg = &q[g * gh * dh..(g + 1) * gh * dh];
        let (o, st) = fab.attend_and_merge(g, qg, 1, &[usize::MAX]);
        out[g * gh * dh..(g + 1) * gh * dh].copy_from_slice(&o);
        net += st.network_bytes;
        mig += st.migration_bytes;
    }
    let mut err = 0.0f32;
    for h in 0..nh {
        let g = h / (nh / cfg.n_kv_heads);
        let kh: Vec<f32> = (0..ctx)
            .flat_map(|t| k.data[t * kd + g * dh..t * kd + (g + 1) * dh].to_vec())
            .collect();
        let vh: Vec<f32> = (0..ctx)
            .flat_map(|t| v.data[t * kd + g * dh..t * kd + (g + 1) * dh].to_vec())
            .collect();
        let mut r = vec![0.0; dh];
        reference_attention(&q[h * dh..(h + 1) * dh], &kh, &vh, dh, &|_| true, &mut r);
        for d in 0..dh {
            err = err.max((r[d] - out[h * dh + d]).abs());
        }
    }
    (err, net, mig)
}

/// Declared Scenario 3 attention wire budget (KB per decode step).
const S3_WIRE_BUDGET_KB: f64 = 34.0;

fn scenario_long_context(contexts: &[u32], g: &Gen, s3_rounds: usize, out: &mut String) {
    let cfg = ModelConfig::long_context();
    let studio = preset("studio-m5max").unwrap();
    let cluster = ClusterDef {
        nodes: (0..4).map(|i| (i, studio)).collect(),
        studio_tb5: true,
    };
    let ids = cluster.ids();
    // heterogeneous RAM budgets (MB) for the four shards
    let ram_mb = [512.0f64, 256.0, 192.0, 128.0];
    let layer_mb = cfg.layer_params() as f64 * 4.0 / 1e6;
    let counts = memory_weighted_partition(cfg.n_layers, &ram_mb);
    let ring = ring_stages(&ids, &counts);
    let free: Vec<u64> = ids
        .iter()
        .enumerate()
        .map(|(i, _)| ((ram_mb[i] - counts[i] as f64 * layer_mb) * 1e6) as u64)
        .collect();
    // calibration: one short KV-fabric run at 8K with an even split measures
    // the per-page local scan cost and the remote exchange overhead (wait for
    // the partner beyond the local scan); the partner is then kept
    // overhead / page-cost pages behind the computing node in every run
    let bias: (u16, String) = {
        let rc = RunCfg {
            label: "calibration".into(),
            engine: EngineKind::Tesseract,
            stages: ring.clone(),
            head: ring.last().unwrap().0,
            tess: TesseractOptions {
                kv_mode: KvMode::Disaggregated,
                budget: 1,
                braid_strands: 1,
                draft_ahead: 0,
                lookahead: false,
                ..Default::default()
            },
            kv_caps: free.clone(),
            inject: 8192,
            note: String::new(),
        };
        eprintln!("  calibrating the paired KV page bias …");
        let m = run_engine(
            &cluster,
            &cfg,
            &rc,
            &Gen {
                max_new: 8,
                ..g.clone()
            },
            None,
        );
        // lower envelopes: load spikes only ever add time, so the minimum
        // exchange wait and the minimum per-page scan are the engine's own
        let mut agg = NodeStats::default();
        for st in &m.stats {
            agg.merge(st);
        }
        let page_ns = agg.kv_page_min_ns.max(1) as f64;
        let wait_ns = agg.kv_wait_min_ns as f64;
        let b0 = (wait_ns / page_ns).round().clamp(0.0, 64.0) as u16;
        // the declared wire budget is a hard constraint on the placement:
        // project the attention bytes of one decode step with a dry run of
        // the page table at every long context (frames sized by the real
        // encoders, partials at the widest code) and step the bias down until
        // all of them fit
        let owners: Vec<(std::ops::Range<usize>, u16)> =
            ring.iter().map(|&(n, s, e)| (s..e, n)).collect();
        let caps_pairs: Vec<(u16, usize)> = ids
            .iter()
            .zip(&free)
            .map(|(&i, &f)| (i, f as usize))
            .collect();
        let tier0 = TesseractOptions::default().tier0_pages as usize;
        let project = |b: u16, ctx: usize| {
            tesseract_fabric::kv_fabric::page_table::project_step_bytes(
                &owners,
                &caps_pairs,
                cfg.n_layers,
                cfg.n_heads,
                cfg.n_kv_heads,
                cfg.d_head,
                256,
                tier0,
                b as usize,
                ctx,
            ) as f64
                / 1024.0
        };
        let lens = [8192usize, 32768, 65536];
        let mut b = b0;
        while b > 0 && lens.iter().any(|&c| project(b, c) >= S3_WIRE_BUDGET_KB) {
            b -= 1;
        }
        let proj: Vec<String> = lens
            .iter()
            .map(|&c| format!("{:.1}", project(b, c)))
            .collect();
        let kb = proj.join(" / ");
        let note = format!("KV remote-holder bias (calibrated once at 8K from lower envelopes): minimum remote exchange overhead {:.0} µs vs minimum {:.0} µs to scan one local page → {b0} pages; capped to {b} so the placement's projected attention traffic at 8K / 32K / 64K ({kb} KB/step, dry-run page layout with real frame sizes) stays under the {S3_WIRE_BUDGET_KB} KB/step wire budget. Every remote holder of a virtual layer is kept {b} pages behind the computing node.", wait_ns / 1e3, page_ns / 1e3);
        (b, note)
    };
    let (bias, bias_note) = bias;
    writeln!(out, "\n## Scenario 3 — Long-Context Scaling Stress Test (4 shards, TB5 links, {} layers, GQA {}q/{}kv heads)\n", cfg.n_layers, cfg.n_heads, cfg.n_kv_heads).unwrap();
    writeln!(out, "Shard RAM budgets (MB): {:?}; layer split (both engines, ∝ RAM): {:?}. Context KV is injected with the layers' real K/V projections of the context tokens (single-pass approximate prefill), then {} tokens are decoded with full forward passes.\n", ram_mb, counts, g.max_new).unwrap();
    writeln!(out, "{bias_note}\n").unwrap();
    writeln!(out, "| Context | Engine | Throughput, median of {s3_rounds} interleaved rounds (tok/s) | min–max | vs ring: fastest-quarter envelope | vs ring: median of paired per-round ratios, load-spike filtered (min–max) | Peak node memory (weights+KV) | Memory balance (max/mean util) | Attention traffic (KB/step) | FP16 page-migration equivalent (KB/step) | LSE merge max abs err vs monolithic (live BFP wire + head-group striping) |").unwrap();
    writeln!(
        out,
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    .unwrap();
    for &ctx in contexts {
        let caps_pairs: Vec<(u16, usize)> = ids
            .iter()
            .zip(&free)
            .map(|(&i, &f)| (i, f as usize))
            .collect();
        // three engines, interleaved round by round so host-load swings hit
        // all of them alike: the ring baseline, Tesseract's KV fabric alone
        // (speculation off, isolating the KV design), and the full engine
        let modes: [(KvMode, &str, u16); 3] = [
            (KvMode::LayerCoupled, "Layer-coupled ring KV (Exo)", 1),
            (
                KvMode::Disaggregated,
                "Tesseract disaggregated Flash-KV (speculation off)",
                1,
            ),
            (
                KvMode::Disaggregated,
                "Tesseract full engine (Flash-KV + speculation)",
                0,
            ),
        ];
        let rcs: Vec<RunCfg> = modes
            .iter()
            .map(|&(kv_mode, label, budget)| {
                let engine = if kv_mode == KvMode::LayerCoupled {
                    EngineKind::ExoRing
                } else {
                    EngineKind::Tesseract
                };
                RunCfg {
                    label: label.into(),
                    engine,
                    stages: ring.clone(),
                    head: ring.last().unwrap().0,
                    tess: TesseractOptions {
                        kv_mode,
                        budget,
                        braid_strands: 1,
                        draft_ahead: 0,
                        lookahead: false,
                        kv_remote_bias_pages: bias,
                        ..Default::default()
                    },
                    kv_caps: free.clone(),
                    inject: ctx,
                    note: String::new(),
                }
            })
            .collect();
        let mut per: Vec<Vec<Measured>> = rcs.iter().map(|_| Vec::new()).collect();
        for r in 0..s3_rounds {
            for (i, rc) in rcs.iter().enumerate() {
                eprint!("  ctx {ctx}: {} (round {}) … ", rc.label, r + 1);
                let t = Instant::now();
                let m = run_engine(&cluster, &cfg, rc, g, None);
                eprintln!("{:.1} tok/s ({:.1}s)", m.tok_s, t.elapsed().as_secs_f64());
                per[i].push(m);
            }
        }
        // paired estimator: each round runs the ring and both Tesseract
        // engines back to back, so a round's ratio compares runs that saw the
        // same host load; report the median of the per-round ratios
        let env3: Vec<f64> = per.iter().map(|v| envelope(v)).collect();
        let kept = kept_rounds(&per);
        let paired: Vec<(f64, f64, f64)> = (0..per.len())
            .map(|mi| {
                let mut r: Vec<f64> = kept
                    .iter()
                    .map(|&k| per[mi][k].tok_s / per[0][k].tok_s.max(1e-9))
                    .collect();
                r.sort_by(|a, b| a.partial_cmp(b).unwrap());
                (r[r.len() / 2], r[0], r[r.len() - 1])
            })
            .collect();
        for (mi, mut reps) in per.into_iter().enumerate() {
            let kv_mode = modes[mi].0;
            let label = modes[mi].1;
            reps.sort_by(|a, b| a.tok_s.partial_cmp(&b.tok_s).unwrap());
            let (lo_t, hi_t) = (reps[0].tok_s, reps[reps.len() - 1].tok_s);
            let mid = reps.len() / 2;
            let m = reps.swap_remove(mid);
            let (pr, plo, phi) = paired[mi];
            let ev = env3[mi] / env3[0].max(1e-9);
            // per-host memory = weights + KV (node + its KV service)
            let mut util = Vec::new();
            let mut peak_mb = 0.0f64;
            for (i, &h) in ids.iter().enumerate() {
                let kv: u64 = m
                    .stats
                    .iter()
                    .filter(|s| Assignment::host_of(s.node) == h)
                    .map(|s| s.kv_bytes)
                    .sum();
                let mem = counts[i] as f64 * layer_mb * 1e6 + kv as f64;
                peak_mb = peak_mb.max(mem / 1e6);
                util.push(mem / (ram_mb[i] * 1e6));
            }
            let mean = util.iter().sum::<f64>() / util.len() as f64;
            let maxu = util.iter().cloned().fold(0.0, f64::max);
            let oom = maxu > 1.0;
            let attn: u64 = m.stats.iter().map(|s| s.attn_bytes).sum();
            let steps = m.res.tokens.len().max(1) as f64;
            let (err_s, mig) = if kv_mode == KvMode::Disaggregated {
                // the live wire and placement: BFP partials, head-group striping
                let owners: Vec<(std::ops::Range<usize>, u16)> =
                    ring.iter().map(|&(n, s, e)| (s..e, n)).collect();
                let gn = cfg.n_kv_heads;
                let policy = PlacementPolicy::Striped {
                    holders: tesseract_fabric::kv_fabric::page_table::group_holders(
                        &owners[..1],
                        &caps_pairs,
                        1,
                        gn,
                        bias as usize,
                    ),
                    owners: vec![(0..gn, owners[0].1)],
                    local_pages: TesseractOptions::default().tier0_pages as usize,
                    mirror: false,
                };
                let (e, _, _) = lse_precision(
                    WirePrecision::Bfp,
                    &cfg,
                    &caps_pairs,
                    ctx as usize,
                    256,
                    policy,
                    gn,
                ); // migration equivalent: every remote page of every layer at FP16, per step
                let table = {
                    let mut t =
                        KvPageTable::new(256, &caps_pairs, PlacementPolicy::ProportionalFreeRam);
                    for p in 0..(ctx as usize).div_ceil(256) {
                        for l in 0..cfg.n_layers {
                            t.locate_or_allocate(l, p, 256 * cfg.kv_dim() * 8).unwrap();
                        }
                    }
                    t
                };
                let mut mig = 0usize;
                for (stage, s, e) in &ring {
                    for l in *s..*e {
                        for (n, pages) in table.ranges_for_layer(l) {
                            if n != *stage {
                                mig += pages.len() * 256 * cfg.kv_dim() * 2 * 2;
                            }
                        }
                    }
                }
                (format!("{e:.2e}"), format!("{:.0}", mig as f64 / 1024.0))
            } else {
                ("— (monolithic per layer)".into(), "—".into())
            };
            writeln!(
                out,
                "| {}K | {} | {:.2} | {lo_t:.2}–{hi_t:.2} | **{ev:.2}×** | {pr:.2}× ({plo:.2}–{phi:.2}) | {:.0} MB{} | {:.2} | {:.1} | {} | {} |",
                ctx / 1024,
                label,
                m.tok_s,
                peak_mb,
                if oom { " (OOM)" } else { "" },
                maxu / mean,
                attn as f64 / 1024.0 / steps,
                mig,
                err_s
            )
            .unwrap();
        }
    }
    writeln!(out, "\nAttention traffic for the disaggregated fabric includes the KV query broadcast, the (O, m, l) partials and the append of each step's newly committed K/V rows to their owning page.").unwrap();
}

// ---------------------------------------------------------------------------
// pillar micro-benchmarks
// ---------------------------------------------------------------------------

/// Target statistics and drafter agreement on one model: (mean top-1 prob,
/// entropy nats, zero-layer agree, 2-layer early-exit agree, early-exit mass).
fn drafter_agreement(cfg: &ModelConfig) -> DrafterAgreement {
    use tesseract_fabric::kv_fabric::{BatchMeta, LocalKvCache};
    use tesseract_fabric::model::{probs_from_logits, sample, NoObserver, ShardModel};
    let m = ReferenceModel::new(cfg);
    let mut rng = Rng::new(7);
    let mut toks: Vec<u32> = (0..16).map(|_| rng.below(cfg.vocab) as u32).collect();
    let mut kv = LocalKvCache::new(cfg.n_heads, cfg.n_kv_heads, cfg.d_head);
    let mut x = m.head.embed_tokens(&toks[..15]);
    let meta = BatchMeta::causal_prefill(0, 15, 0);
    m.shard.forward(&mut x, &meta, &mut kv, &mut NoObserver);
    kv.commit(&meta.keys, &|_| false);
    let (mut top1, mut ent) = (0.0f64, 0.0f64);
    let n = 256;
    for s in 0..n {
        let last = *toks.last().unwrap();
        let mut x = m.head.embed_tokens(&[last]);
        let meta = BatchMeta::causal_prefill(toks.len() - 1, 1, 100 + s as u64);
        m.shard.forward(&mut x, &meta, &mut kv, &mut NoObserver);
        kv.commit(&meta.keys, &|_| false);
        let p = probs_from_logits(m.head.logits(&x, cfg.rms_eps).row(0), 1.0);
        top1 += p.iter().cloned().fold(0.0, f32::max) as f64;
        ent += -p
            .iter()
            .filter(|v| **v > 0.0)
            .map(|v| (*v as f64) * (*v as f64).ln())
            .sum::<f64>();
        toks.push(sample(&p, &mut rng));
    }
    let rl = m.forward_sequence(&toks);
    let early = ShardModel::generate(cfg, 0..2, None);
    let mut kv2 = LocalKvCache::new(cfg.n_heads, cfg.n_kv_heads, cfg.d_head);
    let mut x2 = m.head.embed_tokens(&toks);
    early.forward(
        &mut x2,
        &BatchMeta::causal_prefill(0, toks.len(), 0),
        &mut kv2,
        &mut NoObserver,
    );
    let l2 = m.head.logits(&x2, cfg.rms_eps);
    // attention-only drafter: layer 0 with its FFN skipped (the V4 default)
    let mut attn = ShardModel::generate(cfg, 0..1, None);
    attn.skip_ffn = true;
    let mut kv1 = LocalKvCache::new(cfg.n_heads, cfg.n_kv_heads, cfg.d_head);
    let mut x1 = m.head.embed_tokens(&toks);
    attn.forward(
        &mut x1,
        &BatchMeta::causal_prefill(0, toks.len(), 0),
        &mut kv1,
        &mut NoObserver,
    );
    let l1 = m.head.logits(&x1, cfg.rms_eps);
    // the deployed drafter: same layer, two-stage shortlist LM head
    let (sr, sk) = shortlist_params(cfg);
    let mut sh = ShortlistHead::new(&m.head, sr, sk, 8, cfg.rms_eps);
    sh.certify = false;
    let l0 = m.head.logits(&m.head.embed_tokens(&toks), cfg.rms_eps);
    let (mut a0, mut a1, mut a1s, mut a2, mut q2) = (0usize, 0usize, 0usize, 0usize, 0.0f64);
    for t in 15..toks.len() - 1 {
        let tp = probs_from_logits(rl.row(t), 1.0);
        let ta = argmax(rl.row(t));
        a0 += (argmax(l0.row(t)) == ta) as usize;
        a1 += (argmax(l1.row(t)) == ta) as usize;
        a1s += (argmax(&sh.logits_row(x1.row(t))) == ta) as usize;
        let d2 = argmax(l2.row(t));
        a2 += (d2 == ta) as usize;
        q2 += tp[d2] as f64;
    }
    let k = (toks.len() - 16) as f64;
    DrafterAgreement {
        top1: top1 / n as f64,
        entropy: ent / n as f64,
        zero_layer: a0 as f64 / k,
        attn_only: a1 as f64 / k,
        attn_short: a1s as f64 / k,
        draft_macs: drafter_macs(cfg, sh.mean_macs()),
        step_macs: target_macs(cfg),
        two_layer: a2 as f64 / k,
        mass_two_layer: q2 / k,
    }
}

/// Weight MACs of one target decode step (all layers + LM head; MoE: router
/// + top-k experts). Attention over the context is excluded on both sides.
fn target_macs(cfg: &ModelConfig) -> f64 {
    let d = cfg.d_model as f64;
    let qo = 2.0 * d * (cfg.n_heads * cfg.d_head) as f64;
    let kv = 2.0 * d * (cfg.n_kv_heads * cfg.d_head) as f64;
    let ffn = match &cfg.moe {
        Some(m) => d * m.n_experts as f64 + (m.top_k * 3 * m.d_expert) as f64 * d,
        None => 3.0 * d * cfg.d_ff as f64,
    };
    cfg.n_layers as f64 * (qo + kv + ffn) + d * cfg.vocab as f64
}

/// Weight MACs of one drafted node: layer-0 attention projections + head.
fn drafter_macs(cfg: &ModelConfig, head_macs: f64) -> f64 {
    let d = cfg.d_model as f64;
    2.0 * d * (cfg.n_heads * cfg.d_head) as f64
        + 2.0 * d * (cfg.n_kv_heads * cfg.d_head) as f64
        + head_macs
}

struct DrafterAgreement {
    top1: f64,
    entropy: f64,
    zero_layer: f64,
    attn_only: f64,
    attn_short: f64,
    draft_macs: f64,
    step_macs: f64,
    two_layer: f64,
    mass_two_layer: f64,
}

fn pillar_micro(out: &mut String) {
    {
        writeln!(out, "\n### Drafter agreement with the target (256 sampled tokens, T = 1)\n\n| Benchmark model | Target mean top-1 prob | Target entropy | Zero-layer head top-1 agreement | 1 attention-only layer, full LM head (V4) | **1 attention-only layer + shortlist head (V5 default drafter)** | V5 drafter cost per node vs one target step (weight MACs) | 2-layer early-exit top-1 agreement | Target mass on 2-layer argmax |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|").unwrap();
        for (label, cfg) in [
            (
                "V1 (unstructured random weights)",
                ModelConfig::bench_moe_v1(),
            ),
            (
                "V2 (first-order transition prior)",
                ModelConfig::bench_moe_v2(),
            ),
            (
                "V3 (contextual: current + previous token, prev. via layer-0 attention)",
                ModelConfig::bench_moe(),
            ),
        ] {
            let d = drafter_agreement(&cfg);
            writeln!(
                out,
                "| {label} | {:.3} | {:.2} nats | {:.1}% | {:.1}% | {:.1}% | {:.2}% ({:.2}M / {:.1}M) | {:.1}% | {:.3} |",
                d.top1,
                d.entropy,
                d.zero_layer * 100.0,
                d.attn_only * 100.0,
                d.attn_short * 100.0,
                100.0 * d.draft_macs / d.step_macs,
                d.draft_macs / 1e6,
                d.step_macs / 1e6,
                d.two_layer * 100.0,
                d.mass_two_layer
            )
            .unwrap();
        }
        writeln!(out, "\nV2's next token was dominated by a first-order prior, so a zero-layer head already agreed 86.5% of the time. In V3 the next token depends equally on the current and the previous token, and the previous token only reaches the current position through a layer-0 attention head; a zero-layer head cannot see it, while the 2-layer early exit can. V4's default drafter is the target's own layer 0 with its MoE FFN skipped (attention + residual only, then the target's final norm and LM head): the previous-token head is all it needs, at a small fraction of a target step. It shares weights with the target and is not tuned to the data, so the honesty contrast with the zero-layer head still holds.").unwrap();
    }
    writeln!(out, "\n## Pillar micro-benchmarks\n").unwrap();
    // HRQ-EF on an LLM-like residual trajectory
    writeln!(out, "### Holographic residual compression (d = 4096, 128 autoregressive steps, synthetic low-rank residual trajectory with massive-activation channels)\n").unwrap();
    writeln!(out, "| Bits | Error feedback γ | Payload vs FP16 | Cumulative L2 drift | Min cosine | Drift reduction vs stateless |").unwrap();
    writeln!(out, "|---:|---:|---:|---:|---:|---:|").unwrap();
    let d = 4096;
    let traj = {
        let mut rng = Rng::new(77);
        let mut base: Vec<f32> = (0..d).map(|_| rng.normal()).collect();
        for c in [3usize, 97, 1024, 2900] {
            base[c] = 40.0;
        }
        let r = 16;
        let u: Vec<Vec<f32>> = (0..r)
            .map(|_| (0..d).map(|_| rng.normal()).collect())
            .collect();
        let fr: Vec<f32> = (0..r).map(|_| 0.02 + 0.2 * rng.uniform()).collect();
        (0..128)
            .map(|t| {
                let mut h = base.clone();
                for k in 0..r {
                    let a = 0.12 * (fr[k] * t as f32).sin();
                    for i in 0..d {
                        h[i] += a * u[k][i];
                    }
                }
                for v in h.iter_mut() {
                    *v += 0.03 * rng.normal();
                }
                h
            })
            .collect::<Vec<_>>()
    };
    for bits in [2u8, 3, 4] {
        let run = |gamma: f32| {
            let cfg = HoloConfig {
                bits,
                gamma,
                ..Default::default()
            };
            let mut tx = HolographicCompressor::new(d, cfg.clone());
            let mut rx = HolographicDecompressor::new(d, cfg);
            let (mut drift, mut minc, mut bytes) = (0.0f64, 1.0f64, 0usize);
            for (t, h) in traj.iter().enumerate() {
                let f = tx.compress(h);
                if t > 0 {
                    bytes += f.wire_bytes();
                }
                let hh = rx.decompress(&f).to_vec();
                drift += h
                    .iter()
                    .zip(&hh)
                    .map(|(a, b)| ((a - b) as f64).powi(2))
                    .sum::<f64>()
                    .sqrt();
                minc = minc.min(tesseract_fabric::tensor::cosine(h, &hh));
            }
            (drift, minc, bytes)
        };
        let (de, ce, be) = run(0.95);
        let (ds, _, _) = run(0.0);
        writeln!(
            out,
            "| {bits} | 0.95 | −{:.1}% | {de:.1} | {ce:.5} | ×{:.2} |",
            100.0 * (1.0 - be as f64 / (127 * d * 2) as f64),
            ds / de
        )
        .unwrap();
    }
    // determinism
    {
        use tesseract_fabric::tensor::determinism::{emulate, CanonicalBoundaryAnchor};
        let a = CanonicalBoundaryAnchor::new(64);
        let mut rng = Rng::new(3);
        let (mut naive_div, trials) = (0, 100);
        for _ in 0..trials {
            let x: Vec<f32> = (0..4096).map(|_| rng.normal() * 8.0).collect();
            let s = [
                emulate::sum_sq_scalar(&x),
                emulate::sum_sq_lanes(&x, 4),
                emulate::sum_sq_lanes(&x, 8),
                emulate::sum_sq_lanes(&x, 32),
                emulate::sum_sq_fma(&x),
            ];
            if s.iter().any(|v| v.to_bits() != s[0].to_bits()) {
                naive_div += 1;
            }
            let b = a.encode(&x);
            let mut rb = b.clone();
            rb.reverse();
            assert_eq!(
                a.exact_sum_squares(&b).to_bits(),
                a.exact_sum_squares(&rb).to_bits()
            );
        }
        writeln!(out, "\n### Cross-silicon determinism (Q8.24 CanonicalBoundaryAnchor)\n\nRMSNorm Σx² over 4096-wide activations: scalar / NEON-4 / AVX2-8 / warp-32 / FMA reduction orders disagree bitwise in {naive_div}/{trials} trials with IEEE f32; the Q8.24 integer accumulation is bit-identical in {trials}/{trials}.").unwrap();
    }
    // KV fabric traffic scaling
    {
        writeln!(out, "\n### Disaggregated attention traffic per decode step (one layer, 8 query heads, 4 shards; bytes incl. frame headers)\n\nV1–V4 stripe pages over all 4 shards (3 remote KV services per layer); V5 pairs the layer with one partner (tier 0 local, later pages split evenly between the two), so a step contacts 1 remote service; V6.2 stripes by KV-head group (each KV-head group of a layer is its own virtual layer: group 0 stays with the computing node plus remote help, the others live on remote nodes, and each holder receives only its group's query heads). The BFP columns use the current wire (16-bit block-floating-point queries, weight-bound-coded partials); V4 shipped 14-bit queries (6.3 KB, 6.3e-5 at 4K).\n\n| Context | V1 f32 wire | V3 f16 wire | BFP wire, striped over 4 shards | BFP wire, paired (V5) | **BFP wire, head-group striped (V6.2 default)** | FP16 page migration | V3 reduction | V6.2 reduction | V3 max abs err | striped BFP max abs err | V5 paired max abs err | **V6.2 max abs err** |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|").unwrap();
        let cfg = ModelConfig::long_context();
        let caps = [
            (0u16, 400usize << 20),
            (1, 200 << 20),
            (2, 150 << 20),
            (3, 100 << 20),
        ];
        for ctx in [4096usize, 16384, 65536] {
            let pp = || PlacementPolicy::ProportionalFreeRam;
            let (_, n32, mig) = lse_precision(WirePrecision::F32, &cfg, &caps, ctx, 256, pp(), 1);
            let (e16, n16, _) = lse_precision(WirePrecision::F16, &cfg, &caps, ctx, 256, pp(), 1);
            let (eb, nb, _) = lse_precision(WirePrecision::Bfp, &cfg, &caps, ctx, 256, pp(), 1);
            let owners = vec![(0..cfg.n_layers, 0u16)];
            let paired = PlacementPolicy::Paired {
                partners: tesseract_fabric::kv_fabric::page_table::pair_partners(
                    &owners,
                    &caps,
                    cfg.n_layers,
                ),
                owners,
                local_pages: TesseractOptions::default().tier0_pages as usize,
                remote_bias: 0,
            };
            let (ep, np, _) = lse_precision(WirePrecision::Bfp, &cfg, &caps, ctx, 256, paired, 1);
            let gn = cfg.n_kv_heads;
            let striped = PlacementPolicy::Striped {
                holders: tesseract_fabric::kv_fabric::page_table::group_holders(
                    &[(0..1, 0u16)],
                    &caps,
                    1,
                    gn,
                    0,
                ),
                owners: vec![(0..gn, 0u16)],
                local_pages: TesseractOptions::default().tier0_pages as usize,
                mirror: false,
            };
            let (eg, ng, _) = lse_precision(WirePrecision::Bfp, &cfg, &caps, ctx, 256, striped, gn);
            let red = |n: usize| 100.0 * (1.0 - n as f64 / mig as f64);
            writeln!(out, "| {}K | {:.1} KB | {:.1} KB | {:.1} KB | {:.1} KB | {:.1} KB | {:.0} KB | −{:.2}% | −{:.2}% | {e16:.2e} | {eb:.2e} | {ep:.2e} | {eg:.2e} |", ctx / 1024, n32 as f64 / 1024.0, n16 as f64 / 1024.0, nb as f64 / 1024.0, np as f64 / 1024.0, ng as f64 / 1024.0, mig as f64 / 1024.0, red(n16), red(ng)).unwrap();
        }
    }
    // tiled batch GEMM
    {
        use tesseract_fabric::tensor::Linear;
        let mut rng = Rng::new(5);
        writeln!(out, "\n### Weight-reuse tiled batch GEMM (single core, min of 300 runs)\n\n| Weight matrix | B=1 | B=4 | B=6 | B=8 | B=16 | cost B=6 / B=1 | cost B=8 / B=1 |\n|---|---:|---:|---:|---:|---:|---:|---:|").unwrap();
        for (n, k) in [(2048usize, 512usize), (704, 512), (512, 704)] {
            let lin = Linear::new(Tensor2D::randn(n, k, 0.05, &mut rng));
            let mut t = Vec::new();
            for m in [1usize, 4, 6, 8, 16] {
                let x = Tensor2D::randn(m, k, 1.0, &mut rng);
                let mut best = f64::MAX;
                for _ in 0..300 {
                    let t0 = Instant::now();
                    std::hint::black_box(lin.forward(std::hint::black_box(&x)));
                    best = best.min(t0.elapsed().as_secs_f64());
                }
                t.push((m, best));
            }
            let g = |i: usize| format!("{:.1} GMAC/s", (t[i].0 * n * k) as f64 / t[i].1 / 1e9);
            writeln!(
                out,
                "| {n}×{k} | {} | {} | {} | {} | {} | {:.2}× | {:.2}× |",
                g(0),
                g(1),
                g(2),
                g(3),
                g(4),
                t[2].1 / t[0].1,
                t[3].1 / t[0].1
            )
            .unwrap();
        }
    }
    // intra-node lane pool: exact 4-row verification vs a 1-row step, with the
    // SAME lanes and striping rule for both
    {
        use tesseract_fabric::model::{
            moe_combine, moe_combine_lanes, route_topk, FfnWeights, LayerWeights,
        };
        use tesseract_fabric::tensor::pool::LanePool;
        let cfg = ModelConfig::bench_moe();
        let layer = LayerWeights::generate(&cfg, 5, None);
        if let FfnWeights::Moe {
            router,
            bank,
            top_k,
        } = &layer.ffn
        {
            writeln!(out, "\n### Intra-node lane pool: one MoE layer, exact routing, 1-row step vs 4-row tree (min of 60 trials, interleaved)\n\nBoth batch sizes use the same lanes and the same striping rule (every active expert split into ⌈lanes / active experts⌉ hidden-unit and output stripes), so a 1-row step's 2 experts are parallelised as finely as the tree's 5–6.\n\n| Lanes | 1 row | 4 rows | 4 rows / 1 row | max |Δ| vs sequential |\n|---:|---:|---:|---:|---:|").unwrap();
            let mut rng = Rng::new(21);
            let xs: Vec<Tensor2D> = (0..60)
                .map(|_| Tensor2D::randn(4, cfg.d_model, 1.0, &mut rng))
                .collect();
            for lanes in [1usize, 2, 4, 8] {
                let pool = LanePool::new(lanes);
                let (mut t1, mut t4, mut dmax) = (f64::MAX, f64::MAX, 0.0f32);
                for x in &xs {
                    let x1 = Tensor2D::from_vec(1, cfg.d_model, x.row(0).to_vec());
                    let r1 = route_topk(router, &x1, *top_k);
                    let r4 = route_topk(router, x, *top_k);
                    let t0 = Instant::now();
                    std::hint::black_box(moe_combine_lanes(bank, &x1, &r1, &pool));
                    t1 = t1.min(t0.elapsed().as_secs_f64());
                    let t0 = Instant::now();
                    let y = moe_combine_lanes(bank, x, &r4, &pool);
                    t4 = t4.min(t0.elapsed().as_secs_f64());
                    let yr = moe_combine(bank, x, &r4);
                    for (a, b) in y.data.iter().zip(&yr.data) {
                        dmax = dmax.max((a - b).abs());
                    }
                }
                writeln!(
                    out,
                    "| {lanes} | {:.1} µs | {:.1} µs | {:.2}× | {dmax:.1e} |",
                    t1 * 1e6,
                    t4 * 1e6,
                    t4 / t1
                )
                .unwrap();
            }
        }
    }
    // anchor-consensus expert coalescing (approximate)
    {
        use tesseract_fabric::model::{
            moe_combine, route_coalesced, route_topk, FfnWeights, LayerWeights,
        };
        let cfg = ModelConfig::bench_moe();
        let layer = LayerWeights::generate(&cfg, 5, None);
        if let FfnWeights::Moe {
            router,
            bank,
            top_k,
        } = &layer.ffn
        {
            let mut rng = Rng::new(12);
            let mut row_err = 0.0f64;
            let (mut t1, mut te, mut tc) = (f64::MAX, f64::MAX, f64::MAX);
            let mut experts_exact = 0usize;
            let trials = 40;
            for _ in 0..trials {
                let x = Tensor2D::randn(4, cfg.d_model, 1.0, &mut rng);
                let x1 = Tensor2D::from_vec(1, cfg.d_model, x.row(0).to_vec());
                let re = route_topk(router, &x, *top_k);
                let rc = route_coalesced(router, &x, *top_k);
                experts_exact += re
                    .iter()
                    .flat_map(|r| r.0.clone())
                    .collect::<std::collections::BTreeSet<_>>()
                    .len();
                let t0 = Instant::now();
                std::hint::black_box(moe_combine(bank, &x1, &route_topk(router, &x1, *top_k)));
                t1 = t1.min(t0.elapsed().as_secs_f64());
                let t0 = Instant::now();
                let ye = moe_combine(bank, &x, &re);
                te = te.min(t0.elapsed().as_secs_f64());
                let t0 = Instant::now();
                let yc = moe_combine(bank, &x, &rc);
                tc = tc.min(t0.elapsed().as_secs_f64());
                for i in 1..4 {
                    let num: f64 = ye
                        .row(i)
                        .iter()
                        .zip(yc.row(i))
                        .map(|(a, b)| ((a - b) as f64).powi(2))
                        .sum();
                    let den: f64 = ye.row(i).iter().map(|a| (*a as f64).powi(2)).sum();
                    row_err += (num / den.max(1e-30)).sqrt() / 3.0;
                }
            }
            writeln!(out, "\n### Anchor-consensus expert coalescing (APPROXIMATE; one MoE layer, 4-row tree, min of {trials} trials)\n\n| | Distinct experts touched | Time | vs 1 row |\n|---|---:|---:|---:|\n| 1 row (anchor only) | 2 | {:.1} µs | 1.00× |\n| 4 rows, exact routing | {:.1} (mean) | {:.1} µs | {:.2}× |\n| 4 rows, coalesced (anchor's experts + 1 consensus) | ≤ 3 | {:.1} µs | {:.2}× |\n\nCoalescing changes the candidate rows' FFN outputs by {:.1}% (relative L2, mean over candidate rows), so trees verified with it are not distributed exactly as the target model; it is therefore opt-in (`coalesce_experts`) and off in every scenario run.", t1 * 1e6, experts_exact as f64 / trials as f64, te * 1e6, te / t1, tc * 1e6, tc / t1, 100.0 * row_err / trials as f64).unwrap();
        }
    }
    // slab commit
    {
        use tesseract_fabric::kv_fabric::{node_key, BatchMeta, LocalKvCache};
        let mut kv = LocalKvCache::new(8, 4, 64);
        let mut rng = Rng::new(6);
        let mut ts = Vec::new();
        for it in 0..500u64 {
            let k = Tensor2D::randn(16, 256, 1.0, &mut rng);
            let meta = BatchMeta {
                positions: (0..16).collect(),
                keys: (0..16).map(|i| node_key(it, i)).collect(),
                visible: vec![vec![]; 16],
            };
            for l in 0..16 {
                kv.push_pending(l, &k, &k, &meta);
            }
            let acc: Vec<u64> = [0usize, 1, 4, 9].iter().map(|&i| node_key(it, i)).collect();
            let t0 = Instant::now();
            kv.commit(&acc, &|_| false);
            ts.push(t0.elapsed().as_nanos() as f64 / 1e3);
        }
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        writeln!(out, "\n### Lock-free slab KV commit (16 layers, 16 pending rows → 4 accepted)\n\nMedian {:.2} µs, p90 {:.2} µs, p99 {:.2} µs for the whole shard (V1: per-layer copy + compaction, ≈2 µs per layer in isolation, 14–16 µs per commit inside the loaded V1 cluster).", ts[ts.len() / 2], ts[ts.len() * 9 / 10], ts[ts.len() * 99 / 100]).unwrap();
    }
    // multipath striping + brownout
    {
        let a = Endpoint::bind(1, "127.0.0.1:0").unwrap();
        let b = Endpoint::bind(2, "127.0.0.1:0").unwrap();
        a.shared.set_addr(2, b.addr());
        a.shared
            .set_links(2, vec![LinkProfile::ethernet_2_5g(), LinkProfile::wifi6e()]);
        b.shared.set_addr(1, a.addr());
        b.shared
            .set_links(1, vec![LinkProfile::ethernet_2_5g(), LinkProfile::wifi6e()]);
        let frame_bytes = 512 * 1024;
        // goodput from the median per-frame delivery time (robust to host noise)
        let phase = |n: usize| -> (f64, Vec<u64>) {
            let mut times = Vec::with_capacity(n);
            for _ in 0..n {
                let t0 = mono_ns();
                a.send_payload(2, FrameType::ActivationDelta, 0, 0, vec![7u8; frame_bytes])
                    .unwrap();
                let _ = b.recv_match(&|f| f.header.frame_type == FrameType::ActivationDelta);
                times.push((mono_ns() - t0) as f64 * 1e-9);
            }
            times.sort_by(|x, y| x.partial_cmp(y).unwrap());
            let s = a.shared.scheduler_snapshot(2).unwrap();
            (
                frame_bytes as f64 * 8.0 / times[n / 2] / 1e9,
                s.links.iter().map(|l| l.bytes_sent).collect(),
            )
        };
        a.shared
            .stripe_threshold
            .store(u64::MAX, std::sync::atomic::Ordering::Relaxed);
        let (single, _) = phase(24);
        a.shared
            .stripe_threshold
            .store(64 * 1024, std::sync::atomic::Ordering::Relaxed);
        let (_, warm) = phase(8);
        let (striped, split1) = phase(24);
        let mut brown = LinkProfile::ethernet_2_5g();
        brown.one_way_us = 8000.0;
        brown.bandwidth_gbps = 0.1;
        a.shared.reshape_link(2, 0, brown).unwrap();
        let (_, before) = phase(8);
        let (during, after) = phase(24);
        let shift = |x: &Vec<u64>, y: &Vec<u64>| -> String {
            format!(
                "2.5GbE {:.0}% / WiFi {:.0}%",
                100.0 * (y[0] - x[0]) as f64 / ((y[0] - x[0]) + (y[1] - x[1])) as f64,
                100.0 * (y[1] - x[1]) as f64 / ((y[0] - x[0]) + (y[1] - x[1])) as f64
            )
        };
        writeln!(out, "\n### Multipath striping (512 KB frames, 2.5GbE + Wi-Fi 6E, live sockets)\n\n| Mode | Goodput, median frame (Gbps) | Byte split |\n|---|---:|---|\n| Best single link | {single:.2} | 2.5GbE 100% |\n| ETTC water-filling stripes | {striped:.2} | {} |\n| 2.5GbE brownout (8 ms, 0.1 Gbps) — after telemetry adapts | {during:.2} | {} |", shift(&warm, &split1), shift(&before, &after)).unwrap();
        a.shared.close();
        b.shared.close();
    }
}

// ---------------------------------------------------------------------------

fn main() {
    tesseract_fabric::tensor::prefer_performance_cores();
    let args: Vec<String> = std::env::args().collect();
    let quick = args.iter().any(|a| a == "--quick");
    let out_path = args
        .iter()
        .position(|a| a == "--out")
        .and_then(|i| args.get(i + 1).cloned())
        .unwrap_or_else(|| "benchmarks/bench_results.md".into());
    let t_all = Instant::now();
    let cfg = ModelConfig::bench_moe();
    let mut out = String::new();
    let cpu = std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default();
    let load = std::process::Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default();
    writeln!(out, "# tesseract-bench results\n").unwrap();
    writeln!(out, "Host: {} · {} logical CPUs · every node is a set of OS threads with its own TCP endpoints on 127.0.0.1; all traffic is real `TesseractWire` frames through the kernel's TCP stack, delayed by per-link shapers (FIFO serialisation at link bandwidth + one-way latency + jitter). Compute is real FP32 tensor math on this host; slower devices are emulated by extending each node's measured compute time (its thread-CPU critical path; `TESSERACT_SHAPER=wall` restores the ≤ V10.0 wall-time model) by its memory-bandwidth ratio to the fastest node.", cpu.trim(), std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0)).unwrap();
    writeln!(out, "\nHost load average at start: `{}` — background load on the host competes with the emulated nodes for cores and adds variance to every timing below.", load.trim()).unwrap();
    writeln!(out, "\nModel: synthetic Llama/Qwen-style MoE decoder — {} layers, d_model {}, {} q / {} kv heads, {} experts top-{} (d_expert {}), vocab {}, {:.1}M parameters, same weights on every engine. V3 uses a *contextual* initialisation: the next token depends equally on the current token (+{} nats transition prior) and the previous token (+{} nats), and the previous token reaches the current position only through a layer-0 attention head built with RoPE to attend to offset −1; power-law spectral decay σᵢ ∝ i^-{} in layer weights; logit noise σ={}. It is calibrated to V1's peakedness (mean top-1 ≈ 0.6). V2 used a first-order prior only; V1 unstructured random weights. {} of {} experts per layer fit the hot buffer on every engine; the rest are staged from int8 cold storage.", cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.moe.as_ref().unwrap().n_experts, cfg.moe.as_ref().unwrap().top_k, cfg.moe.as_ref().unwrap().d_expert, cfg.vocab, cfg.total_params() as f64 / 1e6, cfg.transition_prior, cfg.context_prior, cfg.spectral_decay, cfg.logit_std, cfg.moe.as_ref().unwrap().n_experts * 3 / 4, cfg.moe.as_ref().unwrap().n_experts).unwrap();
    let mut rng = Rng::new(2024);
    let g = Gen {
        prompt: (0..16).map(|_| rng.below(cfg.vocab) as u32).collect(),
        max_new: if quick { 48 } else { 128 },
        seed: 11,
        temperature: 0.0,
    };
    writeln!(out, "\nGeneration: single stream, {}-token prompt, {} new tokens, temperature {} (greedy decoding: every engine must emit exactly the FP32 reference's argmax sequence up to its own numerical error; a T = 1 sampled ablation follows Scenario 1). Network bytes count inter-host frames only (incl. prompt prefill). Scenarios 1–2 run all engines back to back in each of N interleaved rounds (identical seeds) and report each engine's median round, so transient host load hits every engine of a round alike.", g.prompt.len(), g.max_new, g.temperature).unwrap();
    let sel = args
        .iter()
        .position(|a| a == "--scenarios")
        .and_then(|i| args.get(i + 1).cloned())
        .unwrap_or_else(|| "1,2,3,micro,v8,v9,v10".into());
    let want = |k: &str| sel.split(',').any(|x| x.trim() == k);
    let rounds: usize = args
        .iter()
        .position(|a| a == "--rounds")
        .and_then(|i| args.get(i + 1))
        .and_then(|x| x.parse().ok())
        .unwrap_or(if quick { 2 } else { 5 });
    eprintln!("building FP32 reference model …");
    let reference = ReferenceModel::new(&cfg);
    let studio = preset("studio-m5max").unwrap();
    if want("1") {
        eprintln!("Scenario 1");
        quiet_host(&mut out, "Scenario 1");
        let s1 = ClusterDef {
            nodes: (0..4).map(|i| (i, studio)).collect(),
            studio_tb5: true,
        };
        scenario_pipeline("Scenario 1 — Homogeneous High-Speed Studio Cluster (4× Mac Studio, Thunderbolt 5 RDMA profile: 40 µs RTT, 80 Gbps)", &s1, &cfg, &g, &reference, rounds, &mut out);
        temperature_ablation(&s1, &cfg, &g, &reference, rounds.min(3), 1.0, &mut out);
        htep_step_ratio(&s1, &cfg, &g, &reference, (3 * rounds).max(3), &mut out);

        bits_ablation(&s1, &cfg, &g, &reference, &mut out);
    }
    if want("codec") && !want("1") {
        // the activation-codec frontier alone (it also runs inside Scenario 1)
        let s1 = ClusterDef {
            nodes: (0..4).map(|i| (i, studio)).collect(),
            studio_tb5: true,
        };
        bits_ablation(&s1, &cfg, &g, &reference, &mut out);
    }
    if want("2") {
        eprintln!("Scenario 2");
        quiet_host(&mut out, "Scenario 2");
        let s2 = ClusterDef {
            nodes: vec![
                (0, studio),
                (1, preset("rtx5090").unwrap()),
                (2, preset("macbook-pro").unwrap()),
                (3, preset("android-snapdragon").unwrap()),
            ],
            studio_tb5: false,
        };
        scenario_pipeline("Scenario 2 — Heterogeneous Multi-Platform Cluster (Mac Studio + RTX 5090 on 2.5GbE, MacBook Pro + Android on Wi-Fi 6E)", &s2, &cfg, &g, &reference, rounds, &mut out);
    }
    if want("3") {
        eprintln!("Scenario 3");
        quiet_host(&mut out, "Scenario 3");
        let lc_gen = Gen {
            max_new: if quick { 8 } else { 48 },
            ..g.clone()
        };
        let contexts: Vec<u32> = match args
            .iter()
            .position(|a| a == "--contexts")
            .and_then(|i| args.get(i + 1))
        {
            Some(c) => c
                .split(',')
                .map(|x| x.trim().parse().expect("--contexts"))
                .collect(),
            None if quick => vec![2048, 8192],
            None => vec![2048, 8192, 32768, 65536],
        };
        // the paired-ratio estimator needs more rounds than a median of
        // throughputs: 7 by default
        scenario_long_context(&contexts, &lc_gen, if quick { 2 } else { 7 }, &mut out);
    }
    if want("micro") {
        eprintln!("Pillar micro-benchmarks");
        pillar_micro(&mut out);
    }
    if want("v8") {
        eprintln!("Section 4 — V8.0 production runtime");
        production_v8(&cfg, &g, &reference, if quick { 1 } else { 3 }, &mut out);
    }
    if want("v9") {
        eprintln!("Section 5 — V9.0 real checkpoints");
        production_v9(&cfg, &reference, &mut out);
    }
    if want("v10") {
        eprintln!("Section 6 — V10.0");
        quiet_host(&mut out, "Section 6");
        production_v10(&mut out);
    }
    writeln!(
        out,
        "\n_Total benchmark wall time: {:.1} s._",
        t_all.elapsed().as_secs_f64()
    )
    .unwrap();
    println!("{out}");
    if let Some(dir) = std::path::Path::new(&out_path).parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir).expect("create results directory");
        }
    }
    std::fs::write(&out_path, &out).expect("write results");
    eprintln!("results written to {out_path}");
}

// ---------------------------------------------------------------------------
// Section 4 — V8.0 production runtime
// ---------------------------------------------------------------------------

/// Greedy single-node decode of the whole model on one node thread with the
/// same 2-lane pool every engine node installs: (prefill tok/s over
/// `prefill_len` tokens, decode tok/s over `n_new` tokens, ms to the first
/// generated token).
fn single_node(
    cfg: &ModelConfig,
    prompt: &[u32],
    prefill_len: usize,
    n_new: usize,
) -> (f64, f64, f64) {
    use tesseract_fabric::kv_fabric::{BatchMeta, LocalKvCache};
    use tesseract_fabric::model::{NoObserver, ShardModel};
    let lanes = TesseractOptions::default().node_lanes as usize;
    tesseract_fabric::tensor::pool::install(lanes);
    let t0 = Instant::now();
    let head = HeadWeights::generate(cfg);
    let model = ShardModel::generate(cfg, 0..cfg.n_layers, None);
    // prefill throughput: one batched causal pass over `prefill_len` tokens
    let long: Vec<u32> = (0..prefill_len).map(|i| prompt[i % prompt.len()]).collect();
    let mut kv = LocalKvCache::new(cfg.n_heads, cfg.n_kv_heads, cfg.d_head);
    let meta = BatchMeta::causal_prefill(0, prefill_len, 1);
    let mut x = head.embed_tokens(&long);
    let tp = Instant::now();
    model.forward(&mut x, &meta, &mut kv, &mut NoObserver);
    let prefill_s = tp.elapsed().as_secs_f64();
    // first token from a fresh start, then greedy decode
    let mut kv = LocalKvCache::new(cfg.n_heads, cfg.n_kv_heads, cfg.d_head);
    let n = prompt.len();
    let meta = BatchMeta::causal_prefill(0, n - 1, 1);
    let mut x = head.embed_tokens(&prompt[..n - 1]);
    model.forward(&mut x, &meta, &mut kv, &mut NoObserver);
    kv.commit(&meta.keys, &|_| false);
    let mut toks = prompt.to_vec();
    let mut first_ms = 0.0;
    let td = Instant::now();
    for step in 0..n_new {
        let pos = toks.len() - 1;
        let meta = BatchMeta::causal_prefill(pos, 1, 2 + pos as u64);
        let mut x = head.embed_tokens(&toks[pos..]);
        model.forward(&mut x, &meta, &mut kv, &mut NoObserver);
        kv.commit(&meta.keys, &|_| false);
        toks.push(argmax(head.logits(&x, cfg.rms_eps).row(0)) as u32);
        if step == 0 {
            first_ms = t0.elapsed().as_secs_f64() * 1e3 - prefill_s * 1e3;
        }
    }
    let decode_s = td.elapsed().as_secs_f64();
    tesseract_fabric::tensor::pool::install(1);
    (
        prefill_len as f64 / prefill_s,
        n_new as f64 / decode_s,
        first_ms,
    )
}

fn production_v8(
    cfg: &ModelConfig,
    g: &Gen,
    reference: &ReferenceModel,
    rounds: usize,
    out: &mut String,
) {
    use tesseract_fabric::loader::sharded_loader::{config_from_file, export_gguf};
    use tesseract_fabric::loader::GgufFile;
    use tesseract_fabric::tensor::accel::{self, Backend};
    use tesseract_fabric::tensor::quant::{active_kernel, WeightFormat};
    let dir = std::env::temp_dir().join(format!("tesseract-bench-v8-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let tok = tesseract_fabric::tokenizer::synthetic_tokenizer(cfg.vocab);
    writeln!(out, "\n## Section 4 — V8.0 production runtime: GGUF files over `mmap`, quantized kernels, native acceleration\n").unwrap();
    writeln!(out, "The Scenario 1–3 model exported to real GGUF v3 files (llama.cpp tensor names; Mixtral-style stacked experts; norms and routers F32; in the Q4_0 file `output.weight` is Q8_0, as llama.cpp keeps it above Q4_0) and loaded back through `mmap`. The file was just written, so its pages are in the OS page cache: load times are *warm-cache* times (a disk-cold load adds the read of the pages a node touches). Integer-dot kernel: {}. `portable` = the in-tree kernels used by every engine comparison above; `accelerate` = Apple Accelerate `cblas_sgemm` (AMX) for dense FP32 GEMMs and for dequantized tiles of batches ≥ {} rows (its internal threads are not part of the equal-lanes rule, so it is reported here only).", active_kernel().name(), accel::ACCEL_MIN_ROWS).unwrap();
    let seq: Vec<u32> = g.prompt.iter().cycle().take(64).copied().collect();
    let ref_logits = reference.forward_sequence(&seq);
    let fid = |c: &ModelConfig| {
        let l = ReferenceModel::new(c).forward_sequence(&seq);
        let mut cos = 0.0;
        let mut agree = 0usize;
        for i in 0..seq.len() {
            cos += logit_fidelity(l.row(i), ref_logits.row(i));
            agree += (argmax(l.row(i)) == argmax(ref_logits.row(i))) as usize;
        }
        (cos / seq.len() as f64, agree as f64 / seq.len() as f64)
    };
    let studio = preset("studio-m5max").unwrap();
    let s1 = ClusterDef {
        nodes: (0..4).map(|i| (i, studio)).collect(),
        studio_tb5: true,
    };
    let caps: Vec<u64> = s1
        .specs()
        .iter()
        .map(|s| (s.available_ram_mb * 1e6 * 0.15) as u64)
        .collect();
    let rc = RunCfg {
        label: "H-TEP".into(),
        engine: EngineKind::TesseractHtep,
        stages: (0..4).map(|i| (i, 0, cfg.n_layers)).collect(),
        head: 0,
        tess: TesseractOptions::default(),
        kv_caps: caps,
        inject: 0,
        note: String::new(),
    };
    let fp32_bytes = cfg.total_params() as f64 * 4.0;
    let mut load_rows = Vec::new();
    let mut tput_rows = Vec::new();
    let formats = [
        None,
        Some(WeightFormat::F32),
        Some(WeightFormat::F16),
        Some(WeightFormat::Q8_0),
        Some(WeightFormat::Q4_0),
    ];
    for fmt in formats {
        let (lcfg, name) = match fmt {
            None => (
                cfg.clone(),
                "in-memory FP32 (synthetic, V7 path)".to_string(),
            ),
            Some(f) => {
                let path = dir.join(format!("bench-{}.gguf", f.name()));
                let path = path.to_string_lossy().into_owned();
                eprintln!("  exporting {} …", f.name());
                let size = export_gguf(cfg, f, Some(&tok), &path).expect("export");
                let t = Instant::now();
                let gf = GgufFile::open(&path).expect("open");
                let map_ms = t.elapsed().as_secs_f64() * 1e3;
                drop(gf);
                let t = Instant::now();
                let heap = std::fs::read(&path).expect("read");
                let read_ms = t.elapsed().as_secs_f64() * 1e3;
                drop(heap);
                let t = Instant::now();
                let lcfg = config_from_file(&path, None).expect("config");
                let model =
                    tesseract_fabric::model::ShardModel::generate(&lcfg, 0..lcfg.n_layers, None);
                let head = HeadWeights::generate(&lcfg);
                let views_ms = t.elapsed().as_secs_f64() * 1e3;
                let resident: usize =
                    model.layers.iter().map(|l| l.bytes()).sum::<usize>() + head.lm_head.bytes();
                drop((model, head));
                let (cos, top1) = fid(&lcfg);
                load_rows.push(format!(
                    "| {} | {:.1} | {:.2}× | {:.2} | {:.1} | {:.1} | {:.1} | {:.5} ({:.1}%) |",
                    f.name(),
                    size as f64 / 1e6,
                    fp32_bytes / size as f64,
                    map_ms,
                    read_ms,
                    views_ms,
                    resident as f64 / 1e6,
                    cos,
                    top1 * 100.0
                ));
                (lcfg, format!("GGUF {}", f.name()))
            }
        };
        let mut cells = vec![name];
        for b in [Backend::Portable, Backend::Accelerate] {
            if b == Backend::Accelerate && !accel::ACCELERATE_AVAILABLE {
                cells.push("n/a".into());
                cells.push("n/a".into());
                cells.push("n/a".into());
                continue;
            }
            accel::set_backend(b);
            let (pf, dec, first) = single_node(&lcfg, &g.prompt, 128, 48);
            // 4-node H-TEP, interleaved rounds, median
            let mut ts: Vec<f64> = (0..rounds)
                .map(|_| {
                    run_engine(
                        &s1,
                        &lcfg,
                        &rc,
                        &Gen {
                            max_new: 64,
                            ..g.clone()
                        },
                        None,
                    )
                    .tok_s
                })
                .collect();
            ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
            cells.push(format!("{pf:.0} / {dec:.1} / {first:.0} ms"));
            cells.push(format!("{:.1}", ts[ts.len() / 2]));
            cells.push(String::new());
        }
        accel::set_backend(Backend::Portable);
        tput_rows.push(format!(
            "| {} | {} | {} | {} | {} |",
            cells[0], cells[1], cells[2], cells[4], cells[5]
        ));
    }
    writeln!(out, "\n#### Footprint, load and fidelity\n").unwrap();
    writeln!(out, "| Format | GGUF file (MB) | FP32 bytes / file bytes | `mmap` + header parse (ms) | heap read of the whole file, for contrast (ms) | all layer views + head ready (ms) | weight bytes addressed by the full model (MB) | teacher-forced fidelity vs in-memory FP32 (cosine, top-1 of 64 positions) |").unwrap();
    writeln!(out, "|---|---:|---:|---:|---:|---:|---:|---:|").unwrap();
    for r in &load_rows {
        writeln!(out, "{r}").unwrap();
    }
    writeln!(out, "\nF32 files are copied into the engine's dense tiled layout at load (that is the \"views ready\" time); F16 / Q8_0 / Q4_0 matrices stay zero-copy views into the mapping, and only the embedding table (decoded to f32), norms, routers and the re-ordered Q/K rows are materialised.").unwrap();
    writeln!(
        out,
        "\n#### Throughput (greedy, {}-token prompt)\n",
        g.prompt.len()
    )
    .unwrap();
    writeln!(out, "| Weights | single node, portable: prefill tok/s (128 tokens) / decode tok/s / first token | 4-node H-TEP decode tok/s, portable (median of {rounds}) | single node, Accelerate: prefill / decode / first token | 4-node H-TEP decode tok/s, Accelerate (median of {rounds}) |").unwrap();
    writeln!(out, "|---|---:|---:|---:|---:|").unwrap();
    for r in &tput_rows {
        writeln!(out, "{r}").unwrap();
    }
    writeln!(out, "\nSingle node = the whole model on one node thread with the same 2-lane pool every engine node installs. 4-node H-TEP = Scenario 1's cluster (TB5-shaped loopback links, speculation on), loading its weights from the file. Fidelity and kernels are deterministic; throughput shares the host with its background load.").unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// Section 5 — V9.0: real checkpoints, K-quants, Accelerate MoE offload
// ---------------------------------------------------------------------------

const V9_MODELS: [&str; 5] = [
    "SmolLM2-135M-Instruct-Q8_0.gguf",
    "SmolLM2-135M-Instruct-Q4_K_M.gguf",
    "qwen2.5-0.5b-instruct-q4_k_m.gguf",
    "Qwen3-0.6B-Q4_K_M.gguf",
    "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
];

const V9_ENGLISH: &str = "The quick brown fox jumps over the lazy dog. Paris is the capital of France, and Rome is the capital of Italy. Water boils at one hundred degrees Celsius at sea level, and it freezes at zero degrees.";

fn ppl_of(cfg: &ModelConfig, ids: &[u32]) -> f64 {
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

/// Greedy single-node decode on one 2-lane node thread: (tokens, tok/s).
fn v9_single(cfg: &ModelConfig, prompt: &[u32], n: usize, stops: &[u32]) -> (Vec<u32>, f64) {
    use tesseract_fabric::kv_fabric::{BatchMeta, LocalKvCache};
    use tesseract_fabric::model::{NoObserver, ShardModel};
    tesseract_fabric::tensor::pool::install(TesseractOptions::default().node_lanes as usize);
    let head = HeadWeights::generate(cfg);
    let model = ShardModel::generate(cfg, 0..cfg.n_layers, None);
    let mut kv = LocalKvCache::new(cfg.n_heads, cfg.n_kv_heads, cfg.d_head).with_exact(cfg.exact);
    let mut toks = prompt.to_vec();
    let m = toks.len();
    let meta = BatchMeta::causal_prefill(0, m - 1, 1);
    let mut x = head.embed_tokens(&toks[..m - 1]);
    model.forward(&mut x, &meta, &mut kv, &mut NoObserver);
    kv.commit(&meta.keys, &|_| false);
    let t = Instant::now();
    for _ in 0..n {
        let pos = toks.len() - 1;
        let meta = BatchMeta::causal_prefill(pos, 1, 2 + pos as u64);
        let mut x = head.embed_tokens(&toks[pos..]);
        model.forward(&mut x, &meta, &mut kv, &mut NoObserver);
        kv.commit(&meta.keys, &|_| false);
        let nt = argmax(head.logits(&x, cfg.rms_eps).row(0)) as u32;
        toks.push(nt);
        if stops.contains(&nt) {
            break;
        }
    }
    let rate = (toks.len() - m) as f64 / t.elapsed().as_secs_f64();
    tesseract_fabric::tensor::pool::install(1);
    (toks[m..].to_vec(), rate)
}

/// 4 loopback workers (unshaped links: the host is the network).
fn v9_cluster(cfg: &ModelConfig, engine: EngineKind, prompt: &[u32], n: u32) -> GenerateResult {
    let ids: Vec<u16> = (0..4).collect();
    let ws = spawn_local_workers(&ids);
    let driver = Driver::bind("127.0.0.1:0").expect("driver");
    let mut links = Vec::new();
    for a in 0..4u16 {
        for b in a + 1..4 {
            links.push((a, b, vec![LinkProfile::unshaped()]));
        }
    }
    let (stages, head): (Vec<(u16, usize, usize)>, u16) = match engine {
        EngineKind::TesseractHtep => (ids.iter().map(|&i| (i, 0, cfg.n_layers)).collect(), 0),
        _ => (
            (0..4usize)
                .map(|i| (i as u16, cfg.n_layers * i / 4, cfg.n_layers * (i + 1) / 4))
                .collect(),
            3,
        ),
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
        tess: TesseractOptions {
            codec: 0,
            htep_reduce_f32: true,
            draft_lookup: true,
            ..Default::default()
        },
        driver_addr: driver.addr(),
        service_addrs: ws.iter().flat_map(|w| w.service_addrs()).collect(),
    };
    driver.assign(&a);
    let r = driver.generate(
        head,
        &GenerateRequest {
            prompt: prompt.to_vec(),
            max_new: n,
            seed: 1,
            temperature: 0.0,
            record_logits: false,
        },
    );
    driver.reset(&a);
    driver.shutdown(&a);
    r
}

fn production_v9(cfg: &ModelConfig, reference: &ReferenceModel, out: &mut String) {
    use tesseract_fabric::loader::sharded_loader::{
        config_from_file, export_gguf_mix, tokenizer_for, QuantMix,
    };
    use tesseract_fabric::loader::GgufFile;
    use tesseract_fabric::tensor::accel::{self, Backend};
    use tesseract_fabric::tensor::quant::{best_kernel, gemm_quant_with, QuantView, WeightFormat};
    use tesseract_fabric::tokenizer::{stop_tokens, ChatFormat};
    writeln!(
        out,
        "\n## Section 5 — V9.0: real third-party checkpoints, K-quants, Accelerate MoE offload\n"
    )
    .unwrap();
    // --- kernels
    writeln!(out, "#### Fused quantized GEMM by format (2048 × 2048 weights, one node thread, kernel {}; min of 30)\n", best_kernel().name()).unwrap();
    writeln!(out, "| Format | bits / weight | 1 row: µs (GMAC/s) | 4-row tree: µs (GMAC/s) |\n|---|---:|---:|---:|").unwrap();
    let mut rng = Rng::new(1);
    let w = Tensor2D::randn(2048, 2048, 0.05, &mut rng);
    for fmt in [
        WeightFormat::Q8_0,
        WeightFormat::Q6_K,
        WeightFormat::Q5_0,
        WeightFormat::Q4_K,
        WeightFormat::Q4_0,
        WeightFormat::F16,
    ] {
        let v = QuantView::from_f32(&w, fmt);
        let mut cells = Vec::new();
        for m in [1usize, 4] {
            let x = Tensor2D::randn(m, 2048, 1.0, &mut rng);
            let mut y = Tensor2D::zeros(m, 2048);
            let mut best = f64::MAX;
            for _ in 0..30 {
                let t = Instant::now();
                gemm_quant_with(&x, &v, &mut y, best_kernel());
                best = best.min(t.elapsed().as_secs_f64());
            }
            cells.push(format!(
                "{:.0} ({:.1})",
                best * 1e6,
                (m * 2048 * 2048) as f64 / best / 1e9
            ));
        }
        writeln!(
            out,
            "| {} | {:.2} | {} | {} |",
            fmt.name(),
            fmt.bits_per_weight(),
            cells[0],
            cells[1]
        )
        .unwrap();
    }
    // --- the benchmark model as Q4_K_M
    let dir = std::env::temp_dir().join(format!("tesseract-bench-v9-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("bench-q4_k_m.gguf").to_string_lossy().into_owned();
    let size = export_gguf_mix(cfg, QuantMix::Q4KM, None, &path).expect("export");
    let qcfg = config_from_file(&path, None).unwrap();
    let seq: Vec<u32> = (0..64).map(|i| (i * 977 % cfg.vocab) as u32).collect();
    let (l0, l1) = (
        reference.forward_sequence(&seq),
        ReferenceModel::new(&qcfg).forward_sequence(&seq),
    );
    let cos = (0..seq.len())
        .map(|i| logit_fidelity(l1.row(i), l0.row(i)))
        .sum::<f64>()
        / seq.len() as f64;
    let top1 = (0..seq.len())
        .filter(|&i| argmax(l1.row(i)) == argmax(l0.row(i)))
        .count();
    writeln!(out, "\nThe Scenario 1–3 model exported with llama.cpp's Q4_K_M recipe (Q4_K; Q6_K LM head and Q6_K `attn_v` / `ffn_down` on the \"more bits\" layers; Q5_0 / Q8_0 where a row is not a multiple of 256): {:.1} MB ({:.2}× smaller than FP32), teacher-forced cosine {cos:.5} and top-1 {:.1}% vs the in-memory FP32 model.", size as f64 / 1e6, cfg.total_params() as f64 * 4.0 / size as f64, 100.0 * top1 as f64 / seq.len() as f64).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    // --- Accelerate vs portable (single node)
    writeln!(out, "\n#### Accelerate (AMX) vs portable kernels, single node (prefill 128 tokens / greedy decode 48 tokens)\n").unwrap();
    writeln!(out, "| Model | portable: prefill tok/s / decode tok/s | Accelerate: prefill tok/s / decode tok/s |\n|---|---:|---:|").unwrap();
    let models_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models");
    let llama = models_dir.join(V9_MODELS[4]).to_string_lossy().into_owned();
    let mut acc_rows: Vec<(String, ModelConfig)> = vec![(
        "benchmark MoE (FP32, 8 experts: expert stripes on AMX)".into(),
        cfg.clone(),
    )];
    if std::path::Path::new(&llama).exists() {
        acc_rows.push((
            "Llama-3.2-1B Q4_K_M (dense; quantized tiles on AMX only from 16 rows)".into(),
            config_from_file(&llama, None).unwrap(),
        ));
    }
    for (name, c) in &acc_rows {
        let prompt: Vec<u32> = (0..16).map(|i| (i * 31 % c.vocab) as u32).collect();
        let mut cells = Vec::new();
        for b in [Backend::Portable, Backend::Accelerate] {
            if b == Backend::Accelerate && !accel::ACCELERATE_AVAILABLE {
                cells.push("n/a".to_string());
                continue;
            }
            accel::set_backend(b);
            let (pf, dec, _) = single_node(c, &prompt, 128, 48);
            cells.push(format!("{pf:.0} / {dec:.1}"));
        }
        accel::set_backend(Backend::Portable);
        writeln!(out, "| {name} | {} | {} |", cells[0], cells[1]).unwrap();
    }
    // --- real checkpoints
    writeln!(out, "\n#### Real third-party GGUF checkpoints (`models/`, downloaded by `cargo test --test production_v9`)\n").unwrap();
    let present: Vec<&str> = V9_MODELS
        .iter()
        .copied()
        .filter(|f| models_dir.join(f).exists())
        .collect();
    if present.is_empty() {
        writeln!(out, "_No checkpoints in `models/`: run `cargo test --release --test production_v9` once to download them._").unwrap();
        return;
    }
    writeln!(out, "Prompt (each model's own chat template): \"Explain in one sentence what distributed AI inference is.\" Greedy, ≤ 48 tokens, stop at the template's end-of-turn token. Perplexity: a fixed 3-sentence English paragraph, teacher forced. Clusters: 4 loopback workers with unshaped links (this host's TCP stack is the network), lossless f32 activations / f32 H-TEP reduction, prompt-lookup speculation.\n").unwrap();
    writeln!(out, "| Checkpoint | arch · layers · d · vocab | weight types | file (MB) | `mmap` + all views (ms) | perplexity | single node tok/s | 4-stage pipeline: tok/s · tokens/step · identical to single node | 4-rank H-TEP: tok/s · tokens identical before first divergence |\n|---|---|---|---:|---:|---:|---:|---:|---:|").unwrap();
    let mut texts = Vec::new();
    for f in present {
        let path = models_dir.join(f).to_string_lossy().into_owned();
        eprintln!("  {f}");
        let g = GgufFile::open(&path).unwrap();
        let mut mix = std::collections::BTreeMap::new();
        for t in &g.tensors {
            if t.dims.len() >= 2 {
                *mix.entry(t.format().map(|f| f.name()).unwrap_or("?"))
                    .or_insert(0u64) += t.byte_size().unwrap_or(0);
            }
        }
        let total: u64 = mix.values().sum();
        let mix_s = mix
            .iter()
            .map(|(k, v)| format!("{k} {:.0}%", 100.0 * *v as f64 / total as f64))
            .collect::<Vec<_>>()
            .join(", ");
        let t = Instant::now();
        let c = config_from_file(&path, None).unwrap();
        let shard = tesseract_fabric::model::ShardModel::generate(&c, 0..c.n_layers, None);
        let head = HeadWeights::generate(&c);
        let load_ms = t.elapsed().as_secs_f64() * 1e3;
        drop((shard, head));
        let tok = tokenizer_for(&c);
        let fmt = ChatFormat::detect(&tok);
        let ids = fmt.encode(
            &tok,
            &[(
                "user".into(),
                "Explain in one sentence what distributed AI inference is.".into(),
            )],
        );
        let stops = stop_tokens(&tok);
        let ppl = ppl_of(&c, &tok.encode(V9_ENGLISH));
        let (single, rate) = v9_single(&c, &ids, 48, &stops);
        let n = single.len() as u32;
        let p = v9_cluster(&c, EngineKind::Tesseract, &ids, n);
        let h = v9_cluster(&c, EngineKind::TesseractHtep, &ids, n);
        let agree = h
            .tokens
            .iter()
            .zip(&single)
            .take_while(|(a, b)| a == b)
            .count();
        let tps = |r: &GenerateResult| r.tokens.len() as f64 / (r.decode_ns as f64 * 1e-9);
        writeln!(
            out,
            "| {f} | {} · {} · {} · {} | {mix_s} | {:.0} | {load_ms:.1} | {ppl:.1} | {rate:.1} | {:.1} · {:.2} · {} | {:.1} · {agree} / {} |",
            g.architecture().unwrap_or("?"),
            c.n_layers,
            c.d_model,
            c.vocab,
            g.map.len() as f64 / 1e6,
            tps(&p),
            p.tokens.len() as f64 / p.per_epoch.len().max(1) as f64,
            if p.tokens == single { "yes" } else { "**no**" },
            tps(&h),
            single.len()
        )
        .unwrap();
        let kept: Vec<u32> = single
            .iter()
            .copied()
            .filter(|t| !stops.contains(t))
            .collect();
        texts.push((f, fmt.name(), tok.decode(&kept)));
    }
    writeln!(out, "\nVerbatim single-node replies (greedy):\n").unwrap();
    for (f, fmt, t) in texts {
        writeln!(out, "* **{f}** ({fmt}): {t:?}").unwrap();
    }
}

// ---------------------------------------------------------------------------
// Host load gate
// ---------------------------------------------------------------------------

/// The 1-minute load average.
fn load_avg() -> f64 {
    extern "C" {
        fn getloadavg(loadavg: *mut f64, nelem: i32) -> i32;
    }
    let mut l = [0.0f64; 3];
    // SAFETY: libc writes at most 3 doubles into `l`.
    let n = unsafe { getloadavg(l.as_mut_ptr(), 3) };
    if n >= 1 {
        l[0]
    } else {
        0.0
    }
}

/// Wait (up to 3 minutes) for the host's 1-minute load average to fall
/// below a quarter of its cores before a timed section, and record it.
/// Up to V10.0, emulated slow devices stretched their measured *wall* time,
/// so host contention was amplified up to 23× on exactly the nodes
/// (drafters on emulated phones) that some configurations use and others do
/// not — the V9/V10.0 run-1 Scenario 2 inversions. Since V10.1 the shaper
/// stretches critical-path CPU time instead; the gate remains so that the
/// un-stretched part of every timing is measured on a quiet host too.
fn quiet_host(out: &mut String, what: &str) {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8) as f64;
    let limit = (cores / 4.0).max(2.0);
    let t = Instant::now();
    let mut l = load_avg();
    while l > limit && t.elapsed().as_secs() < 180 {
        std::thread::sleep(std::time::Duration::from_secs(5));
        l = load_avg();
    }
    let waited = t.elapsed().as_secs();
    eprintln!("  host load {l:.2} (limit {limit:.1}, waited {waited} s)");
    writeln!(out, "\n_{what}: host 1-minute load average {l:.2} at start (gate: ≤ {limit:.1}, waited {waited} s{})._", if l > limit { "; **still above the gate — timings below are load-affected**" } else { "" }).unwrap();
}

// ---------------------------------------------------------------------------
// Section 6 — V10.0
// ---------------------------------------------------------------------------

/// Draft-model pair for Section 6 (same tokenizer).
const V10_TARGET: &str = "SmolLM2-1.7B-Instruct-Q4_K_M.gguf";
const V10_DRAFT: &str = "SmolLM2-135M-Instruct-Q8_0.gguf";

/// A loopback cluster of `world` H-TEP ranks (rank 0 the head).
fn v10_cluster(
    cfg: &ModelConfig,
    world: u16,
    tess: TesseractOptions,
) -> (
    Driver,
    Assignment,
    Vec<tesseract_fabric::cluster::LocalWorker>,
) {
    let ids: Vec<u16> = (0..world).collect();
    let ws = spawn_local_workers(&ids);
    let driver = Driver::bind("127.0.0.1:0").expect("driver");
    let mut links = Vec::new();
    for a in 0..world {
        for b in a + 1..world {
            links.push((a, b, vec![LinkProfile::unshaped()]));
        }
    }
    let a = Assignment {
        engine: EngineKind::TesseractHtep,
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
        stages: ids.iter().map(|&i| (i, 0, cfg.n_layers)).collect(),
        head: 0,
        hot_capacity: 0,
        inject_context: 0,
        tess,
        driver_addr: driver.addr(),
        service_addrs: ws.iter().flat_map(|w| w.service_addrs()).collect(),
    };
    driver.assign(&a);
    (driver, a, ws)
}

fn v10_opts(sessions: u8, draft: &str, budget: u16) -> TesseractOptions {
    TesseractOptions {
        codec: 0,
        htep_reduce_f32: true,
        draft_lookup: true,
        sessions,
        draft_model: draft.into(),
        budget,
        ..Default::default()
    }
}

/// Greedy single-node decoding, one token per step: tokens and their logits.
fn v10_single(cfg: &ModelConfig, prompt: &[u32], n: usize) -> (Vec<u32>, Vec<Vec<f32>>, f64) {
    use tesseract_fabric::kv_fabric::{BatchMeta, LocalKvCache};
    use tesseract_fabric::model::{NoObserver, ShardModel};
    tesseract_fabric::tensor::pool::install(TesseractOptions::default().node_lanes as usize);
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
    let t = Instant::now();
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
    let rate = n as f64 / t.elapsed().as_secs_f64();
    (toks[m..].to_vec(), logits, rate)
}

fn production_v10(out: &mut String) {
    use tesseract_fabric::cluster::batch::{BatchClient, ClientEvent};
    use tesseract_fabric::loader::sharded_loader::{config_from_file, tokenizer_for};
    use tesseract_fabric::tensor::quant::{
        best_kernel, gemm_fixed_with, gemm_quant_with, QuantView, WeightFormat,
    };
    use tesseract_fabric::tokenizer::ChatFormat;
    writeln!(out, "\n## Section 6 — V10.0: fast K-quant kernels, exact distributed inference, draft-model speculation, continuous batching\n").unwrap();
    // --- 6.1 kernels
    writeln!(out, "#### 6.1 Quantized GEMM by format and batch rows (2048 × 2048 weights, one thread, kernel {}; min of 20) — GMAC/s, float / exact (fixed-point) accumulation\n", best_kernel().name()).unwrap();
    writeln!(out, "| Format | V9.0, 1 row (README) | 1 row | 2 rows | 4 rows | 8 rows |\n|---|---:|---:|---:|---:|---:|").unwrap();
    let mut rng = Rng::new(1);
    let w = Tensor2D::randn(2048, 2048, 0.05, &mut rng);
    for (fmt, v9) in [
        (WeightFormat::Q8_0, 14.7),
        (WeightFormat::Q4_0, 15.2),
        (WeightFormat::Q5_0, 13.0),
        (WeightFormat::Q4_K, 10.5),
        (WeightFormat::Q6_K, 12.4),
    ] {
        let v = QuantView::from_f32(&w, fmt);
        let mut cells = Vec::new();
        for m in [1usize, 2, 4, 8] {
            let x = Tensor2D::randn(m, 2048, 1.0, &mut rng);
            let mut y = Tensor2D::zeros(m, 2048);
            let (mut bf, mut bx) = (f64::MAX, f64::MAX);
            for _ in 0..20 {
                let t = Instant::now();
                gemm_quant_with(&x, &v, &mut y, best_kernel());
                bf = bf.min(t.elapsed().as_secs_f64());
                let t = Instant::now();
                std::hint::black_box(gemm_fixed_with(&x, &v, best_kernel()));
                bx = bx.min(t.elapsed().as_secs_f64());
            }
            let g = |t: f64| (m * 2048 * 2048) as f64 / t / 1e9;
            cells.push(format!("{:.1} / {:.1}", g(bf), g(bx)));
        }
        writeln!(out, "| {} | {v9:.1} | {} |", fmt.name(), cells.join(" | ")).unwrap();
    }
    let models_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models");
    let present: Vec<&str> = V9_MODELS
        .iter()
        .copied()
        .filter(|f| models_dir.join(f).exists())
        .collect();
    if present.is_empty() {
        writeln!(out, "\n_No checkpoints in `models/`: run `cargo test --release --test production_v9` once to download them._").unwrap();
        return;
    }
    // --- 6.2 exactness
    writeln!(
        out,
        "\n#### 6.2 Exact distributed inference: 4-rank H-TEP vs a single node, bit for bit\n"
    )
    .unwrap();
    writeln!(out, "Greedy, 40 tokens, each model's chat template, prompt \"List the capitals of France, Italy and Spain, one per line, then explain why Rome is famous.\" The single node decodes one token per step; H-TEP runs the session scheduler (exact all-reduce of fixed-point partials, exact attention, context-copy speculation verified in multi-row chains).\n").unwrap();
    writeln!(out, "| Checkpoint | attention split across ranks | single node tok/s | H-TEP×4 tok/s | tokens identical | logit rows bit-identical | tokens / step |\n|---|---|---:|---:|---:|---:|---:|").unwrap();
    for f in &present {
        let path = models_dir.join(f).to_string_lossy().into_owned();
        eprintln!("  6.2 {f}");
        let c = config_from_file(&path, None).unwrap();
        let tok = tokenizer_for(&c);
        let ids = ChatFormat::detect(&tok).encode(&tok, &[("user".into(), "List the capitals of France, Italy and Spain, one per line, then explain why Rome is famous.".into())]);
        let (want, wl, rate) = v10_single(&c, &ids, 40);
        let (d, a, _w) = v10_cluster(&c, 4, v10_opts(1, "", 0));
        let r = d.generate(
            0,
            &GenerateRequest {
                prompt: ids.clone(),
                max_new: 40,
                seed: 1,
                temperature: 0.0,
                record_logits: true,
            },
        );
        d.reset(&a);
        d.shutdown(&a);
        let same = r
            .tokens
            .iter()
            .zip(&want)
            .take_while(|(x, y)| x == y)
            .count();
        let lsame = r.logits.iter().zip(&wl).take_while(|(x, y)| x == y).count();
        let split = c.n_heads.is_multiple_of(4)
            && c.n_kv_heads.is_multiple_of(4)
            && (c.n_heads / 4 * c.d_head).is_multiple_of(256);
        writeln!(
            out,
            "| {f} | {} | {rate:.1} | {:.1} | {same} / {} | {lsame} / {} | {:.2} |",
            if split {
                "yes (heads)"
            } else {
                "no (replicated)"
            },
            r.tokens.len() as f64 / (r.decode_ns as f64 * 1e-9),
            want.len(),
            want.len(),
            r.tokens.len() as f64 / r.per_epoch.len().max(1) as f64
        )
        .unwrap();
    }
    // --- 6.3 draft model
    let (tp, dp) = (models_dir.join(V10_TARGET), models_dir.join(V10_DRAFT));
    if tp.exists() && dp.exists() {
        let c = config_from_file(&tp.to_string_lossy(), None).unwrap();
        let tok = tokenizer_for(&c);
        let ids = ChatFormat::detect(&tok).encode(
            &tok,
            &[(
                "user".into(),
                "Write a short paragraph about the history of the Roman Empire.".into(),
            )],
        );
        let n = 64;
        eprintln!("  6.3 draft model");
        let (want, _, rate1) = v10_single(&c, &ids, n);
        writeln!(out, "\n#### 6.3 Draft-model speculation: SmolLM2-135M (Q8_0) drafting for SmolLM2-1.7B (Q4_K_M), greedy prose, 64 tokens\n").unwrap();
        writeln!(out, "Single node one token per step: {rate1:.1} tok/s. Every configuration below reproduces it token for token.\n").unwrap();
        writeln!(out, "| Ranks | Drafting | Chain length | tokens / step | draft tokens accepted | tok/s | vs single node | identical |\n|---:|---|---|---:|---:|---:|---:|---|").unwrap();
        let dps = dp.to_string_lossy().into_owned();
        for (world, draft, budget, label) in [
            (1u16, "", 0u16, "context copy only"),
            (1, dps.as_str(), 5, "draft model"),
            (1, dps.as_str(), 0, "draft model"),
            (4, "", 0, "context copy only"),
            (4, dps.as_str(), 5, "draft model"),
            (4, dps.as_str(), 0, "draft model"),
        ] {
            let (d, a, _w) = v10_cluster(&c, world, v10_opts(1, draft, budget));
            let r = d.generate(
                0,
                &GenerateRequest {
                    prompt: ids.clone(),
                    max_new: n as u32,
                    seed: 1,
                    temperature: 0.0,
                    record_logits: false,
                },
            );
            d.reset(&a);
            d.shutdown(&a);
            let drafted: u64 = r.budgets.iter().map(|&b| b.saturating_sub(1) as u64).sum();
            let accepted: u64 = r
                .per_epoch
                .iter()
                .map(|&e| e.saturating_sub(1) as u64)
                .sum();
            let tps = r.tokens.len() as f64 / (r.decode_ns as f64 * 1e-9);
            writeln!(
                out,
                "| {world} | {label} | {} | {:.2} | {} | {tps:.1} | {:.2}× | {} |",
                if budget > 1 {
                    format!("fixed {}", budget - 1)
                } else {
                    "planned (cost model)".into()
                },
                r.tokens.len() as f64 / r.per_epoch.len().max(1) as f64,
                if drafted > 0 {
                    format!(
                        "{accepted} / {drafted} ({:.0}%)",
                        100.0 * accepted as f64 / drafted as f64
                    )
                } else {
                    "—".into()
                },
                tps / rate1,
                if r.tokens == want { "yes" } else { "**no**" }
            )
            .unwrap();
        }
    }
    // --- 6.4 continuous batching
    let qp = models_dir.join(V9_MODELS[2]);
    if qp.exists() {
        eprintln!("  6.4 continuous batching");
        let c = config_from_file(&qp.to_string_lossy(), None).unwrap();
        let tok = tokenizer_for(&c);
        let fmt = ChatFormat::detect(&tok);
        let qs = [
            "Name three rivers in Europe.",
            "What is the boiling point of water at sea level?",
            "Scrivi una frase sul mare.",
            "List the first five prime numbers.",
            "Why is the sky blue?",
            "Describe a cat in one sentence.",
            "What is the capital of Japan?",
            "Give me a synonym for happy.",
        ];
        let ids: Vec<Vec<u32>> = qs
            .iter()
            .map(|q| fmt.encode(&tok, &[("user".into(), q.to_string())]))
            .collect();
        let n = 32u32;
        let want: Vec<Vec<u32>> = ids
            .iter()
            .map(|p| v10_single(&c, p, n as usize).0)
            .collect();
        writeln!(out, "\n#### 6.4 Continuous batching: {} on 2 H-TEP ranks, 8 session slots, greedy, {n} tokens per request\n", V9_MODELS[2]).unwrap();
        writeln!(out, "N requests submitted at once; the head interleaves them in shared forward passes (chunked prefill + every session's draft chain in one batch, one all-reduce per layer for all of them). Aggregate = all tokens / wall time from submission to the last result.\n").unwrap();
        writeln!(out, "| Concurrent requests | aggregate tok/s | per-request tok/s (mean) | vs 1 request | outputs identical to single node |\n|---:|---:|---:|---:|---|").unwrap();
        let (d, a, _w) = v10_cluster(&c, 2, v10_opts(8, "", 0));
        let client = BatchClient::new(d, 0);
        // warm up the step-cost model
        let _ = client.generate(&GenerateRequest {
            prompt: ids[0].clone(),
            max_new: 8,
            seed: 1,
            temperature: 0.0,
            record_logits: false,
        });
        let mut base = 0.0;
        for k in [1usize, 2, 4, 8] {
            let t = Instant::now();
            let rx: Vec<_> = (0..k)
                .map(|i| {
                    client.submit(
                        GenerateRequest {
                            prompt: ids[i].clone(),
                            max_new: n,
                            seed: 1,
                            temperature: 0.0,
                            record_logits: false,
                        },
                        GenerateExt::default(),
                    )
                })
                .collect();
            let res: Vec<GenerateResult> = rx
                .into_iter()
                .map(|r| loop {
                    if let Ok(ClientEvent::Done(g)) = r.recv() {
                        break g;
                    }
                })
                .collect();
            let wall = t.elapsed().as_secs_f64();
            let total: usize = res.iter().map(|r| r.tokens.len()).sum();
            let agg = total as f64 / wall;
            let per = res
                .iter()
                .map(|r| r.tokens.len() as f64 / (r.decode_ns as f64 * 1e-9))
                .sum::<f64>()
                / k as f64;
            if k == 1 {
                base = agg;
            }
            let same = res
                .iter()
                .zip(&want)
                .filter(|(r, w)| &r.tokens == *w)
                .count();
            writeln!(
                out,
                "| {k} | {agg:.1} | {per:.1} | {:.2}× | {same} / {k} |",
                agg / base
            )
            .unwrap();
        }
        client.exec(move |d| {
            d.reset(&a);
            d.shutdown(&a);
        });
    }
}
