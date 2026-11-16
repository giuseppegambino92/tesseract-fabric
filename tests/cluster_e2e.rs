//! Multi-node execution over real loopback TCP sockets: every test spawns
//! worker threads with their own endpoints and exchanges real TesseractWire
//! frames through link shapers.

use tesseract_fabric::cluster::*;
use tesseract_fabric::model::{argmax, logit_fidelity, ModelConfig, ReferenceModel};
use tesseract_fabric::topology::profiles::{links_between, preset};
use tesseract_fabric::topology::{
    footprint, HeteroClusterSolver, LinkSpec, Platform, SolverConfig,
};
use tesseract_fabric::wire::multipath::LinkProfile;
use tesseract_fabric::wire::protocol::FrameType;
use tesseract_fabric::wire::transport::Endpoint;

struct Run {
    res: GenerateResult,
    stats: Vec<NodeStats>,
}

#[allow(clippy::too_many_arguments)]
fn run(
    cfg: &ModelConfig,
    n: u16,
    engine: EngineKind,
    stages: Vec<(u16, usize, usize)>,
    head: u16,
    link: LinkProfile,
    tess: TesseractOptions,
    inject: u32,
    req: GenerateRequest,
) -> Run {
    let ids: Vec<u16> = (0..n).collect();
    let ws = spawn_local_workers(&ids);
    let driver = Driver::bind("127.0.0.1:0").unwrap();
    let mut links = Vec::new();
    for a in 0..n {
        for b in a + 1..n {
            links.push((a, b, vec![link]));
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
        hot_capacity: cfg
            .moe
            .as_ref()
            .map(|m| (m.n_experts - 1) as u16)
            .unwrap_or(0),
        inject_context: inject,
        tess,
        driver_addr: driver.addr(),
        service_addrs: ws.iter().flat_map(|w| w.service_addrs()).collect(),
    };
    driver.assign(&a);
    let res = driver.generate(head, &req);
    // let in-flight frames on shaped links (e.g. the final commit) land before
    // the control-plane stats request overtakes them
    std::thread::sleep(std::time::Duration::from_millis(50));
    let mut all = ids.clone();
    all.extend(ids.iter().map(|i| KV_SERVICE_BASE + i));
    let stats = driver.stats(&all);
    driver.reset(&a);
    driver.shutdown(&a);
    Run { res, stats }
}

fn fidelity(reference: &ReferenceModel, prompt: &[u32], r: &GenerateResult) -> (f64, f64) {
    let mut seq = prompt.to_vec();
    seq.extend(&r.tokens);
    let rl = reference.forward_sequence(&seq);
    let mut cos = 0.0;
    let mut agree = 0;
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

fn req(prompt: &[u32], max_new: u32, temperature: f32) -> GenerateRequest {
    GenerateRequest {
        prompt: prompt.to_vec(),
        max_new,
        seed: 5,
        temperature,
        record_logits: true,
    }
}

fn prompt() -> Vec<u32> {
    (0..10).map(|i| (i * 53 % 256) as u32).collect()
}

#[test]
fn exo_ring_pipeline_over_sockets_matches_fp32_reference() {
    let cfg = ModelConfig::tiny();
    let reference = ReferenceModel::new(&cfg);
    let p = prompt();
    let r = run(
        &cfg,
        3,
        EngineKind::ExoRing,
        vec![(0, 0, 2), (1, 2, 4), (2, 4, 6)],
        2,
        LinkProfile::ethernet_2_5g(),
        TesseractOptions::default(),
        0,
        req(&p, 24, 1.0),
    );
    assert_eq!(r.res.tokens.len(), 24);
    let (cos, top1) = fidelity(&reference, &p, &r.res);
    assert!(cos > 0.9995 && top1 >= 0.95, "fidelity {cos} top1 {top1}");
    // FP16 activations crossed two hops per token
    assert!(r.stats.iter().map(|s| s.act_bytes).sum::<u64>() > 24 * 2 * cfg.d_model as u64 * 2);
}

#[test]
fn tensor_parallel_two_allreduces_per_layer() {
    let cfg = ModelConfig::tiny();
    let reference = ReferenceModel::new(&cfg);
    let p = prompt();
    let r = run(
        &cfg,
        2,
        EngineKind::TensorParallel,
        vec![(0, 0, 0), (1, 0, 0)],
        0,
        LinkProfile::thunderbolt5(),
        TesseractOptions::default(),
        0,
        req(&p, 16, 1.0),
    );
    assert_eq!(r.res.tokens.len(), 16);
    let (cos, top1) = fidelity(&reference, &p, &r.res);
    assert!(cos > 0.9995 && top1 >= 0.9, "fidelity {cos} top1 {top1}");
    // rank 1 sends exactly one partial per barrier: 2 per layer per forward pass
    let rank1_frames = r.stats.iter().find(|s| s.node == 1).unwrap().frames_sent;
    assert!(
        rank1_frames >= (2 * cfg.n_layers * 16) as u64,
        "rank 1 frames {rank1_frames}"
    );
}

#[test]
fn tesseract_speculative_braid_pipeline_generates_with_high_fidelity() {
    let cfg = ModelConfig::tiny();
    let reference = ReferenceModel::new(&cfg);
    let p = prompt();
    let tess = TesseractOptions {
        budget: 6,
        braid_strands: 2,
        ..Default::default()
    };
    let r = run(
        &cfg,
        3,
        EngineKind::Tesseract,
        vec![(0, 0, 2), (1, 2, 4), (2, 4, 6)],
        2,
        LinkProfile::ethernet_2_5g(),
        tess,
        0,
        req(&p, 40, 1.0),
    );
    assert_eq!(r.res.tokens.len(), 40);
    assert_eq!(r.res.logits.len(), 40);
    let (cos, top1) = fidelity(&reference, &p, &r.res);
    // default V2 codec: 3-bit trunk-residual activations
    assert!(cos > 0.995 && top1 >= 0.85, "fidelity {cos} top1 {top1}");
    // every epoch emits 1..=depth+1 tokens and all stages applied every commit
    assert!(r.res.per_epoch.iter().all(|&k| (1..=7).contains(&k)));
    let commits: Vec<u64> = r
        .stats
        .iter()
        .filter(|s| s.node < 3)
        .map(|s| s.commits)
        .collect();
    assert!(
        commits.iter().all(|&c| c == commits[0]),
        "commit counts diverged: {commits:?}"
    );
    assert_eq!(commits[0] as usize, r.res.per_epoch.len() + 1); // + prompt prefill
                                                                // lookahead pre-staging was exercised across shards
    let prewakes: u64 = r.stats.iter().map(|s| s.prewakes_sent).sum();
    assert!(prewakes > 0);
}

#[test]
fn tesseract_greedy_is_deterministic_across_runs() {
    let cfg = ModelConfig::tiny();
    let p = prompt();
    let tess = TesseractOptions {
        budget: 4,
        braid_strands: 2,
        ..Default::default()
    };
    let a = run(
        &cfg,
        2,
        EngineKind::Tesseract,
        vec![(0, 0, 3), (1, 3, 6)],
        1,
        LinkProfile::thunderbolt5(),
        tess.clone(),
        0,
        req(&p, 24, 0.0),
    );
    let b = run(
        &cfg,
        2,
        EngineKind::Tesseract,
        vec![(0, 0, 3), (1, 3, 6)],
        1,
        LinkProfile::wifi6e(),
        tess,
        0,
        req(&p, 24, 0.0),
    );
    assert_eq!(
        a.res.tokens, b.res.tokens,
        "greedy output must not depend on link timing"
    );
}

#[test]
fn disaggregated_kv_fabric_matches_layer_coupled_kv() {
    let mut cfg = ModelConfig::tiny();
    cfg.moe = None;
    cfg.n_kv_heads = 2;
    let p = prompt();
    let stages = vec![(0, 0, 3), (1, 3, 6)];
    let base = TesseractOptions {
        budget: 1,
        braid_strands: 1,
        draft_ahead: 0,
        lookahead: false,
        page_tokens: 128,
        // tier 0 = 4 pages (512 tokens) stays with the computing stage; the
        // rest of the 1500-token context is striped, so both tiers are used
        tier0_pages: 4,
        ..Default::default()
    };
    // 1500 tokens with 128-token pages and a 4-page tier 0 crosses the
    // delegation threshold (2 × 4 × 128 = 1024): mirrored prefix pages are
    // scanned by their mirrors; 700 tokens stays below it. Every variant must
    // reproduce layer-coupled KV (each page scanned exactly once). 1000
    // injected + 40 generated tokens crosses the threshold mid-generation:
    // the prefix pages migrate (move, not copy) to their mirrors.
    for (inject, mirror, n) in [
        (1500u32, true, 12u32),
        (1500, false, 12),
        (700, true, 12),
        (1000, true, 40),
    ] {
        let coupled = run(
            &cfg,
            3,
            EngineKind::Tesseract,
            stages.clone(),
            1,
            LinkProfile::thunderbolt5(),
            base.clone(),
            inject,
            req(&p, n, 0.0),
        );
        let disagg = run(
            &cfg,
            3,
            EngineKind::Tesseract,
            stages.clone(),
            1,
            LinkProfile::thunderbolt5(),
            TesseractOptions {
                kv_mode: KvMode::Disaggregated,
                kv_prefix_mirror: mirror,
                ..base.clone()
            },
            inject,
            req(&p, n, 0.0),
        );
        assert_eq!(
            coupled.res.tokens, disagg.res.tokens,
            "inject {inject} mirror {mirror}"
        );
        for (a, b) in coupled.res.logits.iter().zip(&disagg.res.logits) {
            let err = a
                .iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            assert!(
                err < 2e-3,
                "logit divergence {err} (inject {inject} mirror {mirror})"
            );
        }
        // remote KV services really held pages; past the delegation
        // threshold (with mirroring) every host does, incl. the weight-less
        // node 2
        let kv_on = |n: u16| -> u64 {
            disagg
                .stats
                .iter()
                .filter(|s| s.node == KV_SERVICE_BASE + n)
                .map(|s| s.kv_bytes)
                .sum()
        };
        assert!((0..3).map(kv_on).sum::<u64>() > 0, "no remote KV pages");
        if mirror && inject >= 1024 {
            assert!(kv_on(2) > 0, "no KV pages on the weight-less node");
        }
        assert!(disagg.stats.iter().map(|s| s.attn_bytes).sum::<u64>() > 0);
    }
}

#[test]
fn remote_edge_drafter_serves_trees_over_wifi() {
    let cfg = ModelConfig::tiny();
    let reference = ReferenceModel::new(&cfg);
    let p = prompt();
    let tess = TesseractOptions {
        budget: 4,
        drafter: Some(2),
        ..Default::default()
    };
    let r = run(
        &cfg,
        3,
        EngineKind::Tesseract,
        vec![(0, 0, 3), (1, 3, 6)],
        1,
        LinkProfile::wifi6e(),
        tess,
        0,
        req(&p, 30, 1.0),
    );
    assert_eq!(r.res.tokens.len(), 30);
    assert!(r.res.draft_hits + r.res.draft_misses > 0);
    let (cos, _) = fidelity(&reference, &p, &r.res);
    assert!(cos > 0.995, "fidelity {cos}");
}

#[test]
fn multipath_survives_link_brownout_and_reroutes() {
    let a = Endpoint::bind(1, "127.0.0.1:0").unwrap();
    let b = Endpoint::bind(2, "127.0.0.1:0").unwrap();
    for (x, y) in [(&a, &b), (&b, &a)] {
        x.shared.set_addr(y.id(), y.addr());
        x.shared.set_links(
            y.id(),
            vec![LinkProfile::ethernet_2_5g(), LinkProfile::wifi6e()],
        );
    }
    let frame = |i: u8| vec![i; 300 * 1024];
    for i in 0..6 {
        a.send_payload(2, FrameType::ActivationDelta, i, 0, frame(i as u8))
            .unwrap();
        let f = b.recv();
        assert_eq!(
            f.payload,
            frame(i as u8),
            "striped frame reassembled intact"
        );
    }
    // brownout: the 2.5GbE link degrades to 15 ms / 50 Mbps
    let mut bad = LinkProfile::ethernet_2_5g();
    bad.one_way_us = 15_000.0;
    bad.bandwidth_gbps = 0.05;
    a.shared.reshape_link(2, 0, bad).unwrap();
    for i in 6..20u8 {
        a.send_payload(2, FrameType::ActivationDelta, i as u64, 0, frame(i))
            .unwrap();
        let f = b.recv_match(&|f| f.header.frame_type == FrameType::ActivationDelta);
        assert_eq!(f.payload, frame(i));
    }
    let s = a.shared.scheduler_snapshot(2).unwrap();
    assert!(
        !s.links[0].healthy || s.links[0].est_one_way_ns > 5_000_000.0,
        "brownout not detected: {:?}",
        s.links[0]
    );
    let plan = s.plan(
        tesseract_fabric::wire::transport::mono_ns() as f64,
        300 * 1024,
    );
    assert!(
        plan.iter().all(|p| p.0 == 1),
        "traffic not moved off the browned-out link: {plan:?}"
    );
    a.shared.close();
    b.shared.close();
}

#[test]
fn solver_keeps_edge_and_wifi_nodes_off_the_critical_path() {
    let presets = [
        preset("studio-m5max").unwrap(),
        preset("rtx5090").unwrap(),
        preset("macbook-pro").unwrap(),
        preset("android-snapdragon").unwrap(),
    ];
    let specs: Vec<_> = presets
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let links = presets
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(j, q)| LinkSpec {
                    peer: j as u16,
                    profiles: links_between(p, q, false),
                })
                .collect();
            p.spec(i as u16, links)
        })
        .collect();
    let cfg = ModelConfig::bench_moe();
    let plan = HeteroClusterSolver::new(specs.clone(), footprint(&cfg, 8), SolverConfig::default())
        .solve();
    let stage_nodes: Vec<u16> = plan.stages.iter().map(|s| s.node).collect();
    assert!(!stage_nodes.contains(&3), "Android in pipeline: {plan:?}");
    assert!(
        !stage_nodes.contains(&2),
        "Wi-Fi laptop in pipeline: {plan:?}"
    );
    assert_eq!(
        plan.stages.iter().map(|s| s.layers.len()).sum::<usize>(),
        cfg.n_layers
    );
    // V10.1: the drafter is the node that returns a tree soonest (compute +
    // round trip to the coordinator): the idle cores of the fast
    // non-coordinator pipeline stage, not the 23× phone on Wi-Fi
    let d = plan.drafter.expect("a drafter");
    assert_ne!(d, 3, "drafting on the phone: {plan:?}");
    assert!(
        d != plan.coordinator && stage_nodes.contains(&d),
        "{plan:?}"
    );
    assert!(specs.iter().find(|s| s.node_id == 3).unwrap().platform == Platform::MobileEdgeNpu);
    assert!(
        plan.kv_pools.iter().any(|p| p.0 == 2),
        "laptop should offer an overflow KV pool"
    );
}

#[test]
fn hrq_v5_beats_v21_beats_v2_on_the_live_pipeline_within_25_percent() {
    // bench-scale contextual model, 2-stage live pipeline, speculation off so
    // every emitted token's logits went through exactly one compressed hop
    let cfg = ModelConfig::bench_moe();
    let reference = ReferenceModel::new(&cfg);
    let p: Vec<u32> = (0..16).map(|i| (i * 131 % 2048) as u32).collect();
    let mut res = Vec::new();
    for codec in [2u8, 3, 4] {
        let tess = TesseractOptions {
            codec,
            codec_auto: false,
            holo_bits: 3,
            budget: 1,
            braid_strands: 1,
            draft_ahead: 0,
            ..Default::default()
        };
        let r = run(
            &cfg,
            2,
            EngineKind::Tesseract,
            vec![(0, 0, 8), (1, 8, 16)],
            1,
            LinkProfile::thunderbolt5(),
            tess,
            0,
            req(&p, 96, 1.0),
        );
        res.push(fidelity(&reference, &p, &r.res));
    }
    let v2 = tesseract_fabric::compression::TrunkResidualCodec::new(cfg.d_model, 3, 0).row_bytes()
        as f64
        / (cfg.d_model * 2) as f64;
    let v21 = tesseract_fabric::compression::EntropyResidualCodec::new(cfg.d_model, 0.25, 0)
        .row_bytes_cap() as f64
        / (cfg.d_model * 2) as f64;
    eprintln!("V2 3-bit: cosine {:.5}, top-1 {:.1}% ({:.1}% of FP16) | V2.1: cosine {:.5}, top-1 {:.1}% (≤{:.1}% of FP16) | V5: cosine {:.5}, top-1 {:.1}% (≤25%)", res[0].0, res[0].1 * 100.0, v2 * 100.0, res[1].0, res[1].1 * 100.0, v21 * 100.0, res[2].0, res[2].1 * 100.0);
    assert!(v2 <= 0.25 && v21 <= 0.25);
    assert!(
        res[1].0 > res[0].0,
        "V2.1 must be more faithful than V2 at the same budget"
    );
    assert!(
        res[1].0 > 0.999 && res[1].1 >= 0.90,
        "V2.1 fidelity {:?}",
        res[1]
    );
    assert!(
        res[2].0 > res[1].0 && res[2].1 >= res[1].1,
        "V5 must be more faithful than V2.1 at the same cap: {:?} vs {:?}",
        res[2],
        res[1]
    );
    assert!(res[2].0 > 0.9998, "V5 fidelity {:?}", res[2]);
}

#[test]
fn per_hop_codec_choice_ships_fp16_on_fast_links_and_codes_slow_ones() {
    use tesseract_fabric::wire::multipath::LinkKind;
    let cfg = ModelConfig::tiny();
    let p = prompt();
    let slow = LinkProfile {
        kind: LinkKind::Ethernet2_5G,
        one_way_us: 200.0,
        bandwidth_gbps: 0.002,
        jitter_us: 0.0,
    };
    let mut per_tok = Vec::new();
    for link in [LinkProfile::thunderbolt5(), slow] {
        let r = run(
            &cfg,
            2,
            EngineKind::Tesseract,
            vec![(0, 0, 3), (1, 3, 6)],
            1,
            link,
            TesseractOptions {
                budget: 1,
                braid_strands: 1,
                draft_ahead: 0,
                ..Default::default()
            },
            0,
            req(&p, 12, 0.0),
        );
        let act: u64 = r.stats.iter().map(|s| s.act_bytes).sum();
        per_tok.push(act as f64 / (p.len() - 1 + r.res.tokens.len()) as f64);
    }
    // bytes are whole frames (64-byte header, strand metadata, token ids)
    let fp16_row = (cfg.d_model * 2) as f64;
    eprintln!(
        "inter-stage frame bytes per row: TB5 {:.0}, 2 Mbps link {:.0} (FP16 payload {fp16_row})",
        per_tok[0], per_tok[1]
    );
    assert!(per_tok[0] >= fp16_row, "TB5 hop should ship FP16");
    assert!(
        per_tok[0] - per_tok[1] >= 0.6 * fp16_row,
        "starved link should use the codec (≤ 25% of the FP16 payload)"
    );
}

#[test]
fn htep_speculative_tensor_expert_parallel_over_sockets() {
    let cfg = ModelConfig::tiny();
    let reference = ReferenceModel::new(&cfg);
    let p = prompt();
    for (temp, tess) in [
        (
            1.0f32,
            TesseractOptions {
                budget: 1,
                ..Default::default()
            },
        ),
        (
            0.0,
            TesseractOptions {
                budget: 3,
                ..Default::default()
            },
        ),
    ] {
        let r = run(
            &cfg,
            2,
            EngineKind::TesseractHtep,
            vec![(0, 0, 6), (1, 0, 6)],
            0,
            LinkProfile::thunderbolt5(),
            tess.clone(),
            0,
            req(&p, 24, temp),
        );
        assert_eq!(r.res.tokens.len(), 24);
        let (cos, top1) = fidelity(&reference, &p, &r.res);
        assert!(
            cos > 0.9995 && top1 >= 0.95,
            "H-TEP fidelity {cos} top1 {top1} (budget {})",
            tess.budget
        );
        let commits: Vec<u64> = r
            .stats
            .iter()
            .filter(|s| s.node < 2)
            .map(|s| s.commits)
            .collect();
        assert_eq!(commits[0], commits[1], "ranks diverged: {commits:?}");
        let prewakes: u64 = r.stats.iter().map(|s| s.prewakes_sent).sum();
        assert!(prewakes > 0, "H-TEP lookahead never pre-staged an expert");
        if tess.budget > 1 {
            assert!(
                r.res.per_epoch.iter().any(|&k| k > 1),
                "greedy speculation never accepted a draft"
            );
        }
    }
}

#[test]
fn solver_picks_htep_only_for_low_latency_groups() {
    let cfg = ModelConfig::bench_moe();
    let studio = preset("studio-m5max").unwrap();
    let specs: Vec<_> = (0..4u16)
        .map(|i| {
            studio.spec(
                i,
                (0..4u16)
                    .filter(|&j| j != i)
                    .map(|j| LinkSpec {
                        peer: j,
                        profiles: links_between(studio, studio, true),
                    })
                    .collect(),
            )
        })
        .collect();
    let plan = HeteroClusterSolver::new(specs, footprint(&cfg, 3), SolverConfig::default()).solve();
    assert_eq!(plan.htep, Some(vec![0, 1, 2, 3]));
    let presets = [
        studio,
        preset("rtx5090").unwrap(),
        preset("macbook-pro").unwrap(),
        preset("android-snapdragon").unwrap(),
    ];
    let specs: Vec<_> = presets
        .iter()
        .enumerate()
        .map(|(i, p)| {
            p.spec(
                i as u16,
                presets
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(j, q)| LinkSpec {
                        peer: j as u16,
                        profiles: links_between(p, q, false),
                    })
                    .collect(),
            )
        })
        .collect();
    let plan = HeteroClusterSolver::new(specs, footprint(&cfg, 3), SolverConfig::default()).solve();
    assert_eq!(
        plan.htep, None,
        "2.5GbE / Wi-Fi links must not form an H-TEP group"
    );
}

#[test]
fn speculation_runs_only_while_all_kv_is_local() {
    let mut cfg = ModelConfig::tiny();
    cfg.moe = None;
    cfg.n_kv_heads = 2;
    let p = prompt();
    let base = TesseractOptions {
        kv_mode: KvMode::Disaggregated,
        braid_strands: 1,
        draft_ahead: 0,
        lookahead: false,
        page_tokens: 128,
        tier0_pages: 4,
        ..Default::default()
    };
    let mut budgets = Vec::new();
    // 1500 injected tokens: pages beyond tier 0 live on remote holders;
    // 100 tokens: everything is tier 0, i.e. local
    for inject in [1500u32, 100] {
        let r = run(
            &cfg,
            3,
            EngineKind::Tesseract,
            vec![(0, 0, 3), (1, 3, 6)],
            1,
            LinkProfile::thunderbolt5(),
            base.clone(),
            inject,
            req(&p, 24, 0.0),
        );
        budgets.push(r.res.budgets.clone());
    }
    assert!(
        budgets[0].iter().all(|&b| b == 1),
        "trees with remote KV: {:?}",
        budgets[0]
    );
    assert!(
        budgets[1].iter().any(|&b| b > 1),
        "no trees with all-local KV: {:?}",
        budgets[1]
    );
}

#[test]
fn cross_step_stage0_lookahead_is_exact_and_reused() {
    let cfg = ModelConfig::tiny();
    let p = prompt();
    // stage 0 carries 5 of 6 layers: the pipeline bottleneck
    let stages = vec![(0, 0, 5), (1, 5, 6)];
    let run_with = |lookahead: bool, max_new: u32, temperature: f32| {
        run(
            &cfg,
            2,
            EngineKind::Tesseract,
            stages.clone(),
            1,
            LinkProfile::ethernet_2_5g(),
            TesseractOptions {
                budget: 1,
                braid_strands: 1,
                draft_ahead: 0,
                stage_lookahead: lookahead,
                ..Default::default()
            },
            0,
            req(&p, max_new, temperature),
        )
    };
    // Greedy decoding mostly hits (the early exit agrees with the target);
    // sampled decoding (same seed on both sides) often draws another token,
    // which exercises both rollbacks: the precomputed row is discarded on
    // arrival, or aborted mid-run when the TreeCommit overtakes it.
    let mut total = (0u64, 0u64, 0u64);
    for (max_new, temperature) in [(32u32, 0.0f32), (64, 1.0)] {
        // single-row decoding with and without the cross-step lookahead: the
        // only difference is which rows stage 0 precomputed
        let off = run_with(false, max_new, temperature);
        let on = run_with(true, max_new, temperature);
        assert_eq!(
            off.res.tokens, on.res.tokens,
            "lookahead changed the output at T = {temperature}"
        );
        let mut worst = 0.0f32;
        for (a, b) in off.res.logits.iter().zip(&on.res.logits) {
            for (x, y) in a.iter().zip(b) {
                worst = worst.max((x - y).abs());
            }
        }
        let (h, m, ab) = on.stats.iter().fold((0, 0, 0), |acc, s| {
            (
                acc.0 + s.ahead_hits,
                acc.1 + s.ahead_misses,
                acc.2 + s.ahead_aborts,
            )
        });
        eprintln!("T = {temperature}: stage-0 lookahead {h} rows reused, {m} discarded, {ab} aborted early; max logit Δ vs off {worst:.2e}");
        // the reused row attends to the same tokens; only the f32 summation
        // order of the previous root (pending path vs the blocked committed
        // scan) differs — the same tolerance as the other KV-path tests
        assert!(
            worst < 2e-3,
            "logit divergence {worst} at T = {temperature}"
        );
        total = (total.0 + h, total.1 + m, total.2 + ab);
    }
    assert!(total.0 > 10, "lookahead rows reused only {} times", total.0);
    assert!(
        total.1 + total.2 > 0,
        "no rollback was exercised ({} discarded, {} aborted)",
        total.1,
        total.2
    );
}
