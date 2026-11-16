//! Real checkpoint: single-node greedy vs a 4-rank H-TEP cluster and a 4-stage
//! pipeline (lossless, and KLT with `KLT=1`):
//! `cargo run --release --example cluster_probe -- models/SmolLM2-135M-Instruct-Q8_0.gguf 32`
use tesseract_fabric::cluster::*;
use tesseract_fabric::kv_fabric::{BatchMeta, LocalKvCache};
use tesseract_fabric::loader::sharded_loader::{config_from_file, tokenizer_for};
use tesseract_fabric::model::*;
use tesseract_fabric::tokenizer::{stop_tokens, ChatFormat};
use tesseract_fabric::wire::multipath::LinkProfile;
fn single(cfg: &ModelConfig, prompt: &[u32], n: usize) -> Vec<u32> {
    let head = HeadWeights::generate(cfg);
    let model = ShardModel::generate(cfg, 0..cfg.n_layers, None);
    let mut kv = LocalKvCache::new(cfg.n_heads, cfg.n_kv_heads, cfg.d_head);
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
        toks.push(argmax(head.logits(&x, cfg.rms_eps).row(0)) as u32);
    }
    toks[m..].to_vec()
}
fn cluster(
    cfg: &ModelConfig,
    engine: EngineKind,
    stages: Vec<(u16, usize, usize)>,
    head: u16,
    tess: TesseractOptions,
    prompt: &[u32],
    n: u32,
) -> (GenerateResult, f64) {
    let ids: Vec<u16> = (0..4).collect();
    let ws = spawn_local_workers(&ids);
    let driver = Driver::bind("127.0.0.1:0").unwrap();
    let mut links = vec![];
    for a in 0..4u16 {
        for b in a + 1..4 {
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
    let t = std::time::Instant::now();
    driver.assign(&a);
    let setup = t.elapsed().as_secs_f64();
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
    (r, setup)
}
fn main() {
    let path = std::env::args().nth(1).unwrap();
    let n: u32 = std::env::args()
        .nth(2)
        .map(|s| s.parse().unwrap())
        .unwrap_or(32);
    let cfg = config_from_file(&path, None).unwrap();
    let tok = tokenizer_for(&cfg);
    let fmt = ChatFormat::detect(&tok);
    let prompt = fmt.encode(&tok, &[("user".into(), "List the capitals of France, Italy, Spain, Germany and Portugal, one per line, in the form: The capital of X is Y.".into())]);
    let stops = stop_tokens(&tok);
    let t = std::time::Instant::now();
    let s = single(&cfg, &prompt, n as usize);
    println!(
        "single ({:.1}s): {:?}",
        t.elapsed().as_secs_f64(),
        tok.decode(&s)
    );
    let (h, setup) = cluster(
        &cfg,
        EngineKind::TesseractHtep,
        (0..4).map(|i| (i, 0, cfg.n_layers)).collect(),
        0,
        TesseractOptions {
            htep_reduce_f32: true,
            draft_lookup: true,
            ..Default::default()
        },
        &prompt,
        n,
    );
    let mut hist = std::collections::BTreeMap::new();
    for b in &h.budgets {
        *hist.entry(*b).or_insert(0) += 1;
    }
    println!(
        "H-TEP budgets {hist:?} per_epoch {:?}",
        &h.per_epoch[..h.per_epoch.len().min(20)]
    );
    println!(
        "H-TEP x4 (setup {setup:.1}s, {:.1} tok/s, {:.2} tok/step): {:?}  match={}",
        h.tokens.len() as f64 / (h.decode_ns as f64 * 1e-9),
        h.tokens.len() as f64 / h.per_epoch.len() as f64,
        tok.decode(&h.tokens),
        h.tokens == s
    );
    let q = cfg.n_layers / 4;
    let st: Vec<(u16, usize, usize)> = (0..4)
        .map(|i| {
            (
                i as u16,
                i * q,
                if i == 3 { cfg.n_layers } else { (i + 1) * q },
            )
        })
        .collect();
    let opts = TesseractOptions {
        codec: 0,
        draft_lookup: true,
        budget: std::env::var("BUDGET")
            .ok()
            .and_then(|b| b.parse().ok())
            .unwrap_or(0),
        stage_lookahead: std::env::var("NOAHEAD").is_err(),
        ..Default::default()
    };
    let (p, setup) = cluster(&cfg, EngineKind::Tesseract, st.clone(), 3, opts, &prompt, n);
    println!(
        "pipeline x4 lossless (setup {setup:.1}s, {:.1} tok/s, {:.2} tok/step): {:?}  match={}",
        p.tokens.len() as f64 / (p.decode_ns as f64 * 1e-9),
        p.tokens.len() as f64 / p.per_epoch.len() as f64,
        tok.decode(&p.tokens),
        p.tokens == s
    );
    if std::env::var("KLT").is_err() {
        return;
    }
    let (p, setup) = cluster(
        &cfg,
        EngineKind::Tesseract,
        st,
        3,
        TesseractOptions {
            codec: 4,
            ..Default::default()
        },
        &prompt,
        n,
    );
    println!(
        "pipeline x4 KLT (setup {setup:.1}s, {:.1} tok/s, {:.2} tok/step): {:?}  match={}",
        p.tokens.len() as f64 / (p.decode_ns as f64 * 1e-9),
        p.tokens.len() as f64 / p.per_epoch.len() as f64,
        tok.decode(&p.tokens),
        p.tokens == s
    );
    let _ = stops;
}
