//! Single-node greedy decode of the benchmark model under both backends,
//! from in-memory FP32 and from F16 / Q8_0 GGUF exports:
//! `cargo run --release --example accel_crossover`.
use std::time::Instant;
use tesseract_fabric::kv_fabric::{BatchMeta, LocalKvCache};
use tesseract_fabric::loader::sharded_loader::{config_from_file, export_gguf};
use tesseract_fabric::model::*;
use tesseract_fabric::tensor::accel::{self, Backend};
use tesseract_fabric::tensor::quant::WeightFormat;
fn decode(cfg: &ModelConfig, n: usize) -> f64 {
    let head = HeadWeights::generate(cfg);
    let model = ShardModel::generate(cfg, 0..cfg.n_layers, None);
    let mut kv = LocalKvCache::new(cfg.n_heads, cfg.n_kv_heads, cfg.d_head);
    let mut toks: Vec<u32> = (0..16).map(|i| i * 97 % cfg.vocab as u32).collect();
    let meta = BatchMeta::causal_prefill(0, 15, 1);
    let mut x = head.embed_tokens(&toks[..15]);
    model.forward(&mut x, &meta, &mut kv, &mut NoObserver);
    kv.commit(&meta.keys, &|_| false);
    let t = Instant::now();
    for _ in 0..n {
        let pos = toks.len() - 1;
        let meta = BatchMeta::causal_prefill(pos, 1, 2 + pos as u64);
        let mut x = head.embed_tokens(&toks[pos..]);
        model.forward(&mut x, &meta, &mut kv, &mut NoObserver);
        kv.commit(&meta.keys, &|_| false);
        toks.push(argmax(head.logits(&x, cfg.rms_eps).row(0)) as u32);
    }
    n as f64 / t.elapsed().as_secs_f64()
}
fn main() {
    tesseract_fabric::tensor::pool::install(2);
    let cfg = ModelConfig::bench_moe();
    let dir = std::env::temp_dir();
    let mut cfgs = vec![("in-memory F32", cfg.clone())];
    let mut files = Vec::new();
    for f in [WeightFormat::F16, WeightFormat::Q8_0] {
        let p = dir
            .join(format!("xover-{}-{}.gguf", std::process::id(), f.name()))
            .to_string_lossy()
            .into_owned();
        export_gguf(&cfg, f, None, &p).unwrap();
        cfgs.push((f.name(), config_from_file(&p, None).unwrap()));
        files.push(p);
    }
    for (name, c) in &cfgs {
        for b in [
            Backend::Portable,
            Backend::Accelerate,
            Backend::Portable,
            Backend::Accelerate,
        ] {
            accel::set_backend(b);
            println!("{name:14} {:10} {:6.1} tok/s", b.name(), decode(c, 48));
        }
    }
    for p in files {
        let _ = std::fs::remove_file(p);
    }
}
