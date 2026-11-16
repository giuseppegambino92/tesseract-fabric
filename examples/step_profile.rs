//! Where a decode step's time goes on a real checkpoint (single node, one
//! lane pool): per-component timings for 1, 4 and 12 rows.
//! `cargo run --release --example step_profile -- models/<file>.gguf`
use std::time::Instant;
use tesseract_fabric::kv_fabric::{AttentionBackend, BatchMeta, LocalKvCache};
use tesseract_fabric::loader::sharded_loader::config_from_file;
use tesseract_fabric::model::*;
use tesseract_fabric::tensor::{rms_norm, Tensor2D};
fn main() {
    let path = std::env::args().nth(1).expect("model path");
    let cfg = config_from_file(&path, None).unwrap();
    tesseract_fabric::tensor::pool::install(2);
    let head = HeadWeights::generate(&cfg);
    let layers: Vec<LayerWeights> = (0..cfg.n_layers)
        .map(|l| LayerWeights::generate(&cfg, l, None))
        .collect();
    let rope = cfg.rope_table();
    let ctx = 200;
    for exact_attn in [false, true] {
        for m in [1usize, 4, 12] {
            let mut kv =
                LocalKvCache::new(cfg.n_heads, cfg.n_kv_heads, cfg.d_head).with_exact(exact_attn);
            // a committed context of `ctx` tokens
            let toks: Vec<u32> = (0..ctx as u32).map(|i| 100 + i).collect();
            let meta = BatchMeta::causal_prefill(0, ctx, 1);
            let mut x = head.embed_tokens(&toks);
            let model = ShardModel {
                layers: layers.clone(),
                ..ShardModel::generate(&cfg, 0..0, None)
            };
            model.forward(&mut x, &meta, &mut kv, &mut NoObserver);
            kv.commit(&meta.keys, &|_| false);
            let mut t = [0f64; 6];
            let reps = 5;
            for rep in 0..reps {
                let meta = BatchMeta::causal_prefill(ctx, m, 10 + rep);
                let mut x = head.embed_tokens(&toks[..m]);
                for l in &layers {
                    let t0 = Instant::now();
                    let mut xn = Tensor2D::zeros(m, cfg.d_model);
                    for i in 0..m {
                        rms_norm(x.row(i), &l.attn_norm, cfg.rms_eps, xn.row_mut(i));
                    }
                    let (q, k, v) = project_qkv(
                        &l.wq,
                        &l.wk,
                        &l.wv,
                        &l.extras,
                        &xn,
                        &rope,
                        &meta.positions,
                        cfg.d_head,
                        cfg.rms_eps,
                    );
                    let t1 = Instant::now();
                    let a = kv.attend(l.index, &q, &k, &v, &meta);
                    let t2 = Instant::now();
                    let o = l.wo.forward(&a);
                    let t3 = Instant::now();
                    for (xi, oi) in x.data.iter_mut().zip(&o.data) {
                        *xi += oi;
                    }
                    for i in 0..m {
                        rms_norm(x.row(i), &l.ffn_norm, cfg.rms_eps, xn.row_mut(i));
                    }
                    let FfnWeights::Dense(e) = &l.ffn else {
                        panic!()
                    };
                    let h = e.hidden(&xn);
                    let t4 = Instant::now();
                    let f = e.w_down.forward(&h);
                    let t5 = Instant::now();
                    for (xi, fi) in x.data.iter_mut().zip(&f.data) {
                        *xi += fi;
                    }
                    t[0] += (t1 - t0).as_secs_f64();
                    t[1] += (t2 - t1).as_secs_f64();
                    t[2] += (t3 - t2).as_secs_f64();
                    t[3] += (t4 - t3).as_secs_f64();
                    t[4] += (t5 - t4).as_secs_f64();
                }
                let t0 = Instant::now();
                std::hint::black_box(head.logits(&x, cfg.rms_eps));
                t[5] += t0.elapsed().as_secs_f64();
                kv.commit(&[], &|_| false);
            }
            let ms = |v: f64| v / reps as f64 * 1e3;
            println!(
                "exact attn {exact_attn:5} · {m:2} rows: qkv {:.2} · attention {:.2} · wo {:.2} · gate/up {:.2} · down {:.2} · lm_head {:.2} ms (total {:.2})",
                ms(t[0]), ms(t[1]), ms(t[2]), ms(t[3]), ms(t[4]), ms(t[5]), ms(t.iter().sum())
            );
        }
    }
}
