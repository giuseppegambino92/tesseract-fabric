# tesseract-bench results

Host: Apple M3 Max · 14 logical CPUs · every node is a set of OS threads with its own TCP endpoints on 127.0.0.1; all traffic is real `TesseractWire` frames through the kernel's TCP stack, delayed by per-link shapers (FIFO serialisation at link bandwidth + one-way latency + jitter). Compute is real FP32 tensor math on this host; slower devices are emulated by extending each node's *measured* compute time by its memory-bandwidth ratio to the fastest node.

Host load average at start: `{ 50,22 86,27 69,53 }` — background load on the host competes with the emulated nodes for cores and adds variance to every timing below.

Model: synthetic Llama/Qwen-style MoE decoder — 16 layers, d_model 512, 8 q / 4 kv heads, 8 experts top-2 (d_expert 704), vocab 2048, 153.2M parameters, same weights on every engine. V3 uses a *contextual* initialisation: the next token depends equally on the current token (+8 nats transition prior) and the previous token (+8 nats), and the previous token reaches the current position only through a layer-0 attention head built with RoPE to attend to offset −1; power-law spectral decay σᵢ ∝ i^-0.5 in layer weights; logit noise σ=1.5. It is calibrated to V1's peakedness (mean top-1 ≈ 0.6). V2 used a first-order prior only; V1 unstructured random weights. 6 of 8 experts per layer fit the hot buffer on every engine; the rest are staged from int8 cold storage.

Generation: single stream, 16-token prompt, 128 new tokens, temperature 0 (greedy decoding: every engine must emit exactly the FP32 reference's argmax sequence up to its own numerical error; a T = 1 sampled ablation follows Scenario 1). Network bytes count inter-host frames only (incl. prompt prefill). Scenarios 1–2 run all engines back to back in each of N interleaved rounds (identical seeds) and report each engine's median round, so transient host load hits every engine of a round alike.

## Scenario 1 — Homogeneous High-Speed Studio Cluster (4× Mac Studio, Thunderbolt 5 RDMA profile: 40 µs RTT, 80 Gbps)

| Node | Device profile | Platform | Mem BW (GB/s) | Emulated RAM (MB) | Compute slowdown |
|---|---|---|---:|---:|---:|
| 0 | Mac Studio M5 Max | AppleSiliconUMA | 546 | 640 | 1.00× |
| 1 | Mac Studio M5 Max | AppleSiliconUMA | 546 | 640 | 1.00× |
| 2 | Mac Studio M5 Max | AppleSiliconUMA | 546 | 640 | 1.00× |
| 3 | Mac Studio M5 Max | AppleSiliconUMA | 546 | 640 | 1.00× |

Links: 0↔1: TB5-RDMA, 0↔2: TB5-RDMA, 0↔3: TB5-RDMA, 1↔2: TB5-RDMA, 1↔3: TB5-RDMA, 2↔3: TB5-RDMA

* Exo ring partition (layers ∝ RAM): node 0: L0–3 → node 1: L4–7 → node 2: L8–11 → node 3: L12–15
* HeteroClusterSolver plan: node 0: L0–15 → node 1: L0–15 → node 2: L0–15 → node 3: L0–15 | coordinator = node 0 | drafter = node 0 | est. epoch 0.41 ms
  * Mac Studio M5 Max → asynchronous tree drafter
  * Mac Studio M5 Max → overflow KV page pool
  * low-latency group [0, 1, 2, 3] (RTT < 100 µs) → H-TEP: replicated attention, experts split across ranks, 1 sparse all-reduce per layer

| Engine | Throughput, median of 5 interleaved rounds (tok/s) | min–max | vs Exo | Per-token latency (ms) | Network per token (KB) | Active HW utilization | Cosine fidelity vs FP32 (top-1 agree) | Tokens / epoch |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| Exo-Ring-Pipeline | 29.4 | 24.5–36.6 | 1.00× | 34.02 | 3.79 | 21.8% | 0.99999 (100% top-1) | 1.00 |
| RDMA-Tensor-Parallel | 46.9 | 43.6–57.7 | 1.60× | 21.32 | 456.33 | 41.6% | 0.99999 (100% top-1) | 1.00 |
| Tesseract-Fabric (H-TEP) | 77.2 | 55.6–106.0 | 2.63× | 12.95 | 621.06 | 53.4% | 0.99999 (100% top-1) | 1.47 |
| Tesseract-Fabric (speculation off, ablation) | 53.6 | 35.5–90.7 | 1.82× | 18.64 | 458.83 | 37.8% | 0.99999 (100% top-1) | 1.00 |

Tesseract telemetry: decode 1.66s over 87 epochs; tree budget picks (budget:count) 1:40 2:20 3:3 4:24; draft-ahead hits/misses 0/0; KV commits 352 (mean 1.01 µs); expert staging — blocking misses 1123 (Exo: 973), pre-staged 346, lookahead recall 76.4%, PreWake frames 4576 (0.0 KB out-of-band).

#### Sampled decoding (T = 1) ablation, same cluster, median of 3 interleaved rounds

| Engine | Throughput (tok/s) | min–max | Tokens / epoch | Cosine fidelity vs FP32 (top-1 agree) |
|---|---:|---:|---:|---|
| RDMA-Tensor-Parallel | 43.6 | 31.4–45.3 | 1.00 | 1.00000 (100% top-1) |
| Tesseract-Fabric, speculation off | 48.7 | 34.1–60.6 | 1.00 | 1.00000 (100% top-1) |
| Tesseract-Fabric, speculation on | 40.2 | 35.0–50.1 | 1.15 | 1.00000 (100% top-1) |

### Activation codec frontier (Scenario 1 hardware, fixed 8+8-layer 2-stage pipeline, speculation off, 3 prompts × 128 tokens, temperature 0)

Bytes are whole inter-stage frames (64-byte header + strand metadata + token ids + payload). Exo ships the hidden state as raw FP16 (2 B/element). Fidelity aggregates every generated position of all prompts, so a 0.1-point top-1 difference is about one token.

| Activation coding | Inter-stage bytes per token | vs FP16 frame | Codec payload per row vs FP16 | Cosine fidelity vs FP32 | Top-1 agreement |
|---|---:|---:|---:|---:|---:|
| FP16 (Exo ring, same 2-stage split) | 1256 B | 100% | 100.0% | 0.99999 | 99.74% (383/384) |
| V1 holographic 8-bit | 736 B | 59% | 52.8% | 0.99997 | 98.70% (379/384) |
| V1 holographic 4-bit | 450 B | 36% | 27.8% | 0.99770 | 92.71% (356/384) |
| V2 HRQ trunk-residual 3-bit + sketch | 411 B | 33% | 24.2% | 0.99793 | 92.71% (356/384) |
| V2 HRQ trunk-residual 2-bit + sketch | 339 B | 27% | 18.0% | 0.99263 | 88.02% (338/384) |
| **V2.1 entropy-coded trunk residual, ≤25% cap (default)** | 409 B | 33% | 24.2% | 0.99940 | 95.57% (367/384) |

## Scenario 2 — Heterogeneous Multi-Platform Cluster (Mac Studio + RTX 5090 on 2.5GbE, MacBook Pro + Android on Wi-Fi 6E)

| Node | Device profile | Platform | Mem BW (GB/s) | Emulated RAM (MB) | Compute slowdown |
|---|---|---|---:|---:|---:|
| 0 | Mac Studio M5 Max | AppleSiliconUMA | 546 | 640 | 3.28× |
| 1 | Linux RTX 5090 Workstation | DiscreteGpuWorkstation | 1792 | 160 | 1.00× |
| 2 | MacBook Pro (M-Pro) | AppleSiliconUMA | 273 | 240 | 6.56× |
| 3 | Android Snapdragon Edge | MobileEdgeNpu | 77 | 80 | 23.27× |

Links: 0↔1: 2.5GbE+WiFi6E, 0↔2: WiFi6E, 0↔3: WiFi6E, 1↔2: WiFi6E, 1↔3: WiFi6E, 2↔3: WiFi6E

* Exo ring partition (layers ∝ RAM): node 0: L0–8 → node 1: L9–10 → node 2: L11–14 → node 3: L15–15
* HeteroClusterSolver plan: node 0: L0–12 → node 1: L13–15 | coordinator = node 1 | drafter = node 3 | est. epoch 0.69 ms
  * Android Snapdragon Edge → asynchronous tree drafter
  * MacBook Pro (M-Pro) → overflow KV page pool

| Engine | Throughput, median of 5 interleaved rounds (tok/s) | min–max | vs Exo | Per-token latency (ms) | Network per token (KB) | Active HW utilization | Cosine fidelity vs FP32 (top-1 agree) | Tokens / epoch |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| Exo-Ring-Pipeline | 4.9 | 3.3–5.4 | 1.00× | 206.10 | 5.81 | 21.9% | 0.99999 (100% top-1) | 1.00 |
| RDMA-Tensor-Parallel † | 2.9 | 2.3–3.3 | 0.59× | 350.45 | 479.94 | 23.6% | 0.99999 (100% top-1) | 1.00 |
| Tesseract-Fabric | 8.5 | 7.3–10.2 | 1.76× | 117.03 | 6.01 | 24.3% | 0.99936 (93% top-1) | 1.97 |
| Tesseract-Fabric (speculation off, ablation) | 14.9 | 11.5–16.5 | 3.08× | 66.97 | 4.28 | 21.3% | 0.99936 (93% top-1) | 1.00 |

† RDMA-Tensor-Parallel: 1/4 shard (151 MB) exceeds RAM of Android Snapdragon Edge — would OOM; run for latency only

Tesseract telemetry: decode 14.98s over 65 epochs; tree budget picks (budget:count) 1:19 4:21 6:10 8:5 16:10; draft-ahead hits/misses 0/45; KV commits 132 (mean 1.81 µs); expert staging — blocking misses 571 (Exo: 971), pre-staged 57, lookahead recall 81.8%, PreWake frames 1456 (46.4 KB out-of-band).

## Scenario 3 — Long-Context Scaling Stress Test (4 shards, TB5 links, 8 layers, GQA 8q/2kv heads)

Shard RAM budgets (MB): [512.0, 256.0, 192.0, 128.0]; layer split (both engines, ∝ RAM): [4, 2, 1, 1]. Context KV is injected with the layers' real K/V projections of the context tokens (single-pass approximate prefill), then 24 tokens are decoded with full forward passes.

| Context | Engine | Throughput (tok/s) | Peak node memory (weights+KV) | Memory balance (max/mean util) | Attention traffic (KB/step) | FP16 page-migration equivalent (KB/step) | LSE merge max abs err vs monolithic |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2K | Layer-coupled ring KV (Exo) | 92.03 | 48 MB | 1.09 | 0.0 | — | — (monolithic per layer) |
| 2K | Tesseract disaggregated Flash-KV | 79.77 | 44 MB | 1.09 | 0.0 | 5376 | 6.05e-5 |
| 8K | Layer-coupled ring KV (Exo) | 42.17 | 77 MB | 1.09 | 0.0 | — | — (monolithic per layer) |
| 8K | Tesseract disaggregated Flash-KV | 27.21 | 68 MB | 1.08 | 54.9 | 22016 | 5.99e-5 |
| 32K | Layer-coupled ring KV (Exo) | 9.35 | 174 MB | 1.09 | 0.0 | — | — (monolithic per layer) |
| 32K | Tesseract disaggregated Flash-KV | 17.35 | 162 MB | 1.04 | 54.9 | 87808 | 4.70e-5 |
| 64K | Layer-coupled ring KV (Exo) | 4.81 | 346 MB | 1.09 | 0.0 | — | — (monolithic per layer) |
| 64K | Tesseract disaggregated Flash-KV | 8.06 | 288 MB | 1.02 | 54.9 | 175616 | 5.97e-5 |

Attention traffic for the disaggregated fabric includes the KV query broadcast, the (O, m, l) partials and the append of each step's newly committed K/V rows to their owning page.

### Drafter agreement with the target (256 sampled tokens, T = 1)

| Benchmark model | Target mean top-1 prob | Target entropy | Zero-layer head top-1 agreement | **1 attention-only layer (V4 default drafter)** top-1 agreement | 2-layer early-exit top-1 agreement | Target mass on 2-layer argmax |
|---|---:|---:|---:|---:|---:|---:|
| V1 (unstructured random weights) | 0.591 | 1.45 nats | 48.8% | 47.7% | 51.2% | 0.409 |
| V2 (first-order transition prior) | 0.730 | 1.38 nats | 89.5% | 90.6% | 90.2% | 0.714 |
| V3 (contextual: current + previous token, prev. via layer-0 attention) | 0.585 | 1.69 nats | 13.3% | 82.0% | 85.9% | 0.569 |

V2's next token was dominated by a first-order prior, so a zero-layer head already agreed 86.5% of the time. In V3 the next token depends equally on the current and the previous token, and the previous token only reaches the current position through a layer-0 attention head; a zero-layer head cannot see it, while the 2-layer early exit can. V4's default drafter is the target's own layer 0 with its MoE FFN skipped (attention + residual only, then the target's final norm and LM head): the previous-token head is all it needs, at a small fraction of a target step. It shares weights with the target and is not tuned to the data, so the honesty contrast with the zero-layer head still holds.

## Pillar micro-benchmarks

### Holographic residual compression (d = 4096, 128 autoregressive steps, synthetic low-rank residual trajectory with massive-activation channels)

| Bits | Error feedback γ | Payload vs FP16 | Cumulative L2 drift | Min cosine | Drift reduction vs stateless |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.95 | −85.8% | 236.1 | 0.99974 | ×5.18 |
| 3 | 0.95 | −79.6% | 107.5 | 0.99995 | ×6.41 |
| 4 | 0.95 | −73.3% | 55.9 | 0.99999 | ×6.90 |

### Cross-silicon determinism (Q8.24 CanonicalBoundaryAnchor)

RMSNorm Σx² over 4096-wide activations: scalar / NEON-4 / AVX2-8 / warp-32 / FMA reduction orders disagree bitwise in 100/100 trials with IEEE f32; the Q8.24 integer accumulation is bit-identical in 100/100.

### Disaggregated attention traffic per decode step (one layer, 8 query heads, 4 shards; bytes incl. frame headers)

| Context | V1 f32 wire | V3 f16 wire | V4 BFP wire | FP16 page migration | V3 reduction | V4 reduction | V3 max abs err | V4 max abs err |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4K | 12.7 KB | 6.7 KB | 6.3 KB | 1152 KB | −99.42% | −99.45% | 1.40e-4 | 6.30e-5 |
| 16K | 12.7 KB | 6.7 KB | 6.3 KB | 4352 KB | −99.85% | −99.86% | 6.50e-5 | 3.28e-5 |
| 64K | 12.7 KB | 6.7 KB | 6.3 KB | 17280 KB | −99.96% | −99.96% | 5.62e-5 | 2.35e-5 |

### Weight-reuse tiled batch GEMM (single core, min of 300 runs)

| Weight matrix | B=1 | B=4 | B=6 | B=8 | B=16 | cost B=6 / B=1 | cost B=8 / B=1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2048×512 | 8.1 GMAC/s | 18.7 GMAC/s | 20.0 GMAC/s | 20.0 GMAC/s | 20.1 GMAC/s | 2.44× | 3.25× |
| 704×512 | 8.2 GMAC/s | 18.8 GMAC/s | 20.1 GMAC/s | 20.2 GMAC/s | 20.1 GMAC/s | 2.45× | 3.25× |
| 512×704 | 8.1 GMAC/s | 19.5 GMAC/s | 20.6 GMAC/s | 20.6 GMAC/s | 20.6 GMAC/s | 2.35× | 3.13× |

### Anchor-consensus expert coalescing (APPROXIMATE; one MoE layer, 4-row tree, min of 40 trials)

| | Distinct experts touched | Time | vs 1 row |
|---|---:|---:|---:|
| 1 row (anchor only) | 2 | 407.0 µs | 1.00× |
| 4 rows, exact routing | 5.4 (mean) | 992.3 µs | 2.44× |
| 4 rows, coalesced (anchor's experts + 1 consensus) | ≤ 3 | 693.6 µs | 1.70× |

Coalescing changes the candidate rows' FFN outputs by 70.7% (relative L2, mean over candidate rows), so trees verified with it are not distributed exactly as the target model; it is therefore opt-in (`coalesce_experts`) and off in every scenario run.

### Lock-free slab KV commit (16 layers, 16 pending rows → 4 accepted)

Median 0.33 µs, p90 0.75 µs, p99 4.00 µs for the whole shard (V1: per-layer copy + compaction, ≈2 µs per layer in isolation, 14–16 µs per commit inside the loaded V1 cluster).

### Multipath striping (512 KB frames, 2.5GbE + Wi-Fi 6E, live sockets)

| Mode | Goodput, median frame (Gbps) | Byte split |
|---|---:|---|
| Best single link | 1.02 | 2.5GbE 100% |
| ETTC water-filling stripes | 0.47 | 2.5GbE 56% / WiFi 44% |
| 2.5GbE brownout (8 ms, 0.1 Gbps) — after telemetry adapts | 0.44 | 2.5GbE 0% / WiFi 100% |

_Total benchmark wall time: 1635.9 s._
