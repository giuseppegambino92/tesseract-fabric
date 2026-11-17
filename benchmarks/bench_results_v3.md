# tesseract-bench results

Host: Apple M3 Max · 14 logical CPUs · every node is a set of OS threads with its own TCP endpoints on 127.0.0.1; all traffic is real `TesseractWire` frames through the kernel's TCP stack, delayed by per-link shapers (FIFO serialisation at link bandwidth + one-way latency + jitter). Compute is real FP32 tensor math on this host; slower devices are emulated by extending each node's *measured* compute time by its memory-bandwidth ratio to the fastest node.

Host load average at start: `{ 8,52 10,62 14,93 }` — background load on the host competes with the emulated nodes for cores and adds variance to every timing below.

Model: synthetic Llama/Qwen-style MoE decoder — 16 layers, d_model 512, 8 q / 4 kv heads, 8 experts top-2 (d_expert 704), vocab 2048, 153.2M parameters, same weights on every engine. V3 uses a *contextual* initialisation: the next token depends equally on the current token (+8 nats transition prior) and the previous token (+8 nats), and the previous token reaches the current position only through a layer-0 attention head built with RoPE to attend to offset −1; power-law spectral decay σᵢ ∝ i^-0.5 in layer weights; logit noise σ=1.5. It is calibrated to V1's peakedness (mean top-1 ≈ 0.6). V2 used a first-order prior only; V1 unstructured random weights. 6 of 8 experts per layer fit the hot buffer on every engine; the rest are staged from int8 cold storage.

Generation: single stream, 16-token prompt, 128 new tokens, temperature 1. Network bytes count inter-host frames only (incl. prompt prefill). Scenarios 1–2 run all engines back to back in each of N interleaved rounds (identical seeds) and report each engine's median round, so transient host load hits every engine of a round alike.

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
| Exo-Ring-Pipeline | 63.8 | 61.2–74.6 | 1.00× | 15.68 | 3.79 | 24.4% | 1.00000 (100% top-1) | 1.00 |
| RDMA-Tensor-Parallel | 131.6 | 118.2–159.4 | 2.06× | 7.60 | 456.33 | 53.6% | 1.00000 (100% top-1) | 1.00 |
| Tesseract-Fabric (H-TEP) | 129.1 | 116.9–150.6 | 2.02× | 7.75 | 524.64 | 49.8% | 1.00000 (99% top-1) | 1.13 |
| Tesseract-Fabric (speculation off, ablation) | 133.8 | 115.8–143.1 | 2.10× | 7.48 | 458.83 | 53.8% | 1.00000 (100% top-1) | 1.00 |

Tesseract telemetry: decode 0.99s over 113 epochs; tree budget picks (budget:count) 1:92 2:10 3:5 4:6; draft-ahead hits/misses 0/0; KV commits 456 (mean 0.59 µs); expert staging — blocking misses 4803 (Exo: 958), pre-staged 0, lookahead recall 0.0%, PreWake frames 0 (0.0 KB out-of-band).

#### Greedy decoding (T = 0) ablation, same cluster, median of 3 interleaved rounds

| Engine | Throughput (tok/s) | min–max | Tokens / epoch | Cosine fidelity vs FP32 (top-1 agree) |
|---|---:|---:|---:|---|
| RDMA-Tensor-Parallel | 135.8 | 118.7–138.3 | 1.00 | 0.99999 (100% top-1) |
| Tesseract-Fabric, speculation off | 132.6 | 112.8–145.2 | 1.00 | 0.99999 (100% top-1) |
| Tesseract-Fabric, speculation on | 117.5 | 113.4–134.9 | 1.47 | 0.99999 (100% top-1) |

### Activation codec frontier (Scenario 1 hardware, fixed 8+8-layer 2-stage pipeline, speculation off)

Bytes are whole inter-stage frames (64-byte header + strand metadata + token ids + payload). Exo ships the hidden state as raw FP16 (2 B/element).

| Activation coding | Inter-stage bytes per token | vs FP16 frame | Codec payload per row vs FP16 | Cosine fidelity vs FP32 (top-1 agree) |
|---|---:|---:|---:|---|
| FP16 (Exo ring, same 2-stage split) | 1386 B | 100% | 100.0% | 1.00000 (100% top-1) |
| V1 holographic 8-bit | 810 B | 58% | 52.8% | 0.99998 (100% top-1) |
| V1 holographic 4-bit | 494 B | 36% | 27.8% | 0.99771 (92% top-1) |
| V2 HRQ trunk-residual 3-bit + sketch | 451 B | 33% | 24.2% | 0.99789 (84% top-1) |
| V2 HRQ trunk-residual 2-bit + sketch | 372 B | 27% | 18.0% | 0.99280 (80% top-1) |
| **V2.1 entropy-coded trunk residual, ≤25% cap (default)** | 448 B | 32% | 24.2% | 0.99928 (94% top-1) |

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
| Exo-Ring-Pipeline | 10.0 | 9.7–10.9 | 1.00× | 99.73 | 6.11 | 21.1% | 1.00000 (100% top-1) | 1.00 |
| RDMA-Tensor-Parallel † | 7.7 | 7.0–8.1 | 0.76× | 130.44 | 465.38 | 22.7% | 1.00000 (100% top-1) | 1.00 |
| Tesseract-Fabric | 23.7 | 22.0–25.7 | 2.36× | 42.26 | 2.53 | 41.9% | 0.99947 (94% top-1) | 1.01 |
| Tesseract-Fabric (speculation off, ablation) | 24.9 | 24.7–28.5 | 2.48× | 40.20 | 2.25 | 20.6% | 0.99945 (95% top-1) | 1.00 |

† RDMA-Tensor-Parallel: 1/4 shard (151 MB) exceeds RAM of Android Snapdragon Edge — would OOM; run for latency only

Tesseract telemetry: decode 5.41s over 127 epochs; tree budget picks (budget:count) 1:122 4:2 6:1 8:1 16:1; draft-ahead hits/misses 0/4; KV commits 256 (mean 0.50 µs); expert staging — blocking misses 266 (Exo: 1029), pre-staged 68, lookahead recall 67.7%, PreWake frames 1729 (57.0 KB out-of-band).

## Scenario 3 — Long-Context Scaling Stress Test (4 shards, TB5 links, 8 layers, GQA 8q/2kv heads)

Shard RAM budgets (MB): [512.0, 256.0, 192.0, 128.0]; layer split (both engines, ∝ RAM): [4, 2, 1, 1]. Context KV is injected with the layers' real K/V projections of the context tokens (single-pass approximate prefill), then 24 tokens are decoded with full forward passes.

| Context | Engine | Throughput (tok/s) | Peak node memory (weights+KV) | Memory balance (max/mean util) | Attention traffic (KB/step) | FP16 page-migration equivalent (KB/step) | LSE merge max abs err vs monolithic |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2K | Layer-coupled ring KV (Exo) | 196.27 | 48 MB | 1.09 | 0.0 | — | — (monolithic per layer) |
| 2K | Tesseract disaggregated Flash-KV | 176.48 | 44 MB | 1.09 | 0.0 | 5376 | 6.05e-5 |
| 8K | Layer-coupled ring KV (Exo) | 96.38 | 77 MB | 1.09 | 0.0 | — | — (monolithic per layer) |
| 8K | Tesseract disaggregated Flash-KV | 117.44 | 68 MB | 1.08 | 58.1 | 22016 | 5.99e-5 |
| 32K | Layer-coupled ring KV (Exo) | 27.60 | 174 MB | 1.09 | 0.0 | — | — (monolithic per layer) |
| 32K | Tesseract disaggregated Flash-KV | 49.05 | 162 MB | 1.04 | 58.1 | 87808 | 4.73e-5 |
| 64K | Layer-coupled ring KV (Exo) | 12.29 | 346 MB | 1.09 | 0.0 | — | — (monolithic per layer) |
| 64K | Tesseract disaggregated Flash-KV | 22.95 | 288 MB | 1.02 | 58.1 | 175616 | 4.94e-5 |

Attention traffic for the disaggregated fabric includes the KV query broadcast, the (O, m, l) partials and the append of each step's newly committed K/V rows to their owning page.

### Drafter agreement with the target (96 sampled tokens, T = 1)

| Benchmark model | Target mean top-1 prob | Target entropy | Zero-layer head top-1 agreement | 2-layer early-exit top-1 agreement | Target mass on early-exit argmax |
|---|---:|---:|---:|---:|---:|
| V1 (unstructured random weights) | 0.601 | 1.43 nats | 42.7% | 41.7% | 0.391 |
| V2 (first-order transition prior) | 0.715 | 1.44 nats | 89.6% | 90.6% | 0.700 |
| V3 (contextual: current + previous token, prev. via layer-0 attention) | 0.555 | 1.84 nats | 12.5% | 84.4% | 0.542 |

V2's next token was dominated by a first-order prior, so a zero-layer head already agreed 86.5% of the time. In V3 the next token depends equally on the current and the previous token, and the previous token only reaches the current position through a layer-0 attention head; a zero-layer head cannot see it, while the 2-layer early exit can.

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

| Context | V1 f32 wire | V2 compact wire (f16 Q/O, f32 m,l) | FP16 page migration | V1 reduction | V2 reduction | V2 max abs err |
|---:|---:|---:|---:|---:|---:|---:|
| 4K | 12.7 KB | 6.7 KB | 1152 KB | −98.90% | −99.42% | 1.40e-4 |
| 16K | 12.7 KB | 6.7 KB | 4352 KB | −99.71% | −99.85% | 6.48e-5 |
| 64K | 12.7 KB | 6.7 KB | 17280 KB | −99.93% | −99.96% | 5.66e-5 |

### Weight-reuse tiled batch GEMM (single core, min of 300 runs)

| Weight matrix | B=1 | B=4 | B=6 | B=8 | B=16 | cost B=6 / B=1 | cost B=8 / B=1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2048×512 | 23.0 GMAC/s | 51.5 GMAC/s | 56.2 GMAC/s | 56.6 GMAC/s | 55.9 GMAC/s | 2.45× | 3.24× |
| 704×512 | 23.0 GMAC/s | 51.0 GMAC/s | 56.5 GMAC/s | 56.6 GMAC/s | 55.8 GMAC/s | 2.44× | 3.25× |
| 512×704 | 22.5 GMAC/s | 54.6 GMAC/s | 57.5 GMAC/s | 57.3 GMAC/s | 57.4 GMAC/s | 2.35× | 3.15× |

### Anchor-consensus expert coalescing (APPROXIMATE; one MoE layer, 4-row tree, min of 40 trials)

| | Distinct experts touched | Time | vs 1 row |
|---|---:|---:|---:|
| 1 row (anchor only) | 2 | 142.5 µs | 1.00× |
| 4 rows, exact routing | 5.4 (mean) | 422.4 µs | 2.96× |
| 4 rows, coalesced (anchor's experts + 1 consensus) | ≤ 3 | 265.1 µs | 1.86× |

Coalescing changes the candidate rows' FFN outputs by 70.7% (relative L2, mean over candidate rows), so trees verified with it are not distributed exactly as the target model; it is therefore opt-in (`coalesce_experts`) and off in every scenario run.

### Lock-free slab KV commit (16 layers, 16 pending rows → 4 accepted)

Median 0.12 µs, p90 0.12 µs, p99 0.75 µs for the whole shard (V1: per-layer copy + compaction, ≈2 µs per layer in isolation, 14–16 µs per commit inside the loaded V1 cluster).

### Multipath striping (512 KB frames, 2.5GbE + Wi-Fi 6E, live sockets)

| Mode | Goodput, median frame (Gbps) | Byte split |
|---|---:|---|
| Best single link | 1.46 | 2.5GbE 100% |
| ETTC water-filling stripes | 1.60 | 2.5GbE 83% / WiFi 17% |
| 2.5GbE brownout (8 ms, 0.1 Gbps) — after telemetry adapts | 0.44 | 2.5GbE 0% / WiFi 100% |

_Total benchmark wall time: 548.7 s._
