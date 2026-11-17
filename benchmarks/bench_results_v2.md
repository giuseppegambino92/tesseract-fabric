# tesseract-bench results

Host: Apple M3 Max · 14 logical CPUs · every node is a set of OS threads with its own TCP endpoints on 127.0.0.1; all traffic is real `TesseractWire` frames through the kernel's TCP stack, delayed by per-link shapers (FIFO serialisation at link bandwidth + one-way latency + jitter). Compute is real FP32 tensor math on this host; slower devices are emulated by extending each node's *measured* compute time by its memory-bandwidth ratio to the fastest node.

Host load average at start: `{ 453,36 461,56 465,36 }` — background load on the host competes with the emulated nodes for cores and adds variance to every timing below.

Model: synthetic Llama/Qwen-style MoE decoder — 16 layers, d_model 512, 8 q / 4 kv heads, 8 experts top-2 (d_expert 704), vocab 2048, 153.2M parameters, same weights on every engine. V2 uses a *structured* initialisation: a token-transition prior embedded in the LM head (+12 nats for a token's top successor), power-law spectral decay σᵢ ∝ i^-0.5 in layer weights, and logit noise σ=1.5 — calibrated to keep V1's next-token entropy (≈1.4 nats) while making the next token largely predictable from the current one. V1 used unstructured random weights. 6 of 8 experts per layer fit the hot buffer on every engine; the rest are staged from int8 cold storage.

Generation: single stream, 16-token prompt, 128 new tokens, temperature 1. Network bytes count inter-host frames only (incl. prompt prefill). Scenarios 1–2 report the faster of 2 runs per engine with identical seeds (baseline tokens are identical across repeats; Tesseract's draft-ahead cache hits depend on timing, so its sampled path may differ).

## Scenario 1 — Homogeneous High-Speed Studio Cluster (4× Mac Studio, Thunderbolt 5 RDMA profile: 40 µs RTT, 80 Gbps)

| Node | Device profile | Platform | Mem BW (GB/s) | Emulated RAM (MB) | Compute slowdown |
|---|---|---|---:|---:|---:|
| 0 | Mac Studio M5 Max | AppleSiliconUMA | 546 | 640 | 1.00× |
| 1 | Mac Studio M5 Max | AppleSiliconUMA | 546 | 640 | 1.00× |
| 2 | Mac Studio M5 Max | AppleSiliconUMA | 546 | 640 | 1.00× |
| 3 | Mac Studio M5 Max | AppleSiliconUMA | 546 | 640 | 1.00× |

Links: 0↔1: TB5-RDMA, 0↔2: TB5-RDMA, 0↔3: TB5-RDMA, 1↔2: TB5-RDMA, 1↔3: TB5-RDMA, 2↔3: TB5-RDMA

* Exo ring partition (layers ∝ RAM): node 0: L0–3 → node 1: L4–7 → node 2: L8–11 → node 3: L12–15
* HeteroClusterSolver plan: node 0: L0–13 → node 1: L14–15 | coordinator = node 1 | drafter = node 2 | est. epoch 0.41 ms
  * Mac Studio M5 Max → asynchronous tree drafter
  * Mac Studio M5 Max → overflow KV page pool

| Engine | Throughput (tok/s) | vs Exo | Per-token latency (ms) | Network per token (KB) | Active HW utilization | Cosine fidelity vs FP32 (top-1 agree) | Tokens / epoch |
|---|---:|---:|---:|---:|---:|---|---:|
| Exo-Ring-Pipeline | 39.8 | 1.00× | 25.11 | 3.79 | 23.8% | 0.99991 (100% top-1) | 1.00 |
| RDMA-Tensor-Parallel | 63.5 | 1.59× | 15.75 | 456.33 | 42.0% | 0.99991 (100% top-1) | 1.00 |
| Tesseract-Fabric | 33.1 | 0.83× | 30.19 | 2.15 | 51.3% | 0.99815 (97% top-1) | 3.37 |
| Tesseract-Fabric (speculation off, ablation) | 47.1 | 1.18× | 21.25 | 0.91 | 23.1% | 0.99818 (99% top-1) | 1.00 |

Tesseract telemetry: decode 3.86s over 38 epochs; tree budget picks (budget:count) 1:2 4:3 6:3 8:15 16:15; draft-ahead hits/misses 3/34; KV commits 78 (mean 1.38 µs); expert staging — blocking misses 773 (Exo: 1071), pre-staged 31, lookahead recall 94.1%, PreWake frames 975 (21.6 KB out-of-band).

### Activation codec frontier (Scenario 1 plan, speculation off)

Bytes are whole inter-stage frames (64-byte header + strand metadata + token ids + payload). Exo ships the hidden state as raw FP16 (2 B/element).

| Activation coding | Inter-stage bytes per token | vs FP16 frame | Codec payload per row vs FP16 | Cosine fidelity vs FP32 (top-1 agree) |
|---|---:|---:|---:|---|
| FP16 (Exo ring, same 2-stage split) | 1386 B | 100% | 100.0% | 0.99974 (100% top-1) |
| V1 holographic 8-bit | 811 B | 58% | 52.8% | 0.99992 (98% top-1) |
| V1 holographic 4-bit | 495 B | 36% | 27.8% | 0.99425 (100% top-1) |
| **V2 HRQ trunk-residual 3-bit (default)** | 451 B | 33% | 24.2% | 0.99806 (98% top-1) |
| V2 HRQ trunk-residual 2-bit | 372 B | 27% | 18.0% | 0.99402 (100% top-1) |

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

| Engine | Throughput (tok/s) | vs Exo | Per-token latency (ms) | Network per token (KB) | Active HW utilization | Cosine fidelity vs FP32 (top-1 agree) | Tokens / epoch |
|---|---:|---:|---:|---:|---:|---|---:|
| Exo-Ring-Pipeline | 5.6 | 1.00× | 179.52 | 9.21 | 19.9% | 0.99987 (100% top-1) | 1.00 |
| RDMA-Tensor-Parallel † | 2.5 | 0.45× | 397.97 | 484.47 | 16.9% | 0.99991 (100% top-1) | 1.00 |
| Tesseract-Fabric | 11.9 | 2.13× | 84.18 | 4.74 | 47.1% | 0.99804 (99% top-1) | 3.88 |
| Tesseract-Fabric (speculation off, ablation) | 11.6 | 2.09× | 86.05 | 5.56 | 15.9% | 0.99812 (98% top-1) | 1.00 |

† RDMA-Tensor-Parallel: 1/4 shard (151 MB) exceeds RAM of Android Snapdragon Edge — would OOM; run for latency only

Tesseract telemetry: decode 10.77s over 33 epochs; tree budget picks (budget:count) 1:2 4:2 6:5 8:14 16:10; draft-ahead hits/misses 0/32; KV commits 68 (mean 1.40 µs); expert staging — blocking misses 626 (Exo: 1058), pre-staged 30, lookahead recall 94.2%, PreWake frames 845 (29.7 KB out-of-band).

## Scenario 3 — Long-Context Scaling Stress Test (4 shards, TB5 links, 8 layers, GQA 8q/2kv heads)

Shard RAM budgets (MB): [512.0, 256.0, 192.0, 128.0]; layer split (both engines, ∝ RAM): [4, 2, 1, 1]. Context KV is injected with the layers' real K/V projections of the context tokens (single-pass approximate prefill), then 24 tokens are decoded with full forward passes.

| Context | Engine | Throughput (tok/s) | Peak node memory (weights+KV) | Memory balance (max/mean util) | Attention traffic (KB/step) | FP16 page-migration equivalent (KB/step) | LSE merge max abs err vs monolithic |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2K | Layer-coupled ring KV (Exo) | 94.98 | 48 MB | 1.09 | 0.0 | — | — (monolithic per layer) |
| 2K | Tesseract disaggregated Flash-KV | 49.96 | 44 MB | 1.09 | 60.8 | 5376 | 6.05e-5 |
| 8K | Layer-coupled ring KV (Exo) | 34.24 | 77 MB | 1.09 | 0.0 | — | — (monolithic per layer) |
| 8K | Tesseract disaggregated Flash-KV | 36.20 | 67 MB | 1.07 | 59.7 | 22016 | 5.99e-5 |
| 32K | Layer-coupled ring KV (Exo) | 12.47 | 174 MB | 1.09 | 0.0 | — | — (monolithic per layer) |
| 32K | Tesseract disaggregated Flash-KV | 11.65 | 162 MB | 1.02 | 57.6 | 87808 | 4.73e-5 |
| 64K | Layer-coupled ring KV (Exo) | 7.27 | 346 MB | 1.09 | 0.0 | — | — (monolithic per layer) |
| 64K | Tesseract disaggregated Flash-KV | 6.89 | 288 MB | 1.01 | 57.6 | 175616 | 4.94e-5 |

Attention traffic for the disaggregated fabric includes the KV query broadcast, the (O, m, l) partials and the append of each step's newly committed K/V rows to their owning page.

### Drafter agreement with the target (96 sampled tokens, T = 1)

| Benchmark model | Target mean top-1 prob | Target entropy | Zero-layer head top-1 agreement | 2-layer early-exit top-1 agreement | Target mass on early-exit argmax |
|---|---:|---:|---:|---:|---:|
| V1 (unstructured random weights) | 0.601 | 1.43 nats | 42.7% | 41.7% | 0.391 |
| V2 (structured initialisation) | 0.736 | 1.40 nats | 86.5% | 85.4% | 0.713 |

On the V2 model the next token is dominated by a first-order transition prior, so even the zero-layer head agrees with the full model most of the time; this is a deliberately speculation-friendly workload, and the 2-layer self-speculative drafter adds little over it here.

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
| 4K | 12.6 KB | 6.6 KB | 1152 KB | −98.90% | −99.42% | 1.40e-4 |
| 16K | 12.6 KB | 6.6 KB | 4352 KB | −99.71% | −99.85% | 6.48e-5 |
| 64K | 12.6 KB | 6.6 KB | 17280 KB | −99.93% | −99.96% | 5.66e-5 |

### Weight-reuse tiled batch GEMM (single core, min of 300 runs)

| Weight matrix | B=1 | B=4 | B=6 | B=8 | B=16 | cost B=6 / B=1 | cost B=8 / B=1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2048×512 | 15.2 GMAC/s | 38.2 GMAC/s | 39.9 GMAC/s | 43.8 GMAC/s | 43.8 GMAC/s | 2.29× | 2.78× |
| 704×512 | 15.0 GMAC/s | 38.4 GMAC/s | 41.4 GMAC/s | 38.4 GMAC/s | 38.4 GMAC/s | 2.18× | 3.13× |
| 512×704 | 15.3 GMAC/s | 36.1 GMAC/s | 39.1 GMAC/s | 39.4 GMAC/s | 42.2 GMAC/s | 2.35× | 3.12× |

### Lock-free slab KV commit (16 layers, 16 pending rows → 4 accepted)

Median 0.17 µs, p90 0.25 µs, p99 1.25 µs for the whole shard (V1: per-layer copy + compaction, ≈2 µs per layer in isolation, 14–16 µs per commit inside the loaded V1 cluster).

### Multipath striping (512 KB frames, 2.5GbE + Wi-Fi 6E, live sockets)

| Mode | Goodput, median frame (Gbps) | Byte split |
|---|---:|---|
| Best single link | 1.18 | 2.5GbE 100% |
| ETTC water-filling stripes | 1.05 | 2.5GbE 70% / WiFi 30% |
| 2.5GbE brownout (8 ms, 0.1 Gbps) — after telemetry adapts | 0.42 | 2.5GbE 0% / WiFi 100% |

_Total benchmark wall time: 536.8 s._
