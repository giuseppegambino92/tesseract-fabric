# tesseract-bench results

Host: Apple M3 Max · 14 logical CPUs · every node is a set of OS threads with its own TCP endpoints on 127.0.0.1; all traffic is real `TesseractWire` frames through the kernel's TCP stack, delayed by per-link shapers (FIFO serialisation at link bandwidth + one-way latency + jitter). Compute is real FP32 tensor math on this host; slower devices are emulated by extending each node's *measured* compute time by its memory-bandwidth ratio to the fastest node.

Host load average at start: `{ 142,65 131,21 84,44 }` — background load on the host competes with the emulated nodes for cores and adds variance to every timing below.

Model: synthetic Llama/Qwen-style MoE decoder — 16 layers, d_model 512, 8 q / 4 kv heads, 8 experts top-2 (d_expert 704), vocab 2048, 153.2M parameters (random weights; same weights on every engine). 6 of 8 experts per layer fit the hot buffer on every engine; the rest are staged from int8 cold storage.

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
| Exo-Ring-Pipeline | 68.8 | 1.00× | 14.54 | 3.79 | 23.7% | 0.99988 (100% top-1) | 1.00 |
| RDMA-Tensor-Parallel | 78.6 | 1.14× | 12.72 | 456.33 | 37.6% | 0.99993 (99% top-1) | 1.00 |
| Tesseract-Fabric | 40.3 | 0.59× | 24.79 | 2.17 | 24.8% | 0.99997 (99% top-1) | 1.36 |
| Tesseract-Fabric (speculation off, ablation) | 78.2 | 1.14× | 12.79 | 1.22 | 20.4% | 0.99986 (98% top-1) | 1.00 |

Tesseract telemetry: decode 3.17s over 94 epochs; tree budget picks (budget:count) 1:8 2:82 4:2 8:2; draft-ahead hits/misses 30/63; KV commits 190 (mean 14.32 µs); expert staging — blocking misses 312 (Exo: 1082), pre-staged 26, lookahead recall 77.8%, PreWake frames 2353 (44.5 KB out-of-band).

### Activation bit-width frontier (Scenario 1 plan, speculation off)

Exo ships the same hidden state as raw FP16 (2 B/element).

| Activation coding | Inter-stage activation bytes per token | vs FP16 | Cosine fidelity vs FP32 (top-1 agree) |
|---|---:|---:|---|
| FP16 (Exo ring, 2-stage) | 1386 B | 100% | 0.99990 (100% top-1) |
| Holographic 8-bit | 807 B | 58% | 0.99987 (100% top-1) |
| Holographic 6-bit | 649 B | 47% | 0.99939 (94% top-1) |
| Holographic 4-bit | 491 B | 35% | 0.99414 (84% top-1) |
| Holographic 3-bit | 412 B | 30% | 0.98040 (73% top-1) |

## Scenario 2 — Heterogeneous Multi-Platform Cluster (Mac Studio + RTX 5090 on 2.5GbE, MacBook Pro + Android on Wi-Fi 6E)

| Node | Device profile | Platform | Mem BW (GB/s) | Emulated RAM (MB) | Compute slowdown |
|---|---|---|---:|---:|---:|
| 0 | Mac Studio M5 Max | AppleSiliconUMA | 546 | 640 | 3.28× |
| 1 | Linux RTX 5090 Workstation | DiscreteGpuWorkstation | 1792 | 160 | 1.00× |
| 2 | MacBook Pro (M-Pro) | AppleSiliconUMA | 273 | 240 | 6.56× |
| 3 | Android Snapdragon Edge | MobileEdgeNpu | 77 | 80 | 23.27× |

Links: 0↔1: 2.5GbE+WiFi6E, 0↔2: WiFi6E, 0↔3: WiFi6E, 1↔2: WiFi6E, 1↔3: WiFi6E, 2↔3: WiFi6E

* Exo ring partition (layers ∝ RAM): node 0: L0–8 → node 1: L9–10 → node 2: L11–14 → node 3: L15–15
* HeteroClusterSolver plan: node 1: L0–2 → node 0: L3–15 | coordinator = node 0 | drafter = node 3 | est. epoch 0.71 ms
  * Android Snapdragon Edge → asynchronous tree drafter
  * MacBook Pro (M-Pro) → overflow KV page pool

| Engine | Throughput (tok/s) | vs Exo | Per-token latency (ms) | Network per token (KB) | Active HW utilization | Cosine fidelity vs FP32 (top-1 agree) | Tokens / epoch |
|---|---:|---:|---:|---:|---:|---|---:|
| Exo-Ring-Pipeline | 8.8 | 1.00× | 113.87 | 6.65 | 16.3% | 0.99991 (99% top-1) | 1.00 |
| RDMA-Tensor-Parallel † | 1.8 | 0.20× | 557.58 | 480.58 | 7.4% | 0.99993 (99% top-1) | 1.00 |
| Tesseract-Fabric | 21.9 | 2.49× | 45.75 | 4.39 | 27.0% | 0.99980 (98% top-1) | 1.36 |
| Tesseract-Fabric (speculation off, ablation) | 23.9 | 2.72× | 41.91 | 3.49 | 16.5% | 0.99973 (100% top-1) | 1.00 |

† RDMA-Tensor-Parallel: 1/4 shard (151 MB) exceeds RAM of Android Snapdragon Edge — would OOM; run for latency only

Tesseract telemetry: decode 5.86s over 94 epochs; tree budget picks (budget:count) 1:7 2:81 4:2 8:4; draft-ahead hits/misses 14/79; KV commits 189 (mean 16.35 µs); expert staging — blocking misses 381 (Exo: 1064), pre-staged 26, lookahead recall 79.5%, PreWake frames 1796 (66.8 KB out-of-band).

## Scenario 3 — Long-Context Scaling Stress Test (4 shards, TB5 links, 8 layers, GQA 8q/2kv heads)

Shard RAM budgets (MB): [512.0, 256.0, 192.0, 128.0]; layer split (both engines, ∝ RAM): [4, 2, 1, 1]. Context KV is injected with the layers' real K/V projections of the context tokens (single-pass approximate prefill), then 24 tokens are decoded with full forward passes.

| Context | Engine | Throughput (tok/s) | Peak node memory (weights+KV) | Memory balance (max/mean util) | Attention traffic (KB/step) | FP16 page-migration equivalent (KB/step) | LSE merge max abs err vs monolithic |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2K | Layer-coupled ring KV (Exo) | 91.71 | 44 MB | 1.09 | 0.0 | — | — (monolithic per layer) |
| 2K | Tesseract disaggregated Flash-KV | 97.08 | 44 MB | 1.09 | 108.8 | 5376 | 4.47e-8 |
| 8K | Layer-coupled ring KV (Exo) | 33.95 | 69 MB | 1.09 | 0.0 | — | — (monolithic per layer) |
| 8K | Tesseract disaggregated Flash-KV | 40.60 | 67 MB | 1.07 | 107.7 | 22016 | 2.61e-8 |
| 32K | Layer-coupled ring KV (Exo) | 12.84 | 170 MB | 1.09 | 0.0 | — | — (monolithic per layer) |
| 32K | Tesseract disaggregated Flash-KV | 12.99 | 162 MB | 1.02 | 105.5 | 87808 | 2.24e-8 |
| 64K | Layer-coupled ring KV (Exo) | 7.21 | 304 MB | 1.09 | 0.0 | — | — (monolithic per layer) |
| 64K | Tesseract disaggregated Flash-KV | 8.05 | 288 MB | 1.01 | 105.5 | 175616 | 3.73e-8 |

Attention traffic for the disaggregated fabric includes the KV query broadcast, the (O, m, l) partials and the append of each step's newly committed K/V rows to their owning page.

## Pillar micro-benchmarks

### Holographic residual compression (d = 4096, 128 autoregressive steps, synthetic low-rank residual trajectory with massive-activation channels)

| Bits | Error feedback γ | Payload vs FP16 | Cumulative L2 drift | Min cosine | Drift reduction vs stateless |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.95 | −85.8% | 236.1 | 0.99974 | ×5.18 |
| 3 | 0.95 | −79.6% | 107.5 | 0.99995 | ×6.41 |
| 4 | 0.95 | −73.3% | 55.9 | 0.99999 | ×6.90 |

### Cross-silicon determinism (Q8.24 CanonicalBoundaryAnchor)

RMSNorm Σx² over 4096-wide activations: scalar / NEON-4 / AVX2-8 / warp-32 / FMA reduction orders disagree bitwise in 100/100 trials with IEEE f32; the Q8.24 integer accumulation is bit-identical in 100/100.

### Disaggregated attention traffic per decode step (one layer, 8 query heads, 4 shards)

| Context | LSE-merge traffic | FP16 page migration | Reduction | Max abs err |
|---:|---:|---:|---:|---:|
| 4K | 12.3 KB | 1152 KB | −98.94% | 4.47e-8 |
| 16K | 12.3 KB | 4352 KB | −99.72% | 2.98e-8 |
| 64K | 12.3 KB | 17280 KB | −99.93% | 3.73e-8 |

### Multipath striping (512 KB frames, 2.5GbE + Wi-Fi 6E, live sockets)

| Mode | Goodput, median frame (Gbps) | Byte split |
|---|---:|---|
| Best single link | 0.54 | 2.5GbE 100% |
| ETTC water-filling stripes | 1.22 | 2.5GbE 77% / WiFi 23% |
| 2.5GbE brownout (8 ms, 0.1 Gbps) — after telemetry adapts | 0.46 | 2.5GbE 0% / WiFi 100% |

_Total benchmark wall time: 500.6 s._
