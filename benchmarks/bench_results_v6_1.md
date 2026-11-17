# tesseract-bench results

Host: Apple M3 Max · 14 logical CPUs · every node is a set of OS threads with its own TCP endpoints on 127.0.0.1; all traffic is real `TesseractWire` frames through the kernel's TCP stack, delayed by per-link shapers (FIFO serialisation at link bandwidth + one-way latency + jitter). Compute is real FP32 tensor math on this host; slower devices are emulated by extending each node's *measured* compute time by its memory-bandwidth ratio to the fastest node.

Host load average at start: `{ 100,22 134,70 139,04 }` — background load on the host competes with the emulated nodes for cores and adds variance to every timing below.

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
| Exo-Ring-Pipeline | 16.7 | 12.8–20.4 | 1.00× | 60.04 | 3.79 | 22.2% | 0.99999 (100% top-1) | 1.00 |
| RDMA-Tensor-Parallel | 19.8 | 13.4–22.5 | 1.19× | 50.61 | 456.33 | 21.6% | 0.99999 (100% top-1) | 1.00 |
| Tesseract-Fabric (H-TEP) | 23.0 | 18.5–39.1 | 1.38× | 43.48 | 619.82 | 22.4% | 0.99999 (100% top-1) | 1.88 |
| Tesseract-Fabric (speculation off, ablation) | 16.4 | 11.4–106.2 | 0.98× | 60.97 | 458.83 | 16.3% | 0.99999 (100% top-1) | 1.00 |

Tesseract telemetry: decode 5.56s over 68 epochs; tree budget picks (budget:count) 2:41 3:18 4:2 6:7; draft-ahead hits/misses 0/0; KV commits 276 (mean 2.00 µs); expert staging — blocking misses 1027 (Exo: 973), pre-staged 170, lookahead recall 79.9%, PreWake frames 3588 (0.0 KB out-of-band).

#### Sampled decoding (T = 1) ablation, same cluster, median of 3 interleaved rounds

| Engine | Throughput (tok/s) | min–max | Tokens / epoch | Cosine fidelity vs FP32 (top-1 agree) |
|---|---:|---:|---:|---|
| RDMA-Tensor-Parallel | 56.5 | 24.9–82.0 | 1.00 | 1.00000 (100% top-1) |
| Tesseract-Fabric, speculation off | 66.1 | 8.2–74.9 | 1.00 | 1.00000 (100% top-1) |
| Tesseract-Fabric, speculation on | 102.3 | 14.4–139.8 | 1.27 | 1.00000 (100% top-1) |

#### 4-token verification in the deployed cluster (H-TEP, this hardware, median of 5 interleaved rounds)

Every rank holds a quarter of every expert's hidden units, so a 4-row tree still touches ≈ 5.4 expert slices per rank; the per-layer collectives and the replicated attention are paid once per epoch for any tree size. Verification is exact.

| Epoch | Decode time per epoch | Tokens per epoch |
|---|---:|---:|
| 1-row step | 8.80 ms | 1.00 |
| 4-row tree | 12.52 ms | 1.91 |
| **4-row / 1-row** | **1.43×** (paired rounds 1.03–1.84) | |

### Activation codec frontier (Scenario 1 hardware, fixed 8+8-layer 2-stage pipeline, speculation off, 3 prompts × 128 tokens, temperature 0)

Bytes are whole inter-stage frames (64-byte header + strand metadata + token ids + payload). Exo ships the hidden state as raw FP16 (2 B/element). Fidelity aggregates every generated position of all prompts, so a 0.1-point top-1 difference is about one token.

| Activation coding | Inter-stage bytes per token | vs FP16 frame | Codec payload per row vs FP16 | Cosine fidelity vs FP32 | Top-1 agreement |
|---|---:|---:|---:|---:|---:|
| FP16 (Exo ring, same 2-stage split) | 1256 B | 100% | 100.0% | 0.99999 | 99.74% (383/384) |
| V1 holographic 8-bit | 738 B | 59% | 52.8% | 0.99997 | 98.70% (379/384) |
| V1 holographic 4-bit | 452 B | 36% | 27.8% | 0.99770 | 92.71% (356/384) |
| V2 HRQ trunk-residual 3-bit + sketch | 413 B | 33% | 24.2% | 0.99793 | 92.71% (356/384) |
| V2 HRQ trunk-residual 2-bit + sketch | 341 B | 27% | 18.0% | 0.99263 | 88.02% (338/384) |
| V2.1 entropy-coded trunk residual, ≤25% cap | 411 B | 33% | 25.0% | 0.99940 | 95.57% (367/384) |
| **V5 conditional-KLT transform coding, ≤25% cap (default)** | 422 B | 34% | 25.0% | 0.99997 | 99.48% (382/384) |

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
| Exo-Ring-Pipeline | 9.2 | 5.0–11.3 | 1.00× | 109.12 | 4.50 | 21.0% | 0.99999 (100% top-1) | 1.00 |
| RDMA-Tensor-Parallel † | 4.4 | 1.8–8.3 | 0.48× | 229.56 | 470.72 | 21.7% | 0.99999 (100% top-1) | 1.00 |
| Tesseract-Fabric | 18.1 | 15.2–26.4 | 1.97× | 55.25 | 4.94 | 23.8% | 0.99999 (100% top-1) | 2.17 |
| Tesseract-Fabric (speculation off, ablation) | 13.4 | 4.4–27.5 | 1.46× | 74.69 | 5.42 | 20.7% | 0.99999 (100% top-1) | 1.00 |

† RDMA-Tensor-Parallel: 1/4 shard (151 MB) exceeds RAM of Android Snapdragon Edge — would OOM; run for latency only

Tesseract telemetry: decode 7.07s over 59 epochs; tree budget picks (budget:count) 1:10 2:6 3:3 4:9 6:6 8:22 16:3; draft-ahead hits/misses 0/48; KV commits 120 (mean 1.42 µs); expert staging — blocking misses 501 (Exo: 971), pre-staged 46, lookahead recall 86.8%, PreWake frames 780 (31.0 KB out-of-band).

## Scenario 3 — Long-Context Scaling Stress Test (4 shards, TB5 links, 8 layers, GQA 8q/2kv heads)

Shard RAM budgets (MB): [512.0, 256.0, 192.0, 128.0]; layer split (both engines, ∝ RAM): [4, 2, 1, 1]. Context KV is injected with the layers' real K/V projections of the context tokens (single-pass approximate prefill), then 48 tokens are decoded with full forward passes.

Paired KV page bias (calibrated once, 8K, even split): remote exchange overhead 1371 µs vs 56 µs to scan one local page → the partner holds 24 pages fewer than the computing node.

| Context | Engine | Throughput, median of 7 interleaved rounds (tok/s) | min–max | vs ring: median of 7 paired per-round ratios (min–max) | Peak node memory (weights+KV) | Memory balance (max/mean util) | Attention traffic (KB/step) | FP16 page-migration equivalent (KB/step) | LSE merge max abs err vs monolithic (live BFP wire + paired placement) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2K | Layer-coupled ring KV (Exo) | 62.92 | 31.07–88.15 | **1.00×** (1.00–1.00) | 48 MB | 1.09 | 0.0 | — | — (monolithic per layer) |
| 2K | Tesseract disaggregated Flash-KV (speculation off) | 63.71 | 13.50–87.65 | **1.24×** (0.21–1.71) | 44 MB | 1.09 | 0.0 | 5376 | 4.47e-7 |
| 2K | Tesseract full engine (Flash-KV + speculation) | 88.39 | 24.73–112.00 | **1.26×** (0.31–3.60) | 44 MB | 1.09 | 0.0 | 5376 | 4.47e-7 |
| 8K | Layer-coupled ring KV (Exo) | 32.44 | 5.79–45.73 | **1.00×** (1.00–1.00) | 77 MB | 1.09 | 0.0 | — | — (monolithic per layer) |
| 8K | Tesseract disaggregated Flash-KV (speculation off) | 27.34 | 4.71–45.19 | **0.90×** (0.13–3.34) | 70 MB | 1.10 | 23.1 | 22016 | 1.67e-5 |
| 8K | Tesseract full engine (Flash-KV + speculation) | 35.90 | 13.78–50.27 | **1.22×** (0.39–4.01) | 70 MB | 1.10 | 43.5 | 22016 | 1.67e-5 |
| 32K | Layer-coupled ring KV (Exo) | 8.23 | 4.66–9.92 | **1.00×** (1.00–1.00) | 174 MB | 1.09 | 0.0 | — | — (monolithic per layer) |
| 32K | Tesseract disaggregated Flash-KV (speculation off) | 9.97 | 7.27–20.60 | **1.84×** (0.86–2.40) | 170 MB | 1.12 | 23.1 | 87808 | 5.60e-6 |
| 32K | Tesseract full engine (Flash-KV + speculation) | 10.44 | 6.93–16.07 | **1.24×** (0.70–2.80) | 170 MB | 1.12 | 28.1 | 87808 | 5.60e-6 |
| 64K | Layer-coupled ring KV (Exo) | 4.83 | 3.45–6.64 | **1.00×** (1.00–1.00) | 346 MB | 1.09 | 0.0 | — | — (monolithic per layer) |
| 64K | Tesseract disaggregated Flash-KV (speculation off) | 8.04 | 3.78–10.03 | **1.54×** (0.77–2.61) | 304 MB | 1.13 | 23.1 | 175616 | 6.65e-6 |
| 64K | Tesseract full engine (Flash-KV + speculation) | 5.41 | 2.22–8.62 | **1.32×** (0.45–1.57) | 304 MB | 1.13 | 23.9 | 175616 | 6.65e-6 |

Attention traffic for the disaggregated fabric includes the KV query broadcast, the (O, m, l) partials and the append of each step's newly committed K/V rows to their owning page.

### Drafter agreement with the target (256 sampled tokens, T = 1)

| Benchmark model | Target mean top-1 prob | Target entropy | Zero-layer head top-1 agreement | 1 attention-only layer, full LM head (V4) | **1 attention-only layer + shortlist head (V5 default drafter)** | V5 drafter cost per node vs one target step (weight MACs) | 2-layer early-exit top-1 agreement | Target mass on 2-layer argmax |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V1 (unstructured random weights) | 0.591 | 1.45 nats | 48.8% | 47.7% | 36.7% | 2.24% (1.08M / 48.3M) | 51.2% | 0.409 |
| V2 (first-order transition prior) | 0.730 | 1.38 nats | 89.5% | 90.6% | 89.5% | 2.24% (1.08M / 48.3M) | 90.2% | 0.714 |
| V3 (contextual: current + previous token, prev. via layer-0 attention) | 0.585 | 1.69 nats | 13.3% | 82.0% | 81.2% | 2.24% (1.08M / 48.3M) | 85.9% | 0.569 |

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

V1–V4 stripe pages over all 4 shards (3 remote KV services per layer); V5 pairs the layer with one partner (tier 0 local, later pages split evenly between the two), so a step contacts 1 remote service. The two BFP columns use the current wire (16-bit block-floating-point queries, weight-bound-coded partials); V4 shipped 14-bit queries (6.3 KB, 6.3e-5 at 4K).

| Context | V1 f32 wire | V3 f16 wire | BFP wire, striped over 4 shards | **BFP wire, paired (V5 default)** | FP16 page migration | V3 reduction | V5 reduction | V3 max abs err | striped BFP max abs err | **V5 max abs err** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4K | 12.7 KB | 6.7 KB | 6.7 KB | 2.2 KB | 1152 KB | −99.42% | −99.81% | 1.40e-4 | 4.05e-5 | 5.86e-5 |
| 16K | 12.7 KB | 6.7 KB | 6.7 KB | 2.3 KB | 4352 KB | −99.85% | −99.95% | 6.49e-5 | 2.06e-5 | 1.00e-5 |
| 64K | 12.7 KB | 6.7 KB | 6.7 KB | 2.3 KB | 17280 KB | −99.96% | −99.99% | 5.67e-5 | 2.01e-5 | 7.72e-6 |

### Weight-reuse tiled batch GEMM (single core, min of 300 runs)

| Weight matrix | B=1 | B=4 | B=6 | B=8 | B=16 | cost B=6 / B=1 | cost B=8 / B=1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2048×512 | 12.7 GMAC/s | 29.8 GMAC/s | 32.0 GMAC/s | 31.8 GMAC/s | 31.8 GMAC/s | 2.38× | 3.19× |
| 704×512 | 12.6 GMAC/s | 30.3 GMAC/s | 32.2 GMAC/s | 32.1 GMAC/s | 32.0 GMAC/s | 2.34× | 3.13× |
| 512×704 | 12.7 GMAC/s | 30.5 GMAC/s | 32.6 GMAC/s | 32.8 GMAC/s | 32.8 GMAC/s | 2.33× | 3.10× |

### Intra-node lane pool: one MoE layer, exact routing, 1-row step vs 4-row tree (min of 60 trials, interleaved)

Both batch sizes use the same lanes and the same striping rule (every active expert split into ⌈lanes / active experts⌉ hidden-unit and output stripes), so a 1-row step's 2 experts are parallelised as finely as the tree's 5–6.

| Lanes | 1 row | 4 rows | 4 rows / 1 row | max |Δ| vs sequential |
|---:|---:|---:|---:|---:|
| 1 | 224.4 µs | 838.6 µs | 3.74× | 0.0e0 |
| 2 | 196.9 µs | 519.9 µs | 2.64× | 0.0e0 |
| 4 | 167.2 µs | 364.9 µs | 2.18× | 0.0e0 |
| 8 | 129.4 µs | 285.6 µs | 2.21× | 0.0e0 |

### Anchor-consensus expert coalescing (APPROXIMATE; one MoE layer, 4-row tree, min of 40 trials)

| | Distinct experts touched | Time | vs 1 row |
|---|---:|---:|---:|
| 1 row (anchor only) | 2 | 330.9 µs | 1.00× |
| 4 rows, exact routing | 5.4 (mean) | 648.5 µs | 1.96× |
| 4 rows, coalesced (anchor's experts + 1 consensus) | ≤ 3 | 460.5 µs | 1.39× |

Coalescing changes the candidate rows' FFN outputs by 70.7% (relative L2, mean over candidate rows), so trees verified with it are not distributed exactly as the target model; it is therefore opt-in (`coalesce_experts`) and off in every scenario run.

### Lock-free slab KV commit (16 layers, 16 pending rows → 4 accepted)

Median 0.21 µs, p90 0.83 µs, p99 2.92 µs for the whole shard (V1: per-layer copy + compaction, ≈2 µs per layer in isolation, 14–16 µs per commit inside the loaded V1 cluster).

### Multipath striping (512 KB frames, 2.5GbE + Wi-Fi 6E, live sockets)

| Mode | Goodput, median frame (Gbps) | Byte split |
|---|---:|---|
| Best single link | 0.41 | 2.5GbE 100% |
| ETTC water-filling stripes | 0.41 | 2.5GbE 4% / WiFi 96% |
| 2.5GbE brownout (8 ms, 0.1 Gbps) — after telemetry adapts | 0.41 | 2.5GbE 0% / WiFi 100% |

_Total benchmark wall time: 4116.9 s._
