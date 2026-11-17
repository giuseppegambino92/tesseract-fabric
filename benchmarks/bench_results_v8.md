# tesseract-bench results

Host: Apple M3 Max · 14 logical CPUs · every node is a set of OS threads with its own TCP endpoints on 127.0.0.1; all traffic is real `TesseractWire` frames through the kernel's TCP stack, delayed by per-link shapers (FIFO serialisation at link bandwidth + one-way latency + jitter). Compute is real FP32 tensor math on this host; slower devices are emulated by extending each node's *measured* compute time by its memory-bandwidth ratio to the fastest node.

Host load average at start: `{ 21,03 47,65 57,26 }` — background load on the host competes with the emulated nodes for cores and adds variance to every timing below.

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

| Engine | Throughput, median of 5 interleaved rounds (tok/s) | Fastest-quarter envelope (tok/s) | min–max | vs Exo | Per-token latency (ms) | Network per token (KB) | Active HW utilization | Cosine fidelity vs FP32 (top-1 agree) | Tokens / epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| Exo-Ring-Pipeline | 112.7 | 114.5 | 110.1–115.9 | 1.00× | 8.87 | 3.79 | 24.1% | 0.99999 (100% top-1) | 1.00 |
| RDMA-Tensor-Parallel | 145.3 | 148.1 | 139.6–149.2 | 1.29× | 6.88 | 456.33 | 42.8% | 0.99999 (100% top-1) | 1.00 |
| Tesseract-Fabric (H-TEP) | 204.8 | 209.8 | 170.0–209.8 | 1.82× | 4.88 | 553.45 | 43.4% | 0.99999 (100% top-1) | 1.91 |
| Tesseract-Fabric (speculation off, ablation) | 142.9 | 159.1 | 80.2–163.9 | 1.27× | 7.00 | 458.83 | 37.6% | 0.99999 (100% top-1) | 1.00 |

Paired per-round ratios (median of the 4 of 5 rounds kept by the symmetric load-spike filter, min–max): spec-on / spec-off **1.36×** (1.26–1.42), spec-off / TP **1.10×** (0.92–1.11), TP / Exo **1.32×** (1.24–1.33).

Fastest-quarter envelope ratios (mean of each engine's fastest ⌈5/4⌉ rounds): spec-on / spec-off **1.32×**, spec-off / TP **1.07×**, TP / Exo **1.29×**.

Tesseract telemetry: decode 0.62s over 67 epochs; tree budget picks (budget:count) 2:54 3:2 4:8 6:3; draft-ahead hits/misses 0/0; KV commits 272 (mean 0.65 µs); expert staging — blocking misses 979 (Exo: 973), pre-staged 354, lookahead recall 78.7%, PreWake frames 3536 (0.0 KB out-of-band); stage-0 cross-step lookahead rows reused / discarded / aborted early 0 / 0 / 0.

#### Sampled decoding (T = 1) ablation, same cluster, median of 3 interleaved rounds

| Engine | Throughput (tok/s) | min–max | Tokens / epoch | Cosine fidelity vs FP32 (top-1 agree) |
|---|---:|---:|---:|---|
| RDMA-Tensor-Parallel | 85.2 | 57.7–143.8 | 1.00 | 1.00000 (100% top-1) |
| Tesseract-Fabric, speculation off | 148.2 | 127.3–152.7 | 1.00 | 1.00000 (100% top-1) |
| Tesseract-Fabric, speculation on | 150.1 | 85.6–159.1 | 1.49 | 1.00000 (99% top-1) |

#### 4-token verification in the deployed cluster (H-TEP, this hardware, median of 15 interleaved rounds)

Every rank holds a quarter of every expert's hidden units, so a 4-row tree still touches ≈ 5.4 expert slices per rank; the per-layer collectives and the replicated attention are paid once per epoch for any tree size. Verification is exact. Targets: the original ≤ 1.35× assumed dense-model cost; exact top-2 routing makes a 4-row tree read ≈ 2.7× the expert weights of one row, and V7.0 revised the target to ≤ 1.65× *after* measuring ≈ 1.6× (both thresholds are reported).

| Epoch | Decode time per epoch | Tokens per epoch |
|---|---:|---:|
| 1-row step | 6.26 ms | 1.00 |
| 4-row tree | 9.85 ms | 1.91 |
| 4-row / 1-row, paired per-round median | 1.59× (rounds 0.55–1.72) | |
| **4-row / 1-row, lower envelope** (mean of the fastest quartile of each side: 5.91 vs 9.55 ms) | **1.62×** | |

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

| Engine | Throughput, median of 5 interleaved rounds (tok/s) | Fastest-quarter envelope (tok/s) | min–max | vs Exo | Per-token latency (ms) | Network per token (KB) | Active HW utilization | Cosine fidelity vs FP32 (top-1 agree) | Tokens / epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| Exo-Ring-Pipeline | 6.7 | 8.2 | 6.5–9.0 | 1.00× | 149.63 | 4.48 | 20.2% | 0.99999 (100% top-1) | 1.00 |
| RDMA-Tensor-Parallel † | 6.1 | 8.4 | 4.2–9.0 | 0.91× | 164.38 | 466.97 | 25.7% | 0.99999 (100% top-1) | 1.00 |
| Tesseract-Fabric | 26.4 | 27.0 | 20.3–27.0 | 3.94× | 37.93 | 4.12 | 22.6% | 0.99999 (100% top-1) | 1.71 |
| Tesseract-Fabric (speculation off, ablation) | 18.3 | 19.4 | 15.7–19.4 | 2.73× | 54.74 | 3.34 | 22.6% | 0.99999 (100% top-1) | 1.00 |

Paired per-round ratios (median of the 5 of 5 rounds kept by the symmetric load-spike filter, min–max): spec-on / spec-off **1.39×** (1.29–1.48), spec-off / TP **3.00×** (2.01–4.60), TP / Exo **0.94×** (0.63–1.04).

Fastest-quarter envelope ratios (mean of each engine's fastest ⌈5/4⌉ rounds): spec-on / spec-off **1.39×**, spec-off / TP **2.31×**, TP / Exo **1.02×**.

† RDMA-Tensor-Parallel: 1/4 shard (151 MB) exceeds RAM of Android Snapdragon Edge — would OOM; run for latency only

Tesseract telemetry: decode 4.86s over 75 epochs; tree budget picks (budget:count) 1:23 2:23 3:9 4:3 6:12 8:3 16:2; draft-ahead hits/misses 0/51; KV commits 152 (mean 1.13 µs); expert staging — blocking misses 303 (Exo: 971), pre-staged 46, lookahead recall 81.0%, PreWake frames 871 (34.0 KB out-of-band); stage-0 cross-step lookahead rows reused / discarded / aborted early 19 / 0 / 3.

## Scenario 3 — Long-Context Scaling Stress Test (4 shards, TB5 links, 8 layers, GQA 8q/2kv heads)

Shard RAM budgets (MB): [512.0, 256.0, 192.0, 128.0]; layer split (both engines, ∝ RAM): [4, 2, 1, 1]. Context KV is injected with the layers' real K/V projections of the context tokens (single-pass approximate prefill), then 48 tokens are decoded with full forward passes.

KV remote-holder bias (calibrated once at 8K from lower envelopes): minimum remote exchange overhead 60 µs vs minimum 12 µs to scan one local page → 5 pages; capped to 4 so the placement's projected attention traffic at 8K / 32K / 64K (32.1 / 32.1 / 32.4 KB/step, dry-run page layout with real frame sizes) stays under the 34 KB/step wire budget. Every remote holder of a virtual layer is kept 4 pages behind the computing node.

| Context | Engine | Throughput, median of 7 interleaved rounds (tok/s) | min–max | vs ring: fastest-quarter envelope | vs ring: median of paired per-round ratios, load-spike filtered (min–max) | Peak node memory (weights+KV) | Memory balance (max/mean util) | Attention traffic (KB/step) | FP16 page-migration equivalent (KB/step) | LSE merge max abs err vs monolithic (live BFP wire + head-group striping) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2K | Layer-coupled ring KV (Exo) | 271.56 | 258.13–274.86 | **1.00×** | 1.00× (1.00–1.00) | 48 MB | 1.09 | 0.0 | — | — (monolithic per layer) |
| 2K | Tesseract disaggregated Flash-KV (speculation off) | 252.31 | 236.23–256.76 | **0.94×** | 0.93× (0.89–0.99) | 44 MB | 1.09 | 0.0 | 5376 | 4.47e-7 |
| 2K | Tesseract full engine (Flash-KV + speculation) | 282.18 | 129.31–285.62 | **1.04×** | 1.04× (0.98–1.09) | 44 MB | 1.09 | 0.0 | 5376 | 4.47e-7 |
| 8K | Layer-coupled ring KV (Exo) | 121.57 | 121.23–122.77 | **1.00×** | 1.00× (1.00–1.00) | 77 MB | 1.09 | 0.0 | — | — (monolithic per layer) |
| 8K | Tesseract disaggregated Flash-KV (speculation off) | 199.43 | 195.80–202.98 | **1.66×** | 1.64× (1.59–1.67) | 54 MB | 1.42 | 32.1 | 22016 | 1.11e-5 |
| 8K | Tesseract full engine (Flash-KV + speculation) | 201.24 | 199.62–202.07 | **1.65×** | 1.66× (1.63–1.67) | 54 MB | 1.42 | 32.1 | 22016 | 1.11e-5 |
| 32K | Layer-coupled ring KV (Exo) | 37.84 | 37.25–37.91 | **1.00×** | 1.00× (1.00–1.00) | 174 MB | 1.09 | 0.0 | — | — (monolithic per layer) |
| 32K | Tesseract disaggregated Flash-KV (speculation off) | 98.74 | 96.45–99.67 | **2.62×** | 2.61× (2.55–2.63) | 104 MB | 1.57 | 32.1 | 87808 | 1.14e-5 |
| 32K | Tesseract full engine (Flash-KV + speculation) | 98.40 | 97.52–99.66 | **2.62×** | 2.62× (2.57–2.63) | 104 MB | 1.57 | 32.1 | 87808 | 1.14e-5 |
| 64K | Layer-coupled ring KV (Exo) | 19.55 | 11.67–19.82 | **1.00×** | 1.00× (1.00–1.00) | 346 MB | 1.09 | 0.0 | — | — (monolithic per layer) |
| 64K | Tesseract disaggregated Flash-KV (speculation off) | 53.41 | 45.59–53.61 | **2.71×** | 2.73× (2.69–2.76) | 178 MB | 1.44 | 32.4 | 175616 | 1.34e-5 |
| 64K | Tesseract full engine (Flash-KV + speculation) | 53.38 | 53.22–53.76 | **2.71×** | 2.75× (2.71–2.76) | 178 MB | 1.44 | 32.4 | 175616 | 1.34e-5 |

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

V1–V4 stripe pages over all 4 shards (3 remote KV services per layer); V5 pairs the layer with one partner (tier 0 local, later pages split evenly between the two), so a step contacts 1 remote service; V6.2 stripes by KV-head group (each KV-head group of a layer is its own virtual layer: group 0 stays with the computing node plus remote help, the others live on remote nodes, and each holder receives only its group's query heads). The BFP columns use the current wire (16-bit block-floating-point queries, weight-bound-coded partials); V4 shipped 14-bit queries (6.3 KB, 6.3e-5 at 4K).

| Context | V1 f32 wire | V3 f16 wire | BFP wire, striped over 4 shards | BFP wire, paired (V5) | **BFP wire, head-group striped (V6.2 default)** | FP16 page migration | V3 reduction | V6.2 reduction | V3 max abs err | striped BFP max abs err | V5 paired max abs err | **V6.2 max abs err** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4K | 12.7 KB | 6.7 KB | 6.7 KB | 2.2 KB | 3.4 KB | 1152 KB | −99.42% | −99.70% | 1.40e-4 | 4.05e-5 | 5.86e-5 | 5.54e-5 |
| 16K | 12.7 KB | 6.7 KB | 6.7 KB | 2.3 KB | 3.7 KB | 4352 KB | −99.85% | −99.92% | 6.49e-5 | 2.06e-5 | 1.00e-5 | 1.31e-5 |
| 64K | 12.7 KB | 6.7 KB | 6.7 KB | 2.3 KB | 3.7 KB | 17280 KB | −99.96% | −99.98% | 5.67e-5 | 2.01e-5 | 7.72e-6 | 1.34e-5 |

### Weight-reuse tiled batch GEMM (single core, min of 300 runs)

| Weight matrix | B=1 | B=4 | B=6 | B=8 | B=16 | cost B=6 / B=1 | cost B=8 / B=1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2048×512 | 22.3 GMAC/s | 49.6 GMAC/s | 55.2 GMAC/s | 53.8 GMAC/s | 54.6 GMAC/s | 2.42× | 3.32× |
| 704×512 | 22.2 GMAC/s | 53.2 GMAC/s | 55.4 GMAC/s | 54.0 GMAC/s | 54.9 GMAC/s | 2.40× | 3.28× |
| 512×704 | 22.2 GMAC/s | 53.9 GMAC/s | 57.5 GMAC/s | 57.0 GMAC/s | 57.3 GMAC/s | 2.31× | 3.11× |

### Intra-node lane pool: one MoE layer, exact routing, 1-row step vs 4-row tree (min of 60 trials, interleaved)

Both batch sizes use the same lanes and the same striping rule (every active expert split into ⌈lanes / active experts⌉ hidden-unit and output stripes), so a 1-row step's 2 experts are parallelised as finely as the tree's 5–6.

| Lanes | 1 row | 4 rows | 4 rows / 1 row | max |Δ| vs sequential |
|---:|---:|---:|---:|---:|
| 1 | 125.9 µs | 349.2 µs | 2.77× | 0.0e0 |
| 2 | 83.5 µs | 198.4 µs | 2.38× | 0.0e0 |
| 4 | 54.7 µs | 178.8 µs | 3.27× | 0.0e0 |
| 8 | 48.6 µs | 93.5 µs | 1.92× | 0.0e0 |

### Anchor-consensus expert coalescing (APPROXIMATE; one MoE layer, 4-row tree, min of 40 trials)

| | Distinct experts touched | Time | vs 1 row |
|---|---:|---:|---:|
| 1 row (anchor only) | 2 | 135.5 µs | 1.00× |
| 4 rows, exact routing | 5.4 (mean) | 343.6 µs | 2.54× |
| 4 rows, coalesced (anchor's experts + 1 consensus) | ≤ 3 | 248.6 µs | 1.83× |

Coalescing changes the candidate rows' FFN outputs by 70.7% (relative L2, mean over candidate rows), so trees verified with it are not distributed exactly as the target model; it is therefore opt-in (`coalesce_experts`) and off in every scenario run.

### Lock-free slab KV commit (16 layers, 16 pending rows → 4 accepted)

Median 0.12 µs, p90 0.12 µs, p99 0.58 µs for the whole shard (V1: per-layer copy + compaction, ≈2 µs per layer in isolation, 14–16 µs per commit inside the loaded V1 cluster).

### Multipath striping (512 KB frames, 2.5GbE + Wi-Fi 6E, live sockets)

| Mode | Goodput, median frame (Gbps) | Byte split |
|---|---:|---|
| Best single link | 1.46 | 2.5GbE 100% |
| ETTC water-filling stripes | 1.66 | 2.5GbE 84% / WiFi 16% |
| 2.5GbE brownout (8 ms, 0.1 Gbps) — after telemetry adapts | 0.45 | 2.5GbE 0% / WiFi 100% |

## Section 4 — V8.0 production runtime: GGUF files over `mmap`, quantized kernels, native acceleration

The Scenario 1–3 model exported to real GGUF v3 files (llama.cpp tensor names; Mixtral-style stacked experts; norms and routers F32; in the Q4_0 file `output.weight` is Q8_0, as llama.cpp keeps it above Q4_0) and loaded back through `mmap`. The file was just written, so its pages are in the OS page cache: load times are *warm-cache* times (a disk-cold load adds the read of the pages a node touches). Integer-dot kernel: NEON (sdot). `portable` = the in-tree kernels used by every engine comparison above; `accelerate` = Apple Accelerate `cblas_sgemm` (AMX) for dense FP32 GEMMs and for dequantized tiles of batches ≥ 16 rows (its internal threads are not part of the equal-lanes rule, so it is reported here only).

#### Footprint, load and fidelity

| Format | GGUF file (MB) | FP32 bytes / file bytes | `mmap` + header parse (ms) | heap read of the whole file, for contrast (ms) | all layer views + head ready (ms) | weight bytes addressed by the full model (MB) | teacher-forced fidelity vs in-memory FP32 (cosine, top-1 of 64 positions) |
|---|---:|---:|---:|---:|---:|---:|---:|
| F32 | 612.8 | 1.00× | 2.46 | 52.6 | 77.3 | 608.5 | 1.00000 (100.0%) |
| F16 | 306.6 | 2.00× | 1.06 | 26.4 | 5.2 | 304.4 | 0.99999 (100.0%) |
| Q8_0 | 163.1 | 3.76× | 1.23 | 13.9 | 4.0 | 161.9 | 0.99985 (96.9%) |
| Q4_0 | 87.0 | 7.04× | 0.99 | 7.7 | 3.5 | 86.4 | 0.97145 (85.9%) |

F32 files are copied into the engine's dense tiled layout at load (that is the "views ready" time); F16 / Q8_0 / Q4_0 matrices stay zero-copy views into the mapping, and only the embedding table (decoded to f32), norms, routers and the re-ordered Q/K rows are materialised.

#### Throughput (greedy, 16-token prompt)

| Weights | single node, portable: prefill tok/s (128 tokens) / decode tok/s / first token | 4-node H-TEP decode tok/s, portable (median of 3) | single node, Accelerate: prefill / decode / first token | 4-node H-TEP decode tok/s, Accelerate (median of 3) |
|---|---:|---:|---:|---:|
| in-memory FP32 (synthetic, V7 path) | 1272 / 205.6 / 2431 ms | 204.0 | 1442 / 225.0 / 2389 ms | 203.3 |
| GGUF F32 | 1282 / 207.4 / 90 ms | 221.7 | 1418 / 228.8 / 101 ms | 240.1 |
| GGUF F16 | 491 / 166.1 / 45 ms | 199.7 | 673 / 163.5 / 46 ms | 194.4 |
| GGUF Q8_0 | 948 / 516.2 / 20 ms | 254.3 | 1070 / 499.5 / 21 ms | 265.8 |
| GGUF Q4_0 | 898 / 483.8 / 21 ms | 231.7 | 1033 / 482.2 / 21 ms | 229.0 |

Single node = the whole model on one node thread with the same 2-lane pool every engine node installs. 4-node H-TEP = Scenario 1's cluster (TB5-shaped loopback links, speculation on), loading its weights from the file. Fidelity and kernels are deterministic; throughput shares the host with its background load.

_Total benchmark wall time: 1234.8 s._
