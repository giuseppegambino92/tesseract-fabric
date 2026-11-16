# tesseract-bench results

Host: Apple M3 Max · 14 logical CPUs · every node is a set of OS threads with its own TCP endpoints on 127.0.0.1; all traffic is real `TesseractWire` frames through the kernel's TCP stack, delayed by per-link shapers (FIFO serialisation at link bandwidth + one-way latency + jitter). Compute is real FP32 tensor math on this host; slower devices are emulated by extending each node's *measured* compute time by its memory-bandwidth ratio to the fastest node.

Host load average at start: `{ 4,94 6,13 10,81 }` — background load on the host competes with the emulated nodes for cores and adds variance to every timing below.

Model: synthetic Llama/Qwen-style MoE decoder — 16 layers, d_model 512, 8 q / 4 kv heads, 8 experts top-2 (d_expert 704), vocab 2048, 153.2M parameters, same weights on every engine. V3 uses a *contextual* initialisation: the next token depends equally on the current token (+8 nats transition prior) and the previous token (+8 nats), and the previous token reaches the current position only through a layer-0 attention head built with RoPE to attend to offset −1; power-law spectral decay σᵢ ∝ i^-0.5 in layer weights; logit noise σ=1.5. It is calibrated to V1's peakedness (mean top-1 ≈ 0.6). V2 used a first-order prior only; V1 unstructured random weights. 6 of 8 experts per layer fit the hot buffer on every engine; the rest are staged from int8 cold storage.

Generation: single stream, 16-token prompt, 128 new tokens, temperature 0 (greedy decoding: every engine must emit exactly the FP32 reference's argmax sequence up to its own numerical error; a T = 1 sampled ablation follows Scenario 1). Network bytes count inter-host frames only (incl. prompt prefill). Scenarios 1–2 run all engines back to back in each of N interleaved rounds (identical seeds) and report each engine's median round, so transient host load hits every engine of a round alike.

_Scenario 1: host 1-minute load average 4.58 at start (gate: ≤ 3.5, waited 180 s; **still above the gate — timings below are load-affected**)._

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
| Exo-Ring-Pipeline | 115.1 | 116.7 | 113.6–117.9 | 1.00× | 8.69 | 3.79 | 24.1% | 0.99999 (100% top-1) | 1.00 |
| RDMA-Tensor-Parallel | 152.4 | 154.5 | 149.3–155.2 | 1.32× | 6.56 | 456.33 | 45.5% | 0.99999 (100% top-1) | 1.00 |
| Tesseract-Fabric (H-TEP) | 213.9 | 227.5 | 211.2–229.8 | 1.86× | 4.67 | 805.12 | 49.8% | 0.99999 (100% top-1) | 2.03 |
| Tesseract-Fabric (speculation off, ablation) | 166.9 | 168.6 | 166.6–170.2 | 1.45× | 5.99 | 458.83 | 40.6% | 0.99999 (100% top-1) | 1.00 |

Paired per-round ratios (median of the 5 of 5 rounds kept by the symmetric load-spike filter, min–max): spec-on / spec-off **1.27×** (1.26–1.38), spec-off / TP **1.09×** (1.08–1.14), TP / Exo **1.32×** (1.29–1.35).

Fastest-quarter envelope ratios (mean of each engine's fastest ⌈5/4⌉ rounds): spec-on / spec-off **1.35×**, spec-off / TP **1.09×**, TP / Exo **1.32×**.

Tesseract telemetry: decode 0.60s over 63 epochs; tree budget picks (budget:count) 2:12 3:2 4:38 6:11; draft-ahead hits/misses 0/0; KV commits 256 (mean 0.61 µs); expert staging — blocking misses 1355 (Exo: 973), pre-staged 182, lookahead recall 83.9%, PreWake frames 3328 (0.0 KB out-of-band); stage-0 cross-step lookahead rows reused / discarded / aborted early 0 / 0 / 0.

#### Sampled decoding (T = 1) ablation, same cluster, median of 3 interleaved rounds

| Engine | Throughput (tok/s) | min–max | Tokens / epoch | Cosine fidelity vs FP32 (top-1 agree) |
|---|---:|---:|---:|---|
| RDMA-Tensor-Parallel | 150.4 | 149.3–154.0 | 1.00 | 1.00000 (100% top-1) |
| Tesseract-Fabric, speculation off | 165.7 | 165.0–167.0 | 1.00 | 1.00000 (100% top-1) |
| Tesseract-Fabric, speculation on | 186.1 | 175.4–187.6 | 1.62 | 1.00000 (99% top-1) |

#### 4-token verification in the deployed cluster (H-TEP, this hardware, median of 15 interleaved rounds)

Every rank holds a quarter of every expert's hidden units, so a 4-row tree still touches ≈ 5.4 expert slices per rank; the per-layer collectives and the replicated attention are paid once per epoch for any tree size. Verification is exact. Targets: the original ≤ 1.35× assumed dense-model cost; exact top-2 routing makes a 4-row tree read ≈ 2.7× the expert weights of one row, and V7.0 revised the target to ≤ 1.65× *after* measuring ≈ 1.6× (both thresholds are reported).

| Epoch | Decode time per epoch | Tokens per epoch |
|---|---:|---:|
| 1-row step | 5.98 ms | 1.00 |
| 4-row tree | 9.52 ms | 1.91 |
| 4-row / 1-row, paired per-round median | 1.59× (rounds 1.55–1.65) | |
| **4-row / 1-row, lower envelope** (mean of the fastest quartile of each side: 5.86 vs 9.41 ms) | **1.60×** | |

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

_Scenario 2: host 1-minute load average 5.82 at start (gate: ≤ 3.5, waited 180 s; **still above the gate — timings below are load-affected**)._

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
| Exo-Ring-Pipeline | 5.5 | 5.7 | 5.4–5.7 | 1.00× | 180.41 | 5.46 | 21.1% | 0.99999 (100% top-1) | 1.00 |
| RDMA-Tensor-Parallel † | 3.5 | 3.6 | 3.5–3.6 | 0.63× | 285.46 | 471.68 | 30.3% | 0.99999 (100% top-1) | 1.00 |
| Tesseract-Fabric | 20.9 | 25.0 | 17.7–25.1 | 3.76× | 47.92 | 4.70 | 22.6% | 0.99999 (100% top-1) | 1.83 |
| Tesseract-Fabric (speculation off, ablation) | 15.4 | 16.3 | 13.7–16.6 | 2.78× | 64.93 | 3.79 | 22.4% | 0.99999 (100% top-1) | 1.00 |

Paired per-round ratios (median of the 5 of 5 rounds kept by the symmetric load-spike filter, min–max): spec-on / spec-off **1.49×** (1.29–1.61), spec-off / TP **4.41×** (3.91–4.73), TP / Exo **0.63×** (0.62–0.65).

Fastest-quarter envelope ratios (mean of each engine's fastest ⌈5/4⌉ rounds): spec-on / spec-off **1.53×**, spec-off / TP **4.59×**, TP / Exo **0.63×**.

† RDMA-Tensor-Parallel: 1/4 shard (151 MB) exceeds RAM of Android Snapdragon Edge — would OOM; run for latency only

Tesseract telemetry: decode 6.13s over 70 epochs; tree budget picks (budget:count) 1:17 2:17 3:4 4:13 6:13 8:3 16:3; draft-ahead hits/misses 0/52; KV commits 142 (mean 1.34 µs); expert staging — blocking misses 354 (Exo: 971), pre-staged 43, lookahead recall 83.1%, PreWake frames 793 (32.3 KB out-of-band); stage-0 cross-step lookahead rows reused / discarded / aborted early 16 / 0 / 1.

_Scenario 3: host 1-minute load average 3.89 at start (gate: ≤ 3.5, waited 180 s; **still above the gate — timings below are load-affected**)._

## Scenario 3 — Long-Context Scaling Stress Test (4 shards, TB5 links, 8 layers, GQA 8q/2kv heads)

Shard RAM budgets (MB): [512.0, 256.0, 192.0, 128.0]; layer split (both engines, ∝ RAM): [4, 2, 1, 1]. Context KV is injected with the layers' real K/V projections of the context tokens (single-pass approximate prefill), then 48 tokens are decoded with full forward passes.

KV remote-holder bias (calibrated once at 8K from lower envelopes): minimum remote exchange overhead 34 µs vs minimum 13 µs to scan one local page → 3 pages; capped to 2 so the placement's projected attention traffic at 8K / 32K / 64K (32.1 / 32.1 / 32.4 KB/step, dry-run page layout with real frame sizes) stays under the 34 KB/step wire budget. Every remote holder of a virtual layer is kept 2 pages behind the computing node.

| Context | Engine | Throughput, median of 7 interleaved rounds (tok/s) | min–max | vs ring: fastest-quarter envelope | vs ring: median of paired per-round ratios, load-spike filtered (min–max) | Peak node memory (weights+KV) | Memory balance (max/mean util) | Attention traffic (KB/step) | FP16 page-migration equivalent (KB/step) | LSE merge max abs err vs monolithic (live BFP wire + head-group striping) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2K | Layer-coupled ring KV (Exo) | 276.55 | 271.36–280.70 | **1.00×** | 1.00× (1.00–1.00) | 48 MB | 1.09 | 0.0 | — | — (monolithic per layer) |
| 2K | Tesseract disaggregated Flash-KV (speculation off) | 255.14 | 246.75–262.63 | **0.93×** | 0.92× (0.89–0.96) | 44 MB | 1.09 | 0.0 | 5376 | 4.47e-7 |
| 2K | Tesseract full engine (Flash-KV + speculation) | 287.10 | 284.52–288.75 | **1.03×** | 1.04× (1.02–1.06) | 44 MB | 1.09 | 0.0 | 5376 | 4.47e-7 |
| 8K | Layer-coupled ring KV (Exo) | 121.94 | 120.18–123.03 | **1.00×** | 1.00× (1.00–1.00) | 77 MB | 1.09 | 0.0 | — | — (monolithic per layer) |
| 8K | Tesseract disaggregated Flash-KV (speculation off) | 202.28 | 185.73–205.80 | **1.67×** | 1.64× (1.53–1.70) | 53 MB | 1.41 | 32.1 | 22016 | 1.11e-5 |
| 8K | Tesseract full engine (Flash-KV + speculation) | 202.82 | 196.98–205.60 | **1.67×** | 1.65× (1.62–1.70) | 53 MB | 1.41 | 32.1 | 22016 | 1.11e-5 |
| 32K | Layer-coupled ring KV (Exo) | 37.68 | 37.53–38.02 | **1.00×** | 1.00× (1.00–1.00) | 174 MB | 1.09 | 0.0 | — | — (monolithic per layer) |
| 32K | Tesseract disaggregated Flash-KV (speculation off) | 100.05 | 96.73–100.97 | **2.66×** | 2.66× (2.56–2.69) | 104 MB | 1.56 | 32.1 | 87808 | 1.14e-5 |
| 32K | Tesseract full engine (Flash-KV + speculation) | 99.05 | 97.61–100.77 | **2.64×** | 2.63× (2.57–2.69) | 104 MB | 1.56 | 32.1 | 87808 | 1.14e-5 |
| 64K | Layer-coupled ring KV (Exo) | 19.73 | 19.68–19.89 | **1.00×** | 1.00× (1.00–1.00) | 346 MB | 1.09 | 0.0 | — | — (monolithic per layer) |
| 64K | Tesseract disaggregated Flash-KV (speculation off) | 54.10 | 53.78–54.61 | **2.75×** | 2.75× (2.71–2.77) | 178 MB | 1.44 | 32.4 | 175616 | 1.34e-5 |
| 64K | Tesseract full engine (Flash-KV + speculation) | 53.87 | 53.12–54.28 | **2.73×** | 2.73× (2.68–2.74) | 178 MB | 1.44 | 32.4 | 175616 | 1.34e-5 |

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
| 2048×512 | 23.0 GMAC/s | 50.6 GMAC/s | 55.2 GMAC/s | 54.8 GMAC/s | 55.7 GMAC/s | 2.49× | 3.35× |
| 704×512 | 22.9 GMAC/s | 50.9 GMAC/s | 56.0 GMAC/s | 55.2 GMAC/s | 55.7 GMAC/s | 2.46× | 3.32× |
| 512×704 | 22.7 GMAC/s | 54.8 GMAC/s | 57.6 GMAC/s | 57.2 GMAC/s | 57.3 GMAC/s | 2.36× | 3.18× |

### Intra-node lane pool: one MoE layer, exact routing, 1-row step vs 4-row tree (min of 60 trials, interleaved)

Both batch sizes use the same lanes and the same striping rule (every active expert split into ⌈lanes / active experts⌉ hidden-unit and output stripes), so a 1-row step's 2 experts are parallelised as finely as the tree's 5–6.

| Lanes | 1 row | 4 rows | 4 rows / 1 row | max |Δ| vs sequential |
|---:|---:|---:|---:|---:|
| 1 | 125.6 µs | 346.8 µs | 2.76× | 0.0e0 |
| 2 | 82.4 µs | 192.5 µs | 2.34× | 0.0e0 |
| 4 | 58.3 µs | 110.9 µs | 1.90× | 0.0e0 |
| 8 | 57.3 µs | 131.4 µs | 2.29× | 0.0e0 |

### Anchor-consensus expert coalescing (APPROXIMATE; one MoE layer, 4-row tree, min of 40 trials)

| | Distinct experts touched | Time | vs 1 row |
|---|---:|---:|---:|
| 1 row (anchor only) | 2 | 134.5 µs | 1.00× |
| 4 rows, exact routing | 5.4 (mean) | 340.0 µs | 2.53× |
| 4 rows, coalesced (anchor's experts + 1 consensus) | ≤ 3 | 251.0 µs | 1.87× |

Coalescing changes the candidate rows' FFN outputs by 70.7% (relative L2, mean over candidate rows), so trees verified with it are not distributed exactly as the target model; it is therefore opt-in (`coalesce_experts`) and off in every scenario run.

### Lock-free slab KV commit (16 layers, 16 pending rows → 4 accepted)

Median 0.12 µs, p90 0.12 µs, p99 0.67 µs for the whole shard (V1: per-layer copy + compaction, ≈2 µs per layer in isolation, 14–16 µs per commit inside the loaded V1 cluster).

### Multipath striping (512 KB frames, 2.5GbE + Wi-Fi 6E, live sockets)

| Mode | Goodput, median frame (Gbps) | Byte split |
|---|---:|---|
| Best single link | 1.46 | 2.5GbE 100% |
| ETTC water-filling stripes | 1.55 | 2.5GbE 83% / WiFi 17% |
| 2.5GbE brownout (8 ms, 0.1 Gbps) — after telemetry adapts | 0.43 | 2.5GbE 0% / WiFi 100% |

## Section 4 — V8.0 production runtime: GGUF files over `mmap`, quantized kernels, native acceleration

The Scenario 1–3 model exported to real GGUF v3 files (llama.cpp tensor names; Mixtral-style stacked experts; norms and routers F32; in the Q4_0 file `output.weight` is Q8_0, as llama.cpp keeps it above Q4_0) and loaded back through `mmap`. The file was just written, so its pages are in the OS page cache: load times are *warm-cache* times (a disk-cold load adds the read of the pages a node touches). Integer-dot kernel: NEON (sdot). `portable` = the in-tree kernels used by every engine comparison above; `accelerate` = Apple Accelerate `cblas_sgemm` (AMX) for dense FP32 GEMMs and for dequantized tiles of batches ≥ 16 rows (its internal threads are not part of the equal-lanes rule, so it is reported here only).

#### Footprint, load and fidelity

| Format | GGUF file (MB) | FP32 bytes / file bytes | `mmap` + header parse (ms) | heap read of the whole file, for contrast (ms) | all layer views + head ready (ms) | weight bytes addressed by the full model (MB) | teacher-forced fidelity vs in-memory FP32 (cosine, top-1 of 64 positions) |
|---|---:|---:|---:|---:|---:|---:|---:|
| F32 | 612.8 | 1.00× | 2.82 | 55.8 | 83.0 | 608.5 | 1.00000 (100.0%) |
| F16 | 306.6 | 2.00× | 1.26 | 35.8 | 4.9 | 304.4 | 0.99999 (100.0%) |
| Q8_0 | 163.1 | 3.76× | 1.24 | 14.2 | 3.8 | 161.9 | 0.99985 (96.9%) |
| Q4_0 | 86.8 | 7.06× | 0.99 | 8.6 | 3.4 | 86.1 | 0.97140 (82.8%) |

F32 files are copied into the engine's dense tiled layout at load (that is the "views ready" time); F16 / Q8_0 / Q4_0 matrices stay zero-copy views into the mapping, and only the embedding table (decoded to f32), norms, routers and the re-ordered Q/K rows are materialised.

#### Throughput (greedy, 16-token prompt)

| Weights | single node, portable: prefill tok/s (128 tokens) / decode tok/s / first token | 4-node H-TEP decode tok/s, portable (median of 3) | single node, Accelerate: prefill / decode / first token | 4-node H-TEP decode tok/s, Accelerate (median of 3) |
|---|---:|---:|---:|---:|
| in-memory FP32 (synthetic, V7 path) | 1276 / 207.3 / 2384 ms | 219.9 | 2144 / 411.9 / 2377 ms | 202.2 |
| GGUF F32 | 713 / 193.4 / 90 ms | 195.6 | 870 / 332.1 / 94 ms | 190.7 |
| GGUF F16 | 399 / 151.0 / 54 ms | 187.4 | 750 / 150.7 / 57 ms | 168.5 |
| GGUF Q8_0 | 1103 / 634.7 / 17 ms | 237.2 | 1353 / 638.8 / 17 ms | 221.1 |
| GGUF Q4_0 | 1097 / 641.4 / 17 ms | 205.3 | 1475 / 629.0 / 18 ms | 200.7 |

Single node = the whole model on one node thread with the same 2-lane pool every engine node installs. 4-node H-TEP = Scenario 1's cluster (TB5-shaped loopback links, speculation on), loading its weights from the file. Fidelity and kernels are deterministic; throughput shares the host with its background load.

## Section 5 — V9.0: real third-party checkpoints, K-quants, Accelerate MoE offload

#### Fused quantized GEMM by format (2048 × 2048 weights, one node thread, kernel NEON (sdot); min of 30)

| Format | bits / weight | 1 row: µs (GMAC/s) | 4-row tree: µs (GMAC/s) |
|---|---:|---:|---:|
| Q8_0 | 8.50 | 159 (26.4) | 287 (58.5) |
| Q6_K | 6.56 | 169 (24.9) | 366 (45.8) |
| Q5_0 | 5.50 | 207 (20.3) | 332 (50.5) |
| Q4_K | 4.50 | 140 (30.0) | 306 (54.7) |
| Q4_0 | 4.50 | 167 (25.1) | 297 (56.5) |
| F16 | 16.00 | 1005 (4.2) | 1684 (10.0) |

The Scenario 1–3 model exported with llama.cpp's Q4_K_M recipe (Q4_K; Q6_K LM head and Q6_K `attn_v` / `ffn_down` on the "more bits" layers; Q5_0 / Q8_0 where a row is not a multiple of 256): 101.4 MB (6.04× smaller than FP32), teacher-forced cosine 0.98785 and top-1 82.8% vs the in-memory FP32 model.

#### Accelerate (AMX) vs portable kernels, single node (prefill 128 tokens / greedy decode 48 tokens)

| Model | portable: prefill tok/s / decode tok/s | Accelerate: prefill tok/s / decode tok/s |
|---|---:|---:|
| benchmark MoE (FP32, 8 experts: expert stripes on AMX) | 1304 / 208.7 | 2196 / 411.8 |
| Llama-3.2-1B Q4_K_M (dense; quantized tiles on AMX only from 16 rows) | 76 / 42.1 | 102 / 42.1 |

#### Real third-party GGUF checkpoints (`models/`, downloaded by `cargo test --test production_v9`)

Prompt (each model's own chat template): "Explain in one sentence what distributed AI inference is." Greedy, ≤ 48 tokens, stop at the template's end-of-turn token. Perplexity: a fixed 3-sentence English paragraph, teacher forced. Clusters: 4 loopback workers with unshaped links (this host's TCP stack is the network), lossless f32 activations / f32 H-TEP reduction, prompt-lookup speculation.

| Checkpoint | arch · layers · d · vocab | weight types | file (MB) | `mmap` + all views (ms) | perplexity | single node tok/s | 4-stage pipeline: tok/s · tokens/step · identical to single node | 4-rank H-TEP: tok/s · tokens identical before first divergence |
|---|---|---|---:|---:|---:|---:|---:|---:|
| SmolLM2-135M-Instruct-Q8_0.gguf | llama · 30 · 576 · 49152 | Q8_0 100% | 145 | 103.7 | 9.6 | 254.1 | 210.5 · 1.02 · yes | 118.9 · 41 / 41 |
| SmolLM2-135M-Instruct-Q4_K_M.gguf | llama · 30 · 576 · 49152 | Q4_K 8%, Q5_0 52%, Q6_K 10%, Q8_0 31% | 105 | 105.1 | 10.1 | 231.0 | 201.5 · 1.00 · yes | 122.8 · 37 / 37 |
| qwen2.5-0.5b-instruct-q4_k_m.gguf | qwen2 · 24 · 896 · 151936 | Q4_K 6%, Q5_0 55%, Q6_K 9%, Q8_0 30% | 491 | 272.7 | 8.0 | 78.3 | 73.8 · 1.00 · yes | 78.4 · 30 / 30 |
| Qwen3-0.6B-Q4_K_M.gguf | qwen3 · 28 · 1024 · 151936 | Q4_K 55%, Q6_K 45% | 397 | 314.8 | 14.3 | 75.8 | 72.0 · 1.00 · yes | 53.6 · 33 / 33 |
| Llama-3.2-1B-Instruct-Q4_K_M.gguf | llama · 16 · 2048 · 128256 | Q4_K 58%, Q6_K 42% | 808 | 247.1 | 10.6 | 41.7 | 38.5 · 1.00 · yes | 46.0 · 43 / 43 |

Verbatim single-node replies (greedy):

* **SmolLM2-135M-Instruct-Q8_0.gguf** (ChatML): "Distributed AI inference is a type of AI model that leverages distributed computing resources, such as distributed computing clusters, to simulate the behavior of complex systems, like neural networks, in real-time."
* **SmolLM2-135M-Instruct-Q4_K_M.gguf** (ChatML): "Distributed AI inference is a type of AI model that leverages distributed computing resources to process data in a distributed environment, allowing for faster and more efficient inference of complex data sets."
* **qwen2.5-0.5b-instruct-q4_k_m.gguf** (ChatML): "Distributed AI inference involves processing data across multiple machines or nodes, enabling efficient and scalable AI applications by leveraging parallel processing capabilities and distributed computing techniques."
* **Qwen3-0.6B-Q4_K_M.gguf** (ChatML (Qwen3, thinking off)): "Distributed AI inference refers to the process of deploying and processing large-scale AI models across multiple nodes or systems, enabling efficient and scalable processing of data and tasks."
* **Llama-3.2-1B-Instruct-Q4_K_M.gguf** (Llama 3): "Distributed AI inference refers to the process of training and executing artificial intelligence models in a distributed manner, where multiple machines or nodes work together to process and analyze data, rather than relying on a single central node."

_Section 6: host 1-minute load average 3.39 at start (gate: ≤ 3.5, waited 150 s)._

## Section 6 — V10.0: fast K-quant kernels, exact distributed inference, draft-model speculation, continuous batching

#### 6.1 Quantized GEMM by format and batch rows (2048 × 2048 weights, one thread, kernel NEON (sdot); min of 20) — GMAC/s, float / exact (fixed-point) accumulation

| Format | V9.0, 1 row (README) | 1 row | 2 rows | 4 rows | 8 rows |
|---|---:|---:|---:|---:|---:|
| Q8_0 | 14.7 | 26.4 / 24.2 | 42.3 / 37.8 | 58.5 / 44.7 | 58.3 / 44.8 |
| Q4_0 | 15.2 | 25.1 / 23.2 | 40.6 / 36.5 | 56.5 / 43.7 | 56.2 / 43.7 |
| Q5_0 | 13.0 | 20.7 / 18.6 | 34.7 / 28.8 | 50.6 / 38.8 | 49.7 / 38.8 |
| Q4_K | 10.5 | 30.0 / 28.9 | 39.9 / 38.3 | 54.8 / 43.3 | 55.0 / 43.5 |
| Q6_K | 12.4 | 24.9 / 22.8 | 34.9 / 26.7 | 45.8 / 32.7 | 45.8 / 32.6 |

#### 6.2 Exact distributed inference: 4-rank H-TEP vs a single node, bit for bit

Greedy, 40 tokens, each model's chat template, prompt "List the capitals of France, Italy and Spain, one per line, then explain why Rome is famous." The single node decodes one token per step; H-TEP runs the session scheduler (exact all-reduce of fixed-point partials, exact attention, context-copy speculation verified in multi-row chains).

| Checkpoint | attention split across ranks | single node tok/s | H-TEP×4 tok/s | tokens identical | logit rows bit-identical | tokens / step |
|---|---|---:|---:|---:|---:|---:|
| SmolLM2-135M-Instruct-Q8_0.gguf | no (replicated) | 252.0 | 111.5 | 40 / 40 | 40 / 40 | 1.03 |
| SmolLM2-135M-Instruct-Q4_K_M.gguf | no (replicated) | 227.3 | 113.3 | 40 / 40 | 40 / 40 | 1.00 |
| qwen2.5-0.5b-instruct-q4_k_m.gguf | no (replicated) | 77.9 | 76.1 | 40 / 40 | 40 / 40 | 1.05 |
| Qwen3-0.6B-Q4_K_M.gguf | yes (heads) | 73.7 | 56.4 | 40 / 40 | 40 / 40 | 1.18 |
| Llama-3.2-1B-Instruct-Q4_K_M.gguf | yes (heads) | 41.5 | 42.9 | 40 / 40 | 40 / 40 | 1.00 |

#### 6.3 Draft-model speculation: SmolLM2-135M (Q8_0) drafting for SmolLM2-1.7B (Q4_K_M), greedy prose, 64 tokens

Single node one token per step: 30.6 tok/s. Every configuration below reproduces it token for token.

| Ranks | Drafting | Chain length | tokens / step | draft tokens accepted | tok/s | vs single node | identical |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | context copy only | planned (cost model) | 1.00 | 0 / 6 (0%) | 27.2 | 0.89× | yes |
| 1 | draft model | fixed 4 | 2.46 | 38 / 103 (37%) | 17.6 | 0.57× | yes |
| 1 | draft model | planned (cost model) | 1.73 | 27 / 52 (52%) | 24.2 | 0.79× | yes |
| 4 | context copy only | planned (cost model) | 1.03 | 2 / 13 (15%) | 35.5 | 1.16× | yes |
| 4 | draft model | fixed 4 | 2.46 | 38 / 103 (37%) | 29.6 | 0.97× | yes |
| 4 | draft model | planned (cost model) | 1.78 | 28 / 55 (51%) | 36.2 | 1.19× | yes |

#### 6.4 Continuous batching: qwen2.5-0.5b-instruct-q4_k_m.gguf on 2 H-TEP ranks, 8 session slots, greedy, 32 tokens per request

N requests submitted at once; the head interleaves them in shared forward passes (chunked prefill + every session's draft chain in one batch, one all-reduce per layer for all of them). Aggregate = all tokens / wall time from submission to the last result.

| Concurrent requests | aggregate tok/s | per-request tok/s (mean) | vs 1 request | outputs identical to single node |
|---:|---:|---:|---:|---|
| 1 | 73.2 | 75.1 | 1.00× | 1 / 1 |
| 2 | 90.8 | 58.6 | 1.24× | 2 / 2 |
| 4 | 117.6 | 39.9 | 1.61× | 4 / 4 |
| 8 | 125.0 | 21.9 | 1.71× | 8 / 8 |

_Total benchmark wall time: 2082.5 s._
