# Benchmarks

Two kinds of measurement live in this repository:

1. **Physical runs on real machines** over a real network, recorded in [`docs/REAL_CLUSTER_BENCHMARK.md`](../docs/REAL_CLUSTER_BENCHMARK.md) together with the procedure to reproduce them.
2. **Emulated cluster runs** written by `tesseract-bench` (the files in this folder). Every node runs as threads on one host, with shaped links and compute slowdowns derived from device profiles. Results depend on the host's load, which each file records at the start.

## Summary

### Real hardware: M3 Max laptop + Ryzen 9 9950X3D workstation with RTX 5090, 5 GbE (v1.4.0)

Decode tok/s, two passes, encrypted node traffic. Source: [second physical run](../docs/REAL_CLUSTER_BENCHMARK.md#second-physical-run-rtx-5090-2026-11-17).

| Model | M3 Max (CPU) | Ryzen 9 9950X3D (CPU) | RTX 5090 (CUDA) | M3 Max + RTX 5090, Tesseract-Fabric | same cluster, Exo ring / tensor parallel |
|---|---:|---:|---:|---:|---:|
| SmolLM2-135M | 339–341 | 118–125 | 385–435 | 421–441 | 180–187 / n/a |
| Llama-3.2-1B | 40 | 19 | 253 | 244 | 66–67 / 22 |
| Llama-3.2-3B | 16 | 7.6 | 127–128 | 124–125 | 26–27 / 7–8 |
| Llama-3.2-3B, split 14 + 14 layers | — | — | — | 22–23 | 25–32 / 7–9 |

- With the RTX 5090 declared primary, the cluster runs within 2–4 % of the GPU alone, and 2.3–4.7× the Exo-style ring.
- When the model has to be split across both machines, the M3 Max's CPU stage bounds the speed, and Exo's proportional layer split was faster on both placements tested (25–32 against 22–23 tok/s).
- When a model fits on one machine, the cluster does not beat that machine alone. In the [first physical run](../docs/REAL_CLUSTER_BENCHMARK.md#first-physical-run-2026-11-17) (CPU only) it matched it on SmolLM2-135M and was about 13 % slower on Llama-3.2-1B.
- Encryption costs nothing measurable on the 5 GbE link.

### Exactness

A 4-rank H-TEP cluster matches a single node on 40/40 tokens and 40/40 logit rows for SmolLM2, Qwen2.5, Qwen3 and Llama 3.2 (dense models; MoE expert routing is not covered by the exactness guarantee). This is checked by the test suite, not by a timing run.

### Emulated runs (Apple M3 Max host)

- **Final scorecards (v1.0.0):** 14/15 and 13/15 targets met in two back-to-back runs ([`bench_results_v1_0_0_run1.md`](bench_results_v1_0_0_run1.md), [`bench_results.md`](bench_results.md)). Scenario 2 row 9 reaches parity (0.99–1.00×) rather than a win; run 2's row 15 miss was a host-load spike. The host was in use during both runs.
- **Quantized kernels**, one thread, one row, GMAC/s (from [`bench_results_v10_0.md`](bench_results_v10_0.md) §6.1):

  | | Q8_0 | Q4_0 | Q5_0 | Q4_K | Q6_K |
  |---|---:|---:|---:|---:|---:|
  | V9.0 | 14.7 | 15.2 | 13.0 | 10.5 | 12.4 |
  | v1.0 | 26.4 | 25.1 | 20.7 | 30.0 | 24.9 |

- **Continuous batching**, Qwen2.5-0.5B on 2 H-TEP ranks (from [`bench_results_v10_0.md`](bench_results_v10_0.md) §6.4):

  | Concurrent requests | 1 | 2 | 4 | 8 |
  |---|---:|---:|---:|---:|
  | Aggregate tok/s | 73.2 | 90.8 | 117.6 | 125.0 |

The full discussion of each run, including what changed between versions and why some targets were missed, is in the project history in the main [README](../README.md#full-technical-documentation-and-project-history-english).

## Files

`bench_results.md` is the most recent full run (v1.0.0, run 2). The other files are archived runs, named after the version that produced them; `_run1` marks the first of two back-to-back runs.

| Version | Files |
|---|---|
| v1.0.0 | [`bench_results_v1_0_0_run1.md`](bench_results_v1_0_0_run1.md), [`bench_results.md`](bench_results.md) |
| V10.1 | [`bench_results_v10_1_run1.md`](bench_results_v10_1_run1.md), [`bench_results_v10_1.md`](bench_results_v10_1.md) |
| V10.0 | [`bench_results_v10_0_run1.md`](bench_results_v10_0_run1.md), [`bench_results_v10_0.md`](bench_results_v10_0.md) |
| V9.0 | [`bench_results_v9_run1.md`](bench_results_v9_run1.md), [`bench_results_v9.md`](bench_results_v9.md) |
| V8.0 | [`bench_results_v8.md`](bench_results_v8.md) |
| V7.0 | [`bench_results_v7_0_run1.md`](bench_results_v7_0_run1.md), [`bench_results_v7_0.md`](bench_results_v7_0.md) |
| V6.8 | [`bench_results_v6_8_run1.md`](bench_results_v6_8_run1.md), [`bench_results_v6_8.md`](bench_results_v6_8.md) |
| V6.7 | [`bench_results_v6_7_run1.md`](bench_results_v6_7_run1.md), [`bench_results_v6_7.md`](bench_results_v6_7.md) |
| V6.6 | [`bench_results_v6_6_run1.md`](bench_results_v6_6_run1.md), [`bench_results_v6_6.md`](bench_results_v6_6.md) |
| V6.5 | [`bench_results_v6_5.md`](bench_results_v6_5.md) |
| V6.4 | [`bench_results_v6_4.md`](bench_results_v6_4.md) |
| V6.3 | [`bench_results_v6_3.md`](bench_results_v6_3.md) |
| V6.1 | [`bench_results_v6_1.md`](bench_results_v6_1.md) |
| V6 | [`bench_results_v6.md`](bench_results_v6.md) |
| V5 | [`bench_results_v5.md`](bench_results_v5.md) |
| V4 | [`bench_results_v4.md`](bench_results_v4.md) |
| V3 | [`bench_results_v3.md`](bench_results_v3.md) |
| V2 | [`bench_results_v2.md`](bench_results_v2.md) |
| V1 | [`bench_results_v1.md`](bench_results_v1.md) |

## Running

```bash
cargo run --release --bin tesseract-bench              # all scenarios → benchmarks/bench_results.md
cargo run --release --bin tesseract-bench -- --quick   # shorter run
cargo run --release --bin tesseract-bench -- --out path/to/results.md
```

A full run takes a long time and is sensitive to background load; see the methodology in the main README. To measure real machines, follow [`docs/REAL_CLUSTER_BENCHMARK.md`](../docs/REAL_CLUSTER_BENCHMARK.md).
