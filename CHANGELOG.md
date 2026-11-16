# Changelog

## 1.0.0 — Gold Release (2026-11-16)

Zero external crates. 100 tests (`cargo test --release`), `cargo fmt --check` and `cargo clippy --all-targets -- -D warnings` clean; CI on Linux and macOS (`.github/workflows/ci.yml`).

### Highlights (V8.0 → 1.0.0)
- Real third-party GGUF checkpoints (SmolLM2, Qwen2.5, Qwen3, Llama 3.2): F16 / BF16 / Q8_0 / Q6_K / Q5_0 / Q4_K / Q4_0, zero-copy `mmap`, byte-level BPE tokenizer, chat templates.
- NEON (`sdot`) / AVX2 quantized kernels, e.g. Q4_K 10.5 → 30 GMAC/s at one row; Apple Accelerate (AMX) offload.
- 4-rank H-TEP reproduces a single node bit for bit on all five checkpoints (fixed-point all-reduce, split-invariant attention).
- Continuous batching (`tesseract-serve --sessions N`), companion draft models (`--draft-model`), lossless speculation.
- Disaggregated KV fabric serving successive requests; OpenAI-compatible HTTP/SSE server with dashboard; authenticated LAN/Tailscale discovery.

### Changed in 1.0.0
- Solver places the drafter by cost (compute + round trip to the coordinator); a plan with a remote drafter enables draft-ahead (`TesseractOptions::use_plan_drafter`).
- Speculation planner: tree costs relative to the live 1-row cost (follows the host clock regime), successor-lookahead charge for tree epochs, lower-median cost estimates, per-assignment persistence across requests; `TESSERACT_PLANNER_TRACE=1`.
- Compute shaper stretches thread-CPU critical-path time instead of wall time (V10.1); `TESSERACT_SHAPER=wall` restores the old model, `TESSERACT_SHAPER_TRACE=1` traces it.
- The solver test `solver_keeps_edge_and_wifi_nodes_off_the_critical_path` now checks the cost-based drafter rule instead of requiring the phone.

### Known limits
- Scenario 2 row 9 (spec-on > spec-off): parity (0.99–1.00× paired) at full host speed — the stage-0 cross-step lookahead, also active in the spec-off ablation, captures most of the speculative gain. Final scorecards 14/15 and 13/15 (run 2's row-15 miss is a host-load spike); see README.
- Benchmarks emulate every node as threads on one host; absolute tok/s and Scenario 2 depend on the host's clock regime.
- CI workflow is written but has not run (no git remote here); throughput floors in some tests were calibrated on an M3 Max.
