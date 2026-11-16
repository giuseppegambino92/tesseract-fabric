# Changelog

## Unreleased — security hardening and presentation

### Security
- **HTTP API authentication**: `--api-key` / `TESSERACT_API_KEY` requires `Authorization: Bearer` on every `/v1/*` endpoint (constant-time comparison; OpenAI-style `authentication_error`). The dashboard takes the key from `/#key=…` and keeps it for the tab only.
- **Open API refused beyond loopback**: `tesseract-serve` exits with an error when `--http` is not a loopback address and no key is set, unless `--no-auth` is passed explicitly.
- **CORS** off by default (was `Access-Control-Allow-Origin: *`); `--cors-origin` allows one origin.
- **Bounded request parsing**: head limit enforced while bytes arrive (was checked only after a full line, so a newline-free stream could allocate without bound); at most 100 header lines; `Transfer-Encoding` and duplicate or invalid `Content-Length` refused; the body grows with the bytes received instead of being pre-allocated from the declared length.
- **Connection admission**: at most 64 concurrent connections (`--max-connections`) and 16 per client IP (`--max-per-client`); excess connections get an immediate `503` and no thread. Write timeouts on every connection.
- **Authenticated node-to-node connections** (`wire::auth`): every TesseractWire TCP connection starts with a mutual challenge-response (fresh nonces, 128-bit SipHash-2-4 MAC keyed from `--cluster-secret`); no frame is read from a peer that fails it. Frames are not encrypted: use WireGuard / Tailscale on untrusted networks. `tesseract-node` warns when it listens beyond loopback with the public default secret.
- **Available memory on macOS**: nodes advertise installed RAM minus app, wired and compressed memory (as Activity Monitor computes it) instead of installed RAM (`hw.memsize`).

### Tests and tooling
- 10 new tests (`tests/security.rs`, wire-handshake and memory unit tests): 110 in total.
- The shaper unit test now checks the shaper's own accounting instead of wall time (it failed on a busy GitHub macOS runner that overslept a 15 ms sleep by 50 ms).
- CI can be started by hand (`workflow_dispatch`).
- `LICENSE` (Apache-2.0, as declared in `Cargo.toml`).

### Documentation
- README: badges, architecture diagram, performance charts (light and dark), terminal screenshots and a dashboard screenshot from real runs, and a security section (EN / IT). Regenerate with `scripts/make_figures.py` and `scripts/make_screenshots.py`.

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
