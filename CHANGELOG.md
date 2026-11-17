# Changelog

## Unreleased — encrypted node traffic, fault tolerance, x86 kernels

### Security
- **Encrypted node-to-node traffic** (`wire::crypto`, `wire::auth`). With a cluster secret of your own, every TesseractWire connection now runs a mutual HMAC-SHA256 handshake (protocol `TFA2`), then carries its frames in ChaCha20-Poly1305 records under a per-connection key from HKDF-SHA256.
  - A modified, dropped, replayed or reordered record closes the connection.
  - SHA-256, HMAC, HKDF, ChaCha20, Poly1305 and the AEAD are implemented in-tree (still zero crates). They are checked against the FIPS 180-4, RFC 4231, RFC 5869 and RFC 8439 vectors, and were differentially tested against OpenSSL on 300 random inputs (0 mismatches).
  - Throughput is ≈ 500 MB/s per core.
  - `--wire-plaintext` (on every node) opts out on trusted dedicated links. With the public default secret, traffic stays authenticated but unencrypted.
  - **Nodes of 1.1.0 and this version cannot connect to each other.**

### Reliability
- **A node lost mid-request no longer hangs the request or kills the slot.**
  - The driver tracks peers whose connection ended (`EndpointShared::lost_peers`).
  - `Driver::try_generate_stream` returns `ClusterError::NodeLost` within ≈ 0.3 s and sends `Reset` to the survivors.
  - Worker endpoints are abortable: an abort (a `Reset` carrying `ABORT_MARK`; the ordinary `Reset` between requests is unchanged) interrupts any wait inside the running engine, and the worker drops it and returns to idle (`run_worker` catches the `Aborted` unwind).
  - The server answers `cluster_error` (an SSE event, or a 503), never poisons the slot lock, and re-plans without the dead node within ≈ 3.5 s, 3 s of them spent waiting for missing beacons.
  - Before, the request waited 900 s, panicked and left its slot unusable.
- **Fixed an intermittent server crash**: `tesseract-serve` died with EPIPE (a `println!` panic) when the process reading its stdout closed the pipe after the lines it needed. This was the intermittent Linux CI failure in `production_v9` and `security`. The address lines are now written ignoring a closed pipe, and informational lines go to stderr.
- Slots poisoned by an earlier panic are recovered and re-planned; `Driver::stats` skips lost nodes instead of waiting 30 s for them.

### Server
- **FIFO slot queue**: requests waiting for an engine slot are served in arrival order, and sleep on a condition variable instead of polling every 2 ms.

### Performance
- **x86_64 AVX2/FMA f32 kernels** (runtime-detected): a dot product and register-tiled GEMMs (4×3, 6×2, 2×4 and 1×4 tiles for the 16 ymm registers) for the batch GEMM and the lane-pool striping. They replace a portable SSE2 path that ran row by row. On a native x86 CI runner a 6-row batch now costs 2.44× one row and an 8-row batch 3.29× (before: 7.1× and 7.4×). The amortization check is asserted on x86 with AVX2 as on aarch64.

### Tooling
- **Physical cluster benchmark** (`docs/REAL_CLUSTER_BENCHMARK.md`): `tesseract-node --role coordinator … --rounds N --report FILE`.
  - Every engine runs in interleaved rounds (median and min–max).
  - Each link is measured before the runs (RTT and a 64 KB burst over the real, encrypted connections).
  - The markdown report records version, platform, model, nodes and per-round results, with secrets redacted.
- CI prints the native speed measurements (`Performance report` step).

### Tests
- `tests/faults.rs`: a worker process is SIGKILLed mid-stream; the stream must report `cluster_error`, the server must survive, and the next request must succeed on the survivors. Node traffic is encrypted in this test.
- Crypto vectors, differential checks of incremental hashing and MACs, handshake and record-layer tests (tampering, replay, reordering, wrong key, mode mismatch).
- An end-to-end test through a recording, tampering TCP relay: the plaintext never appears on the wire, and a flipped bit drops the connection.
- A FIFO ordering test (it fails 3/3 with the old polling).
- While writing the tests, two real bugs were found and fixed in the new code before release: partial-block buffering in SHA-256 / Poly1305 `update`, and a transcription error in one test vector caught by the OpenSSL cross-check.

## 1.1.0 — Security hardening and presentation (2026-11-16)

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
- CI asserts every correctness property and prints, without asserting, the speed thresholds calibrated on Apple Silicon (`TESSERACT_SKIP_PERF_ASSERTS=1`): on shared runners they flipped between runs (batching 1.49× vs 1.5×, Q5_0 0.756× vs 0.75× of Q8_0). On CI the servers started by tests log to the job output.
- First native Linux x86_64 CI run: every correctness test passes. Two speed thresholds calibrated on Apple Silicon are now asserted on aarch64 only, with their values printed everywhere: the absolute K-quant floor (the relative K-quant vs Q8_0 check stays on every target), and the tiled f32 batch amortization. The latter exposed a real gap, now listed under known limits.
- `LICENSE` (Apache-2.0, as declared in `Cargo.toml`).

### Upgrading from 1.0.0
- `tesseract-serve --http <non-loopback>` now needs `--api-key KEY` (or `TESSERACT_API_KEY`), or an explicit `--no-auth`.
- Browser clients on another origin need `--cors-origin`.
- Every node of a cluster must use the same `--cluster-secret`: it now authenticates node-to-node connections, not only discovery. Nodes running different versions cannot connect to each other.

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
- x86_64: no register-tiled f32 batch GEMM micro-kernel yet (aarch64 only); a 6-row batch costs ≈ 7× one row on a CI runner.
- Scenario 2 row 9 (spec-on > spec-off): parity (0.99–1.00× paired) at full host speed — the stage-0 cross-step lookahead, also active in the spec-off ablation, captures most of the speculative gain. Final scorecards 14/15 and 13/15 (run 2's row-15 miss is a host-load spike); see README.
- Benchmarks emulate every node as threads on one host; absolute tok/s and Scenario 2 depend on the host's clock regime.
- CI workflow is written but has not run (no git remote here); throughput floors in some tests were calibrated on an M3 Max.
