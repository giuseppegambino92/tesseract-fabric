# Physical cluster benchmark

The benchmarks in `bench_results*.md` run every node as threads on one host, with emulated links and emulated device speeds. This procedure measures the same engines on **real machines over a real network**, and writes a report that records everything needed to reproduce or compare a run.

It compares, on the same machines and model:
- **Exo-Ring-Pipeline**: a layer-sharded ring.
- **RDMA-Tensor-Parallel**: Megatron-style tensor parallelism, when the head counts divide across the machines.
- **Tesseract-Fabric**: whatever the solver picks for the machines and links — H-TEP or the speculative pipeline.

## 1. Prepare every machine

- Install the **same build** everywhere: same commit, `cargo build --release`.
- Check the machine state, which the report can only partly capture:
  - power adapter connected, low-power mode off;
  - no other heavy jobs (emulators, builds, sync daemons);
  - load average low (`sysctl -n vm.loadavg` / `uptime`).
- Open three consecutive TCP ports per worker (node, KV service, draft service), e.g. 7000–7002.
- For a real checkpoint, put the same GGUF file on every machine.

## 2. Start the workers

Use the same long random secret everywhere; it authenticates every connection and encrypts node traffic.

```bash
export TESSERACT_SECRET=$(openssl rand -hex 32)   # generate once, copy to every machine

# on each machine i = 0, 1, 2, …
tesseract-node --role worker --id <i> --listen 0.0.0.0:7000 --cluster-secret "$TESSERACT_SECRET" \
    [--gguf /path/to/model.gguf]
```

On a Linux machine with an NVIDIA GPU, start the worker with `TESSERACT_BACKEND=cuda` to run its layers on the GPU (see [GPU nodes](#gpu-nodes)). The worker's first log line names its kernels, e.g. `kernels cuda (NVIDIA GeForce RTX 5090 · sm_120 · 31.4 GiB)`.

Encryption (X25519 key exchange with forward secrecy, then ChaCha20-Poly1305) costs about 1.4 GB/s (11 Gb/s) of one core; large records are split across up to 4 cores, reaching ≈ 3.7 GB/s (≈ 30 Gb/s) on an M3 Max. The report states the coordinator's own sealing throughput next to the measured links, so you can see whether encryption could limit them. To measure without it on a trusted dedicated link, add `--wire-plaintext` on **every** node and say so when you publish.

## 3. Run the coordinator

```bash
tesseract-node --role coordinator --listen <coordinator-ip>:6999 --cluster-secret "$TESSERACT_SECRET" \
    --peers 10.0.0.2:7000,10.0.0.3:7000,10.0.0.4:7000 \
    --profile studio-m5max,rtx5090,macbook-pro \
    --engine all --model bench --tokens 128 --temperature 0 \
    --rounds 5 --report cluster-report.md
```

- `--peers`: one address per worker, in id order.
- `--profile`: the closest device preset per worker. It is only used by the solver to plan, and is reported as "declared".
- `--model`:
  - `tiny`, `bench` or `long` for the synthetic models;
  - a GGUF / SafeTensors path for a real checkpoint, which the workers load with `--gguf`.
- `--ram-mb A,B,..` (optional): the memory each node may use for the model, in MB of stored weights. Use it to test a split that the machines' real memory would not force.
- `--rounds 5`: every engine runs once per round, interleaved, so drifting load or network conditions hit all engines alike. The report gives the median and the min–max.
- No `--emulate-links`: the real network is the link model and device speeds are not stretched.

Before the runs, the coordinator **measures** every link (three echo pings for RTT, one 64 KB burst for bandwidth) over the same encrypted connections the engines use. The measurements go into the report.

## 4. What the report contains

`cluster-report.md` holds:
- the command, with secrets redacted;
- the tesseract-fabric version, the coordinator's OS and architecture;
- the model and the number of rounds;
- each node's address, declared profile, measured RTT and bandwidth;
- the coordinator's load at each round;
- per engine:
  - median tok/s with its min–max;
  - ms/token;
  - tokens per verification step;
  - network KB per token;
  - every round's tok/s.

Write down separately what the report cannot know:
- each machine's exact hardware and OS version;
- how the machines are connected (switch, cables, Wi-Fi band);
- anything else running.

## 5. Good practice

- Run the whole procedure **twice** and publish both reports: one run is an anecdote.
- Keep `--temperature 0`, so every engine generates the same tokens and the comparison is about speed only.
- The first round includes one-off costs (first connections, page cache). The median over 5 rounds absorbs it; use more rounds on noisy networks.
- To test a hypothesis (another codec, another tree budget), change one flag at a time and keep everything else identical.

## Smoke test on one machine

The procedure can be rehearsed on loopback. The numbers measure the loopback, not a network.

```bash
for i in 0 1 2; do tesseract-node --role worker --id $i --listen 127.0.0.1:$((7100+10*i)) --cluster-secret test-secret & done
tesseract-node --role coordinator --listen 127.0.0.1:0 --cluster-secret test-secret \
    --peers 127.0.0.1:7100,127.0.0.1:7110,127.0.0.1:7120 --profile studio-m5max,studio-m5max,macbook-pro \
    --engine all --model tiny --tokens 32 --temperature 0 --rounds 3 --report smoke-report.md
```

## First physical run (2026-11-17)

Two real machines on the same LAN, run with this procedure (commit after 1.2.0, encrypted and `--wire-plaintext` runs alternated, 5 interleaved rounds each, two passes, 64 tokens, temperature 0):

| Machine | Hardware | Network |
|---|---|---|
| A (coordinator + worker 0) | MacBook Pro, Apple M3 Max, 36 GB, macOS | 10 GbE |
| B (worker 1) | AMD Ryzen 9 9950X3D (16 cores), 91 GB DDR5, Ubuntu 25.04 | 5 GbE (the link runs at 5 Gb/s) |

Machine B also has an RTX 5090, which these runs do not use: the engines run on the CPU. The [second run](#second-physical-run-rtx-5090-2026-11-17) uses it.

**Single-node reference** (same build, `--role chat --local-nodes 1`):

| Model | A alone | B alone |
|---|---:|---:|
| SmolLM2-135M Q4_K_M | 206 tok/s | 102 tok/s |
| Llama-3.2-1B Q4_K_M | 40.1 tok/s | 19.1 tok/s |

A is about 2× faster than B with this engine, so the runs declare `--profile studio-m5max,macbook-pro`. Those presets have a 2 : 1 memory-bandwidth ratio, matching the measurement; the solver only knows what the presets say.

**Two machines, tok/s** (median of 5 rounds; the two passes shown as pass 1 / pass 2):

| Model | Engine | Encrypted | Plaintext | Network KB/token |
|---|---|---:|---:|---:|
| Llama-3.2-1B | Tesseract-Fabric | 35.0 / 34.1 | 35.1 / 34.0 | 0.15 |
| | Exo-Ring-Pipeline | 24.5 / 23.7 | 25.0 / 24.0 | 5.2 |
| | RDMA-Tensor-Parallel | 14.1 / 14.7 | 14.2 / 16.4 | 321 |
| SmolLM2-135M | Tesseract-Fabric | 206.5 / 206.8 | 207.7 / 200.8 | 0.14 |
| | Exo-Ring-Pipeline | 118.6 / 122.8 | 117.7 / 121.5 | 1.6 |
| | RDMA-Tensor-Parallel | n/a (9 heads do not divide across 2 ranks) | | |

What this shows:
- **Encryption costs nothing measurable** on a real 5 GbE link. Every encrypted/plaintext pair is within the round-to-round spread.
- **Tesseract-Fabric plans the cluster around the faster machine.** All layers stay on A, and B serves draft trees. It runs 1.4–1.7× Exo's layer split and 2.4× tensor parallelism, which pay a network hop per token or per layer to a slower machine.
- **It does not beat the faster machine alone.** These models fit in one machine:
  - SmolLM2: 206 tok/s on two machines, the same as A alone.
  - Llama-3.2-1B: 34–35 tok/s against 40 on A alone, **about 13 % slower**. B's drafts were not accepted (1.00 tokens per step), so the remote drafter only added round trips.
  - Splitting a model across machines pays off when it does not fit in one; these runs do not test that case.
- The coordinator's link probe (3 pings, one 64 KB burst) under-reads a 5 GbE link: 0.3–1.2 Gb/s reported. Treat its numbers as a sanity check, not a bandwidth measurement.

## Second physical run: RTX 5090 (2026-11-17)

The same two machines and network as the first run, with version 1.4.0 and machine B's RTX 5090 in use (`TESSERACT_BACKEND=cuda` on B's worker; A runs the portable CPU kernels). All runs are encrypted, with 5 interleaved rounds, 64 tokens, temperature 0 and two passes, shown as pass 1 / pass 2. The coordinator's 1-minute load stayed between 3.8 and 6.2 in every round.

Since 1.4.0 the benchmark runs real checkpoints with the same settings as `chat` and `tesseract-serve`: lossless activations between stages and prompt-lookup drafting. Numbers are therefore not comparable one-to-one with the first run.

### GPU nodes

With `TESSERACT_BACKEND=cuda`, a Linux worker opens the NVIDIA driver, NVRTC and cuBLAS with `dlopen`; no build step or crate is involved.
- It compiles its kernels for the GPU at start-up and uploads the weight matrices of its own layers to VRAM, where they stay.
- Each matrix product then runs on the GPU. Only the activations travel over PCIe.
- Exact layers (attention output, FFN down) use a fixed-point kernel that matches the CPU bit for bit, so mixed CPU / GPU clusters keep the exactness guarantee.
- Attention, norms and sampling stay on the CPU.

### Single node

Measured with the same coordinator, prompt and drafting as the cluster runs, with a single peer (`--peers <one worker> --engine tesseract`). tok/s:

| Model | A: M3 Max, CPU | B: Ryzen 9 9950X3D, CPU | B: RTX 5090, CUDA | 5090 vs Ryzen | 5090 vs M3 Max |
|---|---:|---:|---:|---:|---:|
| SmolLM2-135M Q4_K_M | 338.8 / 340.7 | 124.6 / 118.3 | 385.3 / 434.7 | 3.1–3.7× | 1.1–1.3× |
| Llama-3.2-1B Q4_K_M | 40.2 / 39.8 | 18.9 / 19.0 | 253.6 / 252.7 | 13.4× | 6.3× |
| Llama-3.2-3B Q4_K_M | 16.2 / 16.3 | 7.7 / 7.6 | 127.3 / 128.2 | 16.7× | 7.9× |

SmolLM2's matrices are small: per-product launch and copy latency (≈ 10 µs) and the CPU-side attention dominate, so the GPU gains less. `cargo run --release --example cuda_bench` breaks the latency down per weight shape.

### Two machines, RTX 5090 as primary

`--profile macbook-pro,rtx5090` declares B as the faster machine. The solver keeps every layer on B, B verifies, and A drafts. tok/s:

| Model | Tesseract-Fabric | Exo-Ring-Pipeline | RDMA-Tensor-Parallel | Tesseract vs B alone |
|---|---:|---:|---:|---:|
| SmolLM2-135M | 420.6 / 440.8 (1.94 tokens / step) | 187.0 / 180.0 | n/a (9 heads) | ≈ equal |
| Llama-3.2-1B | 244.2 / 244.7 | 67.0 / 66.3 | 21.6 / 21.9 | −3.4 % |
| Llama-3.2-3B | 124.2 / 125.4 | 26.4 / 26.6 | 7.3 / 8.3 | −2.2 % |

Network per token:
- Tesseract-Fabric: 0.11–0.14 KB;
- Exo: 1.6 / 5.2 / 7.7 KB;
- tensor parallelism: 321 KB (1B) and 838 KB (3B).

### Roles inverted: A declared primary

`--profile studio-m5max,macbook-pro` declares A as a Mac Studio M5 Max and B as a laptop, the presets of the first run. The solver keeps every layer on A, and B's GPU only drafts. tok/s:

| Model | Tesseract-Fabric | Exo-Ring-Pipeline | RDMA-Tensor-Parallel | A alone |
|---|---:|---:|---:|---:|
| SmolLM2-135M | 316.7 / 312.6 (2.1–3.1 tokens / step) | 187.5 / 181.2 | n/a | 338.8 / 340.7 |
| Llama-3.2-1B | 40.7 / 39.8 | 57.8 / 58.5 | 22.5 / 22.2 | 40.2 / 39.8 |
| Llama-3.2-3B | 15.9 / 16.3 | 22.4 / 23.3 | 8.9 / 8.3 | 16.2 / 16.3 |

### The model split across both machines (Llama-3.2-3B)

`--ram-mb` caps the memory each machine may use for the model, so that neither holds all 28 layers and the solver must split them **14 + 14**. Two placements were tested:
- `--ram-mb 1800,1000`: the LM head (128k-token vocabulary) on A;
- `--ram-mb 1000,1760`: the LM head on B.

Exo splits layers in proportion to the same caps instead (18 + 10 and 10 + 18). tok/s:

| B runs on | LM head on | Tesseract-Fabric | Exo-Ring-Pipeline | RDMA-Tensor-Parallel |
|---|---|---:|---:|---:|
| RTX 5090 (CUDA) | A | 22.9 / 22.9 | 25.2 / 25.4 (A 18 layers) | 7.3 / 8.2 |
| RTX 5090 (CUDA) | B | 22.4 / 22.5 | 32.1 / 30.2 (A 10 layers) | 8.5 / 8.0 |
| Ryzen CPU | A | 10.5 / 10.2 | 10.0 / 9.6 | 6.4 / 6.0 |
| Ryzen CPU | B | 8.9 / 9.3 | 8.5 / 8.7 | 6.2 / 6.1 |

For reference, A alone runs this model at 16.2 tok/s and B's GPU alone at 127.3, when each has the memory to hold it.

Network per token on the split:
- Tesseract-Fabric: 15.3 KB (lossless f32 activations);
- Exo: 7.7 KB (f16);
- tensor parallelism: 838 KB.

### What this shows

- **The CUDA backend makes B the fast machine**: 13–17× its own CPU on Llama, and 6–8× A's CPU engine.
- **With B declared as primary, Tesseract-Fabric gives the RTX 5090's speed to the cluster.** It stays within 2–4 % of B alone (the network hop to the coordinator and the drafter's messages), and is 2.3–4.7× Exo and 11–17× tensor parallelism. Those two engines put layers on A's CPU, so every token waits for it.
- **The solver trusts the declared profiles.** When A is declared the faster machine, Tesseract-Fabric keeps everything on A and runs exactly as fast as A alone (no longer 13 % slower as in the first run). Exo splits by declared memory, which happens to put 4–8 layers on the GPU, and wins on Llama (58 against 40 tok/s on 1B). Declare the right presets. Since 1.4.0, workers running `--discover` with `TESSERACT_BACKEND=cuda` advertise their measured GPU bandwidth, SGEMM throughput and free VRAM instead of their CPU's.
- **When the model must be split, both accelerators work and the CPU stage bounds the speed.**
  - With the GPU, the split runs 2.2–2.5× faster than with B's CPU, and 1.4× A alone.
  - A's 14 layers take most of each token's ≈ 44 ms, so moving the LM head between machines changes little for Tesseract-Fabric.
  - Exo's proportional rule put only 10 layers on A when the head sat on B, and was faster there (30–32 tok/s). Tesseract-Fabric's solver reserves 15 % of each cap for the KV cache and budgets the tied embedding / LM head table twice (lookup and readout), so the same caps leave it 14 layers on A.
- **Drafting costs ≈ 11 % on the split.** The drafter sits on A, the bottleneck stage, and this prompt accepts no drafts (1.00 tokens per step). Run without the other engines interleaved, Tesseract-Fabric reaches 25.8–29.4 tok/s with drafting and 30.3–30.4 with `--budget 1` (speculation off), with the LM head on B. In both runs the fifth round dropped to 10.9 tok/s; the cause was not identified (the GPU was shared at the time, see below).
  - Interleaved with Exo and tensor parallelism, as in the table, the same configuration measured 22.4–22.5. That gap is not explained yet.
- **Encryption costs nothing measurable here either.** Alternating plaintext and encrypted runs (Llama-3.2-1B, B primary, Tesseract-Fabric only) gave medians of 244.1 and 244.2 tok/s in plaintext against 243.5 and 243.8 encrypted. The coordinator (on A) seals ≈ 4 GB/s, while the heaviest traffic in these runs is 7.5 MB/s (tensor parallelism on the 3B model).
- **The GPU was shared.** Other software used the RTX 5090 at times during the session (image generation, a local LLM server). Runs made while it was busy showed rounds alternating between ≈ 19 and ≈ 243 tok/s, and were repeated once the GPU had been idle for 30 s.
