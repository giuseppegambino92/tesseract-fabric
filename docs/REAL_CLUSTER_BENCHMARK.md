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

Machine B also has an RTX 5090, which these runs do not use: the engines run on the CPU.

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
