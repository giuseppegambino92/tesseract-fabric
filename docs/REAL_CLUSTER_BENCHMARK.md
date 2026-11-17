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

Encryption (ChaCha20-Poly1305, ≈ 500 MB/s per core in this implementation) is enough for Ethernet and Wi-Fi. On a dedicated, trusted link faster than ≈ 4 Gb/s (e.g. Thunderbolt), it can become the bottleneck. Add `--wire-plaintext` on **every** node to measure without it, and say so when you publish.

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
