"""Regenerate the README figures (``docs/images/<name>-{light,dark}.svg``).

Pure Python, no dependencies. Every number below is copied from a benchmark
file in this repository (named next to each figure), so the charts can be
re-checked against the measurements. Each figure is written twice: a light and
a dark variant, which the README selects with GitHub's ``#gh-light-mode-only``
/ ``#gh-dark-mode-only`` fragments (they follow the viewer's GitHub theme).

Colours follow a validated categorical palette (slot 1 blue, slot 2 orange;
both modes pass lightness, chroma, colour-vision-deficiency and contrast
checks against their surface).

Run: ``python3 scripts/make_figures.py``
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"

FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

THEMES = {
    "light": dict(
        surface="#fcfcfb", panel="#f3f2ef", border="#d9d8d3", grid="#e8e7e3",
        text="#0b0b0b", text2="#52514e", muted="#6f6e69",
        s1="#2a78d6", s2="#eb6834", s3="#1baf7a",
        s1_soft="#e6effb", s2_soft="#fdebe3", s3_soft="#e3f5ee",
    ),
    "dark": dict(
        surface="#1a1a19", panel="#232322", border="#3a3a37", grid="#2f2f2d",
        text="#f0efec", text2="#c3c2b7", muted="#9a9990",
        s1="#3987e5", s2="#d95926", s3="#199e70",
        s1_soft="#1b2a3d", s2_soft="#3a2419", s3_soft="#15302a",
    ),
}


def t(x, y, s, size=13, fill="text", weight=400, anchor="start", family=FONT, th=None):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" fill="{th[fill]}" text-anchor="{anchor}">{escape(s)}</text>')


def bar_up(x, y, w, h, fill, r=4):
    """Vertical bar anchored to the baseline at y+h, rounded data end on top."""
    r = min(r, w / 2, h)
    return (f'<path d="M{x:.1f},{y + h:.1f} L{x:.1f},{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} '
            f'L{x + w - r:.1f},{y:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} L{x + w:.1f},{y + h:.1f} Z" '
            f'fill="{fill}"/>')


def bar_right(x, y, w, h, fill, r=4):
    """Horizontal bar anchored at x, rounded data end on the right."""
    r = min(r, h / 2, w)
    return (f'<path d="M{x:.1f},{y:.1f} L{x + w - r:.1f},{y:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} '
            f'L{x + w:.1f},{y + h - r:.1f} Q{x + w:.1f},{y + h:.1f} {x + w - r:.1f},{y + h:.1f} L{x:.1f},{y + h:.1f} Z" '
            f'fill="{fill}"/>')


def frame(w, h, th, body, label):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-label="{escape(label)}">'
            f'<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="12" fill="{th["surface"]}" '
            f'stroke="{th["border"]}"/>{body}</svg>\n')


# --- 1. kernels: V9 vs v1.0, GMAC/s at one row ------------------------------
# Source: benchmarks/bench_results_v10_0.md Section 6.1 (host load 3.4, M3 Max, one thread,
# NEON sdot); V9 column = the V9.0 README figures quoted in the same table.
KERNELS = [("Q8_0", 14.7, 26.4), ("Q4_0", 15.2, 25.1), ("Q5_0", 13.0, 20.7),
           ("Q4_K", 10.5, 30.0), ("Q6_K", 12.4, 24.9)]


def kernels(th):
    W, H = 760, 400
    x0, y0, pw, ph = 70, 96, 660, 220
    vmax = 35.0
    body = [t(28, 40, "Quantized matrix-vector kernels: V9 → v1.0", 19, weight=700, th=th),
            t(28, 64, "GMAC/s on one thread at one row (higher is better) · Apple M3 Max, NEON sdot", 13, "text2", th=th)]
    # legend (top right)
    lx = W - 230
    for i, (name, key) in enumerate([("V9.0", "s2"), ("v1.0", "s1")]):
        body.append(f'<rect x="{lx + i * 90}" y="30" width="12" height="12" rx="3" fill="{th[key]}"/>')
        body.append(t(lx + i * 90 + 18, 41, name, 13, th=th))
    for v in range(0, 36, 5):
        y = y0 + ph - v / vmax * ph
        body.append(f'<line x1="{x0}" x2="{x0 + pw}" y1="{y:.1f}" y2="{y:.1f}" stroke="{th["grid"]}" stroke-width="1"/>')
        body.append(t(x0 - 10, y + 4, str(v), 12, "muted", anchor="end", th=th))
    group = pw / len(KERNELS)
    bw = 34
    for i, (fmt, old, new) in enumerate(KERNELS):
        cx = x0 + group * i + group / 2
        for j, (val, key) in enumerate([(old, "s2"), (new, "s1")]):
            bx = cx - bw - 1 if j == 0 else cx + 1
            h = val / vmax * ph
            body.append(bar_up(bx, y0 + ph - h, bw, h, th[key]))
            body.append(t(bx + bw / 2, y0 + ph - h - 7, f"{val:.1f}", 12,
                          "text" if j else "text2", weight=600 if j else 400, anchor="middle", th=th))
        body.append(t(cx, y0 + ph + 22, fmt, 13, weight=600, anchor="middle", th=th))
        body.append(t(cx, y0 + ph + 40, f"{new / old:.1f}×", 12, "text2", anchor="middle", th=th))
    body.append(f'<line x1="{x0}" x2="{x0 + pw}" y1="{y0 + ph}" y2="{y0 + ph}" stroke="{th["border"]}" stroke-width="1"/>')
    body.append(t(28, H - 14, "Source: benchmarks/bench_results_v10_0.md §6.1 (kernels unchanged since). Four-row tiles reach 46–59 GMAC/s.",
                  11, "muted", th=th))
    return frame(W, H, th, "".join(body), "Bar chart: kernel throughput per format, V9 vs v1.0")


# --- 2. single-node decode speedup --------------------------------------------
# Source: README V10.0 findings (single-node greedy decode of the real GGUFs).
DECODE = [("Llama-3.2-1B Q4_K_M", 17, 41), ("Qwen2.5-0.5B Q4_K_M", 48, 78), ("SmolLM2-135M Q8_0", 160, 252)]


def decode(th):
    W, H = 760, 270
    x0, y0, pw = 210, 92, 400
    vmax = 3.0
    body = [t(28, 40, "Single-node decode on real checkpoints: V9 → v1.0", 19, weight=700, th=th),
            t(28, 64, "Speedup of greedy decode, tokens per second before → after", 13, "text2", th=th)]
    for v in (0, 1, 2, 3):
        x = x0 + v / vmax * pw
        dash = ' stroke-dasharray="4 3"' if v == 1 else ""
        stroke = th["border"] if v == 1 else th["grid"]
        body.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y0 - 8}" y2="{y0 + 3 * 44}" '
                    f'stroke="{stroke}" stroke-width="1"{dash}/>')
        body.append(t(x, y0 + 3 * 44 + 18, f"{v}×", 12, "muted", anchor="middle", th=th))
    for i, (name, old, new) in enumerate(DECODE):
        y = y0 + i * 44
        sp = new / old
        body.append(t(x0 - 14, y + 19, name, 13, weight=600, anchor="end", th=th))
        body.append(bar_right(x0, y + 4, sp / vmax * pw, 22, th["s1"]))
        body.append(t(x0 + sp / vmax * pw + 10, y + 20, f"{sp:.2f}×  ({old} → {new} tok/s)", 13, th=th))
    body.append(t(28, H - 14, "Source: README, V10.0 findings (kernels + exact modes). Dashed line = no change.",
                  11, "muted", th=th))
    return frame(W, H, th, "".join(body), "Bar chart: single-node decode speedup per checkpoint")


# --- 3. continuous batching ----------------------------------------------------
# Source: benchmarks/bench_results_v10_0.md Section 6.4 (Qwen2.5-0.5B on 2 H-TEP ranks).
BATCH = [(1, 73.2), (2, 90.8), (4, 117.6), (8, 125.0)]


def batching(th):
    W, H = 760, 360
    x0, y0, pw, ph = 70, 96, 660, 200
    vmax = 140.0
    body = [t(28, 40, "Continuous batching: aggregate throughput", 19, weight=700, th=th),
            t(28, 64, "Qwen2.5-0.5B Q4_K_M on 2 H-TEP ranks · every output identical to a single node", 13, "text2", th=th)]
    for v in range(0, 141, 20):
        y = y0 + ph - v / vmax * ph
        body.append(f'<line x1="{x0}" x2="{x0 + pw}" y1="{y:.1f}" y2="{y:.1f}" stroke="{th["grid"]}" stroke-width="1"/>')
        body.append(t(x0 - 10, y + 4, str(v), 12, "muted", anchor="end", th=th))
    group = pw / len(BATCH)
    bw = 70
    base = BATCH[0][1]
    for i, (n, v) in enumerate(BATCH):
        cx = x0 + group * i + group / 2
        h = v / vmax * ph
        body.append(bar_up(cx - bw / 2, y0 + ph - h, bw, h, th["s1"]))
        body.append(t(cx, y0 + ph - h - 8, f"{v:.1f} tok/s · {v / base:.2f}×", 12, weight=600, anchor="middle", th=th))
        body.append(t(cx, y0 + ph + 22, f"{n} concurrent request{'s' if n > 1 else ''}", 13, anchor="middle", th=th))
    body.append(f'<line x1="{x0}" x2="{x0 + pw}" y1="{y0 + ph}" y2="{y0 + ph}" stroke="{th["border"]}" stroke-width="1"/>')
    body.append(t(28, H - 14, "Source: benchmarks/bench_results_v10_0.md §6.4. Requests share forward passes and one all-reduce per layer.",
                  11, "muted", th=th))
    return frame(W, H, th, "".join(body), "Bar chart: aggregate tokens per second by concurrent requests")


# --- 4. architecture -----------------------------------------------------------

def box(x, y, w, h, th, fill="panel", stroke="border", r=10, dash=False):
    d = ' stroke-dasharray="5 4"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{th[fill]}" '
            f'stroke="{th[stroke]}" stroke-width="1.5"{d}/>')


def arrow(x1, y1, x2, y2, th, color="text2", label=None, lx=None, ly=None):
    s = (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{th[color]}" stroke-width="1.6" '
         f'marker-end="url(#ah-{color})"/>')
    if label:
        s += t(lx, ly, label, 11.5, "text2", anchor="middle", th=th)
    return s


def architecture(th):
    W, H = 960, 640
    b = [f'<defs>' + "".join(
        f'<marker id="ah-{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{th[k]}"/></marker>' for k in ("text2", "s1", "s2", "s3")) + '</defs>']
    b.append(t(28, 40, "tesseract-fabric at a glance", 20, weight=700, th=th))
    b.append(t(28, 64, "The solver turns measured nodes and links into one of two layouts; both keep the output exact.", 13, "text2", th=th))

    # clients
    b.append(box(330, 84, 300, 54, th))
    b.append(t(480, 107, "Clients", 14, weight=700, anchor="middle", th=th))
    b.append(t(480, 127, "OpenAI-compatible HTTP / SSE · chat CLI · dashboard", 12, "text2", anchor="middle", th=th))
    b.append(arrow(480, 138, 480, 166, th))

    # coordinator
    b.append(box(210, 168, 540, 104, th, fill="s1_soft", stroke="s1"))
    b.append(t(480, 192, "Coordinator  ·  tesseract-serve / tesseract-node", 14, weight=700, anchor="middle", th=th))
    rows = [("Discovery", "UDP 49777 beacons, SipHash-2-4, unicast seeds (LAN opt-in)"),
            ("Probe + solver", "RTT pings + 64 KB burst → HeteroClusterSolver → assignment"),
            ("Scheduling", "continuous batching · speculation planner (cost per tree size)")]
    for i, (k, v) in enumerate(rows):
        y = 216 + i * 19
        b.append(t(232, y, k, 12, weight=700, th=th))
        b.append(t(352, y, v, 12, "text2", th=th))

    b.append(arrow(380, 272, 250, 312, th))
    b.append(t(306, 284, "low-latency peers", 11.5, "text2", anchor="end", th=th))
    b.append(arrow(580, 272, 710, 312, th))
    b.append(t(654, 284, "heterogeneous links", 11.5, "text2", th=th))

    # H-TEP group (left)
    b.append(box(28, 314, 440, 214, th))
    b.append(t(48, 340, "H-TEP group (e.g. 4 × Mac Studio, TB5)", 14, weight=700, th=th))
    b.append(t(48, 360, "every rank holds a slice of every layer", 12, "text2", th=th))
    for i in range(4):
        x = 48 + i * 102
        b.append(box(x, 376, 92, 82, th, fill="s1_soft", stroke="s1", r=8))
        b.append(t(x + 46, 398, f"rank {i}", 13, weight=700, anchor="middle", th=th))
        b.append(t(x + 46, 418, "attn heads ¼", 11.5, "text2", anchor="middle", th=th))
        b.append(t(x + 46, 434, "experts / FFN ¼", 11.5, "text2", anchor="middle", th=th))
        b.append(t(x + 46, 450, "KV of its heads", 11.5, "text2", anchor="middle", th=th))
    b.append(f'<rect x="48" y="472" width="398" height="34" rx="8" fill="{th["s1"]}"/>')
    b.append(f'<text x="247" y="494" font-family="{FONT}" font-size="12.5" font-weight="700" fill="#ffffff" '
             f'text-anchor="middle">fixed-point all-reduce → bit-identical to one node</text>')
    b.append(t(48, 520, "attention is split only when aligned to quant blocks, else replicated", 11, "muted", th=th))

    # pipeline (right)
    b.append(box(492, 314, 440, 214, th))
    b.append(t(512, 340, "Pipeline (e.g. Studio + RTX over 2.5GbE)", 14, weight=700, th=th))
    b.append(t(512, 360, "stages own layer ranges; trees are verified losslessly", 12, "text2", th=th))
    b.append(box(512, 376, 180, 92, th, fill="s1_soft", stroke="s1", r=8))
    b.append(t(602, 398, "Stage 0 · Mac Studio", 13, weight=700, anchor="middle", th=th))
    b.append(t(602, 418, "layers 0–12", 11.5, "text2", anchor="middle", th=th))
    b.append(t(602, 434, "cross-step lookahead", 11.5, "text2", anchor="middle", th=th))
    b.append(t(602, 450, "+ draft-ahead drafter", 11.5, "text2", anchor="middle", th=th))
    b.append(box(732, 376, 180, 92, th, fill="s1_soft", stroke="s1", r=8))
    b.append(t(822, 398, "Stage 1 · RTX 5090", 13, weight=700, anchor="middle", th=th))
    b.append(t(822, 418, "layers 13–15 + LM head", 11.5, "text2", anchor="middle", th=th))
    b.append(t(822, 434, "verifies token trees", 11.5, "text2", anchor="middle", th=th))
    b.append(t(822, 450, "commits accepted path", 11.5, "text2", anchor="middle", th=th))
    b.append(arrow(692, 410, 730, 410, th, color="s1"))
    b.append(arrow(730, 438, 694, 438, th, color="s1"))
    b.append(box(512, 480, 400, 38, th, fill="s2_soft", stroke="s2", r=8, dash=True))
    b.append(t(712, 504, "KV page holders (laptop, phone): LSE partials, ≈ 32 KB / step", 12, anchor="middle", th=th))

    # wire
    b.append(box(28, 548, 904, 64, th, fill="s3_soft", stroke="s3"))
    b.append(t(480, 572, "TesseractWire over real TCP", 14, weight=700, anchor="middle", th=th))
    b.append(t(480, 593, "typed frames with CRC32C · multipath striping · per-link shaping · lossless or compressed activations",
               12, "text2", anchor="middle", th=th))
    return frame(W, H, th, "".join(b), "Architecture diagram of tesseract-fabric")


FIGURES = {"kernels": kernels, "decode": decode, "batching": batching, "architecture": architecture}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGURES.items():
        for mode, th in THEMES.items():
            p = OUT / f"{name}-{mode}.svg"
            p.write_text(fn(th), encoding="utf-8")
            print(p.relative_to(ROOT))


if __name__ == "__main__":
    main()
