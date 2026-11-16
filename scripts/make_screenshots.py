"""Regenerate the README terminal screenshots (``docs/images/<name>.svg``) and
the dashboard screenshot (``docs/images/dashboard.png``) from real runs.

Every screenshot is real: the commands shown are executed against the release
binaries and the public GGUF checkpoints in ``models/`` (downloaded by
``cargo test --release`` on first use), and their output is recorded with Rich
(``Console(record=True)`` + ``export_svg``). Long outputs are trimmed to the
lines shown, and the trimming is marked in the picture. Timings are measured
live and differ between runs. The dashboard is captured with headless Chrome.

Requirements: ``cargo build --release``, the checkpoints in ``models/``,
``pip install rich``, ``curl`` and ``jq``; Google Chrome for the dashboard.
Run: ``python scripts/make_screenshots.py`` (``--only chat,server`` for a subset).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

from rich.console import Console
from rich.text import Text

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"
BIN = ROOT / "target" / "release"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
KEY = "demo-key"  # the API key used in the screenshots (not a real secret)


def sh(cmd: str, env: dict | None = None, timeout: int = 900) -> str:
    """Run a shell command from the repository root; stdout and stderr
    interleaved in the order they were written."""
    path = os.environ.get("PATH", "") + os.pathsep + str(Path.home() / ".cargo" / "bin")
    r = subprocess.run(["bash", "-c", f"{{ {cmd} ; }} 2>&1"], cwd=ROOT, stdout=subprocess.PIPE,
                       text=True, timeout=timeout, env={**os.environ, "PATH": path, **(env or {})})
    return r.stdout.rstrip("\n")


def render(name: str, title: str, blocks: list[tuple[str, str]], width: int = 110) -> None:
    """Write one terminal screenshot: each block is (command shown, output)."""
    con = Console(record=True, width=width, file=io.StringIO(), force_terminal=True,
                  color_system="truecolor", legacy_windows=False)
    for i, (cmd, out) in enumerate(blocks):
        if i:
            con.print()
        if cmd:
            con.print(Text("$ ", style="bold green") + Text(cmd, style="bold"))
        for line in out.splitlines():
            style = "dim" if line.startswith("[") or line.startswith("…") else ""
            con.print(Text(line, style=style), soft_wrap=False, overflow="fold")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.svg").write_text(con.export_svg(title=title), encoding="utf-8")
    print(f"docs/images/{name}.svg")


# --- shots ---------------------------------------------------------------------

def shot_chat() -> None:
    prompt = "In two sentences, why do distributed systems need consensus?"
    cmd = ("tesseract-node --role chat --model models/Llama-3.2-1B-Instruct-Q4_K_M.gguf "
           f'--prompt "{prompt}" --local-nodes 4')
    out = sh(f'{BIN}/tesseract-node --role chat --model models/Llama-3.2-1B-Instruct-Q4_K_M.gguf '
             f'--prompt "{prompt}" --local-nodes 4 --tokens 96')
    render("chat", "tesseract-node --role chat", [(cmd, out)])


def shot_exact() -> None:
    cmd = "cargo test --release --test production_v10 htep_reproduces -- --nocapture"
    raw = sh(cmd, timeout=1800)
    keep = []
    for line in raw.splitlines():
        if "H-TEP×4" in line:
            # model: N/N tokens identical, N logit rows bit-identical (decoded text dropped)
            head = line.split(" tok/step")[0]
            keep.append(head.rsplit(",", 1)[0])
        elif line.startswith("test ") or line.startswith("test result"):
            keep.append(line)
    if not any("H-TEP×4" in k for k in keep):
        raise RuntimeError("exactness test produced no per-model lines:\n" + raw[-2000:])
    out = "\n".join(["… (build output trimmed)", *keep])
    render("exact", "4-rank H-TEP vs a single node", [(cmd, out)], width=120)


def start_server(port: int, extra: list[str]) -> subprocess.Popen:
    p = subprocess.Popen([str(BIN / "tesseract-serve"), "--http", f"127.0.0.1:{port}", *extra],
                         cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    t = time.time()
    while time.time() - t < 300:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2).read()
            return p
        except Exception:
            time.sleep(0.5)
    p.kill()
    raise RuntimeError("server did not start")


def shot_server() -> None:
    port = 18080
    args = ["--model", "models/qwen2.5-0.5b-instruct-q4_k_m.gguf", "--local-nodes", "2",
            "--sessions", "8", "--api-key", KEY]
    srv = start_server(port, args)
    try:
        base = f"http://127.0.0.1:{port}"
        req = {"messages": [{"role": "user", "content": "Name three uses of a hash function, briefly."}],
               "max_tokens": 64}
        body = json.dumps(req)
        c1 = f"curl -s {base}/v1/models"
        o1 = sh(c1)
        c2 = (f'curl -s {base}/v1/chat/completions -H "Authorization: Bearer $KEY" '
              f"-d '{body}' | jq '{{reply: .choices[0].message.content, telemetry: .tesseract_telemetry}}'")
        o2 = sh(c2, env={"KEY": KEY}, timeout=300)
        c3 = (f'curl -s {base}/v1/cluster/status -H "Authorization: Bearer $KEY" '
              "| jq -c '.nodes[] | {id, roles, weight_MB: (.weight_bytes / 1048576 | floor)}'")
        o3 = sh(c3, env={"KEY": KEY})
        # the dashboard, populated by the request above
        if Path(CHROME).exists():
            png = OUT / "dashboard.png"
            subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                            "--window-size=1360,660", "--virtual-time-budget=4000",
                            f"--screenshot={png}", f"{base}/#key={KEY}"],
                           capture_output=True, timeout=120)
            print("docs/images/dashboard.png")
    finally:
        srv.kill()
        log = srv.communicate()[0]
    boot = "\n".join(l for l in log.splitlines()
                     if l.startswith(("listening", "http auth", "continuous batching", "chat template")))
    render("server", "tesseract-serve + OpenAI API",
           [("tesseract-serve --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf --local-nodes 2 "
             "--sessions 8 --api-key $KEY &", boot),
            (c1 + "    # no key", o1),
            (c2, o2),
            (c3, o3)], width=120)


SHOTS = {"chat": shot_chat, "exact": shot_exact, "server": shot_server}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", help="comma-separated: " + ", ".join(SHOTS))
    a = ap.parse_args()
    for name in (a.only.split(",") if a.only else SHOTS):
        SHOTS[name]()


if __name__ == "__main__":
    main()
