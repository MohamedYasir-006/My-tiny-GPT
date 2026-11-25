from __future__ import annotations

import argparse
import json
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch

from .checkpoint import load_checkpoint, resolve_device
from .model import count_parameters

HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tiny GPT - Shakespeare</title>
<style>
:root {
  color-scheme: dark;
  --bg: #090d18;
  --panel: #10182b;
  --panel-2: #0c1425;
  --line: #24304d;
  --text: #e8ecf7;
  --muted: #8e9bb9;
  --gold: #e9c46a;
  --gold-soft: #f2d78f;
  --danger: #ff8e8e;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  min-height: 100vh;
  background:
    radial-gradient(circle at 15% 0%, rgba(90, 76, 160, 0.25), transparent 32rem),
    radial-gradient(circle at 100% 10%, rgba(198, 146, 57, 0.12), transparent 28rem),
    var(--bg);
  color: var(--text);
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
.shell { width: min(95%, 1600px); margin-inline: auto; padding: 36px 0 56px; }
header { display: flex; gap: 24px; justify-content: space-between; align-items: flex-end; margin-bottom: 26px; }
.intro { min-width: 0; }
.eyebrow { color: var(--gold); letter-spacing: 0.22em; font-size: 12px; margin-bottom: 8px; }
h1 { font-size: clamp(30px, 5vw, 58px); line-height: 0.98; margin: 0; letter-spacing: -0.045em; }
.subtitle { max-width: 760px; color: var(--muted); line-height: 1.6; margin: 14px 0 0; font-family: "Segoe UI", sans-serif; }
.stats { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 8px; max-width: 560px; }
.chip { border: 1px solid var(--line); background: rgba(16, 24, 43, 0.7); border-radius: 999px; padding: 7px 11px; color: var(--muted); font-size: 11px; white-space: nowrap; }
.chip strong { color: var(--text); font-weight: 600; }
.layout { display: grid; grid-template-columns: minmax(0, 0.8fr) minmax(0, 1.2fr); gap: clamp(14px, 1.1vw, 22px); align-items: start; }
.panel { min-width: 0; background: linear-gradient(180deg, rgba(19, 28, 50, 0.96), rgba(12, 20, 37, 0.96)); border: 1px solid var(--line); border-radius: 18px; padding: 20px; box-shadow: 0 20px 60px rgba(0, 0, 0, 0.28); }
.panel-title { margin: 0 0 14px; font-size: 12px; letter-spacing: 0.16em; color: var(--muted); text-transform: uppercase; }
label { display: block; color: var(--muted); font-size: 12px; margin-bottom: 8px; }
textarea, input[type="number"] {
  width: 100%;
  border: 1px solid var(--line);
  background: var(--panel-2);
  color: var(--text);
  border-radius: 10px;
  font: inherit;
  outline: none;
}
textarea { min-height: 110px; resize: vertical; padding: 13px 14px; line-height: 1.6; }
textarea:focus, input[type="number"]:focus { border-color: var(--gold); box-shadow: 0 0 0 3px rgba(233, 196, 106, 0.12); }
.examples { display: flex; flex-wrap: wrap; gap: 7px; margin: 10px 0 18px; }
.examples button { border: 1px solid var(--line); background: transparent; color: var(--muted); border-radius: 999px; padding: 6px 10px; font: inherit; font-size: 11px; cursor: pointer; }
.examples button:hover { border-color: var(--gold); color: var(--text); }
.controls { display: grid; grid-template-columns: 1fr 1fr; gap: 14px 12px; margin-top: 4px; }
.range-row { display: flex; align-items: center; gap: 10px; }
input[type="range"] { width: 100%; accent-color: var(--gold); }
output { color: var(--gold-soft); font-size: 12px; min-width: 40px; text-align: right; }
.hint { color: #657293; font-size: 10px; margin-top: 4px; }
.actions { display: flex; gap: 10px; align-items: center; margin-top: 20px; }
.primary {
  border: 0;
  border-radius: 10px;
  background: var(--gold);
  color: #191307;
  font: inherit;
  font-weight: 700;
  padding: 12px 18px;
  cursor: pointer;
}
.primary:hover { background: var(--gold-soft); }
.primary:disabled { opacity: 0.55; cursor: wait; }
.status { color: var(--muted); font-size: 11px; min-height: 16px; }
.status.error { color: var(--danger); }
.output-head { display: flex; justify-content: space-between; gap: 16px; align-items: center; margin-bottom: 12px; }
.output-meta { color: var(--muted); font-size: 11px; }
pre {
  margin: 0;
  min-height: clamp(460px, 56vh, 680px);
  max-height: 68vh;
  overflow: auto;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  line-height: 1.72;
  font: inherit;
  font-size: 14px;
  background: rgba(6, 11, 22, 0.55);
  border: 1px solid rgba(36, 48, 77, 0.7);
  border-radius: 12px;
  padding: 18px;
}
pre.placeholder { color: #657293; }
footer { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 6px 20px; color: #5e6a88; font-size: 10px; margin-top: 18px; padding: 0 4px; }
@media (min-width: 1440px) {
  .shell { width: min(94%, 1600px); }
}
@media (max-width: 1100px) {
  header { display: block; }
  .stats { justify-content: flex-start; margin-top: 18px; max-width: 640px; }
}
@media (max-width: 1023px) {
  .shell { width: calc(100% - 40px); }
  .layout { grid-template-columns: minmax(0, 0.85fr) minmax(0, 1.15fr); gap: 14px; }
  .panel { padding: 18px; }
  .controls { grid-template-columns: 1fr; }
  .subtitle { max-width: 680px; }
}
@media (max-width: 767px) {
  .shell { width: calc(100% - 24px); padding: 24px 0 40px; }
  header { margin-bottom: 20px; }
  .stats { max-width: none; margin-top: 16px; }
  .layout { grid-template-columns: 1fr; gap: 12px; }
  .panel { padding: 16px; border-radius: 16px; }
  .controls { grid-template-columns: 1fr 1fr; }
  pre { min-height: 360px; max-height: 60vh; }
}
</style>
</head>
<body>
<div class="shell">
  <header>
    <div class="intro">
      <div class="eyebrow">TINY GPT / CHARACTER-LEVEL</div>
      <h1>Shakespeare machine.</h1>
      <p class="subtitle">A transformer trained from scratch on 1.1M characters of Shakespeare. Pick a prompt, tune the sampler, and watch it invent dialogue.</p>
    </div>
    <div class="stats" id="stats"></div>
  </header>
  <div class="layout">
    <section class="panel prompt-panel">
      <h2 class="panel-title">Prompt</h2>
      <label for="prompt">Starting text</label>
      <textarea id="prompt" spellcheck="false">ROMEO:</textarea>
      <div class="examples">
        <button type="button" data-prompt="ROMEO:">ROMEO:</button>
        <button type="button" data-prompt="KING RICHARD III:">KING RICHARD III:</button>
        <button type="button" data-prompt="QUEEN MARGARET:">QUEEN MARGARET:</button>
        <button type="button" data-prompt="To be, or not to be">To be, or not to be</button>
      </div>
      <div class="controls">
        <div>
          <label for="temperature">Temperature</label>
          <div class="range-row">
            <input id="temperature" type="range" min="0.1" max="1.5" step="0.05" value="0.8">
            <output id="temperatureValue" for="temperature">0.80</output>
          </div>
          <div class="hint">Lower is safer, higher is stranger.</div>
        </div>
        <div>
          <label for="topK">Top-k</label>
          <input id="topK" type="number" min="0" max="65" step="1" value="50">
          <div class="hint">0 disables filtering.</div>
        </div>
        <div>
          <label for="tokens">New tokens</label>
          <input id="tokens" type="number" min="20" max="1000" step="10" value="300">
          <div class="hint">Generated after the prompt.</div>
        </div>
        <div>
          <label for="seed">Seed</label>
          <input id="seed" type="number" min="0" step="1" value="1337">
          <div class="hint">Same seed = same sample.</div>
        </div>
      </div>
      <div class="actions">
        <button class="primary" id="generate" type="button">Generate</button>
        <span class="status" id="status"></span>
      </div>
    </section>
    <section class="panel output-panel">
      <div class="output-head">
        <h2 class="panel-title" style="margin:0">Output</h2>
        <span class="output-meta" id="outputMeta">ready</span>
      </div>
      <pre id="output" class="placeholder">Press Generate to sample from the model.</pre>
    </section>
  </div>
  <footer>
    <span>CPU inference, local only. Nothing leaves this machine.</span>
    <span>Ctrl + Enter to generate</span>
  </footer>
</div>
<script>
const promptInput = document.getElementById("prompt");
const temperatureInput = document.getElementById("temperature");
const temperatureValue = document.getElementById("temperatureValue");
const topKInput = document.getElementById("topK");
const tokensInput = document.getElementById("tokens");
const seedInput = document.getElementById("seed");
const generateButton = document.getElementById("generate");
const statusElement = document.getElementById("status");
const outputElement = document.getElementById("output");
const outputMeta = document.getElementById("outputMeta");
const statsElement = document.getElementById("stats");

function addChip(label, value) {
  const chip = document.createElement("span");
  chip.className = "chip";
  const strong = document.createElement("strong");
  strong.textContent = value;
  chip.append(label + " ", strong);
  statsElement.append(chip);
}

async function loadInfo() {
  try {
    const response = await fetch("/api/info");
    const info = await response.json();
    addChip("params", Number(info.parameters).toLocaleString());
    addChip("val loss", Number(info.val_loss).toFixed(4));
    addChip("step", Number(info.step).toLocaleString());
    addChip("vocab", info.vocab_size);
    addChip("device", info.device);
  } catch (error) {
    statsElement.textContent = "Could not load checkpoint info.";
  }
}

async function generate() {
  generateButton.disabled = true;
  statusElement.className = "status";
  statusElement.textContent = "Sampling...";
  outputMeta.textContent = "working";
  const body = {
    prompt: promptInput.value,
    temperature: Number(temperatureInput.value),
    top_k: Number(topKInput.value),
    tokens: Number(tokensInput.value),
    seed: Number(seedInput.value)
  };
  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Generation failed");
    outputElement.classList.remove("placeholder");
    outputElement.textContent = data.text;
    outputMeta.textContent = data.elapsed.toFixed(2) + "s / " + data.text.length + " chars";
    statusElement.textContent = "";
  } catch (error) {
    statusElement.className = "status error";
    statusElement.textContent = error.message;
    outputMeta.textContent = "ready";
  } finally {
    generateButton.disabled = false;
  }
}

temperatureInput.addEventListener("input", () => {
  temperatureValue.textContent = Number(temperatureInput.value).toFixed(2);
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    promptInput.value = button.dataset.prompt;
    promptInput.focus();
  });
});

generateButton.addEventListener("click", generate);
document.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault();
    generate();
  }
});

loadInfo();
</script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a local browser demo for a trained tiny GPT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/gpt.pt"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-open", action="store_true")
    return parser.parse_args()


def make_handler(
    model: torch.nn.Module,
    tokenizer: Any,
    info: dict[str, Any],
    device: torch.device,
    generation_lock: threading.Lock,
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "TinyGPT/0.1"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def send_json(self, data: dict[str, Any], status: int = HTTPStatus.OK) -> None:
            body = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_html(self, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/":
                self.send_html(HTML_PAGE)
            elif self.path == "/api/info":
                self.send_json(info)
            elif self.path == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
            else:
                self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if self.path != "/api/generate":
                self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length <= 0 or content_length > 1_000_000:
                    raise ValueError("Request body is missing or too large")
                payload = json.loads(self.rfile.read(content_length))
                if not isinstance(payload, dict):
                    raise ValueError("Request body must be a JSON object")

                prompt = str(payload.get("prompt", "ROMEO:"))
                if len(prompt) > 500:
                    raise ValueError("Prompt must be at most 500 characters")
                if not prompt:
                    prompt = "\n"
                unknown = sorted(
                    {character for character in prompt if character not in tokenizer.stoi}
                )
                if unknown:
                    raise ValueError(
                        f"Prompt contains characters outside the vocabulary: {unknown!r}"
                    )

                tokens = int(payload.get("tokens", 300))
                temperature = float(payload.get("temperature", 0.8))
                top_k = int(payload.get("top_k", 50))
                seed = int(payload.get("seed", 1337))
                if not 1 <= tokens <= 1000:
                    raise ValueError("tokens must be between 1 and 1000")
                if not 0.05 <= temperature <= 2.0:
                    raise ValueError("temperature must be between 0.05 and 2.0")
                if not 0 <= top_k <= tokenizer.vocab_size:
                    raise ValueError(
                        f"top_k must be between 0 and {tokenizer.vocab_size}"
                    )
                if not 0 <= seed <= 2**31 - 1:
                    raise ValueError("seed must be between 0 and 2147483647")

                with generation_lock:
                    torch.manual_seed(seed)
                    context = torch.tensor(
                        [tokenizer.encode(prompt)],
                        dtype=torch.long,
                        device=device,
                    )
                    start = time.perf_counter()
                    generated = model.generate(
                        context,
                        max_new_tokens=tokens,
                        temperature=temperature,
                        top_k=top_k if top_k > 0 else None,
                    )
                    text = tokenizer.decode(generated[0])
                    elapsed = time.perf_counter() - start

                self.send_json(
                    {
                        "text": text,
                        "elapsed": elapsed,
                        "prompt": prompt,
                        "tokens": tokens,
                    }
                )
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            except Exception as error:
                self.send_json(
                    {"error": f"Generation failed: {error}"},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )

    return Handler


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    if not 0 < args.port < 65536:
        raise SystemExit("port must be between 1 and 65535")

    device = resolve_device(args.device)
    model, tokenizer, checkpoint = load_checkpoint(args.checkpoint, device)
    model.eval()
    model_args = checkpoint.get("model_args", {})
    info = {
        "checkpoint": str(args.checkpoint),
        "model_type": checkpoint.get("model_type", "unknown"),
        "step": checkpoint.get("step"),
        "val_loss": checkpoint.get("val_loss"),
        "parameters": count_parameters(model),
        "vocab_size": tokenizer.vocab_size,
        "device": str(device),
        "model_args": model_args,
    }

    handler = make_handler(
        model,
        tokenizer,
        info,
        device,
        threading.Lock(),
    )
    try:
        server = ThreadingHTTPServer((args.host, args.port), handler)
    except OSError as error:
        raise SystemExit(
            f"Could not start server on {args.host}:{args.port}: {error}"
        )
    url = f"http://{args.host}:{args.port}"
    print(f"Tiny GPT demo: {url}")
    print(f"Checkpoint: {args.checkpoint} | device: {device}")
    print("Press Ctrl+C to stop.")
    if not args.no_open:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
