from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .checkpoint import load_checkpoint, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text from a trained tiny GPT checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/gpt.pt"))
    parser.add_argument("--prompt", default="ROMEO:")
    parser.add_argument("--tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tokens < 0:
        raise ValueError("tokens must be non-negative")
    if args.temperature <= 0:
        raise ValueError("temperature must be greater than zero")

    device = resolve_device(args.device)
    model, tokenizer, payload = load_checkpoint(args.checkpoint, device)
    torch.manual_seed(args.seed)

    prompt = args.prompt or "\n"
    unknown = sorted({character for character in prompt if character not in tokenizer.stoi})
    if unknown:
        raise SystemExit(f"Prompt contains characters outside the vocabulary: {unknown!r}")

    context = torch.tensor(
        [tokenizer.encode(prompt)],
        dtype=torch.long,
        device=device,
    )
    generated = model.generate(
        context,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    text = tokenizer.decode(generated[0])

    step = payload.get("step", "unknown")
    val_loss = payload.get("val_loss")
    loss_text = f"{val_loss:.4f}" if isinstance(val_loss, float) else "unknown"
    print(f"Checkpoint: {args.checkpoint} | step: {step} | val loss: {loss_text}")
    print(f"temperature: {args.temperature} | top-k: {args.top_k}")
    print("-" * 80)
    print(text)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
