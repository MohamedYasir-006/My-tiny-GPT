from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .checkpoint import load_checkpoint, resolve_device


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare temperature and top-k sampling side by side.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/gpt.pt"))
    parser.add_argument("--prompt", default="ROMEO:")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--temperatures", default="0.5,0.8,1.0,1.2")
    parser.add_argument("--top-k-values", default="0,10,50")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("samples/sampling_comparison.txt"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tokens < 0:
        raise ValueError("tokens must be non-negative")

    temperatures = parse_csv_floats(args.temperatures)
    top_k_values = parse_csv_ints(args.top_k_values)
    if not temperatures or not top_k_values:
        raise ValueError("Provide at least one temperature and one top-k value")
    if any(temperature <= 0 for temperature in temperatures):
        raise ValueError("All temperatures must be greater than zero")
    if any(top_k < 0 for top_k in top_k_values):
        raise ValueError("top-k values must be non-negative")

    device = resolve_device(args.device)
    model, tokenizer, payload = load_checkpoint(args.checkpoint, device)
    prompt = args.prompt or "\n"
    unknown = sorted({character for character in prompt if character not in tokenizer.stoi})
    if unknown:
        raise SystemExit(f"Prompt contains characters outside the vocabulary: {unknown!r}")

    step = payload.get("step", "unknown")
    val_loss = payload.get("val_loss")
    loss_text = f"{val_loss:.4f}" if isinstance(val_loss, float) else "unknown"
    lines = [
        f"Checkpoint: {args.checkpoint}",
        f"Step: {step} | val loss: {loss_text}",
        f"Prompt: {prompt!r} | new tokens per sample: {args.tokens}",
        "",
    ]

    for temperature in temperatures:
        for top_k_value in top_k_values:
            top_k = top_k_value if top_k_value > 0 else None
            torch.manual_seed(args.seed)
            context = torch.tensor(
                [tokenizer.encode(prompt)],
                dtype=torch.long,
                device=device,
            )
            generated = model.generate(
                context,
                max_new_tokens=args.tokens,
                temperature=temperature,
                top_k=top_k,
            )
            text = tokenizer.decode(generated[0])
            header = (
                f"temperature={temperature:g} | "
                f"top_k={'none' if top_k is None else top_k}"
            )
            lines.extend([header, "-" * len(header), text, ""])
            print(header)
            print("-" * len(header))
            print(text)
            print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved comparison to {args.out}")


if __name__ == "__main__":
    main()
