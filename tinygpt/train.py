from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .checkpoint import build_model, load_checkpoint, resolve_device, save_checkpoint
from .data import get_batch, prepare_dataset
from .model import count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a character-level bigram or GPT model on Tiny Shakespeare.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=["bigram", "gpt"], default="gpt")
    parser.add_argument("--data", type=Path, default=Path("data/input.txt"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    parser.add_argument("--samples-dir", type=Path, default=Path("samples"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--no-schedule", action="store_true")
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--sample-every", type=int, default=1000)
    parser.add_argument("--sample-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Continue training from a checkpoint up to --max-steps.",
    )
    return parser.parse_args()


def read_initial_val_loss(metrics_path: Path) -> float | None:
    if not metrics_path.exists():
        return None
    with metrics_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = row.get("val_loss")
            if value is not None:
                return float(value)
    return None


def get_learning_rate(
    step: int,
    max_steps: int,
    base_lr: float,
    warmup_steps: int,
    scheduled: bool,
) -> float:
    if not scheduled or max_steps <= 0:
        return base_lr
    warmup = max(1, min(warmup_steps, max_steps))
    minimum_lr = base_lr * 0.1
    if step < warmup:
        return base_lr * (step + 1) / warmup
    if step >= max_steps:
        return minimum_lr
    progress = (step - warmup) / (max_steps - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_lr + cosine * (base_lr - minimum_lr)


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    data: dict[str, torch.Tensor],
    block_size: int,
    batch_size: int,
    eval_iters: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    results = {}
    for split_name, split_data in data.items():
        losses = torch.zeros(eval_iters)
        for index in range(eval_iters):
            x, y = get_batch(split_data, block_size, batch_size, device)
            _, loss = model(x, y)
            assert loss is not None
            losses[index] = loss.item()
        results[split_name] = losses.mean().item()
    model.train()
    return results


def generate_text(
    model: nn.Module,
    tokenizer: Any,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
) -> str:
    was_training = model.training
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(
        context,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    text = tokenizer.decode(generated[0])
    if was_training:
        model.train()
    return text


def save_sample(
    samples_dir: Path,
    run_name: str,
    step: int,
    text: str,
) -> Path:
    samples_dir.mkdir(parents=True, exist_ok=True)
    path = samples_dir / f"{run_name}_step_{step:06d}.txt"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    if args.eval_interval <= 0:
        raise ValueError("eval_interval must be positive")
    if args.eval_iters <= 0:
        raise ValueError("eval_iters must be positive")
    if args.sample_tokens < 0:
        raise ValueError("sample_tokens must be non-negative")

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, train_data, val_data = prepare_dataset(
        args.data,
        train_fraction=args.train_fraction,
    )
    print(f"Device: {device}")
    print(
        f"Vocabulary: {tokenizer.vocab_size} | "
        f"train tokens: {train_data.numel():,} | "
        f"validation tokens: {val_data.numel():,}"
    )

    resume_payload = None
    if args.resume_from is not None:
        if not args.resume_from.exists():
            raise SystemExit(f"Checkpoint not found: {args.resume_from}")
        model, checkpoint_tokenizer, resume_payload = load_checkpoint(
            args.resume_from,
            device,
        )
        checkpoint_model_type = resume_payload["model_type"]
        if args.model != checkpoint_model_type:
            print(f"Using checkpoint model type: {checkpoint_model_type}")
            args.model = checkpoint_model_type
        model_args = resume_payload["model_args"]
        if checkpoint_tokenizer.chars != tokenizer.chars:
            raise SystemExit("Checkpoint tokenizer does not match the dataset")
        tokenizer = checkpoint_tokenizer
        model.train()
        start_step = int(resume_payload.get("step", 0))
        if start_step >= args.max_steps:
            raise SystemExit(
                f"Checkpoint is already at step {start_step}; "
                f"choose --max-steps greater than {start_step}"
            )
        args.block_size = int(model_args.get("block_size", args.block_size))
        print(
            f"Resuming {args.resume_from} at step {start_step} "
            f"(checkpoint val loss {resume_payload.get('val_loss')})"
        )
    else:
        start_step = 0
        if args.model == "gpt":
            model_args = {
                "vocab_size": tokenizer.vocab_size,
                "block_size": args.block_size,
                "n_layer": args.n_layer,
                "n_head": args.n_head,
                "n_embd": args.n_embd,
                "dropout": args.dropout,
            }
        else:
            model_args = {"vocab_size": tokenizer.vocab_size}
        model = build_model(args.model, model_args).to(device)

    run_name = args.run_name or args.model
    parameter_count = count_parameters(model)
    print(f"Parameters: {parameter_count:,} ({parameter_count / 1e6:.3f}M)")

    saved_lr = None
    if resume_payload is not None:
        saved_lr = resume_payload.get("train_args", {}).get("lr")
    base_lr = args.lr
    if base_lr is None:
        base_lr = saved_lr
    if base_lr is None:
        base_lr = 1e-2 if args.model == "bigram" else 3e-4
    weight_decay = 0.0 if args.model == "bigram" else args.weight_decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
    )

    scheduled = args.model == "gpt" and not args.no_schedule
    data = {"train": train_data, "val": val_data}
    metrics_path = out_dir / f"{run_name}_metrics.csv"
    config_path = out_dir / f"{run_name}_config.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "args": vars(args) | {"data": str(args.data), "out_dir": str(args.out_dir)},
                "model_type": args.model,
                "model_args": model_args,
                "parameters": parameter_count,
                "base_lr": base_lr,
            },
            handle,
            indent=2,
            default=str,
        )

    best_val_loss = float("inf")
    best_step = -1
    if resume_payload is not None:
        best_val_loss = float(resume_payload.get("val_loss", float("inf")))
        best_step = start_step
    initial_val_loss = read_initial_val_loss(metrics_path) if resume_payload is not None else None
    if resume_payload is not None and initial_val_loss is None:
        initial_val_loss = float(resume_payload.get("val_loss", float("nan")))
    last_losses = {"train": float("nan"), "val": float("nan")}
    start_time = time.perf_counter()
    loop_start = start_step + 1 if resume_payload is not None else 0
    append_metrics = resume_payload is not None and metrics_path.exists()

    if resume_payload is not None:
        print(f"Continuing {run_name} from step {start_step} to {args.max_steps} ...")
    else:
        print(f"Training {run_name} for {args.max_steps} steps ...")
    with metrics_path.open("a" if append_metrics else "w", newline="", encoding="utf-8") as metrics_handle:
        writer = csv.writer(metrics_handle)
        if not append_metrics:
            writer.writerow(
                ["step", "train_loss", "val_loss", "learning_rate", "elapsed_seconds"]
            )

        for step in range(loop_start, args.max_steps + 1):
            learning_rate = get_learning_rate(
                step,
                args.max_steps,
                base_lr,
                args.warmup_steps,
                scheduled,
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate

            if step % args.eval_interval == 0 or step == args.max_steps:
                losses = estimate_loss(
                    model,
                    data,
                    args.block_size,
                    args.batch_size,
                    args.eval_iters,
                    device,
                )
                last_losses = losses
                elapsed = time.perf_counter() - start_time
                if initial_val_loss is None:
                    initial_val_loss = losses["val"]
                if losses["val"] < best_val_loss:
                    best_val_loss = losses["val"]
                    best_step = step
                    save_checkpoint(
                        out_dir / f"{run_name}.pt",
                        args.model,
                        model_args,
                        tokenizer,
                        model,
                        step,
                        losses["val"],
                        vars(args),
                    )
                writer.writerow(
                    [
                        step,
                        f"{losses['train']:.6f}",
                        f"{losses['val']:.6f}",
                        f"{learning_rate:.8f}",
                        f"{elapsed:.2f}",
                    ]
                )
                metrics_handle.flush()
                print(
                    f"step {step:5d} | train {losses['train']:.4f} | "
                    f"val {losses['val']:.4f} | lr {learning_rate:.2e} | "
                    f"elapsed {elapsed:6.1f}s"
                )

                if (
                    args.sample_tokens > 0
                    and (step % args.sample_every == 0 or step == args.max_steps)
                ):
                    text = generate_text(
                        model,
                        tokenizer,
                        device,
                        args.sample_tokens,
                        args.temperature,
                        args.top_k,
                    )
                    sample_path = save_sample(
                        Path(args.samples_dir),
                        run_name,
                        step,
                        text,
                    )
                    print(f"sample at step {step} ({sample_path}):")
                    print(text)
                    print("-" * 80)

            if step == args.max_steps:
                break

            x, y = get_batch(train_data, args.block_size, args.batch_size, device)
            _, loss = model(x, y)
            assert loss is not None
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % args.log_interval == 0:
                print(
                    f"step {step:5d} | train {loss.item():.4f} | "
                    f"lr {learning_rate:.2e}"
                )

    save_checkpoint(
        out_dir / f"{run_name}_final.pt",
        args.model,
        model_args,
        tokenizer,
        model,
        args.max_steps,
        last_losses["val"],
        vars(args),
    )

    initial = initial_val_loss if initial_val_loss is not None else float("nan")
    print(
        f"Best validation loss: {best_val_loss:.4f} at step {best_step} "
        f"(initial {initial:.4f})"
    )
    print(f"Checkpoints: {out_dir / f'{run_name}.pt'} and {out_dir / f'{run_name}_final.pt'}")
    print(f"Metrics: {metrics_path}")
    if args.model == "gpt":
        print(
            "Resume line: Built and trained a "
            f"{parameter_count / 1e6:.2f}M-parameter character-level transformer "
            f"from scratch in PyTorch, reducing validation loss from {initial:.2f} "
            f"to {best_val_loss:.2f} and generating coherent Shakespeare-style text."
        )
    else:
        print(
            f"Bigram baseline: {parameter_count:,} parameters, validation loss "
            f"from {initial:.2f} to {best_val_loss:.2f}."
        )


if __name__ == "__main__":
    main()
