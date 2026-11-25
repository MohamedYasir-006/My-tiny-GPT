from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .data import CharTokenizer
from .model import BigramLanguageModel, GPT


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(model_type: str, model_args: dict[str, Any]) -> nn.Module:
    if model_type == "bigram":
        return BigramLanguageModel(**model_args)
    if model_type == "gpt":
        return GPT(**model_args)
    raise ValueError(f"Unknown model type: {model_type}")


def save_checkpoint(
    path: str | Path,
    model_type: str,
    model_args: dict[str, Any],
    tokenizer: CharTokenizer,
    model: nn.Module,
    step: int,
    val_loss: float,
    train_args: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_type": model_type,
        "model_args": model_args,
        "chars": tokenizer.chars,
        "model_state": model.state_dict(),
        "step": step,
        "val_loss": val_loss,
        "train_args": train_args or {},
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    device: torch.device,
) -> tuple[nn.Module, CharTokenizer, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = build_model(payload["model_type"], payload["model_args"])
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    tokenizer = CharTokenizer(payload["chars"])
    return model, tokenizer, payload
