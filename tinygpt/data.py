from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import torch
from torch import Tensor

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)


def load_text(path: str | Path, download: bool = True) -> str:
    path = Path(path)
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"Dataset not found: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        print(f"Downloading Tiny Shakespeare to {path} ...")
        urlretrieve(TINY_SHAKESPEARE_URL, temporary_path)
        temporary_path.replace(path)
    return path.read_text(encoding="utf-8")


class CharTokenizer:
    def __init__(self, chars: tuple[str, ...] | list[str]) -> None:
        self.chars = tuple(chars)
        if not self.chars:
            raise ValueError("Tokenizer needs at least one character")
        if len(set(self.chars)) != len(self.chars):
            raise ValueError("Tokenizer characters must be unique")
        self.stoi = {char: index for index, char in enumerate(self.chars)}
        self.itos = {index: char for index, char in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        if not text:
            raise ValueError("Cannot build a tokenizer from empty text")
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[char] for char in text]

    def decode(self, token_ids: list[int] | Tensor) -> str:
        if isinstance(token_ids, Tensor):
            token_ids = token_ids.tolist()
        return "".join(self.itos[int(token_id)] for token_id in token_ids)


def prepare_dataset(
    path: str | Path,
    train_fraction: float = 0.9,
    download: bool = True,
) -> tuple[CharTokenizer, Tensor, Tensor]:
    text = load_text(path, download=download)
    tokenizer = CharTokenizer.from_text(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split_index = int(train_fraction * data.numel())
    if split_index <= 0 or split_index >= data.numel():
        raise ValueError("train_fraction must leave non-empty train and validation splits")
    return tokenizer, data[:split_index], data[split_index:]


def get_batch(
    data: Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device | str,
) -> tuple[Tensor, Tensor]:
    if data.numel() <= block_size:
        raise ValueError("Dataset split must be longer than block_size")
    starts = torch.randint(0, data.numel() - block_size, (batch_size,))
    x = torch.stack([data[start : start + block_size] for start in starts])
    y = torch.stack([data[start + 1 : start + block_size + 1] for start in starts])
    return x.to(device), y.to(device)
