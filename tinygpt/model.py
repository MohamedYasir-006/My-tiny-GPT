from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor


def sample_next_token(
    logits: Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")
    logits = logits / temperature
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        top_values, _ = torch.topk(logits, k, dim=-1)
        logits = logits.masked_fill(logits < top_values[:, [-1]], float("-inf"))
    probabilities = F.softmax(logits, dim=-1)
    return torch.multinomial(probabilities, num_samples=1)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self,
        idx: Tensor,
        targets: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        logits = self.token_embedding_table(idx)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            next_token = sample_next_token(logits[:, -1, :], temperature, top_k)
            idx = torch.cat((idx, next_token), dim=1)
        return idx


class Head(nn.Module):
    def __init__(
        self,
        n_embd: int,
        head_size: int,
        block_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, sequence_length, _ = x.shape
        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)

        attention_scores = queries @ keys.transpose(-2, -1)
        attention_scores = attention_scores * keys.size(-1) ** -0.5
        attention_scores = attention_scores.masked_fill(
            self.tril[:sequence_length, :sequence_length] == 0,
            float("-inf"),
        )
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        return attention_weights @ values


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        output = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.proj(output))


class FeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd, dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_layer: int = 4,
        n_head: int = 4,
        n_embd: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if block_size < 1:
            raise ValueError("block_size must be positive")
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[
                Block(n_embd, n_head, block_size, dropout)
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)
        self.lm_head.weight = self.token_embedding_table.weight
        for name, parameter in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("net.2.weight"):
                nn.init.normal_(
                    parameter,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * n_layer),
                )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: Tensor,
        targets: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        _, sequence_length = idx.shape
        if sequence_length > self.block_size:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds block_size {self.block_size}"
            )

        token_embeddings = self.token_embedding_table(idx)
        positions = torch.arange(sequence_length, device=idx.device)
        position_embeddings = self.position_embedding_table(positions)
        x = self.blocks(token_embeddings + position_embeddings)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> Tensor:
        for _ in range(max_new_tokens):
            idx_conditioned = idx[:, -self.block_size :]
            logits, _ = self(idx_conditioned)
            next_token = sample_next_token(logits[:, -1, :], temperature, top_k)
            idx = torch.cat((idx, next_token), dim=1)
        return idx
