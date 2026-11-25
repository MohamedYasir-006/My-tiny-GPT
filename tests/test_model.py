from __future__ import annotations

import unittest

import torch

from tinygpt.data import CharTokenizer, get_batch
from tinygpt.model import BigramLanguageModel, GPT, count_parameters, sample_next_token


class TokenizerTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        tokenizer = CharTokenizer.from_text("hello\nworld")
        text = "hello\nworld"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        self.assertEqual(tokenizer.vocab_size, 8)


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_bigram_shapes_and_loss(self) -> None:
        model = BigramLanguageModel(vocab_size=11)
        x = torch.randint(0, 11, (2, 8))
        y = torch.randint(0, 11, (2, 8))
        logits, loss = model(x, y)
        self.assertEqual(logits.shape, (2, 8, 11))
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))

    def test_gpt_is_causal(self) -> None:
        model = GPT(
            vocab_size=11,
            block_size=8,
            n_layer=2,
            n_head=2,
            n_embd=16,
            dropout=0.0,
        ).eval()
        x = torch.randint(0, 11, (2, 8))
        changed = x.clone()
        changed[:, 5:] = torch.randint(0, 11, changed[:, 5:].shape)
        with torch.no_grad():
            original_logits, _ = model(x)
            changed_logits, _ = model(changed)
        torch.testing.assert_close(original_logits[:, :5], changed_logits[:, :5])

    def test_gpt_shapes_and_weight_tying(self) -> None:
        model = GPT(
            vocab_size=13,
            block_size=16,
            n_layer=2,
            n_head=2,
            n_embd=16,
            dropout=0.0,
        )
        x = torch.randint(0, 13, (3, 16))
        logits, loss = model(x)
        self.assertEqual(logits.shape, (3, 16, 13))
        self.assertIsNone(loss)
        self.assertIs(model.lm_head.weight, model.token_embedding_table.weight)
        self.assertGreater(count_parameters(model), 0)

    def test_generate_length(self) -> None:
        model = GPT(
            vocab_size=11,
            block_size=8,
            n_layer=1,
            n_head=1,
            n_embd=8,
            dropout=0.0,
        ).eval()
        context = torch.zeros((1, 1), dtype=torch.long)
        output = model.generate(context, max_new_tokens=10, temperature=0.8, top_k=5)
        self.assertEqual(output.shape, (1, 11))

    def test_sampling_rejects_zero_temperature(self) -> None:
        with self.assertRaises(ValueError):
            sample_next_token(torch.zeros(1, 4), temperature=0.0)


class DataTests(unittest.TestCase):
    def test_get_batch(self) -> None:
        data = torch.arange(100)
        x, y = get_batch(data, block_size=8, batch_size=4, device="cpu")
        self.assertEqual(x.shape, (4, 8))
        self.assertEqual(y.shape, (4, 8))
        torch.testing.assert_close(y[:, :-1], x[:, 1:])
        torch.testing.assert_close(y[:, -1], x[:, -1] + 1)


if __name__ == "__main__":
    unittest.main()
