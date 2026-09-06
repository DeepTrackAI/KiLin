"""Offline integration tests for both embedding backends."""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
import unittest

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model

from qachat.embeddings import EmbeddingSession

APPLE_SILICON = (
    platform.system() == "Darwin" and platform.machine().lower() == "arm64"
)


def _save_tokenizer(path: Path) -> None:
    vocabulary = {
        "hello": 0,
        "repository": 1,
        "question": 2,
        "document": 3,
        "<unk>": 4,
        "<pad>": 5,
    }
    raw = Tokenizer(
        WordLevel(
            vocab=vocabulary,
            unk_token="<unk>",
        )
    )
    raw.pre_tokenizer = Whitespace()

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        unk_token="<unk>",
        pad_token="<pad>",
    )
    tokenizer.save_pretrained(path)


def _save_torch_model(path: Path) -> None:
    import torch

    path.mkdir()
    _save_tokenizer(path)
    torch.manual_seed(0)
    configuration = Qwen3Config(
        vocab_size=6,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    Qwen3Model(configuration).save_pretrained(path)


def _save_mlx_model(path: Path) -> None:
    import mlx.core as mx
    from mlx_embeddings.models import qwen3

    path.mkdir()
    _save_tokenizer(path)
    arguments = qwen3.ModelArgs(
        hidden_size=8,
        num_hidden_layers=1,
        intermediate_size=16,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        vocab_size=6,
        tie_word_embeddings=False,
    )
    model = qwen3.Model(arguments)
    model.apply(mx.ones_like)
    model.eval()

    configuration = asdict(arguments)
    (path / "config.json").write_text(
        json.dumps(configuration),
        encoding="utf-8",
    )
    model.save_weights(str(path / "model.safetensors"))


class TestEmbeddingBackends(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary_directory = TemporaryDirectory()
        cls.root = Path(cls.temporary_directory.name)
        cls.torch_model = cls.root / "torch-model"
        _save_torch_model(cls.torch_model)

        cls.mlx_model = cls.root / "mlx-model"
        if APPLE_SILICON:
            _save_mlx_model(cls.mlx_model)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary_directory.cleanup()

    def test_embedding_backends(self) -> None:
        choices = [("torch", self.torch_model)]
        if APPLE_SILICON:
            choices.append(("mlx", self.mlx_model))

        for backend, path in choices:
            with self.subTest(backend=backend):
                session = EmbeddingSession(
                    str(path),
                    backend=backend,
                    allow_download=False,
                )
                try:
                    vectors = session.embed_documents(
                        ["hello document", "repository question"]
                    )
                    query = session.embed_query("hello question")
                    token_count = session.count_tokens("hello document")
                finally:
                    session.close()

                self.assertEqual(token_count, 2)
                self.assertEqual(len(vectors), 2)
                self.assertEqual(len(vectors[0]), 8)
                self.assertEqual(len(query), 8)
                for vector in [*vectors, query]:
                    length = sum(value * value for value in vector) ** 0.5
                    self.assertAlmostEqual(length, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
