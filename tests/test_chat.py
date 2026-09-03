"""Tests for Hugging Face model discovery and chat sessions."""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
from transformers import Qwen3Config
from transformers import Qwen3ForCausalLM

from qachat import chat

APPLE_SILICON = (
    platform.system() == "Darwin" and platform.machine().lower() == "arm64"
)
CHAT_TEMPLATE = """{% for message in messages %}<|im_start|>{{
message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% endfor %}{%
if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"""


def _save_tokenizer(path: Path) -> None:
    vocabulary = {
        "hello": 0,
        "<unk>": 1,
        "<|im_start|>": 2,
        "<|im_end|>": 3,
        "user": 4,
        "assistant": 5,
        "system": 6,
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
        eos_token="<|im_end|>",
        pad_token="<|im_end|>",
        additional_special_tokens=["<|im_start|>"],
    )
    tokenizer.chat_template = CHAT_TEMPLATE
    tokenizer.save_pretrained(path)


def _save_torch_model(path: Path) -> None:
    path.mkdir()
    _save_tokenizer(path)
    configuration = Qwen3Config(
        vocab_size=7,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    model = Qwen3ForCausalLM(configuration)
    for parameter in model.parameters():
        parameter.data.zero_()
    model.save_pretrained(path)


def _save_mlx_model(path: Path) -> None:
    import mlx.core as mx
    from mlx_lm.models import qwen3

    path.mkdir()
    _save_tokenizer(path)
    arguments = qwen3.ModelArgs(
        model_type="qwen3",
        hidden_size=8,
        num_hidden_layers=1,
        intermediate_size=16,
        num_attention_heads=1,
        rms_norm_eps=1e-6,
        vocab_size=7,
        num_key_value_heads=1,
        max_position_embeddings=64,
        rope_theta=10_000.0,
        head_dim=8,
        tie_word_embeddings=False,
    )
    model = qwen3.Model(arguments)
    model.apply(mx.zeros_like)
    model.eval()
    configuration = asdict(arguments)
    configuration["eos_token_id"] = 3
    (path / "config.json").write_text(
        json.dumps(configuration),
        encoding="utf-8",
    )
    model.save_weights(str(path / "model.safetensors"))


class TestChat(unittest.TestCase):
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

    def test_chat_session(self) -> None:
        invalid: list[dict[str, object]] = [
            {"model_id": ""},
            {"backend": "invalid"},
            {"thinking": "yes"},
            {"temperature": -1},
            {"max_new_tokens": 0},
            {"system_prompt": ""},
            {"allow_download": "yes"},
        ]
        with patch.object(chat.ChatSession, "_load"):
            for settings in invalid:
                with self.subTest(settings=settings):
                    with self.assertRaises(ValueError):
                        chat.ChatSession(**settings)

        session = chat.ChatSession(
            str(self.torch_model),
            backend="torch",
            system_prompt="Answer briefly.",
            allow_download=False,
            max_new_tokens=3,
        )
        try:
            self.assertEqual(session.backend, "torch")
            self.assertIn(session.device, {"cpu", "mps", "cuda"})
            self.assertEqual(session.reply("hello"), "hello hello hello")
            self.assertEqual(len(session.messages), 3)
            self.assertEqual(
                [message["role"] for message in session.messages],
                ["system", "user", "assistant"],
            )
            self.assertEqual(session.messages[1]["content"], "hello")
            session.reset()
            self.assertEqual(len(session.messages), 1)
        finally:
            session.close()
        with self.assertRaises(RuntimeError):
            session.reply("hello")

        if APPLE_SILICON:
            session = chat.ChatSession(
                str(self.mlx_model),
                backend="mlx",
                thinking=True,
                allow_download=False,
                max_new_tokens=3,
            )
            try:
                self.assertEqual(session.device, "metal")
                self.assertEqual(
                    session.reply("hello"),
                    "hello hello hello",
                )
                self.assertEqual(len(session.messages), 2)
                self.assertEqual(
                    [message["role"] for message in session.messages],
                    ["user", "assistant"],
                )
                self.assertEqual(session.messages[0]["content"], "hello")
            finally:
                session.close()

        with patch.object(chat.ChatSession, "_load"):
            session = chat.ChatSession(
                str(self.torch_model),
                backend="torch",
                max_new_tokens=None,
            )
        self.assertIsNone(session.max_new_tokens)
        self.assertEqual(chat._check_max_new_tokens(1_000_000), 1_000_000)


if __name__ == "__main__":
    unittest.main()
