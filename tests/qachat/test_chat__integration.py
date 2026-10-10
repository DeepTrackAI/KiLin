"""Integration tests for chat generation with miniature Qwen3 models.

The models are constructed locally with deterministic zero-valued weights.
Tests exercise real PyTorch and MLX loaders, tokenizers, and generation without
network requests or pretrained-model downloads. Missing optional backend
packages and unsupported hardware cause only the affected tests to skip.

"""

from __future__ import annotations

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
import unittest

from qachat import chat


def _has_package(name: str) -> bool:
    """Return whether an optional integration dependency is installed."""

    return importlib.util.find_spec(name) is not None


def _save_tokenizer(path: Path) -> None:
    """Write a tiny tokenizer with a Qwen-compatible chat template."""

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocabulary = {
        "hello": 0,
        "<unk>": 1,
        "<|im_start|>": 2,
        "<|im_end|>": 3,
        "user": 4,
        "assistant": 5,
        "system": 6,
    }
    raw = Tokenizer(WordLevel(vocab=vocabulary, unk_token="<unk>"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        unk_token="<unk>",
        eos_token="<|im_end|>",
        pad_token="<|im_end|>",
        additional_special_tokens=["<|im_start|>"],
    )
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "<|im_start|>{{ message['role'] }}\n"
        "{{ message['content'] }}<|im_end|>\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "<|im_start|>assistant\n"
        "{% endif %}"
    )
    tokenizer.save_pretrained(path)


@unittest.skipUnless(
    all(
        _has_package(name)
        for name in ("torch", "transformers", "tokenizers")
    ),
    "The PyTorch miniature-model dependencies are not installed.",
)
class TestTorchMiniatureModel(unittest.TestCase):
    """Test the real PyTorch loading and text-generation pipeline."""

    @classmethod
    def setUpClass(cls) -> None:
        """Construct a tiny local model once for the PyTorch tests."""

        from transformers import Qwen3Config, Qwen3ForCausalLM

        temporary = TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.model_path = Path(temporary.name) / "torch-model"
        cls.model_path.mkdir()
        _save_tokenizer(cls.model_path)

        config = Qwen3Config(
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
        model = Qwen3ForCausalLM(config)
        for parameter in model.parameters():
            parameter.data.zero_()
        model.save_pretrained(cls.model_path)

    def test_offline_generation_with_local_model(self) -> None:
        """Load, generate, record history, and close a real Torch model."""

        session = chat.ChatSession(
            str(self.model_path),
            backend="torch",
            allow_download=False,
            temperature=0.0,
            max_new_tokens=3,
            system_prompt="Answer briefly.",
        )
        try:
            self.assertEqual(session.model_source, str(self.model_path))
            self.assertEqual(session.backend, "torch")
            self.assertIn(session.device, ("cpu", "mps", "cuda"))
            self.assertEqual(session.reply("hello"), "hello hello hello")
            self.assertEqual(session.output_token_total, 3)
            self.assertGreater(session.input_token_total, 0)
            self.assertEqual(
                [message["role"] for message in session.messages],
                [chat.SYSTEM_ROLE, chat.USER_ROLE, chat.ASSISTANT_ROLE],
            )
        finally:
            session.close()
        self.assertIsNone(session._model)
        self.assertIsNone(session._tokenizer)


@unittest.skipUnless(
    (
        platform.system() == "Darwin"
        and platform.machine().lower() == "arm64"
        and all(
            _has_package(name)
            for name in ("mlx", "mlx_lm", "transformers", "tokenizers")
        )
    ),
    "MLX integration requires Apple Silicon and MLX dependencies.",
)
class TestMlxMiniatureModel(unittest.TestCase):
    """Test the real MLX loading and streaming-generation pipeline."""

    @classmethod
    def setUpClass(cls) -> None:
        """Construct a tiny MLX model and its serialized configuration."""

        import mlx.core as mx
        from mlx_lm.models import qwen3

        temporary = TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.model_path = Path(temporary.name) / "mlx-model"
        cls.model_path.mkdir()
        _save_tokenizer(cls.model_path)

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
        (cls.model_path / "config.json").write_text(
            json.dumps(configuration), encoding="utf-8"
        )
        model.save_weights(str(cls.model_path / "model.safetensors"))

    def test_offline_streaming_with_local_model(self) -> None:
        """Load, stream, record history, and release a real MLX model."""

        session = chat.ChatSession(
            str(self.model_path),
            backend="mlx",
            allow_download=False,
            temperature=0.0,
            max_new_tokens=3,
        )
        try:
            pieces: list[str] = []
            answer = session.reply("hello", token_callback=pieces.append)
            self.assertEqual(answer, "hello hello hello")
            self.assertEqual("".join(pieces), "hello hello hello")
            self.assertEqual(session.device, "metal")
            self.assertEqual(session.output_token_total, 3)
            self.assertGreater(session.input_token_total, 0)
            self.assertEqual(
                [message["role"] for message in session.messages],
                [chat.USER_ROLE, chat.ASSISTANT_ROLE],
            )
        finally:
            session.close()
        self.assertIsNone(session._model)
        self.assertIsNone(session._tokenizer)


if __name__ == "__main__":
    unittest.main()
