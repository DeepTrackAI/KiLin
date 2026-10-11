"""Offline smoke tests for cached default pretrained chat models.

Only the backend-specific default IDs from qachat.model_defaults are used.
Tests skip when an optional backend is unavailable or the corresponding
model has not been fully downloaded. A cached model that fails to load or
produce a reply is a test failure, not a skip. No downloads are permitted.

Note that the default models are large; these tests can be resource intensive
when their weights are present in the local Hugging Face cache.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from huggingface_hub.errors import LocalEntryNotFoundError

from qachat import chat, model_defaults
from qachat import huggingface


def _has_package(name: str) -> bool:
    """Return whether the optional backend dependency can be imported."""

    return importlib.util.find_spec(name) is not None


def _is_complete_snapshot(snapshot: Path) -> bool:
    """Check that cached default weights and tokenizer files are present.

    This is a file-presence check, not a substitute for loading the model.
    Indexed weight files must have every referenced shard available.

    """

    if not (snapshot / "config.json").is_file():
        return False
    if not any(
        (snapshot / name).is_file()
        for name in ("tokenizer.json", "tokenizer.model", "vocab.json")
    ):
        return False

    for index_name in (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ):
        index = snapshot / index_name
        if index.is_file():
            contents = json.loads(index.read_text(encoding="utf-8"))
            files = set(contents["weight_map"].values())
            return bool(files) and all(
                (snapshot / filename).is_file() for filename in files
            )

    return any(path.is_file() for path in snapshot.glob("*.safetensors")) or (
        any(path.is_file() for path in snapshot.glob("pytorch_model*.bin"))
    )


class TestCachedModelRequirements(unittest.TestCase):
    """Test local completeness checks without loading pretrained weights."""

    def test_unindexed_weights_and_tokenizer_are_required(self) -> None:
        """Require a configuration, tokenizer, and existing weight file."""

        with TemporaryDirectory() as directory:
            snapshot = Path(directory)
            self.assertFalse(_is_complete_snapshot(snapshot))
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
            self.assertFalse(_is_complete_snapshot(snapshot))
            (snapshot / "model.safetensors").write_bytes(b"weights")
            self.assertTrue(_is_complete_snapshot(snapshot))

    def test_index_requires_all_referenced_shards(self) -> None:
        """Skip partially downloaded sharded model weights."""

        with TemporaryDirectory() as directory:
            snapshot = Path(directory)
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
            (snapshot / "model.safetensors.index.json").write_text(
                json.dumps(
                    {"weight_map": {"a": "first.safetensors",
                                    "b": "second.safetensors"}}
                ),
                encoding="utf-8",
            )
            (snapshot / "first.safetensors").write_bytes(b"weights")
            self.assertFalse(_is_complete_snapshot(snapshot))
            (snapshot / "second.safetensors").write_bytes(b"weights")
            self.assertTrue(_is_complete_snapshot(snapshot))


class TestDefaultPretrainedChatModels(unittest.TestCase):
    """Verify real inference when the default models are cached locally."""

    def _cached_snapshot(self, model_id: str) -> Path:
        """Skip only when a default model is absent or not downloaded."""

        if not huggingface.model_is_available(model_id):
            self.skipTest(f"Default model is not cached: {model_id}")

        try:
            snapshot = huggingface.cached_model_snapshot(model_id)
        except (FileNotFoundError, LocalEntryNotFoundError):
            self.skipTest(f"Default model snapshot is unavailable: {model_id}")

        if not _is_complete_snapshot(snapshot):
            self.skipTest(f"Default model is not fully cached: {model_id}")
        return snapshot

    def _check_model(self, model_id: str, backend: str) -> None:
        """Perform one offline response and validate the session lifecycle."""

        with patch.dict(
            os.environ,
            {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        ):
            snapshot = self._cached_snapshot(model_id)
            session = chat.ChatSession(
                model_id,
                backend=backend,
                thinking=False,
                temperature=0.0,
                max_new_tokens=12,
                allow_download=False,
            )
            try:
                self.assertEqual(session.model_id, model_id)
                self.assertEqual(session.backend, backend)
                self.assertEqual(session.model_source, str(snapshot))
                answer = session.reply("Say hello briefly.")
                self.assertIsInstance(answer, str)
                self.assertTrue(answer.strip())
                self.assertEqual(
                    [message["role"] for message in session.messages],
                    [chat.USER_ROLE, chat.ASSISTANT_ROLE],
                )
                self.assertEqual(session.messages[0]["content"],
                                 "Say hello briefly.")
                self.assertEqual(session.messages[1]["content"], answer)
                self.assertGreater(session.input_token_total, 0)
                self.assertGreater(session.output_token_total, 0)
                self.assertLessEqual(session.output_token_total, 12)
            finally:
                session.close()
            self.assertIsNone(session._model)
            self.assertIsNone(session._tokenizer)

    def test_default_torch_model(self) -> None:
        """Generate with the cached default PyTorch chat model."""

        if not all(
            _has_package(name) for name in ("torch", "transformers")
        ):
            self.skipTest("PyTorch chat dependencies are not installed.")
        self._check_model(model_defaults.DEFAULT_CHAT_MODEL_TORCH, "torch")

    def test_default_mlx_model(self) -> None:
        """Generate with the cached default MLX model on Apple Silicon."""

        if (
            platform.system() != "Darwin"
            or platform.machine().lower() != "arm64"
            or not all(_has_package(name) for name in ("mlx", "mlx_lm"))
        ):
            self.skipTest("MLX requires Apple Silicon and MLX packages.")
        self._check_model(model_defaults.DEFAULT_CHAT_MODEL_MLX, "mlx")


if __name__ == "__main__":
    unittest.main()
