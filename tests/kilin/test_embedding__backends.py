"""Exercise local embedding models through real optional backends.

These offline integration tests complement the mocked embedding unit tests.
They create small Qwen3-compatible model snapshots and verify that actual
PyTorch or MLX libraries can load, tokenize, embed, and normalize text.

Each backend is tested independently. Missing optional dependencies, or
unavailable MLX hardware, skip only the corresponding backend tests. Errors
from an installed backend remain test failures. No network or pretrained model
weights are needed.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from dataclasses import asdict
from importlib.util import find_spec
import json
import math
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from kilin import embeddings
from kilin.embeddings import EmbeddingSession


__all__ = [
    "TestTorchEmbeddingBackend",
    "TestMLXEmbeddingBackend",
]


APPLE_SILICON = (
    platform.system() == "Darwin"
    and platform.machine().lower() == "arm64"
)


def _require_dependencies(*names: str) -> None:
    """Skip a backend if one of its optional packages is unavailable.

    Parameters
    ----------
    *names: str
        Top-level import names required by the backend.

    Raises
    ------
    unittest.SkipTest
        If a required package is not installed.

    Notes
    -----
    A package that is installed but broken is not silently skipped. Its
    import or execution failure is reported by the integration test.

    """

    missing = [name for name in names if find_spec(name) is None]
    if missing:
        raise unittest.SkipTest(
            "Missing optional backend dependencies: "
            + ", ".join(missing)
        )


def _save_tokenizer(path: Path) -> None:
    """Save a deterministic local word-level tokenizer.

    Parameters
    ----------
    path: pathlib.Path
        Existing directory receiving the tokenizer files.

    """

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocabulary = {
        "hello": 0,
        "repository": 1,
        "question": 2,
        "document": 3,
        "<unk>": 4,
        "<pad>": 5,
    }
    raw = Tokenizer(
        WordLevel(vocab=vocabulary, unk_token="<unk>")
    )
    raw.pre_tokenizer = Whitespace()

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        unk_token="<unk>",
        pad_token="<pad>",
    )
    tokenizer.save_pretrained(path)


def _save_torch_model(path: Path) -> None:
    """Save a tiny, reproducible Qwen3 PyTorch model locally.

    Parameters
    ----------
    path: pathlib.Path
        New directory receiving model and tokenizer files.

    Notes
    -----
    The random-state context avoids modifying the caller's global PyTorch
    random state. Generated weights are for compatibility testing, not for
    evaluating retrieval quality.

    """

    import torch
    from transformers import Qwen3Config, Qwen3Model

    path.mkdir()
    _save_tokenizer(path)
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
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(0)
        Qwen3Model(configuration).save_pretrained(path)


def _save_mlx_model(path: Path) -> None:
    """Save a tiny, deterministic Qwen3 MLX model locally.

    Parameters
    ----------
    path: pathlib.Path
        New directory receiving model and tokenizer files.

    Notes
    -----
    Constant weights provide a stable loading smoke test, not a numerical
    reference for meaningful Qwen3 embeddings or masked-token pooling.

    """

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

    (path / "config.json").write_text(
        json.dumps(asdict(arguments)),
        encoding="utf-8",
    )
    model.save_weights(str(path / "model.safetensors"))


class EmbeddingBackendTestCase(unittest.TestCase):
    """Provide backend-agnostic assertions for real embeddings."""

    def assert_normalized_vectors(
        self,
        vectors: list[list[float]],
        *,
        expected_count: int,
        dimension: int = 8,
    ) -> None:
        """Verify vector shape, finiteness, and L2 normalization.

        Parameters
        ----------
        vectors: list[list[float]]
            Vectors returned by an embedding operation.
        expected_count: int
            Number of input texts.
        dimension: int, optional
            Expected dimensionality of the tiny model. Defaults to ``8``.

        """

        self.assertEqual(len(vectors), expected_count)
        for vector in vectors:
            self.assertEqual(len(vector), dimension)
            self.assertTrue(all(math.isfinite(x) for x in vector))
            norm = math.sqrt(sum(x * x for x in vector))
            self.assertAlmostEqual(norm, 1.0, delta=2e-4)

    def assert_same_vectors(
        self,
        first: list[float],
        second: list[float],
    ) -> None:
        """Compare embeddings independent of batch padding.

        Parameters
        ----------
        first: list[float]
            Embedding obtained in a mixed-length batch.
        second: list[float]
            Embedding obtained for the same text alone.

        """

        self.assertEqual(len(first), len(second))
        for actual, expected in zip(first, second):
            self.assertAlmostEqual(actual, expected, delta=2e-3)

    def exercise_local_session(
        self,
        model_path: Path,
        backend: str,
    ) -> None:
        """Verify a local backend's loading, batching, and lifecycle.

        Parameters
        ----------
        model_path: pathlib.Path
            Directory containing the locally generated model.
        backend: str
            Selected backend, ``'torch'`` or ``'mlx'``.

        Notes
        -----
        A short input is embedded alongside a longer padded input and also
        individually. The results should agree up to backend precision.
        KíLín's snapshot resolvers must not run for these local snapshots.

        """

        with (
            patch.object(
                embeddings,
                "snapshot_download",
                side_effect=AssertionError("Network download requested."),
            ),
            patch.object(
                embeddings,
                "cached_model_snapshot",
                side_effect=AssertionError("Cache lookup requested."),
            ),
        ):
            session = EmbeddingSession(
                str(model_path),
                backend=backend,
                allow_download=False,
            )
            try:
                self.assertEqual(session.model_source, str(model_path))
                self.assertEqual(session.backend, backend)
                self.assertTrue(session.device)

                short_text = "hello"
                long_text = "hello repository question document"
                batch = session.embed_documents([short_text, long_text])
                isolated = session.embed_query(short_text)
                self.assert_normalized_vectors(
                    batch,
                    expected_count=2,
                )
                self.assert_normalized_vectors(
                    [isolated],
                    expected_count=1,
                )
                self.assert_same_vectors(batch[0], isolated)
                self.assertEqual(session.count_tokens("hello document"), 2)
            finally:
                session.close()

            with self.assertRaisesRegex(RuntimeError, "closed"):
                session.count_tokens("hello")
            with self.assertRaisesRegex(RuntimeError, "closed"):
                session.embed_query("hello")


class TestTorchEmbeddingBackend(EmbeddingBackendTestCase):
    """Check the real PyTorch backend using generated local weights."""

    def test_torch_model_integration(self) -> None:
        """Load a tiny model and verify local embedding calculations."""

        _require_dependencies("torch", "transformers", "tokenizers")
        with TemporaryDirectory() as directory:
            model_path = Path(directory) / "torch-model"
            _save_torch_model(model_path)
            self.exercise_local_session(model_path, "torch")


class TestMLXEmbeddingBackend(EmbeddingBackendTestCase):
    """Check the real MLX backend on supported Apple Silicon hardware."""

    def test_mlx_model_integration(self) -> None:
        """Load a tiny model and verify local embedding calculations."""

        if not APPLE_SILICON:
            self.skipTest("MLX requires native Apple Silicon.")
        _require_dependencies(
            "mlx", "mlx_embeddings", "transformers", "tokenizers"
        )
        import mlx.core as mx

        if not mx.metal.is_available():
            self.skipTest("MLX Metal is not available.")

        with TemporaryDirectory() as directory:
            model_path = Path(directory) / "mlx-model"
            _save_mlx_model(model_path)
            self.exercise_local_session(model_path, "mlx")


if __name__ == "__main__":
    unittest.main()
