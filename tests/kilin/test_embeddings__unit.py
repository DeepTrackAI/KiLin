"""Test embedding sessions without downloading or loading embedding models.

The suite exercises source resolution, lifecycle, validation, tracing, backend
loading, and the actual MLX and PyTorch pooling paths with deterministic
in-memory stand-ins. Backend imports are intercepted so that the tests do not
require MLX, PyTorch, transformers, model weights, or network access.

Tests are intentionally independent of local accelerators and model caches.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from kilin import embeddings
from kilin.trace_logging import TRACE_EVENT_EMBEDDING_TOKENS_PREPARED


class FakeTokenizer:
    """Record tokenization calls and return deterministic token IDs."""

    def __init__(self) -> None:
        """Initialize the tokenization call log."""

        self.calls: list[tuple[str, bool, bool]] = []

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool,
    ) -> list[str]:
        """Return one ID per word and optional special-token IDs."""

        self.calls.append((text, add_special_tokens, truncation))
        tokens = text.split()
        if add_special_tokens:
            return ["<bos>", *tokens, "<eos>"]
        return tokens


class FakeTrace:
    """Record embedding trace operations without writing trace files."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        include_content: bool = False,
        include_embedding_vectors: bool = False,
        include_token_ids: bool = False,
    ) -> None:
        """Configure trace flags and initialize captured records."""

        self.enabled = enabled
        self.include_content = include_content
        self.include_embedding_vectors = include_embedding_vectors
        self.include_token_ids = include_token_ids
        self.events: list[tuple[str, dict[str, object]]] = []
        self.spans: list[tuple[str, dict[str, object]]] = []
        self.artifacts: list[tuple[str, object]] = []
        self.ids: list[str] = []

    @contextmanager
    def span(self, name: str, **kwargs: object):
        """Record a span and propagate errors from its body."""

        self.spans.append((name, kwargs))
        yield

    def new_id(self, prefix: str) -> str:
        """Return a predictable operation identifier."""

        identifier = f"{prefix}-{len(self.ids) + 1}"
        self.ids.append(identifier)
        return identifier

    def emit(self, event: str, **kwargs: object) -> None:
        """Capture one trace event and its arguments."""

        self.events.append((event, kwargs))

    def artifact(self, name: str, data: object) -> dict[str, str]:
        """Capture an artifact and return a stable reference."""

        self.artifacts.append((name, data))
        return {"artifact": name}

    def find(self, name: str) -> list[dict[str, object]]:
        """Return arguments for events with the given name."""

        return [kwargs for event, kwargs in self.events if event == name]


class FakeTensor:
    """Implement the small tensor interface used by the PyTorch backend."""

    def __init__(self, values: object, *, device: str = "cpu") -> None:
        """Store a NumPy array with a synthetic device label."""

        self.values = np.asarray(values)
        self.device = device

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the tensor dimensions."""

        return self.values.shape

    def to(self, device: str) -> FakeTensor:
        """Return a tensor labeled with the requested device."""

        return FakeTensor(self.values, device=device)

    def detach(self) -> FakeTensor:
        """Return the same values without gradient tracking."""

        return self

    def cpu(self) -> FakeTensor:
        """Return a CPU-labeled tensor."""

        return self.to("cpu")

    def sum(self, *, dim: int) -> FakeTensor:
        """Sum values along a specified dimension."""

        return FakeTensor(self.values.sum(axis=dim), device=self.device)

    def tolist(self) -> list[object]:
        """Convert the underlying values to Python lists."""

        return self.values.tolist()

    def expand_as(self, other: FakeTensor) -> FakeTensor:
        """Broadcast this tensor to another tensor's shape."""

        return FakeTensor(
            np.broadcast_to(self.values, other.shape),
            device=self.device,
        )

    def masked_fill(self, mask: FakeTensor, value: int) -> FakeTensor:
        """Replace masked elements with a scalar."""

        return FakeTensor(
            np.where(mask.values, value, self.values),
            device=self.device,
        )

    def max(self, *, dim: int) -> SimpleNamespace:
        """Return the maximum values along a dimension."""

        return SimpleNamespace(
            values=FakeTensor(
                self.values.max(axis=dim),
                device=self.device,
            )
        )

    def float(self) -> FakeTensor:
        """Convert tensor values to 32-bit floating point."""

        return FakeTensor(self.values.astype(np.float32), device=self.device)

    def __eq__(self, other: object) -> FakeTensor:
        """Compute elementwise equality."""

        return FakeTensor(self.values == other, device=self.device)

    def __getitem__(self, key: object) -> FakeTensor:
        """Select elements with NumPy-style advanced indexing."""

        if isinstance(key, tuple):
            key = tuple(
                item.values if isinstance(item, FakeTensor) else item
                for item in key
            )
        return FakeTensor(self.values[key], device=self.device)


class FakeBatch(dict):
    """Mimic a token batch that can move to a PyTorch device."""

    def __init__(self, ids: object, mask: object) -> None:
        """Store input IDs and an attention mask."""

        super().__init__(
            input_ids=FakeTensor(ids),
            attention_mask=FakeTensor(mask),
        )
        self.requested_device: str | None = None

    def to(self, device: str) -> FakeBatch:
        """Move each tensor to the requested synthetic device."""

        self.requested_device = device
        for key, tensor in self.items():
            self[key] = tensor.to(device)
        return self


class FakeTorch:
    """Supply tensor operations needed to test last-token pooling."""

    def __init__(self) -> None:
        """Initialize inference-mode and normalization instrumentation."""

        self.inference_depth = 0
        self.normalization_calls: list[tuple[int, int]] = []
        self.nn = SimpleNamespace(
            functional=SimpleNamespace(normalize=self.normalize)
        )

    @contextmanager
    def inference_mode(self):
        """Track execution inside the inference-mode context."""

        self.inference_depth += 1
        try:
            yield
        finally:
            self.inference_depth -= 1

    def arange(self, size: int, *, device: str) -> FakeTensor:
        """Create a device-labeled sequence of positions."""

        return FakeTensor(np.arange(size), device=device)

    def normalize(
        self,
        tensor: FakeTensor,
        *,
        p: int,
        dim: int,
    ) -> FakeTensor:
        """Apply the same row-wise L2 normalization as PyTorch."""

        self.normalization_calls.append((p, dim))
        norms = np.linalg.norm(tensor.values, ord=p, axis=dim, keepdims=True)
        return FakeTensor(tensor.values / norms, device=tensor.device)


class FakeMLXArray:
    """Implement the MLX array operations needed by the embedding path."""

    def __init__(self, values: object) -> None:
        """Store array data."""

        self.values = np.asarray(values)

    def astype(self, dtype: object) -> FakeMLXArray:
        """Cast the array to the requested dtype."""

        return FakeMLXArray(self.values.astype(dtype))

    def __truediv__(self, other: FakeMLXArray) -> FakeMLXArray:
        """Divide array values using NumPy broadcasting."""

        return FakeMLXArray(self.values / other.values)

    def tolist(self) -> list[object]:
        """Convert array values to Python lists."""

        return self.values.tolist()


class FakeMLX:
    """Supply array conversion, norms, and evaluation for MLX tests."""

    float32 = np.float32

    def __init__(self) -> None:
        """Initialize recorded evaluation calls."""

        self.evaluated: list[FakeMLXArray] = []
        self.linalg = SimpleNamespace(norm=self.norm)

    def array(self, values: object) -> FakeMLXArray:
        """Convert input values into an MLX-like array."""

        return FakeMLXArray(values)

    def norm(
        self,
        values: FakeMLXArray,
        *,
        axis: int,
        keepdims: bool,
    ) -> FakeMLXArray:
        """Calculate L2 norms along the requested dimension."""

        return FakeMLXArray(
            np.linalg.norm(values.values, axis=axis, keepdims=keepdims)
        )

    def eval(self, values: FakeMLXArray) -> None:
        """Record the evaluated output array."""

        self.evaluated.append(values)


class EmbeddingTestCase(unittest.TestCase):
    """Provide a model-free embedding session for isolated unit tests."""

    def make_session(
        self,
        *,
        backend: str = "torch",
        trace: FakeTrace | None = None,
    ) -> embeddings.EmbeddingSession:
        """Create a session with a fake model and tokenizer."""

        def fake_load(
            session: embeddings.EmbeddingSession,
            source: str,
        ) -> None:
            """Install fake model objects in place of model weights."""

            self.assertEqual(source, "/fake/snapshot")
            session._model = object()
            session._tokenizer = FakeTokenizer()
            session.device = "test-device"

        with (
            patch.object(
                embeddings,
                "choose_backend",
                return_value=backend,
            ),
            patch.object(
                embeddings,
                "_model_source",
                return_value="/fake/snapshot",
            ),
            patch.object(
                embeddings.EmbeddingSession,
                "_load_torch",
                autospec=True,
                side_effect=fake_load,
            ),
            patch.object(
                embeddings.EmbeddingSession,
                "_load_mlx",
                autospec=True,
                side_effect=fake_load,
            ),
        ):
            return embeddings.EmbeddingSession(
                "group/embedding-model",
                backend=backend,
                allow_download=False,
                trace=trace,
            )


class TestSessionInitialization(EmbeddingTestCase):
    """Check construction, backend selection, and loading failures."""

    def test_initializes_torch_and_mlx_backends(self) -> None:
        """Dispatch each backend to the appropriate loader."""

        for backend in ("torch", "mlx"):
            with self.subTest(backend=backend):
                session = self.make_session(backend=backend)
                self.assertEqual(session.backend, backend)
                self.assertEqual(session.model_id, "group/embedding-model")
                self.assertEqual(session.model_source, "/fake/snapshot")
                self.assertEqual(session.device, "test-device")
                self.assertIsNotNone(session._model)
                self.assertIsInstance(session._tokenizer, FakeTokenizer)

    def test_constructor_passes_normalized_settings(self) -> None:
        """Forward trimmed model ID, revision, and download policy."""

        with (
            patch.object(
                embeddings,
                "choose_backend",
                return_value="torch",
            ) as choose,
            patch.object(
                embeddings,
                "_model_source",
                return_value="/snapshot",
            ) as source,
            patch.object(
                embeddings.EmbeddingSession,
                "_load_torch",
                autospec=True,
            ) as load,
        ):
            session = embeddings.EmbeddingSession(
                "  group/model  ",
                backend="auto",
                allow_download=False,
                revision="revision-1",
            )

        choose.assert_called_once_with("auto")
        source.assert_called_once_with(
            "group/model",
            allow_download=False,
            revision="revision-1",
        )
        load.assert_called_once_with(session, "/snapshot")
        self.assertEqual(session.model_id, "group/model")
        self.assertIsNone(session._model)

    def test_rejects_invalid_model_ids_without_loading(self) -> None:
        """Reject blank or non-string identifiers before backend selection."""

        for model_id in ("", " \t\n ", None, 3, Path("model"), False):
            with self.subTest(model_id=model_id):
                with patch.object(embeddings, "choose_backend") as choose:
                    with self.assertRaisesRegex(ValueError, "model ID"):
                        embeddings.EmbeddingSession(model_id)
                choose.assert_not_called()

    def test_rejects_nonboolean_download_flags(self) -> None:
        """Require an exact Boolean download policy."""

        for value in (0, 1, "yes", None, [], object()):
            with self.subTest(value=value):
                with patch.object(embeddings, "choose_backend") as choose:
                    with self.assertRaisesRegex(ValueError, "allow_download"):
                        embeddings.EmbeddingSession(
                            "group/model", allow_download=value
                        )
                choose.assert_not_called()

    def test_invalid_backend_is_not_silently_replaced(self) -> None:
        """Propagate backend selection errors without source resolution."""

        with (
            patch.object(
                embeddings,
                "choose_backend",
                side_effect=ValueError("invalid backend"),
            ),
            patch.object(embeddings, "_model_source") as source,
        ):
            with self.assertRaisesRegex(ValueError, "invalid backend"):
                embeddings.EmbeddingSession("group/model", backend="other")
        source.assert_not_called()

    def test_source_resolution_failure_propagates(self) -> None:
        """Do not attempt backend loading after source resolution fails."""

        with (
            patch.object(
                embeddings,
                "choose_backend",
                return_value="torch",
            ),
            patch.object(
                embeddings,
                "_model_source",
                side_effect=OSError("offline cache missing"),
            ),
            patch.object(
                embeddings.EmbeddingSession,
                "_load_torch",
            ) as load,
        ):
            with self.assertRaisesRegex(OSError, "offline cache missing"):
                embeddings.EmbeddingSession("group/model")
        load.assert_not_called()

    def test_loader_failure_propagates_without_resolved_event(self) -> None:
        """Report loader failure without recording successful resolution."""

        trace = FakeTrace()
        with (
            patch.object(
                embeddings,
                "choose_backend",
                return_value="torch",
            ),
            patch.object(
                embeddings,
                "_model_source",
                return_value="/snapshot",
            ),
            patch.object(
                embeddings.EmbeddingSession,
                "_load_torch",
                side_effect=ImportError("torch missing"),
            ),
        ):
            with self.assertRaisesRegex(ImportError, "torch missing"):
                embeddings.EmbeddingSession("group/model", trace=trace)
        self.assertEqual(trace.find("model.embedding.resolved"), [])
        self.assertEqual(trace.spans[0][0], "model.embedding.load")

    def test_initialization_traces_requested_and_resolved_model(self) -> None:
        """Include model source and revision in initialization trace data."""

        trace = FakeTrace()
        session = self.make_session(trace=trace)
        self.assertEqual(len(trace.spans), 1)
        self.assertEqual(trace.spans[0][0], "model.embedding.load")
        data = trace.spans[0][1]["data"]
        self.assertEqual(data["backend"], "torch")
        self.assertFalse(data["allow_download"])
        self.assertEqual(data["model_id"], session.model_id)
        resolved = trace.find("model.embedding.resolved")
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0]["data"]["model_source"], "/fake/snapshot")
        self.assertEqual(resolved[0]["data"]["device"], "test-device")

    def test_default_trace_is_null_logger(self) -> None:
        """Use a disabled trace sink when none is supplied."""

        session = self.make_session()
        self.assertFalse(session.trace.enabled)


class TestModelSource(unittest.TestCase):
    """Check local and Hugging Face model resolution."""

    def test_local_directory_takes_precedence(self) -> None:
        """Use local paths without checking or downloading model snapshots."""

        with TemporaryDirectory() as directory:
            with (
                patch.object(embeddings, "cached_model_snapshot") as cached,
                patch.object(embeddings, "snapshot_download") as download,
            ):
                for allow_download in (True, False):
                    self.assertEqual(
                        embeddings._model_source(
                            directory,
                            allow_download=allow_download,
                            revision="v1",
                        ),
                        directory,
                    )
            cached.assert_not_called()
            download.assert_not_called()

    def test_local_home_path_is_expanded(self) -> None:
        """Resolve a tilde-prefixed local directory."""

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model").mkdir()
            with patch.dict(os.environ, {"HOME": str(root)}):
                resolved = embeddings._model_source(
                    "~/model", allow_download=False, revision=None
                )
            self.assertEqual(resolved, str(root / "model"))

    def test_offline_remote_model_uses_cached_snapshot(self) -> None:
        """Forward the revision when downloads are forbidden."""

        with (
            patch.object(
                embeddings,
                "cached_model_snapshot",
                return_value=Path("/cache/snapshot"),
            ) as cached,
            patch.object(embeddings, "snapshot_download") as download,
        ):
            resolved = embeddings._model_source(
                "group/model", allow_download=False, revision="commit-1"
            )
        self.assertEqual(resolved, "/cache/snapshot")
        cached.assert_called_once_with("group/model", revision="commit-1")
        download.assert_not_called()

    def test_online_remote_model_uses_snapshot_download(self) -> None:
        """Download a requested model revision through Hugging Face."""

        with (
            patch.object(
                embeddings,
                "snapshot_download",
                return_value="/downloaded",
            ) as download,
            patch.object(embeddings, "cached_model_snapshot") as cached,
        ):
            resolved = embeddings._model_source(
                "group/model", allow_download=True, revision=None
            )
        self.assertEqual(resolved, "/downloaded")
        download.assert_called_once_with(
            repo_id="group/model",
            revision=None,
            local_files_only=False,
        )
        cached.assert_not_called()

    def test_snapshot_errors_propagate(self) -> None:
        """Do not mask cache and download failures."""

        for allow_download, attribute in (
            (False, "cached_model_snapshot"),
            (True, "snapshot_download"),
        ):
            with self.subTest(allow_download=allow_download):
                with patch.object(
                    embeddings,
                    attribute,
                    side_effect=OSError("snapshot unavailable"),
                ):
                    with self.assertRaisesRegex(
                        OSError, "snapshot unavailable"
                    ):
                        embeddings._model_source(
                            "group/model",
                            allow_download=allow_download,
                            revision=None,
                        )


class TestTokenCounting(EmbeddingTestCase):
    """Check tokenization semantics and closed-session behavior."""

    def test_counts_special_tokens_without_truncation(self) -> None:
        """Use the selected tokenizer with special tokens and no truncation."""

        session = self.make_session()
        self.assertEqual(session.count_tokens("one two three"), 5)
        self.assertEqual(
            session._tokenizer.calls,
            [("one two three", True, False)],
        )

    def test_empty_text_is_passed_to_tokenizer(self) -> None:
        """Count special tokens even for an empty string."""

        session = self.make_session()
        self.assertEqual(session.count_tokens(""), 2)
        self.assertEqual(session._tokenizer.calls, [("", True, False)])

    def test_invalid_token_input_is_rejected(self) -> None:
        """Reject non-string token-count inputs before tokenization."""

        session = self.make_session()
        for text in (None, 1, False, ["words"], b"bytes"):
            with self.subTest(text=text):
                with self.assertRaisesRegex(ValueError, "string"):
                    session.count_tokens(text)
        self.assertEqual(session._tokenizer.calls, [])

    def test_closed_session_rejects_token_counting(self) -> None:
        """Fail before accessing a released tokenizer."""

        session = self.make_session()
        session._tokenizer = None
        with self.assertRaisesRegex(RuntimeError, "closed"):
            session.count_tokens("text")


class TestTextValidation(unittest.TestCase):
    """Check input collection validation without modifying valid text."""

    def test_preserves_exact_text_and_input_list(self) -> None:
        """Preserve whitespace, Unicode, and input list identity."""

        texts = ["  first\n", "\u03b1\u03b2", "last  "]
        self.assertIs(embeddings._check_texts(texts), texts)
        self.assertEqual(texts[0], "  first\n")

    def test_empty_list_is_valid(self) -> None:
        """Allow the empty batch used for no-op indexing."""

        self.assertEqual(embeddings._check_texts([]), [])

    def test_rejects_nonlist_inputs(self) -> None:
        """Require a list rather than a string or another iterable."""

        for texts in (None, "text", ("text",), {"text"}, 3):
            with self.subTest(texts=texts):
                with self.assertRaisesRegex(ValueError, "list"):
                    embeddings._check_texts(texts)

    def test_rejects_blank_and_nonstrings(self) -> None:
        """Reject invalid values anywhere in the input batch."""

        for invalid in ("", " \t\n", None, 3, True, b"bytes", []):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "empty"):
                    embeddings._check_texts(["valid", invalid])


class TestVectorValidation(unittest.TestCase):
    """Check vector count, dimensions, numeric types, and finite values."""

    def test_valid_vectors_are_converted_to_float(self) -> None:
        """Return float-valued vectors without modifying the originals."""

        vectors = [[3, 4.0], [1.5, -2]]
        self.assertEqual(
            embeddings._check_vectors(vectors, 2),
            [[3.0, 4.0], [1.5, -2.0]],
        )
        self.assertEqual(vectors, [[3, 4.0], [1.5, -2]])

    def test_rejects_wrong_vector_count(self) -> None:
        """Reject missing, excess, and empty embedding results."""

        for vectors, expected in (
            ([], 1),
            ([], 0),
            ([[1.0]], 2),
            ([[1.0], [2.0]], 1),
        ):
            with self.subTest(vectors=vectors, expected=expected):
                with self.assertRaisesRegex(RuntimeError, "wrong count"):
                    embeddings._check_vectors(vectors, expected)

    def test_rejects_zero_and_inconsistent_dimensions(self) -> None:
        """Require a positive shared vector dimension."""

        for vectors in (
            [[]],
            [[1.0], [1.0, 2.0]],
            [[1.0, 2.0], [3.0]],
        ):
            with self.subTest(vectors=vectors):
                with self.assertRaisesRegex(RuntimeError, "invalid vectors"):
                    embeddings._check_vectors(vectors, len(vectors))

    def test_rejects_invalid_numeric_values(self) -> None:
        """Reject booleans, nonnumbers, and nonfinite values."""

        for invalid in (
            True,
            False,
            None,
            "1.0",
            1j,
            float("nan"),
            float("inf"),
            -float("inf"),
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(RuntimeError, "invalid vectors"):
                    embeddings._check_vectors([[1.0, invalid]], 1)

    def test_rejects_zero_norm_in_any_vector(self) -> None:
        """Reject all-zero vectors even when other vectors are valid."""

        for vectors in (
            [[0.0]],
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 1.0]],
        ):
            with self.subTest(vectors=vectors):
                with self.assertRaisesRegex(RuntimeError, "invalid vectors"):
                    embeddings._check_vectors(vectors, len(vectors))


class TestEmbeddingOperations(EmbeddingTestCase):
    """Check document and query dispatch, validation, and lifecycle."""

    def test_empty_documents_skip_backend_even_after_close(self) -> None:
        """An empty batch returns immediately without a loaded model."""

        session = self.make_session()
        session._model = None
        session._tokenizer = None
        with patch.object(session, "_embed_torch") as embed:
            self.assertEqual(session.embed_documents([]), [])
        embed.assert_not_called()

    def test_invalid_document_batches_do_not_call_backend(self) -> None:
        """Reject malformed batches before accessing the model."""

        session = self.make_session()
        with patch.object(session, "_embed_torch") as embed:
            for texts in ("text", ("text",), [""], ["ok", "  "]):
                with self.subTest(texts=texts):
                    with self.assertRaises(ValueError):
                        session.embed_documents(texts)
        embed.assert_not_called()

    def test_partial_session_rejects_documents(self) -> None:
        """Require both model and tokenizer for nonempty batches."""

        for missing in ("_model", "_tokenizer"):
            with self.subTest(missing=missing):
                session = self.make_session()
                setattr(session, missing, None)
                with self.assertRaisesRegex(RuntimeError, "closed"):
                    session.embed_documents(["text"])

    def test_document_batch_preserves_text_and_vector_order(self) -> None:
        """Pass exact input strings through without formatting or sorting."""

        session = self.make_session()
        texts = ["  first\n", "second", "\u03b1\u03b2"]
        vectors = [[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]]
        with patch.object(
            session, "_embed_torch", return_value=vectors
        ) as embed:
            result = session.embed_documents(texts)
        embed.assert_called_once_with(texts)
        self.assertEqual(result, vectors)
        self.assertEqual(texts[0], "  first\n")

    def test_mlx_documents_dispatch_to_mlx_backend(self) -> None:
        """Call the MLX embedding path for MLX sessions."""

        session = self.make_session(backend="mlx")
        with (
            patch.object(
                session, "_embed_mlx", return_value=[[1.0, 0.0]]
            ) as mlx,
            patch.object(session, "_embed_torch") as torch,
        ):
            self.assertEqual(session.embed_documents(["text"]), [[1.0, 0.0]])
        mlx.assert_called_once_with(["text"])
        torch.assert_not_called()

    def test_query_does_not_inject_instructions(self) -> None:
        """Leave model-specific query formatting to the caller."""

        session = self.make_session()
        query = "Instruct: search\nQuery: what is diffusion?"
        with patch.object(
            session, "_embed_torch", return_value=[[0.0, 1.0]]
        ) as embed:
            result = session.embed_query(query)
        self.assertEqual(result, [0.0, 1.0])
        embed.assert_called_once_with([query])

    def test_invalid_queries_are_rejected(self) -> None:
        """Reject empty, whitespace-only, and non-string queries."""

        session = self.make_session()
        with patch.object(session, "_embed_torch") as embed:
            for query in ("", " \n", None, 42, False):
                with self.subTest(query=query):
                    with self.assertRaises(ValueError):
                        session.embed_query(query)
        embed.assert_not_called()

    def test_query_forwards_turn_identifier_and_purpose(self) -> None:
        """Associate the query embedding with its conversation turn."""

        session = self.make_session()
        with patch.object(
            session, "embed_documents", return_value=[[1.0, 0.0]]
        ) as embed:
            result = session.embed_query("question", trace_turn_id="turn-7")
        self.assertEqual(result, [1.0, 0.0])
        embed.assert_called_once_with(
            ["question"], trace_turn_id="turn-7", purpose="query"
        )

    def test_invalid_backend_vectors_fail_after_dispatch(self) -> None:
        """Validate results even if a backend returns malformed vectors."""

        session = self.make_session()
        for vectors in (
            [],
            [[1.0], [2.0]],
            [[float("nan")]],
            [[0.0]],
        ):
            with self.subTest(vectors=vectors):
                with patch.object(
                    session, "_embed_torch", return_value=vectors
                ):
                    with self.assertRaises(RuntimeError):
                        session.embed_documents(["text"])

    def test_repeated_batches_reuse_the_same_loaded_objects(self) -> None:
        """Reuse a loaded session across document and query operations."""

        session = self.make_session()
        model = session._model
        tokenizer = session._tokenizer
        with patch.object(
            session, "_embed_torch", return_value=[[1.0, 0.0]]
        ) as embed:
            session.embed_documents(["document"])
            session.embed_query("query")
        self.assertEqual(embed.call_count, 2)
        self.assertIs(session._model, model)
        self.assertIs(session._tokenizer, tokenizer)


class TestEmbeddingTracing(EmbeddingTestCase):
    """Check trace verbosity, identifiers, digests, and artifacts."""

    def test_document_purpose_suppresses_detailed_trace(self) -> None:
        """Avoid per-document trace events even when tracing is enabled."""

        trace = FakeTrace(enabled=True, include_content=True)
        session = self.make_session(trace=trace)
        with patch.object(
            session, "_embed_torch", return_value=[[1.0, 0.0]]
        ) as embed:
            self.assertEqual(session.embed_documents(["text"]), [[1.0, 0.0]])
        embed.assert_called_once_with(
            ["text"],
            operation_id=None,
            trace_turn_id=None,
            record_trace=False,
        )
        self.assertEqual(trace.find("embedding.input.prepared"), [])
        self.assertEqual(trace.find("embedding.output.completed"), [])
        self.assertEqual(len(trace.spans), 1)

    def test_disabled_trace_uses_simple_backend_call(self) -> None:
        """Avoid detailed backend arguments when tracing is disabled."""

        trace = FakeTrace(enabled=False)
        session = self.make_session(trace=trace)
        with patch.object(
            session, "_embed_torch", return_value=[[1.0, 0.0]]
        ) as embed:
            session.embed_documents(["text"], purpose="query")
        embed.assert_called_once_with(["text"])
        self.assertEqual(len(trace.find("embedding.input.prepared")), 1)

    def test_query_trace_records_hashes_without_raw_content(self) -> None:
        """Record hashes and vector digests with content capture disabled."""

        trace = FakeTrace(enabled=True)
        session = self.make_session(trace=trace)
        with patch.object(
            session, "_embed_torch", return_value=[[0.6, 0.8]]
        ) as embed:
            result = session.embed_documents(
                ["private question"],
                trace_turn_id="turn-3",
                purpose="query",
            )
        self.assertEqual(result, [[0.6, 0.8]])
        embed.assert_called_once_with(
            ["private question"],
            operation_id="embedding-1",
            trace_turn_id="turn-3",
            record_trace=True,
        )
        input_event = trace.find("embedding.input.prepared")[0]
        self.assertEqual(input_event["turn_id"], "turn-3")
        data = input_event["data"]
        self.assertEqual(data["operation_id"], "embedding-1")
        self.assertEqual(data["text_count"], 1)
        self.assertEqual(data["purpose"], "query")
        self.assertEqual(
            data["text_sha256"],
            [sha256(b"private question").hexdigest()],
        )
        self.assertNotIn("text_previews", data)
        output_event = trace.find("embedding.output.completed")[0]
        result_data = output_event["data"]
        self.assertEqual(result_data["dimensions"], 2)
        self.assertEqual(result_data["vector_count"], 1)
        self.assertIsNone(result_data["input_token_counts"])
        self.assertIsNone(result_data["input_token_count"])
        digest = sha256(b"[[0.6,0.8]]").hexdigest()
        self.assertEqual(result_data["vectors_sha256"], digest)
        self.assertNotIn("vectors_artifact", result_data)
        self.assertEqual(trace.spans[-1][0], "embedding.calculate")
        self.assertEqual(trace.spans[-1][1]["turn_id"], "turn-3")

    def test_content_previews_are_opt_in(self) -> None:
        """Include text previews only when requested by the trace sink."""

        trace = FakeTrace(include_content=True)
        session = self.make_session(trace=trace)
        with patch.object(
            session, "_embed_torch", return_value=[[1.0]]
        ):
            session.embed_documents(["one two three four"], purpose="search")
        prepared = trace.find("embedding.input.prepared")[0]["data"]
        self.assertEqual(
            prepared["text_previews"],
            [embeddings.trace_text_preview("one two three four")],
        )

    def test_vector_artifacts_are_opt_in(self) -> None:
        """Store vector artifacts only when explicitly requested."""

        trace = FakeTrace(include_embedding_vectors=True)
        session = self.make_session(trace=trace)
        with patch.object(
            session, "_embed_torch", return_value=[[0.0, 1.0]]
        ):
            session.embed_documents(["text"], purpose="query")
        self.assertEqual(
            trace.artifacts,
            [("embedding-1-vectors", [[0.0, 1.0]])],
        )
        result = trace.find("embedding.output.completed")[0]["data"]
        self.assertEqual(
            result["vectors_artifact"],
            {"artifact": "embedding-1-vectors"},
        )
        self.assertNotIn("vectors_sha256", result)

    def test_backend_token_counts_appear_in_trace(self) -> None:
        """Aggregate recorded input token counts in the result event."""

        trace = FakeTrace()
        session = self.make_session(trace=trace)

        def embed(texts: list[str], **kwargs: object) -> list[list[float]]:
            """Simulate a backend that recorded token counts."""

            session._last_embedding_input_token_counts = [2, 4]
            return [[1.0, 0.0], [0.0, 1.0]]

        with patch.object(session, "_embed_torch", side_effect=embed):
            session.embed_documents(["a", "b"], purpose="query")
        result = trace.find("embedding.output.completed")[0]["data"]
        self.assertEqual(result["input_token_counts"], [2, 4])
        self.assertEqual(result["input_token_count"], 6)

    def test_stale_token_counts_are_cleared_between_batches(self) -> None:
        """Do not carry backend token counts into a later operation."""

        trace = FakeTrace()
        session = self.make_session(trace=trace)
        with patch.object(
            session, "_embed_torch", return_value=[[1.0, 0.0]]
        ):
            session._last_embedding_input_token_counts = [99]
            session.embed_documents(["text"], purpose="query")
        result = trace.find("embedding.output.completed")[0]["data"]
        self.assertIsNone(result["input_token_counts"])

    def test_tracing_is_not_completed_when_vectors_are_invalid(self) -> None:
        """Do not emit a success event for malformed backend output."""

        trace = FakeTrace()
        session = self.make_session(trace=trace)
        with patch.object(session, "_embed_torch", return_value=[[0.0]]):
            with self.assertRaises(RuntimeError):
                session.embed_documents(["text"], purpose="query")
        self.assertEqual(trace.find("embedding.output.completed"), [])

    def test_calculation_duration_uses_elapsed_time(self) -> None:
        """Record duration in milliseconds without a wall-clock timestamp."""

        trace = FakeTrace()
        session = self.make_session(trace=trace)
        with (
            patch.object(
                session, "_embed_torch", return_value=[[1.0]]
            ),
            patch.object(
                embeddings,
                "perf_counter",
                side_effect=[1.0, 1.012345],
            ),
        ):
            session.embed_documents(["text"], purpose="query")
        result = trace.find("embedding.output.completed")[0]
        self.assertAlmostEqual(result["duration_ms"], 12.345)
        self.assertAlmostEqual(
            result["data"]["calculation_duration_ms"], 12.345
        )

    def test_mlx_tracing_uses_same_dispatch_contract(self) -> None:
        """Forward operation metadata to the MLX embedding path."""

        trace = FakeTrace()
        session = self.make_session(backend="mlx", trace=trace)
        with patch.object(
            session, "_embed_mlx", return_value=[[1.0, 0.0]]
        ) as embed:
            session.embed_documents(
                ["text"], purpose="query", trace_turn_id="turn-2"
            )
        embed.assert_called_once_with(
            ["text"],
            operation_id="embedding-1",
            trace_turn_id="turn-2",
            record_trace=True,
        )


class TestBackendLoading(EmbeddingTestCase):
    """Check backend import, tokenizer, and model-loading contracts."""

    def test_torch_loader_sets_tokenizer_model_and_device(self) -> None:
        """Load locally, move the model to the device, and enter eval mode."""

        session = self.make_session()
        model = Mock()
        tokenizer = object()
        from_tokenizer = Mock(return_value=tokenizer)
        from_model = Mock(return_value=model)
        torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=Mock(return_value=True)),
            backends=SimpleNamespace(
                mps=SimpleNamespace(is_available=Mock(return_value=False))
            ),
        )
        transformers = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=from_tokenizer),
            AutoModel=SimpleNamespace(from_pretrained=from_model),
        )
        with (
            patch.object(
                embeddings,
                "import_module",
                side_effect=lambda name: {
                    "torch": torch,
                    "transformers": transformers,
                }[name],
            ),
            patch.object(
                embeddings,
                "torch_device",
                return_value="cuda",
            ) as select_device,
        ):
            session._load_torch("/snapshot")
        from_tokenizer.assert_called_once_with(
            "/snapshot",
            local_files_only=True,
            trust_remote_code=False,
            padding_side="right",
        )
        from_model.assert_called_once_with(
            "/snapshot",
            local_files_only=True,
            trust_remote_code=False,
            dtype="auto",
        )
        select_device.assert_called_once_with(True, False)
        model.to.assert_called_once_with("cuda")
        model.eval.assert_called_once_with()
        self.assertIs(session._tokenizer, tokenizer)
        self.assertIs(session._model, model)
        self.assertEqual(session.device, "cuda")

    def test_mlx_loader_requires_metal(self) -> None:
        """Reject MLX when Metal is not available."""

        session = self.make_session(backend="mlx")
        mx = SimpleNamespace(
            metal=SimpleNamespace(is_available=Mock(return_value=False))
        )
        with patch.object(
            embeddings, "import_module", return_value=mx
        ) as import_module:
            with self.assertRaisesRegex(RuntimeError, "Metal"):
                session._load_mlx("/snapshot")
        import_module.assert_called_once_with("mlx.core")

    def test_mlx_loader_uses_local_snapshot(self) -> None:
        """Load MLX weights and tokenizer from the resolved directory."""

        session = self.make_session(backend="mlx")
        model = object()
        tokenizer = object()
        from_tokenizer = Mock(return_value=tokenizer)
        load_model = Mock(return_value=model)
        modules = {
            "mlx.core": SimpleNamespace(
                metal=SimpleNamespace(is_available=lambda: True)
            ),
            "transformers": SimpleNamespace(
                AutoTokenizer=SimpleNamespace(
                    from_pretrained=from_tokenizer
                )
            ),
            "mlx_embeddings.utils": SimpleNamespace(load_model=load_model),
        }
        with patch.object(
            embeddings, "import_module", side_effect=modules.__getitem__
        ):
            session._load_mlx("/snapshot")
        from_tokenizer.assert_called_once_with(
            "/snapshot",
            local_files_only=True,
            trust_remote_code=False,
            padding_side="right",
        )
        load_model.assert_called_once_with(Path("/snapshot"))
        self.assertIs(session._model, model)
        self.assertIs(session._tokenizer, tokenizer)
        self.assertEqual(session.device, "metal")

    def test_backend_import_failure_is_not_suppressed(self) -> None:
        """Propagate missing optional dependencies from each loader."""

        for backend, method in (
            ("torch", "_load_torch"),
            ("mlx", "_load_mlx"),
        ):
            with self.subTest(backend=backend):
                session = self.make_session(backend=backend)
                with patch.object(
                    embeddings,
                    "import_module",
                    side_effect=ImportError("missing dependency"),
                ):
                    with self.assertRaisesRegex(
                        ImportError, "missing dependency"
                    ):
                        getattr(session, method)("/snapshot")


class TestBackendEmbedding(EmbeddingTestCase):
    """Exercise pooling and normalization with deterministic fake arrays."""

    def make_torch_inputs(self) -> tuple[FakeBatch, FakeTorch, Mock]:
        """Return padded inputs and hidden states with known pooled vectors."""

        batch = FakeBatch(
            ids=[[11, 12, 13, 0], [21, 22, 0, 0]],
            mask=[[1, 1, 1, 0], [1, 1, 0, 0]],
        )
        hidden = FakeTensor(
            [
                [[1, 0], [0, 1], [3, 4], [99, 99]],
                [[1, 1], [0, 5], [99, 99], [99, 99]],
            ]
        )
        torch = FakeTorch()

        def model_call(**inputs: object) -> SimpleNamespace:
            """Assert inference mode is active during model evaluation."""

            self.assertEqual(torch.inference_depth, 1)
            self.assertEqual(inputs["attention_mask"].device, "test-device")
            return SimpleNamespace(last_hidden_state=hidden)

        model = Mock(side_effect=model_call)
        return batch, torch, model

    def test_torch_last_unmasked_pooling_and_normalization(self) -> None:
        """Select final real tokens rather than padded trailing positions."""

        session = self.make_session()
        batch, torch, model = self.make_torch_inputs()
        session._model = model
        tokenizer = Mock(return_value=batch)
        session._tokenizer = tokenizer
        with patch.object(
            embeddings, "import_module", return_value=torch
        ) as importer:
            vectors = session._embed_torch(["first", "second"])
        importer.assert_called_once_with("torch")
        tokenizer.assert_called_once_with(
            ["first", "second"],
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        self.assertEqual(batch.requested_device, "test-device")
        self.assertEqual(torch.inference_depth, 0)
        self.assertEqual(torch.normalization_calls, [(2, 1)])
        self.assertEqual(len(vectors), 2)
        np.testing.assert_allclose(vectors, [[0.6, 0.8], [0.0, 1.0]])

    def test_torch_pooling_handles_left_padding(self) -> None:
        """Find the last unmasked token with left-padded sequences."""

        session = self.make_session()
        batch = FakeBatch(
            ids=[[0, 0, 11, 12], [0, 21, 22, 23]],
            mask=[[0, 0, 1, 1], [0, 1, 1, 1]],
        )
        hidden = FakeTensor(
            [
                [[99, 99], [99, 99], [1, 0], [3, 4]],
                [[99, 99], [1, 0], [0, 1], [0, 5]],
            ]
        )
        session._tokenizer = Mock(return_value=batch)
        session._model = Mock(
            return_value=SimpleNamespace(last_hidden_state=hidden)
        )
        with patch.object(
            embeddings, "import_module", return_value=FakeTorch()
        ):
            vectors = session._embed_torch(["first", "second"])
        np.testing.assert_allclose(vectors, [[0.6, 0.8], [0.0, 1.0]])

    def test_torch_public_embedding_pipeline(self) -> None:
        """Exercise public dispatch, pooling, validation, and trace counts."""

        trace = FakeTrace()
        session = self.make_session(trace=trace)
        batch, torch, model = self.make_torch_inputs()
        session._model = model
        session._tokenizer = Mock(return_value=batch)
        with patch.object(
            embeddings, "import_module", return_value=torch
        ):
            vectors = session.embed_documents(
                ["first", "second"],
                purpose="query",
                trace_turn_id="turn-5",
            )
        np.testing.assert_allclose(vectors, [[0.6, 0.8], [0.0, 1.0]])
        result = trace.find("embedding.output.completed")[0]["data"]
        self.assertEqual(result["input_token_counts"], [3, 2])
        self.assertEqual(result["input_token_count"], 5)
        self.assertEqual(result["dimensions"], 2)

    def test_torch_trace_records_tokens_and_artifacts(self) -> None:
        """Record attention-mask counts and optional token-ID artifacts."""

        trace = FakeTrace(include_token_ids=True)
        session = self.make_session(trace=trace)
        batch, torch, model = self.make_torch_inputs()
        session._model = model
        session._tokenizer = Mock(return_value=batch)
        with patch.object(
            embeddings, "import_module", return_value=torch
        ):
            session._embed_torch(
                ["first", "second"],
                operation_id="operation-7",
                trace_turn_id="turn-1",
                record_trace=True,
            )
        self.assertEqual(session._last_embedding_input_token_counts, [3, 2])
        events = trace.find(TRACE_EVENT_EMBEDDING_TOKENS_PREPARED)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["turn_id"], "turn-1")
        data = events[0]["data"]
        self.assertEqual(data["operation_id"], "operation-7")
        self.assertEqual(data["input_token_counts"], [3, 2])
        self.assertEqual(data["input_token_count"], 5)
        self.assertEqual(len(trace.artifacts), 2)
        self.assertEqual(trace.artifacts[0][1], batch["input_ids"].tolist())
        self.assertEqual(
            trace.artifacts[1][1], batch["attention_mask"].tolist()
        )

    def test_torch_trace_omits_token_ids_by_default(self) -> None:
        """Avoid recording token-ID arrays without explicit permission."""

        trace = FakeTrace()
        session = self.make_session(trace=trace)
        batch, torch, model = self.make_torch_inputs()
        session._model = model
        session._tokenizer = Mock(return_value=batch)
        with patch.object(
            embeddings, "import_module", return_value=torch
        ):
            session._embed_torch(["first", "second"])
        self.assertEqual(len(trace.find(
            TRACE_EVENT_EMBEDDING_TOKENS_PREPARED
        )), 1)
        self.assertEqual(trace.artifacts, [])

    def test_torch_trace_is_suppressed_for_documents(self) -> None:
        """Skip token-count trace work when detailed tracing is disabled."""

        trace = FakeTrace(include_token_ids=True)
        session = self.make_session(trace=trace)
        batch, torch, model = self.make_torch_inputs()
        session._model = model
        session._tokenizer = Mock(return_value=batch)
        with patch.object(
            embeddings, "import_module", return_value=torch
        ):
            session._embed_torch(["first", "second"], record_trace=False)
        self.assertEqual(
            trace.find(TRACE_EVENT_EMBEDDING_TOKENS_PREPARED), []
        )
        self.assertEqual(trace.artifacts, [])

    def test_torch_disabled_sink_skips_token_trace(self) -> None:
        """Do not trace backend tokens when the sink is disabled."""

        trace = FakeTrace(enabled=False, include_token_ids=True)
        session = self.make_session(trace=trace)
        batch, torch, model = self.make_torch_inputs()
        session._model = model
        session._tokenizer = Mock(return_value=batch)
        with patch.object(
            embeddings, "import_module", return_value=torch
        ):
            session._embed_torch(["first", "second"])
        self.assertEqual(
            trace.find(TRACE_EVENT_EMBEDDING_TOKENS_PREPARED), []
        )

    def make_mlx_inputs(self) -> tuple[dict[str, np.ndarray], FakeMLX, Mock]:
        """Return MLX inputs and a model producing known embedding vectors."""

        inputs = {
            "input_ids": np.array([[11, 12, 0], [21, 22, 23]]),
            "attention_mask": np.array([[1, 1, 0], [1, 1, 1]]),
        }
        mx = FakeMLX()
        model = Mock(
            return_value=SimpleNamespace(
                text_embeds=FakeMLXArray([[3, 4], [5, 12]])
            )
        )
        return inputs, mx, model

    def test_mlx_normalizes_text_embeds(self) -> None:
        """Normalize model-provided text embeddings and evaluate arrays."""

        session = self.make_session(backend="mlx")
        inputs, mx, model = self.make_mlx_inputs()
        session._model = model
        tokenizer = Mock(return_value=inputs)
        session._tokenizer = tokenizer
        with patch.object(
            embeddings, "import_module", return_value=mx
        ) as importer:
            vectors = session._embed_mlx(["first", "second"])
        importer.assert_called_once_with("mlx.core")
        tokenizer.assert_called_once_with(
            ["first", "second"],
            padding=True,
            truncation=False,
            return_tensors="np",
        )
        self.assertEqual(len(mx.evaluated), 1)
        self.assertEqual(model.call_count, 1)
        args, kwargs = model.call_args
        self.assertEqual(kwargs["attention_mask"].tolist(),
                         inputs["attention_mask"].tolist())
        self.assertEqual(args[0].tolist(), inputs["input_ids"].tolist())
        np.testing.assert_allclose(
            vectors,
            [[0.6, 0.8], [5 / 13, 12 / 13]],
            rtol=1e-6,
        )

    def test_mlx_public_embedding_pipeline(self) -> None:
        """Exercise MLX public dispatch, normalization, and trace counts."""

        trace = FakeTrace()
        session = self.make_session(backend="mlx", trace=trace)
        inputs, mx, model = self.make_mlx_inputs()
        session._model = model
        session._tokenizer = Mock(return_value=inputs)
        with patch.object(
            embeddings, "import_module", return_value=mx
        ):
            vectors = session.embed_documents(
                ["first", "second"],
                purpose="query",
                trace_turn_id="turn-6",
            )
        np.testing.assert_allclose(
            vectors,
            [[0.6, 0.8], [5 / 13, 12 / 13]],
            rtol=1e-6,
        )
        result = trace.find("embedding.output.completed")[0]["data"]
        self.assertEqual(result["input_token_counts"], [2, 3])
        self.assertEqual(result["input_token_count"], 5)
        self.assertEqual(result["dimensions"], 2)

    def test_mlx_trace_records_tokens_and_artifacts(self) -> None:
        """Record MLX attention-mask token counts and optional token IDs."""

        trace = FakeTrace(include_token_ids=True)
        session = self.make_session(backend="mlx", trace=trace)
        inputs, mx, model = self.make_mlx_inputs()
        session._model = model
        session._tokenizer = Mock(return_value=inputs)
        with patch.object(
            embeddings, "import_module", return_value=mx
        ):
            session._embed_mlx(
                ["first", "second"],
                operation_id="operation-3",
                trace_turn_id="turn-4",
            )
        self.assertEqual(session._last_embedding_input_token_counts, [2, 3])
        event = trace.find(TRACE_EVENT_EMBEDDING_TOKENS_PREPARED)[0]
        self.assertEqual(event["turn_id"], "turn-4")
        self.assertEqual(event["data"]["input_token_count"], 5)
        self.assertEqual(len(trace.artifacts), 2)
        self.assertEqual(trace.artifacts[0][1], inputs["input_ids"].tolist())
        self.assertEqual(
            trace.artifacts[1][1], inputs["attention_mask"].tolist()
        )

    def test_mlx_fake_sink_can_decline_token_artifacts(self) -> None:
        """Honor a fake sink that declines token-array recording."""

        trace = FakeTrace()
        session = self.make_session(backend="mlx", trace=trace)
        inputs, mx, model = self.make_mlx_inputs()
        session._model = model
        session._tokenizer = Mock(return_value=inputs)
        with patch.object(
            embeddings, "import_module", return_value=mx
        ):
            session._embed_mlx(["first", "second"])
        self.assertEqual(len(trace.find(
            TRACE_EVENT_EMBEDDING_TOKENS_PREPARED
        )), 1)
        self.assertEqual(trace.artifacts, [])

    def test_mlx_trace_is_suppressed_for_documents(self) -> None:
        """Skip detailed MLX tracing when requested by the caller."""

        trace = FakeTrace(include_token_ids=True)
        session = self.make_session(backend="mlx", trace=trace)
        inputs, mx, model = self.make_mlx_inputs()
        session._model = model
        session._tokenizer = Mock(return_value=inputs)
        with patch.object(
            embeddings, "import_module", return_value=mx
        ):
            session._embed_mlx(["first", "second"], record_trace=False)
        self.assertEqual(
            trace.find(TRACE_EVENT_EMBEDDING_TOKENS_PREPARED), []
        )
        self.assertEqual(trace.artifacts, [])

    def test_mlx_disabled_sink_skips_token_trace(self) -> None:
        """Skip token tracing when the trace sink is disabled."""

        trace = FakeTrace(enabled=False, include_token_ids=True)
        session = self.make_session(backend="mlx", trace=trace)
        inputs, mx, model = self.make_mlx_inputs()
        session._model = model
        session._tokenizer = Mock(return_value=inputs)
        with patch.object(
            embeddings, "import_module", return_value=mx
        ):
            session._embed_mlx(["first", "second"])
        self.assertEqual(
            trace.find(TRACE_EVENT_EMBEDDING_TOKENS_PREPARED), []
        )


class TestSessionCleanup(EmbeddingTestCase):
    """Check reference release and backend cache cleanup."""

    def test_close_releases_references_and_collects_garbage(self) -> None:
        """Clear both loaded objects and request cache cleanup."""

        session = self.make_session()
        with (
            patch.object(embeddings.gc, "collect") as collect,
            patch.object(session, "_clear_backend_cache") as clear,
        ):
            session.close()
        self.assertIsNone(session._model)
        self.assertIsNone(session._tokenizer)
        collect.assert_called_once_with()
        clear.assert_called_once_with()

    def test_close_can_be_called_repeatedly(self) -> None:
        """Repeated close requests leave the session in a closed state."""

        session = self.make_session()
        with (
            patch.object(embeddings.gc, "collect") as collect,
            patch.object(session, "_clear_backend_cache") as clear,
        ):
            session.close()
            session.close()
        self.assertIsNone(session._model)
        self.assertIsNone(session._tokenizer)
        self.assertEqual(collect.call_count, 2)
        self.assertEqual(clear.call_count, 2)

    def test_close_rejects_subsequent_embedding(self) -> None:
        """Reject nonempty embedding operations after closing."""

        session = self.make_session()
        with (
            patch.object(embeddings.gc, "collect"),
            patch.object(session, "_clear_backend_cache"),
        ):
            session.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            session.embed_documents(["text"])
        with self.assertRaisesRegex(RuntimeError, "closed"):
            session.embed_query("question")

    def test_mlx_cache_is_cleared(self) -> None:
        """Call MLX's cache release function for MLX sessions."""

        session = self.make_session(backend="mlx")
        clear = Mock()
        with patch.object(
            embeddings,
            "import_module",
            return_value=SimpleNamespace(clear_cache=clear),
        ) as importer:
            session._clear_backend_cache()
        importer.assert_called_once_with("mlx.core")
        clear.assert_called_once_with()

    def test_torch_cache_cleanup_checks_cuda_and_mps(self) -> None:
        """Clear only caches available on the current PyTorch runtime."""

        session = self.make_session()
        for cuda_available, mps_available in (
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ):
            with self.subTest(cuda=cuda_available, mps=mps_available):
                cuda_clear = Mock()
                mps_clear = Mock()
                torch = SimpleNamespace(
                    cuda=SimpleNamespace(
                        is_available=Mock(return_value=cuda_available),
                        empty_cache=cuda_clear,
                    ),
                    backends=SimpleNamespace(
                        mps=SimpleNamespace(
                            is_available=Mock(return_value=mps_available),
                        )
                    ),
                    mps=SimpleNamespace(empty_cache=mps_clear),
                )
                with patch.object(
                    embeddings, "import_module", return_value=torch
                ):
                    session._clear_backend_cache()
                self.assertEqual(cuda_clear.call_count, int(cuda_available))
                self.assertEqual(mps_clear.call_count, int(mps_available))


if __name__ == "__main__":
    unittest.main()
