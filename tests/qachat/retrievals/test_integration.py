"""Exercise retrieval pipelines using synthetic and real local embeddings.

The first integration path uses real source extraction, chunking, graph
building, lexical ranking, fusion, routing, and persistence with a controlled
embedding adapter. The backend tests use locally generated tiny model weights
and the real QAChat EmbeddingSession when optional libraries are installed.
No test downloads model weights or requires an external service.

"""

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

import numpy as np

from qachat import embeddings
from qachat.chunking import chunk_source_units
from qachat.embeddings import EmbeddingSession
from qachat.retrievals.model_identity import embedding_model_identity
from qachat.retrievals import (
    CodeGraph,
    build_code_graph,
    build_lexical_index,
    build_semantic_index,
    code_search,
    expand_retrieval_context,
    hybrid_search,
    indexes_are_current,
    lexical_search,
    load_indexes,
    plan_retrieval,
    prepare_persistent_indexes,
    read_document,
    save_indexes,
    semantic_matrix,
    semantic_search,
)
from qachat.sources.sources import extract_source_units
from tests.qachat.retrievals._support import (
    FakeEmbeddings,
    chunk,
    source_file,
)


class TestRetrievalEndToEnd(unittest.TestCase):
    """Check source-to-index-to-answer evidence without optional models."""

    def test_python_extraction_to_persisted_hybrid_context(self) -> None:
        """Retain source provenance through the complete retrieval workflow.

        """
        source = source_file(
            "pkg/calculation.py",
            "def add(a, b):\n    return a + b\n\n"
            "def twice(value):\n    return add(value, value)\n",
        )
        units = extract_source_units([source])
        self.assertTrue(units)
        chunks = chunk_source_units(
            units, count_tokens=len, target_tokens=100,
            max_tokens=150, overlap_tokens=0,
        )
        self.assertTrue(chunks)
        graph = build_code_graph([source], units)
        self.assertTrue(graph.entities)
        session = FakeEmbeddings()
        semantic = build_semantic_index(chunks, session, batch_size=2)
        lexical = build_lexical_index(chunks)
        query = "Explain twice"
        semantic_results = semantic_search(
            semantic, query, session, count=5
        )
        lexical_results = lexical_search(lexical, query, count=5)
        code_results = code_search(graph, chunks, query, count=5)
        fused = hybrid_search(
            semantic=semantic_results,
            lexical=lexical_results,
            code=code_results,
            count=5,
        )
        self.assertTrue(fused)
        plan = plan_retrieval(query, graph)
        expanded = expand_retrieval_context(fused, chunks, graph, plan)
        self.assertTrue(expanded)
        self.assertEqual(len({x.identifier for x in expanded}),
                         len(expanded))
        selected = read_document(
            query, chunks, graph, fused, max_tokens=1000
        )
        self.assertIsNotNone(selected)
        self.assertEqual(selected.path, source.path)

        indexes = prepare_persistent_indexes(semantic, lexical, graph)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "retrieval.json"
            save_indexes(indexes, path)
            loaded = load_indexes(path)
            self.assertEqual(loaded, indexes)
            self.assertTrue(indexes_are_current(loaded, chunks, graph))
            search = semantic_search(
                loaded.semantic, query, session, backend="numpy",
                matrix=semantic_matrix(loaded.semantic),
            )
            self.assertTrue(search)


class TestNativeTorchEmbeddingRetrieval(unittest.TestCase):
    """Exercise real torch tensor embeddings without loading Transformers."""

    def test_real_torch_vectors_match_both_search_backends(self) -> None:
        """Normalized torch embeddings work with loop and NumPy ranking.

        """
        if find_spec("torch") is None:
            self.skipTest("PyTorch is not installed.")
        import torch

        class TorchEmbeddings:
            """Use deterministic tensor encodings for a tiny vocabulary."""

            def __init__(self) -> None:
                """Set up fixed word vectors and index bookkeeping.

                """
                self.cleared = 0
                self.table = {
                    "physics": torch.tensor([3.0, 0.0, 0.0]),
                    "biology": torch.tensor([0.0, 3.0, 0.0]),
                    "chemistry": torch.tensor([0.0, 0.0, 3.0]),
                }

            def encode(self, text: str) -> list[float]:
                """Produce a normalized mean of vocabulary vectors.

                Parameters
                ----------
                text: str
                    Vocabulary text to encode.

                Returns
                -------
                list[float]
                    Tensor-computed normalized vector.

                """
                words = text.split()
                vectors = torch.stack([self.table[word] for word in words])
                pooled = vectors.mean(dim=0)
                return torch.nn.functional.normalize(
                    pooled, p=2, dim=0
                ).tolist()

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                """Compute genuine tensor embeddings for source texts.

                Parameters
                ----------
                texts: list[str]
                    Document text.

                Returns
                -------
                list[list[float]]
                    Normalized embeddings.

                """
                return [self.encode(text) for text in texts]

            def embed_query(self, query_text: str) -> list[float]:
                """Compute a tensor embedding for a query.

                Parameters
                ----------
                query_text: str
                    Query text.

                Returns
                -------
                list[float]
                    Normalized embedding.

                """
                return self.encode(query_text)

            def _clear_backend_cache(self) -> None:
                """Record cache-release requests without GPU requirements.

                """
                self.cleared += 1

        session = TorchEmbeddings()
        chunks = [
            chunk("physics", "physics"),
            chunk("biology", "biology"),
            chunk("chemistry", "chemistry"),
        ]
        index = build_semantic_index(chunks, session, batch_size=2)
        loop = semantic_search(index, "physics", session, backend="loop")
        numpy = semantic_search(
            index, "physics", session, backend="numpy"
        )
        self.assertEqual(loop[0].chunk.identifier, "physics")
        self.assertEqual(numpy[0].chunk.identifier, "physics")
        self.assertAlmostEqual(loop[0].score, 1.0)
        self.assertAlmostEqual(numpy[0].score, 1.0, places=6)
        self.assertEqual(session.cleared, 2)


def _require_dependencies(*modules: str) -> None:
    """Skip a real-model integration test only for absent dependencies.

    Parameters
    ----------
    *modules: str
        Optional top-level module names.

    Raises
    ------
    unittest.SkipTest
        If at least one required module is not installed.

    """
    missing = [name for name in modules if find_spec(name) is None]
    if missing:
        raise unittest.SkipTest("Missing optional dependencies: " +
                                ", ".join(missing))


def _save_tokenizer(path: Path) -> None:
    """Save one deterministic local tokenizer with no remote downloads.

    Parameters
    ----------
    path: pathlib.Path
        Directory receiving the tokenizer configuration.

    """
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    words = {
        "physics": 0,
        "biology": 1,
        "chemistry": 2,
        "question": 3,
        "<unk>": 4,
        "<pad>": 5,
    }
    tokenizer = Tokenizer(
        WordLevel(vocab=words, unk_token="<unk>")
    )
    tokenizer.pre_tokenizer = Whitespace()
    pretrained = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
    )
    pretrained.save_pretrained(path)


def _save_tiny_torch_model(path: Path) -> None:
    """Generate one tiny local Qwen3 model without external assets.

    Parameters
    ----------
    path: pathlib.Path
        New snapshot directory.

    """
    import torch
    from transformers import Qwen3Config, Qwen3Model

    path.mkdir()
    _save_tokenizer(path)
    config = Qwen3Config(
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
        Qwen3Model(config).save_pretrained(path)


def _save_tiny_mlx_model(path: Path) -> None:
    """Generate a tiny local MLX embedding-model snapshot.

    Parameters
    ----------
    path: pathlib.Path
        New model directory.

    """
    import mlx.core as mx
    from mlx_embeddings.models import qwen3

    path.mkdir()
    _save_tokenizer(path)
    args = qwen3.ModelArgs(
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
    model = qwen3.Model(args)
    model.apply(mx.ones_like)
    model.eval()
    (path / "config.json").write_text(
        json.dumps(asdict(args)), encoding="utf-8"
    )
    model.save_weights(str(path / "model.safetensors"))


class TestLocalQwen3ModelRetrieval(unittest.TestCase):
    """Exercise real QAChat sessions with tiny offline model snapshots."""

    def _exercise_backend(self, backend: str, model_path: Path) -> None:
        """Check actual embeddings through both semantic search paths.

        Parameters
        ----------
        backend: str
            Actual backend identifier.
        model_path: pathlib.Path
            Local model snapshot with compatible weights.

        """
        with (
            patch.object(
                embeddings,
                "snapshot_download",
                side_effect=AssertionError("Download attempted."),
            ),
            patch.object(
                embeddings,
                "cached_model_snapshot",
                side_effect=AssertionError("Cache lookup attempted."),
            ),
        ):
            session = EmbeddingSession(
                str(model_path), backend=backend, allow_download=False
            )
            try:
                chunks = [
                    chunk("physics", "physics"),
                    chunk("biology", "biology"),
                    chunk("chemistry", "chemistry"),
                ]
                index = build_semantic_index(chunks, session, batch_size=2)
                self.assertEqual(len(index.embeddings), 3)
                for vector in index.embeddings:
                    self.assertTrue(all(math.isfinite(value)
                                        for value in vector))
                    self.assertAlmostEqual(
                        math.sqrt(sum(x * x for x in vector)),
                        1.0, delta=2e-4,
                    )
                loop = semantic_search(
                    index, "physics", session, backend="loop", count=3
                )
                numpy = semantic_search(
                    index, "physics", session, backend="numpy", count=3,
                    matrix=semantic_matrix(index),
                )
                self.assertEqual(len(loop), 3)
                self.assertEqual(len(numpy), 3)
                left = {r.chunk.identifier: r.score for r in loop}
                right = {r.chunk.identifier: r.score for r in numpy}
                self.assertEqual(set(left), set(right))
                for name in left:
                    self.assertAlmostEqual(
                        left[name], right[name], delta=2e-3
                    )
                self.assertGreaterEqual(loop[0].score, 0.0)
                self.assertTrue(np.isfinite([r.score for r in numpy]).all())

                # Persist genuinely computed backend vectors together with
                # the model identity, then query the loaded index.
                identity = embedding_model_identity(
                    session.model_id, session.model_source,
                    cache_directory=model_path.parent / "identity-cache",
                )
                persistent = prepare_persistent_indexes(
                    index,
                    build_lexical_index(chunks),
                    CodeGraph((), ()),
                    embedding_model_identity=identity,
                )
                target = model_path.parent / f"{backend}-index.json"
                save_indexes(persistent, target)
                restored = load_indexes(target)
                self.assertEqual(restored, persistent)
                self.assertEqual(restored.embedding_model_identity, identity)
                restored_results = semantic_search(
                    restored.semantic,
                    "physics",
                    session,
                    backend="numpy",
                )
                self.assertEqual(
                    [item.chunk.identifier for item in restored_results],
                    [item.chunk.identifier for item in numpy],
                )
            finally:
                session.close()

    def test_local_torch_model_retrieval(self) -> None:
        """Load genuine tiny Qwen3 weights through the PyTorch backend.

        """
        _require_dependencies("torch", "transformers", "tokenizers")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "torch-model"
            _save_tiny_torch_model(path)
            self._exercise_backend("torch", path)

    def test_local_mlx_model_retrieval(self) -> None:
        """Load genuine local MLX model weights on Apple Silicon.

        """
        apple = platform.system() == "Darwin" and (
            platform.machine().lower() == "arm64"
        )
        if not apple:
            self.skipTest("MLX requires native Apple Silicon.")
        _require_dependencies(
            "mlx", "mlx_embeddings", "transformers", "tokenizers"
        )
        import mlx.core as mx

        if not mx.metal.is_available():
            self.skipTest("MLX Metal is unavailable.")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "mlx-model"
            _save_tiny_mlx_model(path)
            self._exercise_backend("mlx", path)


if __name__ == "__main__":
    unittest.main()
