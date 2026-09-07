"""Tests for switchable local embedding sessions."""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from qachat import embeddings


class FakeTokenizer:
    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool,
    ) -> list[str]:
        del add_special_tokens, truncation
        return text.split()


def _load_fake(session: embeddings.EmbeddingSession, source: str) -> None:
    del source
    session._model = object()
    session._tokenizer = FakeTokenizer()
    session.device = "test-device"


class TestEmbeddings(unittest.TestCase):

    def test_embedding_session(self) -> None:
        with patch.object(
            embeddings,
            "_model_source",
            return_value="/snapshot",
        ):
            with patch.object(
                embeddings.EmbeddingSession,
                "_load_torch",
                autospec=True,
                side_effect=_load_fake,
            ) as load_torch:
                torch_session = embeddings.EmbeddingSession(
                    "group/torch-embedding",
                    backend="torch",
                    allow_download=False,
                )

            with patch.object(
                embeddings.EmbeddingSession,
                "_load_mlx",
                autospec=True,
                side_effect=_load_fake,
            ) as load_mlx:
                with patch(
                    "qachat.runtime.platform.system",
                    return_value="Darwin",
                ):
                    with patch(
                        "qachat.runtime.platform.machine",
                        return_value="arm64",
                    ):
                        mlx_session = embeddings.EmbeddingSession(
                            "group/mlx-embedding",
                            backend="mlx",
                            allow_download=False,
                        )

        load_torch.assert_called_once()
        load_mlx.assert_called_once()
        self.assertEqual(torch_session.backend, "torch")
        self.assertEqual(mlx_session.backend, "mlx")
        self.assertEqual(torch_session.count_tokens("one two"), 2)

        captured: list[str] = []

        def fake_torch(texts: list[str]) -> list[list[float]]:
            captured.extend(texts)
            return [[3.0, 4.0] for _ in texts]

        with patch.object(
            torch_session,
            "_embed_torch",
            side_effect=fake_torch,
        ):
            documents = torch_session.embed_documents(["document"])
            query = torch_session.embed_query("question")

        self.assertEqual(documents, [[3.0, 4.0]])
        self.assertEqual(query, [3.0, 4.0])
        self.assertEqual(captured, ["document", "question"])

        with patch.object(
            mlx_session,
            "_embed_mlx",
            return_value=[[1.0, 0.0]],
        ) as embed_mlx:
            self.assertEqual(
                mlx_session.embed_documents(["document"]),
                [[1.0, 0.0]],
            )
        embed_mlx.assert_called_once_with(["document"])

        invalid_settings = (
            {"model_id": ""},
            {"model_id": "group/model", "allow_download": "yes"},
        )
        for settings in invalid_settings:
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    embeddings.EmbeddingSession(**settings)

        with self.assertRaises(ValueError):
            torch_session.embed_documents([""])
        with self.assertRaises(ValueError):
            torch_session.embed_documents("text")

        with patch.object(
            embeddings,
            "import_module",
            return_value=SimpleNamespace(
                cuda=SimpleNamespace(
                    is_available=lambda: False,
                    empty_cache=lambda: None,
                ),
                backends=SimpleNamespace(
                    mps=SimpleNamespace(is_available=lambda: False),
                ),
            ),
        ):
            torch_session.close()
        with self.assertRaises(RuntimeError):
            torch_session.count_tokens("closed")

    def test_model_source(self) -> None:
        with TemporaryDirectory() as directory:
            self.assertEqual(
                embeddings._model_source(
                    directory,
                    allow_download=False,
                    revision=None,
                ),
                directory,
            )

        with patch.object(
            embeddings,
            "snapshot_download",
            return_value="/cache/snapshot",
        ) as download:
            source = embeddings._model_source(
                "group/model",
                allow_download=False,
                revision="abc",
            )
        self.assertEqual(source, "/cache/snapshot")
        download.assert_called_once_with(
            repo_id="group/model",
            revision="abc",
            local_files_only=True,
        )

    def test_check_vectors(self) -> None:
        self.assertEqual(
            embeddings._check_vectors([[1, 2]], 1),
            [[1.0, 2.0]],
        )
        invalid = (
            ([], 1),
            ([[1.0], [1.0, 2.0]], 2),
            ([[0.0, 0.0]], 1),
            ([[True, 1.0]], 1),
            ([[float("nan"), 1.0]], 1),
        )
        for vectors, expected in invalid:
            with self.subTest(vectors=vectors):
                with self.assertRaises(RuntimeError):
                    embeddings._check_vectors(vectors, expected)


if __name__ == "__main__":
    unittest.main()
