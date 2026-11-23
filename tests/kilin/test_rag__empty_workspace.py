"""Exercise an empty workspace through real RAG and persistence orchestration.

Only model adapters and backend selection are replaced. Project creation,
source scans, indexing, answers, checkpoints, reopen and refresh use the
production implementation without downloads or native model dependencies.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from kilin import (
    add_project_sources, open_project, prepare_sources_workspace, rag,
)
from kilin.conversation_format import conversation_state
from kilin.trace_logging import TraceConfig
from tests.kilin.retrievals._support import FakeEmbeddings
from tests.kilin._rag_support import _session


class TestEmptyWorkspaceRAG(unittest.TestCase):
    """Support chatting before sources and indexing after imports."""

    def test_empty_chat_reopen_import_and_remove_sources(self) -> None:
        """Keep conversations usable across every source-collection state."""

        for tracing in (False, True):
            with self.subTest(tracing=tracing), TemporaryDirectory() as temp:
                root = Path(temp)
                project = prepare_sources_workspace(
                    [], "code", root / "project"
                )
                embedding = FakeEmbeddings()
                embedding.model_id = "test-model"
                embedding.model_source = "test-model-source"
                embedding.count_tokens = lambda text: len(text.split())
                embedding.close = Mock()
                chat = _session().chat_session
                with (
                    patch.object(rag, "choose_backend", return_value="mlx"),
                    patch.object(rag, "ChatSession", return_value=chat),
                    patch.object(rag, "EmbeddingSession",
                                 return_value=embedding),
                    patch.object(rag, "embedding_model_identity",
                                 return_value="a" * 64) as identity,
                ):
                    session = rag.RAGSession(
                        project.root, trace=TraceConfig(enabled=tracing)
                    )
                    try:
                        self.assertEqual(session.index_status, "empty")
                        self.assertEqual(list(project.indexes.iterdir()), [])
                        self.assertEqual(session._source_files(), {})
                        self.assertEqual(
                            session.ask("Hello", use_router=False),
                            "model answer",
                        )
                        self.assertEqual(session.turns[-1].evidence, ())
                        identity.assert_not_called()
                        self.assertEqual(embedding.calls, [])
                        self.assertEqual(embedding.queries, [])
                        state_path = session.conversation_path
                        state = conversation_state(json.loads(
                            state_path.read_text(encoding="utf-8")
                        ))
                        parameters = state["turns"][0]["parameters"]
                        self.assertEqual(parameters["project_type_id"],
                                         "code")
                        self.assertNotIn("project_kind", parameters)
                    finally:
                        session.close()

                    session = rag.RAGSession(
                        project.root, trace=TraceConfig(enabled=tracing),
                        conversation_path=state_path,
                    )
                    try:
                        self.assertEqual(session.index_status, "empty")
                        self.assertEqual(len(session.turns), 1)
                        self.assertEqual(list(project.indexes.iterdir()), [])
                        source = root / "notes.md"
                        source.write_text(
                            "# Measurements\nDiffusion describes motion.\n",
                            encoding="utf-8",
                        )
                        imported = add_project_sources(project.root, [source])
                        session.refresh_sources(imported)
                        self.assertEqual(session.index_status, "rebuilt")
                        self.assertTrue(session.chunks)
                        self.assertTrue(embedding.calls)
                        self.assertTrue(
                            (project.indexes / "retrieval.json").is_file()
                        )
                        self.assertIn(Path("notes.md"),
                                      session._source_files())
                        session.ask("Explain diffusion", use_router=False)
                        self.assertTrue(session.turns[-1].evidence)
                        self.assertTrue(embedding.queries)
                        self.assertEqual(len(session.turns), 2)

                        (project.source / "notes.md").unlink()
                        session.refresh_sources(open_project(project.root))
                        self.assertEqual(session.index_status, "empty")
                        self.assertEqual(session.chunks, [])
                        self.assertEqual(session._source_files(), {})
                        self.assertEqual(session._unsearchable_sources([]),
                                         [])
                        count = len(embedding.queries)
                        self.assertEqual(session._retrieve("diffusion"), ())
                        self.assertEqual(len(embedding.queries), count)
                        self.assertEqual(len(session.turns), 2)
                    finally:
                        session.close()
