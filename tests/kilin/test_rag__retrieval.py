"""Test RAG retrieval, source preparation, and index recovery.

Embeddings and indexing effects are deterministic or simulated.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch, sentinel
from kilin import rag
from kilin.workspace import (
    list_source_versions,
    prepare_sources_workspace,
    sync_project_sources,
)
from kilin.retrievals import DocumentSelection
from kilin.retrievals.code_graph import CodeGraph
from kilin.trace_logging import NullTraceLogger, TraceConfig, TraceLogger
from tests.kilin.retrievals._support import (
    FakeEmbeddings,
    chunk,
    graph,
    ranked,
)

from tests.kilin._rag_support import (
    _search_patches,
    _session,
    _turn,
)


class TestRAGRetrieval(unittest.TestCase):
    """Exercise RAG search coordination without loading embedding models."""

    def test_retrieval_uses_formatted_semantic_and_raw_lexical_query(
        self,
    ) -> None:
        """Keep embedding-query formatting separate from lexical search."""
        session = _session()
        session.chunks = [chunk("indexed")]
        session._format_query = Mock(return_value="formatted question")
        with _search_patches() as mocks:
            self.assertEqual(session._retrieve("question", 8), ())
        mocks["semantic_search"].assert_called_once_with(
            sentinel.semantic_index,
            "formatted question",
            session.embedding_session,
            count=8,
            trace_turn_id=None,
        )
        mocks["lexical_search"].assert_called_once_with(
            sentinel.lexical_index, "question", count=8
        )
        mocks["code_search"].assert_not_called()
        mocks["hybrid_search"].assert_called_once_with(
            semantic=[], lexical=[], code=[], count=8
        )
        session._format_query.assert_called_once_with("question")

    def test_retrieval_widening_reuses_one_query_embedding(self) -> None:
        """Request-local cache avoids embedding the same text twice."""

        session = _session()
        session.chunks = [chunk("indexed")]
        session.embedding_session.embed_query.return_value = [1.0, 0.0]
        vectors = {}
        with _search_patches() as mocks:
            session._retrieve("question", 2, _query_vectors=vectors)
            session._retrieve("question", 4, _query_vectors=vectors)
        session.embedding_session.embed_query.assert_called_once_with(
            "question"
        )
        self.assertEqual(mocks["semantic_search"].call_count, 2)
        for call in mocks["semantic_search"].call_args_list:
            self.assertEqual(call.kwargs["query_vector"], [1.0, 0.0])
        self.assertEqual(vectors, {"question": [1.0, 0.0]})

    def test_retrieval_reports_all_search_stages(self) -> None:
        """Report the expected retrieval stages in execution order."""
        session = _session()
        session.chunks = [chunk("indexed")]
        messages = []
        with _search_patches() as mocks:
            session._retrieve("question", status_callback=messages.append)
        self.assertIn("Running lexical search", messages)
        self.assertIn("Combining and ranking search results", messages)
        self.assertIn("Expanding related context", messages)
        self.assertLess(
            messages.index("Running lexical search"),
            messages.index("Combining and ranking search results"),
        )
        self.assertEqual(mocks["semantic_search"].call_args.kwargs["count"],
                         5)
        self.assertEqual(messages[0], "Formatting the search query")

    def test_code_graph_search_is_conditional_on_entities(self) -> None:
        """Use code-graph retrieval only when entities are available."""
        session = _session()
        session.indexes.code_graph = graph("module.func")
        session.chunks = [chunk("a", "def func(): pass")]
        with _search_patches() as mocks:
            mocks["code_search"].return_value = [
                ranked(session.chunks[0])
            ]
            session._retrieve("func", trace_turn_id="turn-1")
        mocks["code_search"].assert_called_once_with(
            session.indexes.code_graph, session.chunks, "func", count=5
        )
        self.assertEqual(len(mocks["hybrid_search"].call_args.kwargs["code"]),
                         1)

    def test_retrieval_renumbers_expanded_chunks_and_keeps_cells(self) -> None:
        """Renumber evidence while retaining notebook cell locations."""
        session = _session()
        session.chunks = [chunk("indexed")]
        first = chunk("a", "first", path="a.py", start=3)
        second = replace(chunk("b", "second", path="b.ipynb"), cell=2)
        with _search_patches() as mocks:
            mocks["expand_retrieval_context"].return_value = [
                first, second
            ]
            evidence = session._retrieve("question")
        self.assertEqual([item.citation for item in evidence], [1, 2])
        self.assertEqual([item.path for item in evidence],
                         [Path("a.py"), Path("b.ipynb")])
        self.assertEqual(evidence[1].cell, 2)
        self.assertEqual(evidence[0].text, "first")
        self.assertEqual(
            mocks["expand_retrieval_context"].call_args.args[1],
            session.chunks,
        )

    def test_source_code_preference_ranks_documentation_later(self) -> None:
        """Documentation follows source code only when the intent asks."""
        session = _session()
        session.chunks = [chunk("indexed")]
        tutorial = ranked(
            chunk("t", path="tutorials/a.ipynb", kind="notebook_markdown")
        )
        guide = ranked(chunk("g", path="docs/a.md", kind="markdown_section"))
        test = ranked(chunk("x", path="tests/test_a.py"))
        source = ranked(chunk("s", path="pkg/a.py"))
        # Preferring source code keeps SOURCE_SEARCH_RESULTS fused results.
        for prefer, expected in (
            (True, [source, tutorial, guide, test]),
            (False, [tutorial, guide]),
        ):
            with self.subTest(prefer=prefer), _search_patches() as mocks:
                mocks["hybrid_search"].return_value = [
                    tutorial, guide, test, source
                ]
                session._retrieve("question", 2, prefer_source_code=prefer)
                self.assertEqual(
                    mocks["expand_retrieval_context"].call_args.args[0],
                    expected,
                )
        with _search_patches() as mocks:
            mocks["hybrid_search"].return_value = [tutorial, test, source]
            session._retrieve("question", 3, prefer_source_code=True)
            self.assertEqual(
                mocks["expand_retrieval_context"].call_args.args[0],
                [source, tutorial, test],
            )

    def test_source_code_preference_searches_deeper(self) -> None:
        """Channels rank deeper and keep their best source code."""
        session = _session()
        session.chunks = [chunk("indexed")]
        guides = [
            ranked(chunk(f"g{i}", path=f"docs/{i}.md",
                         kind="markdown_section"))
            for i in range(3)
        ]
        source = ranked(chunk("s", path="pkg/a.py"))
        with _search_patches() as mocks:
            mocks["lexical_search"].return_value = [*guides, source]
            session._retrieve("question", 1, prefer_source_code=True)
            depth = rag.SOURCE_SEARCH_DEPTH
            self.assertEqual(
                mocks["lexical_search"].call_args.kwargs["count"], depth
            )
            self.assertIn(
                source, mocks["hybrid_search"].call_args.kwargs["lexical"]
            )
        with _search_patches() as mocks:
            mocks["lexical_search"].return_value = [*guides, source]
            session._retrieve("question", 1)
            self.assertEqual(
                mocks["lexical_search"].call_args.kwargs["count"], 1
            )

    def test_single_document_scope_uses_full_ranking(self) -> None:
        """Apply full ranking to evidence scoped to a single document."""
        session = _session()
        session.chunks = [chunk("indexed")]
        first = ranked(chunk("a", path="other.py"))
        focused = ranked(chunk("b", path="target.py"))
        session._rank_everything = Mock(return_value=[first, focused])
        with _search_patches() as mocks:
            mocks["hybrid_search"].return_value = [first]
            mocks["expand_retrieval_context"].return_value = [
                focused.chunk
            ]
            session._retrieve("question", 1, scope=[Path("target.py")])
        session._rank_everything.assert_called_once_with(
            "question", None, semantic_results=[]
        )
        self.assertEqual(
            mocks["expand_retrieval_context"].call_args.args[0],
            [focused],
        )

    def test_empty_scoped_ranking_retains_general_results(self) -> None:
        """Fall back to general ranking when scoped results are empty."""
        session = _session()
        session.chunks = [chunk("indexed")]
        general = ranked(chunk("a", path="other.py"))
        session._rank_everything = Mock(return_value=[general])
        with _search_patches() as mocks:
            mocks["hybrid_search"].return_value = [general]
            session._retrieve("question", scope=[Path("missing.py")])
        self.assertEqual(
            mocks["expand_retrieval_context"].call_args.args[0],
            [general],
        )

    def test_multiple_scoped_paths_do_not_focus_ranking(self) -> None:
        """Avoid special single-document handling for multiple paths."""
        session = _session()
        session._rank_everything = Mock()
        with _search_patches():
            session._retrieve("question", scope=[Path("a"), Path("b")])
        session._rank_everything.assert_not_called()

    def test_retrieval_emits_content_when_tracing_enabled(self) -> None:
        """Attach exact retrieved content when tracing is enabled."""
        session = _session()
        session.chunks = [chunk("indexed")]
        session.trace = MagicMock()
        session.trace.include_content = True
        session.trace.artifact.return_value = "artifact-reference"
        sample = chunk("a", "private source")
        with _search_patches() as mocks:
            mocks["expand_retrieval_context"].return_value = [sample]
            session._retrieve("question", trace_turn_id="t")
        event = [
            item for item in session.trace.emit.call_args_list
            if item.args[0] == "retrieval.context.expanded"
        ][0]
        self.assertEqual(
            event.kwargs["data"]["chunks_artifact"],
            "artifact-reference",
        )
        self.assertNotIn("chunks", event.kwargs["data"])
        session.trace.artifact.assert_called_once()
        formatted = [
            item for item in session.trace.emit.call_args_list
            if item.args[0] == "retrieval.query.formatted"
        ][0]
        self.assertEqual(formatted.kwargs["data"]["question"],
                         "question")

    def test_rank_everything_requests_all_chunks(self) -> None:
        """Pass the complete indexed set to unrestricted ranking."""
        session = _session()
        session.chunks = [chunk("a"), chunk("b")]
        session._format_query = Mock(return_value="formatted")
        with _search_patches() as mocks:
            result = session._rank_everything("question", "turn-2")
        self.assertEqual(result, [])
        self.assertEqual(mocks["semantic_search"].call_args.kwargs,
                         {"count": 2, "trace_turn_id": "turn-2"})
        self.assertEqual(mocks["lexical_search"].call_args.kwargs,
                         {"count": 2})
        self.assertEqual(mocks["hybrid_search"].call_args.kwargs["count"],
                         2)
        session.indexes.code_graph = graph("f")
        with _search_patches() as mocks:
            session._rank_everything("question")
        mocks["code_search"].assert_called_once()

    def test_document_selection_skips_empty_index(self) -> None:
        """Skip document selection when no source chunks are indexed."""
        session = _session()
        session._rank_everything = Mock()
        self.assertIsNone(session._read_document("question", 1000))
        session._rank_everything.assert_not_called()

    def test_document_selection_delegates_with_token_budget(self) -> None:
        """Forward document-selection limits and query metadata."""
        session = _session()
        session.chunks = [chunk("a")]
        session._rank_everything = Mock(return_value=[
            ranked(session.chunks[0])
        ])
        selection = DocumentSelection(
            Path("src/example.py"), (), "search",
            tuple(session.chunks), 1, 20, True,
        )
        with patch.object(
            rag, "read_document", return_value=selection
        ) as read:
            result = session._read_document("question", 1200, 2,
                                            trace_turn_id="turn-3")
        self.assertIs(result, selection)
        read.assert_called_once_with(
            "question", session.chunks, session.indexes.code_graph,
            session._rank_everything.return_value,
            max_tokens=1200, top=2,
        )

    def test_query_format_plain_and_qwen3(self) -> None:
        """Apply the selected embedding query format consistently."""
        session = _session()
        self.assertEqual(session._format_query("  Question  "),
                         "  Question  ")
        session.project.profile.query_format = "qwen3"
        with patch.object(
            rag, "format_qwen3_query", return_value="formatted"
        ) as formatter:
            self.assertEqual(session._format_query("Question"),
                             "formatted")
        formatter.assert_called_once_with(
            "Question", rag.project_type("code").retrieval_instruction
        )
        self.assertEqual(session._system_prompt(),
                         rag.project_type("code").system_prompt)


    def test_empty_retrieval_skips_models_and_all_search_stages(self) -> None:
        """An empty project answers without fabricated retrieval results."""

        session = _session()
        messages = []
        with _search_patches() as mocks:
            self.assertEqual(
                session._retrieve("hello", status_callback=messages.append),
                (),
            )
            self.assertEqual(session._rank_everything("hello"), [])
        for operation in mocks.values():
            operation.assert_not_called()
        session.embedding_session.embed_query.assert_not_called()
        self.assertEqual(
            messages, ["No indexed source passages; skipping retrieval"]
        )

    def test_relationship_progress_is_generic_and_reports_real_work(
        self,
    ) -> None:
        """Report graph operations by available entities, not type IDs."""

        session = _session()
        session.project.profile.type_id = "custom"
        session.chunks = [chunk("indexed")]
        for available in (False, True):
            with self.subTest(available=available):
                session.indexes.code_graph = (
                    graph("pkg.func_name") if available else CodeGraph((), ())
                )
                messages = []
                with _search_patches() as mocks:
                    session._retrieve(
                        "How does func_name work?",
                        status_callback=messages.append,
                    )
                if available:
                    mocks["code_search"].assert_called_once()
                    self.assertIn(
                        "Searching relationships between source entities",
                        messages,
                    )
                    self.assertIn("Relationship search ranked 0 results",
                                  messages)
                else:
                    mocks["code_search"].assert_not_called()
                    self.assertIn(
                        "No source relationships indexed; skipping graph "
                        "search", messages,
                    )
                    self.assertNotIn(
                        "Searching relationships between source entities",
                        messages,
                    )


    def test_prose_question_skips_the_code_graph_channel(self) -> None:
        """Without named code, graph search is not run."""
        session = _session()
        session.chunks = [chunk("indexed")]
        session.indexes.code_graph = graph(
            "pkg.Check.valid", kind="python_method"
        )
        messages = []
        with _search_patches() as mocks:
            session._retrieve(
                "Which files are valid?", status_callback=messages.append
            )
        mocks["code_search"].assert_not_called()
        self.assertIn(
            "No code named in the question; skipping graph search", messages
        )


class TestRAGSourcePreparation(unittest.TestCase):
    """Check extracted-text caching and incremental source chunk reuse."""

    def setUp(self) -> None:
        """Create isolated state and arrange test-specific fixtures."""
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.session = _session(self.root)
        self.session.project.source.mkdir()
        self.session.project.indexes.mkdir()
        self.first = self.session.project.source / "a.py"
        self.second = self.session.project.source / "b.py"
        self.first.write_text("def foo():\n    return 1\n", encoding="utf-8")
        self.second.write_text("def bar():\n    return 2\n", encoding="utf-8")
        self.embedding = SimpleNamespace(
            count_tokens=lambda text: len(text.split())
        )

    def _report(self, names: list[str] | None = None) -> dict:
        """Make a selected-file scan report for the temporary repository."""
        return {
            "repository": str(self.session.project.source),
            "complete": True,
            "selected_files": ["a.py", "b.py"] if names is None else names,
            "entries": [],
        }

    def test_preserved_versions_are_separately_indexed(self) -> None:
        """Retained and current copies yield distinct indexed evidence."""

        original = self.root / "external"
        original.mkdir()
        file = original / "analysis.py"
        file.write_text(
            "def evaluate():\n    return 10\n", encoding="utf-8"
        )
        project = prepare_sources_workspace(
            [original], "code", self.root / "managed"
        )
        file.write_text(
            "def evaluate():\n    return 20\n", encoding="utf-8"
        )
        sync_project_sources(project, decisions={
            "sources/external/analysis.py": "keep_both",
        })
        versions = list_source_versions(project)
        self.assertEqual(len(versions), 1)
        self.session.project = project
        embedding = FakeEmbeddings()
        embedding.model_id = "fake-embedding"
        embedding.model_source = "local-test-model"
        embedding.count_tokens = lambda text: len(text.split())
        with patch.object(
            rag, "embedding_model_identity", return_value="fake-id"
        ):
            chunks, indexes, status, _ = self.session._prepare_retrieval(
                embedding
            )
        self.assertEqual(status, "rebuilt")
        expected = {
            Path("external/analysis.py"),
            Path(versions[0]["version"]).relative_to("sources"),
        }
        self.assertEqual({item.path for item in chunks}, expected)
        self.assertEqual(
            {item.path for item in indexes.semantic.chunks}, expected
        )
        manifest = json.loads(
            (project.indexes / "source_manifest.json").read_text()
        )
        self.assertEqual(set(manifest["files"]), {
            item.as_posix() for item in expected
        })
        self.assertEqual(len(embedding.calls), 1)
        self.assertEqual(len(embedding.calls[0]), 2)
        self.assertEqual({
            item.path.as_posix(): item.text for item in chunks
        }["external/analysis.py"], "def evaluate():\n    return 20\n")
        self.assertIn(
            "return 10", next(
                item.text for item in chunks
                if item.path != Path("external/analysis.py")
            )
        )

        # The retained version is unchanged on the next refresh, so the
        # source identity and its embedding can be reused as-is.
        file.write_text(
            "def evaluate():\n    return 30\n", encoding="utf-8"
        )
        sync_project_sources(project, decisions={
            "sources/external/analysis.py": "replace",
        })
        with patch.object(
            rag, "embedding_model_identity", return_value="fake-id"
        ):
            updated, _, refreshed, delta = self.session._prepare_retrieval(
                embedding
            )
        self.assertEqual(refreshed, "rebuilt")
        self.assertEqual(
            {item.path for item in updated}, expected
        )
        self.assertEqual(delta["modified"], ["external/analysis.py"])
        self.assertEqual(len(embedding.calls[-1]), 1)

    def test_source_preparation_extracts_chunks_and_code_graph(self) -> None:
        """Build source chunks and code graphs from the repository."""
        chunks, code_graph = self.session._prepare_source(
            self.embedding, report=self._report()
        )
        self.assertEqual({value.path for value in chunks},
                         {Path("a.py"), Path("b.py")})
        self.assertTrue(code_graph.entities)
        cached = rag._read_text_cache(self.session.project.indexes)
        self.assertEqual(set(cached), {"a.py", "b.py"})
        self.assertIn("return 1", cached["a.py"]["text"])

    def test_html_text_does_not_create_code_graph_links(self) -> None:
        """Preserve code relations but exclude visible HTML examples."""

        page = self.session.project.source / "page.html"
        page.write_text(
            "<pre>&lt;button onclick=\"ghost()\"&gt;"
            "example&lt;/button&gt;</pre>",
            encoding="utf-8",
        )
        self.session.project.profile.extensions = (".py", ".html")
        with patch.object(rag, "project_type") as lookup:
            lookup.return_value.code_graph = "auto"
            chunks, code_graph = self.session._prepare_source(
                self.embedding, report=self._report(["a.py", "page.html"])
            )
        self.assertEqual({c.path for c in chunks}, {
            Path("a.py"), Path("page.html")
        })
        self.assertIn("a.foo", {e.name for e in code_graph.entities})
        self.assertFalse(any(
            relation.target.endswith("ghost")
            for relation in code_graph.relations
        ))

    def test_html_without_visible_text_has_no_graph(self) -> None:
        """An HTML page of scripts alone must not prevent indexing."""

        page = self.session.project.source / "page.html"
        page.write_text(
            '<html><head><script src="app.js"></script></head></html>',
            encoding="utf-8",
        )
        self.session.project.profile.extensions = (".html",)
        with patch.object(rag, "project_type") as lookup:
            lookup.return_value.code_graph = "auto"
            chunks, code_graph = self.session._prepare_source(
                self.embedding, report=self._report(["page.html"])
            )
        self.assertEqual(chunks, [])
        self.assertEqual(code_graph, CodeGraph((), ()))

    def test_html_cache_representation_does_not_invalidate_python(
        self,
    ) -> None:
        """Switching HTML modes rereads HTML but reuses Python text."""

        page = self.session.project.source / "page.html"
        page.write_text("<h1>Notes</h1>", encoding="utf-8")
        self.session.project.profile.extensions = (".py", ".html")
        report = self._report(["a.py", "page.html"])
        self.session._prepare_source(self.embedding, report=report)
        with patch.object(
            rag, "read_repository_files", wraps=rag.read_repository_files
        ) as reader:
            self.session._prepare_source(self.embedding, report=report)
            reader.assert_not_called()
        with patch.object(rag, "project_type") as lookup:
            lookup.return_value.code_graph = "auto"
            with patch.object(
                rag, "read_repository_files", wraps=rag.read_repository_files
            ) as reader:
                self.session._prepare_source(self.embedding, report=report)
                self.assertEqual(
                    reader.call_args.args[0]["selected_files"],
                    ["page.html"],
                )

    def test_unchanged_text_cache_skips_source_reader(self) -> None:
        """Reuse unchanged text without rereading the original source."""
        self.session._prepare_source(self.embedding, report=self._report())
        with patch.object(
            rag, "read_repository_files",
            side_effect=AssertionError("Reader should not be called"),
        ):
            chunks, _ = self.session._prepare_source(
                self.embedding, report=self._report()
            )
        self.assertTrue(chunks)

    def test_changed_source_reuses_other_file_chunks(self) -> None:
        """Reprocess modified sources while preserving unaffected chunks."""
        chunks, _ = self.session._prepare_source(
            self.embedding, report=self._report()
        )
        saved = {Path("a.py"): [
            value for value in chunks if value.path == Path("a.py")
        ]}
        self.second.write_text(
            "def bar():\n    return 999\n", encoding="utf-8"
        )
        with patch.object(
            rag, "read_repository_files", wraps=rag.read_repository_files
        ) as reader, patch.object(
            rag, "chunk_source_units", wraps=rag.chunk_source_units
        ) as chunker:
            updated, _ = self.session._prepare_source(
                self.embedding, report=self._report(),
                cached_chunks=saved,
            )
        self.assertEqual(
            reader.call_args.args[0]["selected_files"], ["b.py"]
        )
        self.assertTrue(all(
            unit.path == Path("b.py")
            for unit in chunker.call_args.args[0]
        ))
        self.assertEqual(
            [value for value in updated if value.path == Path("a.py")],
            saved[Path("a.py")],
        )
        self.assertTrue(any("999" in value.text for value in updated))

    def test_removed_sources_are_pruned_from_text_cache(self) -> None:
        """Remove cached text for files no longer in the repository."""
        self.session._prepare_source(self.embedding, report=self._report())
        chunks, _ = self.session._prepare_source(
            self.embedding, report=self._report(["b.py"])
        )
        self.assertEqual({value.path for value in chunks}, {Path("b.py")})
        self.assertEqual(set(rag._read_text_cache(
            self.session.project.indexes
        )), {"b.py"})

    def test_empty_source_returns_empty_graph_and_chunks(self) -> None:
        """Handle empty source files without generating false entities."""
        chunks, code_graph = self.session._prepare_source(
            self.embedding, report=self._report([])
        )
        self.assertEqual(chunks, [])
        self.assertEqual(code_graph, CodeGraph((), ()))

    def test_disabled_code_graph_skips_graph_extraction(self) -> None:
        """Skip code-graph work when the profile disables that feature."""
        with patch.object(rag, "project_type") as project:
            project.return_value.code_graph = False
            with patch.object(rag, "build_code_graph") as build:
                chunks, graph_result = self.session._prepare_source(
                    self.embedding, report=self._report()
                )
        self.assertTrue(chunks)
        self.assertEqual(graph_result, CodeGraph((), ()))
        build.assert_not_called()

    def test_missing_source_fails_instead_of_using_stale_text(self) -> None:
        """Fail on missing files instead of relying on cached source text."""
        with self.assertRaises(OSError):
            self.session._prepare_source(
                self.embedding, report=self._report(["missing.py"])
            )

    def test_source_preparation_scans_when_no_report_is_supplied(self) -> None:
        """Scan the repository when no source report was provided."""
        with patch.object(
            rag, "scan_repository", return_value=self._report()
        ) as scan:
            chunks, _ = self.session._prepare_source(self.embedding)
        self.assertTrue(chunks)
        self.assertEqual(scan.call_args.args[0],
                         self.session.project.source)


class TestRAGSourceInventory(unittest.TestCase):
    """Keep tool file inventories aligned with the source manifest."""

    def setUp(self) -> None:
        """Create isolated state and arrange test-specific fixtures."""
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.session = _session(Path(temporary.name))
        self.session.project.indexes.mkdir()

    def test_source_files_reads_manifest_sizes(self) -> None:
        """Read source inventory and file sizes from the saved manifest."""
        manifest = self.session.project.indexes / "source_manifest.json"
        manifest.write_text(json.dumps({
            "files": {
                "a.py": {"size_bytes": 123},
                "b.py": {"size_bytes": 0},
                "c.py": {},
            },
        }), encoding="utf-8")
        self.assertEqual(self.session._source_files(), {
            Path("a.py"): 123, Path("b.py"): 0, Path("c.py"): 0,
        })
        self.assertEqual(
            self.session._unsearchable_sources([Path("a.py")]),
            [Path("b.py"), Path("c.py")],
        )

    def test_invalid_source_manifest_yields_empty_inventory(self) -> None:
        """Return an empty inventory for malformed manifests."""
        manifest = self.session.project.indexes / "source_manifest.json"
        self.assertEqual(self.session._source_files(), {})
        for value in ("bad JSON", "{}", '{"files": []}',
                      '{"files": {"a": {"size_bytes": "oops"}}}'):
            with self.subTest(value=value):
                manifest.write_text(value, encoding="utf-8")
                self.assertEqual(self.session._source_files(), {})


class TestRAGIndexMetadataRecovery(unittest.TestCase):
    """Check recovery from malformed manifest and metadata files."""

    def setUp(self) -> None:
        """Create isolated state and arrange test-specific fixtures."""
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.session = _session(Path(temporary.name))
        self.session.project.indexes.mkdir()
        self.session.project.source.mkdir()
        self.embedding = FakeEmbeddings()
        self.embedding.model_id = "embedding"
        self.embedding.model_source = "resolved-source"
        self.manifest = self.session.project.indexes / "source_manifest.json"
        self.settings = self.session.project.indexes / "settings.json"
        self.report = {
            "repository": str(self.session.project.source),
            "complete": True,
            "selected_files": ["a.py"],
            "entries": [],
        }
        self.changes = {
            "matches": False, "added": ["a.py"], "modified": [],
            "removed": [],
        }

    @contextmanager
    def _index_patches(self):
        """Exercise real index writes with a controlled empty scan."""
        with patch.object(
            rag, "scan_repository", return_value=self.report
        ), patch.object(
            rag, "build_repository_manifest", return_value={
                "selected_files": ["a.py"], "files": {},
            }
        ), patch.object(
            rag, "compare_repository_manifests",
            return_value=dict(self.changes),
        ) as compare, patch.object(
            rag, "embedding_model_identity", return_value="model-hash"
        ), patch.object(
            self.session, "_prepare_source", return_value=(
                [chunk("id", "alpha beta", path="a.py")],
                CodeGraph((), ())
            )
        ) as prepare:
            yield SimpleNamespace(compare=compare, prepare=prepare)

    def test_invalid_manifest_json_triggers_clean_rebuild(self) -> None:
        """Rebuild indexes when the saved manifest is malformed."""
        self.manifest.write_text("invalid JSON", encoding="utf-8")
        with self._index_patches() as mocks:
            _, _, status, _ = self.session._prepare_retrieval(
                self.embedding
            )
        self.assertEqual(status, "rebuilt")
        self.assertIsNone(mocks.compare.call_args.args[0])
        self.assertIsInstance(
            json.loads(self.manifest.read_text(encoding="utf-8")), dict
        )

    def test_nonmapping_manifest_is_treated_as_missing(self) -> None:
        """Treat a non-object manifest as unavailable metadata."""
        self.manifest.write_text("[]", encoding="utf-8")
        with self._index_patches() as mocks:
            self.session._prepare_retrieval(self.embedding)
        self.assertIsNone(mocks.compare.call_args.args[0])

    def test_invalid_settings_json_triggers_clean_rebuild(self) -> None:
        """Rebuild indexes after invalid stored settings JSON."""
        self.settings.write_text("invalid JSON", encoding="utf-8")
        with self._index_patches():
            _, _, status, _ = self.session._prepare_retrieval(
                self.embedding
            )
        self.assertEqual(status, "rebuilt")
        self.assertIn("embedding_model_identity",
                      json.loads(self.settings.read_text(encoding="utf-8")))

    def test_nonmapping_settings_are_replaced(self) -> None:
        """Replace saved settings that are not object mappings."""
        self.settings.write_text("[]", encoding="utf-8")
        with self._index_patches():
            self.session._prepare_retrieval(self.embedding)
        self.assertIsInstance(
            json.loads(self.settings.read_text(encoding="utf-8")), dict
        )

    def test_invalid_saved_manifest_is_compared_against_none(self) -> None:
        """Ignore malformed prior manifests during change comparison."""
        self.manifest.write_text('{"files": {}}', encoding="utf-8")
        with self._index_patches() as mocks:
            mocks.compare.side_effect = [
                ValueError("bad manifest"), dict(self.changes)
            ]
            self.session._prepare_retrieval(self.embedding)
        self.assertEqual(mocks.compare.call_count, 2)
        self.assertIsNone(mocks.compare.call_args.args[0])

    def test_missing_model_source_is_rejected(self) -> None:
        """Reject incomplete model-source descriptions."""
        self.embedding.model_source = None
        with self._index_patches() as mocks:
            with self.assertRaisesRegex(ValueError, "source"):
                self.session._prepare_retrieval(self.embedding)
        mocks.prepare.assert_not_called()


class TestRAGFileOwnership(unittest.TestCase):
    """Protect direct persistence and destination switches without tracing."""

    def setUp(self) -> None:
        """Reserve canonical test paths without loading external models."""

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.session = _session(self.root)
        self.session.conversation_path.parent.mkdir(parents=True)

    def test_direct_save_refuses_a_competing_lease(self) -> None:
        """Do not replace JSON when a separate writer owns the same file."""

        from kilin.conversation_ownership import (
            ConversationFileLock,
            ConversationLockedError,
        )

        path = self.session.conversation_path
        self.session._save_conversation()
        before = path.read_bytes()
        old_checkpoint = self.session._checkpoint_id
        with ConversationFileLock(path):
            with self.assertRaises(ConversationLockedError):
                self.session._save_conversation(pending_turn=_turn())
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(self.session.turns, [])
        self.assertEqual(self.session._checkpoint_id, old_checkpoint)

    def test_switch_lock_conflict_preserves_previous_conversation(
        self,
    ) -> None:
        """Retain the old path and its lock if the new file is occupied."""

        from kilin.conversation_ownership import (
            ConversationFileLock,
            ConversationLockedError,
        )

        previous = self.session.conversation_path
        target = previous.with_name("conversation_next.json")
        owner = ConversationFileLock(previous)
        owner.acquire()
        self.addCleanup(owner.release)
        self.session._conversation_ownership = owner
        self.session.turns = [_turn("original")]
        with ConversationFileLock(target):
            with self.assertRaises(ConversationLockedError):
                self.session.switch_conversation(
                    target, trace=NullTraceLogger()
                )
        self.assertEqual(self.session.conversation_path, previous)
        self.assertEqual(self.session.turns[0].question, "original")
        self.assertTrue(owner.holds(previous))
        self.session.switch_conversation(
            target, trace=NullTraceLogger()
        )
        self.assertFalse(owner.holds(previous))
        self.assertTrue(
            self.session._conversation_ownership.holds(target)
        )
        self.session.close()
        self.assertFalse(
            self.session._conversation_ownership is not None
        )

    def test_closed_borrowed_logger_is_rejected_on_switch(self) -> None:
        """Require an enabled borrowed logger to retain its writer lease."""

        target = self.session.conversation_path.with_name(
            "conversation_target.json"
        )
        logger = TraceLogger(
            TraceConfig(enabled=True, output_path=target),
            default_directory=self.root,
        )
        logger.close()
        with self.assertRaisesRegex(RuntimeError, "no file lock"):
            self.session.switch_conversation(target, trace=logger)


if __name__ == "__main__":
    unittest.main()
