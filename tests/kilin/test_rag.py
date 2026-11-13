"""Exercise RAG records, evidence, and coordinator contracts.

External model calls are isolated by the shared offline fixtures.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import FrozenInstanceError, asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch, sentinel
from kilin import rag
from kilin.project_type import RETRIEVAL_MODE_DOCUMENT
from kilin.retrievals import DocumentSelection
from kilin.retrievals.code_graph import CodeGraph
from kilin.runtime import BACKEND_MLX
from kilin.trace_logging import TraceConfig
from tests.kilin.retrievals._support import FakeEmbeddings, chunk, ranked

from tests.kilin._rag_support import (
    _answer_patches,
    _evidence,
    _intent,
    _session,
    _turn,
)


class TestCitationIdentifierValidation(unittest.TestCase):
    """Check citation syntax against supplied evidence in one fast pass."""

    def test_groups_and_deduplication(self) -> None:
        """Recognize the citation groups already supported by KíLín."""
        self.assertEqual(
            rag._invalid_citations(
                "[2] [1, 3] [3] [0]", (_evidence(1), _evidence(2))
            ),
            (3, 0),
        )

    def test_citations_inside_thinking_are_not_answer_citations(
        self,
    ) -> None:
        """Do not flag intermediate model notes as unsupported sources."""

        self.assertEqual(
            rag._invalid_citations(
                "<think>Inspect [99] first.</think>Answer [1].",
                (_evidence(1),),
            ),
            (),
        )

    def test_no_citations_is_a_noop(self) -> None:
        """Return nothing when no citation group is present."""
        self.assertEqual(rag._invalid_citations("Plain answer", ()), ())
        self.assertEqual(
            rag._invalid_citations("Unrelated [ref] [1-3]", ()), ()
        )

    def test_large_ids_do_not_crash_validation(self) -> None:
        """Avoid integer conversion errors for untrusted long numbers."""
        enormous = "9" * 5000
        result = rag._invalid_citations(f"No source [{enormous}]", ())
        self.assertEqual(result, (enormous,))


class TestRAGRecords(unittest.TestCase):
    """Keep the public phase vocabulary and evidence records stable."""

    def test_answer_phases_are_unique_and_ordered(self) -> None:
        """Keep the public answer-phase identifiers unique and ordered."""
        self.assertEqual(len(rag.ANSWER_PHASE_CHOICES), 8)
        self.assertEqual(len(set(rag.ANSWER_PHASE_CHOICES)), 8)
        self.assertEqual(
            rag.ANSWER_PHASE_CHOICES,
            (
                "routing", "routed", "retrieving", "evidence_ready",
                "generating", "saving", "completed", "processing",
            ),
        )

    def test_records_are_frozen_with_optional_fields(self) -> None:
        """Retain immutable RAG records and their optional metadata."""
        evidence = _evidence()
        self.assertIsNone(evidence.cell)
        with self.assertRaises(FrozenInstanceError):
            evidence.citation = 9
        event = rag.ProgressEvent("scan", 50.0, "Halfway")
        self.assertIsNone(event.remaining)
        with self.assertRaises(FrozenInstanceError):
            event.stage = "other"
        turn = _turn(evidence=(evidence,))
        self.assertEqual(turn.evidence, (evidence,))
        with self.assertRaises(FrozenInstanceError):
            turn.answer = "changed"


class TestRAGEvidenceHelpers(unittest.TestCase):
    """Check prompt construction and evidence provenance preservation."""

    def test_answer_prompt_without_evidence_preserves_question(self) -> None:
        """Preserve the question when no supporting evidence is available."""
        self.assertEqual(rag._answer_prompt("how are you?", ()),
                         "how are you?")
        self.assertEqual(
            rag._answer_prompt("how are you?", (), "  Be brief.  "),
            "how are you?\n\nBe brief.",
        )
        for guidance in (None, "  ", 10):
            self.assertEqual(
                rag._answer_prompt("question", (), guidance), "question"
            )

    def test_answer_prompt_formats_citations_and_guidance(self) -> None:
        """Format cited evidence and optional answering instructions."""
        evidence = (
            _evidence(text="First excerpt"),
            replace(_evidence(2), symbol=None, cell=3),
        )
        prompt = rag._answer_prompt("Explain this", evidence, "  Short. ")
        self.assertIn("[1] src/a.py:1-3 (func)\nFirst excerpt", prompt)
        self.assertIn("[2] src/a.py:1-3\nsource text", prompt)
        self.assertIn("bibliography. Short.", prompt)
        self.assertTrue(prompt.endswith("requests:\nExplain this"))
        self.assertNotIn("(None)", prompt)
        self.assertNotIn("cell=3", prompt)
        self.assertNotIn("Short.", rag._answer_prompt("Q", evidence, 3))

<<<<<<< Updated upstream
=======
    def test_answer_prompt_covers_each_document_asked_about(self) -> None:
        """Ask for each source in turn for questions about several."""

        evidence = (
            replace(_evidence(1), path=Path("a.pdf")),
            replace(_evidence(2), path=Path("b.pdf")),
            replace(_evidence(3), path=Path("a.pdf")),
        )
        for question in ("Review the five articles for relevance.",
                         "Compare the datasets of all the papers."):
            with self.subTest(question=question):
                prompt = rag._answer_prompt(question, evidence)
                self.assertIn(
                    "the evidence comes from 2: a.pdf, b.pdf. Cover each of "
                    "them in turn", prompt,
                )
                self.assertIn("not relevant to the question", prompt)
                self.assertTrue(prompt.endswith(question))
        for question, items in (
            ("How does BRAPH 2 differ from BRAPH?", evidence),
            ("Review the five articles for relevance.", evidence[:1]),
        ):
            with self.subTest(question=question, sources=len(items)):
                self.assertNotIn(
                    "Cover each of them",
                    rag._answer_prompt(question, items),
                )

    def test_answer_prompt_keeps_distinct_duplicate_basenames(self) -> None:
        """Different directories must retain distinct source identities."""

        evidence = (
            replace(_evidence(1), path=Path("group_a/paper.pdf")),
            replace(_evidence(2), path=Path("group_b/paper.pdf")),
        )
        prompt = rag._answer_prompt("Compare two papers.", evidence)
        self.assertIn(
            "the evidence comes from 2: group_a/paper.pdf, "
            "group_b/paper.pdf. Cover each of them", prompt,
        )
        self.assertEqual(
            len({item.path for item in evidence}), 2
        )

>>>>>>> Stashed changes
    def test_answer_prompt_normalizes_foreign_platform_paths(self) -> None:
        """Display evidence paths with forward slashes on every host."""

        cases = (
            ("src/module.py", "src/module.py"),
            (r"src\module.py", "src/module.py"),
            (r"C:\repo\src\module.py", "C:/repo/src/module.py"),
            (r"\\server\share\module.py", "//server/share/module.py"),
        )
        for raw, expected in cases:
            with self.subTest(path=raw):
                evidence = (replace(_evidence(), path=Path(raw)),)
                prompt = rag._answer_prompt("Q", evidence)
                self.assertIn(f"[1] {expected}:1-3 (func)", prompt)
                self.assertNotIn("\\", prompt)

    def test_document_helpers_retain_selected_source_order(self) -> None:
        """Preserve document order and metadata when preparing evidence."""
        chunks = (
            replace(chunk("one", "first"), cell=2),
            chunk("two", "second", start=4),
        )
        selection = DocumentSelection(
            Path("src/a.py"), ("f",), "named_function", chunks,
            3, 24, False,
        )
        evidence = rag._document_evidence(selection)
        self.assertEqual([item.citation for item in evidence], [1, 2])
        self.assertEqual(evidence[0].cell, 2)
        self.assertEqual([item.text for item in evidence],
                         ["first", "second"])
        self.assertEqual(
            rag._document_data(selection),
            {
                "path": "src/a.py", "functions": ["f"],
                "reason": "named_function", "total_chunks": 3,
                "selected_chunks": 2, "estimated_tokens": 24,
                "complete": False,
            },
        )
        self.assertEqual(
            rag._document_message(selection),
            "Reading the 2 most relevant of 3 parts of f in a.py",
        )
        self.assertEqual(
            rag._document_message(replace(selection, complete=True)),
            "Reading all 3 parts of f in a.py",
        )
        self.assertEqual(
            rag._document_message(replace(selection, functions=())),
            "Reading the 2 most relevant of 3 parts of a.py",
        )
        self.assertEqual(rag._document_message(None),
                         "No document found to read")

    def test_merge_evidence_deduplicates_and_keeps_newest_limit(self) -> None:
        """Deduplicate evidence and respect the maximum result count."""
        original = _evidence()
        same = replace(original, citation=8, text="newer text")
        merged = rag._merge_evidence([(original,), (same,)])
        self.assertEqual(merged, (original,))
        many = tuple(
            _evidence(1, start=number * 4)
            for number in range(45)
        )
        merged = rag._merge_evidence([many])
        self.assertEqual(len(merged), rag._MAX_REUSED_EVIDENCE)
        self.assertEqual(merged[0].start_line, 20)
        self.assertEqual(merged[0].citation, 1)
        self.assertEqual(merged[-1].citation, 40)
        self.assertEqual(rag._merge_evidence([]), ())

    def test_related_evidence_uses_citations_newest_first(self) -> None:
        """Follow cited references from newer conversation turns first."""
        fresh = (_evidence(),)
        older = _turn(
            answer="Earlier [1, 2]", evidence=(
                _evidence(1, start=1), _evidence(2, start=10)
            )
        )
        newer = _turn(
            answer="Newer [1]", evidence=(_evidence(1, start=20),)
        )
        combined = rag._add_related_evidence(fresh, [older, newer])
        self.assertEqual([item.start_line for item in combined],
                         [1, 20, 10])
        self.assertEqual([item.citation for item in combined], [1, 2, 3])

    def test_related_evidence_ignores_thinking_only_citations(self) -> None:
        """Do not promote sources mentioned only in model thinking."""

        earlier = _turn(
            answer="<think>Inspect [2]</think>Use [1].",
            evidence=(
                _evidence(1, start=10), _evidence(2, start=20)
            ),
        )
        evidence = rag._add_related_evidence((), [earlier])
        self.assertEqual([item.start_line for item in evidence], [10])

    def test_related_evidence_without_citations_is_limited(self) -> None:
        """Limit related evidence when earlier answers cite no sources."""
        earlier = _turn(
            answer="No source numbers", evidence=tuple(
                _evidence(number + 1, start=number * 4)
                for number in range(8)
            )
        )
        added = rag._add_related_evidence((), [earlier])
        self.assertEqual(len(added), rag._UNCITED_RELATED_EVIDENCE)
        self.assertEqual([item.citation for item in added], [1, 2, 3])
        turns = [
            _turn(answer="[1]", evidence=(_evidence(start=i * 4),))
            for i in range(15)
        ]
        self.assertEqual(
            len(rag._add_related_evidence((), turns)),
            rag._MAX_RELATED_EVIDENCE,
        )

    def test_turn_serialization_roundtrip_with_notebook_cell(self) -> None:
        """Round-trip conversation turns with notebook-cell provenance."""
        evidence = replace(_evidence(), cell=4)
        original = _turn(
            "What?", "Answer [1]", (evidence,), {"count": 3}
        )
        payload = rag._turn_to_data(original)
        self.assertEqual(payload["evidence"][0]["path"], "src/a.py")
        self.assertEqual(payload["evidence"][0]["cell"], 4)
        self.assertEqual(rag._turn_from_data(payload), original)
        legacy = json.loads(json.dumps(payload))
        del legacy["evidence"][0]["cell"]
        self.assertIsNone(rag._turn_from_data(legacy).evidence[0].cell)
        legacy["evidence"][0]["citation"] = "2"
        self.assertEqual(rag._turn_from_data(legacy).evidence[0].citation,
                         2)

    def test_turn_deserialization_rejects_invalid_records(self) -> None:
        """Reject malformed persisted turn and evidence records."""
        payload = rag._turn_to_data(_turn(evidence=(_evidence(),)))
        with self.assertRaisesRegex(ValueError, "evidence must be a list"):
            rag._turn_from_data({**payload, "evidence": "bad"})
        with self.assertRaisesRegex(ValueError, "parameters"):
            rag._turn_from_data({**payload, "parameters": []})
        with self.assertRaises(KeyError):
            rag._turn_from_data({**payload, "question": None,
                                 "timestamp": None, "answer": None,
                                 "evidence": [{}]})
        with self.assertRaises(ValueError):
            rag._turn_from_data({
                **payload, "evidence": [
                    {**payload["evidence"][0], "start_line": "bad"}
                ],
            })

    def test_checkpoint_identifier_is_canonical(self) -> None:
        """Derive the same checkpoint identifier from equivalent snapshots."""
        first = rag._resume_checkpoint_id({"b": 2, "a": "å"})
        second = rag._resume_checkpoint_id({"a": "å", "b": 2})
        self.assertEqual(first, second)
        self.assertTrue(first.startswith("checkpoint_"))
        self.assertEqual(len(first), len("checkpoint_") + 64)
        self.assertNotEqual(first, rag._resume_checkpoint_id({"b": 3}))

    def test_chunk_and_rank_trace_hide_text_by_default(self) -> None:
        """Exclude source text from compact chunk and ranking records."""
        value = chunk("id", "secret text")
        row = rag._chunk_trace_data(value, include_text=False)
        self.assertEqual(row["text_sha256"],
                         sha256(b"secret text").hexdigest())
        self.assertNotIn("text_preview", row)
        preview = rag._chunk_trace_data(value, include_text=True)
        self.assertIn("text_preview", preview)
        self.assertEqual(preview["identifier"], "id")
        result = ranked(value)
        data = rag._ranked_trace_data([result])
        self.assertEqual(data["count"], 1)
        self.assertEqual(data["results"][0]["rank"], 1)
        self.assertEqual(data["results"][0]["channels"], ("semantic",))
        self.assertNotIn("text_preview", data["results"][0]["chunk"])
        self.assertIn(
            "text_preview",
            rag._ranked_trace_data([result], include_text=True)
            ["results"][0]["chunk"],
        )
        self.assertEqual(rag._ranked_trace_data([]),
                         {"count": 0, "results": []})


class TestRetrievalText(unittest.TestCase):
    """Index code passages with their file path and module summary."""

    def test_code_passages_get_path_and_module_summary(self) -> None:
        """Only code passages are prefixed; the header keeps its text."""
        header = chunk(
            "h", '"""Extract natural units from Python files.\n\nMore."""\n'
            "import ast\n", path="pkg/py_ext.py",
            kind="python_module_context",
        )
        function = chunk("f", "def _units():\n    pass\n",
                         path="pkg/py_ext.py", start=5)
        guide = chunk("g", "Read the guide.", path="docs/a.md",
                      kind="markdown_section")
        broken = chunk("b", "def (", path="pkg/bad.py",
                       kind="python_module_context")
        summaries = rag._module_summaries([header, function, guide, broken])
        self.assertEqual(
            summaries,
            {
                Path("pkg/py_ext.py"):
                    "Extract natural units from Python files."
            },
        )
        self.assertEqual(
            rag._retrieval_text(function, summaries),
            "pkg/py_ext.py\nExtract natural units from Python files.\n\n"
            "def _units():\n    pass\n",
        )
        self.assertEqual(
            rag._retrieval_text(header, summaries),
            f"pkg/py_ext.py\n\n{header.text}",
        )
        self.assertEqual(rag._retrieval_text(guide, summaries), guide.text)

    def test_passage_relevance_uses_saved_embeddings(self) -> None:
        """Score passages by cosine similarity to the embedded query."""
        session = _session()
        near, far, missing = chunk("n"), chunk("f"), chunk("m")
        session.indexes = SimpleNamespace(
            semantic=SimpleNamespace(
                chunks=(near, far), embeddings=((1.0, 0.0), (0.0, 1.0))
            )
        )
        session.embedding_session = Mock()
        session.embedding_session.embed_query.return_value = [0.8, 0.6]
        results = rag.semantic_search(
            session.indexes.semantic, "query", session.embedding_session
        )
        score = session._passage_relevance(results)
        self.assertAlmostEqual(score(near), 0.8)
        self.assertAlmostEqual(score(far), 0.6)
        self.assertEqual(score(missing), float("-inf"))
        session.embedding_session.embed_query.assert_called_once_with(
            "query"
        )


class TestRAGProgress(unittest.TestCase):
    """Verify progress counts, callback delivery, and trace throttling."""

    def test_progress_events_include_all_counts(self) -> None:
        """Preserve progress counts and percentages for UI consumers."""
        session = _session()
        session.trace = MagicMock()
        received = []
        session.progress_callback = received.append
        session._emit_progress(
            "scan", 50, "Reading", current=3, total=5,
            remaining=2, elapsed_seconds=6, eta_seconds=4,
        )
        self.assertEqual(len(received), 1)
        event = received[0]
        self.assertEqual(event.percent, 50.0)
        self.assertEqual(event.remaining, 2)
        self.assertEqual(event.eta_seconds, 4)
        self.assertEqual(
            session.trace.emit.call_args.args[0],
            rag.TRACE_EVENT_PROGRESS_UPDATED,
        )
        self.assertEqual(
            session.trace.emit.call_args.kwargs["data"], asdict(event)
        )

    def test_progress_trace_throttling_and_completion(self) -> None:
        """Throttle ordinary progress traces while recording completion."""
        session = _session()
        session.trace = MagicMock()
        session._emit_progress("scan", 0, "Start")
        session._emit_progress("scan", 3, "Small change")
        session._emit_progress("scan", 5, "Checkpoint")
        session._emit_progress("scan", 5, "Complete", current=2, total=2)
        self.assertEqual(session.trace.emit.call_count, 3)
        self.assertEqual(session._progress_last_trace_percent["scan"], 5)
        session._emit_progress("other", 1, "Other stage")
        self.assertEqual(session.trace.emit.call_count, 4)

    def test_count_progress_rejects_invalid_counts(self) -> None:
        """Reject invalid progress totals and processed counts."""
        session = _session()
        for current, total in ((-1, 5), (6, 5), (0, -1)):
            with self.subTest(current=current, total=total):
                with self.assertRaisesRegex(ValueError, "counts"):
                    session._emit_count_progress(
                        "stage", 10, 20, "Stage", current, total, 0
                    )

    def test_count_progress_endpoints_and_eta(self) -> None:
        """Report progress endpoints and estimated remaining time."""
        session = _session()
        session._emit_progress = Mock()
        with patch.object(
            rag, "perf_counter", side_effect=[10.0, 10.2, 10.2]
        ):
            session._emit_count_progress(
                "stage", 10, 20, "Reading", 0, 4, 8.0
            )
            session._emit_count_progress(
                "stage", 10, 20, "Reading", 2, 4, 8.0
            )
            session._emit_count_progress(
                "stage", 10, 20, "Reading", 4, 4, 8.0
            )
        self.assertEqual(session._emit_progress.call_count, 3)
        self.assertEqual(
            [item.args[1] for item in session._emit_progress.call_args_list],
            [10.0, 15.0, 20.0],
        )
        self.assertEqual(
            session._emit_progress.call_args.kwargs["eta_seconds"], 0.0
        )
        session._emit_progress.reset_mock()
        with patch.object(rag, "perf_counter", return_value=10.0):
            session._emit_count_progress(
                "empty", 10, 20, "Empty", 0, 0, 8.0
            )
        self.assertEqual(session._emit_progress.call_args.args[1], 20.0)

    def test_count_progress_throttles_intermediate_updates(self) -> None:
        """Avoid redundant intermediate progress notifications."""
        session = _session()
        session._emit_progress = Mock()
        with patch.object(rag, "perf_counter", return_value=10.0):
            session._emit_count_progress(
                "stage", 0, 100, "Stage", 1, 10, 9.0
            )
            session._emit_count_progress(
                "stage", 0, 100, "Stage", 2, 10, 9.0
            )
        session._emit_progress.assert_called_once()
        with patch.object(rag, "perf_counter", return_value=11.0):
            session._emit_count_progress(
                "stage", 0, 100, "Stage", 2, 10, 11.0
            )
        self.assertEqual(session._emit_progress.call_count, 2)
        self.assertIsNone(
            session._emit_progress.call_args.kwargs["eta_seconds"]
        )


class TestRAGAnswerValidation(unittest.TestCase):
    """Reject invalid answer requests before retrieval or generation."""

    def test_closed_session_rejects_questions(self) -> None:
        """Reject new questions after the RAG session is closed."""
        session = _session()
        session._closed = True
        with self.assertRaisesRegex(RuntimeError, "closed"):
            session.ask("Question")

    def test_empty_and_nonstring_questions_are_rejected(self) -> None:
        """Validate question types and nonblank content before work."""
        session = _session()
        for value in ("", " \t ", None, 123, []):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "non-empty"):
                    session.ask(value)

    def test_invalid_optional_arguments_are_rejected(self) -> None:
        """Validate optional question arguments before running models."""
        session = _session()
        invalid = (
            {"answer_guidance": 42},
            {"retrieval_count": 0},
            {"retrieval_count": -1},
            {"retrieval_count": True},
            {"retrieval_count": 1.5},
            {"confirm_callback": "yes"},
            {"phase_callback": "yes"},
            {"activity_callback": "yes"},
            {"retrieval_query": " "},
            {"retrieval_query": 10},
            {"reuse_evidence": 1},
            {"reuse_evidence": "true"},
            {"reuse_evidence": True},
        )
        for options in invalid:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    session.ask("Question", **options)
        session.chat_session.reply.assert_not_called()


class TestRAGSessionConstruction(unittest.TestCase):
    """Verify startup, restored history, and partial-initialization cleanup."""

    def setUp(self) -> None:
        """Create isolated state and arrange test-specific fixtures."""
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "logs").mkdir()
        from kilin.project import QAProject, default_project_profile

        self.project = QAProject(
            self.root,
            self.root / "sources",
            self.root / "indexes",
            self.root / "logs",
            default_project_profile("code"),
        )

    @contextmanager
    def _construction_patches(
        self,
        *,
        restored: tuple | None = None,
    ):
        """Provide real project settings and isolated model constructors."""
        trace = MagicMock()
        trace.enabled = False
        trace.include_content = False
        state = restored or ([], [], None)
        with patch.object(
            rag, "open_project", return_value=self.project
        ) as opened, patch.object(
            rag, "choose_backend", return_value=BACKEND_MLX
        ) as backend, patch.object(
            rag, "_conversation_trace", return_value=(trace, True)
        ) as traced, patch.object(
            rag, "EmbeddingSession"
        ) as embeddings, patch.object(
            rag, "ChatSession"
        ) as chat, patch.object(
            rag.RAGSession, "_prepare_retrieval", return_value=(
                [chunk("a")], sentinel.indexes, "rebuilt", {"added": []}
            )
        ) as prepared, patch.object(
            rag.RAGSession, "_load_conversation", return_value=state
        ) as loaded:
            yield SimpleNamespace(
                trace=trace, opened=opened, backend=backend,
                traced=traced, embeddings=embeddings, chat=chat,
                prepared=prepared, loaded=loaded,
            )

    def test_constructor_rejects_noncallable_progress_callback(self) -> None:
        """Reject progress callbacks that cannot be invoked."""
        with patch.object(rag, "open_project") as opened:
            with self.assertRaisesRegex(ValueError, "progress_callback"):
                rag.RAGSession(self.root, progress_callback=3)
        opened.assert_not_called()

    def test_constructor_loads_models_and_reports_ready(self) -> None:
        """Initialize the session and report successful preparation."""
        progress = []
        with self._construction_patches() as mocks:
            session = rag.RAGSession(
                self.root, progress_callback=progress.append,
                router_model="custom-router",
            )
        self.addCleanup(session.close)
        self.assertEqual(session.index_status, "rebuilt")
        self.assertEqual(len(session.chunks), 1)
        self.assertIs(session.indexes, sentinel.indexes)
        self.assertEqual(session.router_model, "custom-router")
        self.assertIs(session.embedding_session,
                      mocks.embeddings.return_value)
        self.assertIs(session.chat_session, mocks.chat.return_value)
        self.assertEqual(progress[0].stage, "opening_project")
        self.assertEqual(progress[-1].stage, "ready")
        self.assertEqual(progress[-1].percent, 100)
        self.assertEqual(mocks.embeddings.call_args.kwargs["backend"],
                         BACKEND_MLX)
        self.assertEqual(mocks.chat.call_args.kwargs["trace"], mocks.trace)
        self.assertEqual(mocks.prepared.call_count, 1)
        self.assertTrue(session._owns_trace)

    def test_constructor_uses_saved_router_without_gui(self) -> None:
        """Choose the saved project router without an explicit model ID."""

        from kilin.model_settings import write_router_model

        write_router_model(self.root, BACKEND_MLX, "example/saved-router")
        with self._construction_patches():
            session = rag.RAGSession(self.root)
        self.addCleanup(session.close)
        self.assertEqual(session.router_model, "example/saved-router")
        self.assertNotEqual(
            session.router_model,
            self.project.profile.chat_model(BACKEND_MLX),
        )

    def test_explicit_router_overrides_saved_project_model(self) -> None:
        """Keep an explicit session choice without editing the profile."""

        from kilin.model_settings import (
            read_router_model, write_router_model,
        )

        write_router_model(self.root, BACKEND_MLX, "example/saved-router")
        with self._construction_patches():
            session = rag.RAGSession(
                self.root, router_model="example/override-router"
            )
        self.addCleanup(session.close)
        self.assertEqual(session.router_model, "example/override-router")
        self.assertEqual(
            read_router_model(self.root, BACKEND_MLX),
            "example/saved-router",
        )

    def test_explicit_empty_router_is_rejected(self) -> None:
        """Do not silently substitute defaults for an invalid override."""

        with self._construction_patches():
            with self.assertRaises(ValueError):
                rag.RAGSession(self.root, router_model="")

    def test_constructor_without_router_setting_uses_core_default(
        self,
    ) -> None:
        """Core-only projects share the published router default."""

        from kilin.model_settings import default_router_model

        with self._construction_patches():
            session = rag.RAGSession(self.root)
        self.addCleanup(session.close)
        self.assertEqual(
            session.router_model, default_router_model(BACKEND_MLX)
        )

    def test_constructor_restores_history_and_generation_settings(
        self,
    ) -> None:
        """Restore saved conversation and effective model settings."""
        previous = _turn(parameters={
            "thinking": True, "temperature": 0.3,
            "max_new_tokens": 64,
        })
        history = [{"role": "user", "content": "prior"}]
        with self._construction_patches(
            restored=(history, [previous], "checkpoint")
        ) as mocks:
            session = rag.RAGSession(self.root)
        self.addCleanup(session.close)
        self.assertEqual(session.turns, [previous])
        self.assertEqual(session._checkpoint_id, "checkpoint")
        mocks.chat.return_value.restore_history.assert_called_once_with(
            history
        )
        mocks.chat.return_value.configure_generation.assert_called_once_with(
            {"thinking": True, "temperature": 0.3,
             "max_new_tokens": 64}
        )

    def test_constructor_ignores_saved_effective_thinking_for_restore(
        self,
    ) -> None:
        """Newly loaded sessions restore configuration, not intent policy."""

        previous = _turn(parameters={
            "thinking": True, "thinking_effective": False,
            "temperature": 0.3, "max_new_tokens": 64,
        })
        with self._construction_patches(
            restored=([], [previous], "checkpoint")
        ) as mocks:
            session = rag.RAGSession(self.root)
        self.addCleanup(session.close)
        mocks.chat.return_value.configure_generation.assert_called_once_with(
            {"thinking": True, "temperature": 0.3, "max_new_tokens": 64}
        )

    def test_untraced_constructor_holds_lease_until_close(self) -> None:
        """Protect conversation history even with tracing disabled."""

        from kilin.conversation_ownership import (
            ConversationFileLock,
            ConversationLockedError,
        )

        with self._construction_patches():
            session = rag.RAGSession(self.root)
        path = session.conversation_path
        self.assertTrue(session._conversation_ownership.holds(path))
        with self.assertRaises(ConversationLockedError):
            ConversationFileLock(path).acquire()
        session.close()
        self.assertIsNone(session._conversation_ownership)
        with ConversationFileLock(path):
            pass

    def test_early_callback_failure_releases_untraced_lease(
        self,
    ) -> None:
        """Release an acquired lock if preparation fails before model load."""

        from kilin.conversation_ownership import ConversationFileLock

        path = self.root / "logs" / "traces" / "conversation_fixed.json"

        def fail_on_progress(event: rag.ProgressEvent) -> None:
            """Stop during the initial preparation notification."""

            raise RuntimeError("progress failed")

        with self._construction_patches():
            with self.assertRaisesRegex(RuntimeError, "progress failed"):
                rag.RAGSession(
                    self.root,
                    conversation_path=path,
                    progress_callback=fail_on_progress,
                )
        with ConversationFileLock(path):
            pass

    def test_constructor_closes_models_and_trace_on_index_failure(
        self,
    ) -> None:
        """Release owned resources after index initialization fails."""
        with self._construction_patches() as mocks:
            mocks.prepared.side_effect = OSError("index failed")
            with self.assertRaisesRegex(OSError, "index failed"):
                rag.RAGSession(self.root)
        mocks.embeddings.return_value.close.assert_called_once()
        mocks.chat.return_value.close.assert_not_called()
        mocks.trace.close.assert_called_once()
        mocks.trace.exception.assert_called_once()

    def test_constructor_closes_both_models_on_restore_failure(self) -> None:
        """Close loaded models if conversation restoration fails."""
        with self._construction_patches() as mocks:
            mocks.loaded.side_effect = ValueError("bad conversation")
            with self.assertRaisesRegex(ValueError, "bad conversation"):
                rag.RAGSession(self.root)
        mocks.embeddings.return_value.close.assert_called_once()
        mocks.chat.return_value.close.assert_called_once()
        mocks.trace.close.assert_called_once()

    def test_constructor_keeps_primary_error_when_cleanup_fails(
        self,
    ) -> None:
        """Attempt each cleanup step without masking restoration errors."""

        with self._construction_patches() as mocks:
            mocks.loaded.side_effect = ValueError("invalid history")
            mocks.chat.return_value.close.side_effect = OSError(
                "chat cleanup failed"
            )
            mocks.embeddings.return_value.close.side_effect = OSError(
                "embedding cleanup failed"
            )
            mocks.trace.exception.side_effect = RuntimeError(
                "cannot trace failure"
            )
            mocks.trace.close.side_effect = OSError("trace close failed")
            with self.assertRaisesRegex(
                ValueError, "invalid history"
            ) as caught:
                rag.RAGSession(self.root)
        mocks.chat.return_value.close.assert_called_once()
        mocks.embeddings.return_value.close.assert_called_once()
        mocks.trace.close.assert_called_once()
        notes = " ".join(caught.exception.__notes__)
        self.assertIn("chat cleanup failed", notes)
        self.assertIn("embedding cleanup failed", notes)
        self.assertIn("trace close failed", notes)
        self.assertIn("cannot trace failure", notes)

    def test_constructor_closes_trace_on_embedding_load_failure(
        self,
    ) -> None:
        """Close owned tracing when embedding initialization fails."""
        with self._construction_patches() as mocks:
            mocks.embeddings.side_effect = OSError("missing model")
            with self.assertRaisesRegex(OSError, "missing model"):
                rag.RAGSession(self.root)
        mocks.trace.close.assert_called_once()
        mocks.chat.assert_not_called()


class TestRAGAdditionalBranches(unittest.TestCase):
    """Cover optional trace content and metadata-preserving fallbacks."""

    def test_trace_includes_question_in_turn_span(self) -> None:
        """Include exact question text in the detailed turn metadata."""
        session = _session()
        session.trace = MagicMock()
        session.trace.include_content = True
        with _answer_patches(session, _intent()):
            session.ask("Private question", use_router=False)
        call_span = [
            item for item in session.trace.span.call_args_list
            if item.args[0] == "turn"
        ][0]
        self.assertEqual(call_span.kwargs["data"]["question"],
                         "Private question")

    def test_document_override_records_effective_query(self) -> None:
        """Record the effective query used for document retrieval."""
        session = _session()
        with _answer_patches(
            session, _intent(RETRIEVAL_MODE_DOCUMENT)
        ) as mocks:
            session.ask("Read file", retrieval_query="  a.py  ",
                        use_router=False)
        self.assertEqual(session.turns[-1].parameters["retrieval_query"],
                         "a.py")
        self.assertEqual(mocks.read.call_args.args[0], "a.py")

    def test_router_change_without_separate_session(self) -> None:
        """Update router preference without replacing a shared session."""
        session = _session()
        session.use_router_model("another-router")
        self.assertEqual(session.router_model, "another-router")
        session.chat_session.close.assert_not_called()

    def test_save_conversation_rejects_nonmapping_json(self) -> None:
        """Reject a non-object conversation file without overwriting it."""
        with TemporaryDirectory() as directory:
            session = _session(Path(directory))
            session.conversation_path.parent.mkdir(parents=True)
            session.conversation_path.write_text("[]", encoding="utf-8")
            before = session.conversation_path.read_bytes()
            with self.assertRaisesRegex(ValueError, "invalid"):
                session._save_conversation()
            self.assertEqual(session.conversation_path.read_bytes(), before)

    def test_explicit_enabled_trace_config_is_bound(self) -> None:
        """Bind explicit enabled trace configurations to the target path."""
        with TemporaryDirectory() as directory:
            path = (Path(directory) / "conversation_1.json").resolve()
            for config in (
                TraceConfig(enabled=True),
                TraceConfig(enabled=True, output_path=path),
            ):
                sink, owned = rag._conversation_trace(config, path)
                self.assertTrue(sink.enabled)
                self.assertTrue(owned)
                self.assertEqual(sink.output_path.resolve(), path)
                sink.close()

    def test_constructor_does_not_close_caller_owned_trace_on_error(
        self,
    ) -> None:
        """Preserve a caller-owned trace after constructor failure."""
        with TemporaryDirectory() as directory:
            from kilin.project import QAProject, default_project_profile
            root = Path(directory)
            (root / "logs").mkdir()
            project = QAProject(
                root, root / "sources", root / "indexes", root / "logs",
                default_project_profile("code"),
            )
            trace = MagicMock()
            with patch.object(
                rag, "open_project", return_value=project
            ), patch.object(
                rag, "_conversation_trace", return_value=(trace, False)
            ), patch.object(
                rag, "choose_backend", return_value=BACKEND_MLX
            ), patch.object(
                rag, "EmbeddingSession", side_effect=OSError("missing")
            ):
                with self.assertRaises(OSError):
                    rag.RAGSession(root)
            trace.close.assert_not_called()


class TestRAGDefensiveFallbacks(unittest.TestCase):
    """Exercise guards for invalid cached vectors and changed source sets."""

    def setUp(self) -> None:
        """Create isolated state and arrange test-specific fixtures."""
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.session = _session(self.root)
        self.session.project.indexes.mkdir()
        self.session.project.source.mkdir()
        self.embedding = FakeEmbeddings()
        self.embedding.model_id = "embedding"
        self.embedding.model_source = "resolved-source"
        self.current = chunk("id", "alpha beta", path="a.py")
        metadata = {
            **self.session._index_metadata(),
            "embedding_model_source": "resolved-source",
            "embedding_model_identity": "model-hash",
        }
        (self.session.project.indexes / "settings.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        (self.session.project.indexes / "source_manifest.json").write_text(
            json.dumps({"selected_files": ["a.py"], "files": {}}),
            encoding="utf-8",
        )

    @contextmanager
    def _prepared_index(self, stored: object):
        """Provide an unchanged configuration with modified source files."""
        report = {
            "repository": str(self.session.project.source),
            "complete": True,
            "selected_files": ["a.py"],
            "entries": [],
        }
        changes = {
            "matches": False, "added": [], "modified": [],
            "removed": ["obsolete.py"],
        }
        with patch.object(
            rag, "scan_repository", return_value=report
        ), patch.object(
            rag, "build_repository_manifest", return_value={
                "selected_files": ["a.py"], "files": {},
            }
        ), patch.object(
            rag, "compare_repository_manifests", return_value=changes
        ), patch.object(
            rag, "embedding_model_identity", return_value="model-hash"
        ), patch.object(
            rag, "load_indexes", return_value=stored
        ), patch.object(
            self.session, "_prepare_source", return_value=(
                [self.current], CodeGraph((), ())
            )
        ) as source:
            yield source

    def test_invalid_cached_vectors_force_reembedding(self) -> None:
        """Recompute embeddings when cached vectors are invalid."""
        bad = SimpleNamespace(
            embedding_model_identity="model-hash",
            semantic=SimpleNamespace(
                chunks=(self.current,),
                embeddings=((float("nan"), 1.0),),
            ),
        )
        with self._prepared_index(bad):
            chunks, _, status, _ = self.session._prepare_retrieval(
                self.embedding
            )
        self.assertEqual(status, "rebuilt")
        self.assertEqual(chunks, [self.current])
        self.assertEqual(self.embedding.calls, [["a.py\n\nalpha beta"]])

    def test_removed_file_vectors_are_not_reused(self) -> None:
        """Exclude vectors belonging to source files that were removed."""
        obsolete = chunk("old", "obsolete", path="obsolete.py")
        saved = SimpleNamespace(
            embedding_model_identity="model-hash",
            semantic=SimpleNamespace(
                chunks=(self.current, obsolete),
                embeddings=((1.0, 2.0), (3.0, 4.0)),
            ),
        )
        with self._prepared_index(saved) as source:
            chunks, indexes, status, _ = self.session._prepare_retrieval(
                self.embedding
            )
        self.assertEqual(status, "rebuilt")
        self.assertEqual(chunks, [self.current])
        self.assertEqual(self.embedding.calls, [])
        self.assertEqual(
            source.call_args.kwargs["cached_chunks"],
            {Path("a.py"): [self.current]},
        )
        self.assertEqual(indexes.semantic.embeddings, ((1.0, 2.0),))

    def test_conversation_path_rechecks_symlink_before_creation(self) -> None:
        """Revalidate resolved paths before creating conversation files."""
        with TemporaryDirectory() as directory:
            session = _session(Path(directory))
            session.project.logs.mkdir()
            with patch.object(Path, "is_symlink", return_value=True):
                with self.assertRaisesRegex(ValueError, "folder"):
                    session._resolve_conversation_path(
                        "logs/conversation_test.json"
                    )


if __name__ == "__main__":
    unittest.main()
