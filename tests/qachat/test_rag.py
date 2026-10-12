"""Exercise RAG orchestration, conversation state, and retrieval adapters.

These tests isolate model loading and external tools with controlled fakes.
Index reuse and pretrained-model execution have separate integration tests.

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
from unittest.mock import DEFAULT, MagicMock, Mock, patch, sentinel

from qachat import rag
from qachat.conversation_format import (
    CONVERSATION_SESSION_KEY,
    new_conversation_session,
)
from qachat.answering import RouteDecision
from qachat.project_type import (
    HISTORY_POLICY_RECENT,
    Intent,
    RETRIEVAL_MODE_DOCUMENT,
    RETRIEVAL_MODE_NONE,
    RETRIEVAL_MODE_REUSE,
    RETRIEVAL_MODE_SEARCH,
)
from qachat.retrievals import DocumentSelection, RetrievalPlan
from qachat.retrievals.code_graph import CodeGraph
from qachat.runtime import BACKEND_MLX
from qachat.tools import ToolResult, ToolUse
from qachat.trace_logging import NullTraceLogger, TraceConfig
from tests.qachat.retrievals._support import (
    FakeEmbeddings,
    chunk,
    graph,
    ranked,
)


def _profile() -> SimpleNamespace:
    """Return a profile exposing the fields used by the RAG coordinator."""
    return SimpleNamespace(
        kind="code",
        backend="mlx",
        allow_download=False,
        query_format="plain",
        retrieval_count=5,
        thinking=False,
        temperature=0.2,
        max_new_tokens=128,
        extensions=(".py",),
        max_file_bytes=10000,
        target_tokens=100,
        max_tokens=200,
        overlap_tokens=20,
        embedding_model=lambda backend: "embedding-model",
        chat_model=lambda backend: "chat-model",
    )


def _session(root: Path | None = None) -> rag.RAGSession:
    """Create a coordinator without initializing or downloading models."""
    directory = root or Path("/unused-qachat-test-project")
    session = rag.RAGSession.__new__(rag.RAGSession)
    session.project = SimpleNamespace(
        root=directory,
        source=directory / "sources",
        indexes=directory / "indexes",
        logs=directory / "logs",
        profile=_profile(),
    )
    session.backend = BACKEND_MLX
    session.trace = NullTraceLogger()
    session._owns_trace = False
    session._progress_last_emit = {}
    session._progress_last_trace_percent = {}
    session.progress_callback = None
    session.conversation_path = (
        directory / "logs" / "traces" / "conversation_test.json"
    )
    session.embedding_session = MagicMock()
    session.embedding_session.model_id = "embedding-model"
    session.embedding_session.model_source = "embedding-source"
    session.chat_session = MagicMock()
    session.chat_session.model_id = "chat-model"
    session.chat_session.model_source = "chat-source"
    session.chat_session.messages = [
        {"role": "system", "content": "system"}
    ]
    session.chat_session.reply.return_value = "model answer"
    session.chat_session.temperature = 0.2
    session.chat_session.thinking = False
    session.chat_session.max_new_tokens = 128
    session.chat_session.input_token_total = 10
    session.chat_session.output_token_total = 5
    session.chat_session.last_finish_reason = None
    session.router_model = "chat-model"
    session._router_session = None
    session.chunks = []
    session.indexes = SimpleNamespace(
        semantic=sentinel.semantic_index,
        lexical=sentinel.lexical_index,
        code_graph=CodeGraph((), ()),
        fingerprint="index-fingerprint",
    )
    session.index_status = "rebuilt"
    session.source_changes = {}
    session.turns = []
    session._checkpoint_id = None
    session._closed = False
    return session


def _evidence(
    citation: int = 1,
    *,
    path: str = "src/a.py",
    start: int = 1,
    text: str = "source text",
) -> rag.Evidence:
    """Create a source reference with stable line and path provenance."""
    return rag.Evidence(
        citation, Path(path), start, start + 2, "func", text
    )


def _turn(
    question: str = "earlier question",
    answer: str = "earlier answer [1]",
    evidence: tuple[rag.Evidence, ...] = (),
    parameters: dict[str, object] | None = None,
) -> rag.ConversationTurn:
    """Create a complete earlier turn for history and persistence tests."""
    return rag.ConversationTurn(
        "2026-01-01T00:00:00+00:00",
        question,
        answer,
        evidence,
        {} if parameters is None else parameters,
    )


def _intent(
    mode: str = RETRIEVAL_MODE_SEARCH,
    *,
    tool: object = None,
    related: bool = False,
    guidance: str | None = None,
    policy: str = HISTORY_POLICY_RECENT,
) -> Intent:
    """Construct an intent independently of packaged project-type data."""
    return Intent(
        intent_id="test_intent",
        description="Test intent",
        system_prompt="Intent instruction",
        answer_guidance=guidance,
        history_policy=policy,
        retrieval_mode=mode,
        retrieval_strategy="hybrid",
        include_related_evidence=related,
        max_tokens=1200,
        tool=tool,
    )


def _kind(intent: Intent) -> SimpleNamespace:
    """Provide the minimum project-type contract for an answer turn."""
    return SimpleNamespace(
        kind="code",
        default_intent=intent.intent_id,
        intents={intent.intent_id: intent},
        recent_turns=2,
        system_prompt="Project instruction",
        code_graph=True,
    )


class TestRAGRecords(unittest.TestCase):
    """Keep the public phase vocabulary and evidence records stable."""

    def test_answer_phases_are_unique_and_ordered(self) -> None:
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

    def test_document_helpers_retain_selected_source_order(self) -> None:
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

    def test_related_evidence_without_citations_is_limited(self) -> None:
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
        first = rag._resume_checkpoint_id({"b": 2, "a": "å"})
        second = rag._resume_checkpoint_id({"a": "å", "b": 2})
        self.assertEqual(first, second)
        self.assertTrue(first.startswith("checkpoint_"))
        self.assertEqual(len(first), len("checkpoint_") + 64)
        self.assertNotEqual(first, rag._resume_checkpoint_id({"b": 3}))

    def test_chunk_and_rank_trace_hide_text_by_default(self) -> None:
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



@contextmanager
def _search_patches():
    """Replace search channels while preserving the real RAG coordinator."""
    with patch.multiple(
        rag,
        semantic_search=DEFAULT,
        lexical_search=DEFAULT,
        code_search=DEFAULT,
        hybrid_search=DEFAULT,
        plan_retrieval=DEFAULT,
        expand_retrieval_context=DEFAULT,
    ) as mocks:
        for name in (
            "semantic_search", "lexical_search", "code_search",
            "hybrid_search", "expand_retrieval_context",
        ):
            mocks[name].return_value = []
        mocks["plan_retrieval"].return_value = RetrievalPlan(
            (), (), False, (), ()
        )
        yield mocks


class TestRAGRetrieval(unittest.TestCase):
    """Exercise RAG search coordination without loading embedding models."""

    def test_retrieval_uses_formatted_semantic_and_raw_lexical_query(
        self,
    ) -> None:
        session = _session()
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

    def test_retrieval_reports_all_search_stages(self) -> None:
        session = _session()
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
        session = _session()
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

    def test_single_document_scope_uses_full_ranking(self) -> None:
        session = _session()
        first = ranked(chunk("a", path="other.py"))
        focused = ranked(chunk("b", path="target.py"))
        session._rank_everything = Mock(return_value=[first, focused])
        with _search_patches() as mocks:
            mocks["hybrid_search"].return_value = [first]
            mocks["expand_retrieval_context"].return_value = [
                focused.chunk
            ]
            session._retrieve("question", 1, scope=[Path("target.py")])
        session._rank_everything.assert_called_once_with("question", None)
        self.assertEqual(
            mocks["expand_retrieval_context"].call_args.args[0],
            [focused],
        )

    def test_empty_scoped_ranking_retains_general_results(self) -> None:
        session = _session()
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
        session = _session()
        session._rank_everything = Mock()
        with _search_patches():
            session._retrieve("question", scope=[Path("a"), Path("b")])
        session._rank_everything.assert_not_called()

    def test_retrieval_emits_content_only_in_forensic_mode(self) -> None:
        session = _session()
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
        session = _session()
        session._rank_everything = Mock()
        self.assertIsNone(session._read_document("question", 1000))
        session._rank_everything.assert_not_called()

    def test_document_selection_delegates_with_token_budget(self) -> None:
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


class TestRAGProgress(unittest.TestCase):
    """Verify progress counts, callback delivery, and trace throttling."""

    def test_progress_events_include_all_counts(self) -> None:
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
        session = _session()
        for current, total in ((-1, 5), (6, 5), (0, -1)):
            with self.subTest(current=current, total=total):
                with self.assertRaisesRegex(ValueError, "counts"):
                    session._emit_count_progress(
                        "stage", 10, 20, "Stage", current, total, 0
                    )

    def test_count_progress_endpoints_and_eta(self) -> None:
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


@contextmanager
def _answer_patches(
    session: rag.RAGSession,
    intent: Intent,
    decision: RouteDecision | None = None,
):
    """Isolate the real answer state machine from models and disk writes."""
    decision = decision or RouteDecision(intent.intent_id, True)
    with patch.object(rag, "project_type", return_value=_kind(intent)), \
        patch.object(rag, "document_scope", return_value=((), "none")), \
        patch.object(session, "_route", return_value=decision) as route, \
        patch.object(
            session, "_retrieve", return_value=(_evidence(),)
        ) as retrieve, \
        patch.object(session, "_read_document") as read, \
        patch.object(session, "_run_tool") as run, \
        patch.object(session, "_save_conversation") as save:
        read.return_value = None
        run.return_value = ToolResult(
            "tool answer", (asdict(_evidence()),), {"name": "tool"}
        )
        yield SimpleNamespace(
            route=route, retrieve=retrieve, read=read,
            run=run, save=save,
        )


class TestRAGAnswerValidation(unittest.TestCase):
    """Reject invalid answer requests before retrieval or generation."""

    def test_closed_session_rejects_questions(self) -> None:
        session = _session()
        session._closed = True
        with self.assertRaisesRegex(RuntimeError, "closed"):
            session.ask("Question")

    def test_empty_and_nonstring_questions_are_rejected(self) -> None:
        session = _session()
        for value in ("", " \t ", None, 123, []):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "non-empty"):
                    session.ask(value)

    def test_invalid_optional_arguments_are_rejected(self) -> None:
        session = _session()
        invalid = (
            {"answer_guidance": 42},
            {"retrieval_count": 0},
            {"retrieval_count": -1},
            {"retrieval_count": True},
            {"retrieval_count": 1.5},
            {"confirm_callback": "yes"},
            {"phase_callback": "yes"},
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


class TestRAGAnswerOrchestration(unittest.TestCase):
    """Exercise the configured answer modes and their saved turn records."""

    def test_search_answer_tracks_phases_and_persists_turn(self) -> None:
        session = _session()
        session.chunks = [chunk("a", "text")]
        phases = []
        tokens = Mock()
        with _answer_patches(session, _intent()) as mocks:
            answer = session.ask(
                "  Explain the function  ",
                retrieval_count=2,
                retrieval_query="  focused search  ",
                answer_guidance="Be brief.",
                token_callback=tokens,
                phase_callback=lambda phase, details: phases.append(
                    (phase, details)
                ),
                use_router=False,
            )
        self.assertEqual(answer, "model answer")
        self.assertEqual(len(session.turns), 1)
        turn = session.turns[0]
        self.assertEqual(turn.question, "Explain the function")
        self.assertEqual(turn.evidence, (_evidence(),))
        self.assertEqual(turn.parameters["retrieval_count"], 2)
        self.assertEqual(turn.parameters["retrieval_query"],
                         "focused search")
        self.assertEqual(turn.parameters["answer_guidance"], "Be brief.")
        self.assertEqual(turn.parameters["retrieval_mode"], "search")
        self.assertEqual(turn.parameters["representation"]["turn_id"], 1)
        self.assertEqual(
            [phase for phase, _ in phases],
            [
                "routed", "retrieving", "evidence_ready", "generating",
                "saving", "completed",
            ],
        )
        self.assertEqual(phases[-1][1]["evidence_count"], 1)
        self.assertEqual(mocks.retrieve.call_args.args[:2],
                         ("focused search", 2))
        self.assertIn("[1] src/a.py:1-3", session.chat_session.reply.call_args
                      .args[0])
        self.assertIn("Be brief.", session.chat_session.reply.call_args
                      .args[0])
        self.assertIn("token_callback",
                      session.chat_session.reply.call_args.kwargs)
        mocks.save.assert_called_once()
        tokens.assert_not_called()

    def test_search_adds_related_evidence_and_history(self) -> None:
        session = _session()
        session.turns = [
            _turn("Prior", "Prior answer [1]", (_evidence(start=20),))
        ]
        decision = RouteDecision(
            "test_intent", False, related_turn_ids=(1,), refers_back=True
        )
        with _answer_patches(
            session, _intent(related=True), decision
        ) as mocks:
            session.ask("And what about that?", use_router=False)
        turn = session.turns[-1]
        self.assertEqual([item.citation for item in turn.evidence], [1, 2])
        self.assertEqual([item.start_line for item in turn.evidence],
                         [1, 20])
        self.assertEqual(turn.parameters["related_evidence_count"], 1)
        self.assertEqual(turn.parameters["evidence_from_turns"], [1])
        self.assertEqual(turn.parameters["history_turn_ids"], [1])
        history = session.chat_session.restore_history.call_args.args[0]
        self.assertEqual([message["role"] for message in history],
                         ["system", "user", "assistant"])
        self.assertEqual(mocks.retrieve.call_count, 1)

    def test_search_uses_document_scope_candidates(self) -> None:
        session = _session()
        session.chunks = [
            chunk("a", path="one.py"),
            chunk("b", path="one.py"),
            chunk("c", path="two.py"),
        ]
        with _answer_patches(session, _intent()) as mocks:
            with patch.object(
                rag, "document_scope", return_value=(
                    (Path("one.py"),), "explicit"
                ),
            ) as scope:
                session.ask("Question", use_router=False)
        self.assertEqual(scope.call_args.args[2],
                         [Path("one.py"), Path("two.py")])
        self.assertEqual(mocks.retrieve.call_args.kwargs["scope"],
                         (Path("one.py"),))

    def test_document_answer_keeps_document_metadata(self) -> None:
        session = _session()
        sample = chunk("a", "first", path="paper.md")
        selection = DocumentSelection(
            Path("paper.md"), (), "named_file", (sample,), 1, 2, True
        )
        with _answer_patches(
            session, _intent(RETRIEVAL_MODE_DOCUMENT)
        ) as mocks:
            mocks.read.return_value = selection
            session.ask("Read paper", use_router=False)
        turn = session.turns[-1]
        self.assertEqual(turn.evidence[0].path, Path("paper.md"))
        self.assertEqual(turn.parameters["document"]["path"], "paper.md")
        self.assertEqual(turn.parameters["document"]["max_tokens"], 1200)
        mocks.read.assert_called_once()
        self.assertEqual(mocks.read.call_args.args[:2],
                         ("Read paper", 1200))
        mocks.retrieve.assert_not_called()

    def test_document_not_found_produces_no_evidence(self) -> None:
        session = _session()
        with _answer_patches(
            session, _intent(RETRIEVAL_MODE_DOCUMENT)
        ) as mocks:
            session.ask("Read missing", use_router=False)
        self.assertEqual(session.turns[-1].evidence, ())
        self.assertNotIn("document", session.turns[-1].parameters)
        mocks.read.assert_called_once()

    def test_none_mode_skips_retrieval(self) -> None:
        session = _session()
        phases = []
        with _answer_patches(
            session, _intent(RETRIEVAL_MODE_NONE)
        ) as mocks:
            session.ask("Hello", use_router=False,
                        phase_callback=lambda phase, data: phases.append(
                            (phase, data)
                        ))
        mocks.retrieve.assert_not_called()
        mocks.read.assert_not_called()
        self.assertEqual(session.turns[-1].evidence, ())
        self.assertEqual(
            [phase for phase, data in phases if data.get("skipped")],
            [rag.ANSWER_PHASE_EVIDENCE_READY],
        )
        self.assertEqual(session.chat_session.reply.call_args.args[0],
                         "Hello")

    def test_reuse_mode_combines_prior_turns_without_search(self) -> None:
        session = _session()
        session.turns = [
            _turn("One", evidence=(_evidence(start=10),)),
            _turn("Two", evidence=(_evidence(start=20),)),
        ]
        decision = RouteDecision(
            "test_intent", False, related_turn_ids=(1, 2)
        )
        with _answer_patches(
            session, _intent(RETRIEVAL_MODE_REUSE), decision
        ) as mocks:
            session.ask("Reuse", use_router=False)
        turn = session.turns[-1]
        self.assertEqual([item.start_line for item in turn.evidence],
                         [10, 20])
        self.assertTrue(turn.parameters["evidence_reused"])
        self.assertEqual(turn.parameters["evidence_from_turns"], [1, 2])
        mocks.retrieve.assert_not_called()

    def test_reuse_evidence_override_uses_last_turn(self) -> None:
        session = _session()
        session.turns = [_turn(evidence=(_evidence(),))]
        with _answer_patches(session, _intent()) as mocks:
            session.ask("Reuse last", reuse_evidence=True,
                        use_router=False)
        self.assertEqual(session.turns[-1].parameters["retrieval_mode"],
                         RETRIEVAL_MODE_REUSE)
        mocks.retrieve.assert_not_called()

    def test_reuse_without_matching_evidence_falls_back_to_search(
        self,
    ) -> None:
        session = _session()
        session.turns = [_turn(evidence=())]
        with _answer_patches(
            session, _intent(RETRIEVAL_MODE_REUSE)
        ) as mocks:
            session.ask("Follow up", use_router=False)
        self.assertEqual(session.turns[-1].parameters["retrieval_mode"],
                         RETRIEVAL_MODE_SEARCH)
        mocks.retrieve.assert_called_once()

    def test_intent_guidance_overrides_caller_guidance(self) -> None:
        session = _session()
        with _answer_patches(
            session, _intent(guidance="Configured guidance")
        ):
            session.ask("Question", answer_guidance="Caller guidance",
                        use_router=False)
        prompt = session.chat_session.reply.call_args.args[0]
        self.assertIn("Configured guidance", prompt)
        self.assertNotIn("Caller guidance", prompt)
        self.assertEqual(session.turns[-1].parameters["answer_guidance"],
                         "Configured guidance")

    def test_generation_settings_and_finish_reason_are_recorded(self) -> None:
        session = _session()
        session.chat_session.last_finish_reason = "length"
        with _answer_patches(session, _intent()) as mocks:
            session.ask(
                "Question", use_router=False,
                generation_settings={"temperature": 0.5},
            )
        self.assertEqual(
            session.chat_session.configure_generation.call_args.args[0],
            {"temperature": 0.5},
        )
        self.assertIn("trace_turn_id",
                      session.chat_session.configure_generation.call_args
                      .kwargs)
        self.assertEqual(session.turns[-1].parameters["finish_reason"],
                         "length")
        mocks.save.assert_called_once()
        session.turns.clear()
        session.chat_session.last_finish_reason = "unknown"
        with _answer_patches(session, _intent()):
            session.ask("Again", use_router=False)
        self.assertNotIn("finish_reason", session.turns[-1].parameters)

    def test_tool_answer_bypasses_chat_and_streams_once(self) -> None:
        session = _session()
        use = ToolUse("project_inventory", object())
        tokens = Mock()
        with _answer_patches(
            session, _intent(RETRIEVAL_MODE_NONE, tool=use)
        ) as mocks:
            answer = session.ask("Inventory", use_router=False,
                                 token_callback=tokens)
        self.assertEqual(answer, "tool answer")
        session.chat_session.reply.assert_not_called()
        self.assertEqual(session.turns[-1].parameters["tool"],
                         {"name": "tool"})
        self.assertEqual(session.turns[-1].evidence, (_evidence(),))
        tokens.assert_called_once_with("tool answer")
        self.assertEqual(mocks.run.call_args.kwargs["token_callback"],
                         tokens)

    def test_tool_that_streamed_does_not_repeat_answer(self) -> None:
        session = _session()
        use = ToolUse("project_inventory", object())
        tokens = Mock()
        with _answer_patches(
            session, _intent(RETRIEVAL_MODE_NONE, tool=use)
        ) as mocks:
            mocks.run.return_value = ToolResult(
                "streamed", (), {}, streamed=True
            )
            session.ask("Inventory", token_callback=tokens,
                        use_router=False)
        tokens.assert_not_called()
        self.assertEqual(session.turns[-1].answer, "streamed")

    def test_generation_failure_does_not_append_turn(self) -> None:
        session = _session()
        session.chat_session.reply.side_effect = RuntimeError("model failed")
        with _answer_patches(session, _intent()) as mocks:
            with self.assertRaisesRegex(RuntimeError, "model failed"):
                session.ask("Question", use_router=False)
        self.assertEqual(session.turns, [])
        mocks.save.assert_not_called()

    def test_save_failure_keeps_appended_turn(self) -> None:
        session = _session()
        with _answer_patches(session, _intent()) as mocks:
            mocks.save.side_effect = OSError("disk full")
            with self.assertRaisesRegex(OSError, "disk full"):
                session.ask("Question", use_router=False)
        self.assertEqual(len(session.turns), 1)

    def test_phase_callback_exception_propagates(self) -> None:
        session = _session()
        def fail(phase: str, details: dict) -> None:
            """Simulate a GUI callback failure."""
            raise RuntimeError("callback failure")
        with _answer_patches(session, _intent()):
            with self.assertRaisesRegex(RuntimeError, "callback failure"):
                session.ask("Question", use_router=False,
                            phase_callback=fail)
        self.assertEqual(session.turns, [])


class TestRAGLifecycle(unittest.TestCase):
    """Test model ownership, switching, refresh, and token bookkeeping."""

    def test_close_releases_sessions_once(self) -> None:
        session = _session()
        session.close()
        session.close()
        self.assertTrue(session._closed)
        session.chat_session.close.assert_called_once_with()
        session.embedding_session.close.assert_called_once_with()

    def test_close_releases_distinct_router_and_owned_trace(self) -> None:
        session = _session()
        router = MagicMock()
        session._router_session = router
        session._owns_trace = True
        session.trace = MagicMock()
        session.close()
        self.assertIsNone(session._router_session)
        router.close.assert_called_once()
        session.trace.close.assert_called_once()

    def test_close_marks_closed_even_if_model_cleanup_fails(self) -> None:
        session = _session()
        session.chat_session.close.side_effect = OSError("close failed")
        session.trace = MagicMock()
        session._owns_trace = True
        with self.assertRaisesRegex(OSError, "close failed"):
            session.close()
        self.assertTrue(session._closed)
        session.trace.close.assert_called_once()
        session.embedding_session.close.assert_not_called()

    def test_refresh_sources_replaces_only_retrieval_state(self) -> None:
        session = _session()
        previous_chat = session.chat_session
        new_project = SimpleNamespace(root=Path("/new"))
        values = ([chunk("new")], sentinel.indexes, "reused", {"a": 1})
        session._prepare_retrieval = Mock(return_value=values)
        session.refresh_sources(new_project)
        self.assertIs(session.project, new_project)
        self.assertEqual(session.chunks, values[0])
        self.assertIs(session.indexes, sentinel.indexes)
        self.assertEqual(session.index_status, "reused")
        self.assertEqual(session.source_changes, {"a": 1})
        self.assertIs(session.chat_session, previous_chat)

    def test_refresh_failure_keeps_old_index_but_new_project(self) -> None:
        session = _session()
        original_index = session.indexes
        project = SimpleNamespace(root=Path("/new"))
        session._prepare_retrieval = Mock(side_effect=OSError("bad scan"))
        with self.assertRaisesRegex(OSError, "bad scan"):
            session.refresh_sources(project)
        self.assertIs(session.project, project)
        self.assertIs(session.indexes, original_index)

    def test_switch_chat_model_rejects_closed_or_wrong_backend(self) -> None:
        session = _session()
        session._closed = True
        with self.assertRaisesRegex(RuntimeError, "closed"):
            session.use_chat_model("other")
        session._closed = False
        session.backend = "torch"
        with self.assertRaisesRegex(RuntimeError, "MLX"):
            session.use_chat_model("other")

    def test_switch_chat_model_noop_and_new_session(self) -> None:
        session = _session()
        session.router_model = "different-router"
        original = session.chat_session
        session.use_chat_model(original.model_id)
        original.close.assert_not_called()
        replacement = MagicMock()
        replacement.model_id = "other"
        session._new_chat_session = Mock(return_value=replacement)
        session.use_chat_model("other")
        original.close.assert_called_once()
        self.assertIs(session.chat_session, replacement)
        self.assertIsNone(session._router_session)
        self.assertEqual(
            replacement.restore_history.call_args.args[0],
            [{"role": "system", "content": "system"}],
        )
        self.assertEqual(
            session._new_chat_session.call_args.kwargs,
            {"thinking": False, "temperature": 0.2,
             "max_new_tokens": 128},
        )

    def test_switch_chat_model_reuses_loaded_router(self) -> None:
        session = _session()
        session.router_model = "different-router"
        original = session.chat_session
        router = MagicMock()
        router.model_id = "router"
        session._router_session = router
        session.use_chat_model("router")
        self.assertIs(session.chat_session, router)
        self.assertIsNone(session._router_session)
        router.configure_generation.assert_called_once()
        original.close.assert_called_once()

    def test_switch_chat_model_preserves_old_router_when_shared(self) -> None:
        session = _session()
        original = session.chat_session
        session.router_model = original.model_id
        replacement = MagicMock()
        session._new_chat_session = Mock(return_value=replacement)
        session.use_chat_model("other")
        original.close.assert_not_called()
        self.assertIs(session._router_session, original)
        self.assertIs(session.chat_session, replacement)

    def test_switch_chat_model_rolls_back_failed_new_load(self) -> None:
        session = _session()
        session.router_model = "different-router"
        original = session.chat_session
        restored = MagicMock()
        session._new_chat_session = Mock(
            side_effect=[OSError("cannot load"), restored]
        )
        with self.assertRaisesRegex(OSError, "cannot load"):
            session.use_chat_model("missing")
        original.close.assert_called_once()
        self.assertIs(session.chat_session, restored)
        restored.restore_history.assert_called_once_with(
            [{"role": "system", "content": "system"}]
        )

    def test_switch_chat_model_keeps_shared_router_on_failure(self) -> None:
        session = _session()
        original = session.chat_session
        session.router_model = original.model_id
        session._new_chat_session = Mock(side_effect=OSError("failed"))
        with self.assertRaises(OSError):
            session.use_chat_model("other")
        self.assertIs(session.chat_session, original)
        original.close.assert_not_called()

    def test_router_model_change_closes_only_separate_session(self) -> None:
        session = _session()
        router = MagicMock()
        session._router_session = router
        session.use_router_model("other")
        router.close.assert_called_once()
        self.assertIsNone(session._router_session)
        self.assertEqual(session.router_model, "other")
        session.use_router_model("other")
        router.close.assert_called_once()
        session._closed = True
        with self.assertRaisesRegex(RuntimeError, "closed"):
            session.use_router_model("again")

    def test_router_session_is_shared_or_loaded_lazily(self) -> None:
        session = _session()
        self.assertIs(session._router_chat(), session.chat_session)
        session.router_model = "router"
        router = MagicMock()
        session._new_chat_session = Mock(return_value=router)
        self.assertIs(session._router_chat(), router)
        self.assertIs(session._router_chat(), router)
        session._new_chat_session.assert_called_once_with("router")

    def test_router_reply_delegates_to_side_reply(self) -> None:
        session = _session()
        with patch.object(rag, "side_reply", return_value="yes") as side:
            self.assertEqual(session.router_reply("system", "question",
                                                  max_new_tokens=12), "yes")
            self.assertEqual(session._ask_router("s", "q", 9), "yes")
        self.assertEqual(side.call_args_list[0].kwargs,
                         {"max_new_tokens": 12})
        self.assertEqual(side.call_args_list[1].kwargs,
                         {"max_new_tokens": 9})

    def test_new_chat_session_uses_project_backend_and_trace(self) -> None:
        session = _session()
        with patch.object(rag, "ChatSession") as constructor:
            self.assertIs(
                session._new_chat_session("model", temperature=0.6),
                constructor.return_value,
            )
        self.assertEqual(constructor.call_args.args, ("model",))
        self.assertEqual(constructor.call_args.kwargs["backend"],
                         BACKEND_MLX)
        self.assertEqual(constructor.call_args.kwargs["temperature"], 0.6)
        self.assertFalse(constructor.call_args.kwargs["allow_download"])

    def test_usage_counters_include_distinct_router_only(self) -> None:
        session = _session()
        session._router_session = session.chat_session
        self.assertEqual(len(session.usage_mark()), 1)
        router = MagicMock()
        router.input_token_total = 3
        router.output_token_total = 2
        session._router_session = router
        mark = session.usage_mark()
        self.assertEqual(len(mark), 2)
        session.chat_session.input_token_total = 15
        session.chat_session.output_token_total = 9
        router.input_token_total = 8
        router.output_token_total = 4
        self.assertEqual(
            session.usage_since(mark),
            {"input_tokens": 10, "output_tokens": 6},
        )
        self.assertEqual(
            session.usage_since({}),
            {"input_tokens": 23, "output_tokens": 13},
        )
        self.assertEqual(
            session.usage_since({id(session.chat_session): (100, 100)}),
            {"input_tokens": 8, "output_tokens": 4},
        )

    def test_model_sessions_ignores_missing_counters(self) -> None:
        session = _session()
        session._router_session = object()
        self.assertEqual(session._model_sessions(), [session.chat_session])

    def test_index_metadata_and_turn_parameters(self) -> None:
        session = _session()
        metadata = session._index_metadata()
        self.assertEqual(metadata["embedding_model"], "embedding-model")
        self.assertEqual(metadata["extensions"], [".py"])
        self.assertNotIn("pdf_text_extractor", metadata)
        self.assertNotIn("format_version", metadata)
        parameters = session._parameters(8)
        self.assertEqual(parameters["retrieval_count"], 8)
        self.assertEqual(parameters["chat_model"], "chat-model")
        self.assertEqual(parameters["temperature"], 0.2)
        self.assertEqual(session._parameters()["retrieval_count"], 5)
        session.project.profile.extensions = (".pdf",)
        self.assertEqual(
            session._index_metadata()["pdf_text_extractor"],
            rag.PDF_TEXT_EXTRACTOR,
        )
        with patch.object(rag, "project_type") as lookup:
            lookup.return_value.code_graph = False
            self.assertFalse(session._index_metadata()["code_graph"])


class TestRAGRoutingAndTools(unittest.TestCase):
    """Test intent routing, prior-turn references, and tool adapters."""

    def test_routing_off_uses_default_without_loading_router(self) -> None:
        session = _session()
        kind = _kind(_intent())
        session._router_chat = Mock()
        report = Mock()
        decision = session._route("Question", kind, False, "turn", report)
        self.assertEqual(decision.intent, "test_intent")
        self.assertTrue(decision.standalone)
        self.assertEqual(decision.fallback, "routing off")
        session._router_chat.assert_not_called()
        report.assert_not_called()
        session.turns = [_turn()]
        self.assertFalse(session._route(
            "Followup", kind, False, "turn", report
        ).standalone)

    def test_routing_on_records_classification_and_fallback(self) -> None:
        session = _session()
        kind = _kind(_intent())
        session.trace = MagicMock()
        session.trace.include_content = True
        decision = RouteDecision(
            "test_intent", True, raw="raw reply", fallback="bad JSON",
            dropped_turns=((2, "invalid"),),
        )
        session._router_chat = Mock(return_value=sentinel.router)
        report = Mock()
        with patch.object(rag, "route_question", return_value=decision) as rt:
            self.assertIs(session._route(
                "Question", kind, True, "turn", report
            ), decision)
        rt.assert_called_once_with(
            sentinel.router, kind, session.turns, "Question"
        )
        report.assert_called_once_with(
            rag.ANSWER_PHASE_ROUTING, message="Identifying intent"
        )
        event = [
            call for call in session.trace.emit.call_args_list
            if call.args[0] == "intent.classification.decided"
        ][0]
        self.assertEqual(event.kwargs["data"]["raw"], "raw reply")
        self.assertEqual(event.kwargs["data"]["dropped_turns"],
                         [{"turn_id": 2, "reason": "invalid"}])

    def test_earlier_turns_walks_related_chain_and_marks_last(self) -> None:
        session = _session()
        session.turns = [
            _turn("First", evidence=(_evidence(start=1),)),
            _turn("Second", parameters={"related_turn_ids": [1]}),
            _turn("Third"),
        ]
        decision = RouteDecision(
            "test_intent", False, related_turn_ids=(2,)
        )
        earlier = session._earlier_turns(decision)
        self.assertEqual([item.number for item in earlier], [3, 2, 1])
        self.assertEqual([item.linked for item in earlier],
                         [False, True, True])
        self.assertEqual(earlier[-1].paths, (Path("src/a.py"),))
        self.assertEqual(earlier[-1].texts, ("source text",))
        self.assertEqual(earlier[-1].lines, (1,))

    def test_earlier_turns_handles_back_reference_without_links(self) -> None:
        session = _session()
        session.turns = [_turn("First")]
        self.assertEqual(session._earlier_turns(
            RouteDecision("test_intent", False)
        ), ())
        earlier = session._earlier_turns(
            RouteDecision("test_intent", False, refers_back=True)
        )
        self.assertEqual(len(earlier), 1)
        self.assertTrue(earlier[0].linked)

    def test_earlier_turns_rejects_invalid_links_and_caps_length(self) -> None:
        session = _session()
        session.turns = [
            _turn(str(number), parameters={"related_turn_ids": [
                number - 1, "bad", 999
            ]})
            for number in range(1, 15)
        ]
        decision = RouteDecision(
            "test_intent", False, related_turn_ids=(14, 0, 100)
        )
        earlier = session._earlier_turns(decision)
        self.assertEqual(len(earlier), rag._MAX_EARLIER_TURNS)
        self.assertEqual(earlier[0].number, 14)
        self.assertEqual(earlier[-1].number, 9)

    def test_run_tool_builds_context_and_preserves_callbacks(self) -> None:
        session = _session()
        session.chunks = [chunk("one")]
        session._rank_everything = Mock(return_value=[ranked(chunk("one"))])
        session._ask_router = Mock(return_value="router")
        session._earlier_turns = Mock(return_value=())
        session._source_files = Mock(return_value={Path("a.py"): 2})
        session._unsearchable_sources = Mock(return_value=[])
        use = ToolUse("project_inventory", object())
        decision = RouteDecision("test_intent", True)
        phases = Mock()
        callback = Mock()
        writer = Mock()
        choose = Mock()
        tokens = Mock()
        result = ToolResult("result", (), {"ok": True})
        with patch.object(rag, "run_tool", return_value=result) as runner:
            actual = session._run_tool(
                use, "Inventory", decision, trace_turn_id="t",
                report_phase=phases, confirm_callback=callback,
                pdf_writer=writer, choice_callback=choose,
                token_callback=tokens, answer_guidance="Short",
            )
        self.assertIs(actual, result)
        request = runner.call_args.args[1]
        context = runner.call_args.args[2]
        self.assertEqual(request.question, "Inventory")
        self.assertEqual(request.guidance, "Short")
        self.assertIs(context.confirm, callback)
        self.assertIs(context.write_pdf, writer)
        self.assertIs(context.choose, choose)
        self.assertEqual(context.model_id, "chat-model")
        self.assertEqual(context.rank("topic"), [ranked(chunk("one"))])
        self.assertEqual(context.source_files(), {Path("a.py"): 2})
        self.assertEqual(context.unsearchable([]), [])
        context.progress("step", current=1)
        context.done("done")
        self.assertEqual(phases.call_args_list[0].args[0],
                         rag.ANSWER_PHASE_PROCESSING)
        self.assertEqual(phases.call_args_list[1].args[0],
                         rag.ANSWER_PHASE_EVIDENCE_READY)
        self.assertEqual(phases.call_args_list[1].kwargs["tool"],
                         use.name)

    def test_tool_side_question_and_writer_use_isolated_reply(self) -> None:
        session = _session()
        session._rank_everything = Mock(return_value=[])
        use = ToolUse("project_inventory", object())
        tokens = Mock()
        with patch.object(rag, "run_tool") as runner, \
            patch.object(rag, "side_reply", return_value="reply") as side:
            session._run_tool(
                use, "Question", RouteDecision("test_intent", True),
                trace_turn_id=None, report_phase=Mock(),
                token_callback=tokens,
            )
            context = runner.call_args.args[2]
            self.assertEqual(context.ask_model("s", "q", 5), "reply")
            self.assertEqual(context.write("s", "q", 6), "reply")
        self.assertEqual(side.call_args_list[0].kwargs,
                         {"max_new_tokens": 5})
        self.assertEqual(side.call_args_list[1].kwargs,
                         {"max_new_tokens": 6,
                          "token_callback": tokens})


class TestRAGConversationPersistence(unittest.TestCase):
    """Verify resumable turns, checkpoint identities, and path containment."""

    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.session = _session(self.root)
        self.session.conversation_path.parent.mkdir(parents=True)

    def test_absent_conversation_has_empty_state(self) -> None:
        self.assertEqual(self.session._load_conversation(), ([], [], None))

    def test_save_and_reload_grouped_conversation_without_trace(self) -> None:
        session = self.session
        session.turns = [
            _turn("Question", "Answer [1]", (_evidence(),))
        ]
        session._save_conversation(trace_turn_id="turn-1")
        first = session._checkpoint_id
        self.assertTrue(first.startswith("checkpoint_"))
        payload = json.loads(session.conversation_path.read_text())
        saved = rag.conversation_state(payload)
        self.assertEqual(saved["checkpoint_id"], first)
        self.assertEqual(saved["checkpoint"]["turn_count"], 1)
        self.assertIsNone(saved["checkpoint"]["parent_checkpoint_id"])
        messages, turns, checkpoint = session._load_conversation()
        self.assertEqual(turns, session.turns)
        self.assertEqual(checkpoint, first)
        self.assertEqual([message["role"] for message in messages],
                         ["system", "user", "assistant"])
        session.turns.append(_turn("Second", "Answer"))
        session._save_conversation()
        payload = json.loads(session.conversation_path.read_text())
        saved = rag.conversation_state(payload)
        self.assertEqual(saved["checkpoint"]["parent_checkpoint_id"],
                         first)
        self.assertNotEqual(saved["checkpoint_id"], first)

    def test_save_conversation_with_trace_uses_same_record(self) -> None:
        session = self.session
        session.trace = MagicMock()
        session.trace.enabled = True
        session.turns = [_turn(evidence=(_evidence(),))]
        session._save_conversation()
        session.trace.update_conversation.assert_called_once()
        state = session.trace.update_conversation.call_args.args[0]
        self.assertEqual(state["checkpoint_id"], session._checkpoint_id)
        self.assertEqual(state["turns"][0]["evidence"][0]["path"],
                         "src/a.py")
        self.assertFalse(session.conversation_path.exists())

    def test_load_canonical_turns_reconstructs_model_history(self) -> None:
        session = self.session
        data = {"sessions": {CONVERSATION_SESSION_KEY:
                             new_conversation_session({
            "system_prompt": "Saved system prompt",
            "turns": [rag._turn_to_data(_turn())],
            "checkpoint_id": "checkpoint_previous",
        })}}
        session.conversation_path.write_text(json.dumps(data))
        messages, turns, checkpoint = session._load_conversation()
        self.assertEqual(messages[0]["content"], "Saved system prompt")
        self.assertEqual([message["role"] for message in messages],
                         ["system", "user", "assistant"])
        self.assertEqual(len(turns), 1)
        self.assertEqual(checkpoint, "checkpoint_previous")

    def test_load_conversation_rejects_corrupt_records(self) -> None:
        session = self.session
        session.conversation_path.write_text("not JSON")
        with self.assertRaisesRegex(ValueError, "invalid"):
            session._load_conversation()
        corrupt = {"sessions": {CONVERSATION_SESSION_KEY:
                                 new_conversation_session({
            "turns": [{"timestamp": "x", "evidence": "bad"}],
        })}}
        session.conversation_path.write_text(json.dumps(corrupt))
        with self.assertRaises(ValueError):
            session._load_conversation()
        obsolete = {"sessions": {CONVERSATION_SESSION_KEY:
                                  new_conversation_session({
            "turns": [], "chat_messages": [{"role": "user"}],
        })}}
        session.conversation_path.write_text(json.dumps(obsolete))
        with self.assertRaisesRegex(ValueError, "invalid"):
            session._load_conversation()
        for record in ({"conversation": {"turns": []}}, [], None):
            session.conversation_path.write_text(json.dumps(record))
            with self.subTest(record=record):
                with self.assertRaisesRegex(ValueError, "invalid"):
                    session._load_conversation()

    def test_load_conversation_ignores_invalid_checkpoint(self) -> None:
        session = self.session
        session.conversation_path.write_text(json.dumps({
            "sessions": {CONVERSATION_SESSION_KEY:
                         new_conversation_session({
                "turns": [], "checkpoint_id": 100,
            })},
        }))
        self.assertIsNone(session._load_conversation()[2])

    def test_save_preserves_existing_unrelated_sessions(self) -> None:
        session = self.session
        session.conversation_path.write_text(json.dumps({
            "sessions": {"custom": {"title": "Keep"}},
        }))
        session._save_conversation()
        payload = json.loads(session.conversation_path.read_text())
        self.assertEqual(payload["sessions"]["custom"],
                         {"title": "Keep"})

    def test_untraced_save_does_not_convert_obsolete_state(self) -> None:
        session = self.session
        session.conversation_path.write_text(json.dumps({
            "conversation": {"turns": [{"question": "Do not replace"}]},
        }))
        before = session.conversation_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "invalid"):
            session._save_conversation()
        self.assertEqual(session.conversation_path.read_bytes(), before)

    def test_untraced_save_preserves_invalid_grouped_state(self) -> None:
        session = self.session
        bad_group = new_conversation_session({
            "turns": [], "checkpoint_id": "old",
        })
        bad_group["conversation"] = {"turns": [{"question": "Old"}]}
        session.conversation_path.write_text(json.dumps({
            "sessions": {CONVERSATION_SESSION_KEY: bad_group},
        }))
        before = session.conversation_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "invalid"):
            session._save_conversation()
        self.assertEqual(session.conversation_path.read_bytes(), before)

    def test_write_json_is_atomic_and_replaces_existing_data(self) -> None:
        path = self.root / "payload.json"
        rag._write_json(path, {"a": 1})
        rag._write_json(path, {"a": 2})
        self.assertEqual(json.loads(path.read_text()), {"a": 2})
        self.assertFalse((self.root / "payload.json.tmp").exists())

    def test_resolve_conversation_path_validation(self) -> None:
        session = self.session
        expected = session._resolve_conversation_path(
            "logs/traces/conversation_custom.json"
        )
        self.assertEqual(expected, (
            self.root / "logs/traces/conversation_custom.json"
        ).resolve())
        self.assertEqual(
            session._resolve_conversation_path(expected), expected
        )
        for name in (
            "other/conversation_bad.json",
            "logs/traces/not_conversation.json",
            "logs/traces/conversation_bad.txt",
            "../conversation_escape.json",
        ):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    session._resolve_conversation_path(name)

    def test_resolve_conversation_allocates_unique_default_names(self) -> None:
        session = self.session
        first = session._resolve_conversation_path(None)
        first.write_text("{}")
        second = session._resolve_conversation_path(None)
        self.assertNotEqual(first, second)
        self.assertTrue(second.name.startswith("conversation_"))
        self.assertTrue(second.parent.is_dir())

    def test_resolve_rejects_symlink_escape(self) -> None:
        session = self.session
        outside = self.root / "elsewhere"
        outside.mkdir()
        link = self.root / "logs" / "outside"
        link.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "inside project logs"):
            session._resolve_conversation_path(
                "logs/outside/conversation_escape.json"
            )

    def test_archive_conversation_moves_existing_file(self) -> None:
        session = self.session
        self.assertIsNone(session._archive_conversation())
        session.conversation_path.write_text("original")
        archive = session._archive_conversation()
        self.assertTrue(archive.is_file())
        self.assertEqual(archive.read_text(), "original")
        self.assertFalse(session.conversation_path.exists())
        self.assertTrue(archive.name.startswith("conversation_"))

    def test_reset_archives_turns_and_preserves_models(self) -> None:
        session = self.session
        session.trace = MagicMock()
        session.turns = [_turn()]
        session._checkpoint_id = "old-checkpoint"
        session.conversation_path.write_text("old")
        previous_chat = session.chat_session
        archived = session.reset()
        self.assertTrue(archived.exists())
        self.assertEqual(session.turns, [])
        self.assertIsNone(session._checkpoint_id)
        self.assertIs(session.chat_session, previous_chat)
        session.chat_session.reset.assert_called_once()
        session.trace.reset_conversation_state.assert_called_once()
        session._closed = True
        with self.assertRaisesRegex(RuntimeError, "closed"):
            session.reset()

    def test_switch_conversation_reuses_models_and_restores_settings(
        self,
    ) -> None:
        session = self.session
        target = session._resolve_conversation_path(
            "logs/traces/conversation_second.json"
        )
        earlier = _turn(parameters={
            "temperature": 0.7, "thinking": True,
            "max_new_tokens": 64,
        })
        target.write_text(json.dumps({
            "sessions": {CONVERSATION_SESSION_KEY:
                         new_conversation_session({
                "turns": [rag._turn_to_data(earlier)],
                "checkpoint_id": "restored",
            })},
        }))
        old_chat = session.chat_session
        turns = session.switch_conversation(target, trace=NullTraceLogger())
        self.assertEqual(turns, [earlier])
        self.assertIs(session.chat_session, old_chat)
        self.assertEqual(session._checkpoint_id, "restored")
        self.assertEqual(session.conversation_path, target)
        self.assertEqual(
            session.chat_session.configure_generation.call_args.args[0],
            {"temperature": 0.7, "thinking": True,
             "max_new_tokens": 64},
        )
        self.assertEqual(session.chat_session.trace, session.trace)
        self.assertEqual(session.embedding_session.trace, session.trace)

    def test_switch_conversation_uses_profile_defaults_without_turns(
        self,
    ) -> None:
        session = self.session
        session._owns_trace = True
        old_trace = MagicMock()
        old_trace.enabled = False
        session.trace = old_trace
        target = session._resolve_conversation_path(
            "logs/traces/conversation_empty.json"
        )
        session.switch_conversation(target, trace=NullTraceLogger())
        old_trace.close.assert_called_once()
        self.assertEqual(
            session.chat_session.configure_generation.call_args.args[0],
            {"temperature": 0.2, "thinking": False,
             "max_new_tokens": 128},
        )

    def test_switch_rejects_closed_session_and_mismatched_trace(self) -> None:
        session = self.session
        session._closed = True
        with self.assertRaisesRegex(RuntimeError, "closed"):
            session.switch_conversation(
                "logs/traces/conversation_second.json",
                trace=NullTraceLogger(),
            )
        session._closed = False
        bad = MagicMock()
        bad.enabled = True
        bad.output_path = self.root / "logs/traces/conversation_wrong.json"
        with self.assertRaisesRegex(ValueError, "conversation path"):
            session.switch_conversation(
                "logs/traces/conversation_second.json", trace=bad
            )

    def test_trace_configuration_is_bound_to_conversation_path(self) -> None:
        path = self.session.conversation_path.resolve()
        disabled, owned = rag._conversation_trace(
            TraceConfig(enabled=False), path
        )
        self.assertFalse(disabled.enabled)
        self.assertFalse(owned)
        enabled, owned = rag._conversation_trace(None, path)
        self.assertTrue(enabled.enabled)
        self.assertTrue(owned)
        self.assertEqual(enabled.output_path.resolve(), path)
        enabled.close()
        with self.assertRaisesRegex(ValueError, "conversation path"):
            rag._conversation_trace(
                TraceConfig(enabled=True, output_path=self.root / "other"),
                path,
            )

    def test_trace_sink_must_match_exact_path_when_enabled(self) -> None:
        path = self.session.conversation_path.resolve()
        disabled = NullTraceLogger()
        rag._check_conversation_trace(disabled, path)
        bad = MagicMock()
        bad.enabled = True
        bad.output_path = None
        with self.assertRaisesRegex(ValueError, "exact"):
            rag._check_conversation_trace(bad, path)
        bad.output_path = self.root / "other.json"
        with self.assertRaisesRegex(ValueError, "conversation path"):
            rag._check_conversation_trace(bad, path)
        bad.output_path = path
        rag._check_conversation_trace(bad, path)

    def test_source_text_cache_discards_invalid_entries(self) -> None:
        indexes = self.root / "indexes"
        self.assertEqual(rag._read_text_cache(indexes), {})
        rag._write_text_cache(indexes, {
            "good.py": {"sha256": "hash", "text": "source"},
        })
        self.assertEqual(rag._read_text_cache(indexes)["good.py"]["text"],
                         "source")
        cache = indexes / rag._TEXT_CACHE
        cache.write_text(json.dumps({
            "good": {"sha256": "hash", "text": "source"},
            "bad": {"sha256": 1, "text": "source"},
            "missing": {"sha256": "hash"},
            "null": None,
        }))
        self.assertEqual(list(rag._read_text_cache(indexes)), ["good"])
        cache.write_text("not JSON")
        self.assertEqual(rag._read_text_cache(indexes), {})
        cache.write_text("[]")
        self.assertEqual(rag._read_text_cache(indexes), {})


class TestRAGSourcePreparation(unittest.TestCase):
    """Check extracted-text caching and incremental source chunk reuse."""

    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.session = _session(self.root)
        self.session.project.source.mkdir()
        self.session.project.indexes.mkdir()
        self.first = self.session.project.source / "a.py"
        self.second = self.session.project.source / "b.py"
        self.first.write_text("def foo():\n    return 1\n")
        self.second.write_text("def bar():\n    return 2\n")
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

    def test_source_preparation_extracts_chunks_and_code_graph(self) -> None:
        chunks, code_graph = self.session._prepare_source(
            self.embedding, report=self._report()
        )
        self.assertEqual({value.path for value in chunks},
                         {Path("a.py"), Path("b.py")})
        self.assertTrue(code_graph.entities)
        cached = rag._read_text_cache(self.session.project.indexes)
        self.assertEqual(set(cached), {"a.py", "b.py"})
        self.assertIn("return 1", cached["a.py"]["text"])

    def test_unchanged_text_cache_skips_source_reader(self) -> None:
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
        chunks, _ = self.session._prepare_source(
            self.embedding, report=self._report()
        )
        saved = {Path("a.py"): [
            value for value in chunks if value.path == Path("a.py")
        ]}
        self.second.write_text("def bar():\n    return 999\n")
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
        self.session._prepare_source(self.embedding, report=self._report())
        chunks, _ = self.session._prepare_source(
            self.embedding, report=self._report(["b.py"])
        )
        self.assertEqual({value.path for value in chunks}, {Path("b.py")})
        self.assertEqual(set(rag._read_text_cache(
            self.session.project.indexes
        )), {"b.py"})

    def test_empty_source_returns_empty_graph_and_chunks(self) -> None:
        chunks, code_graph = self.session._prepare_source(
            self.embedding, report=self._report([])
        )
        self.assertEqual(chunks, [])
        self.assertEqual(code_graph, CodeGraph((), ()))

    def test_disabled_code_graph_skips_graph_extraction(self) -> None:
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
        with self.assertRaises(OSError):
            self.session._prepare_source(
                self.embedding, report=self._report(["missing.py"])
            )

    def test_source_preparation_scans_when_no_report_is_supplied(self) -> None:
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
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.session = _session(Path(temporary.name))
        self.session.project.indexes.mkdir()

    def test_source_files_reads_manifest_sizes(self) -> None:
        manifest = self.session.project.indexes / "source_manifest.json"
        manifest.write_text(json.dumps({
            "files": {
                "a.py": {"size_bytes": 123},
                "b.py": {"size_bytes": 0},
                "c.py": {},
            },
        }))
        self.assertEqual(self.session._source_files(), {
            Path("a.py"): 123, Path("b.py"): 0, Path("c.py"): 0,
        })
        self.assertEqual(
            self.session._unsearchable_sources([Path("a.py")]),
            [Path("b.py"), Path("c.py")],
        )

    def test_invalid_source_manifest_yields_empty_inventory(self) -> None:
        manifest = self.session.project.indexes / "source_manifest.json"
        self.assertEqual(self.session._source_files(), {})
        for value in ("bad JSON", "{}", '{"files": []}',
                      '{"files": {"a": {"size_bytes": "oops"}}}'):
            with self.subTest(value=value):
                manifest.write_text(value)
                self.assertEqual(self.session._source_files(), {})


class TestRAGSessionConstruction(unittest.TestCase):
    """Verify startup, restored history, and partial-initialization cleanup."""

    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "logs").mkdir()
        from qachat.project import QAProject, default_project_profile

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
        with patch.object(rag, "open_project") as opened:
            with self.assertRaisesRegex(ValueError, "progress_callback"):
                rag.RAGSession(self.root, progress_callback=3)
        opened.assert_not_called()

    def test_constructor_loads_models_and_reports_ready(self) -> None:
        progress = []
        with self._construction_patches() as mocks:
            session = rag.RAGSession(
                self.root, progress_callback=progress.append,
                router_model="custom-router",
            )
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

    def test_constructor_restores_history_and_generation_settings(
        self,
    ) -> None:
        previous = _turn(parameters={
            "thinking": True, "temperature": 0.3,
            "max_new_tokens": 64,
        })
        history = [{"role": "user", "content": "prior"}]
        with self._construction_patches(
            restored=(history, [previous], "checkpoint")
        ) as mocks:
            session = rag.RAGSession(self.root)
        self.assertEqual(session.turns, [previous])
        self.assertEqual(session._checkpoint_id, "checkpoint")
        mocks.chat.return_value.restore_history.assert_called_once_with(
            history
        )
        mocks.chat.return_value.configure_generation.assert_called_once_with(
            {"thinking": True, "temperature": 0.3,
             "max_new_tokens": 64}
        )

    def test_constructor_closes_models_and_trace_on_index_failure(
        self,
    ) -> None:
        with self._construction_patches() as mocks:
            mocks.prepared.side_effect = OSError("index failed")
            with self.assertRaisesRegex(OSError, "index failed"):
                rag.RAGSession(self.root)
        mocks.embeddings.return_value.close.assert_called_once()
        mocks.chat.return_value.close.assert_not_called()
        mocks.trace.close.assert_called_once()
        mocks.trace.exception.assert_called_once()

    def test_constructor_closes_both_models_on_restore_failure(self) -> None:
        with self._construction_patches() as mocks:
            mocks.loaded.side_effect = ValueError("bad conversation")
            with self.assertRaisesRegex(ValueError, "bad conversation"):
                rag.RAGSession(self.root)
        mocks.embeddings.return_value.close.assert_called_once()
        mocks.chat.return_value.close.assert_called_once()
        mocks.trace.close.assert_called_once()

    def test_constructor_closes_trace_on_embedding_load_failure(
        self,
    ) -> None:
        with self._construction_patches() as mocks:
            mocks.embeddings.side_effect = OSError("missing model")
            with self.assertRaisesRegex(OSError, "missing model"):
                rag.RAGSession(self.root)
        mocks.trace.close.assert_called_once()
        mocks.chat.assert_not_called()



class TestRAGIndexMetadataRecovery(unittest.TestCase):
    """Check recovery from malformed manifest and metadata files."""

    def setUp(self) -> None:
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
        self.manifest.write_text("invalid JSON")
        with self._index_patches() as mocks:
            _, _, status, _ = self.session._prepare_retrieval(
                self.embedding
            )
        self.assertEqual(status, "rebuilt")
        self.assertIsNone(mocks.compare.call_args.args[0])
        self.assertIsInstance(json.loads(self.manifest.read_text()), dict)

    def test_nonmapping_manifest_is_treated_as_missing(self) -> None:
        self.manifest.write_text("[]")
        with self._index_patches() as mocks:
            self.session._prepare_retrieval(self.embedding)
        self.assertIsNone(mocks.compare.call_args.args[0])

    def test_invalid_settings_json_triggers_clean_rebuild(self) -> None:
        self.settings.write_text("invalid JSON")
        with self._index_patches():
            _, _, status, _ = self.session._prepare_retrieval(
                self.embedding
            )
        self.assertEqual(status, "rebuilt")
        self.assertIn("embedding_model_identity",
                      json.loads(self.settings.read_text()))

    def test_nonmapping_settings_are_replaced(self) -> None:
        self.settings.write_text("[]")
        with self._index_patches():
            self.session._prepare_retrieval(self.embedding)
        self.assertIsInstance(json.loads(self.settings.read_text()), dict)

    def test_invalid_saved_manifest_is_compared_against_none(self) -> None:
        self.manifest.write_text('{"files": {}}')
        with self._index_patches() as mocks:
            mocks.compare.side_effect = [
                ValueError("bad manifest"), dict(self.changes)
            ]
            self.session._prepare_retrieval(self.embedding)
        self.assertEqual(mocks.compare.call_count, 2)
        self.assertIsNone(mocks.compare.call_args.args[0])

    def test_missing_model_source_is_rejected(self) -> None:
        self.embedding.model_source = None
        with self._index_patches() as mocks:
            with self.assertRaisesRegex(ValueError, "source"):
                self.session._prepare_retrieval(self.embedding)
        mocks.prepare.assert_not_called()


class TestRAGAdditionalBranches(unittest.TestCase):
    """Cover optional trace content and metadata-preserving fallbacks."""

    def test_forensic_trace_includes_question_in_turn_span(self) -> None:
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
        session = _session()
        session.use_router_model("another-router")
        self.assertEqual(session.router_model, "another-router")
        session.chat_session.close.assert_not_called()

    def test_save_conversation_rejects_nonmapping_json(self) -> None:
        with TemporaryDirectory() as directory:
            session = _session(Path(directory))
            session.conversation_path.parent.mkdir(parents=True)
            session.conversation_path.write_text("[]")
            before = session.conversation_path.read_bytes()
            with self.assertRaisesRegex(ValueError, "invalid"):
                session._save_conversation()
            self.assertEqual(session.conversation_path.read_bytes(), before)

    def test_explicit_enabled_trace_config_is_bound(self) -> None:
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
        with TemporaryDirectory() as directory:
            from qachat.project import QAProject, default_project_profile
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
            json.dumps(metadata)
        )
        (self.session.project.indexes / "source_manifest.json").write_text(
            json.dumps({"selected_files": ["a.py"], "files": {}})
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
        self.assertEqual(self.embedding.calls, [["alpha beta"]])

    def test_removed_file_vectors_are_not_reused(self) -> None:
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
