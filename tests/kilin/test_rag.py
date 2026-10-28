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
import warnings

from kilin import rag
from kilin.workspace import (
    list_source_versions,
    prepare_sources_workspace,
    sync_project_sources,
)
from kilin.conversation_format import (
    CONVERSATION_SESSION_KEY,
    new_conversation_session,
)
from kilin.answering import RouteDecision
from kilin.project_type import (
    HISTORY_POLICY_RECENT,
    Intent,
    RETRIEVAL_MODE_DOCUMENT,
    RETRIEVAL_MODE_NONE,
    RETRIEVAL_MODE_REUSE,
    RETRIEVAL_MODE_SEARCH,
)
from kilin.retrievals import DocumentSelection, ContextExpansionPlan
from kilin.retrievals.code_graph import CodeGraph
from kilin.runtime import BACKEND_MLX
from kilin.tools import ToolResult, ToolUse
from kilin.trace_logging import (
    FAILURE_POLICY_RAISE,
    FAILURE_POLICY_WARN,
    FailurePolicy,
    NullTraceLogger,
    TraceConfig,
    TraceLogger,
)
from tests.kilin.retrievals._support import (
    FakeEmbeddings,
    chunk,
    graph,
    ranked,
)


def _profile() -> SimpleNamespace:
    """Return a profile exposing the fields used by the RAG coordinator."""
    return SimpleNamespace(
        type_id="code",
        backend="mlx",
        allow_download=False,
        query_format="plain",
        retrieval_count=5,
        thinking=False,
        temperature=0.2,
        max_new_tokens=128,
        block_unrequested_cjk=True,
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
    directory = root or Path("/unused-kilin-test-project")
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


def _type_config(
    intent: Intent, *, guidance: str = ""
) -> SimpleNamespace:
    """Provide the minimum project-type contract for an answer turn."""
    return SimpleNamespace(
        id="code",
        default_intent=intent.intent_id,
        intents={intent.intent_id: intent},
        recent_turns=2,
        system_prompt="Project instruction",
        code_graph=True,
        answer_guidance=guidance,
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


@contextmanager
def _search_patches():
    """Replace search channels while preserving the real RAG coordinator."""
    with patch.multiple(
        rag,
        semantic_search=DEFAULT,
        lexical_search=DEFAULT,
        code_search=DEFAULT,
        hybrid_search=DEFAULT,
        plan_context_expansion=DEFAULT,
        expand_retrieval_context=DEFAULT,
    ) as mocks:
        for name in (
            "semantic_search", "lexical_search", "code_search",
            "hybrid_search", "expand_retrieval_context",
        ):
            mocks[name].return_value = []
        mocks["plan_context_expansion"].return_value = ContextExpansionPlan(
            (), (), False, (), ()
        )
        yield mocks


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


@contextmanager
def _answer_patches(
    session: rag.RAGSession,
    intent: Intent,
    decision: RouteDecision | None = None,
):
    """Isolate the real answer state machine from models and disk writes."""
    decision = decision or RouteDecision(intent.intent_id, True)
    with patch.object(
        rag, "project_type", return_value=_type_config(intent)
    ), \
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
        # A mocked successful persistence call still commits the pending
        # turn, without touching the filesystem.
        save.side_effect = lambda pending_turn, **options: (
            session.turns.append(pending_turn)
        )
        yield SimpleNamespace(
            route=route, retrieve=retrieve, read=read,
            run=run, save=save,
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
        """Route, retrieve, answer, and save a search-mode turn."""
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
        """Combine related evidence with the selected conversation history."""
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
        """Pass detected document candidates to the retrieval operation."""
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
        """Retain selected-document provenance and answer parameters."""
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
        """Handle missing documents without inventing evidence."""
        session = _session()
        with _answer_patches(
            session, _intent(RETRIEVAL_MODE_DOCUMENT)
        ) as mocks:
            session.ask("Read missing", use_router=False)
        self.assertEqual(session.turns[-1].evidence, ())
        self.assertNotIn("document", session.turns[-1].parameters)
        mocks.read.assert_called_once()

    def test_none_mode_skips_retrieval(self) -> None:
        """Generate an answer without retrieval in no-retrieval mode."""
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
        """Reuse referenced evidence without invoking a new search."""
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
        """Honor explicit reuse of evidence from the previous answer."""
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
        """Search when requested earlier evidence cannot be reused."""
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
        """Give configured intent guidance precedence over caller text."""
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

    def test_project_guidance_is_applied_without_gui(self) -> None:
        """Use project guidance automatically in direct API calls."""
        session = _session()
        intent = _intent()
        with _answer_patches(session, intent):
            with patch.object(
                rag, "project_type",
                return_value=_type_config(intent, guidance="Core instruction"),
            ):
                session.ask("Question", use_router=False)
        prompt = session.chat_session.reply.call_args.args[0]
        self.assertIn("Core instruction", prompt)
        self.assertEqual(
            session.turns[-1].parameters["answer_guidance"],
            "Core instruction",
        )

    def test_caller_guidance_overrides_project_default(self) -> None:
        """Apply explicit caller guidance instead of the config's default."""
        session = _session()
        intent = _intent()
        with _answer_patches(session, intent):
            with patch.object(
                rag, "project_type",
                return_value=_type_config(intent, guidance="Core instruction"),
            ):
                session.ask(
                    "Question", answer_guidance="Caller instruction",
                    use_router=False,
                )
        prompt = session.chat_session.reply.call_args.args[0]
        self.assertIn("Caller instruction", prompt)
        self.assertNotIn("Core instruction", prompt)
        self.assertEqual(
            session.turns[-1].parameters["answer_guidance"],
            "Caller instruction",
        )

    def test_explicit_empty_guidance_disables_project_default(self) -> None:
        """Allow callers to suppress project guidance without a GUI."""
        session = _session()
        intent = _intent()
        with _answer_patches(session, intent):
            with patch.object(
                rag, "project_type",
                return_value=_type_config(intent, guidance="Core instruction"),
            ):
                session.ask(
                    "Question", answer_guidance="", use_router=False
                )
        prompt = session.chat_session.reply.call_args.args[0]
        self.assertNotIn("Core instruction", prompt)
        self.assertEqual(
            session.turns[-1].parameters["answer_guidance"], ""
        )

    def test_intent_guidance_overrides_empty_caller_guidance(self) -> None:
        """Keep intent guidance authoritative even for empty overrides."""
        session = _session()
        intent = _intent(guidance="Intent instruction")
        with _answer_patches(session, intent):
            with patch.object(
                rag, "project_type",
                return_value=_type_config(intent, guidance="Core instruction"),
            ):
                session.ask(
                    "Question", answer_guidance="", use_router=False
                )
        prompt = session.chat_session.reply.call_args.args[0]
        self.assertIn("Intent instruction", prompt)
        self.assertNotIn("Core instruction", prompt)

    def test_tool_receives_resolved_project_guidance(self) -> None:
        """Send the same resolved guidance to tool-written answers."""
        session = _session()
        intent = _intent(tool=ToolUse("test_tool", {}))
        with _answer_patches(session, intent) as mocks:
            with patch.object(
                rag, "project_type",
                return_value=_type_config(intent, guidance="Core instruction"),
            ):
                session.ask("Question", use_router=False)
        self.assertEqual(
            mocks.run.call_args.kwargs["answer_guidance"],
            "Core instruction",
        )
        self.assertEqual(
            session.turns[-1].parameters["answer_guidance"],
            "Core instruction",
        )

    def test_invalid_citations_are_flagged_without_rewriting_answer(
        self,
    ) -> None:
        """Record nonexistent citation IDs without changing model text."""
        session = _session()
        answer = "Supported [1], invented [5, 9], repeated [5]."
        session.chat_session.reply.return_value = answer
        with _answer_patches(session, _intent()):
            self.assertEqual(
                session.ask("Question", use_router=False), answer
            )
        self.assertEqual(session.turns[-1].answer, answer)
        self.assertEqual(
            session.turns[-1].parameters["invalid_citations"], [5, 9]
        )
        session.chat_session.reply.assert_called_once()

    def test_valid_and_missing_citations_need_no_metadata(self) -> None:
        """Do not add warnings when the answer cites only its evidence."""
        for answer in ("Supported [1]", "No numbered citations"):
            with self.subTest(answer=answer):
                session = _session()
                session.chat_session.reply.return_value = answer
                with _answer_patches(session, _intent()):
                    session.ask("Question", use_router=False)
                self.assertNotIn(
                    "invalid_citations", session.turns[-1].parameters
                )

    def test_tool_citations_are_checked_against_tool_evidence(self) -> None:
        """Validate tool citations against their returned evidence IDs."""
        session = _session()
        intent = _intent(tool=ToolUse("test_tool", {}))
        with _answer_patches(session, intent) as mocks:
            mocks.run.return_value = ToolResult(
                "Tool result [2] [99]", (asdict(_evidence(2)),),
                {"name": "test_tool"},
            )
            session.ask("Question", use_router=False)
        self.assertEqual(
            session.turns[-1].parameters["invalid_citations"], [99]
        )

    def test_generation_settings_and_finish_reason_are_recorded(self) -> None:
        """Persist effective generation settings and valid finish reasons."""
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
        """Use the selected tool instead of an answering model."""
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
        """Avoid replaying an answer already streamed by the tool."""
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
        """Avoid appending a turn when generation raises an error."""
        session = _session()
        session.chat_session.reply.side_effect = RuntimeError("model failed")
        with _answer_patches(session, _intent()) as mocks:
            with self.assertRaisesRegex(RuntimeError, "model failed"):
                session.ask("Question", use_router=False)
        self.assertEqual(session.turns, [])
        mocks.save.assert_not_called()

    def test_save_failure_retains_only_committed_turns(self) -> None:
        """Exclude a generated answer when its checkpoint cannot be saved."""
        session = _session()
        with _answer_patches(session, _intent()) as mocks:
            mocks.save.side_effect = OSError("disk full")
            with self.assertRaisesRegex(OSError, "disk full"):
                session.ask("Question", use_router=False)
        self.assertEqual(session.turns, [])
        self.assertIsNone(session._checkpoint_id)
        session.chat_session.restore_history.assert_called()

    def test_phase_callback_exception_propagates(self) -> None:
        """Propagate exceptions raised by the interface phase callback."""
        session = _session()
        def fail(phase: str, details: dict) -> None:
            """Simulate a GUI callback failure."""
            raise RuntimeError("callback failure")
        with _answer_patches(session, _intent()):
            with self.assertRaisesRegex(RuntimeError, "callback failure"):
                session.ask("Question", use_router=False,
                            phase_callback=fail)
        self.assertEqual(session.turns, [])


class TestRAGIntentThinking(unittest.TestCase):
    """Exercise intent policy with real reply orchestration and fake models."""

    def _chat(self, *, thinking: bool) -> rag.ChatSession:
        """Create a chat with real settings and a deterministic tokenizer.

        Parameters
        ----------
        thinking: bool
            Configured setting that intent overrides must not mutate.

        Returns
        -------
        kilin.ChatSession
            Session whose model generation must be patched by the test.

        """

        with patch.object(rag.ChatSession, "_load"):
            chat = rag.ChatSession(
                "chat-model", backend="torch", thinking=thinking,
                max_new_tokens=16, block_unrequested_cjk=False,
            )
        self.addCleanup(chat.close)
        chat._model = object()
        chat._tokenizer = Mock()
        chat._tokenizer.apply_chat_template.return_value = [1, 2, 3]
        chat._tokenizer.encode.return_value = [4]
        return chat

    def test_intent_policy_propagates_without_reconfiguration(self) -> None:
        """Apply generic intent policies without a small-talk special case."""

        for configured in (True, False):
            for override in (None, True, False):
                with self.subTest(configured=configured, override=override):
                    session = _session()
                    session.chat_session.thinking = configured
                    intent = replace(_intent(), thinking=override)
                    with _answer_patches(session, intent) as mocks:
                        session.ask("Question", use_router=False)
                    options = session.chat_session.reply.call_args.kwargs
                    if override is None:
                        self.assertNotIn("thinking", options)
                    else:
                        self.assertIs(options["thinking"], override)
                    expected = (
                        configured if override is None else override
                    )
                    parameters = session.turns[-1].parameters
                    self.assertIs(parameters["thinking"], configured)
                    self.assertIs(parameters["thinking_effective"], expected)
                    self.assertIs(session.chat_session.thinking, configured)
                    configure = session.chat_session.configure_generation
                    configure.assert_not_called()
                    session.chat_session.reply.assert_called_once()
                    mocks.route.assert_called_once()

    def test_intent_override_has_precedence_over_generation_update(
        self,
    ) -> None:
        """Retain explicit session updates while overriding only the answer."""

        for configured, updated, override in (
            (False, True, False), (True, False, True), (False, True, None),
        ):
            with self.subTest(updated=updated, override=override):
                session = _session()
                chat = self._chat(thinking=configured)
                session.chat_session = chat
                intent = replace(_intent(), thinking=override)
                with (
                    _answer_patches(session, intent),
                    patch.object(chat, "_torch_reply", return_value="Answer"),
                ):
                    session.ask(
                        "Question", use_router=False,
                        generation_settings={"thinking": updated},
                    )
                expected = updated if override is None else override
                self.assertIs(chat.thinking, updated)
                self.assertIs(
                    chat._tokenizer.apply_chat_template.call_args.kwargs[
                        "enable_thinking"
                    ], expected,
                )
                parameters = session.turns[-1].parameters
                self.assertIs(parameters["thinking"], updated)
                self.assertIs(parameters["thinking_effective"], expected)

    def test_small_talk_then_normal_answer_uses_configured_thinking(
        self,
    ) -> None:
        """Packaged small talk affects only its answer on the same model."""

        for type_id in ("code", "literature"):
            with self.subTest(type_id=type_id):
                config = rag.project_type(type_id)
                session = _session()
                session.project.profile.type_id = type_id
                chat = self._chat(thinking=True)
                session.chat_session = chat
                small_talk = config.intents["small_talk"]
                normal = config.intents[config.default_intent]
                with patch.object(
                    chat, "_torch_reply", return_value="Answer"
                ) as generate:
                    with (
                        _answer_patches(session, small_talk) as mocks,
                        patch.object(rag, "project_type", return_value=config),
                    ):
                        mocks.route.side_effect = lambda *args: (
                            rag.RAGSession._route(session, *args)
                        )
                        session.ask("Hello")
                    with _answer_patches(session, normal):
                        session.ask("Explain the sources", use_router=False)
                calls = chat._tokenizer.apply_chat_template.call_args_list
                self.assertEqual(
                    [call.kwargs["enable_thinking"] for call in calls],
                    [False, True],
                )
                self.assertEqual(generate.call_count, 2)
                self.assertIs(session.chat_session, chat)
                self.assertTrue(chat.thinking)
                self.assertEqual(
                    [turn.parameters["thinking"] for turn in session.turns],
                    [True, True],
                )
                self.assertEqual(
                    [turn.parameters["thinking_effective"]
                     for turn in session.turns],
                    [False, True],
                )

    def test_failed_small_talk_does_not_disable_following_answer(
        self,
    ) -> None:
        """A failed answer leaves configured thinking available to retries."""

        session = _session()
        chat = self._chat(thinking=True)
        session.chat_session = chat
        with (
            _answer_patches(session, replace(_intent(), thinking=False)),
            patch.object(
                chat, "_torch_reply", side_effect=RuntimeError("failed")
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "failed"):
                session.ask("First", use_router=False)
        self.assertTrue(chat.thinking)
        self.assertEqual(session.turns, [])
        with (
            _answer_patches(session, _intent()),
            patch.object(chat, "_torch_reply", return_value="Answer"),
        ):
            session.ask("Second", use_router=False)
        self.assertTrue(
            chat._tokenizer.apply_chat_template.call_args.kwargs[
                "enable_thinking"
            ]
        )
        self.assertEqual(len(session.turns), 1)

    def test_tool_answer_has_no_direct_model_thinking_metadata(self) -> None:
        """Do not claim one effective policy for tool-owned model calls."""

        session = _session()
        intent = _intent(
            RETRIEVAL_MODE_NONE,
            tool=ToolUse("project_inventory", object()),
        )
        with _answer_patches(session, intent):
            session.ask("Inventory", use_router=False)
        session.chat_session.reply.assert_not_called()
        session.chat_session.configure_generation.assert_not_called()
        self.assertNotIn("thinking_effective", session.turns[-1].parameters)

    def test_saved_small_talk_restores_configured_not_effective_thinking(
        self,
    ) -> None:
        """Round-trip real checkpoints with tracing both enabled and off."""

        for tracing in (False, True):
            with self.subTest(tracing=tracing), TemporaryDirectory() as root:
                session = _session(Path(root))
                chat = self._chat(thinking=True)
                session.chat_session = chat
                trace = (
                    TraceLogger(
                        TraceConfig(
                            enabled=True, output_path=session.conversation_path
                        ),
                        default_directory=Path(root),
                    ) if tracing else NullTraceLogger()
                )
                session.trace = trace
                chat.trace = trace
                try:
                    with (
                        _answer_patches(
                            session, replace(_intent(), thinking=False)
                        ) as mocks,
                        patch.object(
                            chat, "_torch_reply", return_value="Answer"
                        ),
                    ):
                        mocks.save.side_effect = lambda **options: (
                            rag.RAGSession._save_conversation(
                                session, **options
                            )
                        )
                        session.ask("Hello", use_router=False)
                    _, turns, _ = session._load_conversation()
                    self.assertIs(turns[0].parameters["thinking"], True)
                    self.assertIs(
                        turns[0].parameters["thinking_effective"], False
                    )
                    chat.configure_generation({"thinking": False})
                    session.switch_conversation(
                        session.conversation_path, trace=trace
                    )
                    self.assertTrue(chat.thinking)
                    self.assertIs(session.chat_session, chat)
                    self.assertEqual(len(session.turns), 1)
                finally:
                    session.close()
                    trace.close()


class TestRAGLifecycle(unittest.TestCase):
    """Test model ownership, switching, refresh, and token bookkeeping."""

    def test_close_releases_sessions_once(self) -> None:
        """Close model resources exactly once per RAG session."""
        session = _session()
        session.close()
        session.close()
        self.assertTrue(session._closed)
        session.chat_session.close.assert_called_once_with()
        session.embedding_session.close.assert_called_once_with()

    def test_close_releases_distinct_router_and_owned_trace(self) -> None:
        """Release a separate router and owned trace on shutdown."""
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
        """Mark the session closed despite cleanup exceptions."""
        session = _session()
        session.chat_session.close.side_effect = OSError("close failed")
        session.trace = MagicMock()
        session._owns_trace = True
        with self.assertRaisesRegex(OSError, "close failed"):
            session.close()
        self.assertTrue(session._closed)
        session.trace.close.assert_called_once()
        session.embedding_session.close.assert_called_once()

    def test_refresh_sources_replaces_only_retrieval_state(self) -> None:
        """Refresh indexed sources without reloading chat resources."""
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
        """Preserve the existing index after a failed source refresh."""
        session = _session()
        original_index = session.indexes
        project = SimpleNamespace(root=Path("/new"))
        session._prepare_retrieval = Mock(side_effect=OSError("bad scan"))
        with self.assertRaisesRegex(OSError, "bad scan"):
            session.refresh_sources(project)
        self.assertIs(session.project, project)
        self.assertIs(session.indexes, original_index)

    def test_close_attempts_every_resource_after_multiple_failures(
        self,
    ) -> None:
        """Report the first failure after releasing all owned resources."""

        session = _session()
        session._router_session = MagicMock()
        session.chat_session.close.side_effect = OSError("chat failure")
        session._router_session.close.side_effect = ValueError(
            "router failure"
        )
        session.embedding_session.close.side_effect = RuntimeError(
            "embedding failure"
        )
        session.trace = MagicMock()
        session._owns_trace = True
        session.trace.emit.side_effect = OSError("trace event failure")
        session.trace.close.side_effect = OSError("trace close failure")
        owner = MagicMock()
        session._conversation_ownership = owner
        with self.assertRaisesRegex(OSError, "chat failure") as caught:
            session.close()
        self.assertTrue(session._closed)
        self.assertIsNone(session._router_session)
        self.assertEqual(session.chat_session.close.call_count, 1)
        self.assertEqual(session.embedding_session.close.call_count, 1)
        self.assertEqual(session.trace.close.call_count, 1)
        self.assertIsNone(session._conversation_ownership)
        owner.release.assert_called_once_with()
        self.assertIn("router failure", str(caught.exception.__notes__))
        self.assertIn("embedding failure", str(caught.exception.__notes__))
        session.close()
        self.assertEqual(session.chat_session.close.call_count, 1)

    def test_close_releases_direct_ownership_after_trace_failure(
        self,
    ) -> None:
        """Release direct-write ownership even if emitting close fails."""

        session = _session()
        owner = MagicMock()
        session._conversation_ownership = owner
        session.trace = MagicMock()
        session.trace.emit.side_effect = OSError("event write failed")
        with self.assertRaisesRegex(OSError, "event write failed"):
            session.close()
        owner.release.assert_called_once_with()
        session.embedding_session.close.assert_called_once_with()

    def test_switch_chat_model_rejects_closed_or_wrong_backend(self) -> None:
        """Reject unsupported model switches and closed sessions."""
        session = _session()
        session._closed = True
        with self.assertRaisesRegex(RuntimeError, "closed"):
            session.use_chat_model("other")
        session._closed = False
        session.backend = "torch"
        with self.assertRaisesRegex(RuntimeError, "MLX"):
            session.use_chat_model("other")

    def test_switch_chat_model_noop_and_new_session(self) -> None:
        """Reuse the current model or create a replacement as required."""
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
        """Reuse a loaded router when it already matches the chat model."""
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
        """Avoid closing a router still shared with another session."""
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
        """Keep the previous chat model if loading its replacement fails."""
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
        """Preserve shared router ownership after failed model loading."""
        session = _session()
        original = session.chat_session
        session.router_model = original.model_id
        session._new_chat_session = Mock(side_effect=OSError("failed"))
        with self.assertRaises(OSError):
            session.use_chat_model("other")
        self.assertIs(session.chat_session, original)
        original.close.assert_not_called()

    def test_router_model_change_closes_only_separate_session(self) -> None:
        """Close the old router only when it owns a separate session."""
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
        """Share an existing chat model or load a router on demand."""
        session = _session()
        self.assertIs(session._router_chat(), session.chat_session)
        session.router_model = "router"
        router = MagicMock()
        session._new_chat_session = Mock(return_value=router)
        self.assertIs(session._router_chat(), router)
        self.assertIs(session._router_chat(), router)
        session._new_chat_session.assert_called_once_with("router")

    def test_router_reply_delegates_to_side_reply(self) -> None:
        """Use the router session for classification-only replies."""
        session = _session()
        with patch.object(rag, "side_reply", return_value="yes") as side:
            self.assertEqual(session.router_reply("system", "question",
                                                  max_new_tokens=12), "yes")
            self.assertEqual(session._ask_router("s", "q", 9), "yes")
        self.assertEqual(side.call_args_list[0].kwargs,
                         {"max_new_tokens": 12, "language_context": ()})
        self.assertEqual(side.call_args_list[1].kwargs,
                         {"max_new_tokens": 9, "language_context": ()})

    def test_new_chat_session_uses_project_backend_and_trace(self) -> None:
        """Construct new chat models with the project runtime and trace."""
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
        self.assertTrue(
            constructor.call_args.kwargs["block_unrequested_cjk"]
        )

    def test_usage_counters_include_distinct_router_only(self) -> None:
        """Avoid double-counting token usage for a shared router."""
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
        """Tolerate model sessions without optional usage counters."""
        session = _session()
        session._router_session = object()
        self.assertEqual(session._model_sessions(), [session.chat_session])

    def test_index_metadata_and_turn_parameters(self) -> None:
        """Expose index provenance and effective turn parameters."""
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
        """Select the default intent without loading a routing model."""
        session = _session()
        config = _type_config(_intent())
        session._router_chat = Mock()
        report = Mock()
        decision = session._route("Question", config, False, "turn", report)
        self.assertEqual(decision.intent, "test_intent")
        self.assertTrue(decision.standalone)
        self.assertEqual(decision.fallback, "routing off")
        session._router_chat.assert_not_called()
        report.assert_not_called()
        session.turns = [_turn()]
        self.assertFalse(session._route(
            "Followup", config, False, "turn", report
        ).standalone)

    def test_routing_on_records_classification_and_fallback(self) -> None:
        """Trace route decisions and recover from invalid classifications."""
        session = _session()
        config = _type_config(_intent())
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
                "Question", config, True, "turn", report
            ), decision)
        rt.assert_called_once_with(
            sentinel.router, config, session.turns, "Question"
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
        """Follow related-turn links and identify the latest turn."""
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
        """Resolve backward references when explicit links are missing."""
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
        """Reject invalid earlier-turn links and cap the history depth."""
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
        """Pass required context and callbacks to the selected tool."""
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
        """Keep auxiliary tool replies separate from conversation history."""
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
                         {"max_new_tokens": 5,
                          "language_context": ("Question",)})
        self.assertEqual(side.call_args_list[1].kwargs,
                         {"max_new_tokens": 6,
                          "token_callback": tokens,
                          "language_context": ("Question",)})


class TestRAGConversationPersistence(unittest.TestCase):
    """Verify resumable turns, checkpoint identities, and path containment."""

    def setUp(self) -> None:
        """Create isolated state and arrange test-specific fixtures."""
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.session = _session(self.root)
        self.session.conversation_path.parent.mkdir(parents=True)
        self.addCleanup(self._release_conversation)

    def _release_conversation(self) -> None:
        """Release a conversation lock a switch acquired, before cleanup.

        Windows cannot delete the temporary folder while a lock file is
        open.

        """

        ownership = getattr(self.session, "_conversation_ownership", None)
        if ownership is not None:
            ownership.release()

    def test_absent_conversation_has_empty_state(self) -> None:
        """Treat a missing conversation file as a new conversation."""
        self.assertEqual(self.session._load_conversation(), ([], [], None))

    def test_save_and_reload_grouped_conversation_without_trace(self) -> None:
        """Round-trip saved turns and checkpoints without active tracing."""
        session = self.session
        session.turns = [
            _turn("Question", "Answer [1]", (_evidence(),))
        ]
        session._save_conversation(trace_turn_id="turn-1")
        first = session._checkpoint_id
        self.assertTrue(first.startswith("checkpoint_"))
        payload = json.loads(
            session.conversation_path.read_text(encoding="utf-8")
        )
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
        payload = json.loads(
            session.conversation_path.read_text(encoding="utf-8")
        )
        saved = rag.conversation_state(payload)
        self.assertEqual(saved["checkpoint"]["parent_checkpoint_id"],
                         first)
        self.assertNotEqual(saved["checkpoint_id"], first)

    def test_save_conversation_with_trace_uses_same_record(self) -> None:
        """Write resumable state into the enabled trace conversation file."""
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

    def test_failed_untraced_checkpoint_preserves_previous_turns(
        self,
    ) -> None:
        """Keep existing disk and memory state when direct JSON write fails."""
        session = self.session
        session.turns = [_turn("Saved question", "Saved answer")]
        session._save_conversation()
        first = session._checkpoint_id
        before = session.conversation_path.read_bytes()
        pending = _turn("Pending question", "Pending answer")
        with patch.object(
            rag, "_write_json", side_effect=OSError("out of space")
        ):
            with self.assertRaisesRegex(OSError, "out of space"):
                session._save_conversation(pending_turn=pending)
        self.assertEqual(len(session.turns), 1)
        self.assertEqual(session._checkpoint_id, first)
        self.assertEqual(session.conversation_path.read_bytes(), before)
        session._save_conversation(pending_turn=pending)
        state = rag.conversation_state(
            json.loads(session.conversation_path.read_text(encoding="utf-8"))
        )
        self.assertEqual(len(session.turns), 2)
        self.assertEqual(state["checkpoint"]["parent_checkpoint_id"], first)
        self.assertEqual(state["turns"][1]["question"], "Pending question")

    def test_load_canonical_turns_reconstructs_model_history(self) -> None:
        """Restore chat messages and turns from canonical saved records."""
        session = self.session
        data = {"sessions": {CONVERSATION_SESSION_KEY:
                             new_conversation_session({
            "system_prompt": "Saved system prompt",
            "turns": [rag._turn_to_data(_turn())],
            "checkpoint_id": "checkpoint_previous",
        })}}
        session.conversation_path.write_text(
            json.dumps(data), encoding="utf-8"
        )
        messages, turns, checkpoint = session._load_conversation()
        self.assertEqual(messages[0]["content"], "Saved system prompt")
        self.assertEqual([message["role"] for message in messages],
                         ["system", "user", "assistant"])
        self.assertEqual(len(turns), 1)
        self.assertEqual(checkpoint, "checkpoint_previous")

    def test_load_conversation_rejects_corrupt_records(self) -> None:
        """Reject corrupt and obsolete conversation representations."""
        session = self.session
        session.conversation_path.write_text("not JSON", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "invalid"):
            session._load_conversation()
        corrupt = {"sessions": {CONVERSATION_SESSION_KEY:
                                 new_conversation_session({
            "turns": [{"timestamp": "x", "evidence": "bad"}],
        })}}
        session.conversation_path.write_text(
            json.dumps(corrupt), encoding="utf-8"
        )
        with self.assertRaises(ValueError):
            session._load_conversation()
        obsolete = {"sessions": {CONVERSATION_SESSION_KEY:
                                  new_conversation_session({
            "turns": [], "chat_messages": [{"role": "user"}],
        })}}
        session.conversation_path.write_text(
            json.dumps(obsolete), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "invalid"):
            session._load_conversation()
        for record in ({"conversation": {"turns": []}}, [], None):
            session.conversation_path.write_text(
                json.dumps(record), encoding="utf-8"
            )
            with self.subTest(record=record):
                with self.assertRaisesRegex(ValueError, "invalid"):
                    session._load_conversation()

    def test_load_conversation_ignores_invalid_checkpoint(self) -> None:
        """Ignore invalid checkpoint identifiers while restoring turns."""
        session = self.session
        session.conversation_path.write_text(json.dumps({
            "sessions": {CONVERSATION_SESSION_KEY:
                         new_conversation_session({
                "turns": [], "checkpoint_id": 100,
            })},
        }), encoding="utf-8")
        self.assertIsNone(session._load_conversation()[2])

    def test_save_preserves_existing_unrelated_sessions(self) -> None:
        """Keep unrelated trace sessions when saving without tracing."""
        session = self.session
        session.conversation_path.write_text(json.dumps({
            "sessions": {"custom": {"title": "Keep"}},
        }), encoding="utf-8")
        session._save_conversation()
        payload = json.loads(
            session.conversation_path.read_text(encoding="utf-8")
        )
        self.assertEqual(payload["sessions"]["custom"],
                         {"title": "Keep"})

    def test_untraced_save_does_not_convert_obsolete_state(self) -> None:
        """Refuse to overwrite obsolete conversation data implicitly."""
        session = self.session
        session.conversation_path.write_text(json.dumps({
            "conversation": {"turns": [{"question": "Do not replace"}]},
        }), encoding="utf-8")
        before = session.conversation_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "invalid"):
            session._save_conversation()
        self.assertEqual(session.conversation_path.read_bytes(), before)

    def test_untraced_save_preserves_invalid_grouped_state(self) -> None:
        """Leave invalid grouped records unchanged after save failure."""
        session = self.session
        bad_group = new_conversation_session({
            "turns": [], "checkpoint_id": "old",
        })
        bad_group["conversation"] = {"turns": [{"question": "Old"}]}
        session.conversation_path.write_text(json.dumps({
            "sessions": {CONVERSATION_SESSION_KEY: bad_group},
        }), encoding="utf-8")
        before = session.conversation_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "invalid"):
            session._save_conversation()
        self.assertEqual(session.conversation_path.read_bytes(), before)

    def test_write_json_is_atomic_and_replaces_existing_data(self) -> None:
        """Replace JSON atomically without leftover temporary files."""
        path = self.root / "payload.json"
        rag._write_json(path, {"a": 1})
        rag._write_json(path, {"a": 2})
        self.assertEqual(
            json.loads(path.read_text(encoding="utf-8")), {"a": 2}
        )
        self.assertFalse((self.root / "payload.json.tmp").exists())

    def test_resolve_conversation_path_validation(self) -> None:
        """Enforce project containment and supported conversation paths."""
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
        """Allocate distinct conversation names when none are specified."""
        session = self.session
        first = session._resolve_conversation_path(None)
        first.write_text("{}", encoding="utf-8")
        second = session._resolve_conversation_path(None)
        self.assertNotEqual(first, second)
        self.assertTrue(second.name.startswith("conversation_"))
        self.assertTrue(second.parent.is_dir())

    def test_resolve_rejects_symlink_escape(self) -> None:
        """Reject conversation paths escaping through symbolic links."""
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
        """Archive an existing conversation without destroying its file."""
        session = self.session
        self.assertIsNone(session._archive_conversation())
        session.conversation_path.write_text("original", encoding="utf-8")
        archive = session._archive_conversation()
        self.assertTrue(archive.is_file())
        self.assertEqual(archive.read_text(encoding="utf-8"), "original")
        self.assertFalse(session.conversation_path.exists())
        self.assertTrue(archive.name.startswith("conversation_"))

    def test_reset_archives_turns_and_preserves_models(self) -> None:
        """Reset conversation state while retaining loaded model sessions."""
        session = self.session
        session.trace = MagicMock()
        session.turns = [_turn()]
        session._checkpoint_id = "old-checkpoint"
        session.conversation_path.write_text("old", encoding="utf-8")
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
        """Restore another conversation without reloading model resources."""
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
        }), encoding="utf-8")
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
        """Use profile generation defaults for a conversation with no turns."""
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
        """Reject switching after close or with an unrelated trace."""
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
        """Bind newly created RAG traces to the conversation JSON path."""
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
        self.assertTrue(enabled.include_content)
        self.assertTrue(enabled.include_token_ids)
        enabled.close()
        configured, owned = rag._conversation_trace(
            TraceConfig(enabled=True),
            path.with_name("conversation_configured.json"),
        )
        self.assertTrue(owned)
        self.assertTrue(configured.include_content)
        self.assertTrue(configured.include_token_ids)
        configured.close()
        with self.assertRaisesRegex(ValueError, "conversation path"):
            rag._conversation_trace(
                TraceConfig(enabled=True, output_path=self.root / "other"),
                path,
            )

    def test_trace_sink_must_match_exact_path_when_enabled(self) -> None:
        """Reject enabled trace sinks pointing outside the conversation."""
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
        """Ignore malformed cached source text and retain valid records."""
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
        }), encoding="utf-8")
        self.assertEqual(list(rag._read_text_cache(indexes)), ["good"])
        cache.write_text("not JSON", encoding="utf-8")
        self.assertEqual(rag._read_text_cache(indexes), {})
        cache.write_text("[]", encoding="utf-8")
        self.assertEqual(rag._read_text_cache(indexes), {})


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
            str(item.path): item.text for item in chunks
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


class TestRAGTraceIntegration(unittest.TestCase):
    """Verify RAG turn outcomes in real grouped conversation-trace files."""

    def setUp(self) -> None:
        """Isolate conversation files without loading any model or index."""
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.session = _session(self.root)
        self.session.conversation_path.parent.mkdir(parents=True)

    def _bind_trace(
        self,
        *,
        owned: bool = False,
        failure_policy: FailurePolicy = FAILURE_POLICY_WARN,
    ) -> TraceLogger:
        """Connect a real logger to the RAG conversation's exact path."""
        trace = TraceLogger(
            TraceConfig(
                enabled=True,
                output_path=self.session.conversation_path,
                failure_policy=failure_policy,
            ),
            default_directory=self.root,
        )
        self.addCleanup(trace.close)
        self.session.trace = trace
        self.session._owns_trace = owned
        return trace

    def _payload(self, path: Path | None = None) -> dict[str, object]:
        """Read the single persisted trace and conversation document."""
        selected = path or self.session.conversation_path
        return json.loads(selected.read_text(encoding="utf-8"))

    def _conversation(self) -> dict[str, object]:
        """Return the readable conversation session from the active file."""
        return self._payload()["sessions"][CONVERSATION_SESSION_KEY]

    def _ask_with_real_save(
        self,
        question: str,
        intent: Intent,
        *,
        phase_callback: rag.AnswerPhaseCallback | None = None,
    ) -> str:
        """Run the real turn state machine and real checkpoint writer."""
        with _answer_patches(self.session, intent) as mocks:
            mocks.save.side_effect = lambda **options: (
                rag.RAGSession._save_conversation(self.session, **options)
            )
            return self.session.ask(
                question,
                use_router=False,
                phase_callback=phase_callback,
            )

    def test_full_trace_persists_metadata_and_checkpoint(
        self,
    ) -> None:
        """Keep complete diagnostics and resumable state in one file."""
        trace = self._bind_trace()
        answer = self._ask_with_real_save("Sensitive question", _intent())
        trace.close()
        self.assertEqual(answer, "model answer")
        payload = self._payload()
        self.assertEqual(payload["event_count"], 13)
        session = self._conversation()
        self.assertEqual(len(session["turns"]), 1)
        turn = session["turns"]["turn_0001"]
        self.assertEqual(turn["title"], "Turn 1: Sensitive question")
        self.assertEqual(
            turn["stages"]["03_retrieval"]["operations"][
                "02_retrieval_result"
            ]["details"]["evidence_count"],
            1,
        )
        checkpoint = turn["stages"]["05_generation"]["operations"][
            "01_state_checkpoint_committed"
        ]["details"]
        self.assertEqual(
            checkpoint["checkpoint_id"], session["resume_state"][
                "checkpoint_id"
            ]
        )
        self.assertEqual(checkpoint["turn_count"], 1)
        self.assertEqual(
            checkpoint["conversation_path"], str(trace.output_path)
        )
        self.assertEqual(
            session["resume_state"]["turns"][0]["question"],
            "Sensitive question",
        )
        question_ops = turn["stages"]["01_question"]["operations"]
        self.assertEqual(
            question_ops["01_turn"]["details"]["question"],
            "Sensitive question",
        )
        self.assertEqual(
            question_ops["06_conversation_turn_representation"][
                "details"
            ]["question"],
            "Sensitive question",
        )
        self.assertFalse(trace.output_path.with_suffix(".json.tmp").exists())

    def test_trace_keeps_question_and_related_turn_operations(
        self,
    ) -> None:
        """Keep exact question text and every phase in the same turn."""
        self._bind_trace()
        self._ask_with_real_save("Exact forensic question", _intent())
        turn = self._conversation()["turns"]["turn_0001"]
        self.assertEqual(turn["title"], "Turn 1: Exact forensic question")
        question = turn["stages"]["01_question"]["operations"]
        self.assertEqual(
            question["01_turn"]["details"]["question"],
            "Exact forensic question",
        )
        self.assertEqual(question["01_turn"]["event"]["status"],
                         "completed")
        self.assertIn("03_retrieval", turn["stages"])
        self.assertIn("05_generation", turn["stages"])
        self.assertEqual(self._payload()["progress_checkpoint_count"], 0)

    def test_answer_without_retrieval_is_traced_and_saved(self) -> None:
        """Record explicit no-retrieval decisions and a valid checkpoint."""
        self._bind_trace()
        self._ask_with_real_save("Hello", _intent(RETRIEVAL_MODE_NONE))
        turn = self._conversation()["turns"]["turn_0001"]
        retrieval = turn["stages"]["03_retrieval"]["operations"]
        self.assertEqual(
            retrieval["02_retrieval_result"]["details"]["mode"],
            RETRIEVAL_MODE_NONE,
        )
        self.assertEqual(
            retrieval["02_retrieval_result"]["details"]["evidence_count"],
            0,
        )
        self.assertEqual(
            len(self._conversation()["resume_state"]["turns"]), 1
        )

    def test_generation_failure_is_logged_without_committing_a_turn(
        self,
    ) -> None:
        """Persist a failed turn span but no successful checkpoint event."""
        self._bind_trace()
        self.session.chat_session.reply.side_effect = RuntimeError(
            "model unavailable"
        )
        with self.assertRaisesRegex(RuntimeError, "model unavailable"):
            self._ask_with_real_save("Failing question", _intent())
        self.assertEqual(self.session.turns, [])
        turn = self._conversation()["turns"]["turn_0001"]
        summary = turn["stages"]["01_question"]["operations"][
            "01_turn"
        ]
        self.assertEqual(summary["event"]["status"], "failed")
        self.assertEqual(summary["result"]["error_type"], "RuntimeError")
        self.assertIn("model unavailable", summary["result"]["traceback"])
        self.assertNotIn("05_generation", turn["stages"])
        self.assertEqual(self._conversation()["resume_state"], {})

    def test_retrieval_failure_is_logged_before_generation(self) -> None:
        """Report retrieval errors without falsely recording model output."""
        self._bind_trace()
        with _answer_patches(self.session, _intent()) as mocks:
            mocks.retrieve.side_effect = OSError("retrieval failed")
            with self.assertRaisesRegex(OSError, "retrieval failed"):
                self.session.ask("Search failure", use_router=False)
        turn = self._conversation()["turns"]["turn_0001"]
        self.assertEqual(
            turn["stages"]["01_question"]["operations"][
                "01_turn"
            ]["event"]["status"],
            "failed",
        )
        self.assertNotIn("05_generation", turn["stages"])
        self.assertEqual(len(self.session.turns), 0)

    def test_two_turns_preserve_order_and_checkpoint_parent(self) -> None:
        """Maintain distinct trace turns with linked resumable checkpoints."""
        self._bind_trace()
        self._ask_with_real_save("First question", _intent())
        earlier = self.session._checkpoint_id
        self._ask_with_real_save("Second question", _intent())
        session = self._conversation()
        self.assertEqual(list(session["turns"]), [
            "turn_0001", "turn_0002"
        ])
        self.assertEqual(len(session["resume_state"]["turns"]), 2)
        checkpoint = session["turns"]["turn_0002"]["stages"][
            "05_generation"
        ]["operations"]["01_state_checkpoint_committed"]["details"]
        self.assertEqual(checkpoint["parent_checkpoint_id"], earlier)
        self.assertEqual(checkpoint["turn_count"], 2)
        self.assertNotEqual(checkpoint["checkpoint_id"], earlier)

    def test_reset_archives_previous_trace_without_retaining_old_turns(
        self,
    ) -> None:
        """Keep archived evidence intact while starting a fresh transcript."""
        self._bind_trace()
        self._ask_with_real_save("Before reset", _intent())
        prior_checkpoint = self.session._checkpoint_id
        archived = self.session.reset()
        self.assertIsNotNone(archived)
        self.assertTrue(archived.is_file())
        archived_session = self._payload(archived)["sessions"][
            CONVERSATION_SESSION_KEY
        ]
        self.assertEqual(len(archived_session["turns"]), 1)
        self.assertEqual(
            archived_session["resume_state"]["checkpoint_id"],
            prior_checkpoint,
        )
        active = self._conversation()
        self.assertEqual(active["turns"], {})
        self.assertEqual(active["resume_state"], {})
        self._ask_with_real_save("After reset", _intent())
        self.assertEqual(list(self._conversation()["turns"]), ["turn_0001"])
        self.assertEqual(self.session._checkpoint_id,
                         self._conversation()["resume_state"]["checkpoint_id"])

    def test_switch_conversation_keeps_trace_and_resume_paths_separate(
        self,
    ) -> None:
        """Close the owned old logger before switching to a new trace."""
        self._bind_trace(owned=True)
        self._ask_with_real_save("Old conversation", _intent())
        old_path = self.session.conversation_path
        new_path = old_path.with_name("conversation_second.json")
        next_trace = TraceLogger(
            TraceConfig(enabled=True, output_path=new_path),
            default_directory=self.root,
        )
        self.addCleanup(next_trace.close)
        restored = self.session.switch_conversation(
            new_path, trace=next_trace
        )
        self.assertEqual(restored, [])
        old_trace_events = self._payload(old_path)["sessions"][
            "01_trace"
        ]["operations"]
        self.assertIn("02_trace_closed", old_trace_events)
        self.assertEqual(
            self.session.conversation_path, new_path.resolve()
        )
        self._ask_with_real_save("New conversation", _intent())
        self.assertEqual(
            self._payload(old_path)["sessions"][CONVERSATION_SESSION_KEY][
                "resume_state"
            ]["turns"][0]["question"],
            "Old conversation",
        )
        self.assertEqual(
            self._payload(new_path)["sessions"][CONVERSATION_SESSION_KEY][
                "resume_state"
            ]["turns"][0]["question"],
            "New conversation",
        )
        self.assertEqual(len(self._payload(old_path)["sessions"][
            CONVERSATION_SESSION_KEY
        ]["turns"]), 1)
        self.assertEqual(len(self._payload(new_path)["sessions"][
            CONVERSATION_SESSION_KEY
        ]["turns"]), 1)

    def test_conversation_write_failure_records_no_committed_checkpoint(
        self,
    ) -> None:
        """Report a failed save without claiming the checkpoint succeeded."""
        trace = self._bind_trace()
        with patch.object(
            trace, "update_conversation", side_effect=OSError("no space")
        ):
            with self.assertRaisesRegex(OSError, "no space"):
                self._ask_with_real_save("Save failure", _intent())
        self.assertEqual(self.session.turns, [])
        self.assertIsNone(self.session._checkpoint_id)
        self.session.chat_session.restore_history.assert_called()
        turn = self._conversation()["turns"]["turn_0001"]
        self.assertEqual(
            turn["stages"]["01_question"]["operations"][
                "01_turn"
            ]["event"]["status"],
            "failed",
        )
        self.assertNotIn("05_generation", turn["stages"])
        self.assertEqual(self._conversation()["resume_state"], {})

    def test_failed_checkpoint_restores_committed_history_and_retries(
        self,
    ) -> None:
        """Exclude a failed turn from history, then link the next save."""
        trace = self._bind_trace()
        chat = self.session.chat_session

        def restore(messages: list[dict[str, str]]) -> None:
            """Make the model fake retain restored message history."""
            chat.messages = [dict(message) for message in messages]

        def reply(prompt: str, **options: object) -> str:
            """Simulate the real chat model adding generated messages."""
            chat.messages.extend([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "model answer"},
            ])
            return "model answer"

        chat.restore_history.side_effect = restore
        chat.reply.side_effect = reply
        self._ask_with_real_save("First saved", _intent())
        first = self.session._checkpoint_id
        messages = [dict(message) for message in chat.messages]
        original_writer = trace._write_readable_trace

        def fail_pending_state() -> None:
            """Refuse only the attempted checkpoint write."""
            if trace._conversation.get("checkpoint_id") != first:
                raise OSError("checkpoint unavailable")
            original_writer()

        with patch.object(
            trace, "_write_readable_trace", side_effect=fail_pending_state
        ):
            with self.assertRaisesRegex(OSError, "checkpoint unavailable"):
                self._ask_with_real_save("Failed to save", _intent())
        self.assertEqual(len(self.session.turns), 1)
        self.assertEqual(self.session._checkpoint_id, first)
        self.assertEqual(chat.messages, messages)
        self.assertEqual(trace._conversation["checkpoint_id"], first)
        saved = self._conversation()["resume_state"]
        self.assertEqual(saved["checkpoint_id"], first)
        self.assertEqual(len(saved["turns"]), 1)
        self.assertEqual(
            self._conversation()["turns"]["turn_0002"]["stages"][
                "01_question"
            ]["operations"]["01_turn"]["event"]["status"],
            "failed",
        )
        self._ask_with_real_save("Saved after retry", _intent())
        saved = self._conversation()["resume_state"]
        self.assertEqual(len(self.session.turns), 2)
        self.assertEqual(len(saved["turns"]), 2)
        self.assertEqual(
            saved["checkpoint"]["parent_checkpoint_id"], first
        )
        self.assertEqual(
            [turn["question"] for turn in saved["turns"]],
            ["First saved", "Saved after retry"],
        )

    def test_postcommit_trace_error_does_not_roll_back_checkpoint(
        self,
    ) -> None:
        """Keep a saved turn if its subsequent diagnostic event fails."""
        trace = self._bind_trace()
        original_emit = trace.emit

        def fail_event(event_type: str, **options: object) -> None:
            """Fail only the event emitted after checkpoint persistence."""
            if event_type == "state.checkpoint.committed":
                raise RuntimeError("diagnostic unavailable")
            original_emit(event_type, **options)

        with patch.object(trace, "emit", side_effect=fail_event):
            with self.assertRaisesRegex(
                RuntimeError, "diagnostic unavailable"
            ):
                self._ask_with_real_save("Saved despite trace", _intent())
        saved = self._conversation()["resume_state"]
        self.assertEqual(len(self.session.turns), 1)
        self.assertEqual(self.session._checkpoint_id,
                         saved["checkpoint_id"])
        self.assertEqual(
            saved["turns"][0]["question"], "Saved despite trace"
        )
        self.assertEqual(
            self._conversation()["turns"]["turn_0001"]["stages"][
                "01_question"
            ]["operations"]["01_turn"]["event"]["status"],
            "failed",
        )

    def test_checkpoint_event_write_warning_keeps_committed_answer(
        self,
    ) -> None:
        """Return the answer when optional post-save trace I/O fails."""
        trace = self._bind_trace(failure_policy=FAILURE_POLICY_WARN)
        write = trace._write_readable_trace

        def write_or_fail() -> None:
            """Refuse the checkpoint event after its state is durable."""
            if trace._events[-1]["event_type"] == (
                "state.checkpoint.committed"
            ):
                raise OSError("diagnostic write failed")
            write()

        with patch.object(
            trace, "_write_readable_trace", side_effect=write_or_fail
        ), warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", RuntimeWarning)
            answer = self._ask_with_real_save(
                "Saved despite warning", _intent()
            )
        self.assertEqual(answer, "model answer")
        self.assertEqual(len(self.session.turns), 1)
        self.assertTrue(any(
            "diagnostic write failed" in str(item.message)
            for item in captured
        ))
        self.assertEqual(
            self._conversation()["resume_state"]["checkpoint_id"],
            self.session._checkpoint_id,
        )

    def test_checkpoint_event_write_strict_failure_keeps_state(
        self,
    ) -> None:
        """Raise for strict diagnostics without undoing the checkpoint."""
        trace = self._bind_trace(failure_policy=FAILURE_POLICY_RAISE)
        write = trace._write_readable_trace

        def write_or_fail() -> None:
            """Refuse the checkpoint event but permit the state write."""
            if trace._events[-1]["event_type"] == (
                "state.checkpoint.committed"
            ):
                raise OSError("strict diagnostic failure")
            write()

        with patch.object(
            trace, "_write_readable_trace", side_effect=write_or_fail
        ):
            with self.assertRaisesRegex(
                RuntimeError, "Could not write KíLín trace"
            ):
                self._ask_with_real_save(
                    "Saved despite strict failure", _intent()
                )
        self.assertEqual(len(self.session.turns), 1)
        self.assertEqual(
            self._conversation()["resume_state"]["checkpoint_id"],
            self.session._checkpoint_id,
        )

    def test_intermediate_callback_failure_remains_a_failed_turn(
        self,
    ) -> None:
        """Propagate an observer error before the checkpoint is saved."""
        self._bind_trace()
        observed_phases: list[str] = []

        def phase_callback(
            phase: str, details: dict[str, object]
        ) -> None:
            """Interrupt the answer during evidence retrieval."""
            observed_phases.append(phase)
            if phase == rag.ANSWER_PHASE_RETRIEVING:
                raise RuntimeError("early notification failed")

        with self.assertRaisesRegex(
            RuntimeError, "early notification failed"
        ):
            self._ask_with_real_save(
                "Interrupted answer", _intent(),
                phase_callback=phase_callback,
            )
        self.assertIn(rag.ANSWER_PHASE_RETRIEVING, observed_phases)
        self.assertNotIn(rag.ANSWER_PHASE_COMPLETED, observed_phases)
        self.assertEqual(self.session.turns, [])
        self.assertEqual(self._conversation()["resume_state"], {})
        self.assertEqual(
            self._conversation()["turns"]["turn_0001"]["stages"][
                "01_question"
            ]["operations"]["01_turn"]["event"]["status"],
            "failed",
        )

    def test_completed_callback_sees_committed_and_completed_turn(
        self,
    ) -> None:
        """Notify completion after saving and closing the successful span."""
        self._bind_trace()
        completed_state: list[tuple[str, int, object]] = []

        def phase_callback(
            phase: str, details: dict[str, object]
        ) -> None:
            """Inspect the stored trace at the completion notification."""
            if phase == rag.ANSWER_PHASE_COMPLETED:
                turn = self._conversation()["turns"]["turn_0001"]
                turn_status = (
                    turn["stages"]["01_question"]["operations"][
                        "01_turn"
                    ]["event"]["status"]
                )
                completed_state.append((
                    turn_status,
                    len(self._conversation()["resume_state"]["turns"]),
                    details["evidence_count"],
                ))

        answer = self._ask_with_real_save(
            "Completed notification", _intent(),
            phase_callback=phase_callback,
        )
        self.assertEqual(answer, "model answer")
        self.assertEqual(completed_state, [("completed", 1, 1)])

    def test_late_callback_failure_returns_committed_answer(self) -> None:
        """Record an observer error without marking the saved turn failed."""
        self._bind_trace()

        def phase_callback(
            phase: str, details: dict[str, object]
        ) -> None:
            """Fail only after the conversation checkpoint is committed."""
            if phase == rag.ANSWER_PHASE_COMPLETED:
                raise RuntimeError("UI callback failed")

        answer = self._ask_with_real_save(
            "Callback failure", _intent(),
            phase_callback=phase_callback,
        )
        self.assertEqual(answer, "model answer")
        turn = self._conversation()["turns"]["turn_0001"]
        operations = turn["stages"]["01_question"]["operations"]
        self.assertEqual(
            operations["01_turn"]["event"]["status"], "completed"
        )
        saved = turn["stages"]["05_generation"]["operations"][
            "01_state_checkpoint_committed"
        ]
        self.assertEqual(saved["event"]["status"], "completed")
        self.assertEqual(
            len(self._conversation()["resume_state"]["turns"]), 1
        )
        self.assertEqual(len(self.session.turns), 1)
        notification = [
            operation for key, operation in operations.items()
            if key.endswith("turn_notification_failed")
        ]
        self.assertEqual(len(notification), 1)
        self.assertEqual(notification[0]["event"]["status"], "failed")
        self.assertEqual(
            notification[0]["details"]["phase"],
            rag.ANSWER_PHASE_COMPLETED,
        )
        self.assertEqual(
            notification[0]["details"]["error_type"], "RuntimeError"
        )
        self.assertEqual(
            notification[0]["details"]["message"], "UI callback failed"
        )
        self.assertIn(
            "UI callback failed", notification[0]["details"]["traceback"]
        )

        first_checkpoint = self.session._checkpoint_id
        self._ask_with_real_save("Second answer", _intent())
        self.assertEqual(len(self.session.turns), 2)
        self.assertEqual(
            self._conversation()["resume_state"]["checkpoint"][
                "parent_checkpoint_id"
            ],
            first_checkpoint,
        )

    def test_late_callback_failure_without_tracing_returns_answer(
        self,
    ) -> None:
        """Allow a completed turn to succeed with tracing disabled."""
        phases: list[str] = []

        def phase_callback(
            phase: str, details: dict[str, object]
        ) -> None:
            """Record the phase and fail only on final notification."""
            phases.append(phase)
            if phase == rag.ANSWER_PHASE_COMPLETED:
                raise RuntimeError("optional notification unavailable")

        answer = self._ask_with_real_save(
            "Untraced callback failure", _intent(),
            phase_callback=phase_callback,
        )
        self.assertEqual(answer, "model answer")
        self.assertEqual(phases[-1], rag.ANSWER_PHASE_COMPLETED)
        self.assertEqual(len(self.session.turns), 1)
        saved = self._conversation()["resume_state"]["turns"]
        self.assertEqual(saved[0]["question"], "Untraced callback failure")

    def test_tool_answer_survives_final_callback_failure(self) -> None:
        """Keep a tool-written answer when the final observer fails."""
        self._bind_trace()
        intent = _intent(
            RETRIEVAL_MODE_NONE,
            tool=ToolUse("project_inventory", object()),
        )

        def phase_callback(
            phase: str, details: dict[str, object]
        ) -> None:
            """Fail only the post-checkpoint presentation notification."""
            if phase == rag.ANSWER_PHASE_COMPLETED:
                raise RuntimeError("tool view disconnected")

        answer = self._ask_with_real_save(
            "Tool output", intent, phase_callback=phase_callback
        )
        self.assertEqual(answer, "tool answer")
        self.session.chat_session.reply.assert_not_called()
        self.assertEqual(
            self._conversation()["resume_state"]["turns"][0]["answer"],
            "tool answer",
        )
        turn = self._conversation()["turns"]["turn_0001"]
        self.assertEqual(
            turn["stages"]["01_question"]["operations"][
                "01_turn"
            ]["event"]["status"],
            "completed",
        )

    def test_notification_trace_write_warning_keeps_saved_answer(
        self,
    ) -> None:
        """Ignore optional failed-event I/O under warn policy."""
        trace = self._bind_trace(failure_policy=FAILURE_POLICY_WARN)
        write = trace._write_readable_trace

        def write_or_fail() -> None:
            """Simulate a diagnostic write failure after the checkpoint."""
            if trace._events[-1]["event_type"] == (
                "turn.notification.failed"
            ):
                raise OSError("trace disk unavailable")
            write()

        def fail_when_completed(
            phase: str, details: dict[str, object]
        ) -> None:
            """Raise an ordinary presentation exception after commit."""
            if phase == rag.ANSWER_PHASE_COMPLETED:
                raise RuntimeError("observer failed")

        with patch.object(
            trace, "_write_readable_trace", side_effect=write_or_fail
        ), warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", RuntimeWarning)
            answer = self._ask_with_real_save(
                "Warn on failed event", _intent(),
                phase_callback=fail_when_completed,
            )
        self.assertEqual(answer, "model answer")
        self.assertTrue(any(
            "trace disk unavailable" in str(item.message)
            for item in captured
        ))
        self.assertEqual(len(self.session.turns), 1)
        self.assertEqual(
            self._conversation()["resume_state"]["checkpoint_id"],
            self.session._checkpoint_id,
        )
        self.assertEqual(
            self._conversation()["turns"]["turn_0001"]["stages"][
                "01_question"
            ]["operations"]["01_turn"]["event"]["status"],
            "completed",
        )

    def test_notification_trace_write_strict_policy_propagates(
        self,
    ) -> None:
        """Preserve strict trace errors after a committed RAG answer."""
        trace = self._bind_trace(failure_policy=FAILURE_POLICY_RAISE)
        write = trace._write_readable_trace

        def write_or_fail() -> None:
            """Refuse the notification-error diagnostic event only."""
            if trace._events[-1]["event_type"] == (
                "turn.notification.failed"
            ):
                raise OSError("trace disk unavailable")
            write()

        def fail_when_completed(
            phase: str, details: dict[str, object]
        ) -> None:
            """Raise after commit to trigger strict diagnostic writing."""
            if phase == rag.ANSWER_PHASE_COMPLETED:
                raise RuntimeError("observer failed")

        with patch.object(
            trace, "_write_readable_trace", side_effect=write_or_fail
        ):
            with self.assertRaisesRegex(
                RuntimeError, "Could not write KíLín trace"
            ):
                self._ask_with_real_save(
                    "Strict failed event", _intent(),
                    phase_callback=fail_when_completed,
                )
        self.assertEqual(len(self.session.turns), 1)
        self.assertEqual(
            self._conversation()["resume_state"]["checkpoint_id"],
            self.session._checkpoint_id,
        )
        self.assertEqual(
            self._conversation()["turns"]["turn_0001"]["stages"][
                "01_question"
            ]["operations"]["01_turn"]["event"]["status"],
            "completed",
        )

    def test_tool_answer_commits_trace_without_chat_reply(self) -> None:
        """Persist tool-produced turns without invoking the chat model."""
        self._bind_trace()
        intent = _intent(
            RETRIEVAL_MODE_NONE,
            tool=ToolUse("project_inventory", object()),
        )
        answer = self._ask_with_real_save("Inventory", intent)
        self.assertEqual(answer, "tool answer")
        self.session.chat_session.reply.assert_not_called()
        turn = self._conversation()["turns"]["turn_0001"]
        self.assertIn("05_generation", turn["stages"])
        self.assertEqual(
            self._conversation()["resume_state"]["turns"][0]["answer"],
            "tool answer",
        )


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
