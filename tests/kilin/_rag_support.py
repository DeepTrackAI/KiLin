"""Reusable RAG session, evidence, and orchestration test fixtures.

These helpers intentionally avoid model downloads and disk writes.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import DEFAULT, MagicMock, patch, sentinel
from kilin import rag
from kilin.answering import RouteDecision
from kilin.project_type import (
    HISTORY_POLICY_RECENT,
    Intent,
    RETRIEVAL_MODE_SEARCH,
)
from kilin.retrievals import ContextExpansionPlan
from kilin.retrievals.code_graph import CodeGraph
from kilin.runtime import BACKEND_MLX
from kilin.tools import ToolResult
from kilin.trace_logging import NullTraceLogger


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
        exclude_disliked_answers=True,
        system_prompt="Project instruction",
        code_graph=True,
        answer_guidance=guidance,
    )


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
