"""Assemble retrieval and chat into one persistent QAChat session.

Classes
-------
Evidence
    One traceable source chunk supplied to the chat model.
ConversationTurn
    One saved question, answer, evidence set, and settings snapshot.
RAGSession
    Open a project, reuse or build indexes, answer, log, and resume.

"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from qachat.chat import ChatSession
from qachat.chunking import Chunk, chunk_source_units
from qachat.embeddings import EmbeddingSession
from qachat.project import QAProject, open_project
from qachat.query_formats import (
    LITERATURE_REPOSITORY_INSTRUCTION,
    SOFTWARE_REPOSITORY_INSTRUCTION,
    format_qwen3_query,
)
from qachat.repository import (
    build_repository_manifest,
    compare_repository_manifests,
    scan_repository,
)
from qachat.retrievals import (
    CodeGraph,
    PersistentIndexes,
    build_code_graph,
    build_lexical_index,
    build_semantic_index,
    code_search,
    expand_retrieval_context,
    hybrid_search,
    lexical_search,
    load_indexes,
    plan_retrieval,
    prepare_persistent_indexes,
    save_indexes,
    semantic_search,
)
from qachat.runtime import SelectedBackend, choose_backend
from qachat.sources import extract_source_units, read_repository_files


__all__ = [
    "ConversationTurn",
    "Evidence",
    "RAGSession",
]

_CONVERSATION_VERSION = 1
_INDEX_METADATA_VERSION = 1
_SYSTEM_PROMPT = (
    "Answer only from the supplied project evidence. Cite evidence with "
    "[1], [2], and so on. Say when the evidence is insufficient."
)


@dataclass(frozen=True, slots=True)
class Evidence:
    """One source chunk supplied to the chat model.

    Attributes
    ----------
    citation: int
        One-based citation number used in the answer prompt.
    path: pathlib.Path
        Source path relative to the configured source folder.
    start_line, end_line: int
        One-based source line range.
    symbol: str | None
        Function, class, or section name when available.
    text: str
        Exact retrieved source text.

    """

    citation: int
    path: Path
    start_line: int
    end_line: int
    symbol: str | None
    text: str


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    """One persisted question and grounded answer.

    Attributes
    ----------
    timestamp: str
        UTC time in ISO 8601 form.
    question: str
        Original user question.
    answer: str
        Chat-model response.
    evidence: tuple[Evidence, ...]
        Source chunks supplied for this answer.
    parameters: dict[str, object]
        Exact project and model settings used for this turn.

    """

    timestamp: str
    question: str
    answer: str
    evidence: tuple[Evidence, ...]
    parameters: dict[str, object]


class RAGSession:
    """A loaded project, reusable indexes, models, and conversation.

    Parameters
    ----------
    project_directory: str | pathlib.Path
        QAChat project folder containing profile.ini.

    Attributes
    ----------
    project: QAProject
        Resolved project paths and settings.
    backend: "mlx" | "torch"
        Selected model backend.
    index_status: str
        Either reused or rebuilt for this opening.
    turns: list[ConversationTurn]
        Previous and current persisted interactions.

    """

    project: QAProject
    backend: SelectedBackend
    index_status: str
    turns: list[ConversationTurn]

    def __init__(
        self: RAGSession,
        project_directory: str | Path,
    ) -> None:
        """Open a project, its models, indexes, and saved conversation."""

        self.project = open_project(project_directory)
        profile = self.project.profile
        self.backend = choose_backend(profile.backend)
        embedding_model = profile.embedding_model(self.backend)
        chat_model = profile.chat_model(self.backend)

        embedding_session = EmbeddingSession(
            embedding_model,
            backend=self.backend,
            allow_download=profile.allow_download,
        )
        chat_session: ChatSession | None = None
        try:
            chunks, indexes, status, source_changes = self._prepare_retrieval(
                embedding_session,
            )
            chat_session = ChatSession(
                chat_model,
                backend=self.backend,
                thinking=profile.thinking,
                temperature=profile.temperature,
                max_new_tokens=profile.max_new_tokens,
                system_prompt=_SYSTEM_PROMPT,
                allow_download=profile.allow_download,
            )
            messages, turns = self._load_conversation()
            if messages:
                chat_session.restore_history(messages)
        except Exception:
            if chat_session is not None:
                chat_session.close()
            embedding_session.close()
            raise

        self.embedding_session = embedding_session
        self.chat_session = chat_session
        self.chunks = chunks
        self.indexes = indexes
        self.index_status = status
        self.source_changes = source_changes
        self.turns = turns
        self._closed = False

    def ask(self: RAGSession, question: str) -> str:
        """Retrieve evidence, generate an answer, and save the turn.

        Parameters
        ----------
        question: str
            User question about the opened project.

        Returns
        -------
        str
            Grounded chat-model answer.

        Raises
        ------
        RuntimeError
            If the session is closed.
        ValueError
            If the question is empty.

        """

        if self._closed:
            raise RuntimeError("This RAG session is closed.")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Enter a non-empty question.")
        question = question.strip()

        evidence = self._retrieve(question)
        answer = self.chat_session.reply(_answer_prompt(question, evidence))
        turn = ConversationTurn(
            timestamp=datetime.now(timezone.utc).isoformat(),
            question=question,
            answer=answer,
            evidence=evidence,
            parameters=self._parameters(),
        )
        self.turns.append(turn)
        self._save_conversation()
        return answer

    def reset(self: RAGSession) -> Path | None:
        """Archive the active conversation and clear it from memory.

        Returns
        -------
        pathlib.Path | None
            Renamed conversation log, or None when no active log exists.

        """

        if self._closed:
            raise RuntimeError("This RAG session is closed.")
        archived_path = self._archive_conversation()
        self.chat_session.reset()
        self.turns = []
        return archived_path

    def close(self: RAGSession) -> None:
        """Release both models while keeping indexes and logs on disk."""

        if self._closed:
            return
        self.chat_session.close()
        self.embedding_session.close()
        self._closed = True

    def _prepare_source(
        self: RAGSession,
        embedding_session: EmbeddingSession,
        report: dict[str, object] | None = None,
    ) -> tuple[list[Chunk], CodeGraph]:
        """Read configured source and produce chunks and its code graph."""

        profile = self.project.profile
        if report is None:
            report = scan_repository(
                self.project.source,
                extensions=profile.extensions,
                max_file_bytes=profile.max_file_bytes,
            )
        source_files = read_repository_files(report)
        units = extract_source_units(source_files)
        chunks = chunk_source_units(
            units,
            count_tokens=embedding_session.count_tokens,
            target_tokens=profile.target_tokens,
            max_tokens=profile.max_tokens,
            overlap_tokens=profile.overlap_tokens,
        )
        code_graph = build_code_graph(source_files, units)
        return chunks, code_graph

    def _prepare_retrieval(
        self: RAGSession,
        embedding_session: EmbeddingSession,
    ) -> tuple[
        list[Chunk],
        PersistentIndexes,
        str,
        dict[str, object],
    ]:
        """Reuse indexes when source hashes and index settings still match."""

        profile = self.project.profile
        report = scan_repository(
            self.project.source,
            extensions=profile.extensions,
            max_file_bytes=profile.max_file_bytes,
        )
        current_manifest = build_repository_manifest(report)
        index_path = self.project.indexes / "retrieval.json"
        metadata_path = self.project.indexes / "settings.json"
        manifest_path = self.project.indexes / "source_manifest.json"
        expected_metadata = self._index_metadata()

        saved_manifest: dict[str, object] | None = None
        saved_metadata: dict[str, object] | None = None
        if manifest_path.is_file():
            try:
                value = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    saved_manifest = value
            except (OSError, json.JSONDecodeError):
                saved_manifest = None

        try:
            changes = compare_repository_manifests(
                saved_manifest,
                current_manifest,
            )
        except ValueError:
            saved_manifest = None
            changes = compare_repository_manifests(None, current_manifest)

        if metadata_path.is_file():
            try:
                value = json.loads(metadata_path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    saved_metadata = value
            except (OSError, json.JSONDecodeError):
                saved_metadata = None

        configuration_changed = saved_metadata != expected_metadata
        changes["configuration_changed"] = configuration_changed

        if (
            index_path.is_file()
            and saved_metadata is not None
            and saved_manifest is not None
            and not configuration_changed
            and bool(changes["matches"])
        ):
            try:
                indexes = load_indexes(index_path)
                return (
                    list(indexes.semantic.chunks),
                    indexes,
                    "reused",
                    changes,
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass

        chunks, code_graph = self._prepare_source(
            embedding_session,
            report=report,
        )
        semantic = build_semantic_index(chunks, embedding_session)
        lexical = build_lexical_index(chunks)
        indexes = prepare_persistent_indexes(
            semantic,
            lexical,
            code_graph,
        )
        save_indexes(indexes, index_path)
        _write_json(metadata_path, expected_metadata)
        _write_json(manifest_path, current_manifest)
        return chunks, indexes, "rebuilt", changes

    def _retrieve(self: RAGSession, question: str) -> tuple[Evidence, ...]:
        """Run hybrid retrieval and deterministic context expansion."""

        count = self.project.profile.retrieval_count
        query_text = self._format_query(question)
        semantic = semantic_search(
            self.indexes.semantic,
            query_text,
            self.embedding_session,
            count=count,
        )
        lexical = lexical_search(
            self.indexes.lexical,
            question,
            count=count,
        )
        code = code_search(
            self.indexes.code_graph,
            self.chunks,
            question,
            count=count,
        )
        combined = hybrid_search(
            semantic=semantic,
            lexical=lexical,
            code=code,
            count=count,
        )
        plan = plan_retrieval(question, self.indexes.code_graph)
        expanded = expand_retrieval_context(
            combined,
            self.chunks,
            self.indexes.code_graph,
            plan,
        )
        return tuple(
            Evidence(
                citation=index,
                path=chunk.path,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                symbol=chunk.symbol,
                text=chunk.text,
            )
            for index, chunk in enumerate(expanded, start=1)
        )

    def _format_query(self: RAGSession, question: str) -> str:
        """Apply the embedding model's configured query format."""

        profile = self.project.profile
        if profile.query_format == "plain":
            return question
        instruction = (
            SOFTWARE_REPOSITORY_INSTRUCTION
            if profile.kind == "code"
            else LITERATURE_REPOSITORY_INSTRUCTION
        )
        return format_qwen3_query(question, instruction)

    def _index_metadata(self: RAGSession) -> dict[str, object]:
        """Return settings that determine reusable index compatibility."""

        profile = self.project.profile
        return {
            "format_version": _INDEX_METADATA_VERSION,
            "project_kind": profile.kind,
            "backend": self.backend,
            "embedding_model": profile.embedding_model(self.backend),
            "query_format": profile.query_format,
            "extensions": list(profile.extensions),
            "max_file_bytes": profile.max_file_bytes,
            "target_tokens": profile.target_tokens,
            "max_tokens": profile.max_tokens,
            "overlap_tokens": profile.overlap_tokens,
        }

    def _parameters(self: RAGSession) -> dict[str, object]:
        """Return the complete settings snapshot recorded for one turn."""

        profile = self.project.profile
        return {
            **self._index_metadata(),
            "project_kind": profile.kind,
            "source": str(self.project.source),
            "backend_preference": profile.backend,
            "allow_download": profile.allow_download,
            "chat_model": profile.chat_model(self.backend),
            "retrieval_count": profile.retrieval_count,
            "thinking": profile.thinking,
            "temperature": profile.temperature,
            "max_new_tokens": profile.max_new_tokens,
            "system_prompt": _SYSTEM_PROMPT,
        }

    def _load_conversation(
        self: RAGSession,
    ) -> tuple[list[dict[str, str]], list[ConversationTurn]]:
        """Load saved chat messages and human-readable turns."""

        path = self.project.logs / "conversation.json"
        if not path.is_file():
            return [], []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload["format_version"] != _CONVERSATION_VERSION:
                raise ValueError("Unsupported conversation format version.")
            messages = [
                {
                    "role": str(message["role"]),
                    "content": str(message["content"]),
                }
                for message in payload["chat_messages"]
            ]
            turns = [_turn_from_data(item) for item in payload["turns"]]
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("The saved conversation is invalid.") from error
        return messages, turns

    def _save_conversation(self: RAGSession) -> None:
        """Save exact chat state and inspectable interaction records."""

        payload = {
            "format_version": _CONVERSATION_VERSION,
            "chat_messages": self.chat_session.messages,
            "turns": [_turn_to_data(turn) for turn in self.turns],
        }
        _write_json(
            self.project.logs / "conversation.json",
            payload,
        )

    def _archive_conversation(self: RAGSession) -> Path | None:
        """Rename the active conversation log with a readable UTC time."""

        path = self.project.logs / "conversation.json"
        if not path.is_file():
            return None
        timestamp = datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H-%M-%S-%fZ"
        )
        archived_path = path.with_name(
            f"conversation_{timestamp}.json"
        )
        path.replace(archived_path)
        return archived_path


def _answer_prompt(
    question: str,
    evidence: tuple[Evidence, ...],
) -> str:
    """Combine one question with numbered, traceable evidence."""

    sections = [f"Question:\n{question}", "Evidence:"]
    for item in evidence:
        location = f"{item.path}:{item.start_line}-{item.end_line}"
        symbol = f" ({item.symbol})" if item.symbol else ""
        sections.append(f"[{item.citation}] {location}{symbol}\n{item.text}")
    sections.append(
        "Answer using the evidence above and cite it with bracketed numbers."
    )
    return "\n\n".join(sections)


def _turn_to_data(turn: ConversationTurn) -> dict[str, object]:
    """Convert one conversation turn to JSON-safe data."""

    return {
        "timestamp": turn.timestamp,
        "question": turn.question,
        "answer": turn.answer,
        "evidence": [
            {
                **asdict(item),
                "path": item.path.as_posix(),
            }
            for item in turn.evidence
        ],
        "parameters": turn.parameters,
    }


def _turn_from_data(data: dict[str, object]) -> ConversationTurn:
    """Restore one conversation turn from JSON data."""

    evidence_data = data["evidence"]
    if not isinstance(evidence_data, list):
        raise ValueError("Saved evidence must be a list.")
    evidence = tuple(
        Evidence(
            citation=int(item["citation"]),
            path=Path(str(item["path"])),
            start_line=int(item["start_line"]),
            end_line=int(item["end_line"]),
            symbol=(
                str(item["symbol"]) if item["symbol"] is not None else None
            ),
            text=str(item["text"]),
        )
        for item in evidence_data
    )
    parameters = data["parameters"]
    if not isinstance(parameters, dict):
        raise ValueError("Saved parameters must be a mapping.")
    return ConversationTurn(
        timestamp=str(data["timestamp"]),
        question=str(data["question"]),
        answer=str(data["answer"]),
        evidence=evidence,
        parameters=parameters,
    )


def _write_json(path: Path, data: dict[str, object]) -> None:
    """Write readable JSON atomically."""

    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
