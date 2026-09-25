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

from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
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
    "ProgressCallback",
    "ProgressEvent",
    "RAGSession",
]

_CONVERSATION_VERSION = 1
_INDEX_METADATA_VERSION = 1
_EMBEDDING_PROGRESS_BATCH_SIZE = 64
_PROGRESS_MIN_INTERVAL_SECONDS = 0.15
_SYSTEM_PROMPT = (
    "Answer only from the supplied project evidence. Put source numbers such "
    "as [1] and [2] directly after supported claims. Do not write a separate "
    "reference list or copy bibliographic text. Say when evidence "
    "is insufficient."
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
class ProgressEvent:
    """One structured progress update emitted while opening a RAG session.

    Attributes
    ----------
    stage: str
        Stable machine-readable stage name for notebook or GUI clients.
    percent: float
        Approximate overall session-preparation progress from 0 to 100.
    message: str
        Short human-readable description of the current work.
    current, total: int | None
        Optional item counts when a stage has a meaningful count.
    remaining: int | None
        Number of items still remaining in the current counted stage.
    elapsed_seconds: float | None
        Elapsed time for the current counted stage.
    eta_seconds: float | None
        Estimated seconds remaining for the current counted stage. This is an
        estimate based on the average throughput observed so far.

    """

    stage: str
    percent: float
    message: str
    current: int | None = None
    total: int | None = None
    remaining: int | None = None
    elapsed_seconds: float | None = None
    eta_seconds: float | None = None


ProgressCallback = Callable[[ProgressEvent], None]


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
    progress_callback: collections.abc.Callable[[ProgressEvent], None] | None
        Optional UI-agnostic callback that receives structured progress events
        while the session checks or builds indexes and loads models.

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
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        """Open a project, its models, indexes, and saved conversation."""

        if progress_callback is not None and not callable(progress_callback):
            raise ValueError("progress_callback must be callable or None.")

        self.progress_callback = progress_callback
        self._progress_last_emit: dict[str, float] = {}
        self._emit_progress(
            "opening_project",
            0.0,
            "Opening QAChat project...",
        )
        self.project = open_project(project_directory)
        profile = self.project.profile
        self.backend = choose_backend(profile.backend)
        embedding_model = profile.embedding_model(self.backend)
        chat_model = profile.chat_model(self.backend)

        self._emit_progress(
            "loading_embedding_model",
            5.0,
            f"Loading embedding model: {embedding_model}",
        )
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
            self._emit_progress(
                "loading_chat_model",
                90.0,
                f"Loading chat model: {chat_model}",
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
            self._emit_progress(
                "loading_conversation",
                97.0,
                "Loading saved conversation...",
            )
            messages, turns = self._load_conversation()
            if messages:
                chat_session.restore_history(messages)
            if turns:
                previous = turns[-1].parameters
                chat_session.configure_generation(
                    {
                        key: previous[key]
                        for key in (
                            "temperature", "thinking", "max_new_tokens"
                        )
                        if key in previous
                    }
                )
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
        self._emit_progress(
            "ready",
            100.0,
            f"QAChat ready. Index {self.index_status}.",
        )

    def _emit_progress(
        self: RAGSession,
        stage: str,
        percent: float,
        message: str,
        *,
        current: int | None = None,
        total: int | None = None,
        remaining: int | None = None,
        elapsed_seconds: float | None = None,
        eta_seconds: float | None = None,
    ) -> None:
        """Send one structured progress event when a callback is configured."""

        callback = getattr(self, "progress_callback", None)
        if callback is None:
            return
        callback(
            ProgressEvent(
                stage=stage,
                percent=float(percent),
                message=message,
                current=current,
                total=total,
                remaining=remaining,
                elapsed_seconds=elapsed_seconds,
                eta_seconds=eta_seconds,
            )
        )

    def _emit_count_progress(
        self: RAGSession,
        stage: str,
        start_percent: float,
        end_percent: float,
        label: str,
        current: int,
        total: int,
        started_at: float,
    ) -> None:
        """Emit throttled counted progress with elapsed time and ETA."""

        if total < 0 or current < 0 or current > total:
            raise ValueError(
                "Progress counts must satisfy 0 <= current <= total."
            )

        now = perf_counter()
        last = self._progress_last_emit.get(stage, 0.0)
        if (
            current not in {0, total}
            and now - last < _PROGRESS_MIN_INTERVAL_SECONDS
        ):
            return
        self._progress_last_emit[stage] = now

        fraction = 1.0 if total == 0 else current / total
        percent = start_percent + (end_percent - start_percent) * fraction
        remaining = max(0, total - current)
        elapsed = max(0.0, now - started_at)
        eta: float | None = None
        if current > 0 and elapsed > 0.0:
            eta = 0.0 if remaining == 0 else elapsed * remaining / current

        self._emit_progress(
            stage,
            percent,
            label,
            current=current,
            total=total,
            remaining=remaining,
            elapsed_seconds=elapsed,
            eta_seconds=eta,
        )

    def ask(
        self: RAGSession,
        question: str,
        *,
        retrieval_count: int | None = None,
        generation_settings: dict[str, object] | None = None,
    ) -> str:
        """Retrieve evidence, generate an answer, and save the turn.

        Parameters
        ----------
        question: str
            User question about the opened project.
        retrieval_count: int | None
            Optional evidence count for this question only.
        generation_settings: dict[str, object] | None
            Optional temperature, thinking, and answer length for this and
            later questions, until changed again.

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
        if retrieval_count is not None and (
            type(retrieval_count) is not int or retrieval_count < 1
        ):
            raise ValueError(
                "retrieval_count must be a positive whole number."
            )
        question = question.strip()

        if generation_settings is not None:
            self.chat_session.configure_generation(generation_settings)

        evidence = self._retrieve(question, retrieval_count)
        answer = self.chat_session.reply(_answer_prompt(question, evidence))
        turn = ConversationTurn(
            timestamp=datetime.now(timezone.utc).isoformat(),
            question=question,
            answer=answer,
            evidence=evidence,
            parameters=self._parameters(retrieval_count),
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
        selected_count = len(report.get("selected_files", []))

        reading_started = perf_counter()
        self._emit_count_progress(
            "reading_source", 35.0, 42.0, "Reading source files",
            0, selected_count, reading_started,
        )
        source_files = read_repository_files(
            report,
            progress_callback=lambda current, total: self._emit_count_progress(
                "reading_source",
                35.0,
                42.0,
                "Reading source files",
                current,
                total,
                reading_started,
            ),
        )

        extracting_started = perf_counter()
        source_file_count = len(source_files)
        self._emit_count_progress(
            "extracting_source", 43.0, 49.0, "Extracting source structure",
            0, source_file_count, extracting_started,
        )
        units = extract_source_units(
            source_files,
            progress_callback=lambda current, total: self._emit_count_progress(
                "extracting_source",
                43.0,
                49.0,
                "Extracting source structure",
                current,
                total,
                extracting_started,
            ),
        )

        chunking_started = perf_counter()
        unit_count = len(units)
        self._emit_count_progress(
            "chunking_source", 50.0, 55.0, "Chunking source units",
            0, unit_count, chunking_started,
        )
        chunks = chunk_source_units(
            units,
            count_tokens=embedding_session.count_tokens,
            target_tokens=profile.target_tokens,
            max_tokens=profile.max_tokens,
            overlap_tokens=profile.overlap_tokens,
            progress_callback=lambda current, total: self._emit_count_progress(
                "chunking_source",
                50.0,
                55.0,
                "Chunking source units",
                current,
                total,
                chunking_started,
            ),
        )
        self._emit_progress(
            "building_code_graph",
            56.0,
            f"Building code graph from {len(chunks)} chunks...",
            current=len(chunks),
            total=len(chunks),
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
        self._emit_progress(
            "scanning_source",
            15.0,
            "Scanning selected project files...",
        )
        report = scan_repository(
            self.project.source,
            extensions=profile.extensions,
            max_file_bytes=profile.max_file_bytes,
        )
        selected_count = len(report.get("selected_files", []))
        self._emit_progress(
            "checking_index",
            22.0,
            f"Checking {selected_count} files against the saved index...",
            current=selected_count,
            total=selected_count,
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
        self._emit_progress(
            "validating_index",
            28.0,
            "Validating saved index metadata and source manifest...",
        )

        if (
            index_path.is_file()
            and saved_metadata is not None
            and saved_manifest is not None
            and not configuration_changed
            and bool(changes["matches"])
        ):
            try:
                self._emit_progress(
                    "loading_index",
                    32.0,
                    "Source unchanged; loading the saved retrieval index...",
                )
                indexes = load_indexes(index_path)
                self._emit_progress(
                    "index_ready",
                    85.0,
                    "Saved retrieval index loaded.",
                )
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
        embedding_started = perf_counter()
        chunk_count = len(chunks)
        self._emit_count_progress(
            "building_semantic_index",
            58.0,
            74.0,
            "Embedding chunks",
            0,
            chunk_count,
            embedding_started,
        )
        semantic = build_semantic_index(
            chunks,
            embedding_session,
            batch_size=_EMBEDDING_PROGRESS_BATCH_SIZE,
            progress_callback=lambda current, total: self._emit_count_progress(
                "building_semantic_index",
                58.0,
                74.0,
                "Embedding chunks",
                current,
                total,
                embedding_started,
            ),
        )
        self._emit_progress(
            "building_lexical_index",
            75.0,
            (
                "Semantic index complete; building lexical index for "
                f"{len(chunks)} chunks..."
            ),
            current=len(chunks),
            total=len(chunks),
        )
        lexical = build_lexical_index(chunks)
        indexes = prepare_persistent_indexes(
            semantic,
            lexical,
            code_graph,
        )
        self._emit_progress(
            "saving_index",
            82.0,
            "Saving retrieval index and manifests...",
        )
        save_indexes(indexes, index_path)
        _write_json(metadata_path, expected_metadata)
        _write_json(manifest_path, current_manifest)
        self._emit_progress(
            "index_ready",
            85.0,
            "Retrieval index saved.",
        )
        return chunks, indexes, "rebuilt", changes

    def _retrieve(
        self: RAGSession,
        question: str,
        retrieval_count: int | None = None,
    ) -> tuple[Evidence, ...]:
        """Run hybrid retrieval and deterministic context expansion."""

        count = retrieval_count or self.project.profile.retrieval_count
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

    def _parameters(
        self: RAGSession, retrieval_count: int | None = None
    ) -> dict[str, object]:
        """Return the complete settings snapshot recorded for one turn."""

        profile = self.project.profile
        return {
            **self._index_metadata(),
            "project_kind": profile.kind,
            "source": str(self.project.source),
            "backend_preference": profile.backend,
            "allow_download": profile.allow_download,
            "chat_model": profile.chat_model(self.backend),
            "retrieval_count": retrieval_count or profile.retrieval_count,
            "thinking": self.chat_session.thinking,
            "temperature": self.chat_session.temperature,
            "max_new_tokens": self.chat_session.max_new_tokens,
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
        "Answer using the evidence above. Put [1], [2], and so on immediately "
        "after the claims they support. Do not repeat the source text or "
        "write a bibliography."
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
