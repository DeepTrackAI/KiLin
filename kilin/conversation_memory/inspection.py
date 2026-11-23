"""Inspect active conversation messages without embedding or indexing.

The settings interface needs to show excluded messages even after they have
been removed from the derived memory index. This read-only core service
therefore inspects active conversation checkpoints and authoritative feedback,
applying the same policy as indexing without changing any saved file.

Key Features
------------

- **Complete, paged inventory**

    Inspect eligible and excluded questions and answers, including branches.

- **Explicit policy status**

    Distinguish user exclusion, Dislike, and effective indexing eligibility.

Module Structure
----------------

- ``inspect_memory_messages``

    Read and page project-owned messages for a headless API or GUI.

- ``read_memory_summaries``

    Fetch current full answer summaries from registered conversation traces.

Examples
--------
>>> from kilin.conversation_memory.inspection import inspect_memory_messages

"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from kilin.answering import strip_thinking
from kilin.conversation_format import (
    CONVERSATION_SESSION_KEY,
    conversation_state,
)
from kilin.conversation_memory.feedback import FeedbackStore
from kilin.conversation_memory.index import _eligible, _manifest
from kilin.conversation_memory.models import (
    require_saved_turn_ids,
    require_turn_uuid,
)
from kilin.conversations import ConversationStore
from kilin.project_type import ProjectType

__all__ = [
    "MemoryMessagePage", "inspect_memory_messages", "read_memory_summaries",
]


@dataclass(frozen=True, slots=True)
class MemoryMessagePage:
    """An authoritative read-only page of saved message metadata.

    Attributes
    ----------
    revision: str
        Optimistic revision for feedback changes.
    total: int
        Total distinct active question and answer messages.
    messages: tuple[dict[str, object], ...]
        Current page in descending historical timestamp order.

    """

    revision: str
    total: int
    messages: tuple[dict[str, object], ...]


def inspect_memory_messages(
    project_directory: str | Path,
    config: ProjectType,
    *,
    offset: int = 0,
    limit: int = 100,
) -> MemoryMessagePage:
    """Inspect recent active messages, including those explicitly excluded.

    Parameters
    ----------
    project_directory: str | pathlib.Path
        Owning workspace, not its source folder.
    config: ProjectType
        Effective validated project/intent memory policy.
    offset: int, optional
        Number of records to skip, default 0.
    limit: int, optional
        Page size, default 100; bounded between 1 and 200.

    Returns
    -------
    MemoryMessagePage
        Feedback revision and a page of unmodified message previews.

    Raises
    ------
    ValueError
        If pagination or an active checkpoint is malformed.

    Notes
    -----
    The inventory does not tell the model about excluded content. An
    ``eligible`` status means the next synchronization would index it;
    this inspection never requests embeddings or rebuilds indexes.

    """

    if (type(offset) is not int or offset < 0
            or type(limit) is not int or not 1 <= limit <= 200):
        raise ValueError("Memory pagination values are invalid.")
    root = Path(project_directory).resolve()
    feedback = FeedbackStore(root).load()
    seen: dict[tuple[str, str], dict[str, object]] = {}
    originals: dict[tuple[str, str], tuple[str, str]] = {}
    for name, _, _ in _manifest(root):
        path = root / "logs" / "traces" / name
        record = conversation_state(json.loads(
            path.read_text(encoding="utf-8")
        ))
        conversation_id = record.get("conversation_id", path.stem)
        display_name = record.get("name", conversation_id)
        if (not isinstance(conversation_id, str) or not conversation_id
                or not isinstance(display_name, str)):
            raise ValueError("The saved conversation identity is invalid.")
        turns = record.get("turns", [])
        identities = require_saved_turn_ids(turns)
        for number, (turn, identifier) in enumerate(
            zip(turns, identities), start=1
        ):
            params = turn.get("parameters", {})
            timestamp = turn.get("timestamp")
            if (not isinstance(params, dict)
                    or not isinstance(timestamp, str) or not timestamp):
                raise ValueError("The saved memory turn is invalid.")
            for role, field in (("user", "question"),
                                ("assistant", "answer")):
                if role == "assistant" and params.get("completion"):
                    continue
                text = turn.get(field)
                if not isinstance(text, str):
                    raise ValueError("The saved memory text is invalid.")
                text = (
                    strip_thinking(text) if role == "assistant" else text
                ).strip()
                if not text:
                    continue
                key = (identifier, role)
                if key in seen:
                    if originals[key] != (text, timestamp):
                        raise ValueError(
                            "Branches disagree on an inherited memory turn."
                        )
                    continue
                originals[key] = (text, timestamp)
                rating = feedback.get(identifier, role)
                eligible, _ = _eligible(
                    role, params.get("intent"), config, feedback,
                    identifier,
                )
                state = (
                    "excluded" if rating.excluded else
                    "disliked" if rating.reaction == "disliked" else
                    "eligible" if config.conversation_memory.enabled
                    and eligible else "policy"
                )
                seen[key] = {
                    "turn_uuid": identifier,
                    "role": role,
                    "preview": " ".join(text.split())[:180],
                    "timestamp": timestamp,
                    "conversation_id": conversation_id,
                    "conversation_name": display_name,
                    "turn_number": number,
                    "status": state,
                    "saved": rating.saved,
                    "reaction": rating.reaction,
                    "excluded": rating.excluded,
                }
    ordered = sorted(
        seen.values(),
        key=lambda item: (
            item["timestamp"], item["turn_uuid"], item["role"]
        ),
        reverse=True,
    )
    return MemoryMessagePage(
        feedback.revision, len(ordered),
        tuple(ordered[offset:offset + limit]),
    )


def _trace_answer_summary(
    payload: dict[str, object], number: int
) -> str | None:
    """Read the current, potentially edited summary from a trace turn.

    Parameters
    ----------
    payload: dict[str, object]
        Entire conversation trace, including grouped diagnostic operations.
    number: int
        One-based position of the requested saved turn.

    Returns
    -------
    str | None
        The complete trace summary (including a deliberately emptied one),
        or None if the operation is absent (for example, on a branch).

    """

    sessions = payload.get("sessions", {})
    if not isinstance(sessions, dict):
        return None
    session = sessions.get(CONVERSATION_SESSION_KEY, {})
    if not isinstance(session, dict):
        return None
    turns = session.get("turns", {})
    if not isinstance(turns, dict):
        return None
    trace_turn = turns.get(f"turn_{number:04d}", {})
    if not isinstance(trace_turn, dict):
        return None
    stages = trace_turn.get("stages", {})
    if not isinstance(stages, dict):
        return None
    question = stages.get("01_question", {})
    if not isinstance(question, dict):
        return None
    operations = question.get("operations", {})
    if not isinstance(operations, dict):
        return None
    for key, operation in reversed(list(operations.items())):
        if not key.endswith("_conversation_turn_representation"):
            continue
        details = operation.get("details", {}) if isinstance(
            operation, dict
        ) else {}
        if isinstance(details, dict) and isinstance(
            details.get("summary"), str
        ):
            return details["summary"]
    return None


def read_memory_summaries(
    store: ConversationStore, references: object,
) -> list[dict[str, str]]:
    """Read current memory labels directly from active conversation traces.

    Parameters
    ----------
    store: ConversationStore
        Store whose already registered conversations constrain trace access.
    references: object
        Bounded list of turn UUIDs, message roles, and active origin IDs with
        one-based turn numbers. Origins are checked against saved UUIDs.

    Returns
    -------
    list[dict[str, str]]
        One entry per accessible message, with ``turn_uuid``, ``role`` and
        untruncated ``text``. Question text comes from the checkpoint;
        answer summaries prefer the trace's representation operation.

    Raises
    ------
    ValueError
        If the request has an unsupported structure or message identity.

    Notes
    -----
    A trace is read on every invocation, without embedding or model calls.
    Inherited branch checkpoints can lack the original diagnostic operation;
    the saved compact representation then provides a legacy fallback.
    No summary is written to message provenance or another cache.

    """

    if not isinstance(references, list) or len(references) > 100:
        raise ValueError("Invalid memory summary request.")
    available = {item.conversation_id for item in store.conversations()}
    snapshots: dict[str, dict[str, object] | None] = {}
    results: list[dict[str, str]] = []
    for reference in references:
        if not isinstance(reference, dict) or set(reference) != {
            "turn_uuid", "role", "origins",
        }:
            raise ValueError("Invalid memory summary reference.")
        identifier = require_turn_uuid(reference["turn_uuid"])
        role = reference["role"]
        origins = reference["origins"]
        if (role not in ("user", "assistant")
                or not isinstance(origins, list) or len(origins) > 20):
            raise ValueError("Invalid memory summary reference.")
        fallback = ""
        current: str | None = None
        for origin in origins:
            if not isinstance(origin, dict) or set(origin) != {
                "conversation_id", "turn_number",
            }:
                raise ValueError("Invalid memory summary origin.")
            name = origin["conversation_id"]
            number = origin["turn_number"]
            if (not isinstance(name, str) or not name
                    or type(number) is not int or number < 1):
                raise ValueError("Invalid memory summary origin.")
            if name not in available:
                continue
            if name not in snapshots:
                path = store.state_path(name)
                if path.is_symlink() or not path.is_file():
                    snapshots[name] = None
                else:
                    snapshots[name] = json.loads(
                        path.read_text(encoding="utf-8")
                    )
            payload = snapshots[name]
            if not isinstance(payload, dict):
                continue
            state = conversation_state(payload)
            turns = state.get("turns", [])
            ids = require_saved_turn_ids(turns)
            if number > len(ids) or ids[number - 1] != identifier:
                continue
            turn = turns[number - 1]
            if role == "user":
                question = turn.get("question")
                if isinstance(question, str):
                    current = question
                    break
            else:
                current = _trace_answer_summary(payload, number)
                if current is not None:
                    break
                params = turn.get("parameters", {})
                representation = (
                    params.get("representation", {})
                    if isinstance(params, dict) else {}
                )
                if isinstance(representation, dict):
                    value = representation.get("summary")
                    if isinstance(value, str) and value:
                        fallback = value
        text = fallback if current is None else current
        if text:
            results.append({
                "turn_uuid": identifier, "role": role, "text": text,
            })
    return results
