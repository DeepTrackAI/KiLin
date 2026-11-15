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

Examples
--------
>>> from kilin.conversation_memory.inspection import inspect_memory_messages

"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from kilin.answering import strip_thinking
from kilin.conversation_format import conversation_state
from kilin.conversation_memory.feedback import FeedbackStore
from kilin.conversation_memory.index import _eligible, _manifest
from kilin.conversation_memory.models import require_saved_turn_ids
from kilin.project_type import ProjectType

__all__ = ["MemoryMessagePage", "inspect_memory_messages"]


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
