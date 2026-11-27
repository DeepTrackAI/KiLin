"""Compare recorded context and tool outcomes without changing execution.

Key Features
------------
- Match conversation memory and history by recorded turn identities.
- Distinguish functional tool results from elapsed-time measurements.
- Normalize only corroborated memory headers in structured model inputs.
- Retain independently observable answers when alignment is unavailable.

Module Structure
----------------
- ``context_comparison`` measures context selection by mapped identity.
- ``structured_prompt`` prepares a narrowly normalized prompt observation.
- ``tool_comparison`` separates timing, functional results and coverage.
- ``independent_questions`` exposes saved-question candidates and trust.

Examples
--------
>>> from kilin.system_testing import Experiment
>>> # report = Experiment("010_original").compare().data

"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime
import json
import re

from .timeline import TimelineOperation
from .traces import Recording, TraceEvent
from .validation import ValidationReport

__all__ = []

_CONTEXT = ("history.selection.result", "memory.sync.result",
            "project.memory.sync.result", "conversation_memory.retrieved")
_HEADER = re.compile(
    r"Memory (M[1-9]\d*) \| (user|assistant) \| UUID (\S+) \| "
    r"([^\n|]+) \| (.+)"
)
_MEMORY_HEADING = (
    "CONVERSATION MEMORY (previous discussion, NOT source evidence):"
)
_TIMING = {"seconds", "duration_ms", "elapsed_ms"}
_FAILURES = {"incomplete", "partial", "error", "failed", "cancelled",
             "aborted", "interrupted"}


def _key(value: object) -> str:
    """Serialize one observation for stable, order-independent sorting.

    Parameters
    ----------
    value: object
        JSON-compatible recorded observation.

    """
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _mapped(value: str, identities: dict[str, str]) -> str:
    """Require a known correspondence instead of erasing an unknown ID.

    Parameters
    ----------
    value: str
        Recorded identity.
    identities: dict[str, str]
        Known identities mapped into the reference namespace.

    """
    if value not in identities:
        raise ValueError(f"Unmapped recorded identity: {value}")
    return identities[value]


def _memory_record(record: dict, turns: dict[str, str],
                   conversations: dict[str, str]) -> dict:
    """Normalize one memory identity, keeping role and all branch origins.

    Parameters
    ----------
    record: dict
        Recorded inventory entry or selected memory hit.
    turns: dict[str, str]
        Turn identity map into the reference namespace.
    conversations: dict[str, str]
        Conversation identity map into the reference namespace.

    """
    origins = [{"conversation_id": _mapped(o["conversation_id"],
                                           conversations),
                "turn_number": o["turn_number"]}
               for o in record["origins"]]
    return {"turn_uuid": _mapped(record["turn_uuid"], turns),
            "role": record["role"], "origins": sorted(origins, key=_key)}


def _context_payload(event: TraceEvent, op: TimelineOperation,
                     catalog: dict[str, list[str]], turns: dict[str, str],
                     conversations: dict[str, str]) -> dict:
    """Read identity-level context while keeping content metrics separate.

    Parameters
    ----------
    event: TraceEvent
        Recorded history, memory inventory or memory retrieval event.
    op: TimelineOperation
        Enclosing validated operation.
    catalog: dict[str, list[str]]
        Final validated conversation turn prefixes, including archives.
    turns: dict[str, str]
        Map into the reference turn namespace.
    conversations: dict[str, str]
        Map into the reference conversation namespace.

    """
    data = event.details
    kind = event.event_type
    if kind == "history.selection.result":
        prefix = catalog[op.request["conversation_id"]]
        saved = op.matching("gui.question.saved")
        if len(saved) != 1:
            raise ValueError("Cannot bound history to a saved question.")
        count = saved[0].details["turn_count"] - 1
        result = {}
        for field in ("recent_turn_ids", "related_turn_ids",
                      "final_turn_ids"):
            ids = data[field]
            if not isinstance(ids, list) or any(
                    type(i) is not int or not 1 <= i <= count or
                    i > len(prefix) for i in ids):
                raise ValueError(f"Invalid history selection: {field}")
            result[field] = [_mapped(prefix[i - 1], turns) for i in ids]
        return {"identity": result}
    if kind == "conversation_memory.retrieved":
        entries = data["selected_memories"]
        return {"identity": [_memory_record(r, turns, conversations)
                             for r in entries],
                "historical_lookup": data.get("historical_lookup"),
                "scores": [{k: r.get(k) for k in ("relevance", "score")}
                           for r in entries]}
    entries = data.get("records")
    if not isinstance(entries, list):
        raise ValueError("Memory inventory was not recorded.")
    records = []
    for entry in entries:
        identity = _memory_record(entry, turns, conversations)
        records.append({"identity": identity,
                        "fingerprint": entry.get("fingerprint"),
                        "priority": entry.get("priority")})
    records.sort(key=lambda r: _key(r["identity"]))
    return {"identity": [r["identity"] for r in records],
            "fingerprints": [r["fingerprint"] for r in records],
            "priorities": [r["priority"] for r in records],
            "enabled": data.get("enabled")}


def context_comparison(original: TimelineOperation,
                       replayed: TimelineOperation, catalogs: tuple,
                       maps: tuple) -> list[dict]:
    """Compare each context observation by event type and occurrence.

    Parameters
    ----------
    original: TimelineOperation
        Original command.
    replayed: TimelineOperation
        Aligned replay command.
    catalogs: tuple
        Original and replayed conversation-to-turn-prefix maps.
    maps: tuple
        Original-to-replay conversation and turn identity maps.

    Returns
    -------
    list[dict]
        Mapped observations, explicit missing coverage, ordered selection
        agreement and separate content/priority comparisons. Empty recorded
        selections are distinct from unavailable observations.

    """
    conversations, turns = maps
    namespaces = (({k: k for k in turns}, {k: k for k in conversations}),
                  ({v: k for k, v in turns.items()},
                   {v: k for k, v in conversations.items()}))
    rows = []
    for kind in _CONTEXT:
        events = (original.matching(kind), replayed.matching(kind))
        for i in range(max(map(len, events))):
            row = {"event_type": kind, "occurrence": i + 1, "issues": []}
            for side, group, op, catalog, namespace in zip(
                    ("original", "replayed"), events, (original, replayed),
                    catalogs, namespaces, strict=True):
                row[side] = None
                if i < len(group):
                    try:
                        row[side] = _context_payload(
                            group[i], op, catalog, *namespace)
                    except (KeyError, TypeError, ValueError,
                            AttributeError) as error:
                        row["issues"].append(f"{side}: {error}")
            a, b = row["original"], row["replayed"]
            row["identity_equal"] = (a["identity"] == b["identity"]
                                     if a is not None and b is not None
                                     else None)
            if (kind == "history.selection.result" and
                    a is not None and b is not None):
                row["selection_equal"] = {
                    key: a["identity"][key] == b["identity"][key]
                    for key in a["identity"]
                }
            for field in ("fingerprints", "priorities", "historical_lookup",
                          "enabled", "scores"):
                if (a is not None and b is not None and
                        field in a and field in b):
                    row[field + "_equal"] = (a[field] == b[field]
                                               if row["identity_equal"]
                                               else None)
            rows.append(row)
    return rows


def _header(record: dict, line: str, turns: dict[str, str],
            conversations: dict[str, str]) -> str:
    """Normalize a memory header only if its recorded identity agrees.

    Parameters
    ----------
    record: dict
        Selected memory observation corroborating this header.
    line: str
        Exact structured-message line.
    turns: dict[str, str]
        Turn map into the reference namespace.
    conversations: dict[str, str]
        Conversation map into the reference namespace.

    """
    match = _HEADER.fullmatch(line)
    if not match:
        raise ValueError("Memory header does not match its recorded format.")
    label, role, uid, stamp, origins = match.groups()
    # Parse rather than masking arbitrary text in the timestamp position.
    if datetime.fromisoformat(stamp).utcoffset() is None:
        raise ValueError("Memory timestamp has no timezone.")
    expected = ", ".join(f"{o['conversation_id']} turn {o['turn_number']}"
                         for o in record["origins"])
    if (label, role, uid, origins) != (
            record["label"], record["role"], record["turn_uuid"], expected):
        raise ValueError("Memory header does not match selected memory.")
    identity = _memory_record(record, turns, conversations)
    return f"Memory {label} | {_key(identity)} | <recorded timestamp>"


def structured_prompt(event: TraceEvent | None, op: TimelineOperation,
                      turns: dict[str, str],
                      conversations: dict[str, str]) -> dict | None:
    """Measure exact structured messages and corroborated header metadata.

    Parameters
    ----------
    event: TraceEvent or None
        LLM input event, or unavailable.
    op: TimelineOperation
        Enclosing operation with selected memory observations.
    turns: dict[str, str]
        Turn map into the reference namespace.
    conversations: dict[str, str]
        Conversation map into the reference namespace.

    Returns
    -------
    dict or None
        Exact and normalized JSON strings, plus changed header count. Only
        the final user message's explicitly recorded memory block is eligible.
        Unknown or ambiguous blocks yield unavailable normalization.

    """
    if event is None:
        return None
    artifact = event.exact_content.get("01_structured_messages", {})
    messages = artifact.get("value")
    if not isinstance(messages, list) or not all(
            isinstance(m, dict) and isinstance(m.get("content"), str)
            for m in messages):
        return None
    result = {"raw": _key(messages), "normalized": _key(messages),
              "headers": 0, "issue": None}
    # Only the final user message is supplied by the memory formatter.
    if not messages or messages[-1].get("role") != "user":
        return result
    text = messages[-1]["content"]
    if _MEMORY_HEADING not in text:
        return result
    try:
        if text.count(_MEMORY_HEADING) != 1:
            raise ValueError("Ambiguous conversation memory block.")
        prior = [e for e in op.matching("conversation_memory.retrieved")
                 if e.sequence < event.sequence]
        if not prior:
            raise ValueError("Missing preceding selected-memory record.")
        hits = prior[-1].details["selected_memories"]
        prefix, block = text.split(_MEMORY_HEADING, 1)
        lines = block.split("\n")
        indices = [i for i, line in enumerate(lines)
                   if line.startswith("Memory M")]
        if not hits or len(indices) != len(hits):
            raise ValueError("Memory header coverage differs from selection.")
        for i, record in zip(indices, hits, strict=True):
            lines[i] = _header(record, lines[i], turns, conversations)
        normalized = deepcopy(messages)
        normalized[-1]["content"] = (prefix + _MEMORY_HEADING +
                                      "\n".join(lines))
        result.update(normalized=_key(normalized), headers=len(indices))
    except (KeyError, ValueError, TypeError, AttributeError) as error:
        result.update(normalized=None, issue=str(error))
    return result


def _functional_tool(value: object) -> tuple[object, dict]:
    """Separate known tool timing fields, not arbitrary numeric results.

    Parameters
    ----------
    value: object
        Raw saved tool summary or absent value.

    """
    if not isinstance(value, dict):
        return value, {}
    functional = deepcopy(value)
    timing = {key: functional.pop(key) for key in sorted(_TIMING)
              if key in functional}
    plan = functional.get("plan")
    if isinstance(plan, dict):
        for key in ("estimated_seconds", "quick_seconds"):
            if key in plan:
                timing["plan." + key] = plan.pop(key)
    return functional, timing


def _completion(tool: object, completion: object) -> dict:
    """Describe explicit incomplete outcomes without judging answer quality.

    Parameters
    ----------
    tool: object
        Saved tool summary, if recorded.
    completion: object
        Saved completion metadata, if recorded.

    """
    a = tool if isinstance(tool, dict) else {}
    b = completion if isinstance(completion, dict) else {}
    failures = [v for v in (a.get("status"), b.get("status"))
                if isinstance(v, str) and v in _FAILURES]
    if a.get("valid") is False or failures:
        status = "incomplete"
    elif a.get("valid") is True or a.get("status") in (
            "completed", "complete", "success", "ok"):
        status = "complete"
    elif tool is None and not b:
        status = "not_reported"
    else:
        status = "unconfirmed"
    return {"status": status, "tool": a.get("name"),
            "tool_status": a.get("status"), "valid": a.get("valid"),
            "completion": b, "problems": a.get("problems"),
            "saved": {k: a[k] for k in ("report", "saved", "saved_pdf")
                      if k in a}}


def tool_comparison(tool: dict, completion: dict) -> dict:
    """Keep raw tool summaries in the report and compare functional fields.

    Parameters
    ----------
    tool: dict
        Original and replayed saved tool summaries.
    completion: dict
        Original and replayed saved completion metadata.

    """
    functional, timings, outcomes = {}, {}, {}
    for side in ("original", "replayed"):
        functional[side], timings[side] = _functional_tool(tool[side])
        outcomes[side] = _completion(tool[side], completion[side])
    return {"functional": functional, "timing": timings,
            "functional_equal": functional["original"] ==
            functional["replayed"], "outcomes": outcomes}


def independent_questions(recording: Recording,
                          report: ValidationReport) -> list[dict]:
    """Expose unique decodable saved questions independently of alignment.

    Parameters
    ----------
    recording: Recording
        Independently read workspace snapshot.
    report: ValidationReport
        Validation of that snapshot.

    Returns
    -------
    list[dict]
        Saved questions corroborated by a save event and a unique matching
        canonical turn. Invalid-recording rows are explicitly unverified.
        Ambiguous or unavailable saved text is not reconstructed.

    """
    turns = {}
    ambiguous = set()
    for state in recording.states.values():
        for turn in state.get("turns", []):
            uid = turn.get("conversation_turn_uuid")
            if not isinstance(uid, str):
                continue
            if uid in turns and turns[uid] != turn:
                ambiguous.add(uid)
            turns[uid] = turn
    events = [e for e in report.events if e.event_type == "gui.question.saved"]
    counts = Counter(e.details.get("conversation_turn_uuid") for e in events
                     if isinstance(e.details.get("conversation_turn_uuid"),
                                   str))
    rows = []
    for event in events:
        uid = event.details.get("conversation_turn_uuid")
        if not isinstance(uid, str) or counts[uid] != 1 or uid in ambiguous:
            continue
        turn = turns.get(uid, {})
        if not all(isinstance(turn.get(k), str)
                   for k in ("question", "answer")):
            continue
        parameters = turn.get("parameters") or {}
        if not isinstance(parameters, dict):
            parameters = {}
        rows.append({"sequence": event.sequence, "trace": event.path,
                     "turn_uuid": uid, "validated": report.valid,
                     "question": turn["question"], "answer": turn["answer"],
                     "intent": parameters.get("intent"),
                     "tool_outcome": _completion(parameters.get("tool"),
                                                 parameters.get(
                                                     "completion"))})
    return rows
