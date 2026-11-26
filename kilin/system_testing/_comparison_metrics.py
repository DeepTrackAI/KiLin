"""Measure recorded text, retrieval, model calls, and file inventories.

Key Features
------------
- Deterministic lexical and exact-passage metrics, without a model judge.
- Optional cosine similarity from one explicitly selected local evaluator.
- Recorded durations and token counts, with missing values left unavailable.

Module Structure
----------------
- Text, ranking, evidence, and trace-stage measurements.
- ``EmbeddingEvaluator`` owns one optional embedding session per comparison.
- ``inventory`` and ``file_changes`` compare retained file bytes.

Examples
--------
>>> from kilin.system_testing import Experiment
>>> # Experiment("010_original").compare().display()

"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from hashlib import sha256
import math
import os
from pathlib import Path
import re
import unicodedata

from kilin._model_completion import strip_thinking

from .timeline import TimelineOperation
from .traces import TraceEvent, safe_path

__all__ = []

_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_TOKENS = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_RANKINGS = (
    "retrieval.semantic.ranked", "retrieval.lexical.ranked",
    "retrieval.code.ranked", "retrieval.hybrid.ranked",
    "retrieval.sources.ranked",
)


def normalized_text(text: str) -> str:
    """Normalize display text without changing its case or punctuation.

    Parameters
    ----------
    text: str
        Recorded text, potentially including model thinking tags.

    Returns
    -------
    str
        Visible text in NFC with whitespace runs collapsed. Numbered
        citations remain in the text; thinking is excluded.

    """
    return " ".join(unicodedata.normalize("NFC", strip_thinking(text)).split())


def text_metrics(original: str, replayed: str) -> dict:
    """Measure lexical agreement without claiming semantic correctness.

    Parameters
    ----------
    original: str
        Original saved or traced text.
    replayed: str
        Reproduced text.

    Returns
    -------
    dict
        Raw and normalized equality, visible character counts and multiset
        token Dice/F1. Two empty token sets have no measured similarity.

    """
    first, second = normalized_text(original), normalized_text(replayed)
    a, b = Counter(_TOKENS.findall(first)), Counter(_TOKENS.findall(second))
    count = sum(a.values()) + sum(b.values())
    return {
        "exact_equal": original == replayed,
        "normalized_equal": first == second,
        "original_characters": len(first), "replayed_characters": len(second),
        "token_f1": (2 * sum((a & b).values()) / count if count else None),
    }


def _number(value: object) -> int | float | None:
    """Return a finite nonnegative measurement, not a guessed zero.

    Parameters
    ----------
    value: object
        Recorded duration, count, or absent value.

    Returns
    -------
    int | float | None
        Valid measurement or None.

    """
    if type(value) in (int, float) and value >= 0:
        if type(value) is int or math.isfinite(value):
            return value
    return None


def _count(value: object) -> int | None:
    """Return a recorded integer count, including a genuine zero.

    Parameters
    ----------
    value: object
        Recorded count.

    Returns
    -------
    int | None
        Nonnegative integer or None.

    """
    return value if type(value) is int and value >= 0 else None


def _duration(event: TraceEvent) -> int | float | None:
    """Use the explicit event duration, falling back to its result field.

    Parameters
    ----------
    event: TraceEvent
        One decoded diagnostic event.

    Returns
    -------
    int | float | None
        Recorded duration in milliseconds, if available.

    """
    return _number(event.metadata.get("duration_ms",
                                      event.details.get("duration_ms")))


def operation_duration(op: TimelineOperation) -> dict:
    """Measure a command at its own recorded boundaries.

    Parameters
    ----------
    op: TimelineOperation
        Original or replayed command interval.

    Returns
    -------
    dict
        Duration and measurement basis; point events have no duration.
        Cross-logger intervals use recorded UTC wall-clock timestamps.

    """
    first, last = op.events[0], op.events[-1]
    if first.sequence == last.sequence:
        return {"ms": None, "basis": None}
    if (first.span_id is not None and first.span_id == last.span_id and
            first.metadata["trace_id"] == last.metadata["trace_id"] and
            _duration(last) is not None):
        return {"ms": _duration(last), "basis": "span"}
    start = datetime.fromisoformat(first.metadata["timestamp_utc"])
    end = datetime.fromisoformat(last.metadata["timestamp_utc"])
    return {"ms": _number((end - start).total_seconds() * 1000),
            "basis": "utc_boundary"}


def _stage_summary(events: list[TraceEvent]) -> dict:
    """Summarize one event type without adding nested stage times together.

    Parameters
    ----------
    events: list[TraceEvent]
        Events of one component and type in chronological order.

    Returns
    -------
    dict
        Occurrence count, available timing samples and total only when all
        occurrences contain durations.

    """
    durations = [_duration(event) for event in events]
    complete = bool(durations) and all(ms is not None for ms in durations)
    return {"count": len(events), "duration_ms": durations,
            "total_duration_ms": sum(durations) if complete else None,
            "statuses": dict(sorted(Counter(e.metadata["status"]
                                            for e in events).items()))}


def stage_metrics(original: TimelineOperation,
                  replayed: TimelineOperation) -> list[dict]:
    """Compare diagnostic event counts and explicit timing by stage type.

    Parameters
    ----------
    original: TimelineOperation
        Aligned original operation.
    replayed: TimelineOperation
        Aligned replay operation.

    Returns
    -------
    list[dict]
        Per-component, per-event-type observations. Inclusive nested span
        durations are not a decomposition of total wall time.

    """
    groups = []
    for op in (original, replayed):
        values = {}
        for event in op.events:
            key = event.metadata["component"], event.event_type
            values.setdefault(key, []).append(event)
        groups.append(values)
    first, second = groups
    return [{"component": key[0], "event_type": key[1],
             "original": _stage_summary(first.get(key, [])),
             "replayed": _stage_summary(second.get(key, []))}
            for key in sorted(first.keys() | second.keys())]


def passage_identity(record: dict) -> dict:
    """Describe an exact passage by source coordinates and text digest.

    Parameters
    ----------
    record: dict
        Saved evidence or traced chunk metadata, using managed relative
        source paths rather than absolute workspace or external paths.

    Returns
    -------
    dict
        Source coordinates, symbol, optional section/cell, and SHA-256 of
        the exact text. Generated chunk identifiers and scores are excluded.

    """
    path = record["path"]
    if not isinstance(path, str) or not path:
        raise ValueError("Invalid passage source path.")
    start, end = record["start_line"], record["end_line"]
    if (type(start) is not int or type(end) is not int or
            start < 1 or end < start):
        raise ValueError("Invalid passage line coordinates.")
    digest = record.get("text_sha256")
    if "text" in record:
        digest = sha256(record["text"].encode("utf-8")).hexdigest()
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}",
                                                       digest):
        raise ValueError("Missing exact passage content hash.")
    return {"path": path, "start_line": start, "end_line": end,
            "symbol": record.get("symbol"), "cell": record.get("cell"),
            "section": record.get("section"), "text_sha256": digest}


def _key(item: dict) -> tuple:
    """Build a hashable passage identity without comparing ranking scores.

    Parameters
    ----------
    item: dict
        Exact passage identity.

    Returns
    -------
    tuple
        Stable ordered identity fields.

    """
    return tuple(item.get(key) for key in ("path", "start_line", "end_line",
                                      "symbol", "cell", "section",
                                      "text_sha256"))


def overlap(original: list[dict], replayed: list[dict]) -> dict:
    """Measure exact and source-level overlap, including rank displacement.

    Parameters
    ----------
    original: list[dict]
        Original passage identities in recorded order.
    replayed: list[dict]
        Replay passage identities in recorded order.

    Returns
    -------
    dict
        Unique set overlap and mean absolute displacement of common exact
        passages, based on first occurrence positions. Empty unions and no
        common ranks have no measured similarity or displacement.

    """
    ranks = []
    for items in (original, replayed):
        positions = {}
        for i, item in enumerate(items, 1):
            positions.setdefault(_key(item), i)
        ranks.append(positions)
    first, second = ranks
    common = first.keys() & second.keys()
    union = first.keys() | second.keys()
    paths_a = {item["path"] for item in original}
    paths_b = {item["path"] for item in replayed}
    paths = paths_a | paths_b
    depth = min(len(original), len(replayed))
    # Compare the same available prefix depth, without arbitrary top-k cuts.
    prefix_a = {_key(item) for item in original[:depth]}
    prefix_b = {_key(item) for item in replayed[:depth]}
    return {
        "original_count": len(original), "replayed_count": len(replayed),
        "original_unique": len(first), "replayed_unique": len(second),
        "common": len(common), "union": len(union),
        "jaccard": len(common) / len(union) if union else None,
        "source_jaccard": (len(paths_a & paths_b) / len(paths)
                           if paths else None),
        "prefix_depth": depth,
        "prefix_overlap": (len(prefix_a & prefix_b) / depth if depth
                           else None),
        "mean_absolute_rank_shift": (
            sum(abs(first[key] - second[key]) for key in common) / len(common)
            if common else None
        ),
    }


def _evidence(turn: dict) -> tuple[list[dict], list[dict], list[str]]:
    """Resolve visible numeric citations against the saved evidence list.

    Parameters
    ----------
    turn: dict
        Authoritative saved question, answer and evidence record.

    Returns
    -------
    tuple
        Evidence identities, cited identities in first-use order, and
        unmatched citation numbers. This is syntactic, not a support check.

    """
    passages = []
    citations = {}
    for item in turn["evidence"]:
        identity = passage_identity(item)
        number = item["citation"]
        if type(number) is not int or number < 1 or str(number) in citations:
            raise ValueError("Invalid or duplicate saved citation number.")
        passages.append(identity)
        citations[str(number)] = identity
    cited = []
    missing = []
    seen = set()
    for group in _CITATION.findall(strip_thinking(turn["answer"])):
        for number in group.split(","):
            number = number.strip().lstrip("0") or "0"
            if number in seen:
                continue
            seen.add(number)
            if number in citations:
                cited.append(citations[number])
            else:
                missing.append(number)
    return passages, cited, missing


def evidence_metrics(original: dict, replayed: dict) -> dict:
    """Compare saved evidence and actual syntactically cited passages.

    Parameters
    ----------
    original: dict
        Original saved turn.
    replayed: dict
        Corresponding replayed saved turn.

    Returns
    -------
    dict
        Both passage inventories, overlap metrics, and unresolved citations.

    """
    a, cited_a, missing_a = _evidence(original)
    b, cited_b, missing_b = _evidence(replayed)
    return {"original": a, "replayed": b, "overlap": overlap(a, b),
            "citations": {"original": cited_a, "replayed": cited_b,
                          "overlap": overlap(cited_a, cited_b),
                          "original_unresolved": missing_a,
                          "replayed_unresolved": missing_b}}


def _ranked(event: TraceEvent) -> list[dict]:
    """Read one complete recorded ranking in its recorded rank order.

    Parameters
    ----------
    event: TraceEvent
        Ranked retrieval diagnostic.

    Returns
    -------
    list[dict]
        Exact passage identities in rank order.

    """
    if event.event_type == "retrieval.sources.ranked":
        rows = event.details["sources"]
        if (not isinstance(rows, list) or not all(
                isinstance(row.get("path"), str) and row["path"]
                for row in rows)):
            raise ValueError("Invalid source ranking diagnostic.")
        return [{"path": row["path"]} for row in rows]
    rows = event.details["results"]
    if (not isinstance(rows, list) or event.details["count"] != len(rows)
            or any(row["rank"] != i for i, row in enumerate(rows, 1))):
        raise ValueError("Invalid ranked retrieval diagnostic.")
    return [passage_identity(row["chunk"]) for row in rows]


def ranking_metrics(original: TimelineOperation,
                    replayed: TimelineOperation) -> list[dict]:
    """Compare ranked events by channel and occurrence, not generated IDs.

    Parameters
    ----------
    original: TimelineOperation
        Aligned original operation.
    replayed: TimelineOperation
        Aligned replay operation.

    Returns
    -------
    list[dict]
        One row per channel occurrence, with absent sides left unavailable.
        Occurrence pairing is descriptive; a changed pipeline may have
        changed the query or purpose of a particular search.

    """
    comparisons = []
    for kind in _RANKINGS:
        old, new = original.matching(kind), replayed.matching(kind)
        for i in range(max(len(old), len(new))):
            row = {"event_type": kind, "occurrence": i + 1,
                   "identity": ("source_path" if kind ==
                                "retrieval.sources.ranked" else "passage")}
            errors = []
            for side, events in (("original", old), ("replayed", new)):
                row[side] = None
                if i < len(events):
                    try:
                        row[side] = _ranked(events[i])
                    except (ValueError, TypeError, KeyError,
                            AttributeError) as error:
                        errors.append(f"{side}: {error}")
            a, b = row["original"], row["replayed"]
            row["overlap"] = None
            if a is not None and b is not None:
                try:
                    row["overlap"] = overlap(a, b)
                except (ValueError, TypeError, KeyError,
                        AttributeError) as error:
                    errors.append(str(error))
            row["issues"] = errors
            comparisons.append(row)
    return comparisons


def _artifact_text(event: TraceEvent | None, name: str) -> str | None:
    """Read an inline exact-text artifact without following external paths.

    Parameters
    ----------
    event: TraceEvent | None
        Diagnostic containing the artifact, if any.
    name: str
        Exact-content section key from the trace writer.

    Returns
    -------
    str | None
        Exact text or None when absent.

    """
    if event is not None:
        section = event.exact_content.get(name)
        if isinstance(section, dict) and isinstance(section.get("value"), str):
            return section["value"]
    return None


def model_calls(op: TimelineOperation) -> list[dict]:
    """Correlate LLM inputs and outputs by call identity within a command.

    Parameters
    ----------
    op: TimelineOperation
        One recorded operation, including router and tool calls.

    Returns
    -------
    list[dict]
        Chronological call metadata, token counts, timing, finish reason,
        and optional exact-text artifacts. Missing output is not zero tokens.
        Repeated or unpaired call identities remain explicitly ambiguous.

    """
    inputs = op.matching("llm.call.input")
    outputs = op.matching("llm.call.output")
    matches = {}
    for event in outputs:
        matches.setdefault((event.metadata["trace_id"],
                            event.details.get("call_id")), []).append(event)
    counts = Counter((e.metadata["trace_id"], e.details.get("call_id"))
                     for e in inputs)
    rows = []
    for event in inputs:
        data = event.details
        key = event.metadata["trace_id"], data.get("call_id")
        ends = matches.get(key, [])
        valid = (isinstance(key[1], str) and counts[key] == 1 and
                 len(ends) == 1 and ends[0].sequence > event.sequence)
        end = ends[0] if valid else None
        result = end.details if end is not None else {}
        rows.append({
            "input_sequence": event.sequence,
            "output_sequence": end.sequence if end is not None else None,
            "model_id": data.get("model_id"), "backend": data.get("backend"),
            "model_source": data.get("model_source"),
            "input_tokens": _count(data.get("input_token_count")),
            "output_tokens": _count(result.get("output_token_count")),
            "output_token_count_source": result.get(
                "output_token_count_source"),
            "finish_reason": result.get("finish_reason"),
            "generation_duration_ms": _number(result.get(
                "generation_duration_ms")),
            "reply_duration_ms": _number(result.get("reply_duration_ms")),
            "output_available": valid,
            "prompt": _artifact_text(event, "02_rendered_prompt"),
            "answer": _artifact_text(end, "01_generated_answer"),
        })
    used = {row["output_sequence"] for row in rows}
    for event in outputs:
        if event.sequence not in used:
            rows.append({"input_sequence": None,
                         "output_sequence": event.sequence,
                         "output_available": False,
                         "issue": "Unpaired or ambiguous model output."})
    return rows


def usage(turn: dict) -> dict:
    """Read saved total chat-model usage without double-counting calls.

    Parameters
    ----------
    turn: dict
        Saved turn with historical parameter snapshot.

    Returns
    -------
    dict
        Original saved input and output token counts, or None if absent.
        These cover the chat and router counters, not embedding token use.

    """
    values = turn["parameters"].get("usage", {})
    if not isinstance(values, dict):
        values = {}
    return {key: _count(values.get(key)) for key in (
        "input_tokens", "output_tokens")}


def cosine(first: list[float], second: list[float]) -> float:
    """Compute cosine similarity with finite nonzero equal-size vectors.

    Parameters
    ----------
    first: list[float]
        Evaluation embedding of the original text.
    second: list[float]
        Evaluation embedding of the replayed text.

    Returns
    -------
    float
        Cosine in [-1, 1], with only floating-point rounding clamped.

    """
    if (not first or len(first) != len(second) or
            not all(type(value) in (int, float) and math.isfinite(value)
                    for values in (first, second) for value in values)):
        raise ValueError("Invalid or incompatible evaluation embeddings.")
    a, b = math.hypot(*first), math.hypot(*second)
    if not a or not b or not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("Invalid evaluation embedding norm.")
    return max(-1.0, min(1.0, math.fsum((x / a) * (y / b)
                                      for x, y in zip(first, second))))


class EmbeddingEvaluator:
    """Own one local model instance for optional symmetric answer metrics."""

    def __init__(self, model: str | None, backend: str,
                 revision: str | None) -> None:
        """Store evaluation provenance without loading a model yet.

        Parameters
        ----------
        model: str | None
            Explicit local model ID/path, or None to disable embeddings.
        backend: str
            Selected backend preference.
        revision: str | None
            Optional explicit model revision.

        """
        self.session = None
        self.provenance = {"model_id": model, "backend_preference": backend,
                           "requested_revision": revision,
                           "status": "disabled" if model is None else
                           "not_used", "error": None}

    def measure(self, original: str, replayed: str) -> dict:
        """Use the same loaded model for both normalized visible answers.

        Parameters
        ----------
        original: str
            Original saved answer.
        replayed: str
            Replayed saved answer.

        Returns
        -------
        dict
            Cosine or None and a precise unavailable reason. No retries,
            truncation, substitute models, or downloads are requested.

        """
        provenance = self.provenance
        if provenance["status"] == "disabled":
            return {"cosine": None, "reason": "disabled"}
        first = normalized_text(original)
        second = normalized_text(replayed)
        if not first or not second:
            return {"cosine": None, "reason": "empty visible answer"}
        if provenance["status"] == "failed":
            return {"cosine": None, "reason": provenance["error"]}
        if self.session is None:
            try:
                from kilin.embeddings import EmbeddingSession

                self.session = EmbeddingSession(
                    provenance["model_id"],
                    backend=provenance["backend_preference"],
                    revision=provenance["requested_revision"],
                    allow_download=False,
                )
                provenance.update(status="loaded",
                                  backend=self.session.backend,
                                  device=self.session.device,
                                  model_source=self.session.model_source)
            except Exception as error:
                provenance.update(status="failed", error=str(error))
                return {"cosine": None, "reason": str(error)}
        try:
            vectors = self.session.embed_documents([first, second])
            if len(vectors) != 2:
                raise ValueError("Expected two evaluation embeddings.")
            return {"cosine": cosine(*vectors), "reason": None}
        except Exception as error:
            return {"cosine": None, "reason": str(error)}

    def close(self) -> None:
        """Release the evaluator; retain any cleanup problem in provenance."""
        if self.session is not None:
            try:
                self.session.close()
            except Exception as error:
                self.provenance["cleanup_error"] = str(error)
            finally:
                self.session = None


def inventory(root: Path, directory: str) -> dict:
    """Hash retained regular files without opening linked or external paths.

    Parameters
    ----------
    root: pathlib.Path
        Resolved stable workspace root.
    directory: str
        Workspace-relative inventory folder.

    Returns
    -------
    dict
        File records and explicit inspection errors. A missing folder is
        distinguished from an empty folder. No files are created or repaired.

    """
    files = {}
    issues = []
    exists = False

    def on_error(error: OSError) -> None:
        """Collect traversal errors rather than hiding missing content.

        Parameters
        ----------
        error: OSError
            Error reported by the directory walk.

        """
        issues.append(str(error))

    try:
        path = safe_path(root, directory)
        exists = path.exists()
        if exists and not path.is_dir():
            raise ValueError(f"Not a directory: {directory}")
        if exists:
            for parent, dirs, names in os.walk(path, onerror=on_error):
                dirs.sort()
                for name in list(dirs):
                    child = Path(parent) / name
                    if child.is_symlink():
                        dirs.remove(name)
                        issues.append(f"Linked directory: {child}")
                for name in sorted(names):
                    child = Path(parent) / name
                    relative = child.relative_to(root).as_posix()
                    try:
                        safe_path(root, relative)
                        if not child.is_file():
                            raise ValueError(f"Nonregular file: {relative}")
                        with child.open("rb") as stream:
                            before = os.fstat(stream.fileno())
                            digest = sha256()
                            for block in iter(lambda: stream.read(1024 * 1024),
                                              b""):
                                digest.update(block)
                            after = os.fstat(stream.fileno())
                        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns",
                                  "st_ctime_ns")
                        if any(getattr(before, key) != getattr(after, key)
                               for key in fields):
                            raise ValueError(f"File changed: {relative}")
                        files[relative] = {"size_bytes": after.st_size,
                                           "sha256": digest.hexdigest()}
                    except (ValueError, OSError) as error:
                        issues.append(str(error))
    except (ValueError, OSError) as error:
        issues.append(str(error))
    return {"exists": exists, "files": files, "issues": issues}


def file_changes(original: dict, replayed: dict) -> dict:
    """Compare inventories by relative name and bytes, not modification time.

    Parameters
    ----------
    original: dict
        Original folder inventory.
    replayed: dict
        Replay folder inventory.

    Returns
    -------
    dict
        Missing, added, changed, and unchanged names. Incomplete inventories
        have no asserted comparison; their individual errors are retained.

    """
    result = {"original": original, "replayed": replayed, "changes": None}
    if (original["exists"] and replayed["exists"] and
            not original["issues"] and not replayed["issues"]):
        a, b = original["files"], replayed["files"]
        result["changes"] = {
            "removed": sorted(a.keys() - b.keys()),
            "added": sorted(b.keys() - a.keys()),
            "changed": sorted(k for k in a.keys() & b.keys() if a[k] != b[k]),
            "unchanged": sorted(k for k in a.keys() & b.keys()
                                if a[k] == b[k]),
        }
    return result
