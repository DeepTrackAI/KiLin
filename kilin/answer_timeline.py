"""Record lightweight, persistent timing for an answered conversation turn.

Answer timelines own the activity stream for the RAG coordinator. Each
activity is reported to observers, included in optional diagnostic traces,
and stored with the committed conversation turn. No GUI-specific activity
inference is needed. Recording uses a monotonic clock, independent of trace
availability. Only activity identifiers, descriptions, and durations are
persisted with the conversation.

The saved duration ends before the atomic conversation checkpoint. Measuring
that write would require a second checkpoint, so the write and the final
completion notification are intentionally excluded from saved timings.

Key Features
------------

- **Core-owned timing**

    Measure answer activities even when no phase observer or trace is enabled.

- **Single-checkpoint persistence**

    Store a compact, validated timeline alongside its completed turn.

Module Structure
----------------

- ``AnswerTimelineStep``

    One completed activity and its duration in milliseconds.

- ``AnswerTimeline``

    The total recorded duration and ordered activity steps.

- ``_AnswerTimelineRecorder``

    A per-answer monotonic recorder translating phase notifications to one
    ordered stream of user-visible activities.

- ``_timeline_to_data`` and ``_timeline_from_data``

    Convert validated records to and from conversation JSON values.

Examples
--------
>>> import kilin
>>> step = kilin.AnswerTimelineStep("generating", "Generating answer", 5)
>>> timeline = kilin.AnswerTimeline(5, (step,))
>>> timeline.elapsed_ms
5

"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from threading import Lock
from time import perf_counter

__all__ = ["AnswerTimeline", "AnswerTimelineStep"]


@dataclass(frozen=True, slots=True)
class AnswerTimelineStep:
    """One activity identified by its phase, label, and measured duration.

    Parameters
    ----------
    phase: str
        Stable identifier of the originating core phase, or ``preparing``
        if no phases were reported at all.
    label: str
        Human-readable activity description from the core.
    ms: int
        Nonnegative duration in milliseconds.

    Attributes
    ----------
    phase: str
        Core activity identifier.
    label: str
        Human-readable activity description.
    ms: int
        Duration in milliseconds.

    """

    phase: str
    label: str
    ms: int


@dataclass(frozen=True, slots=True)
class AnswerTimeline:
    """Completed answer duration and its ordered execution activities.

    Parameters
    ----------
    elapsed_ms: int
        Duration through answer preparation and generation, ending just
        before the atomic conversation checkpoint. Excludes the checkpoint
        write and any GUI-only operations such as pre-answer model loading.
    steps: tuple[AnswerTimelineStep, ...]
        Non-overlapping sequential activities whose durations total
        ``elapsed_ms``.

    Attributes
    ----------
    elapsed_ms: int
        Total measured answer-processing duration, in milliseconds.
    steps: tuple[AnswerTimelineStep, ...]
        Ordered activity steps in the answer.

    """

    elapsed_ms: int
    steps: tuple[AnswerTimelineStep, ...]


_PHASE_LABELS = {
    "routing": "Identifying intent",
    "retrieving": "Retrieving evidence",
    "evidence_ready": "Preparing answer",
    "generating": "Generating answer",
    "processing": "Processing request",
    "saving": "Saving conversation",
}


def _history_activity(details: Mapping[str, object]) -> str:
    """Describe the selected history using the core's routing decision.

    Parameters
    ----------
    details: Mapping[str, object]
        Metadata accompanying the routed phase.

    Returns
    -------
    str
        Description of the earlier turns supplied to the model.

    """

    chosen = details.get("history_turn_ids")
    related = details.get("related_turn_ids")
    ids = chosen if isinstance(chosen, (tuple, list)) else ()
    related_ids = related if isinstance(related, (tuple, list)) else ()
    if not ids:
        return "No earlier conversation needed"
    recent = [number for number in ids if number not in related_ids]
    parts = []
    if recent:
        name = "turn" if len(recent) == 1 else "turns"
        parts.append(f"the last {len(recent)} {name}")
    if related_ids:
        name = "turn" if len(related_ids) == 1 else "turns"
        numbers = ", ".join(str(number) for number in related_ids)
        parts.append(f"{name} {numbers}")
    return f"Including {' and '.join(parts)} from the conversation"


def _activity_labels(
    phase: str,
    details: Mapping[str, object],
) -> tuple[str, ...]:
    """Resolve canonical activity labels for one core phase notification.

    Parameters
    ----------
    phase: str
        Core answer phase.
    details: Mapping[str, object]
        Phase-specific details already known to the RAG coordinator.

    Returns
    -------
    tuple[str, ...]
        Every new user-facing activity represented by this notification.
        Completion is a post-checkpoint state, not an execution activity.

    """

    if phase == "completed":
        return ()
    if phase == "routed":
        intent = str(details.get("intent", "")).replace("_", " ")
        name = intent.capitalize() if intent else "Intent"
        return (f"{name} identified", _history_activity(details))
    message = details.get("message")
    if isinstance(message, str) and message.strip():
        return (message.strip(),)
    return (_PHASE_LABELS.get(phase, "Processing answer"),)


class _AnswerTimelineRecorder:
    """Accumulate phase transitions before committing one answer timeline.

    Parameters
    ----------
    clock: Callable[[], float], optional
        Monotonic time source in seconds. Defaults to ``perf_counter``.
        The injected clock permits deterministic duration tests.

    Notes
    -----
    Phase notifications describe transitions, not completed spans. The
    recorder measures each span until the following distinct notification.
    The saving notification starts immediately before the checkpoint, but
    its disk write is not timed. Post-commit completion is not an activity.
    Repeated notifications describing the same ongoing activity update that
    activity rather than creating misleading duplicate starts. The recorder
    returns exactly the new starts that form the saved activity list.

    """

    def __init__(
        self,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        """Start one timeline before routing and other answer processing.

        Parameters
        ----------
        clock: Callable[[], float], optional
            Monotonic seconds source, defaulting to ``perf_counter``.

        """

        self._clock = clock
        self._lock = Lock()
        self._started_at = clock()
        self._activities: list[tuple[str, str, float]] = []
        self._finished: AnswerTimeline | None = None

    def record(
        self, phase: str, details: Mapping[str, object],
    ) -> tuple[dict[str, object], ...]:
        """Start activities for one phase and return their exact identities.

        Parameters
        ----------
        phase: str
            Machine-readable answer phase.
        details: Mapping[str, object]
            Existing event metadata needed for the canonical descriptions.

        Returns
        -------
        tuple[dict[str, object], ...]
            New activity events in order, each with its position, phase,
            description, and monotonic start offset. A repeated update of
            the current activity emits no duplicate start event.

        """

        labels = _activity_labels(phase, details)
        with self._lock:
            if self._finished is not None:
                return ()
            events: list[dict[str, object]] = []
            for label in labels:
                if self._activities and (
                    self._activities[-1][:2] == (phase, label)
                ):
                    continue
                now = self._clock()
                started = now if self._activities else self._started_at
                self._activities.append((phase, label, started))
                events.append({
                    "index": len(self._activities) - 1,
                    "phase": phase,
                    "label": label,
                    "started_ms": max(
                        0, round((started - self._started_at) * 1000)
                    ),
                })
            return tuple(events)

    def finish(self) -> AnswerTimeline:
        """Freeze durations just before the conversation checkpoint.

        Returns
        -------
        AnswerTimeline
            Sequential activities measured through the final response.
            Rounded stage boundaries ensure their durations add exactly
            to the total recorded milliseconds.

        """

        with self._lock:
            if self._finished is not None:
                return self._finished
            elapsed_ms = max(
                0, round((self._clock() - self._started_at) * 1000)
            )
            if not self._activities:
                self._activities.append((
                    "preparing", "Preparing answer", self._started_at
                ))
            boundaries = []
            for _, _, started_at in self._activities:
                offset = round((started_at - self._started_at) * 1000)
                boundaries.append(min(elapsed_ms, max(
                    boundaries[-1] if boundaries else 0, offset
                )))
            ends = (*boundaries[1:], elapsed_ms)
            steps = tuple(
                AnswerTimelineStep(phase, label, end - start)
                for (phase, label, _), start, end in zip(
                    self._activities, boundaries, ends
                )
            )
            self._finished = AnswerTimeline(elapsed_ms, steps)
            return self._finished


def _timeline_to_data(timeline: AnswerTimeline | None) -> object:
    """Convert a timeline to JSON-compatible conversation data.

    Parameters
    ----------
    timeline: AnswerTimeline | None
        Completed timeline or None when no timing was recorded.

    Returns
    -------
    object
        Mapping of total and step durations, or None.

    """

    if timeline is None:
        return None
    return {
        "elapsed_ms": timeline.elapsed_ms,
        "steps": [
            {"phase": step.phase, "label": step.label, "ms": step.ms}
            for step in timeline.steps
        ],
    }


def _timeline_from_data(data: object) -> AnswerTimeline | None:
    """Validate and restore a timeline from conversation JSON.

    Parameters
    ----------
    data: object
        Parsed timeline mapping, or None for a turn without timing.

    Returns
    -------
    AnswerTimeline | None
        Reconstructed immutable timeline, or None.

    Raises
    ------
    ValueError
        If the timeline has invalid structure, identifiers, or durations.

    """

    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError("Saved answer timeline must be a mapping.")
    elapsed_ms = data.get("elapsed_ms")
    records = data.get("steps")
    if type(elapsed_ms) is not int or elapsed_ms < 0:
        raise ValueError("Saved answer duration must be nonnegative.")
    if not isinstance(records, list):
        raise ValueError("Saved answer timeline steps must be a list.")
    steps = []
    for item in records:
        if not isinstance(item, dict):
            raise ValueError("Saved answer timeline step must be a mapping.")
        phase, label, ms = (
            item.get("phase"), item.get("label"), item.get("ms")
        )
        if not isinstance(phase, str) or not phase.strip():
            raise ValueError("Saved answer timeline phase is invalid.")
        if not isinstance(label, str) or not label.strip():
            raise ValueError("Saved answer timeline label is invalid.")
        if type(ms) is not int or ms < 0:
            raise ValueError("Saved answer timeline step duration is invalid.")
        steps.append(AnswerTimelineStep(phase, label, ms))
    if sum(step.ms for step in steps) != elapsed_ms:
        raise ValueError("Saved answer timeline durations do not add up.")
    return AnswerTimeline(elapsed_ms, tuple(steps))
