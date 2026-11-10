"""Record lightweight, persistent timing for an answered conversation turn.

Answer timelines measure stages already reported by the RAG coordinator.
They are ordinary conversation data, independent of optional diagnostic
tracing and of the GUI. Recording uses a monotonic clock; only durations and
human-readable stage labels are persisted, not clock values or trace content.

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

    A per-answer monotonic recorder consuming existing phase notifications.

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
        for work before the first notification.
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
    "routed": "Selecting context",
    "retrieving": "Retrieving evidence",
    "evidence_ready": "Preparing answer",
    "generating": "Generating answer",
    "processing": "Processing request",
}


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
    Saving and completion notifications are excluded because they occur at
    or after the conversation checkpoint boundary. Repeated notifications
    with the same phase and label do not create duplicate steps. Worker
    callbacks are serialized to keep observed phase boundaries ordered.

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
        self._activities: list[tuple[str, str, float]] = [
            ("preparing", "Preparing answer", self._started_at)
        ]
        self._finished: AnswerTimeline | None = None

    def record(self, phase: str, details: Mapping[str, object]) -> None:
        """Observe a core phase without altering the caller's notification.

        Parameters
        ----------
        phase: str
            Machine-readable answer phase.
        details: Mapping[str, object]
            Existing event metadata; the ``message`` may supply the label.

        """

        if phase in ("saving", "completed"):
            return
        message = details.get("message")
        label = (
            message.strip()
            if isinstance(message, str) and message.strip()
            else _PHASE_LABELS.get(phase, "Processing answer")
        )
        with self._lock:
            if self._finished is not None:
                return
            previous_phase, previous_label, _ = self._activities[-1]
            if (phase, label) != (previous_phase, previous_label):
                self._activities.append((phase, label, self._clock()))

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
