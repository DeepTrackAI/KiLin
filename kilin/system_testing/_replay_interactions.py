"""Replay only recorded user responses to live interactive prompts.

Key Features
------------
- Match a selected option by its unique label rather than its old index.
- Require compatible confirmation plans and consume responses in order.
- Refuse new, changed, missing, cancelled or ambiguous interactions.

Module Structure
----------------
- ``InteractionReplay`` implements the synchronous engine command callback.

Examples
--------
>>> from kilin.system_testing import Experiment
>>> # Experiment("workspaces/010_original").replay()

"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .timeline import TimelineOperation

__all__ = []


class InteractionReplay:
    """Consume recorded user responses in the scope of one question."""

    def __init__(self) -> None:
        """Initialize an empty interaction queue."""
        self._expected = deque()
        self._prompt = None
        self.failure: str | None = None
        self.differences: list[tuple[str, object, object]] = []

    def begin(self, op: TimelineOperation) -> None:
        """Extract only actual user responses, correlated by interaction ID.

        Parameters
        ----------
        op: TimelineOperation
            Validated reference question and its diagnostics.

        """
        self._expected.clear()
        self._prompt = None
        self.failure = None
        self.differences.clear()
        decisions = {
            e.details["interaction_id"]: e.details
            for e in op.matching("tool.plan.decided", "tool.choice.decided",
                                 "routing.choice.decided")
        }
        for event in op.matching("tool.plan.proposed", "tool.choice.proposed",
                                 "routing.choice.proposed"):
            if event.details["asked"]:
                self._expected.append((
                    event.event_type, event.details,
                    decisions[event.details["interaction_id"]],
                ))

    def observe(self, event: dict) -> None:
        """Capture the actual prompt sent before a synchronous callback.

        Parameters
        ----------
        event: dict
            Ordinary headless engine event.

        """
        if event.get("event") != "answer_phase":
            return
        for kind in ("plan", "choice"):
            if kind in event:
                if self._prompt is not None:
                    self._fail("Overlapping interactive prompts.")
                self._prompt = (kind, event[kind])

    def _fail(self, message: str) -> None:
        """Record a divergence even if the engine catches the exception.

        Parameters
        ----------
        message: str
            Precise mismatch explanation.

        """
        self.failure = message
        raise RuntimeError(message)

    def read_command(self) -> dict:
        """Return the recorded semantic response to the current prompt.

        Returns
        -------
        dict
            Ordinary GUI protocol choice command with the live index.

        """
        if self.failure:
            self._fail(self.failure)
        if not self._expected or self._prompt is None:
            self._fail("Unexpected interaction; no matching user response.")
        kind, live = self._prompt
        self._prompt = None
        prefix, original, decision = self._expected[0]
        if prefix == "tool.plan.proposed":
            if kind != "plan":
                self._fail("Expected a tool plan, not an option question.")
            # Counts and timing estimates can vary. Scope and action meaning
            # must agree, especially for a recorded network approval.
            fields = ("tool", "summary", "scope", "details")
            if (any(original.get(key) != live.get(key) for key in fields)
                    or original.get("operation", "model") !=
                    live.get("operation", "model")):
                self._fail("Tool plan scope or meaning changed.")
            choice = decision["choice"]
            if choice not in ("continue", "quick"):
                self._fail("No recorded affirmative plan response.")
            if choice == "quick" and any(
                original.get(key) != live.get(key)
                for key in ("quick_label", "quick_note")
            ):
                self._fail("Quick-plan meaning changed.")
            for key in ("items", "batches", "batches_to_run", "quick_items"):
                if original.get(key) != live.get(key):
                    if live.get("operation") == "network":
                        self._fail("Network approval scope changed.")
                    self.differences.append((
                        f"plan.{key}", original.get(key), live.get(key)
                    ))
        else:
            if (kind != "choice" or original["question"] !=
                    live.get("question") or original["allow_cancel"] !=
                    live.get("allow_cancel", True)):
                self._fail("Interactive question or choice scope changed.")
            label = decision["selected_option"]
            options = live.get("options", [])
            if not isinstance(options, list) or options.count(label) != 1:
                self._fail(f"Recorded choice is missing or ambiguous: {label}")
            choice = f"option:{options.index(label)}"
            if original["options"] != options:
                self.differences.append(("choice.options", original["options"],
                                         options))
        self._expected.popleft()
        return {"action": "plan_choice", "choice": choice}

    def finish(self) -> None:
        """Reject a question that did not consume its exact user responses."""
        if self.failure:
            self._fail(self.failure)
        if self._expected or self._prompt is not None:
            self._fail("Recorded interaction was not reproduced.")
