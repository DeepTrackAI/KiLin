"""Plan expensive workflows and let the user confirm them.

Workflows that call the model many times first resolve their sources and
estimate the work. When the intent's confirmation policy asks for it, the
plan is offered to the caller, who chooses to continue, run a quick
non-exhaustive version, or cancel. Callers without a confirmation callback
(notebooks, scripts) run the full plan.

Key Features
------------

- **Choices**

    ``WorkflowChoice`` is the answer to a plan: continue, quick, or cancel.

- **Plans**

    ``WorkflowPlan`` describes scope, items, batches, model calls, and an
    approximate duration in a form the interface can show.

- **Policy**

    ``needs_confirmation`` applies an intent's confirmation policy.

Module Structure
----------------

- ``WorkflowChoice``
- ``WorkflowPlan``
- ``needs_confirmation``

"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final, Literal, TypeAlias

from qachat.project_types import (
    CONFIRMATION_ALWAYS,
    CONFIRMATION_NEVER,
    ConfirmationPolicy,
)

__all__ = [
    "WORKFLOW_CHOICE_CANCEL",
    "WORKFLOW_CHOICE_CHOICES",
    "WORKFLOW_CHOICE_CONTINUE",
    "WORKFLOW_CHOICE_QUICK",
    "WorkflowChoice",
    "WorkflowPlan",
    "needs_confirmation",
]

# The user's answer to a plan. The interface mirrors these values.
WorkflowChoice: TypeAlias = Literal["continue", "quick", "cancel"]
WORKFLOW_CHOICE_CONTINUE: Final[WorkflowChoice] = "continue"
WORKFLOW_CHOICE_QUICK: Final[WorkflowChoice] = "quick"
WORKFLOW_CHOICE_CANCEL: Final[WorkflowChoice] = "cancel"
WORKFLOW_CHOICE_CHOICES: Final[tuple[WorkflowChoice, ...]] = (
    WORKFLOW_CHOICE_CONTINUE,
    WORKFLOW_CHOICE_QUICK,
    WORKFLOW_CHOICE_CANCEL,
)


@dataclass(frozen=True, slots=True)
class WorkflowPlan:
    """The work a workflow would do, for the user to confirm.

    Attributes
    ----------
    kind: str
        Workflow kind, such as ``"batch_classify"``.
    summary: str
        One sentence describing the work.
    scope: str
        Which sources are covered and why.
    items: int
        Items to process (for example mentions to classify).
    batches: int
        Model calls needed to process every item.
    batches_to_run: int
        Model calls that would run; fewer than ``batches`` when the
        intent's model-call limit applies.
    estimated_seconds: float
        Approximate duration of ``batches_to_run`` calls.
    quick_items: int
        Items the quick version would process.
    quick_seconds: float
        Approximate duration of the quick version.
    details: tuple[str, ...]
        Further lines to show, such as the categories.

    """

    kind: str
    summary: str
    scope: str
    items: int
    batches: int
    batches_to_run: int
    estimated_seconds: float
    quick_items: int
    quick_seconds: float
    details: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return the plan as JSON-compatible data."""

        data = asdict(self)
        data["details"] = list(self.details)
        return data


def needs_confirmation(
    policy: ConfirmationPolicy, plan: WorkflowPlan, large_batches: int
) -> bool:
    """Tell whether a plan must be confirmed before it runs.

    Parameters
    ----------
    policy: ConfirmationPolicy
        ``always``, ``when_large``, or ``never``.
    plan: WorkflowPlan
        The plan.
    large_batches: int
        With ``when_large``, plans with more model calls than this ask.

    Returns
    -------
    bool
        True when the user should be asked; plans without items never ask.

    """

    if plan.items == 0 or policy == CONFIRMATION_NEVER:
        return False
    if policy == CONFIRMATION_ALWAYS:
        return True
    return plan.batches_to_run > large_batches
