"""Plan expensive tool runs and let the user confirm them or choose.

Tools that call the model many times first resolve their sources and
estimate the work. When the tool's confirmation policy asks for it, the
plan is offered to the caller, who chooses to continue, run a quick
non-exhaustive version, or cancel. Callers without a confirmation callback
(notebooks, scripts) run the full plan. Tools that need one choice from the
user (which paper, save a list or show it) ask through ``ask_choice``;
without a choice callback they receive None and use their best guess.

Every plan, question, and answer is traced, so a conversation trace shows
what the user was offered and what they chose. Durations are rough
estimates for the user's decision, not limits: nothing here stops a run
that takes longer than planned.

Key Features
------------

- **Policy**

    Ask before every run, only before large runs, or never.

- **Plans**

    Describe scope, items, batches, model calls, and an approximate
    duration in a form the interface can show.

- **Choices**

    Continue, run the quick version, or cancel; an unexpected answer
    cancels.

- **Questions**

    Ask the user to pick one of a few options, shown as buttons.

Module Structure
----------------

- ``CONFIRMATION_ALWAYS``, ``CONFIRMATION_WHEN_LARGE``, and
  ``CONFIRMATION_NEVER``

    Canonical string values of a tool's confirmation policy.

- ``CONFIRMATION_CHOICES``

    Ordered collection of accepted confirmation policies.

- ``ConfirmationPolicy``

    Type alias for ``"always"``, ``"when_large"``, or ``"never"``.

- ``PLAN_CHOICE_CONTINUE``, ``PLAN_CHOICE_QUICK``, and
  ``PLAN_CHOICE_CANCEL``

    Canonical string values of the user's answer to a plan.

- ``PLAN_CHOICE_CHOICES``

    Ordered collection of accepted answers to a plan.

- ``PlanChoice``

    Type alias for ``"continue"``, ``"quick"``, or ``"cancel"``.

- ``ToolPlan``

    The work a tool would do, for the user to confirm.

- ``needs_confirmation``

    Tell whether a plan must be confirmed before it runs.

- ``confirm_plan``

    Trace a plan, ask the user when the policy says so, trace the answer.

- ``ask_choice``

    Ask the user to pick one option, tracing the question and answer.

Examples
--------
>>> from qachat.tools.planning import (
...     CONFIRMATION_WHEN_LARGE, ToolPlan, needs_confirmation
... )
>>> plan = ToolPlan(
...     tool="batch_classify",
...     summary="Classify 120 mentions of MRI.",
...     scope="All indexed files.",
...     items=120,
...     batches=12,
...     batches_to_run=12,
...     estimated_seconds=240.0,
...     quick_items=10,
...     quick_seconds=20.0,
... )
>>> needs_confirmation(CONFIRMATION_WHEN_LARGE, plan, 4)
True

"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Final, Literal, TypeAlias

if TYPE_CHECKING:
    from qachat.tools.contract import ToolContext

__all__ = [
    "CONFIRMATION_ALWAYS",
    "CONFIRMATION_CHOICES",
    "CONFIRMATION_NEVER",
    "CONFIRMATION_WHEN_LARGE",
    "PLAN_CHOICE_CANCEL",
    "PLAN_CHOICE_CHOICES",
    "PLAN_CHOICE_CONTINUE",
    "PLAN_CHOICE_QUICK",
    "ConfirmationPolicy",
    "PlanChoice",
    "ToolPlan",
    "ask_choice",
    "confirm_plan",
    "needs_confirmation",
]

# When a tool that calls the model asks the user before running: always,
# only when its plan has more batches than the tool's ``large_batches``,
# or never.
ConfirmationPolicy: TypeAlias = Literal["always", "when_large", "never"]
CONFIRMATION_ALWAYS: Final[ConfirmationPolicy] = "always"
CONFIRMATION_WHEN_LARGE: Final[ConfirmationPolicy] = "when_large"
CONFIRMATION_NEVER: Final[ConfirmationPolicy] = "never"
CONFIRMATION_CHOICES: Final[tuple[ConfirmationPolicy, ...]] = (
    CONFIRMATION_ALWAYS,
    CONFIRMATION_WHEN_LARGE,
    CONFIRMATION_NEVER,
)

# The user's answer to a plan. The interface mirrors these values.
PlanChoice: TypeAlias = Literal["continue", "quick", "cancel"]
PLAN_CHOICE_CONTINUE: Final[PlanChoice] = "continue"
PLAN_CHOICE_QUICK: Final[PlanChoice] = "quick"
PLAN_CHOICE_CANCEL: Final[PlanChoice] = "cancel"
PLAN_CHOICE_CHOICES: Final[tuple[PlanChoice, ...]] = (
    PLAN_CHOICE_CONTINUE,
    PLAN_CHOICE_QUICK,
    PLAN_CHOICE_CANCEL,
)

# Progress shown while the user decides; the interface shows the same
# words for plans and choices.
_WAITING_FOR_CHOICE: Final[str] = "Waiting for your choice"


@dataclass(frozen=True, slots=True)
class ToolPlan:
    """The work a tool would do, for the user to confirm.

    Attributes
    ----------
    tool: str
        Tool name, such as ``"batch_classify"``.
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
        tool's model-call limit applies.
    estimated_seconds: float
        Approximate duration of ``batches_to_run`` calls.
    quick_items: int
        Items the quick version would process.
    quick_seconds: float
        Approximate duration of the quick version.
    details: tuple[str, ...], optional
        Further lines to show, such as the categories. Defaults to ``()``.
    quick_label: str, optional
        Short name of the quick version, when it is more specific than a
        sample (for example "Exact matches only"). Defaults to ``""``.
    quick_note: str, optional
        One line under ``quick_label`` saying what it covers (for example
        "6 passages mention “XYZ”"). Defaults to ``""``.

    """

    tool: str
    summary: str
    scope: str
    items: int
    batches: int
    batches_to_run: int
    estimated_seconds: float
    quick_items: int
    quick_seconds: float
    details: tuple[str, ...] = ()
    quick_label: str = ""
    quick_note: str = ""

    def to_dict(self) -> dict[str, object]:
        """Return the plan as JSON-compatible data.

        Returns
        -------
        dict[str, object]
            Every attribute by name, with ``details`` as a list.

        """

        data = asdict(self)
        data["details"] = list(self.details)
        return data


def needs_confirmation(
    policy: ConfirmationPolicy, plan: ToolPlan, large_batches: int
) -> bool:
    """Tell whether a plan must be confirmed before it runs.

    Parameters
    ----------
    policy: ConfirmationPolicy
        ``always``, ``when_large``, or ``never``.
    plan: ToolPlan
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


def confirm_plan(
    plan: ToolPlan,
    policy: ConfirmationPolicy,
    large_batches: int,
    context: ToolContext,
) -> tuple[PlanChoice, bool]:
    """Trace a plan, ask the user when the policy says so, trace the answer.

    Parameters
    ----------
    plan: ToolPlan
        The plan.
    policy: ConfirmationPolicy
        The tool's confirmation policy.
    large_batches: int
        The tool's threshold for ``when_large``.
    context: ToolContext
        QAChat's services; ``context.confirm`` asks the user.

    Returns
    -------
    tuple[PlanChoice, bool]
        The choice (``continue`` when nobody was asked; anything unexpected
        becomes ``cancel``) and whether the user was asked.

    """

    trace = context.trace
    trace.emit(
        "tool.plan.proposed",
        component="tool",
        data=plan.to_dict(),
        turn_id=context.trace_turn_id,
    )

    asked = context.confirm is not None and needs_confirmation(
        policy, plan, large_batches
    )
    choice: PlanChoice = PLAN_CHOICE_CONTINUE
    if asked:
        context.progress(_WAITING_FOR_CHOICE, plan=plan.to_dict())
        answer = context.confirm(plan)
        choice = (
            answer if answer in PLAN_CHOICE_CHOICES else PLAN_CHOICE_CANCEL
        )

    trace.emit(
        "tool.plan.decided",
        component="tool",
        data={"choice": choice, "asked": asked, "policy": policy},
        turn_id=context.trace_turn_id,
    )

    return choice, asked


def ask_choice(
    question: str, options: tuple[str, ...], context: ToolContext
) -> int | None:
    """Ask the user to pick one option, tracing the question and answer.

    Parameters
    ----------
    question: str
        What to ask, such as "Which paper should I turn into a research
        briefing?".
    options: tuple[str, ...]
        The options, shown as buttons.
    context: ToolContext
        QAChat's services; ``context.choose`` asks the user.

    Returns
    -------
    int | None
        The chosen option's index; None when the user cancels, the answer
        is not an option, or nobody can be asked (``context.choose`` is
        None).

    """

    trace = context.trace
    trace.emit(
        "tool.choice.proposed",
        component="tool",
        data={"question": question, "options": list(options)},
        turn_id=context.trace_turn_id,
    )

    index = None
    if context.choose is not None and options:
        context.progress(
            _WAITING_FOR_CHOICE,
            choice={"question": question, "options": list(options)},
        )
        answer = context.choose(question, options)
        if type(answer) is int and 0 <= answer < len(options):
            index = answer

    trace.emit(
        "tool.choice.decided",
        component="tool",
        data={"index": index, "asked": context.choose is not None},
        turn_id=context.trace_turn_id,
    )

    return index
