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
>>> from kilin.tools._shared.planning import (
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

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Final, Literal, TypeAlias
from uuid import uuid4

if TYPE_CHECKING:
    from kilin.tools._contract import ToolContext
    from kilin.trace_logging import TraceSink

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
    operation: str, optional
        ``"model"`` for the existing model-call plan or ``"network"``
        for explicitly approved bibliographic requests. Defaults to
        ``"model"`` and is omitted from serialized ordinary plans.

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
    operation: str = "model"

    def to_dict(self) -> dict[str, object]:
        """Return the plan as JSON-compatible data.

        Returns
        -------
        dict[str, object]
            Every attribute by name, with ``details`` as a list.

        """

        data = asdict(self)
        data["details"] = list(self.details)
        if self.operation == "model":
            data.pop("operation")
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
        KíLín's services; ``context.confirm`` asks the user.

    Returns
    -------
    tuple[PlanChoice, bool]
        The choice (``continue`` when nobody was asked; anything unexpected
        becomes ``cancel``) and whether the user was asked.

    """

    trace = context.trace
    interaction_id = uuid4().hex
    asked = context.confirm is not None and needs_confirmation(
        policy, plan, large_batches
    )
    trace.emit(
        "tool.plan.proposed",
        component="tool",
        data={
            **plan.to_dict(), "interaction_id": interaction_id,
            "choices": list(PLAN_CHOICE_CHOICES), "asked": asked,
            "policy": policy,
        },
        turn_id=context.trace_turn_id,
    )
    choice: PlanChoice = PLAN_CHOICE_CONTINUE
    answer = None
    if asked:
        try:
            context.progress(_WAITING_FOR_CHOICE, plan=plan.to_dict())
            answer = context.confirm(plan)
            choice = (
                answer if answer in PLAN_CHOICE_CHOICES else PLAN_CHOICE_CANCEL
            )
        except BaseException as error:
            trace.emit(
                "tool.plan.failed", component="tool", status="failed",
                data={"interaction_id": interaction_id,
                      "error_type": type(error).__name__,
                      "message": str(error)},
                turn_id=context.trace_turn_id,
            )
            raise

    trace.emit(
        "tool.plan.decided",
        component="tool",
        data={
            "interaction_id": interaction_id,
            "choice": choice, "asked": asked, "policy": policy,
            "response": answer,
        },
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
        KíLín's services; ``context.choose`` asks the user.

    Returns
    -------
    int | None
        The chosen option's index; None when the user cancels, the answer
        is not an option, or nobody can be asked (``context.choose`` is
        None).

    """

    def notify() -> None:
        """Display the question and options before waiting for the choice."""
        context.progress(
            _WAITING_FOR_CHOICE,
            choice={"question": question, "options": list(options)},
        )

    return _record_choice(
        question, options, context.choose, notify,
        trace=context.trace, turn_id=context.trace_turn_id,
        prefix="tool.choice", component="tool", allow_cancel=True,
    )


def _record_choice(
    question: str,
    options: tuple[str, ...],
    choose: Callable[[str, tuple[str, ...]], int | None] | None,
    notify: Callable[[], None],
    *,
    trace: TraceSink,
    turn_id: str | None,
    prefix: str,
    component: str,
    allow_cancel: bool,
) -> int | None:
    """Record one routing or tool choice at the actual callback boundary.

    Parameters
    ----------
    question: str
        Prompt shown to the caller.
    options: tuple[str, ...]
        Ordered option labels shown to the caller.
    choose: Callable | None
        Existing user-choice callback, or None for automatic behavior.
    notify: Callable[[], None]
        Existing presentation callback invoked before the choice.
    trace: TraceSink
        Owning conversation trace.
    turn_id: str | None
        Correlation identifier of the active turn.
    prefix: str
        Event namespace for routing or tool choices.
    component: str
        Emitting subsystem.
    allow_cancel: bool
        Whether the displayed prompt offers an explicit cancellation option.

    Returns
    -------
    int | None
        Valid selected index, or None when no option was selected.

    Notes
    -----
    Store the label as well as the position; replay must match the meaning,
    not reuse the original position. A proposal without a terminal record
    is incomplete. Automatic fallbacks are not user responses.

    """

    interaction_id = uuid4().hex
    asked = choose is not None and bool(options)
    trace.emit(
        f"{prefix}.proposed", component=component,
        data={
            "interaction_id": interaction_id, "question": question,
            "options": list(options), "asked": asked,
            "allow_cancel": allow_cancel,
        },
        turn_id=turn_id,
    )
    index = None
    answer = None
    if asked:
        try:
            notify()
            answer = choose(question, options)
            if type(answer) is int and 0 <= answer < len(options):
                index = answer
        except BaseException as error:
            trace.emit(
                f"{prefix}.failed", component=component, status="failed",
                data={"interaction_id": interaction_id,
                      "error_type": type(error).__name__,
                      "message": str(error)},
                turn_id=turn_id,
            )
            raise
    trace.emit(
        f"{prefix}.decided", component=component,
        data={
            "interaction_id": interaction_id, "index": index,
            "asked": asked, "response": answer,
            "selected_option": options[index] if index is not None else None,
        },
        turn_id=turn_id,
    )
    return index
