"""The contract between QAChat and its tools.

A tool performs one explicit operation for an intent, such as listing every
mention of a term. An intent in a project type names its tool and settings
(``"tool": {"name": "scan_and_list", ...}``); the router selects the
intent, and QAChat runs the tool instead of retrieval and the answering
model. A tool receives the question as a ``ToolRequest`` and QAChat's
services as a ``ToolContext``, and returns a ``ToolResult``: the answer,
the evidence it cites, and a summary saved with the turn.

Tools never import ``qachat.rag``. Everything they may use (indexed
chunks, ranking, the router and answering models, progress, confirmation,
tracing) arrives through the context, so a tool can be tested with plain
functions in place of models.

Key Features
------------

- **Request**

    The question and the user's own words to take terms from.

- **Context**

    QAChat's services, as plain callables.

- **Result**

    Answer, evidence fields, and a JSON-compatible summary.

- **Settings checks**

    ``check_bool``, ``check_int``, ``check_choice``, and
    ``reject_unknown`` give configuration errors that name the field.

Module Structure
----------------

- ``ToolRequest``
- ``ToolContext``
- ``ToolResult``
- ``check_bool``
- ``check_int``
- ``check_choice``
- ``reject_unknown``

"""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qachat.chunking import Chunk
    from qachat.retrievals import CodeGraph, HybridResult
    from qachat.tools.planning import ToolPlan
    from qachat.trace_logging import TraceSink

__all__ = [
    "EarlierTurn",
    "ToolContext",
    "ToolRequest",
    "ToolResult",
    "check_bool",
    "check_choice",
    "check_int",
    "reject_unknown",
]


@dataclass(frozen=True, slots=True)
class EarlierTurn:
    """An earlier turn the question points back to.

    Attributes
    ----------
    number: int
        One-based turn number in the chat.
    question: str
        The question asked in that turn.
    paths: tuple[pathlib.Path, ...]
        Paths of the evidence that turn used, one per evidence passage.

    """

    number: int
    question: str
    paths: tuple[Path, ...] = ()


@dataclass(frozen=True, slots=True)
class ToolRequest:
    """What the user asked.

    Attributes
    ----------
    question: str
        The question as typed.
    text: str
        The user's own words a tool may take values from: the question,
        followed by the questions of earlier turns it points back to.
    guidance: str | None
        Answer guidance (such as a length target) for tools whose answer
        the model writes.
    earlier: tuple[EarlierTurn, ...]
        The earlier turns the question points back to, newest first,
        including the turns those turns pointed back to ("turn this into a
        briefing" after "summarize that" after "walk me through Smith's
        paper").

    """

    question: str
    text: str
    guidance: str | None = None
    earlier: tuple[EarlierTurn, ...] = ()


@dataclass(frozen=True, slots=True)
class ToolContext:
    """QAChat's services, as a tool may use them.

    Attributes
    ----------
    project_root: pathlib.Path
        The project folder (``outputs/`` and ``templates/`` live here; see
        ``qachat.tools.outputs`` and ``qachat.tools.report_template``).
    chunks: Sequence[Chunk]
        Every indexed chunk, in index order.
    graph: CodeGraph
        The project's code graph (empty without code).
    rank: Callable[[str], list[HybridResult]]
        Ranks every chunk for a text with the fused search channels.
    retrieval_count: int
        The project's retrieval count, for "top results" decisions.
    unsearchable: Callable[[list[pathlib.Path]], list[pathlib.Path]]
        Given the indexed files, returns selected source files that have
        no indexed text (for coverage reports).
    source_files: Callable[[], dict[pathlib.Path, int]]
        Every selected source file (as in the source manifest) with its
        size in bytes.
    ask_router: Callable[[str, str, int], str]
        Asks the router model ``(system, prompt, max_new_tokens)`` with
        thinking off and greedy decoding; the conversation is untouched.
    ask_model: Callable[[str, str, int], str]
        Same for the answering model of the selected effort level.
    write: Callable[[str, str, int], str]
        Like ``ask_model``, but the reply streams to the user as the
        answer; a tool that uses it returns ``streamed=True``.
    model_id: str
        The answering model's ID, for plans and summaries.
    progress: Callable[..., None]
        Reports a step: ``progress(message, **details)``.
    done: Callable[..., None]
        Reports the result before the answer is shown:
        ``done(message, **details)``.
    confirm: Callable[[ToolPlan], str] | None
        Asks the user to confirm a plan and returns ``continue``,
        ``quick``, or ``cancel``; None runs every plan.
    write_pdf: Callable[..., int] | None
        ``write_pdf(markdown, path, label=...)`` writes a one-page PDF and
        returns its page count; None where no PDF writer is available
        (for example notebooks without Qt).
    choose: Callable[[str, tuple[str, ...]], int | None] | None
        ``choose(question, options)`` asks the user to pick one option and
        returns its index, or None when they cancel; None where nobody can
        be asked (notebooks), so tools fall back to their best guess.
    trace: TraceSink
        The conversation's trace; emit events with ``trace_turn_id``.
    trace_turn_id: str | None
        The turn the tool runs for.

    """

    project_root: Path
    chunks: Sequence[Chunk]
    graph: CodeGraph
    rank: Callable[[str], list[HybridResult]]
    retrieval_count: int
    unsearchable: Callable[[list[Path]], list[Path]]
    source_files: Callable[[], dict[Path, int]]
    ask_router: Callable[[str, str, int], str]
    ask_model: Callable[[str, str, int], str]
    write: Callable[[str, str, int], str]
    model_id: str
    progress: Callable[..., None]
    done: Callable[..., None]
    confirm: Callable[[ToolPlan], str] | None
    trace: TraceSink
    trace_turn_id: str | None
    write_pdf: Callable[..., int] | None = None
    choose: Callable[[str, tuple[str, ...]], int | None] | None = None


@dataclass(frozen=True, slots=True)
class ToolResult:
    """What a tool returns.

    Attributes
    ----------
    answer: str
        Markdown answer; ``[n]`` cites evidence ``n``.
    evidence: tuple[dict[str, object], ...]
        Evidence fields (``citation``, ``path``, ``start_line``,
        ``end_line``, ``symbol``, ``text``, ``cell``), numbered from one.
        ``text`` must be exact source lines so previews can locate it.
    summary: dict[str, object]
        JSON-compatible facts saved with the turn under ``"tool"``.
    streamed: bool
        True when the answer already reached the user through
        ``ToolContext.write``.

    """

    answer: str
    evidence: tuple[dict[str, object], ...]
    summary: dict[str, object]
    streamed: bool = False


def reject_unknown(where: str, data: dict, known: Collection[str]) -> None:
    """Reject settings a tool does not have.

    Parameters
    ----------
    where: str
        Location for the error, such as ``"code.intents.x.tool"``.
    data: dict
        The settings.
    known: Collection[str]
        Setting names the tool accepts.

    Raises
    ------
    ValueError
        Naming the unknown settings, so typos are not silently ignored.

    """

    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ValueError(
            f"{where!r} has unknown settings: {', '.join(unknown)}."
        )


def check_bool(where: str, data: dict, name: str, default: bool) -> bool:
    """Return a true/false setting, or its default.

    Parameters
    ----------
    where: str
        Location for the error.
    data: dict
        The settings.
    name: str
        Setting name.
    default: bool
        Value when the setting is absent.

    Returns
    -------
    bool
        The setting.

    Raises
    ------
    ValueError
        If the value is not true or false.

    """

    value = data.get(name, default)
    if type(value) is not bool:
        raise ValueError(f"{where!r} {name} must be true or false.")
    return value


def check_int(
    where: str, data: dict, name: str, default: int, low: int, high: int
) -> int:
    """Return a whole-number setting within a range, or its default.

    Parameters
    ----------
    where: str
        Location for the error.
    data: dict
        The settings.
    name: str
        Setting name.
    default: int
        Value when the setting is absent.
    low: int
        Smallest allowed value.
    high: int
        Largest allowed value.

    Returns
    -------
    int
        The setting.

    Raises
    ------
    ValueError
        If the value is not a whole number from ``low`` to ``high``.

    """

    value = data.get(name, default)
    if type(value) is not int or not low <= value <= high:
        raise ValueError(
            f"{where!r} {name} must be a whole number from {low} to {high}."
        )
    return value


def check_choice(
    where: str, data: dict, name: str, default: str, choices: Sequence[str]
) -> str:
    """Return a setting that must be one of fixed values, or its default.

    Parameters
    ----------
    where: str
        Location for the error.
    data: dict
        The settings.
    name: str
        Setting name.
    default: str
        Value when the setting is absent.
    choices: Sequence[str]
        Allowed values.

    Returns
    -------
    str
        The setting.

    Raises
    ------
    ValueError
        If the value is not one of ``choices``.

    """

    value = data.get(name, default)
    if value not in choices:
        raise ValueError(
            f"{where!r} {name} must be one of: {', '.join(choices)}."
        )
    return value
