"""Define what KíLín gives a tool and what the tool returns.

A tool performs one explicit operation for an intent, such as listing every
mention of a term. An intent in a project type names its tool and settings
(``"tool": {"name": "scan_and_list", ...}``); the router selects the
intent, and KíLín runs the tool instead of retrieval and the answering
model. A tool receives the question as a ``ToolRequest`` and KíLín's
services as a ``ToolContext``, and returns a ``ToolResult``: the answer,
the evidence it cites, and a summary saved with the turn.

Tools never import ``kilin.rag``. Everything they may use (indexed
chunks, ranking, the router and answering models, progress, confirmation,
tracing) arrives through the context, so a tool can be tested with plain
functions in place of models. The context does not check what a tool does
with these services; a tool is trusted code registered in
``kilin.tools._registry``.

The settings checks raise ``ValueError`` with the intent location and the
field name, so a mistake in a project type file points at its line of
configuration. They check types and ranges only; each tool decides which
settings it has.

Key Features
------------

- **Request**

    The question and the user's own words to take terms from, with the
    earlier turns the question points back to.

- **Context**

    KíLín's services, including optional bounded retrieval, as plain
    callables; complete hybrid ranking remains unchanged.

- **Result**

    Answer, evidence fields, and a JSON-compatible summary.

- **Settings checks**

    Configuration errors that name the field, so typos are not silently
    ignored.

Module Structure
----------------

- ``PLUGIN_API_VERSION`` and ``ToolName``

    Independent API generation and extensible tool identifier type.

- ``Tool`` and ``ToolUse``

    A registration descriptor and an intent's validated settings.

- ``EarlierTurn``

    An earlier turn the question points back to.

- ``ToolRequest``

    What the user asked.

- ``ToolContext`` and ``ToolTrace``

    KíLín's services and the tool-facing trace interface.

- ``ToolResult``

    What a tool returns.

- ``ToolRouting``

    Routing a tool adds for one project, from the project's own files.

- ``reject_unknown``

    Reject settings a tool does not have.

- ``check_bool``

    Return a true/false setting, or its default.

- ``check_int``

    Return a whole-number setting within a range, or its default.

- ``check_choice``

    Return a setting that must be one of fixed values, or its default.

Examples
--------
>>> from kilin.tools import ToolRequest, check_int
>>> question = "list every mention of MRI"
>>> request = ToolRequest(question, question)
>>> check_int("code.intents.x.tool", {"max_calls": 5}, "max_calls", 10, 1, 50)
5

"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, TypeAlias

from kilin.tools._defaults import load_tool_defaults
from kilin.tools._model_calls import (
    ModelCallSettings,
    load_optional_model_call_settings,
)
from kilin.workspace_resources import WorkspaceResource

if TYPE_CHECKING:
    from kilin.chunking import Chunk
    from kilin.retrievals import CodeGraph, HybridResult
    from kilin.retrievals.requests import RetrievalRequest, RetrievalResult
    from kilin.tools._shared.planning import ToolPlan
    from kilin.question_suggestions import Suggestion

__all__ = [
    "PLUGIN_API_VERSION",
    "Tool",
    "ToolName",
    "ToolUse",
    "EarlierTurn",
    "ToolContext",
    "ToolRequest",
    "ToolTrace",
    "ToolResult",
    "ToolRouting",
    "check_bool",
    "check_choice",
    "check_int",
    "reject_unknown",
]


# The API generation does not track the KíLín distribution version.
PLUGIN_API_VERSION: Final[int] = 1
ToolName: TypeAlias = str


@dataclass(frozen=True, slots=True)
class EarlierTurn:
    """An earlier turn the question points back to.

    Attributes
    ----------
    number: int
        One-based turn number in the chat.
    question: str
        The question asked in that turn.
    paths: tuple[pathlib.Path, ...], optional
        Paths of the evidence that turn used, one per evidence passage.
        Defaults to ``()``.
    texts: tuple[str, ...], optional
        The evidence passages' texts, in the same order. Defaults to
        ``()``.
    lines: tuple[int, ...], optional
        The evidence passages' first lines, in the same order. Defaults to
        ``()``.
    linked: bool, optional
        True when the router related the question to this turn (or to a
        turn related to it); False for the last turn added only because
        "that paper" usually means the paper just discussed. Defaults to
        ``True``.
    answer: str, optional
        The answer given in that turn. Defaults to ``""``.

    """

    number: int
    question: str
    paths: tuple[Path, ...] = ()
    texts: tuple[str, ...] = ()
    lines: tuple[int, ...] = ()
    linked: bool = True
    answer: str = ""


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
    guidance: str | None, optional
        Answer guidance (such as a length target) for tools whose answer
        the model writes. Defaults to ``None``.
    earlier: tuple[EarlierTurn, ...], optional
        The earlier turns the question points back to, newest first,
        including the turns those turns pointed back to ("turn this into a
        briefing" after "summarize that" after "walk me through Smith's
        paper"). Defaults to ``()``.

    """

    question: str
    text: str
    guidance: str | None = None
    earlier: tuple[EarlierTurn, ...] = ()


class ToolTrace(Protocol):
    """The subset of the conversation trace supported for tool authors.

    The host owns the trace lifetime, persistence, and turn correlation.
    A disabled trace supports the same operations without writing events.

    """

    @property
    def include_content(self) -> bool:
        """Whether the user's trace policy permits content in diagnostics.

        Returns
        -------
        bool
            True when tool diagnostics may include source or prompt text.

        """

        ...

    def emit(
        self,
        event_type: str,
        *,
        component: str,
        data: Mapping[str, object] | None = None,
        status: str = "completed",
        turn_id: str | None = None,
    ) -> None:
        """Record an event through the host's trace sink.

        Parameters
        ----------
        event_type: str
            Nonempty, conventionally dot-separated event name.
        component: str
            Nonempty subsystem identifier, normally ``"tool"``.
        data: Mapping[str, object] | None, optional
            JSON-compatible diagnostic fields. Defaults to ``None``.
        status: str, optional
            Event status. Defaults to ``"completed"``.
        turn_id: str | None, optional
            Active turn identifier, normally ``context.trace_turn_id``.

        """

        ...

    def span(
        self,
        event_type: str,
        *,
        component: str,
        data: Mapping[str, object] | None = None,
        turn_id: str | None = None,
    ) -> AbstractContextManager[str]:
        """Record the lifetime of an operation and propagate its failures.

        Parameters
        ----------
        event_type: str
            Nonempty operation name, conventionally dot-separated.
        component: str
            Nonempty subsystem identifier, normally ``"tool"``.
        data: Mapping[str, object] | None, optional
            JSON-compatible diagnostic fields. Defaults to ``None``.
        turn_id: str | None, optional
            Active turn identifier, normally ``context.trace_turn_id``.

        Returns
        -------
        contextlib.AbstractContextManager[str]
            Context manager yielding the span identifier.

        """

        ...


@dataclass(frozen=True, slots=True)
class ToolContext:
    """KíLín's services, as a tool may use them.

    Attributes
    ----------
    project_root: pathlib.Path
        The project folder (``outputs/`` and ``templates/`` live here; see
        ``kilin.tools._shared.outputs`` and the report-summary tool).
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
    ask_model: Callable[[str, str, int | None], str]
        Asks the selected answering model with inherited temperature and
        the tool-owned MODEL_CALLS thinking policy. Returns only the
        parsed-answer text; any thinking is captured by the coordinator
        for display, not returned to the
        tool. ``None`` inherits the session maximum. Explicit caps apply
        only when effective thinking is off, never exceeding that maximum.
        A thinking call that exhausts its allowance inside unfinished
        thinking is retried once without thinking, with the same request
        and user allowance. Other failures, or a failed retry, raise
        ``IncompleteModelResponse`` rather than returning empty data.
        Independent item loops can retain checked results and report gaps.
    write: Callable[[str, str, int | None], str]
        Inherits session thinking and temperature, independently of the
        internal MODEL_CALLS policy. Preserves thinking in the returned
        answer and streams the complete reply to the user;
        a tool using it returns ``streamed=True``. Completion is validated
        before the reply can become a successful tool result. ``None``
        inherits the user limit; a smaller cap applies only without thinking.
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
    trace: ToolTrace
        The conversation's trace; emit events with ``trace_turn_id``.
    trace_turn_id: str | None
        The turn the tool runs for.
    write_pdf: Callable[..., int] | None, optional
        ``write_pdf(markdown, path, label=...)`` attempts a one-page PDF
        and returns the actual page count. ``None`` when no renderer is
        available (for example notebooks without Qt). Defaults to ``None``.
    retrieve: Callable[[RetrievalRequest], RetrievalResult] | None, optional
        Return bounded original evidence with source-first discovery and
        coverage reporting. None for contexts that provide ranking only.
        Tool-owned limits do not inherit answering-intent budgets.
    choose: Callable[[str, tuple[str, ...]], int | None] | None, optional
        ``choose(question, options)`` asks the user to pick one option and
        returns its index, or None when they cancel; None where nobody can
        be asked (notebooks), so tools fall back to their best guess.
        Defaults to ``None``.
    source_root: pathlib.Path | None, optional
        Saved project source directory. The selected-source reader rejects
        this path when it is outside the project workspace. Defaults to
        ``None`` for legacy test contexts.
    model_attempt_count: Callable[[], int] | None, optional
        Cumulative number of internal answering-model generation attempts,
        including incomplete attempts and bounded recovery, for this tool
        run. ``None`` when the context cannot expose attempt accounting.
        The count can be sampled around ``ask_model`` without an extra call.
    card: Callable[[pathlib.Path], Mapping[str, str] | None] | None, optional
        ``card(path)`` returns the fields of the indexed source's valid
        card by name: its metadata (``title``, ``authors``, ``year``,
        ``venue``, ``identifiers``) and statements (such as ``abstract``,
        ``aim``, ``methods``, ``findings``, ``limitations``), as indexed;
        None when the source has no valid card. Reading a card calls no
        model. None for contexts without cards. Defaults to ``None``.
    ask_check: Callable[[str, str, int], str] | None, optional
        Asks the selected answering model a short check
        ``(system, prompt, max_new_tokens)`` with thinking off and greedy
        decoding, such as whether the passages found answer the question;
        the conversation is untouched. None for contexts without it, where
        tools fall back to ``ask_router``. Defaults to ``None``.
    suggestions: Callable[[], Sequence[Suggestion]] | None, optional
        Returns the project's suggested questions, each with ``intent`` and
        ``question`` (``kilin.question_suggestions.Suggestion``), possibly
        several per intent, without those of the intent being answered;
        empty without sources. None for contexts without them. Defaults to
        ``None``.
    fixed_suggestions: tuple[str, ...], optional
        The project type's fixed example requests, for a project without
        sources. Defaults to ``()``.

    """

    project_root: Path
    chunks: Sequence[Chunk]
    graph: CodeGraph
    rank: Callable[[str], list[HybridResult]]
    retrieval_count: int
    unsearchable: Callable[[list[Path]], list[Path]]
    source_files: Callable[[], dict[Path, int]]
    ask_router: Callable[[str, str, int], str]
    ask_model: Callable[[str, str, int | None], str]
    write: Callable[[str, str, int | None], str]
    model_id: str
    progress: Callable[..., None]
    done: Callable[..., None]
    confirm: Callable[[ToolPlan], str] | None
    trace: ToolTrace
    trace_turn_id: str | None
    write_pdf: Callable[..., int] | None = None
    choose: Callable[[str, tuple[str, ...]], int | None] | None = None
    retrieve: Callable[[RetrievalRequest], RetrievalResult] | None = None
    model_attempt_count: Callable[[], int] | None = None
    source_root: Path | None = None
    card: Callable[[Path], Mapping[str, str] | None] | None = None
    ask_check: Callable[[str, str, int], str] | None = None
    suggestions: Callable[[], Sequence[Suggestion]] | None = None
    fixed_suggestions: tuple[str, ...] = ()


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
        Partial operations set ``status="partial"`` and ``incomplete_items``;
        operations without usable results set ``status="incomplete"``.
        An unassessed item is never a negative finding.
    streamed: bool, optional
        True when the answer already reached the user through
        ``ToolContext.write``. Defaults to ``False``.

    """

    answer: str
    evidence: tuple[dict[str, object], ...]
    summary: dict[str, object]
    streamed: bool = False


@dataclass(frozen=True, slots=True)
class ToolRouting:
    """Routing a tool adds to its intent for one project.

    A tool whose use depends on the project's own files (the report
    templates in ``templates/``) describes them for routing, so a new file
    is routed to without editing the project type.

    Attributes
    ----------
    cues: tuple[str, ...]
        Regular expressions (case-insensitive) added to the intent's cues.
    description: str
        Text added to the intent's description for the router, such as
        the templates the project has; ``""`` adds nothing.

    """

    cues: tuple[str, ...] = ()
    description: str = ""


@dataclass(frozen=True, slots=True)
class Tool:
    """Describe one tool for registration.

    Attributes
    ----------
    name: ToolName
        Name used in project types.
    summary: str
        What the tool does, for documentation and listings.
    parse_settings: Callable[[dict, str], object]
        Validates the tool's settings from a project type and returns a
        dataclass with JSON-compatible fields.
    run: Callable[[object, ToolRequest, ToolContext], ToolResult]
        Runs the tool with validated settings.
    follows_up: bool
        True when the tool works on an earlier answer, so a question
        routed to it always relates to an earlier turn.
    follows_up_when: Callable[[str], bool] | None
        For a tool that works on an earlier answer only for some
        questions: True for those questions, which then relate to an
        earlier turn like ``follows_up``. They are not ruled out for
        naming a file.
    routing: Callable[[object, Path, tuple[object, ...]], ToolRouting | None]
        Optional ``routing(settings, project_root, others)``: cues and a
        router description from the project's own files, added to the
        intent before routing (``with_project_routing``); ``others`` are
        the settings of the project type's other intents using the same
        tool, whose cases the routing can leave to them.
    model_calls: ModelCallSettings
        Tool-owned policy for internal answering-model calls, available
        after registration. Absent policies inherit the session.
    resource_root: pathlib.Path
        Package directory assigned by the registry, not by the author.
    workspace_resources: Callable | None
        Optional declaration of packaged resources for a new project.
    version: str
        Independent version of the tool. Declare it explicitly in a plugin;
        the constructor default is ``"1.0.0"``.
    api_version: int
        Plugin API generation implemented by the tool. Defaults to
        ``1``; this declaration is checked during registration.

    """

    name: ToolName
    summary: str
    parse_settings: Callable[[dict, str], object]
    run: Callable[[object, ToolRequest, ToolContext], ToolResult]
    follows_up: bool = False
    follows_up_when: Callable[[str], bool] | None = None
    routing: (
        Callable[[object, Path, tuple[object, ...]], ToolRouting | None]
        | None
    ) = None
    workspace_resources: (
        Callable[[object], tuple[WorkspaceResource, ...]] | None
    ) = None
    version: str = field(default="1.0.0", kw_only=True)
    api_version: int = field(default=1, kw_only=True)

    @property
    def resource_root(self) -> Path:
        """Return the package root assigned by the registry.

        Returns
        -------
        pathlib.Path
            Absolute directory containing this registered tool's resources.

        Raises
        ------
        RuntimeError
            If the descriptor has not been bound to an installed package.

        """

        raise RuntimeError(f"Tool {self.name!r} is not registered.")

    @property
    def model_calls(self) -> ModelCallSettings:
        """Return the registered tool's validated internal thinking policy.

        Returns
        -------
        ModelCallSettings
            Cached policy from this tool's own packaged defaults.

        """

        return load_optional_model_call_settings(
            self.resource_root / "defaults.json"
        )

    @property
    def internal_defaults(self) -> dict[str, object]:
        """Return the tool's packaged defaults other than ``SETTINGS``.

        These sections, such as ``MODEL_CALLS`` or ``ESTIMATES``, belong to
        the tool: project types and users cannot change them.

        Returns
        -------
        dict[str, object]
            The sections by name, empty when the tool has no defaults file.

        """

        path = self.resource_root / "defaults.json"
        if not path.is_file():
            return {}
        defaults = load_tool_defaults(path)
        defaults.pop("SETTINGS", None)
        return defaults


@dataclass(frozen=True, slots=True)
class ToolUse:
    """An intent's tool with its validated settings.

    Attributes
    ----------
    name: ToolName
        The tool.
    settings: object
        The tool's settings dataclass, such as ``ScanSettings``.

    """

    name: ToolName
    settings: object


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
