"""Load project types from the JSON files in ``kilin/project_types/``.

A project type (for example a code repository or a literature collection)
decides which files a new project reads by default, which intents the
router may choose, and for each intent the instructions, history,
retrieval, thinking override, or tool. Editing a JSON file changes
behaviour without code changes; restart KíLín afterwards. Saved conversations
keep the system prompt they were answered with.

Files are read once per folder and cached for the process. A malformed
file stops loading with a ``ValueError``; no type is skipped silently. This
module accepts any nonempty collection of valid definitions. The standard
``code`` and ``literature`` definitions are not mandatory.

Key Features
------------

- **One file per type**

    ``code.json`` and ``literature.json`` sit in ``kilin/project_types/``,
    a folder that holds only data. The file name is the type's ID, and
    types are listed in file-name order. Add or remove packaged definitions
    to specialize a distribution; at least one valid definition is required.

- **Tools by name**

    An intent may name a registered tool (``kilin.tools._registry``) with
    its settings; the tool validates them.

- **Request-local thinking**

    An ordinary intent may enable or disable thinking for its answer.
    An omitted or null setting inherits the configured chat setting.
    Tools own their model calls and cannot declare a thinking override.

- **Validated on load**

    A malformed file or an unknown field at any configuration level raises
    ``ValueError`` naming the type and field. Tool-specific settings are
    checked by their registered tool.

Module Structure
----------------

- ``CODE_GRAPH_AUTO`` and ``CODE_GRAPH_CHOICES``

    Code-graph setting that builds the graph only for supported code, and
    the ordered collection of accepted code-graph settings.

- ``DOCUMENT_MAX_TOKENS``

    Default evidence budget for document reading, taken from the packaged
    ``PROJECT_TYPES`` section when the intent JSON omits a budget.

- ``HISTORY_POLICY_NONE``, ``HISTORY_POLICY_RECENT``, and
  ``HISTORY_POLICY_RELEVANT``

    Canonical string values for which earlier turns an intent includes.

- ``HISTORY_POLICY_CHOICES``

    Ordered collection of accepted history policies.

- ``PROJECT_TYPES_DIR``

    Packaged folder with one JSON file per project type.

- ``RETRIEVAL_MODE_SEARCH``, ``RETRIEVAL_MODE_DOCUMENT``,
  ``RETRIEVAL_MODE_REUSE``, and ``RETRIEVAL_MODE_NONE``

    Canonical string values for how an intent obtains evidence.

- ``RETRIEVAL_MODE_CHOICES``

    Ordered collection of accepted retrieval modes.

- ``RETRIEVAL_STRATEGY_HYBRID`` and ``RETRIEVAL_STRATEGY_PAPERS``

    Canonical string values for the hybrid search pipeline and for the
    pipeline that chooses papers first.

- ``RETRIEVAL_STRATEGY_CHOICES``

    Ordered collection of accepted retrieval strategies.

- ``CodeGraphSetting``

    Type alias for ``True``, ``False``, or ``"auto"``.

- ``HistoryPolicy``

    Type alias for ``"none"``, ``"recent"``, or ``"relevant"``.

- ``Intent``

    One validated intent of a project type.

- ``ProjectTypeId``

    Open string alias for a registered project-type identifier.

- ``ProjectType``

    One validated project type.

- ``RetrievalMode``

    Type alias for ``"search"``, ``"document"``, ``"reuse"``, or
    ``"none"``.

- ``RetrievalStrategy``

    Type alias for ``"hybrid"``.

- ``load_project_types``

    Read and validate every project type in a folder, cached for the
    process.

- ``project_type``

    Look up one packaged project type by its ID.

Private helpers read and validate single fields; they are not part of the
public API.

Examples
--------
>>> from kilin.project_type import load_project_types, project_type
>>> list(load_project_types())
['code', 'literature']
>>> project_type("code").display_name
'Code repository'

"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Final, Literal, TypeAlias

from kilin.defaults import (
    load_defaults_section,
    require_default_int,
    require_default_nonnegative_int,
)
from kilin.repository import normalize_extensions
from kilin.tools._contract import reject_unknown
from kilin.tools._registry import ToolUse, parse_tool

__all__ = [
    "CODE_GRAPH_AUTO",
    "CODE_GRAPH_CHOICES",
    "DOCUMENT_MAX_TOKENS",
    "HISTORY_POLICY_CHOICES",
    "HISTORY_POLICY_NONE",
    "HISTORY_POLICY_RECENT",
    "HISTORY_POLICY_RELEVANT",
    "PROJECT_TYPES_DIR",
    "RETRIEVAL_MODE_CHOICES",
    "RETRIEVAL_MODE_DOCUMENT",
    "RETRIEVAL_MODE_NONE",
    "RETRIEVAL_MODE_REUSE",
    "RETRIEVAL_MODE_SEARCH",
    "RETRIEVAL_STRATEGY_CHOICES",
    "RETRIEVAL_STRATEGY_HYBRID",
    "RETRIEVAL_STRATEGY_PAPERS",
    "CodeGraphSetting",
    "HistoryPolicy",
    "Intent",
    "ProjectType",
    "ProjectTypeId",
    "RetrievalMode",
    "RetrievalStrategy",
    "load_project_types",
    "project_type",
]

ProjectTypeId: TypeAlias = str

# One JSON file per project type; the file name is the type's ID.
PROJECT_TYPES_DIR: Final[Path] = Path(__file__).parent / "project_types"
# Project type and intent IDs are short, lowercase, and safe in file names
# and in ``profile.ini``.
_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"[a-z][a-z0-9_-]{0,31}")

# Which earlier turns an intent includes as conversation history.
HistoryPolicy: TypeAlias = Literal["none", "recent", "relevant"]
HISTORY_POLICY_NONE: Final[HistoryPolicy] = "none"
HISTORY_POLICY_RECENT: Final[HistoryPolicy] = "recent"
HISTORY_POLICY_RELEVANT: Final[HistoryPolicy] = "relevant"
HISTORY_POLICY_CHOICES: Final[tuple[HistoryPolicy, ...]] = (
    HISTORY_POLICY_NONE,
    HISTORY_POLICY_RECENT,
    HISTORY_POLICY_RELEVANT,
)
# Package-level defaults apply only when the project-type JSON omits a
# value. The maximum below is a validation bound, not a configurable default.
_PROJECT_TYPE_DEFAULTS = load_defaults_section("PROJECT_TYPES")
_DEFAULT_RECENT_TURNS: Final[int] = require_default_nonnegative_int(
    _PROJECT_TYPE_DEFAULTS, "DEFAULT_RECENT_TURNS", "PROJECT_TYPES"
)
_MAX_RECENT_TURNS: Final[int] = 100
if _DEFAULT_RECENT_TURNS > _MAX_RECENT_TURNS:
    raise ValueError(
        "PROJECT_TYPES.DEFAULT_RECENT_TURNS cannot exceed 100."
    )

# How an intent obtains evidence for its answer.
# ``document`` reads one whole file or function in source order.
RetrievalMode: TypeAlias = Literal["search", "document", "reuse", "none"]
RETRIEVAL_MODE_SEARCH: Final[RetrievalMode] = "search"
RETRIEVAL_MODE_DOCUMENT: Final[RetrievalMode] = "document"
RETRIEVAL_MODE_REUSE: Final[RetrievalMode] = "reuse"
RETRIEVAL_MODE_NONE: Final[RetrievalMode] = "none"
RETRIEVAL_MODE_CHOICES: Final[tuple[RetrievalMode, ...]] = (
    RETRIEVAL_MODE_SEARCH,
    RETRIEVAL_MODE_DOCUMENT,
    RETRIEVAL_MODE_REUSE,
    RETRIEVAL_MODE_NONE,
)
# Default evidence budget for document reading, in estimated tokens.
# This leaves room for instructions, conversation history, and the answer.
DOCUMENT_MAX_TOKENS: Final[int] = require_default_int(
    _PROJECT_TYPE_DEFAULTS, "DOCUMENT_MAX_TOKENS", "PROJECT_TYPES"
)
# Accepted range for an intent's own budget, kept as validation policy.
_MIN_DOCUMENT_TOKENS: Final[int] = 1_000
_MAX_DOCUMENT_TOKENS: Final[int] = 200_000
if not _MIN_DOCUMENT_TOKENS <= DOCUMENT_MAX_TOKENS <= _MAX_DOCUMENT_TOKENS:
    raise ValueError(
        "PROJECT_TYPES.DOCUMENT_MAX_TOKENS must be between "
        "1000 and 200000."
    )

# Search pipeline used by RETRIEVAL_MODE_SEARCH: the hybrid pipeline
# (semantic, lexical, code graph, fused), or the same pipeline choosing
# papers first (``retrieval_papers.py``), for types that index papers.
# New strategies need benchmarks before they are added.
RetrievalStrategy: TypeAlias = Literal["hybrid", "papers"]
RETRIEVAL_STRATEGY_HYBRID: Final[RetrievalStrategy] = "hybrid"
RETRIEVAL_STRATEGY_PAPERS: Final[RetrievalStrategy] = "papers"
RETRIEVAL_STRATEGY_CHOICES: Final[tuple[RetrievalStrategy, ...]] = (
    RETRIEVAL_STRATEGY_HYBRID,
    RETRIEVAL_STRATEGY_PAPERS,
)
# Papers the ``papers`` strategy chooses by default, and the accepted range.
_DEFAULT_MAX_PAPERS: Final[int] = 3
_MAX_PAPERS_LIMIT: Final[int] = 50
# Most passages guaranteed to each chosen paper.
_PASSAGES_PER_PAPER_LIMIT: Final[int] = 10

# Whether a project type builds and searches the code graph.
CodeGraphSetting: TypeAlias = bool | Literal["auto"]
CODE_GRAPH_AUTO: Final[Literal["auto"]] = "auto"
CODE_GRAPH_CHOICES: Final[tuple[CodeGraphSetting, ...]] = (
    True,
    False,
    CODE_GRAPH_AUTO,
)


@dataclass(frozen=True, slots=True)
class Intent:
    """One intent a project type supports.

    Attributes
    ----------
    intent_id: str
        Identifier the router returns.
    description: str
        What the intent covers; shown to the router.
    system_prompt: str
        Instruction added to the project type's system prompt.
    answer_guidance: str | None
        Replaces the project type's answer guidance when set.
    history_policy: "none" | "recent" | "relevant"
        ``none``, ``recent``, or ``relevant`` (recent plus related turns).
    retrieval_mode: "search" | "document" | "reuse" | "none"
        ``search``, ``document`` (one whole file or function), ``reuse``
        (the related turn's evidence), or ``none``.
    retrieval_strategy: "hybrid" | "papers"
        Search pipeline used when searching: ``hybrid``, or ``papers``,
        which ranks papers first and fills the evidence from the best ones
        (only for project types with ``paper_index``).
    shortcuts: tuple[re.Pattern[str], ...]
        Case-insensitive patterns for whole messages that obviously have
        this intent, such as greetings; a match skips the router model.
    cues: tuple[re.Pattern[str], ...]
        Case-insensitive patterns searched in a question whose wording
        means this intent ("give me an overview of the Smith paper"); a
        match sets the intent after the router has judged the history.
    requires: tuple[re.Pattern[str], ...]
        Case-insensitive patterns of which a question must match one for
        the router's choice of this intent to stand (a reference check
        needs words such as "references" or "cite"). Empty: no condition.
    include_related_evidence: bool
        When searching, also supply the evidence cited by the earlier turns
        the router marked as related, so follow-ups keep their sources.
    prefer_source_code: bool
        When searching, rank documentation and notebook passages after
        source code unless the question asks about documentation (see
        ``kilin.retrievals.context_expansion.demote_documentation``).
    max_tokens: int
        Evidence budget in estimated tokens for ``document`` reading; a
        longer document keeps its parts most relevant to the question.
    tool: ToolUse | None
        Tool that replaces retrieval and the answering model, if any (see
        ``kilin.tools``).
    thinking: bool | None
        Per-answer thinking override for a compatible chat template.
        ``None`` inherits the session setting; ``True`` or ``False``
        overrides it without changing the session. Defaults to ``None``.
        Tool-backed intents must leave this unset or null.
    max_papers: int
        Papers the ``papers`` strategy fills the evidence from. Defaults to
        ``3``.
    passages_per_paper: int
        Passages the ``papers`` strategy guarantees to each chosen paper;
        zero guarantees one to the best paper only. Defaults to ``0``.

    """

    intent_id: str
    description: str
    system_prompt: str
    answer_guidance: str | None
    history_policy: HistoryPolicy
    retrieval_mode: RetrievalMode
    retrieval_strategy: RetrievalStrategy
    shortcuts: tuple[re.Pattern[str], ...] = ()
    cues: tuple[re.Pattern[str], ...] = ()
    requires: tuple[re.Pattern[str], ...] = ()
    include_related_evidence: bool = False
    prefer_source_code: bool = False
    max_tokens: int = DOCUMENT_MAX_TOKENS
    tool: ToolUse | None = None
    thinking: bool | None = None
    max_papers: int = _DEFAULT_MAX_PAPERS
    passages_per_paper: int = 0


@dataclass(frozen=True, slots=True)
class ProjectType:
    """One project type from ``kilin/project_types/<type>.json``.

    Attributes
    ----------
    id: ProjectTypeId
        Identifier written to ``profile.ini``.
    display_name: str
        Name shown when choosing a type.
    description: str
        One sentence describing the sources; shown to the router.
    extensions: tuple[str, ...]
        File suffixes a new project reads by default, in lowercase.
    max_file_bytes: int
        Default size limit for one source file.
    retrieval_instruction: str
        Task description added to Qwen3-format embedding queries.
    system_prompt: str
        Base instructions for every answer; the intent's are added.
    code_graph: bool | "auto"
        Build and search the code graph: true, false, or "auto" (only when
        selected files include code the graph understands).
    recent_turns: int
        Earlier turns kept as conversation history by default.
    default_intent: str
        Intent used when routing is off or its reply is unusable.
    intents: dict[str, Intent]
        Supported intents, in file order.
    answer_guidance: str
        Default answer instruction applied by the core to all callers
        unless overridden by the caller or selected intent; empty for none.
    welcome_prompts: tuple[str, ...]
        Headings the desktop app shows in an empty chat.
    suggestions: tuple[str, ...]
        Example requests the desktop app shows in an empty chat.
    source_cards_enabled: bool
        Enable the optional independent source-card index.
    source_card_kind: str | None
        Preferred card kind, or None for no default.
    source_cards_automatic: bool
        Generate missing cards during indexing.
    paper_index: bool
        Index documents as papers: search headers with each document's
        title, author, section and page, and paper cards that summarize
        each document. Defaults to ``False``.

    """

    id: ProjectTypeId
    display_name: str
    description: str
    extensions: tuple[str, ...]
    max_file_bytes: int
    retrieval_instruction: str
    system_prompt: str
    code_graph: CodeGraphSetting
    recent_turns: int
    default_intent: str
    intents: dict[str, Intent]
    answer_guidance: str = ""
    welcome_prompts: tuple[str, ...] = ()
    suggestions: tuple[str, ...] = ()
    paper_index: bool = False
    source_cards_enabled: bool = False
    source_card_kind: str | None = None
    source_cards_automatic: bool = False

    @property
    def label(self) -> str:
        """Alias of ``display_name``.

        Returns
        -------
        str
            Name shown when choosing a type.

        """

        return self.display_name


def _text(type_id: str, data: dict, field: str, required: bool = True) -> str:
    """Return one text field, stripped.

    Parameters
    ----------
    type_id: str
        Type or intent location, used in the error message.
    data: dict
        Parsed JSON object that holds the field.
    field: str
        Field name.
    required: bool, optional
        Whether the field must be present and nonblank. An optional field
        that is missing reads as empty text. Defaults to ``True``.

    Returns
    -------
    str
        Field text without surrounding whitespace.

    Raises
    ------
    ValueError
        If the value is not text, or is blank while required.

    """

    value = data.get(field, None if required else "")
    if not isinstance(value, str) or (required and not value.strip()):
        raise ValueError(f"Project type {type_id!r} needs text for {field!r}.")

    return value.strip()


def _patterns(
    where: str, data: dict, field: str
) -> tuple[re.Pattern[str], ...]:
    """Compile a list of case-insensitive regular expressions.

    Parameters
    ----------
    where: str
        Intent location for errors.
    data: dict
        The intent's JSON object.
    field: str
        Field name, such as ``"shortcuts"`` or ``"cues"``.

    Returns
    -------
    tuple[re.Pattern[str], ...]
        The compiled patterns; empty when the field is missing.

    Raises
    ------
    ValueError
        If the field is not a list of texts or a pattern is invalid.

    """

    patterns = []
    for pattern in _texts(where, data, field):
        try:
            patterns.append(re.compile(pattern, re.IGNORECASE))
        except re.error as error:
            raise ValueError(
                f"Intent {where!r} {field} pattern {pattern!r} is not a "
                f"valid regular expression: {error}"
            ) from error
    return tuple(patterns)


def _texts(type_id: str, data: dict, field: str) -> tuple[str, ...]:
    """Return a list of nonblank texts, stripped; missing reads as empty.

    Parameters
    ----------
    type_id: str
        Registered project-type identifier to validate.
    data: dict
        Structured input to inspect or forward.
    field: str
        Field name in the JSON configuration.

    Returns
    -------
    tuple[str, ...]
        a list of nonblank texts, stripped; missing reads as empty.

    Raises
    ------
    ValueError
        If the value is not a list of nonblank texts.

    """

    value = data.get(field, [])
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(
            f"Project type {type_id!r} needs a list of text for {field!r}."
        )

    return tuple(item.strip() for item in value)


def _object(type_id: str, data: dict, field: str) -> dict:
    """Return a nested JSON object; missing reads as empty.

    Parameters
    ----------
    type_id: str
        Registered project-type identifier to validate.
    data: dict
        Structured input to inspect or forward.
    field: str
        Field name in the JSON configuration.

    Returns
    -------
    dict
        a nested JSON object; missing reads as empty.

    Raises
    ------
    ValueError
        If the value is not an object.

    """

    value = data.get(field, {})
    if not isinstance(value, dict):
        raise ValueError(
            f"Project type {type_id!r} needs an object for {field!r}."
        )

    return value


def _parse_intent(type_id: str, intent_id: object, data: object) -> Intent:
    """Validate one intent of a project type.

    Missing settings take their defaults: history policy ``relevant``,
    retrieval mode ``search`` (``none`` when a tool is named), strategy
    ``hybrid``, budget ``DOCUMENT_MAX_TOKENS``, and thinking ``None``
    (inherit the configured chat setting for ordinary model answers).

    Parameters
    ----------
    type_id: str
        Project type ID, used in error messages.
    intent_id: object
        Key of the intent in the type's ``intents`` object.
    data: object
        Parsed JSON value of the intent.

    Returns
    -------
    Intent
        The validated intent.

    Raises
    ------
    ValueError
        If the ID, a field, a shortcut pattern, or the tool settings are
        invalid, an unknown field is present, or a tool is combined with
        a retrieval mode or an explicit thinking override.

    """

    where = f"{type_id}.intents.{intent_id}"
    if not isinstance(intent_id, str) or not _ID_PATTERN.fullmatch(intent_id):
        raise ValueError(
            f"Intent ID {intent_id!r} in {type_id!r} must be lowercase "
            "letters, digits, hyphens, or underscores, starting with a letter."
        )
    if not isinstance(data, dict):
        raise ValueError(f"Intent {where!r} must be an object.")

    reject_unknown(
        where,
        data,
        (
            "description",
            "system_prompt",
            "answer_guidance",
            "history",
            "retrieval",
            "shortcuts",
            "cues",
            "requires",
            "tool",
            "thinking",
        ),
    )
    history = _object(where, data, "history")
    reject_unknown(f"{where}.history", history, ("policy",))
    retrieval = _object(where, data, "retrieval")
    reject_unknown(
        f"{where}.retrieval",
        retrieval,
        (
            "mode",
            "strategy",
            "include_related_evidence",
            "prefer_source_code",
            "max_tokens",
            "max_papers",
            "passages_per_paper",
        ),
    )

    policy = history.get("policy", HISTORY_POLICY_RELEVANT)
    if policy not in HISTORY_POLICY_CHOICES:
        raise ValueError(
            f"Intent {where!r} history policy must be one of: "
            f"{', '.join(HISTORY_POLICY_CHOICES)}."
        )

    # A tool replaces retrieval, so its intent defaults to no retrieval.
    tool = parse_tool(where, data["tool"]) if "tool" in data else None
    thinking = data.get("thinking")
    if thinking is not None and type(thinking) is not bool:
        raise ValueError(
            f"Intent {where!r} thinking must be true, false, or null."
        )
    if tool is not None and thinking is not None:
        raise ValueError(
            f"Intent {where!r} thinking must be null or omitted when "
            "a tool writes the answer."
        )
    mode = retrieval.get(
        "mode",
        RETRIEVAL_MODE_NONE if tool is not None else RETRIEVAL_MODE_SEARCH,
    )
    if tool is not None and mode != RETRIEVAL_MODE_NONE:
        raise ValueError(
            f"Intent {where!r} has a tool, which replaces retrieval; its "
            'retrieval mode must be "none".'
        )
    if mode not in RETRIEVAL_MODE_CHOICES:
        raise ValueError(
            f"Intent {where!r} retrieval mode must be one of: "
            f"{', '.join(RETRIEVAL_MODE_CHOICES)}."
        )

    strategy = retrieval.get("strategy", RETRIEVAL_STRATEGY_HYBRID)
    if strategy not in RETRIEVAL_STRATEGY_CHOICES:
        raise ValueError(
            f"Intent {where!r} retrieval strategy must be one of: "
            f"{', '.join(RETRIEVAL_STRATEGY_CHOICES)}."
        )

    max_papers = retrieval.get("max_papers", _DEFAULT_MAX_PAPERS)
    if (
        type(max_papers) is not int
        or not 1 <= max_papers <= _MAX_PAPERS_LIMIT
    ):
        raise ValueError(
            f"Intent {where!r} retrieval.max_papers must be a whole number "
            f"from 1 to {_MAX_PAPERS_LIMIT}."
        )
    per_paper = retrieval.get("passages_per_paper", 0)
    if (
        type(per_paper) is not int
        or not 0 <= per_paper <= _PASSAGES_PER_PAPER_LIMIT
    ):
        raise ValueError(
            f"Intent {where!r} retrieval.passages_per_paper must be a whole "
            f"number from 0 to {_PASSAGES_PER_PAPER_LIMIT}."
        )

    related = retrieval.get("include_related_evidence", False)
    if type(related) is not bool:
        raise ValueError(
            f"Intent {where!r} retrieval.include_related_evidence must be "
            "true or false."
        )

    prefer_source = retrieval.get("prefer_source_code", False)
    if type(prefer_source) is not bool:
        raise ValueError(
            f"Intent {where!r} retrieval.prefer_source_code must be "
            "true or false."
        )

    max_tokens = retrieval.get("max_tokens", DOCUMENT_MAX_TOKENS)
    if (
        type(max_tokens) is not int
        or not _MIN_DOCUMENT_TOKENS <= max_tokens <= _MAX_DOCUMENT_TOKENS
    ):
        raise ValueError(
            f"Intent {where!r} retrieval.max_tokens must be a whole number "
            f"from {_MIN_DOCUMENT_TOKENS} to {_MAX_DOCUMENT_TOKENS}."
        )

    guidance = data.get("answer_guidance")
    if guidance is not None and not isinstance(guidance, str):
        raise ValueError(f"Intent {where!r} answer_guidance must be text.")

    shortcuts = _patterns(where, data, "shortcuts")

    return Intent(
        intent_id=intent_id,
        description=_text(where, data, "description"),
        # A tool writes its own answer, so no instruction is needed.
        system_prompt=_text(
            where, data, "system_prompt", required=tool is None
        ),
        answer_guidance=guidance.strip() if guidance is not None else None,
        history_policy=policy,
        retrieval_mode=mode,
        retrieval_strategy=strategy,
        shortcuts=shortcuts,
        cues=_patterns(where, data, "cues"),
        requires=_patterns(where, data, "requires"),
        include_related_evidence=related,
        prefer_source_code=prefer_source,
        max_tokens=max_tokens,
        tool=tool,
        thinking=thinking,
        max_papers=max_papers,
        passages_per_paper=per_paper,
    )


def _parse_project_type(type_id: object, data: object) -> ProjectType:
    """Validate one project type file.

    Parameters
    ----------
    type_id: object
        Project type ID (the file name without ``.json``).
    data: object
        Parsed JSON value of the file.

    Returns
    -------
    ProjectType
        The validated type. Its default intent is the first intent unless
        ``default_intent`` names another.

    Raises
    ------
    ValueError
        If the ID, a field, or an intent is invalid, including unknown
        configuration fields.

    """

    if not isinstance(type_id, str) or not _ID_PATTERN.fullmatch(type_id):
        raise ValueError(
            f"Project type ID {type_id!r} must be lowercase letters, digits, "
            "hyphens, or underscores, starting with a letter."
        )
    if not isinstance(data, dict):
        raise ValueError(f"Project type {type_id!r} must be an object.")

    reject_unknown(
        type_id,
        data,
        (
            "display_name",
            "description",
            "extensions",
            "max_file_bytes",
            "retrieval_instruction",
            "system_prompt",
            "capabilities",
            "history",
            "default_intent",
            "intents",
            "answer_guidance",
            "welcome_prompts",
            "suggestions",
        ),
    )
    extensions = _texts(type_id, data, "extensions")
    if not extensions or any(
        not extension.startswith(".") for extension in extensions
    ):
        raise ValueError(
            f"Project type {type_id!r} needs file suffixes such as '.pdf'."
        )
    try:
        extensions = normalize_extensions(extensions)
    except ValueError as error:
        raise ValueError(
            f"Project type {type_id!r} needs literal file suffixes: {error}"
        ) from error

    max_file_bytes = data.get("max_file_bytes")
    if type(max_file_bytes) is not int or max_file_bytes < 1:
        raise ValueError(
            f"Project type {type_id!r} needs a positive whole number for "
            "'max_file_bytes'."
        )

    # The type check rejects 1 and 0, which equal True and False.
    capabilities = _object(type_id, data, "capabilities")
    reject_unknown(
        f"{type_id}.capabilities", capabilities,
        ("code_graph", "paper_index", "source_cards"),
    )
    code_graph = capabilities.get("code_graph", CODE_GRAPH_AUTO)
    if code_graph not in CODE_GRAPH_CHOICES or type(code_graph) not in (
        bool,
        str,
    ):
        raise ValueError(
            f"Project type {type_id!r} capabilities.code_graph must be true, "
            'false, or "auto".'
        )

    paper_index = capabilities.get("paper_index", False)
    if type(paper_index) is not bool:
        raise ValueError(
            f"Project type {type_id!r} capabilities.paper_index must be true "
            "or false."
        )

    source_cards = _object(type_id, capabilities, "source_cards")
    reject_unknown(
        f"{type_id}.capabilities.source_cards", source_cards,
        ("enabled", "kind", "automatic"),
    )
    source_cards_enabled = source_cards.get("enabled", False)
    source_cards_automatic = source_cards.get(
        "automatic", source_cards_enabled
    )
    for setting in (source_cards_enabled, source_cards_automatic):
        if type(setting) is not bool:
            raise ValueError("Source-card project capability must be boolean.")
    source_card_kind = source_cards.get("kind")
    if source_card_kind is not None and (
        not isinstance(source_card_kind, str)
        or not re.fullmatch(r"[a-z][a-z0-9_]*", source_card_kind)
    ):
        raise ValueError("Source-card kind must be a snake-case identifier.")

    history = _object(type_id, data, "history")
    reject_unknown(f"{type_id}.history", history, ("recent_turns",))
    recent_turns = history.get("recent_turns", _DEFAULT_RECENT_TURNS)
    if (
        type(recent_turns) is not int
        or not 0 <= recent_turns <= _MAX_RECENT_TURNS
    ):
        raise ValueError(
            f"Project type {type_id!r} history.recent_turns must be a whole "
            f"number from 0 to {_MAX_RECENT_TURNS}."
        )

    raw_intents = _object(type_id, data, "intents")
    if not raw_intents:
        raise ValueError(
            f"Project type {type_id!r} needs at least one intent."
        )
    intents = {
        intent_id: _parse_intent(type_id, intent_id, value)
        for intent_id, value in raw_intents.items()
    }
    for intent in intents.values():
        if (
            intent.retrieval_strategy == RETRIEVAL_STRATEGY_PAPERS
            and not paper_index
        ):
            raise ValueError(
                f"Intent {type_id}.intents.{intent.intent_id} uses the "
                '"papers" retrieval strategy, which needs '
                "capabilities.paper_index."
            )
    default_intent = data.get("default_intent", next(iter(intents)))
    if default_intent not in intents:
        raise ValueError(
            f"Project type {type_id!r} default_intent "
            f"{default_intent!r} is not "
            "one of its intents."
        )

    return ProjectType(
        id=type_id,
        display_name=_text(type_id, data, "display_name"),
        description=_text(type_id, data, "description"),
        extensions=extensions,
        max_file_bytes=max_file_bytes,
        retrieval_instruction=_text(type_id, data, "retrieval_instruction"),
        system_prompt=_text(type_id, data, "system_prompt"),
        code_graph=code_graph,
        recent_turns=recent_turns,
        default_intent=default_intent,
        intents=intents,
        answer_guidance=_text(
            type_id, data, "answer_guidance", required=False
        ),
        welcome_prompts=_texts(type_id, data, "welcome_prompts"),
        suggestions=_texts(type_id, data, "suggestions"),
        paper_index=paper_index,
        source_cards_enabled=source_cards_enabled,
        source_card_kind=source_card_kind,
        source_cards_automatic=source_cards_automatic,
    )


def _unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build a JSON object, rejecting keys that appear twice.

    ``json`` keeps only the last of duplicate keys, which would silently
    drop an intent or setting.

    Parameters
    ----------
    pairs: list[tuple[str, object]]
        Object key/value pairs from the JSON decoder.

    Returns
    -------
    dict[str, object]
        Dictionary constructed from the unique JSON key/value pairs.

    Raises
    ------
    ValueError
        If a key appears more than once in the same object.

    """

    keys = [key for key, _ in pairs]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"Duplicate IDs: {', '.join(duplicates)}.")

    return dict(pairs)


@lru_cache(maxsize=None)
def load_project_types(
    directory: Path = PROJECT_TYPES_DIR,
) -> dict[str, ProjectType]:
    """Read and validate every project type in a folder.

    Parameters
    ----------
    directory: pathlib.Path, optional
        Folder with one ``<type>.json`` per project type. Defaults to the
        packaged ``kilin/project_types``.

    Returns
    -------
    dict[str, ProjectType]
        Types keyed by ID (the file name without ``.json``), in file-name
        order. The result is cached for the process.

    Raises
    ------
    ValueError
        If the folder has no project types, a file is not valid JSON, or a
        type contains invalid or unknown fields. Every JSON file is checked;
        loading stops at the first invalid definition. No particular type ID
        is required.

    """

    files = sorted(Path(directory).glob("*.json"))
    if not files:
        raise ValueError(f"{directory} has no project type files.")

    types: dict[str, ProjectType] = {}
    for path in files:
        try:
            value = json.loads(
                path.read_text(encoding="utf-8"),
                object_pairs_hook=_unique_keys,
            )
        except json.JSONDecodeError as error:
            raise ValueError(f"{path} is not valid JSON: {error}") from error
        if not isinstance(value, dict):
            raise ValueError(f"{path} must contain one project type object.")
        types[path.stem] = _parse_project_type(path.stem, value)

    return types


def project_type(type_id: ProjectTypeId) -> ProjectType:
    """Return one packaged project type by ID.

    Parameters
    ----------
    type_id: ProjectTypeId
        Project type ID, as written in ``profile.ini``.

    Returns
    -------
    ProjectType
        The validated type.

    Raises
    ------
    ValueError
        If no type has that ID.

    """

    types = load_project_types()
    if type_id not in types:
        raise ValueError(
            f"Unknown project type {type_id!r}. Choose one of: "
            f"{', '.join(types)}."
        )

    return types[type_id]
