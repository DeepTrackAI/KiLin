"""Load project types from the JSON files in ``kilin/project_types/``.

A project type (for example a code repository or a literature collection)
decides which files a new project reads by default, which intents the
router may choose, and for each intent the instructions, history,
retrieval, thinking override, memory policy, or tool. Editing these JSON
files changes behavior without code changes, including project-wide
conversation-memory defaults independent of source indexing and recent
conversation history. Restart KíLín afterwards. Saved conversations
keep the system prompt they were answered with.

The runtime catalog captures the raw and validated definitions once per
process. Invalid definitions remain visible with diagnostics, but cannot
be selected for execution. Explicit folder validation is strict and cached
separately. The standard ``code`` and ``literature`` types are not mandatory.

Key Features
------------

- **One file per type**

    ``code.json`` and ``literature.json`` sit in ``kilin/project_types/``,
    a folder that holds only data. The file name is the type's ID, and
    types are listed in file-name order. Add or remove packaged definitions
    to specialize a distribution. Runtime discovery retains invalid types
    as unavailable; explicit folder validation requires a nonempty folder.

- **Tools by name**

    An intent may name a registered tool (``kilin.tools._registry``) with
    its settings; the tool validates them.

- **New-project generation defaults**

    A type may set ``generation.thinking`` or
    ``generation.temperature`` for new project profiles. Omitted values
    inherit packaged defaults. Saved profiles and explicit creation
    settings take precedence.

- **Independent conversation-memory policy**

    Every project type defines its memory enablement, question/answer
    inclusion, retrieval limits, and disliked-history exclusion. Intents
    may restrict memory indexing to explicit selection, forbid it entirely,
    or disable reading memory. Memory remains distinct from source evidence.

- **Request-local thinking**

    An ordinary intent may enable or disable thinking for its answer.
    An omitted or null setting inherits the configured chat setting.
    Tools own their model calls and cannot declare a thinking override.

- **Validated on load**

    Structural validation is independent of tool availability. The
    runtime catalog then binds each tool reference and records failures
    with the type and field. The independent JSON schema version defaults
    to 1 for existing definitions.

Module Structure
----------------

- ``CODE_GRAPH_AUTO`` and ``CODE_GRAPH_CHOICES``

    Code-graph setting that builds the graph only for supported code, and
    the ordered collection of accepted code-graph settings.

- ``DOCUMENT_MAX_TOKENS``

    Default evidence budget for document reading, taken from the packaged
    ``PROJECT_TYPES`` section when the intent JSON omits a budget.

- ``MEMORY_QUESTION_CHOICES`` and ``MEMORY_ANSWER_CHOICES``

    Allowed project-wide policies for including questions and answers.

- ``MEMORY_INDEXING_CHOICES``

    Automatic, explicit, and never indexing policies for intents.

- ``HISTORY_POLICY_NONE``, ``HISTORY_POLICY_RECENT``, and
  ``HISTORY_POLICY_RELEVANT``

    Canonical string values for which earlier turns an intent includes.

- ``HISTORY_POLICY_CHOICES``

    Ordered collection of accepted history policies.

- ``PROJECT_TYPES_DIR``

    Packaged folder with one JSON file per project type.

- ``SUGGESTION_PLACEHOLDERS``

    Names a suggested-question template may fill in, such as ``{file}``.

- ``RETRIEVAL_MODE_SEARCH``, ``RETRIEVAL_MODE_DOCUMENT``,
  ``RETRIEVAL_MODE_REUSE``, and ``RETRIEVAL_MODE_NONE``

    Canonical string values for how an intent obtains evidence.

- ``RETRIEVAL_MODE_CHOICES``

    Ordered collection of accepted retrieval modes.

- ``RETRIEVAL_STRATEGY_HYBRID`` and
  ``RETRIEVAL_STRATEGY_SOURCE_FIRST``

    Canonical string values for hybrid and four-channel source-first
    retrieval.

- ``RETRIEVAL_STRATEGY_CHOICES``

    Ordered collection of accepted retrieval strategies.

- ``CodeGraphSetting``

    Type alias for ``True``, ``False``, or ``"auto"``.

- ``ConversationMemorySettings``

    Validated project-wide conversation-memory settings and limits.

- ``MemoryQuestionPolicy``, ``MemoryAnswerPolicy``, and
  ``MemoryIndexingPolicy``

    Type aliases for project-wide content and per-intent indexing policies.

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

    Type alias for ``"hybrid"`` or ``"source_first"``.

- ``PROJECT_TYPE_SCHEMA_VERSION``

    Supported project-type JSON schema version, independent of KíLín.

- ``ProjectTypeInfo``, ``project_type_catalog``, and ``project_type_raw``

    Inspect availability and captured source definitions.

- ``load_project_types``

    Return available runtime definitions, or strictly validate an explicit
    folder.

- ``project_type``

    Look up one packaged project type by its ID.

- ``with_project_routing``

    Add the routing that tools derive from one project's own files (its
    report templates) to a project type, before a question is routed.

Private helpers read and validate single fields; they are not part of the
public API.

Examples
--------
>>> from kilin.project_type import load_project_types, project_type
>>> bool(load_project_types())
True
>>> all(item.id == type_id for type_id, item in load_project_types().items())
True

"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import lru_cache
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re
from threading import RLock
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

from kilin.defaults import (
    load_defaults_section,
    require_default_int,
    require_default_nonnegative_int,
)
from kilin.plugins._paths import _runtime_directory
from kilin.retrievals.evidence_budget import (
    COVERAGE_MODES, EvidenceLimits,
)
from kilin.repository import normalize_extensions
from kilin.tools._contract import ToolUse, reject_unknown
from kilin.tools._registry import parse_tool, registered_tools

__all__ = [
    "CODE_GRAPH_AUTO",
    "CODE_GRAPH_CHOICES",
    "MEMORY_ANSWER_CHOICES",
    "MEMORY_INDEXING_CHOICES",
    "MEMORY_QUESTION_CHOICES",
    "DOCUMENT_MAX_TOKENS",
    "HISTORY_POLICY_CHOICES",
    "HISTORY_POLICY_NONE",
    "HISTORY_POLICY_RECENT",
    "HISTORY_POLICY_RELEVANT",
    "PROJECT_TYPES_DIR",
    "PROJECT_TYPE_SCHEMA_VERSION",
    "RETRIEVAL_MODE_CHOICES",
    "RETRIEVAL_MODE_DOCUMENT",
    "RETRIEVAL_MODE_NONE",
    "RETRIEVAL_MODE_REUSE",
    "RETRIEVAL_MODE_SEARCH",
    "RETRIEVAL_STRATEGY_CHOICES",
    "RETRIEVAL_STRATEGY_HYBRID",
    "RETRIEVAL_STRATEGY_SOURCE_FIRST",
    "SUGGESTION_PLACEHOLDERS",
    "CodeGraphSetting",
    "ConversationMemorySettings",
    "HistoryPolicy",
    "Intent",
    "MemoryAnswerPolicy",
    "MemoryIndexingPolicy",
    "MemoryQuestionPolicy",
    "ProjectType",
    "ProjectTypeId",
    "ProjectTypeInfo",
    "RetrievalMode",
    "RetrievalStrategy",
    "load_project_types",
    "project_type",
    "project_type_catalog",
    "project_type_raw",
    "with_project_routing",
]

ProjectTypeId: TypeAlias = str

# One JSON file per project type; the file name is the type's ID.
PROJECT_TYPES_DIR: Final[Path] = Path(__file__).parent / "project_types"
PROJECT_TYPE_SCHEMA_VERSION: Final[int] = 1
# Project type and intent IDs are short, lowercase, and safe in file names
# and in ``profile.ini``.
_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"[a-z][a-z0-9_-]{0,31}")
# What a suggested-question template may fill in: a source file name, a
# second, different one, a topic of the sources, an exact term in them, and
# a broad theme of the whole collection.
SUGGESTION_PLACEHOLDERS: Final[tuple[str, ...]] = (
    "file", "file2", "topic", "term", "theme",
)
_PLACEHOLDER: Final[re.Pattern[str]] = re.compile(r"\{([a-z0-9]+)\}")

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

# Search pipelines for RETRIEVAL_MODE_SEARCH. ``hybrid`` combines passage
# channels; ``source_first`` fuses four source rankings.
RetrievalStrategy: TypeAlias = Literal["hybrid", "source_first"]
RETRIEVAL_STRATEGY_HYBRID: Final[RetrievalStrategy] = "hybrid"
RETRIEVAL_STRATEGY_SOURCE_FIRST: Final[RetrievalStrategy] = "source_first"
RETRIEVAL_STRATEGY_CHOICES: Final[tuple[RetrievalStrategy, ...]] = (
    RETRIEVAL_STRATEGY_HYBRID,
    RETRIEVAL_STRATEGY_SOURCE_FIRST,
)
# Source-first defaults and validation bounds for selected sources.
_DEFAULT_MAX_SOURCES: Final[int] = 3
_MAX_SOURCES_LIMIT: Final[int] = 50
_PASSAGES_PER_SOURCE_LIMIT: Final[int] = 10

# Whether a project type builds and searches the code graph.
CodeGraphSetting: TypeAlias = bool | Literal["auto"]
CODE_GRAPH_AUTO: Final[Literal["auto"]] = "auto"
CODE_GRAPH_CHOICES: Final[tuple[CodeGraphSetting, ...]] = (
    True,
    False,
    CODE_GRAPH_AUTO,
)

# Conversation memory is separate from indexed source evidence and from
# recent conversation history. These policies configure future ingestion;
# they do not turn remembered model output into verifiable source evidence.
MemoryQuestionPolicy: TypeAlias = Literal["all", "none"]
MEMORY_QUESTION_CHOICES: Final[tuple[MemoryQuestionPolicy, ...]] = (
    "all", "none",
)
MemoryAnswerPolicy: TypeAlias = Literal["none", "selected", "all"]
MEMORY_ANSWER_CHOICES: Final[tuple[MemoryAnswerPolicy, ...]] = (
    "none", "selected", "all",
)
MemoryIndexingPolicy: TypeAlias = Literal[
    "automatic", "explicit", "never"
]
MEMORY_INDEXING_CHOICES: Final[tuple[MemoryIndexingPolicy, ...]] = (
    "automatic", "explicit", "never",
)
_MEMORY_MAX_HITS_LIMIT: Final[int] = 20
_MEMORY_MIN_TOKENS: Final[int] = 128
_MEMORY_MAX_TOKENS: Final[int] = 20_000


@dataclass(frozen=True, slots=True)
class ConversationMemorySettings:
    """Validated project-wide policy for independent conversation memory.

    Attributes
    ----------
    enabled: bool
        Whether long-term conversation-memory ingestion and search are on.
    questions: "all" | "none"
        Whether eligible user questions enter memory automatically.
    answers: "none" | "selected" | "all"
        Whether answers are forbidden, explicitly selected, or automatic.
    max_hits: int
        Maximum independently retrieved memory entries per answer.
    max_tokens: int
        Memory-only context budget; never consumes source-evidence budget.

    """

    enabled: bool
    questions: MemoryQuestionPolicy
    answers: MemoryAnswerPolicy
    max_hits: int
    max_tokens: int


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
    retrieval_strategy: "hybrid" | "source_first"
        Search pipeline used when searching: ``hybrid`` fuses passage
        ranks and ``source_first`` fuses independent source ranks.
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
    max_sources: int | None
        Source-first limit on represented sources; None defaults to 3.
    passages_per_source: int | None
        Source-first per-source target; None defaults to one passage.
    max_passages: int
        Hard number of source-first evidence passages.
    evidence_tokens: int
        Hard estimated source-first evidence-token budget including headers.
    coverage: "bounded" | "exhaustive"
        Whether ordinary questions require all indexed sources. Explicit
        questions asking about all sources always request exhaustive mode.
    memory_indexing: "automatic" | "explicit" | "never"
        Whether this intent can enter long-term memory automatically, only
        when selected, or never. Applies independently to questions/answers.
    memory_read: bool
        Whether this intent can retrieve previous conversation memory.

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
    max_sources: int | None = None
    passages_per_source: int | None = None
    max_passages: int = 8
    evidence_tokens: int = 5_000
    coverage: str = "bounded"
    memory_indexing: MemoryIndexingPolicy = "automatic"
    memory_read: bool = True

    def source_limits(self) -> EvidenceLimits:
        """Resolve the validated source-first limits for this intent.

        Returns
        -------
        EvidenceLimits
            Per-intent limits independent of the project retrieval count.

        """

        return EvidenceLimits(
            max_sources=(
                self.max_sources if self.max_sources is not None
                else _DEFAULT_MAX_SOURCES
            ),
            passages_per_source=(
                self.passages_per_source
                if self.passages_per_source is not None else 1
            ),
            max_passages=self.max_passages,
            max_tokens=self.evidence_tokens,
            coverage=self.coverage,
        )


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
    exclude_disliked_answers: bool
        Whether rejected answers are omitted from conversational context.
        Does not delete the original saved question or answer.
    conversation_memory: ConversationMemorySettings
        Independent memory settings and retrieval budget.
    default_intent: str
        Intent used when routing is off or its reply is unusable.
    intents: dict[str, Intent]
        Supported intents, in file order.
    answer_guidance: str
        Default answer instruction applied by the core to all callers
        unless overridden by the caller or selected intent; empty for none.
    welcome_prompts: tuple[str, ...]
        Headings the desktop app shows in an empty chat when there are no
        headings about the project's own sources.
    welcome_templates: tuple[str, ...]
        Fill-in headings, such as "What would you like to understand about
        {theme}?", from which ``kilin.question_suggestions`` writes
        headings about the project's own sources. Empty when the type has
        none.
    suggestions: tuple[str, ...]
        Ready-to-send requests the desktop app shows in an empty chat when
        there are too few questions about the project's own sources.
    suggestion_templates: tuple[tuple[str, str], ...]
        Intent ID and fill-in question pairs, in file order (an intent may
        have several), from which ``kilin.question_suggestions`` writes
        questions about the project's own sources, such as "Give me a
        short summary of {file}.". Empty when the type has none.
    source_cards_enabled: bool
        Enable the optional independent source-card index.
    source_card_kind: str | None
        Preferred card kind, or None for no default.
    source_cards_automatic: bool
        Generate missing cards during indexing.
    document_headers: bool
        Prefix searchable passages with document metadata, such as title,
        author, section, and page. Independent from source-card generation.
        Defaults to ``False``.
    generation_thinking: bool | None
        New-project thinking default, or None to use the packaged setting.
    generation_temperature: float | None
        New-project sampling temperature, or None to use the packaged value.

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
    exclude_disliked_answers: bool
    conversation_memory: ConversationMemorySettings
    default_intent: str
    intents: dict[str, Intent]
    answer_guidance: str = ""
    welcome_prompts: tuple[str, ...] = ()
    welcome_templates: tuple[str, ...] = ()
    suggestions: tuple[str, ...] = ()
    suggestion_templates: tuple[tuple[str, str], ...] = ()
    document_headers: bool = False
    source_cards_enabled: bool = False
    source_card_kind: str | None = None
    source_cards_automatic: bool = False
    generation_thinking: bool | None = None
    generation_temperature: float | None = None

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


def _suggestion_templates(
    type_id: str, data: dict, intents: dict[str, Intent]
) -> tuple[tuple[str, str], ...]:
    """Return the fill-in suggested questions, checked against the intents.

    Parameters
    ----------
    type_id: str
        Registered project-type identifier to validate.
    data: dict
        Project-type JSON object.
    intents: dict[str, Intent]
        The type's parsed intents.

    Returns
    -------
    tuple[tuple[str, str], ...]
        Intent ID and stripped template pairs, in file order; an intent
        with a list of templates gives one pair per template. Empty when
        the field is missing.

    Raises
    ------
    ValueError
        If the field is not an object, names an unknown intent, or has a
        value that is not nonblank text or a nonempty list of it, or a
        placeholder other than ``SUGGESTION_PLACEHOLDERS``.

    """

    field = "suggestion_templates"
    templates = _object(type_id, data, field)
    pairs: list[tuple[str, str]] = []
    for intent_id, value in templates.items():
        if intent_id not in intents:
            raise ValueError(
                f"Project type {type_id!r} {field} names unknown intent "
                f"{intent_id!r}."
            )
        values = value if isinstance(value, list) and value else [value]
        for template in values:
            _check_template(type_id, f"{field}.{intent_id}", template)
            pairs.append((intent_id, template.strip()))
    return tuple(pairs)


def _welcome_templates(type_id: str, data: dict) -> tuple[str, ...]:
    """Return the fill-in welcome headings, with their placeholders checked.

    Parameters
    ----------
    type_id: str
        Registered project-type identifier to validate.
    data: dict
        Project-type JSON object.

    Returns
    -------
    tuple[str, ...]
        Stripped templates, in file order; empty when the field is missing.

    Raises
    ------
    ValueError
        If the field is not a list of nonblank text, or a template has a
        placeholder other than ``SUGGESTION_PLACEHOLDERS``.

    """

    templates = _texts(type_id, data, "welcome_templates")
    for position, template in enumerate(templates):
        _check_template(type_id, f"welcome_templates[{position}]", template)
    return templates


def _check_template(type_id: str, where: str, template: object) -> None:
    """Check one fill-in suggested question.

    Parameters
    ----------
    type_id: str
        Registered project-type identifier to validate.
    where: str
        Location for errors, such as ``suggestion_templates.paper_overview``.
    template: object
        Value from the JSON file.

    Raises
    ------
    ValueError
        If the template is not nonblank text or uses a placeholder other
        than ``SUGGESTION_PLACEHOLDERS``.

    """

    if not isinstance(template, str) or not template.strip():
        raise ValueError(
            f"Project type {type_id!r} {where} must be nonblank text."
        )
    names = set(_PLACEHOLDER.findall(template))
    leftover = _PLACEHOLDER.sub("", template)
    if names - set(SUGGESTION_PLACEHOLDERS) or re.search(r"[{}]", leftover):
        raise ValueError(
            f"Project type {type_id!r} {where} may use only the "
            "placeholders "
            + ", ".join(f"{{{name}}}" for name in SUGGESTION_PLACEHOLDERS)
            + "."
        )


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


def _tool_reference(where: str, data: object) -> ToolUse:
    """Validate the declarative part of a tool reference.

    Parameters
    ----------
    where: str
        Intent location for diagnostics.
    data: object
        Raw tool reference from the JSON definition.

    Returns
    -------
    ToolUse
        Unbound reference carrying raw settings. Never execute it before
        binding to the registered tool and validating those settings.

    """

    if not isinstance(data, dict):
        raise ValueError(f"Intent {where!r} tool must be an object.")
    name = data.get("name")
    if not isinstance(name, str) or not re.fullmatch(
        r"[a-z][a-z0-9_]*", name
    ):
        raise ValueError(f"Intent {where!r} has an invalid tool name.")
    settings = {key: value for key, value in data.items() if key != "name"}
    return ToolUse(name, deepcopy(settings))


def _schema_version(data: dict) -> int:
    """Check the independently versioned project-type JSON schema.

    Parameters
    ----------
    data: dict
        Raw project-type object.

    Returns
    -------
    int
        Supported schema version; missing means the original version 1.

    """

    version = data.get("schema_version", 1)
    if type(version) is not int or version != PROJECT_TYPE_SCHEMA_VERSION:
        raise ValueError(
            f"Project type schema_version {version!r} is unsupported; "
            f"supported version is {PROJECT_TYPE_SCHEMA_VERSION}."
        )
    return version


def _parse_intent(
    type_id: str, intent_id: object, data: object, *,
    bind_tools: bool = True,
) -> Intent:
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
    bind_tools: bool, optional
        Validate settings against installed tools. False validates only
        the reference shape; the result must not be used for execution.

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
            "memory",
        ),
    )
    history = _object(where, data, "history")
    reject_unknown(f"{where}.history", history, ("policy",))
    memory = _object(where, data, "memory")
    reject_unknown(f"{where}.memory", memory, ("indexing", "read"))
    memory_indexing = memory.get("indexing", "automatic")
    if memory_indexing not in MEMORY_INDEXING_CHOICES:
        raise ValueError(
            f"Intent {where!r} memory.indexing must be one of: "
            f"{', '.join(MEMORY_INDEXING_CHOICES)}."
        )
    memory_read = memory.get("read", True)
    if type(memory_read) is not bool:
        raise ValueError(f"Intent {where!r} memory.read must be boolean.")
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
            "max_sources",
            "passages_per_source",
            "max_passages",
            "evidence_tokens",
            "coverage",
        ),
    )

    policy = history.get("policy", HISTORY_POLICY_RELEVANT)
    if policy not in HISTORY_POLICY_CHOICES:
        raise ValueError(
            f"Intent {where!r} history policy must be one of: "
            f"{', '.join(HISTORY_POLICY_CHOICES)}."
        )

    # A tool replaces retrieval, so its intent defaults to no retrieval.
    tool = None
    if "tool" in data:
        tool = (
            parse_tool(where, data["tool"]) if bind_tools
            else _tool_reference(where, data["tool"])
        )
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

    numeric_limits = (
        ("max_sources", 1, _MAX_SOURCES_LIMIT),
        ("passages_per_source", 1, _PASSAGES_PER_SOURCE_LIMIT),
        ("max_passages", 1, 200),
        ("evidence_tokens", 1000, _MAX_DOCUMENT_TOKENS),
    )
    for key, minimum, maximum in numeric_limits:
        if key not in retrieval:
            continue
        value = retrieval[key]
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(
                f"Intent {where!r} retrieval.{key} must be a whole "
                f"number from {minimum} to {maximum}."
            )
    source_only = (
        "max_sources", "passages_per_source", "max_passages",
        "evidence_tokens", "coverage",
    )
    if any(key in retrieval for key in source_only) and (
        strategy != RETRIEVAL_STRATEGY_SOURCE_FIRST
        or mode != RETRIEVAL_MODE_SEARCH
        or tool is not None
    ):
        raise ValueError(
            f"Intent {where!r} source-first evidence limits require "
            'retrieval.mode="search" and strategy="source_first".'
        )
    coverage = retrieval.get("coverage", "bounded")
    if coverage not in COVERAGE_MODES:
        raise ValueError(
            f"Intent {where!r} retrieval.coverage must be bounded "
            "or exhaustive."
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
        max_sources=retrieval.get("max_sources"),
        passages_per_source=retrieval.get("passages_per_source"),
        max_passages=retrieval.get("max_passages", 8),
        evidence_tokens=retrieval.get("evidence_tokens", 5000),
        coverage=coverage,
        memory_indexing=memory_indexing,
        memory_read=memory_read,
    )


def _parse_project_type(
    type_id: object, data: object, *, bind_tools: bool = True,
) -> ProjectType:
    """Validate one project type file.

    Parameters
    ----------
    type_id: object
        Project type ID (the file name without ``.json``).
    data: object
        Parsed JSON value of the file.
    bind_tools: bool, optional
        Resolve and validate tool settings. False checks the declarative
        schema without requiring the tools to be installed.

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

    _schema_version(data)
    reject_unknown(
        type_id,
        data,
        (
            "schema_version",
            "display_name",
            "description",
            "extensions",
            "max_file_bytes",
            "retrieval_instruction",
            "system_prompt",
            "capabilities",
            "history",
            "conversation_memory",
            "generation",
            "default_intent",
            "intents",
            "answer_guidance",
            "welcome_prompts",
            "welcome_templates",
            "suggestions",
            "suggestion_templates",
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
        ("code_graph", "document_headers", "source_cards"),
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

    document_headers = capabilities.get("document_headers", False)
    if type(document_headers) is not bool:
        raise ValueError(
            f"Project type {type_id!r}: capabilities.document_headers "
            "must be true "
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
    reject_unknown(
        f"{type_id}.history", history,
        ("recent_turns", "exclude_disliked_answers"),
    )
    if "exclude_disliked_answers" not in history:
        raise ValueError(
            f"Project type {type_id!r} requires "
            "history.exclude_disliked_answers."
        )
    exclude_disliked = history["exclude_disliked_answers"]
    if type(exclude_disliked) is not bool:
        raise ValueError(
            f"Project type {type_id!r} "
            "history.exclude_disliked_answers must be boolean."
        )
    recent_turns = history.get("recent_turns", _DEFAULT_RECENT_TURNS)
    if (
        type(recent_turns) is not int
        or not 0 <= recent_turns <= _MAX_RECENT_TURNS
    ):
        raise ValueError(
            f"Project type {type_id!r} history.recent_turns must be a whole "
            f"number from 0 to {_MAX_RECENT_TURNS}."
        )

    if "conversation_memory" not in data:
        raise ValueError(
            f"Project type {type_id!r} requires conversation_memory."
        )
    memory = _object(type_id, data, "conversation_memory")
    memory_fields = frozenset((
        "enabled", "questions", "answers", "max_hits", "max_tokens",
    ))
    reject_unknown(f"{type_id}.conversation_memory", memory,
                   memory_fields)
    if set(memory) != memory_fields:
        raise ValueError(
            f"Project type {type_id!r} conversation_memory requires "
            "enabled, questions, answers, max_hits, and max_tokens."
        )
    if type(memory["enabled"]) is not bool:
        raise ValueError(
            f"Project type {type_id!r} "
            "conversation_memory.enabled must be boolean."
        )
    if memory["questions"] not in MEMORY_QUESTION_CHOICES:
        raise ValueError(
            f"Project type {type_id!r} conversation_memory.questions "
            f"must be one of: {', '.join(MEMORY_QUESTION_CHOICES)}."
        )
    if memory["answers"] not in MEMORY_ANSWER_CHOICES:
        raise ValueError(
            f"Project type {type_id!r} conversation_memory.answers "
            f"must be one of: {', '.join(MEMORY_ANSWER_CHOICES)}."
        )
    for key, minimum, maximum in (
        ("max_hits", 1, _MEMORY_MAX_HITS_LIMIT),
        ("max_tokens", _MEMORY_MIN_TOKENS, _MEMORY_MAX_TOKENS),
    ):
        value = memory[key]
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(
                f"Project type {type_id!r} conversation_memory.{key} "
                f"must be a whole number from {minimum} to {maximum}."
            )
    memory_settings = ConversationMemorySettings(
        enabled=memory["enabled"],
        questions=memory["questions"],
        answers=memory["answers"],
        max_hits=memory["max_hits"],
        max_tokens=memory["max_tokens"],
    )

    generation = _object(type_id, data, "generation")
    reject_unknown(
        f"{type_id}.generation", generation,
        ("thinking", "temperature"),
    )
    thinking = generation.get("thinking")
    if "thinking" in generation and type(thinking) is not bool:
        raise ValueError(
            f"Project type {type_id!r} generation.thinking must be "
            "true or false."
        )
    temperature = generation.get("temperature")
    if "temperature" in generation:
        if type(temperature) not in (int, float):
            raise ValueError(
                f"Project type {type_id!r} generation.temperature must "
                "be a finite nonnegative number."
            )
        try:
            temperature = float(temperature)
        except OverflowError as error:
            raise ValueError(
                f"Project type {type_id!r} generation.temperature must "
                "be a finite nonnegative number."
            ) from error
        if not isfinite(temperature) or temperature < 0:
            raise ValueError(
                f"Project type {type_id!r} generation.temperature must "
                "be a finite nonnegative number."
            )

    raw_intents = _object(type_id, data, "intents")
    if not raw_intents:
        raise ValueError(
            f"Project type {type_id!r} needs at least one intent."
        )
    intents = {
        intent_id: _parse_intent(
            type_id, intent_id, value, bind_tools=bind_tools
        )
        for intent_id, value in raw_intents.items()
    }
    default_intent = data.get("default_intent", next(iter(intents)))
    if (not isinstance(default_intent, str)
            or default_intent not in intents):
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
        exclude_disliked_answers=exclude_disliked,
        conversation_memory=memory_settings,
        default_intent=default_intent,
        intents=intents,
        answer_guidance=_text(
            type_id, data, "answer_guidance", required=False
        ),
        welcome_prompts=_texts(type_id, data, "welcome_prompts"),
        welcome_templates=_welcome_templates(type_id, data),
        suggestions=_texts(type_id, data, "suggestions"),
        suggestion_templates=_suggestion_templates(type_id, data, intents),
        document_headers=document_headers,
        source_cards_enabled=source_cards_enabled,
        source_card_kind=source_card_kind,
        source_cards_automatic=source_cards_automatic,
        generation_thinking=thinking,
        generation_temperature=temperature,
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


@dataclass(frozen=True, slots=True)
class ProjectTypeInfo:
    """Describe a discovered project type without exposing mutable settings.

    Attributes
    ----------
    type_id: str
        Identifier derived from the JSON filename.
    path: pathlib.Path
        Resolved source of the captured definition, or its absolute path
        when resolution itself failed.
    origin: str
        Distribution or user-level source of the definition.
    schema_version: int | None
        Declared schema version, if readable and an integer.
    required_tools: tuple[str, ...]
        Tool IDs from the structurally valid definition, in sorted order.
    error: str | None
        Diagnostic if the type is unavailable; otherwise None.
    sha256: str | None
        Digest of the captured JSON bytes, if readable.
    bound: bool
        False for structural-only validation without tool resolution.

    """

    type_id: str
    path: Path
    origin: str
    schema_version: int | None
    required_tools: tuple[str, ...]
    error: str | None
    sha256: str | None = None
    bound: bool = True

    @property
    def available(self) -> bool:
        """Report whether the definition can be selected for execution."""

        return self.error is None and self.bound

    def identity(self) -> dict[str, object]:
        """Return portable provenance without an absolute user path."""

        return {
            "type_id": self.type_id, "origin": self.origin,
            "schema_version": self.schema_version, "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class _ProjectTypeRecord:
    """Own a catalog entry and its captured raw and parsed definitions.

    The registry never returns this private record or its parsed object
    directly. Tool settings can contain mutable JSON-compatible values.

    """

    info: ProjectTypeInfo
    raw: dict | None
    definition: ProjectType | None


def _read_type_json(path: Path, content: bytes | None = None) -> dict:
    """Read one JSON object with duplicate-key validation.

    Parameters
    ----------
    path: pathlib.Path
        Project-type definition file.
    content: bytes | None, optional
        Captured bytes, if already read by the catalog builder.

    Returns
    -------
    dict
        Parsed JSON object before tool binding.

    """

    try:
        value = json.loads(
            (path.read_bytes() if content is None else content).decode(
                "utf-8"
            ), object_pairs_hook=_unique_keys,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one project type object.")
    return value


def _bind_tools(definition: ProjectType) -> ProjectType:
    """Resolve every tool reference in a structurally validated type.

    Parameters
    ----------
    definition: ProjectType
        Structurally validated type with unbound tool references.

    Returns
    -------
    ProjectType
        Fully bound definition with validated settings.

    """

    names = {intent.tool.name for intent in definition.intents.values()
             if intent.tool is not None}
    if not names:
        return definition
    missing = names - registered_tools().keys()
    if missing:
        raise ValueError(
            "Required tools unavailable: " + ", ".join(sorted(missing))
        )
    intents = {}
    for key, intent in definition.intents.items():
        use = intent.tool
        if use is not None:
            use = parse_tool(
                f"{definition.id}.intents.{key}",
                {"name": use.name, **deepcopy(use.settings)},
            )
            intent = replace(intent, tool=use)
        intents[key] = intent
    return replace(definition, intents=intents)


def _type_record(
    path: Path, origin: str, *, bind_tools: bool = True,
) -> _ProjectTypeRecord:
    """Capture and validate one definition independently of other files.

    Parameters
    ----------
    path: pathlib.Path
        Candidate JSON file; its stem is the project type ID.
    origin: str
        Storage scope of the candidate.
    bind_tools: bool, optional
        Resolve tool references and settings after structural validation.

    Returns
    -------
    _ProjectTypeRecord
        Captured definition, raw JSON and individual diagnostic.

    """

    raw = None
    definition = None
    version = None
    required = ()
    error = None
    digest = None
    source = path.absolute()
    try:
        if origin == "user" and (path.is_symlink() or not path.is_file()):
            raise ValueError(f"Expected a regular JSON file: {path}.")
        if path.suffix != ".json":
            raise ValueError(f"Expected a .json project type: {path}.")
        source = path.resolve()
        content = source.read_bytes()
        digest = sha256(content).hexdigest()
        raw = _read_type_json(source, content)
        declared = raw.get("schema_version", 1)
        if type(declared) is int:
            version = declared
        structure = _parse_project_type(path.stem, raw, bind_tools=False)
        required = tuple(sorted({
            intent.tool.name for intent in structure.intents.values()
            if intent.tool is not None
        }))
        if bind_tools:
            definition = _bind_tools(structure)
    except (Exception, SystemExit) as failure:
        error = str(failure) or type(failure).__name__
    info = ProjectTypeInfo(
        path.stem, source, origin, version, required, error, digest,
        bind_tools,
    )
    return _ProjectTypeRecord(info, raw, definition)


def _build_project_type_catalog(
    directory: Path, origin: str = "distribution", *,
    reserved: frozenset[str] = frozenset(),
) -> dict[str, _ProjectTypeRecord]:
    """Discover each definition independently and capture diagnostics.

    Parameters
    ----------
    directory: pathlib.Path
        Folder containing JSON definitions.
    origin: str, optional
        Storage scope of discovered definitions.
    reserved: frozenset[str], optional
        IDs already owned by the distribution. Do not bind duplicates.

    Returns
    -------
    dict[str, _ProjectTypeRecord]
        Entries in deterministic filename order, including failures.

    """

    registered_tools()
    records = {}
    for path in sorted(Path(directory).glob("*.json")):
        if path.stem in reserved:
            info = ProjectTypeInfo(
                path.stem, path.absolute(), origin, None, (),
                f"Project type ID {path.stem!r} is already registered.",
            )
            record = _ProjectTypeRecord(info, None, None)
        else:
            record = _type_record(path, origin)
        records[path.stem] = record
    return records


_catalog: dict[str, _ProjectTypeRecord] | None = None
_catalog_lock = RLock()
_catalog_initializing = False
_rejected_types: tuple[ProjectTypeInfo, ...] = ()


def _runtime_catalog() -> dict[str, _ProjectTypeRecord]:
    """Initialize exactly one process-local catalog, on first use."""

    global _catalog, _catalog_initializing, _rejected_types
    # Tool discovery precedes type discovery, including lock acquisition.
    registered_tools()
    with _catalog_lock:
        if _catalog is None:
            if _catalog_initializing:
                raise RuntimeError("Recursive project-type initialization.")
            _catalog_initializing = True
            try:
                records = _build_project_type_catalog(PROJECT_TYPES_DIR)
                user_dir = _runtime_directory() / "project_types"
                rejected = []
                try:
                    if user_dir.is_symlink():
                        raise ValueError("Linked project types directory.")
                    # No directory is created by discovery. Access failures
                    # remain inspectable rather than hiding other types.
                    if user_dir.exists():
                        if not user_dir.is_dir():
                            raise ValueError("Expected a directory.")
                        list(user_dir.iterdir())
                        user_records = _build_project_type_catalog(
                            user_dir, "user", reserved=frozenset(records),
                        )
                        for name, record in user_records.items():
                            if name in records:
                                rejected.append(record.info)
                            else:
                                records[name] = record
                except (OSError, ValueError) as error:
                    rejected.append(ProjectTypeInfo(
                        "<project types directory>", user_dir, "user",
                        None, (), str(error),
                    ))
                _rejected_types = tuple(rejected)
                _catalog = records
            finally:
                _catalog_initializing = False
        return _catalog


def project_type_catalog() -> Mapping[str, ProjectTypeInfo]:
    """Inspect all discovered runtime types, including unavailable ones.

    Returns
    -------
    Mapping[str, ProjectTypeInfo]
        Read-only map of immutable diagnostics in discovery order.

    """

    return MappingProxyType({
        key: record.info for key, record in _runtime_catalog().items()
    })


def _project_type_infos() -> tuple[ProjectTypeInfo, ...]:
    """Return discovered entries and rejected ID conflicts for inspection."""

    return tuple(record.info for record in _runtime_catalog().values()) + (
        _rejected_types
    )


def _project_type_identity(type_id: str) -> dict[str, object] | None:
    """Return the captured project-type identity, if present.

    Parameters
    ----------
    type_id: str
        Selected project type identifier.

    Returns
    -------
    dict[str, object] | None
        Portable provenance from the catalog, or None for unknown types.

    """

    record = _runtime_catalog().get(type_id)
    return record.info.identity() if record is not None else None


def _available_record(type_id: ProjectTypeId) -> _ProjectTypeRecord:
    """Return a usable runtime entry or report why it cannot be selected.

    Parameters
    ----------
    type_id: ProjectTypeId
        Requested project type ID.

    Returns
    -------
    _ProjectTypeRecord
        Entry with both a raw and a fully bound definition.

    """

    catalog = _runtime_catalog()
    record = catalog.get(type_id) if isinstance(type_id, str) else None
    if record is None:
        available = (key for key, item in catalog.items()
                     if item.info.available)
        raise ValueError(
            f"Unknown project type {type_id!r}. Choose one of: "
            f"{', '.join(available) or '<none>'}."
        )
    if not record.info.available:
        raise ValueError(
            f"Project type {type_id!r} is unavailable: {record.info.error}"
        )
    return record


class _AvailableProjectTypes(Mapping[str, ProjectType]):
    """Read-only runtime mapping returning defensive definition copies.

    This is a view of one captured catalog, not a second registry. A
    caller may edit its own parsed intent dictionary or tool settings
    without changing the catalog or another caller's definition.

    """

    def __init__(self, records: dict[str, _ProjectTypeRecord]) -> None:
        """Capture the available entries of an already built catalog.

        Parameters
        ----------
        records: dict[str, _ProjectTypeRecord]
            Private runtime entries, not published directly to callers.

        """

        self._records = {
            key: value for key, value in records.items()
            if value.info.available
        }

    def __getitem__(self, key: str) -> ProjectType:
        """Return a copy of the validated definition.

        Parameters
        ----------
        key: str
            Available project type ID.

        Returns
        -------
        ProjectType
            Independent copy of the captured definition.

        """

        return deepcopy(self._records[key].definition)

    def __iter__(self) -> Iterator[str]:
        """Iterate available type IDs in discovery order."""

        return iter(self._records)

    def __len__(self) -> int:
        """Return the number of available types."""

        return len(self._records)

    def __contains__(self, key: object) -> bool:
        """Check availability without copying a parsed definition.

        Parameters
        ----------
        key: object
            Candidate type ID.

        Returns
        -------
        bool
            Whether the ID is present in this available catalog view.

        """

        return isinstance(key, str) and key in self._records


@lru_cache(maxsize=None)
def load_project_types(
    directory: Path | None = None,
) -> Mapping[str, ProjectType]:
    """List available runtime types or validate an explicit folder.

    Parameters
    ----------
    directory: pathlib.Path | None, optional
        When None, use the captured runtime catalog. An explicit folder is
        checked strictly and cached separately; it never registers types.

    Returns
    -------
    Mapping[str, ProjectType]
        Types keyed by ID in filename order. Runtime values are defensive
        copies; an explicit folder retains its original cached mapping.

    Raises
    ------
    ValueError
        If an explicit folder is empty or contains an invalid definition.
        Runtime discovery retains per-type diagnostics instead.

    Notes
    -----
    Clearing the function cache refreshes explicit folder validation, not
    the process-local runtime catalog. Restart to activate changes.

    """

    if directory is None:
        return _AvailableProjectTypes(_runtime_catalog())
    files = sorted(Path(directory).glob("*.json"))
    if not files:
        raise ValueError(f"{directory} has no project type files.")
    return {
        path.stem: _parse_project_type(path.stem, _read_type_json(path))
        for path in files
    }


def project_type(type_id: ProjectTypeId) -> ProjectType:
    """Return a defensive copy of one available runtime project type.

    Parameters
    ----------
    type_id: ProjectTypeId
        Project type ID, as written in ``profile.ini``.

    Returns
    -------
    ProjectType
        Fully validated type, independent of other callers' copies.

    Raises
    ------
    ValueError
        If the type is unknown or unavailable.

    """

    return deepcopy(_available_record(type_id).definition)


def project_type_raw(type_id: ProjectTypeId) -> dict:
    """Return the captured raw JSON of an available runtime type.

    Parameters
    ----------
    type_id: ProjectTypeId
        Project type ID, as written in ``profile.ini``.

    Returns
    -------
    dict
        Defensive copy of the original JSON for resolving overrides, not a
        fresh filesystem read. Changes cannot affect the runtime catalog.

    """

    return deepcopy(_available_record(type_id).raw)


def with_project_routing(config: ProjectType, root: Path) -> ProjectType:
    """Return a project type with the routing its tools derive per project.

    A registered tool with ``routing`` (``generate_paper_summary``, from
    the project's ``templates/``) adds cues and a router description to
    its intent, so a template file added to the project is routed to
    without editing the project type.

    Parameters
    ----------
    config: ProjectType
        The project's effective type.
    root: pathlib.Path
        The project's workspace.

    Returns
    -------
    ProjectType
        ``config`` itself when no tool adds routing; otherwise a copy whose
        intents carry the added cues (compiled case-insensitively, after
        their own) and description. A tool that fails to read the project
        adds nothing.

    """

    intents: dict[str, Intent] = {}
    changed = False
    for intent_id, intent in config.intents.items():
        tool = registered_tools()[intent.tool.name] if intent.tool else None
        routing = None
        if tool is not None and tool.routing is not None:
            others = tuple(
                other.tool.settings
                for other_id, other in config.intents.items()
                if other_id != intent_id and other.tool is not None
                and other.tool.name == intent.tool.name
            )
            try:
                routing = tool.routing(intent.tool.settings, root, others)
            except (OSError, ValueError):
                routing = None
        if routing is None or not (routing.cues or routing.description):
            intents[intent_id] = intent
            continue
        changed = True
        intents[intent_id] = replace(
            intent,
            cues=intent.cues + tuple(
                re.compile(cue, re.IGNORECASE) for cue in routing.cues
            ),
            description=(
                f"{intent.description} {routing.description}".strip()
            ),
        )
    return replace(config, intents=intents) if changed else config
