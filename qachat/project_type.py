"""Load project types from the JSON files in ``qachat/project_types/``.

A project type (for example a code repository or a literature collection)
decides which files a new project reads by default, which intents the
router may choose, and for each intent the instructions, history,
retrieval, or tool. Editing a JSON file changes behaviour without code
changes; restart QAChat afterwards. Saved conversations keep the system
prompt they were answered with.

Files are read once per folder and cached for the process. A malformed
file stops loading with a ``ValueError``; no type is skipped silently. This
module does not check that the built-in ``code`` and ``literature`` types
exist: ``qachat.project`` owns those kinds and checks them.

Key Features
------------

- **One file per type**

    ``code.json`` and ``literature.json`` sit in ``qachat/project_types/``,
    a folder that holds only data. The file name is the type's ID, and
    types are listed in file-name order. The built-in ``code`` and
    ``literature`` types must stay; add a file to add a type.

- **Tools by name**

    An intent may name a registered tool (``qachat.tools.registry``) with
    its settings; the tool validates them.

- **Validated on load**

    A malformed file raises ``ValueError`` naming the type and field.

Module Structure
----------------

- ``CODE_GRAPH_AUTO`` and ``CODE_GRAPH_CHOICES``

    Code-graph setting that builds the graph only for supported code, and
    the ordered collection of accepted code-graph settings.

- ``DOCUMENT_MAX_TOKENS``

    Default evidence budget for document reading, in estimated tokens.

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

- ``RETRIEVAL_STRATEGY_HYBRID``

    Canonical string value for the hybrid search pipeline.

- ``RETRIEVAL_STRATEGY_CHOICES``

    Ordered collection of accepted retrieval strategies.

- ``CodeGraphSetting``

    Type alias for ``True``, ``False``, or ``"auto"``.

- ``HistoryPolicy``

    Type alias for ``"none"``, ``"recent"``, or ``"relevant"``.

- ``Intent``

    One validated intent of a project type.

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
>>> from qachat.project_type import load_project_types, project_type
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

from qachat.tools.registry import ToolUse, parse_tool

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
    "CodeGraphSetting",
    "HistoryPolicy",
    "Intent",
    "ProjectType",
    "RetrievalMode",
    "RetrievalStrategy",
    "load_project_types",
    "project_type",
]

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
# Earlier turns kept as history when a type does not set recent_turns, and
# the most a type may set.
_DEFAULT_RECENT_TURNS: Final[int] = 2
_MAX_RECENT_TURNS: Final[int] = 20

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
# Evidence budget for document reading, in estimated tokens. Qwen3 models
# accept 40,960 tokens; this leaves room for instructions, history, and a
# 2,048-token answer.
DOCUMENT_MAX_TOKENS: Final[int] = 30_000
# Accepted range for an intent's own budget; values outside it are treated
# as mistakes in the file.
_MIN_DOCUMENT_TOKENS: Final[int] = 1_000
_MAX_DOCUMENT_TOKENS: Final[int] = 200_000

# Search pipeline used by RETRIEVAL_MODE_SEARCH. Only the existing hybrid
# pipeline (semantic, lexical, code graph, fused) exists; new strategies
# need benchmarks before they are added.
RetrievalStrategy: TypeAlias = Literal["hybrid"]
RETRIEVAL_STRATEGY_HYBRID: Final[RetrievalStrategy] = "hybrid"
RETRIEVAL_STRATEGY_CHOICES: Final[tuple[RetrievalStrategy, ...]] = (
    RETRIEVAL_STRATEGY_HYBRID,
)

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
    retrieval_strategy: "hybrid"
        Search pipeline used when searching; only ``hybrid`` exists.
    shortcuts: tuple[re.Pattern[str], ...]
        Case-insensitive patterns for whole messages that obviously have
        this intent, such as greetings; a match skips the router model.
    include_related_evidence: bool
        When searching, also supply the evidence cited by the earlier turns
        the router marked as related, so follow-ups keep their sources.
    max_tokens: int
        Evidence budget in estimated tokens for ``document`` reading; a
        longer document keeps its parts most relevant to the question.
    tool: ToolUse | None
        Tool that replaces retrieval and the answering model, if any (see
        ``qachat.tools``).

    """

    intent_id: str
    description: str
    system_prompt: str
    answer_guidance: str | None
    history_policy: HistoryPolicy
    retrieval_mode: RetrievalMode
    retrieval_strategy: RetrievalStrategy
    shortcuts: tuple[re.Pattern[str], ...] = ()
    include_related_evidence: bool = False
    max_tokens: int = DOCUMENT_MAX_TOKENS
    tool: ToolUse | None = None


@dataclass(frozen=True, slots=True)
class ProjectType:
    """One project type from ``qachat/project_types/<type>.json``.

    Attributes
    ----------
    kind: str
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
        Extra instruction the desktop app adds to each question; empty for
        none. Notebooks do not add it.
    welcome_prompts: tuple[str, ...]
        Headings the desktop app shows in an empty chat.
    suggestions: tuple[str, ...]
        Example requests the desktop app shows in an empty chat.

    """

    kind: str
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

    @property
    def label(self) -> str:
        """Alias of ``display_name``.

        Returns
        -------
        str
            Name shown when choosing a type.

        """

        return self.display_name


def _text(kind: str, data: dict, field: str, required: bool = True) -> str:
    """Return one text field, stripped.

    Parameters
    ----------
    kind: str
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
        raise ValueError(f"Project type {kind!r} needs text for {field!r}.")

    return value.strip()


def _texts(kind: str, data: dict, field: str) -> tuple[str, ...]:
    """Return a list of nonblank texts, stripped; missing reads as empty.

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
            f"Project type {kind!r} needs a list of text for {field!r}."
        )

    return tuple(item.strip() for item in value)


def _object(kind: str, data: dict, field: str) -> dict:
    """Return a nested JSON object; missing reads as empty.

    Raises
    ------
    ValueError
        If the value is not an object.

    """

    value = data.get(field, {})
    if not isinstance(value, dict):
        raise ValueError(
            f"Project type {kind!r} needs an object for {field!r}."
        )

    return value


def _parse_intent(kind: str, intent_id: object, data: object) -> Intent:
    """Validate one intent of a project type.

    Missing settings take their defaults: history policy ``relevant``,
    retrieval mode ``search`` (``none`` when a tool is named), strategy
    ``hybrid``, and budget ``DOCUMENT_MAX_TOKENS``.

    Parameters
    ----------
    kind: str
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
        invalid, or a tool is combined with a retrieval mode.

    """

    where = f"{kind}.intents.{intent_id}"
    if not isinstance(intent_id, str) or not _ID_PATTERN.fullmatch(
        intent_id
    ):
        raise ValueError(
            f"Intent ID {intent_id!r} in {kind!r} must be lowercase letters, "
            "digits, hyphens, or underscores, starting with a letter."
        )
    if not isinstance(data, dict):
        raise ValueError(f"Intent {where!r} must be an object.")

    history = _object(where, data, "history")
    retrieval = _object(where, data, "retrieval")

    policy = history.get("policy", HISTORY_POLICY_RELEVANT)
    if policy not in HISTORY_POLICY_CHOICES:
        raise ValueError(
            f"Intent {where!r} history policy must be one of: "
            f"{', '.join(HISTORY_POLICY_CHOICES)}."
        )

    # A tool replaces retrieval, so its intent defaults to no retrieval.
    tool = parse_tool(where, data["tool"]) if "tool" in data else None
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

    related = retrieval.get("include_related_evidence", False)
    if type(related) is not bool:
        raise ValueError(
            f"Intent {where!r} retrieval.include_related_evidence must be "
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

    shortcuts = []
    for pattern in _texts(where, data, "shortcuts"):
        try:
            shortcuts.append(re.compile(pattern, re.IGNORECASE))
        except re.error as error:
            raise ValueError(
                f"Intent {where!r} shortcut {pattern!r} is not a valid "
                f"regular expression: {error}"
            ) from error

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
        shortcuts=tuple(shortcuts),
        include_related_evidence=related,
        max_tokens=max_tokens,
        tool=tool,
    )


def _parse_project_type(kind: object, data: object) -> ProjectType:
    """Validate one project type file.

    Parameters
    ----------
    kind: object
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
        If the ID, a field, or an intent is invalid.

    """

    if not isinstance(kind, str) or not _ID_PATTERN.fullmatch(kind):
        raise ValueError(
            f"Project type ID {kind!r} must be lowercase letters, digits, "
            "hyphens, or underscores, starting with a letter."
        )
    if not isinstance(data, dict):
        raise ValueError(f"Project type {kind!r} must be an object.")

    extensions = _texts(kind, data, "extensions")
    if not extensions or not all(
        extension.startswith(".") and len(extension) > 1
        for extension in extensions
    ):
        raise ValueError(
            f"Project type {kind!r} needs file suffixes such as '.pdf'."
        )

    max_file_bytes = data.get("max_file_bytes")
    if type(max_file_bytes) is not int or max_file_bytes < 1:
        raise ValueError(
            f"Project type {kind!r} needs a positive whole number for "
            "'max_file_bytes'."
        )

    # The type check rejects 1 and 0, which equal True and False.
    code_graph = _object(kind, data, "capabilities").get(
        "code_graph", CODE_GRAPH_AUTO
    )
    if code_graph not in CODE_GRAPH_CHOICES or type(code_graph) not in (
        bool,
        str,
    ):
        raise ValueError(
            f"Project type {kind!r} capabilities.code_graph must be true, "
            'false, or "auto".'
        )

    recent_turns = _object(kind, data, "history").get(
        "recent_turns", _DEFAULT_RECENT_TURNS
    )
    if (
        type(recent_turns) is not int
        or not 0 <= recent_turns <= _MAX_RECENT_TURNS
    ):
        raise ValueError(
            f"Project type {kind!r} history.recent_turns must be a whole "
            f"number from 0 to {_MAX_RECENT_TURNS}."
        )

    raw_intents = _object(kind, data, "intents")
    if not raw_intents:
        raise ValueError(f"Project type {kind!r} needs at least one intent.")
    intents = {
        intent_id: _parse_intent(kind, intent_id, value)
        for intent_id, value in raw_intents.items()
    }
    default_intent = data.get("default_intent", next(iter(intents)))
    if default_intent not in intents:
        raise ValueError(
            f"Project type {kind!r} default_intent {default_intent!r} is not "
            "one of its intents."
        )

    return ProjectType(
        kind=kind,
        display_name=_text(kind, data, "display_name"),
        description=_text(kind, data, "description"),
        extensions=tuple(extension.lower() for extension in extensions),
        max_file_bytes=max_file_bytes,
        retrieval_instruction=_text(kind, data, "retrieval_instruction"),
        system_prompt=_text(kind, data, "system_prompt"),
        code_graph=code_graph,
        recent_turns=recent_turns,
        default_intent=default_intent,
        intents=intents,
        answer_guidance=_text(kind, data, "answer_guidance", required=False),
        welcome_prompts=_texts(kind, data, "welcome_prompts"),
        suggestions=_texts(kind, data, "suggestions"),
    )


def _unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build a JSON object, rejecting keys that appear twice.

    ``json`` keeps only the last of duplicate keys, which would silently
    drop an intent or setting.

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
        packaged ``qachat/project_types``.

    Returns
    -------
    dict[str, ProjectType]
        Types keyed by ID (the file name without ``.json``), in file-name
        order. The result is cached for the process.

    Raises
    ------
    ValueError
        If the folder has no project types, a file is not valid JSON, or a
        type is malformed. The built-in project kinds are checked by
        ``qachat.project``, which owns them.

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


def project_type(kind: str) -> ProjectType:
    """Return one packaged project type by ID.

    Parameters
    ----------
    kind: str
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
    if kind not in types:
        raise ValueError(
            f"Unknown project type {kind!r}. Choose one of: "
            f"{', '.join(types)}."
        )

    return types[kind]
