"""Load project types and their instructions from ``project_types.json``.

A project type (for example a code repository or a literature collection)
decides which files a new project reads by default and which instructions
the models receive. Editing the JSON file changes those instructions without
code changes; restart QAChat afterwards. Saved conversations keep the system
prompt they were answered with.

Key Features
------------

- **One file per installation**

    ``qachat/project_types.json`` lists every type. The built-in ``code`` and
    ``literature`` types must stay; more can be added.

- **Validated on load**

    A malformed file raises ``ValueError`` naming the type and field.

Module Structure
----------------

- ``ProjectType``

    One validated type.

- ``load_project_types``

    Read and validate the file, cached for the process.

- ``project_type``

    Look up one type by its ID.

Examples
--------
>>> from qachat.project_types import project_type
>>> project_type("code").display_name
'Code repository'

"""

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Final, Literal, TypeAlias

__all__ = [
    "PROJECT_TYPES_FILE",
    "Intent",
    "Workflow",
    "ProjectType",
    "load_project_types",
    "project_type",
]

PROJECT_TYPES_FILE = Path(__file__).with_name("project_types.json")
_KIND_PATTERN = re.compile(r"[a-z][a-z0-9_-]{0,31}")

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

# Search pipeline used by RETRIEVAL_MODE_SEARCH. Only the existing hybrid
# pipeline (semantic, lexical, code graph, fused) exists; new strategies
# need benchmarks before they are added.
RetrievalStrategy: TypeAlias = Literal["hybrid"]
RETRIEVAL_STRATEGY_HYBRID: Final[RetrievalStrategy] = "hybrid"
RETRIEVAL_STRATEGY_CHOICES: Final[tuple[RetrievalStrategy, ...]] = (
    RETRIEVAL_STRATEGY_HYBRID,
)

# Optional workflow an intent can declare. It replaces retrieval and the
# answering model for that intent; intents without one are unchanged.
WorkflowKind: TypeAlias = Literal["scan_and_list", "batch_classify"]
WORKFLOW_SCAN_AND_LIST: Final[WorkflowKind] = "scan_and_list"
WORKFLOW_BATCH_CLASSIFY: Final[WorkflowKind] = "batch_classify"
WORKFLOW_CHOICES: Final[tuple[WorkflowKind, ...]] = (
    WORKFLOW_SCAN_AND_LIST,
    WORKFLOW_BATCH_CLASSIFY,
)

# How a workflow collects the items it processes. Only exact mentions of
# the question's terms exist; whole-chunk collection comes with batched
# document review.
CandidateSource: TypeAlias = Literal["mentions"]
CANDIDATES_MENTIONS: Final[CandidateSource] = "mentions"
CANDIDATE_CHOICES: Final[tuple[CandidateSource, ...]] = (CANDIDATES_MENTIONS,)

# When a workflow that calls the model asks the user before running:
# always, only when its plan has more batches than ``large_batches``, or
# never.
ConfirmationPolicy: TypeAlias = Literal["always", "when_large", "never"]
CONFIRMATION_ALWAYS: Final[ConfirmationPolicy] = "always"
CONFIRMATION_WHEN_LARGE: Final[ConfirmationPolicy] = "when_large"
CONFIRMATION_NEVER: Final[ConfirmationPolicy] = "never"
CONFIRMATION_CHOICES: Final[tuple[ConfirmationPolicy, ...]] = (
    CONFIRMATION_ALWAYS,
    CONFIRMATION_WHEN_LARGE,
    CONFIRMATION_NEVER,
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
class Workflow:
    """Settings of an intent's workflow.

    Attributes
    ----------
    kind: WorkflowKind
        ``scan_and_list`` (exact mentions, no model) or ``batch_classify``
        (exact mentions, each classified by the answering model).
    case_sensitive: bool
        Whether letter case must match.
    whole_words: bool
        Whether a term must not continue into adjacent letters or digits.
    context_lines: int
        Lines of context kept around each mention (0-5); for
        ``batch_classify`` this is also what the model sees.
    max_listed: int
        Mentions listed with citations (1-1000); counts include all.
    candidates: CandidateSource
        ``batch_classify`` only: how items are collected; only
        ``mentions`` (every exact mention of the question's terms).
    categories: tuple[str, ...]
        ``batch_classify`` only: fixed categories; empty lets the router
        model propose them from the question (shown before running).
    batch_tokens: int
        ``batch_classify`` only: estimated tokens of mentions per model
        call (500-8000).
    max_model_calls: int
        ``batch_classify`` only: most model calls one answer may make
        (1-200); mentions beyond them are reported as not classified.
    confirmation: ConfirmationPolicy
        ``batch_classify`` only: when to ask before running.
    large_batches: int
        ``batch_classify`` only: with ``when_large``, plans with more
        batches than this ask first (0-100).

    """

    kind: WorkflowKind
    case_sensitive: bool = False
    whole_words: bool = True
    context_lines: int = 1
    max_listed: int = 100
    candidates: CandidateSource = CANDIDATES_MENTIONS
    categories: tuple[str, ...] = ()
    batch_tokens: int = 2500
    max_model_calls: int = 30
    confirmation: ConfirmationPolicy = CONFIRMATION_WHEN_LARGE
    large_batches: int = 2


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
    history_policy: HistoryPolicy
        ``none``, ``recent``, or ``relevant`` (recent plus related turns).
    retrieval_mode: RetrievalMode
        ``search``, ``document`` (one whole file or function), ``reuse``
        (the related turn's evidence), or ``none``.
    retrieval_strategy: RetrievalStrategy
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
    workflow: Workflow | None
        Workflow that replaces retrieval and the answering model, if any.

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
    workflow: Workflow | None = None


@dataclass(frozen=True, slots=True)
class ProjectType:
    """One project type from ``project_types.json``.

    Attributes
    ----------
    kind: str
        Identifier written to ``profile.ini``.
    display_name: str
        Name shown when choosing a type.
    description: str
        One sentence describing the sources; shown to the router.
    extensions: tuple[str, ...]
        File suffixes a new project reads by default.
    max_file_bytes: int
        Default size limit for one source file.
    retrieval_instruction: str
        Task description added to Qwen3-format embedding queries.
    system_prompt: str
        Base instructions for every answer; the intent's are added.
    code_graph: CodeGraphSetting
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
        """Alias of ``display_name``."""

        return self.display_name


def _text(kind: str, data: dict, field: str, required: bool = True) -> str:
    value = data.get(field, None if required else "")
    if not isinstance(value, str) or (required and not value.strip()):
        raise ValueError(f"Project type {kind!r} needs text for {field!r}.")
    return value.strip()


def _texts(kind: str, data: dict, field: str) -> tuple[str, ...]:
    value = data.get(field, [])
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(
            f"Project type {kind!r} needs a list of text for {field!r}."
        )
    return tuple(item.strip() for item in value)


def _object(kind: str, data: dict, field: str) -> dict:
    value = data.get(field, {})
    if not isinstance(value, dict):
        raise ValueError(
            f"Project type {kind!r} needs an object for {field!r}."
        )
    return value


def _parse_intent(kind: str, intent_id: object, data: object) -> Intent:
    where = f"{kind}.intents.{intent_id}"
    if not isinstance(intent_id, str) or not _KIND_PATTERN.fullmatch(
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
    workflow = _parse_workflow(where, data.get("workflow"))
    mode = retrieval.get(
        "mode",
        RETRIEVAL_MODE_NONE if workflow is not None else RETRIEVAL_MODE_SEARCH,
    )
    if workflow is not None and mode != RETRIEVAL_MODE_NONE:
        raise ValueError(
            f"Intent {where!r} has a workflow, which replaces retrieval; "
            'its retrieval mode must be "none".'
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
    if type(max_tokens) is not int or not 1_000 <= max_tokens <= 200_000:
        raise ValueError(
            f"Intent {where!r} retrieval.max_tokens must be a whole number "
            "from 1000 to 200000."
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
        # A workflow writes its own answer, so no instruction is needed.
        system_prompt=_text(
            where, data, "system_prompt", required=workflow is None
        ),
        answer_guidance=guidance.strip() if guidance is not None else None,
        history_policy=policy,
        retrieval_mode=mode,
        retrieval_strategy=strategy,
        shortcuts=tuple(shortcuts),
        include_related_evidence=related,
        max_tokens=max_tokens,
        workflow=workflow,
    )


def _parse_workflow(where: str, data: object) -> Workflow | None:
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError(f"Intent {where!r} workflow must be an object.")
    kind = data.get("kind")
    if kind not in WORKFLOW_CHOICES:
        raise ValueError(
            f"Intent {where!r} workflow kind must be one of: "
            f"{', '.join(WORKFLOW_CHOICES)}."
        )
    known = {"kind", "case_sensitive", "whole_words"}
    known |= {"context_lines", "max_listed"}
    if kind == WORKFLOW_BATCH_CLASSIFY:
        known |= {"candidates", "categories", "batch_tokens"}
        known |= {"max_model_calls"}
        known |= {"confirmation", "large_batches"}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(
            f"Intent {where!r} workflow has unknown settings: "
            f"{', '.join(unknown)}."
        )
    defaults = {field.name: field.default for field in fields(Workflow)}
    flags = {}
    for name in ("case_sensitive", "whole_words"):
        value = data.get(name, defaults[name])
        if type(value) is not bool:
            raise ValueError(
                f"Intent {where!r} workflow.{name} must be true or false."
            )
        flags[name] = value
    numbers = {}
    for name, low, high in (
        ("context_lines", 0, 5),
        ("max_listed", 1, 1000),
        ("batch_tokens", 500, 8000),
        ("max_model_calls", 1, 200),
        ("large_batches", 0, 100),
    ):
        value = data.get(name, defaults[name])
        if type(value) is not int or not low <= value <= high:
            raise ValueError(
                f"Intent {where!r} workflow.{name} must be a whole number "
                f"from {low} to {high}."
            )
        numbers[name] = value
    candidates = data.get("candidates", defaults["candidates"])
    if candidates not in CANDIDATE_CHOICES:
        raise ValueError(
            f"Intent {where!r} workflow.candidates must be one of: "
            f"{', '.join(CANDIDATE_CHOICES)}."
        )
    confirmation = data.get("confirmation", defaults["confirmation"])
    if confirmation not in CONFIRMATION_CHOICES:
        raise ValueError(
            f"Intent {where!r} workflow.confirmation must be one of: "
            f"{', '.join(CONFIRMATION_CHOICES)}."
        )
    categories = data.get("categories", [])
    if (
        not isinstance(categories, list)
        or len(categories) == 1
        or len(categories) > 8
        or not all(
            isinstance(item, str) and 0 < len(item.strip()) <= 40
            for item in categories
        )
        or len({item.strip().casefold() for item in categories})
        < len(categories)
    ):
        raise ValueError(
            f"Intent {where!r} workflow.categories must be 2-8 distinct "
            "labels of at most 40 characters, or empty."
        )
    return Workflow(
        kind=kind,
        **flags,
        **numbers,
        candidates=candidates,
        categories=tuple(item.strip() for item in categories),
        confirmation=confirmation,
    )


def _parse(kind: object, data: object) -> ProjectType:
    if not isinstance(kind, str) or not _KIND_PATTERN.fullmatch(kind):
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
    recent_turns = _object(kind, data, "history").get("recent_turns", 2)
    if type(recent_turns) is not int or not 0 <= recent_turns <= 20:
        raise ValueError(
            f"Project type {kind!r} history.recent_turns must be a whole "
            "number from 0 to 20."
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
    keys = [key for key, _ in pairs]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"Duplicate IDs: {', '.join(duplicates)}.")
    return dict(pairs)


@lru_cache(maxsize=None)
def load_project_types(
    path: Path = PROJECT_TYPES_FILE,
) -> dict[str, ProjectType]:
    """Read and validate every project type, in file order.

    Parameters
    ----------
    path: pathlib.Path, optional
        JSON file to read. Defaults to the packaged ``project_types.json``.

    Returns
    -------
    dict[str, ProjectType]
        Types keyed by ID. The result is cached for the process.

    Raises
    ------
    ValueError
        If the file is not valid JSON or a type is malformed. The built-in
        project kinds are checked by ``qachat.project``, which owns them.

    """

    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_unique_keys,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object of project types.")
    return {kind: _parse(kind, data) for kind, data in value.items()}


def project_type(kind: str) -> ProjectType:
    """Return one project type by ID.

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
