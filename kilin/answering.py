"""Route questions by intent and select conversation history.

KíLín has one machine state for now, ``answering``. Within it, a router
model classifies each question into one of the current project type's
intents (see ``kilin/project_types/``). The configuration, not the
router, then decides the intent's instructions, which earlier turns are
included, and whether evidence is searched, reused, or skipped.

Earlier turns enter the prompt as question and answer text only; their
retrieved evidence stays with the turn it was retrieved for. Each turn keeps
a short representation (question, cited sources, first sentence of the
answer) so the router can point back to older turns without reading them.

The router's judgement of which turns a question relates to is checked
before it is used: a related turn is kept only when the question points
back (for example "it", "the second one", "you said") or shares a
distinctive term with that turn, and turns whose intent ignores the
conversation (small talk) are never related. The search query is always the
user's own words; when the question points back, the related turns'
questions are added to it, never model-written text.

Key Features
------------

- **Router decision**

    ``route_question`` asks the router model for one line of JSON and
    validates it; anything unusable falls back to the default intent.

- **Related-turn checks**

    ``route_question`` drops related turns the question gives no sign of
    referring to, and ``search_text`` builds the search from the user's
    own words. Malformed router fields fall back safely instead of
    interrupting answering.

- **History selection**

    ``select_history`` combines the most recent turns with turns the router
    marked as related, depending on the intent's history policy.

- **Turn representations**

    ``turn_representation`` builds a compact, model-free summary of a turn.
    Invalid saved representations are rebuilt from the original turn.

- **Packaged answering defaults**

    The router and history heuristic limits, short-question word count,
    reference-detection pattern, and common-word vocabulary come from the
    ``ANSWERING`` section of ``kilin_defaults.json`` at import time.
    See ``docs/package_defaults.md`` for every setting and its meaning.

Module Structure
----------------

- ``MACHINE_STATE_ANSWERING``

    Canonical string value ``"answering"`` for the only machine state.

- ``MACHINE_STATE_CHOICES``

    Ordered collection of the machine states.

- ``MachineState``

    Type alias for ``"answering"``.

- ``HISTORY_CITATION_MARKER``

    Text that replaces citation numbers in earlier answers.

- ``MAX_LISTED_TURNS``

    Most earlier turns the router sees, set by packaged defaults.

- ``MAX_RELATED_TURNS``

    Most earlier turns the router may mark as related, set by defaults.

- ``RouteDecision``

    What the router decided for one question.

- ``strip_thinking``

    Remove thinking text from model output.

- ``turn_representation``

    Summarize a turn compactly, without a model.

- ``side_reply``

    Ask a chat model a side question, then restore its conversation.

- ``route_question``

    Classify a question with the router model and check its related turns.

- ``refers_back``

    Tell whether a question's wording depends on earlier turns.

- ``search_text``

    Build the search query from the user's own words.

- ``select_history``

    Choose which earlier turns enter the prompt.

- ``history_messages``

    Return selected turns as user and assistant messages.

Private helpers build the router prompt, parse its reply, and compare
terms; they are not part of the public API.

Examples
--------
>>> from kilin.answering import RouteDecision, select_history
>>> decision = RouteDecision(
...     "question_answering", standalone=False, related_turn_ids=(2,)
... )
>>> select_history(5, "relevant", 2, decision)
[2, 4, 5]

"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import json
from pathlib import PureWindowsPath
import re
from typing import Final, Literal, Protocol, TypeAlias

from kilin.chat import ASSISTANT_ROLE, SYSTEM_ROLE, USER_ROLE
from kilin.defaults import (
    load_defaults_section,
    require_default_int,
    require_default_lowercase_words,
    require_default_pattern,
)
from kilin.project_type import (
    HISTORY_POLICY_NONE,
    HISTORY_POLICY_RECENT,
    RETRIEVAL_MODE_NONE,
    RETRIEVAL_MODE_REUSE,
    HistoryPolicy,
    ProjectType,
)
from kilin.tools._registry import TOOL_REGISTRY

__all__ = [
    "HISTORY_CITATION_MARKER",
    "MACHINE_STATE_ANSWERING",
    "MACHINE_STATE_CHOICES",
    "MAX_LISTED_TURNS",
    "MAX_RELATED_TURNS",
    "MachineState",
    "RouteDecision",
    "history_messages",
    "refers_back",
    "route_question",
    "search_text",
    "select_history",
    "side_reply",
    "strip_thinking",
    "turn_representation",
]

# The operational state that decides KíLín's execution flow. Only
# answering exists; later states (for example agentic workflows) would be
# added here, and no state transitions exist yet.
MachineState: TypeAlias = Literal["answering"]
MACHINE_STATE_ANSWERING: Final[MachineState] = "answering"
MACHINE_STATE_CHOICES: Final[tuple[MachineState, ...]] = (
    MACHINE_STATE_ANSWERING,
)
# Import-time configuration is shared with the rest of KíLín core.
_ANSWERING_DEFAULTS = load_defaults_section("ANSWERING")

# The router may point at only a few earlier turns.
MAX_RELATED_TURNS: Final[int] = require_default_int(
    _ANSWERING_DEFAULTS, "MAX_RELATED_TURNS", "ANSWERING"
)
# The router lists the newest eligible turns in chronological order.
MAX_LISTED_TURNS: Final[int] = require_default_int(
    _ANSWERING_DEFAULTS, "MAX_LISTED_TURNS", "ANSWERING"
)
# Token budget for the router's one-line JSON decision.
_ROUTER_MAX_TOKENS: Final[int] = require_default_int(
    _ANSWERING_DEFAULTS, "ROUTER_MAX_TOKENS", "ANSWERING"
)
# Maximum question and answer-summary characters in a turn representation.
_REPRESENTATION_MAX_CHARACTERS: Final[int] = require_default_int(
    _ANSWERING_DEFAULTS, "REPRESENTATION_MAX_CHARACTERS", "ANSWERING"
)
# Maximum source labels stored with a representation.
_REPRESENTATION_MAX_SOURCES: Final[int] = require_default_int(
    _ANSWERING_DEFAULTS, "REPRESENTATION_MAX_SOURCES", "ANSWERING"
)
# Short questions are treated as potentially dependent on earlier turns.
_SHORT_QUESTION_MAX_WORDS: Final[int] = require_default_int(
    _ANSWERING_DEFAULTS, "SHORT_QUESTION_MAX_WORDS", "ANSWERING"
)
# Citation syntax is part of the conversation representation contract.
_CITATION: Final[re.Pattern[str]] = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\]")
# A neutral marker replaces numbered citations whose evidence is not carried
# into the conversation-history prompt.
HISTORY_CITATION_MARKER: Final[str] = "[source]"

# Reference-detection cues are compiled with case-insensitive matching.
_REFERS_BACK: Final[re.Pattern[str]] = require_default_pattern(
    _ANSWERING_DEFAULTS, "REFERS_BACK_PATTERN", "ANSWERING", flags=re.I
)
# Common words are excluded from term matching but keep their configured
# vocabulary independent of Python source code.
_COMMON_WORDS: Final[frozenset[str]] = require_default_lowercase_words(
    _ANSWERING_DEFAULTS, "COMMON_WORDS", "ANSWERING"
)


class _Chat(Protocol):
    """The part of ``ChatSession`` that side questions use.

    Attributes
    ----------
    messages: list[dict[str, str]]
        The chat session's in-memory message history.
    thinking: bool
        Whether the session is configured for a thinking-capable response.
    temperature: float
        The generation temperature used by the session.

    """

    messages: list[dict[str, str]]
    thinking: bool
    temperature: float

    def restore_history(self, messages: list[dict[str, str]]) -> None:
        """Replace the in-memory conversation with ``messages``.

        Parameters
        ----------
        messages: list[dict[str, str]]
            Conversation messages replacing the current history.

        """

    def configure_generation(self, settings: dict[str, object]) -> None:
        """Update generation settings such as thinking and temperature.

        Parameters
        ----------
        settings: dict[str, object]
            Settings used to configure the operation.

        """

    def reply(
        self, message: str, *, max_new_tokens: int | None = None,
        language_context: Sequence[str] | None = None,
    ) -> str:
        """Generate, return, and retain one assistant reply.

        Parameters
        ----------
        message: str
            Message passed to the current session.
        max_new_tokens: int | None, optional
            Maximum generated tokens, or None for the default. Defaults to
            ``None``.
        language_context: Sequence[str] | None, optional
            Original user messages for script permission, or ``None`` for
            direct-chat history. Defaults to ``None``.

        Returns
        -------
        str
            Assistant reply generated for the supplied user message.

        """


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """What the router decided for one question.

    Attributes
    ----------
    intent: str
        Intent ID from the project type.
    standalone: bool
        True when the question can be understood without earlier turns.
    related_turn_ids: tuple[int, ...]
        One-based numbers of earlier turns the question refers to, after
        the related-turn checks.
    raw: str
        The router's reply, for tracing.
    fallback: str | None
        Why defaults replaced part of the reply, if they did.
    shortcut: str | None
        The intent shortcut pattern that matched, when the router model
        was skipped.
    refers_back: bool
        True when the question's wording points back to earlier turns.
    dropped_turns: tuple[tuple[int, str], ...]
        Related turns the router named but the checks removed, each with
        the reason, for tracing.

    """

    intent: str
    standalone: bool
    related_turn_ids: tuple[int, ...] = ()
    raw: str = ""
    fallback: str | None = None
    shortcut: str | None = None
    refers_back: bool = False
    dropped_turns: tuple[tuple[int, str], ...] = ()


def strip_thinking(text: str) -> str:
    """Remove <think> blocks, including one opened by the chat template.

    Parameters
    ----------
    text: str
        Model output.

    Returns
    -------
    str
        The output without thinking text, stripped of surrounding space.

    """

    text = re.sub(r"<think>.*?</think>", "", str(text), flags=re.S)
    # A template that opens <think> itself leaves only the closing tag.
    if "</think>" in text:
        text = text.split("</think>", 1)[1]

    return text.split("<think>", 1)[0].strip()


def _without_citations(text: str) -> str:
    """Remove citation groups such as ``[1]`` or ``[2, 3]``.

    Parameters
    ----------
    text: str
        Text to analyze or transform.

    Returns
    -------
    str
        Text with numbered citation groups removed.

    """

    return _CITATION.sub("", text)


def _neutral_citations(text: str) -> str:
    """Replace citation groups with ``HISTORY_CITATION_MARKER``.

    Parameters
    ----------
    text: str
        Text to analyze or transform.

    Returns
    -------
    str
        Text with citation groups replaced by the history marker.

    """

    return _CITATION.sub(f" {HISTORY_CITATION_MARKER}", text)


def turn_representation(turn: object, number: int) -> dict[str, object]:
    """Summarize a turn compactly, without a model.

    Parameters
    ----------
    turn: object
        A conversation turn with ``question``, ``answer``, and ``evidence``.
    number: int
        One-based position of the turn in its conversation.

    Returns
    -------
    dict[str, object]
        ``turn_id``, ``question``, ``sources`` (cited files and symbols),
        and ``summary`` (the answer's first sentence).

    """

    answer = _without_citations(strip_thinking(getattr(turn, "answer", "")))
    first = re.split(r"(?<=[.!?])\s+", " ".join(answer.split()), maxsplit=1)

    # Only cited evidence is listed; an answer without citations lists all.
    cited = {
        int(citation)
        for group in re.findall(
            r"\[(\d+(?:\s*,\s*\d+)*)\]", str(getattr(turn, "answer", ""))
        )
        for citation in re.split(r"\s*,\s*", group)
    }
    sources: list[str] = []
    for item in getattr(turn, "evidence", ()) or ():
        if cited and getattr(item, "citation", None) not in cited:
            continue
        # Evidence paths can originate on a different operating system.
        name = PureWindowsPath(str(getattr(item, "path", ""))).name
        symbol = getattr(item, "symbol", None)
        label = f"{name}:{symbol}" if symbol else name
        if label and label not in sources:
            sources.append(label)

    question = " ".join(str(getattr(turn, "question", "")).split())
    return {
        "turn_id": number,
        "question": question[:_REPRESENTATION_MAX_CHARACTERS],
        "sources": sources[:_REPRESENTATION_MAX_SOURCES],
        "summary": (
            first[0][:_REPRESENTATION_MAX_CHARACTERS]
            if first and first[0]
            else ""
        ),
    }


def _representation_line(representation: dict[str, object]) -> str:
    """Write a turn representation as one line of the router prompt.

    Parameters
    ----------
    representation: dict[str, object]
        Stored summary of a conversation turn.

    Returns
    -------
    str
        Single-line router prompt text for this conversation turn.

    """

    sources = ", ".join(representation.get("sources") or []) or "none"
    return (
        f"Turn {representation['turn_id']}: Q: {representation['question']}"
        f" | Sources: {sources} | A: {representation.get('summary', '')}"
    )


def side_reply(
    chat: _Chat,
    system: str,
    prompt: str,
    *,
    max_new_tokens: int,
    token_callback: Callable[[str], None] | None = None,
    language_context: Sequence[str] = (),
) -> str:
    """Ask a chat model a side question, then restore its conversation.

    Parameters
    ----------
    chat: ChatSession
        Model session to ask; its history and settings are restored.
    system: str
        System instruction for the side question only.
    prompt: str
        The side question.
    max_new_tokens: int
        Maximum length of the reply.
    token_callback: Callable[[str], None] | None, optional
        Receives the reply as it is generated, for a side answer shown to
        the user. Defaults to ``None``.
    language_context: Sequence[str], optional
        Original user messages authorizing CJK output, oldest first. The
        generated side prompt cannot authorize scripts. Defaults to ``()``.

    Returns
    -------
    str
        The reply with thinking text removed.

    Notes
    -----
    Thinking is off and sampling is greedy for the side question.

    """

    history = [dict(message) for message in chat.messages]
    previous = {"thinking": chat.thinking, "temperature": chat.temperature}

    # Restore the conversation and settings even when the reply fails.
    try:
        chat.restore_history([{"role": SYSTEM_ROLE, "content": system}])
        chat.configure_generation({"thinking": False, "temperature": 0.0})
        options: dict[str, object] = {
            "max_new_tokens": max_new_tokens,
            "language_context": language_context,
        }
        if token_callback is not None:
            options["token_callback"] = token_callback
        reply = chat.reply(prompt, **options)
    finally:
        chat.restore_history(history)
        chat.configure_generation(previous)

    return strip_thinking(reply)


def _router_prompt(
    project: ProjectType,
    representations: list[dict[str, object]],
    question: str,
) -> tuple[str, str]:
    """Build the router's system and user prompts.

    Parameters
    ----------
    project: ProjectType
        Current project type with its intents.
    representations: list[dict[str, object]]
        One representation per earlier turn; those marked ``skip`` (small
        talk) are left out, and only the last ``MAX_LISTED_TURNS`` are
        listed.
    question: str
        The new question.

    Returns
    -------
    tuple[str, str]
        System prompt and user prompt. When the type has an intent that
        reuses evidence and earlier turns are listed, the user prompt ends
        with a structural example for the reuse intent and last listed turn.

    """

    intents = "\n".join(
        f"- {intent_id}: {intent.description}"
        for intent_id, intent in project.intents.items()
    )
    listed = [item for item in representations if not item.get("skip")]
    listed = listed[-MAX_LISTED_TURNS:]
    turns = (
        "\n".join(_representation_line(item) for item in listed)
        if listed
        else "none"
    )
    system = (
        "You route questions for KíLín, which answers from a "
        f"{project.display_name.lower()}: {project.description} You never "
        "answer questions. You reply with one line of JSON only."
    )
    user = f"""Machine state: {MACHINE_STATE_ANSWERING}

Intents:
{intents}

Earlier turns:
{turns}

New question: {question}

Return one line of JSON with:
"intent": one intent ID from the list; when unsure whether a message is
small talk, choose another intent,
"standalone": true if the question is understandable without earlier turns;
a question that names its own topic is standalone even after other turns. A
request about the form of the last answer ("shorter", "in two sentences",
"as a list", "in simpler words") is not standalone: it refers to the last
turn.
Only when standalone is false, also add:
"related_turns": numbers of the earlier turns it refers to.

Example: {{"intent": "{project.default_intent}", "standalone": true}}"""
    reuse = next(
        (
            intent_id
            for intent_id, intent in project.intents.items()
            if intent.retrieval_mode == RETRIEVAL_MODE_REUSE
        ),
        None,
    )
    if reuse and listed:
        last = listed[-1].get("turn_id", len(listed))
        user += f"""
Example for intent "{reuse}" referring to turn {last}: {{"intent": \
"{reuse}", "standalone": false, "related_turns": [{last}]}}"""
    return system, user


def _parse_route(
    raw: str, project: ProjectType, turn_count: int
) -> RouteDecision:
    """Validate the router's reply.

    Parameters
    ----------
    raw: str
        The router's reply; the first ``{...}`` in it is read as JSON.
    project: ProjectType
        Current project type with its intents.
    turn_count: int
        Number of earlier turns, which bounds the related turn numbers.

    Returns
    -------
    RouteDecision
        The decision. An unknown intent becomes the default intent, a
        missing ``standalone`` is true only for a first question, related
        turn numbers outside the conversation are dropped, and at most the
        newest ``MAX_RELATED_TURNS`` are kept. Malformed field types do not
        raise exceptions. ``fallback`` says what was replaced.

    """

    default = RouteDecision(
        intent=project.default_intent,
        standalone=turn_count == 0,
        raw=raw,
    )
    match = re.search(r"\{.*\}", raw, flags=re.S)
    try:
        value = json.loads(match.group(0)) if match else None
    except ValueError:
        value = None
    if not isinstance(value, dict):
        return RouteDecision(
            **{**_fields(default), "fallback": "reply was not JSON"}
        )

    fallback = None
    intent = value.get("intent")
    if not isinstance(intent, str):
        fallback = "intent was not a string"
        intent = project.default_intent
    elif intent not in project.intents:
        fallback = f"unknown intent {intent!r}"
        intent = project.default_intent
    standalone = value.get("standalone")
    if not isinstance(standalone, bool):
        if "standalone" in value and fallback is None:
            fallback = "standalone was not a boolean"
        standalone = turn_count == 0

    related_values = value.get("related_turns", [])
    if related_values is None:
        related_values = []
    if not isinstance(related_values, list):
        return RouteDecision(
            **{
                **_fields(default),
                "fallback": "related_turns was not a list",
            }
        )

    related: list[int] = []
    for item in related_values:
        if type(item) is int and 1 <= item <= turn_count:
            if item not in related:
                related.append(item)

    return RouteDecision(
        intent=intent,
        standalone=standalone and not related,
        related_turn_ids=tuple(sorted(related)[-MAX_RELATED_TURNS:]),
        raw=raw,
        fallback=fallback,
    )


def _fields(decision: RouteDecision) -> dict[str, object]:
    """Return the fields a checked decision copies from the router's.

    Parameters
    ----------
    decision: RouteDecision
        Router decision whose fields are extracted.

    Returns
    -------
    dict[str, object]
        the fields a checked decision copies from the router's.

    """

    return {
        "intent": decision.intent,
        "standalone": decision.standalone,
        "related_turn_ids": decision.related_turn_ids,
        "raw": decision.raw,
    }


def route_question(
    chat: _Chat,
    project: ProjectType,
    turns: list[object],
    question: str,
) -> RouteDecision:
    """Classify a question with the router model.

    Parameters
    ----------
    chat: ChatSession
        Router model session; its conversation is restored afterwards.
    project: ProjectType
        Current project type with its intents.
    turns: list
        Earlier conversation turns.
    question: str
        The new question.

    Returns
    -------
    RouteDecision
        Validated decision. Errors and unusable replies fall back to the
        project type's default intent with no related turns.

    Notes
    -----
    The router's related turns are checked: a turn is kept only when the
    question points back (``refers_back``) or shares a distinctive term
    with the turn's question, sources, or summary. Turns whose intent
    neither reads history nor retrieves (small talk) are not listed to the
    router and are never related. A question left with no related turns
    is standalone.

    """

    # Obvious messages such as greetings skip the router model.
    message = re.sub(r"[\s!?.,]+$", "", question.strip())
    for intent_id, intent in project.intents.items():
        for pattern in intent.shortcuts:
            if pattern.fullmatch(message):
                return RouteDecision(
                    intent=intent_id,
                    standalone=True,
                    shortcut=pattern.pattern,
                )

    representations = [
        {
            **_stored_representation(turn, index + 1),
            "skip": _ignores_conversation(turn, project),
        }
        for index, turn in enumerate(turns)
    ]
    pointing = refers_back(question)
    system, prompt = _router_prompt(project, representations, question)

    try:
        raw = side_reply(
            chat, system, prompt, max_new_tokens=_ROUTER_MAX_TOKENS
        )
    except Exception as error:  # A routing failure must not stop answers.
        return RouteDecision(
            intent=project.default_intent,
            standalone=not turns,
            fallback=f"router failed: {error}",
            refers_back=pointing,
        )
    decision = _parse_route(raw, project, len(turns))

    # Keep a related turn only when the question gives a sign of it.
    terms = _content_terms(question)
    kept: list[int] = []
    dropped: list[tuple[int, str]] = []
    for number in decision.related_turn_ids:
        item = representations[number - 1]
        if item["skip"]:
            dropped.append((number, "turn ignores the conversation"))
        elif pointing or terms & _content_terms(_representation_text(item)):
            kept.append(number)
        else:
            dropped.append(
                (number, "question neither points back nor shares a term")
            )
    standalone = decision.standalone or (bool(dropped) and not kept)

    intent = project.intents.get(decision.intent)
    if (
        not kept
        and intent is not None
        and (
            intent.retrieval_mode == RETRIEVAL_MODE_REUSE
            or (
                intent.tool is not None
                and TOOL_REGISTRY[intent.tool.name].follows_up
            )
        )
    ):
        # An intent that reuses earlier evidence, or a tool that works on
        # an earlier answer, always refers to a turn: without one named,
        # it is the last turn that is not small talk.
        last = next(
            (
                item["turn_id"]
                for item in reversed(representations)
                if not item["skip"]
            ),
            None,
        )
        if last is not None:
            kept, standalone = [last], False

    return RouteDecision(
        **{
            **_fields(decision),
            # Only when every named turn was dropped does the check
            # overrule the router's standalone judgement.
            "standalone": standalone,
            "related_turn_ids": tuple(kept),
        },
        fallback=decision.fallback,
        refers_back=pointing,
        dropped_turns=tuple(dropped),
    )


def refers_back(question: str) -> bool:
    """Tell whether a question's wording depends on earlier turns.

    Parameters
    ----------
    question: str
        The new question.

    Returns
    -------
    bool
        True for pronouns ("it", "they"), ordinal references ("the second
        one"), pointers back ("you said", "earlier", "so far"), elliptical
        openings ("and for X?"), and messages of one or two words.

    """

    words = re.findall(r"\w+", question)
    return len(words) <= _SHORT_QUESTION_MAX_WORDS or bool(
        _REFERS_BACK.search(question)
    )


def search_text(
    question: str, turns: list[object], decision: RouteDecision
) -> str:
    """Build the search query from the user's own words.

    Parameters
    ----------
    question: str
        The new question.
    turns: list
        Earlier conversation turns with ``question``.
    decision: RouteDecision
        The checked routing decision.

    Returns
    -------
    str
        The question as asked. When it points back to related turns, those
        turns' questions follow it, oldest first, so "how do I install
        it?" also searches for what "it" was. Model-written text is never
        added.

    """

    if not (decision.refers_back and decision.related_turn_ids):
        return question

    earlier = [
        " ".join(str(getattr(turns[number - 1], "question", "")).split())
        for number in decision.related_turn_ids
        if 0 < number <= len(turns)
    ]
    return " ".join([question.strip(), *[text for text in earlier if text]])


def _content_terms(text: str) -> set[str]:
    """Return the distinctive terms of a text.

    Lowercase words of three or more characters that are not common words,
    plus capitalized acronyms and short names, such as ``MRI``.

    Parameters
    ----------
    text: str
        Text to analyze or transform.

    Returns
    -------
    set[str]
        Distinctive normalized content words.

    """

    words = {
        word
        for word in re.findall(r"[a-z0-9][a-z0-9_-]+", text.lower())
        if len(word) >= 3 and word not in _COMMON_WORDS
    }
    # Acronyms and short names ("R", "ML", "MRI") count, case-sensitively.
    names = set(re.findall(r"\b[A-Z][A-Z0-9]*\b", text)) - {"I"}
    return words | names


def _representation_text(representation: dict[str, object]) -> str:
    """Join a representation's question, sources, and summary as words.

    Parameters
    ----------
    representation: dict[str, object]
        Stored summary of a conversation turn.

    Returns
    -------
    str
        Question, source names, and summary joined for term matching.

    """

    sources = " ".join(
        re.sub(r"[._:/]", " ", str(source))
        for source in representation.get("sources") or []
    )
    return " ".join(
        [
            str(representation.get("question", "")),
            sources,
            str(representation.get("summary", "")),
        ]
    )


def _ignores_conversation(turn: object, project: ProjectType) -> bool:
    """Whether a turn's intent neither reads history nor retrieves.

    Such turns (small talk) are not listed to the router and are never
    related turns. A turn with an unknown intent does not ignore the
    conversation.

    Parameters
    ----------
    turn: object
        Conversation turn to inspect.
    project: ProjectType
        Current project and its associated metadata.

    Returns
    -------
    bool
        True if the turn neither uses history nor retrieves evidence.

    """

    parameters = getattr(turn, "parameters", None)
    if not isinstance(parameters, dict):
        return False
    intent_id = parameters.get("intent")
    intent = (
        project.intents.get(intent_id)
        if isinstance(intent_id, str)
        else None
    )
    return (
        intent is not None
        and intent.tool is None
        and intent.history_policy == HISTORY_POLICY_NONE
        and intent.retrieval_mode == RETRIEVAL_MODE_NONE
    )


def _stored_representation(turn: object, number: int) -> dict[str, object]:
    """Return the representation saved with a turn, or build one.

    A saved representation is reused only if its question, summary, and
    sources have the expected types and size bounds. Otherwise it is
    reconstructed from the original turn. A valid saved representation
    keeps its content but takes the turn's current number.

    Parameters
    ----------
    turn: object
        Conversation turn to inspect.
    number: int
        Current number assigned to the conversation turn.

    Returns
    -------
    dict[str, object]
        Saved or reconstructed representation with the current turn number.

    """

    parameters = getattr(turn, "parameters", None)
    stored = (
        parameters.get("representation")
        if isinstance(parameters, dict)
        else None
    )
    if isinstance(stored, dict):
        question = stored.get("question")
        summary = stored.get("summary")
        sources = stored.get("sources")
        if (
            isinstance(question, str)
            and bool(question.strip())
            and len(question) <= _REPRESENTATION_MAX_CHARACTERS
            and isinstance(summary, str)
            and len(summary) <= _REPRESENTATION_MAX_CHARACTERS
            and isinstance(sources, list)
            and len(sources) <= _REPRESENTATION_MAX_SOURCES
            and all(isinstance(source, str) for source in sources)
        ):
            return {**stored, "turn_id": number}
    return turn_representation(turn, number)


def select_history(
    turn_count: int,
    policy: HistoryPolicy,
    recent_turns: int,
    decision: RouteDecision,
) -> list[int]:
    """Choose which earlier turns enter the prompt.

    Parameters
    ----------
    turn_count: int
        Number of earlier turns in the conversation.
    policy: HistoryPolicy
        ``none`` includes nothing; ``recent`` the last ``recent_turns``;
        ``relevant`` adds the router's related turns to the recent ones.
        A question the router judged standalone still gets the previous
        turn (its question and answer, never its evidence), so the model
        knows what the user is reacting to.
    recent_turns: int
        How many of the latest turns count as recent.
    decision: RouteDecision
        The router's decision for the new question.

    Returns
    -------
    list[int]
        One-based turn numbers in conversation order.

    """

    if policy == HISTORY_POLICY_NONE or turn_count == 0:
        return []

    recent = list(range(max(1, turn_count - recent_turns + 1), turn_count + 1))
    if policy == HISTORY_POLICY_RECENT:
        return recent

    if decision.standalone:
        return [turn_count]

    return sorted(set(recent) | set(decision.related_turn_ids))


def history_messages(
    turns: list[object], numbers: list[int]
) -> list[dict[str, str]]:
    """Return selected turns as user and assistant messages.

    Parameters
    ----------
    turns: list
        Conversation turns with ``question`` and ``answer``.
    numbers: list[int]
        One-based numbers of the turns to include.

    Returns
    -------
    list[dict[str, str]]
        Alternating user and assistant messages. Only question and answer
        text is included. Thinking text is removed, and citation numbers
        become ``HISTORY_CITATION_MARKER``: the evidence they referred to is
        not in the prompt, but the answers still show the citing style.

    """

    messages: list[dict[str, str]] = []
    for number in numbers:
        turn = turns[number - 1]
        answer = _neutral_citations(strip_thinking(turn.answer)).strip()
        messages.append({"role": USER_ROLE, "content": str(turn.question)})
        messages.append({"role": ASSISTANT_ROLE, "content": answer})

    return messages
