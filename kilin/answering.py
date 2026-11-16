"""Route questions by intent and select conversation history.

KíLín has one machine state for now, ``answering``. Within it, a router
model classifies each question into one of the current project type's
intents (see ``kilin/project_types/``). The configuration, not the
router, then decides the intent's instructions, which earlier turns are
included, and whether evidence is searched, reused, or skipped.

Earlier turns enter the prompt as question and answer text only; their
retrieved evidence stays with the turn it was retrieved for. Each turn keeps
a short representation (question, cited sources, and a one-sentence summary
of the answer) so the router can point back to older turns without reading
them. The router model writes the summary once, when the turn is saved; the
answer's first sentence is kept when the answer has only one sentence or
the model gives no usable summary.

A question asked again, with the same distinctive words as an earlier
question, is still routed by the router model, which is told which turn it
repeats and how that turn was answered. Asked right after itself, the
user chooses between the same way again and another way when they can be
asked.

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

    ``turn_representation`` builds a compact summary of a turn, and
    ``summarize_answer`` asks the router model for its one-sentence answer
    summary. Invalid saved representations are rebuilt from the original
    turn.

- **Repeated questions**

    ``route_question`` tells the router when a question repeats an
    earlier turn, and how that turn was answered, so it can judge.

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

    Summarize a turn compactly, optionally with a model-written summary.

- ``summarize_answer``

    Ask the router model for a one-sentence summary of an answer.

- ``another_way_guidance``

    Tell the answer that the user wants another way than an earlier turn.

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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import PureWindowsPath
import re
from typing import Final, Literal, Protocol, TypeAlias

from kilin._model_completion import (
    require_complete_reply,
    strip_thinking,
)
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
    "another_way_guidance",
    "extract_thinking",
    "history_messages",
    "refers_back",
    "route_question",
    "search_text",
    "select_history",
    "side_reply",
    "strip_thinking",
    "summarize_answer",
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
# Token budget and answer length for the router's one-sentence summary.
_SUMMARY_MAX_TOKENS: Final[int] = require_default_int(
    _ANSWERING_DEFAULTS, "SUMMARY_MAX_TOKENS", "ANSWERING"
)
_SUMMARY_ANSWER_MAX_CHARACTERS: Final[int] = require_default_int(
    _ANSWERING_DEFAULTS, "SUMMARY_ANSWER_MAX_CHARACTERS", "ANSWERING"
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
# A tool that works on an earlier answer is not meant by a question that
# names a file and says nothing about earlier answers ("In Figure 1G of
# smith20.pdf, what does 79.7 represent?").
_NAMED_FILE: Final[re.Pattern[str]] = re.compile(
    r"\b[\w.-]+\.(?:pdf|md|txt|docx?|py|ipynb|m|json)\b", re.I
)
_EARLIER_ANSWER: Final[re.Pattern[str]] = re.compile(
    r"\b(?:you|your|cited|quoted|listed|answer|answers|previous|earlier"
    r"|above|just|that|this|these|those|them)\b",
    re.I,
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
    repeated_turn: int | None
        The earlier turn that asked the same question, which the router
        was told about.
    another_way: bool
        True when the user chose "Try another way" for a question asked
        twice in a row; the answer is then told that the last one missed
        (``another_way_guidance``).

    """

    intent: str
    standalone: bool
    related_turn_ids: tuple[int, ...] = ()
    raw: str = ""
    fallback: str | None = None
    shortcut: str | None = None
    refers_back: bool = False
    dropped_turns: tuple[tuple[int, str], ...] = ()
    repeated_turn: int | None = None
    another_way: bool = False


def extract_thinking(text: str) -> str:
    """Extract model thinking while leaving the original reply untouched.

    Parameters
    ----------
    text: str
        Model output, including possible thinking delimiters.

    Returns
    -------
    str
        Thinking text, or an empty string when there is none. Supports
        multiple blocks and a closing tag opened by the chat template.

    """

    raw = str(text)
    parts: list[str] = []
    first_close = raw.find("</think>")
    first_open = raw.find("<think>")
    if first_close != -1 and (first_open == -1 or first_close < first_open):
        parts.append(raw[:first_close])
        raw = raw[first_close + len("</think>"):]

    for match in re.finditer(r"<think>(.*?)</think>", raw, flags=re.S):
        parts.append(match.group(1))

    last_open = raw.rfind("<think>")
    last_close = raw.rfind("</think>")
    if last_open > last_close:
        parts.append(raw[last_open + len("<think>"):])

    return "\n\n".join(part.strip() for part in parts if part.strip())


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


def turn_representation(
    turn: object, number: int, *, summary: str | None = None
) -> dict[str, object]:
    """Summarize a turn compactly.

    Parameters
    ----------
    turn: object
        A conversation turn with ``question``, ``answer``, and ``evidence``.
    number: int
        One-based position of the turn in its conversation.
    summary: str | None, optional
        Model-written answer summary from ``summarize_answer``. Defaults to
        ``None``, which keeps the answer's first sentence.

    Returns
    -------
    dict[str, object]
        ``turn_id``, ``question``, ``sources`` (cited files and symbols),
        and ``summary`` (the given summary, or the answer's first
        sentence).

    """

    clean = strip_thinking(getattr(turn, "answer", ""))
    first = summary or _answer_sentences(clean)[0]

    # Only cited evidence is listed; an answer without citations lists all.
    cited = {
        int(citation)
        for group in re.findall(
            r"\[(\d+(?:\s*,\s*\d+)*)\]", clean
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
        "summary": first[:_REPRESENTATION_MAX_CHARACTERS],
    }


def _answer_sentences(answer: str) -> list[str]:
    """Split an answer into its first sentence and the rest.

    Parameters
    ----------
    answer: str
        Answer text, possibly with thinking and citations.

    Returns
    -------
    list[str]
        The first sentence, then the remaining text when there is any;
        ``[""]`` for an empty answer. Thinking text and citation groups are
        removed and whitespace is collapsed.

    """

    text = " ".join(_without_citations(strip_thinking(answer)).split())
    return re.split(r"(?<=[.!?])\s+", text, maxsplit=1)


def summarize_answer(chat: _Chat, question: str, answer: str) -> str | None:
    """Ask the router model for a one-sentence summary of an answer.

    The router lists each earlier turn by one line. An answer's first
    sentence is often a status line ("This project has 1 source file."),
    so a short model summary tells the router what the answer said.

    Parameters
    ----------
    chat: ChatSession
        Router model session; its conversation is restored afterwards.
    question: str
        The question the answer replies to.
    answer: str
        The saved answer, possibly with thinking and citations.

    Returns
    -------
    str | None
        One sentence with collapsed whitespace. None when the answer has
        only one sentence, or the call fails, is cut off, or gives an empty
        reply; the caller then keeps the answer's first sentence.

    """

    sentences = _answer_sentences(answer)
    if len(sentences) < 2:
        return None

    text = " ".join(sentences)[:_SUMMARY_ANSWER_MAX_CHARACTERS]
    system = (
        "You summarize answers for KíLín. You reply with one sentence "
        "only, in the language of the answer."
    )
    prompt = f"""Question: {question}

Answer:
{text}

In one sentence of at most 30 words, say what the answer states, keeping
its key names, numbers, and conclusion."""
    try:
        reply = side_reply(
            chat, system, prompt,
            max_new_tokens=_SUMMARY_MAX_TOKENS, require_complete=True,
        )
    except Exception:  # A summary failure must not stop saving a turn.
        return None
    return " ".join(reply.split()) or None


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
    max_new_tokens: int | None,
    token_callback: Callable[[str], None] | None = None,
    language_context: Sequence[str] = (),
    inherit_settings: bool = False,
    preserve_thinking: bool = False,
    require_complete: bool = False,
    thinking: bool | None = None,
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
    max_new_tokens: int | None
        Optional cap for a nonthinking request. Inherited thinking requests
        always use the active session maximum, including their thinking.
        ``None`` inherits that maximum in either mode. Router calls retain
        their separate explicit limits.
    token_callback: Callable[[str], None] | None, optional
        Receives the reply as it is generated, for a side answer shown to
        the user. Defaults to ``None``.
    language_context: Sequence[str], optional
        Original user messages authorizing CJK output, oldest first. The
        generated side prompt cannot authorize scripts. Defaults to ``()``.
    inherit_settings: bool, optional
        Keep session thinking and temperature unless explicitly overridden.
        Only nonthinking calls may use a smaller local cap; it never exceeds
        the user's maximum. False retains deterministic router behavior.
    preserve_thinking: bool, optional
        Return the complete model output, including thinking text, for a
        visible answer or for separate capture by the tool coordinator.
        Other side requests continue to strip thinking by default.
    require_complete: bool, optional
        Reject truncated, thinking-only or empty output before a tool
        consumes it. Defaults to ``False`` for router compatibility.
    thinking: bool | None, optional
        Temporary thinking override for an inherited tool request.
        ``None`` inherits unchanged. Only valid with ``inherit_settings``
        enabled; router calls cannot enable thinking through this option.

    Returns
    -------
    str
        The complete reply when ``preserve_thinking`` is true; otherwise
        the reply with thinking text removed.

    Raises
    ------
    ValueError
        If the cap is not a positive integer, an override is not a boolean,
        or a caller attempts to change router thinking without inheriting
        tool settings.
    IncompleteModelResponse
        If completion is required and the reply is truncated, unfinished
        or has no usable final content.

    Notes
    -----
    Router questions use greedy, nonthinking generation by default.
    Tool questions can inherit the answering session generation settings.

    """

    if max_new_tokens is not None and (
        type(max_new_tokens) is not int or max_new_tokens < 1
    ):
        raise ValueError("max_new_tokens must be a positive integer or None.")
    if thinking is not None:
        if type(thinking) is not bool or not inherit_settings:
            raise ValueError(
                "thinking requires a boolean and inherit_settings=True."
            )

    history = [dict(message) for message in chat.messages]
    previous = {"thinking": chat.thinking, "temperature": chat.temperature}

    # Restore the conversation and settings even when the reply fails.
    try:
        chat.restore_history([{"role": SYSTEM_ROLE, "content": system}])
        if not inherit_settings:
            chat.configure_generation({
                "thinking": False, "temperature": 0.0,
            })
        elif thinking is not None:
            chat.configure_generation({
                "thinking": thinking, "temperature": chat.temperature,
            })
        session_limit = getattr(chat, "max_new_tokens", None)
        token_limit = max_new_tokens
        if inherit_settings:
            if chat.thinking or token_limit is None:
                token_limit = session_limit
            elif session_limit is not None:
                token_limit = min(token_limit, session_limit)
        options: dict[str, object] = {
            "max_new_tokens": token_limit,
            "language_context": language_context,
        }
        if token_callback is not None:
            options["token_callback"] = token_callback
        reply = chat.reply(prompt, **options)
        if require_complete:
            # Some templates put <think> in the prompt. Reconstruct that
            # opening for validation, without changing the generated text.
            check_text = reply
            if (
                getattr(chat, "_last_prompt_opens_thinking", False)
                and "<think>" not in reply and "</think>" not in reply
            ):
                check_text = "<think>" + reply
            require_complete_reply(
                check_text, getattr(chat, "last_finish_reason", None)
            )
    finally:
        chat.restore_history(history)
        if not inherit_settings or thinking is not None:
            chat.configure_generation(previous)

    return reply if preserve_thinking else strip_thinking(reply)


def _router_prompt(
    project: ProjectType,
    representations: list[dict[str, object]],
    question: str,
    exclude: frozenset[str] = frozenset(),
    *,
    note: str = "",
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
    exclude: frozenset[str], optional
        Intent IDs left out of the list. Defaults to none.
    note: str, optional
        A sentence placed after the new question, such as that it repeats
        an earlier turn. Defaults to none.

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
        if intent_id not in exclude
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
    asked = f"{question}\n{note}" if note else question
    user = f"""Machine state: {MACHINE_STATE_ANSWERING}

Intents:
{intents}

Earlier turns:
{turns}

New question: {asked}

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
            and intent_id not in exclude
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
    *,
    excluded_turn_ids: frozenset[int] = frozenset(),
    choose: Callable[[str, tuple[str, ...]], int | None] | None = None,
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
    excluded_turn_ids: frozenset[int], optional
        Saved one-based turn numbers excluded from routing. Their stored
        representations are not read or shown to the router; numbering
        of other turns remains unchanged. Defaults to empty.
    choose: Callable[[str, tuple[str, ...]], int | None] | None, optional
        ``choose(question, options)`` asks the user to pick one option and
        returns its index, or None when they cancel. Used only for a
        question asked twice in a row. Defaults to ``None``: nobody can be
        asked, and the router decides.

    Returns
    -------
    RouteDecision
        Validated decision. Errors and unusable replies fall back to the
        project type's default intent with no related turns.

    Notes
    -----
    A question with the same distinctive words as an earlier question is
    marked as repeating it (``_repeated_turn``); the router prompt names
    that turn and its intent. Only
    the user knows why they asked the last question again, so with
    ``choose`` they pick: the same way again (the last turn's saved route,
    so a repeated inventory stays consistent) or another way (the router
    without the last intent). Cancelling leaves the decision to the router.

    The router's related turns are checked: a turn is kept only when the
    question points back (``refers_back``) or shares a distinctive term
    with the turn's question, sources, or summary. Turns whose intent
    neither reads history nor retrieves (small talk) are not listed to the
    router and are never related. A question left with no related turns
    is standalone. An intent whose cue the question matches replaces the
    router's choice ("give me an overview of the Smith paper"). Otherwise,
    when the wording rules out the router's choice, the router is asked
    once more without the ruled-out intents, and the default intent is
    used when that fails: an intent with ``requires`` patterns needs one
    of them to match, and a tool that works on an earlier answer
    (``follows_up``) is not meant when the question names a file and says
    nothing of earlier answers ("In Figure 1G of smith20.pdf, what does
    79.7 represent?").

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
        (
            {"turn_id": index, "skip": True}
            if index in excluded_turn_ids
            else {
                **_stored_representation(turn, index),
                "skip": _ignores_conversation(turn, project),
            }
        )
        for index, turn in enumerate(turns, 1)
    ]
    # A question asked again is not routed by rule: the router reads that
    # it repeats an earlier turn, with that turn's intent, and decides.
    repeated = _repeated_turn(turns, question, excluded_turn_ids)
    if repeated == len(turns) and choose is not None:
        chosen = _repeat_choice(
            chat, project, representations, question, turns, choose
        )
        if chosen is not None:
            return chosen
    pointing = refers_back(question)
    system, prompt = _router_prompt(
        project, representations, question,
        note=_repeat_note(turns, repeated) if repeated else "",
    )

    try:
        raw = side_reply(
            chat, system, prompt, max_new_tokens=_ROUTER_MAX_TOKENS
        )
    except Exception as error:  # A routing failure must not stop answers.
        return RouteDecision(
            intent=project.default_intent,
            standalone=not any(
                not item["skip"] for item in representations
            ),
            fallback=f"router failed: {error}",
            refers_back=pointing,
            repeated_turn=repeated,
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
    fallback = decision.fallback
    cued = _cued_intent(project, question)
    if cued is not None and cued != decision.intent:
        # Clear wording decides the intent; the router judged the history.
        fallback = (
            f"intent {cued} set by the question's wording (router chose "
            f"{decision.intent})"
        )
        decision = RouteDecision(**{**_fields(decision), "intent": cued})
        intent = project.intents[cued]
    elif intent is not None and cued is None:
        excluded = _unmeant_intents(project, question)
        if decision.intent in excluded:
            # The wording rules out the router's choice: ask once more
            # without it, and use the default intent when that fails.
            replaced = decision.intent
            retry = _reroute(
                chat, project, representations, question, len(turns),
                excluded,
            )
            decision = RouteDecision(
                **{**_fields(decision), "intent": retry}
            )
            intent = project.intents[retry]
            fallback = (
                f"intent {replaced} replaced by {retry}: "
                + excluded[replaced]
            )
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
        fallback=fallback,
        refers_back=pointing,
        dropped_turns=tuple(dropped),
        repeated_turn=repeated,
    )


def _repeated_turn(
    turns: list[object],
    question: str,
    excluded_turn_ids: frozenset[int],
) -> int | None:
    """Return the newest earlier turn that asked the same question.

    Two questions are the same when they have the same distinctive words
    (``_content_terms``), so case, word order, punctuation, and common
    words do not matter, while a different name or number does. A question
    that points back ("and its methods?") depends on the turn before it:
    asked right after the same question, "its" means the same thing both
    times, so it repeats the last turn; asked again later, it may mean
    something else, so it is not treated as repeated, and neither is an
    earlier question that pointed back.

    Parameters
    ----------
    turns: list
        Earlier conversation turns.
    question: str
        The new question.
    excluded_turn_ids: frozenset[int]
        Turns that must not be matched, such as disliked answers.

    Returns
    -------
    int | None
        One-based number of the matching turn, or None.

    """

    terms = _content_terms(question)
    if not terms:
        return None

    pointing = refers_back(question)
    for number in range(len(turns), 0, -1):
        earlier = str(getattr(turns[number - 1], "question", ""))
        if number in excluded_turn_ids or _content_terms(earlier) != terms:
            continue
        # Only the last turn shares the context a pointing word refers to.
        if number == len(turns) or not (pointing or refers_back(earlier)):
            return number
        return None
    return None


def _repeat_note(turns: list[object], number: int) -> str:
    """Tell the router that the new question repeats an earlier one.

    Parameters
    ----------
    turns: list
        Earlier conversation turns.
    number: int
        The repeated turn, from ``_repeated_turn``.

    Returns
    -------
    str
        One sentence for the router prompt, naming the earlier turn and its
        intent, so the same question tends to take the same route. Only the
        user can say that the last answer missed ("Try another way" in
        ``_repeat_choice``); the note never suggests another intent.

    """

    parameters = getattr(turns[number - 1], "parameters", None)
    intent = (
        parameters.get("intent") if isinstance(parameters, dict) else None
    )
    note = f"The new question repeats turn {number}"
    if isinstance(intent, str):
        return f"{note}, which was answered as {intent}."
    return f"{note}."


def _repeat_choice(
    chat: _Chat,
    project: ProjectType,
    representations: list[dict[str, object]],
    question: str,
    turns: list[object],
    choose: Callable[[str, tuple[str, ...]], int | None],
) -> RouteDecision | None:
    """Ask the user how to answer a question asked twice in a row.

    The same words right after an answer can mean "do it again" (a source
    inventory after adding files) or "that was not what I meant". Only
    the user knows, so they choose.

    Parameters
    ----------
    chat: ChatSession
        Router model session, for another way.
    project: ProjectType
        Current project type with its intents.
    representations: list[dict[str, object]]
        One representation per earlier turn.
    question: str
        The new question.
    turns: list
        Earlier conversation turns; the last one asked the same question.
    choose: Callable[[str, tuple[str, ...]], int | None]
        Asks the user to pick one option.

    Returns
    -------
    RouteDecision | None
        The last turn's saved route for "Same way again", or the router's
        choice for "Try another way", without the last intent and without
        intents that reuse earlier passages (``reuse`` retrieval). None when
        the user cancels or the last turn's saved route is unusable; the
        router then decides.

    """

    number = len(turns)
    parameters = getattr(turns[-1], "parameters", None)
    if not isinstance(parameters, dict):
        return None
    intent = parameters.get("intent")
    standalone = parameters.get("standalone")
    related = parameters.get("related_turn_ids")
    if (
        not isinstance(intent, str)
        or intent not in project.intents
        or type(standalone) is not bool
        or not isinstance(related, list)
        or not all(type(item) is int and 0 < item < number for item in related)
    ):
        return None

    label = intent.replace("_", " ").capitalize()
    index = choose(
        "You asked this right after the last answer. How should I answer?",
        (f"Same way again ({label})", "Try another way"),
    )
    if index == 0:
        reason = "the user chose the same way again"
    elif index == 1:
        reason = f"the user chose another way than {intent}"
        # Intents that reuse the last turn's passages would answer from
        # the same evidence again, so another way leaves them out too.
        excluded = {
            other: "it reuses the passages of the last answer"
            for other, setting in project.intents.items()
            if setting.retrieval_mode == RETRIEVAL_MODE_REUSE
        }
        excluded[intent] = "the user asked for another way"
        intent = _reroute(
            chat, project, representations, question, number, excluded
        )
    else:
        return None
    return RouteDecision(
        intent=intent,
        standalone=standalone,
        related_turn_ids=tuple(related),
        fallback=reason,
        repeated_turn=number,
        another_way=index == 1,
    )


def another_way_guidance(number: int) -> str:
    """Tell the answer that the user wants another way than turn ``number``.

    Used only after the user chose "Try another way" for a question asked
    twice in a row: without it, the model sees the same question and its
    earlier answer, and writes that answer again.

    Parameters
    ----------
    number: int
        The turn the user asked again right after.

    Returns
    -------
    str
        Answer guidance asking for a different approach, while keeping
        what was correct.

    """

    return (
        f"The user asked this again right after turn {number} and chose "
        "another way: that answer did not give them what they wanted. If "
        "it was correct, keep it and say so briefly, then add what it "
        "missed; otherwise take a different approach, such as more "
        "specific details from other parts of the documents or a "
        "different structure. Do not repeat that answer."
    )


def _cued_intent(project: ProjectType, question: str) -> str | None:
    """Return the first intent whose cue the question matches, if any.

    Parameters
    ----------
    project: ProjectType
        Current project type with its intents.
    question: str
        The new question.

    Returns
    -------
    str | None
        The intent ID, in the project type's order; None without a match.

    """

    return next(
        (
            intent_id
            for intent_id, intent in project.intents.items()
            if any(pattern.search(question) for pattern in intent.cues)
        ),
        None,
    )


def _unmeant_intents(project: ProjectType, question: str) -> dict[str, str]:
    """Return the intents a question's wording rules out, with the reason.

    Parameters
    ----------
    project: ProjectType
        Current project type with its intents.
    question: str
        The new question.

    Returns
    -------
    dict[str, str]
        Intent ID to reason. An intent with ``requires`` patterns is ruled
        out when the question matches none of them. A tool that works on an
        earlier answer (``follows_up``) is ruled out when the question
        names a file and says nothing of earlier answers.

    """

    new_about_file = bool(
        _NAMED_FILE.search(question) and not _EARLIER_ANSWER.search(question)
    )
    excluded: dict[str, str] = {}
    for intent_id, intent in project.intents.items():
        if intent.requires and not any(
            pattern.search(question) for pattern in intent.requires
        ):
            excluded[intent_id] = "the question lacks its required words"
        elif (
            new_about_file
            and intent.tool is not None
            and TOOL_REGISTRY[intent.tool.name].follows_up
        ):
            excluded[intent_id] = (
                "the question names a file and no earlier answer"
            )
    return excluded


def _reroute(
    chat: _Chat,
    project: ProjectType,
    representations: list[dict[str, object]],
    question: str,
    turn_count: int,
    excluded: Mapping[str, str],
) -> str:
    """Ask the router for an intent again, without the excluded intents.

    Parameters
    ----------
    chat: ChatSession
        Router model session.
    project: ProjectType
        Current project type with its intents.
    representations: list[dict[str, object]]
        One representation per earlier turn.
    question: str
        The new question.
    turn_count: int
        Number of earlier turns.
    excluded: Mapping[str, str]
        Intents left out of the router's list.

    Returns
    -------
    str
        The router's new intent, or the default intent when the call
        fails or names an excluded intent.

    """

    system, prompt = _router_prompt(
        project, representations, question, frozenset(excluded)
    )
    try:
        retry = _parse_route(
            side_reply(
                chat, system, prompt, max_new_tokens=_ROUTER_MAX_TOKENS
            ),
            project,
            turn_count,
        ).intent
    except Exception:  # The default intent is always safe.
        return project.default_intent
    return project.default_intent if retry in excluded else retry


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
    question: str,
    turns: list[object],
    decision: RouteDecision,
    *,
    excluded_turn_ids: frozenset[int] = frozenset(),
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
    excluded_turn_ids: frozenset[int], optional
        Turn numbers whose questions must not extend this query.

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
        and number not in excluded_turn_ids
    ]
    return " ".join([question.strip(), *[text for text in earlier if text]])


def _content_terms(text: str) -> set[str]:
    """Return the distinctive terms of a text.

    Lowercase words of three or more characters that are not common words,
    in singular form ("datasets" and "dataset" match), plus capitalized
    acronyms and short names, such as ``MRI``.

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
        _singular(word)
        for word in re.findall(r"[a-z0-9][a-z0-9_-]+", text.lower())
        if len(word) >= 3 and word not in _COMMON_WORDS
    }
    # Acronyms and short names ("R", "ML", "MRI") count, case-sensitively.
    names = set(re.findall(r"\b[A-Z][A-Z0-9]*\b", text)) - {"I"}
    return words | names


def _singular(word: str) -> str:
    """Return the singular form of a regular English plural.

    Parameters
    ----------
    word: str
        Lowercase word.

    Returns
    -------
    str
        "studies" becomes "study" and "datasets" "dataset"; words ending in
        "ss", "us" or "is" ("process", "corpus", "analysis") and short words
        are unchanged. Irregular plurals ("mice") are not handled.

    """

    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith(
        ("ss", "us", "is")
    ):
        return word[:-1]
    return word


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
    policy = parameters.get("intent_history_policy")
    mode = parameters.get("intent_retrieval_mode")
    uses_tool = parameters.get("intent_tool_used")
    if (
        isinstance(policy, str) and isinstance(mode, str)
        and type(uses_tool) is bool
    ):
        return (
            not uses_tool and policy == HISTORY_POLICY_NONE
            and mode == RETRIEVAL_MODE_NONE
        )
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
    *,
    excluded_turn_ids: frozenset[int] = frozenset(),
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
    excluded_turn_ids: frozenset[int], optional
        Saved turns to omit without renumbering eligible history. Recent
        history counts the most recent *eligible* turns.

    Returns
    -------
    list[int]
        One-based turn numbers in conversation order.

    """

    if policy == HISTORY_POLICY_NONE or turn_count == 0:
        return []

    # Preserve the constant-window cost and unchanged selection when
    # no rejected turns need filtering.
    if not excluded_turn_ids:
        recent = list(range(
            max(1, turn_count - recent_turns + 1), turn_count + 1
        ))
        if policy == HISTORY_POLICY_RECENT:
            return recent
        if decision.standalone:
            return [turn_count]
        return sorted(set(recent) | set(decision.related_turn_ids))

    eligible = [
        number for number in range(1, turn_count + 1)
        if number not in excluded_turn_ids
    ]
    if not eligible:
        return []
    recent = eligible[-recent_turns:] if recent_turns else []
    if policy == HISTORY_POLICY_RECENT:
        return recent

    if decision.standalone:
        return [eligible[-1]]

    return sorted(set(recent) | {
        number for number in decision.related_turn_ids
        if number in eligible
    })


def history_messages(
    turns: list[object],
    numbers: list[int],
    *,
    excluded_turn_ids: frozenset[int] = frozenset(),
) -> list[dict[str, str]]:
    """Return selected turns as user and assistant messages.

    Parameters
    ----------
    turns: list
        Conversation turns with ``question`` and ``answer``.
    numbers: list[int]
        One-based numbers of the turns to include.
    excluded_turn_ids: frozenset[int], optional
        Turns excluded from model context, including their questions.

    Returns
    -------
    list[dict[str, str]]
        Alternating user and assistant messages. Only question and answer
        text is included. Thinking text is removed, and citation numbers
        become ``HISTORY_CITATION_MARKER``: the evidence they referred to is
        not in the prompt, but the answers still show the citing style.
        Incomplete turns and legacy thinking-only replies are omitted as
        complete question/answer pairs; stored turn numbering is unchanged.

    """

    messages: list[dict[str, str]] = []
    for number in numbers:
        if number in excluded_turn_ids:
            continue
        turn = turns[number - 1]
        answer = _neutral_citations(strip_thinking(turn.answer)).strip()
        completion = getattr(turn, "parameters", {}).get("completion", {})
        if not answer or completion.get("status") == "incomplete":
            # Keep the saved turn for inspection, but never restore an
            # empty assistant message or use a failed result as history.
            continue
        messages.append({"role": USER_ROLE, "content": str(turn.question)})
        messages.append({"role": ASSISTANT_ROLE, "content": answer})

    return messages
