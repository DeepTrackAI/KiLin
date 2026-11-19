"""Answer "What can you do for me?" with examples about the project.

The answer lists one ready-to-send example question for each kind of
request the project type supports, such as "Give me a short summary of
tiunn26braph2.pdf." for a paper overview. The examples are the project's
suggested questions (``kilin.question_suggestions``), so they name the
project's own files and topics, and they change from one answer to the
next. Every answer ends with how to add and remove files, so the same
intent answers "How do I add my papers?". A project without sources is
told how to add its files first and that a pasted passage can be asked
about meanwhile, followed by the project type's fixed suggestions. No
model is called.

Key Features
------------

- **Project-specific examples**

    Show what each kind of request does with the user's own files.

- **Ready to send**

    Every example is a complete question the user can copy and send.

- **Adding files**

    Every answer says how to add and remove files.

- **Empty projects**

    Explain how to add files first, and fall back to the project type's fixed
    suggestions.

Module Structure
----------------

- ``GuideSettings``

    Settings of ``assistant_guide`` in a project type.

- ``parse_settings``

    Validate ``assistant_guide`` settings from a project type.

- ``run``

    Answer with one example question per kind of request.

Private helpers implement the normalization, validation, and bookkeeping
used by these operations; they are not added to the package-level API.

Examples
--------
>>> from kilin.tools.assistant_guide import parse_settings
>>> parse_settings({}, "example").max_listed
20

"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import random
from typing import Final

from kilin.tools._contract import (
    ToolContext,
    ToolRequest,
    ToolResult,
    check_int,
    reject_unknown,
)
from kilin.tools.assistant_guide import _configuration as _config

__all__ = [
    "GuideSettings",
    "parse_settings",
    "run",
]

_WITH_SOURCES: Final[str] = (
    "Here is what I can do with the files in this project, with one "
    "example for each kind of request. Copy an example into the question "
    "box and send it to try it:"
)
# How to add and remove files; ends every answer, so the same intent
# also answers "How do I add my papers?".
_FILE_STEPS: Final[tuple[str, ...]] = (
    "1. Drag files or folders onto **sources** in the Workspace panel on "
    "the right.",
    "2. Or click **+** in the message box, choose **Attach file** or "
    "**Attach folder**, and send your question.",
    "",
    "To remove a file, right-click it under **sources** and choose "
    "**Delete**.",
)
_NO_SOURCES: Final[str] = (
    "This project has no sources yet, so I cannot answer from your files. "
    "To add them:"
)
_MORE_SOURCES: Final[str] = "To add more files:"
_PASTE: Final[str] = (
    "Meanwhile, you can paste a short passage into the message box and ask "
    "me about it."
)
_WITHOUT_SOURCES: Final[str] = "Here are the kinds of things you can ask:"
_CLOSING: Final[str] = (
    "You can also ask your own question in your own words; I pick the "
    "right kind of request for it."
)
# Varies the examples from one answer to the next.
_RANDOM: Final[random.Random] = random.Random()


@dataclass(frozen=True, slots=True)
class GuideSettings:
    """Settings of ``assistant_guide`` in a project type.

    Attributes
    ----------
    max_listed: int
        Example questions listed, one per kind of request (1-100).

    """

    max_listed: int = _config.DEFAULT_MAX_LISTED


def parse_settings(data: dict, where: str) -> GuideSettings:
    """Validate ``assistant_guide`` settings from a project type.

    Parameters
    ----------
    data: dict
        Settings from the intent's ``tool`` object, without ``name``.
    where: str
        Location for errors.

    Returns
    -------
    GuideSettings
        Validated settings.

    Raises
    ------
    ValueError
        If a setting is unknown or out of range.

    """

    reject_unknown(where, data, ("max_listed",))
    defaults = GuideSettings()
    return GuideSettings(
        max_listed=check_int(
            where, data, "max_listed", defaults.max_listed, 1, 100
        ),
    )


def run(
    settings: GuideSettings, request: ToolRequest, context: ToolContext
) -> ToolResult:
    """Answer with one example question per kind of request.

    Parameters
    ----------
    settings: GuideSettings
        Validated settings.
    request: ToolRequest
        The user's question; its wording does not change the answer.
    context: ToolContext
        KíLín's services; ``context.suggestions`` gives the examples and
        ``context.fixed_suggestions`` the fallback.

    Returns
    -------
    ToolResult
        The answer, without evidence.

    """

    context.progress("Collecting example questions")
    suggestions = (
        list(context.suggestions()) if context.suggestions is not None else []
    )
    examples = _one_per_intent(suggestions)[: settings.max_listed]
    if examples:
        lines = [_WITH_SOURCES, ""]
        lines += [
            f"- **{_label(intent)}**: {question}"
            for intent, question in examples
        ]
        lines += ["", _CLOSING, "", _MORE_SOURCES, "", *_FILE_STEPS]
    else:
        lines = [_NO_SOURCES, "", *_FILE_STEPS, "", _PASTE]
        fixed = _other_requests(context.fixed_suggestions, request.question)
        if fixed:
            lines += ["", _WITHOUT_SOURCES, ""]
            lines += [f"- {text}" for text in fixed[: settings.max_listed]]

    summary: dict[str, object] = {
        "name": "assistant_guide",
        "examples": len(examples),
        "intents": [intent for intent, _ in examples],
        "fallback": not examples,
    }
    context.trace.emit(
        "tool.assistant_guide.result",
        component="tool",
        data={key: value for key, value in summary.items() if key != "name"},
        turn_id=context.trace_turn_id,
    )
    context.done(f"Listed {len(examples)} examples", evidence_count=0)
    return ToolResult("\n".join(lines), (), summary)


def _one_per_intent(suggestions: Sequence[object]) -> list[tuple[str, str]]:
    """Pick one question at random for each intent, in first-seen order.

    Parameters
    ----------
    suggestions: Sequence[object]
        Items with ``intent`` and ``question``, possibly several per
        intent.

    Returns
    -------
    list[tuple[str, str]]
        Intent ID and question pairs, one per intent.

    """

    by_intent: dict[str, list[str]] = {}
    for item in suggestions:
        by_intent.setdefault(item.intent, []).append(item.question)
    return [
        (intent, _RANDOM.choice(questions))
        for intent, questions in by_intent.items()
    ]


def _other_requests(fixed: Sequence[str], question: str) -> list[str]:
    """Return the fixed suggestions other than the question being answered.

    Parameters
    ----------
    fixed: Sequence[str]
        The project type's fixed suggestions.
    question: str
        The user's question, such as "What can you do for me?".

    Returns
    -------
    list[str]
        The suggestions in order, without the one that is the question,
        ignoring case and the final punctuation.

    """

    asked = question.strip().rstrip("?!. ").casefold()
    return [
        text
        for text in fixed
        if text.strip().rstrip("?!. ").casefold() != asked
    ]


def _label(intent: str) -> str:
    """Return an intent ID as a short label, such as "Paper overview".

    Parameters
    ----------
    intent: str
        Intent ID, such as ``paper_overview``.

    Returns
    -------
    str
        The ID with spaces and a capital first letter, as the desktop app
        names intents.

    """

    return intent.replace("_", " ").capitalize()
