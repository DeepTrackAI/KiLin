"""Shared reading, note-taking and citation checks for document tools.

Several registered tools read full documents by breaking them into
passage groups, asking the model for traceable notes, and checking the
passage citations. This private implementation prevents tools from
importing one another for reusable reading algorithms.

Key Features
------------

- **Deterministic planning**

    Group passages by estimated token sizes.

- **Checked model notes**

    Keep only notes citing passages included in their reading part.

- **Reusable model orchestration**

    Report progress and trace notes using the caller's context.

Module Structure
----------------

- ``NOTHING``

    Sentinel in a model reply when no relevant notes are found.

- ``plan_parts``, ``notes_prompt`` and ``clean_notes``

    Group passages, construct prompts, and validate notes.

- ``answer_citations``

    Find invalid passage citations in a generated answer.

- ``take_notes``

    Read each part using the caller's model and tracing callbacks.

Examples
--------
>>> from kilin.tools._shared.document_notes import clean_notes
>>> clean_notes("- Fact [2]", [1, 2])
['- Fact [2]']

"""

from __future__ import annotations

from collections.abc import Sequence
import re
from time import perf_counter
from typing import Final

from kilin._model_completion import IncompleteModelResponse
from kilin.chunking import Chunk
from kilin.retrievals.document_reading import estimate_tokens
from kilin.tools._contract import ToolContext

__all__ = [
    "NOTHING",
    "answer_citations",
    "clean_notes",
    "notes_prompt",
    "plan_parts",
    "take_notes",
]

NOTHING: Final[str] = "NOTHING"
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def plan_parts(chunks: Sequence[Chunk], part_tokens: int) -> list[list[int]]:
    """Pack consecutive passages into parts by estimated tokens.

    Parameters
    ----------
    chunks: Sequence[Chunk]
        The document's passages, in order.
    part_tokens: int
        Estimated tokens per part; a larger passage gets its own part.

    Returns
    -------
    list[list[int]]
        Passage indices per part, in order.

    """

    parts: list[list[int]] = []
    used = 0
    for index, chunk in enumerate(chunks):
        size = estimate_tokens(chunk.text)
        if parts and used + size <= part_tokens:
            parts[-1].append(index)
            used += size
        else:
            parts.append([index])
            used = size
    return parts


def notes_prompt(
    question: str,
    name: str,
    part: int,
    parts: int,
    passages: Sequence[tuple[int, str]],
) -> tuple[str, str]:
    """Return the prompts that take notes on one part.

    Parameters
    ----------
    question: str
        The user's question.
    name: str
        The document's name.
    part: int
        This part's number, from one.
    parts: int
        Number of parts.
    passages: Sequence[tuple[int, str]]
        Passage numbers and texts of this part.

    Returns
    -------
    tuple[str, str]
        System prompt and user prompt.

    """

    system = (
        "You read one part of a document for KíLín and take notes. You do "
        "not answer the question yet. You reply with short bullet notes "
        "only."
    )
    text = "\n\n".join(
        f"[{number}] {body.strip()}" for number, body in passages
    )
    user = f"""Question: {question}

Below is part {part} of {parts} of {name}. Its passages are numbered.

{text}

Write bullet notes with every fact in this part that helps answer the
question, each ending with its passage number, for example
"- The software was written in Python 3.11 [12]". Copy names, numbers, and
versions exactly. If this part has nothing relevant, reply exactly:
{NOTHING}"""
    return system, user


def clean_notes(reply: str, numbers: Sequence[int]) -> list[str]:
    """Keep note lines that cite passages of their part.

    Parameters
    ----------
    reply: str
        The model's notes.
    numbers: Sequence[int]
        Passage numbers in the part.

    Returns
    -------
    list[str]
        Note lines, each citing at least one passage of the part; empty for
        ``NOTHING`` or notes without valid citations.

    """

    allowed = set(numbers)
    kept: list[str] = []
    for line in reply.splitlines():
        line = line.strip().lstrip("-*• ").strip()
        cited = {
            int(number)
            for group in _CITATION.findall(line)
            for number in re.split(r"\s*,\s*", group)
        }
        if line and cited and cited <= allowed:
            kept.append(f"- {line}")
    return kept



def answer_citations(answer: str, passages: int) -> list[int]:
    """Return citation numbers in an answer that match no passage.

    Parameters
    ----------
    answer: str
        The final answer.
    passages: int
        Number of passages in the document.

    Returns
    -------
    list[int]
        Sorted invalid citation numbers; empty when all are valid.

    """

    cited = {
        int(number)
        for group in _CITATION.findall(answer)
        for number in re.split(r"\s*,\s*", group)
    }
    return sorted(number for number in cited if not 1 <= number <= passages)



def take_notes(
    task: str,
    name: str,
    texts: Sequence[str],
    parts: Sequence[Sequence[int]],
    context: ToolContext,
    *,
    what: str = "document",
    whole: bool = True,
    failures: list[dict[str, object]] | None = None,
) -> list[tuple[int, list[str]]]:
    """Take checked notes on each part of a document with the model.

    Parameters
    ----------
    task: str
        What the notes are for: the user's question, or an instruction
        such as the sections of a summary.
    name: str
        The document's name.
    texts: Sequence[str]
        Every passage's text; passage ``n`` is ``texts[n - 1]``.
    parts: Sequence[Sequence[int]]
        Passage indices per part, in reading order.
    context: ToolContext
        KíLín's services (``ask_model``, ``progress``, ``trace``).
    what: str, optional
        What is read, for progress messages: "paper", "file",
        "function", ... Defaults to ``"document"``.
    whole: bool, optional
        True when the parts cover the whole document; False for a quick
        reading of its most relevant passages. Defaults to ``True``.
    failures: list[dict[str, object]] | None, optional
        Collect independent failed parts and continue when supplied. Failed
        parts are omitted from notes, never represented as NOTHING. None
        retains strict failure propagation.

    Returns
    -------
    list[tuple[int, list[str]]]
        Part numbers with checked note lines. Explicit NOTHING replies
        give empty parts. Failed calls or unparseable notes are omitted
        when a failure collector is supplied; otherwise they raise
        ``IncompleteModelResponse``. They never imply a negative result.

    """

    notes: list[tuple[int, list[str]]] = []
    for number, part in enumerate(parts, start=1):
        if not whole:
            message = f"Reading the most relevant passages: {name}"
        elif len(parts) == 1:
            message = f"Reading the whole {what}: {name}"
        else:
            message = (
                f"Reading the whole {what} (part {number} of "
                f"{len(parts)}): {name}"
            )
        context.progress(message, current=number, total=len(parts))
        started = perf_counter()
        system, prompt = notes_prompt(
            task,
            name,
            number,
            len(parts),
            [(index + 1, texts[index]) for index in part],
        )
        try:
            reply = context.ask_model(system, prompt, None)
            lines = clean_notes(reply, [index + 1 for index in part])
            if not lines and reply.strip() != NOTHING:
                raise IncompleteModelResponse(
                    "The model returned neither cited notes nor NOTHING.",
                    reason="invalid_output", raw_text=reply,
                )
        except IncompleteModelResponse as error:
            if failures is None:
                raise
            failure = {
                "item": f"{name}, reading part {number}",
                "part": number, "passages": len(part),
                "passage_numbers": [index + 1 for index in part],
                **error.to_dict(),
            }
            failures.append(failure)
            context.trace.emit(
                "tool.part.failed", component="tool", data=failure,
                turn_id=context.trace_turn_id,
            )
            continue
        except Exception as error:
            context.trace.emit(
                "tool.part.failed", component="tool",
                data={"part": number, "notes": 0, "error": str(error)},
                turn_id=context.trace_turn_id,
            )
            raise IncompleteModelResponse(
                f"Document reading failed in part {number}: {error}",
                reason="model_error",
            ) from error
        notes.append((number, lines))
        data: dict[str, object] = {
            "part": number,
            "passages": [index + 1 for index in (part[0], part[-1])],
            "notes": len(lines),
            "seconds": round(perf_counter() - started, 2),
        }
        if context.trace.include_content:
            data["raw"] = reply
        context.trace.emit(
            "tool.part.notes",
            component="tool",
            data=data,
            turn_id=context.trace_turn_id,
        )
    return notes
