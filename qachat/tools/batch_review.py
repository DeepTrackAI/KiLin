"""Tool ``batch_review``: read a whole document in parts, then answer.

A long document does not fit a small model's prompt well: on an 8B model,
one 27,000-token paper took 105 seconds and the answer missed what the
paper said. This tool reads the document in order, a few thousand tokens at
a time. For each part the model writes short notes that cite passage
numbers; a final call answers the question from all the notes, and that
answer streams to the user. Every passage of the document is evidence, so
each citation opens the exact passage, and citations are checked against
the passages that exist.

The document is chosen as for document reading: a function named in the
question, a file it names, or the file that dominates a search for the
question (see ``qachat.retrievals.document_reading``). Like other
expensive tools, the plan can be confirmed first; the quick version reads
only the parts most relevant to the question.

Key Features
------------

- **Parts in order**

    ``plan_parts`` packs consecutive passages into parts by estimated
    tokens.

- **Notes with passage numbers**

    ``notes_prompt`` and ``clean_notes`` keep only lines citing passages in
    the part, so notes cannot point elsewhere.

- **Checked answer**

    ``answer_prompt`` writes the final answer; ``check_citations`` reports
    citations to passages that do not exist.

Module Structure
----------------

- ``ReviewSettings``
- ``parse_settings``
- ``run``
- ``plan_parts``
- ``take_notes``
- ``notes_prompt``
- ``clean_notes``
- ``answer_prompt``
- ``check_citations``

"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import re
from time import perf_counter
from typing import Final

from qachat.chunking import Chunk
from qachat.retrievals.document_reading import (
    DOCUMENT_REASON_SEARCH,
    estimate_tokens,
    read_document,
)
from qachat.tools.choose_document import choose_document
from qachat.tools.term_filter import term_hits, term_label
from qachat.tools.contract import (
    ToolContext,
    ToolRequest,
    ToolResult,
    check_choice,
    check_int,
    reject_unknown,
)
from qachat.tools.planning import (
    CONFIRMATION_CHOICES,
    CONFIRMATION_WHEN_LARGE,
    PLAN_CHOICE_CANCEL,
    PLAN_CHOICE_QUICK,
    ConfirmationPolicy,
    ToolPlan,
    confirm_plan,
)

__all__ = [
    "NOTHING",
    "ReviewSettings",
    "answer_prompt",
    "check_citations",
    "clean_notes",
    "notes_prompt",
    "parse_settings",
    "plan_parts",
    "run",
    "take_notes",
]

# What a part's notes say when it has nothing relevant.
NOTHING: Final[str] = "NOTHING"
# Rough speeds of an 8B 4-bit model on Apple silicon, for estimates only.
_PROMPT_TOKENS_PER_SECOND: Final[float] = 700.0
_REPLY_TOKENS_PER_SECOND: Final[float] = 25.0
_PROMPT_OVERHEAD_TOKENS: Final[int] = 300
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass(frozen=True, slots=True)
class ReviewSettings:
    """Settings of ``batch_review`` in a project type.

    Attributes
    ----------
    part_tokens: int
        Estimated tokens of document per part (1000-16000).
    note_tokens: int
        Longest notes for one part, in tokens (100-1500).
    answer_tokens: int
        Longest final answer, in tokens (200-4000).
    max_model_calls: int
        Most part calls one answer may make (1-200); later parts are
        reported as not read.
    confirmation: ConfirmationPolicy
        When to ask before running.
    large_batches: int
        With ``when_large``, plans with more parts than this ask first
        (0-100).

    """

    part_tokens: int = 6000
    note_tokens: int = 400
    answer_tokens: int = 1024
    max_model_calls: int = 20
    confirmation: ConfirmationPolicy = CONFIRMATION_WHEN_LARGE
    large_batches: int = 3


def parse_settings(data: dict, where: str) -> ReviewSettings:
    """Validate ``batch_review`` settings from a project type.

    Parameters
    ----------
    data: dict
        Settings from the intent's ``tool`` object, without ``name``.
    where: str
        Location for errors.

    Returns
    -------
    ReviewSettings
        Validated settings.

    Raises
    ------
    ValueError
        If a setting is unknown or out of range.

    """

    reject_unknown(
        where,
        data,
        (
            "part_tokens",
            "note_tokens",
            "answer_tokens",
            "max_model_calls",
            "confirmation",
            "large_batches",
        ),
    )
    defaults = ReviewSettings()
    return ReviewSettings(
        part_tokens=check_int(
            where, data, "part_tokens", defaults.part_tokens, 1000, 16000
        ),
        note_tokens=check_int(
            where, data, "note_tokens", defaults.note_tokens, 100, 1500
        ),
        answer_tokens=check_int(
            where, data, "answer_tokens", defaults.answer_tokens, 200, 4000
        ),
        max_model_calls=check_int(
            where, data, "max_model_calls", defaults.max_model_calls, 1, 200
        ),
        confirmation=check_choice(
            where,
            data,
            "confirmation",
            defaults.confirmation,
            CONFIRMATION_CHOICES,
        ),
        large_batches=check_int(
            where, data, "large_batches", defaults.large_batches, 0, 100
        ),
    )


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
        "You read one part of a document for QAChat and take notes. You do "
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


def answer_prompt(
    question: str,
    name: str,
    notes: Sequence[tuple[int, list[str]]],
    parts: int,
    guidance: str | None,
) -> tuple[str, str]:
    """Return the prompts that answer from all the notes.

    Parameters
    ----------
    question: str
        The user's question.
    name: str
        The document's name.
    notes: Sequence[tuple[int, list[str]]]
        Part numbers and their note lines.
    parts: int
        Number of parts read.
    guidance: str | None
        Answer guidance, such as a length target.

    Returns
    -------
    tuple[str, str]
        System prompt and user prompt.

    """

    system = (
        "You answer questions for QAChat from notes taken while reading a "
        "document in full. Use only the notes, and cite passage numbers "
        "exactly as the notes give them, like [12]."
    )
    blocks = "\n\n".join(
        f"Part {part}:\n" + ("\n".join(lines) if lines else NOTHING)
        for part, lines in notes
    )
    user = f"""Question: {question}

Notes from reading {parts} parts of {name}:

{blocks}

Answer the question from these notes. If they do not settle it, say what
the document does say."""
    if guidance:
        user += f"\n\n{guidance}"
    return system, user


def check_citations(answer: str, passages: int) -> list[int]:
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


def run(
    settings: ReviewSettings, request: ToolRequest, context: ToolContext
) -> ToolResult:
    """Read the document the question is about, in parts, then answer.

    Parameters
    ----------
    settings: ReviewSettings
        The intent's settings.
    request: ToolRequest
        The question, the user's words, and answer guidance.
    context: ToolContext
        QAChat's services.

    Returns
    -------
    ToolResult
        The streamed answer, every passage of the document as evidence,
        and a summary with parts read, notes, and citation checks.

    Notes
    -----
    The plan (parts, model calls, approximate time) is traced and, when
    the confirmation policy asks, offered through ``context.confirm``. The
    quick version reads the most relevant passages that fit one part.

    """

    trace = context.trace
    context.progress("Finding the document to read")
    ranked = context.rank(request.text)
    selection = read_document(
        request.text,
        context.chunks,
        context.graph,
        ranked,
        max_tokens=10**9,
        top=context.retrieval_count,
    )
    summary: dict[str, object] = {"name": "batch_review"}
    if selection is not None and selection.reason == DOCUMENT_REASON_SEARCH:
        # Only a search guessed the file: use the earlier turns, or ask.
        chosen = choose_document(
            request,
            context,
            question="Which document should I read in full?",
            example="read smith2020.pdf and say which software it uses",
        )
        if isinstance(chosen, str):
            context.done("No document chosen", evidence_count=0)
            return ToolResult(chosen, (), summary)
        summary["chosen"] = chosen[1]
        selection = read_document(
            request.text,
            context.chunks,
            context.graph,
            ranked,
            max_tokens=10**9,
            top=context.retrieval_count,
            path=chosen[0],
        )
    if selection is None:
        context.done("No document found to read", evidence_count=0)
        return ToolResult(
            "I couldn't find a document to read. Name the file, for "
            "example: read smith2020.pdf and say which software it uses.",
            (),
            summary,
        )
    passages = list(selection.chunks)
    name = selection.path.name
    if selection.functions:
        name = f"{', '.join(selection.functions)} in {name}"
    parts = plan_parts(passages, settings.part_tokens)
    to_read = parts[: settings.max_model_calls]
    texts = [chunk.text for chunk in passages]
    order = {chunk.identifier: index for index, chunk in enumerate(passages)}
    relevant = [
        order[result.chunk.identifier]
        for result in ranked
        if result.chunk.identifier in order
    ]
    # Quick: the passages that mention the question's exact terms, when it
    # names any; otherwise the most relevant passages that fit one part.
    terms, hits = term_hits(request.question, context, passages)
    quick_parts = (
        [[hits[i] for i in part] for part in plan_parts(
            [passages[i] for i in hits], settings.part_tokens
        )]
        if terms
        else [sorted(_fit(relevant, texts, settings.part_tokens))]
    )
    quick = [index for part in quick_parts for index in part]
    label, note = term_label(terms, hits) if terms else ("", "")
    plan = ToolPlan(
        tool="batch_review",
        summary=(
            f"Read all {len(passages)} passages of {name} in "
            f"{len(parts)} parts, then answer from the notes."
        ),
        scope=f"{selection.path.as_posix()} ({selection.reason})",
        items=len(passages),
        batches=len(parts),
        batches_to_run=len(to_read),
        estimated_seconds=round(
            _estimate(to_read, texts, settings) + _answer_seconds(settings),
            1,
        ),
        quick_items=len(quick),
        quick_seconds=round(
            (
                _estimate(quick_parts, texts, settings)
                + _answer_seconds(settings)
            )
            if quick
            else 0.0,
            1,
        ),
        details=(
            f"About {selection.estimated_tokens:,} tokens",
            f"Model: {context.model_id}",
        ),
        quick_label=label,
        quick_note=note,
    )
    choice, asked = confirm_plan(
        plan, settings.confirmation, settings.large_batches, context
    )
    summary.update(
        {
            "document": selection.path.as_posix(),
            "functions": list(selection.functions),
            "reason": selection.reason,
            "passages": len(passages),
            "plan": plan.to_dict(),
            "choice": choice,
            "asked": asked,
        }
    )
    if choice == PLAN_CHOICE_CANCEL:
        context.done("Cancelled before reading", evidence_count=0)
        return ToolResult(
            f"Cancelled before reading {name}; no model calls were made.",
            (),
            summary,
        )

    if choice == PLAN_CHOICE_QUICK and terms and not hits:
        # No passage mentions the terms: answered without the model.
        quoted = " or ".join(f"“{term}”" for term in terms)
        context.done(f"No passage mentions {quoted}", evidence_count=0)
        summary.update({"terms": list(terms), "term_hits": 0})
        return ToolResult(
            f"No passage of {name} mentions {quoted}; all {len(passages)} "
            "passages were checked for these words. Ask again and choose "
            "Continue to read it in full for related wording.",
            (),
            summary,
        )
    runs = quick_parts if choice == PLAN_CHOICE_QUICK else to_read
    started = perf_counter()
    notes = take_notes(
        request.question,
        name,
        texts,
        runs,
        settings.note_tokens,
        context,
        what=(
            "function"
            if selection.functions
            else "paper"
            if selection.path.suffix.lower() == ".pdf"
            else "file"
        ),
        whole=choice != PLAN_CHOICE_QUICK,
    )
    read = sorted(index for part in runs for index in part)
    context.done(
        f"Read {len(read)} of {len(passages)} passages of {name}",
        evidence_count=len(passages),
    )
    system, prompt = answer_prompt(
        request.question, name, notes, len(runs), request.guidance
    )
    answer = context.write(system, prompt, settings.answer_tokens)
    invalid = check_citations(answer, len(passages))
    if choice == PLAN_CHOICE_QUICK or len(read) < len(passages):
        answer += (
            f"\n\n*Read {len(read)} of {len(passages)} passages of {name}"
            + (
                (
                    f" (those that mention {' or '.join(terms)})."
                    if terms
                    else " (the most relevant ones)."
                )
                if choice == PLAN_CHOICE_QUICK
                else " (the model-call limit was reached)."
            )
            + "*"
        )
    summary.update(
        {
            "parts_read": len(runs),
            "passages_read": len(read),
            "notes": sum(len(lines) for _, lines in notes),
            "parts_without_notes": [
                part for part, lines in notes if not lines
            ],
            "invalid_citations": invalid,
            "seconds": round(perf_counter() - started, 1),
            "model": context.model_id,
        }
    )
    trace.emit(
        "tool.batch_review.result",
        component="tool",
        data={
            key: value
            for key, value in summary.items()
            if key not in ("name", "plan")
        },
        turn_id=context.trace_turn_id,
    )
    evidence = tuple(
        {
            "citation": index + 1,
            "path": chunk.path,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "symbol": chunk.symbol,
            "text": chunk.text,
            "cell": chunk.cell,
        }
        for index, chunk in enumerate(passages)
    )
    return ToolResult(answer, evidence, summary, streamed=True)


def take_notes(
    task: str,
    name: str,
    texts: Sequence[str],
    parts: Sequence[Sequence[int]],
    note_tokens: int,
    context: ToolContext,
    *,
    what: str = "document",
    whole: bool = True,
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
    note_tokens: int
        Longest notes for one part, in tokens.
    context: ToolContext
        QAChat's services (``ask_model``, ``progress``, ``trace``).
    what: str, optional
        What is read, for progress messages: "paper", "file",
        "function", ... Defaults to ``"document"``.
    whole: bool, optional
        True when the parts cover the whole document; False for a quick
        reading of its most relevant passages. Defaults to ``True``.

    Returns
    -------
    list[tuple[int, list[str]]]
        Part numbers with their note lines (see ``clean_notes``). A failed
        call gives an empty part.

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
            reply = context.ask_model(system, prompt, note_tokens)
        except Exception as error:  # A failed part is reported, not fatal.
            reply = f"failed: {error}"
        lines = clean_notes(reply, [index + 1 for index in part])
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


def _fit(
    order: Sequence[int], texts: Sequence[str], part_tokens: int
) -> list[int]:
    """The leading indices of ``order`` that fit one part."""

    chosen: list[int] = []
    used = 0
    for index in order:
        size = estimate_tokens(texts[index])
        if chosen and used + size > part_tokens:
            break
        chosen.append(index)
        used += size
    return chosen


def _estimate(
    parts: Sequence[Sequence[int]],
    texts: Sequence[str],
    settings: ReviewSettings,
) -> float:
    """Approximate seconds to take notes on the parts."""

    total = 0.0
    for part in parts:
        prompt = _PROMPT_OVERHEAD_TOKENS + sum(
            estimate_tokens(texts[index]) for index in part
        )
        total += 1.0 + prompt / _PROMPT_TOKENS_PER_SECOND
        total += settings.note_tokens / 2 / _REPLY_TOKENS_PER_SECOND
    return total


def _answer_seconds(settings: ReviewSettings) -> float:
    """Approximate seconds for the final answer."""

    return 2.0 + settings.answer_tokens / 3 / _REPLY_TOKENS_PER_SECOND
