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
question (see ``kilin.retrievals.document_reading``). Like other
expensive tools, the plan can be confirmed first; the quick version reads
only the parts most relevant to the question.

Key Features
------------

- **Parts in order**

    Shared ``plan_parts`` packs consecutive passages by estimated
    tokens.

- **Notes with passage numbers**

    Shared ``notes_prompt`` and ``clean_notes`` keep lines citing passages in
    the part, so notes cannot point elsewhere.

- **Checked answer**

    ``answer_prompt`` writes the final answer; ``answer_citations`` reports
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
- ``kilin.tools._shared.document_notes.answer_citations``

Examples
--------
>>> from kilin.tools.batch_review import ReviewSettings
>>> ReviewSettings().part_tokens
6000

"""

from __future__ import annotations

from kilin.tools import partial_result
from kilin.tools._shared.citations import inspect_citations
from kilin.tools._shared.document_notes import (
    NOTHING,
    plan_parts,
    take_notes,
)

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Final

from kilin.retrievals.document_reading import (
    DOCUMENT_REASON_SEARCH,
    estimate_tokens,
    read_document,
)
from kilin.tools._shared.choose_document import choose_document
from kilin.tools._shared.term_filter import term_hits, term_label
from . import _configuration as _config
from kilin.tools import (
    ToolContext,
    ToolRequest,
    ToolResult,
    check_choice,
    check_int,
    reject_unknown,
)
from kilin.tools import (
    CONFIRMATION_CHOICES,
    PLAN_CHOICE_CANCEL,
    PLAN_CHOICE_QUICK,
    ConfirmationPolicy,
    ToolPlan,
    confirm_plan,
)

__all__ = [
    "ReviewSettings",
    "answer_prompt",
    "parse_settings",
    "run",
]

# What a part's notes say when it has nothing relevant.
# Rough speeds of an 8B 4-bit model on Apple silicon, for estimates only.
_PROMPT_TOKENS_PER_SECOND: Final[float] = _config.PROMPT_TOKENS_PER_SECOND
_REPLY_TOKENS_PER_SECOND: Final[float] = _config.REPLY_TOKENS_PER_SECOND
_PROMPT_OVERHEAD_TOKENS: Final[int] = _config.PROMPT_OVERHEAD_TOKENS


@dataclass(frozen=True, slots=True)
class ReviewSettings:
    """Settings of ``batch_review`` in a project type.

    Attributes
    ----------
    part_tokens: int
        Estimated tokens of document per part (1000-16000).
    max_model_calls: int
        Most part calls one answer may make (1-200); later parts are
        reported as not read.
    confirmation: ConfirmationPolicy
        When to ask before running.
    large_batches: int
        With ``when_large``, plans with more parts than this ask first
        (0-100).

    """

    part_tokens: int = _config.DEFAULT_PART_TOKENS
    max_model_calls: int = _config.DEFAULT_MAX_MODEL_CALLS
    confirmation: ConfirmationPolicy = _config.DEFAULT_CONFIRMATION
    large_batches: int = _config.DEFAULT_LARGE_BATCHES


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
        "You answer questions for KíLín from notes taken while reading a "
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
        KíLín's services.

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
            _estimate(to_read, texts) + _answer_seconds(),
            1,
        ),
        quick_items=len(quick),
        quick_seconds=round(
            (
                _estimate(quick_parts, texts)
                + _answer_seconds()
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
    failures: list[dict[str, object]] = []
    notes = take_notes(
        request.question,
        name,
        texts,
        runs,
        context,
        what=(
            "function"
            if selection.functions
            else "paper"
            if selection.path.suffix.lower() == ".pdf"
            else "file"
        ),
        whole=choice != PLAN_CHOICE_QUICK,
        failures=failures,
    )
    read = sorted(index for number, _ in notes for index in runs[number - 1])
    if failures and not any(lines for _, lines in notes):
        return partial_result(
            ToolResult(
                "No usable notes are available to answer.", (), summary,
            ),
            failures, completed=False,
        )
    context.done(
        f"Read {len(read)} of {len(passages)} passages of {name}",
        evidence_count=len(read),
    )
    system, prompt = answer_prompt(
        request.question, name, notes, len(runs), request.guidance
    )
    if failures:
        prompt += (
            "\n\nSome reading parts failed. Answer only from the supplied "
            "notes; do not infer absence from missing notes."
        )
    answer = context.write(system, prompt, None)
    citation_report = inspect_citations(answer, len(passages))
    invalid = list(citation_report.invalid_numbers)
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
                else " (reading coverage is incomplete)."
            )
            + "*"
        )
    summary.update(
        {
            "parts_read": len(notes),
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
    if citation_report.invalid_groups:
        summary["invalid_citation_groups"] = list(
            citation_report.invalid_groups
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
        if not failures or index in read
    )
    return partial_result(
        ToolResult(answer, evidence, summary, streamed=True), failures,
    )


def _fit(
    order: Sequence[int], texts: Sequence[str], part_tokens: int
) -> list[int]:
    """The leading indices of ``order`` that fit one part.

    Parameters
    ----------
    order: Sequence[int]
        Order in which items should be examined.
    texts: Sequence[str]
        Text passages or fields to inspect.
    part_tokens: int
        Token budget for a single batch.

    Returns
    -------
    list[int]
        Leading item indexes that fit within one token budget.

    """

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
) -> float:
    """Approximate seconds to take notes on the parts.

    Parameters
    ----------
    parts: Sequence[Sequence[int]]
        Parts of the document to process.
    texts: Sequence[str]
        Text passages or fields to inspect.

    Returns
    -------
    float
        Estimated seconds needed to process the supplied parts.

    """

    total = 0.0
    for part in parts:
        prompt = _PROMPT_OVERHEAD_TOKENS + sum(
            estimate_tokens(texts[index]) for index in part
        )
        total += 1.0 + prompt / _PROMPT_TOKENS_PER_SECOND
        total += _config.NOTE_REPLY_TOKENS / _REPLY_TOKENS_PER_SECOND
    return total


def _answer_seconds() -> float:
    """Approximate seconds for the final answer.

    Returns
    -------
    float
        Estimated seconds for final answer generation.

    """

    return 2.0 + _config.ANSWER_REPLY_TOKENS / _REPLY_TOKENS_PER_SECOND
