"""Find a fact, or a complete list, in one paper or in every paper.

"What sample size does that paper use?", "which model does each paper
use?", or "give a full list of the datasets in that paper" asks for a fact
or a complete list: a number, a setting, or the named models, datasets,
tools, or other things a paper includes. Search ranks a few passages, and
the ones that state the fact may not be among them, so this tool reads
every passage of the paper, or of every paper, in parts and takes notes
with passage numbers. The answer is written from the notes, per paper.

The scope is the paper the question names or points at ("that paper"),
otherwise every indexed file. Reading everything takes time, so by default
the user always chooses first: read everything, a quick answer, or cancel.
When the question names exact terms, the quick answer takes notes on only
the passages that mention them; otherwise, or when no passage mentions
them, it answers from the best-ranked passages directly. Passages are numbered once across all files, so every
citation is unambiguous, and citations to passages that do not exist are
counted in the summary.

Parts beyond ``max_model_calls`` are not read, and the answer says so. A
failed notes call gives that part no notes; it is not retried. Only the
cited passages are kept as evidence, because a whole project has thousands
of passages.

Key Features
------------

- **Everything, or quick**

    The user chooses between reading every passage in scope and a quick
    answer from exact-term hits or the best-ranked passages.

- **Notes per file**

    Each file is read in parts with the shared document-notes service;
    the answer is written from the notes, one document at a time.

- **Checked citations**

    One passage numbering across all files; invalid citations are counted,
    and only cited passages become evidence.

Module Structure
----------------

- ``FactSettings``

    Settings of ``find_fact`` in a project type.

- ``parse_settings``

    Validate ``find_fact`` settings from a project type.

- ``fact_prompt``

    Prompts that answer a fact question from notes or passages.

- ``run``

    Find a fact by reading every passage in scope, or quickly.

Private helpers read the parts, find the cited passage numbers, and
estimate the time; they are not part of the public API.

Examples
--------
This is an illustrative integration call, not a standalone doctest.
KíLín supplies the required tool context and conversation state.

``context`` is the ``ToolContext`` that KíLín passes to a tool:

>>> from kilin.tools import ToolRequest
>>> from kilin.tools.find_fact import FactSettings, run
>>> question = "how many epochs does each paper use?"
>>> request = ToolRequest(question, question)
>>> result = run(FactSettings(), request, context)

"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import re
from time import perf_counter
from typing import Final

from kilin.tools._shared.model_results import partial_result
from kilin.chunking import Chunk
from kilin.retrievals.document_reading import estimate_tokens
from kilin.tools._shared.document_notes import (
    answer_citations,
    plan_parts,
    take_notes,
)
from kilin.tools._shared.choose_document import document_scope
from kilin.tools.find_fact import _configuration as _config
from kilin.tools._contract import (
    ToolContext,
    ToolRequest,
    ToolResult,
    check_choice,
    check_int,
    reject_unknown,
)
from kilin.tools._shared.planning import (
    CONFIRMATION_CHOICES,
    PLAN_CHOICE_CANCEL,
    PLAN_CHOICE_QUICK,
    ConfirmationPolicy,
    ToolPlan,
    confirm_plan,
)
from kilin.tools._shared.term_filter import term_hits, term_label

__all__ = [
    "FactSettings",
    "fact_prompt",
    "parse_settings",
    "run",
]

# Rough speeds of an 8B 4-bit model on Apple silicon, for estimates only.
_PROMPT_TOKENS_PER_SECOND: Final[float] = _config.PROMPT_TOKENS_PER_SECOND
_REPLY_TOKENS_PER_SECOND: Final[float] = _config.REPLY_TOKENS_PER_SECOND
# For estimates only: a notes prompt without its passages, the fixed cost
# of one notes call, and the fixed cost of a quick answer.
_NOTES_PROMPT_TOKENS: Final[int] = _config.NOTES_PROMPT_TOKENS
_CALL_SECONDS: Final[float] = _config.CALL_SECONDS
_QUICK_ANSWER_SECONDS: Final[float] = _config.QUICK_ANSWER_SECONDS
# A citation such as [3], [3, 5], or [3-5] (with a hyphen or an en dash).
_CITATION_GROUP: Final[re.Pattern[str]] = re.compile(
    r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]"
)
_CITATION_RANGE: Final[re.Pattern[str]] = re.compile(
    r"(\d+)(?:\s*[–-]\s*(\d+))?"
)


@dataclass(frozen=True, slots=True)
class FactSettings:
    """Settings of ``find_fact`` in a project type.

    Attributes
    ----------
    part_tokens: int
        Estimated tokens of passages per reading part (1000-16000).
    max_model_calls: int
        Most reading parts (1-500).
    confirmation: "always" | "when_large" | "never"
        When to ask before reading; ``always`` by default, because the user
        chooses between a thorough and a quick answer.
    large_batches: int
        With ``when_large``, plans with more parts than this ask first.

    """

    part_tokens: int = _config.DEFAULT_PART_TOKENS
    max_model_calls: int = _config.DEFAULT_MAX_MODEL_CALLS
    confirmation: ConfirmationPolicy = _config.DEFAULT_CONFIRMATION
    large_batches: int = _config.DEFAULT_LARGE_BATCHES


def parse_settings(data: dict, where: str) -> FactSettings:
    """Validate ``find_fact`` settings from a project type.

    Parameters
    ----------
    data: dict
        Settings from the intent's ``tool`` object, without ``name``.
    where: str
        Location for errors.

    Returns
    -------
    FactSettings
        Validated settings.

    Raises
    ------
    ValueError
        If a setting is unknown or out of range.

    """

    names = (
        "part_tokens",
        "max_model_calls",
        "confirmation",
        "large_batches",
    )
    reject_unknown(where, data, names)

    defaults = FactSettings()
    return FactSettings(
        part_tokens=check_int(
            where, data, "part_tokens", defaults.part_tokens, 1000, 16000
        ),
        max_model_calls=check_int(
            where, data, "max_model_calls", defaults.max_model_calls, 1, 500
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


def fact_prompt(
    question: str,
    notes: Sequence[tuple[str, list[str]]],
    guidance: str | None,
    *,
    quick: bool = False,
) -> tuple[str, str]:
    """Return the prompts that answer a fact question from notes.

    Parameters
    ----------
    question: str
        The user's question.
    notes: Sequence[tuple[str, list[str]]]
        Per file: its name and its note lines (or, for a quick answer, its
        numbered passages).
    guidance: str | None
        Answer guidance, such as a length target.
    quick: bool, optional
        True when the material is ranked passages, not notes from reading
        everything. Defaults to ``False``.

    Returns
    -------
    tuple[str, str]
        System prompt and user prompt.

    """

    material = "passages" if quick else "notes"
    system = (
        f"You answer a factual question for KíLín from {material} about "
        "documents. Use only them, cite passage numbers exactly as given, "
        "like [12], and never guess a number or name they do not state."
    )
    blocks = "\n\n".join(
        f"{name}:\n" + ("\n".join(lines) if lines else "NOTHING FOUND")
        for name, lines in notes
    )
    user = f"""{material.capitalize()} by document:

{blocks}

Answer only what the question asks, directly. The {material} often repeat
the same fact from different parts of a document: state each fact once and
cite every passage that supports it together, like [6, 64]. Leave out
details the question does not ask for. When it asks for a list, include
every distinct item once. With several documents, give the answer for each
document, and say plainly when a document does not state it. Follow any
length or format the question asks for.\
{f" {guidance}" if guidance else ""}

Question: {question}"""
    return system, user


def run(
    settings: FactSettings, request: ToolRequest, context: ToolContext
) -> ToolResult:
    """Find a fact by reading every passage in scope, or quickly.

    Parameters
    ----------
    settings: FactSettings
        The intent's settings.
    request: ToolRequest
        The question, with the earlier turns it points back to.
    context: ToolContext
        KíLín's services.

    Returns
    -------
    ToolResult
        The streamed answer, the cited passages as evidence, and a summary
        of the plan, the choice, and what was read.

    """

    started = perf_counter()
    summary: dict[str, object] = {"name": "find_fact"}
    context.progress("Finding the documents to read")
    indexed = list(dict.fromkeys(chunk.path for chunk in context.chunks))
    scope, why = document_scope(request, context.ask_router, indexed)
    files = scope or indexed

    # One passage numbering across all files: each file's parts hold
    # indices into ``passages``.
    per_file = [
        (path, [chunk for chunk in context.chunks if chunk.path == path])
        for path in files
    ]
    passages: list[Chunk] = [
        chunk for _, chunks in per_file for chunk in chunks
    ]
    texts = [chunk.text for chunk in passages]
    parts_by_file: list[tuple[Path, list[list[int]]]] = []
    offset = 0
    for path, chunks in per_file:
        parts = plan_parts(chunks, settings.part_tokens)
        parts_by_file.append(
            (path, [[offset + index for index in part] for part in parts])
        )
        offset += len(chunks)
    all_parts = [part for _, parts in parts_by_file for part in parts]
    budget = all_parts[: settings.max_model_calls]

    order = {chunk.identifier: index for index, chunk in enumerate(passages)}
    ranked = [
        order[result.chunk.identifier]
        for result in context.rank(request.text)
        if result.chunk.identifier in order
    ][: context.retrieval_count]

    # Quick: when the question names exact terms, only the passages that
    # mention them (read in parts, per file); otherwise the ranked ones.
    # Terms that no passage mentions are often the question's own wording
    # ("metadata table C6", "modular software extension"), not the paper's,
    # so the quick answer falls back to the ranked passages.
    terms, hits = term_hits(request.question, context, passages)
    missing: tuple[str, ...] = ()
    if terms and not hits:
        missing, terms = terms, ()
    hit_set = set(hits)
    quick_by_file = [
        (
            path,
            [[index for index in part if index in hit_set] for part in parts],
        )
        for path, parts in parts_by_file
    ]
    quick_by_file = [
        (path, [part for part in parts if part])
        for path, parts in quick_by_file
    ]
    if terms:
        label, note = term_label(terms, hits)
        quick_parts = [part for _, parts in quick_by_file for part in parts]
        quick_seconds = _estimate(quick_parts, texts)
    else:
        label = "Quick answer"
        note = (
            f"From the {len(ranked)} best-ranked "
            f"{'passage' if len(ranked) == 1 else 'passages'}; may miss a "
            "fact stated elsewhere"
        )
        if missing:
            note = f"{term_label(missing, ())[1]}. {note}"
        quick_seconds = (
            _QUICK_ANSWER_SECONDS
            + sum(estimate_tokens(texts[index]) for index in ranked)
            / _PROMPT_TOKENS_PER_SECOND
        )

    plan = ToolPlan(
        tool="find_fact",
        summary=(
            f"Read all {len(passages)} passages of "
            + (files[0].name if len(files) == 1 else f"{len(files)} files")
            + " to find the answer."
        ),
        scope=(
            f"{files[0].name} ({why})"
            if scope and len(files) == 1
            else f"All {len(files)} indexed files"
        ),
        items=len(passages),
        batches=len(all_parts),
        batches_to_run=len(budget),
        estimated_seconds=round(_estimate(budget, texts), 1),
        quick_items=len(hits) if terms else len(ranked),
        quick_seconds=round(quick_seconds, 1),
        details=(f"Model: {context.model_id}",),
        quick_label=label,
        quick_note=note,
    )
    choice, asked = confirm_plan(
        plan, settings.confirmation, settings.large_batches, context
    )
    summary.update({"plan": plan.to_dict(), "choice": choice, "asked": asked})
    if missing:
        summary["terms_without_hits"] = list(missing)
    if choice == PLAN_CHOICE_CANCEL:
        context.done("Cancelled before reading", evidence_count=0)
        return ToolResult(
            "Cancelled; no model calls were made.", (), summary
        )

    failures: list[dict[str, object]] = []
    if choice == PLAN_CHOICE_QUICK and terms:
        material = _notes(
            request.question,
            quick_by_file,
            texts,
            settings,
            context,
            whole=False,
            failures=failures,
        )
        read = len(hits)
    elif choice == PLAN_CHOICE_QUICK:
        material = []
        for path in files:
            lines = [
                f"[{index + 1}] {texts[index].strip()}"
                for index in sorted(ranked)
                if passages[index].path == path
            ]
            if lines:
                material.append((path.name, lines))
        read = len(ranked)
    else:
        # Parts beyond the model-call budget are left out.
        allowed = {id(part) for part in budget}
        in_budget = [
            (path, [part for part in parts if id(part) in allowed])
            for path, parts in parts_by_file
        ]
        material = _notes(
            request.question,
            in_budget,
            texts,
            settings,
            context,
            whole=True,
            failures=failures,
        )
        read = sum(len(part) for part in budget)
    read -= sum(int(item["passages"]) for item in failures)
    if failures and not any(lines for _, lines in material):
        return partial_result(
            ToolResult(
                "No usable notes are available to answer.", (), summary,
            ),
            failures, completed=False,
        )
    context.done(
        f"Read {read} of {len(passages)} passages",
        evidence_count=read,
    )

    system, prompt = fact_prompt(
        request.question,
        material,
        request.guidance,
        quick=choice == PLAN_CHOICE_QUICK and not terms,
    )
    if failures:
        prompt += (
            "\n\nSome reading parts failed. Answer only from the supplied "
            "notes; do not infer absence from missing notes."
        )
    answer = context.write(system, prompt, None)
    invalid = answer_citations(answer, len(passages))
    if choice == PLAN_CHOICE_QUICK:
        which = (
            "passages that mention " + " or ".join(terms)
            if terms
            else "best-ranked passages"
        )
        if missing:
            which += " (no passage mentions " + " or ".join(
                f"“{term}”" for term in missing
            ) + ")"
        answer += (
            f"\n\n*Quick answer from the {read} {which}; ask again and "
            "choose Continue to read everything.*"
        )
    elif read < len(passages):
        answer += (
            f"\n\n*Read {read} of {len(passages)} passages; "
            "reading coverage is incomplete.*"
        )
    summary.update(
        {
            "files": len(files),
            "passages": len(passages),
            "passages_read": read,
            "invalid_citations": invalid,
            "seconds": round(perf_counter() - started, 1),
        }
    )

    # Citations keep their numbers, so evidence ``n`` is passage ``n``.
    cited = _cited(answer, len(passages)) - {
        number for failure in failures
        for number in failure["passage_numbers"]
    }
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
        if index + 1 in cited
    )
    return partial_result(
        ToolResult(answer, evidence, summary, streamed=True), failures,
    )


def _notes(
    question: str,
    parts_by_file: Sequence[tuple[Path, list[list[int]]]],
    texts: Sequence[str],
    settings: FactSettings,
    context: ToolContext,
    *,
    whole: bool,
    failures: list[dict[str, object]] | None = None,
) -> list[tuple[str, list[str]]]:
    """Take notes on each file's parts; return the note lines per file.

    Files without parts are left out. ``whole`` is False for a quick
    reading of only some passages, which changes the progress messages.

    Parameters
    ----------
    question: str
        Question or request to analyze.
    parts_by_file: Sequence[tuple[Path, list[list[int]]]]
        Selected text parts grouped by source file.
    texts: Sequence[str]
        Text passages or fields to inspect.
    settings: FactSettings
        Settings used to configure the operation.
    context: ToolContext
        Tool context for the current request.
    whole: bool
        Whether to use the complete source text.
    failures: list[dict[str, object]] | None, optional
        Collect failed parts while keeping validated notes from other parts.

    Returns
    -------
    list[tuple[str, list[str]]]
        Per-file summaries and their extracted note lines.

    """

    material: list[tuple[str, list[str]]] = []
    for path, parts in parts_by_file:
        if not parts:
            continue

        notes = take_notes(
            question, path.name, texts, parts, context,
            what="paper" if path.suffix.lower() == ".pdf" else "file",
            whole=whole, failures=failures,
        )
        if notes:
            material.append(
                (path.name, [line for _, lines in notes for line in lines])
            )
    return material


def _cited(answer: str, count: int) -> set[int]:
    """Return the passage numbers cited in an answer, from 1 to ``count``.

    Citations may list numbers ([3, 5]) or ranges ([3-5]).

    Parameters
    ----------
    answer: str
        Answer text to inspect.
    count: int
        Maximum numbered passage to include.

    Returns
    -------
    set[int]
        the passage numbers cited in an answer, from 1 to ``count``.

    """

    numbers: set[int] = set()
    for group in _CITATION_GROUP.findall(answer):
        for piece in re.split(r"\s*,\s*", group):
            match = _CITATION_RANGE.fullmatch(piece)
            if match is None:
                continue

            first = int(match.group(1))
            last = int(match.group(2) or first)
            numbers.update(
                number
                for number in range(first, last + 1)
                if 1 <= number <= count
            )
    return numbers


def _estimate(
    parts: list[list[int]], texts: list[str]
) -> float:
    """Return the approximate seconds of the notes calls and the answer.

    Reply estimates do not cap model generation.

    Parameters
    ----------
    parts: list[list[int]]
        Parts of the document to process.
    texts: list[str]
        Text passages or fields to inspect.

    Returns
    -------
    float
        the approximate seconds of the notes calls and the answer.

    """

    total = 0.0
    for part in parts:
        prompt = _NOTES_PROMPT_TOKENS + sum(
            estimate_tokens(texts[index]) for index in part
        )
        total += _CALL_SECONDS + prompt / _PROMPT_TOKENS_PER_SECOND
        total += _config.NOTE_REPLY_TOKENS / _REPLY_TOKENS_PER_SECOND
    total += _config.ANSWER_REPLY_TOKENS / _REPLY_TOKENS_PER_SECOND
    return total
