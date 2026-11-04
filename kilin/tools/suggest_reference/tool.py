"""Tool ``suggest_reference``: find papers in the project to cite.

"Suggest a reference for: anomalous diffusion is common in living cells"
is about one statement and the project's papers. The tool:

1. Takes the statement from the question (``statement_of``): quoted text,
   the text after a colon, or what the router model copies out of the
   question, checked to be the user's own words.
2. Ranks project passages and keeps the papers with the best-ranked
   passages (``candidate_files``). Only a document explicitly identified
   as a draft is excluded; verbatim similarity does not prove identity.
3. Judges each paper with the answering model (``_shared.claims``): its
   passages most related to the statement, a verdict, the passage sentence
   it rests on (checked to be in the passages), and a critical comment.
   The plan is shown first when it is large.
4. Rates direct or partial support using the shared half-star rubric.
   Contextual background papers are shown separately, not as evidence
   directly establishing a claim.

Only the project's files are searched: a paper that is not in the folder
cannot be suggested, and the answer says so when none supports the
statement. Each paper is judged from a few passages, not all of it.

Key Features
------------

- **Passage-based suggestions**

    The model judges passages from each paper. A quotation is shown
    only when confirmed in those passages; a positive verdict may
    remain without a confirmed quotation.

- **Sources before repeats**

    The model is told that a paper repeating a claim with a citation is
    not its source, and to say so.

- **No false draft identification**

    A paper containing the exact statement can still be verified.

Module Structure
----------------

- ``SuggestReferenceSettings`` and ``parse_settings``

    Settings of ``suggest_reference`` in a project type.

- ``statement_of``

    The statement a question asks a reference for.

- ``candidate_files``

    The papers most related to the statement.

- ``chat_answer``

    The answer, with rated suggestions first.

- ``_rating.citation_suitability``

    Deterministic suitability score, with no extra model calls.

- ``run``

    Suggest references for the statement in the question.

Examples
--------
This is an illustrative integration call, not a standalone doctest.
KíLín supplies the required tool context and conversation state.

``context`` is the ``ToolContext`` that KíLín passes to a tool:

>>> from kilin.tools import ToolRequest
>>> from kilin.tools.suggest_reference import (
...     SuggestReferenceSettings, run)
>>> question = "suggest a reference for: cells move by diffusion"
>>> request = ToolRequest(question, question)
>>> result = run(SuggestReferenceSettings(), request, context)

"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
import re
from typing import Final

from kilin.tools._shared.model_results import partial_result
from kilin.chunking import Chunk
from kilin.tools._contract import (
    ToolContext,
    ToolRequest,
    ToolResult,
    check_choice,
    check_int,
    reject_unknown,
)
from kilin.tools._shared.claims import (
    VERDICT_BACKGROUND,
    VERDICT_CONTRADICTED,
    VERDICT_FAILED,
    VERDICT_NOT_FOUND,
    VERDICT_NOT_SUPPORTED,
    VERDICT_NO_TEXT,
    VERDICT_PARTLY,
    VERDICT_SUPPORTED,
    VERDICT_UNCLEAR,
    VERDICTS,
    Support,
    claim_passages,
    judge_support,
    quote_line_range,
    verdict_label,
)
from kilin.tools._shared.planning import (
    CONFIRMATION_CHOICES,
    PLAN_CHOICE_CANCEL,
    PLAN_CHOICE_QUICK,
    ConfirmationPolicy,
    ToolPlan,
    confirm_plan,
)
from kilin.tools.suggest_reference import _configuration as _config
from kilin.tools._shared._rating import (
    citation_suitability,
    suitability_stars,
)

__all__ = [
    "SuggestReferenceSettings",
    "candidate_files",
    "chat_answer",
    "parse_settings",
    "run",
    "statement_of",
]

# The verdicts that make a paper a suggestion, best first.
_SUGGESTED: Final[tuple[str, ...]] = (
    VERDICT_SUPPORTED, VERDICT_PARTLY, VERDICT_BACKGROUND,
)
# Order of the papers in the answer.
_ORDER: Final[dict[str, int]] = {
    verdict: index
    for index, verdict in enumerate((
        *_SUGGESTED, VERDICT_BACKGROUND, VERDICT_CONTRADICTED,
        VERDICT_NOT_SUPPORTED,
        VERDICT_NOT_FOUND, VERDICT_UNCLEAR, VERDICT_FAILED, VERDICT_NO_TEXT,
    ))
}
# Text in quotation marks: the statement, when the user quotes it.
_QUOTED: Final[re.Pattern[str]] = re.compile(r"[\"“«]([^\"”»]+)[\"”»]")
# A word, for comparing the statement with texts.
_WORD: Final[re.Pattern[str]] = re.compile(r"[^\W_]+")
# A bracketed number, which the chat would read as a link to evidence.
_BRACKETED: Final[re.Pattern[str]] = re.compile(
    r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]"
)


@dataclass(frozen=True, slots=True)
class SuggestReferenceSettings:
    """Settings of ``suggest_reference`` in a project type.

    Attributes
    ----------
    files: int
        Most papers judged, the most related first (1-20).
    passages: int
        Most passages of each paper read (1-20).
    passage_tokens: int
        Most estimated tokens of those passages (300-16000).
    quick_files: int
        Papers judged by the quick version (1-20).
    confirmation: ConfirmationPolicy
        When to ask before judging: always, when_large, or never.
    large_batches: int
        With ``when_large``, plans with more papers than this ask first
        (0-1000).

    """

    files: int = _config.DEFAULT_FILES
    passages: int = _config.DEFAULT_PASSAGES
    passage_tokens: int = _config.DEFAULT_PASSAGE_TOKENS
    quick_files: int = _config.DEFAULT_QUICK_FILES
    confirmation: ConfirmationPolicy = _config.DEFAULT_CONFIRMATION
    large_batches: int = _config.DEFAULT_LARGE_BATCHES


def parse_settings(data: dict, where: str) -> SuggestReferenceSettings:
    """Validate ``suggest_reference`` settings from a project type.

    Parameters
    ----------
    data: dict
        Settings from the intent's ``tool`` object, without ``name``.
    where: str
        Location for errors.

    Returns
    -------
    SuggestReferenceSettings
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
            "files", "passages", "passage_tokens",
            "quick_files", "confirmation", "large_batches",
        ),
    )
    defaults = SuggestReferenceSettings()
    return SuggestReferenceSettings(
        files=check_int(where, data, "files", defaults.files, 1, 20),
        passages=check_int(where, data, "passages", defaults.passages, 1, 20),
        passage_tokens=check_int(
            where, data, "passage_tokens", defaults.passage_tokens, 300,
            16000,
        ),
        quick_files=check_int(
            where, data, "quick_files", defaults.quick_files, 1, 20
        ),
        confirmation=check_choice(
            where, data, "confirmation", defaults.confirmation,
            CONFIRMATION_CHOICES,
        ),
        large_batches=check_int(
            where, data, "large_batches", defaults.large_batches, 0, 1000
        ),
    )


def statement_of(
    question: str, ask_router: Callable[[str, str, int], str] | None
) -> str:
    """Return the statement a question asks a reference for.

    Parameters
    ----------
    question: str
        The question as typed.
    ask_router: Callable[[str, str, int], str] | None
        The router model (``ToolContext.ask_router``), asked only when the
        statement is neither quoted nor after a colon.

    Returns
    -------
    str
        The longest quoted text, or the text after the first colon, when
        it has at least ``MIN_WORDS`` words; otherwise what the router
        copies out of the question, when most of its words are the
        question's; otherwise the whole question.

    """

    quoted = max(
        (match.group(1).strip() for match in _QUOTED.finditer(question)),
        key=len,
        default="",
    )
    if len(quoted.split()) >= _config.MIN_WORDS:
        return quoted
    _, colon, after = question.partition(":")
    if colon and len(after.split()) >= _config.MIN_WORDS:
        return after.strip()

    copied = ""
    if ask_router is not None:
        system = "You copy text exactly, without comment."
        prompt = (
            "A user asks for a reference for a statement:\n\n"
            f"{question}\n\n"
            "Copy the statement that needs a reference exactly as written, "
            'without the request around it ("suggest a reference for", '
            '"which paper supports"). If there is none, answer NONE.'
        )
        try:
            copied = ask_router(system, prompt, _config.ROUTER_TOKENS)
        except Exception:  # The whole question is used instead.
            copied = ""
        copied = copied.strip().strip("\"“”'").strip()
    words = _words(copied)
    asked = set(_words(question))
    if (
        len(words) >= _config.MIN_WORDS
        and copied.upper() != "NONE"
        and sum(word in asked for word in words) / len(words)
        >= _config.ROUTER_WORD_SHARE
    ):
        return copied
    return question.strip()


def candidate_files(
    statement: str,
    by_file: dict[Path, list[Chunk]],
    rank: Callable[[str], list],
    count: int,
    excluded: Sequence[Path] = (),
) -> tuple[list[Path], list[Path]]:
    """Return the papers most related to a statement.

    Parameters
    ----------
    statement: str
        The statement.
    by_file: dict[pathlib.Path, list[Chunk]]
        Every file's passages, in order.
    rank: Callable[[str], list]
        KíLín's ranking of every passage for a text; results have a
        ``chunk``.
    count: int
        Most papers.
    excluded: Sequence[pathlib.Path], optional
        Explicitly identified draft paths to leave out. A verbatim
        sentence match is not a reliable draft-identity check.

    Returns
    -------
    tuple[list[pathlib.Path], list[pathlib.Path]]
        Ranked papers and explicitly excluded document paths.

    """

    holding = [path for path in excluded if path in by_file]
    picked: list[Path] = []
    for result in rank(statement):
        path = result.chunk.path
        if path in by_file and path not in holding and path not in picked:
            picked.append(path)
            if len(picked) == count:
                break
    return picked, holding


def chat_answer(
    statement: str, supports: Sequence[Support], note: str = ""
) -> tuple[str, tuple[Chunk, ...]]:
    """Return the answer and the quoted passages it cites.

    Parameters
    ----------
    statement: str
        The statement.
    supports: Sequence[Support]
        Each paper's judgement, in any order.
    note: str, optional
        Why some papers were not judged. Defaults to ``""``.

    Returns
    -------
    tuple[str, tuple[Chunk, ...]]
        Markdown for the chat, and the passages its "[n]" cite, in order.

    """

    ratings = {
        support.cited: citation_suitability(support)
        for support in supports
    }
    ordered = sorted(
        supports,
        key=lambda s: (
            _ORDER.get(s.verdict, 99),
            -(ratings[s.cited].score if ratings[s.cited] else 0),
        ),
    )
    suggested = [s for s in ordered if s.verdict in _SUGGESTED]
    background = [s for s in ordered if s.verdict == VERDICT_BACKGROUND]
    others = [
        s for s in ordered
        if s.verdict not in _SUGGESTED and s.verdict != VERDICT_BACKGROUND
    ]
    lines = [f"**Statement:** “{_escape(statement)}”", ""]
    evidence: list[Chunk] = []

    if suggested:
        lines.append(
            f"**{len(suggested)} of the {len(ordered)} papers checked could "
            "be cited for it:**"
        )
        for index, support in enumerate(suggested, start=1):
            label = verdict_label(support.verdict).capitalize()
            rating = ratings[support.cited]
            assert rating is not None  # Each suggested verdict is rated.
            stars = suitability_stars(rating)
            lines += [
                "", f"{index}. **{label}** · "
                f"`{support.cited.as_posix()}` · {stars}"
            ]
            if support.quote and support.quote_chunk is not None:
                evidence.append(support.quote_chunk)
                lines.append(
                    f"   *It says:* “{_escape(support.quote)}” "
                    f"[{len(evidence)}]"
                )
            if support.evidence_status in ("missing", "rejected"):
                lines.append(
                    "   *Provisional: no independently verified quotation "
                    f"(evidence {support.evidence_status}).*"
                )
            elif support.evidence_status == "approximate":
                lines.append("   *Evidence: approximate quotation match.*")
            if support.comment:
                lines.append(f"   {_escape(support.comment)}")
        lines += [
            "",
            "*Stars and scores in parentheses (out of 100) indicate "
            "heuristic citation suitability, based on the support "
            "verdict, confirmed evidence, and textual overlap; "
            "they are not probabilities or paper-quality ratings.*",
        ]
    else:
        lines.append(
            "No direct or partial support was established among the "
            f"{len(ordered)} papers checked. This does not establish "
            "that none supports the statement. Only the project's files "
            "are searched: add the paper you have in mind to the folder "
            "to check it."
        )

    if background:
        listed = ", ".join(
            f"`{s.cited.as_posix()}` ({suitability_stars(ratings[s.cited])})"
            for s in background
        )
        lines += [
            "", "**Contextual or methodological references, not direct "
            f"claim support:** {listed}.",
        ]
    inconclusive = sum(
        s.verdict in (VERDICT_UNCLEAR, VERDICT_FAILED, VERDICT_NO_TEXT,
                      VERDICT_NOT_FOUND)
        for s in ordered
    )
    if inconclusive:
        label = "assessment was" if inconclusive == 1 else "assessments were"
        lines += [
            "", f"*{inconclusive} {label} inconclusive "
            "(unclear, failed, no text, or no matching passage).*",
        ]
    if others:
        listed = ", ".join(
            f"`{s.cited.as_posix()}` ({verdict_label(s.verdict)})"
            for s in others
        )
        lines += ["", f"Also checked: {listed}."]
    if note:
        lines += ["", f"*{note}*"]
    return "\n".join(lines), tuple(evidence)


def run(
    settings: SuggestReferenceSettings,
    request: ToolRequest,
    context: ToolContext,
) -> ToolResult:
    """Suggest papers of the project to cite for a statement.

    Parameters
    ----------
    settings: SuggestReferenceSettings
        The intent's settings.
    request: ToolRequest
        The question.
    context: ToolContext
        KíLín's services.

    Returns
    -------
    ToolResult
        The answer, the quoted passages as evidence, and a summary of the
        verdicts.

    """

    summary: dict[str, object] = {"name": "suggest_reference"}
    statement = statement_of(
        request.question, getattr(context, "ask_router", None)
    )
    summary["statement"] = statement
    if len(_words(statement)) < _config.MIN_WORDS:
        context.done("No statement", evidence_count=0)
        return ToolResult(
            "Which sentence needs a reference? Write it after a colon, "
            "for example: suggest a reference for: anomalous diffusion is "
            "common in living cells.",
            (), summary,
        )

    by_file: dict[Path, list[Chunk]] = defaultdict(list)
    for chunk in context.chunks:
        by_file[chunk.path].append(chunk)
    # The statement is ranked once, for the papers and for each paper's
    # passages.
    rank = cache(context.rank)

    context.progress("Finding the papers most related to the statement")
    files, holding = candidate_files(statement, by_file, rank, settings.files)
    summary["left_out"] = [path.as_posix() for path in holding]
    if not files:
        context.done("No paper to check", evidence_count=0)
        return ToolResult(
            "The project has no other paper to check the statement against.",
            (), summary,
        )

    quick = min(len(files), settings.quick_files)
    plan = ToolPlan(
        tool="suggest_reference",
        summary=(
            f"Check whether the {len(files)} papers most related to the "
            "statement support it, one model call each."
        ),
        scope=f"{settings.passages} passages of each paper.",
        items=len(files),
        batches=len(files),
        batches_to_run=len(files),
        estimated_seconds=len(files) * _paper_seconds(settings),
        quick_items=quick,
        quick_seconds=quick * _paper_seconds(settings),
        quick_label=f"The {quick} most related papers",
    )
    choice, _asked = confirm_plan(
        plan, settings.confirmation, settings.large_batches, context
    )
    if choice == PLAN_CHOICE_CANCEL:
        context.done("Cancelled", evidence_count=0)
        listed = ", ".join(f"`{path.as_posix()}`" for path in files)
        return ToolResult(
            f"Not checked (cancelled). The papers most related to the "
            f"statement are {listed}.",
            (), summary,
        )

    to_check = files[:quick] if choice == PLAN_CHOICE_QUICK else files
    supports: list[Support] = []
    for index, path in enumerate(to_check, start=1):
        context.progress(f"Checking paper {index} of {len(to_check)}")
        passages = claim_passages(
            (statement,), by_file[path], rank, settings.passages,
            settings.passage_tokens,
        )
        support = judge_support(
            statement, path, passages, context.ask_model,
        )
        supports.append(support)
        rating = citation_suitability(support)
        context.trace.emit(
            "tool.suggest_reference.paper",
            component="tool",
            data={
                "paper": path.as_posix(),
                "verdict": support.verdict,
                "quote_found": support.quote_chunk is not None,
                "evidence_status": support.evidence_status,
                "quote_passage_id": (
                    support.quote_chunk.identifier
                    if support.quote_chunk is not None else None
                ),
                "quote_lines": quote_line_range(
                    support.quote, support.quote_chunk,
                ),
                "suitability_stars": rating.stars if rating else None,
                "suitability_score": rating.score if rating else None,
            },
            turn_id=context.trace_turn_id,
        )

    notes = []
    if len(to_check) < len(files):
        notes.append(
            f"Only the {len(to_check)} most related of {len(files)} papers "
            "were checked (the quick version)."
        )
    if holding:
        left_out = ", ".join(f"`{path.as_posix()}`" for path in holding)
        notes.append(
            f"Explicitly excluded draft document(s): {left_out}."
        )
    failures = [
        {"item": support.cited.as_posix(), "message": support.comment,
         "reason": "model_result"}
        for support in supports if support.verdict == VERDICT_FAILED
    ]
    answer, quoted = chat_answer(statement, supports, " ".join(notes))
    verdicts = Counter(support.verdict for support in supports)
    summary.update(
        {
            "papers_checked": len(supports),
            "background": verdicts[VERDICT_BACKGROUND],
            "inconclusive": sum(verdicts[v] for v in (
                VERDICT_UNCLEAR, VERDICT_FAILED, VERDICT_NO_TEXT,
                VERDICT_NOT_FOUND,
            )),
            "evidence_statuses": dict(Counter(
                support.evidence_status for support in supports
            )),
            "suggested": sum(verdicts[v] for v in _SUGGESTED),
            "verdicts": {v: verdicts[v] for v in VERDICTS if verdicts[v]},
            "suitability": {
                support.cited.as_posix(): {
                    "stars": rating.stars,
                    "score": rating.score,
                }
                for support in supports
                if (rating := citation_suitability(support)) is not None
            },
        }
    )
    context.done(
        f"Checked {len(supports)} papers", evidence_count=len(quoted)
    )
    evidence = tuple(
        {
            "citation": number,
            "path": chunk.path,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "symbol": chunk.symbol,
            "text": chunk.text,
            "cell": chunk.cell,
        }
        for number, chunk in enumerate(quoted, start=1)
    )
    return partial_result(
        ToolResult(answer, evidence, summary), failures,
        completed=len(supports) > len(failures),
    )


def _paper_seconds(settings: SuggestReferenceSettings) -> float:
    """Return the estimated time of judging one paper, in seconds.

    Parameters
    ----------
    settings: SuggestReferenceSettings
        For the passage and answer budgets.

    Returns
    -------
    float
        A planning estimate, not a limit.

    """

    prompt = _config.PROMPT_OVERHEAD_TOKENS + settings.passage_tokens
    # Expected output is a planning estimate, never a generation cap.
    reply = _config.JUDGE_REPLY_TOKENS * _config.REPLY_SHARE
    return (
        _config.CALL_SECONDS
        + prompt / _config.PROMPT_TOKENS_PER_SECOND
        + reply / _config.REPLY_TOKENS_PER_SECOND
    )


def _words(text: str) -> list[str]:
    """Return a text's words, case-folded.

    Parameters
    ----------
    text: str
        Any text.

    Returns
    -------
    list[str]
        Its letters-and-digits words, in order.

    """

    return [word.casefold() for word in _WORD.findall(text)]


def _escape(text: str) -> str:
    """Return text without bracketed numbers the chat would link.

    Parameters
    ----------
    text: str
        A quote or comment.

    Returns
    -------
    str
        The text with "[12]" written "(12)".

    """

    return _BRACKETED.sub(r"(\1)", text)
