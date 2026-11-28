"""Summarize one paper, chosen first, from its own passages.

"Give me a short summary of the Smith paper" or "summarize another paper"
is about one paper. A plain search mixes passages from several papers and
cannot tell which one "another paper" means, so this tool first settles the
paper with ``choose_document``: a paper the question names, the paper the
earlier turns discussed, or, when that is unclear, the user's choice. A
request for another paper, in any wording, leaves out the papers already
discussed; the router model judges the wording.

The answer is written from that paper only: its opening passages (title,
abstract, introduction) and its passages most relevant to the question, in
reading order, within ``max_tokens`` estimated tokens. The answer streams,
and citations to passages that do not exist are counted in the summary.
Where nobody can be asked (notebooks, scripts), a guessed paper is named
at the end of the answer.

Passages that do not fit the token budget are left out, so a long paper is
summarized from part of its text. The first passage is always included,
even when it alone is larger than the budget.

Key Features
------------

- **One paper, settled first**

    ``choose_document`` names, infers, or asks for the paper.

- **Opening and relevant passages**

    The opening passages and the best-ranked passages of the paper, in
    reading order, within a token budget.

Module Structure
----------------

- ``OverviewSettings``

    Settings of ``paper_overview`` in a project type.

- ``parse_settings``

    Validate ``paper_overview`` settings from a project type.

- ``overview_prompt``

    Prompts that summarize one paper from its passages.

- ``run``

    Summarize the paper the question means, asking when unclear.

Examples
--------
This is an illustrative integration call, not a standalone doctest.
KíLín supplies the required tool context and conversation state.

``context`` is the ``ToolContext`` that KíLín passes to a tool:

>>> from kilin.tools import ToolRequest
>>> from kilin.tools.paper_overview import OverviewSettings, run
>>> question = "give me a short summary of smith2020.pdf"
>>> request = ToolRequest(question, question)
>>> result = run(OverviewSettings(), request, context)

"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from kilin.chunking import Chunk
from kilin.retrievals.document_reading import estimate_tokens
from kilin.tools._shared.citations import inspect_citations
from kilin.tools._shared.document_notes import answer_citations
from kilin.tools._shared.choose_document import choose_document
from . import _configuration as _config
from kilin.tools import (
    ToolContext,
    ToolRequest,
    ToolResult,
    check_int,
    reject_unknown,
)

__all__ = [
    "OverviewSettings",
    "overview_prompt",
    "parse_settings",
    "run",
]

# Opening passages offered first (title, abstract, introduction); like
# the ranked ones, they are left out when they do not fit the budget.
_OPENING_PASSAGES: Final[int] = _config.OPENING_PASSAGES


@dataclass(frozen=True, slots=True)
class OverviewSettings:
    """Settings of ``paper_overview`` in a project type.

    Attributes
    ----------
    max_tokens: int
        Most estimated tokens of the paper given to the model (1000-20000).

    """

    max_tokens: int = _config.DEFAULT_MAX_TOKENS


def parse_settings(data: dict, where: str) -> OverviewSettings:
    """Validate ``paper_overview`` settings from a project type.

    Parameters
    ----------
    data: dict
        Settings from the intent's ``tool`` object, without ``name``.
    where: str
        Location for errors.

    Returns
    -------
    OverviewSettings
        Validated settings.

    Raises
    ------
    ValueError
        If a setting is unknown or out of range.

    """

    reject_unknown(where, data, ("max_tokens",))

    defaults = OverviewSettings()
    return OverviewSettings(
        max_tokens=check_int(
            where, data, "max_tokens", defaults.max_tokens, 1000, 20000
        ),
    )


def overview_prompt(
    question: str,
    name: str,
    passages: Sequence[tuple[int, Chunk]],
    guidance: str | None,
) -> tuple[str, str]:
    """Return the prompts that summarize one paper from its passages.

    Parameters
    ----------
    question: str
        The user's question.
    name: str
        The paper's file name.
    passages: Sequence[tuple[int, Chunk]]
        Citation numbers and passages, in reading order.
    guidance: str | None
        Answer guidance, such as a length target.

    Returns
    -------
    tuple[str, str]
        System prompt and user prompt.

    """

    system = (
        "You summarize one scientific paper for KíLín from passages of "
        "it. Use only the passages, cite them like [3], and never add "
        "results they do not state."
    )
    listed = "\n\n".join(
        f"[{number}] {chunk.text.strip()}" for number, chunk in passages
    )
    user = f"""Passages of {name}, in reading order:

{listed}

Answer the question below about this paper. If it asks for a length or
format (for example "in two sentences"), follow it exactly; otherwise keep
it short: a few sentences on what the paper does, how, and what it finds.\
{f" {guidance}" if guidance else ""}

Question: {question}"""
    return system, user


def run(
    settings: OverviewSettings, request: ToolRequest, context: ToolContext
) -> ToolResult:
    """Summarize the paper the question means, asking when unclear.

    Parameters
    ----------
    settings: OverviewSettings
        The intent's settings.
    request: ToolRequest
        The question, with the earlier turns it points back to.
    context: ToolContext
        KíLín's services.

    Returns
    -------
    ToolResult
        The streamed answer, the passages it was written from as evidence,
        and a summary of the choice.

    """

    context.progress("Finding the paper")
    chosen = choose_document(
        request,
        context,
        question="Which paper should I summarize?",
        example="give me a short summary of smith2020.pdf",
        prefer=(".pdf",),
    )
    summary: dict[str, object] = {"name": "paper_overview"}
    if isinstance(chosen, str):
        context.done("No paper chosen", evidence_count=0)
        return ToolResult(chosen, (), summary)
    path, reason, guessed = chosen
    summary.update(
        {"paper": path.as_posix(), "reason": reason, "guessed": guessed}
    )

    # The opening passages first, then the best-ranked ones, while they fit
    # the budget; the first passage is always taken.
    paper = [chunk for chunk in context.chunks if chunk.path == path]
    order = {chunk.identifier: index for index, chunk in enumerate(paper)}
    ranked = [
        order[result.chunk.identifier]
        for result in context.rank(request.question)
        if result.chunk.identifier in order
    ]
    picked: list[int] = []
    used = 0
    for index in list(range(_OPENING_PASSAGES)) + ranked:
        if index >= len(paper) or index in picked:
            continue
        size = estimate_tokens(paper[index].text)
        if picked and used + size > settings.max_tokens:
            continue
        picked.append(index)
        used += size
    chunks = [paper[index] for index in sorted(picked)]
    context.done(
        f"Read {len(chunks)} passages of {path.name}",
        evidence_count=len(chunks),
    )

    system, prompt = overview_prompt(
        request.question,
        path.name,
        list(enumerate(chunks, start=1)),
        request.guidance,
    )
    answer = context.write(system, prompt, None)
    citation_report = inspect_citations(answer, len(chunks))
    summary["invalid_citations"] = list(citation_report.invalid_numbers)
    if citation_report.invalid_groups:
        summary["invalid_citation_groups"] = list(
            citation_report.invalid_groups
        )
    if guessed:
        answer += (
            f"\n\n*This is about {path.name}; name the paper if you "
            "meant another one.*"
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
        for number, chunk in enumerate(chunks, start=1)
    )
    return ToolResult(answer, evidence, summary, streamed=True)
