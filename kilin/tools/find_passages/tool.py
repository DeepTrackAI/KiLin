"""List every sentence, or every file, about a topic, judged by meaning.

"List the sentences that discuss network analysis" asks for meaning, not
exact words: a sentence about "graph measures of how the parts of a system
connect" belongs, and one that only names "network" in passing does not.
Ranked retrieval would show only the best few passages. This tool reads
every sentence in scope, in batches, and the model only answers which
numbered sentences are about the topic. Python copies those sentences word
for word with their file and line, so nothing is paraphrased or invented.

One router call names the topic, any one document the question names, and
whether the user wants sentences or files. When its reply is unusable, the
topic is the question without its request words ("list the sentences
that ..."). A document the question names, or points at, narrows the
scope; otherwise every indexed file is read. The plan can be confirmed
first. Its quick version is an exact match without the model when the
question names exact terms, and otherwise the sentences of the
best-ranked passages. When the question asks for files ("which papers use
XYZ 2 as an analysis tool?") and more than one file is in scope, the
answer lists the matching files, each with its number of matching
sentences and one example. A long answer can be saved as a Markdown file
instead of shown in the chat.

Sentences have at least four words, so headings and labels are never
listed. An incomplete or failed batch stops the operation instead of
reporting zero matches. Numbers outside the current batch are ignored
only when valid selected numbers remain; zero matches require NONE.
Batches beyond ``max_model_calls`` are
not read, and the answer says so.

Key Features
------------

- **Sentences without repeats**

    Overlapping passages share sentences; each sentence is kept once, with
    its file and first line.

- **Judged by meaning, copied word for word**

    The model only gives sentence numbers; the answer quotes the sentences
    themselves.

- **Sentences or files**

    List the matching sentences by file, or the matching files with a count
    and an example.

- **Confirmed plans**

    Large runs are offered first with a quick version: an exact match
    without the model, or a sample of the best-ranked passages.

Module Structure
----------------

- ``PassageSettings``

    Settings of ``find_passages`` in a project type.

- ``parse_settings``

    Validate ``find_passages`` settings from a project type.

- ``Sentence``

    One sentence of a passage, with its file and first line.

- ``split_sentences``

    Split passages into sentences, without repeats.

- ``topic_prompt``

    Prompts that ask the router model for the topic and any one document.

- ``judge_prompt``

    Prompts that ask the model which numbered sentences are about a topic.

- ``run``

    List every sentence in scope that talks about the question's topic.

Private helpers find the topic, pack and judge the batches, estimate the
time, and write the answer; they are not part of the public API.

Examples
--------
``context`` is the ``ToolContext`` that KíLín passes to a tool:

>>> from kilin.tools import ToolRequest
>>> from kilin.tools.find_passages import PassageSettings, run
>>> question = "list the sentences that discuss network analysis"
>>> request = ToolRequest(question, question)
>>> result = run(PassageSettings(), request, context)

"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
import json
from pathlib import Path
import re
from typing import Final

from kilin._model_completion import IncompleteModelResponse
from kilin.retrievals.document_reading import estimate_tokens, named_paths
from kilin.tools._shared.choose_document import document_scope
from kilin.tools.find_passages import _configuration as _config
from kilin.tools._shared.model_results import selected_numbers
from kilin.tools._shared.sentences import Sentence, split_sentences
from kilin.tools._contract import (
    ToolContext,
    ToolRequest,
    ToolResult,
    check_choice,
    check_int,
    reject_unknown,
)
from kilin.tools._shared.long_lists import (
    CHAT_LIST_LIMIT,
    file_name,
    offer_export,
)
from kilin.tools._shared.planning import (
    CONFIRMATION_CHOICES,
    PLAN_CHOICE_CANCEL,
    PLAN_CHOICE_QUICK,
    ConfirmationPolicy,
    PlanChoice,
    ToolPlan,
    confirm_plan,
)
from kilin.tools._shared.paths import short_names
from kilin.tools._shared.terms import term_pattern
from kilin.tools._shared.term_filter import term_hits, term_label

__all__ = [
    "PassageSettings",
    "Sentence",
    "judge_prompt",
    "parse_settings",
    "run",
    "split_sentences",
    "topic_prompt",
]

# A sentence ends at ".", "!", or "?" followed by space and a capital,
# digit, quote, or bracket, or at a blank line.

# A passage that starts with a lowercase letter starts inside a sentence.

# Shorter pieces, such as headings and labels, are not sentences.
# Words that ask for passages rather than name the topic; they are removed
# from the question when the router gives no usable topic.
_REQUEST_WORDS: Final[re.Pattern[str]] = re.compile(
    r"^.*?\b(?:sentences?|passages?|parts?|paragraphs?|places?)\b"
    r"(?:\s+(?:that|which|where))?(?:\s+(?:are|is))?"
    r"(?:\s+(?:talking|talk|talks|discussing|discuss|discusses|about|on|"
    r"mentioning|mention|describing|describe))*\s*",
    flags=re.I,
)
# Longest replies: the router's line of JSON and a batch's numbers.
_TOPIC_REPLY_TOKENS: Final[int] = _config.TOPIC_REPLY_TOKENS
_JUDGE_REPLY_TOKENS: Final[int] = _config.JUDGE_REPLY_TOKENS
# Estimated tokens that a sentence's number and line break add to a batch.
_SENTENCE_NUMBER_TOKENS: Final[int] = _config.SENTENCE_NUMBER_TOKENS
# Rough speeds of an 8B 4-bit model on Apple silicon, for estimates only.
_PROMPT_TOKENS_PER_SECOND: Final[float] = _config.PROMPT_TOKENS_PER_SECOND
_REPLY_TOKENS_PER_SECOND: Final[float] = _config.REPLY_TOKENS_PER_SECOND
# For estimates only: a judge prompt without its sentences, a typical judge
# reply, and the fixed cost of one model call.
_JUDGE_PROMPT_TOKENS: Final[int] = _config.JUDGE_PROMPT_TOKENS
_JUDGE_REPLY_ESTIMATE_TOKENS: Final[int] = _config.JUDGE_REPLY_ESTIMATE_TOKENS
_CALL_SECONDS: Final[float] = _config.CALL_SECONDS


@dataclass(frozen=True, slots=True)
class PassageSettings:
    """Settings of ``find_passages`` in a project type.

    Attributes
    ----------
    batch_tokens: int
        Estimated tokens of sentences per model call (500-8000).
    max_model_calls: int
        Most batches (1-500).
    quick_passages: int
        Best-ranked passages read by the quick sample (5-200).
    max_listed: int
        Most sentences listed in the answer (10-1000); all are counted.
    confirmation: "always" | "when_large" | "never"
        When to ask before running.
    large_batches: int
        With ``when_large``, plans with more batches than this ask first.

    """

    batch_tokens: int = _config.DEFAULT_BATCH_TOKENS
    max_model_calls: int = _config.DEFAULT_MAX_MODEL_CALLS
    quick_passages: int = _config.DEFAULT_QUICK_PASSAGES
    max_listed: int = _config.DEFAULT_MAX_LISTED
    confirmation: ConfirmationPolicy = _config.DEFAULT_CONFIRMATION
    large_batches: int = _config.DEFAULT_LARGE_BATCHES


def parse_settings(data: dict, where: str) -> PassageSettings:
    """Validate ``find_passages`` settings from a project type.

    Parameters
    ----------
    data: dict
        Settings from the intent's ``tool`` object, without ``name``.
    where: str
        Location for errors.

    Returns
    -------
    PassageSettings
        Validated settings.

    Raises
    ------
    ValueError
        If a setting is unknown or out of range.

    """

    names = (
        "batch_tokens",
        "max_model_calls",
        "quick_passages",
        "max_listed",
        "confirmation",
        "large_batches",
    )
    reject_unknown(where, data, names)

    defaults = PassageSettings()
    return PassageSettings(
        batch_tokens=check_int(
            where, data, "batch_tokens", defaults.batch_tokens, 500, 8000
        ),
        max_model_calls=check_int(
            where, data, "max_model_calls", defaults.max_model_calls, 1, 500
        ),
        quick_passages=check_int(
            where, data, "quick_passages", defaults.quick_passages, 5, 200
        ),
        max_listed=check_int(
            where, data, "max_listed", defaults.max_listed, 10, 1000
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






def topic_prompt(question: str) -> tuple[str, str]:
    """Return the prompts that ask for the topic and any one document.

    Parameters
    ----------
    question: str
        The question.

    Returns
    -------
    tuple[str, str]
        System prompt and user prompt; the reply is one line of JSON with
        ``topic``, ``document``, and ``list`` (``files`` or
        ``sentences``).

    """

    system = (
        "You pick the topic of a search. You never answer questions. You "
        "reply with one line of JSON only."
    )
    user = f"""Message: {question}

Return one line of JSON with:
"topic": what the sentences the user wants are about, in the user's words,
without words such as "list", "sentences", "passages", or "talking about";
"document": the words naming one particular document to search, copied
exactly from the message (for example "the Smith paper"), or null;
"list": "files" when the user wants the files (or papers) that match,
"sentences" when they want the matching sentences or passages.

Examples:
"list the sentences in the Smith paper that discuss limitations" ->
{{"topic": "limitations", "document": "the Smith paper", "list": "sentences"}}
"which papers use deep learning?" ->
{{"topic": "use deep learning", "document": null, "list": "files"}}"""
    return system, user


def judge_prompt(
    topic: str, sentences: Sequence[Sentence], question: str = ""
) -> tuple[str, str]:
    """Return the prompts that pick the sentences about a topic.

    Parameters
    ----------
    topic: str
        The topic.
    sentences: Sequence[Sentence]
        One batch, numbered from 1 in the prompt.
    question: str, optional
        The user's request, so the judgement follows its meaning (for
        example "uses XYZ 2 as an analysis tool", not just "XYZ 2").
        Defaults to ``""``.

    Returns
    -------
    tuple[str, str]
        System prompt and user prompt; the reply is the numbers of the
        sentences about the topic.

    """

    system = (
        "You judge which sentences are about a topic. You reply with "
        "sentence numbers only, separated by commas, or NONE."
    )
    listed = "\n".join(
        f"{number}. {sentence.text}"
        for number, sentence in enumerate(sentences, start=1)
    )
    request = f"The user asked: {question}\n" if question else ""
    user = f"""{request}Topic: {topic}

Which sentences below match what the user is looking for? Include a
sentence when it matches, even in other words; leave out sentences that
only use a word in passing. Reply with their numbers, for example: 2, 7,
11. Reply NONE when no sentence matches.

{listed}"""
    return system, user


def run(
    settings: PassageSettings, request: ToolRequest, context: ToolContext
) -> ToolResult:
    """List every sentence in scope that talks about the question's topic.

    Parameters
    ----------
    settings: PassageSettings
        The intent's settings.
    request: ToolRequest
        The question.
    context: ToolContext
        KíLín's services.

    Returns
    -------
    ToolResult
        The sentences by file with citations (the first ``max_listed``),
        the passages they come from as evidence, and a summary. When the
        user saves the list as a file, the answer is the counts and the
        file's path, without evidence.

    """

    summary: dict[str, object] = {"name": "find_passages"}
    context.progress("Finding the topic")
    topic, document, by_file = _topic(request.question, context)
    summary.update({"topic": topic, "document": document})
    if not topic:
        context.done("No topic found", evidence_count=0)
        return ToolResult(
            "I couldn't tell which topic to look for. Try, for example: "
            "list the sentences that discuss network analysis.",
            (),
            summary,
        )

    # The router's document phrase is a fallback for a document that
    # ``document_scope`` does not find in the question itself.
    indexed = list(dict.fromkeys(chunk.path for chunk in context.chunks))
    scope, _ = document_scope(request, context.ask_router, indexed)
    if not scope and document:
        scope = named_paths(document, indexed, authors=True)
    chunks = sorted(
        (
            chunk
            for chunk in context.chunks
            if not scope or chunk.path in scope
        ),
        key=lambda chunk: (str(chunk.path), chunk.start_line),
    )
    sentences = split_sentences(chunks, min_words=_config.MIN_WORDS)
    batches = _batches(sentences, settings.batch_tokens)
    to_run = batches[: settings.max_model_calls]

    # Quick: with exact terms in the question, every sentence that contains
    # one, found by Python without the model; otherwise the sentences of
    # the best-ranked passages, judged by the model.
    terms, hits = term_hits(request.question, context, chunks)
    ranked = context.rank(topic) if context.chunks else []
    if terms:
        quick_chunks = [chunks[index] for index in hits]
    else:
        quick_chunks = [
            result.chunk
            for result in ranked[: settings.quick_passages]
            if not scope or result.chunk.path in scope
        ]
    quick_sentences = split_sentences(
        quick_chunks, min_words=_config.MIN_WORDS
    )
    quick = _batches(quick_sentences, settings.batch_tokens)
    patterns = [term_pattern(term, False, True) for term in terms]
    exact = [
        sentence
        for sentence in quick_sentences
        if any(pattern.search(sentence.text) for pattern in patterns)
    ]
    if terms:
        label, note = term_label(terms, exact, "sentence")
        quick_items, quick_seconds = len(exact), 0.0
    else:
        label, note = "", ""
        quick_items = sum(len(batch) for batch in quick)
        quick_seconds = round(_estimate(quick), 1)

    files = len({chunk.path for chunk in chunks})
    plan = ToolPlan(
        tool="find_passages",
        summary=(
            f"Read all {len(sentences)} sentences in {files} files and list "
            f"the ones about “{topic}”."
        ),
        scope=(
            ", ".join(path.name for path in scope)
            if scope
            else f"All {files} indexed files"
        ),
        items=len(sentences),
        batches=len(batches),
        batches_to_run=len(to_run),
        estimated_seconds=round(_estimate(to_run), 1),
        quick_items=quick_items,
        quick_seconds=quick_seconds,
        details=(f"Model: {context.model_id}",),
        quick_label=label,
        quick_note=note,
    )
    choice, asked = confirm_plan(
        plan, settings.confirmation, settings.large_batches, context
    )
    summary.update({"plan": plan.to_dict(), "choice": choice, "asked": asked})
    if choice == PLAN_CHOICE_CANCEL:
        context.done("Cancelled before reading", evidence_count=0)
        return ToolResult(
            "Cancelled; no model calls were made.", (), summary
        )

    exact_only = choice == PLAN_CHOICE_QUICK and bool(terms)
    if choice == PLAN_CHOICE_QUICK:
        runs = [] if terms else quick
    else:
        runs = to_run
    if not runs and terms:
        found = list(exact)  # The exact match needs no reading.
    else:
        found = _judge(topic, runs, request.question, context)

    read = sum(len(batch) for batch in runs)
    summary.update(
        {"sentences": len(sentences), "read": read, "found": len(found)}
    )
    if exact_only:
        summary["exact_terms"] = list(terms)
    context.done(
        f"Found {len(found)} sentences", evidence_count=len(found)
    )

    # Files are listed when the user asked for files and more than one
    # file is in scope; one file in scope means its sentences.
    list_files = by_file and files > 1
    summary["by_file"] = list_files
    write_answer = partial(
        _result,
        topic=topic,
        found=found,
        read=read,
        total=len(sentences),
        files=files,
        choice=choice,
        summary=summary,
        by_file=list_files,
        exact_terms=terms if exact_only else (),
    )
    full = write_answer(limit=len(found))
    count = len({s.path for s in found}) if list_files else len(found)
    saved = offer_export(
        context,
        count=count,
        what=("files" if list_files else "sentences")
        + f" about “{topic}”",
        name=file_name("passages", topic),
        markdown=full.answer,
    )
    if saved:
        summary["saved"] = saved.as_posix()
        counts = full.answer.split("\n", 1)[0]
        return ToolResult(
            f"{counts}\n\nThe full list is saved as `{saved.as_posix()}`.",
            (),
            summary,
        )

    # When the user could be asked, the chat shows at most
    # ``CHAT_LIST_LIMIT`` items, as the offer to save the list said.
    limit = (
        min(settings.max_listed, CHAT_LIST_LIMIT)
        if context.choose is not None
        else settings.max_listed
    )
    return write_answer(limit=limit)


def _topic(
    question: str, context: ToolContext
) -> tuple[str, str | None, bool]:
    """Return the topic, any document phrase, and whether files are wanted.

    The router's topic is used only when it shares a word with the
    question; otherwise the topic is the question without its request
    words. The document phrase is kept only when the question contains it.

    Parameters
    ----------
    question: str
        Question or request to analyze.
    context: ToolContext
        Tool context providing router data and sources.

    Returns
    -------
    tuple[str, str | None, bool]
        the topic, any document phrase, and whether files are wanted.

    """

    system, prompt = topic_prompt(question)
    topic, document, by_file = "", None, False
    try:
        reply = context.ask_router(system, prompt, _TOPIC_REPLY_TOKENS)
        match = re.search(r"\{.*\}", reply, flags=re.S)
        data = json.loads(match.group(0)) if match else {}
        topic = str(data.get("topic") or "").strip()
        value = data.get("document")
        if isinstance(value, str) and value.casefold() in question.casefold():
            document = value
        by_file = data.get("list") == "files"
    except Exception:  # Fall back to the wording below.
        topic = ""

    words = set(re.findall(r"\w+", question.casefold()))
    if not topic or not set(re.findall(r"\w+", topic.casefold())) & words:
        topic = _REQUEST_WORDS.sub("", question).strip(" ?.!")
    return topic, document, by_file


def _batches(sentences: list[Sentence], tokens: int) -> list[list[Sentence]]:
    """Pack consecutive sentences into batches of about ``tokens`` tokens.

    A sentence larger than ``tokens`` gets a batch of its own.

    Parameters
    ----------
    sentences: list[Sentence]
        Sentences to arrange or render.
    tokens: int
        Approximate token budget or limit.

    Returns
    -------
    list[list[Sentence]]
        Consecutive sentence batches respecting the token budget.

    """

    batches: list[list[Sentence]] = []
    used = 0
    for sentence in sentences:
        size = estimate_tokens(sentence.text) + _SENTENCE_NUMBER_TOKENS
        if batches and used + size <= tokens:
            batches[-1].append(sentence)
            used += size
        else:
            batches.append([sentence])
            used = size
    return batches


def _estimate(batches: list[list[Sentence]]) -> float:
    """Return the approximate seconds to judge the batches.

    Parameters
    ----------
    batches: list[list[Sentence]]
        Consecutive sentence groups to evaluate.

    Returns
    -------
    float
        the approximate seconds to judge the batches.

    """

    total = 0.0
    for batch in batches:
        prompt = _JUDGE_PROMPT_TOKENS + sum(
            estimate_tokens(sentence.text) + _SENTENCE_NUMBER_TOKENS
            for sentence in batch
        )
        total += _CALL_SECONDS + prompt / _PROMPT_TOKENS_PER_SECOND
        total += _JUDGE_REPLY_ESTIMATE_TOKENS / _REPLY_TOKENS_PER_SECOND
    return total


def _judge(
    topic: str,
    batches: Sequence[Sequence[Sentence]],
    question: str,
    context: ToolContext,
) -> list[Sentence]:
    """Return the sentences the model judges to be about the topic.

    Failed or malformed batches stop the operation. Only an explicit
    NONE reply establishes that a batch contains no matches.

    Parameters
    ----------
    topic: str
        Topic against which sentence batches are classified.
    batches: Sequence[Sequence[Sentence]]
        Consecutive sentence groups to evaluate.
    question: str
        Question or request to analyze.
    context: ToolContext
        Tool context for the current request.

    Returns
    -------
    list[Sentence]
        the sentences the model judges to be about the topic.

    """

    found: list[Sentence] = []
    for number, batch in enumerate(batches, start=1):
        context.progress(
            f"Reading sentences (batch {number} of {len(batches)})",
            current=number,
            total=len(batches),
        )
        system, prompt = judge_prompt(topic, batch, question)
        try:
            reply = context.ask_model(system, prompt, _JUDGE_REPLY_TOKENS)
        except IncompleteModelResponse:
            raise
        except Exception as error:  # Never turn failure into zero matches.
            context.trace.emit(
                "tool.find_passages.batch_failed",
                component="tool",
                data={"batch": number, "error": str(error)},
                turn_id=context.trace_turn_id,
            )
            raise IncompleteModelResponse(
                f"Sentence classification failed in batch {number}: {error}",
                reason="model_error",
            ) from error

        picked = selected_numbers(reply, range(1, len(batch) + 1))
        found.extend(batch[index - 1] for index in picked)
    return found


def _evidence(
    found: Sequence[Sentence],
) -> tuple[dict[str, int], list[dict[str, object]]]:
    """Number the passages of the sentences, in order of first use.

    Parameters
    ----------
    found: Sequence[Sentence]
        Sentences selected as evidence.

    Returns
    -------
    tuple[dict[str, int], list[dict[str, object]]]
        Citation numbers by passage identifier, and the passages as
        evidence.

    """

    citations: dict[str, int] = {}
    evidence: list[dict[str, object]] = []
    for sentence in found:
        chunk = sentence.chunk
        if chunk.identifier in citations:
            continue

        citations[chunk.identifier] = len(citations) + 1
        evidence.append(
            {
                "citation": citations[chunk.identifier],
                "path": chunk.path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "symbol": chunk.symbol,
                "text": chunk.text,
                "cell": chunk.cell,
            }
        )
    return citations, evidence


def _result(
    *,
    topic: str,
    found: Sequence[Sentence],
    read: int,
    total: int,
    files: int,
    choice: PlanChoice,
    summary: dict[str, object],
    by_file: bool,
    exact_terms: Sequence[str],
    limit: int,
) -> ToolResult:
    """Write the answer: counts, then the sentences or files that match.

    Parameters
    ----------
    topic: str
        The topic.
    found: Sequence[Sentence]
        The matching sentences, in file and line order.
    read: int
        Number of sentences the model read.
    total: int
        Number of sentences in scope.
    files: int
        Number of files in scope.
    choice: "continue" | "quick"
        The user's choice for the plan.
    summary: dict[str, object]
        The run's summary, returned with the answer.
    by_file: bool
        True to list the matching files instead of the sentences.
    exact_terms: Sequence[str]
        The exact terms of a quick exact match; empty otherwise.
    limit: int
        Most sentences, or files, listed.

    Returns
    -------
    ToolResult
        The answer, the passages of the sentences as evidence, and the
        summary.

    """

    citations, evidence = _evidence(found)
    in_files = len({sentence.path for sentence in found})
    if by_file:
        lines = [
            f"**{in_files} of {files} files** match “{topic}”, from "
            f"{len(found)} matching sentences."
        ]
    else:
        noun = "sentence" if len(found) == 1 else "sentences"
        lines = [
            f"Found **{len(found)} {noun}** about “{topic}” in "
            f"**{in_files} of {files} files**."
        ]

    if exact_terms:
        quoted = " or ".join(f"“{term}”" for term in exact_terms)
        lines.append(
            f"Exact match: the sentences that contain {quoted}, found "
            "without the model. Ask again and choose Continue for sentences "
            "about the topic in other words."
        )
    elif choice == PLAN_CHOICE_QUICK:
        lines.append(
            f"Quick sample: {read} sentences from the best-matching "
            f"passages were read, not all {total}."
        )
    elif read < total:
        lines.append(
            f"{read} of {total} sentences were read (the "
            "model-call limit was reached)."
        )

    names = short_names(sorted({sentence.path for sentence in found}, key=str))
    if by_file:
        # One line per file: how many sentences match, and the first one.
        per_file: dict[Path, list[Sentence]] = {}
        for sentence in found:
            per_file.setdefault(sentence.path, []).append(sentence)
        ranked = sorted(per_file.items(), key=lambda item: -len(item[1]))
        for path, matches in ranked[:limit]:
            first = matches[0]
            cite = citations[first.chunk.identifier]
            count = (
                "1 sentence"
                if len(matches) == 1
                else f"{len(matches)} sentences"
            )
            lines.append(
                f"- **{names[path]}** ({count}): “{first.text}” [{cite}]"
            )
        if not found:
            lines.append("No file in scope matches.")
        return ToolResult("\n".join(lines), tuple(evidence), summary)

    listed = found[:limit]
    current: Path | None = None
    for sentence in listed:
        if sentence.path != current:
            current = sentence.path
            lines.append(f"\n**{names[current]}**\n")
        cite = citations[sentence.chunk.identifier]
        lines.append(f"- “{sentence.text}” (line {sentence.line}) [{cite}]")
    if len(found) > len(listed):
        lines.append(
            f"\nListed the first {len(listed)} of {len(found)} sentences."
        )
    if not found:
        lines.append("No sentence in scope is about this topic.")
    return ToolResult("\n".join(lines), tuple(evidence), summary)
