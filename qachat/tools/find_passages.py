"""Tool ``find_passages``: every sentence (or file) that matches a request.

"List the sentences that discuss network analysis" asks for meaning, not
exact words: a sentence about "graph-theoretical measures of brain
connectivity" belongs, and one that only names "network" in passing does
not. Ranked retrieval would show the best few; this tool reads **every**
sentence in scope, in batches, and the model only answers which numbered
sentences are about the topic. Python copies those sentences word for word
with their file and line, so nothing is paraphrased or invented.

1. Find the topic and any one document in the question (one router call;
   a document named in the question narrows the scope).
2. Split the passages in scope into sentences, without the duplicates
   overlapping passages share, and number them.
3. Plan batches by estimated tokens and confirm a large run (with a quick
   sample of the best-ranked passages).
4. Ask the model, batch by batch, for the numbers of the sentences about
   the topic; numbers that are not in the batch are ignored.
5. List the sentences by file, with citations; when the question asks for
   files ("which papers use BRAPH 2 as an analysis tool?"), list the files
   that match, each with its number of matching sentences and an example.

Module Structure
----------------

- ``PassageSettings``
- ``parse_settings``
- ``run``
- ``split_sentences``
- ``topic_prompt``
- ``judge_prompt``

"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Final

from qachat.chunking import Chunk
from qachat.retrievals.document_reading import estimate_tokens, named_paths
from qachat.tools.choose_document import document_scope
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
from qachat.tools.scan_and_list import short_names

__all__ = [
    "PassageSettings",
    "Sentence",
    "judge_prompt",
    "parse_settings",
    "run",
    "split_sentences",
    "topic_prompt",
]

# Sentence ends: ., !, or ? followed by space and a capital, digit, quote,
# or bracket; or a blank line.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n\s*\n")
_MIN_WORDS: Final[int] = 4
_PROMPT_TOKENS_PER_SECOND: Final[float] = 700.0
_REPLY_TOKENS_PER_SECOND: Final[float] = 25.0
# A request for files rather than sentences ("which papers use ...").
_FILES = re.compile(
    r"\b(files?|papers?|documents?|articles?|studies|sources)\b", re.I
)
_SENTENCES = re.compile(
    r"\b(sentences?|passages?|paragraphs?|quotes?)\b", re.I
)
# Words that ask for passages rather than name the topic.
_REQUEST_WORDS = re.compile(
    r"^.*?\b(?:sentences?|passages?|parts?|paragraphs?|places?)\b"
    r"(?:\s+(?:that|which|where))?(?:\s+(?:are|is))?"
    r"(?:\s+(?:talking|talk|talks|discussing|discuss|discusses|about|on|"
    r"mentioning|mention|describing|describe))*\s*",
    flags=re.I,
)


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
    confirmation: ConfirmationPolicy
        When to ask before running.
    large_batches: int
        With ``when_large``, plans with more batches than this ask first.

    """

    batch_tokens: int = 2500
    max_model_calls: int = 200
    quick_passages: int = 40
    max_listed: int = 100
    confirmation: ConfirmationPolicy = CONFIRMATION_WHEN_LARGE
    large_batches: int = 3


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


@dataclass(frozen=True, slots=True)
class Sentence:
    """One sentence of a passage.

    Attributes
    ----------
    path: pathlib.Path
        The file.
    line: int
        Its first line in the file.
    text: str
        The sentence, with whitespace joined.
    chunk: Chunk
        The passage it comes from, for citations.

    """

    path: Path
    line: int
    text: str
    chunk: Chunk


def split_sentences(chunks: Sequence[Chunk]) -> list[Sentence]:
    """Split passages into sentences, without repeats.

    Parameters
    ----------
    chunks: Sequence[Chunk]
        Passages in file and line order.

    Returns
    -------
    list[Sentence]
        Sentences of at least four words. A sentence that overlapping
        passages share is kept once, from its first passage.

    """

    seen: set[tuple[Path, str]] = set()
    sentences: list[Sentence] = []
    for chunk in chunks:
        start = 0
        pieces = _SENTENCE_END.split(chunk.text)
        for piece in pieces:
            offset = chunk.text.find(piece, start)
            start = max(start, offset + len(piece))
            text = " ".join(piece.split())
            if len(text.split()) < _MIN_WORDS:
                continue
            key = (chunk.path, text.casefold())
            if key in seen:
                continue
            seen.add(key)
            line = chunk.start_line + chunk.text[: max(0, offset)].count("\n")
            sentences.append(Sentence(chunk.path, line, text, chunk))
    return sentences


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
        ``topic`` and ``document``.

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
exactly from the message (for example "the Smith paper"), or null.

Example: "list the sentences in the Smith paper that discuss limitations"
-> {{"topic": "limitations", "document": "the Smith paper"}}"""
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
        example "uses BRAPH 2 as an analysis tool", not just "BRAPH 2").
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
        QAChat's services.

    Returns
    -------
    ToolResult
        The sentences by file with citations (the first ``max_listed``),
        the passages they come from as evidence, and a summary.

    """

    summary: dict[str, object] = {"name": "find_passages"}
    context.progress("Finding the topic")
    topic, document = _topic(request.question, context)
    summary.update({"topic": topic, "document": document})
    if not topic:
        context.done("No topic found", evidence_count=0)
        return ToolResult(
            "I couldn't tell which topic to look for. Try, for example: "
            "list the sentences that discuss network analysis.",
            (),
            summary,
        )
    indexed = list(dict.fromkeys(chunk.path for chunk in context.chunks))
    scope, _ = document_scope(request, context, indexed)
    if not scope and document:
        scope = named_paths(document, indexed, authors=True)
    chunks = sorted(
        (c for c in context.chunks if not scope or c.path in scope),
        key=lambda c: (str(c.path), c.start_line),
    )
    sentences = split_sentences(chunks)
    batches = _batches(sentences, settings.batch_tokens)
    to_run = batches[: settings.max_model_calls]
    ranked = context.rank(topic) if context.chunks else []
    quick_chunks = [
        result.chunk
        for result in ranked[: settings.quick_passages]
        if not scope or result.chunk.path in scope
    ]
    quick = _batches(split_sentences(quick_chunks), settings.batch_tokens)
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
        quick_items=sum(len(batch) for batch in quick),
        quick_seconds=round(_estimate(quick), 1),
        details=(f"Model: {context.model_id}",),
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
    runs = quick if choice == PLAN_CHOICE_QUICK else to_run
    found: list[Sentence] = []
    for number, batch in enumerate(runs, start=1):
        context.progress(
            f"Reading sentences (batch {number} of {len(runs)})",
            current=number,
            total=len(runs),
        )
        system, prompt = judge_prompt(topic, batch, request.question)
        try:
            reply = context.ask_model(system, prompt, 120)
        except Exception as error:  # One failed batch does not stop the run.
            context.trace.emit(
                "tool.find_passages.batch_failed",
                component="tool",
                data={"batch": number, "error": str(error)},
                turn_id=context.trace_turn_id,
            )
            continue
        picked = sorted(
            {
                int(value) - 1
                for value in re.findall(r"\d+", reply)
                if 0 < int(value) <= len(batch)
            }
        )
        found.extend(batch[index] for index in picked)
    read = sum(len(batch) for batch in runs)
    summary.update(
        {"sentences": len(sentences), "read": read, "found": len(found)}
    )
    context.done(
        f"Found {len(found)} sentences", evidence_count=len(found)
    )
    summary["by_file"] = bool(
        _FILES.search(request.question)
        and not _SENTENCES.search(request.question)
    )
    return _result(topic, found, read, sentences, files, choice, settings,
                   summary)


def _topic(question: str, context: ToolContext) -> tuple[str, str | None]:
    """The topic and document phrase, from the router or the wording."""

    system, prompt = topic_prompt(question)
    topic, document = "", None
    try:
        reply = context.ask_router(system, prompt, 80)
        match = re.search(r"\{.*\}", reply, flags=re.S)
        data = json.loads(match.group(0)) if match else {}
        topic = str(data.get("topic") or "").strip()
        value = data.get("document")
        if isinstance(value, str) and value.casefold() in question.casefold():
            document = value
    except Exception:  # Fall back to the wording below.
        topic = ""
    words = set(re.findall(r"\w+", question.casefold()))
    if not topic or not set(re.findall(r"\w+", topic.casefold())) & words:
        topic = _REQUEST_WORDS.sub("", question).strip(" ?.!")
    return topic, document


def _batches(sentences: list[Sentence], tokens: int) -> list[list[Sentence]]:
    batches: list[list[Sentence]] = []
    used = 0
    for sentence in sentences:
        size = estimate_tokens(sentence.text) + 2
        if batches and used + size <= tokens:
            batches[-1].append(sentence)
            used += size
        else:
            batches.append([sentence])
            used = size
    return batches


def _estimate(batches: list[list[Sentence]]) -> float:
    total = 0.0
    for batch in batches:
        prompt = 150 + sum(estimate_tokens(s.text) + 2 for s in batch)
        total += 1.0 + prompt / _PROMPT_TOKENS_PER_SECOND
        total += 15 / _REPLY_TOKENS_PER_SECOND
    return total


def _result(
    topic: str,
    found: list[Sentence],
    read: int,
    sentences: list[Sentence],
    files: int,
    choice: str,
    settings: PassageSettings,
    summary: dict[str, object],
) -> ToolResult:
    """Write the answer: counts, then the sentences grouped by file."""

    citations: dict[str, int] = {}
    evidence: list[dict[str, object]] = []
    for sentence in found:
        chunk = sentence.chunk
        if chunk.identifier not in citations:
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
    in_files = len({s.path for s in found})
    if summary.get("by_file"):
        lines = [
            f"**{in_files} of {files} files** match “{topic}”, from "
            f"{len(found)} matching sentences."
        ]
    else:
        lines = [
            f"Found **{len(found)} sentences** about “{topic}” in "
            f"**{in_files} of {files} files**."
        ]
    if choice == PLAN_CHOICE_QUICK:
        lines.append(
            f"Quick sample: {read} sentences from the best-matching "
            f"passages were read, not all {len(sentences)}."
        )
    elif read < len(sentences):
        lines.append(
            f"{read} of {len(sentences)} sentences were read (the "
            "model-call limit was reached)."
        )
    names = short_names(sorted({s.path for s in found}, key=str))
    if summary.get("by_file"):
        # One line per file: how many sentences match, and the first one.
        per_file: dict[Path, list[Sentence]] = {}
        for sentence in found:
            per_file.setdefault(sentence.path, []).append(sentence)
        ranked = sorted(per_file.items(), key=lambda item: -len(item[1]))
        for path, matches in ranked[: settings.max_listed]:
            first = matches[0]
            cite = citations[first.chunk.identifier]
            count = "1 sentence" if len(matches) == 1 else (
                f"{len(matches)} sentences"
            )
            lines.append(
                f"- **{names[path]}** ({count}): “{first.text}” [{cite}]"
            )
        if not found:
            lines.append("No file in scope matches.")
        return ToolResult("\n".join(lines), tuple(evidence), summary)
    listed = found[: settings.max_listed]
    current: Path | None = None
    for sentence in listed:
        if sentence.path != current:
            current = sentence.path
            lines.append(f"\n**{names[current]}**\n")
        cite = citations[sentence.chunk.identifier]
        lines.append(
            f"- “{sentence.text}” (line {sentence.line}) [{cite}]"
        )
    if len(found) > len(listed):
        lines.append(
            f"\nListed the first {len(listed)} of {len(found)} sentences."
        )
    if not found:
        lines.append("No sentence in scope is about this topic.")
    return ToolResult("\n".join(lines), tuple(evidence), summary)
