"""Tool ``batch_classify``: label every exact mention, in batches.

``batch_classify`` answers questions such as "which Python mentions in the
Smith paper are the authors' own code, and which are citations?". It
collects candidates exhaustively with ``scan_and_list`` (exact mentions
with surrounding lines), asks the answering model to label them in
token-sized batches, validates every label, and counts in Python. The model
labels; it never counts, so a skipped or invented item cannot change the
totals unnoticed.

Key Features
------------

- **Batches**

    ``plan_batches`` packs mentions into calls by estimated tokens and
    applies the model-call limit; ``quick_sample`` picks one batch spread
    across files for a quick, non-exhaustive run.

- **Validated labels**

    ``parse_labels`` accepts only listed passage numbers and allowed
    categories; anything else is retried once and then reported as not
    classified.

- **Report**

    ``classification_report`` writes counts per category, the listed
    mentions with reasons, and coverage; ``classification_evidence`` gives
    their exact lines.

Module Structure
----------------

- ``ClassifySettings``
- ``parse_settings``
- ``run``
- ``UNCLEAR``
- ``Label``
- ``validated_categories``
- ``item_text``
- ``plan_batches``
- ``quick_sample``
- ``estimate_seconds``
- ``classify_prompt``
- ``parse_labels``
- ``classification_evidence``
- ``classification_report``

Examples
--------
>>> from kilin.tools.batch_classify import ClassifySettings
>>> ClassifySettings().whole_words
True

"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
import re
from time import perf_counter
from typing import Final, Literal, TypeAlias

from kilin.tools import partial_result
from kilin._model_completion import IncompleteModelResponse
from kilin.retrievals.document_reading import estimate_tokens
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
from kilin.tools._shared.mentions import (
    Mention,
    MentionScan,
    collect_mentions,
    mention_location,
    mention_snippet,
)
from kilin.tools._shared.paths import short_names
from kilin.tools._shared.scan_settings import (
    SCAN_SETTING_NAMES,
    scan_fields,
)

__all__ = [
    "CANDIDATES_MENTIONS",
    "CANDIDATE_CHOICES",
    "UNCLEAR",
    "CandidateSource",
    "ClassifySettings",
    "parse_settings",
    "run",
    "Label",
    "classification_evidence",
    "classification_report",
    "classify_prompt",
    "estimate_seconds",
    "item_text",
    "parse_labels",
    "plan_batches",
    "quick_sample",
    "validated_categories",
]

# Always offered, so the model never has to force a passage into a group.
UNCLEAR: Final[str] = "unclear"
# Rough speeds of an 8B 4-bit model on Apple silicon, for estimates only.
_PROMPT_TOKENS_PER_SECOND: Final[float] = _config.PROMPT_TOKENS_PER_SECOND
_REPLY_TOKENS_PER_SECOND: Final[float] = _config.REPLY_TOKENS_PER_SECOND
_REPLY_TOKENS_PER_ITEM: Final[int] = _config.REPLY_TOKENS_PER_ITEM
_PROMPT_OVERHEAD_TOKENS: Final[int] = _config.PROMPT_OVERHEAD_TOKENS
# Room for one label object per item; reasons are asked to stay short.
# Missing labels are retried in groups of at most this many items.
_RETRY_GROUP: Final[int] = _config.RETRY_GROUP


# How the tool collects the items it labels. Only exact mentions of the
# question's terms exist; whole-chunk collection comes with batched
# document review.
CandidateSource: TypeAlias = Literal["mentions"]
CANDIDATES_MENTIONS: Final[CandidateSource] = "mentions"
CANDIDATE_CHOICES: Final[tuple[CandidateSource, ...]] = (CANDIDATES_MENTIONS,)


@dataclass(frozen=True, slots=True)
class ClassifySettings:
    """Settings of ``batch_classify`` in a project type.

    Attributes
    ----------
    case_sensitive: bool
        Whether letter case must match.
    whole_words: bool
        Whether a term must not continue into adjacent letters or digits.
    context_lines: int
        Lines around each mention that the model sees (0-5).
    max_listed: int
        Mentions listed with citations (1-1000); counts include all.
    candidates: CandidateSource
        How items are collected; only ``mentions``.
    categories: tuple[str, ...]
        Fixed categories; empty lets the router model propose them from
        the question (shown in the plan before running).
    batch_tokens: int
        Estimated tokens of mentions per model call (500-8000).
    max_model_calls: int
        Most model calls one answer may make (1-200); mentions beyond them
        are reported as not classified.
    confirmation: ConfirmationPolicy
        When to ask before running.
    large_batches: int
        With ``when_large``, plans with more calls than this ask first
        (0-100).

    """

    case_sensitive: bool = _config.DEFAULT_CASE_SENSITIVE
    whole_words: bool = _config.DEFAULT_WHOLE_WORDS
    context_lines: int = _config.DEFAULT_CONTEXT_LINES
    max_listed: int = _config.DEFAULT_MAX_LISTED
    candidates: CandidateSource = _config.DEFAULT_CANDIDATES
    categories: tuple[str, ...] = _config.DEFAULT_CATEGORIES
    batch_tokens: int = _config.DEFAULT_BATCH_TOKENS
    max_model_calls: int = _config.DEFAULT_MAX_MODEL_CALLS
    confirmation: ConfirmationPolicy = _config.DEFAULT_CONFIRMATION
    large_batches: int = _config.DEFAULT_LARGE_BATCHES


def parse_settings(data: dict, where: str) -> ClassifySettings:
    """Validate ``batch_classify`` settings from a project type.

    Parameters
    ----------
    data: dict
        Settings from the intent's ``tool`` object, without ``name``.
    where: str
        Location for errors.

    Returns
    -------
    ClassifySettings
        Validated settings; absent ones take their defaults.

    Raises
    ------
    ValueError
        If a setting is unknown, out of range, or categories are not 2-8
        distinct labels of at most 40 characters.

    """

    reject_unknown(
        where,
        data,
        (
            *SCAN_SETTING_NAMES,
            "candidates",
            "categories",
            "batch_tokens",
            "max_model_calls",
            "confirmation",
            "large_batches",
        ),
    )
    defaults = ClassifySettings()
    fields = scan_fields(data, where, defaults)
    categories = data.get("categories", list(defaults.categories))
    if (
        not isinstance(categories, list)
        or len(categories) == 1
        or len(categories) > 8
        or not all(
            isinstance(item, str) and 0 < len(item.strip()) <= 40
            for item in categories
        )
        or len({item.strip().casefold() for item in categories})
        < len(categories)
    ):
        raise ValueError(
            f"{where!r} categories must be 2-8 distinct labels of at most "
            "40 characters, or empty."
        )
    return ClassifySettings(
        **fields,
        candidates=check_choice(
            where, data, "candidates", defaults.candidates, CANDIDATE_CHOICES
        ),
        categories=tuple(item.strip() for item in categories),
        batch_tokens=check_int(
            where, data, "batch_tokens", defaults.batch_tokens, 500, 8000
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


def run(
    settings: ClassifySettings, request: ToolRequest, context: ToolContext
) -> ToolResult:
    """Classify every mention the question asks about.

    Parameters
    ----------
    settings: ClassifySettings
        The intent's settings.
    request: ToolRequest
        The question and the user's words.
    context: ToolContext
        KíLín's services.

    Returns
    -------
    ToolResult
        Counts per category, listed mentions with reasons, and coverage.

    Notes
    -----
    Mentions are collected exhaustively first (see
    ``scan_and_list.collect_mentions``). The plan (items, batches, calls,
    approximate time, categories) is traced and, when the confirmation
    policy asks, offered through ``context.confirm``. Batches use the
    answering model with the tool's explicit thinking policy. A batch's
    missing or invalid labels are retried once, in groups of at most eight
    mentions; mentions still without a label are reported as not
    classified. ``max_model_calls`` bounds initial batches, not the existing
    label repairs or the coordinator's one nonthinking recovery attempt.
    Incomplete batches stay unclassified and do not enter label repair;
    independent batches continue with explicit partial status.

    """

    trace = context.trace
    found = collect_mentions(
        settings,
        request,
        context,
        tool="batch_classify",
        categories=not settings.categories,
    )
    if isinstance(found, ToolResult):
        return found
    scan, summary = found.scan, found.summary
    categories = settings.categories or validated_categories(found.reply)
    summary["categories"] = list(categories)
    summary["categories_from"] = (
        "configuration" if settings.categories else "router"
    )
    if not categories:
        context.done("No categories to sort into", evidence_count=0)
        return ToolResult(
            "I couldn't tell which groups to sort the mentions into. Name "
            "them in the question, for example: which mentions of "
            '"Python" are the authors\' own code and which are citations?',
            (),
            summary,
        )

    names = short_names(scan.files_scanned)
    texts = [
        item_text(index + 1, mention, names[mention.path])
        for index, mention in enumerate(scan.mentions)
    ]
    batches = plan_batches(texts, settings.batch_tokens)
    to_run = batches[: settings.max_model_calls]
    quick = quick_sample(scan.mentions, texts, settings.batch_tokens)
    listed = " or ".join(f"“{term}”" for term in scan.terms)
    details = [
        "Categories: " + ", ".join([*categories, UNCLEAR]),
        f"Model: {context.model_id}",
    ]
    if len(to_run) < len(batches):
        left = len(scan.mentions) - sum(len(batch) for batch in to_run)
        details.append(
            f"The model-call limit ({settings.max_model_calls}) leaves "
            f"{left} mentions unclassified."
        )
    plan = ToolPlan(
        tool="batch_classify",
        summary=(
            f"Classify {len(scan.mentions)} mentions of {listed} into "
            f"{len(categories)} categories with the answering model."
        ),
        scope=found.scope_note
        or f"All {len(scan.files_scanned)} indexed files.",
        items=len(scan.mentions),
        batches=len(batches),
        batches_to_run=len(to_run),
        estimated_seconds=round(estimate_seconds(to_run, texts), 1),
        quick_items=len(quick),
        quick_seconds=round(estimate_seconds([quick], texts), 1),
        details=tuple(details),
    )
    choice, asked = confirm_plan(
        plan, settings.confirmation, settings.large_batches, context
    )
    summary.update({"plan": plan.to_dict(), "choice": choice, "asked": asked})
    if choice == PLAN_CHOICE_CANCEL:
        context.done("Cancelled before classifying", evidence_count=0)
        return ToolResult(
            "Cancelled before classifying; no model calls were made. The "
            f"plan was to classify {plan.items} mentions of {listed} in "
            f"{plan.batches_to_run} model calls (about "
            f"{_duration(plan.estimated_seconds)}).",
            (),
            summary,
        )

    runs = [quick] if choice == PLAN_CHOICE_QUICK else to_run
    labels: dict[int, Label] = {}
    failed: list[int] = []
    failures: list[dict[str, object]] = []
    calls = 0
    started = perf_counter()
    for number, batch in enumerate(runs, start=1):
        message = f"Classifying batch {number} of {len(runs)}"
        if number > 1:
            pace = (perf_counter() - started) / (number - 1)
            message += (
                f" (about {_duration(pace * (len(runs) - number + 1))} "
                "left)"
            )
        context.progress(message, current=number, total=len(runs))
        batch_started = perf_counter()
        calls += 1
        try:
            found_labels = _label_batch(
                request.question, scan.terms, categories, texts, batch, context
            )
        except IncompleteModelResponse as error:
            failures.append({
                "item": f"Classification batch {number}",
                "items": [index + 1 for index in batch], **error.to_dict(),
            })
            failed.extend(batch)
            continue
        missing = [index for index in batch if index not in found_labels]
        # Existing schema-repair retries apply only to completed model replies.
        for start in range(0, len(missing), _RETRY_GROUP):
            group = missing[start : start + _RETRY_GROUP]
            calls += 1
            try:
                found_labels.update(_label_batch(
                    request.question, scan.terms, categories, texts,
                    group, context,
                ))
            except IncompleteModelResponse as error:
                failures.append({
                    "item": f"Classification batch {number}, missing labels",
                    "items": [index + 1 for index in group], **error.to_dict(),
                })
        labels.update(found_labels)
        failed += [index for index in batch if index not in found_labels]
        trace.emit(
            "tool.batch.result",
            component="tool",
            data={
                "batch": number,
                "items": len(batch),
                "labelled": len(found_labels),
                "retried": len(missing),
                "seconds": round(perf_counter() - batch_started, 2),
            },
            turn_id=context.trace_turn_id,
        )
    classified = sorted(index for batch in runs for index in batch)
    evidence = classification_evidence(
        scan, labels, categories, settings.max_listed
    )
    answer = classification_report(
        scan,
        labels,
        categories,
        limit=settings.max_listed,
        classified=classified,
        not_classified=failed,
        quick=choice == PLAN_CHOICE_QUICK,
        scope_note=found.scope_note,
    )
    counts = {category: 0 for category in [*categories, UNCLEAR]}
    for label in labels.values():
        counts[label.category] += 1
    summary.update(
        {
            "classified": len(classified),
            "labelled": len(labels),
            "not_classified": len(failed),
            "not_sent": len(scan.mentions) - len(classified),
            "counts": counts,
            "model_calls": calls,
            "model": context.model_id,
            "seconds": round(perf_counter() - started, 1),
            "listed": len(evidence),
        }
    )
    trace.emit(
        "tool.batch_classify.result",
        component="tool",
        data={
            key: value
            for key, value in summary.items()
            if key not in ("name", "plan")
        },
        turn_id=context.trace_turn_id,
    )
    context.done(
        f"Classified {len(labels)} of {len(scan.mentions)} mentions",
        evidence_count=len(evidence),
    )
    return partial_result(
        ToolResult(answer, evidence, summary), failures,
        completed=bool(labels),
    )


def _label_batch(
    question: str,
    terms: Sequence[str],
    categories: Sequence[str],
    texts: Sequence[str],
    indices: Sequence[int],
    context: ToolContext,
) -> dict[int, Label]:
    """Label one batch of mentions; returns labels by mention index.

    Parameters
    ----------
    question: str
        Question or request to analyze.
    terms: Sequence[str]
        Search terms used to prepare the classification question.
    categories: Sequence[str]
        Available classification categories.
    texts: Sequence[str]
        Text passages or fields to inspect.
    indices: Sequence[int]
        Indexes of mentions included in this classification batch.
    context: ToolContext
        Tool context for the current request.

    Returns
    -------
    dict[int, Label]
        Assigned labels keyed by each mention index.

    """

    system, prompt = classify_prompt(
        question, terms, categories, [texts[index] for index in indices]
    )
    try:
        reply = context.ask_model(
            system, prompt, None
        )
    except IncompleteModelResponse:
        raise
    except Exception as error:  # A failed batch is reported, not fatal.
        reply = f"failed: {error}"
    if context.trace.include_content:
        context.trace.emit(
            "tool.batch.reply",
            component="tool",
            data={"items": [index + 1 for index in indices], "raw": reply},
            turn_id=context.trace_turn_id,
        )
    labelled = parse_labels(
        reply, [index + 1 for index in indices], categories
    )
    return {number - 1: label for number, label in labelled.items()}


def _duration(seconds: float) -> str:
    """Write an approximate duration such as "40 s" or "3 min".

    Parameters
    ----------
    seconds: float
        Estimated duration in seconds.

    Returns
    -------
    str
        Approximate duration expressed in seconds or minutes.

    """

    if seconds < 90:
        return f"{max(1, round(seconds))} s"
    return f"{round(seconds / 60)} min"


@dataclass(frozen=True, slots=True)
class Label:
    """The category the model gave one mention.

    Attributes
    ----------
    category: str
        One of the allowed categories (or ``UNCLEAR``).
    reason: str
        The model's short reason.

    """

    category: str
    reason: str


def validated_categories(reply: str) -> tuple[str, ...]:
    """Return the categories a model proposed, when they are usable.

    Parameters
    ----------
    reply: str
        Model reply containing JSON with ``"categories"``.

    Returns
    -------
    tuple[str, ...]
        2-6 distinct labels of at most 40 characters, without ``unclear``;
        empty when the reply has no usable categories.

    """

    match = re.search(r"\{.*\}", reply, flags=re.S)
    try:
        value = json.loads(match.group(0)) if match else None
    except ValueError:
        value = None
    items = value.get("categories") if isinstance(value, dict) else None
    if not isinstance(items, list):
        return ()
    found: list[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        label = " ".join(item.split())
        if (
            0 < len(label) <= 40
            and label.casefold() != UNCLEAR
            and label.casefold() not in {f.casefold() for f in found}
        ):
            found.append(label)
    return tuple(found) if 2 <= len(found) <= 6 else ()


def item_text(number: int, mention: Mention, name: str) -> str:
    """Show one mention to the model: number, location, marked context.

    Parameters
    ----------
    number: int
        The passage number the model must answer with.
    mention: Mention
        The mention.
    name: str
        How to name its file.

    Returns
    -------
    str
        For example ``"[3] paper.pdf, page 2, line 7"``, then the context
        with the mention marked ``«Python»``.

    """

    end = mention.offset + len(mention.text)
    context = (
        f"{mention.context[: mention.offset]}«{mention.text}»"
        f"{mention.context[end:]}"
    )
    return f"[{number}] {mention_location(mention, name)}\n{context.strip()}"


def plan_batches(
    texts: Sequence[str], batch_tokens: int
) -> list[list[int]]:
    """Pack items into model calls by estimated tokens, in order.

    Parameters
    ----------
    texts: Sequence[str]
        Item texts as the model will see them.
    batch_tokens: int
        Estimated tokens of items per call; an item larger than this gets
        a call of its own.

    Returns
    -------
    list[list[int]]
        Item indices per call.

    """

    batches: list[list[int]] = []
    used = 0
    for index, text in enumerate(texts):
        size = estimate_tokens(text)
        if batches and used + size <= batch_tokens:
            batches[-1].append(index)
            used += size
        else:
            batches.append([index])
            used = size
    return batches


def quick_sample(
    mentions: Sequence[Mention], texts: Sequence[str], batch_tokens: int
) -> list[int]:
    """Pick one batch of items spread across files, for a quick run.

    Parameters
    ----------
    mentions: Sequence[Mention]
        All mentions, in index order.
    texts: Sequence[str]
        Their item texts.
    batch_tokens: int
        Estimated tokens the sample may use.

    Returns
    -------
    list[int]
        Item indices in index order: files take turns, so the sample is not
        only the first file's mentions.

    """

    by_file: dict[object, list[int]] = {}
    for index, mention in enumerate(mentions):
        by_file.setdefault(mention.path, []).append(index)
    queues = list(by_file.values())
    chosen: list[int] = []
    used = 0
    while any(queues):
        for queue in queues:
            if not queue:
                continue
            index = queue[0]
            size = estimate_tokens(texts[index])
            if chosen and used + size > batch_tokens:
                return sorted(chosen)
            chosen.append(queue.pop(0))
            used += size
    return sorted(chosen)


def estimate_seconds(
    batches: Sequence[Sequence[int]], texts: Sequence[str]
) -> float:
    """Approximate how long the calls take on an 8B model.

    Parameters
    ----------
    batches: Sequence[Sequence[int]]
        Item indices per call.
    texts: Sequence[str]
        Item texts.

    Returns
    -------
    float
        Seconds: reading each prompt plus writing a label per item. A
        larger model is slower; the plan says "about".

    """

    total = 0.0
    for batch in batches:
        prompt = _PROMPT_OVERHEAD_TOKENS + sum(
            estimate_tokens(texts[index]) for index in batch
        )
        reply = _REPLY_TOKENS_PER_ITEM * len(batch)
        total += 1.0 + prompt / _PROMPT_TOKENS_PER_SECOND
        total += reply / _REPLY_TOKENS_PER_SECOND
    return total


def classify_prompt(
    question: str,
    terms: Sequence[str],
    categories: Sequence[str],
    items: Sequence[str],
) -> tuple[str, str]:
    """Return the system and user prompts for one batch.

    Parameters
    ----------
    question: str
        The user's question.
    terms: Sequence[str]
        Searched terms.
    categories: Sequence[str]
        Allowed categories, without ``UNCLEAR``.
    items: Sequence[str]
        Item texts (see ``item_text``).

    Returns
    -------
    tuple[str, str]
        System prompt and user prompt; the reply is checked with
        ``parse_labels``.

    """

    system = (
        "You classify passages for KíLín. You never answer the question "
        "itself. You reply with JSON only."
    )
    quoted = " or ".join(f"“{term}”" for term in terms)
    labels = "\n".join(f"- {category}" for category in categories)
    passages = "\n\n".join(items)
    user = f"""Question: {question}

Each passage contains a mention of {quoted}, marked «like this». Give each
passage exactly one category for that mention:
{labels}
- {UNCLEAR} (the passage does not show which category applies)

Passages:

{passages}

Reply with a JSON array with one object per passage, in order:
[{{"id": 1, "category": "...", "reason": "at most ten words"}}]"""
    return system, user


def parse_labels(
    reply: str, numbers: Sequence[int], categories: Sequence[str]
) -> dict[int, Label]:
    """Accept only labels for listed passages with allowed categories.

    Parameters
    ----------
    reply: str
        Model reply.
    numbers: Sequence[int]
        Passage numbers in the batch.
    categories: Sequence[str]
        Allowed categories, without ``UNCLEAR``.

    Returns
    -------
    dict[int, Label]
        Valid labels by passage number. Unknown numbers, other categories,
        and repeated numbers (after the first) are ignored. When the reply
        is not one JSON array (for example cut off by the length limit),
        every complete object in it is used.

    """

    allowed = {category.casefold(): category for category in categories}
    allowed[UNCLEAR] = UNCLEAR
    start, end = reply.find("["), reply.rfind("]")
    try:
        value = json.loads(reply[start : end + 1]) if start >= 0 else None
    except ValueError:
        value = None
    if not isinstance(value, list):
        # A reply cut off by the length limit still has complete objects.
        value = []
        for match in re.finditer(r"\{[^{}]*\}", reply):
            try:
                value.append(json.loads(match.group(0)))
            except ValueError:
                continue
    wanted = set(numbers)
    labels: dict[int, Label] = {}
    for item in value:
        if not isinstance(item, dict):
            continue
        number, category = item.get("id"), item.get("category")
        if type(number) is not int or number not in wanted:
            continue
        if not isinstance(category, str) or number in labels:
            continue
        category = allowed.get(" ".join(category.split()).casefold())
        if category is None:
            continue
        reason = item.get("reason")
        if not isinstance(reason, str):
            reason = ""
        reason = " ".join(reason.split())[:120]
        labels[number] = Label(category, reason)
    return labels


def classification_evidence(
    scan: MentionScan,
    labels: dict[int, Label],
    categories: Sequence[str],
    limit: int,
) -> tuple[dict[str, object], ...]:
    """Return evidence fields for the listed mentions, in report order.

    Parameters
    ----------
    scan: MentionScan
        Scan result.
    labels: dict[int, Label]
        Labels by mention index (zero-based).
    categories: Sequence[str]
        Categories in display order, without ``UNCLEAR``.
    limit: int
        How many mentions to list.

    Returns
    -------
    tuple[dict[str, object], ...]
        Evidence fields numbered from one, grouped by category.

    """

    return tuple(
        {
            "citation": number,
            "path": mention.path,
            "start_line": mention.context_start,
            "end_line": mention.context_end,
            "symbol": mention.symbol,
            "text": mention.context,
            "cell": mention.cell,
        }
        for number, (_, mention) in enumerate(
            _listed(scan, labels, categories, limit), start=1
        )
    )


def classification_report(
    scan: MentionScan,
    labels: dict[int, Label],
    categories: Sequence[str],
    *,
    limit: int,
    classified: Sequence[int],
    not_classified: Sequence[int],
    quick: bool = False,
    scope_note: str | None = None,
) -> str:
    """Write the answer: counts per category, listed mentions, coverage.

    Parameters
    ----------
    scan: MentionScan
        Scan result (every candidate).
    labels: dict[int, Label]
        Valid labels by mention index (zero-based).
    categories: Sequence[str]
        Categories in display order, without ``UNCLEAR``.
    limit: int
        How many mentions are listed with citations.
    classified: Sequence[int]
        Mention indices sent to the model.
    not_classified: Sequence[int]
        Sent indices without a valid label after the retry.
    quick: bool, optional
        True for a quick, non-exhaustive sample. Defaults to ``False``.
    scope_note: str | None, optional
        How the scanned files were chosen. Defaults to ``None``.

    Returns
    -------
    str
        Markdown. Listed mention ``n`` cites evidence ``[n]``.

    """

    quoted = " or ".join(f"“{term}”" for term in scan.terms)
    total = len(scan.mentions)
    names = short_names(scan.files_scanned)
    labelled = len(labels)
    where = (
        names[scan.files_scanned[0]]
        if len(scan.files_scanned) == 1
        else f"{len(scan.files_scanned)} files"
    )
    lines = [
        f"Classified **{labelled} of {total} mentions** of {quoted} in "
        f"{where}."
    ]
    if quick:
        lines.append(
            f"Quick sample: {len(classified)} mentions spread across the "
            "files; the counts describe the sample only."
        )
    if scope_note:
        lines += ["", scope_note]
    order = [*categories, UNCLEAR]
    counts = {category: 0 for category in order}
    for label in labels.values():
        counts[label.category] += 1
    lines += ["", "**By category**", ""]
    lines += [f"- {category}: {counts[category]}" for category in order]
    if not_classified:
        lines.append(
            f"- not classified: {len(not_classified)} (no valid label from "
            "the model, even after a retry)"
        )
    listed = _listed(scan, labels, categories, limit)
    number = 0
    for category in order:
        group = [
            item for item in listed if labels[item[0]].category == category
        ]
        if not group:
            continue
        lines += ["", f"**{category}**", ""]
        for index, mention in group:
            number += 1
            reason = labels[index].reason
            lines.append(
                f"{number}. {mention_location(mention, names[mention.path])}"
                f": {mention_snippet(mention, 40)}"
                + (f" — {_plain(reason)}" if reason else "")
                + f" [{number}]"
            )
    if len(listed) < labelled:
        lines += [
            "",
            f"Listed {len(listed)} of {labelled} classified mentions; the "
            "counts include all of them.",
        ]
    skipped = total - len(classified)
    coverage = [f"Scanned all indexed text in {where}."]
    if skipped and not quick:
        coverage.append(
            f"{skipped} mentions were not sent to the model because the "
            "model-call limit was reached."
        )
    lines += ["", " ".join(coverage)]
    return "\n".join(lines)


def _listed(
    scan: MentionScan,
    labels: dict[int, Label],
    categories: Sequence[str],
    limit: int,
) -> list[tuple[int, Mention]]:
    """Labelled mentions grouped by category in display order, capped.

    Parameters
    ----------
    scan: MentionScan
        Scan result containing candidate items.
    labels: dict[int, Label]
        Assigned mention labels indexed by mention position.
    categories: Sequence[str]
        Available classification categories.
    limit: int
        Maximum number of items or characters to include.

    Returns
    -------
    list[tuple[int, Mention]]
        Sorted, capped list of mention indexes and their records.

    """

    order = {category: rank for rank, category in enumerate(categories)}
    order[UNCLEAR] = len(categories)
    items = sorted(
        labels, key=lambda index: (order[labels[index].category], index)
    )
    return [(index, scan.mentions[index]) for index in items[:limit]]


def _plain(text: str) -> str:
    """Replace square brackets with parentheses for plain-text output.

    Parameters
    ----------
    text: str
        Plain-text output that may contain bracketed labels.

    Returns
    -------
    str
        Text with square brackets replaced by parentheses.

    """

    return text.replace("[", "(").replace("]", ")")
