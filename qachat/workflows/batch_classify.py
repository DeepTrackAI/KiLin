"""Classify every exact mention with the model, in batches.

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

"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
import re
from typing import Final

from qachat.retrievals.document_reading import estimate_tokens
from qachat.workflows.scan_and_list import (
    Mention,
    MentionScan,
    mention_location,
    mention_snippet,
    short_names,
)

__all__ = [
    "UNCLEAR",
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
_PROMPT_TOKENS_PER_SECOND: Final[float] = 700.0
_REPLY_TOKENS_PER_SECOND: Final[float] = 25.0
_REPLY_TOKENS_PER_ITEM: Final[int] = 25
_PROMPT_OVERHEAD_TOKENS: Final[int] = 350


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
        "You classify passages for QAChat. You never answer the question "
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
        and repeated numbers (after the first) are ignored.

    """

    allowed = {category.casefold(): category for category in categories}
    allowed[UNCLEAR] = UNCLEAR
    start, end = reply.find("["), reply.rfind("]")
    try:
        value = json.loads(reply[start : end + 1]) if start >= 0 else None
    except ValueError:
        value = None
    wanted = set(numbers)
    labels: dict[int, Label] = {}
    for item in value if isinstance(value, list) else []:
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
    """Labelled mentions grouped by category in display order, capped."""

    order = {category: rank for rank, category in enumerate(categories)}
    order[UNCLEAR] = len(categories)
    items = sorted(
        labels, key=lambda index: (order[labels[index].category], index)
    )
    return [(index, scan.mentions[index]) for index in items[:limit]]


def _plain(text: str) -> str:
    return text.replace("[", "(").replace("]", ")")
