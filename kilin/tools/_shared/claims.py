"""Judge whether a cited work supports the sentence that cites it.

Shared by the tools that weigh a sentence against a paper. For a sentence
and one paper in the project (a *claim*), the paper's passages most
related to the sentence are given to the answering model, which returns a
verdict, the passage sentence it rests on, and a short critical comment:

- **supported**: the passages state what the sentence attributes to the
  reference;
- **partly supported**: they state part of it, or something weaker,
  narrower, or under other conditions;
- **not supported**: they discuss the topic but do not state it;
- **contradicted**: they state the opposite or a different result;
- **background**: the sentence only names the reference as the source of a
  method, tool, dataset or topic, and the passages are about that;
- **not found**, shown as "no matching passage": the passages are not
  about the sentence (the relevant part of the paper may not have been
  retrieved).

The model can be wrong, so its answer is checked in Python: the quote must
be found in the passages it was given (its words in order; citation
numbers and whitespace aside, with key scientific values and
qualifiers preserved), and is then shown in the
passage's own words, or it is dropped and the comment says what the model
quoted; and a reference is called **contradicted** only with a confirmed
quote, otherwise **not supported**.
An answer that names no verdict is **unclear**; a failed model call is
**failed**; a file without searchable text is **no text**, with no model
call. Only a sample of each cited paper is read, so "no matching passage"
means none of the passages read is about the sentence, not that the paper
has none.

Key Features
------------

- **One claim, one cited work**

    A sentence citing several works is judged once per work, and the
    model is told the work only needs to support its own part.

- **Publication attribution**

    When a citation identifies the publication reporting the preceding
    study, the study description is compared with the cited work. A paper
    need not contain a self-referential sentence saying it was published.

- **Checked quotes**

    Quotes are confirmed against the passages and shown in the passage's
    own words, preserving critical numbers and effect direction; rejected
    or absent quotations are marked separately from the model verdict.

Module Structure
----------------

- ``VERDICTS``, the ``VERDICT_*`` values and ``verdict_label``

    The possible verdicts, and the words shown for them.

- ``Claim`` and ``Verdict``

    A citing sentence with one cited file, and its judgement.

- ``claim_passages``

    The cited file's passages most related to a claim.

- ``verification_prompt`` and ``parse_verdict``

    The model's prompts, and its checked answer.

- ``verify_claim``

    Judge one claim with the answering model.

- ``Support``, ``support_prompt`` and ``judge_support``

    The same judgement for a statement and a work it does not cite yet,
    for suggesting references.

Examples
--------
>>> from kilin.tools._shared.claims import parse_verdict
>>> parse_verdict("VERDICT: not found\\nQUOTE: NONE\\nCOMMENT: Off topic.", ())
('not found', '', None, 'Off topic.')

"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
import re
from typing import Final

from kilin._model_completion import IncompleteModelResponse
from kilin.chunking import Chunk
from kilin.retrievals.document_reading import estimate_tokens
from kilin.tools._shared.sentences import Sentence

__all__ = [
    "VERDICTS",
    "VERDICT_BACKGROUND",
    "VERDICT_CONTRADICTED",
    "VERDICT_FAILED",
    "VERDICT_NOT_FOUND",
    "VERDICT_NO_TEXT",
    "VERDICT_NOT_SUPPORTED",
    "VERDICT_PARTLY",
    "VERDICT_SUPPORTED",
    "VERDICT_UNCLEAR",
    "Claim",
    "Support",
    "Verdict",
    "claim_passages",
    "judge_support",
    "quote_line_range",
    "parse_verdict",
    "support_prompt",
    "verdict_label",
    "verification_prompt",
    "verify_claim",
]

VERDICT_SUPPORTED: Final[str] = "supported"
VERDICT_PARTLY: Final[str] = "partly supported"
VERDICT_NOT_SUPPORTED: Final[str] = "not supported"
VERDICT_CONTRADICTED: Final[str] = "contradicted"
VERDICT_BACKGROUND: Final[str] = "background"
VERDICT_NOT_FOUND: Final[str] = "not found"
VERDICT_UNCLEAR: Final[str] = "unclear"
VERDICT_FAILED: Final[str] = "failed"
VERDICT_NO_TEXT: Final[str] = "no text"
#: Every verdict, from the model's six to the three Python assigns.
VERDICTS: Final[tuple[str, ...]] = (
    VERDICT_SUPPORTED,
    VERDICT_PARTLY,
    VERDICT_NOT_SUPPORTED,
    VERDICT_CONTRADICTED,
    VERDICT_BACKGROUND,
    VERDICT_NOT_FOUND,
    VERDICT_UNCLEAR,
    VERDICT_FAILED,
    VERDICT_NO_TEXT,
)
# What people read for a verdict, where it differs from its name: "not
# found" only means that the passages read are not about the sentence.
_LABELS: Final[dict[str, str]] = {
    VERDICT_NOT_FOUND: "no matching passage",
}
# The verdicts the model may give, longest first so "not supported" is not
# read as "supported".
_MODEL_VERDICTS: Final[tuple[str, ...]] = tuple(
    sorted(VERDICTS[:6], key=len, reverse=True)
)
_LINE: Final[re.Pattern[str]] = re.compile(
    r"^\s*\**(VERDICT|QUOTE|COMMENT)\**\s*:\s*(.*)$", flags=re.I | re.M
)
# What the reference is cited for names a source ("a previously
# established tracking method"): the prompt says so.
_SOURCE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:method|tool|software|package|dataset|data|algorithm|code|model|"
    r"approach|challenge|library|framework|pipeline|protocol|toolbox|"
    r"database|technique|platform|implementation|benchmark)s?\b",
    flags=re.I,
)
# Reciprocal-rank fusion constant: a passage at rank r of one query scores
# 1 / (_FUSION + r).
_FUSION: Final[int] = 60
# A word hyphenated across lines in extracted text ("perfor- mance").
_HYPHENATED: Final[re.Pattern[str]] = re.compile(r"(?<=[a-z])-\s+(?=[a-z])")
# A word, for comparing quotes: letters only.
_WORD: Final[re.Pattern[str]] = re.compile(r"[^\W\d_]+")
# Least share of a quote's words a passage must hold, in order, for the
# quote to be confirmed; the passage's own words are then shown.
_QUOTE_MATCH: Final[float] = 0.8
# Most words of the passage stretch, as a multiple of the quote's words:
# PDF text can interleave a quoted sentence with margin notes or the
# other column.
_QUOTE_SPREAD: Final[float] = 3.0
# How near a single matching word must be to a run of matching words to
# count, in words.
_QUOTE_NEAR: Final[int] = 3
# Longest unconfirmed quote shown in a comment, in characters.
_SHOWN_QUOTE: Final[int] = 200
# Fewest characters of a quote worth checking; shorter ones are dropped.
_QUOTE_MIN_CHARACTERS: Final[int] = 20


@dataclass(frozen=True, slots=True)
class Claim:
    """A citing sentence and one cited work that is in the project.

    Attributes
    ----------
    number: int
        The reference number.
    sentence: Sentence
        The citing sentence.
    cited: pathlib.Path
        The cited work's file.
    also_cited: tuple[int, ...]
        Other reference numbers the same sentence cites.
    before: str
        Up to two preceding sentences in the same paragraph, for
        disambiguating "this study" and similar references; empty when
        no safe context is available.
    marked: str
        The sentence with every citation marker written in brackets where
        it stands ("a tracking method [5,40,88–91] and then"), so the model
        sees what each cites; empty to use the sentence as extracted.
    cited_for: str
        The words the reference is cited for: usually the phrase just
        before its marker ("a previously established tracking method");
        empty when unknown.
    after: str
        A following sentence from the same paragraph, if available.
        Used only to disambiguate the citing sentence.
    match_status: str
        ``found`` or ``probable`` for the cited paper identity.

    """

    number: int
    sentence: Sentence
    cited: Path
    also_cited: tuple[int, ...] = ()
    before: str = ""
    marked: str = ""
    cited_for: str = ""
    after: str = ""
    match_status: str = "found"


@dataclass(frozen=True, slots=True)
class Verdict:
    """The judgement of one claim.

    Attributes
    ----------
    claim: Claim
        The claim.
    verdict: str
        One of ``VERDICTS``.
    quote: str
        The passage sentence the verdict rests on, as the model copied it
        and found in the passages; empty when none.
    quote_chunk: Chunk | None
        The passage holding the quote.
    comment: str
        The model's comment, with any note Python added.
    passages: tuple[Chunk, ...]
        The passages the model read.
    evidence_status: str
        ``exact``, ``approximate``, ``missing``, ``rejected``,
        ``not required``, or ``not checked`` for a constructed result.

    """

    claim: Claim
    verdict: str
    quote: str
    quote_chunk: Chunk | None
    comment: str
    passages: tuple[Chunk, ...] = ()
    evidence_status: str = "not checked"


@dataclass(frozen=True, slots=True)
class Support:
    """How far one work supports a statement it does not yet cite.

    Attributes
    ----------
    statement: str
        The statement, as the user wrote it.
    cited: pathlib.Path
        The work's file.
    verdict: str
        One of ``VERDICTS``.
    quote: str
        The passage sentence the verdict rests on, in the passage's own
        words; empty when none.
    quote_chunk: Chunk | None
        The passage holding the quote.
    comment: str
        The model's comment, with any note Python added.
    passages: tuple[Chunk, ...]
        The passages the model read.
    evidence_status: str
        ``exact``, ``approximate``, ``missing``, ``rejected``,
        ``not required``, or ``not checked`` for a constructed result.

    """

    statement: str
    cited: Path
    verdict: str
    quote: str
    quote_chunk: Chunk | None
    comment: str
    passages: tuple[Chunk, ...] = ()
    evidence_status: str = "not checked"


def verdict_label(verdict: str) -> str:
    """Return the words shown for a verdict.

    Parameters
    ----------
    verdict: str
        One of ``VERDICTS``.

    Returns
    -------
    str
        "no matching passage" for ``not found`` (none of the passages
        read is about the sentence; the rest of the paper may be);
        otherwise the verdict itself.

    """

    return _LABELS.get(verdict, verdict)


def claim_passages(
    texts: Sequence[str],
    cited: Sequence[Chunk],
    rank: Callable[[str], list],
    count: int,
    max_tokens: int,
) -> tuple[Chunk, ...]:
    """Return the cited file's passages most related to a claim.

    Parameters
    ----------
    texts: Sequence[str]
        Texts to rank the passages for: the citing sentence, and what the
        reference is cited for. Their rankings are fused by reciprocal
        rank, so a passage near the top of either comes first.
    cited: Sequence[Chunk]
        The cited file's passages, in order.
    rank: Callable[[str], list]
        KíLín's ranking of every passage for a text
        (``ToolContext.rank``); results have a ``chunk``.
    count: int
        Most passages.
    max_tokens: int
        Most estimated tokens of all passages together; the best-ranked
        passage is always taken.

    Returns
    -------
    tuple[Chunk, ...]
        The best-ranked passages of the cited file that fit, in reading
        order.

    """

    order = {chunk.identifier: index for index, chunk in enumerate(cited)}
    fused: dict[int, float] = {}
    for text in dict.fromkeys(t for t in texts if t.strip()):
        position = 0
        for result in rank(text):
            index = order.get(result.chunk.identifier)
            if index is None:
                continue
            fused[index] = fused.get(index, 0.0) + 1 / (_FUSION + position)
            position += 1

    picked: list[int] = []
    used = 0
    for index in sorted(fused, key=lambda i: (-fused[i], i)):
        size = estimate_tokens(cited[index].text)
        if picked and used + size > max_tokens:
            continue
        picked.append(index)
        used += size
        if len(picked) == count:
            break

    return tuple(cited[index] for index in sorted(picked))


def verification_prompt(
    claim: Claim, passages: Sequence[Chunk]
) -> tuple[str, str]:
    """Return the prompts that judge one claim.

    Parameters
    ----------
    claim: Claim
        The citing sentence and the cited file.
    passages: Sequence[Chunk]
        The cited file's passages to judge it from.

    Returns
    -------
    tuple[str, str]
        System prompt and user prompt.

    """

    number = claim.number
    system = (
        "You check whether a cited scientific work supports the sentence "
        "that cites it. Judge only from the passages given, never from "
        "what you know. Be strict and fair, and copy quotes exactly."
    )
    others = ""
    if claim.also_cited:
        listed = ", ".join(f"[{n}]" for n in claim.also_cited)
        others = (
            f"\nThe sentence also cites {listed}; reference [{number}] only "
            "needs to support its own part of it."
        )
    cited_for = ""
    if claim.cited_for:
        source = (
            " (as the source of it)"
            if _SOURCE.search(claim.cited_for) else ""
        )
        cited_for = (
            f'\nReference [{number}] is cited for: "{claim.cited_for}"'
            f"{source}. Judge whether the cited work backs that; the rest "
            "of the sentence is context."
        )
    listed_passages = "\n\n".join(
        f"Passage {index}:\n{chunk.text.strip()}"
        for index, chunk in enumerate(passages, start=1)
    )
    preceding = (
        f'Preceding context: "{claim.before}"\n'
        if claim.before else ""
    )
    following = (
        f'Following context: "{claim.after}"\n'
        if claim.after else ""
    )
    user = f"""A paper writes this sentence, citing reference [{number}]:

"{claim.marked or claim.sentence.text}"
{preceding}{following}{cited_for}{others}
The surrounding sentences identify what expressions such as "this study"
refer to. Judge only what the cited sentence attributes to [{number}];
do not require the cited paper to restate the citing paper's own wording.
When the citing sentence says that "the results of this study were
published" in [{number}], treat this as attribution to the publication:
use the preceding description to identify the study. If the cited work
reports those specific results, it supports the attribution, even if it
does not say "these results were published". If it merely discusses a
related study, use partly supported or not supported as appropriate. Do
not infer a match just from a shared topic; require specific evidence.
Passages of the cited work [{number}] ({claim.cited.name}), the parts most
related to the sentence:

{listed_passages}

What does the sentence use reference [{number}] for, and do these passages
back it? Choose one verdict:
- supported: the passages state what the sentence cites [{number}] for.
- partly supported: they state part of it, or something weaker, narrower,
  or under other conditions (another species, dataset, method or number).
- not supported: they discuss the topic but do not state it.
- contradicted: they state the opposite or a clearly different result.
- background: the sentence only names [{number}] as the source of a method,
  tool, dataset, challenge or topic, and the passages are about that.
- not found: the passages are not about the sentence's topic.

Two rules:
- Words the citing paper uses to describe the reference ("previously
  established", "recent", "popular", "state-of-the-art", "first") are its
  own view; the cited work does not need to say them about itself.
- When [{number}] is the source of a method, tool, dataset or challenge
  ("using the method of [{number}]") and the passages describe it, the
  verdict is background, even without the sentence's wording.

Answer in exactly three lines:
VERDICT: <supported, partly supported, not supported, contradicted,
background, or not found>
QUOTE: <the one sentence of the passages that best supports or contradicts
the claim, copied exactly, or NONE>
COMMENT: <one or two sentences explaining the verdict critically>"""
    return system, user


def support_prompt(
    statement: str, cited: Path, passages: Sequence[Chunk]
) -> tuple[str, str]:
    """Return the prompts that judge whether a work could be cited.

    Parameters
    ----------
    statement: str
        A sentence that needs a reference.
    cited: pathlib.Path
        The candidate work's file.
    passages: Sequence[Chunk]
        The work's passages to judge it from.

    Returns
    -------
    tuple[str, str]
        System prompt and user prompt, asking for the same three lines as
        ``verification_prompt``.

    """

    system = (
        "You judge whether a scientific work could be cited as the source "
        "of a sentence. Judge only from the passages given, never from "
        "what you know. Be strict and fair, and copy quotes exactly."
    )
    listed_passages = "\n\n".join(
        f"Passage {index}:\n{chunk.text.strip()}"
        for index, chunk in enumerate(passages, start=1)
    )
    user = f"""A writer needs a reference for this sentence:

"{statement}"

Passages of a candidate work ({cited.name}), the parts most related to
the sentence:

{listed_passages}

Could this work be cited for the sentence? Choose one verdict:
- supported: the passages state what the sentence says.
- partly supported: they state part of it, or something weaker, narrower,
  or under other conditions (another species, dataset, method or number).
- not supported: they discuss the topic but do not state it.
- contradicted: they state the opposite or a clearly different result.
- background: the sentence names a method, tool, dataset, challenge or
  topic, and the work is about that, without stating the sentence's claim.
- not found: the passages are not about the sentence's topic.

Two rules:
- Judge what the work itself states. When the passages only repeat the
  claim citing other work for it ("as shown in [12]"), answer partly
  supported and say in the comment that the original source is elsewhere.
- Words of opinion in the sentence ("popular", "recent", "first") need
  not be stated by the work.

Answer in exactly three lines:
VERDICT: <supported, partly supported, not supported, contradicted,
background, or not found>
QUOTE: <the one sentence of the passages that best supports or contradicts
the sentence, copied exactly, or NONE>
COMMENT: <one or two sentences explaining the verdict critically>"""
    return system, user


def _reply_fields(reply: str) -> dict[str, str]:
    """Read model fields, including wrapped quotation lines.

    Parameters
    ----------
    reply: str
        Model answer with VERDICT, QUOTE, and COMMENT headings.

    Returns
    -------
    dict[str, str]
        Fields with continuation lines retained until the next heading.

    """

    matches = list(_LINE.finditer(reply))
    fields: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) \
            else len(reply)
        fields[match.group(1).upper()] = (
            match.group(2) + reply[match.end():end]
        ).strip()
    return fields


def parse_verdict(
    reply: str, passages: Sequence[Chunk]
) -> tuple[str, str, Chunk | None, str]:
    """Return the model's checked verdict, quote and comment.

    Parameters
    ----------
    reply: str
        The model's answer.
    passages: Sequence[Chunk]
        The passages it was given.

    Returns
    -------
    tuple[str, str, Chunk | None, str]
        Verdict, quote, the passage holding the quote, and comment. The
        quote is the passage's own words: the stretch of a passage that
        the model's quote matches word for word, or for at least
        ``_QUOTE_MATCH`` of its words in order (it may drop citation
        numbers glued to words, or mend a hyphenated word). A quote found
        in no passage is dropped, and the comment says what the model
        quoted; ``contradicted`` without a confirmed quote becomes ``not
        supported``. Without a recognised verdict, ``unclear`` with the
        reply as comment.

    """

    fields = _reply_fields(reply)
    said = fields.get("VERDICT", "").casefold().strip(" .*\"'")
    verdict = next(
        (name for name in _MODEL_VERDICTS if said.startswith(name)), None
    )
    comment = fields.get("COMMENT", "")
    if verdict is None:
        return VERDICT_UNCLEAR, "", None, " ".join(reply.split())[:300]

    said_quote = fields.get("QUOTE", "").strip().strip("\"“”'")
    quote, chunk = "", None
    wanted = said_quote.upper() != "NONE"
    if wanted and len(said_quote) >= _QUOTE_MIN_CHARACTERS:
        found = _passage_with(said_quote, passages)
        if found is not None:
            chunk, quote = found
    unfound = bool(said_quote) and wanted and not quote
    shown = said_quote if len(said_quote) <= _SHOWN_QUOTE else (
        said_quote[:_SHOWN_QUOTE].rsplit(" ", 1)[0] + " …"
    )

    if verdict == VERDICT_CONTRADICTED and chunk is None:
        verdict = VERDICT_NOT_SUPPORTED
        why = (
            f"its quote, “{shown}”, is not in the passages" if unfound
            else "it quoted no contradicting sentence"
        )
        comment = f"{comment} (Judged contradicted, but {why}.)".strip()
    elif unfound:
        comment = (
            f"{comment} (Its quote, “{shown}”, is not in the passages.)"
        ).strip()

    return verdict, quote, chunk, comment


def _checked_verdict(
    reply: str, passages: Sequence[Chunk]
) -> tuple[str, str, Chunk | None, str, str]:
    """Parse a verdict and expose independently checked quote status.

    Parameters
    ----------
    reply: str
        Model-generated assessment.
    passages: Sequence[Chunk]
        Retrieved evidence passages.

    Returns
    -------
    tuple[str, str, Chunk | None, str, str]
        The public parser's fields plus exact, approximate, missing,
        rejected or not-required quotation status. Negative verdicts with
        ``QUOTE: NONE`` need no confirming positive-evidence quote.
        A quote's presence does not verify
        the scientific correctness of the model's interpretation.

    """

    verdict, quote, chunk, comment = parse_verdict(reply, passages)
    fields = _reply_fields(reply)
    claimed = fields.get("QUOTE", "").strip().strip("\"“”'")
    if chunk is not None:
        original = _HYPHENATED.sub("", chunk.text)
        normalized = lambda text: " ".join(text.casefold().split())
        status = (
            "exact" if normalized(claimed) in normalized(original)
            else "approximate"
        )
    elif claimed and claimed.upper() != "NONE":
        status = "rejected"
    elif verdict in (
        VERDICT_NOT_SUPPORTED, VERDICT_NOT_FOUND, VERDICT_UNCLEAR,
    ):
        # An unsupported, off-topic or unparsable assessment need not
        # supply a positive-evidence quote. QUOTE: NONE is expected.
        status = "not required"
    else:
        status = "missing"
    return verdict, quote, chunk, comment, status


def verify_claim(
    claim: Claim,
    passages: Sequence[Chunk],
    ask_model: Callable[[str, str, int | None], str],
) -> Verdict:
    """Judge one claim with the answering model.

    Parameters
    ----------
    claim: Claim
        The citing sentence and the cited file.
    passages: Sequence[Chunk]
        The cited file's passages to judge it from.
    ask_model: Callable[[str, str, int | None], str]
        The answering model (``ToolContext.ask_model``).

    Returns
    -------
    Verdict
        ``no text`` without passages (no model call); ``failed`` when
        the model call raises; otherwise the checked answer.

    """

    verdict, quote, chunk, comment, status = _judge(
        verification_prompt(claim, passages) if passages else ("", ""),
        passages, ask_model,
    )
    return Verdict(
        claim, verdict, quote, chunk, comment, tuple(passages), status
    )


def judge_support(
    statement: str,
    cited: Path,
    passages: Sequence[Chunk],
    ask_model: Callable[[str, str, int | None], str],
) -> Support:
    """Judge whether one work supports a statement, with the model.

    Parameters
    ----------
    statement: str
        A sentence that needs a reference.
    cited: pathlib.Path
        The candidate work's file.
    passages: Sequence[Chunk]
        The work's passages to judge it from.
    ask_model: Callable[[str, str, int | None], str]
        The answering model (``ToolContext.ask_model``).

    Returns
    -------
    Support
        Checked as ``verify_claim`` checks a claim: ``no text`` without
        passages, ``failed`` when the call raises, quotes confirmed in the
        passages.

    """

    verdict, quote, chunk, comment, status = _judge(
        support_prompt(statement, cited, passages) if passages else ("", ""),
        passages, ask_model,
    )
    return Support(
        statement, cited, verdict, quote, chunk, comment,
        tuple(passages), status,
    )


def _judge(
    prompts: tuple[str, str],
    passages: Sequence[Chunk],
    ask_model: Callable[[str, str, int | None], str],
) -> tuple[str, str, Chunk | None, str, str]:
    """Ask the model for a verdict and check its answer.

    Parameters
    ----------
    prompts: tuple[str, str]
        System prompt and user prompt.
    passages: Sequence[Chunk]
        The passages the prompt shows; none means no model call.
    ask_model: Callable[[str, str, int | None], str]
        The answering model.

    Returns
    -------
    tuple[str, str, Chunk | None, str, str]
        Verdict, quote, the passage holding it, comment and evidence
        status; ``no text`` without passages and ``failed`` on errors.

    """

    if not passages:
        return (
            VERDICT_NO_TEXT, "", None, "The file has no searchable text.",
            "no text",
        )
    try:
        reply = ask_model(prompts[0], prompts[1], None)
    except IncompleteModelResponse as error:
        return (
            VERDICT_FAILED, "", None,
            f"Not completed ({error.reason}): {error}", "failed",
        )
    except Exception as error:  # A failed call is reported, not fatal.
        return VERDICT_FAILED, "", None, f"Model error: {error}", "failed"
    return _checked_verdict(reply, passages)


def _passage_with(
    quote: str, passages: Sequence[Chunk]
) -> tuple[Chunk, str] | None:
    """Return the passage holding a quote, and the quote as printed there.

    Parameters
    ----------
    quote: str
        Text the model says it copied.
    passages: Sequence[Chunk]
        The passages it was given.

    Returns
    -------
    tuple[Chunk, str] | None
        The first passage whose words contain the quote's words (letters
        only, so spacing, line breaks, hyphenation, quotation marks and
        citation numbers do not matter), or failing that the passage with
        a stretch matching at least ``_QUOTE_MATCH`` of the quote's words
        in order, no longer than ``_QUOTE_SPREAD`` times the quote; the
        stretch runs from the first to the last run of two or more
        matching words, with single matching words only within
        ``_QUOTE_NEAR`` words of those runs; and
        the passage's matching words, as printed, with "…" where words of
        the passage between them are left out. None when no passage holds
        the quote.

    """

    wanted = _words(quote)
    if not wanted:
        return None

    best: tuple[float, Chunk, str] | None = None
    for chunk in passages:
        # Words hyphenated across lines ("perfor- mance") are joined.
        text = _HYPHENATED.sub("", chunk.text)
        spans = [
            (match.group(0).casefold(), match.start(), match.end())
            for match in _WORD.finditer(text)
        ]
        words = [word for word, _, _ in spans]
        matched = [
            block for block in SequenceMatcher(
                None, words, wanted, autojunk=False
            ).get_matching_blocks()
            if block.size
        ]
        # A quote matches in runs of words; single words matching far from
        # them are chance ("We" earlier in the passage).
        runs = [block for block in matched if block.size >= 2]
        if not runs:
            continue
        low = runs[0].a - _QUOTE_NEAR
        high = runs[-1].a + runs[-1].size + _QUOTE_NEAR
        blocks = [
            block for block in matched
            if block.a + block.size >= low and block.a <= high
        ]
        first = blocks[0].a
        last = blocks[-1].a + blocks[-1].size - 1
        share = sum(block.size for block in blocks) / len(wanted)
        if (
            share >= _QUOTE_MATCH
            and last - first + 1 <= _QUOTE_SPREAD * len(wanted)
            and (best is None or share > best[0])
        ):
            # The paper's words from the first match to the last, "…"
            # where more than _QUOTE_NEAR words are skipped (interleaved
            # margin text, say).
            stretches = [[blocks[0].a, blocks[0].a + blocks[0].size - 1]]
            for block in blocks[1:]:
                if block.a - stretches[-1][1] - 1 > _QUOTE_NEAR:
                    stretches.append([block.a, block.a + block.size - 1])
                else:
                    stretches[-1][1] = block.a + block.size - 1
            shown = " … ".join(
                " ".join(text[spans[start][1]:spans[end][2]].split())
                for start, end in stretches
            )
            # Compare even the skipped stretch: a negation or changed
            # numerical result in interleaved PDF text must not be hidden
            # by an ellipsis used only for display.
            full_stretch = text[spans[first][1]:spans[last][2]]
            if not _consistent_quote(quote, full_stretch):
                continue
            best = (share, chunk, shown)
            if share == 1.0:
                break

    return None if best is None else (best[1], best[2])


def _words(text: str) -> list[str]:
    """Return a text's words, lowercase, letters only.

    Parameters
    ----------
    text: str
        Any text.

    Returns
    -------
    list[str]
        Runs of letters, so digits (citation numbers glued to words),
        spaces, punctuation and symbols do not count.

    """

    return [word.casefold() for word in _WORD.findall(text)]


# Terms for which a one-word difference may reverse the scientific
# interpretation even if the remainder of a quotation matches closely.
_CRITICAL_WORDS: Final[frozenset[str]] = frozenset((
    "not", "no", "none", "never", "without", "absent", "absence",
    "present", "presence", "increased", "increases", "increase",
    "decreased", "decreases", "decrease", "reduced", "reduces",
    "reduce", "raised", "raises", "lowered", "lowers", "higher",
    "lower", "greater", "less", "more", "positive", "negative",
    "significant", "insignificant", "significantly", "nonsignificant",
    "before", "after", "above", "below", "equal", "unequal",
    "improved", "worsened", "improves", "worsens", "faster", "slower",
    "only", "all", "every", "some", "most", "least", "at", "up", "down",
))
_CRITICAL_UNITS: Final[frozenset[str]] = frozenset((
    "mg", "kg", "g", "ug", "μg", "ml", "l", "mm", "nm", "um", "μm",
    "ms", "ns", "hz", "khz", "mhz", "kda", "da", "kb", "mb",
))
_NUMERICAL: Final[re.Pattern[str]] = re.compile(
    r"(?<![\w.])[-+−]?\d+(?:[.,]\d+)?(?:\s*[%‰])?(?![\w])"
)


def _consistent_quote(model_quote: str, source_quote: str) -> bool:
    """Reject numeric, negation and directional changes in fuzzy quotes.

    Parameters
    ----------
    model_quote: str
        Sentence claimed to be copied by the model.
    source_quote: str
        Located stretch in the paper's own words.

    Returns
    -------
    bool
        Whether critical values and semantic qualifiers are preserved.
        This is a conservative text-integrity check, not a proof that a
        passage entails the model's verdict.

    """

    model_nums = [
        value.replace(" ", "") for value in _NUMERICAL.findall(model_quote)
    ]
    source_nums = [
        value.replace(" ", "") for value in _NUMERICAL.findall(source_quote)
    ]
    # Source-only citation numbers are allowed; every model-claimed
    # number must nonetheless occur in the located source text.
    remaining = list(source_nums)
    for number in model_nums:
        if number not in remaining:
            return False
        remaining.remove(number)
    claimed = [
        word for word in _words(model_quote) if word in _CRITICAL_WORDS
    ]
    actual = [
        word for word in _words(source_quote) if word in _CRITICAL_WORDS
    ]
    if claimed != actual:
        return False
    units_model = [
        word for word in _words(model_quote) if word in _CRITICAL_UNITS
    ]
    units_source = [
        word for word in _words(source_quote) if word in _CRITICAL_UNITS
    ]
    return units_model == units_source


def quote_line_range(
    quote: str, chunk: Chunk | None,
) -> tuple[int, int] | None:
    """Return exact source lines for a verbatim located quotation.

    Parameters
    ----------
    quote: str
        The validated quotation in its source wording.
    chunk: Chunk | None
        The indexed passage containing the quotation.

    Returns
    -------
    tuple[int, int] | None
        First and last source lines for an exact whitespace-tolerant
        match. None for a fuzzy/interleaved quote, for which callers
        should cite the original chunk's broader line range.

    """

    if chunk is None or not quote:
        return None
    tokens = quote.split()
    if not tokens or "…" in tokens:
        return None
    pattern = r"\s+".join(re.escape(token) for token in tokens)
    matched = re.search(pattern, chunk.text, flags=re.I)
    if matched is None:
        return None
    first = chunk.start_line + chunk.text.count("\n", 0, matched.start())
    last = chunk.start_line + chunk.text.count("\n", 0, matched.end())
    return first, last
