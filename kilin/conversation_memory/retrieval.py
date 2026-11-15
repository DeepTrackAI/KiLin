"""Retrieve conversation memory without promoting it to source evidence.

Memory hits are ranked independently from primary-source retrieval and never
become ``Evidence`` instances. Semantic and lexical relevance determine
eligibility before bounded Save/Like importance is applied. The project
controls its own hit count and separate context-token budget.

Key Features
------------

- **Independent relevance and priority**

    Reject irrelevant saved answers before applying their small rank boost.

- **Distinct provenance**

    Provide conversation, turn UUID, role, time, and source-free memory labels.

- **Bounded context**

    Limit both total results and the complete rendered memory section.

Module Structure
----------------

- ``MemoryHit``

    One retrieved prior message with its measured score and text slice.

- ``search_memory``

    Search a synchronized project memory index.

- ``render_memory_hits``

    Construct a labeled, quoted memory section for the answer prompt.

Examples
--------
>>> from kilin.conversation_memory.retrieval import render_memory_hits
>>> render_memory_hits(())
''

"""

from __future__ import annotations

from dataclasses import dataclass
from math import log1p, sqrt
import re
from typing import Final

from kilin.conversation_memory.index import MemoryIndex, MemoryRecord
from kilin.retrievals.document_reading import estimate_tokens

__all__ = ["MemoryHit", "render_memory_hits", "search_memory"]

_TERMS: Final[re.Pattern[str]] = re.compile(r"\w+", re.UNICODE)
_STOP: Final[frozenset[str]] = frozenset((
    "the", "and", "that", "with", "this", "from", "have", "what",
    "which", "please", "about", "could", "would", "into", "were",
    "then", "your", "their", "also", "when", "does", "there", "are",
    "for", "you", "how", "can", "our", "was", "has", "its", "not",
))
# Weak semantic similarity alone is especially noisy for small local models.
# An exact historical lookup instead requires distinctive lexical overlap.
_MIN_SEMANTIC: Final[float] = 0.48
_MIN_LEXICAL: Final[float] = 0.38


@dataclass(frozen=True, slots=True)
class MemoryHit:
    """One retrieved message with non-evidentiary conversation provenance.

    Attributes
    ----------
    record: MemoryRecord
        Saved message and all of its active conversation origins.
    text: str
        Budget-bounded message content (may be shorter than stored text).
    relevance: float
        Semantic/lexical relevance before importance weighting.
    score: float
        Relevance with a small importance adjustment.

    """

    record: MemoryRecord
    text: str
    relevance: float
    score: float


def _terms(text: str) -> set[str]:
    """Reduce searchable text to meaningful lexical tokens.

    Parameters
    ----------
    text: str
        Current question or remembered message.

    Returns
    -------
    set[str]
        Distinct normalized terms, excluding common function words.

    """

    return {
        word.lower() for word in _TERMS.findall(text)
        if len(word) >= 3 and word.lower() not in _STOP
    }


def _similarity(a: list[float], b: tuple[float, ...]) -> float:
    """Return cosine similarity for nonzero, equal-dimensional vectors.

    Parameters
    ----------
    a: list[float]
        Query embedding.
    b: tuple[float, ...]
        Persisted normalized memory embedding.

    Returns
    -------
    float
        Cosine similarity, or zero when vector geometry is invalid.

    """

    if len(a) != len(b) or not a:
        return 0.0
    norm_a = sqrt(sum(value * value for value in a))
    norm_b = sqrt(sum(value * value for value in b))
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (norm_a * norm_b)


def render_memory_hits(hits: tuple[MemoryHit, ...]) -> str:
    """Render untrusted previous messages in a distinct prompt channel.

    Parameters
    ----------
    hits: tuple[MemoryHit, ...]
        Relevant, budget-limited conversation-memory records.

    Returns
    -------
    str
        Labeled quotation with memory-only references (M1, M2, ...).
        Empty when no memory hits are present.

    Notes
    -----
    No source citation number or path is created here. Text indentation is
    a presentation boundary, not a security sandbox; prompt instructions
    must also prohibit treating remembered text as trusted instructions.

    """

    if not hits:
        return ""
    parts = [
        "CONVERSATION MEMORY (previous discussion, NOT source evidence):"
    ]
    for number, hit in enumerate(hits, 1):
        record = hit.record
        origins = ", ".join(
            f"{origin.conversation_id} turn {origin.turn_number}"
            for origin in record.origins
        )
        parts.append(
            f"Memory M{number} | {record.role} | "
            f"UUID {record.turn_uuid} | {record.timestamp} | {origins}\n"
            + "\n".join(
                "    " + line for line in hit.text.splitlines()
            )
        )
    return "\n\n".join(parts)


def search_memory(
    question: str,
    index: MemoryIndex,
    embedding_session: object,
    *,
    max_hits: int,
    max_tokens: int,
    excluded_uuids: frozenset[str] = frozenset(),
    query_text: str | None = None,
    recall_topic: str | None = None,
) -> tuple[MemoryHit, ...]:
    """Search and budget an independent conversation-memory index.

    Parameters
    ----------
    question: str
        Current researcher's request, used for lexical relevance.
    index: MemoryIndex
        Synchronized project memory records and their embeddings.
    embedding_session: object
        Already loaded embedding model with ``embed_query``.
    max_hits: int
        Maximum number of distinct message occurrences.
    max_tokens: int
        Estimated memory-section token budget, separate from evidence.
    excluded_uuids: frozenset[str], optional
        Recent/related turns already supplied as live chat context.
    query_text: str | None, optional
        Embedding-model-formatted query. Defaults to the raw question.
    recall_topic: str | None, optional
        Explicit historical-lookup topic. When set, verify occurrences
        lexically against saved records without calling the embedder.

    Returns
    -------
    tuple[MemoryHit, ...]
        Relevant hits in rank order, within both independent budgets.

    Notes
    -----
    Saved or liked content without query relevance is not returned. A valid
    memory hit is never assigned a primary-source citation number.

    """

    if max_hits <= 0 or max_tokens <= 0 or not question.strip():
        return ()
    candidates = [
        record for record in index.records
        if record.turn_uuid not in excluded_uuids
    ]
    if not candidates:
        return ()
    ranked: list[tuple[float, float, MemoryRecord]] = []
    if recall_topic is not None:
        # The question itself is not evidence that a discussion occurred.
        # Only actual saved-message terms can establish a match.
        topic_terms = _terms(recall_topic)
        if not topic_terms:
            return ()
        for record in candidates:
            found = _terms(record.text) & topic_terms
            required = max(1, (3 * len(topic_terms) + 3) // 4)
            if len(found) >= required:
                relevance = len(found) / len(topic_terms)
                ranked.append((relevance, relevance, record))
        # A matching question establishes that the researcher asked it.
        # Its eligible answer may supply context, but is never a verified
        # scientific conclusion. A disliked/excluded answer is absent.
        matched_questions = {
            record.turn_uuid for _, _, record in ranked
            if record.role == "user"
        }
        for record in candidates:
            if (record.role == "assistant"
                    and record.turn_uuid in matched_questions
                    and not any(item[2] == record for item in ranked)):
                ranked.append((0.01, 0.01, record))
        ranked.sort(key=lambda item: (
            -int(item[2].role == "user"), -item[0],
            item[2].timestamp, item[2].turn_uuid,
        ))
    else:
        query_vector = embedding_session.embed_query(
            question if query_text is None else query_text
        )
        terms = _terms(question)
        document_terms = [_terms(item.text) for item in candidates]
        term_frequency = {
            term: sum(term in words for words in document_terms)
            for term in terms
        }
        weights = {
            term: log1p((len(candidates) + 1) / (count + 1))
            for term, count in term_frequency.items()
        }
        denominator = sum(weights.values())
        scored: list[tuple[
            float, float, MemoryRecord, float, float
        ]] = []
        for record, record_terms in zip(candidates, document_terms):
            vector = index.vectors.get(record.fingerprint)
            if vector is None:
                continue
            semantic = max(0.0, _similarity(query_vector, vector))
            lexical = (
                sum(weights[term] for term in terms & record_terms)
                / denominator if denominator else 0.0
            )
            if semantic < _MIN_SEMANTIC and lexical < _MIN_LEXICAL:
                continue
            relevance = 0.75 * semantic + 0.25 * lexical
            priority = 0.025 if record.priority == 1 else 0.0
            if record.priority == 2:
                priority = 0.05
            scored.append((
                relevance + priority, relevance, record, semantic, lexical
            ))
        # If distinctive query words have a good matching candidate,
        # avoid filling the context with high-cosine but zero-overlap
        # distractors produced by coarse or anisotropic local embeddings.
        lexical_anchors = [
            item[3] for item in scored if item[4] >= 0.75
        ]
        anchor_semantic = max(lexical_anchors, default=None)
        ranked = [
            (score, relevance, record)
            for score, relevance, record, semantic, lexical in scored
            if (anchor_semantic is None or lexical >= 0.05
                or semantic > anchor_semantic + 0.10)
        ]
        ranked.sort(key=lambda item: (
            -item[0], -item[1], item[2].timestamp,
            item[2].turn_uuid, item[2].role,
        ))
    result: list[MemoryHit] = []
    displayed_fingerprints: set[str] = set()
    for score, relevance, record in ranked:
        if len(result) >= max_hits:
            break
        # Separate feedback remains attached to each occurrence, while
        # identical text need only appear once in the answering prompt.
        if (recall_topic is None
                and record.fingerprint in displayed_fingerprints):
            continue
        # Account for the rendered header and origin metadata, not only text.
        hit = MemoryHit(record, record.text, relevance, score)
        if estimate_tokens(render_memory_hits((*result, hit))) > max_tokens:
            # Allow a shorter last message, but never an orphaned header.
            available = max_tokens * 4
            overhead = len(render_memory_hits((*result, MemoryHit(
                record, "", relevance, score
            ))))
            characters = max(0, available - overhead)
            if characters < 24:
                continue
            shortened = record.text[:characters].rstrip()
            if len(shortened) < len(record.text):
                shortened += "..."
            hit = MemoryHit(record, shortened, relevance, score)
            while (
                shortened and estimate_tokens(render_memory_hits(
                    (*result, hit)
                )) > max_tokens
            ):
                shortened = shortened[:-4].rstrip()
                hit = MemoryHit(record, shortened, relevance, score)
            if not shortened:
                continue
        result.append(hit)
        displayed_fingerprints.add(record.fingerprint)
    return tuple(result)
