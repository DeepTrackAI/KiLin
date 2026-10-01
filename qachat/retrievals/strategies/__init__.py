"""Expose the concrete retrieval strategies and their result records.

Lexical retrieval uses TF-IDF weights, semantic retrieval uses embedding
cosine similarity, code retrieval follows recognized entity relationships,
and hybrid retrieval combines existing rankings through reciprocal ranks.
Each strategy has its own score semantics. This initializer only re-exports
the listed implementations; it does not build indexes or perform searches.

Key Features
------------

- **Independent strategies**

    Keep text, vector, and symbolic matching independently callable.

- **Rank fusion**

    Combine channel rankings without treating their raw scores as
    interchangeable.

Module Structure
----------------

- ``CodeResult``

    One chunk found through code-aware retrieval.

- ``HybridResult``

    One chunk ranked across retrieval channels.

- ``LexicalIndex``

    Chunks paired with sparse TF-IDF vectors.

- ``LexicalResult``

    One chunk returned by lexical ranking.

- ``SemanticIndex``

    Chunks paired with vectors prepared for semantic search.

- ``SemanticResult``

    One chunk returned by semantic ranking.

- ``build_lexical_index``

    Calculate TF-IDF vectors for repository chunks.

- ``build_semantic_index``

    Embed chunks once and return their searchable semantic index.

- ``code_search``

    Find chunks for explicitly named and related code entities.

- ``hybrid_search``

    Fuse ranked retrieval lists without comparing unlike raw scores.

- ``lexical_search``

    Rank indexed chunks by TF-IDF cosine similarity to one query.

- ``lexical_terms``

    Return lowercase one-word and two-word terms from text.

- ``resolve_code_entities``

    Return entities explicitly named by a query.

- ``semantic_search``

    Rank indexed chunks by cosine similarity to one query.

Examples
--------
>>> import qachat
>>> terms = qachat.lexical_terms("read_repository_files")

"""

from .retrieval_code import (
    CodeResult,
    code_search,
    resolve_code_entities,
)
from .retrieval_hybrid import HybridResult, hybrid_search
from .retrieval_lexical import (
    LexicalIndex,
    LexicalResult,
    build_lexical_index,
    lexical_search,
    lexical_terms,
)
from .retrieval_semantic import (
    SemanticIndex,
    SemanticResult,
    build_semantic_index,
    semantic_search,
)


__all__ = [
    "CodeResult",
    "HybridResult",
    "LexicalIndex",
    "LexicalResult",
    "SemanticIndex",
    "SemanticResult",
    "build_lexical_index",
    "build_semantic_index",
    "code_search",
    "hybrid_search",
    "lexical_search",
    "lexical_terms",
    "resolve_code_entities",
    "semantic_search",
]
