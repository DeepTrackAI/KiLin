"""Provide one public entry point for retrieval methods.

Classes
-------
LexicalIndex, LexicalResult
    Data produced by lexical retrieval.
SemanticIndex, SemanticResult
    Data produced by semantic retrieval.

Functions
---------
build_lexical_index, lexical_search, lexical_terms
    Build and search a lexical TF-IDF index.
build_semantic_index, semantic_search
    Build and search an embedding index.

"""

from qachat.retrievals.strategies.retrieval_lexical import (
    LexicalIndex,
    LexicalResult,
    build_lexical_index,
    lexical_search,
    lexical_terms,
)
from qachat.retrievals.strategies.retrieval_semantic import (
    SemanticIndex,
    SemanticResult,
    build_semantic_index,
    semantic_search,
)

__all__ = [
    "LexicalIndex",
    "LexicalResult",
    "SemanticIndex",
    "SemanticResult",
    "build_lexical_index",
    "build_semantic_index",
    "lexical_search",
    "lexical_terms",
    "semantic_search",
]
