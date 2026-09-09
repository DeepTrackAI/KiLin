"""Expose the retrieval strategies bundled with QAChat."""

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
