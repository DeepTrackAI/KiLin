"""Provide one public entry point for retrieval methods.

Classes
-------
CodeEntity, CodeGraph, CodeRelation, CodeResult
    Data produced by code-aware retrieval.
HybridResult
    One result combined from several retrieval channels.
PersistentIndexes
    Semantic, lexical, and code-graph data saved together.
LexicalIndex, LexicalResult
    Data produced by lexical retrieval.
SemanticIndex, SemanticResult
    Data produced by semantic retrieval.

Functions
---------
build_code_graph, code_search, resolve_code_entities
    Build and search deterministic source-code relationships.
build_lexical_index, lexical_search, lexical_terms
    Build and search a lexical TF-IDF index.
build_semantic_index, semantic_search
    Build and search an embedding index.
hybrid_search
    Combine independent rankings with Reciprocal Rank Fusion.
prepare_persistent_indexes, save_indexes, load_indexes, indexes_are_current
    Save, restore, and validate reusable retrieval data.

"""

from qachat.retrievals.persistence import (
    PersistentIndexes,
    indexes_are_current,
    load_indexes,
    prepare_persistent_indexes,
    save_indexes,
)
from qachat.retrievals.code_graph import (
    CodeEntity,
    CodeGraph,
    CodeRelation,
    build_code_graph,
)
from qachat.retrievals.strategies.retrieval_code import (
    CodeResult,
    code_search,
    resolve_code_entities,
)
from qachat.retrievals.strategies.retrieval_hybrid import (
    HybridResult,
    hybrid_search,
)
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
    "CodeEntity",
    "CodeGraph",
    "CodeRelation",
    "CodeResult",
    "HybridResult",
    "LexicalIndex",
    "LexicalResult",
    "PersistentIndexes",
    "SemanticIndex",
    "SemanticResult",
    "build_code_graph",
    "build_lexical_index",
    "build_semantic_index",
    "code_search",
    "hybrid_search",
    "indexes_are_current",
    "lexical_search",
    "lexical_terms",
    "load_indexes",
    "prepare_persistent_indexes",
    "resolve_code_entities",
    "save_indexes",
    "semantic_search",
]
