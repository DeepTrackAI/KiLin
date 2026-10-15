"""Expose QAChat's retrieval, code-graph, and index-persistence API.

This namespace collects the concrete strategy implementations, source-code
relations, query routing helpers, and serialized index containers. It performs
no search or index construction during import. Model-dependent semantic
indexing remains an explicit operation on an embedding session.

Key Features
------------

- **Retrieval strategies**

    Expose semantic, lexical, symbolic, and reciprocal-rank-fusion search
    operations.

- **Index lifecycle**

    Expose index construction, persistence, and chunk/graph fingerprint
    comparison.

- **Context routing**

    Expose deterministic explicit-path and function-neighbor expansion helpers.

- **Whole-document reading**

    Choose one named or dominant document and fit it to a token budget.

Module Structure
----------------

- ``DOCUMENT_REASON_NAMED_FUNCTION``, ``DOCUMENT_REASON_NAMED_FILE``, and
  ``DOCUMENT_REASON_SEARCH``

    Canonical string values for why a document was chosen for reading.

- ``DOCUMENT_REASON_CHOICES``

    Ordered collection of the document reasons.

- ``CodeEntity``

    One named source-code element.

- ``CodeGraph``

    Entities and deterministic code relationships.

- ``CodeRelation``

    One directed relationship between code entities.

- ``CodeResult``

    One chunk found through code-aware retrieval.

- ``DocumentReason``

    Type alias for why a document was chosen for reading.

- ``DocumentSelection``

    The document chosen for reading and the chunks supplied from it.

- ``HybridResult``

    One chunk ranked across retrieval channels.

- ``LexicalIndex``

    Chunks paired with sparse TF-IDF vectors.

- ``LexicalResult``

    One chunk returned by lexical ranking.

- ``PersistentIndexes``

    Retrieval data saved for the same repository state.

- ``RetrievalPlan``

    Deterministic context expansion requested by one question.

- ``SemanticIndex``

    Chunks paired with vectors prepared for semantic search.

- ``SemanticResult``

    One chunk returned by semantic ranking.

- ``build_code_graph``

    Build a code graph without relying on directory roles.

- ``build_lexical_index``

    Calculate TF-IDF vectors for repository chunks.

- ``build_semantic_index``

    Embed chunks once and return their searchable semantic index.

- ``code_search``

    Find chunks for explicitly named and related code entities.

- ``dominant_path``

    Return the file that dominates a ranking.

- ``estimate_tokens``

    Estimate the token count of text without a tokenizer.

- ``expand_retrieval_context``

    Add complete requested context after normal RRF retrieval.

- ``hybrid_search``

    Fuse ranked retrieval lists without comparing unlike raw scores.

- ``indexes_are_current``

    Return whether saved retrieval data matches the current repository.

- ``lexical_search``

    Rank indexed chunks by TF-IDF cosine similarity to one query.

- ``lexical_terms``

    Return lowercase one-word and two-word terms from text.

- ``load_indexes``

    Load semantic, lexical, and code-graph data from JSON.

- ``named_paths``

    Return the files a question names.

- ``prepare_persistent_indexes``

    Validate and combine retrieval data from one repository state.

- ``plan_retrieval``

    Detect exact repository context requested by one question.

- ``read_document``

    Choose one whole document or function and fit it to a token budget.

- ``resolve_code_entities``

    Return entities explicitly named by a query.

- ``save_indexes``

    Save persistent retrieval data as JSON and return its path.

- ``semantic_matrix``

    Stack one index's embeddings into a single numpy array.

- ``semantic_search``

    Rank indexed chunks by cosine similarity to one query.

Examples
--------
>>> import qachat
>>> terms = qachat.lexical_terms("ChatSession.restore_history")

"""

from qachat.retrievals.persistence import (
    PersistentIndexes,
    indexes_are_current,
    load_indexes,
    prepare_persistent_indexes,
    save_indexes,
)
from qachat.retrievals.code_graph.code_graph import (
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
    semantic_matrix,
    semantic_search,
)
from qachat.retrievals.routing import (
    RetrievalPlan,
    expand_retrieval_context,
    plan_retrieval,
)
from qachat.retrievals.document_reading import (
    DOCUMENT_REASON_CHOICES,
    DOCUMENT_REASON_NAMED_FILE,
    DOCUMENT_REASON_NAMED_FUNCTION,
    DOCUMENT_REASON_SEARCH,
    DocumentReason,
    DocumentSelection,
    dominant_path,
    estimate_tokens,
    named_paths,
    read_document,
)

__all__ = [
    "DOCUMENT_REASON_CHOICES",
    "DOCUMENT_REASON_NAMED_FILE",
    "DOCUMENT_REASON_NAMED_FUNCTION",
    "DOCUMENT_REASON_SEARCH",
    "CodeEntity",
    "CodeGraph",
    "CodeRelation",
    "CodeResult",
    "DocumentReason",
    "DocumentSelection",
    "HybridResult",
    "LexicalIndex",
    "LexicalResult",
    "PersistentIndexes",
    "RetrievalPlan",
    "SemanticIndex",
    "SemanticResult",
    "build_code_graph",
    "build_lexical_index",
    "build_semantic_index",
    "code_search",
    "dominant_path",
    "estimate_tokens",
    "expand_retrieval_context",
    "hybrid_search",
    "indexes_are_current",
    "lexical_search",
    "lexical_terms",
    "load_indexes",
    "named_paths",
    "prepare_persistent_indexes",
    "plan_retrieval",
    "read_document",
    "resolve_code_entities",
    "save_indexes",
    "semantic_matrix",
    "semantic_search",
]
