"""Expose the public API of the QAChat package."""

from .chat import ChatSession
from .chunking import Chunk, chunk_source_units
from .embeddings import EmbeddingSession
from .huggingface import (
    available_models,
    download_model,
    model_is_available,
)
from .query_formats import (
    LITERATURE_REPOSITORY_INSTRUCTION,
    SOFTWARE_REPOSITORY_INSTRUCTION,
    format_qwen3_query,
)
from .repository import scan_repository
from .retrievals import (
    CodeEntity,
    CodeGraph,
    CodeRelation,
    CodeResult,
    HybridResult,
    LexicalIndex,
    LexicalResult,
    PersistentIndexes,
    SemanticIndex,
    SemanticResult,
    build_code_graph,
    build_lexical_index,
    build_semantic_index,
    code_search,
    hybrid_search,
    indexes_are_current,
    lexical_search,
    lexical_terms,
    load_indexes,
    prepare_persistent_indexes,
    resolve_code_entities,
    save_indexes,
    semantic_search,
)
from .runtime import Backend, SelectedBackend, choose_backend, torch_device
from .sources import (
    SourceFile,
    SourceUnit,
    extract_source_units,
    read_repository_files,
)


__all__ = [
    "Backend",
    "ChatSession",
    "Chunk",
    "CodeEntity",
    "CodeGraph",
    "CodeRelation",
    "CodeResult",
    "EmbeddingSession",
    "HybridResult",
    "LITERATURE_REPOSITORY_INSTRUCTION",
    "LexicalIndex",
    "LexicalResult",
    "PersistentIndexes",
    "SOFTWARE_REPOSITORY_INSTRUCTION",
    "SelectedBackend",
    "SemanticIndex",
    "SemanticResult",
    "SourceFile",
    "SourceUnit",
    "available_models",
    "build_code_graph",
    "build_lexical_index",
    "build_semantic_index",
    "choose_backend",
    "chunk_source_units",
    "code_search",
    "download_model",
    "extract_source_units",
    "format_qwen3_query",
    "hybrid_search",
    "indexes_are_current",
    "lexical_search",
    "lexical_terms",
    "load_indexes",
    "model_is_available",
    "prepare_persistent_indexes",
    "read_repository_files",
    "resolve_code_entities",
    "save_indexes",
    "scan_repository",
    "semantic_search",
    "torch_device",
]
