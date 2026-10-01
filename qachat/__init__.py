"""Expose the package-level QAChat API.

This initializer re-exports the conversation, source-processing, retrieval,
project, workspace, and tracing objects listed in ``__all__``. Examples in
QAChat documentation use this namespace when an object is exported here.
Objects exported only by a submodule must be accessed through that submodule.

Importing this package does not construct a model session or start the GUI.
Model sessions load their backend dependencies when instantiated; the GUI
has a separate entry point. Imported submodules may nevertheless require
non-model dependencies such as NumPy.

Key Features
------------

- **Public facade**

    Collect the existing public classes, constants, and functions without
    wrapping or changing their behavior.

- **Separate lifecycles**

    Keep model construction and GUI launch as explicit operations performed by
    callers.

Module Structure
----------------

The public names are listed explicitly in ``__all__``. They are grouped
by implementation module in the import section below. This initializer does
not export every public object from every submodule.

"""

from .chat import ASSISTANT_ROLE, SYSTEM_ROLE, USER_ROLE, ChatSession
from .chunking import Chunk, chunk_source_units
from .embeddings import EmbeddingSession
from .huggingface import (
    available_models,
    cached_model_snapshot,
    download_model,
    model_is_available,
)
from .query_formats import (
    LITERATURE_REPOSITORY_INSTRUCTION,
    SOFTWARE_REPOSITORY_INSTRUCTION,
    format_qwen3_query,
)
from .project import (
    PROJECT_KIND_CHOICES,
    PROJECT_KIND_CODE,
    PROJECT_KIND_LITERATURE,
    ProjectProfile,
    QAProject,
    initialize_project,
    open_project,
    prepare_project_workspace,
)
from .rag import (
    ConversationTurn,
    Evidence,
    ProgressCallback,
    ProgressEvent,
    RAGSession,
)
from .repository import (
    build_repository_manifest,
    compare_repository_manifests,
    detect_repository_changes,
    repository_has_changed,
    scan_repository,
)
from .retrievals import (
    CodeEntity,
    CodeGraph,
    CodeRelation,
    CodeResult,
    HybridResult,
    LexicalIndex,
    LexicalResult,
    PersistentIndexes,
    RetrievalPlan,
    SemanticIndex,
    SemanticResult,
    build_code_graph,
    build_lexical_index,
    build_semantic_index,
    code_search,
    expand_retrieval_context,
    hybrid_search,
    indexes_are_current,
    lexical_search,
    lexical_terms,
    load_indexes,
    plan_retrieval,
    prepare_persistent_indexes,
    resolve_code_entities,
    save_indexes,
    semantic_matrix,
    semantic_search,
)
from .runtime import Backend, SelectedBackend, choose_backend, torch_device
from .sources import (
    SourceFile,
    SourceUnit,
    extract_source_units,
    read_repository_files,
)
from .trace_logging import (
    NullTraceLogger,
    TraceConfig,
    TraceLogger,
    TraceMode,
    TraceSink,
)
from .workspace import prepare_sources_workspace

__all__ = [
    "ASSISTANT_ROLE",
    "Backend",
    "ChatSession",
    "Chunk",
    "CodeEntity",
    "CodeGraph",
    "CodeRelation",
    "CodeResult",
    "ConversationTurn",
    "EmbeddingSession",
    "Evidence",
    "HybridResult",
    "LITERATURE_REPOSITORY_INSTRUCTION",
    "LexicalIndex",
    "LexicalResult",
    "PersistentIndexes",
    "PROJECT_KIND_CHOICES",
    "PROJECT_KIND_CODE",
    "PROJECT_KIND_LITERATURE",
    "ProjectProfile",
    "ProgressCallback",
    "ProgressEvent",
    "QAProject",
    "RAGSession",
    "RetrievalPlan",
    "SOFTWARE_REPOSITORY_INSTRUCTION",
    "SYSTEM_ROLE",
    "SelectedBackend",
    "SemanticIndex",
    "SemanticResult",
    "SourceFile",
    "SourceUnit",
    "TraceConfig",
    "TraceLogger",
    "TraceMode",
    "TraceSink",
    "USER_ROLE",
    "NullTraceLogger",
    "available_models",
    "build_code_graph",
    "build_repository_manifest",
    "build_lexical_index",
    "build_semantic_index",
    "cached_model_snapshot",
    "choose_backend",
    "chunk_source_units",
    "code_search",
    "compare_repository_manifests",
    "detect_repository_changes",
    "download_model",
    "expand_retrieval_context",
    "extract_source_units",
    "format_qwen3_query",
    "hybrid_search",
    "initialize_project",
    "indexes_are_current",
    "lexical_search",
    "lexical_terms",
    "load_indexes",
    "model_is_available",
    "open_project",
    "plan_retrieval",
    "prepare_persistent_indexes",
    "prepare_project_workspace",
    "prepare_sources_workspace",
    "read_repository_files",
    "repository_has_changed",
    "resolve_code_entities",
    "save_indexes",
    "scan_repository",
    "semantic_matrix",
    "semantic_search",
    "torch_device",
]
