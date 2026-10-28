"""Expose the package-level KíLín API.

This initializer re-exports the conversation, source-processing, retrieval,
project, workspace, and tracing objects listed in ``__all__``. Examples in
KíLín documentation use this namespace when an object is exported here.
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

Examples
--------
>>> import kilin
>>> kilin.PROJECT_TYPE_ID_CODE
'code'

"""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _distribution_version

try:
    __version__ = _distribution_version("kilin")
except _PackageNotFoundError:
    # Uninstalled source checkouts use the same version from pyproject.toml.
    from pathlib import Path as _Path
    import tomllib as _tomllib

    with (_Path(__file__).resolve().parents[1] / "pyproject.toml").open(
        "rb"
    ) as _stream:
        __version__ = _tomllib.load(_stream)["project"]["version"]

from .chat import ASSISTANT_ROLE, SYSTEM_ROLE, USER_ROLE, ChatSession
from .chunking import Chunk, chunk_source_units
from .embeddings import EmbeddingSession
from .model_defaults import (
    DEFAULT_CHAT_MODEL_MLX,
    DEFAULT_CHAT_MODEL_TORCH,
    DEFAULT_EMBEDDING_MODEL_MLX,
    DEFAULT_EMBEDDING_MODEL_TORCH,
    MODEL_ROLE_CHAT,
    MODEL_ROLE_CHOICES,
    MODEL_ROLE_EMBEDDING,
    ModelRole,
)
from .huggingface import (
    available_models,
    cached_model_snapshot,
    download_model,
    model_is_available,
)
from .query_formats import (
    QUERY_FORMAT_CHOICES,
    QUERY_FORMAT_PLAIN,
    QUERY_FORMAT_QWEN3,
    QueryFormat,
    format_qwen3_query,
)
from .project import (
    PROJECT_TYPE_IDS,
    PROJECT_TYPE_ID_CODE,
    PROJECT_TYPE_ID_LITERATURE,
    ProjectTypeId,
    ProjectProfile,
    QAProject,
    initialize_project,
    ProjectAlreadyExistsError,
    WorkspaceOccupiedError,
    synchronize_project_workspace,
    open_project,
    prepare_project_workspace,
)
from .answering import (
    MACHINE_STATE_ANSWERING,
    MACHINE_STATE_CHOICES,
    MachineState,
)
from .project_type import (
    HISTORY_POLICY_CHOICES,
    HISTORY_POLICY_NONE,
    HISTORY_POLICY_RECENT,
    HISTORY_POLICY_RELEVANT,
    DOCUMENT_MAX_TOKENS,
    RETRIEVAL_MODE_CHOICES,
    RETRIEVAL_MODE_DOCUMENT,
    RETRIEVAL_MODE_NONE,
    RETRIEVAL_MODE_REUSE,
    RETRIEVAL_MODE_SEARCH,
    RETRIEVAL_STRATEGY_CHOICES,
    RETRIEVAL_STRATEGY_HYBRID,
    HistoryPolicy,
    Intent,
    ProjectType,
    RetrievalMode,
    RetrievalStrategy,
    load_project_types,
    project_type,
)
from .tools import (
    CONFIRMATION_ALWAYS,
    CONFIRMATION_CHOICES,
    CONFIRMATION_NEVER,
    CONFIRMATION_WHEN_LARGE,
    PLAN_CHOICE_CANCEL,
    PLAN_CHOICE_CHOICES,
    PLAN_CHOICE_CONTINUE,
    PLAN_CHOICE_QUICK,
    TOOL_BATCH_CLASSIFY,
    TOOL_CHOICES,
    TOOL_SCAN_AND_LIST,
    ConfirmationPolicy,
    PlanChoice,
    ToolContext,
    ToolName,
    ToolPlan,
    ToolRequest,
    ToolResult,
    ToolUse,
)
from .rag import (
    ANSWER_PHASE_CHOICES,
    ANSWER_PHASE_COMPLETED,
    ANSWER_PHASE_EVIDENCE_READY,
    ANSWER_PHASE_GENERATING,
    ANSWER_PHASE_PROCESSING,
    ANSWER_PHASE_RETRIEVING,
    ANSWER_PHASE_ROUTED,
    ANSWER_PHASE_ROUTING,
    ANSWER_PHASE_SAVING,
    AnswerPhase,
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
    DOCUMENT_REASON_CHOICES,
    DOCUMENT_REASON_NAMED_FILE,
    DOCUMENT_REASON_NAMED_FUNCTION,
    DOCUMENT_REASON_SEARCH,
    CodeEntity,
    CodeGraph,
    CodeRelation,
    CodeResult,
    DocumentReason,
    DocumentSelection,
    HybridResult,
    LexicalIndex,
    LexicalResult,
    PersistentIndexes,
    ContextExpansionPlan,
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
    plan_context_expansion,
    prepare_persistent_indexes,
    read_document,
    resolve_code_entities,
    save_indexes,
    semantic_matrix,
    semantic_search,
)
from .runtime import (
    BACKEND_AUTO,
    BACKEND_CHOICES,
    BACKEND_MLX,
    BACKEND_TORCH,
    Backend,
    SelectedBackend,
    choose_backend,
    torch_device,
)
from .sources import (
    SourceFile,
    SourceUnit,
    extract_source_units,
    read_repository_files,
)
from .trace_logging import (
    FAILURE_POLICY_CHOICES,
    FAILURE_POLICY_RAISE,
    FAILURE_POLICY_WARN,
    TRACE_EVENT_EMBEDDING_TOKENS_PREPARED,
    TRACE_EVENT_LLM_CALL_INPUT,
    TRACE_EVENT_LLM_CALL_OUTPUT,
    TRACE_EVENT_LLM_CALL_OUTPUT_TOKENS,
    TRACE_EVENT_PROGRESS_UPDATED,
    FailurePolicy,
    NullTraceLogger,
    TraceConfig,
    TraceLogger,
    TraceSink,
)
from .project_files import (
    add_project_files,
    list_project_files,
    remove_project_files,
    replace_project_file,
)
from .workspace import (
    SourceAction,
    SourceChanges,
    SourceVersion,
    add_project_sources,
    default_workspace_path,
    find_source_changes,
    list_source_versions,
    prepare_sources_workspace,
    remove_workspace_sources,
    sync_project_sources,
)

__all__ = [
    "__version__",
    "ANSWER_PHASE_CHOICES",
    "ANSWER_PHASE_COMPLETED",
    "ANSWER_PHASE_EVIDENCE_READY",
    "ANSWER_PHASE_GENERATING",
    "ANSWER_PHASE_PROCESSING",
    "ANSWER_PHASE_RETRIEVING",
    "ANSWER_PHASE_SAVING",
    "ASSISTANT_ROLE",
    "AnswerPhase",
    "BACKEND_AUTO",
    "BACKEND_CHOICES",
    "BACKEND_MLX",
    "BACKEND_TORCH",
    "Backend",
    "ChatSession",
    "Chunk",
    "CodeEntity",
    "CodeGraph",
    "CodeRelation",
    "CodeResult",
    "ConversationTurn",
    "DEFAULT_CHAT_MODEL_MLX",
    "DEFAULT_CHAT_MODEL_TORCH",
    "DEFAULT_EMBEDDING_MODEL_MLX",
    "DEFAULT_EMBEDDING_MODEL_TORCH",
    "EmbeddingSession",
    "Evidence",
    "FAILURE_POLICY_CHOICES",
    "FAILURE_POLICY_RAISE",
    "FAILURE_POLICY_WARN",
    "FailurePolicy",
    "HybridResult",
    "MODEL_ROLE_CHAT",
    "MODEL_ROLE_CHOICES",
    "MODEL_ROLE_EMBEDDING",
    "ModelRole",
    "LexicalIndex",
    "LexicalResult",
    "PersistentIndexes",
    "PROJECT_TYPE_IDS",
    "PROJECT_TYPE_ID_CODE",
    "PROJECT_TYPE_ID_LITERATURE",
    "ProjectTypeId",
    "QUERY_FORMAT_CHOICES",
    "QUERY_FORMAT_PLAIN",
    "QUERY_FORMAT_QWEN3",
    "QueryFormat",
    "ProjectProfile",
    "ProgressCallback",
    "ProgressEvent",
    "QAProject",
    "RAGSession",
    "ContextExpansionPlan",
    "SYSTEM_ROLE",
    "SelectedBackend",
    "SemanticIndex",
    "SemanticResult",
    "SourceFile",
    "SourceUnit",
    "TRACE_EVENT_EMBEDDING_TOKENS_PREPARED",
    "TRACE_EVENT_LLM_CALL_INPUT",
    "TRACE_EVENT_LLM_CALL_OUTPUT",
    "TRACE_EVENT_LLM_CALL_OUTPUT_TOKENS",
    "TRACE_EVENT_PROGRESS_UPDATED",
    "TraceConfig",
    "TraceLogger",
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
    "ProjectAlreadyExistsError",
    "WorkspaceOccupiedError",
    "synchronize_project_workspace",
    "add_project_sources",
    "default_workspace_path",
    "find_source_changes",
    "list_source_versions",
    "indexes_are_current",
    "lexical_search",
    "lexical_terms",
    "load_indexes",
    "model_is_available",
    "open_project",
    "plan_context_expansion",
    "read_document",
    "DocumentSelection",
    "DocumentReason",
    "DOCUMENT_REASON_CHOICES",
    "DOCUMENT_REASON_NAMED_FILE",
    "DOCUMENT_REASON_NAMED_FUNCTION",
    "DOCUMENT_REASON_SEARCH",
    "prepare_persistent_indexes",
    "prepare_project_workspace",
    "prepare_sources_workspace",
    "remove_workspace_sources",
    "sync_project_sources",
    "SourceAction",
    "SourceChanges",
    "SourceVersion",
    "add_project_files",
    "replace_project_file",
    "remove_project_files",
    "list_project_files",
    "read_repository_files",
    "repository_has_changed",
    "resolve_code_entities",
    "save_indexes",
    "scan_repository",
    "semantic_matrix",
    "semantic_search",
    "torch_device",
    "ANSWER_PHASE_ROUTED",
    "ANSWER_PHASE_ROUTING",
    "MACHINE_STATE_ANSWERING",
    "MACHINE_STATE_CHOICES",
    "MachineState",
    "DOCUMENT_MAX_TOKENS",
    "CONFIRMATION_ALWAYS",
    "CONFIRMATION_CHOICES",
    "CONFIRMATION_NEVER",
    "CONFIRMATION_WHEN_LARGE",
    "PLAN_CHOICE_CANCEL",
    "PLAN_CHOICE_CHOICES",
    "PLAN_CHOICE_CONTINUE",
    "PLAN_CHOICE_QUICK",
    "TOOL_BATCH_CLASSIFY",
    "TOOL_CHOICES",
    "TOOL_SCAN_AND_LIST",
    "ConfirmationPolicy",
    "PlanChoice",
    "ToolContext",
    "ToolName",
    "ToolPlan",
    "ToolRequest",
    "ToolResult",
    "ToolUse",
    "HISTORY_POLICY_CHOICES",
    "HISTORY_POLICY_NONE",
    "HISTORY_POLICY_RECENT",
    "HISTORY_POLICY_RELEVANT",
    "RETRIEVAL_MODE_CHOICES",
    "RETRIEVAL_MODE_DOCUMENT",
    "RETRIEVAL_MODE_NONE",
    "RETRIEVAL_MODE_REUSE",
    "RETRIEVAL_MODE_SEARCH",
    "RETRIEVAL_STRATEGY_CHOICES",
    "RETRIEVAL_STRATEGY_HYBRID",
    "HistoryPolicy",
    "Intent",
    "ProjectType",
    "RetrievalMode",
    "RetrievalStrategy",
    "load_project_types",
    "project_type",
]
