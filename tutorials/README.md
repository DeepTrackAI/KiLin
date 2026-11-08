
# KíLín tutorials: start here

These notebooks explain the local KíLín assistant from model setup to
retrieval-augmented chat and extensions. They are intended for a source
checkout: they import the local `kilin/` package and the repository's
`setup/notebook.py` helper. Each notebook can be opened independently.

## Before opening a notebook

1. Follow the operating-system setup instructions in the [project README](../README.md).
   Use Python 3.12, as required by `pyproject.toml`.
2. Open the notebook in Jupyter from the project checkout. Select the Python
   interpreter in the checkout's `.venv` as its kernel. The initial setup cells
   check the environment; if they request a kernel switch, switch and rerun.
3. Start with [Tutorial 10](./10_setup.ipynb) to check the computing backend.
   Choose `auto`, `mlx` (Apple Silicon), or `torch` where supported.
4. Cache chat and embedding models deliberately. Model IDs depend on the
   selected backend and may require substantial disk space and memory.
   Tutorial 20 explains inspecting and downloading Hugging Face models.

The `prepare_notebook()` helper may invoke `uv sync` or a platform
setup script to repair missing Python dependencies. That step can
require network access even when model downloads are disabled. Install
dependencies before working offline and review setup commands if
automatic package installation is not acceptable.

The notebooks that load models now use `ALLOW_DOWNLOAD = False` by default.
Set it to `True` only when you intend to access the network and download
missing weights. The Hugging Face cache is shared outside the checkout.
An already-cached model may still require a substantial first load.
Tutorials 130 and 140 use the download setting saved in each project
profile: newly created tutorial projects disable downloads, while
resuming a project preserves its existing setting.

## Tutorials

- [10 · Setup](./10_setup.ipynb): Check the environment and computation backend without loading a model.
- [20 · Hugging Face models](./20_huggingface_models.ipynb): Inspect the model cache and optionally download a chat model.
- [30 · Model chat](./30_model_chat.ipynb): Load a model, exchange messages, and release it.
- [40 · Repository files](./40_repository_files.ipynb): Select the files KíLín will read from a repository.
- [50 · Source structure](./50_source_structure.ipynb): Read source files and split their structural units into searchable chunks.
- [60 · Embeddings](./60_embeddings.ipynb): Convert chunks and questions into vectors with an embedding model.
- [70 · Semantic retrieval](./70_semantic_retrieval.ipynb): Rank chunks by similarity in meaning.
- [80 · Lexical retrieval](./80_lexical_retrieval.ipynb): Rank chunks by matching words and identifiers without an embedding model.
- [90 · Code-aware retrieval](./90_code_aware_retrieval.ipynb): Find code through symbols and relationships such as calls and imports.
- [100 · Persistent indexes](./100_persistent_indexes.ipynb): Save retrieval indexes and reuse them when their inputs are unchanged.
- [110 · Hybrid ranking](./110_hybrid_ranking.ipynb): Combine semantic, lexical, and code-aware rankings.
- [120 · Retrieval expansion](./120_context_expansion.ipynb): Add context for explicitly named files, functions, and code relationships.
- [130 · Project chat](./130_project_chat.ipynb): Prepare or resume a project and ask questions with citations and saved history.
- [140 · Execution traces](./140_logging_trace.ipynb): Inspect the recorded inputs, outputs, retrieval steps, and timings of a session.
- [150 · Project types](./150_project_types.ipynb): Customize project types, instructions, and intents using JSON configuration.
- [160 · Custom tools](./160_create_custom_tool.ipynb): Explore a paper-summary tool and build a custom tool.
- [170 · Source cards](./170_source_cards.ipynb): Inspect scientific-paper cards, provenance, independent persistence, and four-channel source-first retrieval without loading a model.

## Choose a learning path

| Goal | Recommended notebooks | Model needed? |
|---|---|---|
| Verify setup and inspect model availability | [10](./10_setup.ipynb), [20](./20_huggingface_models.ipynb) | No model loaded |
| Learn local file processing and retrieval without a language model | [40](./40_repository_files.ipynb), [50](./50_source_structure.ipynb), [80](./80_lexical_retrieval.ipynb), [90](./90_code_aware_retrieval.ipynb) | No |
| Experiment with a local chatbot | [30](./30_model_chat.ipynb) | Chat |
| Understand semantic retrieval and persistent indexes | [60](./60_embeddings.ipynb), [70](./70_semantic_retrieval.ipynb), [100](./100_persistent_indexes.ipynb) | Embedding |
| Combine evidence and examine retrieval guarantees | [110](./110_hybrid_ranking.ipynb), [120](./120_context_expansion.ipynb) | Embedding |
| Use the complete project-question-answering workflow | [130](./130_project_chat.ipynb), [140](./140_logging_trace.ipynb) | Chat and embedding |
| Design project-specific behavior and tools | [150](./150_project_types.ipynb), [160](./160_create_custom_tool.ipynb) | No for the main examples; real-model sections are optional |
| Inspect source-card indexing and source-first retrieval | [170](./170_source_cards.ipynb) | No |

If you only want an introduction to KíLín as an end user, start with 10,
20, and 130. The intermediate notebooks are explanations of the internals,
not prerequisites for using the GUI or project chat.

## Where the examples get their data

Most tutorials work with the included `kilin/` Python package. The source
code is itself a realistic, reproducible example repository. Tutorials 140
and 160 use literature PDFs:

- The complete supplementary article corpus, if available, lives under
  `_si_article_/literature_repo/`, with PDF files under `papers/`.
- Populated `_si_article_/code_repo/` and `_si_article_/literature_repo/`
  are intentionally absent from space-saving source ZIP exports. Their
  contents remain part of the full supplementary materials, not mandatory
  dependencies of the tutorial series.
- When a paper is absent from that corpus, Tutorials 140 and 160 use the
  small PDFs provided in `tests/fixtures/`. They tell you which file was
  selected. If neither location exists, the notebook raises an informative
  `FileNotFoundError` instead of silently choosing unrelated documents.

For a real project, replace the example source directory with your own
existing code or literature folder. The original source files are input;
KíLín's managed state is written to a separate project workspace.

## Runtime cost and data safety

Embedding every chunk of a large repository can be slow. The early semantic
example works with a few source files; the advanced retrieval examples
start from the retrieval package and `rag.py`. Broaden `include_paths`
after checking the smaller examples. Some notebooks use embedding batches
to avoid processing the entire corpus in one model call.

Example workspaces and indexes are created below `sandbox/`, not in the
original source directory. Tutorials 140 and 160 automatically choose a
new numbered workspace if an earlier run exists; they do not recursively
delete an earlier one. The other tutorials may write to documented paths
below `sandbox/`; inspect these paths before changing them. Do not add
`sandbox/`, Hugging Face model caches, or source PDFs to version control.

Traces may contain prompts, retrieved passages, model output, paths, and
other sensitive information. A trace file may also contain the resumable
conversation record. Keep workspaces private and review the
[trace-recording guide](../docs/trace_recording.md) before deleting or
sharing them. Disable detailed tracing in Tutorial 130 if not needed.

## After a tutorial

Every notebook closes model sessions used by its example. It is safe to
restart the notebook kernel to release remaining memory. Saved indexes,
project files, generated reports, and cached model weights are separate
from the live model session. Deleting an example workspace discards its
saved outputs and any conversation history in it; do so only intentionally.

The [developer documentation](../docs/README.md) explains the core
architecture, configuration precedence, project data ownership, retrieval,
and extension interfaces in greater detail.
