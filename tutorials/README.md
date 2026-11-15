
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
Tutorials 140 and 150 use the download setting saved in each project
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
- [130 · Source cards](./130_source_cards.ipynb): Inspect scientific-paper cards, provenance, independent persistence, and four-channel source-first retrieval without loading a model.
- [140 · Project chat](./140_project_chat.ipynb): Prepare or resume a project and ask questions with citations and saved history.
- [150 · Execution traces](./150_logging_trace.ipynb): Inspect the recorded inputs, outputs, retrieval steps, and timings of a session.
- [160 · Project types](./160_project_types.ipynb): Customize project types, instructions, and intents using JSON configuration.
- [170 · Conversation memory](./170_conversation_memory.ipynb): Explore project-local indexing, Save/Like/Dislike, exclusion, branch origins, and verified historical recall using a model-free example.
- [180 · Custom tools](./180_create_custom_tool.ipynb): Explore a paper-summary tool and build a custom tool.
- [190 · Research-assistant capstone](./190_end_to_end_research_assistant.ipynb): Create a literature project from bundled example notes, evaluate offline retrieval and citations, and optionally run a complete RAG session.

## Choose a learning path

| Goal | Recommended notebooks | Model needed? |
|---|---|---|
| Verify setup and inspect model availability | [10](./10_setup.ipynb), [20](./20_huggingface_models.ipynb) | No model loaded |
| Learn local file processing and retrieval without a language model | [40](./40_repository_files.ipynb), [50](./50_source_structure.ipynb), [80](./80_lexical_retrieval.ipynb), [90](./90_code_aware_retrieval.ipynb) | No |
| Experiment with a local chatbot | [30](./30_model_chat.ipynb) | Chat |
| Understand semantic retrieval and persistent indexes | [60](./60_embeddings.ipynb), [70](./70_semantic_retrieval.ipynb), [100](./100_persistent_indexes.ipynb) | Embedding |
| Combine evidence and examine retrieval guarantees | [110](./110_hybrid_ranking.ipynb), [120](./120_context_expansion.ipynb) | Embedding |
| Inspect source-card indexing and source-first retrieval | [130](./130_source_cards.ipynb) | No |
| Use the complete project-question-answering workflow | [140](./140_project_chat.ipynb), [150](./150_logging_trace.ipynb) | Chat and embedding |
| Configure project behavior and conversation memory | [160](./160_project_types.ipynb), [170](./170_conversation_memory.ipynb) | No for the memory exercise (deterministic stand-in vectors) |
| Design tools for research workflows | [180](./180_create_custom_tool.ipynb) | No for the main example; real-model section optional |
| Apply and verify the complete research workflow | [190](./190_end_to_end_research_assistant.ipynb) | No for the core checks; chat and embedding models optional |

If you only want an introduction to KíLín as an end user, start with 10,
20, and 140, then try the integrated example in 190. The intermediate
notebooks explain the internals; they are not prerequisites for the GUI
or project chat.

## Where the examples get their data

Most tutorials work with the included `kilin/` Python package. The source
code is itself a realistic, reproducible example repository.
Tutorials 150 and 180 use literature PDFs:

- The complete supplementary article corpus, if available, lives under
  `_si_article_/literature_repo/`, with PDF files under `papers/`.
- Populated `_si_article_/code_repo/` and `_si_article_/literature_repo/`
  are intentionally absent from space-saving source ZIP exports. Their
  contents remain part of the full supplementary materials, not mandatory
  dependencies of the tutorial series.
- When a paper is absent from that corpus, Tutorials 150 and 180 use the
  small PDFs provided in `tests/fixtures/`. They tell you which file was
  selected. If neither location exists, the notebook raises an informative
  `FileNotFoundError` instead of silently choosing unrelated documents.

Tutorial 170 uses fictional saved conversation turns and a deterministic
stand-in embedding model; it does not download weights or call an LLM.
Tutorial 190 instead uses four small, fictional Markdown research notes
from `tutorials/data/190_research_pilot/`. Its deterministic steps work
without a language model; the full RAG run is explicitly opt-in. The
fictional measurements must not be cited as research findings.

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
original source directory. Tutorials 150, 170, 180, and 190 automatically
choose a new numbered workspace if an earlier run exists; they do not
recursively delete an earlier one. The other tutorials may write to documented paths below `sandbox/`; inspect these paths before changing them. Do not add
`sandbox/`, Hugging Face model caches, or source PDFs to version control.

Traces may contain prompts, retrieved passages, model output, paths, and
other sensitive information. A trace file may also contain the resumable
conversation record. Keep workspaces private and review the
[trace-recording guide](../docs/trace_recording.md) before deleting or
sharing them. Disable detailed tracing in Tutorial 140 if not needed.

## After a tutorial

Every notebook closes model sessions used by its example. It is safe to
restart the notebook kernel to release remaining memory. Saved indexes,
project files, generated reports, and cached model weights are separate
from the live model session. Deleting an example workspace discards its
saved outputs and any conversation history in it; do so only intentionally.

The [developer documentation](../docs/README.md) explains the core
architecture, configuration precedence, project data ownership, retrieval,
and extension interfaces in greater detail.
