# QAChat

QAChat runs locally to answer questions about your code and literature,
with citations to your files and customizable project types, intents, and tools.

## Setup

After downloading the code, open a terminal in the QAChat folder and run
the setup command for your operating system.

macOS or Linux:

```sh
sh setup/setup.sh
```

Windows (PowerShell):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File setup/setup.ps1
```

For notebooks, select the project's
`.venv` Python kernel.

## Graphical User Interface

To open the desktop GUI, run the command:

```sh
uv run qachat --gui
```

The launcher installs missing GUI/backend dependencies; follow the setup
panel to download the required models.

## Tests and formatting

Run these commands from the QAChat folder after setup. `--no-sync` keeps
the installed backend dependencies unchanged.

Run all unit tests:

```sh
uv run --no-sync python -m tests
```

Format Python code with Black, then check it with Ruff:

```sh
uv run --no-sync black qachat setup tests
uv run --no-sync ruff check qachat setup tests
```

To check formatting without changing files, add `--check` to the Black
command. Format only the files you changed when preserving existing style.

## Conversation persistence and exclusive ownership

Each active conversation has a single writable `conversation_*.json` record.
QAChat holds an OS-level advisory lock on a stable `.json.lock` companion
while an active trace or tracing-disabled RAG session owns that file. A second
writer targeting the same conversation raises `ConversationLockedError` and
leaves the saved JSON unchanged. Locks are released on close and by the
operating system after process termination; the companion files can remain.
This is not a substitute for database transactions or a disk-durability
protocol. For behavior during switching, archival, and GUI metadata updates,
see [conversation-file ownership](docs/conversation_ownership.md).

## Tutorials

Follow these notebooks in order, or open the one you need.

- [10 · Setup](tutorials/10_setup.ipynb): Check the environment and computation backend without loading a model.
- [20 · Hugging Face models](tutorials/20_huggingface_models.ipynb): Inspect the model cache and optionally download a chat model.
- [30 · Model chat](tutorials/30_model_chat.ipynb): Load a model, exchange messages, and release it.
- [40 · Repository files](tutorials/40_repository_files.ipynb): Select the files QAChat will read from a repository.
- [50 · Source structure](tutorials/50_source_structure.ipynb): Read source files and split their structural units into searchable chunks.
- [60 · Embeddings](tutorials/60_embeddings.ipynb): Convert chunks and questions into vectors with an embedding model.
- [70 · Semantic retrieval](tutorials/70_semantic_retrieval.ipynb): Rank chunks by similarity in meaning.
- [80 · Lexical retrieval](tutorials/80_lexical_retrieval.ipynb): Rank chunks by matching words and identifiers without an embedding model.
- [90 · Code-aware retrieval](tutorials/90_code_aware_retrieval.ipynb): Find code through symbols and relationships such as calls and imports.
- [100 · Persistent indexes](tutorials/100_persistent_indexes.ipynb): Save retrieval indexes and reuse them when their inputs are unchanged.
- [110 · Hybrid ranking](tutorials/110_hybrid_ranking.ipynb): Combine semantic, lexical, and code-aware rankings.
- [120 · Retrieval expansion](tutorials/120_retrieval_routing.ipynb): Add context for explicitly named files, functions, and code relationships.
- [130 · Project chat](tutorials/130_project_chat.ipynb): Prepare or resume a project and ask questions with citations and saved history.
- [140 · Execution traces](tutorials/140_logging_trace.ipynb): Inspect the recorded inputs, outputs, retrieval steps, and timings of a session.
- [150 · Project types](tutorials/150_project_types.ipynb): Customize project types, instructions, and intents using JSON configuration.
- [160 · Custom tools](tutorials/160_create_custom_tool.ipynb): Explore a paper-summary tool and build a custom tool.
