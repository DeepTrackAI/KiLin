# QAChat

QAChat answers questions about code and literature contained in a local
repository using a local LLM that ensures no data are shared outsde the local
enviroment.

## 1. Install once

**Do this once on each computer.** To install QAChat, open a terminal in the
QAChat repository and run the appropriate command for your operative system.

Mac or Linux:

```sh
sh setup/setup.sh
```

Windows x64 PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File setup/setup.ps1
```

When the installation finishes, you will see:

```text
Setup complete. The project environment is in .venv
```

Select the QAChat `.venv` kernel when opening a notebook.

To launch the optional desktop GUI from the repository, run:

```
uv run qachat --gui
```

That command creates or refreshes the uv environment when needed.
The launcher adds the PySide6 Qt GUI layer and native computation backend
automatically: MLX on Apple silicon, or PyTorch elsewhere. If the .venv is
already active, use `qachat --gui` directly. Use `--backend mlx` or `--backend
torch` to override the automatic settings.

Models are downloaded and stored in the Hugging Face cache.

## 2. Chat with your repository

Open the [Repository Q&A Chat](tutorials/130_project_chat.ipynb).
In its configuration cell:

- Set `PROJECT_KIND` to `"code"` or `"literature"`.
- Set `PROJECT_DIRECTORY` to your original source folder.
- Leave `PROJECT_DIRECTORY = None` to use `example_repos/code_repo` or
  `example_repos/literature_repo`.

Every source, including the bundled examples, is copied into a workspace in
`sandbox/`; the original folder is never modified. For example,
`example_repos/code_repo` becomes:

```text
sandbox/
└── project_code_repo/
    ├── code_repo/
    ├── indexes/
    ├── logs/
    ├── profile.ini
    └── source_snapshot.json
```

On each open, QAChat compares SHA-256 manifests and synchronizes only added,
removed, or modified selected source files. The retrieval index is reused only
when both `indexes/source_manifest.json` and the saved indexing settings still
match. Otherwise QAChat rebuilds the index automatically.

You can also check source changes directly from Python at any time:

```python
from qachat import detect_repository_changes, repository_has_changed

changes = detect_repository_changes(
    "path/to/project",
    "sandbox/project_name/indexes/source_manifest.json",
)

print(changes)
# {"matches": False, "added": [...], "removed": [...], "modified": [...]}

if repository_has_changed(
    "path/to/project",
    "sandbox/project_name/indexes/source_manifest.json",
):
    print("Project changed")
```

Pass the same extensions/include/exclude settings used by the project profile
when you have customized repository selection.

`RAGSession` also accepts an optional UI-agnostic progress callback. The same
callback API can drive notebook output and a GUI progress bar:

```python
from qachat import RAGSession

def on_progress(event):
    print(f"{event.percent:5.1f}%  {event.message}")

session = RAGSession(project.root, progress_callback=on_progress)
```

Each event includes `stage`, `percent`, `message`, and optional `current`,
`total`, `remaining`, `elapsed_seconds`, and `eta_seconds` values. Long stages
such as reading, source extraction, chunking, and embedding emit repeated
updates so notebook or GUI clients can show live counts and an ETA. Embedding
uses small batches during index construction so progress can be reported
between batches.

### Inspect one complete execution trace

QAChat writes an ordered JSON trace for project setup, model and backend
selection, file scanning, chunking, indexing, retrieval ranks and
context-expansion decisions, effective per-turn settings, exact model input
and output, input/output token counts, per-call timings, and failures. Every
event has a UTC timestamp and sequence number. Exact prompts, evidence,
outputs, and checkpoints remain inline so the file is portable and
self-contained.

```python
from qachat import RAGSession, TraceConfig

session = RAGSession(project.root)
print(session.trace.readable_path)
```

Each conversation owns one file below the project's `logs/traces` folder:
`conversation_<timestamp>.json`. It contains identity, name, branch lineage,
resumable messages and turns, checkpoints, and the named hierarchical trace.
There is no separate catalog, JSONL stream, conversation-state file, or
artifact directory. Discovery, switching, renaming, branching, and resuming
scan these files directly. Preparation progress remains compact and sampled.
Effective setting snapshots are accompanied by change events that show each
changed field's `before` and `after` values. Use `mode="standard"` for
major operations, decisions, hashes, ranks, token totals, and timings without
duplicate token-detail events or exact retrieved source and rendered model
prompt payloads. Standard mode still stores the conversation state required
by resume checkpoints. The default is `mode="forensic"`, which also keeps
exact retrieved text, rendered prompts, token inputs, and low-level token
events.

Nested work is connected through `span_id` and `parent_span_id`, for example
one turn containing retrieval, query embedding, context expansion, and model
generation spans. In the readable trace, `operation_duration_ms` is the
duration of its named operation, not time since the previous event.

After each saved turn or effective-setting change, QAChat emits
`state.checkpoint.committed` and records a resumable checkpoint inline. Its
content-addressed `checkpoint_id` covers the conversation, effective settings,
model identities, project paths, and index fingerprint. The next checkpoint
records `parent_checkpoint_id`, so lineage continues after reopening or
branching a conversation.

To customize or disable tracing, pass a configuration explicitly:

```python
session = RAGSession(
    project.root,
    trace=TraceConfig(enabled=False),
)
```

The desktop GUI creates the conversation before preparation, so project setup,
indexing, and subsequent turns are recorded in the same file.

## 3. Understand how QAChat works

These standalone tutorials explain QAChat one layer at a time. Use them when
you want to understand the concepts and code clearly.

1. [Setup check](tutorials/10_setup.ipynb) — verify MLX or PyTorch.
2. [Hugging Face models](tutorials/20_huggingface_models.ipynb) — find or
   download models.
3. [Model chat](tutorials/30_model_chat.ipynb) — load a model and chat.
4. [Repository files](tutorials/40_repository_files.ipynb) — choose files.
5. [Source chunks](tutorials/50_source_structure.ipynb) — read and split
   source.
6. [Embeddings](tutorials/60_embeddings.ipynb) — turn text into vectors.
7. [Semantic retrieval](tutorials/70_semantic_retrieval.ipynb) — search by
   meaning.
8. [Lexical retrieval](tutorials/80_lexical_retrieval.ipynb) — search by
   exact words and identifiers.
9. [Code-aware retrieval](tutorials/90_code_aware_retrieval.ipynb) — follow
   Python and MATLAB relationships.
10. [Persistent indexes](tutorials/100_persistent_indexes.ipynb) — save and
    reuse indexes.
11. [Hybrid ranking](tutorials/110_hybrid_ranking.ipynb) — combine retrieval
    rankings.
12. [Retrieval routing](tutorials/120_retrieval_routing.ipynb) — include
    explicitly requested code.
13. [Project chat](tutorials/130_project_chat.ipynb) — use the complete RAG
    workflow.
14. [Execution trace](tutorials/140_logging_trace.ipynb) — follow project
    preparation, indexing, retrieval, exact model input, and generation one
    logged operation at a time.

## 4. Development checks

Run all tests:

```sh
.venv/bin/python -m tests
```

Format and check the code, then clear notebook output:

```sh
.venv/bin/black qachat setup tests
.venv/bin/ruff check --fix qachat setup tests
.venv/bin/jupyter nbconvert --clear-output --inplace tutorials/*.ipynb
```

On Windows, replace `.venv/bin/` with `.venv\Scripts\`.
