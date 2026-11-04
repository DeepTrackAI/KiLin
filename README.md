# KíLín

KíLín runs locally to answer questions about your code and literature,
with citations to your files and customizable project types, intents, and tools.

## Citation

Please cite KíLín using the metadata in [CITATION.cff](CITATION.cff).
The citation lists Io̍k-uí Tiunn, Joana B. Pereira, Carlo Manzo, and
Giovanni Volpe as authors.

## Setup

After downloading the code, open a terminal in the KíLín folder and run
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
uv run kilin --gui
```

The launcher installs missing GUI/backend dependencies; follow the setup
panel to download the required models.

## Tests and formatting

Run these commands from the KíLín folder after setup. `--no-sync` keeps
the installed backend dependencies unchanged.

Run all unit tests:

```sh
uv run --no-sync python -m tests
```

Format Python code with Black, then check it with Ruff:

```sh
uv run --no-sync black kilin setup tests
uv run --no-sync ruff check kilin setup tests
```

To check formatting without changing files, add `--check` to the Black
command. Format only the files you changed when preserving existing style.

## Diagnostic tracing

KíLín records complete diagnostic events for enabled tracing, including
available prompts, generated answers, retrieved content, token IDs, errors,
and local paths. Embedding vectors are opt-in because they can be large.
There are no selectable trace-detail modes. `RAGSession(trace=None)` enables
tracing, while `TraceConfig(enabled=False)` disables diagnostic events but
retains the resumable conversation checkpoint. Both are written to the same
JSON file when tracing is enabled. Deleting that file also removes the saved
conversation. Protect local workspaces and do not share raw trace files as
sanitized reports. See [trace recording](docs/trace_recording.md).

## Conversation persistence and exclusive ownership

Each active conversation has a single writable `conversation_*.json` record.
KíLín holds an OS-level advisory lock on a stable `.json.lock` companion
while an active trace or tracing-disabled RAG session owns that file. A second
writer targeting the same conversation raises `ConversationLockedError` and
leaves the saved JSON unchanged. Locks are released on close and by the
operating system after process termination; the companion files can remain.
Conversation JSON and related local metadata use unique temporary files and
atomic replacements, avoiding predictable temporary-file collisions. Active
GUI metadata changes are written once through the owning trace, rather than
through two independent snapshots. These guarantees do not replace database
transactions or a disk-durability protocol. For switching, archival, error
recovery, and GUI metadata updates, see
[conversation-file ownership](docs/conversation_ownership.md).

## Tutorials

KíLín includes 16 notebooks introducing the framework, from local models
and source processing to retrieval, project chat, diagnostic traces, project
types, and custom tools.

Start with the [tutorial guide](tutorials/README.md) for prerequisites,
recommended learning paths, model requirements, example datasets, and safe
handling of generated workspaces.

For a first introduction, we recommend:

1. [10 · Setup](tutorials/10_setup.ipynb): Check your environment and
   computational backend without loading a model.
2. [20 · Hugging Face models](tutorials/20_huggingface_models.ipynb): Inspect
   locally cached models and download models explicitly when needed.
3. [130 · Project chat](tutorials/130_project_chat.ipynb): Create a project
   and ask questions about its files with citations and saved history.
   This example requires local chat and embedding models.

The other notebooks explain individual components in more depth; they are
not prerequisites for using the graphical interface. Most examples use
source files distributed with KíLín. Tutorials involving literature can
optionally use the supplementary article corpus in
`_si_article_/literature_repo/` and provide bundled PDF fallbacks when
that corpus is unavailable.
