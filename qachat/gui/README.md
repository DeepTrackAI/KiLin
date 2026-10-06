# QAChat desktop interface

The desktop interface is a local Qt shell around QAChat's project, retrieval,
and model APIs. The web page owns presentation state; a separate Python
process owns filesystem access, indexing, model sessions, and conversation
storage. Closing or stopping that process is not the same as pausing a model.

## Launch and dependencies

With the optional GUI dependencies installed, the module entry point is:

```sh
python -m qachat.gui
```

The existing development instructions also use the console entry point:

```sh
uv run qachat --gui
```

The attached source snapshot does not contain the packaging configuration
that declares that console command or its dependency extras. Those commands
therefore assume the surrounding checkout and environment provide them.

`qachat.cli.main` checks dependency module discoverability and may invoke
`uv sync` in a development checkout before launching the GUI. This can modify
the environment and contact package repositories. Its `--backend` argument
controls dependency preparation; it does not override the project's backend
selection in the GUI. `python -m qachat.gui` delegates directly to the GUI
application and does not perform that synchronization step.

## Components and communication

`app.py` creates the Qt application and window. `window.py` hosts the local
page, exposes `ChatBridge` through QWebChannel, starts the engine subprocess,
and converts bridge calls into JSON-line commands. Engine responses are
relayed to JavaScript as JSON events. Unexpected diagnostic output is kept
separate from protocol messages where the bridge implements that handling.

`engine.py` handles commands synchronously, invokes QAChat's APIs, and emits
project, progress, answer, reference, and failure events. `model_catalog.py`
contains suggested model records and distribution-presence checks. A cache
entry or an installed distribution is not proof that a model can load or run.

`web/index.html` defines the setup form, conversation controls, workspace and
reference panels, activity display, and model dialog. Its identifiers are the
contract used by `web/app.js`. `web/styles.css` defines layout, interaction
states, and responsive breakpoints. JavaScript function comments describe
state changes and bridge calls.

The page's local-content access to remote URLs is disabled in the Qt view.
That setting does not prevent the Python engine from downloading models.
Answers and source excerpts are inserted as text rather than executed as
HTML. Numeric citations are made clickable only when they refer to evidence
records supplied with that answer.

## Source selection and workspace preparation

Select code or literature, then choose local files or directories through the
native dialog or drag-and-drop. The engine previews directory selections
using repository scanning. The tree shows selected and skipped entries;
protected entries remain unavailable. Oversized eligible files can be checked
explicitly, which increases the submitted size limit to accommodate them.
Additional UTF-8 suffixes can be entered in Settings.

The selected sources are copied into the workspace's `sources/` directory.
This multi-source route is cumulative: re-adding a source updates matching
files, but deletion or deselection at the origin does not automatically remove
old workspace copies. It differs from the single-source synchronization API
in `qachat.project`. The original source is not intentionally modified.

In an open project, only the sidebar's `sources` tree accepts dropped files
or folders. It highlights during a native file drag; other workspace areas
reject drops. A confirmation explains the copy and index update before any
import command is sent. Confirmed imports use the saved project filters,
update the source manifest and indexes, and restore the active conversation.
Right-clicking a file or folder inside `sources` (not `sources` itself or any
other workspace item) offers deletion. After confirmation the workspace copy
is deleted and the project is reopened, so the index drops the removed files;
originals outside the workspace and saved chats are untouched. Drops are disabled during a running operation. The dashed target and dimmed
surroundings appear only during dragging. During indexing, the composer,
Ask, attachment, effort, and Branch controls are disabled; the progress area
provides a separate Stop button. Existing draft text is retained.

When indexing settings and embedding-model snapshot match, added or modified
files alone are chunked and embedded. Unchanged chunks and vectors are reused;
removed files are excluded. Lexical statistics and the global code graph are
rebuilt so retrieval behavior matches a full rebuild. Changed embedding or
chunking settings, or an unusable cache, trigger a full rebuild. Older indexes
without embedding-snapshot metadata require one full rebuild.

A workspace also contains `profile.ini`, `indexes/`, and `logs/`. It can be
selected explicitly or chosen through the engine's default workspace route.
A saved workspace can be reopened without adding sources. The default
example-source route requires an example repository that is not included in
this attached archive. A workspace is a directory organization convention,
not an execution sandbox or a protection boundary for hostile inputs.

Settings configure the profile before indexing and model loading.
New GUI projects leave preparation downloads disabled unless explicitly
allowed. Literature defaults permit larger files than code defaults; the
profile and the current GUI fields remain the effective configuration.

## Project types

Project types live in `qachat/project_types.json`. Each entry sets the label
shown in the type menu, default file suffixes and size limit for new
projects, the retrieval instruction added to Qwen3-format embedding queries,
the system prompt for answers, the answer guidance the desktop app adds to
each question (for example the 150-300 word target), and the empty-chat
headings and suggestions. Add an entry to create a new type; `code` and
`literature` must stay. Restart QAChat after editing. Saved conversations
keep the system prompt they were answered with. Only `code` and
`literature` have bundled example repositories, so other types need sources.

## Models and generation settings

The window and Dock icon follow the Appearance setting: `icons/qachat-light.png`
for light, `icons/qachat-dark.png` for dark, and the operating system's color
scheme for System. Missing icon files leave Qt's default icon.

Settings are ordered Appearance, Local models, Answers, then Project. Clicking
the backdrop or Close applies model edits, with confirmation for downloads and
embedding changes; cancelling keeps the panel open.

Local models lists five effort levels. Each has an editable name, a language
model chosen from a dropdown of cached language models, and a reference
count (1-200). The composer's effort picker has two sliders: one chooses the
level whose model answers, the other the level whose reference count is
retrieved, so a small model can be paired with many references or the
reverse. The middle level's model is also the project's chat model. Cached
models are classified as language or embedding models by preset, by an
"embed" name or sentence-transformers configuration, then by architecture.
A selected model that is not downloaded shows a Download button. On PyTorch
every level uses one chat model; levels still set reference counts.

Levels that differ from the defaults are saved per backend in
effort_levels.json; with defaults, no file is kept. An older
model_choices.json is migrated on first read and removed on the next save.
Changing a language model does not rebuild the index; changing the embedding
model (also a dropdown of cached embedding models) or query format requires
confirmation and rebuilds it. Cached models that no level uses are listed
under Other cached models.

Selecting a cached model is not a completeness check. Offline loading,
backend compatibility, model architecture, tokenizer behavior, and pooling
assumptions are enforced or encountered later in the model-loading path.

The GUI allows up to 2,048 generated tokens; the backend profile default
stays 256, and GUI sessions using 256 are raised to 2,048 when opened. The
token cap is a safety limit, not a requested answer length. Model-reported
length stops are
saved with the turn and shown as an incomplete-answer notice. Old answers are
not regenerated automatically.

Answer-time controls set retrieval count, temperature, thinking, and maximum
new tokens without rebuilding the index or reloading models. A blank answer
limit requests the session's automatic budget; it does not remove the model's
finite context limit. The values used for a completed answer are recorded with
that turn. Reopening a saved conversation restores its saved settings when
available.

## Conversations, references, and progress

New chat creates a separate conversation record while retaining earlier
conversations in the picker. It does not archive and delete the previous GUI
conversation. Branch copies resumable parent state and records parent and
checkpoint identifiers; it does not copy the entire parent event history.
Renaming a conversation updates display metadata. The project name can be edited in the heading or floating top bar.
Renaming a project moves
its workspace folder, reloads the conversation registry, and restores the
selected chat. If reopening fails, QAChat attempts to restore the old folder. Changing conversation retains already loaded
models and indexes where the engine implements that reuse.

Select an answer to inspect its evidence. A citation requests a preview of
that evidence's source location through the bridge. Reference responses are
matched to the currently selected turn and citation before being displayed,
so a late response does not overwrite a newer selection.

The interface displays backend phases as they happen: loading a model,
identifying intent, retrieving or reusing references, and reading the
references before the first word streams. Reading time grows with the
conversation, because earlier turns' prompts stay in the chat history. Each label
stays at least 0.6 seconds and a newer phase replaces queued ones, so the
display is at most one short step behind. A timer runs from Ask.

Answers stream as they are generated: the engine forwards text a few times
per second and the page appends it. On MLX this is token by token; PyTorch
delivers the whole reply at once. With thinking enabled, the model's thinking
shows live in a grey box that folds into "Thought for Ns" when the answer
starts; saved answers keep a collapsed "Show thinking". Citation links are
added when the answer completes, under a short "Took Ns" line.

Follow-up questions: before searching, the model that will answer decides
whether the previous answer's references suffice (REUSE) or a new search is
needed (SEARCH), in which case it rewrites the question into a standalone
search query. Reused references skip retrieval; the rewritten query is used
only for searching, and the question is answered as asked. The decision is
traced as gui.intent.decided and saved with the turn (evidence_reused or
retrieval_query). It is skipped for first questions, after answers without
references, and for models under about 2B parameters, which search with the
question as asked. Unclear replies also fall back to searching.

Stop terminates the engine and starts a fresh backend process. Loaded models
and unsaved work are lost; previously completed writes remain on disk. Reopen
the project to continue. This is not transactional cancellation of a copy,
index write, download, or conversation update that was in progress.

## Trace privacy and current limits

The engine entry point enables forensic tracing. Traces and conversation
checkpoints can contain complete questions, answers, source excerpts, paths,
settings, token identifiers, and diagnostic output. Standard trace mode is
not a content-redaction policy: resumable conversation state may still contain
full text. Treat the entire workspace, especially `logs/`, as sensitive data.

Model changes emit gui.models.change.requested/confirmed/applied events;
downloads emit start/completion events. Source imports record the request
and copied files. index.incremental.plan records reused and embedded chunk
counts plus added, modified, and removed paths. Conversation turns retain the
model and generation settings actually used.
