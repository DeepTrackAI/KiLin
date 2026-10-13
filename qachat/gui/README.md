# QAChat desktop interface

The desktop interface is a local Qt shell around QAChat's project, retrieval,
and model APIs. The web page owns presentation state; a separate Python
process owns filesystem access, indexing, model sessions, and conversation
storage. Closing or stopping that process is not the same as pausing a model.

## Launch and dependencies

With synchronization and automatic preparation of missing GUI/backend
dependencies, launch from the repository with:

```sh
uv run qachat --gui
```

`uv run` synchronizes the project environment. The `qachat` command checks
the required modules and may invoke `uv sync --extra gui --extra mlx` on an
Apple Silicon Mac, or `uv sync --extra gui --extra torch` on other supported
platforms, when GUI/backend modules are missing. This can modify the local
environment and contact package repositories. Models are managed separately.

To use an already-installed environment **without synchronization**, run:

```sh
uv run --no-sync python -m qachat.gui
```

This bypasses both uv's initial synchronization and the QAChat CLI's optional
dependency repair. Missing GUI modules will instead cause startup to fail.
The `--no-sync` option is an **uv** option, not a `qachat` option. Running
`uv run --no-sync qachat --gui` skips only uv's initial synchronization:
QAChat's launcher can still call `uv sync` if dependencies are missing.

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
HTML. Answers are rendered as Markdown (headings, lists, quotes, code, bold,
italics) by building DOM nodes, never by parsing HTML, so model output cannot
inject markup; plain answers simply become paragraphs. Markdown is rendered
while the answer streams, and citation links are added when it completes. Numeric citations are made clickable only when they refer to evidence
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
originals outside the workspace and saved chats are untouched.

The `outputs` and `templates` folders are always shown and can be edited
from their right-click menus: **Add files to …** copies files in (a taken
name is replaced after confirmation, or the copy is numbered), **Replace
with another file…** overwrites one file keeping its name, and **Delete**
removes a file or folder. Neither folder is indexed, so these edits only
refresh the tree; the open chat and index stay as they are. Drops are disabled during a running operation. The dashed target and dimmed
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
GUI projects never download models while preparing; missing models are
downloaded from Settings with their Download buttons. Literature defaults permit larger files than code defaults; the
profile and the current GUI fields remain the effective configuration.

## Project types

Project types live in `qachat/project_types/`, one JSON file per type
(`code.json`, `literature.json`); the file name is the type's ID and types
are listed in file-name order. Each file sets the display name shown in the
type menu, a one-line description (shown to the router), default file
suffixes and size limit for new projects, the retrieval instruction added
to Qwen3-format embedding queries, the base system prompt, the answer
guidance the desktop app adds to each question (for example the 150-300
word target), `capabilities.code_graph` (true, false, or "auto": only when
selected files include Python or MATLAB), `history.recent_turns`, the
`default_intent`, the `intents`, and the empty-chat headings and
suggestions. The first suggestion is always shown with two others; a "…"
in a suggestion is selected when it is clicked, ready to be replaced.

Each intent has a description, a system prompt added to the base prompt, an
optional answer guidance that replaces the type's (and always applies, also
in notebooks), a `history.policy` (none, recent, or relevant), and
`retrieval.mode` (search, document, reuse, or none) with
`retrieval.strategy` (only "hybrid" exists). An intent can also list
`shortcuts`: case-insensitive regular expressions for whole messages that
obviously have that intent (the default `small_talk` lists greetings,
thanks, and "how are you"). A match skips the router model, so such
messages are answered in about a second.

An intent may instead name a **tool** (`"tool": {"name": ..., settings}`),
which replaces retrieval and the answering prompt for that intent; its
retrieval mode must be "none" and it needs no system prompt. Tools live in
`qachat/tools/`, one module each, and must be registered in
`qachat/tools/registry.py`; each validates its own settings when the type
is loaded, and unknown settings are rejected. The tools are:

- `project_inventory` (intent `project_inventory`): exact counts and lists
  of the project's files by type and folder, and which have no searchable
  text, from the source manifest and index.
- `scan_and_list` (intent `mention_listing`): every exact mention of the
  question's terms, counted and listed with clickable lines, across all
  files or the one file the question names or describes.
- `batch_classify` (intent `mention_classification`): every exact mention
  sorted into categories by the answering model in batches; Python
  validates the labels and counts.
- `batch_review` (intents `document_reading` and `file_reading`): reads one
  whole document in parts, takes notes with passage numbers, and answers
  from the notes; the answer streams and its citations are checked.
- `generate_paper_summary` (intent `paper_summary`): turns one paper into a
  one-page document that follows a Markdown template from the project's
  `templates/` folder, which starts with a copy of
  `qachat/tools/templates/research_briefing.md` (the Research Briefing
  format of Nature-family journals: a headline of at most 90 characters and
  six sections with word limits, at most 750 words). Edit it, or add your
  own `.md` template and name it in the question ("…with the lab_summary
  template"); the rules (limits in brackets after a heading, `{citation}`
  and `{references}` placeholders) are in tutorial 160 and in
  `qachat/tools/report_template.py`. The headline and each section are
  written separately and checked in Python (word and item limits), with
  targeted fixes; the result is saved as Markdown in the project's
  `outputs/` folder, with a one-page PDF in the desktop app. The answer
  says where both files are and shows the saved Markdown. Nothing
  bibliographic is written by the model: the citation comes from the PDF's
  metadata and first page (or the file name), and reference items are
  entries of the paper's own reference list, which the model only chooses
  by number. A paper can be named by file name or first author ("Smith's
  paper"); a guessed paper is always confirmed first.

Tools that call the model many times (`batch_classify`, `batch_review`,
`generate_paper_summary`) first show a plan (scope, model calls, estimated
time) with **Continue**, **Quick** (a smaller, non-exhaustive version), and
**Cancel**, when the plan is larger than the tool's `large_batches`
setting (`confirmation`: always, when_large, or never).

A project folder has `sources/` (copied in, indexed, searched), `outputs/`
and `templates/` (never indexed). Summaries and reports QAChat writes go to
`outputs/`, so they can never come back as sources; you can also put a
paper you are working on there. Saving never overwrites a file: `-2`,
`-3`, ... is added instead. Projects from earlier versions have their
`target/` folder renamed to `outputs/` when opened.

Types are validated when QAChat starts; mistakes name the type and field.
Add a JSON file to create a new type; `code` and `literature` must stay.
Restart QAChat after editing. Only `code` and `literature` have bundled
example repositories, so other types need sources. Tutorial 150 explains
project types and intents and builds a new type; tutorial 160 shows how to
write and register a new tool.

## Models and generation settings

The window and Dock icon follow the Appearance setting: `icons/qachat-light.png`
for light, `icons/qachat-dark.png` for dark, and the operating system's color
scheme for System. Missing icon files leave Qt's default icon.

Settings are ordered Appearance, Effort levels, Search depth, Embedding
model, Intent model, Answers, then Project. Clicking
the backdrop or Close applies model edits, with confirmation for downloads and
embedding changes; cancelling keeps the panel open.

Effort levels lists five levels. Each has an editable name and a language
model chosen from a dropdown of cached language models. Search depth is a
separate section of five steps, each the number of passages retrieved for
an answer (1-200; default 5, 5, 10, 10, 15). The composer's effort picker
shows the effort name once at the top and has two sliders: Model chooses
the level whose model answers, and Search depth chooses the step, so a
small model can be paired with a deep search or the reverse. The middle
level's model is also the project's chat model. Cached
models are classified as language or embedding models by preset, by an
"embed" name or sentence-transformers configuration, then by architecture.
A selected model that is not downloaded shows a Download button. On PyTorch
every level uses one chat model; search depth still applies.

Each project keeps all its settings, changed or not, in its own
effort_levels.json: the levels per backend, the search depth under
"search_depth", and the intent model under "router". They are written when
the project is created (from the preferences below) or first opened by
this version, so a project resumes exactly as it was set up; later changes
to defaults.json or the preferences apply only to new projects. The model
names, embedding model, and answer settings stay in profile.ini. To make an
existing project follow new defaults, change its levels in Settings, or
delete its effort_levels.json and reopen it. Per-level reference counts saved by earlier
versions become the search depth (the old defaults become the new ones). An older
model_choices.json is migrated on first read and removed on the next save.
Changing a language model does not rebuild the index; changing the embedding
model (also a dropdown of cached embedding models) or query format requires
confirmation and rebuilds it. A note above the levels appears only when the
selected backend's packages are not installed.

The Intent model section, after Embedding model, chooses the router model
that classifies each question's intent and writes chat titles (default: the
8B chat model). A change applies at once: a separately loaded router model is
released and the new one loads when first needed. It is saved per backend in
effort_levels.json under "router".

**All default settings are in one file, `qachat/gui/defaults.json`:** the
appearance, the five effort levels (names and models) per backend, the
search depth, the intent and embedding models per backend, the query
format, and the answer settings (temperature, thinking, longest answer).
Edit it to change what everyone starts with, then restart QAChat; a
mistake is reported with the setting's name. (Notebooks use the core
library's defaults in `qachat/model_defaults.py`.)

Each user's own changes are kept in `qachat/gui/preferences.json` (ignored
by git; `QACHAT_PREFERENCES_FILE` points elsewhere, as the tests do): the
appearance, and the effort levels, router model, embedding model, query
format, and search depth that new projects start from. Settings edited with
no project open are saved there. With a project open, the switch at the top
of Settings chooses **This project** (the default) or **New projects**,
which edits these preferences and leaves the open project unchanged; a note
under the switch says which one is edited. Settings always reopen on
**This project**. Precedence: defaults.json < preferences.json < the project's
own settings. Only values that differ from the built-in defaults are written,
so restoring defaults removes them. The first-run check asks for the models
these preferences use. Its **QAChat setup** card stays at the top of
Settings: folded with "Ready" when everything is installed (click to see
each check), and opened by itself when something is missing or a download
runs. When some are missing but installed models of the
same role exist, **Use installed models** switches to the closest installed
size (for example a missing 8B model is replaced by an installed 14B
rather than a 4B) and saves that choice.

Selecting a cached model is not a completeness check. Offline loading,
backend compatibility, model architecture, tokenizer behavior, and pooling
assumptions are enforced or encountered later in the model-loading path.

The GUI allows up to 2,048 generated tokens; the backend profile default
stays 256, and projects saved with 256 answer with 2,048 in the GUI. The
token cap is a safety limit, not a requested answer length. Model-reported
length stops are
saved with the turn and shown as an incomplete-answer notice. Old answers are
not regenerated automatically.

Answer-time controls set retrieval count, temperature, thinking, and maximum
new tokens without rebuilding the index or reloading models. A blank answer
limit requests the session's automatic budget; it does not remove the model's
finite context limit. The values used for a completed answer are recorded with
that turn. Temperature, thinking, and answer length belong to the project:
a change is saved to its profile.ini when the next question is asked, every
chat in the project uses it, switching chats keeps it, and opening another
project loads that project's own values.

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

Conversation files use exclusive single-writer ownership. The GUI opens a
new conversation trace before closing the previous one, so a conflict on
the target does not discard the active trace. Metadata updates and branching
borrow the active writer's file lock; inactive conversations are locked for
the duration of their update or archival. Deleting an active conversation
moves the saved JSON into `logs/traces/deleted/` and releases the old writer
without appending a redundant closing event. The sidecar `.json.lock` file
may remain after closing; it does not indicate a stale active lock. See
[conversation-file ownership](../../docs/conversation_ownership.md) for the
error and recovery contract.

Select an answer to inspect its evidence. A citation requests a preview of
that evidence's source location through the bridge. Reference responses are
matched to the currently selected turn and citation before being displayed,
so a late response does not overwrite a newer selection.

The interface displays backend phases as they happen: loading a model,
identifying intent, retrieving or reusing references, and reading the
references before the first word streams. Reading time grows with the
references and the selected history; earlier turns enter as question and
answer text only. Each label
stays at least 0.6 seconds and a newer phase replaces queued ones, so the
display is at most one short step behind. A timer runs from Ask.

Answers stream as they are generated: the engine forwards text a few times
per second and the page appends it. On MLX this is token by token; PyTorch
delivers the whole reply at once. With thinking enabled, the model's thinking
shows live in a grey box that folds into "Thought for Ns" when the answer
starts; saved answers keep a collapsed "Show thinking". Citation links are
added when the answer completes, under a short "Took Ns · 3.5k tokens" line.

Tokens are the pieces of text models work in (about ¾ of a word). Each
answer counts the tokens the models read (question, evidence, instructions,
history) and wrote, over the answer model, the router, and every tool call;
the count updates live next to the timer while an answer is generated and
is saved with the turn. The gauge button next to Ask shows the project's
total (every chat, deleted ones included; turns branched chats share count
once); its menu splits it into read and written, and shows this chat's
total.

Routing: each question first goes to the 8B router model (see
`qachat/answering.py`), which picks one of the project type's intents and
says whether the question stands alone and which earlier turns it refers to.
Those turns are kept only when the question points back to them ("it",
"the second one", "you said", "so far") or shares a distinctive term with
them; small talk is never a related turn. The search always uses the
question as typed; when it points back, the related turns' questions are
added, never model-written text. The intent's configuration then selects history (question and answer text
only, never earlier evidence) and retrieval (search, reuse an earlier turn's
references, or none). A shimmering line under the pending answer steps through
what happens: identifying intent, the identified intent, the included
conversation history, the retrieval steps (skipped when not needed), and
reading the references before the answer streams. A finished answer shows only "Took Ns"; expanding it lists every step it
went through (intent, history, retrieval steps including the search text
when earlier questions were added, reading, and generating) with how long each took, measured when the
steps happened. Chat titles are written by
the router model too, so naming never switches models.

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
