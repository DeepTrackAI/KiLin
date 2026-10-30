# KíLín desktop interface

The desktop interface is a local Qt shell around KíLín's project, retrieval,
and model APIs. The web page owns presentation state; a separate Python
process owns filesystem access, indexing, model sessions, and conversation
storage. Closing or stopping that process is not the same as pausing a model.

## Launch and dependencies

With synchronization and automatic preparation of missing GUI/backend
dependencies, launch from the repository with:

```sh
uv run kilin --gui
```

`uv run` synchronizes the project environment. The `kilin` command checks
the required modules and may invoke `uv sync --extra gui --extra mlx` on an
Apple Silicon Mac, or `uv sync --extra gui --extra torch` on other supported
platforms, when GUI/backend modules are missing. This can modify the local
environment and contact package repositories. Models are managed separately.

To use an already-installed environment **without synchronization**, run:

```sh
uv run --no-sync python -m kilin.gui
```

This bypasses both uv's initial synchronization and the KíLín CLI's optional
dependency repair. Missing GUI modules will instead cause startup to fail.
The `--no-sync` option is an **uv** option, not a `kilin` option. Running
`uv run --no-sync kilin --gui` skips only uv's initial synchronization:
KíLín's launcher can still call `uv sync` if dependencies are missing.

`kilin.cli.main` checks dependency module discoverability and may invoke
`uv sync` in a development checkout before launching the GUI. This can modify
the environment and contact package repositories. Its `--backend` argument
controls dependency preparation; it does not override the project's backend
selection in the GUI. `python -m kilin.gui` delegates directly to the GUI
application and does not perform that synchronization step.

## Components and communication

`app.py` creates the Qt application and window. `window.py` hosts the local
page, exposes `ChatBridge` through QWebChannel, starts the engine subprocess,
and converts bridge calls into JSON-line commands. Engine responses are
relayed to JavaScript as JSON events. Unexpected diagnostic output is kept
separate from protocol messages where the bridge implements that handling.

`engine.py` handles commands synchronously, invokes KíLín's APIs, and emits
project, progress, answer, reference, and failure events. `model_catalog.py`
contains suggested model records and distribution-presence checks. A cache
entry or an installed distribution is not proof that a model can load or run.

For the existing GUI command/event names, payloads, and recovery rules,
see [the desktop protocol reference](../../docs/gui_protocol.md).

`web/index.html` defines the setup form, conversation controls, workspace and
reference panels, activity display, and model dialog. Its identifiers are the
contract used by `web/app.js`. `web/styles.css` defines layout, interaction
states, and responsive breakpoints. JavaScript function comments describe
state changes and bridge calls.

The page's local-content access to remote URLs is disabled in the Qt view.
That setting does not prevent the Python engine from downloading models.
Answers and source excerpts are inserted as text rather than executed as
HTML. Answers are rendered as Markdown (headings, lists, quotes, code,
tables, bold, italics, LaTeX maths) by building DOM nodes, never by parsing HTML, so model
output cannot inject markup; plain answers simply become paragraphs. Markdown is rendered
while the answer streams, and citation links are added when it completes. Numeric citations are made clickable only when they refer to evidence
records supplied with that answer.
LaTeX maths (`$…$`, `\(…\)`, `$$…$$`, `\[…\]`) is drawn by the bundled
KaTeX (`web/katex/`, MIT) as MathML, without fonts or network access;
LaTeX it cannot read stays as text.
Files attached to a message are first added to the project's sources and
indexed, without the add dialog; the question then lists their `sources/`
paths. The question shows them as chips that open the workspace tree there.

## Source selection and workspace preparation

Select a registered project type, then choose local files or directories in the
native dialog or drag-and-drop. The engine previews directory selections
using repository scanning. The tree shows selected and skipped entries;
protected entries remain unavailable. Oversized eligible files can be checked
explicitly, which increases the submitted size limit to accommodate them.
The selection protocol uses stable `reason_code` identifiers; displayed
explanations do not control which files are selectable. See
[the protocol reference](../../docs/gui_protocol.md#source-preview-contract).
Additional UTF-8 suffixes can be entered in Settings.

The selected sources are copied into the workspace's `sources/` directory.
This multi-source route is cumulative: new sources are copied, while
previously imported copies change only after explicit synchronization
choices. Deletion or deselection at the origin does not remove snapshots
until approved. It differs from the single-source synchronization API
in `kilin.project`. The original source is not intentionally modified.

In an open project, only the sidebar's `sources` tree accepts dropped files
or folders. It highlights during a native file drag; other workspace areas
reject drops. A confirmation explains the copy and index update before any
import command is sent. Direct empty and oversized files are left out,
missing or linked source paths are rejected, and same-name imports show
their distinct destination names. Confirmed imports use saved filters,
update the source manifest and indexes, and restore the active conversation.
On resume, the GUI checks imported originals for changes and asks which
files to update, preserve as versions, keep, or remove. A kept copy whose
original has disappeared remains detached; restoring the original offers
explicit reconciliation without silently overwriting the kept copy.
Sync rejects updates that exceed the project's saved file-size limit.

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
selected explicitly or created under the GUI's default workspace parent. In
a source checkout (containing `pyproject.toml`, `tests/kilin/`, and the GUI
source), the default is the repository's `sandbox/` folder. In an installed
package, it is a per-user application-data directory:

- macOS: `~/Library/Application Support/kilin/workspaces/`
- Linux: `${XDG_DATA_HOME:-~/.local/share}/kilin/workspaces/`
- Windows: `%LOCALAPPDATA%/kilin/workspaces/` (with a home fallback)

`KILIN_WORKSPACES_DIR` overrides this default in both cases and must
be an absolute path (or a home-expanded `~/...` path). Otherwise the
Workspace folder chosen in Settings → Project (Browse) is used: it is saved
in the preferences file as `workspace_folder`, so it is kept after a
restart, and Use default goes back to the standard folder. A chosen folder
that is missing or not mounted is skipped with a note. Changing it restarts
the engine, so it is offered only on the start screen. The engine
does not create the default directory until a project needs it; existing
projects are never moved. Repeated creation from the same sources allocates
independent numbered folders. An explicitly selected occupied destination is
never reused: the interface offers to resume an existing project separately,
without importing pending sources or applying new-project settings, or lets
you choose another destination. The current session stays open on conflicts.
A saved workspace can be reopened through Resume without adding sources. New projects may also start without sources: they get a free
`new-project` folder in the workspace folder. Their managed `sources/` folder is initially
empty, and sources can be added later. No development example is selected
automatically. A workspace is a directory organization convention, not an
execution sandbox or a boundary for hostile inputs.

Settings configure the profile before indexing and model loading.
GUI projects never download models while preparing; missing models are
downloaded from Settings with their Download buttons. Literature defaults permit larger files than code defaults; the
profile and the current GUI fields remain the effective configuration.

## Project types

Project types live in `kilin/project_types/`, one JSON file per type
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
thanks, and "how are you"). A match selects the intent without a router
inference call.

An intent may instead name a **tool** (`"tool": {"name": ..., settings}`),
which replaces retrieval and the answering prompt for that intent; its
retrieval mode must be "none" and it needs no system prompt. Tools live in
`kilin/tools/`, one module each, and must be registered in
`kilin/tools/_registry.py`; each validates its own settings when the type
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
- `generate_paper_summary` (four literature report intents): turns one
  paper into a structured document following a Markdown template in the
  project's `templates/` folder. New literature projects receive only
  their four configured templates. The included `research_briefing.md`
  follows the Research Briefing format of Nature-family journals: a
  headline of at most 90 characters and six sections with word limits,
  at most 750 words. Edit a supplied template, or add your
  own `.md` template and name it in the question ("…with the lab_summary
  template"); the rules (limits in brackets after a heading, `{citation}`
  and `{references}` placeholders) are in tutorial 160 and in
  `kilin/tools/generate_paper_summary/_report_template.py`. The headline and each section are
  written separately and checked in Python (word and item limits), with
  targeted fixes; the result is saved as Markdown in the project's
  `outputs/` folder. In the desktop app, an additional PDF is generated
  when rendering succeeds. The answer lists the generated files and
  shows the saved Markdown. PDF rendering uses the private Qt adapter
  `kilin/gui/_report_pdf.py`; formatting belongs to the report tool.
  The adapter attempts a one-page A4 layout but can produce more pages
  when the report does not fit. Without a renderer callback, the core
  saves only Markdown. Bibliographic information is not invented by the
  model: the citation comes from the source PDF's
  metadata and first page (or the file name), and reference items are
  entries of the paper's own reference list, which the model only chooses
  by number. A paper can be named by file name or first author ("Smith's
  paper"); a guessed paper is always confirmed first.

Tools that call the model many times (`batch_classify`, `batch_review`,
`generate_paper_summary`) first show a plan (scope, model calls, estimated
time) with **Continue**, **Quick** (a smaller, non-exhaustive version), and
**Cancel**, when the plan is larger than the tool's `large_batches`
setting (`confirmation`: always, when_large, or never).

A GUI-managed project has `sources/` (copied in, indexed, searched),
`outputs/` and `templates/` (never indexed). The core creates the latter
two folders for every project, even when no templates are selected.
Summaries and reports KíLín writes go to
`outputs/`, so they can never come back as sources; you can also put a
paper you are working on there. Saving never overwrites a file: `-2`, `-3`, ... is added instead.
Workspace initialization never renames unrelated directories.

Types are validated when KíLín starts; mistakes name the type and field.
Add or remove packaged JSON files to specialize a distribution; at least
one valid type must remain. Neither `code` nor `literature` is mandatory.
Restart KíLín after editing. The first registered type is the initial GUI
selection. Welcome prompts and their selection history are dynamic for all
types; absent prompts fall back to "What would you like to work on?".
Tutorial 150 explains
project types and intents and builds a new type; tutorial 160 shows how to
write and register a new tool.

## Models and generation settings

KíLín setup sits at the top of Settings. Below it, settings are grouped in
tabs: **General** (Appearance), **Models** (effort levels with their search
depth, embedding and intent models), **Answers** (temperature, answer length,
thinking), and **Project** (workspace folder, backend, file types,
chunking). Messages from Close show below the tabs, whichever tab is open.

The window and Dock icon follow the Appearance setting: `icons/kilin-light.png`
for light, `icons/kilin-dark.png` for dark, and the operating system's color
scheme for System. Missing icon files leave Qt's default icon.
The left-panel qilin logo is gold in light mode and blue-black in dark
mode; `icons/README.md` credits its CC BY-SA 4.0 source and is
installed with the images.

Settings are ordered Appearance, Effort levels, Search depth, Embedding
model, Intent model, Answers, then Project. Clicking
the backdrop or Close applies model edits, with confirmation for downloads and
embedding changes; cancelling keeps the panel open.

Effort levels lists five levels. Each has an editable name and a language
model chosen from a dropdown of cached language models. Search depth is a
separate section of five steps, each the number of passages retrieved for
an answer (1-200; default 5, 8, 10, 15, 20). The composer's effort picker
shows the effort name once at the top and has two sliders: Model chooses
the level whose model answers, and Search depth chooses the step, so a
small model can be paired with a deep search or the reverse. The middle
level's model is also the project's chat model. Cached
models are classified as language or embedding models by preset, by an
"embed" name or sentence-transformers configuration, then by architecture.
A selected model that is not downloaded shows a Download button. On PyTorch
every level uses one chat model; search depth still applies.

Each project stores its effort levels (separately for MLX and PyTorch),
search-depth steps, and per-backend intent router in `profile.ini`, alongside
its chat and embedding models, query format, and generation settings. The GUI
initializes any missing project settings from the effective application
preferences when the project is created or first opened. Saved project
settings remain authoritative: later changes to packaged defaults or user
preferences affect new projects, not existing ones. To change an existing
project's choices, use Settings rather than deleting its saved profile.

Earlier KíLín versions stored effort levels, router models, and search depth
in `effort_levels.json`, and model substitutions in `model_choices.json`.
When opening an older project, the GUI imports compatible values into
`profile.ini` and removes the legacy files after saving the settings.
Customized per-level retrieval counts in legacy data can supply the new
search-depth steps; unchanged legacy defaults do not override the current
search-depth defaults. Changing a language model does not rebuild the index;
changing the embedding model (also a dropdown of cached embedding models)
or query format requires confirmation and rebuilds it. A note above the
levels appears only when the selected backend's packages are not installed.

The Intent model section, after Embedding model, chooses the model
used to classify question intents and write chat titles. The packaged
default is Qwen3-4B on both backends, with backend-specific model IDs in
`defaults.json`. A change applies at once: a separately loaded router
model is released, and the replacement loads when first needed. Project
choices are saved per backend in the `[router]` section of `profile.ini`;
application-wide choices are saved as user-preference overrides.

**GUI defaults** ship in the read-only
`kilin/gui/defaults.json`: appearance, the five effort levels (names and
models), search depth, router and embedding models, and query format.
Application-wide changes are stored separately in a user-owned
`preferences.json`, using only keys (and backend values) that differ from
the shipped defaults. The file is not created just to read defaults.

- macOS: `~/Library/Application Support/kilin/preferences.json`
- Linux: `${XDG_CONFIG_HOME:-~/.config}/kilin/preferences.json`
- Windows: `%APPDATA%/kilin/preferences.json` (with a home fallback)

`KILIN_DEFAULTS_FILE` explicitly overrides the **writable preferences
file**, not the packaged defaults. The override must be an absolute path (or
a home-expanded `~/...` path). Existing complete settings files at that
path are accepted and written as sparse overrides on the next update. Invalid
JSON, unsupported keys, and invalid values produce errors; the file is not
silently reset. Updates use atomic replacement, and no package files are
modified. When working from a source checkout, edits made directly to the
shipped `defaults.json` remain readable as checkout defaults but are not
copied into the user-preferences file. Before reinstalling an earlier KíLín
version with edited installed defaults, manually preserve those edits:
automatic migration from previously mutable package resources is not part
of this PR.

**Generation defaults** for new projects remain core-owned in
`kilin/kilin_defaults.json` (`PROJECT_PROFILE`): `thinking=true`,
`temperature=0.0`, and `max_new_tokens=8192`. All interfaces use these
new-project generation defaults. Existing projects retain the settings in
their own `profile.ini`, including an explicit 256-token limit. A saved
project profile takes precedence over global defaults and user preferences.

With no project open, settings edits update the user preferences. With a
project open, the Settings switch chooses **This project** (the default) or
**New projects**. The latter edits application preferences while preserving
the active project. Settings reopen on **This project**. Returning a setting
to its packaged value removes its override. The first-run check asks for the
models selected by the effective preferences. Its **KíLín setup** card stays at the top of
Settings: folded with "Ready" when everything is installed (click to see
each check), and opened by itself when something is missing or a download
runs. When some are missing but installed models of the
same role exist, **Use installed models** switches to the closest installed
size (for example a missing 8B model is replaced by an installed 14B
rather than a 4B) and saves that choice.

Selecting a cached model is not a completeness check. Offline loading,
backend compatibility, model architecture, tokenizer behavior, and pooling
assumptions are enforced or encountered later in the model-loading path.

The default maximum for a new project's answer is 8,192 generated
tokens, supplied by the core `PROJECT_PROFILE` configuration. The GUI does
not substitute another limit for a saved project's value: for example,
a project saved with a 256-token cap retains that cap. The answer-token
limit is a safety ceiling, not a requested response length; Settings can
change it. The model's finite context limit still applies. Model-reported
length stops are saved with the turn and shown as an incomplete-answer
notice; old answers are not regenerated automatically.

Answer-time controls set retrieval count, temperature, thinking, and maximum
new tokens without rebuilding the index or reloading models. A blank answer
limit requests the session's automatic budget; it does not remove the model's
finite context limit. The values used for a completed answer are recorded with
that turn. Temperature, thinking, and answer length belong to the project:
a change is saved to its profile.ini when the next question is asked, every
chat in the project uses it, switching chats keeps it, and opening another
project loads that project's own values.

An ordinary answer intent can override thinking for that reply without
changing these controls or the saved project setting. Both packaged
`small_talk` intents disable thinking; an omitted or null intent policy
inherits the configured value. A following intent without an override
therefore uses the configured value again. This uses the selected model/template
control, not a model switch, and does not control tool-internal calls.
See [Per-intent thinking policy](../../docs/configuration.md#per-intent-thinking-policy).

## Conversations, references, and progress

New chat creates a separate conversation record while retaining earlier
conversations in the picker. It does not archive and delete the previous GUI
conversation. Branch copies resumable parent state and records parent and
checkpoint identifiers; it does not copy the entire parent event history.
Renaming a conversation updates display metadata. The chat heading and the
floating top bar show only the open chat's name and turn count. The left
panel lists projects; right-click the open project to rename it, and its
settings icon opens its Settings. Other projects have no settings icon,
because Settings save into the open project.
Renaming a project moves
its workspace folder, reloads the conversation registry, and restores the
selected chat. If reopening fails, KíLín attempts to restore the old folder. Changing conversation retains already loaded
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

Routing normally uses the project's selected intent router model
(see `kilin/answering.py`) to choose an intent and decide whether the
question stands alone or refers to earlier turns. Configured intent
shortcuts, such as simple greetings, skip router inference; callers can
also disable routing and use the project type's default intent. When the
router is used, earlier turns are retained only if the question points
back to them ("it", "the second one", "you said", "so far") or shares a
distinctive term with them; small talk is never a related turn. The search
uses the question as typed and, when it points back, adds related turns'
questions rather than model-written text.

The selected intent determines the history (question and answer text only,
never earlier evidence) and retrieval mode (search, reuse earlier
references, or none). A shimmering line under the pending answer shows
the phases that apply: intent identification, included conversation
history, retrieval, and reading references before generation. A finished
answer shows "Took Ns"; expanding it lists the actual steps and measured
durations. The selected router model also writes chat titles without
switching the answering model.

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
