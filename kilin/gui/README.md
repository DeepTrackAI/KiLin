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

When a selected backend is missing, setup displays a command for that
backend, not the platform-default `app` extra. A checkout uses
`uv sync --extra gui --extra mlx` or `--extra torch`. An ordinary
installed Python environment uses the matching `kilin[gui,mlx]` or
`kilin[gui,torch]` extra with that interpreter; if pip is absent, uv
can install into the same interpreter using `uv pip install --python`.
The macOS application uses the copy of uv inside its `.app` bundle and
its original wheel or pinned PyPI distribution, since its managed Python
environment does not include pip. This permits adding PyTorch even when
Apple Silicon initially installed MLX through `kilin[app]`. Run the
displayed repair command explicitly and restart the application. These
instructions do not auto-install packages, replace the current project
backend preference, or download model weights.

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
Fenced `latex` blocks display source text with long lines wrapped, while
preserving explicit line breaks and indentation. Other code blocks retain
their horizontal scrolling behavior.
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
package (the macOS app or pip), the suggested folder is `KíLín` under
Documents. On Linux, KíLín uses the XDG Documents location when configured;
on Windows it uses the registered Documents location (which may be redirected
into OneDrive). On macOS it uses `~/Documents/KíLín`. If a platform setting
cannot be resolved, it falls back to `~/Documents/KíLín`. KíLín does not
create this folder until it is confirmed.

`KILIN_WORKSPACES_DIR` overrides this default in both cases and must
be an absolute path (or a home-expanded `~/...` path). Otherwise the
projects folder chosen in KíLín setup is used: it is saved in the
preferences file as `workspace_folder`, so it is kept after a restart.
KíLín setup asks the user to confirm it once, recommending the folder in
use (Confirm projects folder), choose another (Browse), or choose the
standard folder (Use default); each saves the folder by its path and
creates it. A chosen folder that is missing or not mounted is skipped with
a note, and setup asks again. Changing it restarts the engine, so it is
offered only on the start screen. Existing projects are never moved.

For installed versions upgraded from before 7 November 2026, KíLín also
checks the previous user-data `workspaces/` location for a direct subfolder
containing `profile.ini`. If found, setup offers **Use existing projects**
and displays the old directory; the user must explicitly select it. The
folder is rechecked before saving and the engine restarts to discover its
projects normally. This only changes the saved projects-folder preference:
there is no automatic copy, deletion, or migration of sources, indexes, or
conversations. A saved choice (or `KILIN_WORKSPACES_DIR`) remains authoritative.
Projects with outdated or invalid profiles may still require separate repair.
Cancelling Browse and failures when choosing a folder return to setup so the
user can choose again.

The project folder is a local filesystem path, but the operating system or
services such as iCloud or OneDrive may synchronize it. For sensitive data,
choose a directory that is not synchronized. KíLín does not control external
filesystem synchronization, and it cannot guarantee that Documents is
strictly local.

Before anything else, a welcome dialog summarizes KíLín and shows the full
terms of use (`kilin/gui/terms.txt`), with links to the Apache 2.0 licence
and Hugging Face's terms and privacy policy, opened in the system browser.
The user ticks the agreement and clicks Continue; until Qt confirms the
saved agreement, the dialog cannot be dismissed. A failed save displays an
error inside the dialog and allows a manual retry without clearing the
checkbox. If the version changed, setup reloads the terms and requires a
fresh agreement. The agreed version is saved as `terms_accepted`;
changing the text needs a new `TERMS_VERSION` in `kilin/gui/preferences.py`,
so every user is asked again. Settings reopens the terms from its Terms of
use link, under a copyright line built from the package authors.
The initial setup blocks project interaction until terms acceptance and
projects-folder confirmation succeed. After that, saved projects can be
resumed without downloading new models; creating and answering in new
projects still requires the configured models. Changing the type selected
for a new project does not filter or hide existing resumable projects.
Repeated creation from the same sources allocates independent numbered
folders. An explicitly selected occupied destination is
never reused: the interface offers to resume an existing project separately,
without importing pending sources or applying new-project settings, or lets
you choose another destination. The current session stays open on conflicts.
A saved workspace can be reopened through Resume without adding
sources. New projects may start without sources: they get a free
`new-project` folder in the workspace folder. Their managed `sources/`
folder is initially empty, and sources can be added later. No development example is selected
automatically. A workspace is a directory organization convention, not an
execution sandbox or a boundary for hostile inputs.

Settings configure the profile before indexing and model loading.
GUI projects never download models while preparing; missing models are
downloaded from Settings with their Download buttons. Literature defaults
permit larger files than code defaults; the profile and current GUI fields
remain the effective configuration.

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
  and `{references}` placeholders) are in tutorial 170 and in
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
Tutorial 160 explains
project types and intents and builds a new type; tutorial 170 shows how to
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
The welcome screen displays the locally packaged, full-body
Gaudí-inspired mosaic qilin (dragon head, deer body, cow tail).
In an open project, the sidebar now shows the full mascot alone,
without a separate wordmark, while the welcome card continues to
use the refined vector KíLín logotype with evenly spaced letters
and tapered accents above both i characters. Dark mode keeps the
vivid ceramic colors against navy surfaces. The window and Dock
use a close-up of the dragon head; `icons/README.md` documents the
artwork and the attribution history.
The palette and decorations are CSS-only and do not alter interaction,
project settings, model selection, or accessibility labels.

Settings has two parts. KíLín setup holds the app-wide choices: the
computation backend for new projects, Appearance, and the projects folder.
Selecting Auto uses MLX on native Apple Silicon and PyTorch otherwise.
Selecting MLX or PyTorch explicitly selects that backend for setup checks,
required-model downloads, and compatible cached-model alternatives, with no
silent fallback to the other backend. If the selected runtime is unavailable
or MLX is unsupported on the platform, setup reports that limitation rather
than declaring another backend ready. Changing the preference refreshes the
setup status after the write succeeds; if the preferences file cannot be
saved, the previous selection is restored and an error is shown.
New-project preparation requires the selected default runtime and models,
but Settings can be closed even when they are missing. Saved projects remain
available from Resume and the Projects sidebar, using each project's own
saved backend and models. An unavailable saved-project model fails through
the normal project-opening path; it is not replaced by a new-project default.
If switching projects fails after the former session has been released, the
GUI returns to the start screen and displays the error. A profile rejected
before the former session is released leaves that session open.
The Project part has five tabs: General (the project's name and Delete),
Indexing (file types, file size and chunk sizes), Models (effort levels,
intent model, embedding model), Answers (temperature, answer length,
search depth, thinking) and Tools. Changes are submitted when chosen or
when a typed field is committed (Enter or leaving the field). Answer settings
are not considered saved until the engine acknowledges the profile write.
While one answer-settings save is pending, later edits remain available and
only the latest combination is queued for the next save. The next question
cannot start until saving finishes; its text is preserved. On failure the
engine rereads the actual profile and synchronizes the live chat. The GUI
restores those confirmed values, or requires reopening the project if the
saved state cannot be verified. A download, an embedding change or an
Indexing change asks first, because it can take a while; cancelling puts
the saved value back. During the rebuild, the page reports the operation as
pending, not already saved. Success is confirmed only after the project
reopens with its updated settings. If rebuilding fails, the engine restores
any changed `profile.ini` atomically and, if required, attempts to reopen
the previous conversation. If recovery also fails, the project is closed
where possible and the GUI asks for inspection and reopening. Rollback
covers the profile file, **not** every intermediate index artifact; those
are validated when the project reopens.

The line under the tabs reports save status. On the start screen nothing
is saved: the Project values set there are kept on the page and written only
into the new project's `profile.ini` when it is created, so the next new
project starts from the defaults again.

Effort levels lists five levels. Each has an editable name and a
language model chosen from a dropdown of cached language models. The
GUI renders each cached chat template with `enable_thinking` on and
off for one sample prompt. It shows "· thinking switch detected" when
the rendered prompts differ, "· no template switch detected" when
they are identical, or no label when the template cannot be checked.
This probe neither loads the model nor establishes its reasoning
capability or runtime support. The composer's Thinking switch remains
disabled unless a template prompt difference was detected. In Answers,
"Request thinking via chat template" passes the setting to generation;
actual model behavior is not verified by the probe. Search depth is a
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
application-wide choices are saved as user-preference overrides. The
core (`kilin.model_settings`) reads these same project settings when
`RAGSession` is opened directly from Python, without starting the GUI.

**GUI defaults** ship in the read-only
`kilin/gui/defaults.json`: appearance, the backend new projects are made
with (`auto`, `mlx` or `torch`, chosen in KíLín setup), the five effort
levels (names and models), search depth, router and embedding models,
query format, and the projects folder.
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
version with edited installed defaults, manually preserve those edits.
KíLín does not automatically migrate edits to installed package resources.

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
rather than a 4B) and saves that choice. Setup is guidance for new projects,
not a prerequisite for browsing or resuming existing saved projects.

The GUI's cached-model list uses lightweight, offline file checks: a
configuration, recognized tokenizer assets, and complete indexed weights.
It reads only small metadata and file sizes, never full weights or tokenizer
contents. "Downloaded" means locally complete by these checks, not verified
loadable; backend compatibility, tokenizer behavior, and pooling assumptions
are encountered later in the model-loading path. See
[Model management](../../docs/model_management.md) for the exact limits.

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
a change is saved and acknowledged when committed in Settings or the
Thinking control. Every chat uses the confirmed project-wide settings,
switching chats keeps them, and another project loads its own values. A
question never silently commits unconfirmed settings; it must wait until
the corresponding profile save has been acknowledged.

An ordinary answer intent can override thinking for that reply without
changing these controls or the saved project setting. Both packaged
`small_talk` intents disable thinking; an omitted or null intent policy
inherits the configured value. A following intent without an override
therefore uses the configured value again. This uses the selected model/template
control, not a model switch, and does not control tool-internal calls.
See [Per-intent thinking policy](../../docs/configuration.md#per-intent-thinking-policy).

## Conversations, references, and progress

The message box expands as text is typed, pasted, or restored from a draft.
It contracts when text is removed or after a message is sent. At 250 pixels
it scrolls internally rather than displacing more of the conversation. **Enter** sends; **Shift+Enter** inserts a newline.

New chat creates a separate conversation record while retaining earlier
conversations in the picker; it does not archive or delete the previous GUI
conversation. The chat producing an answer is italicized with a spinner in
the left panel, even while another chat is being read. The indicator clears
on completion, cancellation, or failure and is not saved. Branch copies
resumable parent state and records parent and checkpoint identifiers; it
does not copy the entire parent event history.
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
moves the saved JSON into `logs/traces/.deleted/`, while its hidden
`.conversation_*.json.lock` companion stays at the original path in
`logs/traces/`. The old writer is released without appending a redundant
closing event. The retained lock companion does not indicate an active
writer. Restoration requires moving the JSON back while KíLín is closed;
the original lock companion is reused. See
[conversation-file ownership](../../docs/conversation_ownership.md) for the
error and recovery contract.

Clicking an answer selects its evidence when the side panels are visible;
selecting text does not change the current reference. Ordinary answer clicks
do not open hidden panels. Clicking a citation opens the panels and requests
a preview of the cited source through the bridge. Pressing Enter or Space on
a focused answer also explicitly opens its references. Reference responses
are matched to the currently selected turn and citation before being
displayed, so a late response does not overwrite a newer selection.

The interface displays backend phases as they happen: loading a model,
identifying intent, retrieving or reusing references, and reading the
references before the first word streams. Reading time grows with the
references and the selected history; earlier turns enter as question and
answer text only. Each queued label
is displayed briefly (currently 450 ms), and later stages follow it, so the
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
references, or none). The core reports an ordered activity stream with
intent identification, history decisions, retrieval (including skipped
work), tool progress, and generation. The pending answer shows these same
labels in an expandable live list, including events that describe skipped
operations; the GUI does not filter them. A finished answer shows "Took Ns";
expanding it lists the identical core-recorded activities with measured
durations. The GUI estimates the times of in-flight activities; completed
timings are stored with each turn and appear for notebook-generated answers
as well. Enabled traces also record each activity start. The selected
router model writes chat titles without switching the answering model.

Stop terminates the engine and starts a fresh backend process. Loaded models
and unsaved work are lost; previously completed writes remain on disk. Reopen
the project to continue. This is not transactional cancellation of a copy,
index write, download, or conversation update that was in progress. The GUI
never recursively deletes a project after Stop: it removes only an empty
creation-marker directory whose metadata confirms it was created by the
stopped engine, not an existing empty folder. Completed projects and
nonempty partial workspaces are preserved, with known paths reported for
inspection. A stopped preparation may leave an occupied folder that must be
reviewed and removed manually before that location can be used again.

## Trace privacy and current limits

The engine entry point enables diagnostic tracing. Enabled traces and
conversation checkpoints can contain complete questions, answers, source
excerpts, paths, settings, token identifiers, and diagnostic output. There
is no standard/forensic mode switch or automatic redaction. Disabling
diagnostic events does not disable resumable conversation checkpoints,
which may still contain full text. Treat the entire workspace, especially
`logs/`, as sensitive data.

Model changes emit gui.models.change.requested/confirmed/applied events;
downloads emit start/completion events. Source imports record the request
and copied files. index.incremental.plan records reused and embedded chunk
counts plus added, modified, and removed paths. Conversation turns retain the
model and generation settings actually used.
