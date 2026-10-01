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
state changes and bridge calls. HTML and CSS were retained byte for byte.

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
Additional UTF-8 suffixes can be entered in Advanced settings.

The selected sources are copied into the workspace's `sources/` directory.
This multi-source route is cumulative: re-adding a source updates matching
files, but deletion or deselection at the origin does not automatically remove
old workspace copies. It differs from the single-source synchronization API
in `qachat.project`. The original source is not intentionally modified.

A workspace also contains `profile.ini`, `indexes/`, and `logs/`. It can be
selected explicitly or chosen through the engine's default workspace route.
A saved workspace can be reopened without adding sources. The default
example-source route requires an example repository that is not included in
this attached archive. A workspace is a directory organization convention,
not an execution sandbox or a protection boundary for hostile inputs.

Advanced settings configure the profile before indexing and model loading.
New GUI projects leave preparation downloads disabled unless explicitly
allowed. Literature defaults permit larger files than code defaults; the
profile and the current GUI fields remain the effective configuration.

## Models and generation settings

The Models panel shows suggested chat and embedding models compatible with
the selected backend and any required embedding-query format. Explicit
Download actions request missing suggested models. Other cached model IDs
are listed separately. The chat dropdown offers cached IDs without proving
their role; the embedding dropdown filters known compatible embedding presets.
A custom model ID or local path can still be entered and is checked when used.

Selecting a cached model is not a completeness check. Offline loading,
backend compatibility, model architecture, tokenizer behavior, and pooling
assumptions are enforced or encountered later in the model-loading path.

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
Rename updates display metadata. Changing conversation retains already loaded
models and indexes where the engine implements that reuse.

Select an answer to inspect its evidence. A citation requests a preview of
that evidence's source location through the bridge. Reference responses are
matched to the currently selected turn and citation before being displayed,
so a late response does not overwrite a newer selection.

The interface displays backend phases and progress while work is running.
Once a complete answer arrives, JavaScript reveals it one character at a time.
This is a presentation animation, not streamed model generation. It can keep
the interface busy after generation has already completed.

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

The separate documentation review records existing persistence, path-trust,
and context-budget issues. They were not fixed as part of documentation.
The interface was reviewed from source, but not launched in the documentation
runtime because its Qt dependencies were unavailable.
