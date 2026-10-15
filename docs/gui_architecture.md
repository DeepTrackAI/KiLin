# Desktop GUI architecture and lifecycle

## Optional adapter, not the core application

The desktop interface is implemented with PySide6/Qt WebEngine and a local
HTML/CSS/JavaScript page. Importing `qachat` or running the Python core in a
notebook must not require Qt. The GUI provides a file picker, setup forms,
model management, project/conversation controls, progress, rendered answers
and reference previews; source policy, model sessions, indexing and
persistence remain delegated to Python core APIs.

The more detailed [GUI README](../qachat/gui/README.md) is the user- and
implementation-facing companion, while the exact command/event names and
payloads are owned by [GUI protocol](gui_protocol.md).

## Processes and message boundaries

![Qt hosts the JavaScript page and QWebChannel bridge; JSON-line commands and events connect Qt to a separate engine child process.](figures/gui-processes.svg)

*Figure: QWebChannel traffic stays inside the Qt process; newline-delimited JSON messages cross the engine subprocess boundary.*

`gui/__main__.py` launches `gui.app.main()`. The `qachat --gui` CLI can
first check/prepare optional dependencies using `uv` in a development
checkout, whereas `python -m qachat.gui` does not perform dependency
installation. The GUI process imports Qt; the child engine can run without
Qt for ordinary answering, except optional PDF rendering injected into
report tools. A Qt application's existence is a per-process global
constraint, which matters for integration test isolation.

`ChatWindow` owns the backend subprocess and QWebChannel bridge.
`ChatBridge` offers camelCase slots such as `openProject`, `askQuestion`,
`resumeProjectConversation`, `previewSources`, `choosePlan`, and
`stopOperation`. The bridge translates typed and JSON-encoded arguments into
backend commands or handles native OS actions itself. The engine reads
commands sequentially, writes structured JSON events and forwards runtime
output to diagnostic tracing instead of mixing arbitrary text into stdout.

The page's `onEvent` consumes the events and updates transient view state;
backend `opened`/`answered`/`conversations` events are not replaced by
frontend guesses about project storage. Core events and browser-specific
phase timers are different: a character-by-character answer reveal does not
necessarily reflect a backend's incremental token generation.

## Initial setup and project selection

Startup retrieves available model choices, backend readiness, project types
and the preferred/default settings. `gui/model_catalog.py` describes the
suggested catalog; `huggingface.py` checks the cache. A cached distribution
is not proof it will load. Model downloads are explicit GUI actions and
have progress and error events; previewing source files or creating a
workspace does not silently download inference models.

The user chooses `code` or `literature` and local source paths. The engine
uses the core repository scanner to preview directory selections and the
explicit-file path to preview individual files. Entry rows have structured
`reason_code` from core JSON plus human-readable `reason`; JavaScript uses
the former when making opt-in eligibility decisions. `file_too_large` can
be selected explicitly; protected/linked scanner entries stay unavailable.
The GUI then passes confirmed copies and project settings to core workspace
creation and indexing.

A project already saved under the selected workspace can be listed and
resumed without recopying original sources. Managed projects normally
contain `sources/`, `outputs/`, `templates/`, `indexes/`, `logs/` and
`profile.ini`. Output/template editing is delegated through `project_files`
APIs and does not reindex sources. Adding/removing confirmed `sources/`
copies does refresh the retrieval index, without deleting originals.

## Workspace ownership and paths

Packaged `gui/defaults.json` is **immutable** during normal operation.
`gui/preferences.py` merges its validated values with sparse per-user
preferences; `gui/_paths.py` resolves config/data locations independently
of Qt. In a validated development checkout, the GUI's default workspace
parent remains its root `sandbox/`; in an installed distribution it is a
platform-specific per-user data directory. `QACHAT_WORKSPACES_DIR` and an
explicit workspace selection override that suggested parent.

An existing project's `profile.ini` remains authoritative for backend,
models, search depth, effort and router settings. `gui/model_settings.py`
stores GUI-specific options in sections of that same profile. Its controlled
legacy migration is different from a general project data migration. See
[Configuration](configuration.md) for all precedence rules.

## Operation lifecycle and answer display

The GUI engine owns one `ChatEngine` and an optional live `RAGSession`. A
request to answer:

1. Validates the selected project, chat settings and current operation.
2. Chooses the required effort-model behavior and search-depth step.
3. Calls `RAGSession.ask()` with phase/token/progress callbacks and optional
   callbacks for tool plans, document choices and PDF output.
4. Sends `answer_phase`, progress, usage and answer-text events to the page.
5. Relays the committed turn and evidence in `answered`, so the page can
   render citations and saved timeline information.

For a large tool plan, the backend can request a user choice **while the
answer is running**. The engine's `read_command` uses the same stdin iterator
for plan choices and ordinary dispatch; while awaiting a choice, unrelated
commands are consumed with an error rather than queued. Arbitrary parallel
command execution is not the current design. The page disables relevant
controls during model/index operations to reduce conflicting requests.

A numeric citation becomes a source-preview link only when it corresponds
to the answer's saved evidence record. Preview requests carry turn index
and citation number; the engine returns exact contextual source excerpts or
a `reference_error` if the reference cannot be resolved.

## Conversations, Stop and recovery

`ConversationStore` manages the local conversation catalog and metadata.
The engine can select, create, branch, rename, archive/delete or reset a
conversation. Existing answer-model sessions and indexes are reused when
switching conversations within the same project, subject to backend state.
The GUI stores answer timing as optional presentation metadata; it is not the
same thing as the diagnostic trace or the core `ConversationTurn`.

**Stop is forceful.** `window.py` kills the child process to interrupt a
blocking backend call; this unloads models and discards unsaved partial
output. Qt starts a new engine, the frontend reloads settings and can
resume the earlier project. After PR3a, `resume` may supply the active
`conversation_id` so the engine restores **that** conversation; without an
ID it uses the latest modified. Invalid requested IDs raise an error and do
not silently open another chat. This is conversation recovery, **not**
continuation of interrupted generation.

The UI may show a remembered partial answer after Stop, but such display
state does not imply a committed `ConversationTurn`. Closing the desktop
window also terminates its owned child; ordinary closing is not a separate
multi-file transaction. The detailed process and error events belong to
[GUI protocol](gui_protocol.md).

## PDF and native capabilities

`gui/_report_pdf.py` is an **optional** Qt-based adapter supplied as a
callback to a report-generating tool; it renders prepared Markdown and
returns a page count. Formatting/word limits and template validation belong
to the core report package. The PDF adapter's attempt at A4/one-page layout
does not guarantee every report is exactly one page; tests verify actual
pagination and metadata separately. The core tool must function without
this callback, saving Markdown.

Other native capabilities include file/folder dialogs, safe workspace-file
opening and drag-and-drop. They should stay in `window.py`/`ChatBridge`, not
inside `rag.py` or tools. The local web page constructs DOM nodes rather
than rendering model-supplied HTML; that reduces one injection route but
does not authenticate untrusted source content.

## Known boundaries and tests

- A local `file://` UI plus QWebChannel is not a web server or remotely
  authenticated API. Protocol JSON currently has no version negotiation or
  comprehensive runtime schema validation.
- Separate event producers exist in the engine and Qt window. Browser and
  backend phase names are partly mirrored; PR3 includes contract tests.
- A `settings` event delivers the validated repository reason-code mapping
  to JavaScript. The frontend should not duplicate these strings.
- Qt/WebEngine, native file dialogs, GPU models and macOS appearance require
  actual platform testing; a Qt-free engine mock alone is insufficient.
- Creating `QCoreApplication` in a test process that later needs
  `QApplication` can crash native PDF tests; isolate optional Qt/WebEngine
  smoke tests in a **subprocess**.

Focused regression tests are under `tests/qachat/gui/`, including protocol,
model, project, source-selection, reference preview, PDF and optional native
smoke suites. General persistence and source-management guarantees are
covered by tests under `tests/qachat/` independent of Qt.

Related: [Architecture](architecture.md),
[GUI protocol](gui_protocol.md),
[Conversation ownership](conversation_ownership.md),
[Security](security.md), [Development](development.md).
