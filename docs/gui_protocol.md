# Desktop GUI command and event protocol

This document specifies the **current** desktop protocol between the Python
engine, Qt bridge and JavaScript page. It is not a version-negotiated
schema or an API compatibility guarantee.

## Ownership and transport

The desktop GUI has three communication boundaries:

```text
web/app.js
  | JavaScript calls ChatBridge slots; receives bridge signals
  v
window.py (Qt/QWebChannel, GUI process)
  | newline-delimited JSON on child stdin/stdout
  v
engine.py (synchronous child process)
  | core KíLín project, retrieval, model, and conversation APIs
```

- `ChatBridge` in `kilin/gui/window.py` exposes **camelCase** Qt slots and
  typed Qt signals. Some operations, such as native file dialogs, opening a
  workspace file, appearance, and Stop, are handled locally rather than being
  passed to the child as commands.
- `ChatWindow._send_command()` queues dictionaries until the child runs;
  `_send_pending()` writes one JSON object followed by `\n` per command.
- `kilin.gui.engine.main()` reads commands from stdin and emits one JSON
  object followed by `\n` per event. The event name is under `event`; the
  command action is under `action`. There is currently **no protocol version
  field** or central runtime message schema.
- `ChatWindow._read_events()` buffers partial stdout, decodes complete lines,
  discards malformed/non-object lines, and relays events through the
  `ChatBridge.event` signal as JSON text. It updates `project_root` when a
  valid `opened` event contains a string `project` field.
- Ordinary engine stdout/stderr during a command is redirected to trace
  `runtime.output` events; it must not become unframed protocol stdout.
  Child stderr is also consumed by Qt as a bounded diagnostic tail. The
  engine serializes emitted events under a lock because progress may be
  reported from helper threads; serialization does not imply fixed progress
  event timing.
- The browser's `onEvent()` parses incoming bridge JSON and branches on
  `data.event`. It currently does not independently validate a message
  against a schema; its `JSON.parse()` errors are not caught there.

For orientation, a command and an event look like:

```json
{"action": "ask", "question": "Explain this file", "effort": "lagom",
 "generation_settings": {"temperature": 0.0}}
```

```json
{"event": "answer_delta", "text": "The file defines..."}
```

## Engine command inventory

The following are the *existing engine dispatcher* actions. Required fields
are required by the dispatcher unless otherwise noted; the Qt bridge may
preprocess JSON strings or validate a subset before constructing the command.
An empty `workspace` is meaningful for new-project settings. All project
paths and source paths remain subject to downstream core validation.

| Action | Required fields | Optional fields / current meaning |
| --- | --- | --- |
| `startup` | None | Find a recent saved project (engine event currently has no browser handler) |
| `readiness` | None | Check selected backend and model availability |
| `install_required_models` | None | Download missing required models |
| `use_installed_models` | None | Substitute available models where supported |
| `settings` | `type_id`, `workspace` | Read project profile or new-project defaults |
| `apply_models` | `settings` | Apply model and search-depth choices |
| `download_model` | `model_id` | Download selected catalog model |
| `saved_project` | `type_id`, `workspace` | Find a matching saved project (event currently has no browser handler) |
| `saved_projects` | `workspace` | List saved projects for Resume |
| `resume` | `project` | Optional `conversation_id` selects that exact saved conversation; without it, the most recently updated conversation is selected |
| `close` | None | Release the active project/models; child remains running |
| `preview` | `sources`, `extensions`, `max_file_bytes` | `workspace` defaults to `""` |
| `open` | `sources`, `type_id`, `workspace` | Create only; `selected_files`, `settings` default to `None`. Blank workspace allocates a fresh default name. |
| `preview_reference` | `turn_index`, `citation` | Look up one citation of a saved turn |
| `ask` | `question` | `retrieval_count`, `generation_settings`, `effort` default to `None` |
| `new_conversation` | None | Create an empty conversation |
| `branch_conversation` | None | Create a branch of the active conversation |
| `select_conversation` | `conversation_id` | Switch to a saved conversation |
| `rename_conversation` | `conversation_id`, `name` | Rename the registered conversation |
| `delete_conversation` | `conversation_id` | Archive/remove it from the visible catalog |
| `add_project_sources` | `sources` | Add confirmed source paths and rebuild as needed |
| `remove_project_sources` | `paths` | Remove confirmed copied project sources |
| `sync_sources` | `decisions` (object of `sources/` path → `import`, `replace`, `keep_both`, `delete` or `keep`) | Apply the user's choice per changed file, leave unlisted files and edited copies untouched, and reindex |
| `edit_folder` | None | `folder`, `operation` default to `""`; other fields forwarded as keyword arguments |
| `rename_project` | `name` | Rename the project |
| `save_answer_timeline` | None | `timestamp` defaults to `""`; `timeline` defaults to `None` |
| `reset` | None | Start a new conversation; do not erase the previous trace |
| `plan_choice` | None in the outer dispatcher | **Special:** consumed while awaiting a tool plan/option; otherwise ignored |

For `resume`, a supplied `conversation_id` must identify an existing
conversation in the given project. An unknown, empty, or invalid explicit
ID produces an `error` event rather than selecting a different chat; no
model is activated. Omitting the field selects the most recently updated
conversation. An `opened` event reports the actual selected ID when
activation succeeds. This restores the active chat after Stop or engine
restart; it does not recover unsaved partial model output.

The page invokes the Qt slots rather than writing these dictionaries itself.
The bridge decodes JSON-encoded arguments for source lists, generation
settings, folder edits, and model settings. `openProject()` additionally
rejects a decoded source selection that is not a list of strings, emitting
`error` directly through the Qt bridge. Other malformed slot arguments can
raise JSON decoding errors in Qt and do not become normal engine commands.

Native file selection, `openWorkspaceFile`, appearance changes, drag/drop
state, and `stopOperation` are *not* additional engine commands. They use
Qt signals, platform dialogs, local file opening, or process control.

### Interactive tool decisions

`ChatEngine.ask()` supplies callbacks to the RAG session when the
child-process command reader is available. A tool can emit an `answer_phase`
with phase `processing` and a `plan` or `choice` payload. The engine then
calls the same stdin iterator **during** that answer and accepts the next
`plan_choice` command.

- Tool plans accept `continue`, `quick`, or `cancel`. Invalid values or EOF
  cancel the plan.
- Option questions accept `option:<zero-based index>` or `cancel`. An
  invalid/out-of-range value or EOF cancels the question.
- Other commands received while waiting emit an `error` asking the user to
  answer the pending decision; they are consumed rather than queued for
  execution afterward.
- An idle `plan_choice` that reaches the ordinary dispatcher is ignored.

This path is why the child must keep *one shared iterator* over stdin.

## Frontend event inventory

Events are JSON objects with an `event` string plus event-specific fields.
The table groups the names consumed by the current JavaScript `onEvent()`.
It records the primary fields, not a complete static type specification:
optional details from core progress callbacks and other contextual fields
can accompany an event.

| Group | Events and representative payloads |
| --- | --- |
| Setup | `readiness` (`backend`, `ready`, `environment_ready`, `required_models`, `cached_models`, `missing_models`, `replacements`, `can_use_installed`); `installed_models_used` (`replacements`, updated model settings) |
| Installation | `installation_started` (`total`, `backend`); `installation_progress` (`index`, `total`, `model_id`, progress details); `installation_model_complete` (`model_id`, elapsed times); `installation_complete` (`elapsed_seconds`) |
| Settings/models | `settings` (`profile`, project types, model choices, backend, search depth, `selection_reason_codes`); `models_applied` (updated levels, models and profile); `model_download_started`, `model_download_progress`, `model_downloaded` (model ID and download status) |
| Source/project | `projects` (`projects`); `preview` (`roots`); `opened` (`project`, `type_id`, `profile`, `turns`, workspace and conversation metadata); `folder_changed` (`folder`, `operation`, `paths`, `workspace_files`); `source_changes` (`added`, `changed`, `deleted`, `edited`), sent after a saved project reopens if a file can be imported, updated or removed |
| Conversation | `conversation_opened` (`turns`, settings, usage, conversation metadata, `workspace_files`); `conversations` (catalog, active ID, `workspace_files`); `conversation_delete_started` (`conversation_id`); `reset` (`archived`) |
| Answer | `status` (`message`); `answer_phase` (`phase` and phase-specific details); `answer_delta` (`text`); `usage_progress` (token counts); `answered` (`turn`, `usage`, `workspace_files`); `progress` (core progress record) |
| Evidence | `reference_preview` (`turn_index`, `citation`, `path`, contextual excerpt details); `reference_error` (`turn_index`, `citation`, `message`) |
| Creation conflict | `workspace_conflict` (`workspace`, `existing_project`, `message`); current session remains active and destination is unchanged |
| Diagnostics | `trace_ready` (`trace_path`, `scope`, `conversation_id`); `error` (`message`, optionally `trace_path`); `backend_failed` (`message`) |
| Process recovery | `stopped` (no required extra fields); `backend_restarted` (`message`, `exit_code`, `diagnostic`) |

The engine also emits `startup` and `saved_project`, but the current
`web/app.js` `onEvent()` implementation has no branches for these names.
The process-recovery events `stopped` and `backend_restarted`, and some
`error` events, originate in `window.py`, not `engine.py`. Diagnostic trace
event names such as `gui.project.settings.saved` are a **separate** trace
schema and are not browser events.

### Source-preview contract

`preview.roots` is an ordered array corresponding to selected input roots.
Each root contains `path`, `files` (selected relative paths), and `skipped`
(entries containing `path`, `reason_code`, `reason`, and `size_bytes`). The page
preserves checkbox choices by path. It makes an explicitly skipped item
selectable only when its machine-readable `reason_code` is `file_too_large`;
selecting it increases the submitted size limit. The human-readable `reason`
is still displayed but is never used to decide file eligibility. All other
skipped entries remain unavailable.

The core `kilin.repository.scan_repository()` owns selection decisions.
The identifiers themselves are defined in the packaged
`kilin/repository_reason_codes.json` and loaded and validated once by
`kilin.repository_reason_codes`. The scanner and Python GUI engine import
the corresponding named constants. Its `entries` contain `reason_code` for
both selected and skipped rows. The supported values are:

| `reason_code` | Selection meaning |
| --- | --- |
| `selected` | File matches the selection |
| `file_too_large` | Eligible file exceeds the configured size limit |
| `empty_file` | File has zero bytes |
| `file_type_not_selected` | File suffix is not selected |
| `protected_file` | Protected filename or environment file |
| `blocked_file_type` | Protected suffix, such as a model weight |
| `excluded_path` | File or directory excluded by the user |
| `outside_scope` | Entry outside the included paths |
| `link_not_followed` | Descendant symbolic link or junction |
| `excluded_directory` | Built-in protected or generated directory |
| `unsupported_entry` | Nonregular filesystem entry |
| `inspection_failed` | An entry or directory could not be inspected |

The `settings` event includes a `selection_reason_codes` dictionary with
exactly the symbolic-name/value pairs from the packaged JSON file. The web
page receives this dictionary before normal preview interaction and uses
`selection_reason_codes.FILE_TOO_LARGE` to identify the sole explicitly
selectable skip reason. It does not fetch local files, hard-code reason values,
or infer eligibility from display messages. Missing or malformed packaged
identifiers are errors; there is no silent fallback vocabulary. Other GUI
settings and project profiles do not override this protocol resource.

The GUI forwards codes from directory-scan reports unchanged. Explicitly
selected individual files bypass filename and suffix filtering, as before;
empty or oversized individual files carry the same `empty_file` or
`file_too_large` code. The text of `reason`, selected-file ordering, and
file-copy behavior remain unchanged. This is an additive protocol field,
without a compatibility fallback to interpreting the English message.

### Answer phases and event ordering

The phase strings used by the web page are the values of
`kilin.rag.ANSWER_PHASE_*`: `routing`, `routed`, `retrieving`,
`evidence_ready`, `generating`, `saving`, `completed`, and `processing`.
`kilin.gui.engine.ENGINE_PHASE_LOADING_MODEL` adds `loading_model` when
switching the MLX chat model. The strings are duplicated as JavaScript
constants and are covered by the Qt-free protocol tests.

A common successful answer can send `status`, several `answer_phase` and
`progress` events, zero or more `answer_delta` and `usage_progress` events,
and a final `answered` event with the saved turn. The exact number of phase
updates and intermediate order **depend on intent, tools, model switches,
and retrieval paths**. `answer_delta` reports batched model text; the
browser's further character-by-character reveal is a presentation effect.
An error may terminate the answer path without `answered`.

`opened` establishes an active project and the initial conversation state;
`conversation_opened` switches the visible saved conversation. Catalog
updates can arrive separately as `conversations`. The page treats the
backend's serialized turn as the authoritative completed answer, not a
partial visible stream.

## Errors, Stop, and restart

- The normal engine command loop reads one JSON object per line. An invalid
  JSON line or non-object command becomes an `error` beginning `The
  background task failed:`; missing required fields produce the same general
  error path. An unknown action produces `Unknown command.`. The loop
  continues afterward. Inside the interactive tool wait, malformed JSON is
  reported as `Unreadable command.` and the wait continues.
- `ChatWindow._read_events()` silently ignores malformed stdout lines or
  dictionaries without an `event` key; the bridge only forwards recognized
  frames. `onEvent()` has no branch for unknown event names.
- Stop is **not** an engine `cancel` command. `ChatBridge.stopOperation()`
  emits `stopRequested`; the Qt window clears queued commands, kills its
  child using `QProcess.kill()`, starts a new child, and emits `stopped`.
  Loaded models are lost. Any unsaved partial answer is not committed as a
  completed turn.
- On unexpected child exit, Qt prepares a `backend_restarted` message and
  starts a replacement child. The page requests settings/readiness again
  and may attempt to reopen the former project. The diagnostic text is
  bounded; these events do not claim recovery of an interrupted operation.
- Closing the window terminates its child and does not issue an additional
  graceful checkpoint transaction.

## Known limitations and deferred changes

1. **Frontend event and phase identifiers are not generated from Python.**
   The test suite characterizes current names and checks phase agreement,
   but a shared schema/code generator is not proposed in this PR.
2. **Structured scan codes do not cover all diagnostics.** Source preview
   eligibility now uses the validated JSON `reason_code` vocabulary, but
   general `error` event messages and some other diagnostic text remain
   human-readable strings without a universal structured error schema.
3. **Malformed payload validation is distributed.** Qt slots, the dispatcher,
   the engine methods, and the page have different failure policies. This
   document describes them; it does not introduce global validation.
4. **Native Qt/WebEngine coverage is opt-in.** The ordinary protocol tests
   cannot validate native page rendering, dialogs or PDF interactions. The
   isolated `test__webengine_smoke.py` can be enabled on a supported GUI
   installation; it is not a default full native acceptance suite.

Implementation owners: `kilin/gui/window.py`, `kilin/gui/engine.py`,
`kilin/gui/web/app.js`; core phases in `kilin/rag.py`. Regression tests:
`tests/kilin/gui/test__protocol.py` and
`tests/kilin/gui/test_engine__resume.py`.

## Project-type identifiers

Project selection consistently uses `type_id`, including `startup`,
`projects`, `saved_project`, `opened`, and `settings.project_types` entries.
Serialized `ProjectProfile` snapshots likewise contain `type_id`. The
`open` command accepts `sources: []` for a new empty workspace only when
`workspace` is explicit; no example paths are inferred. Despite its command
name, `open` creates a new project and never reuses an existing one. `resume`
is the separate operation for opening saved state. On `workspace_conflict`,
`existing_project` is true for any existing profile entry, even an invalid one.
The page may ask whether to resume; confirmation sends `resume` without any
creation settings or pending sources. Cancellation keeps the new-project
inputs available for choosing another destination. Non-project occupied
folders are not offered for resume. The final allocated path is reported by
`opened.project`, not guaranteed by a prior preview. The old project
`kind` protocol key is not accepted. Source-preview rows still use `kind`
for file/folder/link classification; that is a distinct contract.
