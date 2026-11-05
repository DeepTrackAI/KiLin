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
engine.py (sequential dispatcher; separate input-reader thread)
  | core KíLín project, retrieval, model, and conversation APIs
```

- `ChatBridge` in `kilin/gui/window.py` exposes **camelCase** Qt slots and
  typed Qt signals. Some operations, such as native file dialogs, opening a
  workspace file, appearance, and forceful Stop, are handled locally rather
  than passed to the child as commands. Answer cancellation instead sends
  `cancel_answer` to the engine.
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

## Project-type identifiers

Project selection uses `type_id` in `startup`, `projects`, `saved_project`,
`opened` and `settings.project_types` entries. The cross-layer identifier
schema is owned by [Configuration](configuration.md#canonical-identifiers-and-schemas). The
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
| `cancel_answer` | None | **Special:** applied by the input thread as soon as it arrives, so it stops a running `ask` at the model's next token; ends a wait for a tool plan or option; otherwise ignored |
| `new_conversation` | None | Create an empty conversation |
| `branch_conversation` | None | Create a branch of the active conversation; optional `turn_count` keeps only its first turns (default: all) |
| `select_conversation` | `conversation_id` | Switch to a saved conversation |
| `rename_conversation` | `conversation_id`, `name` | Rename the registered conversation |
| `delete_conversation` | `conversation_id` | Archive/remove it from the visible catalog |
| `check_sources` | `sources` | Report what adding these files and folders would copy and leave out, without copying; the page asks the user before `add_project_sources` |
| `add_project_sources` | `sources` | Optional `decisions` maps `sources/` paths to sync actions. Copy eligible new sources, apply supplied choices for existing copies, and reindex once. |
| `remove_project_sources` | `paths` | Remove confirmed copied project sources |
| `sync_sources` | None | Optional `decisions` maps `sources/` paths to `import`, `replace`, `keep_both`, `delete` or `keep`; an explicit map leaves unlisted files untouched. Omitted or `null` decisions use the core automatic policy: update changed unedited copies, remove deleted unedited copies, and import no new files. Reindex afterward. |
| `edit_folder` | None | `folder`, `operation` default to `""`; other fields forwarded as keyword arguments |
| `rename_project` | `name` | Rename the project |
| `delete_project` | `project` | Move that closed project to `.deleted/` beside it (`kilin.project.delete_project`); then send `projects` |
| `apply_indexing` | `settings` | Atomically save validated indexing settings, reopen the project and revalidate its index; send `indexing_applied` with the `profile` only after reopening succeeds |
| `save_answer_settings` | None at dispatcher | `settings` is forwarded (default `None`); the handler requires an object and an open project. Accepted updates are `temperature`, `thinking`, and `max_new_tokens`; other keys are ignored. The GUI additionally sends `request_id` (positive integer) and canonical `project` path to correlate the result. |
| `save_answer_timeline` | None | `timestamp` defaults to `""`; `timeline` defaults to `None` |
| `tool_settings` | None | List the open project's tools and their settings |
| `save_tool_settings` | `tool` | `settings` (object of the tool's changeable settings) saved for the open project |
| `reset` | None | Start a new conversation; do not erase the previous trace |
| `plan_choice` | None in the outer dispatcher | **Special:** consumed while awaiting a tool plan/option; otherwise ignored |

`save_answer_settings` updates the saved project profile and the live chat
settings without asking a question. The Qt bridge accepts either the original
settings JSON object or an envelope containing `settings`, `request_id`, and
`project`, which it forwards as command fields. Correlated GUI requests must
match the active project; the engine never applies a late request to a newly
opened project. The page sends only one such request at a time, retains the
latest subsequent edit, and waits for its acknowledgement before sending the
next. Asking a question is blocked while a save is pending; the input is kept.

Success emits `answer_settings_saved` with the full profile values now in
effect. Correlated successes also return `request_id` and `project`. A
correlated failure emits `answer_settings_failed` with the same identity,
`message`, and `verified`: if true, the engine successfully reread
`profile.ini` and reconfigured the live chat, and returns its authoritative
`temperature`, `thinking`, and `max_new_tokens`. If false, the page discards
pending edits and requires reopening the project before answering. The GUI
ignores stale acknowledgements and discards queued edits after any failure.
Uncorrelated callers retain the prior generic `error` event on failure. An
omitted `settings` field reaches the handler as `None` and is rejected,
not treated as a request to reset defaults.

For `resume`, a supplied `conversation_id` must identify an existing
conversation in the given project. An unknown, empty, or invalid explicit
ID produces an `error` event rather than selecting a different chat; no
model is activated. Omitting the field selects the most recently updated
conversation. An `opened` event reports the actual selected ID when
activation succeeds. This restores the active chat after Stop or engine
restart; it does not recover unsaved partial model output.

`readiness` applies to **new-project creation**, not `resume`. The page can
close Settings and issue `resume` from either saved-project list when the
selected default runtime or models are missing. The engine uses the saved
project profile and lets normal loading report unavailable project-specific
models. A failed resume emits `error` with `resume_released`: `false` when
profile validation failed before the previous session was closed; `true`
when switching had already begun releasing the previous session. For the
latter, the page returns to launch and requests session cleanup rather than
showing the former project as active. New-project preparation remains gated
by successful readiness.

The page invokes the Qt slots rather than writing these dictionaries itself.
The bridge decodes JSON-encoded arguments for source lists, generation
settings, folder edits, and model settings. `openProject()` additionally
rejects a decoded source selection that is not a list of strings, emitting
`error` directly through the Qt bridge. Other malformed slot arguments can
raise JSON decoding errors in Qt and do not become normal engine commands.

Native file selection, `openWorkspaceFile`, appearance and backend
preference changes (saved in `preferences.json` by Qt), drag/drop
state, and `stopOperation` are *not* additional engine commands. They use
Qt signals, platform dialogs, local file opening, or process control.
When `setBackendPreference` successfully persists `auto`, `mlx`, or `torch`,
Qt emits `backend_preference_saved` and sends the existing `readiness` engine
command. A write/validation error instead emits `backend_preference_failed`;
no readiness command is sent, and the page restores the last acknowledged
selection. Unsupported values are ignored. The engine resolves the saved
preference at the beginning of each setup operation, while `auto` alone uses
the platform policy. Existing saved project settings remain independent.

### Indexing-settings failure recovery

The frontend shows an in-progress status until `indexing_applied` confirms
that the project has reopened with the new indexing settings. An `error`
means the change failed, not that the values were saved. Before changing
`profile.ini`, the engine snapshots its complete text. If the operation
fails, it compares the actual file with the snapshot and restores any
changed profile using `atomic_write_text` rather than overwriting the file
in place. This includes failures after `configure_project` published a
new profile but before that method returned. A rejection before profile
publication leaves the current session active.

If the engine already began replacing the session, recovery closes the
partial session and tries to reopen the original project and selected
conversation. Successful recovery can emit `opened` before the final
`error` explaining why the indexing change failed. If the profile cannot
be verified or restored, or the original session cannot be reactivated,
the engine emits `backend_failed` followed by `error`, attempts to close
the session, and asks the user to inspect `profile.ini` and reopen the
project. If even closing fails, the user should restart KíLín. No event
claims that the previous index files were transactionally restored: the
core checks or rebuilds index artifacts on the next successful open.

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
- Other commands received while waiting emit an `error` with
  `pending_choice: true`, asking the user to answer the pending decision;
  they are consumed rather than queued for execution afterward. This
  rejects the unrelated command, not the running answer. The page
  displays the notice but retains the prompt and busy state. Ordinary
  errors without this flag still terminate the pending answer.
- An idle `plan_choice` that reaches the ordinary dispatcher is ignored.

The page attaches a complete plan or choice card before showing the
waiting status, replaces any earlier prompt, and focuses the first
button. It clears queued progress labels both when asking and when
answering, so old waiting labels do not outlive the controls. A card
can submit only once; detached or superseded cards cannot send choices.
Document-selection questions always offer Cancel. Network plans retain
their separate Authorize lookup and Stay offline controls; rendering
or focusing a prompt never authorizes it.

This path is why the child must keep *one shared iterator* over stdin.

## Frontend event inventory

Events are JSON objects with an `event` string plus event-specific fields.
The table groups the names consumed by the current JavaScript `onEvent()`.
It records the primary fields, not a complete static type specification:
optional details from core progress callbacks and other contextual fields
can accompany an event.

| Group | Events and representative payloads |
| --- | --- |
| Setup | `readiness` (`backend`, `ready`, `environment_ready`, `unavailable_reason` (empty unless the selected backend is unsupported), `required_models`, `cached_models`, `missing_models`, `replacements`, `can_use_installed`); `installed_models_used` (`replacements`, updated model settings); `backend_preference_saved`, `backend_preference_failed` (Qt events: saved `backend` or failed `backend` and `message`) |
| Installation | `installation_started` (`total`, `backend`); `installation_progress` (`index`, `total`, `model_id`, progress details); `installation_model_complete` (`model_id`, elapsed times); `installation_complete` (`elapsed_seconds`) |
| Projects folder (Qt) | `projects_folder` (`folder`: effective folder, `standard`: default without a saved choice, `saved`: saved choice or empty string, `missing`: whether the saved directory is unavailable) |
| Settings/models | `settings` (`profile`, project types, model choices, backend, search depth, `selection_reason_codes`, `thinking_switches`: for each checked cached model, whether its chat template switches thinking); `models_applied` (updated levels, models and profile, plus `thinking_switches`); `indexing_applied` (`profile` after the index was rebuilt); `answer_settings_saved` (confirmed `temperature`, `thinking`, `max_new_tokens`, plus `request_id` and `project` for correlated saves); `answer_settings_failed` (`request_id`, `project`, `message`, `verified`, plus reconciled settings when verified); `model_download_started`, `model_download_progress`, `model_downloaded` (model ID and download status); `tool_settings` (`tools`, each with `name`, `summary`, `intents`, `settings`, `type_settings`, `per_intent`; after a save also `tool` and `saved` or `error`) |
| Source/project | `projects` (`projects`); `preview` (`roots`); `opened` (`project`, `type_id`, `profile`, `turns`, workspace and conversation metadata); `folder_changed` (`folder`, `operation`, `paths`, `workspace_files`); `source_changes` (`added`, `changed`, `deleted`, `edited`), sent after a saved project reopens if a file can be imported, updated or removed; `sources_checked` (`sources`; `add` with `name`, `kind`, number of `files` and `state`: `new`, `renamed` for a different source with a taken name, or `existing` when imported from the same path; left-out `files` with `path` and `reason_code`; `skipped_folders` without any file to copy; `changes` of existing sources since they were copied, by group), the answer to `check_sources`, which copies nothing |
| Conversation | `conversation_opened` (`turns`, settings, usage, conversation metadata, `workspace_files`); `conversations` (catalog, active ID, `workspace_files`); `conversation_delete_started` (`conversation_id`); `reset` (`archived`); `conversation_viewed` (`conversation_id`, saved `turns`, or `turns` null when unreadable), sent by Qt for a chat read while an answer runs |
| Answer | `status` (`message`); `answer_phase` (`phase` and phase-specific details); `answer_delta` (`text`); `usage_progress` (token counts); `answered` (`turn`, `usage`, `workspace_files`); `answer_cancelled` (`usage`); `progress` (core progress record) |
| Evidence | `reference_preview` (`turn_index`, `citation`, `path`, contextual excerpt details); `reference_error` (`turn_index`, `citation`, `message`) |
| Creation conflict | `workspace_conflict` (`workspace`, `existing_project`, `message`); current session remains active and destination is unchanged |
| Diagnostics | `trace_ready` (`trace_path`, `scope`, `conversation_id`); `error` (`message`, optionally `trace_path`, `pending_choice`; `resume_released` for saved-project opening failures); `backend_failed` (`message`) |
| Process recovery | `stopped` (optional `removed_projects`: empty creation claims removed; `preserved_projects`: known workspaces left intact); `backend_restarted` (`message`, `exit_code`, `diagnostic`) |

The engine also emits `startup` and `saved_project`, but the current
`web/app.js` `onEvent()` implementation has no branches for these names.
Deleting a conversation moves its JSON into `logs/traces/.deleted/`; the
hidden `.conversation_*.json.lock` sidecar stays at its original path.
`conversation_delete_started` is sent before the archive operation, so an
error may be followed by a refreshed catalog instead of deletion.
The process-recovery events `stopped` and `backend_restarted`,
`conversation_viewed`, `projects_folder`, and some `error` events
originate in `window.py`, not `engine.py`. Diagnostic trace event names such as `gui.project.settings.saved` are a **separate** trace
schema and are not browser events.

### Open-project source-add confirmation

`check_sources` delegates the import plan to
`kilin.preview_project_sources()`. A `sources_checked` event lists only
eligible imports in `add`, with core-allocated destination names (`state` is
`new`, `renamed`, or `existing`). The `sources` array in the plan corresponds
to these eligible imports, even when the user's original selection includes
files that cannot be copied. The GUI keeps the originally requested paths in
the event for confirmation and rechecks the plan on the add command.

Direct files bypass directory suffix and protected-name filters but remain
subject to the project's saved nonempty/file-size rule. Empty and oversized
files appear in `files` with `empty_file` and `file_too_large` reason codes.
Missing, linked, or unsupported paths and conflicting destination names
produce an `error` event rather than a misleading addition preview. Name
collisions within one selection batch use the same hashed destination names
as the importer. The plan is read-only, not a reservation: filesystem changes
between preview and import are checked again by the core.

On resume, `source_changes` describes pending source differences without
modifying snapshots. The user must confirm synchronization; accepted changes
are indexed once afterward. A deleted original explicitly kept by the user
is detached. If that original later returns to the same path, the retained
copy is reported as `changed`, and an explicit `replace` or `keep_both`
choice is needed to reconcile it; the automatic core policy keeps it.
Replacing a source with an empty or oversized original fails before copying.

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
An error may terminate the answer path without `answered`; a `cancel_answer`
ends it with `answer_cancelled` instead.

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
- An answer is stopped with the `cancel_answer` command. A thread in the
  engine reads input while `ask` runs and sets the Stop request that every
  model call of the answer checks after each token, including routing and
  tools. The answer ends with `answer_cancelled`; no turn is saved, and the
  project and loaded models stay open.
- The page's answer Stop calls `ChatBridge.cancelAnswer()`, which sends
  `cancel_answer` and starts the window's `CANCEL_GRACE_MS` timer
  (30,000 milliseconds). An `answered`, `answer_cancelled` or `error` event
  stops it; otherwise the window kills the engine as below. Cancellation
  is a request, not a guarantee that it wins a race with normal completion:
  an already completed answer can still end with `answered`.
- `ChatBridge.stopOperation()` is the forceful Stop for other operations,
  and the fallback when an engine does not respond: it emits
  `stopRequested`; the Qt window clears queued commands, kills its child
  using `QProcess.kill()`, starts a new child, and emits `stopped`. Loaded
  models are lost. Any unsaved partial answer is not committed as a
  completed turn. Stop is conservative about project files. The core has
  already published `profile.ini` when `project_created` (`workspace`) is
  emitted, so a delayed `opened` event does not make that project disposable.
  After stopping the child, Qt can remove only a directory containing its
  `.kilin-creating` marker and no other entries, with the marker naming the
  stopped engine's process ID and recording that it created this folder.
  Markers from older versions and pre-existing empty folders are retained.
  Qt never recursively deletes a project or a populated partial
  workspace based on a PID or queued event.
  `stopped` reports removed empty claims as `removed_projects` and paths
  known to be preserved as `preserved_projects`. The latter may be a
  completed project or partial files requiring inspection; it is not an
  exhaustive inventory if an interruption occurred before any event.
- On unexpected child exit, Qt prepares a `backend_restarted` message and
  starts a replacement child. The page requests settings/readiness again
  and may attempt to reopen the former project. The diagnostic text is
  bounded; these events do not claim recovery of an interrupted operation.
- Closing the window terminates its child and does not issue an additional
  graceful checkpoint transaction.

## Known limitations and deferred changes

1. **Frontend event and phase identifiers are not generated from Python.**
   The test suite characterizes current names and checks phase agreement,
   but no shared runtime schema or identifier generator is implemented.
2. **Structured scan codes do not cover all diagnostics.** Source preview
   eligibility uses the validated JSON `reason_code` vocabulary, but
   general `error` event messages and some other diagnostic text remain
   human-readable strings without a universal structured error schema.
3. **Malformed payload validation is distributed.** Qt slots, the dispatcher,
   the engine methods, and the page have different failure policies. This
   document describes them; it does not introduce global validation.
4. **Native Qt/WebEngine coverage is opt-in.** The ordinary protocol tests
   cannot validate native page rendering, dialogs or PDF interactions. The
   isolated `test__webengine_smoke.py` can be enabled on a supported GUI
   installation; it is not a default full native acceptance suite.

## Change impact and verification

The transport separates Qt-local actions, ordinary engine commands and
commands accepted during an active answer. Preserve that distinction when
adding an action; a handler working in isolation does not prove that the
input thread, interactive wait and frontend agree about when it is usable.

Review `kilin/gui/window.py`, `kilin/gui/engine.py`, `kilin/gui/web/app.js`
and the core phase definitions in `kilin/rag.py` together. Update the command
or event inventory in this page, the affected lifecycle description and
[protocol tests](../tests/kilin/gui/test__protocol.py). Exercise
[exact resume](../tests/kilin/gui/test_engine__resume.py) and relevant
frontend recovery/confirmation tests. The
[source-backed inventory check](../tests/setup/test__documentation_assets.py)
detects missing names, not complete payload validation or native acceptance.

## Related

[GUI architecture](gui_architecture.md), [Configuration](configuration.md),
[Answering lifecycle](answering_pipeline.md), [Development](development.md).
