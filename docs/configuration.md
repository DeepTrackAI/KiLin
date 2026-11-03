# Configuration, defaults and precedence

## Principle: settings have different owners

KíLín intentionally does not have one mutable global configuration file.
Packaged default values, project-type definitions, persistent project
profiles, per-user GUI preferences, tool-specific defaults and per-turn
options have **different lifetimes and authority**. A new default does not
rewrite an existing project's configuration.

| Layer | Source | Scope and owner | Mutability |
| --- | --- | --- | --- |
| Core packaged defaults | `kilin/kilin_defaults.json` | Initial models, project profile, chunking, retrieval, routing, scanner heuristics | Read-only packaged resource; loaded/validated by core |
| Project types | `kilin/project_types/<type_id>.json` | Eligible sources, intents, prompts, routing/retrieval behavior for a project type | Packaged declarations; not editable user profiles |
| Tool defaults | `kilin/tools/<tool>/defaults.json` | Parameter defaults owned by each built-in tool | Packaged and validated by that tool |
| GUI packaged defaults | `kilin/gui/defaults.json` | Appearance and initial GUI model/search preferences | Read-only application resource |
| User GUI preferences | Per-user `preferences.json` | Sparse overrides of GUI defaults, e.g. appearance | Atomic user-owned write |
| Project profile | `<workspace>/profile.ini` | Backend, model IDs, source filters, chunk sizes, search and generation values; GUI effort/router choices | Persistent, authoritative for the opened project |
| Per-answer arguments | `RAGSession.ask(...)` | Retrieval count, routing choice, generation updates, answer guidance | In-memory until saved according to the parameter contract |
| Runtime state | Backend availability, cached revisions, loaded sessions | Selected implementation state | Resolved at execution, not a package default |

Read [Package defaults](package_defaults.md) for **every key** in the core
JSON and its validated type/range. For exact GUI user-preference keys and
current model presets see [GUI README](../kilin/gui/README.md),
`gui/defaults.json` and `gui/preferences.py`. For each tool's parameters
use its README and `_configuration.py`.

![Packaged defaults and validated project types initialize a new saved profile, while GUI preferences are separate and an existing profile remains authoritative.](figures/configuration-precedence.svg)

*Figure: Packaged creation defaults do not overwrite saved project profiles; application-wide GUI preferences and tool-owned defaults have separate roles.*

## Initialization versus reopening

A new project's `type_id` selects a validated `ProjectType`; the core then
constructs and validates a `ProjectProfile` using `PROJECT_PROFILE` defaults
and the type's source defaults. `initialize_project()` installs declared
tool resources and commits the new `profile.ini` **last**. It must not
rewrite an existing profile when the packaged defaults change. All creation
APIs reject explicit occupied destinations; default naming allocates a new
folder. Neither settings overrides nor matching project types authorize reuse.
`open_project()` validates the saved profile and the resolved source path;
it does not opportunistically copy/replace templates based on a new
version of a tool. These invariants are detailed in
[Project workspaces](project_workspaces.md).

`kilin.project.configure_project()` is the core API for validated project
settings. GUI-specific project settings (effort levels, search-depth steps,
router model) are saved in additional `profile.ini` sections by
`kilin.gui.model_settings`. They are not a separate active
`effort_levels.json` database. The module still supports a controlled
migration of recognized older files on first use; moving or deleting files
outside that migration is not authorized.

### Important distinctions

- **Model role vs model ID:** role constants describe purposes such as chat
  and embeddings; default IDs are suggested choices, not an allowlist.
- **Backend vs device:** `auto` selects MLX or PyTorch; PyTorch independently
  chooses CUDA/MPS/CPU based on availability. The GUI dependency bootstrap
  does not override an already-saved project backend.
- **Default chunking vs profile chunking:** direct `chunk_source_units()`
  defaults are not identical to a newly created project's persisted sizes.
- **Project-type defaults vs project overrides:** type JSON affects intent
  resolution, instruction and default scope, while saved profile values
  determine current model, filters and indexing settings.
- **Router/effort/depth:** routing selects an intent, effort chooses an
  answering model as supported by the backend, and search depth chooses the
  number of passages; these are separate controls.
- **Question guidance:** intent-specific answer guidance takes precedence
  over caller guidance and project-type guidance; an explicit empty string
  can suppress project default guidance but not a configured intent rule.

## Core packaged defaults

`kilin.defaults.load_defaults_section()` reads the packaged core JSON via
`importlib.resources` and provides typed validators. Individual modules
validate the sections they own, rather than trusting arbitrary JSON values.
The resource is loaded once per process, not watched for hot reload.
Missing keys, invalid types and incompatible values raise errors; the
validator does not rewrite or repair the shipped resource.

The sections are `DEFAULT_MODELS`, `REPOSITORY`, `ANSWERING`, `RAG`, `CHAT`,
`PROJECT_PROFILE`, `CHUNKING`, `PROJECT_TYPES`, and `RETRIEVAL`. The
`DEFAULT_MODELS` section provides initial MLX/PyTorch chat and embedding
identifiers. `PROJECT_TYPES` provides bounded fallbacks for fields omitted
in an intent (for example a default document token budget); it does not
select which intent should answer an arbitrary question. `CHAT` owns the
language/script vocabulary used in positive CJK output requests; a keyword
mention alone does not grant permission. The per-project `[generation]`
`block_unrequested_cjk` flag controls whether suppression is enabled, with
`PROJECT_PROFILE.BLOCK_UNREQUESTED_CJK` as the fallback for older profiles.

Changing scanner exclusions or the answer-routing heuristic vocabulary in
this JSON changes future operations and can affect previously created
projects after restart where those global heuristics remain in use. Such
changes require behavior-focused regression tests, even if no file format
changes.

## Project type JSON and intent precedence

The type loader reads `project_types/*.json` in filename order, uses each
filename stem as its type ID and caches validated records. An invalid
project type is an error, not silently skipped. The top-level object defines
its display metadata, source-folder convention, extensions, size limit,
base prompts,
retrieval instruction, code-graph policy, history defaults, default intent,
intents, welcome prompts and suggestions.

An intent can override inherited prompt/history/retrieval settings, specify
shortcut regexes and select a registered tool. The parser gives missing
fields documented defaults (including `search` for ordinary intents and
`none` when a tool is named). It rejects invalid tool or retrieval
combinations. A project type cannot load arbitrary Python code by naming
it in JSON; only registered tool IDs are accepted. See
[Extension interfaces](extension_interfaces.md).

### Per-intent thinking policy

An intent may set `thinking` to `true`, `false`, or `null`. Omitting the
field is equivalent to `null`: inherit the loaded chat session's configured
setting. This is inheritance, not automatic reasoning-effort selection.
Strings such as `"false"`, numbers, arrays, and objects are rejected.

For ordinary model answers, an explicit boolean overrides the configured
setting for **that reply only**, including a thinking setting passed in
`RAGSession.ask(generation_settings=...)`. Those generation updates still
persist on the session; the intent override never rewrites them, the project
profile, or GUI preferences. It does not switch models, change token budgets,
or add a model call. Both packaged `small_talk` intents set `thinking` to
`false`; other packaged intents inherit the session setting.

For example, a minimal ordinary intent can declare:

```json
{
  "description": "Greetings and thanks.",
  "system_prompt": "Reply briefly without citing sources.",
  "history": {"policy": "none"},
  "retrieval": {"mode": "none"},
  "thinking": false
}
```

This object belongs under an intent key in the project type's `intents`
object. Tool-backed intents must omit `thinking` or set it to `null`;
explicit booleans are rejected because tools own their internal model calls.
The policy does not control the router or other side requests. Disabling
routing still selects the configured default intent and its policy.

The control uses the selected model's existing `enable_thinking`
chat-template argument. Templates or models that do not support this control
cannot be made non-thinking by this setting; no model fallback or output
filter is added. Restart after editing a packaged project-type JSON because
validated types are cached. Existing workspaces use the updated intent policy
on subsequent requests after restart; saved answers are not rewritten and no
profile migration or reindexing is needed.

Saved turn parameters keep `thinking` as the configured value used to restore
a conversation. Direct model answers additionally record `thinking_effective`
as the resolved value requested for that answer; tool answers do not claim a
single effective thinking policy. See [Model management](model_management.md)
and [Trace recording](trace_recording.md).

### Canonical identifiers and schemas

`ProjectTypeId` is an open string alias, not a closed enum. A validated
`ProjectType` has an `id`; an individual `ProjectProfile` has a `type_id`.
`kilin.PROJECT_TYPE_IDS` contains all packaged IDs in filename order.
`PROJECT_TYPE_ID_CODE` and `PROJECT_TYPE_ID_LITERATURE` name the standard
configurations without requiring their presence. Any nonempty collection
of valid packaged definitions is supported. An unknown ID or an empty or
invalid registry is an error. Definitions are cached: restart after edits.

The current representations are:

| Boundary | Identifier field |
| --- | --- |
| Python `ProjectType` | `id` |
| Python `ProjectProfile` and serialized profile snapshots | `type_id` |
| `profile.ini`, `[project]` section | `type_id = code` |
| GUI commands, project lists and project-type choices | `type_id` |
| Index settings, turn parameters and intent trace metadata | `project_type_id` |
| Project-type JSON files | Filename stem; no duplicate ID field |

This is a clean prerelease change. Former Python project-kind names and
old profile/protocol/metadata keys have no aliases or migration readers.
Recreate development workspaces that use the previous profile schema;
do not expect old index or trace metadata to match the new contracts.
The unrelated `kind` fields on source entities and chunks are unchanged.

### New-project source convention

Every new project keeps its sources in `sources/`, whatever its project type
and however it is created: `initialize_project()`,
`prepare_project_workspace()`, and `prepare_sources_workspace()` all create
it. The name is the core constant
`kilin.workspace_resources.PROJECT_SOURCES_DIRECTORY`; it is not a setting.
Project types decide how sources are read, indexed, and answered, not the
folder name, so a project-type JSON file has no `source_directory` field and
one that still names it is rejected as an unknown setting.

Existing projects keep the `source` recorded in their `profile.ini`, for
example `documents/` or `repository/` from before this convention.
`open_project()` reads that path unchanged and performs no filesystem writes;
no folder is renamed or moved. A single-source mirror made before the
convention keeps its copy under the original's name, and
`synchronize_project_workspace()` updates it there.

## GUI defaults and writable preferences

`gui.preferences.built_in_defaults()` reads immutable packaged GUI defaults;
`app_defaults()` merges **validated sparse** user preferences over those
built-ins. `write_defaults()` validates and atomically writes deviations to
a user-owned `preferences.json`, leaving the packaged resource unchanged.
Missing user preferences are normal on first run. Invalid values are
reported, not silently substituted with unknown settings.

By default GUI preferences live in a per-user configuration directory;
`KILIN_DEFAULTS_FILE` is an explicit **absolute-path** override for that
user preferences file (despite its historic name). It does **not** point to
the immutable shipped defaults during normal use.

A recognized development checkout defaults GUI workspaces to its repository
`sandbox/`; an installed package uses a user data folder. Explicit
`KILIN_WORKSPACES_DIR` (absolute or home-expanded) overrides the default,
and a workspace explicitly chosen in the GUI overrides the suggested
parent. No installed package directory should become the normal writable
location for conversations, workspaces or preferences. See
[GUI architecture](gui_architecture.md) and [GUI README](../kilin/gui/README.md).

### Saved project settings take precedence

Once a project exists, `profile.ini` is authoritative for its saved model,
router, search depth and other project settings. The GUI does not silently
replace them when its packaged/global defaults change. GUI-wide preferences
are initial values for a new project and presentation choices; they are not
permission to reinterpret an existing project's saved sources or tools.

## Tool configuration and resources

`tools/_registry.parse_tool()` reads a project-type tool declaration,
resolves a **registered built-in name**, and calls that tool's own
`parse_settings()` implementation. `tools/_defaults.py` validates
`defaults.json` owned by the package. Unknown settings should fail early.
Tool resource declarations are planned by the core and installed only for a
new workspace (or upon explicit restoration), preserving user-edited files.

The main tool configuration reference is
[Tool packages](tool_packages.md); the full installed resource ownership
contract is [Project workspaces](project_workspaces.md).

### Tool settings per project

A project can change the settings of the tools its type uses, for all of
that tool's intents (`kilin.tool_settings`; in the app, Settings → Tools).
The settings offered for a tool are exactly the `SETTINGS` keys of its
`defaults.json`, so a new tool appears without extra code. A setting the
type's intents set differently, such as `generate_paper_summary`'s
`template`, stays with each intent. The tool's other `defaults.json`
sections are internal to the tool and are not shown or changed.

Changes are saved in the project's `profile.ini`, one `[tool.<name>]`
section per tool, as JSON values, and only where they differ from the
project type's values; saving the type's values removes the section.
The tool's own `parse_settings` checks every change when it is saved and
again before each run, so an invalid value never reaches a tool.

## File-selection protocol constants

Repository scan reason identifiers live in one validated
[`repository_reason_codes.json`](../kilin/repository_reason_codes.json)
resource. Python exposes named constants through
`repository_reason_codes.py`, and the GUI settings event supplies the same
mapping to JavaScript. Human-readable skip messages are **not** protocol
identifiers. The JSON resource is packaged with the wheel; invalid,
duplicate or unexpected entries fail validation. See
[GUI protocol](gui_protocol.md) and [Source pipeline](source_pipeline.md).

## Configuration changes and reproducibility

A reproducible session report needs at least project type, saved profile
(including backend/model IDs, chunk/search values and allow-download flag),
selected source snapshot, loaded model revision/identity, query format,
actual retrieval query and optional trace configuration. An ID in a default
JSON is not proof that a particular model revision ran.

Source selection changes can invalidate manifests; chunk/tokenizer/query
format or embedding-model changes can invalidate retrieval caches; route
shortcuts and intent instructions can change answers without reindexing.
Unit tests should validate each ownership/precedence boundary and refusal
of malformed configuration. Include an installed-wheel test when adding a
new packaged JSON resource.

Related: [Package defaults](package_defaults.md),
[Model management](model_management.md), [Project workspaces](project_workspaces.md),
[Tool packages](tool_packages.md), [Development](development.md).

### Internal tool thinking

Every tool inherits the active answering session's output allowance. This
includes per-question `generation_settings`, whether supplied through the
GUI or the Python API. The allowance is **per call**, includes thinking and
final output together, and is not a target length or an aggregate tool budget.
Tool prompts and report templates still control the requested prose length.

Each tool owns one explicit `MODEL_CALLS` policy in
`kilin/tools/<tool>/defaults.json`:

```json
"MODEL_CALLS": {
  "THINKING": "inherit"
}
```

`THINKING` accepts `inherit`, `on`, or `off`. It controls internal
`context.ask_model` requests; `context.write` uses main-session thinking.
The structured selection tools (`find_passages`, `filter_results`,
`batch_classify`) and the two document-processing tools (`batch_review`,
`generate_paper_summary`) default to `off`. Other tools default to
`inherit`.
Changing this value to `inherit` or `on` is sufficient to reset or force
thinking. There is no short-reply threshold and no token-based mode switch.

Output-limit resolution is deliberately independent:

1. A thinking call uses the active session's maximum, ignoring any local cap.
2. A nonthinking call with no explicit cap also uses the session maximum.
3. A nonthinking call with a cap uses the smaller of that cap and the user's
   maximum. A tool never silently raises the user's selected maximum.

When the session maximum is `None`, inherited calls retain the existing
backend/context-derived budget; a nonthinking local cap still applies.
Invalid caps (including booleans, zero and negative values) are rejected.
No minimum generation length is imposed, and no automatic retry is added.

Paper overviews, source explanations, document notes, report sections and
reference judgments no longer have separate local output ceilings. Variable-
length lists from the three selection tools also inherit the user limit,
even though these tools default to nonthinking. Input chunk, passage, batch
and reading budgets are unchanged and must not be confused with output caps.

The retained short-response exceptions use powers of two:

| Request | Local output cap | When it applies |
| --- | ---: | --- |
| Report headline | 128 | Only when effective thinking is off |
| Report reference identifiers | 128 | Only when effective thinking is off |
| Router intent decision | 64 | Separate nonthinking router |
| Router document selection | 8 | Separate nonthinking router |
| Router term/topic extraction | 128 | Separate nonthinking router |
| Router category or statement extraction | 256 | Separate nonthinking router |
| GUI conversation title | 32 | Separate nonthinking router |

User-supplied limits are not rounded or restricted to powers of two. Router
limits remain separate from the answering model's session ceiling. The
report metadata caps are configurable in its `WRITING` section, and are
bypassed automatically when thinking is enabled.

Temperature, selected models, existing call counts and completion checks
are unchanged. Internal replies remain thinking-free for parsers; generated
thinking is retained separately for display as before. All temporary settings
and isolated histories are restored on success or failure. Models still need
to support the existing thinking control. An exhausted user budget remains
an explicit incomplete result, not a false negative or a successful report.

Duration estimates describe expected output, not maximum generation. The
new `*_REPLY_TOKENS` entries under `ESTIMATES` are used only in plan timing;
they do not control the model. Estimates remain approximate, particularly
when thinking uses more of the selected allowance.

#### Migration from the previous tool budgets

The packaged files are updated together. Customized project-type tool
objects must remove `answer_tokens`, `note_tokens`, and `section_tokens`.
The corresponding `SETTINGS` defaults (`ANSWER_TOKENS`, `NOTE_TOKENS`,
`SECTION_TOKENS`) are removed. They are rejected as unknown settings rather
than silently overriding the interface or becoming ineffective settings.

Remove `SHORT_REPLY_THINKING` and `SHORT_REPLY_MAX_TOKENS` from `MODEL_CALLS`.
Only `THINKING` remains. Selection output caps `find_passages`'
`JUDGE_REPLY_TOKENS`, `filter_results`' `REPLY_TOKENS`, and `batch_classify`'s
`REPLY_TOKEN_LIMIT_PER_ITEM` are removed from those tools' old estimate groups.
Restart KíLín or the notebook kernel to clear the existing configuration
cache. No source reindexing or rewriting of historical answers is needed.

The saved tool summary retains `internal_generation` with the resolved
thinking policy, requested cap and completion status for each internal call.
The model trace records the actual effective generation allowance. See the
[completion contract](answering_pipeline.md#model-output-completion-contract).
