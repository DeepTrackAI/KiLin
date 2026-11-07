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
and the type's file-selection defaults. This initializes saved state; it is
not permission to overwrite an existing profile or reinstall user resources.
Creation, read-only reopening and explicit updates are different operations,
owned by [Project workspaces](project_workspaces.md#creation-opening-and-explicit-updates).

In the GUI, readiness for the selected default backend and required models
blocks **new-project creation only**. Setup opens automatically to explain
missing requirements but can be dismissed. Existing projects are offered for
Resume regardless of this global readiness: reopening uses their persisted
backend and model IDs, then reports project-specific loading failures without
silently substituting current creation defaults.

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
- **Backend vs device:** `auto` selects MLX on native Apple Silicon and
  PyTorch elsewhere; an explicit `mlx` or `torch` user preference is never
  silently replaced when setup checks models, suggests substitutes, or
  installs missing models. A missing runtime is reported as unavailable.
  PyTorch separately chooses CUDA/MPS/CPU. New-project GUI preferences do
  not override an already-saved project backend.
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
its display metadata, extensions, size limit, base prompts,
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

Former Python project-kind names and old profile/protocol/metadata keys
have no compatibility aliases or migration readers. Workspaces using that
previous profile schema must be recreated; old index or trace metadata
must not be assumed to match this schema. This restriction is distinct from
opening a current-schema profile with an older source-folder name. The
unrelated `kind` fields on source entities and chunks retain their meaning.

### New-project source convention

Source-folder ownership is fixed by the core, not a project-type setting.
See the [source-directory convention](project_workspaces.md#source-directory-convention)
for new projects and the handling of previously saved source paths.

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
`sandbox/`; an installed package suggests `KíLín` under the OS Documents
location (Linux XDG user directories; Windows registered Documents;
`~/Documents` on macOS or when the location cannot be resolved). The user
must confirm or browse to a directory. Explicit `KILIN_WORKSPACES_DIR`
(absolute or home-expanded) takes precedence over the GUI's saved folder;
a workspace explicitly selected for one project overrides the suggested
parent. No installed package directory should become the normal writable
location for conversations, workspaces or preferences. Documents may be
synchronized by iCloud, OneDrive or another external service; users needing
strictly local project storage should choose a non-synchronized path.

If the previous installed-app `workspaces/` data directory contains project
profiles, GUI setup offers to reuse that folder. Choosing it records a
folder preference without moving or rewriting any project files. Otherwise
KíLín keeps the proposed Documents folder. An existing confirmed choice or
environment override is never silently replaced by legacy detection. See
[GUI architecture](gui_architecture.md) and [GUI README](../kilin/gui/README.md).

### Saved project settings take precedence

Once a project exists, `profile.ini` is authoritative for its saved model,
router, search depth and other project settings. The GUI does not silently
replace them when its packaged/global defaults change. GUI-wide preferences
are initial values for a new project and presentation choices; they are not
permission to reinterpret an existing project's saved sources or tools.

## Tool configuration and resources

Tool settings have three ordered layers:

| Layer, lowest to highest precedence | Validation and scope |
| --- | --- |
| Packaged `SETTINGS` in the tool's `defaults.json` | The tool initializes and validates its settings dataclass. |
| The selected project-type intent's `tool` object | `tools._registry.parse_tool()` resolves a registered name and validates overrides through that tool's `parse_settings()`. |
| Saved `[tool.<name>]` values in the project profile | `tool_settings.with_project_settings()` applies and validates project overrides before execution. |

An omitted setting keeps the preceding layer's value. The implementation
owners are `tools/_registry.py`, the tool's `_configuration.py`/`tool.py`,
and `tool_settings.py`; the GUI exposes these core operations rather than
inventing a fourth settings layer.

### Tool settings per project

A project can change settings shared by all intents using a tool
(`kilin.tool_settings`; in the app, Settings -> Tools). Only keys in the
tool's packaged `SETTINGS` whose effective values agree across those intents
are offered. A value that differs by intent, such as a report template,
stays with that intent. Saved changes therefore apply to all eligible uses
of that tool, not just the currently selected question.

Changes are JSON-encoded in `[tool.<name>]` sections of `profile.ini`, and
only differences from the project type's values are stored. Saving all of
those values removes the tool's override section; resetting a subset removes
only those overrides and preserves the others. The tool parser validates
updates when saved and again before execution. Unknown keys and invalid
types or bounds fail rather than being silently ignored.

### Internal tool thinking

`MODEL_CALLS`, `ESTIMATES`, `HEURISTICS` and other non-`SETTINGS` sections are
internal packaged configuration, not per-intent or saved project overrides.
They are not offered by the tool-settings API. The exact thinking policy,
output-limit resolution, retained short-response exceptions and customized
configuration compatibility notes belong to
[Generation settings for tools](tool_packages.md#generation-settings-for-tools).

Packaged files are cached; restart the application or notebook kernel after
editing them. Timing estimates do not configure generation limits. Core
`kilin_defaults.json` and a tool's defaults are separate resources: a tool
setting cannot override the core JSON.

### Tool resources

A validated tool declaration may request packaged workspace resources.
Planning and installation belong to the core, not the GUI or settings form.
The [workspace resource contract](project_workspaces.md#tool-resource-contract)
defines safe destinations, creation-time installation and explicit repair
without replacing user edits.

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

### Change impact and verification

The separation of creation defaults, saved settings and per-call policy is
what makes a reopened project reproducible. A configuration edit must identify
which lifetime it affects, whether a restart is needed, and whether index
identity changes. Do not treat a new default as an implicit migration.

Inspect the owning validator and every profile/GUI consumer together. Test
[project defaults and profiles](../tests/kilin/test_project.py),
[project-type contracts](../tests/kilin/test__project_type_contract.py),
[tool overrides](../tests/kilin/test_tool_settings.py) and
[GUI generation settings](../tests/kilin/gui/test__generation_settings.py).
The [packaged-default documentation check](../tests/setup/test__documentation_assets.py)
covers literal values, not all precedence behavior. New resources also need
[installed-wheel verification](development.md#packaging-and-source-exports).

## Related

[Package defaults](package_defaults.md),
[Model management](model_management.md), [Project workspaces](project_workspaces.md),
[Tool packages](tool_packages.md), [Development](development.md).
