# Configuration, defaults and precedence

## Principle: settings have different owners

QAChat intentionally does not have one mutable global configuration file.
Packaged default values, project-type definitions, persistent project
profiles, per-user GUI preferences, tool-specific defaults and per-turn
options have **different lifetimes and authority**. A new default does not
rewrite an existing project's configuration.

| Layer | Source | Scope and owner | Mutability |
| --- | --- | --- | --- |
| Core packaged defaults | `qachat/qachat_defaults.json` | Initial models, project profile, chunking, retrieval, routing, scanner heuristics | Read-only packaged resource; loaded/validated by core |
| Project types | `qachat/project_types/code.json`, `literature.json` | Eligible sources, intents, prompts, routing/retrieval behavior for a kind | Packaged declarations; not editable user profiles |
| Tool defaults | `qachat/tools/<tool>/defaults.json` | Parameter defaults owned by each built-in tool | Packaged and validated by that tool |
| GUI packaged defaults | `qachat/gui/defaults.json` | Appearance and initial GUI model/search preferences | Read-only application resource |
| User GUI preferences | Per-user `preferences.json` | Sparse overrides of GUI defaults, e.g. appearance | Atomic user-owned write |
| Project profile | `<workspace>/profile.ini` | Backend, model IDs, source filters, chunk sizes, search and generation values; GUI effort/router choices | Persistent, authoritative for the opened project |
| Per-answer arguments | `RAGSession.ask(...)` | Retrieval count, routing choice, generation updates, answer guidance | In-memory until saved according to the parameter contract |
| Runtime state | Backend availability, cached revisions, loaded sessions | Selected implementation state | Resolved at execution, not a package default |

Read [Package defaults](package_defaults.md) for **every key** in the core
JSON and its validated type/range. For exact GUI user-preference keys and
current model presets see [GUI README](../qachat/gui/README.md),
`gui/defaults.json` and `gui/preferences.py`. For each tool's parameters
use its README and `_configuration.py`.

## Initialization versus reopening

A new project's kind selects a validated `ProjectType`; the core then
constructs and validates a `ProjectProfile` using `PROJECT_PROFILE` defaults
and the type's source defaults. `initialize_project()` installs declared
tool resources and commits the new `profile.ini` **last**. It must not
rewrite an existing profile when the packaged defaults change.
`open_project()` validates the saved profile and the resolved source path;
it does not opportunistically copy/replace templates based on a new
version of a tool. These invariants are detailed in
[Project workspaces](project_workspaces.md).

`qachat.project.configure_project()` is the core API for validated project
settings. GUI-specific project settings (effort levels, search-depth steps,
router model) are saved in additional `profile.ini` sections by
`qachat.gui.model_settings`. They are not a separate active
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

`qachat.defaults.load_defaults_section()` reads the packaged core JSON via
`importlib.resources` and provides typed validators. Individual modules
validate the sections they own, rather than trusting arbitrary JSON values.
The resource is loaded once per process, not watched for hot reload.
Missing keys, invalid types and incompatible values raise errors; the
validator does not rewrite or repair the shipped resource.

The sections are `DEFAULT_MODELS`, `REPOSITORY`, `ANSWERING`, `RAG`,
`PROJECT_PROFILE`, `CHUNKING`, `PROJECT_TYPES`, and `RETRIEVAL`. The
`DEFAULT_MODELS` section provides initial MLX/PyTorch chat and embedding
identifiers. `PROJECT_TYPES` provides bounded fallbacks for fields omitted
in an intent (for example a default document token budget); it does not
select which intent should answer an arbitrary question.

Changing scanner exclusions or the answer-routing heuristic vocabulary in
this JSON changes future operations and can affect previously created
projects after restart where those global heuristics remain in use. Such
changes require behavior-focused regression tests, even if no file format
changes.

## Project type JSON and intent precedence

The type loader reads `project_types/*.json` in filename order, uses each
filename stem as its kind ID and caches validated records. An invalid
project type is an error, not silently skipped. The top-level object defines
its display metadata, extensions, source size limit, base prompts,
retrieval instruction, code-graph policy, history defaults, default intent,
intents, welcome prompts and suggestions.

An intent can override inherited prompt/history/retrieval settings, specify
shortcut regexes and select a registered tool. The parser gives missing
fields documented defaults (including `search` for ordinary intents and
`none` when a tool is named). It rejects invalid tool or retrieval
combinations. A project type cannot load arbitrary Python code by naming
it in JSON; only registered tool IDs are accepted. See
[Extension interfaces](extension_interfaces.md).

## GUI defaults and writable preferences

`gui.preferences.built_in_defaults()` reads immutable packaged GUI defaults;
`app_defaults()` merges **validated sparse** user preferences over those
built-ins. `write_defaults()` validates and atomically writes deviations to
a user-owned `preferences.json`, leaving the packaged resource unchanged.
Missing user preferences are normal on first run. Invalid values are
reported, not silently substituted with unknown settings.

By default GUI preferences live in a per-user configuration directory;
`QACHAT_DEFAULTS_FILE` is an explicit **absolute-path** override for that
user preferences file (despite its historic name). It does **not** point to
the immutable shipped defaults during normal use.

A recognized development checkout defaults GUI workspaces to its repository
`sandbox/`; an installed package uses a user data folder. Explicit
`QACHAT_WORKSPACES_DIR` (absolute or home-expanded) overrides the default,
and a workspace explicitly chosen in the GUI overrides the suggested
parent. No installed package directory should become the normal writable
location for conversations, workspaces or preferences. See
[GUI architecture](gui_architecture.md) and [GUI README](../qachat/gui/README.md).

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

## File-selection protocol constants

Repository scan reason identifiers live in one validated
[`repository_reason_codes.json`](../qachat/repository_reason_codes.json)
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
