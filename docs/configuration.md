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
| Project types | `kilin/project_types/<type_id>.json` | Eligible sources, intents, prompts, retrieval and memory policies, optional new-project generation defaults | Packaged declarations; not editable user profiles |
| Tool defaults | `kilin/tools/<tool>/defaults.json` | Parameter defaults owned by each built-in tool | Packaged and validated by that tool |
| GUI packaged defaults | `kilin/gui/defaults.json` | Appearance and initial GUI model/search preferences | Read-only application resource |
| User GUI preferences | Per-user `preferences.json` | Sparse overrides of GUI defaults, e.g. appearance | Atomic user-owned write |
| Project profile | `<workspace>/profile.ini` | Backend, model IDs, source filters, chunk sizes, search and generation values; GUI effort choices and core-owned router settings | Persistent, authoritative for the opened project |
| Project behavior | `<workspace>/project_behavior.json` | Sparse project-wide, per-intent, memory and source-index-capability overrides | Optional project-owned validated settings |
| Per-answer arguments | `RAGSession.ask(...)` | Retrieval count, routing choice, generation updates, answer guidance | In-memory until saved according to the parameter contract |
| Runtime state | Backend availability, cached revisions, loaded sessions | Selected implementation state | Resolved at execution, not a package default |

Read [Package defaults](package_defaults.md) for **every key** in the core
JSON and its validated type/range. For exact GUI user-preference keys and
current model presets see [GUI README](../kilin/gui/README.md),
`gui/defaults.json` and `gui/preferences.py`. For each tool's parameters
use its README and `_configuration.py`.

![Packaged defaults and validated project types initialize a new saved profile, while GUI preferences are separate and an existing profile remains authoritative.](figures/configuration-precedence.svg)

*Figure: Packaged creation defaults do not overwrite saved project profiles; application-wide GUI preferences and tool-owned defaults have separate roles.*

## Settings information: explanations, not configuration

The public core API exposes explanatory metadata in
`kilin/settings_info.json`, while each registered tool owns its own
`kilin/tools/<name>/settings_info.json`. These resources describe **all
public configurable settings**, including settings that are not displayed
in the desktop GUI. The files use `format: "settings-info-v1"` and map
canonical dotted setting identifiers to `title`, `summary`, and
`description` strings. They do not store values, accepted ranges,
serialization policy or validation rules.

The core provides three functions independent of GUI, model initialization
and project creation:

```python
import kilin

catalog = kilin.get_settings_info()
entry = kilin.get_setting_info("project.profile.temperature")
tool = kilin.get_tool_settings_info("reference_check")
```

`get_settings_info()` returns one versioned document with a `settings`
mapping. Tool entries have IDs such as
`tools.reference_check.max_checked_claims`; the tool-specific function
returns the tool's own unprefixed lowercase keys. Each read returns a
separate copy of cached resource contents. Metadata parsers reject
malformed or incomplete records, duplicates and tool keys that do not
match that tool's `defaults.json` `SETTINGS` section. Invalid metadata
is an error only when a client explicitly requests metadata: answering,
retrieval, loading a project and validating its settings do not depend
on these descriptions. Repository tests enforce coverage of public
project/profile, behavior, source-card and registered tool settings.

Consumers may render `title` as a field label and `summary` as brief
contextual help; `description` is available for CLI help and documentation.
The GUI uses the public core API via its engine and does not read these
JSON files itself. This resource does not supersede the separately owned
[package defaults](package_defaults.md) or project-specific overrides.

## Project behavior overrides

Each existing project's optional `project_behavior.json` changes only that
project's shared and per-intent behavior. Packaged project types remain
immutable. `profile.ini` still owns backend, model, indexing filters, and
answer settings. `[tool.<name>]` owns shared tool settings.

The behavior file uses the **`project-behavior`** format, a fixed
`project_type_id`, and four required sparse objects: `global`, `intents`,
`capabilities`, and `memory`. Only differences from the packaged defaults
are stored. `capabilities` controls source indexing (`code_graph` and
`document_headers`); `memory` controls conversation-memory indexing,
retrieval and disliked-history exclusion. Intent-local memory settings
are stored inside `intents.<id>.memory`, independently of the same intent's
ordinary routing, tool and retrieval settings. Nested differences merge
into packaged settings before the normal project-type validation.
An explicit `{ "$delete": true }` removes an optional inherited field
when changing retrieval strategy. This is an internal marker, not an
editor value.

The APIs `kilin.effective_project_type()`,
`kilin.project_behavior_settings()` and `kilin.save_project_behavior()`
resolve and edit these settings without depending on the GUI. Saving an
entire global, intent, capability or memory editor computes sparse differences and
checks an optional SHA-256 expected revision to reject stale changes.
Prompts and routing edits apply to subsequent requests without model
reload or index rebuild. New turns preserve original intent history and
retrieval classifications so changes do not reinterpret their behavior.

The additional core editor scopes are `memory` for the complete
project-wide conversation-memory policy and `intent_memory` for one
intent's `indexing`/`read` pair. Existing `global` and `intent` payloads
remain unchanged. The current core validates and persists these settings;
turn UUIDs, feedback persistence and disliked-history filtering are
implemented independently of the editor. Conversation-memory indexing
and retrieval are core operations, independent of the GUI. The desktop
**Memory** tab edits project-wide policy, inspects eligible saved messages,
provides Exclude/Restore, and explicitly synchronizes the derived index
through an idle-phase Update action. The **Intents** tab edits indexing and
reading permissions for each intent. These controls use the core's validated
editors and revision checks; the GUI does not own memory persistence.
See [Conversation memory](conversation_memory.md) for their contracts.

### Source-first retrieval limits

Packaged project types define the source-first search caps per intent:
`max_sources`, `passages_per_source`, `max_passages`, and
`evidence_tokens`. The caps are upper bounds, not quotas: a question
with only a few relevant sources may return fewer passages. Writing
uses larger evidence budgets for drafting, research, and verification;
the two brainstorming types prioritize different behavior. Brainstorming (Creative)
defaults to source-free ideation and searches when its
selected intent calls for evidence. Brainstorming (Grounded) searches
the indexed project sources for every substantive ideation intent.
Code Repository uses hybrid search and the saved project-profile
`retrieval_count` instead of these source-first limits. Larger evidence
budgets may improve coverage but consume more model context and increase
latency; projects can override per-intent limits in Settings → Intents.

The revised Writing Assistant definition now includes Word documents,
LaTeX manuscripts and bibliographies, Python/MATLAB code, and notebooks
among its default source suffixes, alongside PDF, Markdown, and text.
These are creation defaults; existing projects retain their saved
`profile.ini` extension list until explicitly reconfigured.

### Conversation memory policies

Every current project-type JSON defines `conversation_memory` with five
required values and `history.exclude_disliked_answers` with a required
Boolean. Values are deliberately independent of the source-retrieval
pipeline:

| Setting | Accepted values | Purpose |
| --- | --- | --- |
| `enabled` | Boolean | Enables project-local conversation-memory indexing and retrieval |
| `questions` | `all`, `none` | Automatic indexing of eligible user questions |
| `answers` | `none`, `selected`, `all` | Forbid answer indexing, require explicit Save/Like, or index eligible non-disliked answers automatically |
| `max_hits` | Integer 1–20 | Maximum retrieved memory entries per answer |
| `max_tokens` | Integer 128–20,000 | Independent memory-context token budget |
| `history.exclude_disliked_answers` | Boolean | Exclude rejected question/answer pairs from ordinary conversational context, routing, evidence reuse and tools; defaults to true |

The packaged defaults use `questions: all`, `answers: selected`,
`max_hits: 4`, and `max_tokens: 2000` for every project type.
Conversation memory is enabled by default for new `writing`,
`creative_brainstorming` (Brainstorming (Creative)), `grounded_brainstorming`,
and `reviewing` projects and disabled for new `code`
and `literature` projects. Every type defaults to excluding disliked
answers from chat context, independently of memory enablement.

An intent may set `memory.indexing` to `automatic`, `explicit`, or `never`.
`automatic` follows the project-wide content policies; `explicit` requires
Save/Like before indexing the eligible content; `never` forbids indexing,
even after Save/Like. `memory.read` independently controls whether the
intent may retrieve earlier memories. Intents omitting `memory` default
to `automatic` and `read: true`. The built-in small-talk intents use
`never` and `read: false`; file inventories and whole-document overviews
are typically `explicit`, some with memory reading disabled. The reviewing
type has no small-talk intent and retains its strict no-review-writing
role boundary regardless of memory settings.

Source passages, retrieved memories, and recent conversation history
have different purposes. Memory cannot constitute primary evidence or
produce numbered scientific citations; answering integration must place
source evidence and remembered discussions in distinct prompt
sections with independent budgets and provenance.

The policy scopes do not affect the document index compatibility
signature. Changing memory settings will not refresh or invalidate
document/source-card indexes. The behavior format is defined once for this prerelease. A file with an
unknown format raises an error without being rewritten. Existing workspace
data is never silently deleted.

Advanced indexing exposes `code_graph` (true, false, or `"auto"`) and
`document_headers` (true or false). Changes alter index metadata and require
reindexing. In the desktop GUI, failed refreshes restore the previous
behavior configuration and retry the old index. Other index settings and
source-card policy remain owned by the project profile.

## Initialization versus reopening

A new project's `type_id` selects a validated `ProjectType`; the core then
constructs and validates a `ProjectProfile` using `PROJECT_PROFILE` defaults
and the type's file-selection and optional generation defaults. Explicit
creation settings take precedence over those defaults. This initializes
saved state; it is
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
profile settings. The router is separately managed by the GUI-independent
`kilin.model_settings.read_router_model()` and `write_router_model()`
functions, which use the project's `[router]` section. Independently,
`kilin.model_settings.read_source_card_generation_settings()` and
`write_source_card_generation_settings()` own the dedicated card-generation
model and parameters in `[source_card_generation]`. These are core settings,
not a way to change active answering or routing sessions. The GUI maintains
effort levels and search-depth steps in the same `profile.ini` through
`kilin.gui.model_settings`; it no longer owns router persistence. These
settings are not a separate active `effort_levels.json` database. The GUI
still supports controlled migration of recognized older files on first use;
moving or deleting files outside that migration is not authorized.

### Source-card indexing policy

Project-type JSON can declare `capabilities.source_cards` with `enabled`,
`kind`, and `automatic` defaults. The literature type enables automatic
`scientific_paper` cards; other packaged types leave cards disabled. These
indexing choices are independent of `[source_card_generation]`, which
configures the dedicated generation model rather than selecting sources.

Project-local `profile.ini` can override the type defaults:

```ini
[source_cards]
enabled = true
kind = scientific_paper
automatic = true
```

`kilin.source_card_policy.read_source_card_policy()` and
`write_source_card_policy()` validate these values and retain unrelated
profile sections. The project-level enabled switch is authoritative.

Individual source-relative paths can override the default or opt out in
`source_card_sources.json` at the project root:

```json
{
  "format": "source-card-source-settings-v1",
  "sources": {
    "certificates/certificate.pdf": {"enabled": false},
    "papers/study.pdf": {"kind": "scientific_paper"}
  }
}
```

Keys are case-sensitive source-relative paths; absolute or traversing
paths, unknown card kinds, and malformed settings are rejected. Overrides
do not bypass the kind's deterministic and model-assisted eligibility
checks. Cards are reconciled on project opening and source refresh.
Missing/outdated cards are generated automatically only when policy permits;
valid cached cards are reused, and other retrieval channels work without
cards. See [Retrieval pipeline](retrieval_pipeline.md#automatic-source-card-indexing).

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
  resolution, instructions and new-project generation defaults. The saved
  profile determines active models, temperature, thinking, filters and
  indexing settings; `project_behavior.json` may override an existing
  project's prompts, routing and per-intent retrieval behavior.
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

The sections are `DEFAULT_MODELS`, `SOURCE_CARD_GENERATION`, `REPOSITORY`,
`ANSWERING`, `RAG`, `CHAT`, `PROJECT_PROFILE`, `CHUNKING`, `PROJECT_TYPES`,
and `RETRIEVAL`. `DEFAULT_MODELS` provides initial MLX/PyTorch chat,
router, and embedding identifiers. `SOURCE_CARD_GENERATION` provides
independent source-card inference defaults, documented in the
[packaged defaults](package_defaults.md).

`PROJECT_TYPES` provides bounded fallbacks for fields omitted in an intent
(for example a default document token budget);
it does not select which intent should answer an arbitrary question. `CHAT` owns the
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
retrieval instruction, capabilities, history defaults, optional
generation defaults, default intent, intents, welcome prompts and suggestions.

`suggestion_templates` maps intent IDs to one fill-in question, or a list
of them, such as `"paper_overview": "Give me a short summary of {file}."`.
It may use only `{file}`, `{file2}` (a different file), `{topic}`,
`{term}` and `{theme}` (a broad subject of the whole collection), and
must name the type's own intents. Intents that need an
earlier answer or the user's own text (follow-ups, rephrasing, small
talk) have no template.

`kilin/question_suggestions.py` fills them in two steps:

1. **Once per source change:** the router model reads the first two files
   (usually the user's own) and a random mix of ten others, and picks the
   pool of up to three broad themes (one subject of up to four words
   each, such as "Alzheimer's disease"), ten topics and ten exact terms.
   A theme, topic or term is kept only when it is printed in the sources.
   The pool is saved in
   `indexes/suggestion_words.json` and picked again only when the sources
   or the router model change.
2. **Every time questions are needed** (when a project opens, for each new
   empty chat, and for "What can you do for me?"): each template is filled
   up to three times with files, themes, topics and terms chosen at
   random from the project and the pool. No model is called, so this is instant and
   the examples change every time. A question with an unfilled
   placeholder is left out; a code project's questions name only code
   files.

The empty chat picks three intents and one of their questions at random;
clicking one fills the question box, so Enter sends it. Without sources,
or with fewer than three questions, the fixed `suggestions` fill the rest.

`welcome_templates` is a list of fill-in headings for the top of an empty
chat, such as `"What would you like to understand about {theme}?"`, with
the same placeholders. The packaged types use only `{theme}`, so the
heading covers the whole collection, and the specific files and topics are
left to the questions below it. They are filled from the same pool, one
heading per template with the themes taken in a random order, and the
empty chat shows one of them at random, so a new chat may name another
theme. The fixed
`welcome_prompts` are shown instead when the project has no sources, when
no template can be filled, or when the project's own settings change the
welcome prompts, so a user's own headings always win. In short, the fixed
`welcome_prompts` and `suggestions` are the generic examples of a type,
and the templates turn them into prompts about one project's sources.

Every packaged type also has an `assistant_help` intent, routed by
shortcuts such as "What can you do for me?", "help" and "How do I add my
papers?". Its `assistant_guide` tool answers with one of these questions
per intent, so the user sees what each kind of request does with their
own files, and always ends with how to add files (drag them onto
**sources** in the Workspace panel, or attach them with **+** in the
message box) and how to remove one. A project without sources is told
how to add files first and that a pasted passage can be asked about
meanwhile, followed by the fixed `suggestions`. No model is called.
After the answer, the desktop app briefly flashes the **sources** folder
and the **+** button so the user sees where they are.

The fixed `welcome_prompts` and `suggestions` are general, since they are
shown when a project has no sources: the suggestions start with "What can
you do for me?" and "How do I add my files?", which the empty chat always
shows first.

An intent can override inherited prompt/history/retrieval settings, specify
shortcut regexes and select a registered tool. The parser gives missing
fields documented defaults (including `search` for ordinary intents and
`none` when a tool is named). It rejects invalid tool or retrieval
combinations. A project type cannot load arbitrary Python code by naming
it in JSON; only registered tool IDs are accepted. See
[Extension interfaces](extension_interfaces.md).

### New-project generation defaults

A project type can optionally set `generation.temperature` (a finite,
nonnegative number) and `generation.thinking` (a boolean). Each field is
independent. When omitted, the corresponding value comes from the
packaged `PROJECT_PROFILE` defaults. For example:

```json
{
  "generation": {
    "temperature": 0.9,
    "thinking": false
  }
}
```

The six bundled project types explicitly set the following new-project
answer-generation defaults:

| Project type | Temperature | Thinking |
| --- | ---: | --- |
| Code Repository (`code`) | `0.2` | Off |
| Literature (`literature`) | `0.3` | Off |
| Reviewing (`reviewing`) | `0.2` | Off |
| Writing Assistant (`writing`) | `0.6` | Off |
| Brainstorming (Grounded) (`grounded_brainstorming`) | `0.6` | Off |
| Brainstorming (Creative) (`creative_brainstorming`) | `0.9` | Off |

Lower temperatures favor controlled code, factual, and review support;
higher temperatures allow more variety in writing and ideation. The global
`PROJECT_PROFILE` settings remain fallbacks for custom types that omit
these fields. They do not override the explicit bundled type settings.

These settings initialize `profile.ini` only when creating a project.
Explicit creation settings override them. Once saved, a project's own
profile is authoritative; reopening it never re-applies project-type
or packaged creation defaults. An ordinary intent may separately
override thinking for a single direct reply, while tool-internal model
settings remain tool-owned.

### Source-first evidence limits per intent

Ordinary model-answering intents with `retrieval.strategy` set to
`"source_first"` may declare `max_sources`, `passages_per_source`,
`max_passages`, `evidence_tokens`, and `coverage` in their `retrieval`
object. The first two govern source quotas; the latter two are hard
passage and *estimated evidence-token* bounds after context expansion
and related-turn evidence. `coverage` accepts `bounded` (default) or
`exhaustive`; explicit requests about all sources demand exhaustive
source coverage regardless of that default. Omitted sources cause a
saved coverage record and, in exhaustive mode, a deterministic notice.

These settings belong to the **selected effective intent**, not the
answering model's context-token setting or a tool's internal batch
parameters. Missing values have validated core defaults; packaged
literature intent values are documented in
[Retrieval pipeline](retrieval_pipeline.md#per-intent-source-first-configuration).
An existing project can change them through its validated per-intent
`project_behavior.json` overrides, without modifying the packaged type.
Changes affect subsequent retrieval, not previously saved answers.
A tool-backed intent instead uses its tool's validated settings and does
not automatically execute source-first retrieval. See the tool precedence
section below.

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
or add a model call. Packaged `small_talk` intents set `thinking` to
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
The JSON filename stem is the type ID: applications use the IDs returned
by the registry, rather than one Python constant per packaged type. Any
nonempty collection of valid packaged definitions is supported. An unknown
ID or an empty or invalid registry is an error. Definitions are cached:
restart after edits. Callers of the removed `PROJECT_TYPE_ID_CODE` and
`PROJECT_TYPE_ID_LITERATURE` constants should use the corresponding `"code"`
and `"literature"` identifiers, or query the registry for generic selection.

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
shortcuts, cues, required words and intent instructions can change answers
without reindexing. Unit tests should validate each ownership/precedence
boundary and refusal of malformed configuration. Include an installed-wheel test when adding a
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
