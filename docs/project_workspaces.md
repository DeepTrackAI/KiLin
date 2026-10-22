# Project workspaces and tool resources

KíLín creates and opens project workspaces through **core Python APIs**.
The GUI does not decide which folders or packaged tool resources a project
needs. This document describes the filesystem contract and how to extend it
without adding tool-specific conditions to project creation.

## Workspace structure

![Read-only packaged defaults, writable user preferences and per-project files have separate owners.](figures/workspace-ownership.svg)

*Figure: Defaults, preferences, default workspace locations and persisted project state are intentionally different ownership layers.*

Every initialized project has the following core entries:

```text
project/
    profile.ini
    indexes/
    logs/
    outputs/
    templates/
```

The `templates/` directory **always exists**, including in projects that do
not currently use a templated tool. Users can add Markdown templates or
other resources manually. The `outputs/` directory is a general area for
project-owned results. Neither `outputs/` nor `templates/` is indexed as a
source directory by default. When a project's source is its own root
(`source = .`), KíLín's RAG scans explicitly exclude these managed
folders and the project-owned metadata files `profile.ini`, `sources.json`,
and `source_snapshot.json`; ordinary external source directories and files
with these names are not globally excluded.

Source layout depends on which core API creates the workspace:

- `kilin.initialize_project(path, type_id)` uses an existing folder; it selects
  an existing `sources/`, the type's configured `source_directory`, or
  the project root, in that order, when no saved profile exists. An existing
  profile's explicit source always wins.
- `kilin.prepare_sources_workspace(...)` creates or reuses a cumulative
  `sources/` tree and records imported source provenance in `sources.json`.
- `kilin.prepare_project_workspace(...)` mirrors one original source
  directory inside a managed workspace, preserving its directory name and
  recording `source_snapshot.json`.

Empty managed projects are supported without placeholder sources:

```python
import kilin

project = kilin.prepare_sources_workspace(
    [], kilin.PROJECT_TYPE_IDS[0], "/path/to/new_workspace"
)
project = kilin.prepare_sources_workspace(
    ["/path/to/notes.txt"], project.profile.type_id, project.root
)
```

An explicit workspace location is required when there are no sources.
Creation installs the normal profile, directories and selected tool
resources; `sources/` starts empty. `sources.json` is written when sources
are first imported. Later imports remain cumulative and respect saved
profile overrides. The GUI follows the same core path, without an implicit
example repository. Model loading remains a separate session concern.

The different source layouts are intentional. All three APIs use the same
core initializer for `profile.ini`, shared directories, and resources.

## Initial project creation

`kilin.project` calls `kilin.project_layout.plan_project_layout(type_id)` for
**new profiles**. The planner reads the validated `ProjectType` and
examines its intents. When an intent uses a registered tool, the tool may
provide an optional `workspace_resources(settings)` declaration. The settings
are those already validated by the project's tool registry.

`plan_project_layout` verifies that each declared file belongs to the
registered tool's package, validates its workspace-relative destination,
and deduplicates resources required by multiple intents. Conflicting destinations are rejected rather than
resolved by silent precedence. Planning does not write files.

For new projects, the core first validates the profile contents and resource
plan, creates required directories, installs **only the resources in the
plan**, and prepares the selected source snapshots and provenance. The
`profile.ini` file is then written atomically as the final initialization
step. No tool-specific resource selection occurs in the GUI.

With the current bundled project types:

| Type | `templates/` contents after new project creation |
| --- | --- |
| `code` | Empty directory, ready for user templates |
| `literature` | `research_briefing.md`, `popular_science_summary.md`, `technical_summary.md`, `experimental_protocol.md` |

These four files are selected because the corresponding literature intents
configure `generate_paper_summary` with those template identifiers, not
because the project initializer hard-codes their names.

## Existing projects and user ownership

`kilin.open_project()` validates the existing profile and source path and
ensures the four core directories exist. It **does not reinstall templates**.
Likewise, calling `initialize_project()` on a directory with an existing
profile does not install resources again. As a result:

- Edits and new user-created templates remain untouched on reopening.
- Templates deliberately deleted by a user stay deleted.
- Existing projects with no `templates/` receive an **empty** folder; no
  bundled resources are copied implicitly.
- Existing, unrelated template files are never removed simply because they
  are not mentioned by the current project type.
- A missing selected template causes the report tool to report an error
  rather than silently fall back to the globally shipped version.

An explicit, **opt-in** restoration is possible from the Python API:

```python
from kilin.project_layout import (
    install_project_resources,
    plan_project_layout,
)

install_project_resources(project.root, plan_project_layout(project.profile.type_id))
```

This installs only missing packaged resources; it never overwrites existing
files. Tool authors can offer more specific repair or reset commands later.

The core does not move or delete unrelated folders or generated files
implicitly. An existing `target/` directory is left untouched; `outputs/`
is created independently. No legacy migration is performed.

## Tool resource contract

An independently implemented tool may declare resources by adding an
optional callback to its `Tool` entry in `kilin.tools._registry`. The
callback accepts validated tool settings and returns a tuple of
`kilin.workspace_resources.WorkspaceResource` objects.

For example, the existing report tool declares a packaged file under
`kilin/tools/generate_paper_summary/templates/` and a destination under the project's
`templates/` folder. A custom template name that has no bundled file is
not declared as a packaged resource; its `templates/` folder is still
created, and the user can add the file manually before running the tool.
Declaring a resource does not copy it. The core
performs all installation.

A resource source must be a regular file inside the declaring tool's
package, not a symlink or a file owned by another tool. A declaration's
destination must be a safe, relative POSIX file path.
Absolute paths, traversal components, empty components, Windows drive
paths, and destinations inside source, index, log, or output areas are
rejected. The destination and its parents may not be symbolic links.
Existing files are preserved even during explicit resource restoration.

Every registered tool currently has its own validated `defaults.json`
and README. The `generate_paper_summary` package also owns its shipped
Markdown templates; other tools declare no workspace files at present.

## Explicit imports and source selection

Automatic directory scanning uses the project's configured extension list,
protected-filename rules, and maximum file size. Directly selected source
files may have other extensions or normally excluded names. The core copies
such files without changing `profile.ini`; the managed source manifest
records their explicit selection. The RAG source scan then includes those
regular, size-compliant snapshots even if the ordinary directory scan would
exclude them. Actual extraction may still fail for unsupported or binary
content, and such errors must be reported to the user.

When a source snapshot is deleted through
`kilin.remove_workspace_sources(project, paths)`, only files within a
managed `sources/` folder can be removed. Original source paths are not
modified, and `sources.json` is updated to reflect removed entries. Projects
with external or root-as-source locations cannot use that deletion API.
Index refreshing is a separate core operation.

## Project-owned outputs and templates

The following GUI-independent core functions manage user-owned files:

- `kilin.add_project_files(project, folder, paths, replace=False)` copies
  selected local files or directories into `outputs/` or `templates/`.
  Naming collisions are given numerical suffixes unless replacement is
  explicitly requested.
- `kilin.replace_project_file(project, path, source)` replaces an existing
  regular file in one of those folders.
- `kilin.remove_project_files(project, paths)` removes files or directories
  inside **either** `outputs/` or `templates/`, but cannot remove the managed
  folders themselves.
- `kilin.list_project_files(project)` provides a filesystem inventory,
  including empty folders, without importing the GUI.

All paths are checked for containment and symlink traversal. These
operations do not require a model or reindexing, and the GUI only delegates
the file operations and reports their results.

## Failure behavior and limitations

An invalid or conflicting resource plan fails before creating a new profile.
The first initialization of a project is committed by writing `profile.ini`
**last**, using a temporary file and atomic replacement. If directory,
template, source-copy, or final profile writing fails, KíLín attempts to
remove only entries created by that initialization attempt. Existing user
files are preserved. A process termination can still leave uncommitted
files on disk; without `profile.ini` they do not constitute an initialized
project. Such conflicting paths are not silently removed.

After a project is established, cumulative source additions and later source
synchronization are deliberately **not** all-or-nothing transactions.
Successful earlier imports remain when a later import fails, and provenance
is updated after each successful imported source. Removal likewise updates
provenance to reflect what remains after a partial failure.

These operations are not a filesystem security sandbox and do not defend
against concurrent writers changing paths during an operation. No model is
loaded by initializing folders or managing project-owned files.

The GUI reads the core file inventory. Empty directories are represented
with a trailing `/` rather than invented by the frontend.
