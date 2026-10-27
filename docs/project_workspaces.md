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

## Creation, opening, and explicit updates

**Creating a new project never reuses an existing project or a nonempty
folder.** This rule is enforced in the core, independently of the GUI.

| Operation | Contract |
| --- | --- |
| `kilin.initialize_project(path, type_id)` | Create in a nonexistent or empty directory; create the type's configured `source_directory` subfolder. |
| `kilin.prepare_sources_workspace(...)` | Create a new managed `sources/` tree and record imports in `sources.json`. Never resume a saved workspace. |
| `kilin.open_project(path)` | Read and validate a saved profile and source path without changing files or creating missing folders. |
| `kilin.add_project_sources(path, source_paths, ...)` | Explicit cumulative import using the saved profile. Preserve existing snapshots unless `replace_existing=True` is requested. |
| `kilin.prepare_project_workspace(source, type_id, parent)` | Create a fresh single-source mirror and record `source_snapshot.json`. |
| `kilin.synchronize_project_workspace(source, path)` | Explicitly synchronize that saved mirror. May replace local snapshot edits or remove copies deleted from the original. |
| `kilin.configure_project(path, settings)` | Explicitly validate and update the saved profile. |

An explicit occupied project raises `kilin.ProjectAlreadyExistsError`, a
subclass of `kilin.WorkspaceOccupiedError`. Other nonempty destinations,
partial initialization, linked roots, and in-progress creation are rejected.
Even malformed `profile.ini` entries count as an existing project. Creating
inside another project's directory is also rejected. There is no creation
option that overrides these safeguards. A single-source mirror also rejects
source basenames that would alias managed state, such as `logs`, `indexes`,
`profile.ini`, or the creation marker. Import such folders with the cumulative
`sources/` API instead; their basenames are safe beneath that managed root.

For automatic multi-source naming, omit `workspace_directory` and supply
`workspace_root`. The core derives a base name from the first source's safe
stem and eight-character path hash, then tries `-2`, `-3`, and subsequent
free names. Every existing entry is occupied, including empty folders and
symbolic links. Single-source mirrors use `project_<source-name>` and the
same numbered-sibling allocation. The final name is reserved exclusively;
`default_workspace_path()` merely suggests the base and is not a reservation.

```python
import kilin

first = kilin.prepare_sources_workspace(
    ["/path/to/source"], kilin.PROJECT_TYPE_IDS[0],
    workspace_root="/path/to/projects",
)
second = kilin.prepare_sources_workspace(
    ["/path/to/source"], kilin.PROJECT_TYPE_IDS[0],
    workspace_root="/path/to/projects",
)
assert first.root != second.root
```

Empty managed projects require an explicit destination and contain no
placeholder sources or dummy indexes:

```python
project = kilin.prepare_sources_workspace(
    [], kilin.PROJECT_TYPE_IDS[0], "/path/to/new_workspace"
)
project = kilin.add_project_sources(
    project.root, ["/path/to/notes.txt"]
)
```

Creation installs the normal profile, directories, and selected resources.
`sources.json` is written when sources are first imported. For each
copied file it records the copy's size and modification time at import
(`stamps`), so a later change to the original or to the copy can be told
apart; manifests written before stamps existed remain valid. Later imports
respect the saved profile and never modify it implicitly. Existing source
folders are copied into a new destination rather than adopted in place.
Previously saved root-as-source or external-source profiles remain readable;
opening them does not reinterpret their source layout. Model loading is a
separate session operation.

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
`profile.ini` file is then published atomically, without replacement, as the
final initialization step. No tool-specific resource selection occurs in the GUI.

With the current bundled project types:

| Type | `templates/` contents after new project creation |
| --- | --- |
| `code` | Empty directory, ready for user templates |
| `literature` | `research_briefing.md`, `popular_science_summary.md`, `technical_summary.md`, `experimental_protocol.md` |

These four files are selected because the corresponding literature intents
configure `generate_paper_summary` with those template identifiers, not
because the project initializer hard-codes their names.

## Existing projects and user ownership

`kilin.open_project()` validates the existing profile and source path without
writing anything. It **does not reinstall templates or recreate folders**.
Calling a creation API on that directory raises an ownership error instead
of implicitly reopening it. As a result:

- Edits and new user-created templates remain untouched on reopening.
- Templates deliberately deleted by a user stay deleted.
- Missing folders remain missing on read-only opening. Explicitly call
  `kilin.project_layout.ensure_project_folders(root)` to restore empty core
  folders, and `install_project_resources()` to repair selected resources.
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

An invalid profile or resource plan fails before reservation. The core uses
exclusive filesystem creation to claim a `.kilin-creating` marker and holds
it throughout initialization. Competing processes cannot initialize the same
workspace. An explicit destination is checked again after the marker is
claimed, not just before it. The marker is removed only by its owner.

The profile is published **last** using a same-directory temporary file and
an exclusive hard link. Publication never replaces an existing profile,
including one that unexpectedly appears during initialization. Filesystems
that cannot provide this operation fail closed, without an overwrite fallback.
This is atomic visibility, not a guarantee against power loss.

On failure, the journal rolls back resources created by this attempt; it does
not recursively delete arbitrary contents. A pre-existing empty destination
is retained. A root created by this attempt is removed only if empty; shared
parent directories are not removed. Unexpected unrelated files are preserved.
A process termination may leave the marker and partial resources on disk.
Such a directory remains occupied and requires deliberate inspection/recovery;
creation never guesses that a marker is stale or silently clears the folder.

After a project is established, cumulative source additions and later source
synchronization are deliberately **not** all-or-nothing transactions.
Successful earlier imports remain when a later import fails, and provenance
is updated after each successful imported source. Removal likewise updates
provenance to reflect what remains after a partial failure.

Exclusive creation coordinates cooperating creators, not arbitrary hostile
processes that change files or links while an operation is running. Source
updates are not serialized by the creation marker. These APIs are not a
filesystem security sandbox. No model is loaded by initializing folders or
managing project-owned files.

The GUI reads the core file inventory. Empty directories are represented
with a trailing `/` rather than invented by the frontend.

## Prerelease API change

Old calls that used preparation as an implicit reopen must now choose an
operation explicitly. Replace repeated cumulative preparation with
`add_project_sources(existing_path, sources)`, and repeated mirror preparation
with `synchronize_project_workspace(original, existing_path)`. Use
`open_project(existing_path)` for read-only opening. No compatibility switch
reintroduces implicit reuse. Existing project files and schemas need no
migration for this creation-safety change.
