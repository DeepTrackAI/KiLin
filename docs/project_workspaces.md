# Project workspaces and tool resources

QAChat creates and opens project workspaces through **core Python APIs**.
The GUI does not decide which folders or packaged tool resources a project
needs. This document describes the filesystem contract and how to extend it
without adding tool-specific conditions to project creation.

## Workspace structure

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
(`source = .`), QAChat's RAG scans explicitly exclude these managed
folders; ordinary source directories with the same names elsewhere are
not globally excluded.

Source layout depends on which core API creates the workspace:

- `qachat.initialize_project(path, kind)` uses an existing folder; it selects
  `sources/`, a recognized existing source folder, or the project root as
  the source location according to existing profile rules.
- `qachat.prepare_sources_workspace(...)` creates or reuses a cumulative
  `sources/` tree and records imported source provenance in `sources.json`.
- `qachat.prepare_project_workspace(...)` mirrors one original source
  directory inside a managed workspace, preserving its directory name and
  recording `source_snapshot.json`.

The different source layouts are intentional. All three APIs use the same
core initializer for `profile.ini`, shared directories, and resources.

## Initial project creation

`qachat.project` calls `qachat.project_layout.plan_project_layout(kind)` for
**new profiles**. The planner reads the validated `ProjectType` and
examines its intents. When an intent uses a registered tool, the tool may
provide an optional `workspace_resources(settings)` declaration. The settings
are those already validated by the project's tool registry.

`plan_project_layout` verifies that declared source files exist, validates each
workspace-relative destination, and deduplicates resources required by
multiple intents. Conflicting destinations are rejected rather than
resolved by silent precedence. Planning does not write files.

After the initial profile is written, the core creates the shared directories
and copies **only the resources in the plan**. No tool-specific resource
selection occurs in the GUI or in the generic workspace copier.

With the current bundled project types:

| Type | `templates/` contents after new project creation |
| --- | --- |
| `code` | Empty directory, ready for user templates |
| `literature` | `research_briefing.md`, `popular_science_summary.md`, `technical_summary.md`, `experimental_protocol.md` |

These four files are selected because the corresponding literature intents
configure `generate_paper_summary` with those template identifiers, not
because the project initializer hard-codes their names.

## Existing projects and user ownership

`qachat.open_project()` validates the existing profile and source path and
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
from qachat.project_layout import (
    install_project_resources,
    plan_project_layout,
)

install_project_resources(project.root, plan_project_layout(project.profile.kind))
```

This installs only missing packaged resources; it never overwrites existing
files. Tool authors can offer more specific repair or reset commands later.

On opening an old workspace, the core also moves a legacy `target/` directory
to `outputs/` if `outputs/` does not exist. Only identifiable automatically
generated template guides inside `outputs/templates/` are cleaned up;
manual files are preserved. This migration no longer depends on GUI startup.

## Tool resource contract

An independently implemented tool may declare resources by adding an
optional callback to its `Tool` entry in `qachat.tools.registry`. The
callback accepts validated tool settings and returns a tuple of
`qachat.workspace_resources.WorkspaceResource` objects.

For example, the existing report tool declares a packaged file under
`qachat/tools/templates/` and a destination under the project's
`templates/` folder. A custom template name that has no bundled file is
not declared as a packaged resource; its `templates/` folder is still
created, and the user can add the file manually before running the tool.
Declaring a resource does not copy it. The core
performs all installation.

A declaration's destination must be a safe, relative POSIX file path.
Absolute paths, traversal components, empty components, Windows drive
paths, and destinations inside source, index, log, or output areas are
rejected. The destination and its parents may not be symbolic links.
Existing files are preserved even during explicit resource restoration.

The declaration interface does **not** require any tool to own a separate
`defaults.json` today. Tools can later move into self-contained packages
with private defaults, templates, and code without changing the core
project-creation workflow.

## Failure behavior and limitations

A missing packaged resource or conflicting declaration prevents a new
profile from being created because the resource plan is checked first.
Creating a profile, importing sources, and copying resources are not one
atomic filesystem transaction; a disk or permissions error can leave a
partially initialized project. In that case, verify the workspace before
explicitly repairing missing resources.

The workspace APIs organize project files. They do not provide a security
sandbox for untrusted tools, processes, or concurrently modified directories.
No model is loaded merely by creating project folders.

The GUI reads the actual core folder inventory. Empty directories are
reported with a trailing `/` rather than being invented in the browser.
The global README remains unchanged for this internal lifecycle refactor.
