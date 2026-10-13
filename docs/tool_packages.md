# Modular tools in QAChat

## Ownership and boundaries

QAChat's registered tools are Python subpackages under `qachat/tools/`.
The central registry (`qachat/tools/_registry.py`) connects a stable tool
identifier from a project type to `parse_settings` and `run`. It can also
register an optional `workspace_resources` function. Project-type JSON may
select and configure registered tools, but cannot load arbitrary code.

Each tool owns its implementation, its validated packaged `defaults.json`,
and its local `README.md`. When a tool needs packaged templates, those
resources also live in its package. The workspace planner rejects files
outside their declaring tool package. Only the configured resources are
installed into new project workspaces. The core always creates the
`templates/` folder, including when the selected tools have no templates.

The following files and directories form private infrastructure rather
than tools callable by project types:

- `_contract.py`: request, context, result, and validation contracts.
- `_registry.py`: controlled registration and tool selection.
- `_defaults.py`: cached, defensive loading of tool-owned JSON files.
- `_shared/`: reusable source-processing, conversation, planning, and
  output services. Shared algorithms never import executable tool packages.

Do not import one executable tool to reuse an implementation detail from
another. Extract the reusable operation into a private `_shared/` module
with its own tests. Tool packages may import private shared services,
but shared infrastructure should not depend on a tool's configuration.

## Standard tool layout

```text
qachat/tools/example_tool/
    __init__.py        # Tool's documented exports
    tool.py            # Settings dataclass, parser, runner
    _configuration.py  # Strictly validate defaults on import
    defaults.json      # Tool-owned settings and internal parameters
    README.md          # Explanation of every default
    templates/         # Only when this tool ships resources
```

The reference `generate_paper_summary` tool additionally has a private
report-template parser. Other tools can add private modules as needed.
All ten built-in registered tools use packages, not flat `.py` modules.
No compatibility aliases are maintained for the old flat file locations;
QAChat is a prerelease package.

## Configuration precedence

For each registered tool, the validated `SETTINGS` values in the packaged
`defaults.json` initialize its settings dataclass. A project type may
specify a subset of those settings in its intent's `tool` object; the
settings parser validates those overrides with the same bounds. Missing
settings retain the packaged defaults.

Other sections, such as `ESTIMATES` and `HEURISTICS`, define internal
parameters used by the tool and are not project-type overrides.

`qachat_defaults.json` is independent of these tool-specific defaults.
Allowing project-type definitions to override *core* QAChat defaults is
an intended separate change and is not implemented by this refactor.

Configuration files are loaded once per process. Changing a packaged
file requires restarting QAChat. Incorrect JSON, missing/unknown keys,
invalid types, and values outside the documented constraints fail early.

## Creating a tool

1. Create the package and add its `defaults.json` and `README.md`.
2. Implement validated settings and `run` in `tool.py`.
3. Reuse only the private shared services necessary for the tool.
4. Register the package in `_registry.py`, including a resource provider
   if workspace files must be copied during project creation.
5. Add the tool to a project type intent to make it selectable.
6. Add `tests/qachat/tools/test_<tool_name>.py` with deterministic unit
   tests for settings, rejection of invalid overrides, the registry entry,
   and logic that does not require live language models. Test resource
   declarations if applicable.
7. Include tool-owned JSON, documentation, and resource files in the
   relevant package-data section of `pyproject.toml`. Verify a wheel build.

Tests for shared services belong in `tests/qachat/tools/`, where they can
be exercised independently from every registered tool. Full QAChat
regression tests must pass after every migration. QAChat's coding and
DeepTrack-style documentation conventions apply to all Python modules.
