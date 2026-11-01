# Modular tools in KíLín

## Ownership and boundaries

KíLín's registered tools are Python subpackages under `kilin/tools/`.
The central registry (`kilin/tools/_registry.py`) connects a stable tool
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
kilin/tools/example_tool/
    __init__.py        # Tool's documented exports
    tool.py            # Settings dataclass, parser, runner
    _configuration.py  # Strictly validate defaults on import
    defaults.json      # Tool-owned settings and internal parameters
    README.md          # Explanation of every default
    templates/         # Only when this tool ships resources
```

The reference `generate_paper_summary` tool additionally has a private
report-template parser and a pure report HTML formatter. Qt-based PDF writing
remains an optional private GUI adapter (`kilin/gui/_report_pdf.py`), passed
to the tool as a callback. The tool does not import the GUI, and it
saves validated Markdown even without a PDF backend. Other tools can add private
modules as needed.
All eleven built-in registered tools use packages, not flat `.py` modules.
No compatibility aliases are maintained for the old flat file locations;
KíLín is a prerelease package.

## Generated output safety

Tools that create files use the private shared output service in
`kilin/tools/_shared/outputs.py`. Names must be single, portable basenames;
absolute paths, parent traversal, path separators, and Windows device names
are rejected. Existing filenames and symbolic links are not overwritten.

Markdown and optional PDF reports are written to exclusively created,
hidden staging files inside the project's `outputs/` directory. Complete
files are preferably published through an atomic, no-replace hard link.
Where the filesystem cannot create hard links, an exclusive-create copy
preserves the no-overwrite guarantee, though the copy is not atomic. If a
name is already occupied, the next numbered name is used. Staging files
are cleaned up on both success and failure. Tool writers must not choose
a free name and open that path for ordinary truncating output.

The PDF callback receives a staging filename, **not** the eventual public
filename. A renderer can replace the temporary file's contents; only the
tool publishes the finished report. If PDF rendering fails, the saved
Markdown report remains available.

## Configuration precedence

For each registered tool, the validated `SETTINGS` values in the packaged
`defaults.json` initialize its settings dataclass. A project type may
specify a subset of those settings in its intent's `tool` object; the
settings parser validates those overrides with the same bounds. Missing
settings retain the packaged defaults.

Other sections, such as `ESTIMATES` and `HEURISTICS`, define internal
parameters used by the tool and are not project-type overrides.

`kilin_defaults.json` is independent of these tool-specific defaults.
Allowing project-type definitions to override *core* KíLín defaults is
an intended separate change and is not implemented by this refactor.

Configuration files are loaded once per process. Changing a packaged
file requires restarting KíLín. Incorrect JSON, missing/unknown keys,
invalid types, and values outside the documented constraints fail early.

## Creating a tool

1. Create the package and add its `defaults.json` and `README.md`.
2. Implement validated settings and `run` in `tool.py`.
3. Reuse only the private shared services necessary for the tool.
4. Register the package in `_registry.py`, including a resource provider
   if workspace files must be copied during project creation.
5. Add the tool to a project type intent to make it selectable.
6. Add `tests/kilin/tools/<tool_name>/test_tool.py` with deterministic unit
   tests for settings, rejection of invalid overrides, the registry entry,
   and logic that does not require live language models. Test resource
   declarations if applicable.
7. Include tool-owned JSON, documentation, and resource files in the
   relevant package-data section of `pyproject.toml`. Verify a wheel build.

Tests for shared services belong in `tests/kilin/tools/_shared/`. They
can be exercised independently from every registered tool. Full KíLín
regression tests must pass after changes to shared services. KíLín's coding
and DeepTrack-style documentation conventions apply to all Python modules.

## Generation settings for tools

The `ask_model` and `write` adapters use the active session's model,
temperature and output allowance. `write` inherits session thinking;
`ask_model` uses the tool's explicit `MODEL_CALLS.THINKING` policy.
Thinking requests always inherit the user-selected token limit. Only
nonthinking requests may supply a smaller cap, which cannot exceed the
user's maximum. Pass `None` for ordinary analysis, prose, or variable-length
structured output; reserve explicit caps for genuinely brief responses.
Do not infer thinking from the cap or use duration estimates as limits.

The allowance is per model call, includes thinking and final output, and is
not a target response length. Source-input budgets and template word limits
serve different purposes. See [configuration](configuration.md#internal-tool-thinking)
for defaults, retained exceptions, and removed configuration keys.

The returned text differs intentionally: `ask_model` removes thinking
from the value it passes to tool logic, so classification, reference
selection, and other structured parsing remain reliable. The coordinator
captures any thinking from these calls separately, shows it in the answer's
collapsible thinking section, and saves it with the turn. The captured
thinking is also forwarded to the answer callback after each call completes.
A tool using `write` preserves and streams the model's full final reply.
The internal adapter may retry an exhausted, unfinished thinking call once
without thinking. It keeps the original prompt and user allowance and records
both attempts. Final streaming `write` calls are not retried automatically.

The `ask_router` adapter is unaffected: router side questions remain
nonthinking, deterministic, and isolated from the conversation. All tool
side calls restore the original conversation history after success or
failure.

## Partial tool completion

Keep recovery at two existing boundaries: `RAGSession` owns the single
nonthinking retry for an internal `ask_model` request; each tool owns whether
independent items can continue after `IncompleteModelResponse`. Do not add
per-tool generation retries or parse truncated text. Use the shared
`partial_result` helper to retain valid results with safe item labels, reasons,
and a visible **Incomplete items** notice. The summary is partial when useful
work remains, incomplete otherwise. Never turn a failed read into `NOTHING`, a
failed selection into `NONE`, or a failed evidence judgment into a negative.

Report placeholders are visible omissions, not source findings, and must not
feed later model requests. Partial reports have distinct Markdown filenames,
no PDF and explicit invalid/partial status. Keep normal template and evidence
checks. See [completion and recovery](answering_pipeline.md#model-output-completion-and-bounded-recovery)
for exact retry eligibility, accounting, persistence and dependency behavior.
