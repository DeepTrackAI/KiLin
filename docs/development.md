# Development, tests, packaging and repository maintenance

## Scope and supported environment

This guide applies to developers working with the current QAChat codebase,
not a separately distributed deployment procedure. QAChat declares
`requires-python = ">=3.12,<3.13"` in `pyproject.toml`. A test run under
Python 3.13 may be useful for early review but **does not establish**
compatibility on the target Python 3.12. Model backends and Qt/PDF behavior
add native/platform requirements that unit-test substitutes cannot cover.

The root [README](../README.md) intentionally remains the short setup and
entry-point guide. This document explains development contracts, the test
matrix, how to inspect changes, and what maintainers should verify.

## Source organization

```text
qachat/                # Python core and optional gui/ package
    project_types/     # Packaged declarations for code and literature
    sources/           # Format-specific text readers/extractors
    retrievals/        # Search, graph, index persistence
    tools/             # Registered built-in tools and their defaults
    gui/               # Optional engine, Qt bridge and web UI

tests/qachat/          # Mirrored core + gui/ + tool tests
assets/                # Maintenance/benchmark notebooks and helpers
tutorials/             # Progressive notebooks, some SVG examples
docs/                  # Architecture and reference documentation
sandbox/               # Local development workspaces; export empty
```

The repository includes 16 numbered tutorial notebooks. They introduce
installation, model cache, generation, repository scanning, extraction,
embeddings, semantic/lexical/code retrieval, indexes, routing, RAG sessions,
trace logging, project types and custom tools. Notebook examples are not a
substitute for production API tests and must not rely on changes to hidden
per-user files. `assets/clean_repository.ipynb` supports a controlled clean
source export; `assets/benchmark_traces.ipynb` analyzes diagnostic traces
without production timing hooks.

## Architectural conventions

- Core behavior must remain callable without constructing Qt or importing
  optional PySide6 from the `qachat` facade.
- A project owns its source snapshot, profile, indexes, outputs/templates,
  conversations and traces. GUI presentation must not silently reinterpret
  saved project settings.
- Package defaults are validated and read-only. A tool owns its `defaults.json`
  and README; shared tool internals are private and not cross-imported from
  sibling tool packages.
- A source reader returns provenance-bearing records rather than running
  user code. Retrieval indexes must be checked against source/model identity.
- Conversation-file ownership and atomic writes are explicit; do not bypass
  the lock to add an apparently harmless rename or metadata update.
- Preserve local operation and user control. No extra downloads, model
  calls, tool execution or persistence changes without agreed behavior.

See [Architecture](architecture.md) and [Security](security.md) for the
cross-layer rationale. The complete current inventory is
[Module map](module_map.md).

## Python and documentation style

Use the established DeepTrack-oriented NumPy docstrings for modules,
classes, methods and functions: purpose, important behavior, parameters,
returns, errors, notes and examples where useful. Multiline docstrings keep
an empty line before the closing triple quotes. Examples for public APIs
should use `qachat`-qualified imports. Comments should explain design
rationale, preconditions or constraints rather than restating an assignment.

Python source follows a strict **79-character** line limit, with the
repository's normal Pylint suppression header retained in test files where
needed. Public exports should follow each module's `__all__` convention.
`private` (`_`-prefixed) modules and names are not made public by describing
them here. The authoritative workflow includes baseline comparison,
preservation of tests and review of design changes **one at a time**.

Documentation belongs in layers: root README for setup; `docs/` for
architecture; module docstrings for API details; tools' own READMEs for
settings and usage; tutorials for executable examples. Link to the owner of
a mutable setting or contract rather than duplicating values in every page.
Editable SVGs are indexed in [Architecture diagrams](diagrams.md) and
maintained alongside the technical documents that explain them.

## Test runners and expected isolation

Run from the repository root in the appropriate environment:

```sh
python -m pytest -q
python -m tests
```

The independent `tests/__main__.py` uses Python `unittest` discovery.
Keep new tests discoverable by **both** runners. Its naming convention is
`test_<module>.py` for a module, `test_<module>__<aspect>.py` for a focused
aspect, and `test__<topic>.py` for a cross-module concern. Tests under
`tests/qachat/gui/`, `tests/qachat/retrievals/`,
`tests/qachat/sources/` and `tests/qachat/tools/` mirror the package.

Unit tests should use isolated temporary files, deterministic fake models
and controlled clocks where appropriate. Never require model downloads for
the base suite. Test model/GUI integrations separately on supported
hardware; optional skips must be counted explicitly rather than described
as passes. A checked-in ZIP must not include transient caches, the contents
of local `sandbox/`, user secrets or weights.

### Qt-specific warning

A Qt test that creates a `QCoreApplication` in the same process can prevent
later tests from initializing a full `QApplication` for PDF rendering; this
previously caused a native Qt abort on macOS. Prefer tests of pure protocol
logic without any Qt application. The optional
`tests/qachat/gui/test__webengine_smoke.py` launches a **separate** process
and runs only with `QACHAT_RUN_NATIVE_GUI_TESTS=1` and appropriate PySide6
support. It cannot be treated as an always-on CI test without a verified
headless WebEngine environment. Avoid sharing application instances between
PDF and WebEngine fixtures.

On macOS, a temporary path under `/var/` can canonically resolve to
`/private/var/`. Compare `Path.resolve()` results when filesystem identity
is intended, and compare literal spellings only when testing path
serialization policy.

## Focused test map

| Changed subsystem | Focused tests (representative) |
| --- | --- |
| Project and user files | `test_project.py`, `test_project_transaction.py`, `test_project_layout.py`, `test_project_files.py`, `test_workspace*.py` |
| Sources and scanning | `test_repository__scan.py`, `test_repository__changes.py`, `sources/`, `test_chunking.py` |
| Retrieval and indexes | `retrievals/`, `test_rag__index_provenance.py`, `test_rag.py` |
| Routing and answering | `test_answering.py`, `test_rag.py`, `test_chat__unit.py` |
| Models | `test_huggingface.py`, `test_runtime.py`, `test_embeddings__unit.py`, `test_chat__models.py` |
| Conversation ownership | `test_conversation_ownership.py`, `test_conversations.py`, `test_trace_logging.py`, GUI ownership tests |
| Tools | `tools/test__architecture.py`, package-specific `tools/<name>/test_tool.py`, `_shared/` tests |
| GUI/protocol | `gui/test__protocol.py`, `gui/test_engine__*.py`, `gui/test_preferences.py`, optional Qt smoke |

This is an orientation list; use the actual test tree and test docstrings for
full coverage, including failure/recovery and invalid-input cases.

## Packaging and source exports

`pyproject.toml` uses setuptools and explicitly ships JSON project types,
core defaults, reason codes, GUI HTML/CSS/JS/assets and tool-specific
defaults/templates. A new packaged data file must be added to the package
data configuration and verified in a **built and installed wheel**, not only
with a local checkout import. Optional native dependencies are split into
`mlx`, `torch`, `gui` and `bench` extras; avoid adding GUI requirements to
the core dependencies merely to make a test run.

For reproducible PR review, compare against the **actual** last approved
archive/commit, not an earlier working directory. Provide a clean full
repository ZIP containing the selected top-level source entries, `.gitignore`
and an explicit empty `sandbox/`, plus (1) a Git patch and (2) a changed-files
ZIP with repository paths and a deletion manifest if necessary.

A source export is not a user-data backup. It should exclude `.DS_Store`,
`__MACOSX`, AppleDouble files, caches, model weights, `sandbox/` contents,
`.env` secrets, generated build files and unselected top-level entries. The
`clean_repository.ipynb` notebook has explicit top-level selection controls;
export should not overwrite an existing ZIP without explicit intervention.

## Pull-request verification procedure

1. Record the exact baseline archive/commit and checksum. Inventory modules,
   tests, current docs and affected API/data contracts.
2. Discuss and approve material behavior, compatibility and ownership
   decisions before implementation.
3. Make the narrowest verified change. Preserve existing tests; for moves,
   compare test classes, methods and meaningful assertions.
4. Run focused tests and both full runners. For structural changes, check
   import boundaries, source style and installed-wheel resource presence.
5. Inspect Python changes for the 79-character/docstring conventions.
6. Validate Markdown links, source module/API mentions, config identifiers,
   diagrams and executable examples against the actual final tree.
7. Build the patch against the exact baseline, apply it in a fresh tree and
   compare the resulting files byte-for-byte.
8. Verify changed-only and full ZIP inventories; report native integrations
   that were not executed rather than claiming universal success.

For a **documentation-only PR**, no production or test assertions should
change. Newly discovered defects belong in review notes or separately
approved corrective PRs. The root README need not be expanded merely
because new technical pages exist.

## Performance and reproducibility

Trace-based benchmarking uses `assets/benchmark_traces.ipynb`, with
supporting analysis in `assets/trace_benchmark.py`. The production trace
system remains as documented in [Trace recording](trace_recording.md).
Measurements need model revision, hardware, backend, project state,
question set, warm/cold session distinction and tracing mode. No arbitrary
CI wall-clock threshold should be used as a functional correctness test.
Optimization work on trace persistence or retrieval remains deferred until
evidence and invariance checks justify it.

Related: [Architecture](architecture.md),
[Configuration](configuration.md), [Security](security.md),
[Performance](performance.md), [Module map](module_map.md).
