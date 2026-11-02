# Development, tests, packaging and repository maintenance

## Scope and supported environment

This guide applies to developers working with the current KíLín codebase,
not a separately distributed deployment procedure. KíLín declares
`requires-python = ">=3.12,<3.13"` in `pyproject.toml`. A test run under
Python 3.13 may be useful for early review but **does not establish**
compatibility on the target Python 3.12. Model backends and Qt/PDF behavior
add native/platform requirements that unit-test substitutes cannot cover.

The root [README](../README.md) intentionally remains the short setup and
entry-point guide. This document explains development contracts, the test
matrix, how to inspect changes, and what maintainers should verify.

## Source organization

```text
kilin/                # Python core and optional gui/ package
    project_types/     # Packaged declarations for code and literature
    sources/           # Format-specific text readers/extractors
    retrievals/        # Search, graph, index persistence
    tools/             # Registered built-in tools and their defaults
    gui/               # Optional engine, Qt bridge and web UI

tests/kilin/          # Mirrored core + gui/ + tool tests
assets/                # Maintenance/benchmark notebooks and helpers
.github/workflows/     # CI tests, package and documentation validation
tutorials/             # Progressive notebooks, some SVG examples
docs/                  # Architecture and reference documentation
_si_article_/          # Paper work, benchmarks and source corpora
    benchmarks/         # Tracked notebooks/gold; empty run workspaces
    code_repo/          # Git-tracked corpus; export directory empty
    literature_repo/    # Git-tracked corpus; export directory empty
    code_analysis/      # Tracked code-analysis manuscript files
    reference_analysis/ # Placeholder .gitkeep for now
sandbox/               # Local development workspaces; export empty
```

The repository includes 16 numbered tutorial notebooks. They introduce
installation, model cache, generation, repository scanning, extraction,
embeddings, semantic/lexical/code retrieval, indexes, routing, RAG sessions,
trace logging, project types and custom tools. Notebook examples are not a
substitute for production API tests and must not rely on changes to hidden
per-user files. `assets/clean_repository.ipynb` supports a controlled clean
source export. Within `_si_article_/`, benchmark sources are included but
benchmark run workspaces and both code/literature corpus directories are
exported empty. The populated `code_repo/` and `literature_repo/` folders
may remain tracked in Git; ZIP exclusion is independent of `.gitignore`.
Neither corpus needs a `.gitkeep`. The `code_analysis/` directory includes
Git-tracked analysis scripts, questions and recorded results. Its scripts
exclude `_si_article_/` from the indexed evaluation corpus by default;
this is separate from the source ZIP export policy. See the
[article guide](../_si_article_/README.md) for commands and paths.
`reference_analysis/` contains only `.gitkeep` for now.
`assets/benchmark_traces.ipynb` analyzes diagnostic traces without
production timing hooks.

## Architectural conventions

- Core behavior must remain callable without constructing Qt or importing
  optional PySide6 from the `kilin` facade.
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
should use `kilin`-qualified imports. Comments should explain design
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
`tests/kilin/gui/`, `tests/kilin/retrievals/`,
`tests/kilin/sources/` and `tests/kilin/tools/` mirror the package.

Unit tests should use isolated temporary files, deterministic fake models
and controlled clocks where appropriate. Never require model downloads for
the base suite. Test model/GUI integrations separately on supported
hardware; optional skips must be counted explicitly rather than described
as passes. A checked-in ZIP must not include transient caches, the contents
of local `sandbox/`, user secrets or weights.

### Debugging tests in VS Code

The shared `.vscode/` settings run tests from the repository root, so
imports such as `from tests.kilin...` resolve. In **Run and Debug**, choose:

- **Debug tests in this file**: runs every test in the open test file and
  stops at its breakpoints.
- **Debug the selected test**: select a test method's name on its `def`
  line first; runs only that test.

Start these from the Run and Debug panel, not the editor's "Run Python
File" button, which runs a test file as a plain script outside the
repository's import path. The Testing panel discovers the same tests.

### Qt-specific warning

A Qt test that creates a `QCoreApplication` in the same process can prevent
later tests from initializing a full `QApplication` for PDF rendering; this
previously caused a native Qt abort on macOS. Prefer tests of pure protocol
logic without any Qt application. The optional
`tests/kilin/gui/test__webengine_smoke.py` launches a **separate** process
and runs only with `KILIN_RUN_NATIVE_GUI_TESTS=1` and appropriate PySide6
support. It cannot be treated as an always-on CI test without a verified
headless WebEngine environment. Avoid sharing application instances between
PDF and WebEngine fixtures.

On macOS, a temporary path under `/var/` can canonically resolve to
`/private/var/`. Compare `Path.resolve()` results when filesystem identity
is intended, and compare literal spellings only when testing path
serialization policy.

## GitHub Actions and Python compatibility

KíLín currently supports **Python 3.12** as declared in `pyproject.toml`.
The `uv.lock` file is scoped to the same interpreter range. CI is divided
into three independent workflows under `.github/workflows/`:

| Workflow | Validation | Required checks |
| --- | --- | --- |
| `tests.yml` | Core tests, newer Python probes, isolated Qt/PDF tests | Python 3.12 on Ubuntu and macOS; Qt/PDF on the same systems (Windows runs but does not block yet) |
| `package.yml` | Wheel and source distribution, isolated wheel installation | Ubuntu / Python 3.12 |
| `docs.yml` | Markdown links, SVG metadata, complete module map | Ubuntu / Python 3.12 |

Windows is a target platform, but support is not fully verified yet.
Its Python 3.12 and Qt/PDF jobs run on every change and report their
results, but `continue-on-error` keeps them non-blocking until the
remaining Windows differences (open-file deletion, home-directory lookup,
and native Qt/PDF behavior) are checked or corrected. Windows then becomes
a required platform.

All workflows run on pull requests and pushes to `main` or `master`; they
can also be launched manually. The test matrix has `fail-fast: false`, so
one platform failure does not cancel the other configurations. A Node.js
runtime is installed for the JavaScript checks in the base test suite.
A newer push to a pull request cancels that pull request's runs still in
progress; pushes to `main` are never cancelled, so each keeps a complete
result.

**Python 3.13 and 3.14 are experimental**, each tested on Ubuntu, macOS
and Windows. These six jobs use `pip --ignore-requires-python` to
probe the current pinned core dependencies without changing the supported
range or `uv.lock`. They require binary distributions rather than building
native dependencies from source. They do not use the Python 3.12 lock file
and are non-blocking, but installation failures, skips and test failures
remain visible in each GitHub Actions job and its summary. Passing such a
probe does not establish support for optional MLX, PyTorch or Qt backends.
Python 3.15 can enter the experimental matrix once the pinned core
dependencies install reproducibly from binary distributions on all three
platforms; its optional backends require separate validation.

To promote a newer Python version, first verify core and optional backend
compatibility, update the package metadata and dependency constraints where
justified, regenerate the lock file for the new range, and make that
version's matrix jobs required. Do not conceal incompatibilities through
unconditional skips or unreviewed dependency upgrades.

The Qt job installs the `gui` extra on Python 3.12 and executes the PDF
adapter and GUI protocol tests in distinct processes. The optional native
WebEngine smoke test stays opt-in; CI does not create a browser session or
download language-model weights. `HF_HUB_OFFLINE=1` prevents unintended
Hub access during core and GUI tests.

The distribution job builds both wheel and source distributions, installs
the wheel into a separate environment, then runs
`tests/setup/check_installed_wheel.py` with `python -I` from outside the
checkout. This checks the installed import path, console entry point and
packaged resources. The documentation job runs
`tests.setup.test__documentation_assets`, which is also discovered by the
two standard test runners. The package workflow uploads build artifacts
for review but never publishes them to a package index.

On a developer machine with Python 3.12 and the locked core dependencies,
use these equivalent commands:

```sh
uv sync --locked --no-dev --python 3.12
uv pip install --python .venv "pytest>=8,<11"
uv run --no-sync python -m pytest -q
uv run --no-sync python -m tests
python -m unittest tests.setup.test__documentation_assets -v
uv build
```

The CI service provides the macOS and Windows environments; a local Linux
run cannot establish compatibility on those platforms. Failures in
experimental Python environments should be assessed as dependency or
runtime incompatibilities before expanding the supported range.

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
`clean_repository.ipynb` notebook includes Git-tracked `.github/`
workflows in its default top-level selection; export should not overwrite an existing ZIP without explicit intervention.

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
