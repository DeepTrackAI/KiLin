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
    project_types/     # Validated packaged project type definitions
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

The repository includes 19 numbered tutorial notebooks. They introduce
installation, model cache, generation, repository scanning, extraction,
embeddings, semantic/lexical/code retrieval, indexes, routing, RAG sessions,
trace logging, project types, custom tools, and an end-to-end research
assistant capstone. Notebook examples are not a
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

`tests/setup/test__documentation_assets.py` runs using only the standard
library, without importing the application or loading models. It checks
local links and Markdown heading fragments, SVG metadata, module coverage
and owner totals, GUI action/event inventories, workspace API exports,
registered-tool references, JSON snippets and documented core defaults.
Organization checks also guard terminal related-links sections, evergreen
prose/diagram labels, complete index navigation and actionable change-impact
links. They are not a semantic deduplication or factual-proof engine.
These checks detect structural drift; they do not prove prose semantics or
native GUI behavior. Review the implementation and focused runtime tests
when changing a contract.

### Maintaining an integrated reference

Choose the [primary contract](README.md#contract-ownership) before editing.
Read its source implementation, validators, callers and failure tests;
resolve discrepancies in favor of the executable code and log suspected
implementation defects separately. Comments, an earlier PR description or a
passing structural test do not override the current behavior.

Integrate a new rule where the lifecycle reaches it: configuration with
precedence, recovery with failure handling, and persistence with commitment.
Do not append a feature below the final related-links section. Preserve the
conditions, exclusions, failure residue and compatibility action when moving
text; a shorter summary is not a substitute for a lost edge case.

Keep detailed normative rules in one owner page and use short local context
plus a link elsewhere. Mutable values belong beside their actual validated
resource, not in repeated global tables. State why a boundary matters, what
must remain true and which producers/consumers/tests change together. Label
proposals and unsupported guarantees separately from implemented behavior;
keep historical notes only when they tell a current caller what to do.

After restructuring, check heading anchors and inbound references, examples,
module ownership, tests and editable diagrams. Inspect the diff for deleted
conditions, not just broken links. Human/agent reviewers should use the
[change-impact navigator](README.md#change-impact-navigator) as a starting
point, then confirm the current tree rather than treating the map as exhaustive.

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

Preserve a private production module's leading underscore: `_matching.py`
is covered by `test__matching.py`. Prefer private, non-discovered support
modules for shared fixtures rather than importing another test module.
For example, `tests/kilin/_project_type_support.py` supplies independent
project-type definitions to the loader and cross-module contract tests.

Shared test adapters belong in private, non-discovered support modules.
`tests/kilin/gui/_frontend_support.py` owns the existing top-level
JavaScript function extraction and isolated Node subprocess invocation;
individual GUI tests still check their own event/DOM behavior. The
extractor intentionally uses the existing source terminator, not a full
JavaScript parser. Prefer ``extract_function()`` to repeated manual
function slicing. ``run_node()`` accepts optional ``args`` appended
after Node's ``-e`` script (``process.argv[1:]``), plus a timeout and
environment overrides. Tests retain their own Node-availability skip
gates and assertions. Prefer behavioral tests over additional
source-text checks when a future frontend refactor is considered.
`tests/kilin/tools/_contract_support.py` owns common tool registry and
packaged-default assertions; each package retains its own tests for
invalid settings and tool-specific behavior.

The large RAG coordinator tests are organized by contract:
`test_rag.py` retains records, evidence, and coordinator helpers;
`test_rag__retrieval.py`, `test_rag__answering.py`,
`test_rag__conversation.py`, and `test_rag__trace.py` exercise focused
subsystems. The private `tests/kilin/_rag_support.py` holds deterministic
session, evidence, intent, and model-patching fixtures used by those tests
and the existing focused RAG suites. Do not import fixtures from
`test_rag.py`: discoverable test modules are not fixture libraries.

Frontend behavior checks should run the actual JavaScript function or
handler with a minimal Node-based Qt/DOM substitute when practical. Source
text checks remain appropriate for structural contracts, such as event
registration, but should not stand in for observable behavior. Neither
isolated Node tests nor mocked engine tests replace the optional native
WebEngine smoke test.

Unit tests should use isolated temporary files, deterministic fake models
and controlled clocks where appropriate. Never require model downloads for
the base suite. Test model/GUI integrations separately on supported
hardware; optional skips must be counted explicitly rather than described
as passes. A checked-in ZIP must not include transient caches, the contents
of local `sandbox/`, user secrets or weights.

### Selecting tests and diagnosing slow runs

For a short local change cycle, select the directly affected module and
its integration boundary, for example:

```sh
python -m pytest -q tests/kilin/test_rag__retrieval.py
python -m pytest -q tests/kilin/gui/test__source_add_frontend.py
```

Run **both complete runners** before delivering a PR; a selected subset is
not a substitute for their discovery and integration checks. To identify
slow individual tests in a given environment, run:

```sh
python -m pytest -q --durations=20
```

Inspect the reported test names and any optional backend skips before
changing setup or introducing caches. Duration results are measurements,
not stable pass/fail thresholds. For macOS and Python 3.12, rerun the
full suite and native/real-model gates separately; Linux measurements and
mocked GUI tests do not establish native platform performance.

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
into five independent workflows under `.github/workflows/`:

| Workflow | Validation | Required checks |
| --- | --- | --- |
| `tests.yml` | Core tests, newer Python probes, isolated Qt/PDF tests | Python 3.12 on Ubuntu and macOS; Qt/PDF on the same systems (Windows runs but does not block yet) |
| `package.yml` | Wheel and source distribution, isolated wheel installation | Ubuntu / Python 3.12 |
| `docs.yml` | Local links/anchors, SVG metadata, source-backed inventories and examples | Ubuntu / Python 3.12 |
| `macos_app.yml` | Real Apple Silicon .app/.dmg build, signature, mounted-image inspection and installer regression tests | macOS / Python 3.12, ad-hoc signing |
| `windows_app.yml` | Real Inno Setup build, silent install, runtime and shortcut checks, uninstall | Windows x64, unsigned installer smoke |

Windows is a target platform, but support is not fully verified yet.
Its Python 3.12 and Qt/PDF jobs run on every change and report their
results, but `continue-on-error` keeps them non-blocking until the
remaining Windows differences (open-file deletion, home-directory lookup,
and native Qt/PDF behavior) are checked or corrected. Windows then becomes
a required platform.

All workflows run on pull requests and pushes to `main` or `develop`; they
can also be launched manually. Tag pushes do not currently start a separate
publishing or release-verification workflow. The test matrix has
`fail-fast: false`, so
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
for review but never publishes them to a package index. The macOS
application workflow can upload a development DMG on manual dispatch; it
never uses Apple Developer credentials or publishes a public release.

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

The [change-impact navigator](README.md#change-impact-navigator) pairs common
changes with their primary contract and concrete regression suites. Each
owner page gives narrower invariants and failure checks. Use the actual test
tree and test docstrings to expand that selection; neither the navigator nor
a passing focused subset establishes complete coverage.

## Packaging and source exports

`pyproject.toml` uses setuptools and explicitly ships JSON project types,
core defaults, reason codes, GUI HTML/CSS/JS/assets and tool-specific
defaults/templates. A new packaged data file must be added to the package
data configuration and verified in a **built and installed wheel**, not only
with a local checkout import. Optional native dependencies are split into
`mlx`, `torch`, `gui` and `bench` extras; avoid adding GUI requirements to
the core dependencies merely to make a test run. The `app` extra is what
users install: Qt plus the platform's backend, chosen by environment
markers. `kilin-gui` is a GUI script, so Windows opens it without a
console.

### Public prerelease sequence

The first public beta is `0.1.0b1`, then `0.1.0rc1`, then stable `0.1.0`.
Use matching Git tags: `v0.1.0b1`, `v0.1.0rc1`, `v0.1.0`. If further
beta or RC iterations are needed, increment their sequence numbers. Do
not upload `0.1.0` to PyPI until it is the intended stable release.
Public prereleases remain subject to the security and platform limits in
[Release guidance](release.md); a public package is not a signed native app.

1. On the **exact release commit**, update the package version in
   `pyproject.toml`, synchronize the root entry in `uv.lock`, and match
   `CITATION.cff`. Keep the `Repository` project URL and all public README
   links on `https://github.com/DeepTrackAI/KiLin`.
   Confirm the public repository exists and its raw logo URLs load:
   `https://raw.githubusercontent.com/DeepTrackAI/KiLin/main/assets/brand/`.
   Verify the artwork's redistribution rights before making it public.
2. Run both complete test suites on Python 3.12, the formatting checks,
   documentation checks, and the package/installation CI workflows. Build
   the distributions with `uv build`; review their contents and metadata.
   Install the wheel into a fresh Python 3.12 environment using
   `uv pip install "dist/kilin-<version>-py3-none-any.whl[app]"`, and open
   `kilin-gui`. Run `twine check dist/*` when Twine is available. This
   checks rendering syntax, not whether external image URLs resolve.
3. Merge the reviewed commit to `main`, verify the GitHub checks against
   that commit, and create the matching `v<version>` tag. Create a GitHub
   release marked **prerelease** for `b` and `rc` versions. Do not mark
   the stable `0.1.0` tag as prerelease. Keep tags pointing at the tested
   commit; this repository does not currently publish automatically on
   a tag push.
4. Publish the two **reviewed** files to the existing PyPI `kilin`
   project with an account authorized to publish it:

   ```sh
   uv publish dist/kilin-<version>-py3-none-any.whl dist/kilin-<version>.tar.gz
   ```

   Provide a project-scoped PyPI API token via the publishing tool's
   documented secure input method; do not commit tokens. A released PyPI
   version cannot be re-uploaded with different files, so publish any fix
   under a new version. TestPyPI is an optional rehearsal, not a
   substitute for checking the production release. Confirm the published
   README images, links, citation, license and exact version on PyPI.
5. On a native Apple Silicon Mac, optionally build an **unsigned beta
   development artifact** with `bash setup/macos/build_app.sh` after the
   exact version is available on PyPI, or `--wheel` with a checkout build.
   Its `.app` and `.dmg` are ad-hoc signed and not notarized; follow the
   [macOS build guide](release.md). On Windows, build the experimental
   unsigned installer with
   `powershell -NoProfile -ExecutionPolicy Bypass -File setup\windows\build_installer.ps1`.
   Both builders can embed local wheels for verification. Do not present
   these artifacts as signed or independently security-validated binaries.

The builder resolves `uv` symlinks (including Homebrew's), checks that
its target is an Apple-silicon executable, and bundles the resolved binary.
`UV_BINARY` may specify an executable path instead of searching `PATH`.
It builds in a private temporary directory under `dist/`. The builder
leaves existing `dist/kilin-*.whl` files untouched and no longer deletes
the checkout's `build/` directory; the packaging backend may still use
`build/` for its own intermediate files. With `--wheel`, the temporary
wheel is embedded in the app rather than written to public `dist/`.
The existing named app
and DMG outputs are retained until the replacement build is complete.
Ordinary publication errors restore the previous pair of outputs. Build
interruption during the final two renames is not a power-loss-atomic
transaction; retain separately archived release artifacts as needed.
Temporary staging directories are removed after successful builds and
handled failures. The macOS Finder layout operation may still request
permission; denying it leaves a DMG with the default layout. Set
`KILIN_SKIP_FINDER_LAYOUT=1` for noninteractive CI and test builds.

The app holds only a launcher, the icon and `uv`. First launch installs
Python 3.12 and `kilin[app]` under
`~/Library/Application Support/kilin/runtime/` (with `install.log`; the
application-owned download cache is removed after activation). Models
are downloaded later by the desktop setup flow. Removing `runtime/`
removes application environments, not projects or preferences beside it.

The macOS launcher uses stable, **version-specific virtual environments**
at `runtime/releases/<release-id>/`. The identifier is the SHA-256 digest
of the expanded installation requirement plus the bundled wheel checksum,
when provided. A rebuilt wheel with the same version but different bytes
therefore receives a distinct destination. Python virtual environments are
never renamed after installation: console scripts may contain absolute
interpreter paths that would break after relocation.

The small `runtime/active` file selects the current environment as
`releases/<release-id>`. The launcher validates the `kilin-gui` and Python
entry points and installed distribution metadata, records a complete
`.installed` marker, and **atomically replaces** the active pointer only
after installation succeeds. A crash before activation leaves the previous
pointer intact; a crash afterwards selects the completed new release. On
retry, an incomplete *inactive* directory for the requested release is
rebuilt, while a completed but not yet active release can be reused.

Older versions used `runtime/app` without an active pointer. Such a
complete installation remains usable when its requirement matches the
running application; installing a different version preserves the legacy
environment and activates the versioned release only after validation.
Failed updates preserve the previous working installation and selection,
but **do not silently launch the previous version** from a newer app. The
launcher reports the failure for correction and retry. Previous releases
are **not** automatically removed: another desktop process may
still be using one, and model/backend environments can consume significant
disk space. Keep this in mind when upgrading or rebuilding wheels.

The installer serializes runtime changes using the system `lockf` utility
and a persistent `runtime/.install.lock` file. The kernel releases the
lock when an installer exits, including after a crash; **do not delete
the lock file** to clear a stale install. A concurrent launch waits up to
five minutes by default and reports an error instead of waiting forever.
For diagnostics, `KILIN_INSTALL_LOCK_TIMEOUT` sets a wait of 0-600 seconds.
If the runtime directory or lock file cannot be created, installation
stops with an error without invoking `uv`. An older app may have left a
`runtime/.installing` directory: the launcher does not remove it because
an old installer might still be running. Quit older instances, verify
that no installation is in progress, and only then manually remove that
legacy directory before retrying. The file lock and legacy directory are
distinct.

The launcher continues to show native macOS alerts and notifications for
real installations. Unit tests use temporary copies with `osascript`
redirected to an isolated test executable; negative-path tests must not
open modal dialogs, require approval, or depend on a desktop session. The
mocked locking/installation suite does not replace a native Apple Silicon
acceptance test of the actual application and the system `lockf` utility.

For reproducible PR review, compare against the **actual** last approved
archive/commit, not an earlier working directory. Provide a Git patch and a
ZIP of only changed/new files in repository-relative paths, with explicit
deletions when needed. A separate full source export is useful when requested
for release or transfer; it is not needed to apply a focused revision.

The root `LICENSE` holds the Apache License 2.0 text declared by
`pyproject.toml`. It must be committed to Git and present in source ZIPs,
source distributions, and installed wheels. The source-export notebook
requires `"LICENSE": True` and refuses missing or untracked licence files.
The isolated wheel check verifies the actual installed licence text; the
release test verifies the checkout file. Check the generated source
archive and source distribution as part of release verification.

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
8. Verify the changed-files ZIP against the patch and baseline; verify any
   separately requested full export too. Report unexecuted native checks.

For a **documentation-only PR**, preserve executable production behavior and
every existing test/assertion. Add focused documentation regressions where
useful, without weakening behavioral tests to fit the prose. Newly discovered
runtime defects belong in review notes or separately approved corrective PRs.
The root README need not expand merely because technical pages change.

## Performance and reproducibility

Trace-based benchmarking uses `assets/benchmark_traces.ipynb`, with
supporting analysis in `assets/trace_benchmark.py`. The production trace
system remains as documented in [Trace recording](trace_recording.md).
Measurements need model revision, hardware, backend, project state,
question set, warm/cold session distinction, diagnostic tracing enabled or
disabled, and any embedding-vector opt-in. There is no selectable trace mode.
Do not use arbitrary CI wall-clock thresholds as correctness tests. Further
trace/retrieval optimization requires measurements and invariance checks.

## Related

[Architecture](architecture.md),
[Configuration](configuration.md), [Security](security.md),
[Performance](performance.md), [Module map](module_map.md).
