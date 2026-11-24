<!-- Generated from README.md by setup/generate_pypi_readme.py; do not edit. -->

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)"
            srcset="https://raw.githubusercontent.com/DeepTrackAI/KiLin/v0.1.0b1/assets/brand/11-lockup-horizontal-dark.svg">
    <img src="https://raw.githubusercontent.com/DeepTrackAI/KiLin/v0.1.0b1/assets/brand/10-lockup-horizontal-light.svg"
         alt="KíLín mosaic qilin mascot and wordmark" width="620">
  </picture>
</p>

<p align="center"><strong>Your AI. Your Knowledge. Your Control.</strong></p>

<p align="center">
  <a href="https://github.com/DeepTrackAI/KiLin/releases">
    <img src="https://img.shields.io/badge/release-0.1.0b1%20beta-orange"
         alt="0.1.0b1 beta release"></a>
  <img src="https://img.shields.io/badge/Python-3.12-blue"
       alt="Python 3.12">
  <a href="https://github.com/DeepTrackAI/KiLin/blob/v0.1.0b1/LICENSE">
    <img src="https://img.shields.io/badge/license-Apache--2.0-blue"
         alt="Apache License 2.0"></a>
</p>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#installation">Installation</a> ·
  <a href="#getting-started">Getting started</a> ·
  <a href="#tutorials-and-documentation">Documentation</a> ·
  <a href="#citation">Citation</a> ·
  <a href="#license-and-attribution">License</a>
</p>

## Overview

**KíLín** is a local, extensible AI assistant for scientific, technical,
and general knowledge work. It combines locally run language models,
project-specific sources, retrieval-augmented generation (RAG), configurable
tools, and persistent conversations. Users control their models, projects,
and data without depending on a remote AI service for every answer.

KíLín provides six built-in project types for code, literature, writing,
grounded brainstorming, creative brainstorming, and researcher-directed
reviewing. Source-grounded responses reference project files for independent
inspection.

**Beta release:** This is the `0.1.0b1`, the first public prerelease.
The planned sequence is `0.1.0b1` → `0.1.0rc1` → `0.1.0`. Features, public
interfaces, and saved formats may change before `0.1.0`.

## Installation

The current package officially supports **Python 3.12**. The CI matrix
also runs experimental compatibility tests for Python **3.13 and 3.14** on
Linux, macOS, and Windows. Those jobs bypass the package's 3.12 requirement
and do not yet establish official installation support for newer versions.
The desktop installers use Python 3.12. Model execution is local, but
installation and explicitly requested downloads require network access.

### Python package

On macOS, Linux, or experimental Windows, use a Python 3.12 environment:

```sh
python -m pip install "kilin[app]==0.1.0b1"
kilin-gui
```

The `app` extra installs the graphical interface and the computation backend
appropriate to the platform (MLX on supported Apple Silicon Macs, PyTorch
elsewhere). For the core without the GUI, use `kilin==0.1.0b1`; to select a
backend explicitly, see the
[model-management guide](https://github.com/DeepTrackAI/KiLin/blob/v0.1.0b1/docs/model_management.md).
The beta version is installable from PyPI.

### macOS desktop application (Apple Silicon, macOS 14+)

The experimental `KíLín.dmg`, when provided as a
[GitHub release asset](https://github.com/DeepTrackAI/KiLin/releases),
contains a launcher that installs Python 3.12 and dependencies on first use.
Current builds are signed ad hoc, **not** with an Apple Developer ID, and
are **not notarized**. macOS may block them. Only use an artifact whose
origin you have verified; do not disable Gatekeeper or remove quarantine as
a general installation procedure. See the
[macOS release guide](https://github.com/DeepTrackAI/KiLin/blob/v0.1.0b1/docs/release.md).

### Windows desktop installer (64-bit Windows 10 or 11)

The experimental `KiLin-Setup-<version>.exe`, when available, installs to
the current user's account without administrator privileges. It downloads
Python 3.12 and dependencies (including a suitable PyTorch build). The
installer is **unsigned**, so Windows SmartScreen may warn. Follow the
[Windows installer guide](https://github.com/DeepTrackAI/KiLin/blob/v0.1.0b1/docs/release.md#windows-installer)
and do not run an unverified executable.

## Getting started

After installation, open `kilin-gui`, select models and a project workspace,
create a project of the appropriate type, and add source files. Ask a
question, inspect the retrieved evidence, and save useful conversations.

The desktop beta includes a **local invitation-passkey prompt** on first
launch for each user account. After successful activation, the GUI remembers
that user. Contact the project maintainers for a beta passkey. This is a
client-side access prompt, **not a security boundary or access restriction
on the open-source Python API**. See the
[GUI guide](https://github.com/DeepTrackAI/KiLin/blob/v0.1.0b1/kilin/gui/README.md)
for further details.

To run KíLín from a source checkout, on macOS or Linux:

```sh
sh setup/setup.sh
uv run kilin --gui
```

On Windows, perform development setup with PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File setup/setup.ps1
```

Then run `uv run kilin --gui`. Select the checkout's `.venv` Python
kernel when following the tutorials. The
[development guide](https://github.com/DeepTrackAI/KiLin/blob/v0.1.0b1/docs/development.md)
covers setup, dependency and test requirements.

## Tutorials and documentation

KíLín includes **19 progressive Jupyter notebooks** covering models,
source processing, semantic/lexical/code retrieval, indexing, project chat,
diagnostic traces, project types, conversation memory, custom tools, and
an end-to-end research assistant.

Begin with the
[complete tutorial guide](https://github.com/DeepTrackAI/KiLin/blob/v0.1.0b1/tutorials/README.md),
or start directly with
[10 — Setup](https://github.com/DeepTrackAI/KiLin/blob/v0.1.0b1/tutorials/10_setup.ipynb),
[20 — Hugging Face models](https://github.com/DeepTrackAI/KiLin/blob/v0.1.0b1/tutorials/20_huggingface_models.ipynb),
and [140 — Project chat](https://github.com/DeepTrackAI/KiLin/blob/v0.1.0b1/tutorials/140_project_chat.ipynb).
Some notebooks need previously downloaded models; others work offline with
bundled examples.

The [technical documentation](https://github.com/DeepTrackAI/KiLin/blob/v0.1.0b1/docs/README.md)
explains the architecture, configuration, sources, retrieval, model
management, tools, GUI, testing, and release procedures. In particular,
see [workspace ownership](https://github.com/DeepTrackAI/KiLin/blob/v0.1.0b1/docs/project_workspaces.md),
[conversation memory](https://github.com/DeepTrackAI/KiLin/blob/v0.1.0b1/docs/conversation_memory.md),
and [extension interfaces](https://github.com/DeepTrackAI/KiLin/blob/v0.1.0b1/docs/extension_interfaces.md).

## Privacy and limitations

KíLín is designed for local control, but local model execution does not
make installation and all optional actions offline. Conversations, source
snapshots, and diagnostic traces may contain sensitive material. KíLín
does not provide encrypted workspaces, an execution sandbox, or guaranteed
scientific correctness. Never share a workspace or trace archive assuming
it has been anonymized.

Read the
[security notes](https://github.com/DeepTrackAI/KiLin/blob/v0.1.0b1/docs/security.md)
and [trace-recording guide](https://github.com/DeepTrackAI/KiLin/blob/v0.1.0b1/docs/trace_recording.md)
for the implemented guarantees and limitations.

## Development

Development tests and formatting checks from the checkout include:

```sh
uv run --no-sync python -m tests
uv run --no-sync black --check kilin setup tests
uv run --no-sync ruff check kilin setup tests
```

Use the [development guide](https://github.com/DeepTrackAI/KiLin/blob/v0.1.0b1/docs/development.md)
for the complete test matrix, style conventions, and release validation.
The [README publication guide](https://github.com/DeepTrackAI/KiLin/blob/v0.1.0b1/docs/readme_publication.md) explains how
branch-relative GitHub links become tag-pinned links in PyPI metadata.
Contributions and issue reports can be filed at
[DeepTrackAI/KiLin](https://github.com/DeepTrackAI/KiLin).

## Citation

If KíLín contributes to your research, please cite the **specific version**
used. The [CITATION.cff](https://github.com/DeepTrackAI/KiLin/blob/v0.1.0b1/CITATION.cff)
file provides GitHub-compatible software citation metadata for Io̍k-uí
Tiunn, Joana B. Pereira, Carlo Manzo, and Giovanni Volpe. A DOI will be
included only if one is assigned to an archived release.

## License and attribution

KíLín's software is available under the
[Apache License 2.0](https://github.com/DeepTrackAI/KiLin/blob/v0.1.0b1/LICENSE).
The KíLín name, wordmark, logos, mascot, and other brand artwork have
[separate branding terms](https://github.com/DeepTrackAI/KiLin/blob/v0.1.0b1/kilin/gui/icons/BRAND_LICENSE.md).
They may accompany authentic, unmodified KíLín releases, while modified
software must not imply official endorsement. See the
[brand-asset notes](https://github.com/DeepTrackAI/KiLin/blob/v0.1.0b1/assets/brand/README.md)
for the current artwork and its use.
Bundled KaTeX assets retain their
[third-party license](https://github.com/DeepTrackAI/KiLin/blob/v0.1.0b1/kilin/gui/web/katex/LICENSE).
