<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)"
            srcset="https://raw.githubusercontent.com/DeepTrackAI/KiLin/main/assets/brand/11-lockup-horizontal-dark.svg">
    <img src="https://raw.githubusercontent.com/DeepTrackAI/KiLin/main/assets/brand/10-lockup-horizontal-light.svg"
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
  <a href="https://github.com/DeepTrackAI/KiLin/blob/main/LICENSE">
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
and data instead of depending on a remote AI service for every answer.

KíLín provides six built-in project types for code, literature, writing,
grounded brainstorming, creative brainstorming, and researcher-directed
reviewing. Source-grounded responses reference project files for independent
inspection. Optional conversation memory can reuse eligible earlier
interactions without presenting them as independent scientific evidence.
Scientific interpretation and the final evaluation of reports remain with
the researcher.

**Beta release:** This is the preparation for `0.1.0b1`, the first public
prerelease. The planned sequence is `0.1.0b1` → `0.1.0rc1` → `0.1.0`.
Features, public interfaces, and saved formats may change before `0.1.0`.
A release appearing in this README is not a claim that its distribution
artifacts have already been published.

## Installation

The Python package requires **Python 3.12**. KíLín runs its models locally,
but installation, dependency resolution, and explicitly requested model
downloads require network access.

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
[model-management guide](https://github.com/DeepTrackAI/KiLin/blob/main/docs/model_management.md).
The beta version is installable from PyPI once its artifacts are published.

### macOS desktop application (Apple Silicon, macOS 14+)

The experimental `KíLín.dmg`, when provided as a
[GitHub release asset](https://github.com/DeepTrackAI/KiLin/releases),
contains a launcher that installs Python 3.12 and dependencies on first use.
Current builds are signed ad hoc, **not** with an Apple Developer ID, and
are **not notarized**. macOS may block them. Only use an artifact whose
origin you have verified; do not disable Gatekeeper or remove quarantine as
a general installation procedure. See the
[macOS release guide](https://github.com/DeepTrackAI/KiLin/blob/main/docs/release.md).

### Windows desktop installer (64-bit Windows 10 or 11)

The experimental `KiLin-Setup-<version>.exe`, when available, installs to
the current user's account without administrator privileges. It downloads
Python 3.12 and dependencies (including a suitable PyTorch build). The
installer is **unsigned**, so Windows SmartScreen may warn. Follow the
[Windows installer guide](https://github.com/DeepTrackAI/KiLin/blob/main/docs/release.md#windows-installer)
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
[GUI guide](https://github.com/DeepTrackAI/KiLin/blob/main/kilin/gui/README.md)
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
[development guide](https://github.com/DeepTrackAI/KiLin/blob/main/docs/development.md)
covers setup, dependency and test requirements.

## Tutorials and documentation

KíLín includes **19 progressive Jupyter notebooks** covering models,
source processing, semantic/lexical/code retrieval, indexing, project chat,
diagnostic traces, project types, conversation memory, custom tools, and
an end-to-end research assistant.

Begin with the
[complete tutorial guide](https://github.com/DeepTrackAI/KiLin/blob/main/tutorials/README.md),
or start directly with
[10 — Setup](https://github.com/DeepTrackAI/KiLin/blob/main/tutorials/10_setup.ipynb),
[20 — Hugging Face models](https://github.com/DeepTrackAI/KiLin/blob/main/tutorials/20_huggingface_models.ipynb),
and [140 — Project chat](https://github.com/DeepTrackAI/KiLin/blob/main/tutorials/140_project_chat.ipynb).
Some notebooks need previously downloaded models; others work offline with
bundled examples.

The [technical documentation](https://github.com/DeepTrackAI/KiLin/blob/main/docs/README.md)
explains the architecture, configuration, sources, retrieval, model
management, tools, GUI, testing, and release procedures. In particular,
see [workspace ownership](https://github.com/DeepTrackAI/KiLin/blob/main/docs/project_workspaces.md),
[conversation memory](https://github.com/DeepTrackAI/KiLin/blob/main/docs/conversation_memory.md),
and [extension interfaces](https://github.com/DeepTrackAI/KiLin/blob/main/docs/extension_interfaces.md).

## Privacy and limitations

KíLín is designed for local control, but local model execution does not
make installation and all optional actions offline. Conversations, source
snapshots, and diagnostic traces may contain sensitive material. KíLín
does not provide encrypted workspaces, an execution sandbox, or guaranteed
scientific correctness. Never share a workspace or trace archive assuming
it has been anonymized.

Read the
[security notes](https://github.com/DeepTrackAI/KiLin/blob/main/docs/security.md)
and [trace-recording guide](https://github.com/DeepTrackAI/KiLin/blob/main/docs/trace_recording.md)
for the implemented guarantees and limitations.

## Development

Development tests and formatting checks from the checkout include:

```sh
uv run --no-sync python -m tests
uv run --no-sync black --check kilin setup tests
uv run --no-sync ruff check kilin setup tests
```

Use the [development guide](https://github.com/DeepTrackAI/KiLin/blob/main/docs/development.md)
for the complete test matrix, style conventions, and release validation.
Contributions and issue reports can be filed at
[DeepTrackAI/KiLin](https://github.com/DeepTrackAI/KiLin).

## Citation

If KíLín contributes to your research, please cite the **specific version**
used. The [CITATION.cff](https://github.com/DeepTrackAI/KiLin/blob/main/CITATION.cff)
file provides GitHub-compatible software citation metadata for Io̍k-uí
Tiunn, Joana B. Pereira, Carlo Manzo, and Giovanni Volpe. A DOI will be
included only if one is assigned to an archived release.

## License and attribution

KíLín's software is available under the
[Apache License 2.0](https://github.com/DeepTrackAI/KiLin/blob/main/LICENSE).
Logo and mascot artwork have separate attribution and redistribution
considerations; see the
[brand-asset notes](https://github.com/DeepTrackAI/KiLin/blob/main/assets/brand/README.md)
and [GUI icon notes](https://github.com/DeepTrackAI/KiLin/blob/main/kilin/gui/icons/README.md).
Bundled KaTeX assets retain their
[third-party license](https://github.com/DeepTrackAI/KiLin/blob/main/kilin/gui/web/katex/LICENSE).
