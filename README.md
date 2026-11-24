<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)"
            srcset="assets/brand/11-lockup-horizontal-dark.svg">
    <img src="assets/brand/10-lockup-horizontal-light.svg"
         alt="KíLín mosaic qilin mascot and wordmark" width="620">
  </picture>
</p>

<p align="center"><strong>Your AI. Your Knowledge. Your Control.</strong></p>

<p align="center">
  <a href="https://github.com/DeepTrackAI/KiLin/releases/tag/v0.1.0b1">
    <img src="https://img.shields.io/badge/release-0.1.0b1%20beta-orange"
         alt="0.1.0b1 beta release"></a>
  <img src="https://img.shields.io/badge/Python-3.12-blue"
       alt="Python 3.12">
  <a href="LICENSE">
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

**Version `0.1.0b1` (beta):** This is the first public prerelease.
Backward compatibility of public interfaces and saved project
formats is not guaranteed before the stable release.

## Installation

**KíLín `0.1.0b1` requires Python 3.12.** Supported platforms are
Apple Silicon macOS and Linux; Windows support is experimental. Language
model inference runs locally. Package and model downloads require network
access unless previously cached.

### Python package

In a Python 3.12 environment, install the GUI and its model backend:

```sh
python -m pip install "kilin[app]==0.1.0b1"
kilin-gui
```

The `app` extra installs the graphical interface and the computation backend
appropriate to the platform (MLX on supported Apple Silicon Macs, PyTorch
elsewhere). For the core without the GUI, use `kilin==0.1.0b1`; to select a
backend explicitly, see the
[model-management guide](docs/model_management.md).

### Experimental desktop installers

The [v0.1.0b1 GitHub release](https://github.com/DeepTrackAI/KiLin/releases/tag/v0.1.0b1)
includes two experimental desktop installers built from that version's
source: `KiLin-0.1.0b1-macos-arm64.dmg` for Apple Silicon Macs (macOS 14+)
and `KiLin-Setup-0.1.0b1.exe` for 64-bit Windows 10 or 11.

The macOS launcher downloads Python 3.12 and dependencies on first launch;
the Windows installer downloads them during setup. Model weights are
obtained separately when requested. **Neither installer is a signed,
notarized production distribution:** the macOS app is ad-hoc signed and
not notarized, while the Windows EXE is unsigned. Gatekeeper or Windows
SmartScreen may display security warnings or block installation. Verify
the downloaded files against the release checksums and read the
[native installer guidance](docs/release.md) before using them.

Because these installers are not signed, macOS and Windows show a
security warning the first time you open KíLín. To open it anyway:

- **macOS:** open KíLín once and close the warning. Then open **System
  Settings → Privacy & Security**, scroll to **Security**, click **Open
  Anyway** next to the message about KíLín, and confirm. macOS remembers
  the choice for this app.
- **Windows:** when SmartScreen shows "Windows protected your PC", click
  **More info**, then **Run anyway**.

Do this only for an installer from the official KíLín repository on
GitHub or from the KíLín team.

## Getting started

After installation, open `kilin-gui`, select models and a project workspace,
create a project of the appropriate type, and add source files. Ask a
question, inspect the retrieved evidence, and save useful conversations.

The GUI requires a **beta invitation passkey** on first launch for each
user account. Contact the project maintainers to obtain one. This is a
local client-side check, not a security boundary; the open-source
Python API does not require a passkey. See the
[GUI guide](kilin/gui/README.md) for details.

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
[development guide](docs/development.md)
covers setup, dependency and test requirements.

## Tutorials and documentation

KíLín includes **19 progressive Jupyter notebooks** covering models,
source processing, semantic/lexical/code retrieval, indexing, project chat,
diagnostic traces, project types, conversation memory, custom tools, and
an end-to-end research assistant.

Begin with the
[complete tutorial guide](tutorials/README.md),
or start directly with
[10 — Setup](tutorials/10_setup.ipynb),
[20 — Hugging Face models](tutorials/20_huggingface_models.ipynb),
and [140 — Project chat](tutorials/140_project_chat.ipynb).
Some notebooks need previously downloaded models; others work offline with
bundled examples.

The [technical documentation](docs/README.md)
explains the architecture, configuration, sources, retrieval, model
management, tools, GUI, testing, and release procedures. In particular,
see [workspace ownership](docs/project_workspaces.md),
[conversation memory](docs/conversation_memory.md),
and [extension interfaces](docs/extension_interfaces.md).

## Privacy and limitations

KíLín is designed for local control, but local model execution does not
make installation and all optional actions offline. Conversations, source
snapshots, and diagnostic traces may contain sensitive material. KíLín
does not provide encrypted workspaces, an execution sandbox, or guaranteed
scientific correctness. Never share a workspace or trace archive assuming
it has been anonymized.

Read the
[security notes](docs/security.md)
and [trace-recording guide](docs/trace_recording.md)
for the implemented guarantees and limitations.

## Development

Development tests and formatting checks from the checkout include:

```sh
uv run --no-sync python -m tests
uv run --no-sync black --check kilin setup tests
uv run --no-sync ruff check kilin setup tests
```

Use the [development guide](docs/development.md)
for the complete test matrix, style conventions, and release validation.
The [README publication guide](docs/readme_publication.md) explains how
branch-relative GitHub links become tag-pinned links in PyPI metadata.
Contributions and issue reports can be filed at
[DeepTrackAI/KiLin](https://github.com/DeepTrackAI/KiLin).

## Citation

If KíLín contributes to your research, please cite the **specific version**
used. The [CITATION.cff](CITATION.cff)
file provides version-specific software citation metadata for Io̍k-uí
Tiunn, Joana B. Pereira, Carlo Manzo, and Giovanni Volpe.

## License and attribution

KíLín's software is available under the
[Apache License 2.0](LICENSE).
The KíLín name, wordmark, logos, mascot, and other brand artwork have
[separate branding terms](kilin/gui/icons/BRAND_LICENSE.md).
They may accompany authentic, unmodified KíLín releases, while modified
software must not imply official endorsement. See the
[brand-asset notes](assets/brand/README.md)
for the current artwork and its use.
Bundled KaTeX assets retain their
[third-party license](kilin/gui/web/katex/LICENSE).
