# QAChat developer notes

This file explains the repository for a co-developer in plain words. The
README is the short user path. PLAN is the future work.

## What works now

QAChat can prepare its Python environment, verify MLX and PyTorch, find models
in Hugging Face's shared cache, download a selected model through Hugging Face,
load it, keep a chat in memory, and preview repository file metadata. It does
not yet read repository contents, create an embedding index, answer with RAG,
or provide the Qt interface.

## Folder map

| Path | Purpose | In Git? |
| --- | --- | --- |
| `qachat/` | Shared chat, Hugging Face, and repository code. | Yes |
| `setup/` | One-time setup launchers and the engine check. | Yes |
| `tests/` | Repeatable unittest checks. | Yes |
| `tutorials/` | Small independent examples using shared code. | Yes |
| `.vscode/` | Shared formatting and debugger settings. | Yes |
| `.venv/` | This checkout's Python and installed packages. | No |
| `uv.lock` | Exact resolved package versions. | Yes |
| Hugging Face cache | Model files shared with other HF programs. | No |

`qachat` is the Python package inside the outer QAChat project folder. It is
not an engine folder. MLX and PyTorch are engines selected by this package.

Old ignored `models/`, `app_core/`, or `scripts/` data can still exist in a
developer's checkout after an earlier version. Current code does not import or
write those paths. Do not automatically delete them: model weights may be
large user data. A developer may remove or archive them separately after
checking their contents.

## Python setup in simple words

Python runs the code. `uv` reads `pyproject.toml`, prepares Python 3.12, creates
`.venv`, and installs the required packages. `.venv` keeps this project's
packages separate from other Python projects; it is not a security sandbox.

TOML is a small settings format. `pyproject.toml` lists the project packages
and tool settings. `uv.lock` records the exact package solution, including
packages required by other packages. Commit the lock file so another computer
can repeat the same setup.

The optional dependency groups have different jobs:

- `mlx` installs MLX-LM on native Apple Silicon.
- `torch` installs PyTorch and Transformers.
- `dev` installs Jupyter, Black, and Ruff.

The setup launchers select the appropriate groups, then run
`setup/check_setup.py`. That file asks the actual engine to add two numbers and
compares its result with ordinary Python. A printed success sentence reports
the result; the print command does not perform the calculation.

## Coding style

The attached DTDV411 guide is the source of the style decisions.

- Follow PEP 8 and use Black with a 79-character target.
- Ruff checks common code errors and the configured line length.
- `from __future__ import annotations` is the first import.
- Type hints are specific: use `dict[str, object]`, not bare `dict`; use
  `Literal` when a text value has a fixed set of choices.
- Declare class attribute types before `__init__`.
- Imports are absolute. Combine names from the same module when the result is
  short and clear. Optional or expensive engines can be imported inside the
  function that needs them.
- All modules, classes, public functions, and private functions use NumPy-style
  documentation. Put a blank line before the closing quotes of a long
  docstring and another blank line before the function body.
- Use blank lines inside functions to separate logical steps. Conditions,
  loops, validation, calculation, and return values should be easy to see.
- Keep a collection on one line when it is short and clear. Put one value per
  line when the collection is long or easier to review vertically.
- Prefer direct, descriptive names such as `is_apple_silicon` over names that
  make the reader infer whether the value is Boolean.
- Tests use `unittest`, one main test class per test module, deterministic
  inputs, no real model download, and both MLX and PyTorch paths where the
  feature supports both engines.
- Tests check current behavior and meaningful edge cases. They do not search
  source files for names that belonged to an older implementation.

Run the complete checks from the project root:

```sh
.venv/bin/black --check qachat setup tests
.venv/bin/ruff check qachat setup tests
.venv/bin/python -m unittest discover -s tests -v
```

VS Code uses the same Black configuration from `pyproject.toml`; the settings
file turns on format-on-save. VS Code is useful but not required to run QAChat.

## Hugging Face model flow

There is no QAChat model catalog and no QAChat model-storage setting.

1. `qachat/huggingface.py` contains the thin Hugging Face connection.
2. `available_models()` calls Hugging Face's cache scanner.
3. It returns cached model IDs, paths, sizes, and revisions.
4. `download_model(model_id)` calls Hugging Face's normal snapshot download.
5. `ChatSession(model_id)` passes that ID or local directory to the selected
   engine.
6. The engine reuses correct cached files. If `allow_download=True`, missing
   files are downloaded. If it is false, only local files are accepted.

The default cache is normally outside this repository. Standard Hugging Face
environment settings may move it for the whole computer; QAChat does not add a
second configuration. A cached model is not guaranteed to work in both
engines. MLX usually needs converted MLX weights. PyTorch uses Transformers
weights. QAChat does not keep a default-model catalog. The caller must provide
the model ID and choose a compatible engine.

## ChatSession flow

`ChatSession` is the stable chat object that later RAG and the GUI will call.
RAG retrieval will remain outside this class and supply context as messages.

Construction does five easy-to-follow things:

1. Choose `mlx` or `torch`. `auto` chooses MLX on Apple Silicon and PyTorch
   elsewhere.
2. Require and validate a supplied model ID or local model directory.
3. Check chat settings such as thinking, temperature, and token choice.
4. Load the tokenizer and weights through Hugging Face-aware engine calls.
5. Start in-memory history, optionally with a system prompt.

`reply()` formats all current messages with the model's chat template. It
passes the thinking choice to compatible templates, generates through the
selected engine, then stores the user message and successful answer.

`max_new_tokens=None` means QAChat imposes no chosen reply cap. The code uses
the remaining context reported by the model. That physical model context still
limits generation. A caller can provide a positive number when a short reply
or quick experiment is wanted.

`reset()` clears conversation turns but keeps the model loaded. A system
prompt remains. `close()` clears messages, releases model references, runs
garbage collection, and clears the selected accelerator cache.

Useful debugger breakpoints are:

- `ChatSession.__init__` to follow configuration.
- `ChatSession._load` to see model and device selection.
- `ChatSession.reply` to see prompt creation and history.
- `ChatSession._mlx_reply` or `_torch_reply` to see generation.
- `check_setup` and `run_calculation` to follow the setup proof.

The tracked `.vscode/launch.json` starts the setup checker with `auto` or
PyTorch. Put a breakpoint on a line, select a launcher, and press F5.

## Why the model tests are real but small

Tests must be fast, repeatable, and not depend on the internet. The chat tests
create tiny Qwen-shaped weights and a tiny tokenizer in a temporary folder.
They load and generate through the real MLX path and the real PyTorch path.
The folder is removed after the tests. These checks prove both code paths run;
they do not prove that a full model fits every computer's memory.

On native Apple Silicon, an MLX skip or missing engine is not accepted as a
two-engine pass. On systems where MLX cannot run, its platform-specific test is
skipped and PyTorch remains required. Full supported-platform checks still
need real machines.

## Tutorial rule

Every tutorial is an independent starting point. It never expects another
tutorial's variables, kernel memory, or generated files. A new user still
selects QAChat's `.venv` kernel as explained in the README.

Keep tutorial cells small and visible. One cell should normally perform one
step, such as finding the project, adding it to the Python path, showing the
computer, choosing settings, importing a feature, or calling it. Put tutorial
imports close to the step that first uses them. Print useful intermediate
values so a learner can follow the flow one cell at a time.

Tutorials explain one layer at a time, but application code is not duplicated
inside them. Tests check that notebooks are valid JSON, contain no saved
outputs, compile as Python, keep code lines within 79 characters, use the
shared application functions, and remain safe to commit.

## Decisions from the collaborator revision

The collaborator's revision made the model ID explicit. Keep these decisions
in future work:

- `ChatSession` requires a model ID. Tests include that requirement and cover
  both MLX and PyTorch without downloading full models.
- The small verified MLX tutorial model is
  `mlx-community/Qwen3-0.6B-4bit`. The small PyTorch example is
  `Qwen/Qwen3-0.6B`.
- Tutorials show the backend and model as a compatible pair. `auto` chooses an
  engine from the computer, but it does not inspect or convert model weights.

Do not restore a hidden model catalog. A product interface may offer visible
saved choices later, but the shared chat class still receives an explicit
model ID.

## Git rule

Commit source, tests, clean tutorials, documentation, `.vscode`,
`pyproject.toml`, and `uv.lock`. Do not commit `.venv`, model weights, caches,
secrets, logs, or notebook outputs. `.gitignore` prevents ordinary accidental
adds; it does not delete files, make private data safe, or stop cloud syncing.

Do not stage, commit, or push during assisted development unless the user asks.
