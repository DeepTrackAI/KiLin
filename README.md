# QAChat

QAChat currently sets up MLX and PyTorch, uses Hugging Face models, runs a
local chat, reads selected repository source, turns it into traceable chunks,
and creates embedding vectors. Nothing is searched or retrieved in this
branch.

## Start once on a new computer

From the QAChat folder, run one command with internet access.

Mac or Linux:

```sh
sh setup/setup.sh
```

Windows x64 PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File setup/setup.ps1
```

The script installs `uv` if needed, prepares Python 3.12 and `.venv`, installs
the packages from `uv.lock`, and asks the selected engine to calculate
`2 + 3`. Expect `result: 5.0` and `Setup complete`. No model is downloaded.

## Try one tutorial

Open any tutorial directly, select QAChat's `.venv` Python kernel, and run one
cell at a time with Shift+Enter. Each tutorial starts from a clean session and
does not require another tutorial to run first.

1. [Setup check](tutorials/10_setup.ipynb) — enter two numbers and confirm that
   MLX or PyTorch calculates the correct sum.
2. [Use Hugging Face models](tutorials/20_huggingface_models.ipynb) — list
   models in Hugging Face's normal cache and optionally download one there.
3. [Load a model and chat](tutorials/30_model_chat.ipynb) — choose MLX or
   PyTorch, load a compatible model, configure it, chat, and close.
4. [Choose repository files](tutorials/40_repository_files.ipynb) — preview
   file names, sizes, selections, and skip reasons without reading source.
5. [Read source and make chunks](tutorials/50_source_structure.ipynb) — find
   natural source units, make bounded chunks, and keep their file and lines.
6. [Create embeddings](tutorials/60_embeddings.ipynb) — count model tokens and
   turn real chunks and one question into vectors without searching them.

Hugging Face owns the model cache. If another Hugging Face program already
downloaded a model, QAChat can see it. Loading a model ID reuses cached files;
when allowed, missing files are downloaded to that same cache. Model weights
are not committed to QAChat. MLX and PyTorch may require different weights.

## Verify the repository

These commands are optional when only using the tutorials. Run them to verify
the installation or before committing code.

Run all automated tests:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

Verify that Python files already follow the 79-character format and contain no
common code errors:

```sh
.venv/bin/black --check qachat setup tests
.venv/bin/ruff check qachat setup tests
```

If Black reports formatting problems, let it format the files:

```sh
.venv/bin/black qachat setup tests
```

Clear saved notebook outputs before committing. This keeps the code and
Markdown:

```sh
.venv/bin/jupyter nbconvert --clear-output --inplace tutorials/*.ipynb
```

On Windows, replace `.venv/bin/` with `.venv\Scripts\`.

Semantic, lexical, and hybrid retrieval, the repository graph, question
intent, RAG answers, and the Qt and optional MCP interfaces come later.
