# QAChat

QAChat currently sets up MLX and PyTorch, finds or downloads Hugging Face
models, runs a local chat, and retrieves relevant source from a local
repository.

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
3. [Load a model and chat](tutorials/30_model_chat.ipynb) — choose MLX or PyTorch,
   load a compatible model, configure thinking and sampling, chat, and close.
4. [Choose repository files](tutorials/40_repository_files.ipynb) — preview
   file names, sizes, selections, and skip reasons without reading source.
5. [Read source and make chunks](tutorials/50_source_structure.ipynb) — read
   selected text safely, identify natural source units, and keep provenance.
6. [Semantic retrieval](tutorials/60_semantic_retrieval.ipynb) — load an MLX
   or PyTorch embedding model and find source with a similar meaning.
7. [Lexical and structural retrieval](tutorials/70_lexical_retrieval.ipynb) — find exact
   code words, identifiers, paths, and symbols without loading a model.

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

On Windows, replace `.venv/bin/` with `.venv\Scripts\`.

Hybrid retrieval, the repository graph, RAG answers, and the Qt interface come
in later steps.
