# QAChat

QAChat currently sets up MLX and PyTorch, uses Hugging Face models, runs a
local chat, reads selected repository source, turns it into traceable chunks,
creates embedding vectors, searches semantic and lexical indexes, follows code
relationships, saves reusable indexes, combines retrieval rankings, and runs
a persistent RAG conversation.

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

Each tutorial also checks this environment near the top. If a declared package
is missing, the tutorial repairs `.venv` with `uv`; it never needs `pip`. A
running notebook cannot change its own Python kernel, so select QAChat's
`.venv` kernel and rerun the setup cell if asked.

## Try one tutorial

Open any tutorial directly and run one cell at a time with Shift+Enter. The
first cells check the packages and show whether QAChat's `.venv` kernel is
selected. Each tutorial is standalone.

1. [Setup check](tutorials/10_setup.ipynb) — enter two numbers and confirm that
   MLX or PyTorch calculates the correct sum.
2. [Use Hugging Face models](tutorials/20_huggingface_models.ipynb) — list
   models in Hugging Face's normal cache and optionally download one there.
3. [Load a model and chat](tutorials/30_model_chat.ipynb) — choose MLX or
   PyTorch, load a compatible model, configure it, chat, and close.
4. [Choose repository files](tutorials/40_repository_files.ipynb) — preview
   file names, sizes, selections, and skip reasons without reading source.
   Python, MATLAB, Markdown, notebooks, and text-based PDFs are supported.
5. [Read source and make chunks](tutorials/50_source_structure.ipynb) — find
   natural source units, make bounded chunks, and keep their file and lines.
6. [Create embeddings](tutorials/60_embeddings.ipynb) — count model tokens and
   turn real chunks and one question into vectors without searching them.
7. [Semantic retrieval](tutorials/70_semantic_retrieval.ipynb) — build an
   embedding index and rank chunks by meaning with cosine similarity.
8. [Lexical retrieval](tutorials/80_lexical_retrieval.ipynb) — build a
   TF-IDF index and rank exact words, phrases, and software identifiers.

9. [Code-aware retrieval](tutorials/90_code_aware_retrieval.ipynb) — reuse
   natural source units and follow Python or MATLAB code relationships.
10. [Persistent indexes](tutorials/100_persistent_indexes.ipynb) — save,
    reload, and check semantic and lexical indexes.
11. [Hybrid ranking](tutorials/110_hybrid_ranking.ipynb) — combine semantic,
    lexical, and code-aware rankings with Reciprocal Rank Fusion.
12. [Retrieval routing](tutorials/120_retrieval_routing.ipynb) — preserve
    ordinary ranking while guaranteeing explicitly requested code context.
13. [RAG application](tutorials/130_rag_application.ipynb) — open a configured
    project, ask grounded questions, inspect evidence, and restore the session.
14. [Project chat](tutorials/140_project_chat.ipynb) — choose a project, resume
    its conversation, ask repeatedly, and inspect saved evidence and settings.

## Try a sandbox project

The committed **sandbox/code** and **sandbox/literature** folders are small
project templates. Each **profile.ini** selects the source, models, chunking,
retrieval, and generation settings.

QAChat writes reusable retrieval data to **indexes/** and full conversations,
evidence, and parameter snapshots to **logs/**. Their generated contents are
excluded from Git. Reopening the same project restores its current indexes and
conversation. Calling **session.reset()** archives the active conversation
under a timestamped file name and starts with empty conversation memory.

Hugging Face owns the model cache. If another Hugging Face program already
downloaded a model, QAChat can see it. Loading a model ID reuses cached files;
when allowed, missing files are downloaded to that same cache. Model weights
are not committed to QAChat. MLX and PyTorch may require different weights.

## Verify the repository

These steps are optional when only using the tutorials. Run them when
developing QAChat and before committing changes.

### Run the tests

Run all automated tests:

```sh
.venv/bin/python -m tests
```

This short command uses `tests/__main__.py` to discover every test under the
nested `tests/` folders. The equivalent standard-library command is:

```sh
.venv/bin/python -m unittest discover -s tests -t . -v
```

The `-t .` keeps the repository root as the import root, so `tests/qachat/`
does not hide the real `qachat/` package.

### Clean and check before committing
Format the Python code with Black, check and fix common code issues with
Ruff, and remove saved outputs from the tutorial notebooks:

```sh
.venv/bin/black qachat setup tests
.venv/bin/ruff check --fix qachat setup tests
.venv/bin/jupyter nbconvert --clear-output --inplace tutorials/*.ipynb
```

 [Black](https://black.readthedocs.io/) automatically reformats files when needed.
 [Ruff](https://docs.astral.sh/ruff/) automatically fixes
issues that can be safely corrected and reports any remaining problems.
Clearing notebook outputs keeps committed notebooks clean.
The 79-character line limit is defined in three places:

- pyproject.toml under [tool.black]
- pyproject.toml under [tool.ruff]
- tests/tutorials/test_tutorials.py, which checks tutorial code lines

On Windows, replace `.venv/bin/` with `.venv\Scripts\`.
