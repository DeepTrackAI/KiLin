# QAChat

QAChat answers questions about local code and literature using a local LLM
that ensures no data are shared outsde the local enviroment.

## 1. Install once

**Do this once on each computer.** Open a terminal inside the QAChat
repository, then run the command for your system.

Mac or Linux:

```sh
sh setup/setup.sh
```

Windows x64 PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File setup/setup.ps1
```

When it finishes, you will see:

```text
Setup complete. The project environment is in .venv
```

Select the QAChat `.venv` kernel when opening a notebook.

## 2. Chat with your project

Open [Project chat](tutorials/130_project_chat.ipynb). In its configuration
cell:

- Set `PROJECT_KIND` to `"code"` or `"literature"`.
- Set `PROJECT_DIRECTORY` to your project folder.
- Leave `PROJECT_DIRECTORY = None` to try the included sandbox project.

The first open creates the project settings, indexes, and conversation log.
Opening the same project later resumes it. Generated data and model weights
are not committed to Git; models use the normal Hugging Face cache.

## 3. Understand how QAChat works

These standalone tutorials explain QAChat one layer at a time. Use them when
you want to understand the concepts and code clearly.

1. [Setup check](tutorials/10_setup.ipynb) — verify MLX or PyTorch.
2. [Hugging Face models](tutorials/20_huggingface_models.ipynb) — find or
   download models.
3. [Model chat](tutorials/30_model_chat.ipynb) — load a model and chat.
4. [Repository files](tutorials/40_repository_files.ipynb) — choose files.
5. [Source chunks](tutorials/50_source_structure.ipynb) — read and split
   source.
6. [Embeddings](tutorials/60_embeddings.ipynb) — turn text into vectors.
7. [Semantic retrieval](tutorials/70_semantic_retrieval.ipynb) — search by
   meaning.
8. [Lexical retrieval](tutorials/80_lexical_retrieval.ipynb) — search by
   exact words and identifiers.
9. [Code-aware retrieval](tutorials/90_code_aware_retrieval.ipynb) — follow
   Python and MATLAB relationships.
10. [Persistent indexes](tutorials/100_persistent_indexes.ipynb) — save and
    reuse indexes.
11. [Hybrid ranking](tutorials/110_hybrid_ranking.ipynb) — combine retrieval
    rankings.
12. [Retrieval routing](tutorials/120_retrieval_routing.ipynb) — include
    explicitly requested code.
13. [Project chat](tutorials/130_project_chat.ipynb) — use the complete RAG
    workflow.

## 4. Development checks

Run all tests:

```sh
.venv/bin/python -m tests
```

Format and check the code, then clear notebook output:

```sh
.venv/bin/black qachat setup tests
.venv/bin/ruff check --fix qachat setup tests
.venv/bin/jupyter nbconvert --clear-output --inplace tutorials/*.ipynb
```

On Windows, replace `.venv/bin/` with `.venv\Scripts\`.
