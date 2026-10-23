# KíLín benchmarks

These notebooks evaluate KíLín using published retrieval tasks and
hand-annotated code-question sets. They call KíLín's existing source extraction,
chunking, embedding, indexing, retrieval and answer-generation APIs rather than
introducing a separate benchmark implementation.

## Benchmark inventory

| Notebook | Data and evaluation |
| --- | --- |
| `10_coderag_bench_humaneval_mbpp.ipynb` | CodeRAG-Bench-style HumanEval and MBPP retrieval |
| `20_coderag_bench_odex.ipynb` | ODEX code retrieval |
| `30_coderag_bench_ds1000.ipynb` | DS-1000 code retrieval |
| `40_beir_fiqa_retrieval.ipynb` | BEIR FiQA document retrieval |
| `50_beir_scifact_retrieval.ipynb` | BEIR SciFact document retrieval |
| `60_kilin_dogfood_code.ipynb` | Answers about the current KíLín source and tutorials |
| `70_deeptrack_dogfood_code.ipynb` | Answers about an external DeepTrack2 checkout |

`common.py` defines evaluation records and scoring utilities. Ranking metrics
use `pytrec_eval`; citation, abstention and verbatim-overlap checks are
mechanical indicators, **not substitutes for checking answer faithfulness**.
The dogfooding notebooks export Tier-2 worksheets for human review.

## Environment and execution

Run from a KíLín source checkout with Python 3.12 and the repository's
virtual environment selected as the notebook kernel. Install the relevant
backend, notebook dependencies and `bench` extra as described in the main
[development guide](../docs/development.md) and `pyproject.toml`. The notebooks
run `setup.notebook.prepare_notebook()` and check for their optional datasets
and evaluation dependencies. Published datasets may need network access and
access to their upstream repositories. Embedding and chat models may also need
a local cache or an explicitly permitted download.

Run the numbered notebooks individually; they are not a mandatory part of
KíLín's offline unit-test suite. The retrieval notebooks compare KíLín's
chunked indexing with whole-document baselines where appropriate. Changing
chunking, model weights, tokenization, data revisions or corpus selection may
change metrics. Record those choices before comparing runs.

## Generated files and data ownership

`benchmarks/sandbox/` holds local corpus mirrors, staging copies and benchmark
workspaces. `benchmarks/results/` holds scoring JSON and human grading
worksheets. **Neither directory's contents belong in a source backup.** Both
empty directory entries should remain in the source ZIP.

Dogfooding workspaces are separate from repository source directories. The
current `prepare_project_workspace()` call creates a new workspace on each
run, and `synchronize_project_workspace()` updates an existing mirror after
changing its source-selection settings. The benchmark source and any external
DeepTrack checkout are never modified by these operations.

## Gold annotations and comparability

- `gold/dogfood_kilin_code.json` contains 22 questions, including five
  deliberate abstention cases, verified for source-chunk overlap against the
  **23 October 2026** KíLín archive. Citation paths and line numbers describe
  that snapshot; review them whenever code or notebooks change. An overlapping
  chunk does not, by itself, establish that the written reference answer is
  scientifically or technically correct.
- `gold/dogfood_qachat_code.json` is the **unaltered historical QAChat** gold
  dataset. Its old package paths and implementation claims must not be
  relabeled or scored as if they describe the current KíLín repository.
- `gold/dogfood_deeptrack_code.json` retains the historical DeepTrack2
  reference set. The selected external checkout may differ; notebook 70
  reports unmatched gold citations before scoring. Do not silently repair
  these annotations merely to improve measured retrieval.

The two dogfooding notebooks default to `USE_ROUTER = False`: this selects the
code project's default search intent, holding retrieval behavior constant for
citation evaluation. Setting `USE_ROUTER = True` exercises KíLín's current
intent router, possibly including tool-generated answers; results from that
mode measure a different pipeline and should be reported separately.
`USE_STRONGER_GUIDANCE` applies optional answer instructions through the
public `RAGSession.ask()` API. Neither notebook mutates private chat state.

## Lightweight verification

The repository's `tests/kilin/test__benchmark_contract.py` checks helper
behavior, notebook syntax and benchmark JSON without loading models or
fetching datasets. A separate source-chunk overlap review should be performed
when updating the KíLín gold annotations.
