# Supplementary information for the KíLín article

The `clean_repository.ipynb` export includes this directory and applies
explicit safety rules to its children:

| Directory | Source ZIP policy |
|---|---|
| `benchmarks/` | Include Git-tracked source notebooks, gold data, and helpers; export `sandbox/` and `results/` empty. |
| `code_repo/` | Include the directory, but no local files, even if tracked. |
| `literature_repo/` | Include the directory, but no local files, even if tracked. |
| `code_analysis/` | Include Git-tracked analysis files and manuscripts. |
| `reference_analysis/` | Include only `.gitkeep` until analysis is added. |

If the article is still in a separate `paper/` folder, move its contents to
`_si_article_/code_analysis/` and `git add` the intended source files.
The original `paper/` folder is not present in the provided archive, so
its contents have not been recreated or inferred.

Benchmark notebooks were relocated from the previous root `benchmarks/`
directory. Their storage paths and local helper imports were updated, but
some still refer to older QAChat package APIs. Their migration to current
KíLín interfaces requires separate verification before running them.
