# Retrieval and index architecture

## Responsibilities and public interfaces

QAChat separates **source processing** from **retrieval**. The former yields
provenance-bearing `Chunk` records; the latter produces several rankings,
combines them and expands evidence for explicit code or document references.
The main orchestration belongs to `qachat.rag.RAGSession` and the algorithms
live under `qachat/retrievals/`.

```text
Selected sources -> SourceUnit -> Chunk
                      |
                      +--> embedding model -> SemanticIndex
                      +--> lexical_terms() -> LexicalIndex
                      `--> code entity extraction -> CodeGraph
                              |
Question -> embedding + lexical + code lookup -> ranked channel lists
                              |
                      reciprocal-rank fusion
                              |
                      named-source context / document scope
                              |
                      Evidence(path, line/cell/page, exact text)
```

`qachat.retrievals.__init__` exposes the supported records and functions;
private helpers and individual extension internals should not be treated as
stable public interfaces. The three ranking channels can operate over the
**same ordered chunks** but employ different scores and heuristics.

## Semantic channel

`EmbeddingSession` loads the configured embedding model and tokenizer, then
exposes `count_tokens()`, `embed_documents()` and `embed_query()`; token and
vector behavior is backend-specific. `build_semantic_index()` couples
retrieval chunks with document embeddings. `semantic_search()` embeds a query
and ranks candidates using the stored numerical vectors. The implementation
validates embedding shape, cardinality and finite numeric values; persistent
semantic content must match the referenced chunk order.

Embedding-query formatting is not universal. `query_formats.py` defines
`plain` and Qwen3-compatible modes, while the project's `profile.ini` stores
the configured query format. Do not conflate display text, retrieval query
text, model-specific embedding query format and the original user question.
The Qwen3 query formatter is applied by the relevant semantic-search path,
not by modifying original source files.

## Lexical channel

`retrieval_lexical.py` tokenizes text into lowercase words, source-oriented
identifier components and adjacent bigrams. It builds a normalized sparse
TF-IDF representation using term counts and inverse document frequency.
`lexical_search()` computes its channel-specific rank. The tokenizer is
ASCII-oriented and does **not** perform stemming or general multilingual
linguistic analysis. A lexical score should never be compared directly to a
semantic score as though they represented the same scale.

## Code-aware channel

`code_graph/code_graph.py` represents code entities/relations and constructs
an indexed graph from source units. The Python extension uses AST-oriented
heuristics for definitions, references and imports; the MATLAB extension
uses text patterns. These are useful structured hints, **not complete
semantic call graphs** and not a code execution or static type checker.

`retrieval_code.py` resolves likely code entities in a question and finds
related chunks. `retrievals/routing.py` can detect an explicit file/function
reference and expand retrieved context accordingly. Graph structure and
chunk provenance must agree: a stale graph can misdirect named-function
context even when semantic vectors remain readable.

## Hybrid combination: ranks rather than scores

`retrieval_hybrid.hybrid_search()` takes ordered semantic, lexical and code
channel results. It performs **reciprocal-rank fusion (RRF)**: for each
channel, a chunk's contribution is `1 / (rank_constant + rank)` using a
one-based rank. Each channel has equal weight and raw channel scores are
discarded; repeated chunk identifiers inside one channel contribute only
once. The combined results are sorted by descending fused score, then
stable chunk identifier for ties. `RETRIEVAL.RRF_RANK_CONSTANT` is the
packaged default smoothing constant. Per-query retrieval count limits the
ranked output; explicit context expansion may yield more evidence.

This has two implications for developers: (1) changing the relative channel
scales does not directly influence fusion, but changing **order** does;
(2) an optimization must preserve the content and ordering of evidence,
not just the number of embeddings or the final answer string.

## Named-document retrieval and full-document reading

There are several distinct retrieval modes determined by the selected
`ProjectType.Intent`:

| Mode | Meaning |
| --- | --- |
| `search` | Search the indexed corpus with the configured strategy and count. |
| `document` | Identify and read a named or contextually selected whole document, fitted to a token budget. |
| `reuse` | Reuse evidence from related earlier turns without a new general search. |
| `none` | Generate or otherwise handle a request without retrieval. |

See the canonical values in `project_type.py`. The `document` path uses
`retrievals/document_reading.py`: `named_paths()`, `dominant_path()`,
`read_document()` and `DocumentSelection` distinguish explicit names from
search-ranked document selection, then fit content into a budget. Tools also
use shared document resolution under `tools/_shared/choose_document.py`.
A document-scoped request may require ranking enough material **inside that
document**, even if global top-k omitted its relevant passages; therefore
simply deleting an additional ranking query can be a correctness regression.

`retrievals/routing.py` separately expands evidence near named source
references using `RetrievalPlan` and the graph. This expansion is not the
same as model-based intent routing in `answering.py`; the latter decides
which **operation** to perform, not which chunk matches a function name.

## Index persistence and identity

`PersistentIndexes` stores aligned semantic and lexical data plus a
`CodeGraph`, source fingerprint and embedding-model identity. The persisted
JSON includes chunks, vectors, lexical statistics, code relations and a
payload SHA-256 checksum. `save_indexes()` uses atomic single-file
replacement. `load_indexes()` validates the serialized structure, vector
cardinality/finite values, lexical vocabulary consistency, identifier
uniqueness, content fingerprint and full payload checksum.

Two checks address different failure modes:

- **Source fingerprint / `indexes_are_current()`:** does this bundle represent
  the current ordered chunks and code graph?
- **Payload checksum / identity:** were the serialized vectors, statistics
  and model reference preserved as expected? A matching checksum is an
  integrity check, **not an authenticity or tamper-proof security scheme**.

Backend/library versions are not all encoded in the index identity. A source
snapshot alone does not guarantee semantic compatibility after changing
models or indexing parameters. Older index formats without recorded model
snapshot identity can require a full rebuild.

### Incremental updates

`RAGSession` can reuse cached embeddings for unchanged chunks when a saved
index and the current embedding-model snapshot and chunking configuration
allow it. Changed/new chunks are embedded; removed entries disappear from
the new index. The lexical index and global code graph are rebuilt for the
complete updated corpus. If cache validation fails, rebuilding is safer than
mixing vectors from different settings.

A model ID string is not necessarily sufficient to identify the actual
cached weights. `retrievals/model_identity.py` resolves embedding-model
identity as used by cache validation. For downloads and explicit local
snapshots see [Model management](model_management.md).

## Retrieval versus generation

`RAGSession.ask()` maps retrieved chunks or selected documents to `Evidence`
records with source location and exact text. These are inputs to the
answer-generation prompt or a registered tool. It may append evidence cited
by related prior turns as configured. Retrieval **does not guarantee** that
the model uses, cites or correctly interprets its evidence; QAChat performs a
fast deterministic check for citation identifiers that do not correspond to
available evidence, records the invalid identifiers, and does not retry or
rewrite the answer solely to repair citations.

No universal evidence-token budget guarantees every retrieval result fits
in a model's context. Full-document reading has its own configured budget.
The index is a convenience for local content discovery; it does not assert
that an answer is scientifically valid.

## Tracing and performance analysis

The RAG and retrieval paths emit timed spans/events for preparation,
retrieval and generation; trace details can be expensive and sensitive.
[Performance](performance.md) analyzes existing diagnostic traces without
changing the retrieval implementation. Proposed retrieval optimization is
**deferred** until reproducible measurements and evidence-order invariance
checks justify it.

Relevant sources: `retrievals/persistence.py`, `strategies/retrieval_*`,
`code_graph/`, `document_reading.py`, `routing.py`, `rag.py`.
Relevant regression suites are those matching `retrieval`, `rag`,
`embeddings`, `chunking` and `project` in `tests/qachat/`.

Related: [Source pipeline](source_pipeline.md),
[Answering pipeline](answering_pipeline.md),
[Package defaults](package_defaults.md), [Model management](model_management.md).
