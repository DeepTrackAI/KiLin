# Retrieval and index architecture

## Responsibilities and public interfaces

KíLín separates **source processing** from **retrieval**. The former yields
provenance-bearing `Chunk` records; the latter produces several rankings,
combines them and expands evidence for explicit code or document references.
The main orchestration belongs to `kilin.rag.RAGSession` and the algorithms
live under `kilin/retrievals/`.

![Semantic, lexical and code-aware channels rank aligned chunks before reciprocal-rank fusion and evidence selection.](figures/retrieval-indexes.svg)

*Figure: The three channels preserve different ranking signals; reciprocal-rank fusion combines their ranks, and persisted indexes must pass integrity and model-identity validation.*

`kilin.retrievals.__init__` exposes the supported records and functions;
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

Indexed text is not always the displayed passage. A Python or MATLAB code
passage is embedded and lexically indexed with its file path in front and,
for Python, the first line of its module docstring (`_retrieval_text` in
`rag.py`), so a function can be found by what its module is about even when
its own name and code use other words. Evidence, citations and the answering
model keep the passage text itself. The format is recorded in the index
settings (`retrieval_text`); changing it rebuilds saved indexes.

In project types with the `paper_index` capability (the packaged
`literature` type), each document passage is indexed after a short header:
the document's title, its first author and file name, and the section
heading and page it comes from (`retrievals/papers.py`). Titles and first
authors come from the PDF metadata when plausible, from the first `#`
heading of a Markdown file, or else from the file name; section headings
are recognized by name ("Methods", "Data availability") or numbering
("3.1. Software architecture"). A methods paragraph is then found for a
question about its paper even when it never repeats the paper's name. The
header format is recorded in the index settings (`document_headers`).

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
related chunks. The channel runs only when the question names a file,
function, or class (`names_code` in `context_expansion.py`). For ordinary
prose, it would match common words to identifiers and introduce noise.
Graph structure and chunk provenance must agree: a stale graph can misdirect
named-function context even when semantic vectors remain readable.

Expansion also follows retrieved documentation, module headers, and test
calls to implementing functions. Exact qualified names and uniquely resolved
suffixes are preferred. An unresolved qualified name can still suggest a
unique leaf-name alternative. Supported matches consume the documentation
bridge budget first; ambiguous leaf names are not guessed. Alternatives
remain labeled as such rather than being asserted to identify the named code.

Related functions compete for `RETRIEVAL.RELATED_PASSAGES_MAX` passages,
using the **same validated cosine scores** already computed by semantic
search. Non-unit embeddings retain the ordinary cosine contract. No extra
query embedding, vector conversion, or rescoring is performed for expansion.
The current loop backend already sorts all chunks; RAG retains that ranking
and slices each channel's intended depth before fusion, so reuse does not
silently deepen the fused semantic channel. Scores live only for one query.

Explicit files and named functions are included in full. Every direct
neighbor requested by a relationship question contributes context outside
the shared passage budget, but each long related function is still limited
to `RELATED_CHUNKS_MAX` chunks (currently two). Those chunks prioritize call
sites and question terms. Zero means unlimited for either passage cap;
these settings are not a promise that every related function is included
in full. Alternatives follow supported related passages in the shared budget.

For source-preferring intents, each channel ranks `SOURCE_SEARCH_DEPTH`
times the retrieval count. Its best source-code results retain their order,
followed by its overall best `SOURCE_SEARCH_OTHERS`, before fusion.
`SOURCE_SEARCH_RESULTS` fused results are kept, or the retrieval count when
larger. Documentation and tests count as wanted results when asked for.
Tutorials and tests can otherwise outrank the implementing code merely
because they use a question's words more often.

### Requested targets and broader evidence

A named-file scope is a **focus**, not a hard exclusion of other files.
The requested file's best passages seed retrieval; expansion can still find
helpers, tests, documentation, and plausible alternatives elsewhere. The
full semantic ranking is reused when ranking a focused file, preserving
passages outside the original top results without another embedding call.

Final RAG evidence prioritizes requested targets, retains the existing
ordering within each relevance group, and labels `context_role` as `direct`,
`related`, or `alternative`. Direct targets win over other labels. The prompt
instructs the model to attribute alternatives to their actual paths and
symbols, not to equate differently qualified names without evidence. The
optional label is persisted with evidence and included in traces; older
records without it load as unclassified (`None`). When evidence is reused
from an earlier turn, its old direct-target label becomes related; an
alternative warning is retained. Source text is unchanged.

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

## Generic source-card contracts (PR3)

The `retrievals/source_cards/` package defines reusable **source-card
records** without activating a new indexing or retrieval path yet. A
`SourceDescriptor` describes a selected file by its relative path, known
title, suffix and existing chunk identifiers, including files without
extractable text. A file extension **does not classify its content**: a PDF
is not necessarily a scientific paper, and a chunkless image need not have
a card.

A `SourceCard` contains the source path, registered card-kind ID, schema and
validation versions, source SHA-256, optional model/generation fingerprints,
kind-specific metadata and substantive `CardStatement` records. Statements
refer to original chunk identifiers rather than PDF-only page coordinates.
Generic validation checks source ownership, nonempty fields, duplicate
references and that citations resolve to chunks permitted for card creation.
A kind-specific validator remains responsible for content rules; citations
establish traceability, **not proof of semantic support**. Only original
source chunks can provide final answer evidence.

`SourceCardResult` separately represents `valid`, `not_applicable` and
`failed` outcomes. The `SourceCardRegistry` stores trusted Python kind
implementations and rejects unknown/duplicate IDs or obsolete versions.
A kind's deterministic `precheck` may rule out clearly ineligible sources;
positive results are only candidates, with full eligibility deferred to the
kind's generation/verification logic. Project JSON never imports kind code.

**This is a contract-only addition.** `paper_cards.py`, the existing
`paper_index` capability, `indexes/paper_cards.json` and the `papers`
retrieval strategy continue unchanged. Independent card indexing,
registration of `scientific_paper`, new generation models and automatic
card creation are separate later PRs; no migration layer is needed.

## Paper cards and paper retrieval

In project types with `paper_index`, every document may get a **paper
card** (`retrievals/paper_cards.py`): title, authors, year, a short
abstract, and one or two sentences each on its aim, methods, data, main
findings and limitations, with the pages each field cites. The answering
model writes a card once per document, without thinking, from the
document's opening passages and the passages that best match each field,
within about 5,000 estimated tokens. Only summary fields citing a passage
the model was shown are retained. At least one substantive cited field is
required; unusable replies produce a title-only incomplete card without
embeddings.

Cards are saved in `indexes/paper_cards.json` under a versioned format
(`paper-card-v2`), with the SHA-256 of the source bytes. Cards are reused
while the source and card format are unchanged. An incomplete card is
retried when the source or answering model changes; failed regeneration
still updates the saved card state. Three texts of each complete card
(abstract and aim; methods and data; findings and limitations) are
embedded separately. Missing, malformed or non-finite cached embeddings
are rebuilt without repeating card generation when the embedding model
is available. A cached card marked complete without summary content is
discarded and regenerated. Card-generation or embedding failures never stop a session
from opening. Cards guide retrieval only: answers cite passages, never
cards.

Intents whose retrieval strategy is `papers` (in `literature`:
`question_answering`, `compare_sources` and `report_generation`) choose
papers before passages (`strategies/retrieval_papers.py`). Papers are
ranked by reciprocal-rank fusion of two lists: the similarity of the
question to each card's vectors, and the rank of each paper's best passage
in the full hybrid ranking. The query is embedded once for both. Unless
the question asks about every paper ("all five papers", "each study",
"across the articles"), only papers with a passage among the best `retrieval_count`
passages of the collection are kept, at least two in a comparison, so that
comparing two papers does not take in every paper of the collection. The
best `max_papers` of them are chosen; with `passages_per_paper` zero, the
evidence is the best passages of those papers with the best paper always
represented, and otherwise each chosen paper receives that many passages
before the best remaining ones fill the evidence. A question that means
one document keeps the named-document behavior below, except in a
comparison, where that paper is chosen first together with the other
relevant papers; several named documents limit the papers that can be
chosen. Without cards, papers are ranked by their passages only. In these
projects a paper named in the question is searched, never added in full:
the code-oriented expansion that appends every passage of a named file
applies to code files only.

Which document a follow-up means (`document_scope` in
`tools/_shared/choose_document.py`) is decided by its wording when that is
clear: words for one document ("its limitations", "that study") keep the
document the earlier turns were about, and words for several ("these
articles", "the other papers", "all") search the whole project. The router
model judges only the remaining requests.

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

`retrievals/context_expansion.py` separately expands evidence near named source
references using `ContextExpansionPlan` and the graph. This expansion is not the
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

### Projects without indexed passages

A project with no selected sources, or no extractable chunks, has index
status `empty`. RAG keeps an empty retrieval container and the current
source inventory in memory; it does not persist a dummy index or call the
query embedder. Previously saved caches are not served as evidence in this
state. Adding sources invokes normal indexing on refresh. Existing model
loading and the selected type's prompt/tool policies are unchanged.

## Retrieval versus generation

`RAGSession.ask()` maps retrieved chunks or selected documents to `Evidence`
records with source location and exact text. These are inputs to the
answer-generation prompt or a registered tool. It may append evidence cited
by related prior turns as configured. Retrieval **does not guarantee** that
the model uses, cites or correctly interprets its evidence; KíLín performs a
fast deterministic check for citation identifiers that do not correspond to
available evidence. [Answering](answering_pipeline.md#citation-checks-and-answer-truthfulness)
owns that diagnostic and its limitations.

No universal evidence-token budget guarantees every retrieval result fits
in a model's context. Full-document reading has its own configured budget.
The index is a convenience for local content discovery; it does not assert
that an answer is scientifically valid.

## Tracing and performance analysis

The RAG and retrieval paths emit timed spans/events for preparation,
retrieval and generation; trace details can be expensive and sensitive.
[Performance](performance.md) analyzes existing diagnostic traces without
changing retrieval. Further optimization requires reproducible measurements
and evidence-order invariance checks, not only fewer model calls.

Progress wording describes actual work, not project IDs. Relationship
search is announced only when the graph has entities, with actual result
counts; otherwise the message explicitly reports that it was skipped.

## Change impact and verification

Rank fusion deliberately compares channel order rather than incompatible
raw score scales. A change to indexing text, tokenizer, embedding identity,
context expansion or rank depth can therefore alter evidence without changing
the final answer API. Review persistence identity, RAG orchestration and
evidence labels alongside the specific ranking algorithm.

Run [retrieval algorithms and persistence](../tests/kilin/retrievals/),
[index provenance](../tests/kilin/test_rag__index_provenance.py),
[retrieval corrections](../tests/kilin/test_rag__retrieval_corrections.py)
and [empty workspaces](../tests/kilin/test_rag__empty_workspace.py).
Check stable source coordinates, tie ordering, direct/related/alternative
labels, old-cache rejection and explicit failure cases. A deterministic
unit substitute validates structure, not live embedding quality; evaluate
ranking quality separately before claiming a retrieval improvement.

## Related

[Source pipeline](source_pipeline.md),
[Answering pipeline](answering_pipeline.md),
[Package defaults](package_defaults.md), [Model management](model_management.md).
