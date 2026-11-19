# project_summary

This package owns the `project_summary` registered tool: one report on all
the documents of a project, written from a project template (`scope:
project`). Where a search-based answer reads the few passages a question
retrieves, this tool considers all indexed documents, up to the
configured `MAX_DOCUMENTS` limit. A skipped or failed document makes the
saved report explicitly partial. It is offered by the
`templated_project_report` intent of the Literature and Writing Assistant
project types.

For "write a project overview", the tool:

1. **Chooses the template**: the project template the question names (by
   title or complete file name), otherwise the project's only project
   template, or the one the user chooses from a list. Names that overlap
   are resolved by their complete boundaries; naming multiple templates
   is an error. Invalid choices cancel without reading any documents.
   The GUI receives the shared traced choice event before it waits for
   an answer. A document-scoped template named in the request or set
   as the fixed default is rejected; broken project templates are reported,
   including invalid UTF-8 files.
2. **Plans the work**: one model call per document and one per written
   section, with an estimated time (`ESTIMATES`). With `CONFIRMATION`
   `when_large`, a plan of more than `LARGE_BATCHES` calls asks first; the
   quick version ("Shorter reading") reads only each document's card and
   opening (`QUICK_DOCUMENT_TOKENS`). Documents beyond `MAX_DOCUMENTS`, in
   path order, are left out and the plan says so. The saved report
   also identifies omitted files in a **Coverage** section (up to ten
   filenames; all omitted paths remain in result metadata).
3. **Reads each selected document** (the map step, below).
4. **Writes the report** (the reduce step, below), checks it against the
   template (`TEMPLATE_VALIDATION`), and saves it to
   `outputs/<template>_project_report.md`, closed by the numbered list of
   the documents. The chat answer is the report; citation `[n]` names
   its indexed document, with a representative source passage shown as
   evidence. What the checks still found (sections
   outside their limits or citing doubtfully, sections that could not be
   written, documents that could not be read) is saved in the report, in
   a **Check** list before the documents. In any partial result, a
   prominent notice immediately below the headline identifies the report
   as incomplete even before the reader reaches the checks. The metadata
   reports `status: partial` and `valid: false` when a document failed,
   a document was omitted, a per-document evidence check is unresolved,
   or validation found a problem. `uncertain_checks` records every such
   condition with its document number and reason; the saved **Check**
   section shows up to ten. The `coverage` counts distinguish indexed,
   selected, and successfully read documents.

## Project templates

A project template is a report template (see the projects'
`templates/README.md` and `kilin/tools/_shared/report_templates.py`) with
`scope: project` in its front matter. Besides sections written from all
documents (`## Main Themes (180-260 words)`), it can have:

- **per-document sections**: `## Models (per document)` with `When:` (which
  documents) and `Do:` (what to write for each, optionally `(max 30
  words)`); `(per document, table)` with `Do: | Dataset | Size |` gives a
  table row per document. Templates support at most 26 per-document
  sections (lettered A–Z); larger templates are rejected before use;
- `**Documents:** {documents}`: the number of documents, written by code;
- `{key_documents}` in a section's instruction: the model picks documents
  (with an item limit), code writes their titles.

Two templates are shipped and installed in new projects: **Project
Overview** (`project_overview.md`: scope, themes, methods, findings,
gaps, where to start; scope, themes and methods cover all documents) and **Literature Table**
(`literature_table.md`: an overview covering all documents, a table row
per study with its aim, methods, data and main finding, and a line per
review or other document).
Templates for one document are left to `generate_paper_summary`, which
also lists the project templates when asked "which templates do I have?".

## Reading the selected documents (the map step)

It reads a short excerpt of each document (`_documents.py`), of about
`DOCUMENT_TOKENS` tokens:

- what the document's source card says (title, aim, methods, findings),
  when it has one;
- its opening passages (title, abstract); the first is always read;
- the passages that share most words with what the template asks for
  (its sections' names and instructions, and its per-document conditions
  and actions), in document order.

Documents are the project's indexed sources, numbered in path order; the
number is how the report cites a document (`[3]`). Notebook passages are
read by cell number and then by their cell-local line ranges. PDF pages
are likewise ordered by page number before page-local lines.

One model call per document (`_mapping.py`) asks for:

1. a note for each section written from all documents (`notes`
   sections): what the document contributes to it (for a section about
   the collection as a whole, what the document is about), or `NONE`;
2. for each per-document section, whether the document meets its `When:`
   condition (`YES`, `NO` or `UNCLEAR`), the excerpt's words that decide
   it, and for a yes the entry its `Do:` asks for; for a table, one
   `VALUE` line per column, each asked for by its column's name.

Code checks the model's output against **original indexed passages**,
separately from source cards and model-generated notes:

- Every segment of a quoted phrase must occur **in full and in order** in
  the supplied excerpt and original source. An ellipsis may separate
  segments from different passages, but a continuous quote may not be
  constructed by joining passage boundaries. A four-word prefix is no
  longer sufficient. Numerical signs, decimal values, percentages and
  mathematical inequalities are retained.
- A proposed `ENTRY` must be found verbatim in an original passage and
  the excerpt, before its word-limit trimming. Unsupported prose or
  invented model names become unclear rather than appearing as verified
  entries. The model is asked to use source wording; legitimate paraphrases
  may be flagged as unverified.
- Nonmissing `VALUE` cells must occur verbatim (after whitespace and
  punctuation normalization) in an original source passage as well as in
  the excerpt. If any cell is not found, the entire proposed row becomes
  **unclear** rather than publishing the unsupported values. `not
  reported` may represent a missing cell; a repeated column heading does
  not count as a value. Paraphrased but legitimate values may therefore
  also remain unclear.
- `CHECK ...: NO` is **not** established by silence or by a general quote.
  A negative result requires an exact quotation with explicit negation
  and a topic term overlapping the template's `When:` clause in the
  *negated clause*. The negation must apply to the condition's main
  action, not a related activity or a restricted subset. Double negatives
  ("did not fail to train"), partial quantifiers ("not all"), and limited
  conditions ("not trained on labeled data") remain **unclear**. For
  compound ``When:`` conditions, negating one alternative is not enough.
  This conservative lexical check cannot prove document-wide absence;
  unfamiliar grammar also remains **unclear**.
- Missing or invalid answers remain unclear. The model is instructed to
  copy quotations, entries and table values without inventing or silently
  normalizing them. Source validation is deterministic. It may trigger
  the existing bounded section-repair calls, but adds no checking model.

The report identifies unclear documents and their reasons in each
per-document section. When no matches are verified but some documents are
unresolved or unread, the section says "No confirmed matches" rather than
claiming none exist. Every unresolved check also makes the saved result
`status: partial`, `valid: false`, with details in the result metadata
and its **Check** section. Unverified table values are never published
as confirmed cells. A document whose call fails is reported as not read;
other documents continue.

## Writing the report (the reduce step)

`_reduce.py` writes each section in the template's order:

- **Sections from all documents**: one call per section from every
  document's note for it, citing documents as `[n]`, comparing and
  grouping them rather than describing them one by one. When the notes
  exceed `GROUP_TOKENS`, each group is first condensed into short lines of
  facts that keep their document numbers. The prompt asks for citations
  such as `[2]` or `[1, 4]` (never "Document 2"), agreements and
  disagreements between at least two documents, and no claims about what
  the documents lack unless the notes show it. Code then checks each
  section: every bracket must cite existing documents (`[n1]` or `[7]` of
  five does not), documents must not be named in the text ("Document 4"),
  and a section written from notes must cite at least one document. Each
  substantive sentence, independent list item and semicolon-separated
  assertion must also carry its own citation; a citation in a preceding
  sentence cannot validate an uncited claim. Text following the final
  citation must not introduce another unsupported assertion. Each
  citation is additionally checked against **original indexed source
  passages**, not the mapping model's notes or table entries. Numerical
  values, value-unit pairs and statistical inequalities must have
  matching literal anchors in their cited source, including signs,
  scientific exponents (`5e-3`) and comparison direction. The existing
  word-overlap check uses the clause preceding each citation and
  compares distinctive word stems against the original indexed passages,
  ignoring vocabulary common to most documents. If the cited document
  shares no distinctive term while another matches at least two, the
  model is asked to reconsider the citation; it is never silently changed
  by code. In strict mode, a cited phrase with no recognizable source
  anchor is flagged even without a better alternative. These checks do
  not prove semantic entailment; statements can remain unsupported even
  when some vocabulary overlaps the source. Literal checks identify
  absent numbers, reversed signs and inequalities, and uncited
  substantive assertions, but cannot prove scientific correctness. Short nonnumerical
  fragments may not constitute separate claims. A sentence saying
  documents agree, disagree, concur or conflict must cite at least
  two documents in each citation (the shipped Project Overview has no
  agreements section: notes taken one document at a time cannot show
  agreement between documents) ("agree on X [1], and on Y [2]" is sent
  back). A section marked `all documents` in its heading's brackets
  (`## Scope of the Collection (80-120 words, all documents)`) must cite
  every document read: one without a note for that section (a paper
  says nothing of "the scope of the collection") is represented by its
  title and its other notes, and the repair names those left out. A
  section with such a problem, or outside its limits, is rewritten up to
  `REPAIR_ATTEMPTS` times. Then code writes documents still named in the
  text as citations ("Document 1 and 4 both…" becomes "[1] and [4]
  both…", only when every number is a document), removes brackets still
  citing no document, and lists what remains wrong in the report's Check
  list.
- **Per-document sections**: assembled by code, never by the model: a
  count ("3 of 12 documents."), the unclear documents with why and those
  not read, then one entry or table row per matching document.
- **`{documents}`** and **`{key_documents}`**: the count, and the chosen
  documents' titles, written by code.
- **Citation evidence**: the tool's `[n]` citation identifies the entire
  document. The displayed evidence excerpt is selected from an original
  indexed passage containing a checked positive or negative quotation,
  where possible, rather than automatically taking only the first page.
  One excerpt cannot establish every claim about a long document; the
  report's document list retains the complete source path.
- **The headline**, when the template asks for one, in one call; a
  headline over its limit is asked for again, shorter, and if still too
  long cut at a word, without a final "and" or "of". Citations are
  removed from it.

## Defaults (`defaults.json`)

### `SETTINGS`

Overridable by a project type (`tool` object of an intent).

- `TEMPLATE` (default `"choose"`): the project template's file name
  without `.md`, or `choose` for the one the question names, the only one,
  or the user's choice.
- `DOCUMENT_TOKENS` (default `2500`): tokens read from each document
  (300-16000).
- `MAX_DOCUMENTS` (default `200`): most documents read (1-2000).
- `REPAIR_ATTEMPTS` (default `2`): rewrites of a section outside its
  limits or with a citation problem (0-4).
- `CONFIRMATION` (default `"when_large"`): `always`, `when_large` or
  `never`.
- `LARGE_BATCHES` (default `10`): with `when_large`, plans with more model
  calls than this ask first (0-1000).

### `READING`

- `QUICK_DOCUMENT_TOKENS` (default `600`): tokens read from each document
  in the quick version (100-16000).
- `GROUP_TOKENS` (default `6000`): most tokens of notes written from in
  one call; more are condensed in groups first (1000-64000).

### `TEMPLATE_VALIDATION`

- `LIMIT_TOLERANCE` (default `0.1`): fraction by which word limits may be
  missed either way (0-1).
- `MIN_TOTAL_WORDS` (default `50`): smallest `max_words` a template may
  set.

### `ESTIMATES`

Planning assumptions for the time shown in the plan, not limits.

- `PROMPT_TOKENS_PER_SECOND` (default `700.0`) and
  `REPLY_TOKENS_PER_SECOND` (default `25.0`): model speeds.
- `CALL_SECONDS` (default `1.0`): fixed time per call.
- `MAP_PROMPT_TOKENS` (default `300`) and `MAP_REPLY_TOKENS` (default
  `250`): a document call's instructions and reply.
- `SECTION_PROMPT_TOKENS` (default `300`) and `SECTION_REPLY_TOKENS`
  (default `400`): a section call's instructions and reply.

### `MODEL_CALLS`

- `THINKING` (default `"off"`): `inherit`, `on` or `off` for the tool's
  model calls.

## Limits

A document is read through a bounded excerpt, so some relevant facts
may never be provided to the model. The verification step checks source
wording, table values, numerical assertions, comparisons and citation
anchors deterministically. It cannot prove that a paraphrase is true, that
all relevant passages were read, or that a quoted fact supports the model's
broader interpretation. In particular, a valid source quotation may be
about a different experiment or data subset. `valid: true` therefore means
**the automated structural, coverage and source-anchoring checks passed**,
not that the scientific content has been independently established.
Unresolved source evidence remains visible in a partial report. Researchers
must check substantive conclusions against their original cited documents.
