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
   the documents. The chat answer is the report; each document's first
   passage is its evidence `[n]`. What the checks still found (sections
   outside their limits or citing doubtfully, sections that could not be
   written, documents that could not be read) is saved in the report, in
   a **Check** list before the documents. The result metadata
   reports `status: partial` and `valid: false` when a document failed,
   a document was omitted, or validation found a problem. The `coverage`
   counts distinguish indexed, selected, and successfully read documents.

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

Code then checks the reply. A yes becomes unclear when its quoted words
are not in the excerpt (ignoring case, punctuation and line breaks; a long
quote may be found by its first four words), when it has no entry, or
when a table row has no value, so no document is listed without evidence.
In a table, a column name written before a value ("Aim: ...") is removed,
and a value that only repeats its column's name is "not reported", so an
echoed header is not taken for a row. Entries are cut to their word limit. A
document whose call fails is reported as not read; the others go on.

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
  citation must also be supported: when the words it is cited for (the
  clause before it: since the previous citation, from the last comma,
  dash or "such as" when that is enough to judge) share no distinctive word with the cited
  document's title, notes and entries, while another document shares at
  least two, the section is sent back saying which document it may be
  ("“genomic analysis of psychiatric disorders” cites [4] … it may be
  [5]"); words found in more than half the documents are not
  distinctive, and code never changes a citation itself. A sentence
  saying documents agree, disagree, concur or conflict must cite at least
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

A document is read through an excerpt, not in full, so a fact outside it
is missed; per-document checks say "unclear" rather than guess. The
synthesis can still misstate what the notes say: check claims against the
cited documents. Validation of model-generated quotes and table values
against independent source passages is deferred to the evidence-validation
PR; `valid` here describes structural checks and document coverage only.
