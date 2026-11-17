# project_summary

This package owns the `project_summary` registered tool: one report on all
the documents of a project, written from a project template (`scope:
project`). Where a search-based answer reads the few passages a question
retrieves, this tool reads every document once, so the report covers the
whole project. It is offered by the `templated_project_report` intent of
the Literature and Writing Assistant project types.

For "write a project overview", the tool:

1. **Chooses the template**: the project template the question names (by
   title or file name), otherwise the project's only project template, or
   the one the user chooses from a list. Where nobody can be asked, the
   answer lists the project templates and how to name one.
2. **Plans the work**: one model call per document and one per written
   section, with an estimated time (`ESTIMATES`). With `CONFIRMATION`
   `when_large`, a plan of more than `LARGE_BATCHES` calls asks first; the
   quick version ("Shorter reading") reads only each document's card and
   opening (`QUICK_DOCUMENT_TOKENS`). Documents beyond `MAX_DOCUMENTS`, in
   path order, are left out and the plan says so.
3. **Reads every document** (the map step, below).
4. **Writes the report** (the reduce step, below), checks it against the
   template (`TEMPLATE_VALIDATION`), and saves it to
   `outputs/<template>_project_report.md`, closed by the numbered list of
   the documents. The chat answer is the report; each document's first
   passage is its evidence `[n]`. Sections still outside their limits,
   sections that could not be written and documents that could not be
   read are listed under the report.

## Project templates

A project template is a report template (see the projects'
`templates/README.md` and `kilin/tools/_shared/report_templates.py`) with
`scope: project` in its front matter. Besides sections written from all
documents (`## Main Themes (180-260 words)`), it can have:

- **per-document sections**: `## Models (per document)` with `When:` (which
  documents) and `Do:` (what to write for each, optionally `(max 30
  words)`); `(per document, table)` with `Do: | Dataset | Size |` gives a
  table row per document;
- `**Documents:** {documents}`: the number of documents, written by code;
- `{key_documents}` in a section's instruction: the model picks documents
  (with an item limit), code writes their titles.

Two templates are shipped and installed in new projects: **Project
Overview** (`project_overview.md`: scope, themes, methods, findings,
agreements and disagreements, gaps, where to start) and **Literature
Table** (`literature_table.md`: a table row per study with its aim,
methods, data and main finding, and a line per review or other document).
Templates for one document are left to `generate_paper_summary`, which
also lists the project templates when asked "which templates do I have?".

## Reading every document (the map step)

It reads a short excerpt of each document (`_documents.py`), of about
`DOCUMENT_TOKENS` tokens:

- what the document's source card says (title, aim, methods, findings),
  when it has one;
- its opening passages (title, abstract); the first is always read;
- the passages that share most words with what the template asks for
  (its sections' names and instructions, and its per-document conditions
  and actions), in document order.

Documents are the project's indexed sources, numbered in path order; the
number is how the report cites a document (`[3]`).

One model call per document (`_mapping.py`) asks for:

1. a note for each section written from all documents (`notes`
   sections): the document's facts about it, or `NONE`;
2. for each per-document section, whether the document meets its `When:`
   condition (`YES`, `NO` or `UNCLEAR`), the excerpt's words that decide
   it, and for a yes the entry its `Do:` asks for (for a table, one value
   per column).

Code then checks the reply. A yes becomes unclear when its quoted words
are not in the excerpt (ignoring case, punctuation and line breaks; a long
quote may be found by its first four words) or when it has no entry, so no
document is listed without evidence. Entries are cut to their word limit,
and table rows padded with "not reported" or cut to their columns. A
document whose call fails is reported as not read; the others go on.

## Writing the report (the reduce step)

`_reduce.py` writes each section in the template's order:

- **Sections from all documents**: one call per section from every
  document's note for it, citing documents as `[n]`, comparing and
  grouping them rather than describing them one by one. When the notes
  exceed `GROUP_TOKENS`, each group is first condensed into short lines of
  facts that keep their document numbers. A section outside its limits is
  rewritten up to `REPAIR_ATTEMPTS` times.
- **Per-document sections**: assembled by code, never by the model: a
  count ("3 of 12 documents."), the unclear documents with why and those
  not read, then one entry or table row per matching document.
- **`{documents}`** and **`{key_documents}`**: the count, and the chosen
  documents' titles, written by code.
- **The headline**, when the template asks for one, in one call.

## Defaults (`defaults.json`)

### `SETTINGS`

Overridable by a project type (`tool` object of an intent).

- `TEMPLATE` (default `"choose"`): the project template's file name
  without `.md`, or `choose` for the one the question names, the only one,
  or the user's choice.
- `DOCUMENT_TOKENS` (default `2500`): tokens read from each document
  (300-16000).
- `MAX_DOCUMENTS` (default `200`): most documents read (1-2000).
- `REPAIR_ATTEMPTS` (default `1`): rewrites of a section outside its
  limits (0-4).
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
cited documents.
