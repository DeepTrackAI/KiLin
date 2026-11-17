# project_summary

This package will own the `project_summary` registered tool: one report on
all the documents of a project, written from a project template
(`scope: project`, see `kilin/tools/_shared/report_templates.py`).

## Reading every document (the map step)

A project report needs every document to contribute, so it does not
search. It reads a short excerpt of each document (`_documents.py`):

- what the document's source card says (title, aim, methods, findings),
  when it has one;
- its opening passages (title, abstract), always;
- the passages that share most words with what the template asks for
  (its sections' names and instructions, and its per-document conditions
  and actions), within a token budget, in document order.

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
and table rows padded with "not reported" or cut to their columns.
