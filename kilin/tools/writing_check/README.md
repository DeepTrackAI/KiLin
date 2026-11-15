# writing_check

This package will own the `writing_check` registered tool, which checks the
grammar and style of one long document (an article, thesis chapter, report
or proposal) and returns suggested edits; it never rewrites the text. It is
being built in stages and is not registered yet.

So far it reads a document into sections and paragraphs (`_document.py`)
and chooses the part of it a question asks to check (`_scope.py`).

## Sections and paragraphs


- **LaTeX** (`.tex`): the document text between `\begin{document}` and
  `\end{document}` (the whole file for an `\input` chapter). Sections come
  from `\part` to `\subsubsection`, numbered as LaTeX prints them, and from
  the `abstract` environment. Math, float, verbatim and bibliography
  environments, and display math, are not prose. Each line is reduced to
  the text it prints with `kilin/tools/_shared/latex_text.py`; citations
  become `[cite]`, `\citeauthor` and `\citeyear` `[author]` and `[year]`.
- **Markdown** (`.md`): ATX headings, numbered when the title starts with
  a number ("2.1 Data"); code fences and table rows are not prose; links
  and emphasis are reduced to their words.
- **Plain text** (`.txt` and other suffixes): paragraphs only.

A paragraph ends at a blank line, a heading, a list item, or a line that
prints nothing; a comment line does not end it. Each paragraph keeps its
source lines, its readable text, its innermost section and its position
among that section's paragraphs, so a finding can point at the file and a
question can name "the second paragraph of the introduction".

## The part to check

The part is read from the question in code, without a model call, in this
order:

| The question | Part checked |
| --- | --- |
| Text in quotes, or after a colon (four words or more) | the paragraphs containing it; the text on its own when it is not in the document |
| "lines 120–180", "line 12" | the paragraphs touching those lines |
| "the second paragraph of the introduction", "paragraph 2 of section 1", "the last paragraph" | that paragraph, counted within the section (subsections included) or the document |
| "section 2.3", "§2" | that section, subsections included |
| "the methods section", "the section on data analysis" | the section with that title, or the only one whose title has all its words |
| "check the results", "check the methods" | a section whose title (or the only title containing the words) follows "the" |
| anything else | the whole document |

Names ignore case, numbers and a final "s"; file names in the question
("methods.tex") and words naming what to check ("the grammar", "the
style") are not section names. A section number or name the document does
not have, several sections matching a name, or a paragraph position past
the section's end gives a problem listing the headings, rather than
checking the whole document.
