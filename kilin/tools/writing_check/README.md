# writing_check

This package will own the `writing_check` registered tool, which checks the
grammar and style of one long document (an article, thesis chapter, report
or proposal) and returns suggested edits; it never rewrites the text. It is
being built in stages and is not registered yet.

So far it reads a document into sections and paragraphs (`_document.py`),
chooses the part of it a question asks to check (`_scope.py`), and runs the
checks that need no model (`_checks.py`).

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

## Checks done in code

Problems visible only across the whole document are found by code, in one
pass with no model call, before any paragraph is sent to a model:

| Check | Level | Finding |
| --- | --- | --- |
| Acronyms | consistency | first use before the definition; never defined (unless in `KNOWN_ACRONYMS`); defined again |
| Terms | consistency | a term written as one word, two words or hyphenated ("data set", "dataset"): each less frequent form, with the most frequent as suggestion |
| Spelling | consistency | US and UK spellings mixed: each use of the less frequent variant (`SPELLING_PAIRS`, with -d, -s, -ing, -ation and plural forms) |
| Promotional words | style | each use of a word in `PROMOTIONAL_WORDS` |
| Repeated words | grammar | a word written twice ("the the"); "that that" and "had had" allowed |
| Long sentences | style | sentences over `LONG_SENTENCE_WORDS` words |

A definition is "long form (ACR)" or "ACR (long form)", where the long
form's initials spell the acronym (hyphenated parts count as words, small
words such as "of" may be skipped). The abstract is checked on its own, as
journals expect it to define its acronyms again. Ties (as many "data set"
as "dataset", as many US as UK spellings) are not reported.

Each finding has its paragraph, source line, level, the words as they read
in the text, a suggestion (or none when it only flags) and a one-line
reason. The same pass builds the glossary given to every model call: the
defined acronyms, the preferred form of each term, and the spelling
variant.

### Settings (`checks.json`, section `CHECKS`)

The file becomes the tool's `defaults.json` when the tool is registered.

- `LONG_SENTENCE_WORDS` (default `40`, range 10–200): sentence length, in
  words, reported as long.
- `PROMOTIONAL_WORDS`: lowercase words or hyphenated phrases reported as
  promotional ("novel", "state-of-the-art", ...).
- `SPELLING_PAIRS`: `us:uk` entries ("color:colour"); "-ize"/"-ise" and
  "-yze"/"-yse" verbs and "-or"/"-our" and "-er"/"-re" nouns also match
  their inflected forms.
- `KNOWN_ACRONYMS`: lowercase acronyms readers know without a definition
  ("dna", "mri", roman numerals).
