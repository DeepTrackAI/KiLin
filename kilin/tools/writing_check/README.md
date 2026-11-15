# writing_check

This package owns the `writing_check` registered tool. It checks the
grammar and style of one document of the project (an article, thesis
chapter, report or proposal), or a part of it, and suggests changes; it
never changes the document. The checks across the whole document
(acronyms, terms, spelling) are done in code, deterministically; each
paragraph is checked by the answering model, and every suggestion it
makes is checked before it is reported. A plan is shown for confirmation
when required by the confirmation policy.

The `writing_check` intent is offered in the literature and the writing
(Writing Assistant) project types. Its cues send questions such as "check
the grammar of …", "proofread the manuscript" or "are there typos …" to it
without the router, ahead of the writing assistant's own rewriting intent;
"rewrite this", "polish the introduction" or "proofread this: <text>" stay
with that intent. Other questions reach it only when they use words such as
grammar, spelling, typos, style or wording (`requires`).

For "check the grammar and style of the methods of draft.tex", the tool:

1. Chooses the document with `choose_document` (the named file, the file
   the earlier turns discussed, or the user's choice) and reads its
   selected source file (`_shared/source_access.py`): `.tex`, `.md` and
   `.txt` as written; `.docx` through its source reader, as Markdown;
   `.doc` and `.pdf` through theirs, as plain text (their line numbers are
   those of the extracted text).
2. Reads it into sections and paragraphs (`_document.py`) and chooses the
   part the question names (`_scope.py`). When it names a part the
   document does not have, the user chooses among the headings (or the
   whole document); where nobody can be asked, the answer lists them.
3. Runs the checks done in code on the whole document (`_checks.py`) and
   keeps the findings in the chosen part; their glossary goes into every
   model call.
4. Plans one model call per paragraph and check (grammar, and style when the
   question asks for it or at level `all` or `style`), plus one to describe
   the document when it has no card, and asks first when the plan is large.
   The quick version checks the grammar of the first `QUICK_PARAGRAPHS`
   paragraphs; cancelling keeps the findings made in code.
5. Checks each paragraph (`_review.py`) and drops suggestions that fail
   the checks below.
6. Answers with the first `CHAT_FINDINGS` findings under their section
   headings ("line 12 · grammar: “We tracks” → “We tracked”
   (agreement)") and saves the full report, one table per section, as
   `outputs/<document>_writing_check.md`.

Failed model calls are reported per paragraph (a partial result); the
other paragraphs are still checked.

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

## Checks done by the model

Each paragraph is sent to the answering model on its own: once for
grammar (clear errors only: agreement, verb forms, tense, articles,
prepositions, punctuation, spelling, missing or extra words) and, when
asked, once for style (at most five changes, only where a reader would
stumble: wordy phrases, ambiguous references, awkward constructions).
Neither may replace a correct word with a synonym or to sound more
formal, nor add, remove or soften a claim, hedge or modal verb. Each call
is told:

1. the document: its card's title, venue, aim, methods and findings
   (`ToolContext.card`), or, without a card, a two- or three-sentence brief
   written by one model call from the headings and the first paragraphs;
2. the glossary from the code checks (acronyms, preferred terms,
   spelling), to keep;
3. the section, with what it is for when the title says ("methods: what
   was done, usually in the past tense");
4. the paragraph's readable text, to check, and the last sentence before
   it and the first sentence after it, marked as context not to check or
   quote.

The model answers with `ORIGINAL`, `SUGGESTION` and `REASON` blocks, or
`NONE`. Its quoted words are found in the paragraph with spaces, quotation
marks, dashes and case ignored, and without a full stop, comma or ellipsis
it added at either end; the finding then shows the paragraph's own words,
cut to the words that change with up to two words of context on each side
(models often quote a whole sentence to change one word).
A suggested change is dropped, not reported, when its original
words are not in the paragraph, when nothing changes, or when it changes
numbers, negations, citation placeholders (`[cite]`, `(ref)`, ...), hedges
or modal verbs ("virtually", "approximately", "would", "may", ...), removes
a defined acronym or preferred term, or writes a term form the document
does not use, or when it quotes the surrounding sentences instead of the
paragraph. Dropped suggestions are counted by reason in the chat and listed
in the report's "Dropped suggestions" table with why each was dropped,
except those that only quote the context or change nothing, which are only
counted. A failed call is reported for that paragraph and the others
are still checked; without a brief, the paragraphs are checked without it.

## Settings (`defaults.json`)

### `SETTINGS`

Project types may override these in the intent's `tool` object (lower
case), and users per project.

- `LEVEL` (default `auto`): model checks to run: `auto` (grammar, and
  style when the question asks about style, wording, clarity, concision,
  readability, flow or tone), `grammar`, `style` or `all` (both). The
  checks done in code always run.
- `CONFIRMATION` (default `when_large`): when to show the plan first:
  `always`, `when_large` or `never`.
- `LARGE_BATCHES` (default `30`, range 0–1000): with `when_large`, plans
  with more model calls than this ask first.
- `QUICK_PARAGRAPHS` (default `10`, range 1–1000): paragraphs checked for
  grammar by the quick version.
- `CHAT_FINDINGS` (default `20`, range 0–200): findings shown in the chat;
  all are in the saved report.

### `CHECKS`

- `LONG_SENTENCE_WORDS` (default `40`, range 10–200): sentence length, in
  words, reported as long.
- `PROMOTIONAL_WORDS`: lowercase words or hyphenated phrases reported as
  promotional ("novel", "state-of-the-art", ...).
- `SPELLING_PAIRS`: `us:uk` entries ("color:colour"); "-ize"/"-ise" and
  "-yze"/"-yse" verbs and "-or"/"-our" and "-er"/"-re" nouns also match
  their inflected forms.
- `KNOWN_ACRONYMS`: lowercase acronyms readers know without a definition
  ("dna", "mri", roman numerals).

### `ESTIMATES`

Planning assumptions for one paragraph call, shown in the plan; not
limits: `PROMPT_TOKENS_PER_SECOND` (`700`), `REPLY_TOKENS_PER_SECOND`
(`25`), `CALL_SECONDS` (`1`), `PROMPT_OVERHEAD_TOKENS` (`700`, the
context, paragraph and instructions) and `REPLY_TOKENS` (`150`).

### `MODEL_CALLS`

- `THINKING` (default `off`): the tool's paragraph calls are short and
  many, so thinking is off, as for the batch tools.
