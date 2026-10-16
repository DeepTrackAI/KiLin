# reference_check

This package owns the `reference_check` registered tool. It checks one
paper's reference list against the other files of the project and finds
where each reference is cited in the paper's text. It calls no language
model: every result is deterministic and states its reason.

For "check the references of the Smith paper", the tool:

1. Chooses the citing paper with `choose_document` (the named paper, the
   paper the earlier turns discussed, or the user's choice).
2. Reads the reference list with the shared extractor
   (`kilin/tools/_shared/references.py`). A PDF's list is read again in
   column order, and each entry keeps its printed number.
3. Matches each entry to a project file (`_matching.py`): by the file's
   DOI, by its PDF metadata title and the entry's first author, or by the
   file's first page. Each entry is **found**, **probable** (for example
   the same title with another first author: usually a preprint of the
   published file), **missing**, or **unreadable** (too short to identify).
4. Finds numbered in-text citations (`_citations.py`): bracketed (`[3]`,
   `[2, 5–7]`) in any file, and superscripts in PDFs, confirmed by the
   printed font size. It reports references never cited and numbers cited
   beyond the list.
5. Saves `outputs/<paper>_reference_check.md` with every reference, its
   file, and the sentences citing it, and answers in the chat with the
   counts, the found, probable and missing references, and the problems.

Limits: author-year citations ("Smith et al., 2020") are not recognised;
superscripts are recognised only in PDFs; an unusual layout can still split
or join reference entries. Whether a citing sentence is supported by the
cited work is not checked here.

Its `defaults.json` is independent of `kilin_defaults.json`.
Project-type tool settings override the keys in `SETTINGS`. Other sections
are internal thresholds. All values are read and validated on import.

## Defaults

### `SETTINGS`

- `MAX_LISTED` (default `20`): Most references listed in the chat for each
  status (found, probable, missing); the saved report lists all. Allowed
  range: 1–1000.

### `REFERENCES`

Thresholds passed to the shared reference-list extractor as a
`ReferenceLayout`.

- `REFERENCE_CHARACTERS` (default `100000`): Most characters read after the
  references heading; long reviews need more than a report does. Allowed
  range: 1–1000000.
- `NUMBERED_ENTRIES_MIN` (default `5`): Fewest numbered lines for entries
  to be split at their numbers. Allowed range: 1–1000.
- `RUNNING_LINE_REPEATS` (default `3`): Repetitions that make a short line
  a running head or footer. Allowed range: 1–1000.
- `RUNNING_LINE_WORDS` (default `12`): Most words of a running head or
  footer. Allowed range: 1–1000.
- `ENTRY_MIN_WORDS` (default `5`): Fewest words of an entry. Allowed range:
  1–1000.

### `MATCHING`

Thresholds for matching entries to files (`MatchRules`).

- `FIRST_CHARACTERS` (default `1500`): Characters at the start of a file
  read as its first page. Allowed range: 100–100000.
- `TITLE_WORDS_MIN` (default `3`): Fewest distinctive words of a metadata
  title worth trusting. Allowed range: 1–100.
- `TITLE_ON_PAGE` (default `0.6`): Share of the metadata title's words that
  must be printed on the first page for it to be trusted (so "Microsoft
  Word - draft.docx" is not). Allowed range: 0–1.
- `TITLE_FOUND` (default `0.8`): Share of a trusted title's words an entry
  must contain to be found. Allowed range: 0–1.
- `TITLE_PROBABLE` (default `0.5`): Lower share that makes the file
  probable, with a matching first author. At most `TITLE_FOUND`.
- `PAGE_FOUND` (default `0.85`): Without a trusted title, share of the
  entry's words that must be on the first page. Allowed range: 0–1.
- `PAGE_PROBABLE` (default `0.65`): Lower share that makes the file
  probable, with a matching first author. At most `PAGE_FOUND`.
- `ENTRY_WORDS_MIN` (default `4`): Fewest distinctive words of an entry;
  shorter entries are unreadable. Allowed range: 1–100.
- `COMMON_WORDS` (default 17 words): Lowercase words that do not tell works
  apart (`journal`, `review`, ...); distinct lowercase words.

### `CITATIONS`

Thresholds for finding in-text citations (`CitationRules`).

- `BRACKET_MARKERS_MIN` (default `5`): Fewest bracketed markers for a PDF
  to use brackets rather than superscripts; a file without superscripts
  uses any brackets. Allowed range: 1–1000.
- `SUPERSCRIPT_RATIO` (default `0.85`): A glyph smaller than this share of
  its line's median letter size is a superscript. Allowed range: 0.1–1.
- `SIGNATURE_CHARACTERS` (default `8`): Letters and digits before a
  superscript compared between the PDF and the text. Allowed range: 3–100.
- `MAX_RANGE` (default `50`): Longest range a marker may span ("1–4" spans
  3). Allowed range: 1–1000.
- `UNIT_WORDS` (default 15 words): Lowercase words after which digits are
  exponents, not citations (`cm`, `log`, ...); distinct lowercase words.
- `BYLINE_SHARE` (default `0.5`): Share of capitalized words above which a
  sentence is an author list or affiliations, whose superscripts are not
  citations. Allowed range: 0–1.
