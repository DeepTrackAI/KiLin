# reference_check

This package owns the `reference_check` registered tool. It checks one
paper's reference list against the other files of the project and finds
where each reference is cited in the paper's text. Finding the references,
their files and their citations calls no language model: those results are
deterministic and state their reasons. Checking whether each citing
sentence is supported by the cited work uses the answering model. A plan
is shown for confirmation when required by the confirmation policy.

For "check the references of the Smith paper", the tool:

1. Chooses the citing paper with `choose_document` (the named paper, the
   paper the earlier turns discussed, or the user's choice).
2. Reads the reference list with the shared extractor
   (`kilin/tools/_shared/references.py`). A PDF's list is read again in
   column order, and each entry keeps its printed number. The list ends at
   the next heading (Acknowledgements, Supplementary information, another
   reference list), a table or figure caption, or numbering that starts
   again at 1, so a table printed after it adds no entries; a number that
   does not follow the previous entry's (a page range wrapped onto a new
   line, "…919–\n927. https://doi…") stays inside its entry. A paper with
   several lists (for example main and supplementary references, each
   numbered from 1) has one list checked: the one the question names
   ("the supplementary references"), otherwise the user chooses; without
   anyone to ask, the first list is checked and the answer says so. The
   in-text citations are looked for in the text that cites that list.
3. Matches each entry to a project file (`_matching.py`): by the file's
   DOI, by its PDF metadata title and the entry's first author, or by the
   file's first page. Each entry is **found**, **probable** (for example
   the same title with another first author: usually a preprint of the
   published file), **missing**, or **unreadable** (too short to identify).
4. Finds numbered in-text citations (`_citations.py`): bracketed (`[3]`,
   `[2, 5–7]`) or parenthesized (`(6,7)`, not equation or step numbers)
   in any file, written out (`Ref. 50`, `refs. 12,13`, `refs 8,19,20`),
   and superscripts in PDFs, confirmed by the printed font size and
   raised position (exponents in formulas and subscripts are left out).
   Papers using only written-out numbered references are recognized.
   A sentence broken by a page break (page number, footer, running head)
   is joined first; a superscript marker broken over two lines
   ("accumulation⁵⁹⁻" / "⁶¹") is read whole when the rest is printed small
   at the start of the next line; and a count in parentheses before a
   superscript ("high (3)⁴⁵,⁴⁶") is not taken for a function argument.
   It reports references never cited and numbers cited beyond the list.

   A LaTeX paper (`.tex`) is read by `_latex.py` instead of steps 2 and 4.
   Its references come from a `thebibliography` environment in the file,
   else a **selected and indexed** `.bbl` companion, else the **selected
   and indexed** `.bib` files named by `\bibliography` or
   `\addbibresource` (with `@string` abbreviations expanded). The
   checker never opens unselected files, even when a LaTeX command names
   them or an unselected copy exists beside the paper. All files must
   already be in the project source inventory. A matching bibliography
   beside the referring `.tex` file takes precedence; otherwise one
   uniquely named selected file can be used. Ambiguous names, rejected
   paths and unavailable files are reported rather than guessed. The
   workspace source boundary, including symlink and traversal checks,
   is described in [Security](../../../docs/security.md). `\bibitem` entries keep their
   printed order; `.bib` entries are assigned inferred numbers in
   order of first citation and only cited ones are listed, since a
   `.bib` file often holds a whole library. These inferred numbers need
   not match the order in a compiled bibliography. Each `\cite`-family
   command (`\citep`, `\citet`,
   `\parencite`, `\autocite`, ...) is read as the numbers it prints, the
   rest of the LaTeX is reduced to readable text line by line, and the
   bibliography environment is left out of the text searched for
   citations. Keys cited but missing from the bibliography are listed in
   a note, not numbered; `\nocite` does not count as a citation.
5. Checks each claim (a citing sentence and one cited work that is in the
   project) with the answering model (`_shared/claims.py`). The model reads
   the cited file's passages most related to the sentence and to what
   the reference is cited for. To resolve phrases such as "this study",
   up to two preceding sentences and one following sentence from the
   same paragraph are supplied as context, within 140 words including
   the cited sentence. For anaphoric claims, the preceding context also
   augments the existing retrieval query, without another model call.
   Publication-attribution sentences (such as "The results of this
   study were published [22]") retrieve using the specific preceding
   study description, rather than requiring the cited work to say that
   it was published. The model must still find evidence that the cited
   paper actually reports the described study. The model answers
   with a verdict (supported, partly supported, not supported,
   contradicted, background, or no matching passage), the passage
   sentence it rests on, and a critical comment. A sentence citing
   several references together receives a model-proposed focus for each
   reference (one call per sentence), used only as an additional retrieval
   query. The original citing sentence stays authoritative for the
   verdict; a model-generated rewrite cannot establish support for
   something not asserted in that sentence.
   A claim found not supported, or without a matching passage, is read
   again with the next best-ranked passages (`RETRY_PASSAGES`) in a
   second call; a better verdict from that reading replaces the first,
   the comment identifies the second reading. Traces include passages
   from both readings and identify the final verdict stage. A cited
   file without searchable text requires no model call. Python
   checks the answer: a quote not found in the passages is dropped, and a
   reference is called contradicted only with a confirmed quote, otherwise
   not supported. The plan is traced, and the user can choose all
   claims, the first few, or none when confirmation is requested.
6. Rates eligible verdicts with the shared half-star citation-suitability
   rubric: 0–9 gives `½☆☆☆☆`, 10–19 `★☆☆☆☆`, and each additional ten
   points adds a half-star, with 90–100 giving `★★★★★`. The score is
   shown in parentheses, for example `★★★★½ (88/100)`. The score is the same
   value already stored in structured results and traces, not a new
   calculation. Unsupported, contradicted, unclear,
   failed, no-text and no-matching-passage results show unfilled stars
   marked **unrated**, not a fabricated zero-confidence score. Ratings
   reflect verified quotations and textual overlap within verdict bands,
   never scientific quality or a calibrated probability. The detailed
   report shows the preceding and following context and exact retrieval
   query strings for each checked claim; trace records the context,
   selected passage identifiers and location, quote passage identifier,
   evidence status, bibliography match status, exact quote line range
   when identifiable, and the rating. The report and chat separately
   give logical model requests by stage (focusing, initial checks,
   second looks). When the KíLín adapter is used, they also give actual
   generation attempts, including an eligible recovery attempt. The
   confirmation plan estimates requests, not a strict bound on attempts.
   No additional model calls or network operations are used.
7. Saves `outputs/<paper>_reference_check.md` with every reference, its
   file, the sentences citing it and their verdicts, and answers in the
   chat with the counts, the found, probable and missing references, the
   problems, and the claims that are not fully supported, citing the
   quoted passages as evidence. Citing sentences are shown with their
   markers as superscripts and the checked reference in bold; reference
   numbers are never written "[12]", which the chat would read as a link
   to evidence. Missing references have offline Google Scholar searches:
   by article title when recognizable, or by author, journal, year,
   volume, and first page/article number for title-less references.
   References with a printed DOI link directly to that identifier,
   labeled as unverified. An explicitly numbered, very short reference
   with a DOI or journal coordinates is retained by the extractor,
   even when below its normal minimum word count. An entry classified
   as too short for local matching still receives a search link when it
   contains journal coordinates.
   Links appear in the chat and beside every linkable entry in the report.
8. When the user requests online links, or enables `ONLINE_LOOKUP`,
   explicitly asks permission **before** any request to Crossref or
   OpenAlex. Refusal or absence of a confirmation interface leaves the
   existing offline links intact. An approved request sends only the
   missing entries and verified identifiers, never the manuscript body.
   A web-link request skips model-based claim checking. Lookup and
   claim-verification permissions are distinct, and approval does not
   authorize downloading PDFs. Search hits become verified DOI links
   only after corroborating the author, year and article title, or for
   title-less entries the journal, volume, first page/article number,
   year, and author when printed. Short entries with these fields remain
   eligible for optional lookup even if local file matching is uncertain.
   Ambiguous coordinate-only matches are
   not verified. Failed or uncertain lookups retain offline links.

Links are rendered as clickable HTTPS links in the KíLín chat GUI. The
system browser opens only when the user clicks one; generating a link
never opens a website. No PDFs are downloaded or added to sources.


For **probable** bibliography matches, verification results are explicitly
provisional until the identity of the cited file has been checked. Quotations
are labeled `exact`, `approximate`, `missing`, or `rejected` (and error states
are retained), so an unquoted model verdict is not presented as verified
source evidence. Quote validation rejects altered numbers, negations and
common changes of direction, while accepting harmless PDF extraction
artifacts. When the quote occurs verbatim, the reported source location is
narrowed from the chunk to the matching line(s). Repeated claim queries use
request-local ranking memoization, without changing the retrieved results.

The **answering model** used for internal verification inherits the main
interface's thinking, temperature and maximum output tokens. The tool
imposes no separate output cap for verification; `PASSAGE_TOKENS` instead
limits the *input* evidence. See [Generation settings](#generation-settings)
and the [shared policy](../../../docs/tool_packages.md#generation-settings-for-tools).
The router remains nonthinking and deterministic.

Limits: author-year citations ("Smith et al., 2020") are not recognised;
superscripts are recognised only in PDFs; an unusual layout can still split
or join reference entries. A claim is judged from a few passages of the
cited work, not all of it, so "no matching passage" means none of the
passages read is about the sentence, not that the paper has none; and the
model's verdicts can be wrong, so they are pointers for the
reader to check, with the quote and the place to look.

Its `defaults.json` is independent of `kilin_defaults.json`.
Project-type tool settings override the keys in `SETTINGS`; saved
project-specific settings can override the project type. See
[configuration precedence](../../../docs/configuration.md) for the full
order. Other sections are internal thresholds. All values are read and
validated on import.

## Defaults

### `SETTINGS`

- `MAX_LISTED` (default `20`): Most items listed in the chat for each kind
  (found, probable and missing references, checked claims, those needing
  attention first); the saved report lists all. Allowed range: 1–1000.
- `VERIFY` (default `true`): Whether to check claims with the answering
  model. Allowed values: true, false.
- `PASSAGES` (default `6`): Most passages of the cited work read per claim.
  Allowed range: 1–20.
- `PASSAGE_TOKENS` (default `5000`): Most estimated tokens of those
  passages (an indexed passage is often 700-1000); the best-ranked one is
  always read. Allowed range: 300–16000 (integer, not `null`). This is an
  *input* limit; the generation allowance is inherited separately.
- `RETRY_PASSAGES` (default `8`): When a claim comes out not supported
  or without a matching passage, this many further passages of the cited
  work (the next best-ranked, within `PASSAGE_TOKENS`) are read in a
  second model call before deciding; 0 turns the second look off.
  Allowed range: 0–20.
- `FOCUS_CLAIMS` (default `true`): For a sentence citing several
  references, the answering model proposes a reference-specific search
  hint. The original citing sentence remains authoritative for the
  verdict: the proposed wording is never used as independent evidence.
  One extra model call per such sentence. Allowed values: true, false.
- `MAX_CHECKED_CLAIMS` (default `100`): Most claims selected for
  verification; later ones are reported unchecked. This is not a model-
  call budget: a claim can need a focusing call and a second reading.
  Allowed range: 1–1000. For older `reference_check` overrides, rename
  `max_model_calls` to `max_checked_claims`; the old name is rejected.
  Other tools' `max_model_calls` settings are unaffected.
- `QUICK_CLAIMS` (default `5`): Claims checked by the quick version, in
  reading order. Allowed range: 1–100.
- `CONFIRMATION` (default `"when_large"`): When to ask before checking
  claims. Allowed values: always, when_large, never.
- `LARGE_BATCHES` (default `10`): With `when_large`, plans with more
  claims than this ask first. Allowed range: 0–1000.
- `ONLINE_LOOKUP` (default `false`): Offer online lookup during an
  ordinary reference check (without a link request). This is **not**
  authorization: an actual confirmation callback and affirmative user
  choice are required for each operation. Allowed values: true, false.
- `MAX_ONLINE` (default `100`): Maximum number of missing entries offered
  for one online lookup. Other missing references keep offline search
  links. Allowed range: 1–1000.

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
- `UNIT_WORDS` (default 15 words): Lowercase words after which a single
  digit is an exponent, not a citation (`cm2`, but `nm23` is cited);
  distinct lowercase words.
- `BYLINE_SHARE` (default `0.5`): Share of capitalized words above which a
  sentence is an author list or affiliations, whose superscripts are not
  citations. Allowed range: 0–1.

### `ESTIMATES`

- `JUDGE_REPLY_TOKENS` (default `256`): expected reply tokens for
  duration estimates only; never a generation cap. Range: 1-131072.

Planning assumptions for the time shown before claims are checked; they
are estimates, not limits.

- `PROMPT_TOKENS_PER_SECOND` (default `700.0`): Estimated prompt-processing
  throughput. Allowed range: 0.001–1000000000.
- `REPLY_TOKENS_PER_SECOND` (default `25.0`): Estimated generation
  throughput. Allowed range: 0.001–1000000000.
- `CALL_SECONDS` (default `1.0`): Fixed time of one model call, in seconds.
  Allowed range: 0–1000000000.
- `PROMPT_OVERHEAD_TOKENS` (default `450`): Instruction tokens around the
  passages of one claim. Allowed range: 0–100000.
- `REPLY_SHARE` (default `0.6`): Share of the answer budget a verdict uses
  on average. Allowed range: 0–1.


## Generation settings

Output allowances come from the active session, including per-question GUI
or API settings. They are ceilings for each model call, not target lengths
or a total budget for a multi-call tool. Prose length is controlled by the
prompt or report template; input passage budgets remain separate.

`MODEL_CALLS` contains one key: `THINKING` (packaged value `"inherit"`).
It accepts `inherit`, `on`, or `off` for internal `context.ask_model` calls.
It never infers thinking from a token count. `context.write` always inherits
main-session thinking. Router calls remain nonthinking and separately bounded.

Thinking requests use the user's full allowance. Only nonthinking requests
may use an explicit smaller cap, never exceeding the user's maximum.
Changing `THINKING` to `inherit` resets this tool to the main setting; `on`
forces thinking and bypasses local short-response caps. Restart after editing
cached defaults. See the [shared policy](../../../docs/configuration.md#internal-tool-thinking)
for precedence, the retained exceptions, and migration of obsolete keys.

## Partial completion

An incomplete evidence judgment is `failed` and unrated, not unsupported.
Other claims are still checked, and both the saved report and answer identify
incomplete items with partial status. Existing support and quotation checks
apply to every completed judgment. The shared internal adapter retries only
length-exhausted unfinished thinking, once without thinking.
