# suggest_reference

This package owns the `suggest_reference` registered tool. Given a
statement the user wants to cite, it finds the papers in the project that
could be cited for it, and reports each model assessment with a
confirmed quotation when available. Finding the candidate papers
calls no language model; judging each one uses the answering model,
with the same checks as `reference_check`.

For "suggest a reference for: anomalous diffusion is common in living
cells", the tool:

1. Takes the statement from the question: quoted text, or the text after a
   colon; otherwise the router model copies it out of the question, and
   its copy is used only when most of its words are the question's. A
   statement shorter than `MIN_WORDS` words gets a request to write it.
   A request for the paper to cite for a named software, method or
   dataset ("which paper should I cite for BRAPH2?", "the reference for
   TrackMate") is recognized without the router: the name has at most four
   words, each with a capital letter or digit, or a word such as
   "software" or "dataset". The statement becomes "BRAPH2 (BRAPH 2) was
   introduced (first presented) in this work", with both spellings of a
   version number, so that the paper that introduces it is found and
   judged, rather than papers that merely use it. A lower-case name
   ("scikit-learn") is not recognized and is taken as a statement.
2. Uses the configured `DISCOVERY_STRATEGY`: `hybrid` (default) ranks
   project passages and keeps the `FILES` best-related papers, while
   `source_first` ranks sources across semantic, lexical, graph, and
   scientific-paper-card channels within a separate `DISCOVERY_TOKENS`
   evidence budget. An independent paper is
   **not** excluded merely because it contains the requested sentence
   verbatim; only a positively identified draft should be excluded.
3. Judges each paper with the answering model (`_shared/claims.py`): its
   `PASSAGES` passages most related to the statement, and a verdict
   (supported, partly supported, background, not supported, contradicted,
   or no matching passage, when none of the passages read is about the
   statement), the passage sentence it rests on, and a critical comment.
   A file without searchable text is reported as no text, without a model
   call.
   The model is told that a paper repeating the claim with a citation is
   not its source: it answers partly supported and says the original is
   elsewhere. Python checks the answer: a quote not found in the passages
   is dropped, and a paper is called contradicted only with a confirmed
   quote. The plan is shown first when it has more than `LARGE_BATCHES`
   papers.
4. Assigns each directly/partly supporting paper a half-star citation
   suitability rating (`kilin/tools/_shared/_rating.py`). The support
   verdict determines the score band: supported 70–100, partly supported
   40–70, and background 10–40. A checked quotation and the coverage of
   distinctive statement words in that quote (or the best available
   passage) refine the rating within its band. Unconfirmed quotations
   are subject to a lower score cap within each verdict band. Ratings are
   heuristic, **not** calibrated probabilities or measures of scientific
   quality. They require no extra model, embedding, or network call.
5. Answers with directly or partly supporting papers first, best within
   each verdict, each with its rating, comment,
   and any confirmed quotation as evidence. Background (context-only)
   papers are listed separately, without claiming direct support. Each
   star display includes
   its numerical suitability score in parentheses, for example
   `★★★★½ (88/100)`, using the unchanged score in the structured
   result and trace. Within a verdict group,
   more suitable papers appear first. Other checked papers are listed
   without stars, rather than recommended.

Limits: only the project's files are searched, so a paper that is not in
the folder cannot be suggested (the answer says so when none supports the
statement); each paper is judged from a few passages, not all of it; and
the model's verdicts can be wrong, so they are pointers for the reader to
check. A positive verdict may have no confirmed quotation and is then marked
provisional. Quote validation rejects changed numbers, negations and
common direction-of-effect substitutions. Evidence status is also recorded
in traces. An exact quote's source location is retained with its passage.
Answering-model calls inherit main-interface thinking, temperature and
maximum output tokens by default (`ANSWER_TOKENS` is `null`). An explicit
`ANSWER_TOKENS` value imposes a per-call cap. Router calls remain
deterministic.

With `source_first`, card matches identify candidates but do not establish
claim support. The tool still calls the complete hybrid `rank()` interface
for each candidate's original passages and may therefore perform a second
query embedding. Without the shared retrieval callback, opting in raises
an explicit error rather than silently reverting to hybrid discovery.

Its `defaults.json` is independent of `kilin_defaults.json`.
Project-type tool settings override the keys in `SETTINGS`. Other sections
are internal thresholds. All values are read and validated on import.

## Defaults

### `SETTINGS`

- `FILES` (default `5`): Most papers judged, those with the best-ranked
  passages first. Allowed range: 1–20.
- `PASSAGES` (default `4`): Most passages of each paper read. Allowed
  range: 1–20.
- `PASSAGE_TOKENS` (default `4000`): Most estimated tokens of those
  passages; the best-ranked one is always read. Allowed range: 300–16000.
  With `null`, the answering model inherits the main interface's maximum
  output tokens (or its available context if no maximum was configured).
  An explicit override accepts 100–131072 and is capped by any session
  maximum.
- `QUICK_FILES` (default `2`): Papers judged by the quick version, the most
  related first. Allowed range: 1–20.
- `CONFIRMATION` (default `"when_large"`): When to ask before judging the
  papers. Allowed values: always, when_large, never.
- `LARGE_BATCHES` (default `5`): With `when_large`, plans with more papers
  than this ask first. Allowed range: 0–1000.
- `DISCOVERY_STRATEGY` (default `"hybrid"`): Candidate selection by
  complete passage ranking (`hybrid`) or four-channel source ranking
  (`source_first`); both preserve the existing model-based evidence check.
- `DISCOVERY_TOKENS` (default `8000`): Maximum estimated evidence tokens
  in a bounded `source_first` discovery request. Allowed range: 300–16000.
  This is separate from per-paper `PASSAGE_TOKENS`, model output tokens,
  and ordinary answering-intent retrieval limits.

### `STATEMENT`

How the statement is taken from the question.

- `MIN_WORDS` (default `4`): Fewest words of a statement, quoted or after a
  colon, and of the router's copy. Allowed range: 1–100.
- `ROUTER_TOKENS` (default `256`): Longest router answer when copying the
  statement, in tokens. Allowed range: 20–2000.
- `ROUTER_WORD_SHARE` (default `0.8`): Least share of the router's words
  that must be words of the question; otherwise the whole question is
  used. Allowed range: 0–1.

### `ESTIMATES`

- `JUDGE_REPLY_TOKENS` (default `256`): expected reply tokens for
  duration estimates only; never a generation cap. Range: 1-131072.

Planning assumptions for the time shown before the papers are judged;
they are estimates, not limits.

- `PROMPT_TOKENS_PER_SECOND` (default `700.0`): Estimated prompt-processing
  throughput. Allowed range: 0.001–1000000000.
- `REPLY_TOKENS_PER_SECOND` (default `25.0`): Estimated generation
  throughput. Allowed range: 0.001–1000000000.
- `CALL_SECONDS` (default `1.0`): Fixed time of one model call, in seconds.
  Allowed range: 0–1000000000.
- `PROMPT_OVERHEAD_TOKENS` (default `450`): Instruction tokens around the
  passages of one paper. Allowed range: 0–100000.
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

An incomplete candidate judgment is `failed` and unrated, not evidence against
citing that paper. Other candidates continue and valid suggestions remain
available. The answer explicitly lists incomplete items; all failed judgments
produce an incomplete/inconclusive result rather than a false zero-support
conclusion. The shared internal adapter owns the one eligible retry.
