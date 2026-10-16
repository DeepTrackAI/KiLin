# suggest_reference

This package owns the `suggest_reference` registered tool. Given a
statement the user wants to cite, it finds the papers in the project that
could be cited for it, and shows for each the sentence of the paper that
backs it. Finding the candidate papers calls no language model; judging
each one uses the answering model, with the same checks as
`reference_check`.

For "suggest a reference for: anomalous diffusion is common in living
cells", the tool:

1. Takes the statement from the question: quoted text, or the text after a
   colon; otherwise the router model copies it out of the question, and
   its copy is used only when most of its words are the question's. A
   statement shorter than `MIN_WORDS` words gets a request to write it.
2. Ranks every passage of the project for the statement (KíLín's hybrid
   ranking) and keeps the `FILES` papers with the best-ranked passages. A
   file that contains the statement word for word is not judged, and the
   answer names it: it is the user's draft, or the paper the sentence was
   copied from.
3. Judges each paper with the answering model (`_shared/claims.py`): its
   `PASSAGES` passages most related to the statement, and a verdict
   (supported, partly supported, background, not supported, contradicted,
   not found), the passage sentence it rests on, and a critical comment.
   The model is told that a paper repeating the claim with a citation is
   not its source: it answers partly supported and says the original is
   elsewhere. Python checks the answer: a quote not found in the passages
   is dropped, and a paper is called contradicted only with a confirmed
   quote. The plan is shown first when it has more than `LARGE_BATCHES`
   papers.
4. Answers with the papers that could be cited (supported, partly
   supported, background), best first, each with its confirmed quote as
   evidence and the comment, then lists the other papers checked.

Limits: only the project's files are searched, so a paper that is not in
the folder cannot be suggested (the answer says so when none supports the
statement); each paper is judged from a few passages, not all of it; and
the model's verdicts can be wrong, so they are pointers for the reader to
check, with the quote and the place to look.

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
- `ANSWER_TOKENS` (default `250`): Longest verdict answer, in tokens.
  Allowed range: 100–1000.
- `QUICK_FILES` (default `2`): Papers judged by the quick version, the most
  related first. Allowed range: 1–20.
- `CONFIRMATION` (default `"when_large"`): When to ask before judging the
  papers. Allowed values: always, when_large, never.
- `LARGE_BATCHES` (default `5`): With `when_large`, plans with more papers
  than this ask first. Allowed range: 0–1000.

### `STATEMENT`

How the statement is taken from the question.

- `MIN_WORDS` (default `4`): Fewest words of a statement, quoted or after a
  colon, and of the router's copy. Allowed range: 1–100.
- `ROUTER_TOKENS` (default `200`): Longest router answer when copying the
  statement, in tokens. Allowed range: 20–2000.
- `ROUTER_WORD_SHARE` (default `0.8`): Least share of the router's words
  that must be words of the question; otherwise the whole question is
  used. Allowed range: 0–1.

### `ESTIMATES`

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
