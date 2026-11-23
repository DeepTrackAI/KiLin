# batch_classify

This package owns the `batch_classify` registered tool. Its `defaults.json` is independent of `kilin_defaults.json`. Project-type tool settings override the keys in `SETTINGS`. Other sections are internal execution parameters. All values are read and validated on import.

## Defaults

### `SETTINGS`

- `CASE_SENSITIVE` (default `false`): Whether letter case must match.
- `WHOLE_WORDS` (default `true`): Whether a term must not continue into adjacent letters or digits.
- `CONTEXT_LINES` (default `2`): Lines around each mention that the model sees (0-5). Allowed range: 0–5.
- `MAX_LISTED` (default `100`): Mentions listed with citations (1-1000); counts include all. Allowed range: 1–1000.
- `CANDIDATES` (default `"mentions"`): How items are collected; only ``mentions``. Allowed values: mentions.
- `CATEGORIES` (default `[]`): Fixed labels (two to eight distinct names of at most 40 characters). Empty lets the router suggest labels, which are shown in the plan before execution. Project types may override this list.
- `BATCH_TOKENS` (default `2500`): Estimated tokens of mentions per model call (500-8000). Allowed range: 500–8000.
- `MAX_MODEL_CALLS` (default `30`): Most model calls one answer may make (1-200); mentions beyond them are reported as not classified. Allowed range: 1–200.
- `CONFIRMATION` (default `"when_large"`): When to ask before running. Allowed values: always, when_large, never.
- `LARGE_BATCHES` (default `2`): With ``when_large``, plans with more calls than this ask first (0-100). Allowed range: 0–100.

### `ESTIMATES`

- `PROMPT_TOKENS_PER_SECOND` (default `700.0`): Estimated prompt-processing throughput, tokens per second; planning only. Allowed range: 0.001–1000000000.0.
- `REPLY_TOKENS_PER_SECOND` (default `25.0`): Estimated model generation throughput, tokens per second; planning only. Allowed range: 0.001–1000000000.0.
- `REPLY_TOKENS_PER_ITEM` (default `25`): Reply tokens per item used by this tool during execution. Allowed range: 0–100000.
- `PROMPT_OVERHEAD_TOKENS` (default `350`): Prompt overhead tokens used by this tool during execution. Allowed range: 0–100000.
- `RETRY_GROUP` (default `8`): Maximum number of missing labels included in a classification retry. Allowed range: 1–10000.


## Generation settings

Output allowances come from the active session, including per-question GUI
or API settings. They are ceilings for each model call, not target lengths
or a total budget for a multi-call tool. Prose length is controlled by the
prompt or report template; input passage budgets remain separate.

`MODEL_CALLS` contains one key: `THINKING` (packaged value `"off"`).
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

Incomplete batches remain unclassified while later batches continue. They do
not enter the existing missing-label repair loop. Completed replies with
missing labels still use that existing loop. Partial status and item-specific
failure records distinguish unassessed mentions from classified categories.
The shared internal adapter owns the eligible one-time nonthinking fallback;
logical batch/repair counts are distinct from actual retry inference counts.
