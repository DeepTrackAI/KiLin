# scan_and_list

This package owns the `scan_and_list` registered tool. Its `defaults.json` is independent of `kilin_defaults.json`. Project-type tool settings override the keys in `SETTINGS`. Other sections are internal execution parameters. All values are read and validated on import.

## Defaults

### `SETTINGS`

- `CASE_SENSITIVE` (default `false`): Whether letter case must match.
- `WHOLE_WORDS` (default `true`): Whether a term must not continue into adjacent letters or digits.
- `CONTEXT_LINES` (default `1`): Lines of context kept around each mention (0-5). Allowed range: 0–5.
- `MAX_LISTED` (default `100`): Mentions listed with citations (1-1000); counts include all. Allowed range: 1–1000.

### `HEURISTICS`

- `UNSEARCHABLE_EXAMPLES` (default `5`): Maximum examples of files without indexed searchable text. Allowed range: 0–100.


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
