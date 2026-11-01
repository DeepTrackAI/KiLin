# show_source

This package owns the `show_source` registered tool. Its `defaults.json` is independent of `kilin_defaults.json`. Project-type tool settings override the keys in `SETTINGS`. Other sections are internal execution parameters. All values are read and validated on import.

## Defaults

### `SETTINGS`

- `CONTEXT_SENTENCES` (default `2`): Sentences shown before and after the cited sentence (0-10). Allowed range: 0–10.
- `MAX_PASSAGES` (default `3`): Most cited passages shown (1-10). Allowed range: 1–10.

### `HEURISTICS`

- `MIN_WORDS` (default `4`): Minimum number of words required when splitting cited passages into sentences. Allowed range: 1–100.

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

If the optional model explanation remains incomplete, verbatim source
passages and citations are still returned. The result is explicitly partial
and labels the missing explanation. The shared internal adapter may retry
unfinished, exhausted thinking once; it never supplies truncated text as the
explanation. Source selection and routing are unchanged.
