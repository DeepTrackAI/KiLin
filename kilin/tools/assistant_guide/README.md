# assistant_guide

This package owns the `assistant_guide` registered tool. It answers "What can you do for me?" with one ready-to-send example question for each kind of request the project type supports, written about the project's own files and topics by `kilin.question_suggestions`. Every answer ends with how to add files (drag them onto **sources** in the Workspace panel, or attach them with **+** in the message box) and how to remove one, and a question about files, such as "how to drag files" or "What do you mean by adding files?", gets only what sources are and these steps, without the examples. A project without sources is told how to add its files first, followed by the project type's fixed suggestions. No model writes the answer; when the source-specific pool is not saved yet, the tool can call the router once to pick phrases for the examples. Projects with indexed sources but no fillable examples keep the fixed suggestions without being described as empty. Its `defaults.json` is independent of `kilin_defaults.json`. Project-type tool settings override the keys in `SETTINGS`. Other sections are internal execution parameters. All values are read and validated on import.

## Defaults

### `SETTINGS`

- `MAX_LISTED` (default `20`): Example questions listed, one per kind of request (1-100). Allowed range: 1–100.


## Generation settings

The answer is written without a model call, so output allowances do not
apply.

`MODEL_CALLS` contains one key: `THINKING` (packaged value `"inherit"`).
It accepts `inherit`, `on`, or `off` for internal `context.ask_model` calls,
which this tool does not make. See the [shared policy](../../../docs/configuration.md#internal-tool-thinking)
for precedence.
