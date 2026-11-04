"""Exercise the dialog asking what to do with changed original sources.

Run the production ``sourceSyncSummary``, ``sourceSyncChoices``,
``sourceSyncPlan`` and ``sourceSyncGroups`` functions in Node, and check
that the ``source_changes`` event opens the dialog and that Update all,
Keep both, Keep old copies and Apply choices send the right choices.
Rendering is covered only by the Qt/WebEngine smoke test and manual
checks.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from pathlib import Path
import shutil
import unittest

from tests.kilin.gui._frontend_support import run_node

_ROOT = Path(__file__).resolve().parents[3]
_APP = _ROOT / "kilin/gui/web/app.js"


class TestSourceChangesFrontend(unittest.TestCase):
    """Ask the user what to do with each changed file."""

    def test_event_opens_the_dialog_and_buttons_send_choices(self):
        """The event shows the dialog; each button sends its own choices."""

        source = _APP.read_text(encoding="utf-8")
        start = source.index('data.event === "source_changes"')
        end = source.index("} else if (data.event", start)
        self.assertIn("showSourceSync(data);", source[start:end])

        def handler(button: str) -> str:
            start = source.index(f'byId("{button}").addEventListener')
            return source[start : source.index("\n", start)]

        self.assertIn(".close()", handler("source-sync-keep"))
        self.assertNotIn("syncSources", handler("source-sync-keep"))
        self.assertIn(
            'sourceSyncPlan(sourceSyncChanges, "both")',
            handler("source-sync-both"),
        )
        self.assertIn(
            'sourceSyncPlan(sourceSyncChanges, "update")',
            handler("source-sync-update"),
        )
        start = source.index('byId("source-sync-apply")')
        apply = source[start : source.index("});", start)]
        self.assertIn(
            "decisions[input.dataset.path] = input.value;", apply
        )
        start = source.index("function syncSources(")
        self.assertIn(
            "bridge.syncProjectSources(JSON.stringify(decisions));",
            source[start : source.index("\n}\n", start)],
        )

    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_summary_choices_and_groups(self):
        """Count files, offer valid choices and keep edited copies."""

        source = _APP.read_text(encoding="utf-8")
        functions = []
        for name in (
            "sourceSyncSummary", "sourceSyncChoices", "sourceSyncPlan",
            "sourceSyncGroups",
        ):
            start = source.index(f"function {name}(")
            functions.append(source[start : source.index("\n}\n", start) + 3])
        script = r"""
const assert = require("node:assert/strict");
"""
        script += "\n".join(functions)
        script += r"""
const changes = {
  added: ["sources/repository/new.py"],
  changed: ["sources/notes.md"],
  deleted: ["sources/repository/a.py", "sources/repository/b.py"],
  edited: ["sources/repository/c.py"],
};
assert.equal(
  sourceSyncSummary(changes),
  "4 files changed in the original folders since KíLín copied them. "
    + "KíLín keeps the 1 file you edited in the project.",
);
assert.equal(
  sourceSyncSummary({
    added: [], changed: ["sources/a.md"], deleted: [], edited: [],
  }),
  "1 file changed in the original folders since KíLín copied them.",
);
// The first choice is the default; every action is one the core accepts.
const allowed = {
  added: ["import", "keep"],
  changed: ["replace", "keep", "keep_both"],
  deleted: ["delete", "keep"],
};
for (const [group, actions] of Object.entries(allowed)) {
  const choices = sourceSyncChoices(group).map(([action]) => action);
  assert.deepEqual([...choices].sort(), [...actions].sort());
}
assert.equal(sourceSyncChoices("added")[0][0], "import");
assert.equal(sourceSyncChoices("changed")[0][0], "replace");
assert.equal(sourceSyncChoices("deleted")[0][0], "delete");
assert.deepEqual(sourceSyncChoices("edited"), []);
assert.deepEqual(
  sourceSyncGroups({ ...changes, added: [], changed: [] })
    .map(({ group, title, paths }) => [group, title, paths]),
  [
    ["deleted", "Deleted",
      ["sources/repository/a.py", "sources/repository/b.py"]],
    ["edited", "Edited by you", ["sources/repository/c.py"]],
  ],
);
// Update all brings every change in; Keep both also keeps old copies.
// Edited copies are never sent, so they are kept.
assert.deepEqual(sourceSyncPlan(changes, "update"), {
  "sources/repository/new.py": "import",
  "sources/notes.md": "replace",
  "sources/repository/a.py": "delete",
  "sources/repository/b.py": "delete",
});
assert.deepEqual(sourceSyncPlan(changes, "both"), {
  "sources/repository/new.py": "import",
  "sources/notes.md": "keep_both",
  "sources/repository/a.py": "keep",
  "sources/repository/b.py": "keep",
});
console.log("source sync passed");
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("source sync passed", result.stdout)
