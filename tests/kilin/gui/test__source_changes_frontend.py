"""Exercise the dialog asking whether to sync changed original sources.

Run the production ``sourceSyncSummary`` and ``sourceSyncGroups`` functions
in Node, and check that the ``source_changes`` event opens the dialog and
that Update asks the bridge to sync. Rendering is covered only by the
Qt/WebEngine smoke test and manual checks.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from pathlib import Path
import shutil
import subprocess
import unittest

_ROOT = Path(__file__).resolve().parents[3]
_APP = _ROOT / "kilin/gui/web/app.js"


class TestSourceChangesFrontend(unittest.TestCase):
    """Ask the user whether to update the project's copies."""

    def test_event_opens_the_dialog_and_update_syncs(self):
        """The event shows the dialog; Update starts the sync."""

        source = _APP.read_text(encoding="utf-8")
        start = source.index('data.event === "source_changes"')
        end = source.index("} else if (data.event", start)
        self.assertIn("showSourceSync(data);", source[start:end])
        start = source.index('byId("source-sync-update")')
        end = source.index("});", start)
        self.assertIn("bridge.syncProjectSources();", source[start:end])

    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_summary_counts_and_groups_list_the_files(self):
        """The summary counts files; groups list them without sources/."""

        source = _APP.read_text(encoding="utf-8")
        functions = []
        for name in ("sourceSyncSummary", "sourceSyncGroups"):
            start = source.index(f"function {name}(")
            functions.append(source[start : source.index("\n}\n", start) + 3])
        script = r"""
const assert = require("node:assert/strict");
"""
        script += "\n".join(functions)
        script += r"""
const changes = {
  changed: ["sources/notes.md"],
  deleted: ["sources/repository/a.py", "sources/repository/b.py"],
  edited: ["sources/repository/c.py"],
};
assert.equal(
  sourceSyncSummary(changes),
  "KíLín found 3 files that changed in the original sources. Update the "
    + "project's copies? KíLín keeps the 1 file you edited in the project.",
);
assert.equal(
  sourceSyncSummary({ changed: ["sources/a.md"], deleted: [], edited: [] }),
  "KíLín found 1 file that changed in the original sources. Update the "
    + "project's copies?",
);
assert.deepEqual(sourceSyncGroups({ ...changes, changed: [] }), [
  ["Deleted from the original", ["repository/a.py", "repository/b.py"]],
  ["Edited in this project (kept)", ["repository/c.py"]],
]);
console.log("source sync passed");
"""
        result = subprocess.run(
            [shutil.which("node"), "-e", script],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("source sync passed", result.stdout)
