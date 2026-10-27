"""Exercise the browser's shared file-tree builder without native Qt.

Run the production ``branch`` function in Node with a minimal DOM adapter.
The source preview needs checkbox labels; the workspace tree needs plain
rows, because a label forwards a click on its text to its first control,
which there is a file's delete button. Native label activation itself is
covered only by the Qt/WebEngine smoke test and manual checks.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from pathlib import Path
import shutil
import subprocess
import unittest

_ROOT = Path(__file__).resolve().parents[3]


@unittest.skipUnless(shutil.which("node"), "Node.js is not available")
class TestWorkspaceTreeFrontend(unittest.TestCase):
    """Build selection and workspace trees from the same builder."""

    def test_workspace_rows_are_plain_and_preview_rows_are_labels(self):
        """Only the selectable preview tree contains labels and checkboxes."""

        source = (_ROOT / "kilin/gui/web/app.js").read_text(encoding="utf-8")
        functions = []
        for name in ("branch", "allFiles"):
            start = source.index(f"function {name}(")
            end = source.index("\n}\n", start) + 3
            functions.append(source[start:end])
        start = source.index("const expandedFolders")
        state = source[start : source.index(";", start) + 1]
        script = r"""
const assert = require("node:assert/strict");
function element(tagName) {
  return {
    tagName: tagName.toUpperCase(), children: [], dataset: {},
    append(...children) { this.children.push(...children); },
    addEventListener() {},
  };
}
const document = {
  createElement: element,
  createTextNode: (text) => ({ tagName: "#text", children: [], text }),
};
const renderPreview = () => {};
function walk(node) {
  return [node, ...node.children.flatMap(walk)];
}
const tree = {
  directories: new Map([[
    "reports",
    { directories: new Map(), files: ["outputs/reports/b.md"] },
  ]]),
  files: ["outputs/a.pdf"],
};
"""
        script += "\n".join([state, *functions])
        script += r"""
const files = allFiles(tree);
const workspace = walk(branch("outputs", tree, files, new Set(), 0, null,
                              {}, false));
const workspaceRows = workspace.filter(
  (node) => node.className === "tree-file");
assert.equal(workspaceRows.length, 2);
assert.ok(workspaceRows.every((row) => row.tagName === "DIV"));
assert.ok(workspace.every((node) => !["LABEL", "INPUT"].includes(
  node.tagName)));
assert.deepEqual(workspaceRows.map((row) => row.dataset.path).sort(),
                 ["outputs/a.pdf", "outputs/reports/b.md"]);

const preview = walk(branch("outputs", tree, files, new Set(files)));
const previewRows = preview.filter((node) => node.className === "tree-file");
assert.equal(previewRows.length, 2);
assert.ok(previewRows.every((row) => row.tagName === "LABEL"));
assert.equal(preview.filter((node) => node.tagName === "INPUT").length, 4);
console.log("workspace tree passed");
"""
        result = subprocess.run(
            [shutil.which("node"), "-e", script],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("workspace tree passed", result.stdout)
