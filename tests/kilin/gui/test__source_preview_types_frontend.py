"""Exercise how the source preview follows the new project's file types.

Run the production ``projectFileTypes`` and ``skippedReason`` functions in
Node, and check that the preview and project creation use the same list, so
every file the user can tick is one creation accepts. Rendering is covered
only by the Qt/WebEngine smoke test and manual checks.

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


def _function(source: str, name: str) -> str:
    """Return one top-level function's source text."""

    start = source.index(f"function {name}(")
    return source[start : source.index("\n}\n", start) + 3]


class TestSourcePreviewTypesFrontend(unittest.TestCase):
    """Offer only files the new project accepts."""

    def test_preview_and_creation_share_the_file_types(self):
        """Both the preview request and the settings use one list."""

        source = _APP.read_text(encoding="utf-8")
        self.assertIn(
            "JSON.stringify(projectFileTypes())",
            _function(source, "previewSources"),
        )
        self.assertIn(
            "extensions: projectFileTypes(),",
            _function(source, "collectSettings"),
        )

    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_types_and_reasons(self):
        """Merge the type's and the user's types; explain skipped files."""

        source = _APP.read_text(encoding="utf-8")
        functions = [
            _function(source, name)
            for name in (
                "fileTypes", "projectFileTypes", "isOversizedSkippedFile",
                "skippedReason",
            )
        ]
        script = r"""
const assert = require("node:assert/strict");
const field = { value: " .HTML, .md ,," };
const byId = (id) => {
  assert.equal(id, "extensions");
  return field;
};
const profile = { extensions: [".py", ".md"] };
const selectionReasonCodes = {
  FILE_TOO_LARGE: "file_too_large",
  FILE_TYPE_NOT_SELECTED: "file_type_not_selected",
};
"""
        script += "\n".join(functions)
        script += r"""
assert.deepEqual(projectFileTypes(), [".py", ".md", ".html"]);
assert.equal(
  skippedReason({ reason_code: "file_type_not_selected", reason: "x" }),
  "Not a file type of this project · add it under Additional file types",
);
assert.equal(
  skippedReason({ reason_code: "file_too_large", reason: "x" }),
  "Above current limit · check to include",
);
assert.equal(
  skippedReason({ reason_code: "protected_file", reason: "protected" }),
  "protected",
);
console.log("preview types passed");
"""
        result = subprocess.run(
            [shutil.which("node"), "-e", script],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("preview types passed", result.stdout)
