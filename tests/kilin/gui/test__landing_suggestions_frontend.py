"""Keep customized landing text ahead of generated suggestions.

Run the production JavaScript helpers with Node, so project-specific
presentation overrides are checked without requiring Qt WebEngine.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
import shutil
import unittest

from tests.kilin.gui._frontend_support import extract_function, run_node

_WEB = Path(__file__).resolve().parents[3] / "kilin/gui/web"


class TestLandingSuggestionsFrontend(unittest.TestCase):
    """Prioritize project settings and keep source headings centered."""

    @unittest.skipUnless(shutil.which("node"), "Node.js is unavailable")
    def test_project_overrides_are_not_hidden_by_generated_text(self):
        """Use the production selection and override-check helpers."""

        source = (_WEB / "app.js").read_text(encoding="utf-8")
        script = r"""
const assert = require("node:assert/strict");
let activeProjectTypeId = "literature";
let projectTypes = {
  literature: {
    suggestions: ["Help", "Add", "Built-in"],
    welcome_prompts: ["Built-in heading"],
  },
};
let projectPresentation = {
  suggestions: ["Help", "Add", "Built-in"],
  welcome_prompts: ["Built-in heading"],
};
"""
        script += extract_function(source, "hasCustomLandingText")
        script += extract_function(source, "chooseSuggestions")
        script += r"""
const generated = [
  { intent: "a", question: "Generated A" },
  { intent: "b", question: "Generated B" },
  { intent: "c", question: "Generated C" },
];
assert.equal(hasCustomLandingText("suggestions"), false);
assert.equal(hasCustomLandingText("welcome_prompts"), false);
assert.deepEqual(chooseSuggestions(generated, projectPresentation.suggestions)
  .sort(), ["Generated A", "Generated B", "Generated C"].sort());
projectPresentation.suggestions = ["My first", "My second", "My third"];
projectPresentation.welcome_prompts = ["My own welcome"];
assert.equal(hasCustomLandingText("suggestions"), true);
assert.equal(hasCustomLandingText("welcome_prompts"), true);
const questions = hasCustomLandingText("suggestions") ? [] : generated;
assert.deepEqual(chooseSuggestions(questions, projectPresentation.suggestions)
  .sort(), ["My first", "My second", "My third"].sort());
console.log("landing overrides passed");
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("landing overrides passed", result.stdout)

    def test_renderer_and_events_respect_overrides(self):
        """Render both overrides and refresh on a new question event."""

        source = (_WEB / "app.js").read_text(encoding="utf-8")
        render = extract_function(source, "renderConversationEmpty")
        self.assertIn('hasCustomLandingText("suggestions")', render)
        self.assertIn('hasCustomLandingText("welcome_prompts")', render)
        self.assertIn(
            'if (conversation.querySelector(".conversation-empty")) '
            'renderConversationEmpty();',
            source,
        )
        self.assertIn(
            "pendingSourceQuestions = { project: data.project, "
            "questions, welcome };",
            source,
        )

    def test_heading_is_centered_independently_of_suggestions(self):
        """Retain the previous new-chat CSS alignment correction."""

        styles = (_WEB / "styles.css").read_text(encoding="utf-8")
        heading = next(
            line for line in styles.splitlines()
            if line.startswith(".conversation-empty h2 {")
        )
        self.assertIn("justify-self: center;", heading)


if __name__ == "__main__":
    unittest.main()
