"""Verify source-card GUI controls without starting Qt or loading models.

Tests exercise independent model settings, project-creation choices,
explicit regeneration confirmation, and card indexing progress displays.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from pathlib import Path
import shutil
import unittest

from tests.kilin.gui._frontend_support import extract_function, run_node

_WEB = Path(__file__).resolve().parents[3] / "kilin/gui/web"


class TestSourceCardFrontend(unittest.TestCase):
    """Exercise the independent card settings and action wiring."""

    def test_indexing_and_model_controls_are_distinct(self):
        """Expose core-owned model and policy fields in separate tabs."""
        page = (_WEB / "index.html").read_text(encoding="utf-8")
        source = (_WEB / "app.js").read_text(encoding="utf-8")
        for field in (
            "source-card-enabled", "source-card-kind",
            "source-card-automatic", "source-card-mlx",
            "source-card-torch", "source-card-thinking",
            "source-card-temperature", "source-card-input",
            "source-card-output", "source-card-summary",
            "source-card-progress", "save-source-card-models",
            "save-source-card-policy", "regenerate-source-cards",
        ):
            with self.subTest(field=field):
                self.assertIn(f'id="{field}"', page)
        self.assertIn(
            'bridge.saveSourceCardSettings(JSON.stringify(', source
        )
        self.assertIn('bridge.regenerateSourceCards();', source)
        self.assertIn('data.event === "source_cards_applied"', source)
        self.assertIn('data.event === "source_cards_regenerated"', source)
        # The existing normal index save handler remains functional.
        self.assertEqual(
            source.count('data.event === "indexing_applied"'), 1
        )

    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_settings_validation_save_and_confirmed_regeneration(self):
        """Keep two model IDs separate and require user confirmation."""
        source = (_WEB / "app.js").read_text(encoding="utf-8")
        script = r"""
const assert = require("node:assert/strict");
const controls = {};
const byId = (key) => (controls[key] ||= {
  _value: "", checked: false, disabled: false, textContent: "",
  set value(next) { this._value = String(next); },
  get value() { return this._value; },
});
let sourceCardSettings = null;
let activeBackend = "mlx";
let ready = false;
let busy = false;
let applyingSourceCards = false;
let answer = false;
const actions = [];
const bridge = {
  saveSourceCardSettings: (data) => actions.push([
    "save", JSON.parse(data),
  ]),
  regenerateSourceCards: () => actions.push(["regenerate"]),
};
const setBusy = (flag) => { busy = flag; };
const askKilin = async () => answer;
const settings = {
  policy: { enabled: true, kind: "scientific_paper", automatic: true },
  models: {
    mlx: { model_id: "example/mlx", thinking: false,
      temperature: 0, max_input_tokens: 5000, max_new_tokens: 1200 },
    torch: { model_id: "example/torch", thinking: false,
      temperature: 0, max_input_tokens: 5000, max_new_tokens: 1200 },
  },
};
"""
        for name in (
            "showSourceCardSettings",
            "sourceCardValues",
            "sourceCardProblem",
            "showSourceCardSummary",
            "saveSourceCardSettings",
        ):
            script += extract_function(source, name)
        script += "async " + extract_function(
            source, "regenerateSourceCards"
        )
        script += r"""
(async () => {
  showSourceCardSettings(settings);
  assert.equal(sourceCardProblem(), "");
  assert.equal(byId("source-card-mlx").value, "example/mlx");
  assert.equal(byId("source-card-torch").value, "example/torch");
  assert.equal(byId("regenerate-source-cards").disabled, true);
  showSourceCardSummary({valid: 2, not_applicable: 1,
    failed: 3, embedded: 2, total: 6});
  assert.match(byId("source-card-summary").textContent, /3 failed/);
  byId("source-card-input").value = "-1";
  assert.match(sourceCardProblem(), /token limits/);
  byId("source-card-input").value = "5000";
  saveSourceCardSettings();
  assert.equal(actions.length, 0);
  assert.equal(sourceCardSettings.models.mlx.model_id, "example/mlx");
  ready = true;
  showSourceCardSettings(settings);
  assert.equal(byId("regenerate-source-cards").disabled, false);
  saveSourceCardSettings();
  assert.equal(actions[0][0], "save");
  assert.equal(actions[0][1].models.torch.model_id, "example/torch");
  assert.equal(actions[0][1].models.mlx.model_id, "example/mlx");
  busy = false;
  await regenerateSourceCards();
  assert.equal(actions.length, 1);
  answer = true;
  await regenerateSourceCards();
  assert.deepEqual(actions[1], ["regenerate"]);
  console.log("source-card controls passed");
})().catch((error) => { console.error(error); process.exit(1); });
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("source-card controls passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
