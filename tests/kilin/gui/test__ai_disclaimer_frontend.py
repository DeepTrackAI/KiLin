"""Check the AI-use disclaimer shown above the message box.

Read the production ``index.html`` and ``styles.css``: the chat screen shows
one short, grey line right above the composer. Rendering is covered only by
the Qt/WebEngine smoke test and manual checks.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from pathlib import Path
import unittest

_WEB = Path(__file__).resolve().parents[3] / "kilin/gui/web"


class TestAiDisclaimerFrontend(unittest.TestCase):
    """Remind the user that answers can be wrong."""

    def test_disclaimer_sits_right_above_the_composer(self):
        """One line in the chat screen, directly before the message box."""

        page = (_WEB / "index.html").read_text(encoding="utf-8")
        chat = page[page.index('<section id="chat"') :]
        chat = chat[: chat.index("</section>")]
        line = (
            '<p class="ai-disclaimer">AI-generated answers can be wrong. '
            "Always verify them.</p>"
        )
        self.assertEqual(page.count('class="ai-disclaimer"'), 1)
        self.assertIn(line, chat)
        after = chat[chat.index(line) + len(line) :].lstrip()
        self.assertTrue(after.startswith('<form id="question-form"'))

    def test_disclaimer_is_small_and_grey(self):
        """It uses the muted colour token, so it follows dark mode."""

        styles = (_WEB / "styles.css").read_text(encoding="utf-8")
        start = styles.index(".ai-disclaimer {")
        rule = styles[start : styles.index("}", start)]
        self.assertIn("color: var(--muted)", rule)
        self.assertIn("font-size: 11px", rule)


if __name__ == "__main__":
    unittest.main()
