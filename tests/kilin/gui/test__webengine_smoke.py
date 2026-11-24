"""Optionally smoke-test the actual GUI page in an isolated Qt process.

Run with ``KILIN_RUN_NATIVE_GUI_TESTS=1`` and PySide6 WebEngine installed.
The child owns its QApplication, so this test cannot interfere with the
native PDF tests running in the parent process. Check the conversation
layout at several widths, as well as page events and native source drops.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


# Use the production renderers and layout rather than a synthetic stylesheet.
_CONVERSATION_WIDTH_CHECK = r"""
(() => {
  const longName = "ProteinFoldingDynamics".repeat(9) + ".pdf";
  const width = (element) => {
    const panel = conversationScroll.getBoundingClientRect();
    const box = element.getBoundingClientRect();
    return {
      client: element.clientWidth,
      scroll: element.scrollWidth,
      overflow: getComputedStyle(element).overflowX,
      inside: box.left >= panel.left - 1 && box.right <= panel.right + 1,
    };
  };
  const result = {};
  showConversations(true);
  showWorkspace(true);
  projectPresentation = null;
  sourceWelcome = ["Explore your sources"];
  sourceQuestions = [
    {intent: "test", question: `Explain ${longName}`},
  ];
  byId("conversation-name").classList.add("hidden");
  renderConversationEmpty();
  result.suggestion = width(conversationScroll);
  result.suggestionButton = width(conversation.querySelector("button"));
  sourceWelcome = [longName];
  sourceQuestions = [];
  renderConversationEmpty();
  result.welcome = width(conversationScroll);
  result.welcomeHeading = width(conversation.querySelector("h2"));
  conversation.replaceChildren();
  byId("conversation-name").classList.remove("hidden");
  byId("conversation-name").textContent = longName;
  result.title = width(conversationScroll);
  result.titleHeading = width(byId("conversation-name"));
  byId("conversation-name").textContent = "Test";
  const code = "x".repeat(240);
  const headers = Array(12).fill("Column header").join(" | ");
  const dividers = Array(12).fill("---").join(" | ");
  const cells = Array(12).fill("Value").join(" | ");
  const formula = Array(100).fill("x^2").join("+");
  const answer = ["```python", code, "```", "", `| ${headers} |`,
    `| ${dividers} |`, `| ${cells} |`, "", `$$ ${formula} $$`].join("\n");
  addTurn({question: longName, answer, evidence: [], parameters: {}});
  result.question = width(conversationScroll);
  result.questionBubble = width(
    conversation.querySelector(".user-message"));
  for (const [name, selector] of [["code", "pre"],
    ["table", ".answer-table"], ["math", ".answer-math.display"]]) {
    const element = conversation.querySelector(selector);
    result[name] = width(element);
    element.scrollLeft = 40;
    result[name].moves = element.scrollLeft > 0;
  }
  result.verticalOverflow = getComputedStyle(conversationScroll).overflowY;
  return JSON.stringify(result);
})()
"""


class TestNativeGUIPage(unittest.TestCase):
    """Run an optional end-to-end check outside the unittest process."""

    def test_webengine_loads_page_and_accepts_a_source_drop(self) -> None:
        """Check page events, native drops, and conversation layout."""

        if os.environ.get("KILIN_RUN_NATIVE_GUI_TESTS") != "1":
            self.skipTest("Enable the optional native Qt smoke test")
        if importlib.util.find_spec("PySide6") is None:
            self.skipTest("PySide6 is not installed")
        with TemporaryDirectory() as temporary:
            environment = os.environ.copy()
            environment.setdefault("QT_QPA_PLATFORM", "offscreen")
            environment.setdefault(
                "QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    str(Path(temporary).resolve()),
                ],
                capture_output=True,
                check=False,
                text=True,
                timeout=45,
                env=environment,
            )
        self.assertEqual(
            result.returncode,
            0,
            "Native GUI subprocess failed.\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        )


def _native_smoke(directory: Path) -> None:
    """Check the WebEngine UI inside a dedicated QApplication process.

    Parameters
    ----------
    directory: pathlib.Path
        Temporary, disposable workspace and source-file location.

    """

    from PySide6.QtCore import QEventLoop, QPointF, QTimer, QUrl, QMimeData
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QDropEvent
    from PySide6.QtWidgets import QApplication
    from PySide6.QtWebChannel import QWebChannel
    from PySide6.QtWebEngineCore import QWebEnginePage

    from kilin.gui.window import ChatBridge, SourceDropView

    app = QApplication([])

    class _Page(QWebEnginePage):
        """Accept test-source confirmations inside the isolated page."""

        def javaScriptConfirm(self, url, message) -> bool:
            """Confirm a source-import prompt inside the smoke test."""

            return True

    view = SourceDropView()
    view.resize(1180, 820)
    page = _Page(view)
    view.setPage(page)
    bridge = ChatBridge(view)
    channel = QWebChannel(page)
    channel.registerObject("kilin", bridge)
    page.setWebChannel(channel)
    bridge.sourceDropTargetChanged.connect(view.set_drop_target)
    view.pathsDropped.connect(bridge.pathsSelected)
    view.folderDropped.connect(bridge.folderDropped)
    view.dragChanged.connect(bridge.sourceDragChanged)
    commands: list[dict[str, object]] = []
    bridge.commandRequested.connect(commands.append)
    view.show()

    def pause(milliseconds: int = 100) -> None:
        """Allow the Qt event loop to dispatch page and bridge updates."""

        loop = QEventLoop()
        QTimer.singleShot(milliseconds, loop.quit)
        loop.exec()

    def javascript(expression: str) -> object:
        """Evaluate one expression and require a WebEngine callback."""

        loop = QEventLoop()
        values: list[object] = []

        def finished(value: object) -> None:
            """Collect the result and resume the Qt test thread."""

            values.append(value)
            loop.quit()

        page.runJavaScript(expression, finished)
        QTimer.singleShot(5000, loop.quit)
        loop.exec()
        if not values:
            raise AssertionError(
                f"JavaScript callback timed out: {expression}"
            )
        return values[0]

    try:
        loaded: list[bool] = []
        view.loadFinished.connect(loaded.append)
        html = (
            Path(__file__).resolve().parents[3]
            / "kilin/gui/web/index.html"
        )
        view.load(QUrl.fromLocalFile(str(html)))
        for _ in range(50):
            pause()
            if loaded and javascript("Boolean(bridge)"):
                break
        if loaded != [True]:
            raise AssertionError("The KíLín WebEngine page did not load")

        answer = "The response must reach the web page."
        bridge.event.emit(
            json.dumps(
                {
                    "event": "opened",
                    "project": str(directory / "project"),
                    "type_id": "literature",
                    "workspace_files": ["sources/papers/paper.md"],
                    "turns": [
                        {
                            "question": "What happened?",
                            "answer": answer,
                            "evidence": [],
                            "parameters": {},
                        }
                    ],
                    "temperature": 0.1,
                    "max_new_tokens": 2048,
                    "thinking": False,
                    "active_conversation_id": "chat-1",
                    "conversations": [
                        {
                            "conversation_id": "chat-1",
                            "name": "KíLín test",
                            "empty": False,
                        }
                    ],
                }
            )
        )
        pause(250)
        javascript(
            "showConversations(false); showWorkspace(true); "
            "syncSourceDropTarget();"
        )
        pause(250)
        if not javascript(
            "document.querySelector('.answer-message')"
            ".textContent.includes('response must reach')"
        ):
            raise AssertionError("The answer did not render")
        if view.drop_target.isEmpty():
            raise AssertionError(
                "The page did not register a source drop zone"
            )

        source = directory / "new-paper.md"
        source.write_text("# New source\n", encoding="utf-8")
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(source))])
        event = QDropEvent(
            QPointF(view.drop_target.center()),
            Qt.DropAction.CopyAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        view.dropEvent(event)
        pause(200)
        if not event.isAccepted():
            raise AssertionError("Native source drop was refused")
        if not any(
            command.get("action") == "add_project_sources"
            and command.get("sources") == [str(source)]
            for command in commands
        ):
            raise AssertionError(f"Missing source command: {commands!r}")

        # Check the complete chat layout at ordinary and narrow widths.
        # Each scenario uses a fresh rendered conversation.
        for window_width in (1180, 940, 760):
            view.resize(window_width, 820)
            pause(150)
            measured = javascript(_CONVERSATION_WIDTH_CHECK)
            if not isinstance(measured, str):
                raise AssertionError("Conversation layout did not render")
            _assert_conversation_width(json.loads(measured), window_width)
    finally:
        view.close()
        app.quit()


def _assert_conversation_width(metrics: dict, window_width: int) -> None:
    """Require contained chat text and independently scrolling blocks.

    Parameters
    ----------
    metrics: dict
        Measurements from the rendered conversation and its child elements.
    window_width: int
        Requested window width, used in failure diagnostics.

    """

    for name in (
        "suggestion", "suggestionButton", "welcome", "welcomeHeading",
        "title", "titleHeading", "question", "questionBubble",
    ):
        box = metrics[name]
        if (
            box["client"] <= 0
            or box["scroll"] > box["client"] + 1
            or not box["inside"]
        ):
            raise AssertionError(
                f"{window_width}px window: {name} overflows: {box!r}"
            )
    for name in ("suggestion", "welcome", "title", "question"):
        if metrics[name]["overflow"] != "hidden":
            raise AssertionError(f"Horizontal chat scrolling: {name}")
    for name in ("code", "table", "math"):
        box = metrics[name]
        if (
            not box["inside"] or box["overflow"] != "auto"
            or box["scroll"] <= box["client"] or not box["moves"]
        ):
            raise AssertionError(
                f"{window_width}px window: {name} cannot scroll: {box!r}"
            )
    if metrics["verticalOverflow"] != "auto":
        raise AssertionError("Vertical conversation scrolling was lost")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        _native_smoke(Path(sys.argv[1]))
    else:
        unittest.main()
