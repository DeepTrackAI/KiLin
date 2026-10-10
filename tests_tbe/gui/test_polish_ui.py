"""Exercise the actual Qt page and native file drop bridge without models."""

import json
from dataclasses import asdict
import os
from pathlib import Path
import unittest


class TestPolishUI(unittest.TestCase):
    def test_sources_drop_and_full_answer(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu")
        try:
            from PySide6.QtCore import (
                QEventLoop,
                QTimer,
                QUrl,
                QPointF,
                QMimeData,
                Qt,
            )
            from PySide6.QtGui import (
                QDropEvent,
                QDragEnterEvent,
            )
            from PySide6.QtWidgets import QApplication
            from PySide6.QtWebChannel import QWebChannel
            from PySide6.QtWebEngineCore import QWebEnginePage
            from qachat.gui.window import ChatBridge, SourceDropView

        except ImportError:
            self.skipTest(
                "Optional PySide6 GUI dependencies are not installed"
            )
        self.app = QApplication.instance() or QApplication([])

        class Page(QWebEnginePage):
            confirmed = False
            questions = []
            errors = []

            def javaScriptConfirm(self, url, message):
                self.questions.append(message)
                return self.confirmed

            def javaScriptConsoleMessage(self, level, message, line, source):
                levels = QWebEnginePage.JavaScriptConsoleMessageLevel
                if level == levels.ErrorMessageLevel:
                    self.errors.append(message)
                    print("JS ERROR:", message, line)

        view = SourceDropView()
        view.resize(1180, 820)
        page = Page(view)
        view.setPage(page)
        bridge = ChatBridge(view)
        channel = QWebChannel(page)
        channel.registerObject("qachat", bridge)
        page.setWebChannel(channel)
        bridge.sourceDropTargetChanged.connect(view.set_drop_target)
        view.pathsDropped.connect(bridge.pathsSelected)
        view.folderDropped.connect(bridge.folderDropped)
        view.dragChanged.connect(bridge.sourceDragChanged)
        commands = []
        bridge.commandRequested.connect(commands.append)
        view.show()
        self.addCleanup(view.close)

        def pause(ms=100):
            loop = QEventLoop()
            QTimer.singleShot(ms, loop.quit)
            loop.exec()

        def js(script):
            loop = QEventLoop()
            out = []

            def done(value):
                out.append(value)
                loop.quit()

            page.runJavaScript(script, done)
            QTimer.singleShot(5000, loop.quit)
            loop.exec()
            assert out, script
            return out[0]

        loaded = []
        view.loadFinished.connect(loaded.append)
        view.load(
            QUrl.fromLocalFile(
                str(
                    Path(__file__).resolve().parents[2]
                    / "qachat/gui/web/index.html"
                )
            )
        )
        for _ in range(50):
            pause()
            if loaded and js("Boolean(bridge)"):
                break
        assert loaded == [True], loaded
        turn = {
            "question": "What is memory capacity?",
            "answer": (
                "Memory capacity measures how much past input "
                "a network can retain. " * 50
            )
            + "FINAL SENTENCE.",
            "evidence": [],
            "parameters": {"finish_reason": "length"},
        }
        event = {
            "event": "opened",
            "project": "/tmp/Research",
            "kind": "literature",
            "workspace_files": [
                "indexes/retrieval.json",
                "logs/traces/chat.json",
                "profile.ini",
                "sources/papers/paper.md",
            ],
            "turns": [turn],
            "temperature": 0.1,
            "max_new_tokens": 2048,
            "thinking": False,
            "active_conversation_id": "chat-1",
            "conversations": [
                {
                    "conversation_id": "chat-1",
                    "name": "Memory capacity",
                    "empty": False,
                }
            ],
        }
        bridge.event.emit(json.dumps(event))
        pause(300)
        js(
            "showConversations(false); showWorkspace(true); "
            "syncSourceDropTarget();"
        )
        pause(300)
        assert not view.drop_target.isEmpty(), js("document.body.innerText")
        assert js(
            'document.querySelector(".answer-message").textContent'
            '.includes("FINAL SENTENCE.")'
        )
        assert (
            js('document.querySelectorAll(".answer-limit-notice").length') == 1
        )
        js("conversationScroll.scrollTop = conversationScroll.scrollHeight;")
        pause()
        geometry = json.loads(
            js(
                'JSON.stringify({answer:document.querySelector(".answer-message")'
                ".getBoundingClientRect().bottom, "
                'composer:document.querySelector(".composer")'
                ".getBoundingClientRect().top})"
            )
        )
        assert geometry["answer"] <= geometry["composer"], geometry

        assert js('sourceDropSpotlight.classList.contains("hidden")')
        assert js(
            'getComputedStyle(byId("workspace-source-drop"))'
            '.borderTopWidth === "0px"'
        )
        js('byId("question").value = "Keep my draft"')

        # Native invalid target never enters the confirmation flow.
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile("/tmp/new-paper.md")])

        def drop(point):
            event = QDropEvent(
                point,
                Qt.DropAction.CopyAction,
                mime,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
            view.dropEvent(event)
            pause()
            return event.isAccepted()

        commands.clear()
        child = view.focusProxy()
        drag_signals = []
        view.dragChanged.connect(drag_signals.append)
        enter = QDragEnterEvent(
            view.drop_target.center().toPoint(),
            Qt.DropAction.CopyAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        QApplication.sendEvent(child, enter)
        pause()

        assert drag_signals == [
            True
        ], "Native drag did not reach SourceDropView"
        assert (
            js(
                'document.querySelectorAll(".source-drop-target.drag-over").length'
            )
            == 1
        )
        assert js('!sourceDropSpotlight.classList.contains("hidden")')
        assert js('getComputedStyle(sourceDropSpotlight).boxShadow !== "none"')
        assert js(
            'getComputedStyle(sourceDropSpotlight).pointerEvents === "none"'
        )
        if os.environ.get("QACHAT_TEST_SCREENSHOT"):
            view.grab().save(os.environ["QACHAT_TEST_SCREENSHOT"])
        index_point = json.loads(
            js(
                "JSON.stringify((() => { const r = document.querySelector("
                '"#workspace-tree .tree-folder summary")'
                ".getBoundingClientRect();"
                "return {x:r.x+10,y:r.y+10}; })())"
            )
        )
        assert not drop(QPointF(index_point["x"], index_point["y"]))
        assert not drop(QPointF(20, 20))
        assert not page.questions
        assert drop(view.drop_target.center())
        assert len(page.questions) == 1
        assert not any(c["action"] == "add_project_sources" for c in commands)
        page.confirmed = True
        assert drop(view.drop_target.center())
        assert any(
            c["action"] == "add_project_sources"
            and c["sources"] == ["/tmp/new-paper.md"]
            for c in commands
        ), commands
        pause()
        assert (
            view.drop_target.isEmpty()
        ), "Drops must be disabled during rebuild"
        for control in (
            "question",
            "send",
            "branch-chat",
            "attach-button",
            "effort-button",
        ):
            assert js(f'byId("{control}").disabled'), control
        assert js('!byId("stop").disabled')
        assert js('!byId("stop").classList.contains("hidden")')
        assert js('sourceDropSpotlight.classList.contains("hidden")')
        assert js('byId("question").value') == "Keep my draft"
        commands.clear()
        js('byId("send").click(); byId("branch-chat").click();')
        pause()
        assert not commands
        bridge.event.emit(json.dumps(event))
        pause()
        assert js('!byId("question").disabled && !byId("send").disabled')
        assert js('!byId("branch-chat").disabled')
        assert js('byId("question").value') == "Keep my draft"
        # Answer generation still offers its normal Stop button.
        js('addPendingTurn("Question"); setBusy(true, "Answering");')
        assert js('!byId("send").disabled')
        assert (
            js('byId("send").getAttribute("aria-label")') == "Stop generation"
        )
        assert js('byId("stop").classList.contains("hidden")')
        js("stopPhaseAnimation(); resetAnswerStages(); setBusy(false);")
        # Floating title edits use the same rename path as the main heading.
        commands.clear()
        js(
            'byId("topbar-project-name").textContent = "Renamed"; '
            'byId("topbar-project-name").dispatchEvent(new Event("input")); '
            'byId("topbar-project-name").dispatchEvent(new Event("blur"));'
        )
        pause()
        assert any(
            c == {"action": "rename_project", "name": "Renamed"}
            for c in commands
        ), commands
        bridge.event.emit(json.dumps(event))
        pause()
        from qachat.project import default_project_profile
        from qachat.gui.model_catalog import model_choices, MODEL_CHOICES
        from qachat.gui.preferences import default_effort_levels

        test_profile = asdict(default_project_profile("literature"))
        cached_ids = [choice.model_id for choice in MODEL_CHOICES]
        cached_ids += [
            str(level["model"])
            for level in default_effort_levels("mlx")
            if level["model"] not in cached_ids
        ]
        cached_ids.append("example/other-cached")
        settings = {
            "event": "settings",
            "profile": test_profile,
            "backend": "mlx",
            "cached_models": cached_ids,
            "model_choices": model_choices(set(cached_ids)),
            "installed_backends": {"mlx": True, "torch": False},
            "supported_extensions": [".txt", ".md", ".pdf"],
            "workspace": "",
            "effort_levels": {},
            "default_effort_levels": {
                backend: default_effort_levels(backend)
                for backend in ("mlx", "torch")
            },
        }
        bridge.event.emit(json.dumps(settings))
        pause()
        js("openModels();")
        order = json.loads(
            js(
                "JSON.stringify([...document.querySelectorAll("
                '".models-card > .settings-section > h3")].filter('
                "n => n.offsetParent).map(n => n.textContent))"
            )
        )
        # Effort levels head the models section through their own heading.
        assert order == ["Appearance", "Answers", "Project"], order
        # The removed controls stay removed.
        for selector in (".other-models", "#allow-download"):
            assert js(f'document.querySelector("{selector}") === null')
        assert js('byId("models-backend").classList.contains("hidden")')
        js('document.querySelector(".models-card").click();')
        assert js('!byId("models-panel").classList.contains("hidden")')
        js('byId("models-panel").click();')
        assert js('byId("models-panel").classList.contains("hidden")')
        js("openModels();")
        if os.environ.get("QACHAT_SETTINGS_SCREENSHOT"):
            view.grab().save(os.environ["QACHAT_SETTINGS_SCREENSHOT"])
        page.questions.clear()
        page.confirmed = False
        commands.clear()
        rows = (
            '[...document.querySelectorAll(".level-row:not(.level-columns)")]'
        )
        assert js(f"{rows}.length") == 5
        assert not js('byId("chat-model").offsetParent')
        # Model choices are dropdowns of cached language models only.
        options = json.loads(
            js(
                f"JSON.stringify([...{rows}[0].querySelector('select')"
                ".options].map(o => o.value))"
            )
        )
        assert "example/other-cached" not in options
        assert not any("Embedding" in option for option in options)
        js(
            'draftLevels[0].model = "example/new-model"; '
            "renderModelsPanel();"
        )
        assert js(
            'document.querySelector(".model-missing button").textContent'
        ) == ("Download")
        js('byId("models-panel").click();')
        pause()
        assert any("download" in question for question in page.questions)
        assert not any(c["action"] == "apply_models" for c in commands)
        assert js('!byId("models-panel").classList.contains("hidden")')
        page.confirmed = True
        js('byId("close-models").click();')
        pause()
        applied = next(c for c in commands if c["action"] == "apply_models")
        assert applied["settings"]["download"] is True
        levels = applied["settings"]["effort_levels"]
        assert levels[0]["model"] == "example/new-model"
        bridge.event.emit(
            json.dumps(
                {
                    "event": "models_applied",
                    "effort_levels": {"mlx": levels},
                    "profile": test_profile,
                    "cached_models": cached_ids + ["example/new-model"],
                    "model_roles": {"example/new-model": "chat"},
                }
            )
        )
        pause()
        assert js('byId("models-panel").classList.contains("hidden")')
        js("openModels();")
        assert not js('document.querySelector(".model-missing") !== null')
        assert js('document.querySelector(".model-group h3").textContent') == (
            "Effort levels"
        )
        restore = 'document.querySelector(".restore-defaults")'
        assert not js(f"{restore}.disabled")
        js(f"{restore}.click();")
        assert js("draftLevels[0].model") != "example/new-model"
        assert js(f"{restore}.disabled")
        # Names and reference counts are editable; the picker follows them.
        js(
            f"const first = {rows}[0].querySelector('.level-name'); "
            "first.value = 'Quick'; first.dispatchEvent(new Event('input'));"
        )
        assert js("draftLevels[0].name") == "Quick"
        commands.clear()
        page.questions.clear()
        page.confirmed = False
        js(
            'cachedModels.add("example/new-embedding"); '
            'byId("embedding-model").value = "example/new-embedding"; '
            'byId("close-models").click();'
        )
        assert any("rebuild" in question for question in page.questions)
        assert not any(c["action"] == "apply_models" for c in commands)
        page.confirmed = True
        js('byId("models-panel").click();')
        pause()
        applied = next(c for c in commands if c["action"] == "apply_models")
        assert (
            applied["settings"]["embedding_model"] == "example/new-embedding"
        )
        assert applied["settings"]["download"] is False
        bridge.event.emit(
            json.dumps(
                {
                    "event": "models_applied",
                    "effort_levels": {
                        "mlx": applied["settings"]["effort_levels"]
                    },
                    "profile": test_profile,
                    "cached_models": cached_ids + ["example/new-embedding"],
                    "model_roles": {},
                }
            )
        )
        pause()
        assert js('byId("models-panel").classList.contains("hidden")')
        # While QAChat is busy, Settings opens and closes freely ...
        js('setBusy(true, "Answering…"); openModels();')
        assert js('!byId("models-panel").classList.contains("hidden")')
        assert not js('byId("close-models").disabled')
        js('byId("close-models").click();')
        pause()
        assert js('byId("models-panel").classList.contains("hidden")')
        # ... but unsaved edits keep it open until QAChat is idle.
        js('openModels(); draftDepth[0] = 3; byId("close-models").click();')
        pause()
        assert js('!byId("models-panel").classList.contains("hidden")')
        assert "busy" in js('byId("model-download-status").textContent')
        js("draftDepth[0] = currentDepth()[0]; setBusy(false);")
        js('byId("close-models").click();')
        pause()
        assert js('byId("models-panel").classList.contains("hidden")')
        # Outputs and templates are always shown and can be edited.
        for folder in ("outputs", "templates"):
            js(
                "document.querySelector('#workspace-tree "
                f'details[data-path="{folder}"] > summary\')'
                '.dispatchEvent(new MouseEvent("contextmenu", '
                "{bubbles: true, clientX: 40, clientY: 40}));"
            )
            menu = js('document.querySelector(".context-menu").innerText')
            assert f"Add files to {folder}" in menu, menu
        # A Finder drop onto templates copies the file there (no indexing).
        js(
            'renderWorkspace(["profile.ini", "sources/papers/paper.md", '
            '"outputs/paper_summary.md", "templates/research_briefing.md"]);'
            "showWorkspace(true); showConversations(false);"
            "byId('models-panel').classList.add('hidden');"
            "setBusy(false); syncSourceDropTarget();"
        )
        pause(300)
        box = json.loads(
            js(
                "JSON.stringify(document.querySelector('#workspace-tree "
                "details[data-path=\"templates\"]').getBoundingClientRect())"
            )
        )
        commands.clear()
        assert drop(QPointF(box["x"] + 20, box["y"] + box["height"] / 2))
        pause()
        edit = next(c for c in commands if c["action"] == "edit_folder")
        assert edit["folder"] == "templates", edit
        assert edit["operation"] == "add"
        assert edit["paths"] == ["/tmp/new-paper.md"]
        js("setBusy(false);")
        # Dragging an outputs file onto sources asks to add it as a source.
        page.questions.clear()
        page.confirmed = False
        js(
            "const data = new DataTransfer();"
            "data.setData(WORKSPACE_DRAG_TYPE, 'outputs/paper_summary.md');"
            "byId('workspace-source-drop').dispatchEvent("
            "new DragEvent('drop', {bubbles: true, dataTransfer: data}));"
        )
        pause()
        assert any(
            "Add these sources" in q and "paper_summary.md" in q
            for q in page.questions
        ), page.questions
        # Files in outputs and templates are draggable and deletable.
        assert js(
            "document.querySelector('#workspace-tree "
            "details[data-path=\"outputs\"] .tree-file').draggable"
        )
        assert js(
            "document.querySelectorAll('#workspace-tree .tree-delete').length"
        ) == 2
        # Dragging the conversations border far left hides the panel.
        js(
            "showConversations(true); positionPanelResizers();"
            "const h = panelResizers.left;"
            "h.setPointerCapture = () => {};"
            "h.releasePointerCapture = () => {};"
            "h.hasPointerCapture = () => true;"
            "window.dragTo = (x) => {"
            "  h.dispatchEvent(new PointerEvent('pointerdown',"
            "    {pointerId: 1, clientX: 200, bubbles: true}));"
            "  h.dispatchEvent(new PointerEvent('pointermove',"
            "    {pointerId: 1, clientX: x, bubbles: true}));"
            "};"
            "window.release = () => h.dispatchEvent(new PointerEvent("
            "  'pointerup', {pointerId: 1, bubbles: true}));"
        )
        js("dragTo(260); release();")
        assert not js('shell.classList.contains("conversations-hidden")')
        assert js('shell.style.getPropertyValue("--left-width")') == "260px"
        js("dragTo(60);")
        assert js('shell.classList.contains("collapsing-left")')
        js("release();")
        assert js('shell.classList.contains("conversations-hidden")')
        assert not js('shell.classList.contains("collapsing-left")')
        # It reopens at its last width.
        assert js('shell.style.getPropertyValue("--left-width")') == "260px"
        js("showConversations(false);")
        # Token counts are short and readable; the composer shows totals.
        assert [
            js(f"formatTokens({n})")
            for n in (850, 3456, 42_000, 1_234_567, 2_500_000_000)
        ] == ["850", "3.5k", "42k", "1.2M", "2.5B"]
        js(
            "usageTotals = {project: {input_tokens: 3000, output_tokens: 456,"
            " answers: 4}, chat: {input_tokens: 900, output_tokens: 100,"
            " answers: 1}}; renderUsage();"
        )
        # The composer shows only the gauge icon; the menu has the numbers.
        assert js('byId("usage-button").textContent.trim()') == ""
        js('byId("effort-button").click();')
        js('byId("usage-button").click();')
        assert js('!byId("usage-menu").classList.contains("hidden")')
        # One composer menu at a time.
        assert js('byId("effort-menu").classList.contains("hidden")')
        assert js('byId("usage-project-total").textContent') == "3.5k tokens"
        assert js('byId("usage-chat-answers").textContent') == "1 answer"
        read = js('byId("usage-project-read-text").textContent')
        # Numbers follow the system locale ("3,000", "3 000", ...).
        assert "".join(c for c in read if c.isdigit()) == "3000", read
        written = js('byId("usage-project-written-text").textContent')
        assert "written" in written
        # The bar is full: read, then written.
        widths = js(
            'parseFloat(byId("usage-project-read").style.width)'
            ' + parseFloat(byId("usage-project-written").style.width)'
        )
        assert abs(widths - 100) < 0.2, widths
        js('byId("effort-button").click();')
        assert js('byId("usage-menu").classList.contains("hidden")')
        js('byId("effort-menu").classList.add("hidden");')
        js("document.body.click();")
        assert js('byId("usage-menu").classList.contains("hidden")')
        # The setup card stays available, folded when everything is ready.
        readiness = {
            "event": "readiness",
            "ready": True,
            "environment_ready": True,
            "backend": "mlx",
            "required_models": cached_ids[:1],
            "missing_models": [],
            "cached_models": cached_ids,
        }
        bridge.event.emit(json.dumps(readiness))
        pause()
        card = 'byId("first-run-setup")'
        assert js(f'!{card}.classList.contains("hidden") && !{card}.open')
        assert js('byId("readiness-status").textContent') == "Ready"
        assert js('byId("install-required").classList.contains("hidden")')
        bridge.event.emit(
            json.dumps(
                {
                    **readiness,
                    "ready": False,
                    "missing_models": ["example/missing"],
                    "required_models": ["example/missing"],
                }
            )
        )
        pause()
        assert js(f"{card}.open")
        assert js('byId("readiness-status").textContent') == "1 missing"
        js('byId("models-panel").classList.add("hidden");')
        assert not page.errors, page.errors

        view.close()


if __name__ == "__main__":
    unittest.main()
