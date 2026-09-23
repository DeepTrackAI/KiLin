# QAChat desktop foundation

Launch the desktop interface from the repository with:

~~~sh
uv run qachat --gui
~~~

The launcher prepares the missing GUI layer and native model backend
automatically. If .venv is active, qachat --gui is enough; --backend mlx and
--backend torch override Automatic.

The optional desktop UI follows GenRAG's separation of a Qt shell, a
JavaScript layout, and a Python RAG backend. Qt's WebChannel connects the
page to its shell; a child process owns the backend so Stop can end a long
operation without terminating the window.

- `app.py` starts Qt; `window.py` hosts the web page and bridge.
- `engine.py` runs project loading and questions in the child process. It
  calls public `qachat` functions; retrieval and model logic stay in `qachat/`.
- `web/` contains the HTML, CSS, and JavaScript interface. It renders backend
  text as text, including answers and evidence.

The current UI opens code or literature projects, shows progress, restores
conversation, asks questions, displays evidence, and archives a conversation
with **New chat**. The landing screen stays on source selection; when a saved
sandbox project exists, a small Resume action opens it. **New project** returns
to source selection. The workspace panel is hidden when no project is open
and can be toggled in chat.

Click the source area or drag in several files and folders. Review the scanned
file tree and uncheck anything you do not want copied. Known importable file
types are offered automatically; Advanced settings can add UTF-8 text types.
Checked files determine what is copied and add their types to the project
profile. Larger files start unchecked; checking one raises the saved size
limit to fit it. Protected-file exclusions still apply. QAChat
copies checked files under one workspace's `sources/` folder, preserving folder
layout.
Adding more sources to the same workspace is cumulative. Choose an empty
workspace folder anywhere, or leave it blank to use QAChat's `sandbox/`.
Selecting an existing QAChat workspace with no new sources resumes it. Source
files stay untouched; `profile.ini`, `indexes/`, and `logs/` live in the
workspace. Re-adding a source updates matching copied files; previously copied
files remain until removed from the workspace explicitly.

Advanced settings edit `profile.ini` before model loading and indexing. The
**Models** button opens a hideable panel with QAChat's suggested chat and
embedding models for the selected backend. It marks which are cached and lets
you explicitly download a missing suggestion. Other cached models are visible
but not assumed to have the right role. The language-model menu shows every
model in the local Hugging Face cache.
The embedding menu stays limited to cached embedding choices for the selected
backend and query format; a custom model ID can still be typed and is checked
when loaded. New GUI projects do not silently download
models during preparation. New literature projects allow PDFs
up to 20 MiB by default. Existing projects keep their saved file-size limit,
which you can change in Advanced settings. The workspace sidebar shows saved
files.

Conversation settings beside chat control evidence count, temperature,
thinking, and answer length without rebuilding the index or reloading models.
The values used for each answer are saved with that turn. Reopening a project
restores the last answer's conversation settings.

Stop terminates the active backend process. Already saved indexes and
conversations remain on disk; reopen the project to continue.
