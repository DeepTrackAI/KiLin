# QAChat desktop foundation

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
with **New chat**. Model selection and GenRAG's advanced panels can be added
later without moving RAG logic into this folder.

Stop terminates the active backend process. Already saved indexes and
conversations remain on disk; reopen the project to continue.
