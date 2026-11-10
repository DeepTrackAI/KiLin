/**
 * Present the local KíLín interface inside Qt WebEngine.
 *
 * The page owns transient UI state and delegates filesystem/model operations
 * through QWebChannel. Engine events update project, conversation, evidence,
 * and progress displays. Model answers and source text are inserted as text,
 * not interpreted as HTML. Source selection is a preview, not a filesystem
 * security boundary. Stop terminates the engine rather than pausing inference.
 *
 * Completed answers arrive as complete strings. The character-by-character
 * reveal is a presentation timer, not streamed model output. Conversation
 * persistence and tracing remain backend responsibilities. Answer-settings
 * edits are provisional until the engine confirms the saved project profile.
 */

"use strict";

// Resolve fixed DOM identifiers after the deferred script can see the page.
const byId = (id) => document.getElementById(id);
// Answer phases sent by RAGSession (kilin.rag.ANSWER_PHASE_*); the strings
// are the core's wire values and must stay identical to it.
const ANSWER_PHASE_ROUTING = "routing";
const ANSWER_PHASE_ROUTED = "routed";
const ANSWER_PHASE_RETRIEVING = "retrieving";
const ANSWER_PHASE_EVIDENCE_READY = "evidence_ready";
const ANSWER_PHASE_GENERATING = "generating";
const ANSWER_PHASE_SAVING = "saving";
const ANSWER_PHASE_COMPLETED = "completed";
const ANSWER_PHASE_PROCESSING = "processing";
// Mirrors kilin.tools.planning.PlanChoice.
const PLAN_CHOICE_CONTINUE = "continue";
const PLAN_CHOICE_QUICK = "quick";
const PLAN_CHOICE_CANCEL = "cancel";
// Phase added by the GUI engine (kilin.gui.engine.ENGINE_PHASE_*).
const ENGINE_PHASE_LOADING_MODEL = "loading_model";
const setup = byId("setup");
const chat = byId("chat");
const conversation = byId("conversation");
const conversationScroll = byId("conversation-scroll");
const activity = byId("activity");
const error = byId("error");
const shell = document.querySelector(".app-shell");
let bridge = null;
let busy = false;
let ready = false;
// The Settings status text that showError last wrote, cleared with it.
let shownErrorStatus = "";
// The setup reminder shown in the Settings status line, until it is done.
let shownSetupReminder = "";
const selectedSources = [];
const previewRoots = new Map();
const previewSkipped = new Map();
// Codes arrive from the backend's validated, packaged JSON resource.
let selectionReasonCodes = null;
const checkedFiles = new Map();
const expandedFolders = new Set();
// Workspace folders that start open but the user has folded.
const collapsedWorkspaceFolders = new Set();
// Paths shown in the workspace tree, to warn before replacing files.
let workspacePaths = new Set();
// The user's last show/hide choice for the side panels; null before any.
let sidePanelChoice = null;
// The Settings tab shown when Settings open; kept while KíLín runs.
let settingsTab = "general";
// The last projects_folder event, to redraw the folder's note.
let projectsFolderInfo = null;
// Workspace folders the user edits directly; neither is indexed.
const EDITABLE_FOLDERS = ["outputs", "templates"];
// Drag data for a workspace file dragged within the panel.
const WORKSPACE_DRAG_TYPE = "application/x-kilin-workspace-path";
let profile = null;
// Answer settings changed on the start screen: kept until the project is
// created, instead of being reset by the next settings event.
let answerSettingsEdited = false;
// The last engine-confirmed answer settings; the UI may show newer edits.
let savedAnswerNumbers = { temperature: 0, max_new_tokens: null };
let savedAnswerThinking = true;
// One save at a time. Only the latest newer edit is queued, and every
// acknowledgement is checked against its request and project identity.
const answerSettingsSave = {
  nextId: 0, pending: null, queued: null, uncertain: false,
};
let cachedModels = new Set();
let modelChoices = [];
// Effort levels per backend: saved for the project, built-in defaults, and
// the Settings panel's unsaved copy for the active backend.
let savedLevels = {};
let defaultLevels = {};
let draftLevels = [];
let defaultEmbeddings = {};
// Search depth: passages retrieved at each of five steps (project-wide).
let savedDepth = null;
let defaultDepth = [5, 8, 10, 15, 20];
let draftDepth = [];
// Intent router model per backend: saved, built-in, and unsaved choice.
let savedRouters = {};
let defaultRouters = {};
let draftRouter = "";
let modelRoles = {};
// Whether the tested chat template changes its rendered prompt when
// enable_thinking changes; missing entries mean unknown.
let thinkingSwitches = {};
let modelsBaseline = null;
let applyingModels = false;
let closeAfterModelsApply = false;
// Set while the engine saves indexing settings and rebuilds the index.
let applyingIndexing = false;
// Source-card settings are project-owned and independent of answering models.
let sourceCardSettings = null;
let applyingSourceCards = false;
let downloadingModel = null;
let selectedTurn = null;
let activeReference = null;
let installedBackends = {};
let requiredModels = [];
let readinessEnvironmentReady = false;
// Repair commands from Python, one per backend and installation type.
let installCommand = "";
let installCommands = {};
// KíLín setup also asks to confirm the projects folder and to agree to the
// current terms of use (``termsVersion``), once each.
let folderConfirmed = true;
let termsAccepted = true;
let termsVersion = "";
// The in-flight terms version identifies the Qt acknowledgement to accept.
let acceptingTerms = false;
let pendingTermsVersion = "";
// Retain a successful Qt write if an earlier readiness reply arrives late.
let acknowledgedTermsVersion = "";
// New-project readiness; saved projects use their own profiles.
let kilinReady = true;
// Required models still missing, and whether they are being downloaded.
let readinessMissingCount = 0;
let installingModels = false;
let setupCompletionVisible = false;
let supportedExtensions = new Set();
let automaticBackend = "mlx";
// The last backend acknowledged by Qt, not the page's pending selection.
let confirmedBackendPreference = "auto";
let activeBackend = "mlx";
let activeTracePath = "";
let activeConversationId = "";
// Last conversation registry, and the empty chat whose first question is
// being answered; that chat is listed before it has saved turns.
let lastConversationData = null;
let askingConversation = null;
let pendingTurn = null;
// A stopped answer's partial text, shown again after the engine restarts.
let stoppedTurn = null;
// The question put back in the message box once a branch or new chat opens.
let branchDraft = null;
// The chat being answered while an answer runs; other chats can be read
// meanwhile.
let answeringConversationId = "";
// A chat read while another one is answered: its id and name, and the
// answering chat's own content, put back by returnToAnswer().
let reading = null;
// Chats whose answer finished while the user was reading another one.
const unreadConversations = new Set();
let phaseTimer = null;
let answerTimer = null;
// Tokens used by the open chat and the project, from the engine; the
// running answer's tokens are added live while it is generated.
let usageTotals = null;
let answerStageTimer = null;
let answerStageKey = null;
let answerStageQueue = [];
let selectedModelLevel = 2;
let selectedDepthStep = 2;
let resumeAfterStop = false;
let activeProjectTypeId = "";
const lastEmptyPrompt = Object.create(null);
let messageAttachments = [];
// A question whose attached files are being added to sources; it is asked
// once the index is refreshed.
let attachedAsk = null;
// The question being answered, put back in the message box if the answer
// fails or is stopped.
let unansweredQuestion = "";
// Set while Qt restarts the engine to use another projects folder.
let switchingProjectsFolder = false;
let currentTurnCount = 0;
// Project types sent by the engine from kilin/project_types/*.json.
let projectTypes = Object.create(null);
// Saved projects last listed, shown again once the types' names arrive.
let savedProjects = [];
// Folder of the open project, highlighted in the left panel's Projects.
let openProjectPath = "";
// Used only if a type lists no welcome prompts or suggestions.
const genericEmptyChat = {
  welcome_prompts: ["What would you like to work on?"],
  suggestions: ["Explain a concept", "Develop an idea", "Compare two approaches"],
};

/**
 * Rebuild the project type dropdown, keeping the current choice.
 *
 * @param types Project types sent by the engine.
 */
function renderProjectTypes(types) {
  if (!Array.isArray(types) || !types.length) return;
  projectTypes = Object.fromEntries(types.map((item) => [item.type_id, item]));
  if (savedProjects.length) renderSavedProjects(savedProjects);
  if (openProjectPath) renderSidebarProjects(savedProjects, openProjectPath);
  const select = byId("type-id");
  const current = select.value;
  select.replaceChildren(...types.map((item) => {
    const option = document.createElement("option");
    option.value = item.type_id;
    option.textContent = item.label;
    return option;
  }));
  // The first type in registry filename order is the initial default.
  select.value = projectTypes[current] ? current : types[0].type_id;
  activeProjectTypeId = ready ? activeProjectTypeId : select.value;
}

/**
 * Format the number of conversation turns for display.
 *
 * @param count Number of conversation turns.
 * @returns Singular or plural turn-count label.
 */
function turnCountLabel(count) {
  return `${count} ${count === 1 ? "turn" : "turns"}`;
}

/**
 * Refresh the chat's name and turn count in its heading and the top bar.
 *
 * @param count Latest saved count; defaults to the current count.
 */
function updateChatSummary(count = currentTurnCount) {
  currentTurnCount = Number(count) || 0;
  byId("topbar-chat-name").textContent = byId("conversation-name").textContent.trim() || "New chat";
  byId("topbar-turn-count").textContent = turnCountLabel(currentTurnCount);
  byId("conversation-detail").textContent = currentTurnCount ? turnCountLabel(currentTurnCount) : "";
}

/**
 * Keep the compact chat header visible after scrolling the conversation.
 */
function updateFloatingHeader() {
  const scrolled = ready && conversationScroll.scrollTop > 96;
  document.querySelector(".topbar").classList.toggle("floating", scrolled);
  byId("topbar-chat").classList.toggle("hidden", !scrolled);
}

/**
 * Hide and clear the previously offered saved-project shortcut.
 */
function hideResume() { byId("saved-projects").classList.add("hidden"); }

/**
 * Ask the engine for a saved project only on an idle setup screen.
 */
function refreshSavedProject() {
  if (!bridge || busy || ready) return;
  bridge.listSavedProjects("");
}

/**
 * Toggle the workspace panels and their control label.
 *
 * @param visible Whether the sidebar layout should be shown.
 */
function showWorkspace(visible) {
  // Keep the workspace out of the layout until a project is actually open.
  shell.classList.toggle("sidebar-open", visible);
  byId("toggle-workspace").classList.toggle("hidden", !ready);
  const control = byId("toggle-workspace");
  control.setAttribute("aria-label", visible ? "Hide side panels" : "Show side panels");
  control.title = control.getAttribute("aria-label");
  control.classList.toggle("panels-hidden", !visible);
}

/**
 * Show or hide the side panels because the user asked, and remember it.
 *
 * Automatic layout changes (opening a project, a new chat) then follow
 * this choice instead of overriding it.
 *
 * @param visible Whether the user wants the side panels shown.
 */
function chooseSidePanels(visible) {
  sidePanelChoice = visible;
  showWorkspace(visible);
}

/**
 * Show or hide the conversations panel and synchronize its toggle.
 *
 * @param visible Whether the conversations panel should be visible.
 */
function showConversations(visible) {
  shell.classList.toggle("conversations-hidden", !visible);
  const control = byId("toggle-conversations");
  control.classList.toggle("hidden", !ready);
  control.classList.toggle("panels-hidden", !visible);
  control.setAttribute("aria-label", visible ? "Hide conversations" : "Show conversations");
  control.title = control.getAttribute("aria-label");
}

/**
 * Apply and persist the page appearance preference.
 *
 * @param value Requested theme: system, light, or dark.
 */
function applyAppearance(value) {
  const selected = ["system", "light", "dark"].includes(value) ? value : "system";
  document.documentElement.dataset.appearance = selected;
  localStorage.setItem("kilin-appearance", selected);
  if (byId("appearance")) byId("appearance").value = selected;
  bridge?.setAppearance(selected);
}

/**
 * Render the resumable projects provided by the backend.
 *
 * @param projects Saved projects ordered by recent use.
 */
function renderSavedProjects(projects) {
  savedProjects = projects;
  const host = byId("saved-projects");
  // The list is refreshed every few seconds; rebuilding it unchanged would
  // jump a scrolled list back to the top.
  const shown = JSON.stringify(projects.map((project) => [
    project.project, project.name, projectTypes[project.type_id]?.label || project.type_id,
  ]));
  if (host.dataset.shown === shown) return;
  host.dataset.shown = shown;
  const scroll = host.querySelector(".saved-project-list")?.scrollTop || 0;
  host.replaceChildren();
  if (projects.length) {
    // One list of every project type, most recently used first (engine order).
    const group = document.createElement("div");
    group.className = "saved-project-group";
    const heading = document.createElement("strong");
    heading.textContent = "Resume a previous project";
    // Five projects show; the rest scroll, so the setup card stays in view.
    const list = document.createElement("div");
    list.className = "saved-project-list";
    group.append(heading, list);
    for (const project of projects) {
      const button = document.createElement("button");
      button.type = "button";
      const name = document.createElement("span");
      name.className = "saved-project-name";
      name.textContent = project.name;
      const typeLabel = document.createElement("span");
      typeLabel.className = "saved-project-type";
      // The type's own name, or its ID until the types have arrived.
      typeLabel.textContent = projectTypes[project.type_id]?.label || project.type_id;
      button.append(name, typeLabel);
      button.title = `Open ${project.type_id} project from ${project.project}`;
      button.addEventListener("click", () => {
        if (busy) return;
        setBusy(true, "Opening project…");
        bridge.resumeProject(project.project);
      });
      list.append(button);
    }
    host.append(group);
    list.scrollTop = scroll;
  }
  host.classList.toggle("hidden", !host.children.length);
}

/**
 * Read a left-panel choice remembered on this computer.
 *
 * @param key Choice name, such as "projects-sort".
 * @param fallback Value when nothing is saved or storage is unavailable.
 * @returns The saved value, or ``fallback``.
 */
function sidebarPref(key, fallback) {
  try {
    const saved = localStorage.getItem(`kilin-sidebar-${key}`);
    return saved == null ? fallback : JSON.parse(saved);
  } catch {
    return fallback;
  }
}

/**
 * Remember a left-panel choice on this computer.
 *
 * @param key Choice name, such as "projects-sort".
 * @param value JSON-serializable value.
 */
function setSidebarPref(key, value) {
  try { localStorage.setItem(`kilin-sidebar-${key}`, JSON.stringify(value)); } catch {}
}

/**
 * Put items in the user's dragged order; items not yet placed come first,
 * in their given order.
 *
 * @param items Items in their default order.
 * @param idOf Returns an item's identifier.
 * @param saved Identifiers in the user's order.
 * @returns New list of the same items.
 */
function manualOrder(items, idOf, saved) {
  const rank = new Map((saved || []).map((id, index) => [id, index]));
  const fresh = items.filter((item) => !rank.has(idOf(item)));
  const placed = items.filter((item) => rank.has(idOf(item)))
    .sort((a, b) => rank.get(idOf(a)) - rank.get(idOf(b)));
  return [...fresh, ...placed];
}

/**
 * Move one identifier to the place of another, as a dropped row does.
 *
 * @param ids Identifiers in their shown order.
 * @param moved The dragged row's identifier.
 * @param target The identifier of the row it was dropped on.
 * @returns New list of identifiers.
 */
function moveInOrder(ids, moved, target) {
  const index = ids.indexOf(target);
  const rest = ids.filter((id) => id !== moved);
  rest.splice(index, 0, moved);
  return rest;
}

/**
 * Let a row be dragged onto another row of the same list.
 *
 * @param row Row element.
 * @param id The row's identifier.
 * @param ids Identifiers of the whole list, in shown order.
 * @param save Receives the new order after a drop.
 */
function makeReorderable(row, id, ids, save) {
  const type = "application/x-kilin-order";
  row.draggable = true;
  row.addEventListener("dragstart", (event) => {
    event.dataTransfer.setData(type, id);
    event.dataTransfer.effectAllowed = "move";
    row.classList.add("dragging");
  });
  row.addEventListener("dragend", () => row.classList.remove("dragging"));
  // Stop here, so the page-wide guard against file drops does not refuse it.
  row.addEventListener("dragover", (event) => {
    if (!event.dataTransfer.types.includes(type)) return;
    event.preventDefault();
    event.stopPropagation();
    event.dataTransfer.dropEffect = "move";
  });
  row.addEventListener("drop", (event) => {
    const moved = event.dataTransfer.getData(type);
    if (!moved) return;
    event.preventDefault();
    event.stopPropagation();
    if (moved !== id && ids.includes(moved)) save(moveInOrder(ids, moved, id));
  });
}

/**
 * Order the projects for the left panel.
 *
 * @param projects Saved projects ordered by recent use.
 * @param current Folder of the open project, listed first when it is not
 *   among the saved projects.
 * @param currentType Project type of the open project.
 * @param mode "updated" keeps recent use; "manual" uses ``saved``.
 * @param saved Project folders in the user's dragged order.
 * @returns New list of projects to show.
 */
function sidebarProjectOrder(projects, current, currentType = "", mode = "updated", saved = []) {
  const listed = !current || projects.some((project) => project.project === current)
    ? [...projects]
    : [{ project: current, name: current.split(/[\\/]/).pop(), type_id: currentType }, ...projects];
  return mode === "manual" ? manualOrder(listed, (project) => project.project, saved) : listed;
}

/**
 * Order the conversations for the left panel.
 *
 * @param items Conversation records with ``updated_at_utc``.
 * @param mode "updated" puts the latest change first; "manual" uses
 *   ``saved``.
 * @param saved Conversation identifiers in the user's dragged order.
 * @returns New list of conversations.
 */
function sidebarConversationOrder(items, mode = "updated", saved = []) {
  const recent = [...items].sort((a, b) =>
    (Date.parse(b.updated_at_utc) || 0) - (Date.parse(a.updated_at_utc) || 0));
  return mode === "manual" ? manualOrder(recent, (item) => item.conversation_id, saved) : recent;
}

/**
 * List the most recent projects in the left panel, each with its type, the
 * open one highlighted; clicking another one switches to it, and More shows
 * the rest.
 *
 * @param projects Saved projects ordered by recent use.
 * @param current Folder of the open project.
 */
function renderSidebarProjects(projects, current) {
  const list = byId("project-list");
  list.replaceChildren();
  const mode = sidebarPref("projects-sort", "updated");
  const ordered = sidebarProjectOrder(
    projects, current, activeProjectTypeId, mode, sidebarPref("projects-order", []));
  const ids = ordered.map((project) => project.project);
  // Five projects show; the list scrolls for the rest (see styles.css).
  for (const project of ordered) {
    const row = document.createElement("div");
    row.className = "project-row";
    const button = document.createElement("button");
    button.type = "button";
    button.className = "project-item";
    const name = document.createElement("span");
    name.className = "project-item-name";
    name.textContent = project.name;
    const type = document.createElement("span");
    type.className = "project-item-type";
    type.textContent = projectTypes[project.type_id]?.label || project.type_id || "";
    button.append(name, type);
    button.title = project.project;
    const active = project.project === current;
    button.classList.toggle("active", active);
    if (active) button.setAttribute("aria-current", "true");
    button.disabled = busy;
    const open = () => {
      if (busy || active) return;
      setBusy(true, `Opening ${project.name}…`);
      bridge.resumeProject(project.project);
    };
    button.addEventListener("click", () => open());
    // Settings save into the open project, so only its row has the icon.
    const settings = document.createElement("button");
    settings.type = "button";
    settings.className = "project-settings";
    settings.setAttribute("aria-label", `Settings for ${project.name}`);
    settings.title = "Project settings";
    settings.innerHTML = '<svg aria-hidden="true" viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/></svg>';
    settings.disabled = busy;
    settings.addEventListener("click", () => openModels());
    button.addEventListener("contextmenu", (event) => {
      event.preventDefault();
      showWorkspaceMenu(event.clientX, event.clientY, active
        ? [["Rename", () => renameProjectInSidebar(button, project)], ["Settings", openModels]]
        : [["Open", open]]);
    });
    row.append(button);
    if (active) row.append(settings);
    if (mode === "manual") {
      makeReorderable(row, project.project, ids, (order) => {
        setSidebarPref("projects-order", order);
        renderSidebarProjects(projects, current);
      });
    }
    list.append(row);
  }
}

/**
 * Rename the open project in place: Enter or leaving the field saves,
 * Escape cancels.
 *
 * @param button The project's row button, replaced while editing.
 * @param project The open project.
 */
function renameProjectInSidebar(button, project) {
  const input = document.createElement("input");
  input.type = "text";
  input.className = "conversation-rename";
  input.value = project.name;
  input.setAttribute("aria-label", `Rename ${project.name}`);
  let done = false;
  const finish = (save) => {
    if (done) return;
    done = true;
    const name = input.value.trim();
    input.replaceWith(button);
    if (save && name && name !== project.name && !busy && ready) {
      setBusy(true, "Renaming project…");
      bridge.renameProject(name);
    }
  };
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") { event.preventDefault(); finish(true); }
    if (event.key === "Escape") { event.preventDefault(); finish(false); }
  });
  input.addEventListener("blur", () => finish(true));
  button.replaceWith(input);
  input.focus();
  input.select();
}

/**
 * Show the ⋯ menu of a left-panel section: sort by last update, or in the
 * user's own dragged order.
 *
 * @param anchor The ⋯ button.
 * @param section "projects" or "conversations".
 * @param refresh Redraws the section's list.
 */
function showSidebarSortMenu(anchor, section, refresh) {
  const mode = sidebarPref(`${section}-sort`, "updated");
  const heading = document.createElement("div");
  heading.className = "menu-heading";
  heading.textContent = "Sort by";
  const options = [["updated", "Last updated"], ["manual", "Manual order"]].map(([value, label]) => {
    const button = document.createElement("button");
    button.type = "button";
    button.setAttribute("role", "menuitemradio");
    button.setAttribute("aria-checked", String(value === mode));
    button.textContent = label;
    if (value === mode) {
      const check = document.createElement("span");
      check.className = "menu-check";
      check.textContent = "✓";
      button.append(check);
    }
    button.addEventListener("click", () => {
      hideSourceMenu();
      setSidebarPref(`${section}-sort`, value);
      refresh();
    });
    return button;
  });
  sourceMenu.replaceChildren(heading, ...options);
  sourceMenu.classList.remove("hidden");
  const box = anchor.getBoundingClientRect();
  const { width } = sourceMenu.getBoundingClientRect();
  sourceMenu.style.left = `${Math.max(8, Math.min(box.left, window.innerWidth - width - 8))}px`;
  sourceMenu.style.top = `${box.bottom + 4}px`;
}

/**
 * Check sources for an open project, or add them to a new project's preview.
 *
 * @param paths Local paths provided by a native dialog or drag-and-drop.
 */
function addSources(paths) {
  if (busy || !paths.length) return;
  if (ready) {
    // Check first: the dialog shows what would be added and left out.
    error.classList.add("hidden");
    setBusy(true, "Checking the sources…");
    bridge.checkProjectSources(JSON.stringify(paths));
    return;
  }
  for (const path of paths) {
    const clean = String(path).trim();
    if (clean && !selectedSources.includes(clean)) selectedSources.push(clean);
  }
  previewSources();
}

/**
 * Return nonempty, lowercase custom suffixes from the settings field.
 * @returns Nonempty custom suffix strings.
 */
function fileTypes() {
  return byId("extensions").value.split(",").map((value) => value.trim().toLowerCase()).filter(Boolean);
}

/**
 * List the file types the new project copies: its type's own and the extra
 * types typed in Settings. The preview offers only these, so every file the
 * user can tick is one project creation accepts.
 */
function projectFileTypes() {
  return [...new Set([...profile.extensions, ...fileTypes()])];
}

/**
 * Request a backend metadata scan for the current source selection.
 */
function previewSources() {
  if (bridge && selectedSources.length) {
    setBusy(true, "Scanning selected sources…");
    bridge.previewSources(
      JSON.stringify(selectedSources), JSON.stringify(projectFileTypes()),
      Number(byId("max-file-bytes").value), "",
    );
  }
}

/**
 * Add "(1 folder, 25 files)" after a workspace folder's name: what the
 * folder directly contains, not its nested levels.
 *
 * @param folder A tree-folder element built by branch().
 */
function showFolderCount(folder) {
  const summary = folder.querySelector(":scope > summary");
  const name = [...summary.childNodes].find((node) => node.nodeType === Node.TEXT_NODE);
  if (!name) return;
  const plural = (count, word) => `${count} ${word}${count === 1 ? "" : "s"}`;
  const folders = Number(folder.dataset.folders || 0);
  const files = Number(folder.dataset.files || 0);
  const parts = [
    folders ? plural(folders, "folder") : "",
    files ? plural(files, "file") : "",
  ].filter(Boolean);
  const count = document.createElement("span");
  count.className = "tree-count";
  count.textContent = ` (${parts.join(", ") || "empty"})`;
  name.after(count);
}

/**
 * Build a nested selection tree with tri-state folder checkboxes.
 *
 * @param name Displayed folder label.
 * @param children Tree node with directory children and file paths.
 * @param paths Descendant paths used to compute folder selection.
 * @param selection Mutable Set of checked paths.
 * @param depth Nesting depth; the top-level branch starts expanded.
 * @param enabled Allowed paths, or null to enable every path.
 * @param reasons Path-to-explanation mapping for skipped entries.
 * @param selectable True gives each row a checkbox (the source preview).
 *   False gives plain rows (the workspace tree), so clicking a file's
 *   name does not also click its delete button.
 * @returns New details element containing the branch.
 */
function branch(name, children, paths, selection, depth = 0, enabled = null, reasons = {}, selectable = true) {
  const details = document.createElement("details");
  details.className = "tree-folder";
  const key = `${name}|${paths[0] || ""}`;
  if (depth < 1 || expandedFolders.has(key)) details.open = true;
  details.addEventListener("toggle", () => {
    if (!details.isConnected) return;
    details.open ? expandedFolders.add(key) : expandedFolders.delete(key);
  });
  // Root-relative trees (the workspace) can recover the folder's own path.
  if (paths[0]) details.dataset.path = paths[0].split("/").slice(0, depth + 1).join("/");
  // Direct contents only (not nested levels), shown by the workspace tree.
  details.dataset.folders = String(children.directories.size);
  details.dataset.files = String(children.files.length);
  const summary = document.createElement("summary");
  if (selectable) {
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    const available = enabled ? paths.filter((path) => enabled.has(path)) : paths;
    checkbox.checked = available.length > 0 && available.every((path) => selection.has(path));
    checkbox.indeterminate = available.some((path) => selection.has(path)) && !checkbox.checked;
    checkbox.disabled = available.length === 0;
    checkbox.addEventListener("click", (event) => event.stopPropagation());
    checkbox.addEventListener("change", () => {
      available.forEach((path) => checkbox.checked ? selection.add(path) : selection.delete(path));
      renderPreview();
    });
    summary.append(checkbox);
  }
  summary.append(document.createTextNode(name));
  details.append(summary);
  const content = document.createElement("div");
  content.className = "tree-children";
  for (const [folder, node] of [...children.directories].sort()) {
    content.append(branch(folder, node, allFiles(node), selection, depth + 1, enabled, reasons, selectable));
  }
  for (const file of children.files.sort()) {
    // A label forwards clicks to its first control, so plain rows are divs.
    const row = document.createElement(selectable ? "label" : "div");
    row.className = "tree-file";
    row.dataset.path = file;
    if (selectable) {
      const check = document.createElement("input");
      check.type = "checkbox";
      check.checked = selection.has(file);
      check.disabled = enabled !== null && !enabled.has(file);
      check.addEventListener("change", () => {
        check.checked ? selection.add(file) : selection.delete(file);
        renderPreview();
      });
      row.append(check);
    }
    row.append(document.createTextNode(file.split("/").pop()));
    if (reasons[file]) {
      const reason = document.createElement("small");
      reason.className = "skip-reason";
      reason.textContent = reasons[file];
      row.append(reason);
    }
    content.append(row);
  }
  details.append(content);
  return details;
}

/**
 * Flatten a tree node into its descendant file paths.
 *
 * @param node Node containing files and a Map of directories.
 * @returns Flat array of descendant paths.
 */
function allFiles(node) {
  return [...node.files, ...[...node.directories.values()].flatMap(allFiles)];
}

/**
 * Group forward-slash-separated file paths into directory nodes.
 *
 * @param files Relative file paths to group; the input is not modified.
 * @returns Root tree node.
 */
function fileTree(files) {
  const root = { directories: new Map(), files: [] };
  for (const file of files) {
    // Trailing slashes represent real empty or nonempty folders, not files.
    const directory = file.endsWith("/");
    const parts = file.split("/").filter(Boolean);
    let node = root;
    const folders = directory ? parts : parts.slice(0, -1);
    for (const folder of folders) {
      if (!node.directories.has(folder)) node.directories.set(folder, { directories: new Map(), files: [] });
      node = node.directories.get(folder);
    }
    if (!directory) node.files.push(file);
  }
  return root;
}

/**
 * Identify files that may be included by explicitly raising the size limit.
 *
 * @param item Skipped-file preview row with a reason code.
 * @returns Whether explicit selection can override the size limit.
 */
function isOversizedSkippedFile(item) {
  return selectionReasonCodes !== null
    && item.reason_code === selectionReasonCodes.FILE_TOO_LARGE;
}

/**
 * Say briefly why a preview file cannot be ticked.
 *
 * @param item Skipped-file preview row with a reason code.
 * @returns Short text shown next to the file.
 */
function skippedReason(item) {
  if (isOversizedSkippedFile(item)) return "Above current limit · check to include";
  if (selectionReasonCodes !== null
      && item.reason_code === selectionReasonCodes.FILE_TYPE_NOT_SELECTED) {
    return "File type not included";
  }
  return item.reason;
}

/**
 * Rebuild source trees, counts, and removal controls from preview state.
 */
function renderPreview() {
  const tree = byId("file-tree");
  tree.replaceChildren();
  let total = 0;
  let selected = 0;
  let unavailable = 0;
  for (const rootPath of selectedSources) {
    const files = previewRoots.get(rootPath) || [];
    const skipped = previewSkipped.get(rootPath) || [];
    const oversized = skipped.filter((item) => isOversizedSkippedFile(item) && Number(item.size_bytes) > 0);
    const offered = new Set([...files, ...oversized.map((item) => item.path)]);
    const chosen = checkedFiles.get(rootPath) || new Set();
    total += files.length + skipped.length;
    unavailable += skipped.length - oversized.length;
    selected += [...chosen].filter((file) => offered.has(file)).length;
    const holder = document.createElement("div");
    holder.className = "source-root";
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "source-remove";
    remove.textContent = "Remove";
    remove.addEventListener("click", () => {
      selectedSources.splice(selectedSources.indexOf(rootPath), 1);
      previewRoots.delete(rootPath);
      previewSkipped.delete(rootPath);
      checkedFiles.delete(rootPath);
      renderPreview();
    });
    const reasons = Object.fromEntries(skipped.map((item) => [item.path, skippedReason(item)]));
    const all = [...files, ...skipped.map((item) => item.path)];
    holder.append(branch(rootPath, fileTree(all), all, chosen, 0, offered, reasons), remove);
    tree.append(holder);
  }
  byId("scope-count").textContent = `${selected} of ${total} files selected`
    + (unavailable ? ` · ${unavailable} unavailable` : "");
  byId("scope").classList.toggle("hidden", selectedSources.length === 0);
}

/**
 * Retain the current backend model fields in the local profile object.
 */
function saveModelFields() {
  if (!profile) return;
  profile[`chat_${activeBackend}`] = byId("chat-model").value.trim();
  profile[`embedding_${activeBackend}`] = byId("embedding-model").value.trim();
}

/**
 * Filter presets by role, backend, and any required query format.
 *
 * @param role Model role to show: chat or embedding.
 * @returns Matching model-preset records.
 */
function compatibleChoices(role) {
  // Chat and embedding models need not share a family. Each must support its
  // own role, the chosen backend, and (for embedding) the query format.
  return modelChoices.filter((choice) => choice.role === role
    && choice.backend === activeBackend
    && (!choice.query_format || choice.query_format === byId("query-format").value));
}

/**
 * Show whether a compatible cached chat-and-embedding pair is known.
 */
function refreshModelStatus() {
  const hasBoth = ["chat", "embedding"].every((role) =>
    compatibleChoices(role).some((choice) => cachedModels.has(choice.model_id)));
  byId("models-hint").classList.add("hidden");
  byId("open-models").classList.toggle("models-needed", !hasBoth);
}

const DEFAULT_LEVEL = 2;
const MAX_SEARCH_DEPTH = 200;

const cloneLevels = (levels) => (levels || []).map((level) => ({ ...level }));

/**
 * Return the saved effort levels for the active backend.
 *
 * @returns Five {name, model} records, or none before settings load.
 */
function currentLevels() {
  return savedLevels[activeBackend] || defaultLevels[activeBackend] || [];
}

/**
 * Return the saved search-depth steps (passages per step).
 */
function currentDepth() {
  return savedDepth || defaultDepth;
}

/**
 * Return the saved intent router model for the active backend.
 */
function currentRouter() {
  return savedRouters[activeBackend] || defaultRouters[activeBackend] || "";
}

/**
 * Report a cached model's role: chat, embedding, or other.
 *
 * @param id Model ID.
 * @returns Role from the cache scan, the preset catalog, or the name.
 */
function modelRole(id) {
  if (modelRoles[id]) return modelRoles[id];
  const preset = modelChoices.find((choice) => choice.model_id === id);
  if (preset) return preset.role;
  return /embed/i.test(id) ? "embedding" : "other";
}

/**
 * List dropdown choices: cached models of a role plus the current value.
 *
 * @param role chat or embedding.
 * @param current Model already chosen, kept even when not downloaded.
 * @returns Sorted model IDs.
 */
function roleModelIds(role, current) {
  const ids = [...cachedModels].filter((id) => modelRole(id) === role);
  if (current && !ids.includes(current)) ids.push(current);
  return ids.sort((left, right) => left.localeCompare(right));
}

/**
 * Build a model dropdown; models that are not downloaded are marked.
 * @param ids Model IDs offered in the dropdown.
 * @param value Currently chosen model ID.
 * @param label Accessible label of the model selector.
 * @param markThinking Whether to add the template-switch probe label.
 */
function modelSelect(ids, value, label, markThinking = false) {
  const select = document.createElement("select");
  select.className = "model-select";
  select.setAttribute("aria-label", label);
  for (const id of ids) {
    const option = document.createElement("option");
    option.value = id;
    const name = markThinking ? `${id}${thinkingMark(id)}` : id;
    option.textContent = cachedModels.has(id) ? name : `${name} (not downloaded)`;
    select.append(option);
  }
  select.value = value;
  select.disabled = busy;
  return select;
}

/**
 * Check whether a cached model's template rendered different prompts
 * with enable_thinking on and off. This does not test model reasoning.
 *
 * Missing models or templates that could not be checked are unknown.
 *
 * @param id Model identifier or local path.
 * @returns True only for a detected template prompt difference.
 */
function modelHasThinkingSwitch(id) {
  return thinkingSwitches[id] === true;
}

/**
 * Label the template probe outcome, or return "" when unknown.
 *
 * @param id Model identifier or local path.
 * @returns A template-switch label, or "" when no result is available.
 */
function thinkingMark(id) {
  if (thinkingSwitches[id] === true) {
    return " · thinking switch detected";
  }
  return thinkingSwitches[id] === false
    ? " · no template switch detected" : "";
}

/**
 * Show a Download button under a row whose model is not cached.
 * @param id Model identifier shown by this row.
 * @param installed Whether the chosen backend is installed.
 */
function downloadNotice(id, installed) {
  if (!id || cachedModels.has(id) || id.startsWith("/") || id.startsWith(".")) return null;
  const notice = document.createElement("div");
  notice.className = "model-missing";
  const text = document.createElement("small");
  text.textContent = "Not downloaded";
  const download = document.createElement("button");
  download.type = "button";
  download.className = "primary";
  download.textContent = "Download";
  download.disabled = busy || !installed;
  // An explicit Download click authorizes this download. On the start
  // screen it only downloads: nothing is saved before a project exists.
  download.addEventListener("click", () => (ready ? applyModelEdits(false, id) : downloadModelOnly(id)));
  notice.append(text, download);
  return notice;
}

/**
 * Build a model group heading with its Restore defaults button.
 * @param role Model role whose defaults can be restored.
 * @param title Visible model-group heading.
 */
function modelGroupHeader(role, title) {
  const header = document.createElement("div");
  header.className = "model-group-header";
  const heading = document.createElement("h3");
  heading.textContent = title;
  const restore = document.createElement("button");
  restore.type = "button";
  restore.className = "secondary restore-defaults";
  restore.textContent = "Restore defaults";
  restore.setAttribute("aria-label", `Restore default ${title.toLowerCase()}`);
  restore.dataset.role = role;
  restore.disabled = busy || !hasModelChanges(role);
  restore.addEventListener("click", () => restoreDefaultModels(role));
  header.append(heading, restore);
  return header;
}

/**
 * Remember a field's value when the user enters it, so a refused edit can
 * be put back.
 *
 * @param input Text or number field.
 */
function rememberValue(input) {
  input.addEventListener("focus", () => { input.dataset.before = input.value; });
}

/**
 * Check a start-screen edit at once, as Prepare project would: a refused
 * value is put back and explained; nothing is saved before the project
 * exists.
 *
 * @param input The edited field, if known.
 * @param problem Why the values are refused, or "".
 */
function keepNewProjectValue(input, problem) {
  if (problem) {
    if (input && input.dataset.before !== undefined) input.value = input.dataset.before;
    byId("model-download-status").textContent = problem;
  } else if (input && input.value !== input.dataset.before) {
    byId("model-download-status").textContent = "Kept for the new project.";
  }
}

/**
 * Save a typed field when the user presses Enter or leaves it, not on each
 * key or number-arrow click.
 *
 * @param input Text or number field.
 * @param save Function that saves the field's change.
 */
function saveOnEnter(input, save) {
  rememberValue(input);
  input.addEventListener("blur", save);
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") { event.preventDefault(); event.target.blur(); }
  });
}

/**
 * Build the effort-level rows, the embedding choice, and the cache listing.
 */
function renderModelsPanel() {
  const host = byId("model-list");
  const focused = document.activeElement?.closest?.("[data-focus-key]")?.dataset.focusKey;
  host.replaceChildren();
  const depthHost = byId("search-depth");
  depthHost.replaceChildren();
  const installed = installedBackends[activeBackend];
  const mlx = activeBackend === "mlx";
  // Only a missing backend needs a note above the levels.
  const repair = installCommands[activeBackend] || "the KíLín installer";
  byId("models-backend").textContent = installed ? ""
    : `${activeBackend.toUpperCase()} packages are not installed. Install them with ${repair}, then restart KíLín.`;
  byId("models-backend").classList.toggle("hidden", Boolean(installed));

  const chat = document.createElement("section");
  chat.className = "model-group";
  chat.append(modelGroupHeader("chat", "Effort levels"));
  if (!mlx) {
    // PyTorch cannot switch models per question.
    const hint = document.createElement("p");
    hint.className = "hint";
    hint.textContent = "PyTorch answers every level with one model.";
    chat.append(hint);
  }
  const columns = document.createElement("div");
  columns.className = "level-row level-columns";
  columns.setAttribute("aria-hidden", "true");
  for (const label of ["Level", "Language model"]) {
    const cell = document.createElement("span");
    cell.textContent = label;
    columns.append(cell);
  }
  chat.append(columns);
  draftLevels.forEach((level, index) => {
    const row = document.createElement("div");
    row.className = "level-row";
    const name = document.createElement("input");
    name.type = "text";
    name.className = "level-name";
    name.maxLength = 24;
    name.value = level.name;
    name.disabled = busy;
    name.dataset.focusKey = `name-${index}`;
    name.setAttribute("aria-label", `Name of effort level ${index + 1}`);
    name.addEventListener("input", () => {
      level.name = name.value;
      refreshRestoreButtons();
    });
    saveOnEnter(name, saveModelEdits);
    // Answers think only with these models; routing never does.
    const select = modelSelect(roleModelIds("chat", level.model), level.model,
      `Language model for ${level.name || `level ${index + 1}`}`, true);
    select.dataset.focusKey = `model-${index}`;
    select.addEventListener("change", () => {
      // Other backends cannot switch models per level: one model for all.
      for (const target of mlx ? [level] : draftLevels) target.model = select.value;
      byId("chat-model").value = draftLevels[DEFAULT_LEVEL].model;
      renderModelsPanel();
      saveModelEdits();
    });
    row.append(name, select);
    chat.append(row);
    const notice = downloadNotice(level.model, installed);
    if (notice) chat.append(notice);
  });
  host.append(chat);

  // Search depth: how many passages each step of the composer's second
  // slider retrieves. Shown in Answers; applied with the model changes.
  const depth = document.createElement("section");
  depth.className = "model-group";
  depth.append(modelGroupHeader("depth", "Search depth"));
  const steps = document.createElement("div");
  steps.className = "depth-row";
  draftDepth.forEach((value, index) => {
    const cell = document.createElement("label");
    cell.className = "depth-step";
    const caption = document.createElement("span");
    caption.textContent = `Step ${index + 1}`;
    const input = document.createElement("input");
    input.type = "number";
    input.min = "1";
    input.max = String(MAX_SEARCH_DEPTH);
    input.step = "1";
    input.value = String(value);
    input.disabled = busy;
    input.dataset.focusKey = `depth-${index}`;
    input.setAttribute("aria-label", `Search depth step ${index + 1}, in passages`);
    input.addEventListener("input", () => {
      draftDepth[index] = Number(input.value);
      refreshRestoreButtons();
    });
    saveOnEnter(input, saveModelEdits);
    cell.append(caption, input);
    steps.append(cell);
  });
  depth.append(steps);
  depthHost.append(depth);

  const router = document.createElement("section");
  router.className = "model-group";
  router.append(modelGroupHeader("router", "Intent model"));
  const routerRow = document.createElement("div");
  routerRow.className = "model-row";
  // Marked like the levels; picking the intent runs with thinking off.
  const routerSelect = modelSelect(roleModelIds("chat", draftRouter), draftRouter, "Intent model", true);
  routerSelect.dataset.focusKey = "router";
  routerSelect.addEventListener("change", () => {
    draftRouter = routerSelect.value;
    renderModelsPanel();
    saveModelEdits();
  });
  routerRow.append(routerSelect);
  router.append(routerRow);
  const routerNotice = downloadNotice(draftRouter, installed);
  if (routerNotice) router.append(routerNotice);
  host.append(router);

  const embedding = document.createElement("section");
  embedding.className = "model-group";
  embedding.append(modelGroupHeader("embedding", "Embedding model"));
  const current = byId("embedding-model").value.trim();
  const row = document.createElement("div");
  row.className = "model-row";
  const select = modelSelect(roleModelIds("embedding", current), current, "Embedding model");
  select.dataset.focusKey = "embedding";
  select.addEventListener("change", () => {
    byId("embedding-model").value = select.value;
    renderModelsPanel();
    saveModelEdits();
  });
  row.append(select);
  embedding.append(row);
  const notice = downloadNotice(current, installed);
  if (notice) embedding.append(notice);
  host.append(embedding);
  if (focused) {
    (host.querySelector(`[data-focus-key="${focused}"]`)
      || depthHost.querySelector(`[data-focus-key="${focused}"]`))?.focus();
  }
}

/**
 * Enable each Restore defaults button only when it would change something.
 */
function refreshRestoreButtons() {
  for (const button of document.querySelectorAll("#model-list .restore-defaults, #search-depth .restore-defaults")) {
    button.disabled = busy || !hasModelChanges(button.dataset.role);
  }
}

/**
 * Return the built-in embedding model for the active backend.
 */
function defaultEmbeddingModel() {
  // From defaults.json (sent by the engine); the catalog is a fallback.
  return defaultEmbeddings[activeBackend]
    || compatibleChoices("embedding")[0]?.model_id
    || backendChoices("embedding")[0]?.model_id || "";
}

/**
 * Report whether a group differs from its built-in defaults.
 *
 * @param role chat (effort levels), depth (search depth), router, or
 *   embedding.
 * @returns True when restoring defaults would change something.
 */
function hasModelChanges(role) {
  if (role === "chat") {
    return JSON.stringify(draftLevels) !== JSON.stringify(defaultLevels[activeBackend] || []);
  }
  if (role === "depth") return JSON.stringify(draftDepth) !== JSON.stringify(defaultDepth);
  if (role === "router") return draftRouter !== (defaultRouters[activeBackend] || "");
  return byId("embedding-model").value.trim() !== defaultEmbeddingModel()
    || byId("query-format").value !== "qwen3";
}

/**
 * Return a role's presets for the active backend, ignoring query format.
 *
 * @param role Model role: chat or embedding.
 * @returns Matching model-preset records.
 */
function backendChoices(role) {
  return modelChoices.filter((choice) =>
    choice.role === role && choice.backend === activeBackend);
}

/**
 * Reset a group to its built-in values; Close applies the change.
 *
 * @param role chat (effort levels) or embedding.
 */
function restoreDefaultModels(role) {
  if (busy) return;
  if (role === "chat") {
    draftLevels = cloneLevels(defaultLevels[activeBackend]);
    if (draftLevels.length) byId("chat-model").value = draftLevels[DEFAULT_LEVEL].model;
  } else if (role === "depth") {
    draftDepth = [...defaultDepth];
  } else if (role === "router") {
    draftRouter = defaultRouters[activeBackend] || draftRouter;
  } else {
    byId("query-format").value = "qwen3";
    byId("embedding-model").value = defaultEmbeddingModel();
  }
  refreshModelStatus();
  renderModelsPanel();
  saveModelEdits();
}

/**
 * Show one Settings tab and hide the sections of the others.
 *
 * @param name Tab name: project, models, answers, or tools.
 */
function showSettingsTab(name) {
  settingsTab = name;
  // The open project's tools, read fresh each time the tab is shown.
  if (name === "tools") requestToolSettings();
  for (const tab of document.querySelectorAll(".settings-tab")) {
    const selected = tab.dataset.tab === name;
    tab.classList.toggle("active", selected);
    tab.setAttribute("aria-selected", String(selected));
    // Arrow keys move between tabs; Tab moves into the shown one.
    tab.tabIndex = selected ? 0 : -1;
  }
  for (const section of document.querySelectorAll("#models-panel [data-settings-tab]")) {
    section.classList.toggle("tab-hidden", section.dataset.settingsTab !== name);
  }
}

/**
 * Ask for the open project's tools, or say that a project must be open.
 */
function requestToolSettings() {
  if (bridge && openProjectPath) {
    bridge.requestToolSettings();
    return;
  }
  const hint = document.createElement("p");
  hint.className = "hint";
  hint.textContent = "Open a project to see its tools.";
  byId("tool-settings").replaceChildren(hint);
}

/**
 * Turn a setting name such as "max_model_calls" into "Max model calls".
 *
 * @param key Setting name from a tool's defaults.json.
 * @returns Label for the setting.
 */
function settingLabel(key) {
  const words = key.replaceAll("_", " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}

/**
 * Build the field for one tool setting, chosen by its value's type.
 *
 * @param key Setting name.
 * @param value Current value: a Boolean, number, string, or list.
 * @returns Label element holding the field.
 */
function toolSettingField(key, value) {
  const label = document.createElement("label");
  const input = document.createElement("input");
  input.dataset.key = key;
  if (typeof value === "boolean") {
    label.className = "check-row";
    input.type = "checkbox";
    input.checked = value;
    input.dataset.kind = "boolean";
    label.append(input, ` ${settingLabel(key)}`);
    return label;
  }
  if (typeof value === "number") {
    input.type = "number";
    input.step = Number.isInteger(value) ? "1" : "any";
    input.value = String(value);
    input.dataset.kind = "number";
  } else if (Array.isArray(value)) {
    input.type = "text";
    input.value = value.join(", ");
    input.placeholder = "Comma-separated";
    input.dataset.kind = "list";
  } else {
    input.type = "text";
    input.value = String(value ?? "");
    input.dataset.kind = "text";
  }
  label.append(settingLabel(key), input);
  return label;
}

/**
 * Read a tool's fields back into setting values.
 *
 * @param fields Container of the tool's setting fields.
 * @returns Setting names and values to save.
 */
function toolSettingValues(fields) {
  const values = {};
  for (const input of fields.querySelectorAll("input[data-key]")) {
    const kind = input.dataset.kind;
    values[input.dataset.key] = kind === "boolean" ? input.checked
      : kind === "number" ? Number(input.value)
        : kind === "list" ? input.value.split(",").map((item) => item.trim()).filter(Boolean)
          : input.value;
  }
  return values;
}

/**
 * Show the project's tools, each with its settings built from the tool's
 * defaults.json as fields.
 *
 * @param data The ``tool_settings`` event: ``tools``, and for a save the
 *   ``tool`` with ``saved`` or ``error``.
 */
function renderToolSettings(data) {
  const holder = byId("tool-settings");
  const open = new Set([...holder.querySelectorAll("details.tool-settings[open]")].map((item) => item.dataset.tool));
  if (data.tool) open.add(data.tool);
  // Saving redraws the list; keep the field the user moved to.
  const focused = document.activeElement?.closest?.("#tool-settings") && document.activeElement.dataset.key
    ? [document.activeElement.closest("details.tool-settings").dataset.tool, document.activeElement.dataset.key]
    : null;
  holder.replaceChildren();
  if (!data.tools.length) {
    const hint = document.createElement("p");
    hint.className = "hint";
    hint.textContent = "This project type uses no tools.";
    holder.append(hint);
    return;
  }
  for (const tool of data.tools) {
    const item = document.createElement("details");
    item.className = "tool-settings";
    item.dataset.tool = tool.name;
    item.open = open.has(tool.name);
    const summary = document.createElement("summary");
    const name = document.createElement("code");
    name.textContent = tool.name;
    summary.append(name, ` ${tool.summary}`);
    const fields = document.createElement("div");
    fields.className = "tool-fields";
    // Values first, then the on/off switches together, one per row.
    const entries = Object.entries(tool.settings);
    for (const [key, value] of entries) if (typeof value !== "boolean") fields.append(toolSettingField(key, value));
    for (const [key, value] of entries) if (typeof value === "boolean") fields.append(toolSettingField(key, value));
    item.append(summary, fields);
    // Settings each use sets differently, such as a template, stay as set.
    const perIntent = Object.entries(tool.per_intent);
    if (perIntent.length) {
      const fixed = document.createElement("p");
      fixed.className = "hint";
      fixed.textContent = `Set by each use: ${perIntent.map(([intent, values]) =>
        `${settingLabel(intent)} (${Object.entries(values).map(([key, value]) => `${key} ${value}`).join(", ")})`).join("; ")}`;
      item.append(fixed);
    }
    const status = document.createElement("p");
    status.className = "hint tool-status";
    status.setAttribute("role", "status");
    if (data.tool === tool.name) status.textContent = data.error || (data.saved ? "Saved." : "");
    status.classList.toggle("tool-error", Boolean(data.tool === tool.name && data.error));
    // Like the other settings, a change is saved as soon as it is made.
    fields.addEventListener("change", () => {
      status.textContent = "Saving…";
      status.classList.remove("tool-error");
      bridge.saveToolSettings(tool.name, JSON.stringify(toolSettingValues(fields)));
    });
    const actions = document.createElement("div");
    actions.className = "tool-actions";
    const restore = document.createElement("button");
    restore.type = "button";
    restore.className = "secondary restore-defaults";
    restore.textContent = "Restore defaults";
    restore.setAttribute("aria-label", `Restore the default ${tool.name} settings`);
    restore.disabled = JSON.stringify(tool.settings) === JSON.stringify(tool.type_settings);
    restore.addEventListener("click", () => {
      status.textContent = "Saving…";
      bridge.saveToolSettings(tool.name, JSON.stringify(tool.type_settings));
    });
    actions.append(restore);
    item.append(actions, status);
    holder.append(item);
  }
  if (focused) holder.querySelector(`details[data-tool="${focused[0]}"] input[data-key="${focused[1]}"]`)?.focus();
}

/**
 * Render and open the models panel, then focus its close control.
 */
function openModels() {
  if (byId("models-panel").classList.contains("hidden")) {
    clearSettingsStatus();
    // An open project's saved values, not an edit left unsaved earlier.
    if (ready && profile) showIndexingFields();
    draftLevels = cloneLevels(currentLevels());
    draftDepth = [...currentDepth()];
    draftRouter = currentRouter();
    modelsBaseline = currentModelSettings();
  }
  renderModelsPanel();
  showProjectSettings();
  // The Tools tab may still be selected from another project.
  if (settingsTab === "tools") requestToolSettings();
  byId("models-panel").classList.remove("hidden");
  byId("close-models").focus();
}

/**
 * Clear the message under the tabs, so it only ever answers the last
 * action; one still in progress (a download or a rebuild) is kept.
 */
function clearSettingsStatus() {
  if (downloadingModel || applyingModels || applyingIndexing) return;
  byId("model-download-status").textContent = "";
}

/**
 * Collect the unsaved model, router, and retrieval choices.
 *
 * @returns Current settings object to send to the engine.
 */
function currentModelSettings() {
  return {
    chat_model: draftLevels[DEFAULT_LEVEL]?.model || byId("chat-model").value.trim(),
    embedding_model: byId("embedding-model").value.trim(),
    query_format: byId("query-format").value,
    backend: activeBackend,
    effort_levels: cloneLevels(draftLevels),
    search_depth: [...draftDepth],
    router_model: draftRouter,
  };
}

/**
 * Report whether Settings hold edits that are not saved yet.
 */
function modelEditsPending() {
  const settings = currentModelSettings();
  const before = modelsBaseline || settings;
  return JSON.stringify(settings.effort_levels) !== JSON.stringify(before.effort_levels)
    || JSON.stringify(settings.search_depth) !== JSON.stringify(before.search_depth)
    || settings.embedding_model !== before.embedding_model || settings.query_format !== before.query_format
    || settings.router_model !== before.router_model;
}
/**
 * Close the model settings panel and refresh source-drop bounds.
 */
function hideModels() {
  setupCompletionVisible = false;
  byId("models-panel").classList.add("hidden");
  syncSourceDropTarget();
}

/**
 * Explain the first invalid effort level, or return an empty string.
 * @param levels Current draft effort-level settings.
 * @param depth Current draft retrieval-depth steps.
 */
function levelProblem(levels, depth) {
  for (const [index, level] of levels.entries()) {
    if (!level.name.trim()) return `Give effort level ${index + 1} a name.`;
  }
  for (const [index, value] of depth.entries()) {
    if (!Number.isInteger(value) || value < 1 || value > MAX_SEARCH_DEPTH) {
      return `Search depth step ${index + 1} must be a whole number from 1 to ${MAX_SEARCH_DEPTH}.`;
    }
  }
  return "";
}

/**
 * Validate pending model changes and request that the engine apply them.
 *
 * @param closeAfter Whether to close Settings after applying the changes.
 * @param explicitDownload Model ID to download, or null for ordinary edits.
 */
async function applyModelEdits(closeAfter = false, explicitDownload = null) {
  if (busy || !bridge) return false;
  const settings = currentModelSettings();
  const before = modelsBaseline || settings;
  const changed = modelEditsPending();
  if (!changed && !explicitDownload) { if (closeAfter) hideModels(); return false; }
  const problem = levelProblem(settings.effort_levels, settings.search_depth);
  if (problem) {
    byId("model-download-status").textContent = problem;
    return false;
  }
  for (const level of settings.effort_levels) level.name = level.name.trim();
  if (!settings.chat_model || !settings.embedding_model) {
    byId("model-download-status").textContent = "Choose language and embedding models.";
    return false;
  }
  const requested = [...new Set([settings.embedding_model,
    ...settings.effort_levels.map((level) => level.model),
    ...(settings.router_model ? [settings.router_model] : [])])];
  const missing = requested.filter((id) => !cachedModels.has(id)
    && !id.startsWith("/") && !id.startsWith("."));
  const reindex = ready && (settings.embedding_model !== before.embedding_model
    || settings.query_format !== before.query_format);
  const warnings = [];
  if (missing.some((id) => id !== explicitDownload)) warnings.push(`KíLín downloads ${missing.join(", ")}, which may take a while.`);
  if (reindex) warnings.push("KíLín rebuilds this project's index; saved chats are kept.");
  if (warnings.length && (!await askKilin("Apply model settings?", warnings.join(" "), "Apply") || busy)) return false;
  applyingModels = true;
  closeAfterModelsApply = closeAfter;
  error.classList.add("hidden");
  byId("model-download-status").textContent = reindex ? "Applying models and rebuilding the index…" : "Applying model settings…";
  setBusy(true, byId("model-download-status").textContent);
  bridge.applyModelSettings(JSON.stringify({ ...settings, download: missing.length > 0 }));
  return true;
}

/**
 * Save a Models or Search depth change as soon as it is made, so Settings
 * always shows what is in effect. A refused or cancelled change (for
 * example a download the user declines) puts the saved values back.
 */
async function saveModelEdits() {
  if (!modelEditsPending()) return;
  if (!ready) return keepNewProjectModels();
  if (busy) {
    byId("model-download-status").textContent = "KíLín is busy. Change this again when it finishes.";
  } else if (await applyModelEdits()) {
    return;
  }
  const problem = byId("model-download-status").textContent;
  showSavedModels();
  byId("model-download-status").textContent = problem;
}

/**
 * Keep start-screen model choices on the page for the next new project,
 * which saves them in its own profile.ini. Nothing is written before a
 * project exists, so the next new project starts from the defaults again.
 */
function keepNewProjectModels() {
  const problem = levelProblem(draftLevels, draftDepth);
  if (problem) {
    showSavedModels();
    byId("model-download-status").textContent = problem;
    return;
  }
  savedLevels = { ...savedLevels, [activeBackend]: cloneLevels(draftLevels) };
  savedRouters = { ...savedRouters, [activeBackend]: draftRouter };
  savedDepth = [...draftDepth];
  saveModelFields();
  modelsBaseline = currentModelSettings();
  refreshRestoreButtons();
  byId("model-download-status").textContent = "Kept for the new project.";
}

/**
 * Download one model from the start screen, saving no settings.
 *
 * @param id Model identifier to download.
 */
function downloadModelOnly(id) {
  if (busy || !bridge) return;
  downloadingModel = id;
  setBusy(true, `Downloading ${id}…`);
  bridge.downloadModel(id);
}

/**
 * Put the saved model choices and search depth back into Settings.
 */
function showSavedModels() {
  draftLevels = cloneLevels(currentLevels());
  draftDepth = [...currentDepth()];
  draftRouter = currentRouter();
  if (modelsBaseline) {
    byId("embedding-model").value = modelsBaseline.embedding_model;
    byId("query-format").value = modelsBaseline.query_format;
  }
  if (draftLevels.length) byId("chat-model").value = draftLevels[DEFAULT_LEVEL].model;
  renderModelsPanel();
}

/**
 * Update one required model's line in the KíLín setup steps.
 *
 * @param modelId Model whose line to update; unknown models are ignored.
 * @param text New text, such as "Downloading: <model> · 40%".
 * @param complete Whether the model is now ready.
 */
function showModelStep(modelId, text, complete = false) {
  const item = [...byId("readiness-steps").children]
    .find((step) => step.dataset.model === modelId);
  if (!item) return;
  item.textContent = text;
  item.classList.toggle("complete", complete);
}

/**
 * Show the terms of use in KíLín setup, one paragraph per blank-line block.
 *
 * @param text Terms of use from the readiness check.
 */
function renderTerms(text) {
  const host = byId("terms-text");
  // Readiness is checked often; keep the reader's place in unchanged terms.
  if (host.dataset.shown === text) return;
  host.dataset.shown = text;
  host.replaceChildren(...text.split(/\n\s*\n/).filter(Boolean).map((block) => {
    const paragraph = document.createElement("p");
    // A short lead such as "Research software." is the paragraph's title.
    const [, lead, rest] = block.trim().match(/^([^.]{1,40}\.)\s+(.*)$/s) || [];
    if (lead && lead.split(" ").length <= 5) {
      const title = document.createElement("strong");
      title.textContent = lead;
      paragraph.append(title, ` ${rest}`);
    } else {
      paragraph.textContent = block.trim();
    }
    return paragraph;
  }));
}

/**
 * Show the welcome dialog with the terms of use. Until they are accepted it
 * cannot be dismissed; afterwards it only shows them again.
 */
function showWelcome() {
  const dialog = byId("welcome");
  // Do not erase a checked agreement after a failed save: retry is allowed.
  if (termsAccepted) byId("terms-agree").checked = true;
  byId("terms-agree").closest("label").classList.toggle("hidden", termsAccepted);
  byId("welcome-hint").classList.add("hidden");
  showSetupConfirmations();
  if (dialog.open) return;
  dialog.showModal();
  // Opening focuses "Read the full terms of use"; the dialog takes it.
  dialog.focus?.();
}

/**
 * Display a recoverable save error inside the non-dismissible terms dialog.
 *
 * @param message Message to display, or an empty string to clear it.
 */
function setWelcomeError(message) {
  const host = byId("welcome-error");
  host.textContent = message;
  host.classList.toggle("hidden", !message);
}

/**
 * Answer an attempt to leave the welcome dialog before the terms are
 * accepted: scroll to its end and say what is still needed there.
 */
function remindWelcome() {
  const dialog = byId("welcome");
  const hint = byId("welcome-hint");
  hint.classList.remove("hidden");
  dialog.scrollTo?.({ top: dialog.scrollHeight, behavior: "smooth" });
  byId("terms-agree").focus?.({ preventScroll: true });
  drawAttention(hint);
}

/**
 * Show Confirm projects folder until it is done, and enable Continue in the
 * welcome dialog. The folder is confirmed only on the start screen.
 */
function showSetupConfirmations() {
  const confirm = byId("confirm-workspace");
  confirm.classList.toggle("hidden", folderConfirmed);
  byId("confirm-workspace-note").classList.toggle("hidden", folderConfirmed);
  confirm.disabled = busy || ready || !bridge;
  confirm.title = ready ? "Confirm it from the start screen." : "";
  // Continue needs the box ticked; Close needs no additional confirmation.
  const continueButton = byId("accept-terms");
  continueButton.textContent = termsAccepted ? "Close"
    : acceptingTerms ? "Saving agreement…" : "Continue";
  continueButton.disabled = !termsAccepted
    && (acceptingTerms || !bridge || !termsVersion || !byId("terms-agree").checked);
}

/**
 * Show the setup buttons, Close and Prepare project for the current setup
 * state: while the required models download, Install becomes Stop and
 * new-project preparation waits for setup. Settings can always close.
 */
function showSetupLock() {
  const install = byId("install-required");
  install.textContent = installingModels ? "Stop downloading" : "Install required models";
  install.disabled = installingModels ? false : busy || !readinessEnvironmentReady || !readinessMissingCount;
  byId("use-installed").disabled = busy || !readinessEnvironmentReady;
  byId("use-legacy-workspace").disabled = busy || ready;
  byId("open").disabled = busy || !bridge || !kilinReady;
  byId("close-models").disabled = false;
  byId("close-models").title = "";
  showSetupConfirmations();
}

/**
 * Require the selected default backend and models only for new projects.
 *
 * @returns Whether new-project setup still needs to be finished.
 */
function needsNewProjectSetup() {
  if (kilinReady) return false;
  openModels();
  byId("first-run-setup").open = true;
  byId("model-download-status").textContent = "Finish KíLín setup before preparing a new project.";
  return true;
}

/**
 * Keep Settings open and point to the projects folder or terms of use that
 * KíLín setup still needs.
 */
function remindSetupConfirmations() {
  byId("first-run-setup").open = true;
  if (!termsAccepted) showWelcome();
  else pointToProjectsFolder();
  byId("model-download-status").textContent = !folderConfirmed && !termsAccepted
    ? "Confirm the projects folder and accept the terms of use to continue."
    : !folderConfirmed
      ? "Confirm the projects folder to continue."
      : "Accept the terms of use to continue.";
  shownSetupReminder = byId("model-download-status").textContent;
  // The terms have their own reminder in the welcome dialog.
  if (termsAccepted) drawAttention(byId("model-download-status"));
}

/**
 * Remove the setup reminder and its highlights once it no longer applies.
 *
 * The status line is cleared only while it still shows the reminder, so a
 * later message there (such as download progress) stays.
 */
function clearSetupReminder() {
  const status = byId("model-download-status");
  if (shownSetupReminder && status.textContent === shownSetupReminder) {
    status.textContent = "";
  }
  shownSetupReminder = "";
  status.classList.remove("attention");
  byId("confirm-workspace").classList.remove("attention");
}

/**
 * Bring the Confirm projects folder button into view and highlight it.
 */
function pointToProjectsFolder() {
  const confirm = byId("confirm-workspace");
  confirm.scrollIntoView?.({ block: "center" });
  confirm.focus?.({ preventScroll: true });
  drawAttention(confirm);
}

/**
 * Briefly highlight an element (see `.attention` in the styles).
 * @param element Button or message to point to.
 */
function drawAttention(element) {
  // Restarting the animation highlights again on every reminder.
  element.classList.remove("attention");
  void element.offsetWidth;
  element.classList.add("attention");
}

/**
 * Close Settings; every change has been saved when it was made. Until the
 * projects folder and terms of use are confirmed, Settings stays open.
 */
function closeModels() {
  if (!folderConfirmed || !termsAccepted) return remindSetupConfirmations();
  if (document.activeElement?.matches(".model-id-input")) document.activeElement.blur();
  // A save still running closes Settings once it is done.
  if (applyingModels) closeAfterModelsApply = true;
  else hideModels();
}

/**
 * Return the open project's changed indexing settings, or null for none.
 *
 * The index records these, so changing any of them rebuilds it.
 *
 * @returns Changed values by profile key, or null.
 */
function indexingEdits() {
  if (!ready || !profile) return null;
  // The field lists only the types beyond the ones KíLín reads anyway.
  const extensions = [...new Set([
    ...profile.extensions.filter((extension) => supportedExtensions.has(extension)),
    ...fileTypes(),
  ])];
  const values = {
    extensions,
    max_file_bytes: Number(byId("max-file-bytes").value),
    target_tokens: Number(byId("target-tokens").value),
    max_tokens: Number(byId("max-tokens").value),
    overlap_tokens: Number(byId("overlap-tokens").value),
  };
  const same = (key, value) => (key === "extensions"
    ? JSON.stringify([...value].sort()) === JSON.stringify([...profile[key]].sort())
    : value === profile[key]);
  const edits = Object.fromEntries(Object.entries(values).filter(([key, value]) => !same(key, value)));
  return Object.keys(edits).length ? edits : null;
}

/**
 * Explain the first invalid indexing value, or return an empty string.
 */
function indexingProblem() {
  const [size, target, most, overlap] = ["max-file-bytes", "target-tokens", "max-tokens", "overlap-tokens"]
    .map((id) => Number(byId(id).value));
  if (!Number.isInteger(size) || size < 1) return "The maximum file size must be a whole number of bytes.";
  if (!Number.isInteger(target) || target < 1) return "Target chunk tokens must be a whole number of at least 1.";
  if (!Number.isInteger(most) || most < target) return "Maximum chunk tokens must be at least the target.";
  if (!Number.isInteger(overlap) || overlap < 0 || overlap >= target) return "Overlap tokens must be below the target.";
  return "";
}

/**
 * Save an open project's indexing edit once it is entered and the user
 * agrees to rebuild, so the fields always show the settings in effect.
 * The engine reopens the project, which rebuilds its index; a refused or
 * cancelled value is put back. On the start screen it is only checked.
 *
 * @param event The field's blur event.
 */
async function saveIndexingEdits(event) {
  if (!ready) return keepNewProjectValue(event?.target, indexingProblem());
  const edits = indexingEdits();
  if (!edits) return;
  const problem = busy ? "KíLín is busy. Change this again when it finishes." : indexingProblem();
  if (problem) {
    byId("model-download-status").textContent = problem;
    showIndexingFields();
    return;
  }
  if (!await askKilin(
    "Rebuild the index?",
    "The new value takes effect once the project's index is rebuilt from its files. That takes seconds for a few files and can take several minutes for a large project.",
    "Rebuild",
  )) {
    showIndexingFields();
    return;
  }
  applyingIndexing = true;
  setBusy(true, "Rebuilding the index…");
  byId("model-download-status").textContent = "Applying indexing settings and rebuilding the index…";
  bridge.applyIndexingSettings(JSON.stringify(edits));
}

/**
 * Hide both model suggestion menus and reset their accessibility state.
 */
function closeModelMenus() {
  for (const id of ["chat-model", "embedding-model"]) {
    byId(`${id}-menu`).classList.add("hidden");
    byId(`${id}-options`).setAttribute("aria-expanded", "false");
  }
}

/**
 * Toggle cached choices for a model field without loading a model.
 *
 * @param id Model input identifier: chat-model or embedding-model.
 */
function toggleModelMenu(id) {
  const menu = byId(`${id}-menu`);
  const shouldOpen = menu.classList.contains("hidden");
  closeModelMenus();
  if (!shouldOpen) return;

  const role = id === "chat-model" ? "chat" : "embedding";
  const modelIds = role === "chat"
    ? [...cachedModels].sort((left, right) => left.localeCompare(right))
    : compatibleChoices(role)
      .filter((choice) => cachedModels.has(choice.model_id))
      .map((choice) => choice.model_id)
      .sort((left, right) => left.localeCompare(right));
  const options = modelIds.map((modelId) => {
    const option = document.createElement("button");
    option.type = "button";
    option.setAttribute("role", "option");
    option.textContent = modelId;
    option.addEventListener("click", () => {
      byId(id).value = modelId;
      byId(id).dispatchEvent(new Event("input"));
      closeModelMenus();
    });
    return option;
  });
  if (!options.length) {
    const message = document.createElement("small");
    message.className = "hint";
    message.textContent = "No suggested model cached. Open Models to download one.";
    options.push(message);
  }
  menu.replaceChildren(...options);
  menu.classList.remove("hidden");
  byId(`${id}-options`).setAttribute("aria-expanded", "true");
}

/**
 * Return a size limit large enough for explicitly selected oversized files.
 * @returns Requested maximum file size in bytes.
 */
function selectedFileSizeLimit() {
  let limit = Number(byId("max-file-bytes").value);
  for (const rootPath of selectedSources) {
    const chosen = checkedFiles.get(rootPath) || new Set();
    for (const item of previewSkipped.get(rootPath) || []) {
      if (chosen.has(item.path) && isOversizedSkippedFile(item)) {
        limit = Math.max(limit, Number(item.size_bytes));
      }
    }
  }
  return limit;
}

/**
 * Resolve the selected backend and restore its profile model fields.
 */
function showModelFields() {
  if (!profile) return;
  activeBackend = byId("backend").value === "auto" ? automaticBackend : byId("backend").value;
  byId("chat-model").value = profile[`chat_${activeBackend}`];
  byId("embedding-model").value = profile[`embedding_${activeBackend}`];
  draftLevels = cloneLevels(currentLevels());
  draftDepth = [...currentDepth()];
  draftRouter = currentRouter();
  renderEffortPicker();
  refreshModelStatus();
  if (!byId("models-panel").classList.contains("hidden")) renderModelsPanel();
}

/**
 * Apply a settings response and refresh model and workspace controls.
 *
 * @param data Settings event with profile, catalog, cache, and backend
 *   metadata.
 */
function showSettings(data) {
  if (!data.selection_reason_codes
      || typeof data.selection_reason_codes.FILE_TOO_LARGE !== "string") {
    throw new Error("Missing file-selection codes in GUI settings.");
  }
  selectionReasonCodes = data.selection_reason_codes;
  profile = data.profile;
  savedLevels = data.effort_levels || {};
  defaultLevels = data.default_effort_levels || defaultLevels;
  savedRouters = data.router_models || {};
  defaultRouters = data.default_router_models || defaultRouters;
  defaultEmbeddings = data.default_embedding_models || defaultEmbeddings;
  savedDepth = data.search_depth || null;
  defaultDepth = data.default_search_depth || defaultDepth;
  // The appearance is an app-wide preference; a choice from before
  // preferences existed (kept by the page) is carried over once.
  const remembered = localStorage.getItem("kilin-appearance");
  if (data.appearance && data.appearance !== "system") applyAppearance(data.appearance);
  else if (remembered && remembered !== "system") applyAppearance(remembered);
  renderProjectTypes(data.project_types);
  modelRoles = data.model_roles || modelRoles;
  thinkingSwitches = data.thinking_switches || thinkingSwitches;
  automaticBackend = data.backend;
  cachedModels = new Set(data.cached_models);
  modelChoices = data.model_choices || [];
  installedBackends = data.installed_backends || {};
  supportedExtensions = new Set(data.supported_extensions);
  closeModelMenus();
  // Saved projects use their profile; new projects use the last
  // acknowledged user preference, independently of the automatic policy.
  confirmedBackendPreference = data.backend_preference || "auto";
  byId("backend").value = data.workspace ? profile.backend
    : confirmedBackendPreference;
  byId("query-format").value = profile.query_format;
  showIndexingFields();
  const keepAnswers = ready || answerSettingsEdited;
  for (const key of ["temperature", "max_new_tokens"]) {
    if (!keepAnswers) byId(key.replaceAll("_", "-")).value = profile[key] ?? "";
  }
  if (!keepAnswers) byId("thinking").checked = profile.thinking;
  showModelFields();
  if (data.source_cards) showSourceCardSettings(data.source_cards);
  if (data.workspace) {
    byId("workspace-location").textContent = data.workspace;
    renderWorkspace(data.workspace_files);
  }
  if (selectedSources.length) previewSources();
}

/**
 * Show the profile's indexing settings in the Indexing tab.
 */
function showIndexingFields() {
  byId("extensions").value = profile.extensions.filter((extension) => !supportedExtensions.has(extension)).join(", ");
  for (const key of ["max_file_bytes", "target_tokens", "max_tokens", "overlap_tokens"]) {
    byId(key.replaceAll("_", "-")).value = profile[key] ?? "";
  }
}

/**
 * Collect preparation settings and selected file-type/size adjustments.
 * @returns Preparation settings object for the engine.
 */
function collectSettings() {
  // Preparation settings may require a model reload or index rebuild.
  saveModelFields();
  return {
    ...newProjectAnswerSettings(),
    backend: byId("backend").value,
    _gui_effort_levels: { [activeBackend]: cloneLevels(currentLevels()) },
    _gui_search_depth: [...currentDepth()],
    _gui_router_models: { [activeBackend]: currentRouter() },
    _gui_source_cards: sourceCardValues(),
    chat_mlx: profile.chat_mlx, chat_torch: profile.chat_torch,
    embedding_mlx: profile.embedding_mlx, embedding_torch: profile.embedding_torch,
    query_format: byId("query-format").value,
    extensions: projectFileTypes(),
    max_file_bytes: selectedFileSizeLimit(),
    target_tokens: Number(byId("target-tokens").value),
    max_tokens: Number(byId("max-tokens").value),
    overlap_tokens: Number(byId("overlap-tokens").value),
    // Models are downloaded from Settings, never silently while preparing.
    allow_download: false,
  };
}

/**
 * Display the core-owned card policy and independent backend models.
 * @param settings Validated settings supplied by the engine.
 */
function showSourceCardSettings(settings) {
  sourceCardSettings = JSON.parse(JSON.stringify(settings));
  byId("source-card-enabled").checked = settings.policy.enabled;
  byId("source-card-automatic").checked = settings.policy.automatic;
  byId("source-card-kind").value = settings.policy.kind || "";
  byId("source-card-mlx").value = settings.models.mlx.model_id;
  byId("source-card-torch").value = settings.models.torch.model_id;
  const model = settings.models[activeBackend] || settings.models.mlx;
  byId("source-card-thinking").checked = model.thinking;
  byId("source-card-temperature").value = model.temperature;
  byId("source-card-input").value = model.max_input_tokens;
  byId("source-card-output").value = model.max_new_tokens;
  byId("regenerate-source-cards").disabled = !ready || !settings.policy.enabled;
}

/**
 * Read source-card controls, with shared inference parameters per backend.
 * @returns Complete request for the core-owned settings validator.
 */
function sourceCardValues() {
  const common = {
    thinking: byId("source-card-thinking").checked,
    temperature: Number(byId("source-card-temperature").value),
    max_input_tokens: Number(byId("source-card-input").value),
    max_new_tokens: Number(byId("source-card-output").value),
  };
  return {
    policy: {
      enabled: byId("source-card-enabled").checked,
      kind: byId("source-card-kind").value || null,
      automatic: byId("source-card-automatic").checked,
    },
    models: {
      mlx: { ...common, model_id: byId("source-card-mlx").value.trim() },
      torch: { ...common, model_id: byId("source-card-torch").value.trim() },
    },
  };
}

/**
 * Reject malformed numerical card settings before submitting them to core.
 * @returns Error message, or an empty string for valid values.
 */
function sourceCardProblem() {
  const config = sourceCardValues();
  if (!config.models.mlx.model_id || !config.models.torch.model_id) {
    return "Choose valid MLX and PyTorch source-card model identifiers.";
  }
  const value = config.models.mlx;
  if (!Number.isFinite(value.temperature) || value.temperature < 0
      || byId("source-card-temperature").value.trim() === "") {
    return "Source-card temperature must be finite and nonnegative.";
  }
  for (const key of ["max_input_tokens", "max_new_tokens"]) {
    if (!Number.isInteger(value[key]) || value[key] < 1
        || value[key] > 200000) {
      return "Source-card token limits must be between 1 and 200,000.";
    }
  }
  if (config.policy.enabled && !config.policy.kind) {
    return "Choose a registered card kind when cards are enabled.";
  }
  return "";
}

/**
 * Report valid, excluded, failed and embedded cards from current indexing.
 * @param summary Core-provided persisted card outcome counts.
 */
function showSourceCardSummary(summary) {
  if (!summary) return;
  byId("source-card-summary").textContent =
    `${summary.valid} valid · ${summary.not_applicable} ineligible · `
    + `${summary.failed} failed · ${summary.embedded} embedded `
    + `(${summary.total} recorded)`;
}

/**
 * Persist card settings independently of normal model and index settings.
 */
function saveSourceCardSettings() {
  if (!bridge || busy) return;
  const problem = sourceCardProblem();
  if (problem) {
    byId("model-download-status").textContent = problem;
    return;
  }
  if (!ready) {
    sourceCardSettings = sourceCardValues();
    byId("model-download-status").textContent =
      "Source-card settings kept for the new project.";
    return;
  }
  applyingSourceCards = true;
  setBusy(true, "Updating source-card settings…");
  bridge.saveSourceCardSettings(JSON.stringify(sourceCardValues()));
}

/**
 * Request fresh cards only after confirmation; never rebuild chunk indexes.
 */
async function regenerateSourceCards() {
  if (!ready || !bridge || busy) return;
  if (!await askKilin(
    "Regenerate source cards?",
    "This replaces derived card summaries using the saved generation "
      + "model. Original sources and passage indexes are unchanged.",
    "Regenerate",
  )) return;
  applyingSourceCards = true;
  setBusy(true, "Regenerating source cards…");
  bridge.regenerateSourceCards();
}

/**
 * Return the most recently verified project-wide answer settings.
 * @returns Temperature, Thinking, and answer-token limit.
 */
function confirmedAnswerSettings() {
  return { ...savedAnswerNumbers, thinking: savedAnswerThinking };
}

/**
 * Compare two complete answer-settings snapshots.
 * @param left First settings object.
 * @param right Second settings object.
 * @returns Whether every setting is equal.
 */
function sameAnswerSettings(left, right) {
  return left.temperature === right.temperature
    && left.thinking === right.thinking
    && left.max_new_tokens === right.max_new_tokens;
}

/**
 * Restore the actual or most recently verified values in both controls.
 * @param settings Complete answer-settings snapshot.
 */
function displayAnswerSettings(settings) {
  byId("temperature").value = settings.temperature;
  byId("max-new-tokens").value = settings.max_new_tokens ?? "";
  byId("thinking").checked = settings.thinking;
  renderEffortPicker();
}

/**
 * Adopt the configuration reported by the engine after opening a project.
 * @param settings Complete settings from the opened project.
 */
function resetAnswerSettings(settings) {
  savedAnswerNumbers = {
    temperature: settings.temperature,
    max_new_tokens: settings.max_new_tokens ?? null,
  };
  savedAnswerThinking = settings.thinking;
  answerSettingsSave.pending = null;
  answerSettingsSave.queued = null;
  answerSettingsSave.uncertain = false;
  displayAnswerSettings(confirmedAnswerSettings());
  updateSendButton();
}

/**
 * Save an entire snapshot, or retain its latest version while saving.
 *
 * Only the engine's acknowledgement advances confirmed settings. A request
 * includes the current project so that late replies cannot affect another.
 *
 * @param settings Valid, complete answer settings.
 */
function queueAnswerSettings(settings) {
  if (!ready || !bridge || !openProjectPath) return;
  if (answerSettingsSave.uncertain) {
    displayAnswerSettings(confirmedAnswerSettings());
    byId("model-download-status").textContent =
      "Reopen the project to verify its answer settings.";
    return;
  }
  if (answerSettingsSave.pending) {
    answerSettingsSave.queued = settings;
    byId("model-download-status").textContent =
      "Saving answer settings… latest edits queued.";
    updateSendButton();
    return;
  }
  if (sameAnswerSettings(settings, confirmedAnswerSettings())) return;
  const request = {
    id: ++answerSettingsSave.nextId,
    project: openProjectPath,
    settings,
  };
  answerSettingsSave.pending = request;
  byId("model-download-status").textContent = "Saving answer settings…";
  updateSendButton();
  bridge.saveAnswerSettings(JSON.stringify({
    request_id: request.id,
    project: request.project,
    settings: request.settings,
  }));
}

/**
 * Check whether an asynchronous save result belongs to this project.
 * @param data Engine acknowledgement or operation-specific failure.
 * @returns Whether it matches the active request and project.
 */
function matchesAnswerSettingsSave(data) {
  const pending = answerSettingsSave.pending;
  return pending !== null && data.request_id === pending.id
    && data.project === pending.project && data.project === openProjectPath;
}

/**
 * Confirm a successful write, then send only the latest queued edit.
 * @param data Saved settings and correlated request identity.
 */
function answerSettingsSaved(data) {
  if (!matchesAnswerSettingsSave(data)) return;
  savedAnswerNumbers = {
    temperature: data.temperature,
    max_new_tokens: data.max_new_tokens ?? null,
  };
  savedAnswerThinking = data.thinking;
  const queued = answerSettingsSave.queued;
  answerSettingsSave.pending = null;
  answerSettingsSave.queued = null;
  if (queued && !sameAnswerSettings(queued, confirmedAnswerSettings())) {
    queueAnswerSettings(queued);
    return;
  }
  byId("model-download-status").textContent =
    "Saved in profile.ini. The next answer uses it.";
  updateSendButton();
}

/**
 * Recover the saved profile on failure and discard all unconfirmed edits.
 *
 * If the engine could not reread the file and update the live chat, the
 * project must be reopened before answering again.
 *
 * @param data Failure, verified profile (if available), and request identity.
 */
function answerSettingsFailed(data) {
  if (!matchesAnswerSettingsSave(data)) return;
  answerSettingsSave.pending = null;
  answerSettingsSave.queued = null;
  answerSettingsSave.uncertain = data.verified !== true;
  if (!answerSettingsSave.uncertain) {
    savedAnswerNumbers = {
      temperature: data.temperature,
      max_new_tokens: data.max_new_tokens ?? null,
    };
    savedAnswerThinking = data.thinking;
  }
  displayAnswerSettings(confirmedAnswerSettings());
  const message = answerSettingsSave.uncertain
    ? `${data.message} Reopen the project to verify its answer settings.`
    : data.message;
  byId("model-download-status").textContent = message;
  error.textContent = message;
  error.classList.remove("hidden");
  updateSendButton();
}

/**
 * Validate the Answers numbers, then save them with the current Thinking.
 *
 * The start screen retains these values for project creation. While an
 * existing project's save is pending, later changes replace the queued
 * snapshot rather than starting a second write.
 *
 * @param event Change event for an answer number, if any.
 */
function saveAnswerNumbers(event) {
  const temperature = Number(byId("temperature").value);
  const tokens = byId("max-new-tokens").value.trim();
  const maxNewTokens = tokens ? Number(tokens) : null;
  const invalid = !byId("temperature").value.trim() || !Number.isFinite(temperature) || temperature < 0
    ? "Choose a temperature of zero or higher."
    : tokens && (!Number.isInteger(maxNewTokens) || maxNewTokens < 1)
      ? "Choose a positive answer-token limit or leave it blank."
      : "";
  if (!ready) return keepNewProjectValue(event?.target, invalid);
  const problem = invalid || (busy ? "KíLín is busy. Change this again when it finishes." : "");
  if (problem) {
    byId("model-download-status").textContent = problem;
    const last = answerSettingsSave.queued
      || answerSettingsSave.pending?.settings
      || confirmedAnswerSettings();
    byId("temperature").value = last.temperature;
    byId("max-new-tokens").value = last.max_new_tokens ?? "";
    return;
  }
  queueAnswerSettings({
    temperature, max_new_tokens: maxNewTokens,
    thinking: byId("thinking").checked,
  });
}

/**
 * Return the Answers settings for a new project, so its profile starts
 * with them; later changes are saved when a question is asked.
 *
 * @returns ``thinking``, plus ``temperature`` and ``max_new_tokens`` when
 *   valid ("none" for a blank token limit); invalid values are left out so
 *   the project keeps its defaults.
 */
function newProjectAnswerSettings() {
  const settings = { thinking: byId("thinking").checked };
  const temperature = Number(byId("temperature").value);
  if (byId("temperature").value.trim() && Number.isFinite(temperature) && temperature >= 0) {
    settings.temperature = temperature;
  }
  const tokens = byId("max-new-tokens").value.trim();
  if (!tokens) settings.max_new_tokens = "none";
  else if (Number.isInteger(Number(tokens)) && Number(tokens) > 0) settings.max_new_tokens = Number(tokens);
  return settings;
}

/**
 * Tell whether the workspace tree shows a file.
 *
 * The core keeps a lock file next to each open conversation, so that only
 * one window writes it, and macOS adds .DS_Store files to folders it shows
 * in Finder. Neither is the user's file, so both are hidden.
 *
 * @param path Workspace-relative file path reported by the backend.
 * @returns False for a conversation lock under logs/ or a .DS_Store file,
 *   true otherwise.
 */
function shownInWorkspace(path) {
  if (path.split("/").pop() === ".DS_Store") return false;
  return !(path.startsWith("logs/") && path.endsWith(".json.lock"));
}

/**
 * Render the saved workspace file tree without selection checkboxes.
 *
 * @param paths Workspace-relative file paths reported by the backend.
 */
function renderWorkspace(paths) {
  workspacePaths = new Set(paths);
  const host = byId("workspace-tree");
  host.replaceChildren();
  // The core lists every file; the tree leaves out conversation locks.
  const root = fileTree(paths.filter(shownInWorkspace));
  // Every folder comes from the core's actual workspace inventory.
  for (const [name, node] of [...root.directories].sort()) {
    const files = allFiles(node);
    // Plain rows: the delete button added below must not act as a label's
    // control, or clicking a file's name would also ask to delete it.
    const item = branch(name, node, files, new Set(), 0, null, {}, false);
    // Sources and outputs start open; the other folders stay folded.
    const key = `${name}|${files[0] || ""}`;
    const startsOpen = name === "sources" || name === "outputs";
    item.open = startsOpen ? !collapsedWorkspaceFolders.has(key) : expandedFolders.has(key);
    item.addEventListener("toggle", () => {
      if (!item.isConnected || !startsOpen) return;
      item.open ? collapsedWorkspaceFolders.delete(key) : collapsedWorkspaceFolders.add(key);
    });
    [item, ...item.querySelectorAll(".tree-folder")].forEach(showFolderCount);
    item.querySelectorAll(".tree-file").forEach((line) => {
      line.tabIndex = 0;
      line.setAttribute("role", "button");
      line.addEventListener("click", () => openWorkspaceFile(line.dataset.path));
      line.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          openWorkspaceFile(line.dataset.path);
        }
      });
    });
    if (name === "sources") {
      item.id = "workspace-source-drop";
      item.classList.add("source-drop-target");
      const hint = document.createElement("p");
      hint.className = "source-drop-hint";
      hint.textContent = "Drop files or folders here to add sources";
      item.querySelector("summary").append(hint);
      item.addEventListener("dragover", (event) => {
        if (busy || !ready) return;
        event.preventDefault();
        event.dataTransfer.dropEffect = "copy";
        showNativeSourceDrag(true);
      });
      item.addEventListener("dragleave", (event) => {
        if (!item.contains(event.relatedTarget)) showNativeSourceDrag(false);
      });
      item.addEventListener("drop", (event) => {
        event.preventDefault();
        showNativeSourceDrag(false);
        // A file dragged from outputs or templates in this panel.
        const inside = event.dataTransfer.getData(WORKSPACE_DRAG_TYPE);
        if (inside) return copyToSources(inside);
        addSources(Array.from(event.dataTransfer.files, (file) => file.path).filter(Boolean));
      });
      item.addEventListener("contextmenu", (event) => {
        const row = event.target.closest(".tree-file, .tree-folder > summary");
        const path = row?.classList.contains("tree-file")
          ? row.dataset.path : row?.parentElement.dataset.path;
        // The sources folder itself is not deletable; only its contents are.
        if (!path || !path.startsWith("sources/")) return;
        event.preventDefault();
        showSourceMenu(event.clientX, event.clientY, path, row.classList.contains("tree-file"));
      });
    }
    if (EDITABLE_FOLDERS.includes(name)) {
      item.dataset.path = name;
      item.querySelectorAll(".tree-file").forEach((line) => {
        const path = line.dataset.path;
        // Drag a file onto sources to add it there (and index it).
        line.draggable = true;
        line.addEventListener("dragstart", (event) => {
          event.dataTransfer.setData(WORKSPACE_DRAG_TYPE, path);
          event.dataTransfer.effectAllowed = "copy";
        });
        const remove = document.createElement("button");
        remove.type = "button";
        remove.className = "tree-delete";
        remove.textContent = "×";
        remove.title = `Delete ${path.split("/").pop()}`;
        remove.setAttribute("aria-label", remove.title);
        remove.addEventListener("click", (event) => {
          event.stopPropagation();
          deleteFromFolder(name, path, true);
        });
        line.append(remove);
      });
      const hint = document.createElement("p");
      hint.className = "source-drop-hint";
      hint.textContent = "Right-click to add, replace, or delete; never indexed";
      item.querySelector("summary").append(hint);
      item.addEventListener("contextmenu", (event) => {
        const row = event.target.closest(".tree-file, .tree-folder > summary");
        const isFile = Boolean(row?.classList.contains("tree-file"));
        const path = isFile ? row.dataset.path : row?.parentElement.dataset.path;
        if (!path || !(path === name || path.startsWith(`${name}/`))) return;
        event.preventDefault();
        showFolderMenu(event.clientX, event.clientY, name, path, isFile);
      });
    }
    host.append(item);
  }
  for (const file of root.files) {
    const line = document.createElement("div");
    line.className = "tree-file";
    line.textContent = file;
    line.tabIndex = 0;
    line.setAttribute("role", "button");
    line.addEventListener("click", () => openWorkspaceFile(file));
    host.append(line);
  }
  markActiveConversationFile();
}

/**
 * Bold the workspace entry of the conversation file now in use.
 *
 * The engine's trace path is the active conversation's JSON file, an
 * absolute path; workspace rows carry project-relative paths.
 */
function markActiveConversationFile() {
  const active = activeTracePath.replaceAll("\\", "/");
  byId("workspace-tree").querySelectorAll(".tree-file").forEach((row) => {
    const path = row.dataset.path;
    row.classList.toggle(
      "active-conversation",
      Boolean(active && path && active.endsWith(`/${path}`)),
    );
  });
}

// Send the visible drop rectangle to Qt; all other workspace rows reject drops.
let sourceDragActive = false;
const sourceDropSpotlight = document.createElement("div");
sourceDropSpotlight.className = "source-drop-spotlight hidden";
sourceDropSpotlight.setAttribute("aria-hidden", "true");
document.body.append(sourceDropSpotlight);
let lastSourceDropBounds = "";
/**
 * The visible part of an element, clipped by scrolling parents, or null.
 * @param target DOM element whose visible rectangle is needed.
 */
function visibleRect(target) {
  const rect = target?.getBoundingClientRect();
  const visible = !busy && rect?.width && rect?.height && target.getClientRects().length
    && !target.closest(".hidden") && byId("models-panel").classList.contains("hidden");
  if (!visible) return null;
  let left = Math.max(0, rect.left), top = Math.max(0, rect.top);
  let right = Math.min(innerWidth, rect.right), bottom = Math.min(innerHeight, rect.bottom);
  for (let parent = target.parentElement; parent; parent = parent.parentElement) {
    const style = getComputedStyle(parent);
    const bounds = parent.getBoundingClientRect();
    if (/(auto|scroll|hidden)/.test(style.overflowY)) {
      top = Math.max(top, bounds.top); bottom = Math.min(bottom, bounds.bottom);
    }
    if (/(auto|scroll|hidden)/.test(style.overflowX)) {
      left = Math.max(left, bounds.left); right = Math.min(right, bounds.right);
    }
  }
  return {x: left, y: top, width: Math.max(0, right - left), height: Math.max(0, bottom - top)};
}

/**
 * Tell Qt where files may be dropped: sources, and outputs or templates.
 */
function syncSourceDropTarget() {
  if (!bridge?.setSourceDropTarget) return;
  const sources = visibleRect(ready ? byId("workspace-source-drop") : byId("drop-zone"));
  const folders = ready ? EDITABLE_FOLDERS.flatMap((name) => {
    const rect = visibleRect(document.querySelector(`#workspace-tree details[data-path="${name}"]`));
    return rect ? [{ name, ...rect }] : [];
  }) : [];
  sendSourceDropBounds(JSON.stringify({ ...(sources || {}), folders }));
}
/**
 * Update the native source-drop overlay and send its bounds to Qt.
 *
 * @param bounds JSON string containing the visible target rectangle.
 */
function sendSourceDropBounds(bounds) {
  const region = JSON.parse(bounds);
  const visible = sourceDragActive && region.width > 0 && region.height > 0;
  sourceDropSpotlight.classList.toggle("hidden", !visible);
  if (visible) {
    Object.assign(sourceDropSpotlight.style, {
      left: `${region.x}px`, top: `${region.y}px`,
      width: `${region.width}px`, height: `${region.height}px`,
    });
  }
  if (bounds === lastSourceDropBounds) return;
  lastSourceDropBounds = bounds;
  bridge.setSourceDropTarget(bounds);
}
/**
 * Reflect native dragging over the editable source locations.
 *
 * @param active Whether a native file drag is currently in progress.
 */
function showNativeSourceDrag(active) {
  sourceDragActive = Boolean(active && !busy);
  const target = ready ? byId("workspace-source-drop") : byId("drop-zone");
  target?.classList.toggle("drag-over", sourceDragActive);
  for (const name of EDITABLE_FOLDERS) {
    document.querySelector(`#workspace-tree details[data-path="${name}"]`)
      ?.classList.toggle("drag-over", sourceDragActive);
  }
  syncSourceDropTarget();
}
window.addEventListener("resize", syncSourceDropTarget);
document.addEventListener("scroll", syncSourceDropTarget, true);
document.addEventListener("toggle", syncSourceDropTarget, true);
new ResizeObserver(syncSourceDropTarget).observe(shell);
new MutationObserver(() => requestAnimationFrame(syncSourceDropTarget)).observe(shell, {
  subtree: true, childList: true, attributes: true, attributeFilter: ["class", "open"],
});
// Never let a file drop navigate the page or write to indexes/profile areas.
document.addEventListener("dragover", (event) => {
  if (!event.target.closest("#workspace-source-drop, #drop-zone")) {
    event.preventDefault(); event.dataTransfer.dropEffect = "none";
  }
});
document.addEventListener("drop", (event) => {
  event.preventDefault();
  showNativeSourceDrag(false);
});
window.addEventListener("blur", () => showNativeSourceDrag(false));

// One shared right-click menu for items inside the workspace sources folder.
const sourceMenu = document.createElement("div");
sourceMenu.className = "context-menu hidden";
sourceMenu.setAttribute("role", "menu");
document.body.append(sourceMenu);

/**
 * Offer deletion for one workspace source file or folder at the pointer.
 *
 * @param x Viewport x coordinate of the right-click.
 * @param y Viewport y coordinate of the right-click.
 * @param path Workspace-relative path beginning with sources/.
 * @param isFile Whether the path is a file rather than a folder.
 */
function showSourceMenu(x, y, path, isFile) {
  showWorkspaceMenu(x, y, [[
    isFile ? "Delete file from project…" : "Delete folder from project…",
    () => removeSource(path, isFile),
  ]]);
}

/**
 * Show the shared workspace menu with labelled actions at the pointer.
 *
 * @param x Viewport x coordinate of the right-click.
 * @param y Viewport y coordinate of the right-click.
 * @param items Pairs of menu label and action.
 */
function showWorkspaceMenu(x, y, items) {
  if (busy || !ready) return;
  const buttons = items.map(([label, action]) => {
    const button = document.createElement("button");
    button.type = "button";
    button.setAttribute("role", "menuitem");
    button.textContent = label;
    button.addEventListener("click", () => {
      hideSourceMenu();
      action();
    });
    return button;
  });
  sourceMenu.replaceChildren(...buttons);
  sourceMenu.classList.remove("hidden");
  const { width, height } = sourceMenu.getBoundingClientRect();
  sourceMenu.style.left = `${Math.min(x, window.innerWidth - width - 8)}px`;
  sourceMenu.style.top = `${Math.min(y, window.innerHeight - height - 8)}px`;
  buttons[0]?.focus();
}

/**
 * Offer edits in an editable folder: add files, replace a file, or delete.
 *
 * @param x Viewport x coordinate of the right-click.
 * @param y Viewport y coordinate of the right-click.
 * @param folder Editable folder: outputs or templates.
 * @param path Workspace-relative path: the folder or inside it.
 * @param isFile Whether the path is a file rather than a folder.
 */
function showFolderMenu(x, y, folder, path, isFile) {
  const items = [[`Add files to ${folder}…`, () => bridge.chooseFolderFiles(folder, "add")]];
  if (isFile) {
    items.push(["Replace with another file…", () => bridge.chooseFolderFiles(folder, path)]);
    items.push(["Copy to sources (and index it)…", () => copyToSources(path)]);
  }
  if (path !== folder) {
    items.push([isFile ? `Delete file from ${folder}…` : `Delete folder from ${folder}…`,
      () => deleteFromFolder(folder, path, isFile)]);
  }
  showWorkspaceMenu(x, y, items);
}

/**
 * Confirm, then delete a file or folder in outputs or templates.
 * @param folder Editable workspace folder name.
 * @param path Workspace-relative target path.
 * @param isFile Whether the target is a file.
 */
async function deleteFromFolder(folder, path, isFile) {
  const name = path.split("/").pop();
  const what = isFile ? `“${name}”` : `the folder “${name}” and everything in it`;
  if (!await askKilin(`Delete ${what}?`, "It is deleted permanently, not moved to the Trash.", "Delete")) return;
  editFolder({ folder, operation: "remove", paths: [path] }, `Deleting from ${folder}…`);
}

/**
 * Copy a workspace file (from outputs or templates) into the sources.
 *
 * @param path Workspace-relative path, such as outputs/summary.md.
 */
function copyToSources(path) {
  const root = openProjectPath.replace(/[\\/]+$/, "");
  if (!root || !EDITABLE_FOLDERS.some((name) => path.startsWith(`${name}/`))) return;
  addSources([`${root}/${path}`]);
}

/**
 * Send a folder edit; the engine only refreshes the workspace tree.
 * @param request Folder-edit request sent to the backend.
 * @param message Progress message displayed during the edit.
 */
function editFolder(request, message) {
  if (busy || !bridge) return;
  error.classList.add("hidden");
  setBusy(true, message);
  bridge.editFolder(JSON.stringify(request));
}

/**
 * Copy chosen files into an editable folder, or replace one file there.
 *
 * @param folder Editable folder: outputs or templates.
 * @param purpose "add", or the workspace path to replace.
 * @param paths Local files chosen in the native dialog.
 */
async function folderFilesChosen(folder, purpose, paths) {
  if (purpose !== "add") {
    const name = purpose.split("/").pop();
    const source = paths[0].split(/[\\/]/).pop();
    if (!await askKilin(`Replace “${name}”?`, `KíLín puts the content of ${source} in it; the name stays.`, "Replace")) return;
    editFolder({ folder, operation: "replace", path: purpose, source: paths[0] }, `Replacing a file in ${folder}…`);
    return;
  }
  const names = paths.map((path) => String(path).split(/[\\/]/).pop());
  const taken = names.filter((name) => workspacePaths.has(`${folder}/${name}`));
  let replace = false;
  if (taken.length) {
    const what = taken.length === 1 ? `“${taken[0]}” is` : `${taken.map((name) => `“${name}”`).join(", ")} are`;
    const answer = await askKilin(`${what} already in ${folder}`,
      "Keep both gives the new copies a number.", "Replace", "Keep both");
    if (!answer) return;
    replace = answer === true;
  }
  editFolder({ folder, operation: "add", paths, replace }, `Adding to ${folder}…`);
}

/**
 * Hide the context menu for workspace source files.
 */
function hideSourceMenu() {
  sourceMenu.classList.add("hidden");
}

/**
 * Confirm, then delete a workspace source copy and update the index.
 *
 * @param path Workspace-relative path beginning with sources/.
 * @param isFile Whether the path is a file rather than a folder.
 */
async function removeSource(path, isFile) {
  if (busy || !bridge) return;
  const name = path.split("/").pop();
  const what = isFile ? `“${name}”` : `the folder “${name}” and everything in it`;
  if (!await askKilin(`Delete ${what}?`, "", "Delete") || busy) return;
  error.classList.add("hidden");
  setBusy(true, "Removing sources and updating the index…");
  bridge.removeProjectSources(JSON.stringify([path]));
}

document.addEventListener("click", (event) => {
  if (!sourceMenu.contains(event.target)) hideSourceMenu();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") hideSourceMenu();
});
window.addEventListener("blur", hideSourceMenu);
document.addEventListener("scroll", hideSourceMenu, true);

/**
 * Confirm opening a project file with its default application.
 *
 * @param path Workspace-relative path to the selected file.
 */
async function openWorkspaceFile(path) {
  if (!path || !bridge) return;
  if (!await askKilin(`Open “${path.split("/").pop()}” in its default app?`, "", "Open")) return;
  bridge.openWorkspaceFile(path);
}

/**
 * Put a question whose answer failed or was stopped back in the message
 * box, unless the user has already started typing another one.
 */
function restoreUnansweredQuestion() {
  const question = unansweredQuestion;
  unansweredQuestion = "";
  if (question && !byId("question").value.trim()) byId("question").value = question;
}

/**
 * Restore the interface after an engine failure and display the error.
 *
 * @param message Error message for the user.
 * @param tracePath Path to the saved trace, if available.
 */
function showError(message, tracePath = "") {
  clearAskingConversation();
  applyingModels = false;
  closeAfterModelsApply = false;
  byId("model-download-status").textContent = message;
  stopPhaseAnimation();
  resetAnswerStages();
  if (pendingTurn) {
    pendingTurn.answer.textContent = "Could not complete this answer.";
    pendingTurn.item.classList.add("failed");
    pendingTurn = null;
    restoreUnansweredQuestion();
  }
  error.textContent = tracePath ? `${message} Trace: ${tracePath}` : message;
  error.classList.remove("hidden");
  setupCompletionVisible = false;
  byId("readiness-meter").classList.add("hidden");
  byId("readiness-meter").classList.remove("indeterminate");
  if (downloadingModel) {
    byId("model-download-status").textContent = error.textContent;
    byId("cancel-model-download").classList.add("hidden");
    downloadingModel = null;
  }
  installingModels = false;
  shownErrorStatus = byId("model-download-status").textContent;
  setBusy(false);
  byId("connection").textContent = "Error";
  requestAnimationFrame(() => error.scrollIntoView({ block: "nearest" }));
}

/**
 * Hide an error that no longer applies, such as after a project opens.
 *
 * The Settings status line is cleared only while it still shows that
 * error's text, so later messages there (such as download progress) stay.
 */
function clearError() {
  const status = byId("model-download-status");
  if (shownErrorStatus && status.textContent === shownErrorStatus) {
    status.textContent = "";
  }
  shownErrorStatus = "";
  error.classList.add("hidden");
}

/**
 * Count the files a sync can import, update or remove, and those it keeps.
 *
 * @param changes The ``source_changes`` event: lists of copy paths under
 *   ``added``, ``changed``, ``deleted`` and ``edited``.
 * @returns The sentence at the top of the sync dialog.
 */
function sourceSyncSummary(changes) {
  const files = (count) => `${count} file${count === 1 ? "" : "s"}`;
  const count = changes.added.length + changes.changed.length + changes.deleted.length;
  let text = `${files(count)} changed in the original folders since KíLín copied them.`;
  if (changes.edited.length) {
    text += ` KíLín keeps the ${files(changes.edited.length)} you edited in the project.`;
  }
  return text;
}

/**
 * List the choices for one group of files; the first one is the default.
 *
 * @param group ``added``, ``changed``, ``deleted`` or ``edited``.
 * @returns Pairs of sync action and label; empty for edited copies, which
 *   are always kept.
 */
function sourceSyncChoices(group) {
  return {
    added: [["import", "Add"], ["keep", "Skip"]],
    changed: [["replace", "Update"], ["keep_both", "Keep both"], ["keep", "Keep old"]],
    deleted: [["delete", "Remove"], ["keep", "Keep"]],
    edited: [],
  }[group];
}

/**
 * Choose the same kind of action for every file at once.
 *
 * @param changes The ``source_changes`` event.
 * @param mode ``update`` brings every change in; ``both`` also keeps the
 *   old copy of changed files and the copy of deleted ones.
 * @returns Sync actions keyed by ``sources/`` path; edited copies are left
 *   out, so they are kept.
 */
function sourceSyncPlan(changes, mode) {
  const actions = mode === "both"
    ? { added: "import", changed: "keep_both", deleted: "keep" }
    : { added: "import", changed: "replace", deleted: "delete" };
  const decisions = {};
  for (const [group, action] of Object.entries(actions)) {
    for (const path of changes[group]) decisions[path] = action;
  }
  return decisions;
}

/**
 * Group the differing files under short headings, leaving out empty groups.
 *
 * @param changes The ``source_changes`` event.
 * @returns Groups with their key, heading and ``sources/`` paths.
 */
function sourceSyncGroups(changes) {
  return [
    ["added", "New"],
    ["changed", "Changed"],
    ["deleted", "Deleted"],
    ["edited", "Edited by you"],
  ].filter(([group]) => changes[group].length).map(([group, title]) => ({
    group, title, paths: changes[group],
  }));
}

let sourceSyncChanges = null;

/**
 * Build one file row: its name, folder and choices.
 *
 * @param group The file's change group.
 * @param path The copy's ``sources/`` path.
 * @param index A number that keeps each row's radio group separate.
 * @returns The list item.
 */
function sourceSyncRow(group, path, index) {
  const item = document.createElement("li");
  item.className = `sync-row ${group}`;
  const relative = path.replace(/^sources\//, "");
  const slash = relative.lastIndexOf("/");
  const name = document.createElement("span");
  name.className = "sync-name";
  const file = document.createElement("strong");
  file.textContent = relative.slice(slash + 1);
  name.append(file);
  if (slash > 0) {
    const folder = document.createElement("small");
    folder.textContent = relative.slice(0, slash);
    name.append(folder);
  }
  item.append(name);
  const choices = sourceSyncChoices(group);
  if (!choices.length) {
    const note = document.createElement("span");
    note.className = "sync-kept";
    note.textContent = "Your edit is kept";
    item.append(note);
    return item;
  }
  const options = document.createElement("div");
  options.className = "segmented";
  options.setAttribute("role", "radiogroup");
  options.setAttribute("aria-label", `Action for ${relative}`);
  choices.forEach(([action, label], position) => {
    const option = document.createElement("label");
    const input = document.createElement("input");
    input.type = "radio";
    input.name = `source-sync-${index}`;
    input.value = action;
    input.dataset.path = path;
    input.checked = position === 0;
    const text = document.createElement("span");
    text.textContent = label;
    option.append(input, text);
    options.append(option);
  });
  item.append(options);
  return item;
}

/**
 * Switch the sync dialog between the quick buttons and the file list.
 *
 * @param fileByFile Whether to show the list of files with their choices.
 */
function setSourceSyncFileByFile(fileByFile) {
  byId("source-sync-files").classList.toggle("hidden", !fileByFile);
  byId("source-sync-quick").classList.toggle("hidden", fileByFile);
  byId("source-sync-manual").classList.toggle("hidden", !fileByFile);
}

/**
 * Ask what to do with the files whose originals changed.
 *
 * The quick buttons update everything, keep old and new copies, or keep
 * the old copies; "Choose file by file" lists each file with its own
 * choice. Copies the user edited are always kept.
 *
 * @param changes The ``source_changes`` event.
 */
function showSourceSync(changes) {
  sourceSyncChanges = changes;
  byId("source-sync-summary").textContent = sourceSyncSummary(changes);
  const counts = byId("source-sync-counts");
  const list = byId("source-sync-files");
  counts.replaceChildren();
  list.replaceChildren();
  let index = 0;
  for (const { group, title, paths } of sourceSyncGroups(changes)) {
    const count = document.createElement("li");
    count.className = `sync-count ${group}`;
    count.textContent = `${paths.length} ${title.toLowerCase()}`;
    counts.append(count);
    const heading = document.createElement("h3");
    heading.textContent = title;
    const items = document.createElement("ul");
    for (const path of paths) items.append(sourceSyncRow(group, path, index++));
    list.append(heading, items);
  }
  byId("source-sync-both").classList.toggle("hidden", !changes.changed.length);
  setSourceSyncFileByFile(false);
  byId("source-sync").showModal();
}

/**
 * Close the sync dialog and send the chosen actions to the core.
 *
 * @param decisions Sync actions keyed by ``sources/`` path.
 */
function syncSources(decisions) {
  byId("source-sync").close();
  if (busy || !bridge) return;
  setBusy(true, "Updating sources and the index…");
  bridge.syncProjectSources(JSON.stringify(decisions));
}

byId("source-sync-keep").addEventListener("click", () => byId("source-sync").close());
byId("source-sync-both").addEventListener("click", () => syncSources(sourceSyncPlan(sourceSyncChanges, "both")));
byId("source-sync-update").addEventListener("click", () => syncSources(sourceSyncPlan(sourceSyncChanges, "update")));
byId("source-sync-pick").addEventListener("click", () => setSourceSyncFileByFile(true));
byId("source-sync-back").addEventListener("click", () => setSourceSyncFileByFile(false));
byId("source-sync-apply").addEventListener("click", () => {
  const decisions = {};
  for (const input of byId("source-sync-files").querySelectorAll("input:checked")) {
    decisions[input.dataset.path] = input.value;
  }
  syncSources(decisions);
});

/**
 * Count the copies of already imported sources that changed.
 *
 * @param changes The check's ``changes``: ``sources/`` paths by group.
 * @returns Files the user can bring in, update or remove.
 */
function sourceAddPending(changes) {
  return changes.added.length + changes.changed.length + changes.deleted.length;
}

/**
 * Sum up what adding the checked sources would copy and leave out.
 *
 * @param check The ``sources_checked`` event: ``add`` (``name``, ``kind``,
 *   ``files``, ``state``), left-out ``files``, ``skipped_folders`` and the
 *   ``changes`` of sources already in the project.
 * @returns The sentence at the top of the add dialog.
 */
function sourceAddSummary(check) {
  const files = (count) => `${count} file${count === 1 ? "" : "s"}`;
  const adding = check.add.filter((item) => item.state !== "existing");
  const existing = check.add.filter((item) => item.state === "existing");
  const count = adding.reduce((total, item) => total + item.files, 0);
  if (!count && !existing.length) return "Nothing can be added: this project does not index these files.";
  const parts = [];
  if (count) parts.push(`KíLín will add ${files(count)}.`);
  if (existing.length) {
    const one = existing.length === 1;
    const names = existing.map((item) => item.name).join(", ");
    const pending = sourceAddPending(check.changes);
    parts.push(pending
      ? `${names} ${one ? "is" : "are"} already in the project; ${files(pending)} changed since ${one ? "it was" : "they were"} copied.`
      : `${names} ${one ? "is" : "are"} already in the project, with no changes.`);
  }
  const left = check.files.length;
  if (left) parts.push(`${files(left)} ${left === 1 ? "is" : "are"} left out because this project does not index ${left === 1 ? "it" : "them"}.`);
  return parts.join(" ");
}

/**
 * Say why a checked file is left out.
 *
 * @param file A left-out row with a reason code.
 * @returns Short text shown next to the file.
 */
function sourceAddReason(file) {
  if (file.reason_code === selectionReasonCodes?.FILE_TOO_LARGE) return "Too large";
  if (file.reason_code === selectionReasonCodes?.EMPTY_FILE) return "Empty file";
  return "File type not included";
}

let sourceAddPaths = [];

/**
 * Build one row of the add dialog.
 *
 * @param group ``added`` for files to add, ``left-out`` otherwise.
 * @param relative Folder and file name to show.
 * @param detail Muted text on the right.
 * @returns The list item.
 */
function sourceAddRow(group, relative, detail) {
  const item = document.createElement("li");
  item.className = `sync-row ${group}`;
  const slash = relative.lastIndexOf("/");
  const name = document.createElement("span");
  name.className = "sync-name";
  const file = document.createElement("strong");
  file.textContent = relative.slice(slash + 1);
  name.append(file);
  if (slash > 0) {
    const folder = document.createElement("small");
    folder.textContent = relative.slice(0, slash);
    name.append(folder);
  }
  const side = document.createElement("span");
  side.className = "sync-kept";
  side.textContent = detail;
  item.append(name, side);
  return item;
}

/**
 * Show what adding the dropped sources would copy, and ask to go on.
 *
 * Sources already in the project are matched by their original path: an
 * unchanged one needs nothing, and files that changed since they were
 * copied get the same choices as in the sync dialog. A different file
 * with a taken name is copied under a new name.
 *
 * @param check The ``sources_checked`` event.
 */
function showSourceAdd(check) {
  sourceAddPaths = check.sources;
  const files = (count) => `${count} file${count === 1 ? "" : "s"}`;
  const adding = check.add.filter((item) => item.state !== "existing");
  const existing = check.add.filter((item) => item.state === "existing");
  const count = adding.reduce((total, item) => total + item.files, 0);
  const pending = sourceAddPending(check.changes);
  byId("source-add-summary").textContent = sourceAddSummary(check);
  const counts = byId("source-add-counts");
  const list = byId("source-add-files");
  counts.replaceChildren();
  list.replaceChildren();
  const { added, changed, deleted } = check.changes;
  const differing = (item) => [...added, ...changed, ...deleted].filter((path) => (
    path === `sources/${item.name}` || path.startsWith(`sources/${item.name}/`)
  )).length;
  const sections = [
    ["added", "To add", count ? `${count} to add` : null, adding.map((item) => sourceAddRow(
      "added", item.name, item.state === "renamed"
        ? "Same name, different file · added under a new name"
        : item.kind === "folder" ? files(item.files) : "File",
    ))],
    ["", "Already in the project", existing.length ? `${existing.length} already added` : null, existing.map((item) => sourceAddRow(
      "", item.name, differing(item) ? `${files(differing(item))} changed since copied` : "No changes",
    ))],
  ];
  // A chip only: the changed files are listed under their own headings.
  sections.push(["changed", "", pending ? `${pending} changed` : null, []]);
  // Radio names must not repeat those of the sync dialog's rows.
  let index = 100000;
  for (const { group, paths } of sourceSyncGroups(check.changes)) {
    const heading = {
      added: "New in the original", changed: "Changed in the original",
      deleted: "Deleted from the original", edited: "Edited by you",
    }[group];
    sections.push([group, heading, null, paths.map((path) => sourceSyncRow(group, path, index++))]);
  }
  sections.push(
    ["left-out", "Left out", check.files.length ? `${check.files.length} left out` : null, [
      ...check.files.slice(0, 100).map((file) => sourceAddRow("left-out", file.path, sourceAddReason(file))),
      ...(check.files.length > 100 ? [sourceAddRow("left-out", `and ${check.files.length - 100} more`, "")] : []),
    ]],
    ["left-out", "Not added", null, check.skipped_folders.map((name) => sourceAddRow(
      "left-out", name, "No files this project indexes",
    ))],
  );
  for (const [group, title, chip, rows] of sections) {
    if (chip) {
      const item = document.createElement("li");
      item.className = `sync-count ${group}`;
      item.textContent = chip;
      counts.append(item);
    }
    if (!rows.length) continue;
    const heading = document.createElement("h3");
    heading.textContent = title;
    const items = document.createElement("ul");
    items.append(...rows);
    list.append(heading, items);
  }
  const confirm = byId("source-add-confirm");
  confirm.classList.toggle("hidden", !count && !pending);
  confirm.textContent = count ? "Add" : "Update";
  byId("source-add-cancel").textContent = count || pending ? "Cancel" : "Close";
  byId("source-add").showModal();
}

byId("source-add-cancel").addEventListener("click", () => byId("source-add").close());
byId("source-add-confirm").addEventListener("click", () => {
  byId("source-add").close();
  if (busy || !bridge) return;
  const decisions = {};
  for (const input of byId("source-add-files").querySelectorAll("input:checked")) {
    decisions[input.dataset.path] = input.value;
  }
  setBusy(true, "Adding sources and updating the index…");
  bridge.addProjectSources(JSON.stringify(sourceAddPaths), JSON.stringify(decisions));
});

/**
 * Show Send, or Stop while an answer is generated. A chat read meanwhile
 * shows a disabled Send: Stop belongs to the chat being answered.
 */
function updateSendButton() {
  const answering = busy && ready && activity.classList.contains("answer-progress");
  const stop = answering && !reading;
  byId("send").disabled = !ready || (busy && !stop)
    || Boolean(answerSettingsSave.pending) || answerSettingsSave.uncertain;
  byId("send").classList.toggle("stopping", stop);
  byId("send").querySelector("span").textContent = stop ? "■" : "↑";
  const sendLabel = stop ? "Stop generation"
    : answerSettingsSave.uncertain ? "Reopen project to verify settings"
      : answerSettingsSave.pending ? "Saving answer settings…"
        : "Send message";
  byId("send").setAttribute("aria-label", sendLabel);
  byId("send").title = sendLabel;
}

/**
 * Gate operation controls and update the visible activity state.
 *
 * @param value Whether a backend operation is active.
 * @param message Activity label; the default empty string clears the old
 *   text.
 */
function setBusy(value, message = "") {
  busy = value;
  byId("backend").disabled = ready || value;
  for (const id of ["chat-model", "embedding-model", "query-format"]) byId(id).disabled = value;
  // Closing Settings is independent of new-project readiness.
  showSetupLock();
  byId("conversation-name").contentEditable = String(!value);
  syncSourceDropTarget();
  if (!value) {
    activity.classList.remove("answer-progress");
    byId("send").classList.remove("waiting");
    answeringConversationId = "";
  }
  // A project can be prepared only once KíLín setup is done.
  byId("open").disabled = value || !bridge || !kilinReady;
  const answering = value && ready && activity.classList.contains("answer-progress");
  byId("question-form").classList.toggle("composer-disabled", value && !answering);
  byId("question-form").setAttribute("aria-busy", String(value));
  for (const id of ["attach-button", "attach-files", "attach-folder", "effort-button", "effort-model-range", "effort-depth-range"]) {
    byId(id).disabled = value || !ready;
  }
  // The next question can be typed while an answer is generated; sending
  // waits until it is done (the submit handler checks busy).
  byId("question").disabled = !ready || (value && !answering);
  if (value) {
    showNativeSourceDrag(false);
    byId("attach-menu").classList.add("hidden");
    byId("effort-menu").classList.add("hidden");
  }
  updateSendButton();
  byId("stop").classList.toggle("hidden", !value || answering);
  byId("new-chat").disabled = value || !ready;
  // While an answer runs, other chats can still be read, not changed.
  document.querySelectorAll(".conversation-item").forEach((item) => { item.disabled = (value && !answering) || !ready; });
  document.querySelectorAll(".conversation-delete, .project-item, .project-settings, .message-branch").forEach((item) => { item.disabled = value || !ready; });
  byId("home-logo").disabled = value;
  byId("new-project").disabled = value;
  byId("stop").disabled = !value;
  activity.classList.toggle("hidden", !value);
  byId("activity-text").textContent = message;
  byId("connection").textContent = value ? "Busy" : (ready || bridge ? "Ready" : "Init");
  if (value) {
    byId("activity-percent").textContent = "Working…";
    byId("activity-detail").textContent = "";
  }
  if (!byId("models-panel").classList.contains("hidden")) renderModelsPanel();
  if (!value) focusComposer();
}

/**
 * Format a numeric duration for display in seconds or minutes.
 *
 * @param value Elapsed or estimated seconds.
 * @returns Formatted duration text.
 */
function seconds(value) {
  const rounded = Math.max(0, Math.round(value));
  const minutes = Math.floor(rounded / 60);
  const remainder = rounded % 60;
  return minutes ? `${minutes}m ${remainder}s` : `${remainder}s`;
}

/**
 * Format a byte count using compact binary units.
 *
 * @param value Raw byte count.
 * @returns Human-readable binary-unit size.
 */
function bytes(value) {
  const amount = Math.max(0, Number(value) || 0);
  const units = ["B", "KB", "MB", "GB", "TB"];
  let size = amount;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  const digits = unit === 0 || size >= 100 ? 0 : (size >= 10 ? 1 : 2);
  return `${size.toFixed(digits)} ${units[unit]}`;
}

/**
 * Describe a model-download progress update.
 *
 * @param data Model-download heartbeat data.
 * @returns Elapsed time, progress, and estimated time details.
 */
function downloadProgress(data) {
  const downloaded = Math.max(0, Number(data.completed_bytes) || 0);
  const total = Math.max(0, Number(data.total_bytes) || 0);
  const rate = Math.max(0, Number(data.bytes_per_second) || 0);
  const elapsed = Math.max(0, Number(data.elapsed_seconds) || 0);
  const percent = total ? Math.min(100, (downloaded / total) * 100) : null;
  const parts = [`${seconds(elapsed)} elapsed`];
  if (total) parts.push(`${bytes(downloaded)} of ${bytes(total)}`);
  if (rate) parts.push(`${bytes(rate)}/s`);
  if (data.eta_seconds != null && Number.isFinite(Number(data.eta_seconds))) {
    parts.push(Number(data.eta_seconds) <= 0 ? "finishing…" : `about ${seconds(data.eta_seconds)} left`);
  } else {
    parts.push("estimating time remaining…");
  }
  return { percent, text: parts.join(" · ") };
}

/**
 * Update the setup card's download progress meter.
 *
 * @param percent Current completion percentage, or null.
 */
function updateDownloadMeter(percent) {
  const meter = byId("readiness-meter");
  meter.classList.remove("hidden");
  meter.classList.toggle("indeterminate", percent == null);
  if (percent == null) {
    meter.removeAttribute("aria-valuenow");
    meter.querySelector("span").style.width = "";
  } else {
    const value = Math.min(100, Math.max(0, percent));
    meter.setAttribute("aria-valuenow", value.toFixed(1));
    meter.querySelector("span").style.width = `${value}%`;
  }
}

/**
 * Render backend progress, elapsed time, and optional remaining time.
 *
 * @param data Progress event; percentages are presentation values, not
 *   guarantees.
 */
function showProgress(data) {
  setBusy(true, data.message);
  const percent = Math.min(100, Math.max(0, Number(data.percent)));
  byId("activity-percent").textContent = `${percent.toFixed(1)}% overall`;
  const details = [];
  if (data.current != null && data.total != null) {
    details.push(`${data.current.toLocaleString()} / ${data.total.toLocaleString()}`);
  }
  if (data.remaining != null) {
    details.push(`${data.remaining.toLocaleString()} remaining`);
  }
  if (data.elapsed_seconds != null) {
    details.push(`${seconds(data.elapsed_seconds)} elapsed`);
  }
  if (data.eta_seconds != null) {
    details.push(`about ${seconds(data.eta_seconds)} left`);
  }
  byId("activity-detail").textContent = details.join(" · ");
  if (data.stage === "building_source_cards") {
    byId("source-card-progress").textContent =
      `Generating source cards: ${data.current ?? 0} / ${data.total ?? 0}`;
  }
}

/**
 * Remove the active reference preview and reset its selection state.
 */
function clearSourcePreview() {
  activeReference?.button?.classList.remove("active");
  activeReference?.holder?.classList.remove("expanded");
  activeReference?.preview?.remove();
  activeReference = null;
}

/**
 * Clear selected-turn and reference presentation state.
 */
function clearReferences() {
  selectedTurn?.classList.remove("selected");
  selectedTurn = null;
  byId("references-for").textContent = "Select an answer to see its sources.";
  byId("references-list").replaceChildren();
  clearSourcePreview();
}

/**
 * Create a source-preview container and return its DOM references.
 * @returns Object containing the preview DOM references.
 */
function makeSourcePreview() {
  const preview = document.createElement("section");
  preview.className = "source-preview";
  preview.setAttribute("aria-label", "Cited source preview");
  const heading = document.createElement("header");
  heading.className = "source-preview-heading";
  const note = document.createElement("p");
  note.className = "hint";
  const close = document.createElement("button");
  close.type = "button";
  close.className = "sidebar-toggle";
  close.textContent = "Close";
  const lines = document.createElement("pre");
  lines.className = "source-preview-lines";
  heading.append(note, close);
  preview.append(heading, lines);
  return { preview, close, note, lines };
}

/**
 * Toggle a citation preview or request its source lines from the engine.
 *
 * @param item Conversation article containing the answer.
 * @param source Evidence record carrying a citation number and provenance.
 * @param button Reference button whose active state is updated.
 * @param holder Container in which to place the preview.
 */
function openReference(item, source, button, holder) {
  byId("references-section").open = true;
  const turnIndex = Number(item.dataset.turnIndex);
  const citation = Number(source.citation);
  if (activeReference?.turnIndex === turnIndex && activeReference?.citation === citation) {
    clearSourcePreview();
    return;
  }
  clearSourcePreview();
  const nodes = makeSourcePreview();
  nodes.note.textContent = "Loading nearby lines…";
  nodes.close.addEventListener("click", clearSourcePreview);
  holder.append(nodes.preview);
  holder.classList.add("expanded");
  button.classList.add("active");
  activeReference = { turnIndex, citation, button, holder, ...nodes };
  if (reading) {
    // The engine looks up lines in the chat it has open, not this one.
    nodes.note.textContent = "Open this chat to see the source lines.";
    return;
  }
  bridge?.previewReference(turnIndex, activeReference.citation);
  nodes.preview.scrollIntoView({ block: "nearest" });
}

/**
 * Select an answer and render its evidence list, optionally opening a
 * citation.
 *
 * @param item Conversation article to mark as selected.
 * @param turn Saved or newly completed question/answer/evidence record.
 * @param citation Optional evidence number to open immediately.
 */
function selectTurn(item, turn, citation = null) {
  selectedTurn?.classList.remove("selected");
  selectedTurn = item;
  item.classList.add("selected");
  clearSourcePreview();
  const evidence = turn.evidence || [];
  const { labels, cited } = labelsForTurn(turn);
  const sources = referencedSources(evidence, labels, cited);
  byId("references-for").textContent = !evidence.length
    ? "No sources saved for this answer."
    : cited && cited < evidence.length
      ? `Sources for: ${splitAttached(turn.question).question} · ${cited} cited of ${evidence.length} passages read`
      : `Sources for: ${splitAttached(turn.question).question}`;
  const list = byId("references-list");
  list.replaceChildren();
  for (const source of sources) {
    const holder = document.createElement("div");
    holder.className = "reference-entry";
    const button = document.createElement("button");
    button.type = "button";
    button.className = "reference-item";
    button.dataset.citation = String(source.citation);
    const number = document.createElement("span");
    number.className = "reference-number";
    number.textContent = String(labels.get(Number(source.citation)));
    const path = document.createElement("span");
    path.className = "reference-path";
    path.textContent = `${source.path}:${source.start_line}-${source.end_line}`;
    button.append(number, path);
    button.addEventListener("click", () => openReference(item, source, button, holder));
    holder.append(button);
    list.append(holder);
  }
  if (citation !== null) {
    const source = (turn.evidence || []).find((entry) => Number(entry.citation) === citation);
    const button = list.querySelector(`[data-citation="${citation}"]`);
    if (source && button) openReference(item, source, button, button.parentElement);
  }
}
/**
 * Render numbered source lines only for the currently active reference.
 *
 * @param data Reference response with matching turn/citation identifiers
 *   and lines.
 */
function renderSourcePreview(data) {
  if (!activeReference || data.turn_index !== activeReference.turnIndex
      || data.citation !== activeReference.citation) return;
  const location = data.page
    ? `Page ${data.page} · lines ${data.start_line}–${data.end_line}`
    : `Lines ${data.start_line}–${data.end_line}`;
  activeReference.note.textContent = data.note
    ? `${location} · ${data.note}` : location;
  const host = activeReference.lines;
  host.replaceChildren();
  for (const line of data.lines) {
    const row = document.createElement("span");
    row.className = line.highlighted ? "source-line highlighted" : "source-line";
    const number = document.createElement("span");
    number.className = "source-line-number";
    number.textContent = String(line.number);
    const content = document.createElement("span");
    content.className = "source-line-text";
    content.textContent = line.text || " ";
    row.append(number, content);
    host.append(row);
  }
  activeReference.preview.scrollIntoView({ block: "nearest" });
}


/**
 * Number the cited sources 1, 2, 3… in the order the answer cites them.
 *
 * Evidence numbers count all retrieved passages, so they can be large
 * (143, 286…). Only the labels the user sees change; links still use the
 * evidence numbers.
 *
 * @param markers Evidence numbers of each linked citation marker, in
 *   reading order, such as [[143], [286, 288]].
 * @param evidence The turn's evidence records.
 * @returns Map from evidence number to shown label. Evidence the answer
 *   does not cite follows, in its saved order.
 */
function citationLabels(markers, evidence) {
  const labels = new Map();
  const label = (number) => {
    if (!labels.has(number)) labels.set(number, labels.size + 1);
  };
  markers.flat().forEach(label);
  evidence.forEach((source) => label(Number(source.citation)));
  return labels;
}

// Shown labels of each rendered turn and how many sources its answer
// cites, shared by its answer and references.
const turnCitationLabels = new WeakMap();

/**
 * Choose and order the sources the References panel lists.
 *
 * An answer may read many passages but cite only a few: list those, in
 * their label order, unless the answer cites none.
 *
 * @param evidence The turn's evidence records.
 * @param labels Map from evidence number to shown label.
 * @param cited Number of distinct sources the answer cites.
 * @returns The evidence records to list.
 */
function referencedSources(evidence, labels, cited) {
  return evidence
    .filter((source) => !cited || labels.get(Number(source.citation)) <= cited)
    .sort((a, b) => labels.get(Number(a.citation)) - labels.get(Number(b.citation)));
}

/**
 * Return the shown citation labels of a turn, numbering its evidence in
 * order when its answer has not been rendered.
 *
 * @param turn Saved question, answer, and evidence data.
 * @returns ``labels`` (evidence number to shown label) and ``cited``, the
 *   number of distinct sources the answer cites; those have labels 1 to
 *   ``cited``.
 */
function labelsForTurn(turn) {
  return turnCitationLabels.get(turn)
    || { labels: citationLabels([], turn.evidence || []), cited: 0 };
}

/**
 * Render answer text with links only for recognized numeric citations.
 *
 * @param answer Destination DOM element.
 * @param turn Answer text and its evidence records.
 * @param item Conversation article used when selecting a linked citation.
 * @param thinkingMs Time spent on thinking, or null if absent.
 */
function renderAnswer(answer, turn, item, thinkingMs = null) {
  const citations = new Set((turn.evidence || []).map((source) => Number(source.citation)));
  const parts = splitThinking(turn.answer);
  if (parts.thinking) {
    answer.append(thoughtsBlock(parts.thinking,
      thinkingMs == null ? "Show thinking" : `Thought for ${formatDuration(thinkingMs)}`));
  }
  // Markdown and citations are both recognized; only evidence numbers that
  // belong to this answer become links.
  const markerNumbers = (marker) => {
    const numbers = marker.slice(1, -1).split(",").map((value) => Number(value.trim()));
    return numbers.every((number) => citations.has(number)) ? numbers : null;
  };
  // A first, unseen pass finds the linked markers in reading order, so
  // they can be shown as 1, 2, 3… instead of evidence numbers.
  const markers = [];
  renderMarkdown(document.createElement("div"), parts.answer, (marker) => {
    const numbers = markerNumbers(marker);
    if (numbers) markers.push(numbers);
    return null;
  });
  const labels = citationLabels(markers, turn.evidence || []);
  turnCitationLabels.set(turn, { labels, cited: new Set(markers.flat()).size });
  const body = document.createElement("div");
  body.className = "answer-body";
  renderMarkdown(body, parts.answer, (marker) => {
    const numbers = markerNumbers(marker);
    if (!numbers) return null;
    const superscript = document.createElement("sup");
    superscript.className = "answer-citation";
    numbers.forEach((number, index) => {
      if (index) superscript.append(document.createTextNode(","));
      const link = document.createElement("button");
      link.type = "button";
      link.textContent = String(labels.get(number));
      link.setAttribute("aria-label", `Show reference ${labels.get(number)}`);
      link.addEventListener("click", (event) => {
        event.stopPropagation();
        chooseSidePanels(true);
        selectTurn(item, turn, number);
      });
      superscript.append(link);
    });
    return superscript;
  });
  answer.append(body);
  if (turn.parameters?.finish_reason === "length") {
    const notice = document.createElement("p");
    notice.className = "answer-limit-notice";
    notice.setAttribute("role", "status");
    notice.textContent = "This answer reached the token limit and may be incomplete. Ask for a shorter answer, or raise Maximum answer tokens in Settings and retry.";
    answer.append(notice);
  }
}

// Markdown blocks the renderer recognizes at the start of a line.
const MARKDOWN_FENCE = /^\s*```/;
const MARKDOWN_HEADING = /^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$/;
const MARKDOWN_RULE = /^\s{0,3}([-*_])(\s*\1){2,}\s*$/;
const MARKDOWN_QUOTE = /^\s{0,3}>\s?/;
const MARKDOWN_BULLET = /^\s{0,3}[-*+]\s+/;
const MARKDOWN_NUMBER = /^\s{0,3}\d{1,3}[.)]\s+/;
// Display maths: a line starting with $$ or \[, up to the closing $$ or \].
const MARKDOWN_MATH = /^\s*(\$\$|\\\[)/;
// The line under a table's header: dashes per column, colons for alignment.
const MARKDOWN_TABLE_DIVIDER = /^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$/;

/**
 * Split one Markdown table row into its cell texts.
 *
 * Outer pipes are optional; an escaped pipe (\|) stays inside its cell.
 *
 * @param line Table row source line.
 * @returns Trimmed cell texts, left to right.
 */
function markdownTableCells(line) {
  let row = line.trim();
  if (row.startsWith("|")) row = row.slice(1);
  if (row.endsWith("|") && !row.endsWith("\\|")) row = row.slice(0, -1);
  return row.split(/(?<!\\)\|/).map((cell) => cell.trim().replace(/\\\|/g, "|"));
}

/**
 * Detect a Markdown table: a header row, then a divider row with the same
 * number of columns.
 *
 * @param lines All source lines.
 * @param index Index of the candidate header row.
 * @returns Whether a table starts at this line.
 */
function isMarkdownTableStart(lines, index) {
  const header = lines[index];
  const divider = lines[index + 1];
  if (!header?.includes("|") || !divider?.includes("|") || !MARKDOWN_TABLE_DIVIDER.test(divider)) return false;
  return markdownTableCells(header).length === markdownTableCells(divider).length;
}

/**
 * Detect Markdown lines that begin a block-level structure.
 *
 * @param line Candidate Markdown source line.
 * @returns Whether the line starts a supported Markdown block.
 */
function isMarkdownBlockStart(line) {
  return [MARKDOWN_FENCE, MARKDOWN_MATH, MARKDOWN_HEADING, MARKDOWN_RULE, MARKDOWN_QUOTE, MARKDOWN_BULLET,
    MARKDOWN_NUMBER].some((pattern) => pattern.test(line));
}

/**
 * Build LaTeX maths as MathML, which Qt WebEngine draws natively.
 *
 * KaTeX (``katex/katex.min.js``) writes the markup itself and runs
 * without ``trust``, so commands such as \href are refused. LaTeX it
 * cannot read, or a missing KaTeX, shows the original text instead.
 *
 * @param tex LaTeX between the delimiters.
 * @param source The original text, delimiters included.
 * @param display Whether it is display maths on its own line.
 * @returns New element holding the maths.
 */
function mathNode(tex, source, display) {
  const node = document.createElement(display ? "div" : "span");
  node.className = display ? "answer-math display" : "answer-math";
  try {
    katex.render(tex, node, { displayMode: display, output: "mathml", throwOnError: true });
  } catch {
    node.textContent = source;
  }
  return node;
}

/**
 * Render model text as Markdown using DOM nodes only, never HTML parsing.
 *
 * Supports headings, paragraphs, bullet and numbered lists, block quotes,
 * fenced code, rules, tables, and inline bold, italics, and code. Plain
 * text simply becomes paragraphs.
 *
 * @param host Element that receives the blocks.
 * @param text Model output without thinking text.
 * @param linkCitation Turns a "[1, 2]" marker into a node, or null to keep
 *   it as text; omit while streaming.
 */
function renderMarkdown(host, text, linkCitation = null) {
  const lines = String(text || "").replace(/\r\n?/g, "\n").split("\n");
  let index = 0;
  while (index < lines.length) {
    const line = lines[index];
    if (!line.trim()) { index += 1; continue; }
    if (MARKDOWN_FENCE.test(line)) {
      const code = [];
      index += 1;
      while (index < lines.length && !MARKDOWN_FENCE.test(lines[index])) code.push(lines[index++]);
      index += 1;
      const pre = document.createElement("pre");
      if (/^\s*```[ \t]*latex[ \t]*$/i.test(line)) {
        pre.className = "answer-latex-source";
      }
      const element = document.createElement("code");
      element.textContent = code.join("\n");
      pre.append(element);
      host.append(pre);
      continue;
    }
    if (MARKDOWN_MATH.test(line)) {
      // Everything up to the matching $$ or \] is one display formula; it
      // may also fit on one line, as in "$$ x^2 $$".
      const first = line.trim();
      const close = first.startsWith("$$") ? "$$" : "\\]";
      let end = first.length > 2 && first.endsWith(close) ? index : -1;
      for (let next = index + 1; end < 0 && next < lines.length; next += 1) {
        if (lines[next].trim().endsWith(close)) end = next;
      }
      if (end < 0) {
        // Never closed (or still streaming): keep the line as text.
        const element = document.createElement("p");
        element.textContent = line;
        host.append(element);
        index += 1;
        continue;
      }
      const source = lines.slice(index, end + 1).map((part) => part.trim()).join("\n");
      host.append(mathNode(source.slice(2, -2).trim(), source, true));
      index = end + 1;
      continue;
    }
    if (isMarkdownTableStart(lines, index)) {
      const header = markdownTableCells(line);
      // Alignment per column comes from the colons in the divider row.
      const align = markdownTableCells(lines[index + 1]).map((cell) => (
        cell.startsWith(":") && cell.endsWith(":") ? "center"
          : cell.endsWith(":") ? "right" : cell.startsWith(":") ? "left" : ""));
      index += 2;
      const table = document.createElement("table");
      const row = (cells, tag) => {
        const element = document.createElement("tr");
        header.forEach((_, column) => {
          // Short rows get empty cells; extra cells are dropped.
          const cell = document.createElement(tag);
          if (align[column]) cell.style.textAlign = align[column];
          appendInlineMarkdown(cell, cells[column] || "", linkCitation);
          element.append(cell);
        });
        return element;
      };
      const head = document.createElement("thead");
      head.append(row(header, "th"));
      const body = document.createElement("tbody");
      while (index < lines.length && lines[index].trim() && lines[index].includes("|")
        && !isMarkdownBlockStart(lines[index])) {
        body.append(row(markdownTableCells(lines[index++]), "td"));
      }
      table.append(head, body);
      // A wide table scrolls sideways instead of widening the chat.
      const frame = document.createElement("div");
      frame.className = "answer-table";
      frame.append(table);
      host.append(frame);
      continue;
    }
    const heading = line.match(MARKDOWN_HEADING);
    if (heading) {
      // Chat answers sit inside a page, so # starts at a small heading.
      const element = document.createElement(`h${Math.min(6, heading[1].length + 2)}`);
      appendInlineMarkdown(element, heading[2], linkCitation);
      host.append(element);
      index += 1;
      continue;
    }
    if (MARKDOWN_RULE.test(line)) {
      host.append(document.createElement("hr"));
      index += 1;
      continue;
    }
    if (MARKDOWN_QUOTE.test(line)) {
      const quoted = [];
      while (index < lines.length && MARKDOWN_QUOTE.test(lines[index])) {
        quoted.push(lines[index++].replace(MARKDOWN_QUOTE, ""));
      }
      const quote = document.createElement("blockquote");
      renderMarkdown(quote, quoted.join("\n"), linkCitation);
      host.append(quote);
      continue;
    }
    const pattern = MARKDOWN_NUMBER.test(line) ? MARKDOWN_NUMBER
      : MARKDOWN_BULLET.test(line) ? MARKDOWN_BULLET : null;
    if (pattern) {
      const list = document.createElement(pattern === MARKDOWN_NUMBER ? "ol" : "ul");
      const start = line.match(/\d+/);
      if (pattern === MARKDOWN_NUMBER && start && start[0] !== "1") list.start = Number(start[0]);
      while (index < lines.length && pattern.test(lines[index])) {
        let content = lines[index++].replace(pattern, "");
        // Indented lines continue the item.
        while (index < lines.length && /^\s{2,}\S/.test(lines[index]) && !isMarkdownBlockStart(lines[index])) {
          content += `\n${lines[index++].trim()}`;
        }
        const item = document.createElement("li");
        appendInlineMarkdown(item, content, linkCitation);
        list.append(item);
      }
      host.append(list);
      continue;
    }
    const paragraph = [];
    while (index < lines.length && lines[index].trim() && !isMarkdownBlockStart(lines[index])
      && !isMarkdownTableStart(lines, index)) {
      paragraph.push(lines[index++]);
    }
    const element = document.createElement("p");
    appendInlineMarkdown(element, paragraph.join("\n"), linkCitation);
    host.append(element);
  }
}

// Inline code, bold, italics (not inside words such as snake_case),
// external https links such as [text](https://…), citation markers such as
// [1] or [2, 3], and maths: $$…$$, \(…\), or $…$ with no space just inside,
// so prices such as "$5 and $10" stay text. Groups: 1 code, 2 bold,
// 3 italics, 4 link, 5 citation, 6 maths.
const MARKDOWN_INLINE = /(`[^`\n]+`)|(\*\*[^*\n]+\*\*|__[^_\n]+__)|((?<![\w*])\*[^*\s][^*\n]*\*(?![\w*])|(?<![\w_])_[^_\s][^_\n]*_(?![\w_]))|(\[[^\]\n]{1,100}\]\(https:\/\/[^\s)\n]+\))|(\[\d+(?:\s*,\s*\d+)*\])|(\$\$[^$]+?\$\$|\\\(.+?\\\)|(?<![\\\w$])\$(?![\s$])[^$\n]*?[^\s\\$]\$(?![\w$])|(?<![\\\w$])\$[^\s$\\]\$(?![\w$]))/g;

/**
 * Render escaped inline Markdown and optionally link evidence citations.
 *
 * @param element DOM element receiving the parsed content.
 * @param text Inline Markdown text to render.
 * @param linkCitation Callback constructing a citation link when valid.
 */
function appendInlineMarkdown(element, text, linkCitation) {
  let last = 0;
  for (const match of text.matchAll(MARKDOWN_INLINE)) {
    let before = text.slice(last, match.index);
    last = match.index + match[0].length;
    if (match[1]) {
      element.append(before);
      const code = document.createElement("code");
      code.textContent = match[1].slice(1, -1);
      element.append(code);
    } else if (match[6]) {
      element.append(before);
      const source = match[6];
      const wrap = source.startsWith("$$") ? 2 : source.startsWith("\\(") ? 2 : 1;
      element.append(mathNode(source.slice(wrap, -wrap).trim(), source, false));
    } else if (match[2] || match[3]) {
      element.append(before);
      const node = document.createElement(match[2] ? "strong" : "em");
      appendInlineMarkdown(node, match[0].slice(match[2] ? 2 : 1, match[2] ? -2 : -1), linkCitation);
      element.append(node);
    } else if (match[4]) {
      element.append(before);
      const parts = match[4].match(/^\[([^\]]+)\]\((https:\/\/[^)]+)\)$/);
      let url = null;
      try { url = parts ? new URL(parts[2]) : null; }
      catch (_) { /* Invalid external links stay plain text. */ }
      if (url && url.protocol === "https:" && url.hostname &&
          !url.username && !url.password) {
        // No href: WebEngine cannot speculatively fetch a link it only
        // displays. A deliberate click delegates opening to the browser.
        const link = document.createElement("button");
        link.type = "button";
        link.className = "external-link";
        link.textContent = parts[1];
        link.title = url.href;
        link.addEventListener("click", (event) => {
          event.preventDefault();
          event.stopPropagation();
          bridge?.openExternalLink(url.href);
        });
        element.append(link);
      } else {
        element.append(match[4]);
      }
    } else {
      const link = linkCitation ? linkCitation(match[5]) : null;
      // A linked citation sits right after its claim.
      if (link) before = before.replace(/[ \t]+$/, "");
      element.append(before, link || match[5]);
    }
  }
  element.append(text.slice(last));
}

const MESSAGE_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/**
 * Format a turn's saved time for the actions under its messages.
 *
 * @param timestamp ISO 8601 time the turn was saved.
 * @param now Current time, so today's messages show only the time.
 * @returns "14:05" today, "28 Oct 14:05" this year, "28 Oct 2025 14:05"
 *   before, or "" if the time is missing or invalid.
 */
function messageTime(timestamp, now = new Date()) {
  const date = new Date(timestamp || "");
  if (!timestamp || Number.isNaN(date.getTime())) return "";
  const time = `${String(date.getHours()).padStart(2, "0")}:${String(date.getMinutes()).padStart(2, "0")}`;
  if (date.toDateString() === now.toDateString()) return time;
  const day = `${date.getDate()} ${MESSAGE_MONTHS[date.getMonth()]}`;
  return date.getFullYear() === now.getFullYear()
    ? `${day} ${time}` : `${day} ${date.getFullYear()} ${time}`;
}

/**
 * Return an answer as the user sees it, for copying: no thinking, and
 * citations with their shown numbers.
 *
 * @param turn Saved answer and its evidence.
 * @returns Answer Markdown with linked markers such as "[143, 7]" shown as
 *   "[1, 2]".
 */
function answerCopyText(turn) {
  const { labels } = labelsForTurn(turn);
  return splitThinking(turn.answer).answer.trim().replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, (marker, inner) => {
    const numbers = inner.split(",").map((value) => Number(value.trim()));
    if (!numbers.every((number) => labels.has(number))) return marker;
    return `[${numbers.map((number) => labels.get(number)).join(", ")}]`;
  });
}

/**
 * Put text on the clipboard. In the app Qt copies it, since the page may
 * not; in a plain browser use the clipboard API, falling back to a hidden
 * text box.
 *
 * @param text Text to copy.
 */
async function copyToClipboard(text) {
  if (bridge) {
    bridge.copyText(text);
    return;
  }
  try {
    await navigator.clipboard.writeText(text);
    return;
  } catch {
    // Fall back below.
  }
  const box = document.createElement("textarea");
  box.value = text;
  box.setAttribute("readonly", "");
  box.style.position = "fixed";
  box.style.opacity = "0";
  document.body.append(box);
  box.select();
  document.execCommand("copy");
  box.remove();
}

/**
 * Build the row shown under a message on hover or focus: Copy, Branch from
 * here, and the message's time.
 *
 * @param text Text the Copy button puts on the clipboard.
 * @param timestamp The turn's saved time, or "" while it is not saved.
 * @param label What is copied, for the button's accessible name.
 * @param onBranch Starts a branch from this message.
 * @returns New element placed right after the message.
 */
function messageActions(text, timestamp, label, onBranch) {
  const row = document.createElement("div");
  row.className = "message-actions";
  const copy = document.createElement("button");
  copy.type = "button";
  copy.className = "message-action";
  copy.title = "Copy";
  copy.setAttribute("aria-label", `Copy ${label}`);
  copy.innerHTML = '<svg aria-hidden="true" viewBox="0 0 24 24"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V6a2 2 0 0 1 2-2h8"/></svg>';
  copy.addEventListener("click", async (event) => {
    event.stopPropagation();
    await copyToClipboard(text);
    copy.title = "Copied";
    copy.classList.add("copied");
    setTimeout(() => {
      copy.title = "Copy";
      copy.classList.remove("copied");
    }, 1500);
  });
  const branch = document.createElement("button");
  branch.type = "button";
  branch.className = "message-action message-branch";
  branch.title = "Branch from here";
  branch.setAttribute("aria-label", `Branch from this ${label}`);
  branch.disabled = busy;
  branch.innerHTML = '<svg aria-hidden="true" viewBox="0 0 24 24"><path d="M3 12h5c3.5 0 4.5-6 8-6h4M17 3l3 3-3 3M8 12c3.5 0 4.5 6 8 6h4M17 15l3 3-3 3"/></svg>';
  branch.addEventListener("click", (event) => {
    event.stopPropagation();
    onBranch();
  });
  row.append(copy, branch);
  const shown = messageTime(timestamp);
  if (shown) {
    const time = document.createElement("time");
    time.dateTime = timestamp;
    time.title = new Date(timestamp).toLocaleString();
    time.textContent = shown;
    row.append(time);
  }
  return row;
}

/**
 * Add Copy, Branch from here, and time rows under a turn's question and
 * answer.
 *
 * @param item Conversation article holding the two messages.
 * @param turn Saved question and answer.
 */
function addMessageActions(item, turn) {
  const question = item.querySelector(".user-message");
  const answer = item.querySelector(".answer-message");
  const asked = splitAttached(turn.question).question;
  if (question) {
    const row = messageActions(asked, turn.timestamp, "question", () => branchFromMessage(item, false, asked));
    row.classList.add("user-actions");
    question.after(row);
  }
  if (answer) answer.after(messageActions(answerCopyText(turn), turn.timestamp, "answer", () => branchFromMessage(item, true, "")));
}

/**
 * Open an empty chat: the one on screen when it has no turns yet, otherwise
 * one the engine reuses or creates, so unused empty chats do not pile up.
 *
 * @param draft Question to put in the message box, or "".
 */
function openEmptyChat(draft) {
  if (busy) return;
  if (!conversation.querySelector(".turn")) {
    if (draft) byId("question").value = draft;
    byId("question").focus();
    return;
  }
  branchDraft = draft;
  setBusy(true, "Starting a new conversation…");
  bridge.newConversation();
}

/**
 * Branch the conversation at one message. From an answer, the branch keeps
 * every turn up to it; from a question, the turns before it, with the
 * question put back in the message box to edit and send again.
 *
 * @param item Conversation article holding the message.
 * @param withAnswer Whether the branch keeps this turn's answer.
 * @param draft Question to put back in the message box, or "".
 */
function branchFromMessage(item, withAnswer, draft) {
  // A chat being read is not the one the engine has open.
  if (busy || reading) return;
  // Count saved turns only: a stopped answer on screen was never saved.
  const saved = [...conversation.querySelectorAll(".turn:not(.pending):not(.stopped)")];
  const index = saved.indexOf(item);
  if (index < 0) return;
  if (index === 0 && !withAnswer) {
    // Nothing to keep before the first question: an empty chat will do.
    openEmptyChat(draft);
    return;
  }
  branchDraft = draft;
  setBusy(true, "Branching this conversation…");
  bridge.branchConversation(withAnswer ? index + 1 : index);
}

// The list KíLín adds under a question with attached files; older chats
// listed the original paths under "Attached paths to consider".
const ATTACHED_LIST = /\n\nAttached (?:files|paths to consider):\n((?:- [^\n]+(?:\n|$))+)$/;

/**
 * Split a saved question into the user's text and its attached files.
 *
 * @param text Question as sent to the engine.
 * @returns ``question`` (the user's text) and ``files`` (attached paths).
 */
function splitAttached(text) {
  const value = String(text || "");
  const match = value.match(ATTACHED_LIST);
  if (!match) return { question: value, files: [] };
  return {
    question: value.slice(0, match.index),
    files: match[1].trim().split("\n").map((line) => line.slice(2).trim()),
  };
}

/**
 * Show a question with its attached files as chips under the text.
 *
 * A file in sources opens its place in the workspace tree; a path from an
 * older chat, never copied, is shown by name only.
 *
 * @param element The question bubble.
 * @param text Question as sent to the engine.
 */
function fillQuestion(element, text) {
  const { question, files } = splitAttached(text);
  element.textContent = question;
  if (!files.length) return;
  const row = document.createElement("div");
  row.className = "question-attachments";
  for (const path of files) {
    const inSources = path.startsWith("sources/");
    const chip = document.createElement(inSources ? "button" : "span");
    chip.className = "question-attachment";
    chip.textContent = path.replace(/\/$/, "").split(/[\\/]/).pop() || path;
    chip.title = inSources ? `Show ${path} in the workspace` : path;
    if (inSources) {
      chip.type = "button";
      chip.addEventListener("click", () => revealWorkspacePath(path));
    }
    row.append(chip);
  }
  element.append(row);
}

/**
 * Open the Workspace panel at a file or folder and briefly highlight it.
 *
 * @param path Workspace-relative path; a folder ends with "/".
 */
function revealWorkspacePath(path) {
  showWorkspace(true);
  const target = path.replace(/\/$/, "");
  const tree = byId("workspace-tree");
  const row = [...tree.querySelectorAll(".tree-file")].find((line) => line.dataset.path === target)
    || [...tree.querySelectorAll(".tree-folder")].find((folder) => folder.dataset.path === target)?.querySelector("summary");
  if (!row) {
    // Only a notice: an answer may be on its way.
    error.textContent = `${target} is no longer in the workspace.`;
    error.classList.remove("hidden");
    return;
  }
  // Unfold every folder above it.
  for (let node = row.parentElement; node && node !== tree; node = node.parentElement) {
    if (node.tagName === "DETAILS") node.open = true;
  }
  row.scrollIntoView({ block: "center" });
  row.focus({ preventScroll: true });
  row.classList.remove("revealed");
  void row.offsetWidth;
  row.classList.add("revealed");
}

/**
 * Append a completed turn, enable answer selection, and show its
 * references.
 *
 * @param turn Saved question, answer, and evidence data.
 */
function addTurn(turn) {
  const item = document.createElement("article");
  item.className = "turn";
  item.dataset.turnIndex = String(conversation.children.length);
  const question = document.createElement("div");
  question.className = "user-message";
  fillQuestion(question, turn.question);
  const answer = document.createElement("div");
  answer.className = "answer-message";
  answer.tabIndex = 0;
  answer.setAttribute("aria-label", "Select this answer to view its references");
  answer.addEventListener("click", () => {
    chooseSidePanels(true);
    selectTurn(item, turn);
  });
  answer.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      chooseSidePanels(true);
      selectTurn(item, turn);
    }
  });
  renderAnswer(answer, turn, item);
  // A saved answer shows the timing and tokens it showed when it arrived.
  const timeline = renderTimeline(turn.timeline || null, turn);
  item.append(...[question, timeline, answer].filter(Boolean));
  addMessageActions(item, turn);
  conversation.append(item);
  selectTurn(item, turn);
  conversationScroll.scrollTop = conversationScroll.scrollHeight;
}

/**
 * Replace the conversation picker using the backend registry.
 *
 * @param data Event containing conversation records and the active
 *   identifier.
 */
function renderConversations(data) {
  lastConversationData = data;
  activeConversationId = data.active_conversation_id || activeConversationId;
  const list = byId("conversation-list");
  list.replaceChildren();
  let activeName = "";
  let activeEmpty = false;
  const mode = sidebarPref("conversations-sort", "updated");
  const orderKey = `conversations-order:${openProjectPath}`;
  const ordered = sidebarConversationOrder(data.conversations || [], mode, sidebarPref(orderKey, []));
  const rows = [];
  for (const item of ordered) {
    if (item.conversation_id === activeConversationId) {
      activeName = item.name;
      activeEmpty = Boolean(item.empty);
    }
    const asking = askingConversation?.id === item.conversation_id;
    if (item.empty && !asking) continue;
    // Until the generated title arrives, label the chat with its question.
    const provisional = asking && /^Conversation \d+$/.test(item.name);
    const name = provisional ? provisionalTitle(askingConversation.question) : item.name;
    const row = document.createElement("div");
    row.className = "conversation-row";
    const button = document.createElement("button");
    button.type = "button";
    const shown = reading ? reading.id : activeConversationId;
    button.className = `conversation-item${item.conversation_id === shown ? " active" : ""}${provisional ? " provisional" : ""}${unreadConversations.has(item.conversation_id) ? " unread" : ""}`;
    button.dataset.conversationId = item.conversation_id;
    button.textContent = name;
    let opening = null;
    button.addEventListener("click", () => {
      if (readConversation(item)) return;
      if (busy || item.conversation_id === activeConversationId) return;
      // Wait briefly so a double-click renames instead of opening.
      clearTimeout(opening);
      opening = setTimeout(() => {
        if (busy) return;
        setBusy(true, "Opening conversation…");
        bridge.selectConversation(item.conversation_id);
      }, RENAME_CLICK_DELAY_MS);
    });
    button.addEventListener("dblclick", (event) => {
      event.preventDefault();
      clearTimeout(opening);
      if (!busy && !provisional) renameInSidebar(row, button, item);
    });
    button.title = `${name}\nDouble-click or right-click to rename`;
    button.addEventListener("contextmenu", (event) => {
      event.preventDefault();
      if (provisional) return;
      showWorkspaceMenu(event.clientX, event.clientY, [
        ["Rename", () => renameInSidebar(row, button, item)],
        ["Delete…", () => remove.click()],
      ]);
    });
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "conversation-delete";
    remove.textContent = "×";
    remove.setAttribute("aria-label", `Delete ${item.name}`);
    remove.title = "Delete conversation";
    remove.addEventListener("click", async () => {
      if (busy) return;
      const deleting = await askKilin(`Delete “${item.name}”?`, "", "Delete");
      if (!deleting || busy) return;
      setBusy(true, "Deleting conversation…");
      bridge.deleteConversation(item.conversation_id);
    });
    row.append(button, remove);
    list.append(row);
    rows.push([row, item.conversation_id]);
  }
  if (mode === "manual") {
    const ids = rows.map(([, id]) => id);
    for (const [row, id] of rows) {
      makeReorderable(row, id, ids, (order) => {
        setSidebarPref(orderKey, order);
        renderConversations(lastConversationData);
      });
    }
  }
  const heading = reading ? reading.name : activeEmpty ? "" : activeName;
  byId("conversation-name").textContent = heading;
  byId("conversation-name").classList.toggle("hidden", !heading);
  updateChatSummary();
}

/**
 * Show a chat read-only while another one is answered, or go back to the
 * answering chat.
 *
 * @param item The clicked chat's record from the backend.
 * @returns Whether the click was handled here; false opens the chat as
 *   usual.
 */
function readConversation(item) {
  const id = item.conversation_id;
  if (reading && id === activeConversationId) {
    returnToAnswer();
    return true;
  }
  if (!answeringConversationId || id === activeConversationId) return false;
  if (reading?.id !== id) {
    // The engine is busy answering, so Qt reads the saved file.
    reading = { ...reading, id, name: item.name };
    bridge.viewConversation(id);
  }
  return true;
}

/**
 * Show the saved turns of the chat being read, keeping the answering chat's
 * content aside.
 *
 * @param turns Saved turns from ``conversation_viewed``, or null when the
 *   chat could not be read.
 */
function showReadConversation(turns) {
  if (!reading.live) {
    reading.live = [...conversation.childNodes];
    reading.scroll = conversationScroll.scrollTop;
    reading.count = currentTurnCount;
  }
  conversation.replaceChildren();
  clearReferences();
  if (turns) {
    turns.forEach(addTurn);
  } else {
    const note = document.createElement("p");
    note.className = "reading-failed";
    note.textContent = "KíLín could not read this conversation.";
    conversation.append(note);
  }
  renderConversations(lastConversationData);
  updateChatSummary(turns?.length || 0);
  updateSendButton();
}

/**
 * Leave the chat being read and show the answering chat again.
 */
function returnToAnswer() {
  if (!reading) return;
  const { live, scroll, count } = reading;
  reading = null;
  if (live) {
    conversation.replaceChildren(...live);
    clearReferences();
    conversationScroll.scrollTop = scroll;
    updateChatSummary(count);
  }
  unreadConversations.delete(activeConversationId);
  renderConversations(lastConversationData);
  updateSendButton();
}

/**
 * Ask a yes/no question in a KíLín dialog instead of the system pop-up.
 *
 * Cancel has focus first, so Enter does not confirm by accident; Escape
 * also cancels.
 *
 * @param title Short question, such as "Delete “Notes”?".
 * @param message One sentence on what KíLín will do; empty hides it.
 * @param action Label of the confirming button, such as "Delete".
 * @param other Label of an optional third button, such as "Keep both".
 * @returns Promise of true when the user confirms, "other" for the third
 *   button, false otherwise.
 */
function askKilin(title, message, action, other = "") {
  const dialog = byId("kilin-ask");
  byId("kilin-ask-title").textContent = title;
  byId("kilin-ask-message").textContent = message;
  byId("kilin-ask-message").classList.toggle("hidden", !message);
  byId("kilin-ask-confirm").textContent = action;
  byId("kilin-ask-other").textContent = other;
  byId("kilin-ask-other").classList.toggle("hidden", !other);
  return new Promise((resolve) => {
    dialog.addEventListener("close", () => resolve(
      dialog.returnValue === "confirm" || (dialog.returnValue === "other" && "other")),
    { once: true });
    dialog.returnValue = "";
    dialog.showModal();
    byId("kilin-ask-cancel").focus();
  });
}

/**
 * Keep the text an answer had generated when the user pressed Stop.
 *
 * A stopped answer is never saved; it stays on screen, marked as stopped,
 * until the user opens another chat.
 *
 * @param current The pending turn that was stopped.
 */
function keepStoppedTurn(current) {
  stoppedTurn = null;
  if (!current) return;
  if (!current.streaming || !current.body?.textContent.trim()) {
    current.item.remove();
    return;
  }
  const elapsed = current.elapsedMs ?? performance.now() - current.started;
  current.item.classList.remove("pending");
  current.item.classList.add("stopped");
  current.answer.classList.remove("answer-phase", "streaming");
  // Keep the steps it went through, ending with the Stop, like "Took …".
  const record = {
    elapsed_ms: elapsed,
    steps: timelineSteps(current.timeline || [], current.started + elapsed),
  };
  const line = renderTimeline(record, { parameters: { usage: current.usage } });
  const stop = document.createElement("li");
  stop.className = "timeline-stopped";
  stop.textContent = "Stopped at your request";
  // The last step, before the token count.
  const list = line.querySelector(".answer-timeline");
  list.insertBefore(stop, list.querySelector(".timeline-usage"));
  const tokens = usageSum(current.usage);
  line.querySelector("summary").textContent = [
    `Stopped after ${formatDuration(elapsed)}`,
    tokens ? `${formatTokens(tokens)} tokens` : "",
  ].filter(Boolean).join(" · ");
  current.durationLine.replaceWith(line);
  current.durationLine = line;
  stoppedTurn = { item: current.item, conversationId: activeConversationId };
}

/**
 * Once the answer has ended, or a restarted engine is ready, turn the answer
 * showing "Stopping…" into a stopped one, kept on screen when it had text.
 */
function finishStopping() {
  if (!pendingTurn?.stopping) return;
  stopPhaseAnimation();
  keepStoppedTurn(pendingTurn);
  pendingTurn = null;
}

/**
 * Show the stopped answer again after the restarted engine redraws its chat.
 */
function restoreStoppedTurn() {
  const kept = stoppedTurn;
  stoppedTurn = null;
  if (kept && kept.conversationId === activeConversationId) conversation.append(kept.item);
}

/**
 * Rename a conversation in place in the sidebar: Enter or clicking away
 * saves, Escape cancels.
 *
 * @param row The conversation's sidebar row.
 * @param button The row's conversation button, replaced while editing.
 * @param item The conversation record from the backend.
 */
function renameInSidebar(row, button, item) {
  const input = document.createElement("input");
  input.type = "text";
  input.className = "conversation-rename";
  input.value = item.name;
  input.setAttribute("aria-label", `Rename ${item.name}`);
  let done = false;
  const finish = (save) => {
    if (done) return;
    done = true;
    const name = input.value.trim();
    if (save && name && name !== item.name && !busy) {
      bridge.renameConversation(item.conversation_id, name);
    }
    // The backend's conversation list redraws the sidebar after a rename.
    input.replaceWith(button);
  };
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") { event.preventDefault(); finish(true); }
    if (event.key === "Escape") { event.preventDefault(); finish(false); }
  });
  input.addEventListener("blur", () => finish(true));
  button.replaceWith(input);
  input.focus();
  input.select();
}

/**
 * Shorten a first question into a temporary sidebar label.
 *
 * @param question The question as typed.
 * @returns Up to six words, without trailing punctuation.
 */
function provisionalTitle(question) {
  const words = question.trim().replace(/[?.!]+$/, "").split(/\s+/);
  return words.slice(0, 6).join(" ").slice(0, 80) + (words.length > 6 ? "…" : "");
}

/**
 * List the active empty chat as soon as its first question is sent.
 *
 * @param question The question as typed.
 */
function showAskingConversation(question) {
  const active = lastConversationData?.conversations?.find(
    (item) => item.conversation_id === activeConversationId);
  if (!active?.empty) return;
  askingConversation = { id: activeConversationId, question };
  renderConversations(lastConversationData);
}

/**
 * Stop listing an unsaved chat, for example after a failed first answer.
 */
function clearAskingConversation() {
  if (!askingConversation) return;
  askingConversation = null;
  if (lastConversationData) renderConversations(lastConversationData);
}

/**
 * Render the project-type welcome screen for an empty conversation.
 */
function renderConversationEmpty() {
  conversation.replaceChildren();
  const type = projectTypes[activeProjectTypeId] || {};
  const content = {
    prompts: type.welcome_prompts?.length ? type.welcome_prompts : genericEmptyChat.welcome_prompts,
    suggestions: type.suggestions?.length ? type.suggestions : genericEmptyChat.suggestions,
  };
  let promptIndex = Math.floor(Math.random() * content.prompts.length);
  if (content.prompts.length > 1 && promptIndex === lastEmptyPrompt[activeProjectTypeId]) {
    promptIndex = (promptIndex + 1) % content.prompts.length;
  }
  lastEmptyPrompt[activeProjectTypeId] = promptIndex;
  const landing = document.createElement("section");
  landing.className = "conversation-empty";
  const prompt = document.createElement("h2");
  prompt.textContent = content.prompts[promptIndex];
  const suggestions = document.createElement("div");
  suggestions.className = "prompt-suggestions";
  // The type's first suggestion is always shown; two others vary.
  const [first, ...rest] = content.suggestions;
  const choices = [first, ...rest.sort(() => Math.random() - 0.5).slice(0, 2)].filter(Boolean);
  for (const text of choices) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = text;
    button.addEventListener("click", () => {
      const question = byId("question");
      question.value = text;
      question.focus();
      // Select the "…" so typing the missing part replaces it.
      const gap = text.indexOf("…");
      if (gap >= 0) question.setSelectionRange(gap, gap + 1);
    });
    suggestions.append(button);
  }
  landing.append(prompt, suggestions);
  conversation.append(landing);
  conversationScroll.scrollTop = 0;
  updateFloatingHeader();
}

/**
 * Render removable chips for source files attached to the next prompt.
 */
function renderAttachments() {
  const host = byId("attachment-list");
  host.replaceChildren();
  for (const path of messageAttachments) {
    const chip = document.createElement("span");
    chip.className = "attachment-chip";
    chip.textContent = path.split(/[\\/]/).pop() || path;
    chip.title = path;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "×";
    remove.setAttribute("aria-label", `Remove ${chip.textContent}`);
    remove.addEventListener("click", () => {
      messageAttachments = messageAttachments.filter((item) => item !== path);
      renderAttachments();
    });
    chip.append(remove);
    host.append(chip);
  }
  host.classList.toggle("hidden", !messageAttachments.length);
}

/**
 * Choose the effort level or the search-depth step for the next answers.
 *
 * @param kind "model" or "depth".
 * @param index Zero-based level or step.
 */
function setEffortLevel(kind, index) {
  const level = Math.max(0, Math.min(4, Number(index)));
  if (kind === "model") selectedModelLevel = level;
  else selectedDepthStep = level;
  renderEffortPicker();
}

/**
 * Show the chosen levels on both sliders and on the composer button.
 */
function renderEffortPicker() {
  const levels = currentLevels();
  const depth = currentDepth();
  if (levels.length !== 5 || depth.length !== 5) return;
  // The effort is named by the model level; search depth is a count.
  byId("effort-level-name").textContent = levels[selectedModelLevel].name;
  showThinkingSwitch(levels[selectedModelLevel].model);
  for (const [kind, index] of [["model", selectedModelLevel], ["depth", selectedDepthStep]]) {
    byId(`effort-${kind}-detail`).textContent = kind === "model"
      ? levels[index].model.split("/").pop()
      : `${depth[index]} passages`;
    const range = byId(`effort-${kind}-range`);
    range.value = String(index);
    range.style.setProperty("--effort-percent", `${(index / (levels.length - 1)) * 100}%`);
    range.parentElement.querySelectorAll(".effort-points i").forEach((point, pointIndex) => {
      point.classList.toggle("selected", pointIndex === index);
    });
  }
  byId("effort-button").textContent =
    `${levels[selectedModelLevel].name} · ${depth[selectedDepthStep]}`;
  byId("effort-button").title =
    `${levels[selectedModelLevel].name}: ${levels[selectedModelLevel].model.split("/").pop()} · search depth ${depth[selectedDepthStep]} passages`;
}

/**
 * Show the Thinking switch under "Lagom ›" as Settings → Answers has it.
 *
 * Both are one setting: the switch only sets the Answers checkbox, which
 * saves it to the open project. Without a detected template prompt
 * difference, the switch remains disabled; capability is not inferred.
 *
 * @param model Model of the chosen effort level.
 */
function showThinkingSwitch(model) {
  const detected = modelHasThinkingSwitch(model);
  const on = byId("thinking").checked;
  for (const input of document.querySelectorAll('input[name="effort-thinking"]')) {
    input.checked = (input.value === "on") === on;
    input.disabled = busy || !detected;
  }
}

/**
 * Clear the interval used by the pending-operation label.
 */
function stopPhaseAnimation() {
  clearInterval(phaseTimer);
  phaseTimer = null;
}

// Each stage stays readable briefly. Routing and retrieval report several
// steps at once; they queue (oldest dropped past the limit) and finish well
// within the time the model spends reading before it writes.
const ANSWER_STAGE_DURATION_MS = 450;
// A click on another chat waits this long, so a double-click renames it.
const RENAME_CLICK_DELAY_MS = 250;
const ANSWER_STAGE_QUEUE_LIMIT = 8;

/**
 * Clear any queued answer-stage labels and the running stage timer.
 */
function resetAnswerStages() {
  clearTimeout(answerStageTimer);
  answerStageTimer = null;
  answerStageKey = null;
  answerStageQueue = [];
}

/**
 * Show the next queued stage and schedule the following stage.
 */
function advanceAnswerStage() {
  if (answerStageTimer) return;
  const stage = answerStageQueue.shift();
  if (!stage) {
    answerStageKey = null;
    return;
  }
  answerStageKey = stage.key;
  showTurnPhase("Generating", stage.detail);
  answerStageTimer = setTimeout(() => {
    answerStageTimer = null;
    answerStageKey = null;
    advanceAnswerStage();
  }, ANSWER_STAGE_DURATION_MS);
}

/**
 * Queue a distinct stage label while a question is being answered.
 *
 * @param key Stable identifier used to deduplicate stage labels.
 * @param detail Human-readable status shown by the activity display.
 */
function queueAnswerStage(key, detail) {
  if (!pendingTurn) return;
  if (answerStageKey === key) return;
  if (answerStageQueue.some((stage) => stage.key === key)) return;
  answerStageQueue.push({ key, detail });
  answerStageQueue = answerStageQueue.slice(-ANSWER_STAGE_QUEUE_LIMIT);
  advanceAnswerStage();
}

/**
 * Display an activity start emitted by the core timeline recorder.
 *
 * @param activity Indexed activity with the core's phase and description.
 */
function markTimelineStep(activity) {
  const timeline = pendingTurn?.timeline;
  if (!timeline || activity.index !== timeline.length) return;
  timeline.push({
    phase: activity.phase,
    label: activity.label,
    at: performance.now(),
  });
}

/**
 * Estimate in-flight step durations for the live activity display.
 *
 * @param timeline Steps with ``label`` and start time ``at``, in order.
 * @param end Time the last step ended, or now for a step still running.
 * @returns Each reported activity's label and estimated duration in
 *   milliseconds. No core activity is filtered by the GUI.
 */
function timelineSteps(timeline, end) {
  return timeline.map((step, index) => ({
    label: step.label,
    ms: Math.round(Math.max(0, (timeline[index + 1]?.at ?? end) - step.at)),
  }));
}

/**
 * Add one row per step, with its duration, to a timeline list.
 *
 * Used both while an answer is made and for the saved "Took …" list, so
 * they look the same.
 *
 * @param list The ``answer-timeline`` list to fill.
 * @param steps Step labels and durations in milliseconds.
 * @param running Whether the last step is still going on.
 */
function appendTimelineRows(list, steps, running = false) {
  steps.forEach((step, index) => {
    const row = document.createElement("li");
    if (running && index === steps.length - 1) row.className = "timeline-running";
    const label = document.createElement("span");
    label.textContent = step.label;
    const time = document.createElement("span");
    time.className = "timeline-time";
    time.textContent = step.ms < 100 ? "<0.1s" : `${(step.ms / 1000).toFixed(1)}s`;
    row.append(label, time);
    list.append(row);
  });
}

/**
 * Build the collapsed "Took Ns · Xk tokens" line; expanding it lists each
 * step and how long it took.
 *
 * @param record Core-owned answer timeline stored with the turn, or null
 *   for an older answer without timing data.
 * @param turn Saved turn with its token usage.
 * @returns The line, or null when there is neither a timing nor tokens.
 */
function renderTimeline(record, turn) {
  const usage = turn.parameters?.usage;
  const tokens = usageSum(usage);
  if (!record && !tokens) return null;
  const details = document.createElement("details");
  details.className = "answer-duration";
  const summary = document.createElement("summary");
  summary.textContent = [
    record ? `Took ${formatDuration(record.elapsed_ms)}` : "",
    tokens ? `${formatTokens(tokens)} tokens` : "",
  ].filter(Boolean).join(" · ");
  if (tokens) {
    summary.title = `${usage.input_tokens.toLocaleString()} tokens read, ${usage.output_tokens.toLocaleString()} written`;
  }
  const list = document.createElement("ol");
  list.className = "answer-timeline";
  appendTimelineRows(list, record?.steps || []);
  if (tokens) {
    const row = document.createElement("li");
    row.className = "timeline-usage";
    const label = document.createElement("span");
    const parameters = turn.parameters || {};
    const model = typeof parameters.chat_model === "string"
      ? parameters.chat_model : "";
    const settings = [];
    if (typeof parameters.thinking === "boolean") {
      settings.push(parameters.thinking ? "thinking on" : "thinking off");
    }
    if (Number.isInteger(parameters.max_new_tokens)
        && parameters.max_new_tokens > 0) {
      settings.push(`${parameters.max_new_tokens.toLocaleString()} max tokens`);
    }
    label.textContent = [
      `Tokens: ${usage.input_tokens.toLocaleString()} read`,
      `${usage.output_tokens.toLocaleString()} written`,
    ].join(", ");
    row.append(label);
    list.append(row);
    if (model) {
      const modelRow = document.createElement("li");
      modelRow.className = "timeline-model";
      const modelLabel = document.createElement("span");
      modelLabel.textContent = `Model: ${model}${settings.length
        ? ` (${settings.join(", ")})` : ""}`;
      modelRow.append(modelLabel);
      list.append(modelRow);
    }
  }
  details.append(summary, list);
  return details;
}

/**
 * Complete an answer immediately and discard pending stage animations.
 *
 * @param turn Completed answer and saved conversation metadata.
 */
function completeAfterAnswerStages(turn) {
  unansweredQuestion = "";
  // Freeze the display timer; final timing comes from the saved turn.
  if (pendingTurn && pendingTurn.elapsedMs === null) {
    pendingTurn.elapsedMs = performance.now() - pendingTurn.started;
  }
  // Show the answer as soon as it arrives; queued stage labels are dropped.
  resetAnswerStages();
  completePendingTurn(turn);
}

/**
 * Show the current backend phase under the pending answer's label.
 *
 * @param label Current backend phase label.
 * @param detail Optional supporting status text.
 */
function showTurnPhase(label, detail = "") {
  // Steps reported just before Stop took effect do not undo "Stopping".
  if (!pendingTurn || pendingTurn.stopping) return;
  pendingTurn.phase = label;
  pendingTurn.detail.textContent = detail;
  pendingTurn.detail.classList.toggle("hidden", !detail);
  paintPendingTurn();
}

/**
 * Format a token count compactly: 850, 3.5k, 42k, 1.2M.
 *
 * @param count Number of tokens.
 * @returns Short human-readable count.
 */
function formatTokens(count) {
  const n = Math.max(0, Number(count) || 0);
  const short = (value, unit) => `${value >= 10 ? Math.round(value) : value.toFixed(1).replace(/\.0$/, "")}${unit}`;
  if (n < 1000) return String(Math.round(n));
  if (n < 1e6) return short(n / 1e3, "k");
  if (n < 1e9) return short(n / 1e6, "M");
  return short(n / 1e9, "B");
}

/**
 * Total tokens of a usage record (read plus written).
 * @param usage Input and output token-count record.
 */
function usageSum(usage) {
  return (Number(usage?.input_tokens) || 0) + (Number(usage?.output_tokens) || 0);
}

/**
 * Show the composer's token pill and its menu, including a running answer.
 */
function renderUsage() {
  const live = pendingTurn?.usage || null;
  const plus = (base) => ({
    input_tokens: (Number(base?.input_tokens) || 0) + (Number(live?.input_tokens) || 0),
    output_tokens: (Number(base?.output_tokens) || 0) + (Number(live?.output_tokens) || 0),
    answers: Number(base?.answers) || 0,
  });
  const show = (key, usage) => {
    const total = usageSum(usage);
    byId(`usage-${key}-total`).textContent = `${formatTokens(total)} tokens`;
    byId(`usage-${key}-answers`).textContent = usage.answers === 1 ? "1 answer" : `${usage.answers} answers`;
    // The bar is full: read (green-grey) then written (orange).
    const read = total ? (100 * usage.input_tokens) / total : 0;
    byId(`usage-${key}-read`).style.width = `${read.toFixed(1)}%`;
    byId(`usage-${key}-written`).style.width = total ? `${(100 - read).toFixed(1)}%` : "0";
    byId(`usage-${key}-read-text`).textContent = `${usage.input_tokens.toLocaleString()} read`;
    byId(`usage-${key}-written-text`).textContent = `${usage.output_tokens.toLocaleString()} written`;
    return total;
  };
  const total = show("project", plus(usageTotals?.project));
  show("chat", plus(usageTotals?.chat));
  byId("usage-button").title = `${formatTokens(total)} tokens used in this project`;
}

/**
 * Format a duration compactly: 8s, 42s, 1m 05s.
 *
 * @param ms Duration in milliseconds.
 * @returns Short human-readable duration.
 */
function formatDuration(ms) {
  const seconds = Math.max(0, Math.round(ms / 1000));
  if (seconds < 60) return `${seconds}s`;
  return `${Math.floor(seconds / 60)}m ${String(seconds % 60).padStart(2, "0")}s`;
}

/**
 * Repaint the pending label (cycling dots) and its elapsed-time counter.
 */
function paintPendingTurn() {
  if (!pendingTurn) return;
  const dots = Math.floor(performance.now() / 400) % 4;
  pendingTurn.label.textContent = `${pendingTurn.phase}${".".repeat(dots)}`;
  if (pendingTurn.stopping) {
    // Shown above a streamed answer too, where the phase label is gone.
    pendingTurn.timer.textContent = `Stopping${".".repeat(dots)}`;
    return;
  }
  const tokens = usageSum(pendingTurn.usage);
  pendingTurn.timer.textContent = formatDuration(
    pendingTurn.elapsedMs ?? performance.now() - pendingTurn.started)
    + (tokens ? ` · ${formatTokens(tokens)} tokens` : "");
  // Live estimates only; the completed turn supplies core timings.
  const list = pendingTurn.steps;
  if (list && pendingTurn.durationLine.open) {
    list.replaceChildren();
    appendTimelineRows(list, timelineSteps(pendingTurn.timeline, performance.now()), true);
  }
}

/**
 * Append a pending question and begin its retrieval-phase indicator.
 *
 * @param question Validated question text submitted by the user.
 */
function addPendingTurn(question) {
  resetAnswerStages();
  activity.classList.add("answer-progress");
  if (conversation.querySelector(".conversation-empty")) conversation.replaceChildren();
  const item = document.createElement("article");
  item.className = "turn pending";
  item.dataset.turnIndex = String(conversation.children.length);
  const questionNode = document.createElement("div");
  questionNode.className = "user-message";
  fillQuestion(questionNode, question);
  const answer = document.createElement("div");
  answer.className = "answer-message answer-phase";
  const phaseHeader = document.createElement("div");
  phaseHeader.className = "answer-phase-header";
  const phaseLabel = document.createElement("span");
  phaseLabel.className = "answer-phase-label";
  phaseHeader.append(phaseLabel);
  const phaseDetail = document.createElement("div");
  phaseDetail.className = "answer-phase-detail";
  answer.append(phaseHeader, phaseDetail);
  // The steps are listed as they happen, folded like the "Took …" summary
  // it becomes when the answer arrives; expanding it follows the route live.
  const durationLine = document.createElement("details");
  durationLine.className = "answer-duration";
  const timer = document.createElement("summary");
  const steps = document.createElement("ol");
  steps.className = "answer-timeline";
  durationLine.append(timer, steps);
  item.append(questionNode, durationLine, answer);
  conversation.append(item);
  pendingTurn = {
    item, answer, label: phaseLabel, timer, steps, durationLine, detail: phaseDetail,
    phase: "Generating", started: performance.now(), elapsedMs: null,
    timeline: [],
  };
  // One ticker drives both the cycling dots and the elapsed time.
  stopPhaseAnimation();
  paintPendingTurn();
  phaseTimer = setInterval(paintPendingTurn, 200);
  queueAnswerStage("starting", "Starting");
  conversationScroll.scrollTop = conversationScroll.scrollHeight;
}

/**
 * Present one actionable prompt before displaying a waiting status.
 *
 * @param card Fully built tool plan or option question.
 */
function showToolPrompt(card) {
  if (!pendingTurn) return;
  pendingTurn.prompt?.remove();
  pendingTurn.prompt = card;
  pendingTurn.answer.append(card);
  resetAnswerStages();
  showTurnPhase("Waiting", "Waiting for your choice");
  byId("activity-text").textContent = "Waiting for your choice";
  byId("activity-detail").textContent = "";
  conversationScroll.scrollTop = conversationScroll.scrollHeight;
  card.querySelector("button")?.focus({ preventScroll: true });
}

/**
 * Submit a prompt once and immediately leave its waiting display.
 *
 * @param card The currently active prompt, not an earlier detached card.
 * @param value Engine plan-choice value or option index.
 * @param message Progress label after the explicit user decision.
 */
function submitToolPrompt(card, value, message) {
  if (!bridge || !pendingTurn || pendingTurn.prompt !== card) return;
  card.querySelectorAll("button").forEach((button) => { button.disabled = true; });
  pendingTurn.prompt = null;
  card.remove();
  error.classList.add("hidden");
  resetAnswerStages();
  queueAnswerStage(`choice:${value}`, message);
  byId("activity-text").textContent = message;
  byId("activity-detail").textContent = "";
  bridge.choosePlan(value);
}

/**
 * Show a tool plan with Continue, Quick, and Cancel under the pending
 * answer. The backend waits for the choice; one click answers it.
 *
 * @param plan Plan fields from kilin.tools.planning.ToolPlan.
 */
function showToolPlan(plan) {
  if (!pendingTurn) return;
  const card = document.createElement("div");
  card.className = "tool-plan";
  const summary = document.createElement("p");
  summary.className = "tool-plan-summary";
  summary.textContent = plan.summary;
  const facts = document.createElement("p");
  facts.className = "tool-plan-facts";
  facts.textContent = [plan.scope, ...(plan.details || [])].join(" · ");
  const duration = (seconds) => (seconds < 1 ? "instant" : `about ${formatDuration(seconds * 1000)}`);
  const network = plan.operation === "network";
  const calls = network ? `${plan.batches_to_run} reference lookups`
    : plan.batches_to_run === 1 ? "1 model call" : `${plan.batches_to_run} model calls`;
  const actions = document.createElement("div");
  actions.className = "tool-plan-actions";
  const choices = [
    [PLAN_CHOICE_CONTINUE, "primary", network ? "Authorize lookup" : "Read everything",
      network ? calls : `${calls} · ${duration(plan.estimated_seconds)}`],
    [PLAN_CHOICE_QUICK, "secondary", plan.quick_label || "Quick sample",
      network ? `Authorize ${plan.quick_items} reference lookups`
        : `${plan.quick_note || `A sample of ${plan.quick_items}`} · ${duration(plan.quick_seconds)}`],
  ];
  const choose = (choice) => submitToolPrompt(
    card, choice, choice === PLAN_CHOICE_CANCEL ? "Cancelling" : "Starting",
  );
  for (const [choice, style, label, note] of choices) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `tool-plan-option ${style}`;
    const title = document.createElement("span");
    title.className = "tool-plan-option-title";
    title.textContent = label;
    const detail = document.createElement("span");
    detail.className = "tool-plan-option-note";
    detail.textContent = note;
    button.append(title, detail);
    button.addEventListener("click", () => choose(choice));
    actions.append(button);
  }
  const cancel = document.createElement("button");
  cancel.type = "button";
  cancel.className = "tool-plan-cancel";
  cancel.textContent = network ? "Stay offline" : "Cancel";
  cancel.addEventListener("click", () => choose(PLAN_CHOICE_CANCEL));
  actions.append(cancel);
  card.append(summary, facts, actions);
  card.setAttribute("role", "group");
  card.setAttribute("aria-label", plan.summary);
  showToolPrompt(card);
}

/**
 * Show a tool's question with one button per option, plus Cancel.
 *
 * @param choice Question and option labels from the tool.
 */
function showToolChoice(choice) {
  if (!pendingTurn) return;
  const card = document.createElement("div");
  card.className = "tool-plan tool-choice";
  const question = document.createElement("p");
  question.className = "tool-plan-summary";
  question.textContent = choice.question;
  const actions = document.createElement("div");
  actions.className = "tool-choice-options";
  const answer = (value, label) => submitToolPrompt(
    card, value, value === "cancel" ? "Cancelling" : `Using ${label}`,
  );
  (choice.options || []).forEach((label, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "secondary";
    button.textContent = label;
    button.addEventListener("click", () => answer(`option:${index}`, label.replace(/ \(.*\)$/, "")));
    actions.append(button);
  });
  const cancel = document.createElement("button");
  cancel.type = "button";
  cancel.className = "secondary tool-choice-cancel";
  cancel.textContent = "Cancel";
  cancel.addEventListener("click", () => answer("cancel", ""));
  actions.append(cancel);
  card.append(question, actions);
  card.setAttribute("role", "group");
  card.setAttribute("aria-label", choice.question);
  showToolPrompt(card);
}

/**
 * Enable pointer and keyboard selection of a completed answer.
 *
 * @param answer Focusable answer element.
 * @param item Parent conversation article.
 * @param turn Completed turn whose references will be displayed.
 */
function bindAnswerSelection(answer, item, turn) {
  answer.tabIndex = 0;
  answer.setAttribute("aria-label", "Select this answer to view its references");
  const choose = () => {
    chooseSidePanels(true);
    selectTurn(item, turn);
  };
  answer.addEventListener("click", choose);
  answer.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      choose();
    }
  });
}

/**
 * Reveal an already-completed answer through a local typing animation.
 *
 * @param turn Completed backend response; the animation is not token
 *   streaming.
 */
function completePendingTurn(turn) {
  stopPhaseAnimation();
  resetAnswerStages();
  if (reading?.live) {
    // Finished while another chat is read: mark the answering chat unread.
    reading.count += 1;
    unreadConversations.add(activeConversationId);
  } else {
    updateChatSummary(currentTurnCount + 1);
  }
  const current = pendingTurn;
  if (!current) {
    addTurn(turn);
    setBusy(false);
    return;
  }
  pendingTurn = null;
  current.item.classList.remove("pending");
  // A completed answer uses only the timeline committed by the core.
  // Local phase timings remain a visual estimate for a running/stopped turn.
  const timeline = renderTimeline(turn.timeline || null, turn);
  if (current.durationLine) {
    if (timeline) current.durationLine.replaceWith(timeline);
    else current.durationLine.remove();
  } else if (timeline) {
    current.item.insertBefore(timeline, current.answer);
  }
  current.durationLine = timeline;
  // The text was shown live as it was generated; now add citation links.
  current.answer.classList.remove("answer-phase", "streaming");
  current.answer.replaceChildren();
  bindAnswerSelection(current.answer, current.item, turn);
  renderAnswer(current.answer, turn, current.item, current.thinkingMs);
  addMessageActions(current.item, turn);
  if (reading?.live) {
    renderConversations(lastConversationData);
  } else {
    selectTurn(current.item, turn);
    conversationScroll.scrollTop = conversationScroll.scrollHeight;
  }
  setBusy(false);
}

/**
 * Put the cursor back in the message box when KíLín is free again, unless
 * the user is typing somewhere else (a rename, a field) or Settings is open.
 */
function focusComposer() {
  const question = byId("question");
  const active = document.activeElement;
  const elsewhere = active && active !== question && active !== document.body
    && (active.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(active.tagName));
  const settingsOpen = !byId("models-panel").classList.contains("hidden");
  if (!ready || question.disabled || elsewhere || settingsOpen) return;
  question.focus();
}

/**
 * Split model output into its thinking and its answer.
 *
 * Thinking appears inside <think>…</think>; a chat template may open the
 * block in the prompt, leaving only the closing tag in the output.
 *
 * @param raw Model output so far.
 * @returns Thinking text, answer text, and whether thinking is still open.
 */
function splitThinking(raw) {
  let thinking = "";
  let text = String(raw || "").replace(/<think>([\s\S]*?)<\/think>/g, (_, inner) => {
    thinking += inner;
    return "";
  });
  const close = text.indexOf("</think>");
  if (close !== -1) {
    thinking = text.slice(0, close) + thinking;
    text = text.slice(close + "</think>".length);
  }
  const open = text.indexOf("<think>");
  if (open !== -1) {
    thinking += text.slice(open + "<think>".length);
    text = text.slice(0, open);
  }
  return { thinking: thinking.trim(), answer: text.replace(/^\s+/, ""), thinkingOpen: open !== -1 };
}

/**
 * Build the collapsible box that shows the model's thinking.
 *
 * @param text Thinking text.
 * @param label Summary line, such as "Thought for 12s".
 * @param open Whether the box starts expanded.
 */
function thoughtsBlock(text, label, open = false) {
  const details = document.createElement("details");
  details.className = "answer-thoughts";
  details.open = open;
  const summary = document.createElement("summary");
  summary.textContent = label;
  // Expanding the thoughts should not also select the answer.
  summary.addEventListener("click", (event) => event.stopPropagation());
  const body = document.createElement("div");
  body.className = "answer-thoughts-text";
  body.textContent = text;
  details.append(summary, body);
  return details;
}

/**
 * Show generated text live in the pending answer.
 *
 * @param text Newly generated text.
 */
function appendAnswerText(text) {
  const current = pendingTurn;
  if (!current || !text) return;
  current.raw = (current.raw || "") + text;
  const parts = splitThinking(current.raw);
  const thinkingStarted = parts.thinking || parts.thinkingOpen;
  if (!current.streaming && !thinkingStarted && !parts.answer) return;
  if (!current.streaming) {
    current.streaming = true;
    resetAnswerStages();
    current.answer.classList.remove("answer-phase");
    current.answer.classList.add("streaming");
    current.answer.replaceChildren();
    current.body = document.createElement("div");
    current.body.className = "answer-body";
    current.answer.append(current.body);
    paintPendingTurn();
  }
  if (thinkingStarted) {
    if (!current.thoughts) {
      current.thinkingStarted = performance.now();
      current.thoughts = thoughtsBlock("", "Thinking…", true);
      current.answer.prepend(current.thoughts);
    }
    const body = current.thoughts.querySelector(".answer-thoughts-text");
    body.textContent = parts.thinking;
    body.scrollTop = body.scrollHeight;
    if (parts.thinkingOpen) current.thinkingSeen = true;
    if (!parts.thinkingOpen && !current.thinkingDone) {
      current.thinkingDone = true;
      // Thinking that arrives finished in one piece, as from a tool's
      // model calls, took an unknown time: it is not "Thought for 0s".
      if (current.thinkingSeen) current.thinkingMs = performance.now() - current.thinkingStarted;
      current.thoughts.open = false;
      current.thoughts.querySelector("summary").textContent = current.thinkingMs == null
        ? "Show thinking" : `Thought for ${formatDuration(current.thinkingMs)}`;
    }
  }
  current.body.replaceChildren();
  renderMarkdown(current.body, parts.answer);
  const nearBottom = conversationScroll.scrollHeight - conversationScroll.scrollTop
    - conversationScroll.clientHeight < 120;
  // The chat being read keeps its own scroll position.
  if (nearBottom && !reading?.live) conversationScroll.scrollTop = conversationScroll.scrollHeight;
}

/**
 * Dispatch a JSON event from the Qt bridge into interface state changes.
 *
 * @param raw JSON string using the engine event protocol; parsing errors
 *   propagate.
 */
function onEvent(raw) {
  const data = JSON.parse(raw);
  if (data.event === "trace_ready") {
    activeTracePath = data.trace_path || "";
    markActiveConversationFile();
  } else if (data.event === "backend_preference_saved") {
    confirmedBackendPreference = data.backend;
    // Qt requests a fresh readiness check only after its write succeeds.
    // Do not use the former backend's status while that check is pending.
    kilinReady = false;
    readinessEnvironmentReady = false;
    readinessMissingCount = 0;
    byId("readiness-status").textContent = "Checking";
    byId("readiness-summary").textContent = "Checking the selected backend…";
    showSetupLock();
  } else if (data.event === "backend_preference_failed") {
    // A second selection may already be pending: only revert the failed one.
    if (!ready && byId("backend").value === data.backend) {
      byId("backend").value = confirmedBackendPreference;
      showModelFields();
    }
    error.textContent = data.message;
    error.classList.remove("hidden");
    byId("model-download-status").textContent = data.message;
  } else if (data.event === "terms_acceptance_saved") {
    if (!acceptingTerms || data.version !== pendingTermsVersion) return;
    acceptingTerms = false;
    pendingTermsVersion = "";
    if (data.version !== termsVersion) {
      // The page now shows different terms than the version Qt saved.
      termsAccepted = false;
      acknowledgedTermsVersion = "";
      termsVersion = "";
      byId("terms-agree").checked = false;
      setWelcomeError("The terms have changed. Review and accept them again.");
      bridge?.requestReadiness();
    } else {
      acknowledgedTermsVersion = data.version;
      termsAccepted = true;
      setWelcomeError("");
      byId("welcome").close?.();
    }
    showSetupConfirmations();
  } else if (data.event === "terms_acceptance_failed") {
    if (!acceptingTerms || data.version !== pendingTermsVersion) return;
    acceptingTerms = false;
    pendingTermsVersion = "";
    setWelcomeError(data.message || "Could not save the agreement. Try again.");
    if (data.reason === "outdated_version") {
      // Do not reuse a checkbox checked for terms that are no longer current.
      termsAccepted = false;
      acknowledgedTermsVersion = "";
      termsVersion = "";
      byId("terms-agree").checked = false;
      bridge?.requestReadiness();
    }
    showSetupConfirmations();
  } else if (data.event === "readiness") {
    requiredModels = data.required_models || [];
    readinessEnvironmentReady = Boolean(data.environment_ready);
    installCommand = data.install_command || installCommand;
    installCommands = data.install_commands || {
      [data.backend]: installCommand,
    };
    cachedModels = new Set(data.cached_models || []);
    const card = byId("first-run-setup");
    const steps = byId("readiness-steps");
    steps.replaceChildren();
    const runtime = document.createElement("li");
    runtime.textContent = data.environment_ready
      ? `${String(data.backend).toUpperCase()} runtime is ready`
      : data.unavailable_reason
        || `${String(data.backend).toUpperCase()} runtime is missing; install it with ${installCommands[data.backend] || "the KíLín installer"}, then restart KíLín`;
    runtime.classList.toggle("complete", data.environment_ready);
    steps.append(runtime);
    const replacements = data.replacements || {};
    for (const modelId of requiredModels) {
      const item = document.createElement("li");
      const present = cachedModels.has(modelId);
      item.textContent = present
        ? `Ready: ${modelId}`
        : replacements[modelId]
          ? `Missing: ${modelId} (installed alternative: ${replacements[modelId]})`
          : `Missing: ${modelId}`;
      item.classList.toggle("complete", present);
      // Found again while downloading, to show that model's progress.
      item.dataset.model = modelId;
      steps.append(item);
    }
    // A readiness reply can predate a Qt terms-write acknowledgement.
    folderConfirmed = Boolean(data.folder_confirmed);
    const nextTermsVersion = data.terms_version || "";
    if (termsVersion && termsVersion !== nextTermsVersion) {
      byId("terms-agree").checked = false;
    }
    termsVersion = nextTermsVersion;
    const savedReadiness = acknowledgedTermsVersion === termsVersion
      && Boolean(acknowledgedTermsVersion);
    const termsJustAccepted = savedReadiness && Boolean(data.terms_accepted);
    if (termsJustAccepted || (acknowledgedTermsVersion && !savedReadiness)) {
      acknowledgedTermsVersion = "";
    }
    if (!acceptingTerms) {
      termsAccepted = Boolean(data.terms_accepted) || savedReadiness;
    }
    const folderStep = document.createElement("li");
    folderStep.textContent = folderConfirmed
      ? `Projects folder: ${data.projects_folder}`
      : "Confirm the projects folder below";
    folderStep.classList.toggle("complete", folderConfirmed);
    const termsStep = document.createElement("li");
    termsStep.textContent = termsAccepted
      ? "Terms of use accepted"
      : "Accept the terms of use";
    termsStep.classList.toggle("complete", termsAccepted);
    steps.append(folderStep, termsStep);
    renderTerms(data.terms || "");
    byId("copyright").textContent = data.copyright || "";
    // The terms dialog closes only on a confirmed write, not on readiness.
    if (!termsAccepted && termsVersion) showWelcome();
    else if (termsAccepted && !acceptingTerms) byId("welcome").close?.();
    const setupReady = Boolean(data.ready) && folderConfirmed && termsAccepted;
    if (folderConfirmed && termsAccepted) clearSetupReminder();
    // Always available; it opens by itself when something is missing.
    card.classList.remove("hidden");
    if (!setupReady) card.open = true;
    const missing = data.missing_models.length + (data.environment_ready ? 0 : 1)
      + (folderConfirmed ? 0 : 1) + (termsAccepted ? 0 : 1);
    byId("readiness-status").textContent = setupReady ? "Ready" : `${missing} missing`;
    byId("readiness-status").classList.toggle("ready", setupReady);
    byId("readiness-summary").textContent = setupReady
      ? "KíLín is ready."
      : data.ready
        ? !folderConfirmed && !termsAccepted
          ? "Confirm the projects folder and accept the terms of use to finish setup."
          : !folderConfirmed
            ? "Confirm the projects folder to finish setup."
            : "Accept the terms of use to finish setup."
      : !data.environment_ready
        ? data.unavailable_reason || `${String(data.backend).toUpperCase()} runtime is not installed.`
      : data.missing_models.length === 1
        ? "1 required model needs to be downloaded."
        : `${data.missing_models.length} required models need to be downloaded.`;
    // The user's last setup step completed it: back to the normal view.
    const setupJustFinished = setupReady && !kilinReady;
    kilinReady = setupReady;
    readinessMissingCount = data.missing_models.length;
    byId("install-required").classList.toggle("hidden", !data.missing_models.length);
    // Compatible models already on this computer can be used instead.
    byId("use-installed").classList.toggle("hidden", !data.can_use_installed);
    showSetupLock();
    refreshModelStatus();
    if (!setupReady) {
      // Explain missing new-project requirements without locking Settings.
      byId("models-panel").classList.remove("hidden");
      byId("first-run-setup").focus?.();
      if (termsJustAccepted && !folderConfirmed) pointToProjectsFolder();
    } else if (setupJustFinished) {
      card.open = false;
      closeModels();
    }
  } else if (data.event === "installed_models_used") {
    savedLevels = data.effort_levels || savedLevels;
    savedRouters = data.router_models || savedRouters;
    savedDepth = data.search_depth || savedDepth;
    if (profile && data.embedding_models) {
      for (const [backend, model] of Object.entries(data.embedding_models)) profile[`embedding_${backend}`] = model;
    }
    showModelFields();
    setBusy(false);
    const used = Object.entries(data.replacements || {}).map(([from, to]) => `${to} for ${from}`);
    byId("readiness-time").textContent = used.length ? `Using ${used.join("; ")}.` : "";
  } else if (data.event === "installation_started") {
    setupCompletionVisible = false;
    setBusy(true, `Preparing ${data.total} required models…`);
    byId("first-run-setup").classList.remove("hidden");
    byId("first-run-setup").open = true;
    byId("readiness-status").textContent = "Downloading";
    byId("readiness-status").classList.remove("ready");
    installingModels = true;
    showSetupLock();
    byId("readiness-time").textContent = "Starting download. Remaining time depends on model size and network speed.";
    updateDownloadMeter(null);
    // One model downloads at a time; the others wait their turn.
    for (const modelId of data.models || []) showModelStep(modelId, `Waiting: ${modelId}`);
  } else if (data.event === "installation_progress") {
    const progress = downloadProgress(data);
    const percent = progress.percent == null ? "" : ` · ${progress.percent.toFixed(1)}%`;
    byId("readiness-summary").textContent = `Downloading ${data.index} of ${data.total}${percent}`;
    byId("readiness-time").textContent = `${progress.text} · ${data.model_id}`;
    byId("activity-percent").textContent = progress.percent == null
      ? `Model ${data.index} of ${data.total}`
      : `${progress.percent.toFixed(1)}% · model ${data.index} of ${data.total}`;
    byId("activity-text").textContent = `Downloading ${data.model_id}`;
    byId("activity-detail").textContent = progress.text;
    updateDownloadMeter(progress.percent);
    showModelStep(data.model_id, `Downloading: ${data.model_id}${percent}`);
  } else if (data.event === "installation_model_complete") {
    cachedModels.add(data.model_id);
    showModelStep(data.model_id, `Ready: ${data.model_id}`, true);
    byId("readiness-time").textContent = `${data.model_id} completed in ${seconds(data.model_seconds)} · ${seconds(data.elapsed_seconds)} total`;
  } else if (data.event === "installation_complete") {
    installingModels = false;
    setupCompletionVisible = true;
    byId("readiness-status").textContent = "Ready";
    byId("readiness-status").classList.add("ready");
    byId("readiness-summary").textContent = "Required models are installed.";
    byId("readiness-time").textContent = `Setup completed in ${seconds(data.elapsed_seconds)}.`;
    updateDownloadMeter(100);
    setBusy(false);
  } else if (data.event === "settings") {
    showSettings(data);
  } else if (data.event === "projects") {
    renderSavedProjects(data.projects || []);
    renderSidebarProjects(savedProjects, openProjectPath);
  } else if (data.event === "conversation_delete_started") {
    const button = [...document.querySelectorAll(".conversation-item")]
      .find((item) => item.dataset.conversationId === data.conversation_id);
    button?.closest(".conversation-row")?.remove();
    byId("activity-text").textContent = "Conversation removed. Preserving its trace and opening the next chat…";
    byId("activity-detail").textContent = "The archived trace remains in logs/traces/.deleted/.";
  } else if (data.event === "preview") {
    data.roots.forEach((root, index) => {
      const original = selectedSources[index];
      const previous = checkedFiles.get(original) || checkedFiles.get(root.path);
      selectedSources[index] = root.path;
      if (original !== root.path) {
        previewRoots.delete(original);
        checkedFiles.delete(original);
      }
      previewRoots.set(root.path, root.files);
      previewSkipped.set(root.path, root.skipped || []);
      const offered = [
        ...root.files,
        ...(root.skipped || []).filter((item) => isOversizedSkippedFile(item)).map((item) => item.path),
      ];
      checkedFiles.set(
        root.path,
        previous ? new Set(offered.filter((file) => previous.has(file))) : new Set(root.files),
      );
    });
    renderPreview();
    setBusy(false);
  } else if (data.event === "status") {
    setBusy(true, data.message);
  } else if (data.event === "answer_delta") {
    appendAnswerText(data.text || "");
  } else if (data.event === "answer_activity") {
    // The core emits exactly the activities persisted with the turn.
    // Phase messages update status, but must not invent timeline rows.
    markTimelineStep(data);
    queueAnswerStage(`activity:${data.index}`, data.label);
  } else if (data.event === "answer_phase") {
    activity.classList.add("answer-progress");
    if (data.phase === ANSWER_PHASE_ROUTING) {
      byId("activity-text").textContent = "Identifying what the question asks for";
      byId("activity-detail").textContent = "";
    } else if (data.phase === ANSWER_PHASE_ROUTED) {
      byId("activity-text").textContent = "Selecting conversation context";
      byId("activity-detail").textContent = "";
    } else if (data.phase === ENGINE_PHASE_LOADING_MODEL) {
      const name = String(data.model_id || "the model").split("/").pop();
      byId("activity-text").textContent = `Loading ${name}`;
      byId("activity-detail").textContent = "";
    } else if (data.phase === ANSWER_PHASE_RETRIEVING) {
      byId("activity-text").textContent = data.message || "Searching project evidence";
      byId("activity-detail").textContent = "";
    } else if (data.phase === ANSWER_PHASE_PROCESSING && data.choice) {
      showToolChoice(data.choice);
    } else if (data.phase === ANSWER_PHASE_PROCESSING && data.plan) {
      showToolPlan(data.plan);
    } else if (data.phase === ANSWER_PHASE_PROCESSING) {
      byId("activity-text").textContent = data.message || "Working";
      byId("activity-detail").textContent = "";
    } else if (data.phase === ANSWER_PHASE_EVIDENCE_READY) {
      byId("activity-text").textContent = data.message || "Preparing answer";
      byId("activity-detail").textContent = "";
    } else if (data.phase === ANSWER_PHASE_GENERATING) {
      byId("activity-text").textContent = data.message || "Generating answer";
      byId("activity-detail").textContent = "";
    } else if (data.phase === ANSWER_PHASE_SAVING) {
      byId("activity-text").textContent = data.message || "Saving conversation";
      byId("activity-detail").textContent = "";
    } else if (data.phase === ANSWER_PHASE_COMPLETED) {
      stopPhaseAnimation();
      byId("activity-text").textContent = data.message || "Answer ready";
      byId("activity-detail").textContent = "";
    }
  } else if (data.event === "model_download_started") {
    byId("model-download-status").textContent = `Downloading ${data.model_id}…`;
  } else if (data.event === "model_download_progress") {
    const progress = downloadProgress(data);
    byId("model-download-status").textContent = `${data.model_id} · ${progress.text}`;
    byId("activity-percent").textContent = progress.percent == null
      ? "Connecting…"
      : `${progress.percent.toFixed(1)}% downloaded`;
    byId("activity-text").textContent = `Downloading ${data.model_id}`;
    byId("activity-detail").textContent = progress.text;
  } else if (data.event === "models_applied") {
    applyingModels = false;
    cachedModels = new Set(data.cached_models || [...cachedModels]);
    modelRoles = data.model_roles || modelRoles;
    thinkingSwitches = data.thinking_switches || thinkingSwitches;
    // Without an open project, keep the levels for project creation.
    if (data.effort_levels) savedLevels = data.effort_levels;
    else savedLevels = { ...savedLevels, [activeBackend]: cloneLevels(draftLevels) };
    if (data.router_models) savedRouters = data.router_models;
    else savedRouters = { ...savedRouters, [activeBackend]: draftRouter };
    savedDepth = data.search_depth || [...draftDepth];
    if (data.profile) { profile = data.profile; byId("query-format").value = profile.query_format; showModelFields(); }
    else saveModelFields();
    modelsBaseline = currentModelSettings();
    byId("model-download-status").textContent = ready
      ? "Saved in profile.ini. The next answer uses it."
      : "Saved in preferences.json for new projects.";
    setBusy(false);
    renderModelsPanel();
    if (closeAfterModelsApply) hideModels();
    closeAfterModelsApply = false;
  } else if (data.event === "answer_settings_saved") {
    answerSettingsSaved(data);
  } else if (data.event === "answer_settings_failed") {
    answerSettingsFailed(data);
  } else if (data.event === "source_cards_applied"
      || data.event === "source_cards_regenerated") {
    applyingSourceCards = false;
    if (data.source_cards) showSourceCardSettings(data.source_cards);
    showSourceCardSummary(data.source_card_summary);
    byId("source-card-progress").textContent = "Card indexing complete.";
    byId("model-download-status").textContent =
      data.event === "source_cards_applied"
        ? "Saved source-card settings in profile.ini."
        : "Source cards regenerated; other indexes unchanged.";
    setBusy(false);
  } else if (data.event === "indexing_applied") {
    applyingIndexing = false;
    profile = data.profile;
    showIndexingFields();
    byId("model-download-status").textContent = "Saved in profile.ini and the index is rebuilt. The next answer uses it.";
    setBusy(false);
  } else if (data.event === "model_downloaded") {
    cachedModels.add(data.model_id);
    thinkingSwitches = data.thinking_switches || thinkingSwitches;
    downloadingModel = null;
    byId("cancel-model-download").classList.add("hidden");
    byId("model-download-status").textContent = `${data.model_id} downloaded in ${seconds(data.elapsed_seconds || 0)}.`;
    if (!applyingModels) setBusy(false);
    refreshModelStatus();
    if (!byId("models-panel").classList.contains("hidden")) renderModelsPanel();
  } else if (data.event === "usage_progress") {
    if (pendingTurn) {
      pendingTurn.usage = { input_tokens: data.input_tokens, output_tokens: data.output_tokens };
      paintPendingTurn();
      renderUsage();
    }
  } else if (data.event === "progress") {
    showProgress(data);
  } else if (data.event === "opened") {
    finishStopping();
    reading = null;
    // An error from an earlier attempt no longer applies to this project.
    clearError();
    usageTotals = data.usage || usageTotals;
    renderUsage();
    ready = true;
    if (data.profile) {
      profile = data.profile;
      byId("query-format").value = profile.query_format;
      savedLevels = data.effort_levels || savedLevels;
      savedRouters = data.router_models || savedRouters;
      savedDepth = data.search_depth || savedDepth;
      // Settings shows this project's own values, never start-screen edits.
      byId("backend").value = profile.backend;
      showIndexingFields();
      showModelFields();
      if (data.source_cards) showSourceCardSettings(data.source_cards);
    }
    showSourceCardSummary(data.source_card_summary);
    byId("source-card-progress").textContent = "";
    activeProjectTypeId = data.type_id;
    // Reopening the same project need not discard a queued save that has
    // not yet been acknowledged. A different project starts fresh.
    const keepPendingAnswers = openProjectPath === data.project
      && Boolean(answerSettingsSave.pending);
    // Set before the lists are drawn: chats keep their order per project.
    openProjectPath = data.project;
    messageAttachments = [];
    renderAttachments();
    selectedSources.length = 0;
    previewRoots.clear();
    previewSkipped.clear();
    checkedFiles.clear();
    renderPreview();
    setup.classList.add("hidden");
    chat.classList.remove("hidden");
    shell.classList.add("chat-open");
    byId("conversations-panel").classList.remove("hidden");
    showConversations(true);
    conversation.replaceChildren();
    clearReferences();
    renderConversations(data);
    data.turns.forEach(addTurn);
    restoreStoppedTurn();
    // A stopped first answer stays on screen instead of the welcome.
    if (!conversation.children.length) renderConversationEmpty();
    byId("workspace-location").textContent = data.project;
    renderSidebarProjects(savedProjects, openProjectPath);
    // A renamed project keeps Settings open: show its new name.
    showProjectSettings();
    if (bridge) bridge.listSavedProjects("");
    renderWorkspace(data.workspace_files);
    // Open by default, unless the user closed the panel.
    showWorkspace(sidePanelChoice ?? true);
    if (!keepPendingAnswers) resetAnswerSettings(data);
    // The open project now owns its Answers settings.
    answerSettingsEdited = false;
    updateChatSummary(data.turns.length);
    updateFloatingHeader();
    setBusy(false);
    // The attached files are now in sources and indexed.
    if (attachedAsk) {
      const ask = attachedAsk;
      attachedAsk = null;
      sendQuestion(ask);
    }
  } else if (data.event === "conversation_opened") {
    finishStopping();
    reading = null;
    unreadConversations.delete(data.active_conversation_id);
    // An error from an earlier attempt no longer applies to this conversation.
    clearError();
    usageTotals = data.usage || usageTotals;
    renderUsage();
    stopPhaseAnimation();
    resetAnswerStages();
    clearInterval(answerTimer);
    answerTimer = null;
    pendingTurn = null;
    messageAttachments = [];
    renderAttachments();
    conversation.replaceChildren();
    clearReferences();
    renderConversations(data);
    if (data.workspace_files) renderWorkspace(data.workspace_files);
    data.turns.forEach(addTurn);
    restoreStoppedTurn();
    // A stopped first answer stays on screen instead of the welcome.
    if (!conversation.children.length) renderConversationEmpty();
    if (!answerSettingsSave.pending) resetAnswerSettings(data);
    // The open project now owns its Answers settings.
    answerSettingsEdited = false;
    updateChatSummary(data.turns.length);
    updateFloatingHeader();
    setBusy(false);
    if (branchDraft) {
      // A branch from a question: edit it and send it again.
      byId("question").value = branchDraft;
      byId("question").focus();
    }
    branchDraft = null;
  } else if (data.event === "conversations") {
    renderConversations(data);
    if (data.workspace_files) renderWorkspace(data.workspace_files);
    setBusy(false);
  } else if (data.event === "reference_preview") {
    renderSourcePreview(data);
  } else if (data.event === "reference_error") {
    if (activeReference && data.turn_index === activeReference.turnIndex
        && data.citation === activeReference.citation) {
      activeReference.note.textContent = data.message;
    }
  } else if (data.event === "source_changes") {
    showSourceSync(data);
  } else if (data.event === "projects_folder") {
    showProjectsFolder(data);
  } else if (data.event === "projects_folder_cancelled") {
    finishProjectsFolderSwitch();
  } else if (data.event === "projects_folder_failed") {
    finishProjectsFolderSwitch(data.message);
  } else if (data.event === "sources_checked") {
    if (attachedAsk) return addAttachedSources(data);
    setBusy(false);
    showSourceAdd(data);
  } else if (data.event === "folder_changed") {
    renderWorkspace(data.workspace_files);
    setBusy(false);
  } else if (data.event === "answered") {
    if (data.usage) usageTotals = data.usage;
    completeAfterAnswerStages(data.turn);
    renderUsage();
    if (data.workspace_files) renderWorkspace(data.workspace_files);
  } else if (data.event === "tool_settings") {
    renderToolSettings(data);
  } else if (data.event === "conversation_viewed") {
    if (reading?.id === data.conversation_id) showReadConversation(data.turns);
  } else if (data.event === "answer_cancelled") {
    // Stopped cleanly: nothing was saved, and the project stays open.
    resumeAfterStop = false;
    if (data.usage) usageTotals = data.usage;
    renderUsage();
    finishStopping();
    // The stopped answer is already on screen; there is no restart to
    // show it again after.
    stoppedTurn = null;
    restoreUnansweredQuestion();
    setBusy(false);
  } else if (data.event === "reset") {
    renderConversationEmpty();
    clearReferences();
    updateChatSummary(0);
    updateFloatingHeader();
    setBusy(false);
  } else if (data.event === "stopped") {
    if (answerSettingsSave.pending) answerSettingsSave.uncertain = true;
    answerSettingsSave.pending = null;
    answerSettingsSave.queued = null;
    restoreUnansweredQuestion();
    setupCompletionVisible = false;
    byId("readiness-meter").classList.add("hidden");
    byId("readiness-meter").classList.remove("indeterminate");
    if (downloadingModel) {
      byId("model-download-status").textContent = "Download stopped. Reopen Models to check what is cached.";
      downloadingModel = null;
      byId("cancel-model-download").classList.add("hidden");
    }
    if (installingModels) {
      // Check again what is installed; setup stays open until it is done.
      installingModels = false;
      showSetupLock();
      bridge?.requestReadiness();
    }
    if (resumeAfterStop) {
      resumeAfterStop = false;
      resetAnswerStages();
      ready = false;
      // The answer keeps showing "Stopping…" while the project reopens; the
      // reopening steps stay hidden, as during the answer.
      setBusy(true, "Reopening the chat after Stop…");
      bridge.resumeProjectConversation(openProjectPath, activeConversationId);
      return;
    }
    ready = false;
    updateFloatingHeader();
    chat.classList.add("hidden");
    shell.classList.remove("chat-open");
    byId("conversations-panel").classList.add("hidden");
    byId("toggle-conversations").classList.add("hidden");
    setup.classList.remove("hidden");
    showWorkspace(false);
    clearReferences();
    // Stop preserves any committed or nonempty project directory.
    const preservedProjects = data.preserved_projects || [];
    byId("connection").textContent = preservedProjects.length
      ? "Stopped · project files preserved"
      : data.removed_projects?.length
        ? "Stopped · empty preparation folder removed"
        : "Stopped · reopen your project";
    if (preservedProjects.length) {
      const locations = preservedProjects.join(", ");
      error.textContent = `Project files were kept at: ${locations}. `
        + "Reopen a completed project or inspect an unfinished folder "
        + "before removing it.";
      error.classList.remove("hidden");
    }
    setBusy(false);
    bridge.requestSettings(byId("type-id").value, "");
    if (data.removed_projects?.length || preservedProjects.length) {
      bridge.listSavedProjects("");
    }
  } else if (data.event === "backend_failed") {
    answerSettingsSave.pending = null;
    answerSettingsSave.queued = null;
    answerSettingsSave.uncertain = true;
    ready = false;
    updateFloatingHeader();
    chat.classList.add("hidden");
    shell.classList.remove("chat-open");
    byId("conversations-panel").classList.add("hidden");
    byId("toggle-conversations").classList.add("hidden");
    setup.classList.remove("hidden");
    showWorkspace(false);
    clearReferences();
    showError(data.message, data.trace_path || activeTracePath);
  } else if (data.event === "backend_restarted") {
    answerSettingsSave.pending = null;
    answerSettingsSave.queued = null;
    answerSettingsSave.uncertain = true;
    restoreUnansweredQuestion();
    const reopenProject = ready && openProjectPath && activeConversationId;
    error.classList.add("hidden");
    setupCompletionVisible = true;
    byId("readiness-time").textContent = data.message;
    byId("model-download-status").textContent = data.message;
    downloadingModel = null;
    byId("cancel-model-download").classList.add("hidden");
    ready = false;
    setBusy(false);
    byId("connection").textContent = "Ready";
    bridge.requestSettings(byId("type-id").value, "");
    bridge.requestReadiness();
    if (reopenProject) {
      setBusy(true, "Reopening the project after backend recovery…");
      bridge.resumeProjectConversation(openProjectPath, activeConversationId);
    }
  } else if (data.event === "workspace_conflict") {
    // Creation never reuses a project; opening one is a separate decision.
    setBusy(false);
    error.textContent = data.message;
    error.classList.remove("hidden");
    const asking = data.existing_project
      ? askKilin("Open the existing project?",
        `A project already exists at ${data.workspace}. KíLín opens it without the new sources and settings; Cancel to choose another place.`,
        "Open project")
      : Promise.resolve(false);
    asking.then((resume) => {
      if (resume) {
        setBusy(true, "Opening the existing project...");
        bridge.resumeProject(data.workspace);
      } else {
        openModels();
        byId("workspace-path").focus();
      }
    });
  } else if (data.event === "error") {
    // A failed switch may already have released the previous session.
    // Do not show its stale conversation or claim it is still active.
    if (data.resume_released) returnToLaunch();
    if (typeof applyingSourceCards !== "undefined" && applyingSourceCards) {
      applyingSourceCards = false;
      if (sourceCardSettings) showSourceCardSettings(sourceCardSettings);
    }
    if (applyingIndexing) {
      // Not saved: show the settings still in effect.
      applyingIndexing = false;
      showIndexingFields();
    }
    if (data.pending_choice === true) {
      // The engine rejected an unrelated command, not this answer.
      // It is still waiting; preserve the controls and busy state.
      error.textContent = data.message;
      error.classList.remove("hidden");
      pendingTurn?.prompt?.querySelector("button")?.focus({ preventScroll: true });
      return;
    }
    if (attachedAsk) {
      // Adding the attached files failed: give the question back.
      byId("question").value = attachedAsk.question;
      messageAttachments = attachedAsk.attachments;
      attachedAsk = null;
      renderAttachments();
    }
    branchDraft = null;
    showError(data.message, data.trace_path || activeTracePath);
  }
}

// UI handlers validate user input before issuing bridge commands.
byId("browse-workspace").addEventListener("click", () => changeProjectsFolder(false));
byId("default-workspace").addEventListener("click", () => changeProjectsFolder(true));
byId("use-legacy-workspace").addEventListener(
  "click", useLegacyProjectsFolder,
);
byId("confirm-workspace").addEventListener("click", () => {
  if (!bridge || busy) return;
  if (ready) return changeProjectsFolder(false);
  clearSetupReminder();
  // The engine restarts with the confirmed folder, then setup is rechecked.
  switchingProjectsFolder = true;
  setBusy(true, "Confirming the projects folder…");
  bridge.confirmProjectsFolder();
});
byId("terms-agree").addEventListener("change", () => {
  if (byId("terms-agree").checked) byId("welcome-hint").classList.add("hidden");
  showSetupConfirmations();
});
byId("accept-terms").addEventListener("click", () => {
  if (termsAccepted) return byId("welcome").close();
  if (acceptingTerms || !bridge || !termsVersion || !byId("terms-agree").checked) return;
  // Only the Qt write acknowledgement permits closing the dialog.
  acceptingTerms = true;
  pendingTermsVersion = termsVersion;
  setWelcomeError("");
  showSetupConfirmations();
  bridge.acceptTerms(pendingTermsVersion);
});
// Escape cannot skip the terms before they are accepted. The browser may
// close a dialog opened without a click despite a cancelled "cancel", so
// Escape is stopped first and a dialog closed anyway is reopened.
byId("welcome").addEventListener("keydown", (event) => {
  if (event.key !== "Escape" || termsAccepted) return;
  event.preventDefault();
  remindWelcome();
});
byId("welcome").addEventListener("cancel", (event) => {
  if (termsAccepted) return;
  event.preventDefault();
  remindWelcome();
});
byId("welcome").addEventListener("close", () => {
  if (termsAccepted || !termsVersion) return;
  byId("welcome").showModal();
  remindWelcome();
});
byId("view-terms").addEventListener("click", showWelcome);
for (const link of document.querySelectorAll("#welcome .external-link")) {
  // Opened by the system browser, never inside KíLín.
  link.addEventListener("click", () => bridge?.openExternalLink(link.dataset.url));
}
byId("project-title").addEventListener("change", renameProjectFromSettings);
byId("project-title").addEventListener("keydown", (event) => {
  if (event.key === "Enter") { event.preventDefault(); event.target.blur(); }
  if (event.key === "Escape") {
    // Escape puts the name back; it does not close Settings here.
    event.preventDefault();
    event.stopPropagation();
    event.target.value = projectName(openProjectPath);
    event.target.blur();
  }
});
byId("delete-project").addEventListener("click", deleteOpenProject);
// Enter or leaving the field saves an open project's indexing settings;
// the number arrows only edit, so each click does not start a rebuild.
for (const id of ["extensions", "max-file-bytes", "target-tokens", "max-tokens", "overlap-tokens"]) {
  saveOnEnter(byId(id), saveIndexingEdits);
}
const dropZone = byId("drop-zone");
dropZone.addEventListener("click", () => bridge?.chooseSources());
dropZone.addEventListener("dragover", (event) => { event.preventDefault(); dropZone.classList.add("drag-over"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("drag-over");
  const paths = Array.from(event.dataTransfer.files, (file) => file.path).filter(Boolean);
  addSources(paths);
});
byId("stop").addEventListener("click", () => {
  if (!busy || !bridge) return;
  byId("stop").disabled = true;
  byId("activity-text").textContent = "Stopping…";
  bridge.stopOperation();
});
byId("send").addEventListener("click", (event) => {
  if (!busy || !ready || !bridge || !activity.classList.contains("answer-progress")) return;
  event.preventDefault();
  resumeAfterStop = true;
  byId("send").disabled = true;
  // The button spins until KíLín is ready again, so Stop never looks frozen.
  byId("send").classList.add("waiting");
  byId("activity-text").textContent = "Stopping generation…";
  if (pendingTurn) {
    pendingTurn.elapsedMs ??= performance.now() - pendingTurn.started;
    // "Stopping…" shows once, in the summary line; the phase lines go.
    resetAnswerStages();
    pendingTurn.label.closest(".answer-phase-header")?.remove();
    pendingTurn.detail.remove();
    pendingTurn.stopping = true;
    paintPendingTurn();
  }
  // The engine ends the answer and stays loaded (answer_cancelled); only an
  // engine that does not respond is restarted (stopped).
  bridge.cancelAnswer();
});
byId("open-models").addEventListener("click", openModels);
byId("models-hint-button").addEventListener("click", openModels);
byId("close-models").addEventListener("click", closeModels);
for (const tab of document.querySelectorAll(".settings-tab")) {
  tab.addEventListener("click", () => { clearSettingsStatus(); showSettingsTab(tab.dataset.tab); });
  tab.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const tabs = [...document.querySelectorAll(".settings-tab")];
    const index = tabs.indexOf(tab);
    const next = event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1
      : (index + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
    showSettingsTab(tabs[next].dataset.tab);
    tabs[next].focus();
  });
}
showSettingsTab(settingsTab);
byId("models-panel").addEventListener("click", (event) => {
  if (event.target === byId("models-panel")) closeModels();
});
byId("install-required").addEventListener("click", () => {
  if (!bridge) return;
  // While the required models download, this button stops them.
  if (installingModels) {
    byId("install-required").disabled = true;
    byId("install-required").textContent = "Stopping…";
    bridge.stopOperation();
    return;
  }
  if (busy) return;
  setBusy(true, "Checking required models…");
  bridge.installRequiredModels();
});
byId("use-installed").addEventListener("click", () => {
  if (busy || !bridge) return;
  setBusy(true, "Switching to installed models…");
  bridge.useInstalledModels();
});
byId("cancel-model-download").addEventListener("click", () => {
  if (downloadingModel) bridge?.stopOperation();
});
byId("project-form").addEventListener("submit", (event) => {
  event.preventDefault();
  if (busy || !bridge || needsNewProjectSetup()) return;
  if (!profile) return showError("Settings are still loading. Try again in a moment.");
  if (!installedBackends[activeBackend]) {
    openModels();
    byId("model-download-status").textContent = `Install the ${activeBackend.toUpperCase()} packages before preparing this project.`;
    return;
  }
  for (const [field, role] of [["chat-model", "chat"], ["embedding-model", "embedding"]]) {
    const chosen = modelChoices.find((choice) => choice.model_id === byId(field).value.trim());
    if (chosen && !compatibleChoices(role).includes(chosen)) {
      openModels();
      return showError(`The selected ${role} model does not match the backend or embedding query format.`);
    }
  }
  const localModel = (id) => cachedModels.has(id)
    || id.startsWith("/") || id.startsWith(".") || /^[A-Za-z]:[\\/]/.test(id);
  if (![byId("chat-model").value.trim(), byId("embedding-model").value.trim()].every(localModel)) {
    openModels();
    byId("model-download-status").textContent = "Download the selected models in Settings before preparing the project.";
    return;
  }
  const cardProblem = sourceCardProblem();
  if (cardProblem) {
    openModels();
    return showError(cardProblem);
  }
  const selected = Object.fromEntries(
    selectedSources.map((path) => [path, [...(checkedFiles.get(path) || [])]]),
  );
  const included = selectedSources.filter((path) => selected[path].length);
  if (selectedSources.length && !included.length) return showError("Select at least one file.");
  error.classList.add("hidden");
  // A name set in Settings names the project. Otherwise the engine names
  // it after the first source; an empty project gets a free "new-project"
  // name in the projects folder.
  const name = byId("project-title").value.trim();
  if (/[\\/]/.test(name) || name.startsWith(".")) {
    return showError("A project name cannot start with a dot or contain / or \\.");
  }
  if (savedProjects.some((project) => project.name === name)) {
    return showError(`A project named ${name} already exists. Choose another name in Settings.`);
  }
  const destination = name ? projectPath(name) : included.length ? "" : emptyProjectPath();
  setBusy(true, "Opening project…");
  requestAnimationFrame(() => activity.scrollIntoView({ block: "nearest" }));
  bridge.openProject(
    JSON.stringify(included), byId("type-id").value, destination,
    JSON.stringify(selected), JSON.stringify(collectSettings()),
  );
});
byId("question-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const question = byId("question").value.trim();
  if (!question || busy || !ready) return;
  if (answerSettingsSave.pending || answerSettingsSave.uncertain) {
    error.textContent = answerSettingsSave.uncertain
      ? "Reopen the project to verify its answer settings before asking."
      : "Answer settings are still saving. Send after they are confirmed.";
    error.classList.remove("hidden");
    return;
  }
  error.classList.add("hidden");
  const temperature = Number(byId("temperature").value);
  if (!byId("temperature").value.trim() || !Number.isFinite(temperature) || temperature < 0) {
    return showError("Choose a temperature of zero or higher.");
  }
  const answerTokens = byId("max-new-tokens").value.trim();
  const maxNewTokens = answerTokens ? Number(answerTokens) : null;
  if (answerTokens && (!Number.isInteger(maxNewTokens) || maxNewTokens < 1)) {
    return showError("Choose a positive answer-token limit or leave it blank.");
  }
  const answerSettings = {
    temperature, max_new_tokens: maxNewTokens,
    thinking: byId("thinking").checked,
  };
  // An input can still contain a draft when Submit fires before Change.
  // Never let ask() persist that draft as a side effect of answering.
  if (!sameAnswerSettings(answerSettings, confirmedAnswerSettings())) {
    queueAnswerSettings(answerSettings);
    error.textContent = "Saving answer settings. Send after confirmation.";
    error.classList.remove("hidden");
    return;
  }
  byId("question").value = "";
  // A question goes to the chat the engine has open.
  returnToAnswer();
  const ask = {
    question, attachments: messageAttachments, files: [], leftOut: [],
    effort: JSON.stringify({ model: selectedModelLevel, search_depth: selectedDepthStep }),
    settings: confirmedAnswerSettings(),
  };
  messageAttachments = [];
  renderAttachments();
  if (!ask.attachments.length) return sendQuestion(ask);
  // Copy attached files into sources first, so the answer can search them.
  attachedAsk = ask;
  setBusy(true, "Adding the attached files to sources…");
  bridge.checkProjectSources(JSON.stringify(ask.attachments));
});

/**
 * Show the projects folder in Settings; after a switch, list its projects.
 *
 * @param data The ``projects_folder`` event: ``folder`` in use, ``saved``
 *   choice ("" for the default), and whether that choice is ``missing``.
 */
function showProjectsFolder(data) {
  projectsFolderInfo = data;
  byId("workspace-path").value = data.folder;
  byId("workspace-path").title = data.folder;
  showProjectSettings();
  if (!switchingProjectsFolder) return;
  // The engine restarted with the new folder: list the projects there.
  switchingProjectsFolder = false;
  setBusy(false);
  bridge.requestSettings(byId("type-id").value, "");
  bridge.requestReadiness();
  refreshSavedProject();
}

/**
 * Show what Settings can change: the backend and projects folder, in KíLín
 * setup, only on the start screen (disabled with a project open); Delete
 * only with a project open.
 */
function showProjectSettings() {
  // The buttons always show; they are disabled while a project is open,
  // and Use default also while the default folder is in use.
  const why = ready ? "Change it from the start screen." : "";
  byId("browse-workspace").disabled = ready;
  // Also disabled while the standard folder is the one saved; without a
  // saved folder, Confirm projects folder confirms the one in use.
  byId("default-workspace").disabled = ready || !projectsFolderInfo?.saved
    || projectsFolderInfo.saved === projectsFolderInfo.standard;
  byId("use-legacy-workspace").disabled = ready || busy;
  byId("use-legacy-workspace").title = why;
  byId("browse-workspace").title = why;
  byId("default-workspace").title = why;
  // A note only when the chosen folder is missing.
  const missing = Boolean(projectsFolderInfo?.missing);
  byId("workspace-note").textContent = missing
    ? `${projectsFolderInfo.saved} is not available, so KíLín uses the default folder.`
    : "";
  byId("workspace-note").classList.toggle("hidden", !missing);
  const legacy = projectsFolderInfo?.legacy || "";
  const canReuseLegacy = Boolean(
    legacy && legacy !== projectsFolderInfo?.folder,
  );
  byId("legacy-workspace-choice").classList.toggle(
    "hidden", !canReuseLegacy,
  );
  byId("legacy-workspace-note").textContent = canReuseLegacy
    ? `Existing KíLín projects may be at ${legacy}. Use that folder to `
      + "keep working with them, or confirm a different folder. "
      + "No project files will be moved."
    : "";
  showSetupConfirmations();
  // An open project keeps the backend it was made with.
  byId("backend").title = ready ? "Set when the project was created." : "";
  // On the start screen the name is for the next new project.
  byId("delete-project").classList.toggle("hidden", !ready);
  if (ready) byId("project-title").value = projectName(openProjectPath);
}

/**
 * Return a project's name: the last part of its folder path.
 *
 * @param path The project folder.
 */
function projectName(path) {
  return path.split(/[\\/]/).filter(Boolean).pop() || "";
}

/**
 * Rename the open project from Settings: Enter or leaving the field saves,
 * Escape puts the name back.
 */
function renameProjectFromSettings() {
  const input = byId("project-title");
  // On the start screen the name waits for the next new project.
  if (!ready) return;
  const current = projectName(openProjectPath);
  const name = input.value.trim();
  if (!name || name === current || busy) {
    input.value = current;
    return;
  }
  setBusy(true, "Renaming project…");
  bridge.renameProject(name);
}

/**
 * Confirm, then close the open project and move it to the deleted folder
 * in the projects folder.
 */
async function deleteOpenProject() {
  if (!bridge || !ready || busy) return;
  const folder = openProjectPath;
  const name = projectName(folder);
  if (!await askKilin(
    `Delete ${name}?`,
    "Are you sure you want to delete this project?",
    "Delete",
  )) return;
  hideModels();
  returnToLaunch();
  // Sent after closing: the engine refuses to delete an open project.
  bridge.deleteProject(folder);
}

/**
 * Pick a free folder for a project created without sources.
 *
 * @returns "<projects folder>/new-project", numbered when that name is
 *   taken by a listed project.
 */
function emptyProjectPath() {
  const taken = new Set(savedProjects.map((project) => project.name));
  let name = "new-project";
  for (let number = 2; taken.has(name); number += 1) name = `new-project-${number}`;
  return projectPath(name);
}

/**
 * Return where a new project with this name goes.
 *
 * @param name Folder name of the new project.
 * @returns "<projects folder>/<name>".
 */
function projectPath(name) {
  const root = byId("workspace-path").value.replace(/[\\/]+$/, "");
  return `${root}/${name}`;
}

/**
 * Choose another projects folder, or go back to the default one.
 *
 * Qt saves the choice and restarts the engine, so this is offered only on
 * the start screen while KíLín is idle.
 *
 * @param useDefault Whether to go back to the default folder.
 */
function changeProjectsFolder(useDefault) {
  if (!bridge || busy) return;
  if (ready) {
    error.textContent = "Go back to the start screen to change the projects folder.";
    error.classList.remove("hidden");
    return;
  }
  // Set before Browse: Qt answers only once a folder is chosen.
  switchingProjectsFolder = true;
  if (!useDefault) return bridge.chooseProjectsFolder();
  setBusy(true, "Switching to the default projects folder…");
  bridge.useDefaultProjectsFolder();
}

/**
 * Reuse a previous installed-app projects directory without moving files.
 *
 * Qt rechecks the old folder before saving it as the user's choice.
 */
function useLegacyProjectsFolder() {
  if (!bridge || busy || !projectsFolderInfo?.legacy) return;
  if (ready) return changeProjectsFolder(false);
  switchingProjectsFolder = true;
  setBusy(true, "Opening the previous projects folder…");
  bridge.useLegacyProjectsFolder();
}

/**
 * Restore the start screen after a cancelled or failed folder selection.
 *
 * @param message Error text, or an empty string for a cancelled chooser.
 */
function finishProjectsFolderSwitch(message = "") {
  if (switchingProjectsFolder) {
    switchingProjectsFolder = false;
    setBusy(false);
  }
  if (message) showError(message);
}

/**
 * Ask a question; attached files, already in sources, are listed under it.
 *
 * @param ask Question, its attached ``sources/`` paths (``files``), names
 *   that could not be added (``leftOut``), and the answer settings.
 */
function sendQuestion(ask) {
  if (answerSettingsSave.pending || answerSettingsSave.uncertain) {
    byId("question").value = ask.question;
    messageAttachments = ask.attachments;
    renderAttachments();
    error.textContent = "Confirm answer settings before sending the question.";
    error.classList.remove("hidden");
    return;
  }
  unansweredQuestion = ask.question;
  answeringConversationId = activeConversationId;
  const listed = ask.files.map((path) => `- ${path}`).join("\n");
  const text = listed ? `${ask.question}\n\nAttached files:\n${listed}` : ask.question;
  addPendingTurn(text);
  showAskingConversation(ask.question);
  setBusy(true, "Finding evidence and answering…");
  bridge.askQuestion(text, ask.effort, JSON.stringify({ ...ask.settings, _title_question: ask.question }));
  if (ask.leftOut.length) {
    error.textContent = `KíLín could not add ${ask.leftOut.join(", ")}, so the answer does not use ${ask.leftOut.length === 1 ? "it" : "them"}.`;
    error.classList.remove("hidden");
  }
}

/**
 * Add a question's attached files to sources without the add dialog.
 *
 * Files already in sources are only referred to; the question is asked
 * when the refreshed project arrives (``opened``), or now when nothing new
 * has to be copied.
 *
 * @param check The ``sources_checked`` event for the attached paths.
 */
function addAttachedSources(check) {
  const ask = attachedAsk;
  ask.files = check.add.map((item) => `sources/${item.name}${item.kind === "folder" ? "/" : ""}`);
  ask.leftOut = [...check.files.map((file) => file.path), ...check.skipped_folders];
  if (!check.add.some((item) => item.state !== "existing")) {
    attachedAsk = null;
    return sendQuestion(ask);
  }
  setBusy(true, "Adding the attached files and updating the index…");
  bridge.addProjectSources(JSON.stringify(check.sources), "{}");
}
byId("question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    byId("question-form").requestSubmit();
  }
});
conversationScroll.addEventListener("scroll", updateFloatingHeader, { passive: true });
byId("conversation-name").addEventListener("input", () => updateChatSummary());
byId("new-chat").addEventListener("click", () => openEmptyChat(""));
/**
 * Send a valid edited conversation title to the backend.
 */
function commitConversationName() {
  const name = byId("conversation-name").textContent.trim();
  if (!busy && activeConversationId && name) bridge.renameConversation(activeConversationId, name);
}
for (const [id, commit] of [["conversation-name", commitConversationName]]) {
  byId(id).addEventListener("blur", commit);
  byId(id).addEventListener("keydown", (event) => {
    if (event.key === "Enter") { event.preventDefault(); event.target.blur(); }
  });
}
/**
 * Close the open project and show the start screen, where a project can be
 * resumed or created.
 */
function returnToLaunch() {
  answerSettingsSave.pending = null;
  answerSettingsSave.queued = null;
  answerSettingsSave.uncertain = false;
  bridge.closeProject();
  openProjectPath = "";
  ready = false;
  updateFloatingHeader();
  chat.classList.add("hidden");
  shell.classList.remove("chat-open");
  byId("conversations-panel").classList.add("hidden");
  byId("toggle-conversations").classList.add("hidden");
  setup.classList.remove("hidden");
  selectedSources.length = 0;
  previewRoots.clear();
  previewSkipped.clear();
  checkedFiles.clear();
  renderPreview();
  showWorkspace(false);
  clearReferences();
  byId("project-title").value = "";
  showProjectSettings();
  bridge.requestSettings(byId("type-id").value, "");
  setBusy(false);
  refreshSavedProject();
}

// The logo and the Projects ＋ lead back to the start screen, where new
// projects are made; both are disabled while busy.
for (const id of ["home-logo", "new-project"]) {
  byId(id).addEventListener("click", () => {
    if (busy) return;
    byId("leave-project").showModal();
  });
}
// Each left-panel section folds, and sorts from its ⋯ menu; both choices
// are remembered on this computer.
for (const section of document.querySelectorAll(".sidebar-section")) {
  const name = section.dataset.section;
  const toggle = section.querySelector(".sidebar-section-toggle");
  const fold = (collapsed) => {
    section.classList.toggle("collapsed", collapsed);
    toggle.setAttribute("aria-expanded", String(!collapsed));
  };
  fold(sidebarPref(`${name}-collapsed`, false));
  toggle.addEventListener("click", () => {
    const collapsed = !section.classList.contains("collapsed");
    fold(collapsed);
    setSidebarPref(`${name}-collapsed`, collapsed);
  });
}
byId("sort-projects").addEventListener("click", (event) => {
  event.stopPropagation();
  showSidebarSortMenu(event.currentTarget, "projects",
    () => renderSidebarProjects(savedProjects, openProjectPath));
});
byId("sort-conversations").addEventListener("click", (event) => {
  event.stopPropagation();
  showSidebarSortMenu(event.currentTarget, "conversations",
    () => lastConversationData && renderConversations(lastConversationData));
});
byId("leave-project-cancel").addEventListener("click", () => byId("leave-project").close());
byId("kilin-ask-cancel").addEventListener("click", () => byId("kilin-ask").close("cancel"));
byId("kilin-ask-other").addEventListener("click", () => byId("kilin-ask").close("other"));
byId("kilin-ask-confirm").addEventListener("click", () => byId("kilin-ask").close("confirm"));
byId("leave-project-confirm").addEventListener("click", () => {
  byId("leave-project").close();
  if (busy) return;
  returnToLaunch();
});
byId("toggle-workspace").addEventListener("click", () => {
  chooseSidePanels(!shell.classList.contains("sidebar-open"));
});
byId("toggle-conversations").addEventListener("click", () => {
  showConversations(shell.classList.contains("conversations-hidden"));
});
byId("usage-button").addEventListener("click", (event) => {
  event.stopPropagation();
  const menu = byId("usage-menu");
  renderUsage();
  byId("effort-menu").classList.add("hidden");
  byId("effort-button").setAttribute("aria-expanded", "false");
  menu.classList.toggle("hidden");
  byId("usage-button").setAttribute("aria-expanded", String(!menu.classList.contains("hidden")));
});
byId("usage-menu").addEventListener("click", (event) => event.stopPropagation());
byId("effort-button").addEventListener("click", (event) => {
  event.stopPropagation();
  const menu = byId("effort-menu");
  // One composer menu at a time.
  byId("usage-menu").classList.add("hidden");
  byId("usage-button").setAttribute("aria-expanded", "false");
  // The project may have changed Thinking since the menu was last open.
  renderEffortPicker();
  menu.classList.toggle("hidden");
  byId("effort-button").setAttribute("aria-expanded", String(!menu.classList.contains("hidden")));
});
byId("effort-name").addEventListener("click", () => {
  const open = byId("effort-thinking").classList.toggle("hidden") === false;
  byId("effort-name").setAttribute("aria-expanded", String(open));
});
for (const input of document.querySelectorAll('input[name="effort-thinking"]')) {
  input.addEventListener("change", () => {
    // The Answers checkbox is the setting; its change listener keeps a
    // start-screen choice for the next new project.
    byId("thinking").checked = input.value === "on";
    byId("thinking").dispatchEvent(new Event("change"));
  });
}
byId("thinking").addEventListener("change", () => {
  // Both Thinking controls share the Answers setting and its save queue.
  // On the start screen the selected value belongs to the next project.
  if (ready && busy) {
    byId("thinking").checked = answerSettingsSave.queued?.thinking
      ?? answerSettingsSave.pending?.settings.thinking
      ?? savedAnswerThinking;
  } else if (ready) {
    saveAnswerNumbers();
  }
  renderEffortPicker();
});
byId("effort-model-range").addEventListener("input", (event) => {
  setEffortLevel("model", event.target.value);
});
byId("effort-depth-range").addEventListener("input", (event) => {
  setEffortLevel("depth", event.target.value);
});
byId("attach-button").addEventListener("click", (event) => {
  event.stopPropagation();
  const menu = byId("attach-menu");
  menu.classList.toggle("hidden");
  byId("attach-button").setAttribute("aria-expanded", String(!menu.classList.contains("hidden")));
});
byId("attach-files").addEventListener("click", () => {
  byId("attach-menu").classList.add("hidden");
  bridge?.chooseAttachmentFiles();
});
byId("attach-folder").addEventListener("click", () => {
  byId("attach-menu").classList.add("hidden");
  bridge?.chooseAttachmentFolder();
});
byId("appearance").addEventListener("change", (event) => applyAppearance(event.target.value));
for (const id of ["thinking", "temperature", "max-new-tokens"]) {
  byId(id).addEventListener("change", () => { if (!ready) answerSettingsEdited = true; });
}
for (const id of ["temperature", "max-new-tokens"]) {
  rememberValue(byId(id));
  byId(id).addEventListener("change", saveAnswerNumbers);
}
byId("type-id").addEventListener("change", () => {
  hideResume();
  bridge?.requestSettings(byId("type-id").value, "");
  refreshSavedProject();
});
byId("backend").addEventListener("change", () => {
  if (!ready) bridge?.setBackendPreference(byId("backend").value);
  saveModelFields();
  showModelFields();
});
byId("query-format").addEventListener("change", () => {
  refreshModelStatus();
  if (!byId("models-panel").classList.contains("hidden")) renderModelsPanel();
});
for (const id of ["chat-model", "embedding-model"]) {
  byId(`${id}-options`).addEventListener("click", () => toggleModelMenu(id));
  byId(id).addEventListener("input", () => {
    refreshModelStatus();
  });
}
document.addEventListener("click", (event) => {
  if (!event.target.closest(".model-picker")) closeModelMenus();
  if (!event.target.closest(".effort-picker")) byId("effort-menu").classList.add("hidden");
  if (!event.target.closest(".usage-picker")) byId("usage-menu").classList.add("hidden");
  if (!event.target.closest(".attach-picker")) byId("attach-menu").classList.add("hidden");
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeModelMenus();
    if (!byId("models-panel").classList.contains("hidden")) closeModels();
  }
});
for (const id of ["extensions", "max-file-bytes"]) byId(id).addEventListener("change", previewSources);

byId("save-source-card-models").addEventListener(
  "click", saveSourceCardSettings,
);
byId("save-source-card-policy").addEventListener(
  "click", saveSourceCardSettings,
);
byId("regenerate-source-cards").addEventListener(
  "click", regenerateSourceCards,
);

// Attach the bridge only after Qt exposes the channel; then request state.
new QWebChannel(qt.webChannelTransport, (channel) => {
  bridge = channel.objects.kilin;
  bridge.setAppearance(document.documentElement.dataset.appearance || "system");
  bridge.event.connect(onEvent);
  bridge.pathsSelected.connect(addSources);
  bridge.folderFilesSelected.connect(folderFilesChosen);
  bridge.sourceDragChanged.connect(showNativeSourceDrag);
  syncSourceDropTarget();
  bridge.attachmentsSelected.connect((paths) => {
    for (const path of paths) {
      if (path && !messageAttachments.includes(path)) messageAttachments.push(path);
    }
    renderAttachments();
  });
  bridge.requestSettings(byId("type-id").value, "");
  bridge.requestReadiness();
  bridge.requestProjectsFolder();
  showWorkspace(false);
  renderEffortPicker();
  setBusy(false);
  refreshSavedProject();
});

// Refresh the setup shortcut when external changes may invalidate a project.
window.addEventListener("focus", refreshSavedProject);
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) refreshSavedProject();
});
// Also catch files removed while the project chooser stays in the foreground.
window.setInterval(refreshSavedProject, 3000);
applyAppearance(localStorage.getItem("kilin-appearance") || "system");

// Draggable borders beside the conversation list and the side panels.
const PANEL_LIMITS = { left: [180, 420], right: [260, 640] };
const MIDDLE_MIN_WIDTH = 420;
// Dragging a panel below this share of its minimum width hides it.
const COLLAPSE_SHARE = 0.55;
const panelResizers = {};

/**
 * Apply a remembered or dragged panel width, within its limits.
 *
 * @param side "left" (conversations) or "right" (workspace and references).
 * @param width Requested width in pixels; null restores the default.
 */
function setPanelWidth(side, width) {
  const property = side === "left" ? "--left-width" : "--right-width";
  if (width === null) {
    shell.style.removeProperty(property);
    return;
  }
  const [min, max] = PANEL_LIMITS[side];
  shell.style.setProperty(property, `${Math.round(Math.min(max, Math.max(min, width)))}px`);
}

/**
 * Place each handle on its visible border, or hide it.
 */
function positionPanelResizers() {
  const grid = getComputedStyle(shell).display === "grid" && shell.classList.contains("chat-open");
  const left = byId("conversations-panel");
  const right = document.querySelector(".side-panels");
  for (const [side, panel] of [["left", left], ["right", right]]) {
    const handle = panelResizers[side];
    const box = panel.getBoundingClientRect();
    const visible = grid && box.width > 0 && getComputedStyle(panel).display !== "none";
    handle.classList.toggle("hidden", !visible);
    if (visible) handle.style.left = `${(side === "left" ? box.right : box.left) - 3}px`;
  }
}

for (const side of ["left", "right"]) {
  const handle = document.createElement("div");
  handle.className = "panel-resizer hidden";
  handle.setAttribute("role", "separator");
  handle.setAttribute("aria-orientation", "vertical");
  handle.title = "Drag to resize; double-click to reset";
  let lastPress = 0;
  let collapse = false;
  let startWidth = "";
  const panelClass = `collapsing-${side}`;
  const reset = () => {
    setPanelWidth(side, null);
    try { localStorage.removeItem(`kilin-${side}-width`); } catch {}
    positionPanelResizers();
    syncSourceDropTarget();
  };
  handle.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    // Detect double-clicks here: pointer capture can suppress dblclick.
    if (event.timeStamp - lastPress < 400) {
      lastPress = 0;
      reset();
      return;
    }
    lastPress = event.timeStamp;
    collapse = false;
    startWidth = shell.style.getPropertyValue(`--${side}-width`);
    handle.setPointerCapture(event.pointerId);
    handle.classList.add("dragging");
    document.body.classList.add("resizing-panels");
  });
  handle.addEventListener("pointermove", (event) => {
    if (!handle.hasPointerCapture(event.pointerId)) return;
    lastPress = 0;  // A drag is not the first half of a double-click.
    const other = side === "left"
      ? document.querySelector(".side-panels").getBoundingClientRect().width
      : byId("conversations-panel").getBoundingClientRect().width;
    const requested = side === "left" ? event.clientX : window.innerWidth - event.clientX;
    // Far below the minimum: preview hiding the panel instead of resizing.
    collapse = requested < PANEL_LIMITS[side][0] * COLLAPSE_SHARE;
    shell.classList.toggle(panelClass, collapse);
    if (collapse) return;
    // Keep the conversation column usable.
    setPanelWidth(side, Math.min(requested, window.innerWidth - other - MIDDLE_MIN_WIDTH));
    positionPanelResizers();
  });
  const finish = (event) => {
    if (!handle.hasPointerCapture(event.pointerId)) return;
    handle.releasePointerCapture(event.pointerId);
    handle.classList.remove("dragging");
    document.body.classList.remove("resizing-panels");
    shell.classList.remove(panelClass);
    if (collapse) {
      // Hide the panel as its toggle does; it reopens at its last width.
      collapse = false;
      if (startWidth) shell.style.setProperty(`--${side}-width`, startWidth);
      else shell.style.removeProperty(`--${side}-width`);
      if (side === "left") showConversations(false);
      else chooseSidePanels(false);
      positionPanelResizers();
      syncSourceDropTarget();
      return;
    }
    try {
      localStorage.setItem(`kilin-${side}-width`, shell.style.getPropertyValue(`--${side}-width`));
    } catch {}
    syncSourceDropTarget();
  };
  handle.addEventListener("pointerup", finish);
  handle.addEventListener("pointercancel", finish);
  document.body.append(handle);
  panelResizers[side] = handle;
  try {
    const saved = parseFloat(localStorage.getItem(`kilin-${side}-width`));
    if (Number.isFinite(saved)) setPanelWidth(side, saved);
  } catch {}
}
// Panels appear, hide, and change size on many events; follow them all.
const panelObserver = new ResizeObserver(positionPanelResizers);
panelObserver.observe(shell);
panelObserver.observe(byId("conversations-panel"));
panelObserver.observe(document.querySelector(".side-panels"));
window.addEventListener("resize", positionPanelResizers);

// Draggable border between the Workspace and References panels.
const WORKSPACE_SHARE_LIMITS = [15, 85];
const sideDivider = byId("side-divider");

/**
 * Apply a remembered or dragged Workspace height, within its limits.
 *
 * @param share Requested percentage of the side column for Workspace;
 *   null restores the default even split.
 */
function setWorkspaceShare(share) {
  const panels = document.querySelector(".side-panels");
  if (share === null) {
    panels.style.removeProperty("--workspace-share");
    return;
  }
  const [min, max] = WORKSPACE_SHARE_LIMITS;
  panels.style.setProperty("--workspace-share", `${Math.round(Math.min(max, Math.max(min, share)))}%`);
}

let sideDividerPress = 0;
sideDivider.addEventListener("pointerdown", (event) => {
  event.preventDefault();
  // Detect double-clicks here: pointer capture can suppress dblclick.
  if (event.timeStamp - sideDividerPress < 400) {
    sideDividerPress = 0;
    setWorkspaceShare(null);
    try { localStorage.removeItem("kilin-workspace-share"); } catch {}
    return;
  }
  sideDividerPress = event.timeStamp;
  sideDivider.setPointerCapture(event.pointerId);
  sideDivider.classList.add("dragging");
  document.body.classList.add("resizing-panels", "resizing-rows");
});
sideDivider.addEventListener("pointermove", (event) => {
  if (!sideDivider.hasPointerCapture(event.pointerId)) return;
  sideDividerPress = 0;  // A drag is not the first half of a double-click.
  const box = document.querySelector(".side-panels").getBoundingClientRect();
  setWorkspaceShare(((event.clientY - box.top) / box.height) * 100);
});
const finishSideDivider = (event) => {
  if (!sideDivider.hasPointerCapture(event.pointerId)) return;
  sideDivider.releasePointerCapture(event.pointerId);
  sideDivider.classList.remove("dragging");
  document.body.classList.remove("resizing-panels", "resizing-rows");
  try {
    localStorage.setItem("kilin-workspace-share", document.querySelector(".side-panels").style.getPropertyValue("--workspace-share"));
  } catch {}
  syncSourceDropTarget();
};
sideDivider.addEventListener("pointerup", finishSideDivider);
sideDivider.addEventListener("pointercancel", finishSideDivider);
try {
  const saved = parseFloat(localStorage.getItem("kilin-workspace-share"));
  if (Number.isFinite(saved)) setWorkspaceShare(saved);
} catch {}
