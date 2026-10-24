# Code example: grading worksheet

Indexed revision: f0fdcd5ab (HEAD)

## python_units (location, evaluation)

**Question.** How does KíLín divide a Python file into the parts it indexes, and what happens to a file that is not valid Python?

**Intent:** code_explanation

**Answer with retrieval:**

The provided evidence does not explicitly describe how KíLín divides Python files into parts for indexing or handles invalid Python files. The code snippets focus on checking report templates [1][2][3] and managing output files [4], but no Python-specific parsing or validation logic is detailed. The `ReportCheck` class validates report structure against a template, not Python syntax, and the `load_output` function processes files into passages without specifying Python file handling. Thus, the evidence is insufficient to answer the question about Python file division or invalid Python handling.

**Cited passages:**

- [1] kilin/tools/generate_paper_summary/_report_template.py `ReportCheck.valid` (retrieved by: code)
- [2] kilin/tools/generate_paper_summary/_report_template.py `ReportCheck.problems` (retrieved by: code)
- [3] kilin/tools/generate_paper_summary/_report_template.py `ReportCheck` (retrieved by: code)
- [4] kilin/tools/_shared/outputs.py `outputs` (retrieved by: lexical)

**Gold functions:** `_python_units`, `_python_class_unit`

- in evidence: []
- cited: []

**Facts (tick those the answer states correctly):**

- [ ] The file is parsed with Python's ast module; nothing is executed.
- [ ] Each top-level function becomes one unit, and the code between definitions (imports, constants) becomes module-context units.
- [ ] A class becomes one class unit whose children are its methods and the class-level code between them.
- [ ] A file that does not parse (SyntaxError) becomes a single python_file unit.

**Errors or unsupported statements:** 

**Answer without retrieval:**

KíLín divides a Python file into parts by parsing it into an **Abstract Syntax Tree (AST)** using Python's built-in `ast` module. This process breaks the code into structured elements like functions, classes, variables, and statements, which are then indexed for features like autocompletion or navigation. The division is based on the syntactic structure of the code, not line breaks.

For files that are **not valid Python** (e.g., syntax errors), KíLín will **fail to parse** the file. This results in the entire file being ignored or an error being raised, depending on how the parser handles the invalid input. KíLín does not attempt to recover or index partial content from invalid files.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## pdf_text (location, evaluation)

**Question.** How does KíLín get the text out of a PDF file, and what happens when a PDF contains no text?

**Intent:** code_explanation

**Answer with retrieval:**

KíLín extracts text from PDFs using **pdfminer.six** for layout analysis, which processes glyph boxes and font metrics to preserve word spacing and reading order [1]. It leverages **pypdf** to handle encryption and page counting but relies on **pdfminer.six** for text extraction [5]. The `pdf_text` function (in [4]) checks if the file is a valid PDF and then calls `_pdf_text` (in [2]), which extracts text lines per page, sorts them for two-column layouts, and joins pages with form feeds (`\f`). If the PDF is unreadable (e.g., encrypted, corrupted, or lacks extractable text), `_pdf_text` returns an empty string, and `pdf_text` propagates this as an empty result [4]. When a PDF contains no text, KíLín silently returns an empty string, leading to no units or text being processed. This behavior ensures unsupported or encrypted files are explicitly marked as having no text rather than silently failing [1].

**Cited passages:**

- [1] kilin/sources/extensions/sources_pdf.py `sources_pdf` (retrieved by: semantic)
- [2] kilin/tools/_shared/references.py `_pdf_text` (retrieved by: semantic)
- [4] kilin/tools/_shared/references.py `pdf_text` (retrieved by: semantic)
- [5] kilin/sources/extensions/sources_pdf.py `sources_pdf` (retrieved by: semantic)

**Gold functions:** `read_source_file`, `_layout_pages`, `extract_source_units`

- in evidence: []
- cited: []

**Facts (tick those the answer states correctly):**

- [ ] Pages are extracted with pdfminer.six layout analysis; if that fails or gives a different page count, pypdf's plain extraction is used.
- [ ] Encrypted or unreadable PDFs raise an error.
- [ ] The text is cleaned (invalid Unicode surrogates, unmapped glyphs, trailing spaces, repeated blank lines).
- [ ] A PDF with no extractable text raises an error saying scanned PDFs need OCR.
- [ ] Pages are joined with a page break, and each non-empty page becomes one unit labelled with its page number.

**Errors or unsupported statements:** 

**Answer without retrieval:**

KíLín uses the **PyMuPDF** (also known as **fitz**) library to extract text from PDF files. Here's how it works:

1. **Text Extraction Process**:  
   - KíLín leverages PyMuPDF's ability to parse the PDF's internal text layers. It reads the text content from each page by accessing the PDF's "text" stream, which is part of the PDF's structure. This includes extracting text, fonts, and layout information.

2. **Handling PDFs with No Text**:  
   - If a PDF contains no text (e.g., it consists solely of images, vector graphics, or other non-text elements), PyMuPDF will return an **empty string** or `None` for the text extraction. KíLín will propagate this result directly, meaning the extracted text will be empty or absent.

3. **Fallback or Special Cases**:  
   - KíLín does not inherently add fallback text or guess content in such cases. If no text is present, the output will reflect this absence. However, if the PDF contains non-text elements (e.g., images), KíLín might still extract metadata (e.g., page numbers, author, title) if available, depending on the PDF's structure.

In summary, KíLín extracts text by parsing the PDF's text stream, and if no text exists, the result will be empty. For purely graphical PDFs, additional tools (e.g., OCR) would be required to extract content, but KíLín does not support OCR by default.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## call_relations (location, evaluation)

**Question.** How does KíLín work out which functions call which other functions in Python code, and what happens when a called name could refer to several functions?

**Intent:** code_explanation

**Answer with retrieval:**

KíLín determines function calls by analyzing a `CodeGraph` that maps relationships between functions, specifically "calls" edges [1]. The `_ranked_callees` function processes this graph to identify which functions are called by a given set of functions. It prioritizes callees based on shared words with the query text, call depth, and the callee's role (e.g., functions that call more other functions are ranked higher). When a called name could refer to multiple functions, the system uses the query text to prefer matches with shared terms and relies on the graph's structure to resolve ambiguity [1]. For example, a query like "Which functions call `parse`?" would use the graph to identify the most relevant callee, as demonstrated in a test case where `pkg.loader.read` was the result [5]. The ranking ensures the most contextually relevant and functionally significant callees are prioritized.

**Cited passages:**

- [1] kilin/retrievals/context_expansion.py `_ranked_callees` (retrieved by: semantic)
- [5] tests/kilin/retrievals/test_context_expansion.py `TestContextExpansionPlanning.test_singular_call_is_a_relationship_phrase` (retrieved by: lexical)

**Gold functions:** `extract_code_relations`, `_Visitor.visit_Call`, `_resolve_name`

- in evidence: []
- cited: []

**Facts (tick those the answer states correctly):**

- [ ] Each Python file is parsed with ast and walked; the code is never imported or run, and a file with a syntax error contributes no relations.
- [ ] For every call, the called expression is turned into a dotted name (a plain name or attribute chain) and recorded as a calls relation from the enclosing function, method, class, or module.
- [ ] The name is resolved by its last component against the known entities: a unique match gives the qualified name.
- [ ] An ambiguous name is kept as written (unless only one candidate differs from the name itself), so no call is guessed; calls through non-static expressions are not recorded.

**Errors or unsupported statements:** 

**Answer without retrieval:**

KíLín, a Python code analysis tool, determines function call relationships through **static analysis** of the source code. Here's how it works and how it handles ambiguous references:

---

### **1. How KíLín Identifies Function Calls**
- **AST Parsing**: KíLín parses Python code into an **Abstract Syntax Tree (AST)**, which represents the code's structure. This allows it to identify function definitions and calls explicitly.
- **Call Graph Construction**: It traverses the AST to detect **function calls** (e.g., `func()` or `var()` where `var` refers to a function). For each call, it records the **caller** (the function or variable making the call) and the **callee** (the function being called).
- **Scope Resolution**: KíLín uses **scope information** (e.g., local, global, or module-level variables) to resolve names to their definitions. This helps disambiguate function references within nested scopes.

---

### **2. Handling Ambiguous References**
When a called name could refer to **multiple functions** (e.g., due to dynamic assignments or overlapping scopes), KíLín behaves as follows:
- **Ambiguity Detection**: If a name is assigned to multiple functions (e.g., `f = some_function()` and later `f = another_function()`), KíLín may flag this as a potential issue during analysis.
- **Contextual Resolution**: If the context (e.g., the scope or type) allows disambiguation, KíLín resolves the reference. For example, if a variable is assigned a function in a specific scope, it uses that scope to determine the callee.
- **Reporting Ambiguity**: If the ambiguity cannot be resolved statically (e.g., dynamic function calls like `func = getattr(obj, 'method')`), KíLín may report the ambiguity as a **potential bug** or **uncertain call path**, depending on its configuration.

---

### **3. Example Scenarios**
- **Direct Call**:  
  ```python
  def add(a, b):
      return a + b

  def multiply(a, b):
      return a * b

  result = add(2, 3)  # KíLín identifies `add` as the caller and `add` as the callee.
  ```
- **Ambiguous Reference**:  
  ```python
  def foo():
      print("Hello")

  def bar():
      print("World")

  func = foo  # func refers to foo
  func()      # KíLín resolves `func` to `foo` here.
  ```
  If `func` is reassigned later (e.g., `func = bar`), KíLín may flag the call as ambiguous unless the context is clear.

---

### **4. Limitations**
- **Dynamic Calls**: KíLín cannot resolve calls that depend on runtime values (e.g., `getattr(obj, 'method')` or `locals()['func']`), as these require dynamic execution.
- **Overlapping Scopes**: If multiple functions share the same name in different scopes (e.g., a function and a class with the same name), KíLín may report ambiguity unless the scope is explicitly resolved.

---

In summary, KíLín uses static analysis to map function calls and resolves ambiguities using scope and context. However, it may report unresolved ambiguities as potential issues, especially in cases involving dynamic or overlapping references.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## keyword_scoring (location, evaluation)

**Question.** How are passages scored in KíLín's keyword search, and how are code identifiers broken into searchable words?

**Intent:** code_explanation

**Answer with retrieval:**

Passages in KíLín's keyword search are scored using TF-IDF cosine similarity for lexical search [2], while code search uses a graph-based distance metric, assigning scores as `1.0 / (distance + 1)` [7]. Code identifiers are split into searchable words by inserting spaces before uppercase letters and underscores, then extracting lowercase words via regex (`_WORD.findall`), excluding common words and short terms [1][8]. This process ensures identifiers like `_parse_route` become `parse route` [8]. The scoring and tokenization are tied to retrieval strategies like `lexical_terms` [2], which extracts one- and two-word terms for indexing.

**Cited passages:**

- [1] kilin/tools/reference_check/_matching.py `_words` (retrieved by: code)
- [2] kilin/retrievals/strategies/__init__.py `__init__` (retrieved by: semantic)
- [7] kilin/retrievals/strategies/retrieval_code.py `code_search` (retrieved by: context_expansion)
- [8] kilin/retrievals/context_expansion.py `_identifier_words` (retrieved by: context_expansion)

**Gold functions:** `lexical_terms`, `lexical_search`, `_tfidf_vector`

- in evidence: []
- cited: []

**Facts (tick those the answer states correctly):**

- [ ] Text is split into lowercase words of letters, digits, and underscores; identifiers also contribute their camelCase and snake_case parts longer than one character.
- [ ] Adjacent words also form two-word terms; there is no stemming or stop-word removal.
- [ ] Each passage and the query become TF-IDF vectors with sublinear term frequency (1 + log f) and smoothed inverse document frequency, normalized to unit length.
- [ ] Passages are ranked by cosine similarity, ties by chunk identifier, with no minimum score.

**Errors or unsupported statements:** 

**Answer without retrieval:**

KíLín, a Python package for searching through code, employs specific strategies for scoring keyword matches and tokenizing code identifiers. Here's a breakdown of the process:

### **Passage Scoring in Keyword Search**
1. **Relevance Scoring**: KíLín likely uses a relevance-based scoring system, where the score is determined by factors such as:
   - **Keyword Frequency**: The number of times a keyword appears in the code.
   - **Contextual Importance**: Keywords appearing in function names, variable names, or other syntactically significant locations may receive higher scores.
   - **Proximity**: Matches that are closer to each other or to specific code elements (e.g., function definitions) might be prioritized.
   - **Inverse Document Frequency (IDF)**: Less common keywords might be weighted more heavily to avoid trivial matches.

2. **Search Context**: The scoring may also consider the context in which keywords appear, such as within strings, comments, or code blocks, to differentiate between meaningful matches and accidental overlaps.

### **Code Identifiers to Searchable Words**
KíLín tokenizes code identifiers (e.g., variable names, function names) into searchable terms by:
1. **Handling Naming Conventions**:
   - **CamelCase** (e.g., `myVariable`): Split into `my` and `Variable` by inserting spaces between lowercase and uppercase letters.
   - **snake_case** (e.g., `my_variable`): Split into `my` and `variable` by splitting on underscores.
   - **PascalCase** (e.g., `MyClass`): Split into `My` and `Class` (or treat as a single term if contextually relevant).

2. **Tokenization Process**: 
   - A tokenizer identifies and splits identifiers into individual words, which are then indexed for search. This allows users to search for terms like `my` or `variable` even if they appear as part of a larger identifier.

3. **Case Sensitivity**: KíLín may normalize case (e.g., treating `MyClass` and `myClass` as the same) or preserve case depending on configuration, depending on the intended use case.

### **Key Considerations**
- **Customization**: Users may configure tokenization rules or scoring weights to adapt to specific codebases or search priorities.
- **Efficiency**: The tokenization and scoring mechanisms are optimized for performance, especially when dealing with large codebases.

For precise details, refer to KíLín's documentation or source code, as the exact implementation may vary.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## follow_up_query (mechanism, evaluation)

**Question.** Explain how refers_back decides that a question depends on the earlier conversation, and how search_text uses that to build the search query.

**Intent:** code_explanation

**Answer with retrieval:**

The `refers_back` function determines if a question depends on earlier conversation by checking for pronouns ("it", "they"), ordinal references ("the second one"), pointers back ("you said", "earlier"), elliptical openings ("and for X?"), or short questions (≤ 2 words) [2]. If these patterns are present, it returns `True`, indicating the question relies on prior context. 

The `search_text` function uses this information to build the search query. If `refers_back` is `True` and there are related turn IDs from the routing decision, it retrieves the earlier conversation turns' questions (oldest first) and appends them to the original question. This creates a combined query that includes both the current question and relevant prior context, ensuring the search incorporates historical dialogue without adding model-generated text [1]. For example, a question like "how do I install it?" would include the prior question "what is NOVA" in the search query, linking the current question to its context [5]. This approach helps the model leverage explicit user references to earlier turns for more accurate retrieval.

**Cited passages:**

- [1] kilin/answering.py `search_text` (retrieved by: lexical, semantic)
- [2] kilin/answering.py `refers_back` (retrieved by: lexical, semantic)
- [5] tests/kilin/test_answering.py `TestRelatedTurnChecks.test_pointing_back_adds_related_questions_to_search` (retrieved by: lexical, semantic)

**Gold functions:** `refers_back`, `search_text`

- in evidence: ['refers_back', 'search_text']
- cited: ['refers_back', 'search_text']

**Facts (tick those the answer states correctly):**

- [ ] refers_back is true for messages of at most two words or when a configured pattern matches: pronouns (it, they, this), ordinal references (the second one), pointers back (you said, earlier, so far), or elliptical openings (and, what about).
- [ ] search_text returns the question unchanged unless the routing decision refers back and lists related turns.
- [ ] Otherwise it appends the questions of the related turns, with whitespace normalized, after the new question.
- [ ] Model-written text (earlier answers) is never added to the query.

**Errors or unsupported statements:** 

**Answer without retrieval:**

The `refers_back` function in the KíLín Python package determines if a question depends on earlier conversation by analyzing the current question's text for linguistic cues (e.g., pronouns, references to entities, or topics mentioned previously). It checks whether the question implicitly or explicitly connects to information from the conversation history, such as specific entities, topics, or prior answers. For example, if the user asks, "What's the weather like today?" and then follows up with "Is it going to rain tomorrow?" the second question refers back to the weather topic discussed earlier.

Once `refers_back` identifies that the question is context-dependent, the `search_text` function uses this information to build a more targeted search query. It combines the current question with relevant context from the conversation history to refine the search. This ensures the search is focused on the specific topic or entity mentioned earlier, improving the relevance of results. For instance, if the prior conversation was about a product, `search_text` might prioritize information related to that product, making the search more efficient and aligned with the user's ongoing dialogue. 

If you need precise implementation details, consult the KíLín documentation or source code, as the exact logic may vary based on the package's design.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## download_progress (mechanism, evaluation)

**Question.** Explain how download_model reports download progress to its caller, and where the downloaded model files are stored.

**Intent:** code_explanation

**Answer with retrieval:**

The `download_model` function reports progress to its caller through a callback mechanism and emits events via the `self._send` method. When initiated by `ChatEngine.download` [4], it triggers `_download_with_progress` [2], which uses a `progress_callback` linked to a custom `tqdm` class (`_progress_tqdm`) [9]. This class aggregates download bytes and forwards them to the callback, which updates progress metrics (completed/expected bytes, rate, ETA) via `send_progress` [6]. These metrics are then sent to the frontend as events with `progress_event="model_download_progress"` [7]. 

Downloaded model files are stored in Hugging Face's shared cache, managed by `snapshot_download` [3]. The exact path is determined by Hugging Face, and the function returns a `Path` object pointing to the model directory. KíLín does not copy or manage the files, relying on Hugging Face's caching system. The stored path is returned to the caller, but the actual location depends on Hugging Face's configuration.

**Cited passages:**

- [2] kilin/gui/engine.py `ChatEngine._download_with_progress` (retrieved by: code, semantic)
- [3] kilin/huggingface.py `download_model` (retrieved by: code, lexical)
- [4] kilin/gui/engine.py `ChatEngine.download` (retrieved by: semantic)
- [6] kilin/gui/engine.py `ChatEngine._download_with_progress` (retrieved by: context_expansion)
- [7] kilin/gui/engine.py `ChatEngine._download_with_progress` (retrieved by: context_expansion)
- [9] kilin/huggingface.py `_progress_tqdm` (retrieved by: context_expansion)

**Gold functions:** `download_model`, `_progress_tqdm`

- in evidence: ['download_model', '_progress_tqdm']
- cited: ['download_model', '_progress_tqdm']

**Facts (tick those the answer states correctly):**

- [ ] download_model calls huggingface_hub.snapshot_download, so files go to (or are reused from) Hugging Face's shared cache; KíLín does not copy them.
- [ ] With a progress callback, it passes a progress class built by _progress_tqdm; without one, the Hub shows its own progress.
- [ ] That class disables terminal output and reports only the aggregate byte bar as (completed bytes, total bytes).
- [ ] Because a disabled tqdm stops counting, the class keeps the count itself.
- [ ] It returns the snapshot path and does not load the model.

**Errors or unsupported statements:** 

**Answer without retrieval:**

The `download_model` function in the KíLín Python package reports download progress to its caller through a callback mechanism. It allows users to pass a custom callback function that is invoked periodically during the download process. This callback typically receives parameters such as the current download progress (e.g., percentage completed), the total size of the download, and the current download speed. This enables users to integrate progress tracking into their applications, such as updating a GUI or logging status messages.

By default, the downloaded model files are stored in a user-specific cache directory. The exact path is typically something like `~/.kilin/models` or a subdirectory within a system-wide cache location (e.g., `~/.cache/kilin/models`). The function may also accept an optional parameter to specify a custom download directory, allowing users to override the default storage location. The files are often saved with a unique identifier (e.g., a hash of the model name) to avoid conflicts and ensure proper versioning. 

If you need precise details, consult the KíLín documentation or source code for confirmation.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## reference_matching (mechanism, evaluation)

**Question.** Explain how match_reference decides which project file a reference entry points to, and when a match is only probable.

**Intent:** code_explanation

**Answer with retrieval:**

The `match_reference` function determines which project file a reference entry points to by comparing the entry's words against candidate files (papers) using a scoring system. It splits the entry into words and checks for matches against each paper's content, considering rules like minimum word count and DOI matches [6]. If no candidates match and the entry is too short, it returns "unreadable." If no candidates match at all, it returns "missing." 

Matches are sorted by certainty: exact matches (`MATCH_FOUND`) are prioritized, followed by probable matches (`MATCH_PROBABLE`) if the entry's words align sufficiently with a paper but lack exact matches. The function returns the best match along with alternative candidates. A match is only probable when the entry's words meet the rules' thresholds but do not exactly match a file's content, indicating a potential but uncertain link [6]. This approach ensures references are matched based on keyword similarity and predefined criteria, while distinguishing between definitive and tentative associations.

**Cited passages:**

- [6] kilin/tools/reference_check/_matching.py `match_reference` (retrieved by: code, semantic)

**Gold functions:** `match_reference`, `_candidate`

- in evidence: ['match_reference']
- cited: ['match_reference']

**Facts (tick those the answer states correctly):**

- [ ] A file's DOI found in the entry gives a found match.
- [ ] So does the entry's first author appearing in the file's authors or first page together with the same journal citation.
- [ ] Otherwise the score is the share of the file's title words found in the entry (or of the entry's words found on the first page when there is no title), compared with found and probable thresholds.
- [ ] A high score with a different first author, or a lower score with the same first author, is only probable.
- [ ] Found matches come before probable ones, then higher scores; other matches are listed as alternatives; an entry with too few distinctive words is unreadable, and one with no candidate is missing.

**Errors or unsupported statements:** 

**Answer without retrieval:**

The `match_reference` function in the KíLín Python package is designed to resolve which project file (e.g., a `.tex` file) a reference entry (e.g., a bibliography entry) points to. It uses a combination of exact and heuristic matching strategies to determine the correct file, and it can flag matches as "probable" when ambiguity exists. Here's how it works:

---

### **1. Exact Matching**
- **Key Fields**: KíLín compares the reference entry's metadata (e.g., title, author, year, journal, volume, etc.) with the metadata of project files (e.g., `.tex` files) in the project directory.
- **Exact Match**: If the reference entry's fields exactly match a project file's metadata, the match is considered **definite**. For example, if a reference entry has the title "The Art of Computer Programming" and a project file has the same title, KíLín will map the reference to that file.

---

### **2. Fuzzy/Heuristic Matching**
- **Ambiguity Handling**: When exact matches are not found, KíLín uses **fuzzy matching** or **heuristics** to find the best possible match. This might involve:
  - **Partial Matches**: Comparing substrings of titles, authors, or other fields.
  - **Similarity Scores**: Calculating a similarity score (e.g., using algorithms like Levenshtein distance) between the reference entry and project file metadata.
  - **Prioritization**: Prioritizing matches based on fields with higher importance (e.g., title over author) or other rules (e.g., most recent file, longest match).

---

### **3. Probable Matches**
- **Ambiguity Flag**: If multiple project files have similar metadata (e.g., two files with nearly identical titles), KíLín may flag the match as **probable**. This occurs when:
  - The similarity score is high but not 100%.
  - Multiple files could plausibly match the reference entry.
  - No definitive metadata fields (e.g., unique title or author) are present.
- **User Intervention**: In such cases, the user is typically prompted to manually verify or resolve the ambiguity (e.g., by selecting the correct file or providing additional metadata).

---

### **4. Contextual Factors**
- **Project File Structure**: KíLín may also consider the structure of the project files (e.g., whether a file is a main document, a bibliography, or a section) to improve accuracy.
- **Configuration**: The behavior of `match_reference` can be influenced by configuration settings, such as thresholds for similarity scores or prioritization rules.

---

### **When a Match is Only Probable**
A match is considered **only probable** when:
- There is **no exact match**.
- The **similarity score** is high but not definitive.
- **Multiple files** could plausibly match the reference entry.
- The **metadata is incomplete or ambiguous** (e.g., missing author or title).

In such cases, KíLín may suggest the match but require the user to confirm or refine the mapping.

---

### **Summary**
`match_reference` combines exact and heuristic matching to resolve references to project files. Matches are definitive when metadata aligns precisely, but they are marked as probable when ambiguity exists (e.g., due to partial matches or similar metadata). This ensures accuracy while allowing users to resolve uncertainties manually. For precise details, consult the KíLín documentation or source code.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## term_hits_callers (dependency, evaluation)

**Question.** Explain which functions call term_hits, and what each of them does with the passages it returns.

**Intent:** code_explanation

**Answer with retrieval:**

The `term_hits` function is called by `find_passages`, `find_fact`, and `batch_review`.  

- **`find_passages`** uses `term_hits` to identify passages containing exact terms from the question. If terms exist, it filters sentences from those passages for quick listing; otherwise, it processes top-ranked passages [4][15].  
- **`find_fact`** calls `term_hits` to determine which passages mention the question's terms. If no hits, it defaults to analyzing top-ranked passages, noting missed facts [5][13].  
- **`batch_review`** uses `term_hits` to split passages into parts for model review, either focusing on term-containing passages or the most relevant ones [10].  

Each function leverages the returned passage indices (`hits`) to narrow scope, generate labels (e.g., "Exact matches only"), or structure processing steps, aligning with their goal of efficient or targeted analysis.

**Cited passages:**

- [4] kilin/tools/find_passages/tool.py `run` (retrieved by: code)
- [5] kilin/tools/find_fact/tool.py `run` (retrieved by: code)
- [10] kilin/tools/batch_review/tool.py `run` (retrieved by: context_expansion)
- [13] kilin/tools/find_fact/tool.py `run` (retrieved by: context_expansion)
- [15] kilin/tools/find_passages/tool.py `run` (retrieved by: context_expansion)

**Gold functions:** `run`, `run`, `run`

- in evidence: ['run', 'run', 'run']
- cited: ['run', 'run', 'run']

**Facts (tick those the answer states correctly):**

- [ ] term_hits asks the router model for the question's exact terms and returns them with the passages that mention any of them (whole words, ignoring case).
- [ ] batch_review's run uses them for its quick version: only the passages that mention the terms, read in parts.
- [ ] find_fact's run uses them for its quick version: only the parts that contain a hit, per file.
- [ ] find_passages' run uses them for its quick version: the sentences of those passages, found without the model.
- [ ] Without terms, each falls back to the best-ranked passages.

**Errors or unsupported statements:** 

**Answer without retrieval:**

The KíLín Python package is designed for information retrieval tasks, leveraging Elasticsearch under the hood. While specific implementation details may vary, here's a general explanation of how functions might interact with `term_hits` and what they do with the returned passages:

---

### **Functions That Call `term_hits`**
1. **`search` or `search_terms`**  
   - These are likely high-level functions that accept a query (e.g., a string or list of terms) and internally use `term_hits` to retrieve documents containing the specified terms.  
   - Example: `search("climate change")` might break the query into terms like "climate" and "change," then call `term_hits` for each term to gather matching documents.

2. **`term_hits` (lower-level function)**  
   - This function directly queries Elasticsearch for documents containing a specific term. It returns a list of "hits" (documents) that match the term.  
   - Example: `term_hits("climate")` would return all documents with the word "climate" in their content.

---

### **What Each Function Does with the Returned Passages**
1. **`term_hits`**  
   - **Input**: A term (e.g., "climate").  
   - **Output**: A list of documents (hits) that contain the term. Each hit typically includes metadata (e.g., document ID, score) and the raw text snippet matching the term.  
   - **Processing**: The function may return the raw hits or pre-processed snippets (e.g., highlighting the matched term).

2. **High-level functions (e.g., `search`)**  
   - **Aggregation**: Combines results from multiple terms (e.g., using `term_hits` for each term) and merges them into a unified list of documents.  
   - **Filtering/Sorting**: Removes duplicates, sorts documents by relevance (e.g., using Elasticsearch's scoring), or applies filters (e.g., limiting results to a specific number).  
   - **Formatting**: Returns structured results, such as a list of documents with metadata (e.g., title, score) and highlighted passages.

---

### **Example Workflow**
1. User calls `search("climate change")`.  
2. The function splits the query into terms: ["climate", "change"].  
3. For each term, `term_hits` is called to retrieve matching documents.  
4. Results are merged, duplicates are removed, and documents are sorted by relevance.  
5. The final output includes a list of documents with their matching passages and scores.

---

### **Notes**
- The exact behavior depends on the KíLín implementation. For precise details, consult the [KíLín documentation](https://kilin.readthedocs.io/).  
- If `term_hits` is not explicitly documented, it may be an internal utility function used by higher-level methods.  

If you have access to the package's source code or documentation, checking the `search` and `term_hits` functions directly would clarify the exact workflow.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## explicit_imports_callers (dependency, evaluation)

**Question.** Explain which functions call include_explicit_imports, and why each of them needs it.

**Intent:** code_explanation

**Answer with retrieval:**

The function `include_explicit_imports` is called by `RAGSession._prepare_source` [3] and `RAGSession._prepare_retrieval` [6]. Both functions need it to refine the scan report by explicitly including files directly imported by the project, overriding default scan policies that exclude files based on filename suffixes or protected-name patterns. This ensures that explicitly imported files (e.g., `.env.local` or `.xyz`) are indexed even if they would otherwise be filtered. The test case in [4] also calls `include_explicit_imports` to validate its behavior, confirming it correctly selects explicitly imported files. By integrating this function, the RAG system ensures comprehensive coverage of project dependencies while adhering to safety constraints like file size and symlink checks.

**Cited passages:**

- [3] kilin/rag.py `RAGSession._prepare_source` (retrieved by: lexical)
- [4] tests/kilin/test_workspace_source_management.py `TestWorkspaceSourceManagement.test_explicit_suffix_and_protected_file_are_imported` (retrieved by: code, semantic)
- [6] kilin/rag.py `RAGSession._prepare_retrieval` (retrieved by: context_expansion)

**Gold functions:** `RAGSession._prepare_retrieval`, `RAGSession._prepare_source`

- in evidence: ['RAGSession._prepare_retrieval', 'RAGSession._prepare_source']
- cited: ['RAGSession._prepare_retrieval', 'RAGSession._prepare_source']

**Facts (tick those the answer states correctly):**

- [ ] Both scan the project's source folder with scan_repository and then call include_explicit_imports on the report.
- [ ] _prepare_retrieval uses the scan to build the source manifest and compare it with the saved index, to decide whether to reuse or rebuild it.
- [ ] _prepare_source uses it (when no report is supplied) to choose which files to read and split into chunks.
- [ ] include_explicit_imports adds files the user imported one by one (recorded in sources.json) even when their suffix or name would otherwise be skipped; size limits and link safeguards still apply, and imported folders keep the normal rules.

**Errors or unsupported statements:** 

**Answer without retrieval:**

The KíLín Python package is not a widely known or standard library, and I do not have specific information about its internal implementation, including functions like `include_explicit_imports`. If this is a custom or niche package, its documentation or source code would need to be consulted for precise details.

In general, a function like `include_explicit_imports` might be used in code analysis tools to distinguish between **explicit** (directly written) and **implicit** (e.g., inferred or dynamic) imports. For example:

1. **`parse_imports()`**  
   - **Why it needs `include_explicit_imports`:** To filter or flag only the imports explicitly written in the code, excluding dynamically generated or inferred ones. This ensures accurate analysis of the code's structure.

2. **`generate_report()`**  
   - **Why it needs `include_explicit_imports`:** To include only explicitly declared imports in the report, avoiding misleading information from implicit imports (e.g., those resolved at runtime).

3. **`check_imports()`**  
   - **Why it needs `include_explicit_imports`:** To enforce coding standards, ensuring that all imports are explicitly stated rather than relying on implicit resolution (e.g., `from __future__` imports or dynamic `importlib` calls).

For exact details, refer to the KíLín package's documentation or source code. If you can provide more context or specific code snippets, I can offer a more tailored explanation.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## extension_callers (dependency, evaluation)

**Question.** Explain which functions call normalize_extensions, and what each of them checks with it.

**Intent:** code_explanation

**Answer with retrieval:**

The function `normalize_extensions` is called by three functions: `scan_repository` [2], `_read_profile` [3], and `_parse_project_type` [4]. Each uses it to validate and normalize file extensions according to specific rules.  

1. **`scan_repository`** [2] calls `normalize_extensions(extensions)` to process user-specified file extensions for repository scanning. It checks that extensions are valid literal suffixes (e.g., `.py`, `.md`), without wildcards or whitespace, and ensures they are lowercase and unique. This ensures only valid files are selected during repository scanning.  

2. **`_read_profile`** [3] calls `normalize_extensions(extensions)` to validate project configuration extensions. It enforces the same format rules, ensuring extensions like `.py` are correctly formatted and reject invalid inputs, maintaining consistency with repository scanning logic.  

3. **`_parse_project_type`** [4] calls `normalize_extensions(extensions)` to validate project type-specific extensions. It checks that extensions start with a period and follow the same literal suffix grammar, ensuring project types define valid file types (e.g., `.pdf`).  

Each caller uses `normalize_extensions` to enforce consistent formatting and validity, preventing malformed or ambiguous extensions in configuration and scanning contexts.

**Cited passages:**

- [2] kilin/repository.py `scan_repository` (retrieved by: code)
- [3] kilin/project.py `_read_profile` (retrieved by: code)
- [4] kilin/project_type.py `_parse_project_type` (retrieved by: lexical)

**Gold functions:** `_read_profile`, `_parse_project_type`, `scan_repository`

- in evidence: ['_read_profile', '_parse_project_type', 'scan_repository']
- cited: ['_read_profile', '_parse_project_type', 'scan_repository']

**Facts (tick those the answer states correctly):**

- [ ] normalize_extensions lowercases the suffixes, adds a leading dot, rejects wildcards, whitespace, and non-text values, and removes duplicates.
- [ ] _read_profile checks the extensions setting of a project's profile.ini.
- [ ] _parse_project_type checks the extensions declared by a project-type JSON file.
- [ ] scan_repository checks the extensions it is asked to select before scanning.

**Errors or unsupported statements:** 

**Answer without retrieval:**

The `normalize_extensions` function in the KíLín Python package is used to standardize file extensions, ensuring consistency in how files are processed and checked. While the exact implementation details may vary, here's a general explanation of how it might be used in practice:

### Functions That Call `normalize_extensions`
1. **Main Checker Functions**  
   - Functions like `check_pep8` or `check_style` might call `normalize_extensions` to ensure file extensions (e.g., `.py`, `.PY`) are standardized before applying checks. This ensures that rules like "only check `.py` files" are applied consistently.

2. **Configuration Setup Functions**  
   - When parsing configuration files (e.g., `kilin.ini`), functions that process extension lists (e.g., `load_config`) might use `normalize_extensions` to standardize user-specified extensions (e.g., converting `.PY` to `.py`).

3. **File Filtering Functions**  
   - Functions that determine which files to include/exclude based on extensions (e.g., `filter_files`) might call `normalize_extensions` to ensure the file's actual extension matches the expected format.

### What Each Function Checks
- **Consistency of Extensions**: Ensures file extensions match expected formats (e.g., `.py` instead of `.PY` or `.PY`).
- **Relevance to Checkers**: Validates that files with normalized extensions are compatible with the rules being applied (e.g., only `.py` files are checked for PEP 8 compliance).
- **Configuration Alignment**: Ensures user-specified extensions in configuration files are normalized to match the internal representation used by KíLín.

If you're using KíLín, the `normalize_extensions` function is typically part of the internal logic to maintain consistency in how files are processed, but its exact usage depends on the package's implementation. For precise details, refer to the KíLín source code or documentation.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---
