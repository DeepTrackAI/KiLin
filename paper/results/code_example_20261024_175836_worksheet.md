# Code example: grading worksheet

Indexed revision: 8bd20824c (HEAD)

## python_units (location, evaluation)

**Question.** How does KíLín divide a Python file into the parts it indexes, and what happens to a file that is not valid Python?

**Intent:** code_explanation

**Answer with retrieval:**

KíLín divides a Python file into parts by using the `extract_source_units` function, which splits the file into lines and processes them through `_python_units` to identify natural units (e.g., functions, classes, or code blocks) [6]. These units are then indexed as `SourceUnit` objects. For invalid Python files, the evidence does not explicitly describe error handling; however, if a file lacks valid syntax, the processing might fail or return an empty list if the file is empty [6]. The repository scanning tools (e.g., `scan_repository`) focus on file metadata and structure rather than syntax validation [11], implying invalid Python files may not be explicitly flagged unless they trigger errors during processing.

**Cited passages:**

- [6] kilin/sources/extensions/sources_py.py `extract_source_units` (retrieved by: context_expansion)
- [11] kilin/repository.py `scan_repository` (retrieved by: context_expansion)

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

KíLín extracts text from PDFs using a combination of `pdfminer.six` and `pypdf`. The `read_source_file` function first checks if the PDF is encrypted; if so, it raises an error [12]. Otherwise, it uses `pdfminer.six`'s layout analysis to extract text, splitting pages by form-feed characters (`\f`) [12]. If `pdfminer.six` fails, it falls back to `pypdf`'s plain text extraction [12]. The extracted text is cleaned to remove malformed Unicode characters and isolated surrogate pairs via `_clean_extracted_text` [9]. 

When a PDF contains no text (e.g., images, encrypted files, or non-extractable content), `read_source_file` checks if any page has non-empty text after cleaning. If not, it raises a `ValueError`, indicating the PDF lacks extractable text and suggests OCR might be needed [12]. Encrypted PDFs are explicitly rejected during reading [12]. Thus, KíLín relies on layout analysis and fallback methods but cannot extract text from entirely image-based or encrypted PDFs.

**Cited passages:**

- [9] kilin/sources/extensions/sources_pdf.py `_clean_extracted_text` (retrieved by: context_expansion)
- [12] kilin/sources/extensions/sources_pdf.py `read_source_file` (retrieved by: context_expansion)

**Gold functions:** `read_source_file`, `_layout_pages`, `extract_source_units`

- in evidence: ['read_source_file', '_layout_pages', 'extract_source_units']
- cited: ['read_source_file']

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

KíLín identifies function calls by traversing a `CodeGraph` that explicitly maps "calls" relationships between functions [1]. When analyzing which functions call others, it starts with named functions, then follows call steps up to a specified depth, prioritizing callees whose names share words with the query [6]. If a called name could refer to multiple functions (e.g., ambiguous identifiers), KíLín resolves this by ranking candidates based on: (1) shared words with the query, (2) proximity to the named functions (closer call steps first), and (3) the callee's role in the repository (functions that call more other functions are preferred) [1]. Ambiguity is further addressed by ignoring the named functions' own words during ranking, focusing instead on the query's context, and using semantic search to match identifier variants (e.g., "written" matches "write") [7]. If multiple functions share the same name, KíLín may exclude them unless the query explicitly mentions tests or clarifies the context [7]. This approach ensures relevant, context-aware call relationships are prioritized.

**Cited passages:**

- [1] kilin/retrievals/context_expansion.py `_ranked_callees` (retrieved by: semantic)
- [6] kilin/retrievals/context_expansion.py `plan_context_expansion` (retrieved by: context_expansion)
- [7] kilin/retrievals/context_expansion.py `plan_context_expansion` (retrieved by: context_expansion)

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

Passages in KíLín's keyword search are scored using TF-IDF cosine similarity, as implemented in `lexical_search` [12]. This function converts query text into lowercase one-word and two-word terms via `lexical_terms` [2], then computes cosine similarity between these terms and indexed chunks. Scores are ranked, with higher scores first, and tied results ordered by chunk identifier [13]. 

Code identifiers are split into searchable words using `_identifier_words` [8], which inserts spaces between lowercase/uppercase letters and replaces underscores with spaces (e.g., `_parse_route` becomes `parse route`). Words are then extracted using `_WORD.findall`, converted to lowercase, and filtered to exclude common words and short terms [1]. This process ensures identifiers are broken into meaningful, case-folded components for matching. 

These methods align with KíLín's focus on lexical and semantic retrieval, balancing precision with efficiency.

**Cited passages:**

- [1] kilin/tools/reference_check/_matching.py `_words` (retrieved by: code)
- [2] kilin/retrievals/strategies/__init__.py `__init__` (retrieved by: semantic)
- [8] kilin/retrievals/context_expansion.py `_identifier_words` (retrieved by: context_expansion)
- [12] kilin/retrievals/strategies/retrieval_lexical.py `lexical_search` (retrieved by: context_expansion)
- [13] kilin/retrievals/strategies/retrieval_lexical.py `lexical_search` (retrieved by: context_expansion)

**Gold functions:** `lexical_terms`, `lexical_search`, `_tfidf_vector`

- in evidence: ['lexical_search']
- cited: ['lexical_search']

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

The `refers_back` function determines if a question depends on earlier conversation by checking for pronouns ("it"), ordinal references ("the second one"), pointers back ("you said"), elliptical openings ("and for X?"), or short questions (≤ 2 words) [2]. If any of these patterns are present, it returns `True`, indicating the question references prior turns. 

The `search_text` function uses this information to build a search query by appending the questions from related earlier turns (identified by `decision.related_turn_ids`) to the original question. If `refers_back` is `True` and there are related turns, it concatenates the user’s question with the earlier questions (oldest first), ensuring context is included without adding model-generated text [1]. For example, if the question is "how do I install it?" and a prior turn’s question is "what is NOVA," the search query becomes "how do I install it? what is NOVA" [5]. This linkage ensures the search query retains conversational context.

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

The `download_model` function reports progress to its caller through a progress event emitted by the `ChatEngine._download_with_progress` method. When a model is downloaded, the `progress_callback` parameter (defined in `kilin/huggingface.py:303-341`) is used to track bytes transferred. This callback is linked to a custom progress bar (`_progress_tqdm` in `kilin/huggingface.py:379-425`), which updates a shared state with completed and expected byte counts. These values are periodically sent to the frontend via the `progress_event` ("model_download_progress") using `self._send` in `kilin/gui/engine.py:790-837`. The progress includes metrics like elapsed time, download rate, and estimated time remaining.  

Downloaded model files are stored in Hugging Face's shared cache, determined by the `snapshot_download` function. The `download_model` function returns a `Path` object pointing to the directory containing the model files. This path is derived from the Hugging Face cache location, and the function does not copy or manage the files directly [3]. The cache directory is typically controlled by the Hugging Face library, and the `model_is_available` function can check if a model exists in the cache [11].

**Cited passages:**

- [3] kilin/huggingface.py `download_model` (retrieved by: code, lexical)
- [11] kilin/huggingface.py `model_is_available` (retrieved by: context_expansion)

**Gold functions:** `download_model`, `_progress_tqdm`

- in evidence: ['download_model', '_progress_tqdm']
- cited: ['download_model']

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

The `match_reference` function determines which project file a reference entry points to by evaluating matches based on **DOI**, **author**, **citation**, and **content similarity**. It uses the `rules` parameter to define thresholds for distinguishing between "found" and "probable" matches. 

1. **Exact Matches (Found):**  
   - If the entry's DOI matches the candidate file's DOI, or the first author and citation match, the match is marked as `MATCH_FOUND` [6][7].  
   - Content similarity is calculated by comparing distinctive words (filtered to exclude common terms) between the entry and the candidate file's title or first page [9][8]. If the score meets or exceeds the strict threshold (`rules.title_found` or `rules.page_found`) and the author matches, it is a found match [8].

2. **Probable Matches:**  
   - If the score is high but does not meet the strict threshold (e.g., `rules.title_probable`), or the author does not match, the match is marked as `MATCH_PROBABLE` [8]. This occurs when the entry and file share significant content but lack definitive identifiers like a DOI or exact author match.  
   - Probable matches are prioritized lower than found matches, and their reasons include discrepancies like differing authors or partial content overlap [6][8].

The function ensures that only the best match (highest score) for each file is retained, with ties resolved by DOI precedence and score [13]. A match is "only probable" when the evidence is strong but insufficient to confirm an exact match, such as partial word overlap or missing author information [8].

**Cited passages:**

- [6] kilin/tools/reference_check/_matching.py `match_reference` (retrieved by: code, semantic)
- [7] kilin/tools/reference_check/_matching.py `_candidate` (retrieved by: context_expansion)
- [8] kilin/tools/reference_check/_matching.py `_candidate` (retrieved by: context_expansion)
- [9] kilin/tools/reference_check/_matching.py `_words` (retrieved by: context_expansion)
- [13] kilin/tools/reference_check/_matching.py `match_references` (retrieved by: lexical, semantic)

**Gold functions:** `match_reference`, `_candidate`

- in evidence: ['match_reference', '_candidate']
- cited: ['match_reference', '_candidate']

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

The `term_hits` function is called by `batch_review`, `find_fact`, and `find_passages` to identify passages containing exact terms from a question.  

- **`batch_review`** [10]: Uses `term_hits` to get terms and their matching passage indices (`hits`). If no hits, it alerts the user no passages mention the terms. Otherwise, it partitions passages into quick review parts based on hits.  
- **`find_fact`** [13]: Calls `term_hits` to filter passages (`hits`) containing terms. These are used to create `quick_by_file`, a list of passage indices per file, which are prioritized for rapid fact-checking.  
- **`find_passages`** [15]: Uses `term_hits` to determine `quick_chunks` (passages with hits) for quick sentence listing. If no terms, it defaults to ranked passages.  

Each function leverages `term_hits`'s output (`terms` and `hits`) to narrow scope, either for rapid filtering (e.g., `quick_parts`) or to guide further processing (e.g., `ranked` passages). The returned passage indices (`hits`) are critical for focusing on relevant content, ensuring efficiency in subsequent steps.

**Cited passages:**

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

The function `include_explicit_imports` is called by `RAGSession._prepare_source` [3] and `RAGSession._prepare_retrieval` [6]. Both require it to refine scan reports by adding files explicitly imported via `sources.json`, ensuring they are included in the RAG system’s processing even if they bypass default scan criteria (e.g., non-standard extensions or protected names). 

In `RAGSession._prepare_source` [3], it adjusts the scan report to include explicitly imported files, which are critical for building the code graph and embeddings. This ensures all relevant files are indexed, even if they wouldn’t be selected by the initial scan. 

In `RAGSession._prepare_retrieval` [6], it ensures the retrieval process includes explicitly imported files, maintaining consistency between the scan and the index. This is necessary for accurate retrieval, as excluded files might be required for query relevance. 

Test cases like `TestWorkspaceSourceManagement.test_explicit_suffix_and_protected_file_are_imported` [4] also call `include_explicit_imports` to validate its behavior, confirming it correctly selects explicitly imported files regardless of their filename patterns. This ensures the function adheres to its design intent.

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

The function `normalize_extensions` is called by three main functions: `scan_repository` [2], `_read_profile` in `project.py` [6], and `_parse_project_type` in `project_type.py` [4].  

1. **`scan_repository`** [2] uses `normalize_extensions` to process user-specified file extensions, ensuring they follow the literal suffix grammar (e.g., `.py`, `.md`). It checks for valid formatting, removes duplicates, and enforces lowercase, rejecting wildcards or invalid patterns.  
2. **`_read_profile`** [6] validates project configuration extensions, ensuring they start with a period and conform to the same literal suffix rules. It also ensures extensions are normalized and free of invalid characters.  
3. **`_parse_project_type`** [4] checks project type definitions, verifying that extensions are valid literals and properly formatted. It raises errors if extensions lack a leading period or contain invalid characters.  

Each caller ensures extensions adhere to the same normalization rules, rejecting invalid inputs like wildcards or malformed strings, while standardizing them for consistent use in file selection and configuration.

**Cited passages:**

- [2] kilin/repository.py `scan_repository` (retrieved by: code)
- [4] kilin/project_type.py `_parse_project_type` (retrieved by: lexical)
- [6] kilin/project.py `_read_profile` (retrieved by: context_expansion)

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
