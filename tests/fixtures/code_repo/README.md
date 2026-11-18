# Mock code repository

A tiny calculator written once per language, used to test how KíLín reads
code. Nothing here is run. Names say which language they belong to, so an
answer is easy to check:

- `python/calculator_py.py`: `py_function_A` calls `py_function_B`;
  `PyChildClass` extends `PyBaseClass`, and its `py_method_C` calls
  `py_function_A`.
- `matlab/m_function_A.m` calls `m_function_B` in `matlab/m_function_B.m`.
- `js/calculator_js.js`: `js_function_A` calls `js_function_B`;
  `JsChildClass` extends `JsBaseClass`, and its `js_method_C` calls
  `js_function_A`; `js_function_D` uses `JsChildClass`.
- `html/page_html.html` loads `js/calculator_js.js`, and its `html_button_D`
  button calls `js_function_A`. A Code project reads it as code; other
  project types read only its text (`html_heading_A`, `html_heading_B`).

Each pull request that teaches KíLín a new file type adds one file here and
the questions that check it in
`tests/workflows/KiLin_Code_Fixture_Evaluation.ipynb`.
