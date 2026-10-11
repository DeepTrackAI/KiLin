# Source-extraction fixtures

These small, synthetic files are checked into the test suite. They are not
imported or executed. The Python fixture intentionally raises an exception
at module scope to catch accidental execution by integration tests.

`sample.pdf` is a two-page, text-only PDF created with ReportLab. Tests
read the committed PDF and do not require ReportLab at runtime. PDF parser
dependencies remain optional and are checked by the PDF tests.

Edge cases and failures use in-memory `SourceFile` objects or mocks instead
of adding large numbers of specialized files. The large scientific-paper
PDF regression is optional and is not distributed as a test fixture.
