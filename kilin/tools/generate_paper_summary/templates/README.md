# Report templates

Each Markdown file in this folder (other than this one) is a report
template. To use one, name it in the chat by its title or file name:

- "write a journal club handout of smith20"
- "fill the lab notes template for patil25"
- "use the minutes template for smith20"

KíLín reads the document in parts, writes each section of the template
from what it read, checks every section against its limits, and saves the
report to `outputs/`. When you ask for a report without naming a template,
KíLín asks which one to use. "Which templates do I have?" lists them, and
"check my journal_club template" tells you whether one can be used, or
what to fix.

To make your own, copy one of these files, rename it, and edit it. A new
file is available at the next question; nothing else needs changing.

## A complete example

```markdown
---
title: Journal Club Handout
description: One-page handout for a lab journal club discussion.
max_words: 500
---

# Headline (max 90 characters)

A title naming the finding, not the paper's title.

**Source paper:** {citation}

**In one sentence:** What the paper shows, for someone who reads nothing else.

## The Question (60-90 words)

What the authors asked and why it matters to our lab.

## What They Did (80-120 words)

The design, the data and the key methods, without detail a newcomer does not
need.

## Discussion Points (3-5 items)

Questions to discuss: assumptions, alternative explanations, what we could
use. One per line.

## Further Reading (2-3 items)

Papers from {references} that give the best context.
```

## The rules

**Front matter** (between the two `---` lines at the top):

- `title`: the template's name as you will say it ("Journal Club Handout").
- `description`: one line on what the report is for. KíLín uses it to
  recognise requests for this template.
- `max_words`: the most words for all sections together (at least 50).
- `reference` (optional): where the format comes from.

**Headline**: a line starting with `# `. With a limit, `# Headline (max 90
characters)`, KíLín writes it; without one, the line is printed as the
report's title.

**Sections**: lines starting with `## `, in the order the report uses. The
text under a heading is the instruction for writing that section, so say
what you want, and for whom. A limit in brackets at the end of the heading
is checked, and a section outside it is rewritten:

- `(60-90 words)` or `(max 100 words)`: a paragraph of that length;
- `(3-5 items)` or `(3 items)`: a list, one item per line.

A heading without brackets has no limit.

**One-line fields**: `**Label:** instruction` puts a single line in the
report, written from the instruction ("**In one sentence:** ...").

**Placeholders**:

- `{citation}`: the document's citation, taken from its metadata by code,
  never written by the model (`**Source paper:** {citation}`).
- `{references}`: in a section's instruction, its items are chosen from
  the document's own reference list and copied word for word. Such a
  section needs an item limit, such as `(2-3 items)`.

No other `{...}` placeholders are allowed.

## When a template cannot be used

KíLín says which line breaks which rule, for example a section limit it
cannot read or an unknown placeholder. "check my <name> template" shows
the problem without running on a document; "which templates do I have?"
lists every template with its problem, if any.

The model can still get facts wrong: the checks cover the report's
structure and lengths, not whether its content is correct.
