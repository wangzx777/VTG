# Output Contract

## Paths

For a source PDF at:

```text
literature/sources/CVPR 2025/example-paper.pdf
```

write exactly:

```text
literature/sources/CVPR 2025/translations/example-paper.md
literature/sources/CVPR 2025/notes/example-paper.md
```

Do not add frontmatter, hashes, generated timestamps, managed-region markers, or
auxiliary JSON files. Do not write to `work/`. Existing outputs are immutable by
default; replace one only after an explicit refresh/regenerate request.

## Translation

Translate all prose from title through appendices, excluding the bibliography.
Preserve source order and approximate one translated paragraph per source
paragraph. Translate headings and prose; retain mathematical symbols and
formulas in place.

Omit these source blocks entirely:

- figures and screenshots;
- figure captions;
- tables and their cell contents;
- table captions and table notes;
- bibliography/reference entries.

Do not replace omitted blocks with copied English, summaries, or long
placeholders. A short plain-text marker such as `此处为图 2，已省略。` is allowed
only when it materially helps paragraph-level PDF alignment.

Only headings may use Markdown heading syntax. Body content must not use bold,
italics, bullet lists, blockquotes, fenced code, or Markdown tables. Display and
inline math are allowed. Preserve inline citation numbers only when helpful for
locating the paragraph; do not translate cited titles or bibliography entries.

## AI note

Use the template exactly. Keep it compact: prefer one to three bullets per
section and one line per method-flow stage. Values not supported by the paper
must be `N/A`, `Not Checked`, or `未确认`.

The tag pool is a vocabulary, not a checklist. Select only the few labels that
directly describe this paper; do not add labels merely to cover categories. The
five required tag lines are grouping slots, not new vocabulary:

- `task:` contains `task:*` labels.
- `role:` contains `role:*` labels.
- `training:` may contain `train:*`, `opt:*`, and `data:*` labels.
- `time:` may contain `time:*`, `reason:*`, `out:*`, and `long:*` labels.
- `visual:` may contain `visual:*`, `arch:*`, `setting:*`, and `modal:*` labels.

Separate multiple selected labels with `; `. Use `N/A` when no label in the pool
applies to a slot. The validator checks only that chosen labels are allowed and
placed in a compatible slot; it never requires every label or category.

## Scope

This skill creates two files only. It does not update `papers.xlsx`, change Read
Status, triage priority, create code maps, clone repositories, or claim
reproduction. Persistent extraction is optional and lives under
`literature/extracted/<pdf-stem>/` only when requested or already present.
