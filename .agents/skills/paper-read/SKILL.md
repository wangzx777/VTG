---
name: paper-read
description: "Read one research-paper PDF and create two Chinese Markdown files beside its venue folder: a full paragraph-aligned translation and a concise AI reading note in the VTG template. Use for 读论文, 翻译论文, 生成论文笔记, 论文翻译和摘要, or an explicit $paper-read request. Reuse an existing extraction cache when available; create a persistent OCR/MinerU cache only when the user explicitly asks for cache, OCR, MinerU, or higher-fidelity extraction. Do not update papers.xlsx or perform reproduction work."
---

# Paper Read

Produce exactly two durable files for one PDF:

```text
literature/sources/<venue-year>/translations/<pdf-stem>.md
literature/sources/<venue-year>/notes/<pdf-stem>.md
```

Keep the PDF filename stem unchanged. Do not create candidate files under
`work/`, update Excel, or require the paper to be triaged first.

## Load the contracts

Read `references/output-contract.md` before writing either file. Read
`references/tag-pool.md` when filling `## Tags`. Copy the structure in
`assets/note-template.md` exactly for the note.

## Resolve the paper

Accept either an exact PDF stem/paper ID or a PDF path. Resolve it before
reading:

```bash
python3 <skill-dir>/scripts/resolve_paper.py \
  --workspace <workspace-root> \
  --paper <paper-id-or-pdf-path> \
  --create-dirs
```

Require exactly one matching PDF under `literature/sources/`. Use the JSON
paths returned by the script. If an output already exists, skip that output by
default; overwrite it only when the user explicitly requests refresh,
regeneration, or replacement.

## Choose the reading source

Use the least expensive adequate source in this order:

1. Reuse a valid `literature/extracted/<pdf-stem>/paper.md` cache when it already
   exists and matches the PDF hash.
2. Otherwise read the PDF directly with available PDF text/layout tools. A
   temporary extraction may be used but must not be saved in the workspace.
3. Create or refresh the persistent extraction cache only when the user says
   `with cache`, `生成缓存`, `OCR`, `MinerU`, `高精度提取`, or equivalent:

```bash
python3 <workspace-root>/.agents/tools/paper-workspace/extract_paper.py \
  --workspace <workspace-root> \
  --paper-id <pdf-stem>
```

Add `--force-ocr` only when OCR is explicitly requested or the PDF has no usable
text layer. Do not silently start MinerU merely because direct extraction is
imperfect. If the PDF cannot be read well enough for a full translation, stop
and recommend rerunning with cache/OCR.

## Create the translation

Translate the complete translatable prose into Chinese in source order. Keep
paragraph boundaries aligned with the original as closely as possible so the
user can compare paragraph by paragraph.

- Use Markdown headings only for the paper title and section/subsection titles.
- Keep body paragraphs as plain text: no bold, bullets, blockquotes, or added
  commentary.
- Preserve formulas in place. Exact LaTeX is preferred; approximate recognition
  is acceptable when it remains easy to locate in the PDF.
- Omit figures, figure captions, tables, table captions, and bibliography
  entries. Do not copy their original text into the translation.
- Keep short inline citation markers such as `[12]` only when useful for source
  alignment; omit the References/Bibliography section itself.
- Do not summarize, critique, or insert explanations into the translation.

Before finishing, compare the source section sequence with the translated
heading sequence and confirm that every prose section before the bibliography
was covered.

## Create the AI note

Fill `assets/note-template.md` in Chinese without changing its heading order.
Be brief and useful for later recall; this is a reading aid, not a peer review.

- Derive claims from the paper. Use `N/A` or `未确认` rather than guessing.
- Make the method flow concrete and end it at `Temporal Boundary`.
- Select only a small, paper-specific subset from `references/tag-pool.md`;
  never try to cover the whole pool. Multiple selected labels use `; `, and a
  slot with no applicable label is `N/A`.
- Fill `Repo` only from a URL stated in the paper or already verified in the
  workspace. Otherwise write `Not Checked`.
- Fill `关键文件` only for a Core paper whose repository was actually inspected.
  Otherwise write `N/A` or `Not Checked`; never invent paths.
- In `我还没懂`, record the three most important unresolved questions after the
  first read. Write `无` only when nothing material remains.

Do not inspect a repository solely to complete this two-file task unless the
user explicitly asks. Do not make reproduction claims.

## Validate and finish

After both files are written, run:

```bash
python3 <skill-dir>/scripts/validate_outputs.py \
  --workspace <workspace-root> \
  --paper <paper-id-or-pdf-path>
```

Fix structural errors before reporting completion. The validator checks paths,
forbidden translation formatting, note headings, required fields, method flow,
and tag vocabulary; semantic completeness still requires comparing against the
PDF.

Report the two final paths, whether an existing cache was reused or a new cache
was requested, and any prose that could not be translated reliably. Do not
claim that figures, tables, bibliography, code, or experiments were reviewed
unless the user separately requested that work.
