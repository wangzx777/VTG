# Extraction and cache policy

## Source hierarchy

`paper.md` is a reusable reading cache. The source PDF remains authoritative for
paper claims, equations, tables, captions, and page layout.

## Cache identity

A reusable cache requires:

- matching SHA-256 for the synchronized source PDF;
- `paper.md` with a matching Markdown hash in `extraction.json`;
- a non-failed validation status;
- a supported extraction schema version.

File existence alone never proves cache validity.

## Backend policy

Prefer:

1. MinerU;
2. Marker;
3. Docling.

Use automatic text/OCR routing where supported. Do not OCR every page of a
normal born-digital paper by default: unnecessary OCR can degrade equations and
characters. Force OCR only when the PDF is scanned, has a broken text layer, or
the automatic result fails quality checks.

Keep the adapter boundary stable so the backend can be changed after benchmark
testing without changing the triage workflow.

The adapter is the non-triggering shared tool layer at
`.agents/tools/paper-workspace/`. Both `paper-triage` and `paper-read` call the
same extractor/validator contract; neither owns the normalized cache.

MinerU's local hybrid engine uses MLX on Apple silicon. In a restricted macOS
sandbox, native framework access or Python multiprocessing may fail before
parsing begins. Treat `SC_SEM_NSEMS_MAX` permission failures and native
`NSException` crashes as runtime blockers: stop and rerun the same extractor
command in a normal local terminal. Do not patch MinerU inside `.venv` or
misdiagnose these failures as missing model weights.

## Normalized cache

```text
literature/extracted/<paper-id>/
├── paper.md
├── assets/
├── extraction.json
└── source.sha256
```

`extraction.json` records source identity, backend and version, command, time,
page count when available, output hashes, asset inventory, quality signals, and
warnings. Do not store secrets or a full environment dump.

## Quality gate

Hard failures include unreadably short Markdown, excessive replacement/control
characters, missing source identity, or a Markdown hash mismatch.

Warnings include suspiciously low text per page, missing headings, or referenced
but absent images. Warnings require targeted PDF comparison before using
affected content.

Verify against PDF pages when:

- a formula appears flattened, truncated, or converted into prose;
- a table loses column/row structure;
- two-column reading order is interleaved;
- a caption is detached from its figure;
- an extracted number conflicts with surrounding prose;
- the extraction report carries a relevant warning.

Never silently edit `paper.md` to make a claim look cleaner. Prefer re-extraction
or refer to the PDF; if manual correction is unavoidable, record it in metadata.
