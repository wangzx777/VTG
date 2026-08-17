---
name: paper-triage
description: "Idempotently triage synchronized research papers: reuse or create formula-preserving Markdown caches, perform bounded first-pass reading, assign Core/Important/Scan, and safely update literature/papers.xlsx. Use for initial triage, dynamic Not Triaged queues, or an explicitly requested retriage/refresh; automatically skip Excel rows already marked Triaged, Read, or Deep Read unless refresh was explicit. Do not use for deep paper notes, paper-to-code mapping, repository review, or reproduction work."
---

# Paper Triage

Turn one synchronized paper at a time into a reusable extraction cache and a
validated row in `literature/papers.xlsx`. Let the model interpret the paper;
use bundled scripts for identity, caching, validation, diffing, and workbook
writes.

## Read the policies

Before acting, read:

- `references/research-profile.md` for VTG relevance.
- `references/triage-schema.md` for every field and evidence rule.
- `references/priority-policy.md` before assigning priority.
- `references/extraction-policy.md` when creating or judging a cache.
- `references/backend-setup.md` when no supported parser is available.
- `references/xlsx-merge-policy.md` before changing the workbook.

## Workflow

### 1. Resolve identity and enforce the Excel idempotency gate

Read `literature/zotero-sync.json`. Treat its stored `paper_id`, `source_path`,
and Zotero item key as the identity contract. Never derive a replacement ID from
the title or filename.

If `literature/papers.xlsx` is absent, initialize it from sync state:

```bash
node scripts/init_papers_xlsx.mjs \
  --workspace <workspace-root> \
  --dry-run

node scripts/init_papers_xlsx.mjs \
  --workspace <workspace-root> \
  --apply \
  --preview work/papers-preview.png
```

Initialization reads metadata only; it creates `Not Triaged` rows without
reading PDFs. Then run the gate before reading a PDF, extraction cache, paper
text, or code source:

```bash
node <workspace-root>/.agents/tools/paper-workspace/triage_gate.mjs \
  --workbook <workspace-root>/literature/papers.xlsx \
  --paper-id <paper-id>
```

Treat its result as authoritative:

- `continue_triage` with `not_triaged`: proceed.
- `skipped_already_triaged`: stop immediately and report the existing Excel
  status. Do not read the paper, check code, or write Excel.
- `continue_triage` with `explicit_refresh`: proceed only when the user
  explicitly asked to “重新 triage”, “retriage”, or “refresh”. Pass `--refresh`
  to the gate only in that case.

Excel is the sole triage-completion ledger. A changed Zotero PDF does not bypass
this gate. The user must explicitly request refresh. Do not create a persistent
`triage/<paper-id>.json` completion ledger.

For batch work, build the queue live from Excel; never hard-code a paper list:

```bash
node <workspace-root>/.agents/tools/paper-workspace/list_untriaged_papers.mjs \
  --workbook <workspace-root>/literature/papers.xlsx
```

Set `CODEX_NODE_MODULES` to the bundled Node dependency path returned by the
workspace dependency loader if `@oai/artifact-tool` is not locally resolvable.
Do not continue on duplicate IDs, header drift, or identity ambiguity.

### 2. Reuse or create the extraction cache

Run:

```bash
python3 <workspace-root>/.agents/tools/paper-workspace/extract_paper.py \
  --workspace <workspace-root> \
  --paper-id <paper-id> \
  --backend auto
```

`auto` first reuses a hash-matched usable cache; otherwise it chooses the first
installed backend in this order: MinerU, Marker, Docling. Do not silently install
a heavy backend. If none is available, report the supported commands.

The extractor automatically adds `<workspace-root>/.venv/bin` to `PATH` when
that project environment exists. MinerU defaults to its native `hybrid-engine`.
When accelerated-runtime capability is uncertain, test MLX/MPS in an isolated
process before choosing an override. Use `--mineru-engine pipeline
--mineru-device cpu` only when acceleration is unavailable. The MinerU child
preserves an inherited `HF_HOME`; if none is inherited (common for desktop-app
tasks), it explicitly uses `~/.cache/huggingface`. If a macOS sandbox denies
native framework or multiprocessing access, stop and rerun the same extractor
command in a normal local terminal; do not patch MinerU inside `.venv`.

After extraction, read `extraction.json`. A failed cache blocks triage. A cache
with warnings requires targeted comparison with rendered PDF pages, especially
method equations, the main result table, and missing figures. The PDF remains
the claim source of truth.

Validate a cache independently when provenance or quality is in doubt:

```bash
python3 <workspace-root>/.agents/tools/paper-workspace/validate_extraction.py \
  --workspace <workspace-root> \
  --paper-id <paper-id>
```

### 3. Build a bounded reading packet

Run:

```bash
python3 scripts/build_triage_packet.py \
  --paper-md literature/extracted/<paper-id>/paper.md \
  --output work/<paper-id>-triage-packet.md
```

Read the packet in this order:

1. Abstract
2. Introduction
3. Method overview and main method figure/caption
4. Main experiments/results and primary table
5. Conclusion or limitations
6. Dataset, metric, model, training, and code-link evidence snippets

Open more of `paper.md` or the original PDF only to resolve a field. Do not turn
triage into a full read.

### 4. Produce an evidence-backed candidate record

Use `assets/triage.schema.json` as the contract and write a working
`triage.json` under `work/`. Populate all 20 Excel fields. Attach evidence for
semantic or externally checked values and record uncertainty rather than
guessing.

Hard rules:

- Set `Read Status` to `Triaged`; never downgrade `Read` or `Deep Read`.
- Use `Code = Not Checked` when code availability was not actually checked.
- Use `Code = None` only after a documented official-source check found no code.
- Never infer `Reproduce Status` from the paper. Preserve its existing value.
- Use `N/A` for genuinely inapplicable or unresolved values; explain material
  unknowns in `uncertainties`.
- Keep `Priority Reason` and `Core Idea` to one concrete sentence each.

Validate:

```bash
python3 scripts/validate_triage.py \
  --triage work/<paper-id>-triage.json \
  --schema assets/triage.schema.json \
  --extraction literature/extracted/<paper-id>/extraction.json
```

Fix every error. Treat warnings as explicit review items.

### 5. Dry-run and apply the workbook update

Always dry-run first:

```bash
node scripts/update_papers_xlsx.mjs \
  --workbook literature/papers.xlsx \
  --triage work/<paper-id>-triage.json \
  --dry-run
```

Review the field-level diff. Default conflict policy preserves different
non-placeholder existing values. Use `--conflict-policy replace` only after
checking evidence and only for triage-owned fields. An explicit refresh may
replace triage-owned fields but must preserve a higher `Read` or `Deep Read`
status and every workflow-owned field; the updater enforces both rules.

Apply in place:

```bash
node scripts/update_papers_xlsx.mjs \
  --workbook literature/papers.xlsx \
  --triage work/<paper-id>-triage.json \
  --apply \
  --preview work/<paper-id>-papers-preview.png
```

Confirm the exported workbook reopens, contains one matching row, preserves
identity fields, and has no obvious clipping or corruption in the preview.

## Completion contract

For an idempotent skip, report `skipped_already_triaged`, Paper ID, and existing
Read Status. Otherwise report:

- cache action: reused, refreshed, or created;
- backend, source hash match, extraction status, and warnings;
- Paper ID, priority, and specific reason;
- fields changed, preserved, conflicted, or uncertain;
- whether workbook apply completed;
- next action: deep read, scan only, code check, or resolve uncertainty.

Do not create translations, reading notes, topic syntheses, repository reviews,
or reproduction claims in this skill.
