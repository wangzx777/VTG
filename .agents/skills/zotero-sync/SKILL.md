---
name: zotero-sync
description: Synchronize papers from the direct child collections of a local Zotero Desktop Sync collection into literature/sources/, preserving each child collection as a Venue folder and using deterministic Zotero-item-key Paper IDs. Use when importing or refreshing the workspace paper source mirror from Zotero. This skill never creates or edits papers.xlsx and never reads or summarizes papers.
---

# Zotero Sync

Synchronize the local Zotero Desktop `Sync` collection into the VTG workspace.

## Preconditions

- Zotero Desktop is running on the same Mac as the workspace.
- Zotero local API is enabled and reachable at `http://localhost:23119/api/`.
- Zotero contains one root collection named `Sync` (or another root explicitly supplied with `--sync-root`).
- The direct children of `Sync` are Venue collections, for example:

```text
Sync/
├── CVPR 2025
├── CVPR 2026
├── ICCV 2025
├── NeurIPS 2025
├── WACV 2025
└── WACV 2026
```

## Scope

This skill only synchronizes Zotero papers into `literature/sources/`.

It may:

- read the `Sync` root collection and its direct child collections;
- read items and PDF attachments from those Venue collections;
- copy/update PDFs in the workspace;
- maintain `literature/zotero-sync.json` as machine sync state.

It must not:

- create or edit `literature/papers.xlsx`;
- create `literature/extracted/` files;
- triage, read, summarize, or classify papers;
- create notes, topics, or code maps;
- modify Zotero.

## Identity and naming

Use a deterministic Paper ID:

```text
Paper ID = zotero-<lowercase Zotero parent item key>
```

Example:

```text
Zotero item key: ABCD1234
Paper ID:        zotero-abcd1234
```

Never derive Paper ID from title, year, Venue, DOI, or PDF filename.

A Venue collection maps directly to one source directory:

```text
Sync/CVPR 2025/<paper>
→ literature/sources/CVPR 2025/zotero-abcd1234.pdf
```

Attachment filename changes in Zotero never change the Paper ID or workspace filename.

## Collection rules

- Only direct children of the configured `Sync` root are synchronized.
- Collections outside `Sync` are ignored.
- A paper may belong to any number of collections outside `Sync`; this does not create duplicates.
- Inside `Sync`, one paper must belong to exactly one direct Venue child.
- If the same Zotero item appears in two or more direct `Sync` children, report `CONFLICT` and do not create duplicate PDFs or guess the Venue.
- Nested collections below a Venue child are not traversed.

## Safety

- Run `--dry-run` before the first real sync or after reorganizing `Sync`.
- Never guess a missing/ambiguous Venue.
- Never change Paper IDs for already synchronized Zotero item keys.
- Never delete a paper merely because it disappears from `Sync`; mark it out of scope in state and leave the local PDF untouched.
- If a paper moves from one valid Venue child to another, update its canonical source path after a successful copy and remove only its previous sync-managed source copy.

## Commands

### Verify local Zotero API

```bash
curl -H 'Zotero-API-Version: 3' \
  'http://localhost:23119/api/users/0/collections'
```

### List collections

```bash
python3 .agents/skills/zotero-sync/scripts/zotero_sync.py --list-collections
```

### Dry run Sync

```bash
python3 .agents/skills/zotero-sync/scripts/zotero_sync.py \
  --sync-root Sync \
  --dry-run
```

### Real sync

```bash
python3 .agents/skills/zotero-sync/scripts/zotero_sync.py \
  --sync-root Sync
```

Repeat the real-sync command whenever Zotero changes.

## Expected output

```text
literature/
├── zotero-sync.json
└── sources/
    ├── CVPR 2025/
    │   └── zotero-xxxxxxxx.pdf
    ├── ICCV 2025/
    │   └── zotero-yyyyyyyy.pdf
    └── ...
```

## Validation

Run the offline test suite:

```bash
python3 .agents/skills/zotero-sync/scripts/test_zotero_sync.py
```

Then run the live API check, `--list-collections`, a `--dry-run`, and finally a real sync.
