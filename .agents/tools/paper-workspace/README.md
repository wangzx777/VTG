# Paper Workspace Tools

Deterministic shared utilities used by `paper-triage` and by the optional
extraction path in `paper-read`. This directory intentionally has no `SKILL.md`,
so it is never an automatic trigger.

Stable cache interfaces:

```bash
python3 extract_paper.py --workspace <workspace> --paper-id <paper-id>
python3 validate_extraction.py --workspace <workspace> --paper-id <paper-id>
node inspect_papers_xlsx.mjs --workbook <workspace>/literature/papers.xlsx --paper-id <paper-id>
```

Additional workflow gates:

- `triage_gate.mjs`: Excel-only triage idempotency decision.
- `list_untriaged_papers.mjs`: dynamic `Not Triaged` queue.

The extraction cache contract remains:

```text
literature/extracted/<paper-id>/
├── paper.md
├── assets/
├── extraction.json
└── source.sha256
```

Legacy entry points under `.agents/skills/paper-triage/scripts/` are lightweight
compatibility wrappers around this directory.
