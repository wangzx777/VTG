# Triage record schema

The JSON envelope is an audit artifact used before writing the Excel row. The
canonical machine contract is `assets/triage.schema.json`.

## Excel fields

| Field | Contract |
|---|---|
| Paper ID | Existing ID from Zotero sync state; lowercase kebab-case. Never invent or change during triage. |
| Year | Publication/preprint year as an integer. Record version ambiguity in `uncertainties`. |
| Venue | CVPR, ICCV, NeurIPS, journal, or `arXiv`; `N/A` only if unresolved. |
| Title | Full title from the paper. |
| Task | Concrete task, e.g. `Video Temporal Grounding`. |
| Method Family | Primary paradigm, e.g. `Video-LLM`, `DETR-based`, `Transformer-based`, `Specialized Model`. |
| Focus | Main problem emphasis; separate co-primary labels with `;`. |
| Priority | Exactly `Core`, `Important`, or `Scan`. |
| Priority Reason | One concrete sentence explaining reading value for the VTG program. |
| Read Status | `Triaged` after this workflow; never downgrade `Read` or `Deep Read`. |
| Base Model | Main backbone/foundation model; `N/A` if absent or unresolved. |
| Core Idea | One sentence stating the central method innovation. |
| Training | Main regime such as `SFT`, `SFT+GRPO`, `End-to-End`; `N/A` if unclear. |
| Datasets | Primary datasets separated with `;`; no unverified expansion. |
| Metrics | Primary metrics separated with `;`; preserve spelling where protocol differences matter. |
| Code | Exactly `Official`, `Unofficial`, `None`, or `Not Checked`. |
| Repo | Repository URL or `N/A`. Never guess a URL. |
| Reproduce Status | Preserve the workbook value. New records use `N/A`. |
| Zotero Key | Stable Zotero parent-item key from sync state. Never infer or replace. |
| Remarks | Only important exceptions not represented elsewhere; otherwise `N/A`. |

## Evidence

Use field-keyed evidence items:

```json
{
  "source": "paper",
  "locator": "paper.md:L120-L138 (3 Method)",
  "quote": "Short supporting excerpt or faithful compact paraphrase"
}
```

Allowed sources are `paper`, `pdf`, `xlsx`, `zotero`, `official-page`, `repo`,
and `inference`. Label taxonomy or priority conclusions as `inference`, while
also citing the paper facts on which they rest.

Require evidence for `Title`, `Year`, `Venue`, `Task`, `Method Family`, `Focus`,
`Priority`, `Priority Reason`, `Base Model`, `Core Idea`, `Training`, `Datasets`,
`Metrics`, `Code`, and `Repo`. Identity/workflow fields may cite the workbook or
Zotero state.

## Unknowns and conflicts

Do not disguise uncertainty with a confident-looking cell. Put a conservative
value in `record` and add:

```json
{
  "field": "Venue",
  "reason": "The PDF is an arXiv version and does not state an accepted venue.",
  "action": "Check the official proceedings page."
}
```

- `Not Checked`: a required external check was not performed (`Code`).
- `N/A`: not applicable, absent, or unresolved under the field contract.
- uncertainty item: explains why a material value is provisional.
