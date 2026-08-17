# Papers workbook merge policy

## Precondition

The `Papers` sheet must contain each canonical header exactly once. Paper ID must
resolve to exactly one row. Refuse duplicate IDs, missing headers, identity
mismatches, formulas in a target row, or a missing row unless exact sync-state
identity authorizes append.

## Field ownership

Identity fields:

- `Paper ID`: immutable.
- `Zotero Key`: preserve; mismatch is an error.

Workflow-owned field:

- `Reproduce Status`: preserve; triage cannot advance or reset it.

Monotonic field:

- `Read Status`: advance `Not Triaged` to `Triaged`; preserve `Read` and
  `Deep Read`.

Triage-owned fields:

- Year, Venue, Title, Task, Method Family, Focus, Priority, Priority Reason,
  Base Model, Core Idea, Training, Datasets, Metrics, Code, Repo, Remarks.

## Conflict behavior

Always show a dry-run diff before applying.

Default `preserve` behavior:

- fill blanks or placeholders;
- apply identical normalized values without reporting a change;
- preserve different non-placeholder values and report a conflict;
- upgrade `Code` from `Not Checked` only when the candidate contains required
  evidence;
- preserve `Remarks` unless the candidate adds a distinct important exception.

Use `replace` only after reviewing evidence. It may replace triage-owned fields,
but cannot replace identity, reproduction status, or downgrade read status.

## Write integrity

- Import and inspect the existing workbook before changing values.
- Change only the resolved data row.
- Preserve styles, formulas, tables, filters, validations, and other sheets.
- Export to a sibling temporary file, reopen it, verify the target row, then use
  an atomic rename for in-place apply.
- Render the relevant range after modification and inspect it visually.
- Report changed, preserved, conflicted, and rejected fields.
