#!/usr/bin/env node
/** Decide whether paper-triage may proceed, using papers.xlsx as the sole completion ledger. */

import path from "node:path";
import {
  ContractError, assertSyncIdentity, findPaperRows, importWorkbook, normalizeCell,
  parseArgs, readJson, readPapersTable, reportError, requireString,
  syncItemsByPaperId, validateUniqueIdentities,
} from "./xlsx_common.mjs";

const VALID_STATUSES = new Set(["Not Triaged", "Triaged", "Read", "Deep Read"]);
const COMPLETED_STATUSES = new Set(["Triaged", "Read", "Deep Read"]);

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const workbookPath = requireString(args, "workbook");
  const paperId = requireString(args, "paper_id");
  const workbook = await importWorkbook(workbookPath);
  const table = readPapersTable(workbook);
  validateUniqueIdentities(table.rows);
  const matches = findPaperRows(table.rows, paperId);
  if (matches.length !== 1) {
    throw new ContractError(`Paper ID ${paperId} matched ${matches.length} rows; expected exactly one.`);
  }
  const row = matches[0];
  const workspace = path.dirname(path.dirname(path.resolve(workbookPath)));
  const state = await readJson(path.join(workspace, "literature", "zotero-sync.json"));
  const syncItem = syncItemsByPaperId(state).get(paperId);
  assertSyncIdentity(syncItem, row.record);
  const status = normalizeCell(row.record["Read Status"]);
  if (!VALID_STATUSES.has(status)) throw new ContractError(`Invalid Read Status for ${paperId}: ${JSON.stringify(status)}`);

  let action = "continue_triage";
  let reason = "not_triaged";
  if (args.refresh) {
    reason = "explicit_refresh";
  } else if (COMPLETED_STATUSES.has(status)) {
    action = "skipped_already_triaged";
    reason = "excel_status_completed";
  }
  console.log(JSON.stringify({
    action,
    reason,
    workbook: workbookPath,
    sheet: "Papers",
    row: row.rowNumber,
    paper_id: paperId,
    zotero_key: syncItem.zotero_key,
    read_status: status,
    record: row.record,
  }, null, 2));
}

main().catch(reportError);
