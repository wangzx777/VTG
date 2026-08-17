#!/usr/bin/env node
// Inspect one Papers row without mutating the workbook.
import path from "node:path";
import {
  ContractError, assertSyncIdentity, findPaperRows, importWorkbook, parseArgs,
  readJson, readPapersTable, reportError, requireString, syncItemsByPaperId,
  validateUniqueIdentities,
} from "./xlsx_common.mjs";

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const workbookPath = requireString(args, "workbook");
  const paperId = requireString(args, "paper_id");
  const workbook = await importWorkbook(workbookPath);
  const table = readPapersTable(workbook);
  validateUniqueIdentities(table.rows);
  const matches = findPaperRows(table.rows, paperId);
  if (matches.length !== 1) throw new ContractError(`Paper ID ${paperId} matched ${matches.length} rows; expected exactly one.`);
  const row = matches[0];
  const workspace = path.dirname(path.dirname(path.resolve(workbookPath)));
  const state = await readJson(path.join(workspace, "literature", "zotero-sync.json"));
  const syncItem = syncItemsByPaperId(state).get(paperId);
  assertSyncIdentity(syncItem, row.record);
  console.log(JSON.stringify({
    workbook: workbookPath,
    sheet: "Papers",
    row: row.rowNumber,
    record: row.record,
    sync_identity: {
      paper_id: syncItem.paper_id,
      zotero_key: syncItem.zotero_key,
      source_path: syncItem.source_path,
    },
  }, null, 2));
}

main().catch(reportError);
