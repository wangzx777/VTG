#!/usr/bin/env node
/** Build a live paper-triage queue from rows whose Excel Read Status is Not Triaged. */

import path from "node:path";
import {
  assertSyncIdentity, importWorkbook, normalizeCell, parseArgs, readJson,
  readPapersTable, reportError, requireString, syncItemsByPaperId,
  validateUniqueIdentities,
} from "./xlsx_common.mjs";

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const workbookPath = requireString(args, "workbook");
  const workbook = await importWorkbook(workbookPath);
  const table = readPapersTable(workbook);
  validateUniqueIdentities(table.rows);
  const workspace = path.dirname(path.dirname(path.resolve(workbookPath)));
  const state = await readJson(path.join(workspace, "literature", "zotero-sync.json"));
  const syncItems = syncItemsByPaperId(state);
  const untriagedRows = table.rows.filter((row) => normalizeCell(row.record["Read Status"]) === "Not Triaged");
  for (const row of untriagedRows) assertSyncIdentity(syncItems.get(normalizeCell(row.record["Paper ID"])), row.record);
  const queue = untriagedRows.map((row) => ({
      row: row.rowNumber,
      paper_id: normalizeCell(row.record["Paper ID"]),
      zotero_key: normalizeCell(row.record["Zotero Key"]),
      title: normalizeCell(row.record.Title),
      priority: normalizeCell(row.record.Priority),
    }));
  console.log(JSON.stringify({
    action: "queue_built",
    source: "papers.xlsx",
    workbook: workbookPath,
    count: queue.length,
    paper_ids: queue.map((item) => item.paper_id),
    queue,
  }, null, 2));
}

main().catch(reportError);
