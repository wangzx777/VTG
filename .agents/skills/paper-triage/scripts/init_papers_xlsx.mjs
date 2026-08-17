#!/usr/bin/env node
import fs from "node:fs/promises";
import path from "node:path";
import {
  ContractError, HEADERS, exportAtomic, loadArtifactTool, parseArgs, readJson,
  readPapersTable, renderPapers, reportError, syncItemsByPaperId,
} from "./xlsx_common.mjs";

function yearFrom(item) {
  for (const value of [item.date, item.venue, item.paper_id]) {
    const match = String(value || "").match(/(?:19|20)\d{2}/);
    if (match) return Number(match[0]);
  }
  throw new ContractError(`Cannot resolve year for ${item.paper_id}.`);
}

function venueFrom(item) {
  const venue = String(item.venue || "").trim();
  return venue ? venue.replace(/\s+(?:19|20)\d{2}\s*$/, "").trim() : "N/A";
}

function rowFrom(item) {
  return [
    item.paper_id,
    yearFrom(item),
    venueFrom(item),
    String(item.title || "N/A").trim() || "N/A",
    "N/A", "N/A", "N/A", "N/A", "N/A", "Not Triaged", "N/A", "N/A",
    "N/A", "N/A", "N/A", "Not Checked", "N/A", "N/A", item.zotero_key, "N/A",
  ];
}

async function buildWorkbook(rows) {
  const { Workbook } = await loadArtifactTool();
  const workbook = Workbook.create();
  const sheet = workbook.worksheets.add("Papers");
  const matrix = [HEADERS, ...rows];
  sheet.getRangeByIndexes(0, 0, matrix.length, HEADERS.length).values = matrix;
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(1);
  sheet.showGridLines = false;

  const full = sheet.getRange(`A1:T${matrix.length}`);
  full.format.font = { name: "Aptos", size: 10, color: "#172033" };
  full.format.verticalAlignment = "center";
  const header = sheet.getRange("A1:T1");
  header.format.fill = "#17365D";
  header.format.font = { name: "Aptos", size: 10, bold: true, color: "#FFFFFF" };
  header.format.rowHeight = 30;
  header.format.horizontalAlignment = "center";
  const body = sheet.getRange(`A2:T${Math.max(matrix.length, 2)}`);
  body.format.wrapText = true;
  body.format.rowHeight = 42;
  sheet.getRange(`B2:B${Math.max(matrix.length, 2)}`).format.numberFormat = "0";

  const widths = [190, 65, 90, 320, 170, 150, 160, 80, 300, 100, 150, 330, 130, 210, 190, 100, 260, 130, 100, 260];
  widths.forEach((width, index) => {
    sheet.getRangeByIndexes(0, index, matrix.length, 1).format.columnWidthPx = width;
  });
  sheet.getRange(`H2:H2000`).dataValidation = { rule: { type: "list", values: ["Core", "Important", "Scan"] } };
  sheet.getRange(`J2:J2000`).dataValidation = { rule: { type: "list", values: ["Not Triaged", "Triaged", "Read", "Deep Read"] } };
  sheet.getRange(`P2:P2000`).dataValidation = { rule: { type: "list", values: ["Official", "Unofficial", "None", "Not Checked"] } };
  sheet.getRange(`R2:R2000`).dataValidation = { rule: { type: "list", values: ["N/A", "Not Started", "Env Ready", "Inference", "Evaluation", "Training", "Reproduced", "Failed"] } };
  const table = sheet.tables.add(`A1:T${matrix.length}`, true, "PapersTable");
  table.style = "TableStyleMedium2";
  table.showFilterButton = true;
  return workbook;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const workspace = path.resolve(String(args.workspace || process.cwd()));
  const workbookPath = path.resolve(String(args.output || path.join(workspace, "literature", "papers.xlsx")));
  const statePath = path.join(workspace, "literature", "zotero-sync.json");
  const state = await readJson(statePath);
  const items = [...syncItemsByPaperId(state).values()].filter((item) => item.in_scope !== false);
  items.sort((a, b) => yearFrom(b) - yearFrom(a) || venueFrom(a).localeCompare(venueFrom(b)) || String(a.title).localeCompare(String(b.title)));
  const rows = items.map(rowFrom);
  const summary = { action: args.apply ? "apply" : "dry-run", workbook: workbookPath, rows: rows.length, first_paper_ids: rows.slice(0, 5).map((row) => row[0]) };
  if (!args.apply) {
    console.log(JSON.stringify(summary, null, 2));
    return;
  }
  try {
    await fs.access(workbookPath);
    if (!args.replace) throw new ContractError(`Workbook already exists: ${workbookPath}. Refusing to replace without --replace.`);
  } catch (error) {
    if (error instanceof ContractError) throw error;
    if (error.code !== "ENOENT") throw error;
  }
  const workbook = await buildWorkbook(rows);
  await exportAtomic(workbook, workbookPath, async (reopened) => {
    const table = readPapersTable(reopened);
    if (table.rows.length !== rows.length) throw new ContractError(`Reopened row count mismatch: ${table.rows.length} vs ${rows.length}`);
  });
  await renderPapers(workbook, args.preview || null);
  console.log(JSON.stringify({ ...summary, applied: true, preview: args.preview || null }, null, 2));
}

main().catch(reportError);
