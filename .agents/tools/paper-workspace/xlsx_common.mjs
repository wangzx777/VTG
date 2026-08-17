#!/usr/bin/env node
// Shared deterministic workbook contracts for paper-workspace skills.
import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

export const HEADERS = [
  "Paper ID", "Year", "Venue", "Title", "Task", "Method Family", "Focus",
  "Priority", "Priority Reason", "Read Status", "Base Model", "Core Idea",
  "Training", "Datasets", "Metrics", "Code", "Repo", "Reproduce Status",
  "Zotero Key", "Remarks",
];

export class ContractError extends Error {}

export function parseArgs(argv) {
  const args = { _: [] };
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (!token.startsWith("--")) {
      args._.push(token);
      continue;
    }
    const key = token.slice(2).replaceAll("-", "_");
    const next = argv[i + 1];
    if (!next || next.startsWith("--")) {
      args[key] = true;
    } else {
      args[key] = next;
      i += 1;
    }
  }
  return args;
}

export function requireString(args, name) {
  const value = args[name];
  if (typeof value !== "string" || !value.trim()) {
    throw new ContractError(`--${name.replaceAll("_", "-")} is required.`);
  }
  return value;
}

export async function loadArtifactTool() {
  try {
    return await import("@oai/artifact-tool");
  } catch (firstError) {
    const root = process.env.CODEX_NODE_MODULES;
    if (!root) {
      throw new ContractError(
        "@oai/artifact-tool is unavailable. Set CODEX_NODE_MODULES to the bundled Node node_modules path returned by the workspace dependency loader.",
        { cause: firstError },
      );
    }
    const modulePath = path.join(root, "@oai", "artifact-tool", "dist", "artifact_tool.mjs");
    try {
      await fs.access(modulePath);
      return await import(pathToFileURL(modulePath).href);
    } catch (secondError) {
      throw new ContractError(`Cannot load @oai/artifact-tool from ${modulePath}`, { cause: secondError });
    }
  }
}

export async function readJson(filePath) {
  let text;
  try {
    text = await fs.readFile(filePath, "utf8");
  } catch (error) {
    throw new ContractError(`Cannot read ${filePath}: ${error.message}`);
  }
  try {
    const parsed = JSON.parse(text);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      throw new Error("expected a JSON object");
    }
    return parsed;
  } catch (error) {
    throw new ContractError(`Invalid JSON in ${filePath}: ${error.message}`);
  }
}

export async function importWorkbook(filePath) {
  const { FileBlob, SpreadsheetFile } = await loadArtifactTool();
  try {
    const blob = await FileBlob.load(filePath);
    return await SpreadsheetFile.importXlsx(blob);
  } catch (error) {
    throw new ContractError(`Cannot import workbook ${filePath}: ${error.message}`, { cause: error });
  }
}

export function normalizeCell(value) {
  if (value === null || value === undefined) return "";
  return String(value).trim();
}

export function getPapersSheet(workbook) {
  try {
    return workbook.worksheets.getItem("Papers");
  } catch (error) {
    throw new ContractError("Workbook does not contain a Papers sheet.", { cause: error });
  }
}

export function readPapersTable(workbook) {
  const sheet = getPapersSheet(workbook);
  const used = sheet.getUsedRange(true);
  if (!used) throw new ContractError("Papers sheet is empty.");
  const values = used.values;
  if (!Array.isArray(values) || values.length < 1) throw new ContractError("Papers sheet has no readable values.");
  const actualHeaders = (values[0] || []).slice(0, HEADERS.length).map(normalizeCell);
  const extraHeaders = (values[0] || []).slice(HEADERS.length).map(normalizeCell).filter(Boolean);
  const missing = HEADERS.filter((header) => !actualHeaders.includes(header));
  const duplicates = actualHeaders.filter((header, index) => header && actualHeaders.indexOf(header) !== index);
  const unexpected = actualHeaders.filter((header, index) => header !== HEADERS[index]);
  if (missing.length || duplicates.length || unexpected.length || extraHeaders.length || actualHeaders.length !== HEADERS.length) {
    throw new ContractError(
      `Papers header contract failed. Missing=${JSON.stringify(missing)} duplicates=${JSON.stringify([...new Set(duplicates)])} extra=${JSON.stringify(extraHeaders)} actual=${JSON.stringify(actualHeaders)}`,
    );
  }
  const rows = values.slice(1).map((raw, index) => {
    const record = Object.fromEntries(HEADERS.map((header, col) => [header, raw[col] ?? ""]));
    return { rowNumber: index + 2, values: raw.slice(0, HEADERS.length), record };
  }).filter((row) => HEADERS.some((header) => normalizeCell(row.record[header])));
  return { sheet, used, rows };
}

export function findPaperRows(rows, paperId) {
  return rows.filter((row) => normalizeCell(row.record["Paper ID"]) === paperId);
}

export function validateUniqueIdentities(rows) {
  for (const field of ["Paper ID", "Zotero Key"]) {
    const seen = new Map();
    for (const row of rows) {
      const value = normalizeCell(row.record[field]);
      if (!value || value === "N/A") continue;
      const prior = seen.get(value);
      if (prior) throw new ContractError(`Duplicate ${field} ${JSON.stringify(value)} in rows ${prior} and ${row.rowNumber}.`);
      seen.set(value, row.rowNumber);
    }
  }
}

export function syncItemsByPaperId(state) {
  const byId = new Map();
  for (const [zoteroKey, raw] of Object.entries(state.items || {})) {
    if (!raw || typeof raw !== "object" || !raw.paper_id) continue;
    if (byId.has(raw.paper_id)) throw new ContractError(`Duplicate Paper ID ${raw.paper_id} in zotero-sync.json.`);
    byId.set(raw.paper_id, { ...raw, zotero_key: zoteroKey });
  }
  return byId;
}

export function assertSyncIdentity(syncItem, record) {
  if (!syncItem) throw new ContractError(`Paper ID ${record["Paper ID"]} is absent from zotero-sync.json.`);
  if (syncItem.paper_id !== record["Paper ID"]) throw new ContractError("Paper ID does not match sync state.");
  if (syncItem.zotero_key !== record["Zotero Key"]) {
    throw new ContractError(
      `Zotero Key mismatch for ${record["Paper ID"]}: triage=${record["Zotero Key"]} sync=${syncItem.zotero_key}`,
    );
  }
}

export function placeholder(value) {
  const normalized = normalizeCell(value).toLowerCase();
  return normalized === "" || normalized === "n/a" || normalized === "not triaged" || normalized === "not checked";
}

export function columnName(index) {
  let number = index + 1;
  let name = "";
  while (number > 0) {
    const remainder = (number - 1) % 26;
    name = String.fromCharCode(65 + remainder) + name;
    number = Math.floor((number - 1) / 26);
  }
  return name;
}

export async function renderPapers(workbook, previewPath, rowNumber = null) {
  if (!previewPath) return;
  const endRow = rowNumber ? Math.max(rowNumber, 2) : Math.min(readPapersTable(workbook).rows.length + 1, 35);
  const startRow = rowNumber ? Math.max(1, rowNumber - 1) : 1;
  const blob = await workbook.render({
    sheetName: "Papers",
    range: `A${startRow}:T${endRow}`,
    scale: 1.5,
    format: "png",
  });
  await fs.mkdir(path.dirname(path.resolve(previewPath)), { recursive: true });
  await fs.writeFile(previewPath, new Uint8Array(await blob.arrayBuffer()));
}

export async function exportAtomic(workbook, destination, verify) {
  const { SpreadsheetFile } = await loadArtifactTool();
  const absolute = path.resolve(destination);
  await fs.mkdir(path.dirname(absolute), { recursive: true });
  const temp = path.join(path.dirname(absolute), `.${path.basename(absolute)}.${process.pid}.tmp.xlsx`);
  try {
    const output = await SpreadsheetFile.exportXlsx(workbook);
    await output.save(temp);
    const reopened = await importWorkbook(temp);
    await verify(reopened);
    await fs.rm(`${temp}.inspect.ndjson`, { force: true });
    await fs.rename(temp, absolute);
  } catch (error) {
    await fs.rm(temp, { force: true });
    await fs.rm(`${temp}.inspect.ndjson`, { force: true });
    throw error;
  }
}

export function reportError(error) {
  const prefix = error instanceof ContractError ? "CONTRACT ERROR" : "ERROR";
  process.stderr.write(`${prefix}: ${error.message}\n`);
  if (!(error instanceof ContractError) && process.env.DEBUG) process.stderr.write(`${error.stack}\n`);
  process.exitCode = 2;
}
