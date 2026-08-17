#!/usr/bin/env node
import fs from "node:fs/promises";
import path from "node:path";
import {
  ContractError, HEADERS, assertSyncIdentity, exportAtomic, findPaperRows,
  importWorkbook, normalizeCell, parseArgs, placeholder, readJson, readPapersTable,
  renderPapers, reportError, requireString, syncItemsByPaperId, validateUniqueIdentities,
} from "./xlsx_common.mjs";

const IDENTITY_FIELDS = new Set(["Paper ID", "Zotero Key"]);
const TRIAGE_FIELDS = new Set([
  "Year", "Venue", "Title", "Task", "Method Family", "Focus", "Priority",
  "Priority Reason", "Base Model", "Core Idea", "Training", "Datasets",
  "Metrics", "Code", "Repo", "Remarks",
]);
const READ_RANK = new Map([["Not Triaged", 0], ["Triaged", 1], ["Read", 2], ["Deep Read", 3]]);

function evidencePresent(triage, field) {
  return Array.isArray(triage.evidence?.[field]) && triage.evidence[field].length > 0;
}

function nextValue(field, current, candidate, triage, policy) {
  const currentText = normalizeCell(current);
  const candidateText = normalizeCell(candidate);
  if (IDENTITY_FIELDS.has(field)) {
    if (currentText && currentText !== candidateText) return { action: "reject", value: current, reason: "immutable identity mismatch" };
    return { action: currentText === candidateText ? "same" : "change", value: candidate, reason: "identity fill" };
  }
  if (field === "Reproduce Status") {
    if (!currentText && candidateText === "N/A") {
      return { action: "change", value: candidate, reason: "initialize missing workflow status" };
    }
    return currentText === candidateText
      ? { action: "same", value: current, reason: "workflow-owned field preserved" }
      : { action: "preserve", value: current, reason: "workflow-owned field preserved" };
  }
  if (field === "Read Status") {
    const currentRank = READ_RANK.get(currentText);
    const candidateRank = READ_RANK.get(candidateText);
    if (currentRank === undefined || candidateRank === undefined) return { action: "reject", value: current, reason: "invalid Read Status" };
    if (candidateRank > currentRank) return { action: "change", value: candidate, reason: "monotonic status advance" };
    if (candidateRank < currentRank) return { action: "preserve", value: current, reason: "prevent status downgrade" };
    return { action: "same", value: current, reason: "same value" };
  }
  if (!TRIAGE_FIELDS.has(field)) return { action: "reject", value: current, reason: "unowned field" };
  if (currentText === candidateText) return { action: "same", value: current, reason: "same value" };
  if (field === "Code" && currentText === "Not Checked" && candidateText !== "Not Checked" && !evidencePresent(triage, "Code")) {
    return { action: "reject", value: current, reason: "Code upgrade lacks evidence" };
  }
  if (field === "Code" && candidateText !== "Not Checked" && triage.checks?.official_code_source_checked !== true) {
    return { action: "reject", value: current, reason: "Code decision lacks an official-source check" };
  }
  if (placeholder(currentText)) return { action: "change", value: candidate, reason: "fill placeholder" };
  if (policy === "replace") return { action: "change", value: candidate, reason: "explicit replace policy" };
  return { action: "conflict", value: current, reason: "different non-placeholder value" };
}

function buildDiff(currentRecord, candidateRecord, triage, policy) {
  return HEADERS.map((field) => {
    const decision = nextValue(field, currentRecord[field], candidateRecord[field], triage, policy);
    return { field, current: currentRecord[field], candidate: candidateRecord[field], ...decision };
  });
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const workbookPath = path.resolve(requireString(args, "workbook"));
  const triagePath = path.resolve(requireString(args, "triage"));
  if (Boolean(args.apply) === Boolean(args.dry_run)) throw new ContractError("Specify exactly one of --dry-run or --apply.");
  const policy = String(args.conflict_policy || "preserve");
  if (!["preserve", "replace"].includes(policy)) throw new ContractError("--conflict-policy must be preserve or replace.");

  const triage = await readJson(triagePath);
  const candidate = triage.record;
  if (!candidate || typeof candidate !== "object") throw new ContractError("triage.record is missing.");
  if (candidate["Paper ID"] !== triage.paper_id) throw new ContractError("triage.paper_id differs from record Paper ID.");
  if (new Set(Object.keys(candidate)).size !== HEADERS.length || HEADERS.some((field) => !(field in candidate))) {
    throw new ContractError("triage.record does not contain the exact 20-field workbook contract.");
  }
  const enums = {
    Priority: new Set(["Core", "Important", "Scan"]),
    "Read Status": new Set(["Not Triaged", "Triaged", "Read", "Deep Read"]),
    Code: new Set(["Official", "Unofficial", "None", "Not Checked"]),
    "Reproduce Status": new Set(["N/A", "Not Started", "Env Ready", "Inference", "Evaluation", "Training", "Reproduced", "Failed"]),
  };
  for (const [field, allowed] of Object.entries(enums)) {
    if (!allowed.has(candidate[field])) throw new ContractError(`Invalid ${field}: ${JSON.stringify(candidate[field])}`);
  }
  if (candidate.Code === "Not Checked" && candidate.Repo !== "N/A") {
    throw new ContractError("Repo must be N/A when Code is Not Checked.");
  }
  if (candidate.Code === "None" && candidate.Repo !== "N/A") {
    throw new ContractError("Repo must be N/A when Code is None.");
  }
  if (["Official", "Unofficial"].includes(candidate.Code) && !/^https?:\/\//.test(String(candidate.Repo))) {
    throw new ContractError(`Code=${candidate.Code} requires an http(s) Repo URL.`);
  }

  const workspace = path.dirname(path.dirname(workbookPath));
  const state = await readJson(path.join(workspace, "literature", "zotero-sync.json"));
  const syncItem = syncItemsByPaperId(state).get(candidate["Paper ID"]);
  assertSyncIdentity(syncItem, candidate);

  const workbook = await importWorkbook(workbookPath);
  const table = readPapersTable(workbook);
  validateUniqueIdentities(table.rows);
  const matches = findPaperRows(table.rows, candidate["Paper ID"]);
  if (matches.length !== 1) throw new ContractError(`Paper ID ${candidate["Paper ID"]} matched ${matches.length} rows; expected exactly one. Initialize/refresh papers.xlsx first.`);
  const target = matches[0];
  if (target.values.some((value) => typeof value === "string" && value.startsWith("="))) {
    throw new ContractError(`Target row ${target.rowNumber} appears to contain a formula; refusing value overwrite.`);
  }
  const diff = buildDiff(target.record, candidate, triage, policy);
  const rejected = diff.filter((item) => item.action === "reject");
  if (rejected.length) throw new ContractError(`Rejected field decisions: ${JSON.stringify(rejected)}`);
  const summary = Object.fromEntries(["change", "same", "preserve", "conflict"].map((action) => [action, diff.filter((item) => item.action === action).map((item) => item.field)]));
  const report = { action: args.apply ? "apply" : "dry-run", policy, workbook: workbookPath, row: target.rowNumber, paper_id: candidate["Paper ID"], summary, diff };
  if (!args.apply) {
    console.log(JSON.stringify(report, null, 2));
    return;
  }
  if (summary.conflict.length) throw new ContractError(`Unresolved conflicts under preserve policy: ${summary.conflict.join(", ")}. Review evidence or use --conflict-policy replace.`);
  const finalRow = diff.map((item) => item.value);
  table.sheet.getRangeByIndexes(target.rowNumber - 1, 0, 1, HEADERS.length).values = [finalRow];
  await exportAtomic(workbook, workbookPath, async (reopened) => {
    const reopenedTable = readPapersTable(reopened);
    validateUniqueIdentities(reopenedTable.rows);
    const reopenedMatches = findPaperRows(reopenedTable.rows, candidate["Paper ID"]);
    if (reopenedMatches.length !== 1) throw new ContractError("Reopened workbook lost the target identity.");
    for (let index = 0; index < HEADERS.length; index += 1) {
      const actual = normalizeCell(reopenedMatches[0].record[HEADERS[index]]);
      const expected = normalizeCell(finalRow[index]);
      if (actual !== expected) throw new ContractError(`Reopened value mismatch for ${HEADERS[index]}: ${actual} vs ${expected}`);
    }
  });
  await renderPapers(workbook, args.preview || null, target.rowNumber);
  console.log(JSON.stringify({ ...report, applied: true, preview: args.preview || null }, null, 2));
}

main().catch(reportError);
