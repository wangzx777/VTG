#!/usr/bin/env node
/** Atomically advance one Papers row's Read Status after validating its paper-read note. */

import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import {
  ContractError, HEADERS, assertSyncIdentity, exportAtomic, findPaperRows,
  importWorkbook, normalizeCell, parseArgs, readJson, readPapersTable,
  renderPapers, reportError, requireString, syncItemsByPaperId, validateUniqueIdentities,
} from "./xlsx_common.mjs";

const READ_RANK = new Map([["Not Triaged", 0], ["Triaged", 1], ["Read", 2], ["Deep Read", 3]]);
const LEVEL_FOR_STATUS = new Map([["Read", "read"], ["Deep Read", "deep"]]);
const REQUIRED_FRONTMATTER = [
  "schema_version", "paper_id", "source_sha256", "paper_md_sha256",
  "read_level", "extraction_status", "generated_at",
];
const REQUIRED_HEADINGS = [
  "一句话结论", "研究问题与任务定义", "输入、输出和关键假设",
  "方法总览、模块职责与端到端数据流", "关键公式和时间表示",
  "训练阶段、损失/奖励与数据", "推理流程", "数据集、指标与实验设置",
  "主结果及数值一致性检查", "消融实验", "Claim–Evidence Map",
  "论文级可复现性清单", "局限、失败模式和未确认事项",
  "对当前 VTG 研究的启示", "后续 code-map 应追踪的概念", "抽取质量与核验",
  "我的笔记",
];
const MARKERS = [
  "<!-- PAPER-READ:MANAGED-START -->", "<!-- PAPER-READ:MANAGED-END -->",
  "<!-- PAPER-READ:USER-NOTES-START -->", "<!-- PAPER-READ:USER-NOTES-END -->",
];
const DEEP_SUBHEADINGS = [
  "公式逐项解释与推导关系", "关键图表与核心实验链路",
  "虚拟实现、伪代码与必要配置", "隐含假设、反例与证据强度",
  "论文文本缺失的复现细节",
];

function countLiteral(text, value) {
  return text.split(value).length - 1;
}

function parseFrontmatter(text) {
  const match = text.match(/^---\r?\n([\s\S]*?)\r?\n---(?:\r?\n|$)/);
  if (!match) throw new ContractError("Note must begin with YAML frontmatter.");
  const values = {};
  for (const line of match[1].split(/\r?\n/)) {
    if (!line.trim() || line.trimStart().startsWith("#")) continue;
    const separator = line.indexOf(":");
    if (separator < 1) throw new ContractError(`Unsupported frontmatter line: ${JSON.stringify(line)}`);
    const key = line.slice(0, separator).trim();
    let value = line.slice(separator + 1).trim();
    if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1);
    }
    if (Object.hasOwn(values, key)) throw new ContractError(`Duplicate frontmatter key: ${key}`);
    values[key] = value;
  }
  for (const key of REQUIRED_FRONTMATTER) {
    if (!values[key]) throw new ContractError(`Missing note frontmatter field: ${key}`);
  }
  return values;
}

async function sha256File(filePath) {
  let data;
  try {
    data = await fs.readFile(filePath);
  } catch (error) {
    throw new ContractError(`Cannot read ${filePath}: ${error.message}`);
  }
  return crypto.createHash("sha256").update(data).digest("hex");
}

async function validateNote({ notePath, workspace, paperId, targetStatus, syncItem }) {
  const text = await fs.readFile(notePath, "utf8").catch((error) => {
    throw new ContractError(`Cannot read note ${notePath}: ${error.message}`);
  });
  const frontmatter = parseFrontmatter(text);
  const expectedLevel = LEVEL_FOR_STATUS.get(targetStatus);
  if (frontmatter.schema_version !== "1.0") throw new ContractError(`Unsupported note schema_version: ${frontmatter.schema_version}`);
  if (frontmatter.paper_id !== paperId) throw new ContractError("Note paper_id does not match --paper-id.");
  if (frontmatter.read_level !== expectedLevel) {
    throw new ContractError(`Note read_level=${frontmatter.read_level} does not match target ${targetStatus}.`);
  }
  const generatedAt = new Date(frontmatter.generated_at);
  if (Number.isNaN(generatedAt.valueOf()) || !/(?:Z|[+-]\d{2}:\d{2})$/.test(frontmatter.generated_at)) {
    throw new ContractError("generated_at must be a timezone-aware ISO-8601 timestamp.");
  }
  if (!new Set(["pass", "warning"]).has(frontmatter.extraction_status)) {
    throw new ContractError(`Invalid extraction_status: ${frontmatter.extraction_status}`);
  }
  for (const marker of MARKERS) {
    if (countLiteral(text, marker) !== 1) throw new ContractError(`Note must contain exactly one ${marker}`);
  }
  const positions = MARKERS.map((marker) => text.indexOf(marker));
  if (!(positions[0] < positions[1] && positions[1] < positions[2] && positions[2] < positions[3])) {
    throw new ContractError("Note managed/user markers are out of order.");
  }
  const headingPositions = [];
  for (const heading of REQUIRED_HEADINGS) {
    const matches = text.match(new RegExp(`^## ${heading.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\s*$`, "gm")) || [];
    if (matches.length !== 1) throw new ContractError(`Note must contain exactly one level-2 heading: ${heading}`);
    headingPositions.push(text.indexOf(`## ${heading}`));
  }
  if (!headingPositions.every((value, index) => index === 0 || headingPositions[index - 1] < value)) {
    throw new ContractError("Required level-2 headings are out of contract order.");
  }
  if (expectedLevel === "deep") {
    for (const heading of DEEP_SUBHEADINGS) {
      const matches = text.match(new RegExp(`^### ${heading.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\s*$`, "gm")) || [];
      if (matches.length !== 1) throw new ContractError(`Deep note must contain exactly one level-3 heading: ${heading}`);
    }
  }
  const managed = text.slice(positions[0], positions[1]);
  if (!/待确认|未确认/.test(managed)) throw new ContractError("Managed note must explicitly record unconfirmed items.");
  const cacheDir = path.join(workspace, "literature", "extracted", paperId);
  const extraction = await readJson(path.join(cacheDir, "extraction.json"));
  if (!["pass", "warning"].includes(extraction.status)) throw new ContractError(`Extraction status is ${extraction.status}.`);
  const sourcePath = path.resolve(workspace, syncItem.source_path);
  if (!sourcePath.startsWith(`${path.resolve(workspace)}${path.sep}`)) throw new ContractError("Synchronized PDF path escapes workspace.");
  const [sourceHash, paperMdHash, recordedSourceHash] = await Promise.all([
    sha256File(sourcePath),
    sha256File(path.join(cacheDir, "paper.md")),
    fs.readFile(path.join(cacheDir, "source.sha256"), "utf8").then((value) => value.trim()),
  ]);
  const expected = {
    source_sha256: sourceHash,
    paper_md_sha256: paperMdHash,
    extraction_status: extraction.status,
  };
  for (const [field, value] of Object.entries(expected)) {
    if (frontmatter[field] !== value) throw new ContractError(`Note ${field} is stale or mismatched.`);
  }
  if (extraction.source_sha256 !== sourceHash || recordedSourceHash !== sourceHash) {
    throw new ContractError("Extraction cache does not match the current synchronized PDF.");
  }
  if (extraction.paper_md_sha256 !== paperMdHash) throw new ContractError("paper.md hash does not match extraction.json.");
  if (extraction.status === "warning") {
    const section = text.slice(text.indexOf("## 抽取质量与核验"), text.indexOf("## 我的笔记"));
    if (!/warning|告警|警告/i.test(section)) {
      throw new ContractError("Extraction warning is not acknowledged in the note's extraction-quality section.");
    }
    if (!/pdf/i.test(section)) {
      throw new ContractError("Extraction warning requires a targeted PDF check recorded in the note.");
    }
  }
  return { frontmatter, sourceHash, paperMdHash, extractionStatus: extraction.status };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (Boolean(args.apply) === Boolean(args.dry_run)) throw new ContractError("Specify exactly one of --dry-run or --apply.");
  const workbookPath = path.resolve(requireString(args, "workbook"));
  const notePath = path.resolve(requireString(args, "note"));
  const paperId = requireString(args, "paper_id");
  const targetStatus = requireString(args, "status");
  if (!LEVEL_FOR_STATUS.has(targetStatus)) throw new ContractError("--status must be Read or Deep Read.");
  const workspace = path.dirname(path.dirname(workbookPath));

  const state = await readJson(path.join(workspace, "literature", "zotero-sync.json"));
  const syncItem = syncItemsByPaperId(state).get(paperId);
  if (!syncItem) throw new ContractError(`Paper ID ${paperId} is absent from zotero-sync.json.`);
  const workbook = await importWorkbook(workbookPath);
  const table = readPapersTable(workbook);
  validateUniqueIdentities(table.rows);
  const matches = findPaperRows(table.rows, paperId);
  if (matches.length !== 1) throw new ContractError(`Paper ID ${paperId} matched ${matches.length} rows; expected exactly one.`);
  const target = matches[0];
  assertSyncIdentity(syncItem, target.record);
  const priority = normalizeCell(target.record.Priority);
  if (!new Set(["Core", "Important", "Scan"]).has(priority)) {
    throw new ContractError(`Paper has no valid triage Priority: ${JSON.stringify(priority)}`);
  }
  const currentStatus = normalizeCell(target.record["Read Status"]);
  if (!READ_RANK.has(currentStatus)) throw new ContractError(`Invalid current Read Status: ${currentStatus}`);
  if (currentStatus === "Not Triaged") throw new ContractError("Paper must be triaged before paper-read can advance its status.");
  const readStatusIndex = HEADERS.indexOf("Read Status");
  if (String(target.values[readStatusIndex] ?? "").startsWith("=")) {
    throw new ContractError("Read Status is formula-backed; refusing to overwrite it.");
  }
  const noteValidation = await validateNote({ notePath, workspace, paperId, targetStatus, syncItem });
  const currentRank = READ_RANK.get(currentStatus);
  const targetRank = READ_RANK.get(targetStatus);
  if (currentStatus === "Deep Read" && targetStatus === "Read" && noteValidation.frontmatter.read_level !== "deep") {
    throw new ContractError("A read-level note cannot replace or justify an existing Deep Read status.");
  }
  const nextStatus = targetRank > currentRank ? targetStatus : currentStatus;
  const action = nextStatus === currentStatus ? "preserved_higher_or_equal_status" : (args.apply ? "apply" : "dry-run");
  const report = {
    action, workbook: workbookPath, note: notePath, paper_id: paperId,
    row: target.rowNumber, current_status: currentStatus, requested_status: targetStatus,
    next_status: nextStatus, note_validation: noteValidation,
  };
  if (!args.apply || nextStatus === currentStatus) {
    console.log(JSON.stringify(report, null, 2));
    return;
  }

  const before = table.rows.map((row) => row.values.slice(0, HEADERS.length).map(normalizeCell));
  table.sheet.getRangeByIndexes(target.rowNumber - 1, readStatusIndex, 1, 1).values = [[nextStatus]];
  await exportAtomic(workbook, workbookPath, async (reopened) => {
    const reopenedTable = readPapersTable(reopened);
    validateUniqueIdentities(reopenedTable.rows);
    if (reopenedTable.rows.length !== before.length) throw new ContractError("Reopened workbook row count changed.");
    for (let rowIndex = 0; rowIndex < before.length; rowIndex += 1) {
      for (let colIndex = 0; colIndex < HEADERS.length; colIndex += 1) {
        const expected = rowIndex === target.rowNumber - 2 && colIndex === readStatusIndex
          ? nextStatus : before[rowIndex][colIndex];
        const actual = normalizeCell(reopenedTable.rows[rowIndex].values[colIndex]);
        if (actual !== expected) {
          throw new ContractError(`Unexpected reopened value change at row ${rowIndex + 2}, ${HEADERS[colIndex]}: ${actual} vs ${expected}`);
        }
      }
    }
  });
  await renderPapers(workbook, args.preview || null, target.rowNumber);
  console.log(JSON.stringify({ ...report, action: "applied", applied: true, preview: args.preview || null }, null, 2));
}

main().catch(reportError);
