#!/usr/bin/env node
// Offline integration test for workbook initialization and guarded updates.

import assert from "node:assert/strict";
import crypto from "node:crypto";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const sharedToolsDir = path.resolve(scriptDir, "../../../tools/paper-workspace");
const node = process.execPath;

function run(script, args, expectedCode = 0, baseDir = scriptDir) {
  return new Promise((resolve, reject) => {
    const child = spawn(node, [path.join(baseDir, script), ...args], {
      env: process.env,
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code !== expectedCode) {
        reject(new Error(`${script} exited ${code}, expected ${expectedCode}\nstdout:\n${stdout}\nstderr:\n${stderr}`));
      } else {
        resolve({ stdout, stderr });
      }
    });
  });
}

function makeTriage(zoteroKey = "ABCDEFGH", readStatus = "Triaged") {
  const evidenceFields = [
    "Year", "Venue", "Title", "Task", "Method Family", "Focus", "Priority",
    "Priority Reason", "Base Model", "Core Idea", "Training", "Datasets",
    "Metrics", "Code", "Repo",
  ];
  return {
    schema_version: "1.0",
    paper_id: "2026-test-paper",
    source_sha256: crypto.createHash("sha256").update("fixture").digest("hex"),
    record: {
      "Paper ID": "2026-test-paper", Year: 2026, Venue: "CVPR", Title: "Test Paper",
      Task: "Video Temporal Grounding", "Method Family": "Video-LLM", Focus: "Boundary Modeling",
      Priority: "Important", "Priority Reason": "Provides a reusable boundary-modeling baseline for VTG.",
      "Read Status": readStatus, "Base Model": "N/A", "Core Idea": "Predicts an interval with a boundary-aware decoder.",
      Training: "End-to-End", Datasets: "Charades-STA", Metrics: "mIoU", Code: "Not Checked",
      Repo: "N/A", "Reproduce Status": "N/A", "Zotero Key": zoteroKey, Remarks: "N/A",
    },
    evidence: Object.fromEntries(evidenceFields.map((field) => [field, [{ source: ["Priority", "Priority Reason", "Method Family", "Focus", "Code", "Repo"].includes(field) ? "inference" : "paper", locator: "fixture", quote: "fixture evidence" }]])),
    checks: { paper_read: true, official_code_source_checked: false },
    uncertainties: [],
  };
}

async function main() {
  assert.ok(process.env.CODEX_NODE_MODULES, "Set CODEX_NODE_MODULES to the bundled Node dependency path.");
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "paper-triage-xlsx-test-"));
  try {
    const literature = path.join(root, "literature");
    await fs.mkdir(literature, { recursive: true });
    const state = {
      schema_version: 3,
      items: {
        ABCDEFGH: {
          paper_id: "2026-test-paper", title: "Test Paper", date: "2026", venue: "CVPR 2026",
          source_path: "literature/sources/CVPR 2026/2026-test-paper.pdf", in_scope: true,
        },
      },
    };
    await fs.writeFile(path.join(literature, "zotero-sync.json"), JSON.stringify(state));
    const workbook = path.join(literature, "papers.xlsx");
    const triagePath = path.join(root, "triage.json");
    await fs.writeFile(triagePath, JSON.stringify(makeTriage()));

    await run("init_papers_xlsx.mjs", ["--workspace", root, "--apply"]);
    const inspectedBefore = JSON.parse((await run(
      "inspect_papers_xlsx.mjs", ["--workbook", workbook, "--paper-id", "2026-test-paper"], 0, sharedToolsDir,
    )).stdout);
    assert.equal(inspectedBefore.record["Read Status"], "Not Triaged");
    assert.equal(inspectedBefore.record["Zotero Key"], "ABCDEFGH");

    const dryRun = JSON.parse((await run("update_papers_xlsx.mjs", ["--workbook", workbook, "--triage", triagePath, "--dry-run"])).stdout);
    assert.ok(dryRun.summary.change.includes("Read Status"));
    assert.equal(dryRun.summary.conflict.length, 0);
    await run("update_papers_xlsx.mjs", ["--workbook", workbook, "--triage", triagePath, "--apply"]);
    const inspectedAfter = JSON.parse((await run(
      "inspect_papers_xlsx.mjs", ["--workbook", workbook, "--paper-id", "2026-test-paper"], 0, sharedToolsDir,
    )).stdout);
    assert.equal(inspectedAfter.record.Priority, "Important");
    assert.equal(inspectedAfter.record["Read Status"], "Triaged");

    const triagedGate = JSON.parse((await run(
      "triage_gate.mjs", ["--workbook", workbook, "--paper-id", "2026-test-paper"], 0, sharedToolsDir,
    )).stdout);
    assert.equal(triagedGate.action, "skipped_already_triaged");
    const refreshGate = JSON.parse((await run(
      "triage_gate.mjs", ["--workbook", workbook, "--paper-id", "2026-test-paper", "--refresh"], 0, sharedToolsDir,
    )).stdout);
    assert.equal(refreshGate.action, "continue_triage");
    assert.equal(refreshGate.reason, "explicit_refresh");

    const queueAfterTriage = JSON.parse((await run(
      "list_untriaged_papers.mjs", ["--workbook", workbook], 0, sharedToolsDir,
    )).stdout);
    assert.deepEqual(queueAfterTriage.paper_ids, []);

    for (const completedStatus of ["Read", "Deep Read"]) {
      await fs.writeFile(triagePath, JSON.stringify(makeTriage("ABCDEFGH", completedStatus)));
      await run("update_papers_xlsx.mjs", ["--workbook", workbook, "--triage", triagePath, "--apply"]);
      const completedGate = JSON.parse((await run(
        "triage_gate.mjs", ["--workbook", workbook, "--paper-id", "2026-test-paper"], 0, sharedToolsDir,
      )).stdout);
      assert.equal(completedGate.action, "skipped_already_triaged");
      assert.equal(completedGate.read_status, completedStatus);
    }

    await fs.writeFile(triagePath, JSON.stringify(makeTriage("ABCDEFGH", "Triaged")));
    const refreshDryRun = JSON.parse((await run(
      "update_papers_xlsx.mjs", ["--workbook", workbook, "--triage", triagePath, "--dry-run", "--conflict-policy", "replace"],
    )).stdout);
    assert.ok(refreshDryRun.summary.preserve.includes("Read Status"));
    await run("update_papers_xlsx.mjs", ["--workbook", workbook, "--triage", triagePath, "--apply", "--conflict-policy", "replace"]);
    const afterRefresh = JSON.parse((await run(
      "inspect_papers_xlsx.mjs", ["--workbook", workbook, "--paper-id", "2026-test-paper"], 0, sharedToolsDir,
    )).stdout);
    assert.equal(afterRefresh.record["Read Status"], "Deep Read");

    const badPath = path.join(root, "bad.json");
    await fs.writeFile(badPath, JSON.stringify(makeTriage("WRONGKEY")));
    const failed = await run("update_papers_xlsx.mjs", ["--workbook", workbook, "--triage", badPath, "--dry-run"], 2);
    assert.match(failed.stderr, /Zotero Key mismatch/);
    console.log("xlsx integration test: PASS");
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
}

main().catch((error) => {
  console.error(error.stack || error.message);
  process.exitCode = 1;
});
