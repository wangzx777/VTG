#!/usr/bin/env python3
"""Validate a triage JSON record without optional third-party dependencies."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlparse

from paper_triage_common import HEADERS, PAPER_ID_RE, TRIAGE_SCHEMA_VERSION, read_json

ENUMS = {
    "Priority": {"Core", "Important", "Scan"},
    "Read Status": {"Not Triaged", "Triaged", "Read", "Deep Read"},
    "Code": {"Official", "Unofficial", "None", "Not Checked"},
    "Reproduce Status": {"N/A", "Not Started", "Env Ready", "Inference", "Evaluation", "Training", "Reproduced", "Failed"},
}
EVIDENCE_FIELDS = {
    "Title", "Year", "Venue", "Task", "Method Family", "Focus", "Priority",
    "Priority Reason", "Base Model", "Core Idea", "Training", "Datasets",
    "Metrics", "Code", "Repo",
}
EVIDENCE_SOURCES = {"paper", "pdf", "xlsx", "zotero", "official-page", "repo", "inference"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triage", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--extraction", type=Path, required=True)
    return parser.parse_args()


def is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def main() -> int:
    args = parse_args()
    data = read_json(args.triage)
    schema = read_json(args.schema)
    extraction = read_json(args.extraction)
    errors: list[str] = []
    warnings: list[str] = []

    if schema.get("title") != "VTG paper triage record":
        errors.append("Unexpected triage schema file.")
    allowed_top = {"schema_version", "paper_id", "source_sha256", "record", "evidence", "checks", "uncertainties"}
    missing_top = allowed_top - set(data)
    extra_top = set(data) - allowed_top
    if missing_top:
        errors.append(f"Missing top-level keys: {sorted(missing_top)}")
    if extra_top:
        errors.append(f"Unexpected top-level keys: {sorted(extra_top)}")
    if data.get("schema_version") != TRIAGE_SCHEMA_VERSION:
        errors.append(f"schema_version must be {TRIAGE_SCHEMA_VERSION!r}.")
    paper_id = data.get("paper_id")
    if not isinstance(paper_id, str) or not PAPER_ID_RE.fullmatch(paper_id):
        errors.append("paper_id is invalid.")
    if data.get("source_sha256") != extraction.get("source_sha256"):
        errors.append("source_sha256 does not match extraction.json.")
    if extraction.get("status") not in {"pass", "warning"}:
        errors.append(f"Extraction status {extraction.get('status')!r} blocks triage.")
    elif extraction.get("status") == "warning":
        warnings.append("Extraction has warnings; verify affected content against the PDF.")

    record = data.get("record")
    if not isinstance(record, dict):
        errors.append("record must be an object.")
        record = {}
    missing_fields = set(HEADERS) - set(record)
    extra_fields = set(record) - set(HEADERS)
    if missing_fields:
        errors.append(f"Missing record fields: {sorted(missing_fields)}")
    if extra_fields:
        errors.append(f"Unexpected record fields: {sorted(extra_fields)}")
    if record.get("Paper ID") != paper_id:
        errors.append("record['Paper ID'] must equal paper_id.")
    year = record.get("Year")
    if not isinstance(year, int) or not 1900 <= year <= 2100:
        errors.append("Year must be an integer from 1900 to 2100.")
    elif isinstance(paper_id, str) and paper_id[:4].isdigit() and int(paper_id[:4]) != year:
        warnings.append("Year differs from the Paper ID prefix; confirm version identity.")
    for field in HEADERS:
        if field == "Year":
            continue
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{field} must be a non-empty string.")
    for field, allowed in ENUMS.items():
        if record.get(field) not in allowed:
            errors.append(f"{field} must be one of {sorted(allowed)}.")
    if record.get("Read Status") == "Not Triaged":
        errors.append("A completed triage record cannot remain Not Triaged.")
    repo = record.get("Repo")
    if repo != "N/A" and (not isinstance(repo, str) or not is_http_url(repo)):
        errors.append("Repo must be N/A or an http(s) URL.")

    checks = data.get("checks")
    if not isinstance(checks, dict):
        errors.append("checks must be an object.")
        checks = {}
    if checks.get("paper_read") is not True:
        errors.append("checks.paper_read must be true.")
    checked_code = checks.get("official_code_source_checked")
    if not isinstance(checked_code, bool):
        errors.append("checks.official_code_source_checked must be boolean.")
    code = record.get("Code")
    if code == "Not Checked":
        if checked_code is True:
            warnings.append("Code is Not Checked but official_code_source_checked is true.")
        if repo != "N/A":
            errors.append("Repo must be N/A when Code is Not Checked.")
    elif code == "None":
        if checked_code is not True:
            errors.append("Code=None requires an official-source check.")
        if repo != "N/A":
            errors.append("Repo must be N/A when Code=None.")
    elif code in {"Official", "Unofficial"}:
        if not is_http_url(str(repo)):
            errors.append(f"Code={code} requires a repository URL.")
        if checked_code is not True:
            errors.append(f"Code={code} requires an official-source check.")

    evidence = data.get("evidence")
    if not isinstance(evidence, dict):
        errors.append("evidence must be an object.")
        evidence = {}
    uncertainties = data.get("uncertainties")
    if not isinstance(uncertainties, list):
        errors.append("uncertainties must be an array.")
        uncertainties = []
    uncertain_fields = set()
    for index, item in enumerate(uncertainties):
        if not isinstance(item, dict) or set(item) != {"field", "reason", "action"}:
            errors.append(f"uncertainties[{index}] must contain exactly field, reason, action.")
            continue
        if item["field"] not in HEADERS:
            errors.append(f"uncertainties[{index}].field is not a workbook field.")
        uncertain_fields.add(item["field"])
        if not all(isinstance(item[k], str) and item[k].strip() for k in ("reason", "action")):
            errors.append(f"uncertainties[{index}] has an empty reason/action.")

    for field, items in evidence.items():
        if field not in HEADERS:
            errors.append(f"Evidence key {field!r} is not a workbook field.")
            continue
        if not isinstance(items, list):
            errors.append(f"Evidence for {field} must be an array.")
            continue
        for index, item in enumerate(items):
            if not isinstance(item, dict) or set(item) != {"source", "locator", "quote"}:
                errors.append(f"evidence[{field}][{index}] has an invalid shape.")
                continue
            if item["source"] not in EVIDENCE_SOURCES:
                errors.append(f"evidence[{field}][{index}] has an invalid source.")
            if not isinstance(item["locator"], str) or not item["locator"].strip():
                errors.append(f"evidence[{field}][{index}] has no locator.")
            quote = item.get("quote")
            if not isinstance(quote, str) or not quote.strip() or len(quote) > 500:
                errors.append(f"evidence[{field}][{index}] quote must contain 1-500 characters.")

    for field in EVIDENCE_FIELDS:
        value = record.get(field)
        has_evidence = isinstance(evidence.get(field), list) and bool(evidence[field])
        if not has_evidence and not (value == "N/A" and field in uncertain_fields):
            errors.append(f"{field} requires evidence, or an uncertainty when its value is N/A.")

    for field in ("Priority Reason", "Core Idea"):
        value = str(record.get(field) or "")
        sentence_marks = len(re.findall(r"[.!?。！？]", value))
        if len(value) > 300:
            errors.append(f"{field} must be concise (<=300 characters).")
        if sentence_marks > 1:
            warnings.append(f"{field} appears to contain more than one sentence.")

    result = {"valid": not errors, "errors": errors, "warnings": warnings}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
