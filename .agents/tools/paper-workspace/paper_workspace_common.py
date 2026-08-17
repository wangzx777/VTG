#!/usr/bin/env python3
"""Shared deterministic contracts for papers, sources, and extraction caches."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

EXTRACTION_SCHEMA_VERSION = "1.0"
PAPER_ID_RE = re.compile(r"^[0-9]{4}-[a-z0-9]+(?:-[a-z0-9]+)*$")


class ContractError(RuntimeError):
    """Raised when workspace identity or a machine contract is invalid."""


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractError(f"Required file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ContractError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ContractError(f"Expected a JSON object in {path}")
    return data


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def ensure_within(path: Path, root: Path) -> Path:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ContractError(f"Path escapes workspace: {path}") from exc
    return resolved_path


def load_sync_identity(workspace: Path, paper_id: str) -> dict[str, Any]:
    if not PAPER_ID_RE.fullmatch(paper_id):
        raise ContractError(f"Invalid Paper ID: {paper_id}")
    state_path = workspace / "literature" / "zotero-sync.json"
    state = read_json(state_path)
    matches: list[dict[str, Any]] = []
    for zotero_key, raw in (state.get("items") or {}).items():
        if isinstance(raw, dict) and raw.get("paper_id") == paper_id:
            item = dict(raw)
            item["zotero_key"] = zotero_key
            matches.append(item)
    if len(matches) != 1:
        raise ContractError(
            f"Paper ID {paper_id!r} resolves to {len(matches)} sync-state items; expected exactly one."
        )
    identity = matches[0]
    source_raw = identity.get("source_path")
    if not isinstance(source_raw, str) or not source_raw.strip():
        raise ContractError(f"Sync-state item {paper_id} has no source_path")
    source_path = ensure_within(workspace / source_raw, workspace)
    if not source_path.is_file():
        raise ContractError(f"Synchronized source PDF not found: {source_path}")
    if source_path.suffix.lower() != ".pdf":
        raise ContractError(f"Synchronized source is not a PDF: {source_path}")
    identity["source_path_absolute"] = str(source_path)
    identity["state_path"] = str(state_path)
    return identity


def pdf_page_count(path: Path) -> int | None:
    try:
        completed = subprocess.run(
            ["pdfinfo", str(path)], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    match = re.search(r"^Pages:\s+(\d+)\s*$", completed.stdout, flags=re.MULTILINE)
    return int(match.group(1)) if match else None


def markdown_quality(markdown: str, page_count: int | None, assets_dir: Path) -> dict[str, Any]:
    nonspace = sum(not ch.isspace() for ch in markdown)
    replacement_count = markdown.count("\ufffd")
    controls = sum(ord(ch) < 32 and ch not in "\n\r\t" for ch in markdown)
    heading_count = len(re.findall(r"(?m)^#{1,6}\s+\S", markdown))
    formula_count = len(re.findall(r"\$\$[\s\S]*?\$\$|(?<!\\)\$[^\n$]+(?<!\\)\$|\\\[|\\begin\{(?:equation|align)", markdown))
    table_rows = len(re.findall(r"(?m)^\s*\|.*\|\s*$", markdown))
    html_tables = len(re.findall(r"<table\b", markdown, flags=re.I))
    image_refs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)|<img\b[^>]*?src=[\"']([^\"']+)", markdown, flags=re.I)
    refs = [next(value for value in pair if value) for pair in image_refs]
    missing_assets: list[str] = []
    for raw in refs:
        target = raw.strip().split(" ", 1)[0].strip("<>")
        if re.match(r"^(?:https?:|data:|#)", target):
            continue
        normalized = target[7:] if target.startswith("assets/") else target
        if not (assets_dir / normalized).is_file():
            missing_assets.append(target)

    warnings: list[str] = []
    errors: list[str] = []
    minimum_chars = 1000 if not page_count else max(1000, page_count * 120)
    if nonspace < minimum_chars:
        errors.append(f"Markdown is too short: {nonspace} non-space characters for {page_count or 'unknown'} pages.")
    if page_count and nonspace < page_count * 800:
        warnings.append(f"Low extracted text density: {nonspace / page_count:.0f} non-space characters/page.")
    replacement_ratio = replacement_count / max(len(markdown), 1)
    control_ratio = controls / max(len(markdown), 1)
    if replacement_ratio > 0.005 or control_ratio > 0.005:
        errors.append("Excessive replacement or control characters suggest corrupt extraction.")
    elif replacement_ratio > 0.001 or control_ratio > 0.001:
        warnings.append("Replacement/control character rate is suspicious; inspect affected pages.")
    if heading_count < 3:
        warnings.append("Fewer than three Markdown headings were detected.")
    if missing_assets:
        warnings.append(f"{len(missing_assets)} referenced asset(s) are missing.")
    asset_count = sum(1 for p in assets_dir.rglob("*") if p.is_file()) if assets_dir.exists() else 0
    status = "failed" if errors else ("warning" if warnings else "pass")
    return {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "signals": {
            "characters": len(markdown),
            "nonspace_characters": nonspace,
            "characters_per_page": round(nonspace / page_count, 2) if page_count else None,
            "headings": heading_count,
            "formula_markers": formula_count,
            "markdown_table_rows": table_rows,
            "html_tables": html_tables,
            "referenced_assets": len(refs),
            "asset_files": asset_count,
            "missing_assets": sorted(set(missing_assets)),
            "replacement_character_ratio": round(replacement_ratio, 6),
            "control_character_ratio": round(control_ratio, 6),
        },
    }


def validate_cached_extraction(cache_dir: Path, source_sha256: str | None = None) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    metadata_path = cache_dir / "extraction.json"
    paper_md = cache_dir / "paper.md"
    try:
        metadata = read_json(metadata_path)
    except ContractError as exc:
        return False, [str(exc)]
    if metadata.get("schema_version") != EXTRACTION_SCHEMA_VERSION:
        reasons.append("unsupported extraction schema version")
    if metadata.get("status") not in {"pass", "warning"}:
        reasons.append(f"cache status is {metadata.get('status')!r}")
    if not paper_md.is_file():
        reasons.append("paper.md is missing")
    else:
        actual_markdown_hash = sha256_file(paper_md)
        if actual_markdown_hash != metadata.get("paper_md_sha256"):
            reasons.append("paper.md hash does not match extraction.json")
    expected_source_hash = source_sha256 or metadata.get("source_sha256")
    if metadata.get("source_sha256") != expected_source_hash:
        reasons.append("source PDF hash changed")
    source_hash_file = cache_dir / "source.sha256"
    if not source_hash_file.is_file() or source_hash_file.read_text(encoding="utf-8").strip() != expected_source_hash:
        reasons.append("source.sha256 is missing or mismatched")
    return not reasons, reasons
