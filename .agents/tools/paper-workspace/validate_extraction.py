#!/usr/bin/env python3
"""Validate a paper cache against the synchronized source PDF and report quality signals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from paper_workspace_common import (
    ContractError,
    load_sync_identity,
    markdown_quality,
    pdf_page_count,
    sha256_file,
    validate_cached_extraction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--paper-id", required=True)
    return parser.parse_args()


def merge_recorded_warnings(quality: dict[str, object], metadata: dict[str, object]) -> dict[str, object]:
    """Keep targeted manual-review warnings visible in independent validation."""
    computed = quality.get("warnings", [])
    recorded = metadata.get("warnings", [])
    combined = list(dict.fromkeys(
        warning
        for warnings in (computed, recorded)
        if isinstance(warnings, list)
        for warning in warnings
        if isinstance(warning, str) and warning.strip()
    ))
    quality["warnings"] = combined
    if combined and quality.get("status") == "pass":
        quality["status"] = "warning"
    return quality


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    identity = load_sync_identity(workspace, args.paper_id)
    source = Path(identity["source_path_absolute"])
    source_hash = sha256_file(source)
    cache_dir = workspace / "literature" / "extracted" / args.paper_id
    reusable, integrity_reasons = validate_cached_extraction(cache_dir, source_hash)
    result: dict[str, object] = {
        "paper_id": args.paper_id,
        "cache": str(cache_dir),
        "source_path": identity["source_path"],
        "source_sha256": source_hash,
        "integrity_valid": reusable,
        "integrity_reasons": integrity_reasons,
    }
    paper_md = cache_dir / "paper.md"
    if paper_md.is_file():
        markdown = paper_md.read_text(encoding="utf-8", errors="replace")
        quality = markdown_quality(markdown, pdf_page_count(source), cache_dir / "assets")
        try:
            metadata = json.loads((cache_dir / "extraction.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}
        result["quality"] = merge_recorded_warnings(quality, metadata)
    else:
        result["quality"] = {"status": "failed", "errors": ["paper.md is missing"], "warnings": [], "signals": {}}
    result["valid"] = reusable and result["quality"]["status"] in {"pass", "warning"}  # type: ignore[index]
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
