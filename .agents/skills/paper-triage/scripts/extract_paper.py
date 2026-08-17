#!/usr/bin/env python3
"""Compatibility wrapper for the shared paper-workspace extractor."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SHARED_DIR = Path(__file__).resolve().parents[3] / "tools" / "paper-workspace"
sys.path.insert(0, str(SHARED_DIR))
SPEC = importlib.util.spec_from_file_location("paper_workspace_extract_paper", SHARED_DIR / "extract_paper.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Cannot load shared extract_paper.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
for NAME in dir(MODULE):
    if not NAME.startswith("_"):
        globals()[NAME] = getattr(MODULE, NAME)

if __name__ == "__main__":
    try:
        raise SystemExit(MODULE.main())
    except MODULE.ContractError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
