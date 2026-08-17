#!/usr/bin/env python3
"""Compatibility imports for legacy paper-triage scripts."""

from __future__ import annotations

import sys
from pathlib import Path

SHARED_DIR = Path(__file__).resolve().parents[3] / "tools" / "paper-workspace"
sys.path.insert(0, str(SHARED_DIR))

from paper_workspace_common import *  # noqa: F401,F403,E402

TRIAGE_SCHEMA_VERSION = "1.0"
HEADERS = [
    "Paper ID", "Year", "Venue", "Title", "Task", "Method Family", "Focus",
    "Priority", "Priority Reason", "Read Status", "Base Model", "Core Idea",
    "Training", "Datasets", "Metrics", "Code", "Repo", "Reproduce Status",
    "Zotero Key", "Remarks",
]
