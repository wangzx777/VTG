#!/usr/bin/env python3
"""Validate paper-read translation and AI-note outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from paper_read_common import ContractError, validate_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--paper", required=True, help="Exact PDF stem/paper ID or PDF path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    skill_dir = Path(__file__).resolve().parent.parent
    result = validate_outputs(args.workspace, args.paper, skill_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"CONTRACT ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
