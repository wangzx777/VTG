#!/usr/bin/env python3
"""Resolve one source PDF and the two paper-read output paths."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from paper_read_common import ContractError, paper_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--paper", required=True, help="Exact PDF stem/paper ID or PDF path")
    parser.add_argument("--create-dirs", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = paper_paths(args.workspace, args.paper, create_dirs=args.create_dirs)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"CONTRACT ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
