#!/usr/bin/env python3
"""Build a bounded, line-addressable triage packet from cached paper Markdown."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Section:
    title: str
    start: int
    end: int


SECTION_RULES = [
    ("Abstract", r"\babstract\b"),
    ("Introduction", r"\b(?:introduction|background and motivation)\b"),
    ("Method", r"\b(?:method|approach|framework|architecture|algorithmic designs?|optimization|model)\b"),
    ("Experiments", r"\b(?:experiment|evaluation|result|benchmark)"),
    ("Conclusion", r"\b(?:conclusion|discussion|limitation)"),
]

KEYWORDS = re.compile(
    r"dataset|benchmark|metric|evaluation|backbone|base model|foundation model|"
    r"training|train(ed|ing)?|fine-tun|SFT|GRPO|reinforcement|code|github|repository|"
    r"R@|mIoU|IoU|recall|Figure\s*1|Fig\.\s*1|Table\s*1",
    flags=re.I,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-md", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-chars", type=int, default=30000)
    return parser.parse_args()


def sections(lines: list[str]) -> list[Section]:
    headings: list[tuple[int, str]] = []
    for index, line in enumerate(lines, start=1):
        match = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if match:
            headings.append((index, match.group(1).strip()))
    result: list[Section] = []
    for pos, (start, title) in enumerate(headings):
        end = headings[pos + 1][0] - 1 if pos + 1 < len(headings) else len(lines)
        result.append(Section(title, start, end))
    return result


def line_block(lines: list[str], start: int, end: int) -> str:
    return "\n".join(f"L{number}: {lines[number - 1]}" for number in range(start, end + 1))


def main() -> int:
    args = parse_args()
    content = args.paper_md.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()
    parsed = sections(lines)
    selected: list[tuple[str, int, int]] = []
    used: list[tuple[int, int]] = []

    def add(label: str, start: int, end: int) -> None:
        start = max(1, start)
        end = min(len(lines), end)
        if start > end or any(not (end < a or start > b) for a, b in used):
            return
        selected.append((label, start, end))
        used.append((start, end))

    for label, pattern in SECTION_RULES:
        matches = [section for section in parsed if re.search(pattern, section.title, flags=re.I)]
        if matches:
            section = matches[0]
            cap = 260 if label in {"Method", "Experiments"} else 180
            add(f"{label}: {section.title}", section.start, min(section.end, section.start + cap))

    for index, line in enumerate(lines, start=1):
        if KEYWORDS.search(line):
            add("Evidence snippet", index - 3, index + 5)
        if len(selected) >= 28:
            break

    if not selected:
        add("Document opening", 1, min(len(lines), 500))

    header = [
        "# Triage Reading Packet",
        "",
        f"Source: `{args.paper_md}`",
        "",
        "Line numbers refer to `paper.md`. This packet is a bounded reading aid; verify ambiguous equations, tables, figures, and claims against the PDF.",
        "",
    ]
    body: list[str] = []
    used_chars = len("\n".join(header))
    for label, start, end in selected:
        block = f"## {label} (L{start}-L{end})\n\n{line_block(lines, start, end)}\n"
        if used_chars + len(block) > args.max_chars:
            remaining = args.max_chars - used_chars
            if remaining > 1000:
                body.append(block[:remaining] + "\n\n[packet truncated by max-chars]\n")
            break
        body.append(block)
        used_chars += len(block)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(header + body), encoding="utf-8")
    print(f"Wrote {args.output} ({used_chars} characters, {len(body)} blocks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
