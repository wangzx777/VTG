#!/usr/bin/env python3
"""Shared path resolution and lightweight output validation for paper-read."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any


class ContractError(RuntimeError):
    """Raised when a paper or generated artifact violates the skill contract."""


REQUIRED_NOTE_HEADINGS = (
    (1, "Paper"),
    (2, "Problem"),
    (2, "Method"),
    (2, "Training"),
    (2, "Experiment"),
    (2, "Tags"),
    (2, "Key Figure / Table"),
    (2, "Code"),
    (2, "我还没懂"),
)

REQUIRED_FIELDS = (
    "Training paradigm",
    "Loss / Reward",
    "特殊训练策略",
    "Dataset",
    "Metric",
    "主要结果 / 结论",
    "Repo",
    "关键文件",
)

TAG_GROUPS = {
    "task": {"task"},
    "role": {"role"},
    "training": {"train", "opt", "data"},
    "time": {"time", "reason", "out", "long"},
    "visual": {"visual", "arch", "setting", "modal"},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def resolve_pdf(workspace: Path, paper: str) -> Path:
    workspace = workspace.resolve()
    sources = (workspace / "literature" / "sources").resolve()
    if not sources.is_dir():
        raise ContractError(f"Source directory not found: {sources}")

    supplied = Path(paper).expanduser()
    direct_candidates = [supplied]
    if not supplied.is_absolute():
        direct_candidates.append(workspace / supplied)

    for candidate in direct_candidates:
        if candidate.is_file():
            resolved = candidate.resolve()
            if resolved.suffix.lower() != ".pdf":
                raise ContractError(f"Paper path is not a PDF: {resolved}")
            if not _inside(resolved, sources):
                raise ContractError(f"Paper must be under {sources}: {resolved}")
            return resolved

    requested_name = supplied.name
    requested_stem = requested_name[:-4] if requested_name.lower().endswith(".pdf") else requested_name
    matches = [path.resolve() for path in sources.rglob("*.pdf") if path.stem == requested_stem]
    if not matches:
        raise ContractError(f"No PDF with exact stem {requested_stem!r} under {sources}")
    if len(matches) > 1:
        rendered = "; ".join(str(path) for path in sorted(matches))
        raise ContractError(f"Multiple PDFs share stem {requested_stem!r}: {rendered}")
    return matches[0]


def paper_paths(workspace: Path, paper: str, create_dirs: bool = False) -> dict[str, Any]:
    workspace = workspace.resolve()
    source = resolve_pdf(workspace, paper)
    stem = source.stem
    venue_dir = source.parent
    translation_dir = venue_dir / "translations"
    note_dir = venue_dir / "notes"
    if create_dirs:
        translation_dir.mkdir(parents=True, exist_ok=True)
        note_dir.mkdir(parents=True, exist_ok=True)

    translation = translation_dir / f"{stem}.md"
    note = note_dir / f"{stem}.md"
    cache_dir = workspace / "literature" / "extracted" / stem
    paper_md = cache_dir / "paper.md"
    source_hash_file = cache_dir / "source.sha256"
    source_hash = sha256_file(source)
    recorded_hash = ""
    if source_hash_file.is_file():
        recorded_hash = source_hash_file.read_text(encoding="utf-8", errors="replace").strip()
    cache_reusable = (
        paper_md.is_file()
        and paper_md.stat().st_size > 0
        and recorded_hash == source_hash
    )
    return {
        "paper_id": stem,
        "source_pdf": str(source),
        "venue_dir": str(venue_dir),
        "translation_dir": str(translation_dir),
        "note_dir": str(note_dir),
        "translation": str(translation),
        "note": str(note),
        "translation_exists": translation.is_file(),
        "note_exists": note.is_file(),
        "source_sha256": source_hash,
        "cache_dir": str(cache_dir),
        "paper_md": str(paper_md),
        "cache_reusable": cache_reusable,
    }


def _read_utf8(path: Path, label: str) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ContractError(f"{label} not found: {path}") from exc
    except UnicodeDecodeError as exc:
        raise ContractError(f"{label} is not UTF-8: {path}") from exc
    if len(text.strip()) < 100:
        raise ContractError(f"{label} is unexpectedly short: {path}")
    return text


def validate_translation(path: Path) -> dict[str, Any]:
    text = _read_utf8(path, "Translation")
    errors: list[str] = []
    headings = re.findall(r"(?m)^(#{1,6})\s+(.+?)\s*$", text)
    if not headings or len(headings[0][0]) != 1:
        errors.append("Translation must begin its heading structure with one level-1 paper title.")

    forbidden = (
        (r"!\[[^\]]*\]\(", "Markdown image"),
        (r"<img\b", "HTML image"),
        (r"\*\*", "bold formatting"),
        (r"(?m)^\s*>\s+", "blockquote"),
        (r"```", "fenced code"),
        (r"(?mi)^#{1,6}\s+(references|bibliography|参考文献|文献目录)\s*$", "bibliography heading"),
        (r"(?m)^\s*\|.*\|.*\|\s*$", "Markdown table"),
        (r"(?m)^\s*[-+*]\s+\S", "body bullet list"),
    )
    for pattern, label in forbidden:
        if re.search(pattern, text):
            errors.append(f"Translation contains forbidden {label}.")
    if errors:
        raise ContractError(" ".join(errors))
    return {
        "valid": True,
        "path": str(path),
        "characters": len(text),
        "headings": [heading for _, heading in headings],
    }


def _section(text: str, heading: str) -> str:
    pattern = re.compile(
        rf"(?ms)^## {re.escape(heading)}\s*$\n(.*?)(?=^##\s+|\Z)"
    )
    match = pattern.search(text)
    if not match:
        raise ContractError(f"Missing note section: {heading}")
    return match.group(1)


def _field_value(text: str, field: str) -> str:
    match = re.search(rf"(?m)^-?\s*{re.escape(field)}:\s*(.*?)\s*$", text)
    if not match or not match.group(1).strip():
        raise ContractError(f"Missing or empty note field: {field}")
    return match.group(1).strip()


def _allowed_tags(skill_dir: Path) -> set[str]:
    pool = (skill_dir / "references" / "tag-pool.md").read_text(encoding="utf-8")
    return set(re.findall(r"`([a-z]+:[a-z0-9-]+)`", pool))


def validate_note(path: Path, skill_dir: Path) -> dict[str, Any]:
    text = _read_utf8(path, "Note")
    positions: list[int] = []
    for level, heading in REQUIRED_NOTE_HEADINGS:
        pattern = rf"(?m)^{'#' * level} {re.escape(heading)}\s*$"
        matches = list(re.finditer(pattern, text))
        if len(matches) != 1:
            raise ContractError(
                f"Note must contain exactly one {'#' * level} {heading}; found {len(matches)}."
            )
        positions.append(matches[0].start())
    if positions != sorted(positions):
        raise ContractError("Note headings are out of template order.")

    sentence = re.search(r"(?ms)^一句话：\s*\n(.*?)(?=^## Problem\s*$)", text)
    sentence_text = sentence.group(1).strip() if sentence else ""
    if not all(word in sentence_text for word in ("解决", "通过", "实现")):
        raise ContractError("一句话 must use the 解决…通过…实现… pattern.")

    problem = _section(text, "Problem")
    if len(re.findall(r"(?m)^-\s+\S", problem)) < 2:
        raise ContractError("Problem must contain at least two non-empty bullets.")

    method = _section(text, "Method")
    if method.count("↓") < 2 or "Temporal Boundary" not in method:
        raise ContractError("Method must contain a downward flow ending in Temporal Boundary.")
    if re.search(r"(?m)^\.\.\.\s*$", method):
        raise ContractError("Method flow still contains template placeholders.")
    for index in (1, 2, 3):
        if not re.search(rf"(?m)^{index}\.\s+\S", method):
            raise ContractError(f"Core innovation {index} is missing or empty.")

    for field in REQUIRED_FIELDS:
        _field_value(text, field)

    tag_section = _section(text, "Tags")
    allowed = _allowed_tags(skill_dir)
    selected: list[str] = []
    for group, prefixes in TAG_GROUPS.items():
        match = re.search(rf"(?m)^{group}:\s*(.*?)\s*$", tag_section)
        if not match or not match.group(1).strip():
            raise ContractError(f"Missing or empty tag group: {group}")
        value = match.group(1).strip()
        if value == "N/A":
            continue
        for tag in [item.strip() for item in value.split(";")]:
            if tag not in allowed:
                raise ContractError(f"Tag is not in the allowed pool: {tag}")
            prefix = tag.split(":", 1)[0]
            if prefix not in prefixes:
                raise ContractError(f"Tag {tag} is in the wrong group {group}.")
            selected.append(tag)

    unresolved = _section(text, "我还没懂")
    for index in (1, 2, 3):
        if not re.search(rf"(?m)^{index}\.\s+\S", unresolved):
            raise ContractError(f"我还没懂 item {index} is missing or empty.")

    placeholders = ("___", "Fig.__", "Table.__", "<fill>")
    remaining = [placeholder for placeholder in placeholders if placeholder in text]
    if remaining:
        raise ContractError("Note still contains template placeholders: " + ", ".join(remaining))

    return {
        "valid": True,
        "path": str(path),
        "characters": len(text),
        "tags": selected,
    }


def validate_outputs(workspace: Path, paper: str, skill_dir: Path) -> dict[str, Any]:
    paths = paper_paths(workspace, paper)
    translation = Path(paths["translation"])
    note = Path(paths["note"])
    return {
        "valid": True,
        "paper_id": paths["paper_id"],
        "source_pdf": paths["source_pdf"],
        "cache_reusable": paths["cache_reusable"],
        "translation": validate_translation(translation),
        "note": validate_note(note, skill_dir),
    }
