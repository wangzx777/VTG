#!/usr/bin/env python3
"""Create or reuse a normalized paper Markdown cache using a mature PDF backend."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from paper_workspace_common import (
    ContractError,
    EXTRACTION_SCHEMA_VERSION,
    load_sync_identity,
    markdown_quality,
    pdf_page_count,
    sha256_file,
    validate_cached_extraction,
    write_json_atomic,
)

BACKEND_COMMANDS = {"mineru": "mineru", "marker": "marker_single", "docling": "docling"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--paper-id")
    parser.add_argument("--backend", choices=["auto", *BACKEND_COMMANDS], default="auto")
    parser.add_argument(
        "--mineru-engine",
        choices=["auto", "pipeline", "vlm-engine", "hybrid-engine"],
        default="auto",
        help="MinerU parsing engine. Auto uses the native hybrid-engine and never falls back to CPU automatically.",
    )
    parser.add_argument(
        "--mineru-device",
        choices=["auto", "cpu", "mps", "cuda", "npu"],
        default="auto",
        help="MinerU pipeline device override. Auto leaves MinerU device selection unchanged.",
    )
    parser.add_argument("--force", action="store_true", help="Refresh even when the cache is reusable.")
    parser.add_argument("--force-ocr", action="store_true")
    parser.add_argument("--check", action="store_true", help="Only report backend availability.")
    parser.add_argument("--timeout", type=int, default=3600)
    return parser.parse_args()


def prefer_workspace_venv(workspace: Path) -> None:
    """Make a project-local virtual environment discoverable without global installs."""
    venv_bin = workspace / ".venv" / "bin"
    if not venv_bin.is_dir():
        return
    current = os.environ.get("PATH", "")
    entries = current.split(os.pathsep) if current else []
    if str(venv_bin) not in entries:
        os.environ["PATH"] = str(venv_bin) + (os.pathsep + current if current else "")


def executable_versions() -> dict[str, dict[str, str | bool]]:
    result: dict[str, dict[str, str | bool]] = {}
    for backend, command in BACKEND_COMMANDS.items():
        path = shutil.which(command)
        version = "N/A"
        if path:
            for flag in ("--version", "-v"):
                try:
                    done = subprocess.run([path, flag], capture_output=True, text=True, timeout=20, check=False)
                except (OSError, subprocess.SubprocessError):
                    continue
                text = (done.stdout or done.stderr).strip().splitlines()
                if text:
                    version = text[0][:200]
                    break
        result[backend] = {"available": bool(path), "command": path or command, "version": version}
    return result


def choose_backend(requested: str, versions: dict[str, dict[str, str | bool]]) -> str:
    if requested != "auto":
        if not versions[requested]["available"]:
            raise ContractError(f"Requested backend {requested!r} is not installed ({BACKEND_COMMANDS[requested]} not found).")
        return requested
    for name in ("mineru", "marker", "docling"):
        if versions[name]["available"]:
            return name
    raise ContractError(
        "No supported PDF backend is installed. Install and test one of: MinerU (`mineru`), "
        "Marker (`marker_single`), or Docling (`docling`)."
    )


def resolve_mineru_runtime(
    requested_engine: str,
    requested_device: str,
) -> tuple[str, str]:
    """Prefer MinerU's native accelerated runtime unless CPU is explicitly requested."""
    engine = "hybrid-engine" if requested_engine == "auto" else requested_engine
    device = requested_device
    if requested_device == "cpu" and engine != "pipeline":
        raise ContractError(
            "Explicit CPU execution requires --mineru-engine pipeline --mineru-device cpu. "
            "MinerU's accelerated engines do not use this device override."
        )
    return engine, device


def backend_command(
    backend: str,
    source: Path,
    output: Path,
    force_ocr: bool,
    mineru_engine: str | None = None,
) -> list[str]:
    if backend == "mineru":
        method = "ocr" if force_ocr else "auto"
        command = ["mineru", "-p", str(source), "-o", str(output)]
        if mineru_engine:
            command.extend(["-b", mineru_engine])
        command.extend(["-m", method, "-f", "true", "-t", "true"])
        return command
    if backend == "marker":
        command = ["marker_single", str(source), "--output_dir", str(output), "--output_format", "markdown"]
        if force_ocr:
            command.append("--force_ocr")
        return command
    command = [
        "docling", "convert", str(source), "--from", "pdf", "--to", "md",
        "--image-export-mode", "referenced", "--output", str(output), "--ocr", "--enrich-formula",
    ]
    if force_ocr:
        command.extend(["--ocr-mode", "full_page"])
    return command


def backend_environment(
    backend: str,
    *,
    mineru_device: str = "auto",
) -> dict[str, str]:
    """Return a stable backend environment while preserving user overrides."""
    environment = dict(os.environ)
    if backend == "mineru":
        # Desktop apps do not necessarily inherit exports made in an already-open
        # interactive shell. Make Hugging Face's normal per-user default explicit
        # for the MinerU child process, while preserving any configured override.
        environment.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
        if mineru_device != "auto":
            environment["MINERU_DEVICE_MODE"] = mineru_device
    return environment


def huggingface_cache_location(environment: dict[str, str]) -> tuple[str, str]:
    """Report the inherited stable Hugging Face cache without changing HF_HOME."""
    configured = environment.get("HF_HOME")
    if configured:
        return str(Path(configured).expanduser()), "HF_HOME"
    return str(Path.home() / ".cache" / "huggingface"), "default"


def mineru_failure_hint(detail: str) -> str:
    """Explain the required escalation path for likely sandbox runtime failures."""
    lowered = detail.lower()
    sandbox_signals = (
        "metal",
        "mlx",
        "operation not permitted",
        "permissionerror",
        "permission denied",
        "sc_sem_nsems_max",
        "sock.bind",
    )
    if not any(signal in lowered for signal in sandbox_signals):
        return ""
    return (
        "\nThe native MinerU runtime appears to be blocked by the current sandbox. "
        "Rerun this extractor command in a normal local terminal. Keep "
        "--mineru-engine pipeline --mineru-device cpu for the CPU pipeline, or use the native "
        "accelerated engine outside the sandbox."
    )


def select_markdown(output: Path, source: Path) -> Path:
    candidates = [p for p in output.rglob("*.md") if p.is_file()]
    if not candidates:
        raise ContractError("Backend completed but produced no Markdown file.")
    stem = source.stem.lower()
    candidates.sort(key=lambda p: ((stem in p.stem.lower()), p.stat().st_size), reverse=True)
    return candidates[0]


def normalize_markdown(markdown_path: Path, cache_stage: Path) -> tuple[str, list[str]]:
    markdown = markdown_path.read_text(encoding="utf-8", errors="replace")
    assets_dir = cache_stage / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    source_root = markdown_path.parent
    warnings: list[str] = []

    for candidate in source_root.rglob("*"):
        if not candidate.is_file() or candidate == markdown_path or candidate.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative = candidate.relative_to(source_root)
        target = assets_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate, target)

    def rewrite_target(raw: str) -> str:
        stripped = raw.strip()
        if re.match(r"^(?:https?:|data:|#)", stripped):
            return raw
        target, *title = stripped.split(" ", 1)
        target = target.strip("<>")
        if os.path.isabs(target) or target.startswith("../"):
            warnings.append(f"Skipped unsafe/absolute Markdown asset reference: {target}")
            return raw
        normalized = target[2:] if target.startswith("./") else target
        if normalized.startswith("assets/"):
            return raw
        suffix = f" {title[0]}" if title else ""
        return f"assets/{normalized}{suffix}"

    markdown = re.sub(
        r"(!\[[^\]]*\]\()([^)]+)(\))",
        lambda m: m.group(1) + rewrite_target(m.group(2)) + m.group(3),
        markdown,
    )
    markdown = re.sub(
        r"(<img\b[^>]*?src=[\"'])([^\"']+)([\"'])",
        lambda m: m.group(1) + rewrite_target(m.group(2)) + m.group(3),
        markdown,
        flags=re.I,
    )
    return markdown, warnings


def replace_cache_atomically(staged: Path, destination: Path) -> None:
    backup = destination.with_name(f".{destination.name}.backup-{os.getpid()}")
    if backup.exists():
        shutil.rmtree(backup)
    if destination.exists():
        os.replace(destination, backup)
    try:
        os.replace(staged, destination)
    except Exception:
        if backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    prefer_workspace_venv(workspace)
    versions = executable_versions()
    if args.check:
        print(json.dumps({"backends": versions}, indent=2))
        return 0
    if not args.paper_id:
        raise ContractError("--paper-id is required unless --check is used")

    identity = load_sync_identity(workspace, args.paper_id)
    source = Path(identity["source_path_absolute"])
    source_hash = sha256_file(source)
    extracted_root = workspace / "literature" / "extracted"
    cache_dir = extracted_root / args.paper_id
    cache_existed = cache_dir.exists()
    reusable, reasons = validate_cached_extraction(cache_dir, source_hash)
    if reusable and not args.force:
        metadata = json.loads((cache_dir / "extraction.json").read_text(encoding="utf-8"))
        print(json.dumps({"action": "reused", "cache": str(cache_dir), "metadata": metadata}, indent=2))
        return 0

    backend = choose_backend(args.backend, versions)
    mineru_engine: str | None = None
    mineru_device = "auto"
    if backend == "mineru":
        mineru_engine, mineru_device = resolve_mineru_runtime(
            args.mineru_engine,
            args.mineru_device,
        )
    extracted_root.mkdir(parents=True, exist_ok=True)
    run_root = Path(tempfile.mkdtemp(prefix=f".{args.paper_id}.extract-", dir=extracted_root))
    backend_output = run_root / "backend-output"
    staged_cache = run_root / "cache"
    backend_output.mkdir()
    staged_cache.mkdir()
    command = backend_command(backend, source, backend_output, args.force_ocr, mineru_engine)
    child_environment = backend_environment(
        backend,
        mineru_device=mineru_device,
    )
    hf_cache_path, hf_cache_source = huggingface_cache_location(child_environment)
    started = dt.datetime.now(dt.timezone.utc)
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=args.timeout,
            check=False,
            env=child_environment,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout)[-4000:]
            hint = mineru_failure_hint(detail) if backend == "mineru" else ""
            raise ContractError(f"{backend} failed with exit {completed.returncode}:\n{detail}{hint}")
        markdown_path = select_markdown(backend_output, source)
        markdown, normalization_warnings = normalize_markdown(markdown_path, staged_cache)
        (staged_cache / "paper.md").write_text(markdown, encoding="utf-8")
        (staged_cache / "source.sha256").write_text(source_hash + "\n", encoding="utf-8")
        pages = pdf_page_count(source)
        quality = markdown_quality(markdown, pages, staged_cache / "assets")
        quality["warnings"].extend(normalization_warnings)
        if quality["warnings"] and quality["status"] == "pass":
            quality["status"] = "warning"
        finished = dt.datetime.now(dt.timezone.utc)
        metadata = {
            "schema_version": EXTRACTION_SCHEMA_VERSION,
            "paper_id": args.paper_id,
            "source_path": identity["source_path"],
            "source_sha256": source_hash,
            "paper_md_sha256": sha256_file(staged_cache / "paper.md"),
            "backend": backend,
            "backend_version": versions[backend]["version"],
            "backend_options": {
                "mineru_engine": mineru_engine or "N/A",
                "mineru_device": mineru_device if backend == "mineru" else "N/A",
                "huggingface_cache": hf_cache_path if backend == "mineru" else "N/A",
                "huggingface_cache_source": hf_cache_source if backend == "mineru" else "N/A",
            },
            "ocr_mode": "forced" if args.force_ocr else "auto",
            "command": command,
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "duration_seconds": round((finished - started).total_seconds(), 3),
            "page_count": pages,
            "status": quality["status"],
            "errors": quality["errors"],
            "warnings": quality["warnings"],
            "quality_signals": quality["signals"],
        }
        if cache_existed:
            metadata["previous_cache_rejection_reasons"] = reasons
        write_json_atomic(staged_cache / "extraction.json", metadata)
        if metadata["status"] == "failed":
            raise ContractError("Extraction failed the quality gate: " + "; ".join(metadata["errors"]))
        replace_cache_atomically(staged_cache, cache_dir)
        print(json.dumps({"action": "refreshed" if cache_existed else "created", "cache": str(cache_dir), "metadata": metadata}, indent=2))
        return 0
    finally:
        if run_root.exists():
            shutil.rmtree(run_root)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
