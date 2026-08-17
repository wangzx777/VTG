#!/usr/bin/env python3
"""Offline tests for paper-triage identity, cache, quality, and record validation."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from paper_triage_common import (  # noqa: E402
    EXTRACTION_SCHEMA_VERSION,
    ContractError,
    load_sync_identity,
    markdown_quality,
    sha256_file,
    validate_cached_extraction,
)
from extract_paper import (  # noqa: E402
    backend_environment,
    huggingface_cache_location,
    mineru_failure_hint,
    prefer_workspace_venv,
    resolve_mineru_runtime,
)
from validate_extraction import merge_recorded_warnings  # noqa: E402


class PaperTriageTests(unittest.TestCase):
    def make_workspace(self, root: Path, paper_id: str = "2026-test-paper") -> tuple[Path, Path]:
        source = root / "literature" / "sources" / "CVPR 2026" / f"{paper_id}.pdf"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"%PDF-1.4\nfixture\n%%EOF\n")
        state = {
            "schema_version": 3,
            "items": {
                "ABCDEFGH": {
                    "paper_id": paper_id,
                    "source_path": str(source.relative_to(root)),
                    "title": "Test Paper",
                    "date": "2026",
                    "venue": "CVPR 2026",
                    "in_scope": True,
                }
            },
        }
        state_path = root / "literature" / "zotero-sync.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state), encoding="utf-8")
        return root, source

    def test_sync_identity_uses_state(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root, source = self.make_workspace(Path(raw))
            identity = load_sync_identity(root, "2026-test-paper")
            self.assertEqual(identity["zotero_key"], "ABCDEFGH")
            self.assertEqual(Path(identity["source_path_absolute"]), source.resolve())

    def test_sync_identity_rejects_unknown_id(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root, _ = self.make_workspace(Path(raw))
            with self.assertRaises(ContractError):
                load_sync_identity(root, "2026-other-paper")

    def test_cache_integrity_and_source_invalidation(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root, source = self.make_workspace(Path(raw))
            cache = root / "literature" / "extracted" / "2026-test-paper"
            cache.mkdir(parents=True)
            markdown = "# Test\n\n" + ("A paper sentence with enough content.\n" * 100)
            paper_md = cache / "paper.md"
            paper_md.write_text(markdown, encoding="utf-8")
            source_hash = sha256_file(source)
            (cache / "source.sha256").write_text(source_hash + "\n", encoding="utf-8")
            metadata = {
                "schema_version": EXTRACTION_SCHEMA_VERSION,
                "status": "pass",
                "source_sha256": source_hash,
                "paper_md_sha256": sha256_file(paper_md),
            }
            (cache / "extraction.json").write_text(json.dumps(metadata), encoding="utf-8")
            self.assertTrue(validate_cached_extraction(cache, source_hash)[0])
            source.write_bytes(source.read_bytes() + b"changed")
            self.assertFalse(validate_cached_extraction(cache, sha256_file(source))[0])

    def test_quality_gate_rejects_short_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            result = markdown_quality("# Tiny\n", 10, Path(raw))
            self.assertEqual(result["status"], "failed")

    def test_quality_gate_counts_html_tables(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            markdown = "# Test\n\n" + ("Useful paper content. " * 100) + "\n<table><tr><td>42</td></tr></table>\n"
            result = markdown_quality(markdown, 1, Path(raw))
            self.assertEqual(result["signals"]["html_tables"], 1)

    def test_mineru_environment_preserves_hf_home_and_pythonpath(self) -> None:
        original = {"HF_HOME": "/stable/huggingface", "PYTHONPATH": "/user/modules"}
        with mock.patch.dict(os.environ, original, clear=True):
            environment = backend_environment("mineru")
        self.assertEqual(environment["HF_HOME"], original["HF_HOME"])
        self.assertEqual(environment["PYTHONPATH"], original["PYTHONPATH"])
        self.assertNotIn("MINERU_PDF_RENDER_THREADS", environment)
        self.assertNotIn("MINERU_TOOLS_CONFIG_JSON", environment)

    def test_mineru_auto_uses_native_hybrid_runtime(self) -> None:
        engine, device = resolve_mineru_runtime("auto", "auto")
        self.assertEqual(engine, "hybrid-engine")
        self.assertEqual(device, "auto")

    def test_explicit_mineru_pipeline_cpu_is_sufficient_acknowledgement(self) -> None:
        self.assertEqual(resolve_mineru_runtime("pipeline", "cpu"), ("pipeline", "cpu"))

    def test_mineru_rejects_cpu_with_accelerated_engine(self) -> None:
        with self.assertRaises(ContractError):
            resolve_mineru_runtime("hybrid-engine", "cpu")

    def test_independent_validation_keeps_recorded_review_warnings(self) -> None:
        quality = {"status": "pass", "errors": [], "warnings": [], "signals": {}}
        metadata = {"warnings": ["Equation 3 needs targeted PDF comparison."]}
        merged = merge_recorded_warnings(quality, metadata)
        self.assertEqual(merged["status"], "warning")
        self.assertEqual(merged["warnings"], metadata["warnings"])

    def test_huggingface_cache_location_reports_without_mutation(self) -> None:
        configured = {"HF_HOME": "/stable/huggingface"}
        self.assertEqual(huggingface_cache_location(configured), ("/stable/huggingface", "HF_HOME"))
        self.assertNotIn("HF_HOME", {})
        with mock.patch("extract_paper.Path.home", return_value=Path("/Users/researcher")):
            self.assertEqual(
                huggingface_cache_location({}),
                ("/Users/researcher/.cache/huggingface", "default"),
            )

    def test_mineru_environment_sets_default_hf_home_when_not_inherited(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch(
            "extract_paper.Path.home", return_value=Path("/Users/researcher")
        ):
            environment = backend_environment("mineru", mineru_device="cpu")
        self.assertEqual(environment["HF_HOME"], "/Users/researcher/.cache/huggingface")
        self.assertEqual(environment["MINERU_DEVICE_MODE"], "cpu")

    def test_mineru_sandbox_failure_requests_unsandboxed_retry(self) -> None:
        hint = mineru_failure_hint("PermissionError: Metal initialization operation not permitted")
        self.assertIn("normal local terminal", hint)
        self.assertIn("--mineru-engine pipeline --mineru-device cpu", hint)

    def test_workspace_venv_is_preferred(self) -> None:
        with tempfile.TemporaryDirectory() as raw, mock.patch.dict(os.environ, {"PATH": "/usr/bin"}):
            root = Path(raw)
            (root / ".venv" / "bin").mkdir(parents=True)
            prefer_workspace_venv(root)
            self.assertTrue(os.environ["PATH"].startswith(str(root / ".venv" / "bin")))

    def test_extract_cli_creates_then_reuses_normalized_cache(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root, _ = self.make_workspace(Path(raw))
            bin_dir = root / "fake-bin"
            bin_dir.mkdir()
            fake_mineru = bin_dir / "mineru"
            fake_mineru.write_text(
                "#!/usr/bin/env python3\n"
                "import pathlib, sys\n"
                "if '--version' in sys.argv or '-v' in sys.argv:\n"
                "    print('mineru, version test')\n"
                "    raise SystemExit(0)\n"
                "output = pathlib.Path(sys.argv[sys.argv.index('-o') + 1]) / '2026-test-paper'\n"
                "(output / 'images').mkdir(parents=True)\n"
                "(output / 'images' / 'figure-1.png').write_bytes(b'PNG')\n"
                "body = '# Test Paper\\n\\n## Abstract\\n\\n' + ('Video temporal grounding content. ' * 120) + "
                "'\\n\\n## Method\\n\\n$$x = y + z$$\\n\\n![Figure](images/figure-1.png)\\n\\n## Experiments\\n\\n| Metric | Value |\\n|---|---:|\\n| mIoU | 42 |\\n'\n"
                "(output / '2026-test-paper.md').write_text(body, encoding='utf-8')\n",
                encoding="utf-8",
            )
            fake_mineru.chmod(0o755)
            environment = dict(os.environ)
            environment["PATH"] = str(bin_dir) + os.pathsep + environment.get("PATH", "")
            command = [
                sys.executable, str(SCRIPT_DIR / "extract_paper.py"),
                "--workspace", str(root), "--paper-id", "2026-test-paper", "--backend", "mineru",
            ]
            first = subprocess.run(command, capture_output=True, text=True, env=environment, check=False)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            self.assertEqual(json.loads(first.stdout)["action"], "created")
            cache = root / "literature" / "extracted" / "2026-test-paper"
            self.assertTrue((cache / "paper.md").is_file())
            self.assertTrue((cache / "assets" / "images" / "figure-1.png").is_file())
            self.assertIn("assets/images/figure-1.png", (cache / "paper.md").read_text(encoding="utf-8"))
            self.assertIn(
                json.loads((cache / "extraction.json").read_text(encoding="utf-8"))["status"],
                {"pass", "warning"},
            )
            metadata = json.loads((cache / "extraction.json").read_text(encoding="utf-8"))
            self.assertNotIn("runtime_compatibility", metadata)
            self.assertNotIn("previous_cache_rejection_reasons", metadata)
            second = subprocess.run(command, capture_output=True, text=True, env=environment, check=False)
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            self.assertEqual(json.loads(second.stdout)["action"], "reused")

    def test_validate_triage_accepts_contract_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            extraction = root / "extraction.json"
            triage = root / "triage.json"
            source_hash = hashlib.sha256(b"source").hexdigest()
            extraction.write_text(json.dumps({"source_sha256": source_hash, "status": "pass"}), encoding="utf-8")
            record = {
                "Paper ID": "2026-test-paper", "Year": 2026, "Venue": "CVPR", "Title": "Test Paper",
                "Task": "Video Temporal Grounding", "Method Family": "Video-LLM", "Focus": "Boundary Modeling",
                "Priority": "Important", "Priority Reason": "Provides a relevant boundary-modeling baseline for VTG.",
                "Read Status": "Triaged", "Base Model": "N/A", "Core Idea": "Predicts intervals with a boundary-aware decoder.",
                "Training": "End-to-End", "Datasets": "Charades-STA", "Metrics": "mIoU", "Code": "Not Checked",
                "Repo": "N/A", "Reproduce Status": "N/A", "Zotero Key": "ABCDEFGH", "Remarks": "N/A",
            }
            evidence_fields = {
                "Title", "Year", "Venue", "Task", "Method Family", "Focus", "Priority", "Priority Reason",
                "Base Model", "Core Idea", "Training", "Datasets", "Metrics", "Code", "Repo",
            }
            evidence = {field: [{"source": "paper" if field not in {"Priority", "Priority Reason", "Method Family", "Focus", "Code", "Repo"} else "inference", "locator": "fixture", "quote": "fixture evidence"}] for field in evidence_fields}
            triage.write_text(json.dumps({
                "schema_version": "1.0", "paper_id": "2026-test-paper", "source_sha256": source_hash,
                "record": record, "evidence": evidence,
                "checks": {"paper_read": True, "official_code_source_checked": False}, "uncertainties": [],
            }), encoding="utf-8")
            completed = subprocess.run([
                sys.executable, str(SCRIPT_DIR / "validate_triage.py"),
                "--triage", str(triage), "--schema", str(SKILL_DIR / "assets" / "triage.schema.json"),
                "--extraction", str(extraction),
            ], capture_output=True, text=True, check=False)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()
