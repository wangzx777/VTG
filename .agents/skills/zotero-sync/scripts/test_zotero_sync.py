#!/usr/bin/env python3
"""Offline tests for the local Sync/Venue zotero-sync workflow."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

SCRIPT = Path(__file__).with_name("zotero_sync.py")
spec = importlib.util.spec_from_file_location("zotero_sync", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class FakeData:
    def __init__(self):
        self.reset()

    def reset(self):
        self.collections = [
            self.collection("SYNC0001", "Sync", False),
            self.collection("CVPR2025", "CVPR 2025", "SYNC0001"),
            self.collection("ICCV2025", "ICCV 2025", "SYNC0001"),
            self.collection("OTHER001", "VTG", False),
        ]
        self.items = {
            "CVPR2025": [self.paper("PAPER001", "Time-R1", 1), self.attachment("ATT00001", "PAPER001", 1, "random.pdf")],
            "ICCV2025": [],
            "OTHER001": [self.paper("PAPER001", "Time-R1", 1)],
        }

    @staticmethod
    def collection(key, name, parent):
        return {"key": key, "version": 1, "data": {"key": key, "version": 1, "name": name, "parentCollection": parent}}

    @staticmethod
    def paper(key, title, version=1):
        return {"key": key, "version": version, "data": {"key": key, "version": version, "itemType": "conferencePaper", "title": title}}

    @staticmethod
    def attachment(key, parent, version=1, filename="paper.pdf"):
        return {
            "key": key,
            "version": version,
            "data": {
                "key": key,
                "version": version,
                "itemType": "attachment",
                "parentItem": parent,
                "linkMode": "imported_file",
                "contentType": "application/pdf",
                "filename": filename,
            },
        }


class Handler(BaseHTTPRequestHandler):
    data: FakeData = None

    def log_message(self, fmt, *args):
        pass

    def send_json(self, obj, total=None):
        raw = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        if total is not None:
            self.send_header("Total-Results", str(total))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        start = int(qs.get("start", [0])[0])
        limit = int(qs.get("limit", [100])[0])

        if path == "/users/0/collections":
            rows = self.data.collections
            return self.send_json(rows[start:start + limit], total=len(rows))

        if path.startswith("/users/0/collections/") and path.endswith("/items"):
            key = path.split("/")[-2]
            rows = self.data.items.get(key, [])
            return self.send_json(rows[start:start + limit], total=len(rows))

        if path.startswith("/users/0/items/") and path.endswith("/children"):
            parent = path.split("/")[-2]
            rows = []
            for values in self.data.items.values():
                rows.extend(x for x in values if x.get("data", {}).get("parentItem") == parent)
            # deduplicate attachment keys
            unique = {x["key"]: x for x in rows}
            rows = list(unique.values())
            return self.send_json(rows[start:start + limit], total=len(rows))

        self.send_error(404)


class ZoteroSyncTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fake = FakeData()
        Handler.data = cls.fake
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.api_base = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        self.fake.reset()
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp.name) / "VTG"
        self.data_dir = Path(self.tmp.name) / "Zotero"
        self.workspace.mkdir()
        self._write_pdf("ATT00001", "random.pdf", b"PDF-v1")

    def tearDown(self):
        self.tmp.cleanup()

    def _write_pdf(self, attachment_key, filename, content):
        p = self.data_dir / "storage" / attachment_key / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)

    def run_sync(self, *extra):
        argv = [
            "--workspace", str(self.workspace),
            "--sync-root", "Sync",
            "--api-base", self.api_base,
            "--zotero-data-dir", str(self.data_dir),
            *extra,
        ]
        rc = module.main(argv)
        self.assertEqual(rc, 0)

    def test_initial_sync_uses_venue_folder_and_zotero_key_id(self):
        self.run_sync()
        pdf = self.workspace / "literature" / "sources" / "CVPR 2025" / "zotero-paper001.pdf"
        self.assertTrue(pdf.exists())
        self.assertEqual(pdf.read_bytes(), b"PDF-v1")
        self.assertFalse((self.workspace / "literature" / "papers.xlsx").exists())
        state = json.loads((self.workspace / "literature" / "zotero-sync.json").read_text())
        self.assertEqual(state["items"]["PAPER001"]["paper_id"], "zotero-paper001")
        self.assertEqual(state["items"]["PAPER001"]["venue"], "CVPR 2025")

    def test_attachment_rename_keeps_same_paper_id_and_path(self):
        self.run_sync()
        old = self.workspace / "literature" / "sources" / "CVPR 2025" / "zotero-paper001.pdf"
        self.fake.items["CVPR2025"] = [
            self.fake.paper("PAPER001", "Time-R1", 2),
            self.fake.attachment("ATT00001", "PAPER001", 2, "renamed.pdf"),
        ]
        self._write_pdf("ATT00001", "renamed.pdf", b"PDF-v2")
        self.run_sync()
        self.assertTrue(old.exists())
        self.assertEqual(old.read_bytes(), b"PDF-v2")

    def test_same_item_outside_sync_does_not_duplicate(self):
        self.run_sync()
        matches = list((self.workspace / "literature" / "sources").rglob("zotero-paper001.pdf"))
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].parent.name, "CVPR 2025")

    def test_same_item_in_two_sync_venues_is_conflict_and_not_copied(self):
        self.fake.items["ICCV2025"] = [self.fake.paper("PAPER001", "Time-R1", 1)]
        self.run_sync()
        matches = list((self.workspace / "literature" / "sources").rglob("zotero-paper001.pdf"))
        self.assertEqual(matches, [])

    def test_new_paper_adds_only_new_source(self):
        self.run_sync()
        self.fake.items["ICCV2025"] = [
            self.fake.paper("PAPER002", "UniTime", 1),
            self.fake.attachment("ATT00002", "PAPER002", 1, "unitime.pdf"),
        ]
        self._write_pdf("ATT00002", "unitime.pdf", b"PDF-unitime")
        self.run_sync()
        p1 = self.workspace / "literature" / "sources" / "CVPR 2025" / "zotero-paper001.pdf"
        p2 = self.workspace / "literature" / "sources" / "ICCV 2025" / "zotero-paper002.pdf"
        self.assertEqual(p1.read_bytes(), b"PDF-v1")
        self.assertEqual(p2.read_bytes(), b"PDF-unitime")

    def test_venue_move_updates_canonical_path(self):
        self.run_sync()
        old = self.workspace / "literature" / "sources" / "CVPR 2025" / "zotero-paper001.pdf"
        self.fake.items["CVPR2025"] = []
        self.fake.items["ICCV2025"] = [
            self.fake.paper("PAPER001", "Time-R1", 2),
            self.fake.attachment("ATT00001", "PAPER001", 1, "random.pdf"),
        ]
        self.run_sync()
        new = self.workspace / "literature" / "sources" / "ICCV 2025" / "zotero-paper001.pdf"
        self.assertFalse(old.exists())
        self.assertTrue(new.exists())

    def test_dry_run_writes_nothing(self):
        self.run_sync("--dry-run")
        self.assertFalse((self.workspace / "literature").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
