#!/usr/bin/env python3
"""Mirror Zotero Desktop Sync/Venue collections into literature/sources/.

Contract:
- Local Zotero Desktop API only; no API key.
- A configured root collection (default: Sync) is a sync boundary.
- Only its direct child collections are synchronized; each child is a Venue.
- Paper ID is human-readable and deterministic on first assignment: <year>-<normalized-title>.
- Once assigned, Paper ID is persisted in state and reused even if Zotero metadata changes.
- The script never creates or edits papers.xlsx.
- State is stored in literature/zotero-sync.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

API_VERSION = "3"
DEFAULT_API_BASE = "http://localhost:23119/api"
STATE_SCHEMA_VERSION = 3
NON_PAPER_TYPES = {"attachment", "note", "annotation"}


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def title_slug(title: str, max_length: int = 140) -> str:
    """Convert a paper title into a deterministic, filesystem-safe slug."""
    value = unicodedata.normalize("NFKD", (title or "").strip())
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    if not value:
        value = "untitled"
    if len(value) > max_length:
        value = value[:max_length].rstrip("-")
    return value


def paper_year(date_value: str, venue_name: str) -> str:
    """Prefer Zotero item year; fall back to a year embedded in the Venue collection name."""
    for source in (date_value or "", venue_name or ""):
        match = re.search(r"(?<!\d)(?:19|20)\d{2}(?!\d)", source)
        if match:
            return match.group(0)
    return "undated"


def allocate_paper_id(
    *,
    title: str,
    date_value: str,
    venue_name: str,
    zotero_key: str,
    used_ids: Dict[str, str],
) -> str:
    """Allocate a unique readable Paper ID using a hard title-based rule."""
    base = f"{paper_year(date_value, venue_name)}-{title_slug(title)}"
    candidate = base
    suffix = 2
    while candidate in used_ids and used_ids[candidate] != zotero_key:
        candidate = f"{base}-{suffix}"
        suffix += 1
    used_ids[candidate] = zotero_key
    return candidate


def safe_venue_dir(name: str) -> str:
    """Preserve human-readable Venue names while preventing path traversal."""
    value = (name or "").strip()
    if not value or value in {".", ".."}:
        raise ValueError(f"Invalid Venue collection name: {name!r}")
    value = value.replace("/", "_").replace("\\", "_")
    return value


class ZoteroError(RuntimeError):
    pass


@dataclass
class HttpResponse:
    body: bytes
    headers: dict
    status: int


class ZoteroClient:
    def __init__(self, api_base: str = DEFAULT_API_BASE):
        self.api_base = api_base.rstrip("/")

    def _headers(self) -> dict:
        return {
            "Zotero-API-Version": API_VERSION,
            "User-Agent": "vtg-zotero-sync-local/3.0",
        }

    def request(self, path: str, params: Optional[dict] = None) -> HttpResponse:
        url = self.api_base + path
        if params:
            url += "?" + urllib.parse.urlencode(params, doseq=True)
        req = urllib.request.Request(url, headers=self._headers(), method="GET")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return HttpResponse(resp.read(), dict(resp.headers.items()), resp.status)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise ZoteroError(f"Zotero API {exc.code} for {url}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ZoteroError(
                f"Cannot reach Zotero Desktop at {self.api_base}. Is Zotero running and local API enabled?"
            ) from exc

    def get_json(self, path: str, params: Optional[dict] = None):
        return json.loads(self.request(path, params=params).body.decode("utf-8"))

    def get_paginated(self, path: str, params: Optional[dict] = None) -> List[dict]:
        out: List[dict] = []
        start = 0
        limit = 100
        while True:
            p = dict(params or {})
            p.update({"start": start, "limit": limit})
            resp = self.request(path, params=p)
            batch = json.loads(resp.body.decode("utf-8"))
            if not isinstance(batch, list):
                raise ZoteroError(f"Expected list from {path}")
            out.extend(batch)
            total_raw = resp.headers.get("Total-Results") or resp.headers.get("total-results")
            if total_raw is not None:
                total = int(total_raw)
                if len(out) >= total:
                    break
            elif len(batch) < limit:
                break
            start += len(batch)
            if not batch:
                break
        return out


@dataclass
class PaperRecord:
    zotero_key: str
    title: str
    date_value: str
    item_version: int
    venues: Dict[str, str] = field(default_factory=dict)  # collection key -> name
    attachment: Optional[dict] = None


def load_state(path: Path) -> dict:
    if not path.exists():
        return {"schema_version": STATE_SCHEMA_VERSION, "items": {}}
    with path.open("r", encoding="utf-8") as f:
        state = json.load(f)
    version = int(state.get("schema_version") or 1)
    if version not in {1, 2, STATE_SCHEMA_VERSION}:
        raise RuntimeError(
            f"Unsupported zotero-sync state schema {version}; expected 1, 2, or {STATE_SCHEMA_VERSION}."
        )
    state.setdefault("items", {})
    if version != STATE_SCHEMA_VERSION:
        # Keep old paths/versions so a real sync can move files cleanly, but
        # regenerate every Paper ID using the new title-based naming rule.
        state["_reassign_paper_ids"] = True
        state["schema_version"] = STATE_SCHEMA_VERSION
    return state


def save_json_atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def collection_index(collections: List[dict]) -> Tuple[Dict[str, dict], Dict[str, List[str]]]:
    by_key = {c["key"]: c for c in collections}
    by_name: Dict[str, List[str]] = {}
    for key, c in by_key.items():
        name = str(c.get("data", {}).get("name") or "").strip()
        by_name.setdefault(name, []).append(key)
    return by_key, by_name


def resolve_collection(selector: str, by_key: Dict[str, dict], by_name: Dict[str, List[str]]) -> str:
    if selector in by_key:
        return selector
    matches = by_name.get(selector, [])
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f"Collection name '{selector}' is ambiguous; use its Zotero collection key.")
    raise RuntimeError(f"Collection not found: {selector}")


def direct_children(root_key: str, by_key: Dict[str, dict]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key, c in by_key.items():
        data = c.get("data", {})
        if data.get("parentCollection") == root_key:
            out[key] = str(data.get("name") or key).strip()
    return out


def pick_pdf_attachment(attachments: List[dict]) -> Optional[dict]:
    pdfs = []
    for att in attachments:
        data = att.get("data", {})
        content_type = str(data.get("contentType") or "").lower()
        filename = str(data.get("filename") or "").lower()
        if content_type == "application/pdf" or filename.endswith(".pdf"):
            pdfs.append(att)
    if not pdfs:
        return None

    def rank(att: dict):
        data = att.get("data", {})
        stored = 1 if data.get("linkMode") in {"imported_file", "imported_url"} else 0
        version = int(att.get("version") or data.get("version") or 0)
        return stored, version

    return sorted(pdfs, key=rank, reverse=True)[0]


def gather_sync_papers(
    client: ZoteroClient,
    prefix: str,
    venue_children: Dict[str, str],
) -> Dict[str, PaperRecord]:
    papers: Dict[str, PaperRecord] = {}
    attachment_pool: Dict[str, List[dict]] = {}

    for venue_key, venue_name in venue_children.items():
        items = client.get_paginated(
            f"{prefix}/collections/{venue_key}/items",
            params={"format": "json", "include": "data"},
        )
        for obj in items:
            data = obj.get("data", {})
            key = obj.get("key") or data.get("key")
            if not key:
                continue
            item_type = data.get("itemType")
            parent_key = data.get("parentItem")
            if item_type == "attachment" and parent_key:
                attachment_pool.setdefault(parent_key, []).append(obj)
                continue
            if item_type in NON_PAPER_TYPES or parent_key:
                continue
            title = str(data.get("title") or "").strip()
            if not title:
                continue
            date_value = str(data.get("date") or "").strip()
            version = int(obj.get("version") or data.get("version") or 0)
            paper = papers.get(key)
            if paper is None:
                paper = PaperRecord(key, title, date_value, version)
                papers[key] = paper
            elif version > paper.item_version:
                paper.title = title
                paper.date_value = date_value
                paper.item_version = version
            paper.venues[venue_key] = venue_name

    for key, paper in papers.items():
        candidates = attachment_pool.get(key, [])
        if not candidates:
            children = client.get_paginated(
                f"{prefix}/items/{key}/children",
                params={"format": "json", "include": "data"},
            )
            candidates = [x for x in children if x.get("data", {}).get("itemType") == "attachment"]
        paper.attachment = pick_pdf_attachment(candidates)

    return papers


def copy_local_attachment(attachment: dict, zotero_data_dir: Path, dest: Path) -> Path:
    data = attachment.get("data", {})
    key = attachment.get("key") or data.get("key")
    filename = data.get("filename")
    link_mode = data.get("linkMode") or ""
    raw_path = data.get("path")

    candidates: List[Path] = []
    if link_mode in {"imported_file", "imported_url"} and key and filename:
        candidates.append(zotero_data_dir / "storage" / str(key) / str(filename))
    if raw_path:
        raw_path = str(raw_path)
        if raw_path.startswith("storage:") and key:
            candidates.append(zotero_data_dir / "storage" / str(key) / raw_path.split(":", 1)[1])
        else:
            p = Path(raw_path).expanduser()
            if p.is_absolute():
                candidates.append(p)

    for src in candidates:
        if src.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(prefix=dest.name + ".", dir=str(dest.parent))
            os.close(fd)
            try:
                shutil.copy2(src, tmp_name)
                os.replace(tmp_name, dest)
            except Exception:
                try:
                    os.unlink(tmp_name)
                except FileNotFoundError:
                    pass
                raise
            return src

    shown = ", ".join(str(x) for x in candidates) if candidates else "no usable local path"
    raise ZoteroError(f"Local PDF not found for attachment {key}: {shown}")


def list_collections(collections: List[dict]) -> None:
    by_key = {c["key"]: c for c in collections}

    def path_for(key: str) -> str:
        parts = []
        seen = set()
        cur = key
        while cur in by_key and cur not in seen:
            seen.add(cur)
            data = by_key[cur].get("data", {})
            parts.append(str(data.get("name") or cur))
            cur = data.get("parentCollection")
            if not cur:
                break
        return "/".join(reversed(parts))

    for key in sorted(by_key, key=lambda k: path_for(k).casefold()):
        print(f"{key}\t{path_for(key)}")


def sync(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    api_base = args.api_base or DEFAULT_API_BASE
    client = ZoteroClient(api_base)
    prefix = "/users/0"
    data_dir = Path(args.zotero_data_dir or os.getenv("ZOTERO_DATA_DIR") or "~/Zotero").expanduser().resolve()

    collections = client.get_paginated(f"{prefix}/collections", params={"format": "json", "include": "data"})
    if args.list_collections:
        list_collections(collections)
        return 0

    by_key, by_name = collection_index(collections)
    root_key = resolve_collection(args.sync_root, by_key, by_name)
    venues = direct_children(root_key, by_key)
    if not venues:
        raise RuntimeError(f"Sync root '{args.sync_root}' has no direct child collections.")

    print(f"Sync root: {by_key[root_key]['data'].get('name')} ({root_key})")
    for key, name in sorted(venues.items(), key=lambda kv: kv[1].casefold()):
        print(f"  Venue: {name} ({key})")

    papers = gather_sync_papers(client, prefix, venues)
    literature_dir = workspace / "literature"
    state_path = literature_dir / "zotero-sync.json"
    state = load_state(state_path)
    old_items = state.setdefault("items", {})

    stats = {
        "new": 0,
        "updated": 0,
        "unchanged": 0,
        "conflict": 0,
        "no_pdf": 0,
        "out_of_scope": 0,
    }
    current_valid_keys = set()
    reassign_ids = bool(state.get("_reassign_paper_ids"))
    used_ids: Dict[str, str] = {}
    if not reassign_ids:
        for existing_zkey, entry in old_items.items():
            existing_id = str(entry.get("paper_id") or "").strip()
            if existing_id:
                used_ids[existing_id] = existing_zkey

    for zkey in sorted(papers, key=lambda k: (papers[k].title.casefold(), k)):
        paper = papers[zkey]

        if len(paper.venues) != 1:
            stats["conflict"] += 1
            venue_names = sorted(paper.venues.values())
            print(f"[CONFLICT] {paper.title}: appears in multiple Sync Venue collections: {', '.join(venue_names)}")
            continue

        current_valid_keys.add(zkey)
        venue_key, venue_name = next(iter(paper.venues.items()))
        previous = old_items.get(zkey, {})
        previous_id = str(previous.get("paper_id") or "").strip()
        if previous_id and not reassign_ids:
            paper_id = previous_id
            used_ids[paper_id] = zkey
        else:
            paper_id = allocate_paper_id(
                title=paper.title,
                date_value=paper.date_value,
                venue_name=venue_name,
                zotero_key=zkey,
                used_ids=used_ids,
            )

        venue_dir = safe_venue_dir(venue_name)
        source_rel = f"literature/sources/{venue_dir}/{paper_id}.pdf"
        source_path = workspace / source_rel

        attachment = paper.attachment
        attachment_key = None
        attachment_version = None
        if attachment:
            ad = attachment.get("data", {})
            attachment_key = attachment.get("key") or ad.get("key")
            attachment_version = int(attachment.get("version") or ad.get("version") or 0)

        changed = (
            not previous
            or previous.get("item_version") != paper.item_version
            or previous.get("attachment_key") != attachment_key
            or previous.get("attachment_version") != attachment_version
            or previous.get("venue_collection_key") != venue_key
            or previous.get("source_path") != source_rel
            or not source_path.exists()
        )

        if not previous:
            action = "NEW"
            stats["new"] += 1
        elif changed:
            action = "UPDATE"
            stats["updated"] += 1
        else:
            action = "OK"
            stats["unchanged"] += 1

        if attachment is None:
            stats["no_pdf"] += 1
            print(f"[{action}] {paper_id}: no PDF attachment — {paper.title}")
        elif changed:
            print(f"[{action}] {paper_id}: {venue_name} -> {source_rel}")
            if not args.dry_run:
                src = copy_local_attachment(attachment, data_dir, source_path)
                print(f"         copied from {src}")
                old_rel = previous.get("source_path")
                if old_rel and old_rel != source_rel:
                    old_path = workspace / old_rel
                    if old_path.is_file():
                        old_path.unlink()
        else:
            print(f"[{action}] {paper_id}: unchanged")

        if not args.dry_run:
            old_items[zkey] = {
                "paper_id": paper_id,
                "title": paper.title,
                "date": paper.date_value,
                "item_version": paper.item_version,
                "venue": venue_name,
                "venue_collection_key": venue_key,
                "attachment_key": attachment_key,
                "attachment_version": attachment_version,
                "source_path": source_rel,
                "in_scope": True,
            }

    # Never delete files merely because a paper leaves Sync.
    for zkey, entry in old_items.items():
        if zkey not in current_valid_keys and entry.get("in_scope", True):
            stats["out_of_scope"] += 1
            if not args.dry_run:
                entry["in_scope"] = False

    if not args.dry_run:
        state.pop("_reassign_paper_ids", None)
        state.update(
            {
                "schema_version": STATE_SCHEMA_VERSION,
                "sync_root_key": root_key,
                "sync_root_name": by_key[root_key]["data"].get("name"),
                "venue_collections": {k: v for k, v in sorted(venues.items())},
                "last_sync_unix": int(time.time()),
            }
        )
        save_json_atomic(state_path, state)

    print(
        "Summary: "
        + ", ".join(f"{k}={v}" for k, v in stats.items())
        + (" [DRY RUN]" if args.dry_run else "")
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sync local Zotero Desktop Sync/Venue collections into literature/sources/.")
    p.add_argument("--workspace", default=".", help="VTG workspace root (default: current directory)")
    p.add_argument("--sync-root", default="Sync", help="Root collection name or key (default: Sync)")
    p.add_argument("--list-collections", action="store_true", help="List Zotero collection keys/paths and exit")
    p.add_argument("--dry-run", action="store_true", help="Show planned changes without writing PDFs or state")
    p.add_argument("--api-base", default=DEFAULT_API_BASE, help="Zotero Desktop local API base")
    p.add_argument("--zotero-data-dir", help="Zotero Data Directory (default: $ZOTERO_DATA_DIR or ~/Zotero)")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return sync(args)
    except (RuntimeError, ZoteroError, ValueError, json.JSONDecodeError) as exc:
        eprint(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
