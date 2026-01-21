"""Import BibTeX into JSON Lines (NDJSON) index entries.

This utility is meant for *index* ingestion (metadata only). It does NOT create
deep curated pages automatically.

By default it prints NDJSON lines to stdout so you can append them to:
    scripts/research_db.ndjson

Usage:
    python scripts/import_bibtex_to_json.py path/to/papers.bib > new.ndjson
    cat new.ndjson >> scripts/research_db.ndjson

Optional:
    python scripts/import_bibtex_to_json.py path/to/papers.bib --append

Requirements:
    pip install bibtexparser
"""

from __future__ import annotations

import argparse
import re
import sys
import json
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "scripts" / "research_db.ndjson"


def slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9\s_-]+", "", s)
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "paper"


def first_author(authors: str) -> str:
    # BibTeX authors are usually "A and B and C"
    a = (authors or "").split(" and ")[0].strip()
    if "," in a:
        return a.split(",")[0].strip()
    return a.split(" ")[-1].strip() or "author"


def parse_bibtex(path: Path) -> List[Dict[str, Any]]:
    try:
        import bibtexparser  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependency bibtexparser. Install with: pip install bibtexparser") from e

    bib_db = bibtexparser.parse_file(str(path))
    entries = []
    for ent in getattr(bib_db, "entries", []):
        if not isinstance(ent, dict):
            continue
        entries.append(ent)
    return entries


def entry_to_index(ent: Dict[str, Any]) -> Dict[str, Any]:
    title = (ent.get("title") or "").strip().strip("{}")
    year = ent.get("year")
    try:
        year_int = int(str(year)) if year is not None else None
    except Exception:
        year_int = None

    authors = (ent.get("author") or "").replace("\n", " ").strip()
    venue = (ent.get("booktitle") or ent.get("journal") or "").strip()

    # Try to extract links
    doi = (ent.get("doi") or "").strip()
    url = (ent.get("url") or "").strip()

    # Stable-ish slug: <firstauthor>-<year>-<first-6-title-words>
    words = [w for w in re.split(r"\W+", title.lower()) if w]
    slug = slugify("-".join([first_author(authors), str(year_int or ""), "-".join(words[:6])]))

    links: Dict[str, str] = {}
    if url:
        links["paper"] = url
    if doi:
        links["doi"] = doi

    obj: Dict[str, Any] = {
        "slug": slug,
        "status": "index",
        "full_title": title,
        "authors": authors,
        "year": year_int,
        "venue": venue,
        "links": links,
    }
    # Keep bibtex if present (for BibTeX export)
    if ent.get("ID"):
        obj["bibkey"] = str(ent["ID"])
    return obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bibfile", type=str, help="Path to .bib file")
    ap.add_argument("--append", action="store_true", help=f"Append to {DEFAULT_OUT}")
    args = ap.parse_args()

    bib_path = Path(args.bibfile)
    if not bib_path.exists():
        print(f"BibTeX file not found: {bib_path}", file=sys.stderr)
        return 1

    entries = parse_bibtex(bib_path)
    if not entries:
        print("No entries found.", file=sys.stderr)
        return 1

    ndjson_lines = []
    for ent in entries:
        obj = entry_to_index(ent)
        ndjson_lines.append(json.dumps(obj, ensure_ascii=False))

    if args.append:
        DEFAULT_OUT.parent.mkdir(parents=True, exist_ok=True)
        with DEFAULT_OUT.open("a", encoding="utf-8") as f:
            for line in ndjson_lines:
                f.write(line + "\n")
        print(f"Appended {len(ndjson_lines)} entries to {DEFAULT_OUT}", file=sys.stderr)
    else:
        for line in ndjson_lines:
            print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
