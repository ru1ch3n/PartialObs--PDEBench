#!/usr/bin/env python3
"""
Import a BibTeX file (.bib) into YAML stubs under data/papers/.

This is the most reliable "bulk ingest" workflow for conference proceedings:
- Download a BibTeX export for a venue/year (ICLR/ICML/NeurIPS).
- Convert BibTeX -> many YAML files (status=index).
- Curate a subset over time by editing YAML and setting status=curated.

Examples
--------
# Import everything
python scripts/import_bibtex_to_yaml.py --bib path/to/ICLR_2025.bib --venue ICLR --out data/papers/import/iclr2025

# Import only AI4PDE-ish titles
python scripts/import_bibtex_to_yaml.py \
  --bib path/to/ICLR_2025.bib --venue ICLR --out data/papers/import/iclr2025 \
  --keywords "pde,operator,physics-informed,navier,burgers,sde,diffusion"

Notes
-----
- This script stores the raw BibTeX entry under `bibtex:` in YAML, when possible.
- If you prefer to avoid gigantic YAML, you can remove `bibtex:` for index placeholders;
  the website can still generate a minimal BibTeX from metadata.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import yaml

try:
    import bibtexparser  # type: ignore
    from bibtexparser.bparser import BibTexParser  # type: ignore
    from bibtexparser.bwriter import BibTexWriter  # type: ignore
    from bibtexparser.bibdatabase import BibDatabase  # type: ignore
except Exception as e:  # pragma: no cover
    bibtexparser = None  # type: ignore


def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[’']", "", s)
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "paper"


def norm_authors(author_field: str) -> str:
    # BibTeX uses "and" between authors
    return " ; ".join([a.strip() for a in author_field.split(" and ") if a.strip()])


def keep_if_keywords(title: str, keywords: Optional[List[str]]) -> bool:
    if not keywords:
        return True
    t = title.lower()
    return any(k.lower() in t for k in keywords)


def to_bibtex(entry: Dict[str, str]) -> str:
    """Serialize a single bib entry back to text (best-effort)."""
    if bibtexparser is None:
        return ""
    db = BibDatabase()
    db.entries = [entry]
    writer = BibTexWriter()
    writer.indent = "  "
    writer.order_entries_by = None
    return writer.write(db).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bib", required=True, help="Path to .bib file")
    ap.add_argument("--venue", required=True, help="Venue name (e.g., ICLR / ICML / NeurIPS)")
    ap.add_argument("--out", required=True, help="Output directory, e.g., data/papers/import/iclr2025")
    ap.add_argument("--status", default="index", choices=["index", "curated"], help="Default status")
    ap.add_argument("--keywords", default="", help="Comma-separated keywords to filter by title (optional)")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of entries (0 = no limit)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing YAML files")
    args = ap.parse_args()

    bib_path = Path(args.bib)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()] or None

    if bibtexparser is None:
        raise SystemExit(
            "Missing dependency: bibtexparser\n"
            "Install it with: pip install bibtexparser\n"
            "Then re-run this script."
        )

    parser = BibTexParser(common_strings=True)
    with bib_path.open("r", encoding="utf-8") as f:
        bib_db = bibtexparser.load(f, parser=parser)

    n_written = 0
    n_skipped = 0

    for entry in bib_db.entries:
        title = (entry.get("title") or "").strip().strip("{}")
        if not title:
            n_skipped += 1
            continue

        if not keep_if_keywords(title, keywords):
            n_skipped += 1
            continue

        year = (entry.get("year") or "").strip()
        key = (entry.get("ID") or "").strip()

        slug_base = slugify(title)
        prefix = f"{args.venue.lower()}"
        if year:
            prefix += f"{year}"
        slug = f"{prefix}-{slug_base}"

        authors = ""
        if "author" in entry:
            authors = norm_authors(entry["author"])

        links: Dict[str, str] = {}
        if "url" in entry:
            links["paper"] = entry["url"].strip()
        elif "doi" in entry:
            links["paper"] = f"https://doi.org/{entry['doi'].strip()}"

        y: Dict[str, object] = {
            "slug": slug,
            "status": args.status,
            "category": "AI4PDE",
            "method_class": "SciML",
            "full_title": title,
            "short_title": key or title[:80],
            "authors": authors or "",
            "year": int(year) if year.isdigit() else (year or ""),
            "venue": args.venue,
            "links": links,
        }

        bibtex_txt = to_bibtex(entry)
        if bibtex_txt:
            y["bibkey"] = key or slug
            y["bibtex"] = bibtex_txt + "\n"

        out_path = out_dir / f"{slug}.yaml"
        if out_path.exists() and not args.overwrite:
            n_skipped += 1
            continue

        with out_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(y, f, sort_keys=False, allow_unicode=True)

        n_written += 1
        if args.limit and n_written >= args.limit:
            break

    print(f"Wrote {n_written} YAML files to {out_dir} (skipped {n_skipped}).")


if __name__ == "__main__":
    main()
