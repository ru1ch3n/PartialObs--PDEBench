#!/usr/bin/env python3
"""Build a larger AI4PDE paper database (NDJSON).

This script is intended as a *one-shot builder* for the website database.
It merges:
  - The existing curated entries in scripts/research_db.ndjson
  - Additional paper lists (PINNpapers, PINN_Paper_List, etc.)

Notes
-----
* This project is a static site; we prefer being *honest* over hallucinating.
  Imported papers that are not manually curated will be marked as "index" and
  show placeholders on their paper pages.

* The optional HTML sources used for extraction in this environment are:
    /mnt/data/pinnpapers.html
    /mnt/data/pinn_paper_list.html
    /mnt/data/awesome_sciml.html
    /mnt/data/ai4phys_awesome.html
  If you don't have them, you can re-download the GitHub pages (HTML) and
  point this script at those paths.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from bs4 import BeautifulSoup


REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = REPO_ROOT / "scripts" / "research_db.ndjson"


def norm_title(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def slugify(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s[:60] if len(s) > 60 else s


def extract_year(text: str) -> Optional[int]:
    # Prefer 20xx/19xx in parentheses at the end, but fall back to last match.
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", text)
    if not years:
        return None
    return int(years[-1])


def guess_short_title(full_title: str) -> str:
    # Heuristic: if the title starts with an acronym-ish token before ':'
    if ":" in full_title:
        head = full_title.split(":", 1)[0].strip()
        if 2 <= len(head) <= 18 and re.fullmatch(r"[A-Za-z0-9\-]+", head):
            if sum(c.isupper() for c in head) >= max(2, len(head) // 2):
                return head
    # Otherwise, keep it readable but short.
    if len(full_title) <= 60:
        return full_title
    return full_title[:57].rstrip() + "…"


def guess_method_class(category: str, title: str) -> str:
    t = f"{category} {title}".lower()
    if "diffusion" in t or "score" in t:
        return "Diffusion"
    if "neural operator" in t or "operator" in t or "deeponet" in t or "fno" in t:
        return "Operator learning"
    if "graph" in t or "mesh" in t:
        return "Graph / mesh"
    if "transformer" in t or "attention" in t:
        return "Transformers"
    if "benchmark" in t or "dataset" in t or "arena" in t or "pdebench" in t:
        return "Benchmark"
    if "pinn" in t or "physics-informed" in t or "physics informed" in t or "deep ritz" in t or "dgm" in t:
        return "PINN / physics-constrained"
    return "SciML"


def extract_pde_tags(text: str) -> List[str]:
    # Keep this conservative: only tag when the string is explicitly mentioned.
    t = text.lower()
    tags: List[str] = []
    def add(name: str, *keys: str) -> None:
        if any(k in t for k in keys):
            tags.append(name)
    add("Navier–Stokes", "navier", "stokes")
    add("Burgers", "burgers")
    add("Darcy", "darcy")
    add("Poisson", "poisson")
    add("Helmholtz", "helmholtz")
    add("Wave", "wave equation", "wave")
    add("Heat", "heat equation", "heat")
    add("Advection(-diffusion)", "advection", "transport")
    add("Reaction–diffusion", "reaction-diffusion", "reaction diffusion")
    add("Allen–Cahn", "allen-cahn", "allen cahn")
    add("Shallow water", "shallow water")
    add("Maxwell", "maxwell")
    # Dedup while preserving order
    out=[]
    seen=set()
    for x in tags:
        if x not in seen:
            out.append(x); seen.add(x)
    return out


def load_curated_db() -> List[Dict[str, Any]]:
    papers: List[Dict[str, Any]] = []
    with DB_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            papers.append(json.loads(line))
    for p in papers:
        p.setdefault("status", "curated")
        p.setdefault("method_class", guess_method_class(p.get("category", ""), p.get("full_title", "")))
    return papers


def iter_headings_and_listitems(article: Any) -> Iterable[Tuple[str, Any]]:
    """Yield (current_heading, li_tag) in document order."""
    heading = "(unknown)"
    for el in article.find_all(["h1", "h2", "h3", "h4", "li"]):
        if el.name in ("h1", "h2", "h3", "h4"):
            txt = el.get_text(" ", strip=True)
            if txt:
                heading = txt
        elif el.name == "li":
            yield heading, el


def parse_generic_list(html_path: Path, source: str) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html_path.read_text("utf-8", errors="ignore"), "html.parser")
    article = soup.find("article") or soup
    out: List[Dict[str, Any]] = []
    for heading, li in iter_headings_and_listitems(article):
        txt = li.get_text(" ", strip=True)
        year = extract_year(txt)
        if year is None:
            continue

        # ignore nav-like items
        if len(txt) < 25:
            continue

        # title heuristics
        clean = re.sub(r"^\s*\d+\.?\s*", "", txt)
        title = None
        # quoted titles
        m = re.search(r"[\"\u201c\u201d](.+?)[\"\u201c\u201d]", clean)
        if m and len(m.group(1)) > 6:
            title = m.group(1).strip()
        if not title:
            # split on first comma
            parts = re.split(r",\s*", clean, maxsplit=1)
            title = parts[0].strip()
        if not title or len(title) < 6:
            continue

        # links
        paper_link = None
        code_link = None
        project_link = None
        for a in li.find_all("a", href=True):
            href = a["href"]
            ah = href.lower()
            if any(x in ah for x in ["arxiv.org", "doi.org", "sciencedirect", "springer", "openreview", "ieee", "acm.org", "proceedings.mlr.press", "nature.com", "iopscience"]):
                if paper_link is None:
                    paper_link = href
            if "github.com" in ah and code_link is None:
                code_link = href
            if any(x in ah for x in ["project", "pages", "website"]) and project_link is None:
                project_link = href

        method_class = guess_method_class(heading, title)
        pde_tags = extract_pde_tags(clean)

        out.append(
            {
                "source": source,
                "category": heading,
                "method_class": method_class,
                "full_title": title,
                "short_title": guess_short_title(title),
                "authors": "",
                "year": year,
                "month": None,
                "links": {k: v for k, v in {"paper": paper_link, "code": code_link, "project": project_link}.items() if v},
                "badges": [method_class, source] + pde_tags,
                "tagline": "",
                "quick_facts": [f"Source: {source}", "Summary: index-only (needs manual curation)", f"Class: {method_class}"],
                "tldr": f"Imported from {source} (section: {heading}). This page is currently an index entry; a full experiment + theory summary is not yet curated.",
                "contrib": [],
                "theory": [],
                "core_math": [],
                "pdes": pde_tags,
                "tasks": [],
                "baselines": [],
                "setting": [],
                "results_tables": [],
                "benchmark_note": "",
                "status": "index",
            }
        )
    return out


def parse_ai4phys_with_descriptions(html_path: Path) -> List[Dict[str, Any]]:
    """Parse AI4Phys/Awesome-AI-for-Physics entries that include date+description+domain."""
    soup = BeautifulSoup(html_path.read_text("utf-8", errors="ignore"), "html.parser")
    article = soup.find("article") or soup

    out: List[Dict[str, Any]] = []
    # In rendered GitHub markdown, we have a repeating pattern:
    #   <div><h6><a href=...>TITLE</a></h6> ...
    #   <ul><li><...>Date: ...</li><li>Description: ...</li><li>Domain: ...</li>
    for div in article.find_all("div"):
        h6 = div.find("h6")
        if not h6:
            continue
        a = h6.find("a", href=True)
        if not a:
            continue
        title = h6.get_text(" ", strip=True)
        href = a["href"]

        ul = None
        for sib in div.next_siblings:
            if getattr(sib, "name", None) == "ul":
                ul = sib
                break
            if getattr(sib, "name", None) in ("div", "h1", "h2", "h3", "h4", "h5", "h6"):
                break
        if ul is None:
            continue
        li_txt = "\n".join(li.get_text(" ", strip=True) for li in ul.find_all("li"))

        date_m = re.search(r"Date:\s*([0-9]{4})\.([0-9]{2})", li_txt)
        year = int(date_m.group(1)) if date_m else extract_year(li_txt) or extract_year(title) or None
        month = int(date_m.group(2)) if date_m else None
        if year is None:
            continue

        desc_m = re.search(r"Description:\s*(.+?)(?:\nDomain:|$)", li_txt, flags=re.S)
        desc = desc_m.group(1).strip() if desc_m else ""
        domain_m = re.search(r"Domain:\s*(.+)$", li_txt, flags=re.S)
        domain = domain_m.group(1).strip() if domain_m else ""

        method_class = guess_method_class(domain, title)
        pde_tags = extract_pde_tags(f"{title} {desc}")

        out.append(
            {
                "source": "AI4Phys/Awesome-AI-for-Physics",
                "category": "AI4Phys (annotated)",
                "method_class": method_class,
                "full_title": title,
                "short_title": guess_short_title(title),
                "authors": "",
                "year": year,
                "month": month,
                "links": {"paper": href, "paper_label": "link"},
                "badges": [method_class, "AI4Phys"] + pde_tags,
                "tagline": "",
                "quick_facts": ["Source: AI4Phys (annotated)", f"Class: {method_class}"],
                "tldr": desc or "Imported from AI4Phys list (description not parsed).",
                "contrib": [],
                "theory": [],
                "core_math": [],
                "pdes": pde_tags,
                "tasks": [],
                "baselines": [],
                "setting": [],
                "results_tables": [],
                "benchmark_note": "",
                "status": "index",
            }
        )
    return out


def merge(papers: List[Dict[str, Any]], limit: int = 300) -> List[Dict[str, Any]]:
    # Deduplicate by normalized title.
    by_key: Dict[str, Dict[str, Any]] = {}
    def score(p: Dict[str, Any]) -> int:
        s = 0
        if p.get("status") == "curated":
            s += 100
        if p.get("tldr"):
            s += 10
        if p.get("links", {}).get("paper"):
            s += 5
        if p.get("pdes"):
            s += 2
        return s

    for p in papers:
        key = norm_title(p.get("full_title", ""))
        if not key:
            continue
        if key not in by_key or score(p) > score(by_key[key]):
            by_key[key] = p

    merged = list(by_key.values())
    # assign slugs and ensure unique
    used_slugs = set()
    for p in merged:
        base = p.get("slug") or slugify(p.get("short_title") or p.get("full_title") or "paper")
        slug = base
        i = 2
        while slug in used_slugs or not slug:
            slug = f"{base}-{i}"
            i += 1
        p["slug"] = slug
        used_slugs.add(slug)

    # Sort: curated first, then by year desc, then title.
    merged.sort(key=lambda x: (0 if x.get("status") == "curated" else 1, -(x.get("year") or 0), (x.get("short_title") or "")))

    # Keep only recent-ish AI4PDE relevant papers
    keywords = [
        "pde", "partial differential", "navier", "stokes", "burgers", "darcy", "poisson", "helmholtz",
        "wave", "heat", "advection", "reaction", "diffusion", "operator", "deeponet", "neural operator",
        "pinn", "physics-informed", "physics informed", "fourcastnet", "weather", "turbulence",
        "shallow water", "maxwell",
    ]
    def relevant(p: Dict[str, Any]) -> bool:
        blob = f"{p.get('full_title','')} {p.get('category','')} {p.get('method_class','')} {p.get('tldr','')}".lower()
        return any(k in blob for k in keywords)

    filtered = [p for p in merged if relevant(p) and (p.get("year") or 0) >= 2015]
    if len(filtered) < limit:
        # Fall back: include everything if our heuristic is too strict.
        filtered = merged

    return filtered[:limit]


def write_ndjson(papers: List[Dict[str, Any]]) -> None:
    header = (
        "# Paper metadata for PartialObs–PDEBench website (NDJSON: one JSON object per line)\n"
        "# Fields: slug, short_title, full_title, authors, year, category, method_class, status, badges, tagline, quick_facts, tldr, core_math, theory, contrib, pdes, tasks, baselines, setting, results_tables, benchmark_note, links\n"
    )
    with DB_PATH.open("w", encoding="utf-8") as f:
        f.write(header)
        for p in papers:
            # remove None values for cleanliness
            clean = {k: v for k, v in p.items() if v not in (None, "")}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")


def main() -> None:
    curated = load_curated_db()

    extras: List[Dict[str, Any]] = []
    # Optional sources (paths can be changed)
    srcs = [
        (Path("/mnt/data/pinnpapers.html"), "idrl-lab/PINNpapers"),
        (Path("/mnt/data/pinn_paper_list.html"), "Event-AHU/PINN_Paper_List"),
        (Path("/mnt/data/awesome_sciml.html"), "awesome-scientific-machine-learning"),
    ]

    for path, name in srcs:
        if path.exists():
            extras.extend(parse_generic_list(path, name))

    ai4phys_path = Path("/mnt/data/ai4phys_awesome.html")
    if ai4phys_path.exists():
        extras.extend(parse_ai4phys_with_descriptions(ai4phys_path))

    merged = merge(curated + extras, limit=300)
    write_ndjson(merged)
    print(f"Wrote {len(merged)} papers to {DB_PATH}")


if __name__ == "__main__":
    main()
