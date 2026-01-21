/* Generic paper page (index placeholders) */

function qs(sel) { return document.querySelector(sel); }

function sanitizeBibKey(s) {
  return String(s || "paper")
    .trim()
    .replace(/[^A-Za-z0-9_:-]+/g, "");
}

function pickUrl(links) {
  if (!links) return "";
  return links.paper || links.arxiv || links.openreview || links.doi || links.code || "";
}

function makeMinimalBibtex(p) {
  const key = sanitizeBibKey(p.bibkey || p.slug || "paper");
  const title = (p.full_title || p.short_title || "").trim();
  const authors = (p.authors || "").trim();
  const year = (p.year || "").toString().trim();
  const venue = (p.venue || "").trim();
  const url = pickUrl(p.links);

  const entryType = (venue && venue.toLowerCase() !== "arxiv") ? "inproceedings" : "article";
  const fields = [];
  if (title) fields.push(`  title={${title}}`);
  if (authors) fields.push(`  author={${authors}}`);
  if (year) fields.push(`  year={${year}}`);
  if (venue) fields.push(entryType === "inproceedings" ? `  booktitle={${venue}}` : `  journal={${venue}}`);
  if (url) fields.push(`  url={${url}}`);

  return `@${entryType}{${key},\n${fields.join(",\n")}\n}`;
}

async function loadDB() {
  const url = window.PAPERS_DB_URL || "../../assets/papers_db.json";
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to load papers DB: ${res.status}`);
  return await res.json();
}

function getSlugFromQuery() {
  const u = new URL(window.location.href);
  return u.searchParams.get("slug") || "";
}

function linkLi(label, href) {
  const safe = String(href || "").trim();
  if (!safe) return "";
  return `<li><b>${label}:</b> <a href="${safe}" target="_blank" rel="noopener noreferrer">${safe}</a></li>`;
}

function escapeHtml(s) {
  return String(s || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;");
}

function renderPaper(p) {
  const title = p.full_title || p.short_title || p.slug;
  const authors = p.authors || "";
  const year = p.year || "";
  const venue = p.venue || "";
  const method = p.method_class || "";
  const status = p.status || "index";

  const links = p.links || {};
  const bib = (p.bibtex || "").trim() || makeMinimalBibtex(p);

  const pdes = (p.pdes || []).join(", ");
  const tasks = (p.tasks || []).join(", ");

  const tldr = (p.tldr || "").trim();
  const problem = (p.problem || "").trim();

  const contributeHint = (status === "curated")
    ? `<div class="note">This paper has a curated page. Redirecting…</div>`
    : `<div class="note">
         This is an <b>index placeholder</b>. Want a full summary here?
         Edit <code>data/papers/*.yaml</code> and set <code>status: curated</code>.
         See the <a href="../../contribute/">Contribute</a> tab.
       </div>`;

  return `
    <section class="section">
      <h2>${escapeHtml(title)}</h2>
      <div class="muted">${escapeHtml(authors)}</div>
      <div class="meta" style="margin-top:10px;">
        <div><b>Venue:</b> ${escapeHtml(venue)} ${escapeHtml(year)}</div>
        <div><b>Method:</b> ${escapeHtml(method)}</div>
        <div><b>Status:</b> <span class="badge ${status === "curated" ? "badge-curated" : "badge-index"}">${escapeHtml(status)}</span></div>
      </div>

      ${contributeHint}

      <h3>Links</h3>
      <ul>
        ${linkLi("paper", links.paper)}
        ${linkLi("arxiv", links.arxiv)}
        ${linkLi("openreview", links.openreview)}
        ${linkLi("code", links.code)}
      </ul>

      <h3>Tags</h3>
      <ul>
        <li><b>PDEs:</b> ${escapeHtml(pdes || "(none)")}</li>
        <li><b>Tasks:</b> ${escapeHtml(tasks || "(none)")}</li>
      </ul>

      <h3>TL;DR</h3>
      <p>${escapeHtml(tldr || "No TL;DR yet (add tldr: to the YAML).")}</p>

      <h3>Problem</h3>
      <p>${escapeHtml(problem || "No problem statement yet (add problem: to the YAML).")}</p>

      <h3>Citation (BibTeX)</h3>
      <pre class="code"><code>${escapeHtml(bib)}</code></pre>
      <button id="copyBib" class="btn">Copy BibTeX</button>
    </section>
  `;
}

async function main() {
  const slug = getSlugFromQuery();
  const mount = qs("#paperMount");
  const subtitle = qs("#paperSubtitle");

  if (!slug) {
    if (mount) mount.innerHTML = `<div class="note">Missing <code>?slug=...</code> in URL.</div>`;
    if (subtitle) subtitle.textContent = "Missing slug";
    return;
  }

  const papers = await loadDB();
  const p = papers.find(x => x.slug === slug);
  if (!p) {
    if (mount) mount.innerHTML = `<div class="note">Paper not found: <code>${escapeHtml(slug)}</code></div>`;
    if (subtitle) subtitle.textContent = "Not found";
    return;
  }

  // If curated, redirect to the real page
  if ((p.status || "index") === "curated") {
    window.location.replace(`../${encodeURIComponent(p.slug)}/`);
    return;
  }

  document.title = `${p.full_title || p.short_title || p.slug} — PartialObs–PDEBench`;
  if (subtitle) subtitle.textContent = `${p.venue || ""} ${p.year || ""}`.trim();

  if (mount) mount.innerHTML = renderPaper(p);

  const btn = qs("#copyBib");
  btn?.addEventListener("click", async () => {
    const bib = (p.bibtex || "").trim() || makeMinimalBibtex(p);
    try {
      await navigator.clipboard.writeText(bib);
      btn.textContent = "Copied!";
      setTimeout(() => { btn.textContent = "Copy BibTeX"; }, 1200);
    } catch (e) {
      alert("Clipboard copy failed.");
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  main().catch(err => {
    console.error(err);
    const mount = qs("#paperMount");
    if (mount) mount.innerHTML = `<div class="note">Failed to load paper DB. Check console.</div>`;
  });
});
