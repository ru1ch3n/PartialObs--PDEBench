/* PartialObs–PDEBench: Research index client-side UI
 *
 * - Loads window.PAPERS_DB_URL (docs/assets/papers_db.json)
 * - Renders a filterable table (scales to thousands of papers)
 * - Allows batch BibTeX export (Download .bib / Copy)
 */

function qs(sel) { return document.querySelector(sel); }
function qsa(sel) { return Array.from(document.querySelectorAll(sel)); }

function uniq(arr) {
  return Array.from(new Set(arr.filter(x => x !== null && x !== undefined && String(x).trim() !== "")));
}

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

function paperInternalLink(p) {
  // research/index.html -> curated pages at ./<slug>/ ; placeholders at ./paper/?slug=...
  if ((p.status || "index") === "curated") return `./${encodeURIComponent(p.slug)}/`;
  return `./paper/?slug=${encodeURIComponent(p.slug)}`;
}

function fmtList(xs) {
  if (!xs || xs.length === 0) return "";
  return xs.join(", ");
}

async function loadDB() {
  const url = window.PAPERS_DB_URL || "../assets/papers_db.json";
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to load papers DB: ${res.status}`);
  return await res.json();
}

function buildOptions(selectEl, values, placeholder) {
  const cur = selectEl.value;
  selectEl.innerHTML = "";
  const opt0 = document.createElement("option");
  opt0.value = "";
  opt0.textContent = placeholder;
  selectEl.appendChild(opt0);

  for (const v of values) {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = v;
    selectEl.appendChild(opt);
  }

  // preserve selection if still present
  if (cur && values.includes(cur)) selectEl.value = cur;
}

function matchesQuery(p, q) {
  if (!q) return true;
  const hay = [
    p.full_title, p.short_title, p.authors, p.venue,
    (p.method_class || ""), (p.category || "")
  ].join(" ").toLowerCase();
  return hay.includes(q.toLowerCase());
}

function matchesFilter(p, method, venue, pde, status) {
  if (method && (p.method_class || "") !== method) return false;
  if (venue && (p.venue || "") !== venue) return false;
  if (status && (p.status || "index") !== status) return false;
  if (pde) {
    const xs = p.pdes || [];
    if (!xs.includes(pde)) return false;
  }
  return true;
}

function sortPapers(a, b) {
  // year desc then title asc
  const ya = a.year || 0;
  const yb = b.year || 0;
  if (yb !== ya) return yb - ya;
  const ta = (a.full_title || a.short_title || "").toLowerCase();
  const tb = (b.full_title || b.short_title || "").toLowerCase();
  if (ta < tb) return -1;
  if (ta > tb) return 1;
  return 0;
}

function createRow(p, selected) {
  const tr = document.createElement("tr");

  // pick
  const tdPick = document.createElement("td");
  tdPick.className = "nowrap";
  const cb = document.createElement("input");
  cb.type = "checkbox";
  cb.className = "pick";
  cb.dataset.slug = p.slug;
  cb.checked = selected.has(p.slug);
  tdPick.appendChild(cb);
  tr.appendChild(tdPick);

  // year
  const tdYear = document.createElement("td");
  tdYear.className = "nowrap";
  tdYear.textContent = (p.year || "").toString();
  tr.appendChild(tdYear);

  // paper
  const tdPaper = document.createElement("td");
  tdPaper.className = "paper-col";
  const a = document.createElement("a");
  a.href = paperInternalLink(p);
  a.textContent = (p.full_title || p.short_title || p.slug);
  tdPaper.appendChild(a);

  const small = document.createElement("div");
  small.className = "muted small";
  const subtitle = (p.tldr || "").trim() || (p.authors || "").trim();
  small.textContent = subtitle;
  tdPaper.appendChild(small);

  tr.appendChild(tdPaper);

  // venue
  const tdVenue = document.createElement("td");
  tdVenue.className = "nowrap";
  tdVenue.textContent = p.venue || "";
  tr.appendChild(tdVenue);

  // method
  const tdMethod = document.createElement("td");
  tdMethod.className = "nowrap";
  tdMethod.textContent = p.method_class || "";
  tr.appendChild(tdMethod);

  // PDEs
  const tdPde = document.createElement("td");
  tdPde.textContent = fmtList(p.pdes || []);
  if (p.pdes_auto) tdPde.title = "Auto-tagged (needs verification)";
  tr.appendChild(tdPde);

  // tasks
  const tdTasks = document.createElement("td");
  tdTasks.textContent = fmtList(p.tasks || []);
  if (p.tasks_auto) tdTasks.title = "Auto-tagged (needs verification)";
  tr.appendChild(tdTasks);

  // status
  const tdStatus = document.createElement("td");
  tdStatus.className = "nowrap";
  const s = document.createElement("span");
  s.className = `badge ${((p.status || "index") === "curated") ? "badge-curated" : "badge-index"}`;
  s.textContent = p.status || "index";
  tdStatus.appendChild(s);
  tr.appendChild(tdStatus);

  return tr;
}

function createLoadMoreRow(remaining, onClick) {
  const tr = document.createElement("tr");
  const td = document.createElement("td");
  td.colSpan = 8;
  td.className = "center muted";
  const btn = document.createElement("button");
  btn.className = "btn";
  btn.textContent = `Load more (+${remaining})`;
  btn.addEventListener("click", onClick);
  td.appendChild(btn);
  tr.appendChild(td);
  return tr;
}

function getSelectedBibtex(papersBySlug, selected) {
  const entries = [];
  for (const slug of selected) {
    const p = papersBySlug.get(slug);
    if (!p) continue;
    const bib = (p.bibtex || "").trim();
    entries.push(bib ? bib : makeMinimalBibtex(p));
  }
  return entries.join("\n\n") + (entries.length ? "\n" : "");
}

function downloadText(filename, text) {
  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function setBtnEnabled(id, enabled) {
  const el = qs(id);
  if (!el) return;
  el.disabled = !enabled;
}

async function main() {
  const tbody = qs("#paperRows");
  const qInput = qs("#q");
  const fMethod = qs("#f_method");
  const fVenue = qs("#f_venue");
  const fPde = qs("#f_pde");
  const fStatus = qs("#f_status");

  const selCountEl = qs("#selCount");
  const shownCountEl = qs("#shownCount");

  const btnCopy = qs("#btnCopyBib");
  const btnDownload = qs("#btnDownloadBib");
  const btnClear = qs("#btnClearSel");

  let papers = await loadDB();
  papers.sort(sortPapers);

  const papersBySlug = new Map(papers.map(p => [p.slug, p]));
  const selected = new Set();

  // Build filter option lists
  buildOptions(fMethod, uniq(papers.map(p => p.method_class)).sort(), "All methods");
  buildOptions(fVenue, uniq(papers.map(p => p.venue)).sort(), "All venues");

  const pdeVals = uniq(papers.flatMap(p => (p.pdes || []))).sort();
  buildOptions(fPde, pdeVals, "All PDEs");

  // Deep-link support: pre-fill search/filters from URL query parameters.
  // Example: /research/?method=Operator%20learning
  //          /research/?q=diffusion&pde=Navier%E2%80%93Stokes
  try {
    const params = new URLSearchParams(window.location.search || "");
    const q0 = params.get("q");
    const method0 = params.get("method");
    const venue0 = params.get("venue");
    const pde0 = params.get("pde");
    const status0 = params.get("status");

    if (q0 && qInput) qInput.value = q0;
    if (method0 && fMethod) fMethod.value = method0;
    if (venue0 && fVenue) fVenue.value = venue0;
    if (pde0 && fPde) fPde.value = pde0;
    if (status0 && fStatus) fStatus.value = status0;
  } catch (e) {
    // ignore malformed URLs
  }

  // Rendering state
  let renderLimit = 500;

  function updateSelUI() {
    const n = selected.size;
    if (selCountEl) selCountEl.textContent = n.toString();
    setBtnEnabled("#btnCopyBib", n > 0);
    setBtnEnabled("#btnDownloadBib", n > 0);
    setBtnEnabled("#btnClearSel", n > 0);
  }

  function render() {
    const q = (qInput?.value || "").trim();
    const method = fMethod?.value || "";
    const venue = fVenue?.value || "";
    const pde = fPde?.value || "";
    const status = fStatus?.value || "";

    const filtered = papers.filter(p => matchesQuery(p, q) && matchesFilter(p, method, venue, pde, status));
    const shown = filtered.slice(0, renderLimit);

    tbody.innerHTML = "";
    const frag = document.createDocumentFragment();
    for (const p of shown) frag.appendChild(createRow(p, selected));
    if (filtered.length > shown.length) {
      const remaining = Math.min(500, filtered.length - shown.length);
      frag.appendChild(createLoadMoreRow(remaining, () => { renderLimit += 500; render(); }));
    }
    tbody.appendChild(frag);

    if (shownCountEl) shownCountEl.textContent = `${Math.min(renderLimit, filtered.length)}`;
    updateSelUI();
  }

  // Events: filters + search
  for (const el of [qInput, fMethod, fVenue, fPde, fStatus]) {
    if (!el) continue;
    el.addEventListener("input", () => { renderLimit = 500; render(); });
    el.addEventListener("change", () => { renderLimit = 500; render(); });
  }

  // Event delegation for checkbox selection
  tbody.addEventListener("change", (ev) => {
    const t = ev.target;
    if (!(t instanceof HTMLInputElement)) return;
    if (!t.classList.contains("pick")) return;
    const slug = t.dataset.slug;
    if (!slug) return;
    if (t.checked) selected.add(slug);
    else selected.delete(slug);
    updateSelUI();
  });

  btnClear?.addEventListener("click", () => {
    selected.clear();
    render();
  });

  btnCopy?.addEventListener("click", async () => {
    const bib = getSelectedBibtex(papersBySlug, selected);
    try {
      await navigator.clipboard.writeText(bib);
      btnCopy.textContent = "Copied!";
      setTimeout(() => { btnCopy.textContent = "Copy BibTeX"; }, 1200);
    } catch (e) {
      alert("Clipboard copy failed. Use Download .bib instead.");
    }
  });

  btnDownload?.addEventListener("click", () => {
    const bib = getSelectedBibtex(papersBySlug, selected);
    const date = new Date().toISOString().slice(0, 10);
    downloadText(`PartialObs-PDEBench-${date}.bib`, bib);
  });

  render();
}

document.addEventListener("DOMContentLoaded", () => {
  main().catch(err => {
    console.error(err);
    const tbody = qs("#paperRows");
    if (tbody) tbody.innerHTML = `<tr><td colspan="8" class="muted">Failed to load papers DB. Check console.</td></tr>`;
  });
});
