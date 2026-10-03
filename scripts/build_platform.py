"""Build the public evaluation website from local, versioned sources."""

import hashlib
import html
import json
import math
from pathlib import Path
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DATA = json.loads((ROOT / "data/platform.json").read_text(encoding="utf-8"))
SNAPSHOT_PATH = DOCS / f"assets/snapshots/{DATA['snapshot']}.json"
SNAPSHOT = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
PUBLIC = "https://github.com/ru1ch3n/PDE-OBS"
WEBSITE = "https://github.com/ru1ch3n/PartialObs--PDEBench"
PIN = PUBLIC + "/blob/" + SNAPSHOT["source_commit"]
VERSION = DATA["version"] + "-platform-v1"
NAV = [
    ("benchmark", "Benchmark"),
    ("results", "Results"),
    ("methods", "Methods"),
    ("studies", "Research Studies"),
    ("releases", "Releases"),
    ("contribute", "Contribute"),
    ("guide", "Docs"),
]


def esc(value):
    return html.escape(str(value), quote=True)


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def link(url, label, cls=""):
    return f'<a href="{esc(url)}" class="{cls}">{label}</a>'


def issue(template, title=None):
    query = {"template": template + ".yml"}
    if title:
        query["title"] = title
    return WEBSITE + "/issues/new?" + urlencode(query)


def shell(key, title, content, description="", extra_head=""):
    root = "./" if key == "home" else "../"
    navigation = "".join(
        f'<a href="{root}{slug}/"'
        + (' aria-current="page"' if key == slug else "")
        + f">{label}</a>"
        for slug, label in NAV
    )
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)} · PDE-OBS</title><meta name="description" content="{esc(description or "Open, reproducible evaluation of PDE models under changing observation conditions.")}">
<meta name="theme-color" content="#122a2a"><link rel="icon" href="{root}assets/platform-mark.svg" type="image/svg+xml">
<link rel="stylesheet" href="{root}assets/platform.css?v={VERSION}"><link rel="stylesheet" href="{root}assets/platform-layout.css?v={VERSION}">{extra_head}</head>
<body class="platform" data-root="{root}"><a class="skip" href="#main">Skip to content</a>
<header class="site-header"><div class="wrap header-inner"><a class="brand" href="{root}" aria-label="PDE-OBS home"><span class="brand-mark" aria-hidden="true">▦</span>PDE-OBS<span class="brand-note">OBSERVE. COMPARE. UNDERSTAND.</span></a>
<button class="menu-toggle" aria-expanded="false" aria-controls="primary-nav">Menu <span aria-hidden="true">＋</span></button>
<nav class="nav" id="primary-nav" aria-label="Primary">{navigation}</nav></div></header>
<main id="main">{content}</main>
<footer class="site-footer"><div class="wrap footer-top"><div><a class="brand" href="{root}">PDE-OBS</a><p>Same physics. Different observations.<br>Evidence you can inspect.</p></div><div class="footer-links">
{link(root + "research/", "Literature")}{link(root + "guide/#citation", "Paper & citation")}{link(root + "contribute/#credit", "Credit & authorship")}{link(root + "guide/#archive", "Earlier tools & archive")}{link(WEBSITE, "Website source ↗")}{link(PUBLIC, "Benchmark code ↗")}</div></div>
<div class="wrap footer-bottom"><span>Open evaluation · Reproducible evidence · Collaborative research</span><span>Website {DATA["version"]} · Paper snapshot frozen</span></div></footer>
<script src="{root}assets/platform.js?v={VERSION}" defer></script></body></html>'''


def intro(kicker, title, subtitle):
    return f'<section class="page-intro wrap"><p class="eyebrow">{kicker}</p><h1>{title}</h1><p class="lede">{subtitle}</p></section>'


def section_head(number, title, subtitle="", action=""):
    return f'<div class="section-head"><div><p class="eyebrow">{number}</p><h2>{title}</h2><p>{subtitle}</p></div>{action}</div>'


def actions(root="./"):
    cards = [
        (
            "01",
            "Submit a Method",
            "Bring an adapter, configuration, checkpoint, information permissions and evaluation evidence.",
            "submit_method",
            "Submit a method ↗",
        ),
        (
            "02",
            "Reproduce a Result",
            "Specify what you checked: configuration, scoring, inference or training. Attach reproducible evidence.",
            "reproduce_result",
            "Share a reproduction ↗",
        ),
        (
            "03",
            "Join a Study",
            "Help shape a question, design an experiment, analyze a result or develop the manuscript.",
            "join_study",
            "Propose a contribution ↗",
        ),
    ]
    return (
        '<div class="cards three">'
        + "".join(
            f'<article class="card action-card"><span class="card-index">{n}</span><h3>{title}</h3><p>{desc}</p>{link(issue(template), label, "text-link")}</article>'
            for n, title, desc, template, label in cards
        )
        + "</div>"
    )


def matrix(pde="darcy", method="fno"):
    views = DATA["views"]
    values = [
        SNAPSHOT["records"][f"{pde}/{method}/{v['id']}"]["blocks"][w["id"]]["joint"]["mean"]
        for v in views
        for w in views
    ]
    lo, hi = min(math.log10(v) for v in values), max(math.log10(v) for v in values)
    out = '<div class="matrix-scroll"><table class="matrix"><caption>Darcy · FNO adaptation · mean relative L2<br><span>Rows: training observation · Columns: test observation</span></caption><thead><tr><th scope="col">Train / Test</th>'
    out += (
        "".join(
            f'<th scope="col"><abbr title="{v["label"]}">{v["short"]}</abbr></th>' for v in views
        )
        + "</tr></thead><tbody>"
    )
    for v in views:
        out += f'<tr><th scope="row"><abbr title="{v["label"]}">{v["short"]}</abbr></th>'
        for w in views:
            cell = SNAPSHOT["records"][f"{pde}/{method}/{v['id']}"]["blocks"][w["id"]]["joint"]
            val = cell["mean"]
            scale = (math.log10(val) - lo) / (hi - lo)
            light = 94 - scale * 67
            color = "#122a2a" if light > 60 else "#fff"
            q = urlencode({"pde": pde, "method": method, "train": v["id"], "test": w["id"]})
            title = f"{v['label']} → {w['label']}: {val:.6g}, SD {cell['std']:.6g}"
            out += f'<td class="{"diagonal" if v == w else ""}" style="background:hsl(170 34% {light:.1f}%);color:{color}"><a href="./results/?{q}" title="{title}">{val:.2f}</a></td>'
        out += "</tr>"
    return out + "</tbody></table></div>"


def home():
    body = f"""<section class="home-hero"><div class="wrap hero-grid"><div class="hero-copy"><p class="eyebrow"><span class="status-dot"></span> AN OPEN EVALUATION & RESEARCH PLATFORM</p>
<h1>Same physics.<br>Different<br><em>observations.</em></h1>
<p class="hero-statement">Open, reproducible evaluation of PDE models under changing observation conditions.</p>
<p class="hero-detail">Compare reconstruction and forecasting methods on shared physical records while varying what is observed. Explore the reference benchmark, inspect reproducible results, and contribute new evaluations.</p>
<div class="button-row">{link("./results/", 'Explore Results <span aria-hidden="true">↗</span>', "button primary")}{link("./guide/", "Get Started", "button outline")}{link("./contribute/", "Contribute", "hero-text-link")}</div>
<div class="resource-links">{link(DATA["paper"]["url"], "Paper ↗")}{link("https://huggingface.co/datasets/ru1ch3n/PDE-OBS", "Data ↗")}{link(PUBLIC, "Code ↗")}{link("https://huggingface.co/ru1ch3n/PDE-OBS", "Checkpoints ↗")}</div></div>
<div class="observation-art" aria-label="Illustration of a shared physical field viewed through changing observation layouts"><div class="art-top"><span>ONE PHYSICAL RECORD</span><span>↘ MANY OBSERVATION VIEWS</span></div><div class="field-layer"><div class="field-mesh"></div><span class="field-label">u(x, y)</span></div><div class="mask-cards"><div class="mask-tile random-mask"><span>RANDOM</span></div><div class="mask-tile line-mask"><span>LINE SENSORS</span></div><div class="mask-tile block-mask"><span>BLOCK</span></div></div><p class="art-caption">Hold the record fixed.<br>Change what the model can see.</p><span class="art-footnote">SCHEMATIC · NOT A SIMULATION</span></div></div></section>
<section class="fact-strip wrap" aria-label="Benchmark scope"><div><span>DATA RESOURCE</span><strong>560,000</strong><p>physical records · 7 PDE families</p></div><div><span>REFERENCE STUDY</span><strong>14,000</strong><p>records · 7 adapted baselines · 9 test patterns</p></div><div><span>REFERENCE RESULTS</span><strong>3,969</strong><p>evaluation blocks · 441 trained models</p></div></section>
<section class="wrap section">{section_head("01 / REFERENCE EVIDENCE", "Observation shift, made visible.", "A real slice of the frozen paper snapshot. Every cell links to its evidence.", link("./results/", "Explore all results ↗", "text-link"))}
<div class="results-preview"><div class="matrix-panel">{matrix()}<div class="matrix-legend"><span class="legend-gradient"></span> Lower error → Higher error <span>Logarithmic color scale within this matrix</span></div></div><div class="preview-copy"><span class="tag">PAPER SNAPSHOT</span><h3>A good matched result is only part of the story.</h3><p>The diagonal keeps training and test observations matched. Every other cell changes the observation pattern.</p><p>Read these as adapter-specific results. Training budgets are heterogeneous; the snapshot does not establish a universal method ranking.</p><a class="text-link" href="./benchmark/">Understand the protocol ↗</a><p class="fine">200 held-out physical records per cell. Values show mean joint relative L2. Full precision and sample SD are available in the explorer.</p></div></div></section>
<section class="band"><div class="wrap section">{section_head("02 / THE PLATFORM", "A stable reference. Room for new questions.")}
<div class="cards three layers"><article class="card"><span class="tag">FROZEN</span><h3>Reference Benchmark</h3><p>The original paper’s protocol, results and checkpoint identities stay available as a versioned reference.</p>{link("./benchmark/", "Inspect the benchmark ↗", "text-link")}</article><article class="card"><span class="tag">OPEN FOR CONTRIBUTIONS</span><h3>Evaluations</h3><p>New methods and scoped reproductions enter separate, traceable releases. No community results have been published yet.</p>{link("./results/?collection=community", "Community evaluations ↗", "text-link")}</article><article class="card"><span class="tag">PROPOSED</span><h3>Research Studies</h3><p>Independent questions, explicit controls and new evidence. Three directions are open for study design.</p>{link("./studies/", "Explore research directions ↗", "text-link")}</article></div></div></section>
<section class="wrap section">{section_head("03 / LATEST UPDATE", "The platform takes its next step.", action=link("./releases/", "All releases ↗", "text-link"))}<a class="release-highlight" href="./releases/#website-20261003"><span class="release-date">03 OCT 2026<br><span class="tag">WEBSITE RELEASE</span></span><div><h3>A new home for evaluation and collaboration</h3><p>Versioned results, transparent method cards and practical ways to contribute. The paper snapshot remains fixed.</p></div><span class="arrow" aria-hidden="true">↗</span></a></section>
<section class="wrap section section-topless">{section_head("04 / OPEN RESEARCH", "Start with a question.", "Proposed directions. Study leads and experiments have not yet been assigned.", link("./studies/", "View study plans ↗", "text-link"))}<div class="study-list">"""
    for study in DATA["studies"]:
        body += f'<a href="./studies/#{study["id"]}" class="study-row"><span>{study["number"]}</span><div><span class="tag">{study["status"]}</span><h3>{study["title"]}</h3><p>{study["question"]}</p></div><span class="arrow" aria-hidden="true">↗</span></a>'
    body += f'</div></section><section class="contribution-band"><div class="wrap section">{section_head("05 / CONTRIBUTE", "Build the next piece of evidence.", "Methods, reproductions and research contributions each have a clear starting point.")}{actions()}</div></section>'
    return shell("home", "Open evaluation & research", body)


def benchmark():
    body = intro(
        "REFERENCE BENCHMARK / FROZEN",
        "A reference you can return to.",
        "The original PDE-OBS study measures what happens when observation patterns change while the underlying physical records stay fixed.",
    )
    body += f"""<div class="wrap section section-topless"><div class="button-row">{link(DATA["paper"]["url"], "Read paper v2 ↗", "button dark")}{link("../results/", "Explore paper results", "button")}{link(SNAPSHOT["manifest_url"], "Source manifest ↗", "text-link")}</div>
<div class="cards three spaced"><article class="card"><p class="eyebrow">DATA RESOURCE</p><h2>7 PDE families</h2><p>560,000 physical records in the released resource. The reference study uses 14,000 records; these are different denominators.</p></article><article class="card"><p class="eyebrow">EVALUATION GRID</p><h2>9 × 9 views</h2><p>Seven adapted methods across seven PDEs, each trained on nine observation layouts and evaluated on all nine.</p></article><article class="card"><p class="eyebrow">PREDICTION EVIDENCE</p><h2>200 records / cell</h2><p>3,969 evaluation blocks from 441 checkpoints. Repeated views of a physical record are paired observations.</p></article></div>
<div class="split-section"><div><p class="eyebrow">WHAT WAS EVALUATED</p><h2>Two empirical tasks.</h2></div><div><article class="text-block"><h3>Stationary reconstruction</h3><p>Darcy, Poisson and Helmholtz: reconstruct the solution field from partial observations.</p></article><article class="text-block"><h3>Short-horizon forecasting</h3><p>Heat, reaction–diffusion, Burgers and Navier–Stokes: observe an initial state and forecast the next three stored states.</p></article><p class="notice">Forward and inverse tasks are implemented extensions, not evaluated at the same scale as these reference experiments. Training uses full target fields for supervision.</p></div></div>
<div class="split-section"><div><p class="eyebrow">OBSERVATION CONDITIONS</p><h2>Change the view,<br>keep the record.</h2></div><div class="view-grid">"""
    body += "".join(
        f"<div><strong>{v['short']}</strong><span>{v['label']}</span></div>" for v in DATA["views"]
    )
    body += f"""</div></div><div class="split-section"><div><p class="eyebrow">READING THE EVIDENCE</p><h2>Compare with context.</h2></div><div><ul class="prose-list"><li>Cell metric: mean per-record joint relative-L2 error, with sample SD (ddof = 1) across 200 records.</li><li>Forecasting scores combine three frames jointly; they are not the mean of three horizon errors.</li><li>SD is neither a confidence interval nor random-seed uncertainty. Histories are single-seed and heterogeneous.</li><li>PINO has additional training-time physics information. Its inference input remains the masked observation.</li><li>Comparable compute measurements are unavailable here. Actual epochs and training cohorts are shown per result.</li></ul>{link("../methods/", "Read the method cards ↗", "text-link")}</div></div>
<div class="notice"><h3>Data generators have a different role.</h3><p>Numerical solvers produce complete fields and trajectories. They are not automatically reconstruction or forecasting competitors with the same information permissions. Residual checks demonstrate discrete consistency; they do not establish independent numerical accuracy or grid convergence.</p>{link(PIN + "/docs/dataset_card.md", "Dataset documentation ↗", "text-link")}</div>
<div class="split-section" id="versions"><div><p class="eyebrow">VERSIONED REFERENCE</p><h2>Trace the snapshot.</h2></div><div><dl class="metadata"><dt>Paper</dt><dd>{link(DATA["paper"]["url"], "arXiv:2609.36521v2 · 30 Sep 2026")}</dd><dt>Result snapshot</dt><dd>{DATA["snapshot"]}</dd><dt>Evaluator</dt><dd>{SNAPSHOT["evaluator_version"]}</dd><dt>Software identity</dt><dd>{link(PUBLIC + "/tree/" + SNAPSHOT["source_commit"], SNAPSHOT["source_commit"])}</dd><dt>Data & split bindings</dt><dd>{link(SNAPSHOT["dataset_bindings_url"], "Per-shard hashes, split identities and physical-time mappings ↗")}</dd><dt>Protocol & checkpoints</dt><dd>Contract and checkpoint hashes are included per block in the result export.</dd></dl></div></div></div>"""
    return shell("benchmark", "Reference benchmark", body)


def select(name, label, options, all_label="All"):
    return (
        f'<label>{label}<select id="filter-{name}" name="{name}"><option value="">{all_label}</option>'
        + "".join(f'<option value="{esc(k)}">{esc(v)}</option>' for k, v in options)
        + "</select></label>"
    )


def results():
    body = intro(
        "RESULTS / INSPECT THE EVIDENCE",
        "Compare conditions. Follow the evidence.",
        "Filter individual evaluation blocks, inspect their provenance and export the comparison. Every result belongs to a named snapshot.",
    )
    body += """<section class="wrap section section-topless" id="result-explorer"><div class="collection-tabs" role="tablist" aria-label="Result collection"><button role="tab" id="tab-paper" aria-controls="paper-panel" aria-selected="true" data-collection="paper">Paper Snapshot <span>3,969</span></button><button role="tab" id="tab-community" aria-controls="community-panel" aria-selected="false" tabindex="-1" data-collection="community">Community Evaluations <span>0</span></button><button role="tab" id="tab-study" aria-controls="study-panel" aria-selected="false" tabindex="-1" data-collection="study">Study Results <span>0</span></button></div>
<div id="paper-panel" role="tabpanel" aria-labelledby="tab-paper"><div class="snapshot-bar"><span><span class="status-dot"></span> FROZEN · paper-20260925-v1</span><a href="#verification">Artifacts-checked · source audit ↗</a></div>
<p class="notice compact">Mean joint relative L2 ± sample SD over 200 held-out records. Training histories and permissions differ across methods. Compute cost is unavailable; no universal ranking is implied.</p><form class="filter-grid" id="result-filters">"""
    body += select(
        "task",
        "Task",
        [("recovery", "Stationary reconstruction"), ("rollout", "Short-horizon forecasting")],
        "All tasks",
    )
    body += select("pde", "PDE family", DATA["pdes"].items(), "All PDEs")
    body += select(
        "method", "Method adapter", [(m["id"], m["name"]) for m in DATA["methods"]], "All methods"
    )
    for name, label in [("train", "Training observation"), ("test", "Test observation")]:
        body += select(name, label, [(v["id"], v["label"]) for v in DATA["views"]], "All patterns")
    body += select(
        "match",
        "Observation relation",
        [("matched", "Matched only"), ("shifted", "Shifted only")],
        "Matched & shifted",
    )
    body += """<div class="filter-action"><button type="reset" class="text-button">Reset filters ↺</button></div></form>
<div class="table-toolbar"><p id="result-count" role="status">Loading the frozen snapshot…</p><div class="button-row"><label class="sort-label">Order<select id="result-sort"><option value="identity">PDE / method / observation</option><option value="error">Error: low to high (filtered rows)</option></select></label><button class="button small" id="export-csv" disabled>Export CSV ↓</button><button class="button small" id="export-json" disabled>Export JSON ↓</button></div></div>
<div class="table-scroll" tabindex="0" aria-label="Evaluation blocks table"><table class="results-table"><thead><tr><th scope="col">PDE / task</th><th scope="col">Method adapter</th><th scope="col">Train → test</th><th scope="col">Mean ± SD</th><th scope="col">Training budget</th><th scope="col">Evidence</th></tr></thead><tbody id="result-rows"></tbody></table></div>
<div class="pagination"><button class="button small" id="previous-page" disabled>← Previous</button><span id="page-status" aria-live="polite"></span><button class="button small" id="next-page" disabled>Next →</button></div>
<noscript><p class="notice">Enable JavaScript to filter results, or download the complete JSON snapshot below. The benchmark, method cards and reference matrix work without JavaScript.</p></noscript></div>"""
    for key, title, desc, target, label in [
        (
            "community",
            "The next result could be yours.",
            "No community evaluation has been published yet. Submit a versioned adapter and evidence package; checked contributions will receive their own snapshot.",
            issue("submit_method"),
            "Submit a Method ↗",
        ),
        (
            "study",
            "Questions first. Results when there is evidence.",
            "Research directions are proposed. No study results or completed experiments are claimed.",
            "../studies/",
            "Explore Research Studies ↗",
        ),
    ]:
        body += f'<div id="{key}-panel" role="tabpanel" aria-labelledby="tab-{key}" hidden><div class="empty-state"><span class="empty-icon" aria-hidden="true">＋</span><h2>{title}</h2><p>{desc}</p>{link(target, label, "button dark")}</div></div>'
    body += f'''<div class="split-section" id="verification"><div><p class="eyebrow">VERIFICATION HAS A SCOPE</p><h2>What does the label mean?</h2></div><div><dl class="verification-list"><dt>01 · Author-reported</dt><dd>A submitted claim with declared provenance. It has not yet passed an artifact check.</dd><dt>02 · Artifacts-checked</dt><dd>Specified files, identities, configurations and scores have been checked. The paper snapshot uses the <a href="{SNAPSHOT["audit_url"]}">published source audit</a>, which checked prediction integrity and scoring. This website import is not an independent rerun.</dd><dt>03 · Rerun-verified</dt><dd>Requires a recorded reproduction scope and evidence: rescoring predictions, rerunning inference and retraining are separate operations. The source audit did not rerun inference or training.</dd></dl></div></div>
<div class="download-strip"><div><h3>Keep a complete, versioned copy.</h3><p>Full precision, checkpoint identities, block hashes, configurations and source provenance.</p></div>{link("../assets/snapshots/" + DATA["snapshot"] + ".json", "Download snapshot JSON ↓", "button")}{link("../assets/snapshots/manifest.json", "SHA-256 manifest ↗", "text-link")}</div></section>
<dialog id="evidence-dialog"><div class="dialog-top"><h2>Result evidence</h2><button class="button small" id="close-evidence" aria-label="Close result evidence">Close ×</button></div><div id="evidence-content"></div></dialog>
<script src="../assets/results.js?v={VERSION}" defer></script>'''
    return shell("results", "Results explorer", body)


def methods():
    body = intro(
        "METHODS / IMPLEMENTATION MATTERS",
        "Seven adapters. Explicit boundaries.",
        "These cards describe the implementations evaluated in the paper snapshot. A literature entry, an adapter and an evaluated checkpoint are distinct records.",
    )
    body += '<section class="wrap section section-topless"><div class="notice">All seven methods have 567 evaluation blocks: seven PDEs × nine training views × nine test views. Inspected upstream revisions are reference anchors; evaluated checkpoint identities are recorded separately.</div><div class="method-grid">'
    for method in DATA["methods"]:
        records = [r for r in SNAPSHOT["records"].values() if r["method"] == method["id"]]
        epochs = [r["actual_epochs"] for r in records]
        extra = (
            " Full target supervision plus physics residuals using coefficients/source terms."
            if method["id"] == "pino"
            else " Full target-field supervision."
        )
        body += f'''<article class="card method-card" id="{method["id"]}"><div class="method-card-top"><span class="tag">{method["kind"]}</span><span class="mono">567 BLOCKS</span></div><h2>{method["name"]}</h2><p>{method["note"]}</p>
<div class="resource-links">{link(method["paper"], "Original paper ↗")}{link(method["upstream"] + "/tree/" + method["revision"], "Inspected upstream ↗")}</div>
<dl class="metadata"><dt>Tasks</dt><dd>Stationary reconstruction & short-horizon forecasting</dd><dt>Training inputs</dt><dd>Masked observations and observation mask.{extra}</dd><dt>Inference inputs</dt><dd>Masked observations and mask; coordinates where used by the adapter.</dd><dt>Pretraining</dt><dd>External pretraining data are not specified in this snapshot; no contamination-free claim is made.</dd><dt>Actual epochs</dt><dd>{min(epochs):,}–{max(epochs):,} across 63 checkpoints; heterogeneous training cohorts, single seed.</dd><dt>Compute cost</dt><dd>Comparable runtime, memory and hardware-normalized budgets unavailable.</dd><dt>Verification</dt><dd>{link("../results/#verification", "Artifacts-checked · source audit")}</dd></dl>
<details><summary>Source identity & checkpoint evidence</summary><p>Inspected upstream: <code>{method["revision"]}</code></p><p>Evaluated implementation: {link(PUBLIC + "/tree/" + SNAPSHOT["source_commit"], SNAPSHOT["source_commit"][:12])}. {link(PIN + "/docs/methods_card.md", "Public implementation notes ↗")}</p><p>{link("https://huggingface.co/ru1ch3n/PDE-OBS", "Checkpoint resource ↗")} · Match each downloaded checkpoint to the SHA-256 in the result evidence. Tensor accessibility has not been independently verified by this website.</p></details>
{link("../results/?method=" + method["id"], "Inspect evaluation blocks ↗", "text-link")}</article>'''
    body += f'</div><div class="download-strip"><div><h3>Bring a new method into the comparison.</h3><p>Start with a small, declared coverage set and a reproducible adapter.</p></div>{link(issue("submit_method"), "Submit a Method ↗", "button dark")}</div></section>'
    return shell("methods", "Method cards", body)


def studies():
    body = intro(
        "RESEARCH STUDIES / OPEN QUESTIONS",
        "New evidence starts with a better question.",
        "Independent studies extend the reference benchmark through new hypotheses, controlled experiments and explicit contribution plans.",
    )
    body += '<section class="wrap section section-topless"><div class="notice">All three directions are proposed. Study leads, start dates and experiments are unassigned. The intended sequence is to scope the first direction before starting a pilot; no active study or result is implied.</div>'
    for study in DATA["studies"]:
        body += f'''<article class="study-detail" id="{study["id"]}"><div class="study-number">{study["number"]}</div><div><span class="tag">{study["status"]}</span><h2>{study["title"]}</h2><p class="study-question">{study["question"]}</p><div class="cards two plain"><div><h3>Beyond the reference paper</h3><p>{study["difference"]}</p></div><div><h3>Evidence that would be needed</h3><p>{study["evidence"]}</p></div></div><p class="fine"><strong>Possible contributions:</strong> {study["roles"]}</p><p class="fine"><strong>Lead:</strong> Unassigned · <strong>Results:</strong> None yet</p>{link(issue("join_study", "[Study] " + study["title"]), "Help design this study ↗", "button")}</div></article>'''
    body += """<div class="split-section"><div><p class="eyebrow">FROM QUESTION TO STUDY</p><h2>Evidence sets the pace.</h2></div><div><ol class="prose-list"><li>Define the new question, why the reference paper cannot answer it, and which experiment could falsify the working hypothesis.</li><li>Agree on a lead, roles, permissions, controls and a minimal pilot before expanding computation.</li><li>Publish a scoped technical report when evidence supports a coherent finding. A calendar date alone does not trigger a paper.</li><li>Determine authorship separately for each study, with manuscript participation and accountability.</li></ol><a href="../contribute/#credit" class="text-link">Credit & authorship policy ↗</a></div></div></section>"""
    return shell("studies", "Research studies", body)


def releases():
    body = intro(
        "RELEASES / A TRACEABLE RECORD",
        "Keep the history. Add the next result.",
        "Website changes, evaluation snapshots and research reports have different release records. New evidence never silently replaces a paper result.",
    )
    body += '<section class="wrap section section-topless"><div class="release-timeline">'
    for i, release in enumerate(DATA["releases"]):
        body += f'<article class="release-entry" id="{"website-20261003" if i == 0 else "release-" + release["date"]}"><time datetime="{release["date"]}">{release["date"]}</time><div><span class="tag">{release["kind"]}</span><h2>{release["title"]}</h2><p>{release["description"]}</p>'
        if i == 1:
            body += link(DATA["paper"]["url"], "Versioned preprint ↗", "text-link")
        if i == 2:
            body += link(
                "../assets/snapshots/" + DATA["snapshot"] + ".json",
                "Inspect frozen snapshot ↗",
                "text-link",
            )
        body += "</div></article>"
    body += f"""</div><div class="split-section"><div><p class="eyebrow">RELEASE POLICY</p><h2>Publish when there is something to show.</h2></div><div><h3>Evaluation releases</h3><p>A monthly review cadence is intended for checked new results, corrections or maintenance. No release is promised without qualifying work; partial coverage is labeled explicitly.</p><h3>Study reports</h3><p>A quarterly review can produce a technical note when there is a clear question and useful evidence. Engineering-only changes remain release notes.</p><h3>Corrections & versions</h3><p>A correction receives a new snapshot and a change note linking the prior version. Dataset, split, protocol, evaluator, software and checkpoints retain distinct identities.</p></div></div><div class="notice"><h3>Current snapshot integrity</h3><p><code>{DATA["snapshot"]}</code> · SHA-256 <code>{hashlib.sha256(SNAPSHOT_PATH.read_bytes()).hexdigest()}</code></p>{link("../assets/snapshots/manifest.json", "Download the manifest ↗", "text-link")}</div></section>"""
    return shell("releases", "Releases & versions", body)


def contribute():
    body = intro(
        "CONTRIBUTE / OPEN COLLABORATION",
        "Make your contribution reproducible.",
        "Start with a method, a scoped reproduction or a research question. The issue forms below capture the evidence needed to make the work useful.",
    )
    body += f'<section class="wrap section section-topless">{actions()}<div class="split-section"><div><p class="eyebrow">THE CONTRIBUTION PATH</p><h2>From submission<br>to public evidence.</h2></div><div><ol class="prose-list"><li><strong>Declare the scope.</strong> Name the task, PDEs, observation conditions, information permissions and intended coverage.</li><li><strong>Bind the artifacts.</strong> Supply code and evaluator versions, configuration, data/split identity, checkpoint hashes and result files. Keep large tensors in a suitable artifact store.</li><li><strong>Record the check.</strong> State which configurations or files were inspected, and whether scoring, inference or training was rerun. Attach commands, environment and evidence.</li><li><strong>Release separately.</strong> Accepted results receive a method card, coverage statement, verification scope and versioned snapshot. The original paper snapshot stays fixed.</li></ol><p class="notice compact">A source-code merge is not scientific validation. Evaluation records need explicit evidence and scope before receiving a verification label.</p></div></div>'
    body += """<div class="split-section" id="credit"><div><p class="eyebrow">CREDIT & AUTHORSHIP</p><h2>Credit the work.<br>Agree on responsibilities.</h2></div><div><p>Accepted contributions are credited in the project and release records. Research-paper authorship is determined separately for each study, based on substantive contributions, manuscript participation, accountability, and the applicable venue policy.</p><p><strong>Participation, compute donation, or a pull request does not guarantee authorship.</strong></p><p>Contribution records may identify software, validation, formal analysis, conceptualization and writing roles. These describe work; they are not an automatic authorship checklist. New collaboration does not imply changes to the existing paper’s author list.</p></div></div>"""
    body += f'<div class="download-strip"><div><h3>A website correction or literature suggestion?</h3><p>Use the website issue tracker. Include the affected page and a supporting public source.</p></div>{link(WEBSITE + "/issues/new/choose", "Website issue tracker ↗", "button")}</div></section>'
    return shell("contribute", "Contribute", body)


def guide():
    paper = DATA["paper"]
    bibtex = (
        "@article{xu2026pdeobs,\n  title = {"
        + paper["title"]
        + "},\n  author = {"
        + " and ".join(paper["authors"])
        + "},\n  journal = {arXiv preprint arXiv:2609.36521},\n  year = {2026},\n  doi = {"
        + paper["doi"]
        + "},\n  url = {"
        + paper["url"]
        + "}\n}"
    )
    body = intro(
        "DOCS / GET STARTED",
        "Choose your starting point.",
        "Inspect the reference evidence, work with the public code and data, or prepare a contribution with a clearly defined scope.",
    )
    body += f'''<section class="wrap section section-topless"><div class="cards three"><article class="card"><span class="card-index">01</span><h2>Understand the benchmark</h2><p>Start with the empirical tasks, observation patterns and information permissions.</p>{link("../benchmark/", "Reference benchmark ↗", "text-link")}{link(DATA["paper"]["url"], "Read paper v2 ↗", "text-link")}</article><article class="card"><span class="card-index">02</span><h2>Use the public release</h2><p>Install the current benchmark from its canonical project. Follow its environment and artifact instructions.</p>{link(PIN + "/docs/installation.md", "Installation guide ↗", "text-link")}{link("https://huggingface.co/datasets/ru1ch3n/PDE-OBS", "Data resource ↗", "text-link")}{link("https://huggingface.co/ru1ch3n/PDE-OBS", "Checkpoints ↗", "text-link")}</article><article class="card"><span class="card-index">03</span><h2>Inspect or extend</h2><p>Export results with provenance, examine an adapter, then submit a scoped contribution.</p>{link("../results/", "Result explorer ↗", "text-link")}{link(PIN + "/docs/extending.md", "Extension guide ↗", "text-link")}{link("../contribute/", "Contribution checklist ↗", "text-link")}</article></div>
<div class="split-section"><div><p class="eyebrow">REPRODUCIBILITY</p><h2>Decide what you are reproducing.</h2></div><div><p>A statistical re-analysis of the released index does not rerun a model. An inference reproduction requires matching checkpoints, data, observation contracts and evaluator versions. Retraining adds configuration, environment, seed and budget requirements.</p><p>The public <a href="{SNAPSHOT["audit_url"]}">prediction-audit README</a> provides CPU-only analysis commands and states the evidence boundary. Follow its instructions for a new output path; preserve the original snapshot.</p>{link("../results/#verification", "Verification labels and scope ↗", "text-link")}</div></div>
<div class="split-section" id="citation"><div><p class="eyebrow">PAPER & CITATION</p><h2>{paper["title"]}</h2><p>Public preprint · v2 · {paper["updated"]}</p></div><div><p>{", ".join(paper["authors"])}.</p><p>{link(paper["url"], "arXiv:2609.36521v2 ↗")} · {link("https://doi.org/" + paper["doi"], "DOI ↗")}</p><pre class="citation">{esc(bibtex)}</pre><p class="fine">Website citation metadata follows the public arXiv v2 record. Reference results retain their independent snapshot and source identities.</p></div></div>
<div class="split-section" id="archive"><div><p class="eyebrow">LITERATURE & EARLIER TOOLS</p><h2>Keep useful context accessible.</h2></div><div><p>The literature collection is a research resource, not a list of evaluated methods. Earlier planning tools describe historical protocols and remain separate from the current paper snapshot.</p><div class="archive-links">{link("../research/", "Literature index ↗")}{link("../builder/", "Earlier Benchmark Builder ↗")}{link("../server/", "Earlier server & Slurm tools ↗")}{link("../benchmark/archive.html", "Earlier benchmark plan ↗")}{link("../contribute/literature.html", "Literature record editor ↗")}{link("../index-202608-archive.html", "August website archive ↗")}</div></div></div></section>'''
    return shell("guide", "Documentation & citation", body)


def build(literature):
    # Keep the existing literature URL while giving its index the platform shell.
    literature = literature.split('<main class="container">', 1)[1].split("<footer", 1)[0]
    literature = literature.split("</aside>", 1)[1].strip()
    literature = literature.replace("Research index", "Literature index").replace(
        "use the <b>Contribute</b> tab",
        'use the <a href="../contribute/literature.html">literature record editor</a>',
    )
    literature = (
        intro(
            "LITERATURE / CONTEXT & SOURCES",
            "Explore the research landscape.",
            "A collection of related papers. Inclusion here does not imply an implemented adapter or a completed PDE-OBS evaluation.",
        )
        + '<div class="wrap literature-content">'
        + literature
        + "</div>"
    )
    head = (
        '<link rel="stylesheet" href="../assets/literature.css?v='
        + VERSION
        + '"><script>window.PAPERS_DB_URL="../assets/papers_db.json";</script><script defer src="../assets/research.js"></script>'
    )
    write(
        DOCS / "research/index.html", shell("research", "Literature", literature, extra_head=head)
    )
    for key, render in [
        ("home", home),
        ("benchmark", benchmark),
        ("results", results),
        ("methods", methods),
        ("studies", studies),
        ("releases", releases),
        ("contribute", contribute),
        ("guide", guide),
    ]:
        write(DOCS / ("index.html" if key == "home" else key + "/index.html"), render())
    write(DOCS / "assets/platform-data.json", json.dumps(DATA, ensure_ascii=False, indent=2) + "\n")
    manifest = {
        "snapshot": DATA["snapshot"],
        "file": SNAPSHOT_PATH.name,
        "sha256": hashlib.sha256(SNAPSHOT_PATH.read_bytes()).hexdigest(),
        "source_commit": SNAPSHOT["source_commit"],
        "source_url": SNAPSHOT["source_url"],
        "source_sha256": SNAPSHOT["source_sha256"],
        "models": 441,
        "blocks": 3969,
    }
    write(DOCS / "assets/snapshots/manifest.json", json.dumps(manifest, indent=2) + "\n")
    print("Generated evaluation platform: 8 pages, 1 immutable paper snapshot.")


if __name__ == "__main__":
    from generate_research_site import main

    main()
