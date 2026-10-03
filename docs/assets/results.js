(() => {
  'use strict';
  const root = document.querySelector('#result-explorer');
  if (!root) return;
  const form = document.querySelector('#result-filters');
  const body = document.querySelector('#result-rows');
  const count = document.querySelector('#result-count');
  const previous = document.querySelector('#previous-page');
  const next = document.querySelector('#next-page');
  const pageStatus = document.querySelector('#page-status');
  const sort = document.querySelector('#result-sort');
  const dialog = document.querySelector('#evidence-dialog');
  const tabs = [...root.querySelectorAll('[role=tab]')];
  const keys = ['task', 'pde', 'method', 'train', 'test', 'match'];
  let data, metadata, rows = [], filtered = [], page = 0, collection = 'paper', opener;
  const pageSize = 30;
  const node = (tag, text, className) => {
    const element = document.createElement(tag);
    if (text !== undefined) element.textContent = text;
    if (className) element.className = className;
    return element;
  };
  const name = (kind, id) => kind === 'pde' ? metadata.pdes[id] : kind === 'method'
    ? metadata.methods.find(m => m.id === id).name : metadata.views.find(v => v.id === id).label;
  const taskName = task => task === 'recovery' ? 'Reconstruction' : 'Forecasting';
  const number = value => value.toFixed(4);
  const addLink = (parent, href, text) => {
    const anchor = node('a', text); anchor.href = href; parent.append(anchor); return anchor;
  };

  function syncURL() {
    const url = new URL(location.href);
    keys.forEach(key => {
      const value = form.elements.namedItem(key).value;
      if (value) url.searchParams.set(key, value); else url.searchParams.delete(key);
    });
    if (collection !== 'paper') url.searchParams.set('collection', collection);
    else url.searchParams.delete('collection');
    if (sort.value !== 'identity') url.searchParams.set('sort', sort.value);
    else url.searchParams.delete('sort');
    history.replaceState(null, '', url);
  }

  function chooseCollection(value, sync = true) {
    collection = ['paper', 'community', 'study'].includes(value) ? value : 'paper';
    tabs.forEach(tab => {
      const selected = tab.dataset.collection === collection;
      tab.setAttribute('aria-selected', String(selected));
      tab.tabIndex = selected ? 0 : -1;
      document.getElementById(tab.getAttribute('aria-controls')).hidden = !selected;
    });
    if (sync) syncURL();
  }

  function readURL() {
    const params = new URLSearchParams(location.search);
    keys.forEach(key => {
      const control = form.elements.namedItem(key);
      const value = params.get(key) || '';
      control.value = [...control.options].some(o => o.value === value) ? value : '';
    });
    sort.value = params.get('sort') === 'error' ? 'error' : 'identity';
    chooseCollection(params.get('collection'), false);
    if (data) filter(false);
  }

  function evidence(row, button) {
    opener = button;
    const container = document.querySelector('#evidence-content');
    container.replaceChildren();
    container.append(node('p', `${name('pde', row.pde)} / ${name('method', row.method)} · ${name('view', row.train_view)} → ${name('view', row.test_view)}`));
    container.append(node('p', `${row.mean} ± ${row.std} · n = ${row.n}; ddof = 1.`, 'mono'));
    container.append(node('p', data.metric));
    container.append(node('h3', data.verification));
    container.append(node('p', data.verification_scope));
    const audit = node('p'); addLink(audit, data.audit_url, 'Read the published audit scope ↗'); container.append(audit);
    const record = data.records[row.identity];
    container.append(node('h3', 'Coverage, permissions and compute'));
    container.append(node('p', 'Paper coverage: 567 blocks per adapter; 81 per PDE/method pair. This block contains 200 held-out physical records. Full target fields supervise training. PINO additionally uses training-time physics metadata.'));
    container.append(node('p', data.compute_cost_note));
    container.append(node('h3', 'Version & artifact identities'));
    container.append(node('pre', JSON.stringify({
      snapshot: data.id, source_commit: data.source_commit, source_index_sha256: data.source_sha256,
      evaluator: data.evaluator_version, identity: row.identity, test_view: row.test_view,
      checkpoint_sha256: record.checkpoint_released_sha256,
      contract_sha256: row.block.contract_sha256, file_hashes: row.block.file_hashes,
      training_cohort: record.training_cohort, actual_epochs: record.actual_epochs,
      training_config: record.training_config, model_file_hashes: record.model_file_hashes,
      horizon_summaries: row.block.horizons
    }, null, 2)));
    const sources = node('p');
    addLink(sources, data.source_url, 'Pinned source index ↗'); sources.append(document.createTextNode(' · '));
    addLink(sources, data.dataset_bindings_url, 'Data & split bindings ↗'); container.append(sources);
    dialog.showModal();
  }

  function render() {
    body.replaceChildren();
    const pages = Math.ceil(filtered.length / pageSize);
    page = Math.max(0, Math.min(page, Math.max(0, pages - 1)));
    const checkpoints = new Set(filtered.map(r => r.identity)).size;
    count.textContent = `${filtered.length.toLocaleString('en-US')} of 3,969 blocks · ${checkpoints.toLocaleString('en-US')} checkpoint${checkpoints === 1 ? '' : 's'}`;
    if (!filtered.length) {
      const tr = node('tr'); const td = node('td', 'No blocks match these conditions. Try another task or PDE, or reset the filters.'); td.colSpan = 6; tr.append(td); body.append(tr);
    }
    for (const row of filtered.slice(page * pageSize, (page + 1) * pageSize)) {
      const tr = node('tr');
      const pde = node('td', name('pde', row.pde)); pde.append(node('small', taskName(row.task)));
      const method = node('td'); addLink(method, `../methods/#${row.method}`, name('method', row.method));
      method.append(node('small', row.method === 'pino' ? 'Extra training physics' : row.method === 'cno' ? 'Inspired implementation' : 'PDE-OBS adaptation'));
      const view = node('td', `${name('view', row.train_view)} →`); view.append(node('div', name('view', row.test_view)));
      view.append(node('small', row.train_view === row.test_view ? 'Matched observation' : 'Observation shift'));
      const score = node('td', `${number(row.mean)} ± ${number(row.std)}`); score.append(node('small', 'n = 200 · lower is better'));
      const budget = node('td', `${row.actual_epochs.toLocaleString('en-US')} epochs`); budget.append(node('small', row.training_cohort), node('small', 'Compute cost unavailable'));
      const proof = node('td'); const button = node('button', 'Inspect evidence ↗', 'text-button');
      button.type = 'button'; button.addEventListener('click', () => evidence(row, button));
      proof.append(button, node('small', 'Artifacts-checked · source audit'));
      tr.append(pde, method, view, score, budget, proof); body.append(tr);
    }
    pageStatus.textContent = pages ? `Page ${page + 1} of ${pages} · ${pageSize} rows per page` : 'No matching blocks';
    previous.disabled = page <= 0; next.disabled = page + 1 >= pages;
    document.querySelector('#export-csv').disabled = !filtered.length;
    document.querySelector('#export-json').disabled = !filtered.length;
  }

  function filter(sync = true) {
    if (!data) return;
    const values = Object.fromEntries(keys.map(key => [key, form.elements.namedItem(key).value]));
    filtered = rows.filter(r => (!values.task || r.task === values.task) && (!values.pde || r.pde === values.pde) &&
      (!values.method || r.method === values.method) && (!values.train || r.train_view === values.train) &&
      (!values.test || r.test_view === values.test) && (!values.match ||
        (values.match === 'matched' ? r.train_view === r.test_view : r.train_view !== r.test_view)));
    if (sort.value === 'error') filtered.sort((a, b) => a.mean - b.mean || a.key.localeCompare(b.key));
    page = 0; render(); if (sync) syncURL();
  }

  function exportRows() {
    return filtered.map(r => ({snapshot: data.id, source_commit: data.source_commit,
      source_url: data.source_url, source_sha256: data.source_sha256,
      evaluator_version: data.evaluator_version, dataset_bindings_url: data.dataset_bindings_url,
      identity: r.identity, task: r.task, pde: r.pde, method: r.method,
      train_view: r.train_view, test_view: r.test_view, mean: r.mean, std: r.std, n: r.n, ddof: 1,
      actual_epochs: r.actual_epochs, training_cohort: r.training_cohort,
      compute_cost: null, compute_cost_note: data.compute_cost_note,
      verification: data.verification, verification_scope: data.verification_scope,
      checkpoint_sha256: data.records[r.identity].checkpoint_released_sha256,
      contract_sha256: r.block.contract_sha256, score_sha256: r.block.file_hashes['score.json'],
      prediction_sha256: r.block.file_hashes['predictions.h5']}));
  }

  function download(format) {
    const exported = exportRows();
    if (!exported.length) return;
    let content;
    if (format === 'json') {
      content = JSON.stringify({snapshot: data.id, metric: data.metric, filters: Object.fromEntries(new FormData(form)), rows: exported}, null, 2);
    } else {
      const headers = Object.keys(exported[0]);
      const quote = value => '"' + String(value ?? '').replaceAll('"', '""') + '"';
      content = [headers.map(quote).join(','), ...exported.map(r => headers.map(k => quote(r[k])).join(','))].join('\r\n') + '\r\n';
    }
    const url = URL.createObjectURL(new Blob([content], {type: format === 'json' ? 'application/json' : 'text/csv;charset=utf-8'}));
    const anchor = node('a'); anchor.href = url; anchor.download = `${data.id}-filtered.${format}`;
    document.body.append(anchor); anchor.click(); anchor.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  tabs.forEach((tab, i) => {
    tab.addEventListener('click', () => chooseCollection(tab.dataset.collection));
    tab.addEventListener('keydown', event => {
      let target;
      if (event.key === 'ArrowRight') target = (i + 1) % tabs.length;
      if (event.key === 'ArrowLeft') target = (i + tabs.length - 1) % tabs.length;
      if (event.key === 'Home') target = 0;
      if (event.key === 'End') target = tabs.length - 1;
      if (target !== undefined) { event.preventDefault(); tabs[target].focus(); chooseCollection(tabs[target].dataset.collection); }
    });
  });
  form.addEventListener('submit', event => event.preventDefault());
  form.addEventListener('change', () => filter());
  form.addEventListener('reset', () => { sort.value = 'identity'; setTimeout(() => filter(), 0); });
  sort.addEventListener('change', () => filter());
  previous.addEventListener('click', () => { page--; render(); });
  next.addEventListener('click', () => { page++; render(); });
  document.querySelector('#export-csv').addEventListener('click', () => download('csv'));
  document.querySelector('#export-json').addEventListener('click', () => download('json'));
  document.querySelector('#close-evidence').addEventListener('click', () => dialog.close());
  dialog.addEventListener('close', () => opener?.focus());
  window.addEventListener('popstate', readURL);
  readURL();
  Promise.all([
    fetch('../assets/snapshots/paper-20260925-v1.json').then(r => { if (!r.ok) throw Error('snapshot'); return r.json(); }),
    fetch('../assets/platform-data.json').then(r => { if (!r.ok) throw Error('metadata'); return r.json(); })
  ]).then(([snapshot, platform]) => {
    data = snapshot; metadata = platform;
    for (const record of Object.values(data.records)) {
      for (const [test_view, block] of Object.entries(record.blocks)) {
        rows.push({identity: record.identity, task: record.task, pde: record.pde, method: record.method,
          train_view: record.train_view, test_view, actual_epochs: record.actual_epochs,
          training_cohort: record.training_cohort, mean: block.joint.mean, std: block.joint.std,
          n: block.joint.n, block, key: `${record.identity}/${test_view}`});
      }
    }
    rows.sort((a, b) => a.key.localeCompare(b.key));
    filter(false);
  }).catch(() => {
    count.textContent = 'The snapshot could not be loaded. Reload this page, or use the complete JSON download below.';
    count.setAttribute('role', 'alert');
  });
})();
