// Exercise the shipped explorer in a minimal DOM adapter, with real snapshot data.
// No browser, network, model execution or external dependency is used.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(__dirname, '..');
const metadata = JSON.parse(fs.readFileSync(path.join(root, 'data/platform.json')));
const snapshot = JSON.parse(fs.readFileSync(path.join(root, 'docs/assets/snapshots/paper-20260925-v1.json')));
class Element {
  constructor(tag = '') { this.tag = tag; this.children = []; this.listeners = {}; this.attributes = {}; this.dataset = {}; this.value = ''; }
  append(...items) { this.children.push(...items); }
  replaceChildren(...items) { this.children = items; }
  addEventListener(type, fn) { this.listeners[type] = fn; }
  setAttribute(key, value) { this.attributes[key] = value; }
  getAttribute(key) { return this.attributes[key]; }
  click() { this.listeners.click?.({}); }
  remove() {}
  focus() {}
  showModal() { this.open = true; }
  close() { this.open = false; this.listeners.close?.(); }
}
const ids = new Map();
const get = id => { if (!ids.has(id)) ids.set(id, new Element()); return ids.get(id); };
const controls = Object.fromEntries(['task','pde','method','train','test','match'].map(k => [k, new Element()]));
Object.values(controls).forEach(c => { c.options = [{value:''}]; });
get('#result-filters').elements = {namedItem: key => controls[key]};
const tabs = ['paper','community','study'].map(key => {
  const tab = new Element(); tab.dataset.collection = key; tab.attributes['aria-controls'] = key+'-panel'; return tab;
});
get('#result-explorer').querySelectorAll = () => tabs;
let exported;
class CaptureURL extends URL {
  static createObjectURL(blob) { exported = blob; return 'blob:test'; }
  static revokeObjectURL() {}
}
const context = vm.createContext({
  document: {querySelector: get, getElementById: id => get('#'+id), createElement: tag => new Element(tag),
    createTextNode: text => ({textContent:text}), body: new Element()},
  window: {addEventListener() {}}, location: {href:'https://example.test/project/results/',search:''},
  history: {replaceState() {}}, URL: CaptureURL, URLSearchParams, Blob,
  FormData: class { *[Symbol.iterator]() { for (const [key, c] of Object.entries(controls)) yield [key,c.value]; } },
  setTimeout: fn => fn(),
  fetch: async url => ({ok:true,json:async () => url.includes('snapshots/') ? snapshot : metadata})
});
vm.runInContext(fs.readFileSync(path.join(root,'docs/assets/results.js'),'utf8'), context);
(async () => {
  await new Promise(resolve => setImmediate(resolve));
  assert.match(get('#result-count').textContent, /^3,969 of 3,969/);
  assert.equal(get('#result-rows').children.length, 30);
  get('#next-page').click();
  assert.match(get('#page-status').textContent, /^Page 2 of 133/);
  controls.pde.value = 'darcy'; controls.method.value = 'fno'; controls.match.value = 'matched';
  get('#result-filters').listeners.change();
  assert.match(get('#result-count').textContent, /^9 of 3,969 blocks · 9 checkpoints/);
  get('#export-json').click();
  const result = JSON.parse(await exported.text());
  assert.equal(result.rows.length, 9);
  for (const row of result.rows) {
    const original = snapshot.records[row.identity];
    assert.equal(row.train_view, row.test_view);
    assert.equal(row.mean, original.blocks[row.test_view].joint.mean);
    assert.equal(row.std, original.blocks[row.test_view].joint.std);
    assert.equal(row.checkpoint_sha256, original.checkpoint_released_sha256);
    assert.equal(row.compute_cost, null);
  }
  get('#export-csv').click();
  const csv = await exported.text();
  assert.equal(csv.trim().split('\r\n').length, 10);
  assert.ok(csv.includes('"source_sha256"'));
  assert.ok(csv.includes(String(result.rows[0].mean)));
  controls.task.value = 'rollout'; get('#result-filters').listeners.change();
  assert.match(get('#result-count').textContent, /^0 of 3,969/);
  assert.equal(get('#export-json').disabled, true);
  assert.match(get('#result-rows').children[0].children[0].textContent, /No blocks match/);
  tabs[1].click();
  assert.equal(get('#community-panel').hidden, false);
  assert.equal(get('#paper-panel').hidden, true);
  console.log('Explorer checks passed: pagination, filters, empty states, exact CSV/JSON exports, provenance and collections.');
})().catch(error => { console.error(error); process.exitCode = 1; });
