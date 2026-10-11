import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {SourceTextModule, SyntheticModule, createContext} from 'node:vm';
import * as geometry from '../review_ui/geometry.js';

// Exercise the real event handlers with a small DOM and canvas adapter; no browser automation.
class Element {
  constructor() {this.value = ''; this.children = []; this.checked = false; this.open = false;}
  append(...items) {this.children.push(...items);}
  add(item) {this.children.push(item); if (!this.value) this.value = item.value;}
  replaceChildren(...items) {this.children = items; if (items[0]?.value) this.value = items[0].value;}
  showModal() {this.open = true;}
  close() {this.open = false;}
  scrollIntoView() {this.scrolled = true;}
  setAttribute() {}
}

class CanvasAdapter {
  setReview() {}
  render() {}
}

async function setup(status = 'candidate') {
  const elements = new Map(), requests = [], storage = new Map();
  const get = id => {if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id);};
  const prediction = {id: 'a'.repeat(32), model_name: 'Test YOLO', elapsed_seconds: .1, confidence_threshold: .15,
    boxes: [{id: 'box-1', label: 'bird', x: .2, y: .2, width: .4, height: .4, flags: [], confidence: .9}]};
  const document = {getElementById: get, createElement: () => new Element(), createTextNode: value => value,
    querySelectorAll: () => [], addEventListener() {}, activeElement: null};
  const fetch = async (path, options = {}) => {
    const payload = options.body ? JSON.parse(options.body) : null; requests.push({path, payload});
    if (path === '/api/queue') return {ok: true, json: async () => []};
    if (path === '/api/models') return {ok: true, json: async () => ({models: [{id: 'yolo26n', name: 'Test YOLO'}], confidence: .15})};
    if (path.startsWith('/api/predict/')) return {ok: true, json: async () => structuredClone(prediction)};
    return {ok: true, json: async () => ({review: {...payload, revision: 'saved', status: 'draft'}, reviewed_digest: 'missing'})};
  };
  const context = createContext({document, fetch, structuredClone, console,
    location: {href: 'http://localhost/'}, URL, history: {replaceState() {}},
    window: {addEventListener() {}}, setTimeout: () => 0, clearTimeout() {}, setInterval() {},
    Option: class {constructor(text, value) {this.text = text; this.value = value;}},
    localStorage: {getItem: key => storage.get(key), setItem: (key, value) => storage.set(key, value), removeItem: key => storage.delete(key)}});
  const source = await readFile(new URL('../review_ui/app.js', import.meta.url), 'utf8');
  const app = new SourceTextModule(source + '\nexport {state, runModel, useModelBoxes, applyModelBoxes, undo, save, renderInspector};', {context});
  const canvasModule = new SyntheticModule(['BoxCanvas'], function() {this.setExport('BoxCanvas', CanvasAdapter);}, {context});
  const geometryModule = new SyntheticModule(Object.keys(geometry), function() {for (const [key, value] of Object.entries(geometry)) this.setExport(key, value);}, {context});
  await app.link(name => name === './canvas.js' ? canvasModule : geometryModule);
  await app.evaluate(); await new Promise(resolve => setImmediate(resolve));
  const api = app.namespace;
  api.state.item = {id: 'image', width: 100, height: 80, boxes: [], source_version: 'source', reviewed_digest: 'missing'};
  api.state.review = {boxes: [{id: 'human', label: 'bird', x: .1, y: .1, width: .2, height: .2, flags: []}],
    notes: 'Keep notes', flags: ['blur'], negative: false, model_run_id: null, revision: 'initial', status};
  api.state.canvasReady = true; api.state.selected = 'human';
  get('adjust-mode').value = 'move'; get('show-model').checked = true;
  api.renderInspector();
  return {api, get, prediction, requests};
}

test('preview does not edit the review and the footer requires applying boxes first', async () => {
  const {api, get, requests} = await setup('needs_work');
  const before = JSON.stringify(api.state.review);
  await api.runModel();
  assert.equal(JSON.stringify(api.state.review), before);
  assert.equal(get('approve').textContent, 'Use model boxes');
  assert.equal(get('canvas').scrolled, true);
  assert.equal(requests.some(request => request.path.startsWith('/api/review/')), false);
});

test('replacement is confirmed, undoable, and saved only as a draft with run provenance', async () => {
  const {api, get, prediction, requests} = await setup();
  await api.runModel(); api.useModelBoxes();
  assert.equal(get('model-replace-dialog').open, true);
  assert.equal(api.state.review.boxes[0].id, 'human');
  get('confirm-model').onclick();
  assert.equal(get('model-replace-dialog').open, false);
  assert.equal(api.state.review.model_run_id, prediction.id);
  assert.equal(api.state.review.boxes[0].confidence, undefined);
  assert.equal(api.state.review.notes, 'Keep notes');
  assert.equal(get('approve').textContent, 'Approve & next');
  api.undo(); assert.equal(api.state.review.boxes[0].id, 'human'); assert.equal(api.state.review.model_run_id, null);
  api.undo(true); assert.equal(api.state.review.model_run_id, prediction.id);
  await api.save();
  const saved = requests.find(request => request.path.startsWith('/api/review/')).payload;
  assert.equal(saved.action, 'draft'); assert.equal(saved.model_run_id, prediction.id);
});

test('approved review allows preview but model boxes cannot replace its labels', async () => {
  const {api, get} = await setup('approved');
  await api.runModel(); api.useModelBoxes(); api.applyModelBoxes();
  assert.equal(api.state.review.boxes[0].id, 'human');
  assert.equal(get('use-model').disabled, true);
  assert.equal(get('approve').textContent, 'Reopen review');
});

test('empty prediction is not automatically a reviewed negative', async () => {
  const {api, prediction} = await setup(); prediction.boxes = [];
  await api.runModel(); api.applyModelBoxes();
  assert.equal(api.state.review.boxes.length, 0);
  assert.equal(api.state.review.negative, false);
  assert.equal(api.state.review.status, 'candidate');
});
