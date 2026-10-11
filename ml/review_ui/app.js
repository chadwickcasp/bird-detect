import {BoxCanvas} from './canvas.js';
import {clamp, moveBox, nudgeBox} from './geometry.js';

const $ = id => document.getElementById(id);
const state = {
  queue: [], queueLimit: 80, item: null, review: null, selected: null,
  undo: [], redo: [], dirty: false, loading: false, editing: 0,
  saving: null, timer: null, canvasReady: false, locked: false, recovery: null, requested: null,
  models: [], predicting: false, prediction: null,
};
const STATUS = {candidate: 'Candidate', draft: 'Draft', needs_work: 'Needs work', approved: 'Approved'};
const IMAGE_FLAGS = {blur: 'Blur', tiny: 'Tiny birds', noise: 'Noise', backlit: 'Backlit', flock: 'Flock', uncertain: 'Uncertain'};
const BOX_FLAGS = {occluded: 'Occluded', truncated: 'Truncated', groupof: 'Group', uncertain: 'Uncertain'};
const coordinates = {x: 'box-x', y: 'box-y', width: 'box-width', height: 'box-height'};
const canvas = new BoxCanvas($('canvas'), {
  select: id => {state.selected = id; renderInspector();},
  change: (boxes, selected) => commit(review => {review.boxes = boxes; review.negative = false;}, selected),
  zoom: zoom => $('zoom').textContent = `${zoom}%`,
});

function notice(message, retry = true) {
  $('notice').hidden = !message; $('notice-text').textContent = message;
  $('retry').hidden = !retry; $('reload-item').hidden = !state.item;
  $('recover-draft').hidden = !state.recovery;
  $('recover-draft').disabled = !editable();
  $('export-draft').hidden = !state.recovery && !state.dirty;
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const record = await response.json();
  if (!response.ok) {
    const error = new Error(record.error || 'The request could not complete.');
    error.status = response.status; throw error;
  }
  return record;
}

function localKey() { return state.item ? `bird-review-draft:${state.item.id}` : null; }

function stashDraft() {
  if (!state.item || !state.review) return;
  try {
    localStorage.setItem(localKey(), JSON.stringify({review: state.review,
      source_version: state.item.source_version, reviewed_digest: state.item.reviewed_digest,
      saved_at: new Date().toISOString()}));
  } catch { $('save-status').textContent = 'Device storage full; keep this page open'; }
}

function clearStash() { try {localStorage.removeItem(localKey());} catch {} }

function snapshot() {
  return {boxes: structuredClone(state.review.boxes), notes: state.review.notes,
    flags: [...state.review.flags], negative: state.review.negative, selected: state.selected, model_run_id: state.review.model_run_id || null};
}

function editable() { return state.review && state.review.status !== 'approved' && state.canvasReady && !state.loading && !state.locked; }

function commit(change, selected = state.selected) {
  if (!editable()) return;
  state.undo.push(snapshot()); if (state.undo.length > 100) state.undo.shift();
  state.redo = []; change(state.review); state.selected = selected;
  markDirty(); renderInspector();
}

function markDirty() {
  state.dirty = true; state.editing += 1; stashDraft();
  $('save-status').textContent = 'Saving draft…';
  clearTimeout(state.timer); state.timer = setTimeout(() => save('draft'), 700);
}

function updateCanvas() {
  canvas.predictions = state.prediction && $('show-model').checked ? state.prediction.boxes : [];
  canvas.setReview(state.review?.boxes || [], state.selected, state.item?.boxes || [], editable());
}

function selectedBox() { return state.review?.boxes.find(box => box.id === state.selected); }

function pendingModelBoxes() { return !!state.prediction && state.prediction.id !== state.review?.model_run_id; }

function renderInspector() {
  if (!state.review || !state.item) return;
  const box = selectedBox(), allowed = editable();
  const select = $('box-select'); select.replaceChildren();
  if (!state.review.boxes.length) select.add(new Option('No boxes', ''));
  state.review.boxes.forEach((box, index) => select.add(new Option(`Box ${index + 1} · bird`, box.id)));
  if (state.selected) select.value = state.selected;
  for (const [key, id] of Object.entries(coordinates)) {
    const input = $(id), size = ['x', 'width'].includes(key) ? state.item.width : state.item.height;
    input.value = box ? Math.round(box[key] * size) : '';
    input.max = size; input.disabled = !box || !allowed;
  }
  const operation = $('adjust-mode').value;
  document.querySelectorAll('[data-nudge]').forEach(button => {
    const horizontal = ['left', 'right'].includes(button.dataset.nudge);
    button.disabled = !box || !allowed || (operation !== 'move' && horizontal !== ['left', 'right'].includes(operation));
  });
  $('focus').disabled = !box || !state.canvasReady;
  $('delete').disabled = !box || !allowed; $('duplicate').disabled = !box || !allowed;
  $('undo').disabled = !state.undo.length || !allowed; $('redo').disabled = !state.redo.length || !allowed;
  $('notes').disabled = !allowed;
  if (document.activeElement !== $('notes')) $('notes').value = state.review.notes;
  $('negative').checked = state.review.negative; $('negative').disabled = !allowed;
  $('import-labels').disabled = !allowed;
  $('recover-draft').disabled = !allowed;
  $('run-model').disabled = !state.canvasReady || state.loading || state.predicting || !!state.saving || !state.models.length;
  $('model-select').disabled = state.predicting || !state.models.length;
  $('model-confidence').disabled = state.predicting;
  $('use-model').disabled = !allowed || !state.prediction;
  $('dismiss-model').disabled = state.predicting;
  $('queue-open').disabled = state.predicting;
  document.querySelectorAll('[data-mode]').forEach(button => button.disabled = !state.canvasReady || (button.dataset.mode === 'add' && !allowed));
  document.querySelectorAll('#image-flags input').forEach(input => {input.checked = state.review.flags.includes(input.value); input.disabled = !allowed;});
  document.querySelectorAll('#box-flags input').forEach(input => {input.checked = box?.flags.includes(input.value) || false; input.disabled = !box || !allowed;});
  $('review-status').textContent = STATUS[state.review.status] || 'Candidate';
  $('approve').textContent = state.review.status === 'approved' ? 'Reopen review' : pendingModelBoxes() ? 'Use model boxes' : 'Approve & next';
  $('approve').disabled = !state.canvasReady || state.loading || !!state.saving || state.predicting;
  $('needs-work').disabled = !allowed || !!state.saving;
  $('skip').disabled = state.loading || !!state.saving || state.predicting;
  updateCanvas();
}

async function save(action = 'draft') {
  if (!state.item || !state.review) return false;
  clearTimeout(state.timer);
  if (state.saving) {await state.saving; if (action === 'draft' && !state.dirty) return true;}
  if (action === 'draft' && (!state.dirty || state.review.status === 'approved')) return true;
  const generation = state.editing, identifier = state.item.id;
  state.locked = action !== 'draft';
  const payload = {...structuredClone(state.review), action,
    source_version: state.item.source_version, reviewed_digest: state.item.reviewed_digest};
  $('save-status').textContent = action === 'approve' ? 'Approving…' : 'Saving draft…';
  const operation = async () => {
    try {
      const result = await api(`/api/review/${identifier}`, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
      state.review.revision = result.review.revision; state.item.reviewed_digest = result.reviewed_digest;
      state.review.status = result.review.status;
      state.dirty = state.editing !== generation;
      if (state.dirty) {stashDraft(); state.timer = setTimeout(() => save('draft'), 700);}
      else clearStash();
      $('save-status').textContent = action === 'approve' ? 'Approved' : 'Draft saved';
      notice('');
      const row = state.queue.find(row => row.id === identifier);
      if (row) {row.status = result.review.status; row.box_count = result.review.boxes.length; row.available = true;}
      return true;
    } catch (error) {
      stashDraft(); $('save-status').textContent = 'Not saved'; notice(error.message, error.status !== 409);
      return false;
    }
  };
  state.saving = operation(); renderInspector();
  const result = await state.saving;
  state.saving = null; state.locked = false; renderInspector(); renderQueue();
  return result;
}

async function openItem(identifier, force = false) {
  if (state.loading || state.predicting) return;
  if (!force && state.dirty && !await save('draft')) return;
  state.requested = identifier;
  state.loading = true; state.canvasReady = false; clearTimeout(state.timer);
  $('canvas-message').hidden = false; $('canvas-message').textContent = 'Loading photo and candidate labels…';
  $('save-status').textContent = 'Loading photo…'; notice(''); renderInspector();
  try {
    const item = await api(`/api/item/${identifier}`);
    state.item = item; state.review = structuredClone(item.review); state.dirty = false; state.recovery = null;
    state.prediction = null; $('model-preview').hidden = true;
    $('model-status').textContent = state.review.model_run_id ? 'This review started from a saved model run. Run again to compare predictions.' : 'Generate starting boxes for this frame, then review them.';
    state.undo = []; state.redo = []; state.selected = state.review.boxes[0]?.id || null;
    $('filename').textContent = item.name; $('filename').title = item.path;
    $('source-info').textContent = `${item.width} × ${item.height} pixels · ${item.source} · ${item.attribution}${item.source === 'Open Images' ? ' · Camera and location unverified' : ''}`;
    $('source-link').hidden = !item.source_url; $('source-link').href = item.source_url || '#';
    canvas.original = false; $('compare').checked = false;
    await canvas.load(item.image_url);
    state.canvasReady = true; $('canvas-message').hidden = true;
    $('save-status').textContent = state.review.status === 'approved' ? 'Approved' : state.review.status === 'candidate' ? 'Not reviewed' : 'Draft saved';
    try {
      const saved = JSON.parse(localStorage.getItem(localKey()) || 'null');
      if (!force && saved && saved.source_version === item.source_version && saved.review.revision === item.review.revision && item.review.status !== 'approved') {
        state.review = saved.review; state.selected = state.review.boxes[0]?.id || null; markDirty();
        notice('Recovered your unsaved draft from this device.', false);
      } else if (saved) {state.recovery = saved; notice('A device draft is available. Recover it into the current review, or save a copy before replacing it.', false);}
    } catch {}
    if (item.proposal_errors.length) notice('Some source labels are invalid. Check all boxes before approving.', false);
    if (item.review.source_changed) notice('The source proposal changed since the last draft. Use Compare source boxes to check the new proposal.', false);
  } catch (error) {
    $('canvas-message').textContent = 'Photo unavailable. Use Retry or choose another image in Queue.';
    $('save-status').textContent = 'Could not load'; notice(error.message);
  } finally {state.loading = false; renderInspector();}
}

function filteredQueue() {
  const filter = $('queue-filter').value, search = $('queue-search').value.toLowerCase();
  return state.queue.filter(item => (filter === 'all' || (filter === 'pending' ? item.status !== 'approved' : item.status === filter))
    && `${item.name} ${item.path}`.toLowerCase().includes(search));
}

function renderQueue() {
  const queue = filteredQueue(), list = $('queue-list'); list.replaceChildren();
  const counts = state.queue.reduce((counts, item) => ({...counts, [item.status]: (counts[item.status] || 0) + 1}), {});
  $('queue-count').textContent = `${queue.length} in this view · ${counts.approved || 0} approved · New arrivals refresh automatically`;
  for (const item of queue.slice(0, state.queueLimit)) {
    const button = document.createElement('button'); button.className = 'queue-row';
    const name = document.createElement('span'); name.className = 'title'; name.textContent = item.name;
    const meta = document.createElement('span'); meta.className = 'meta'; meta.textContent = `${item.source} · ${item.has_proposal ? 'Candidate boxes' : 'No proposals yet'}${!item.available ? ' · Downloads when opened' : ''}`;
    name.append(meta); const status = document.createElement('span'); status.className = 'state'; status.textContent = STATUS[item.status];
    button.append(name, status); button.onclick = async () => {await openItem(item.id); $('queue-dialog').close();}; list.append(button);
  }
  if (!queue.length) {const empty = document.createElement('p'); empty.className = 'small'; empty.textContent = 'No images in this view. Add photos or change the filter.'; list.append(empty);}
  $('queue-more').hidden = queue.length <= state.queueLimit;
}

async function refreshQueue() {
  try {state.queue = await api('/api/queue'); renderQueue(); return true;}
  catch (error) {notice(error.message); return false;}
}

async function advance() {
  if (state.predicting || (state.dirty && !await save('draft'))) return;
  const current = state.queue.findIndex(item => item.id === state.item?.id);
  const choices = [...state.queue.slice(current + 1), ...state.queue.slice(0, current + 1)]
    .filter(item => item.status !== 'approved' && item.id !== state.item?.id);
  if (choices.length) await openItem(choices[0].id);
  else {await refreshQueue(); $('queue-dialog').showModal(); notice('You reached the end of the current queue. New candidates will appear automatically.', false);}
}

function undo(redo = false) {
  if (!editable()) return;
  const source = redo ? state.redo : state.undo, target = redo ? state.undo : state.redo;
  const previous = source.pop(); if (!previous) return;
  target.push(snapshot()); Object.assign(state.review, {boxes: previous.boxes, notes: previous.notes, flags: previous.flags, negative: previous.negative});
  state.review.model_run_id = previous.model_run_id || null;
  state.selected = previous.selected; markDirty(); renderInspector();
}

function nudge(direction, step = Number($('nudge-step').value)) {
  const box = selectedBox(); if (!box || !editable()) return;
  const dx = direction === 'left' ? -step : direction === 'right' ? step : 0;
  const dy = direction === 'up' ? -step : direction === 'down' ? step : 0;
  const next = nudgeBox(box, $('adjust-mode').value, dx, dy, state.item);
  commit(review => review.boxes = review.boxes.map(box => box.id === next.id ? next : box));
}

function createFlags(container, values, boxLevel) {
  for (const [value, title] of Object.entries(values)) {
    const label = document.createElement('label'); label.className = 'check';
    const input = document.createElement('input'); input.type = 'checkbox'; input.value = value;
    input.onchange = () => commit(review => {
      const target = boxLevel ? review.boxes.find(box => box.id === state.selected) : review;
      if (target) target.flags = input.checked ? [...new Set([...target.flags, value])] : target.flags.filter(flag => flag !== value);
    });
    label.append(input, document.createTextNode(title)); $(container).append(label);
  }
}

async function showHistory() {
  if (!state.item) return;
  try {
    const history = await api(`/api/history/${state.item.id}`), list = $('history-list'); list.replaceChildren();
    for (const version of history) {
      const row = document.createElement('section'); row.className = 'history-row';
      const description = document.createElement('p'); description.textContent = `${new Date(version.updated_at).toLocaleString()} · ${STATUS[version.status]} · ${version.boxes.length} boxes`;
      const notes = document.createElement('p'); notes.className = 'small'; notes.textContent = version.notes;
      const button = document.createElement('button'); button.textContent = 'Restore as draft'; button.disabled = !editable();
      button.onclick = () => {commit(review => Object.assign(review, {boxes: structuredClone(version.boxes), notes: version.notes, flags: [...version.flags], negative: version.negative, model_run_id: version.model_run_id || null}), version.boxes[0]?.id || null); $('history-dialog').close();};
      row.append(description, notes, button); list.append(row);
    }
    if (!history.length) list.textContent = 'No saved revisions yet.';
    $('history-dialog').showModal();
  } catch (error) {notice(error.message);}
}

async function uploadImages(files) {
  if (state.dirty && !await save('draft')) return;
  let last;
  for (const file of files) {
    if (file.size > 30 * 1024 * 1024) {notice(`${file.name} exceeds the 30 MB limit.`); continue;}
    try {
      $('queue-count').textContent = `Adding ${file.name}…`;
      const content = await new Promise((resolve, reject) => {const reader = new FileReader(); reader.onload = () => resolve(reader.result.split(',')[1]); reader.onerror = reject; reader.readAsDataURL(file);});
      last = await api('/api/upload', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({name: file.name, content})});
    } catch (error) {notice(`Could not add ${file.name}: ${error.message}`);}
  }
  await refreshQueue(); if (last) {await openItem(last.id); $('queue-dialog').close();}
}

async function importLabels(file) {
  if (!file || !editable()) return;
  try {
    const text = await file.text(), boxes = [];
    for (const [index, line] of text.split(/\r?\n/).entries()) {
      if (!line.trim()) continue;
      const parts = line.trim().split(/\s+/).map(Number);
      if (parts.length !== 5 || parts[0] !== 0 || parts.some(value => !Number.isFinite(value))) throw new Error(`Invalid bird label on line ${index + 1}.`);
      const [, cx, cy, width, height] = parts, x = cx - width / 2, y = cy - height / 2;
      if (width <= 0 || height <= 0 || x < -.0001 || y < -.0001 || x + width > 1.0001 || y + height > 1.0001) throw new Error(`Box outside the image on line ${index + 1}.`);
      boxes.push({id: `import-${Date.now()}-${index}`, label: 'bird', x: Math.max(0, x), y: Math.max(0, y), width: Math.min(width, 1 - Math.max(0, x)), height: Math.min(height, 1 - Math.max(0, y)), flags: []});
    }
    commit(review => {review.boxes = boxes; review.negative = false; review.model_run_id = null;}, boxes[0]?.id || null);
    notice('Imported labels into this draft. Review every box before approving.', false);
  } catch (error) {notice(error.message, false);}
}

createFlags('image-flags', IMAGE_FLAGS, false); createFlags('box-flags', BOX_FLAGS, true);
$('run-model').onclick = runModel;
$('show-model').onchange = updateCanvas;
$('dismiss-model').onclick = () => {state.prediction = null; $('model-preview').hidden = true; $('model-status').textContent = 'Preview dismissed. Your review boxes are unchanged.'; renderInspector();};
$('use-model').onclick = useModelBoxes;
$('confirm-model').onclick = () => { $('model-replace-dialog').close(); applyModelBoxes(); };

function useModelBoxes() {
  if (!editable() || !state.prediction) return;
  if (state.review.boxes.length) {
    $('model-replace-message').textContent = `Replace the ${state.review.boxes.length} draft boxes with ${state.prediction.boxes.length} model boxes? Notes and image flags stay. Undo restores the previous boxes.`;
    $('model-replace-dialog').showModal();
    return;
  }
  applyModelBoxes();
}

function applyModelBoxes() {
  if (!editable() || !state.prediction) return;
  const prediction = state.prediction;
  const boxes = prediction.boxes.map(({confidence, ...box}) => ({...structuredClone(box), id: `model-${prediction.id}-${box.id}`}));
  commit(review => {review.boxes = boxes; review.negative = false; review.model_run_id = prediction.id;}, boxes[0]?.id || null);
  $('show-model').checked = false; updateCanvas();
  $('model-status').textContent = `Using ${boxes.length} model boxes in this draft. Refine them before approving.${!boxes.length ? ' No detections is not a confirmed negative.' : ''}`;
}
$('queue-open').onclick = () => {$('queue-dialog').showModal(); refreshQueue();};
document.querySelectorAll('[data-close]').forEach(button => button.onclick = () => $(button.dataset.close).close());
$('queue-filter').onchange = $('queue-search').oninput = () => {state.queueLimit = 80; renderQueue();};
$('queue-more').onclick = () => {state.queueLimit += 80; renderQueue();};
$('refresh').onclick = refreshQueue;
$('upload').onclick = () => $('upload-input').click();
$('upload-input').onchange = event => {uploadImages([...event.target.files]); event.target.value = '';};
$('import-labels').onclick = () => $('label-input').click();
$('label-input').onchange = event => {importLabels(event.target.files[0]); event.target.value = '';};
$('retry').onclick = () => state.canvasReady ? save('draft') : state.requested ? openItem(state.requested, true) : start();
$('reload-item').onclick = () => {
  if (!state.item) return;
  if (state.dirty && !confirm('Reload the server version? Your unsaved draft remains in device storage.')) return;
  openItem(state.item.id, true);
};
$('recover-draft').onclick = () => {
  const saved = state.recovery;
  if (!saved || !editable()) return;
  commit(review => Object.assign(review, {boxes: structuredClone(saved.review.boxes), notes: saved.review.notes,
    flags: [...saved.review.flags], negative: saved.review.negative, model_run_id: saved.review.model_run_id || null}), saved.review.boxes[0]?.id || null);
  state.recovery = null; notice('Recovered into a draft. Check the current image and boxes before approving.', false);
};
$('export-draft').onclick = () => {
  const draft = state.recovery || {review: state.review, source_version: state.item?.source_version};
  const url = URL.createObjectURL(new Blob([JSON.stringify(draft, null, 2)], {type: 'application/json'}));
  const link = document.createElement('a'); link.href = url; link.download = `bird-device-draft-${state.item?.id || 'recovery'}.json`; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
};
$('fit').onclick = () => canvas.fit(); $('zoom').onclick = () => canvas.zoom(1.5);
$('focus').onclick = () => canvas.focus();
$('box-select').onchange = () => {state.selected = $('box-select').value || null; renderInspector();};
document.querySelectorAll('[data-mode]').forEach(button => button.onclick = () => {
  canvas.mode = button.dataset.mode;
  document.querySelectorAll('[data-mode]').forEach(button => button.classList.toggle('selected', button.dataset.mode === canvas.mode));
});
$('boxes-visible').onclick = () => {canvas.visible = !canvas.visible; canvas.render(); $('boxes-visible').textContent = canvas.visible ? 'Hide boxes' : 'Show boxes'; $('boxes-visible').setAttribute('aria-pressed', String(canvas.visible));};
$('compare').onchange = () => {canvas.original = $('compare').checked; canvas.render();};
$('delete').onclick = () => {const selected = state.selected; commit(review => review.boxes = review.boxes.filter(box => box.id !== selected), state.review.boxes.find(box => box.id !== selected)?.id || null);};
$('duplicate').onclick = () => {
  const box = selectedBox(); if (!box) return;
  const next = moveBox({...structuredClone(box), id: `copy-${Date.now()}-${Math.random().toString(16).slice(2)}`}, 10 / state.item.width, 10 / state.item.height);
  commit(review => review.boxes.push(next), next.id);
};
$('undo').onclick = () => undo(); $('redo').onclick = () => undo(true);
$('adjust-mode').onchange = renderInspector;
document.querySelectorAll('[data-nudge]').forEach(button => button.onclick = () => nudge(button.dataset.nudge));
for (const [key, id] of Object.entries(coordinates)) $(id).onchange = () => {
  const box = selectedBox(), value = Number($(id).value);
  if (!box || !Number.isFinite(value)) return;
  const size = ['x', 'width'].includes(key) ? state.item.width : state.item.height;
  commit(review => {
    const next = {...box, [key]: value / size};
    next.width = clamp(next.width, 1 / state.item.width, 1);
    next.height = clamp(next.height, 1 / state.item.height, 1);
    if (key === 'width') next.width = Math.min(next.width, 1 - box.x);
    if (key === 'height') next.height = Math.min(next.height, 1 - box.y);
    next.x = clamp(next.x, 0, 1 - next.width); next.y = clamp(next.y, 0, 1 - next.height);
    review.boxes = review.boxes.map(box => box.id === next.id ? next : box);
  });
};
$('notes').oninput = () => commit(review => review.notes = $('notes').value);
$('negative').onchange = () => {const checked = $('negative').checked; commit(review => {review.negative = checked; if (checked) review.boxes = [];}, checked ? null : state.selected);};
$('history-open').onclick = showHistory;
$('approve').onclick = async () => {
  if (state.review?.status === 'approved') {
    if (confirm('Reopen this image? Its previous approved labels will be archived and withdrawn from the reviewed export until you approve it again.')) {
      if (await save('reopen')) {state.undo = []; state.redo = []; renderInspector();}
    }
    return;
  }
  if (pendingModelBoxes()) {useModelBoxes(); return;}
  if (await save('approve')) {await refreshQueue(); await advance();}
};
$('needs-work').onclick = async () => {if (await save('needs_work')) await advance();};
$('skip').onclick = advance;
document.addEventListener('keydown', event => {
  if (['INPUT', 'TEXTAREA', 'SELECT'].includes(event.target.tagName) || document.querySelector('dialog[open]')) return;
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') {event.preventDefault(); undo(event.shiftKey);}
  if (event.key === 'Delete' || event.key === 'Backspace') {event.preventDefault(); if (!$('delete').disabled) $('delete').click();}
  const direction = {ArrowLeft: 'left', ArrowRight: 'right', ArrowUp: 'up', ArrowDown: 'down'}[event.key];
  if (direction && editable()) {event.preventDefault(); nudge(direction, event.shiftKey ? 5 : Number($('nudge-step').value));}
});
window.addEventListener('beforeunload', event => {if (state.dirty || state.saving) {stashDraft(); event.preventDefault(); event.returnValue = '';}});
document.addEventListener('visibilitychange', () => {if (document.hidden && state.dirty) {stashDraft(); save('draft');}});

async function start() {
  // The server exchanges the access link for an HttpOnly cookie. Keep the token out of copied URLs.
  if (new URL(location.href).searchParams.has('access')) history.replaceState(null, '', '/');
  if (!await refreshQueue()) { $('canvas-message').textContent = 'Open the access link to unlock this editor.'; return; }
  await loadModels();
  const candidate = state.queue.find(item => item.status !== 'approved') || state.queue[0];
  if (candidate) await openItem(candidate.id);
  else { $('canvas-message').textContent = 'No candidates yet. Open Queue to add images.'; $('save-status').textContent = 'Queue ready'; $('queue-dialog').showModal(); }
}

async function loadModels() {
  try {
    const config = await api('/api/models'); state.models = config.models;
    $('model-select').replaceChildren(...config.models.map(model => new Option(model.name, model.id)));
    if (!config.models.length) {$('model-select').add(new Option('No local checkpoint available', '')); $('model-status').textContent = 'No local model checkpoint is available on this computer.';}
    $('model-select').disabled = !config.models.length;
    $('model-confidence').value = config.confidence;
  } catch (error) {$('model-status').textContent = error.message;}
}

async function runModel() {
  if (!state.item || !state.canvasReady || state.predicting) return;
  const confidence = Number($('model-confidence').value);
  if (!Number.isFinite(confidence) || confidence < .01 || confidence > .99) { $('model-status').textContent = 'Choose a minimum score between 0.01 and 0.99.'; return; }
  if (!await save('draft')) return;
  state.predicting = true; state.locked = true; state.prediction = null; $('model-preview').hidden = true; renderInspector();
  $('run-model').textContent = 'Running…'; $('model-status').textContent = 'Running on the computer. The first run may take longer while the model loads.';
  try {
    const prediction = await api(`/api/predict/${state.item.id}`, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({model: $('model-select').value, confidence, source_version: state.item.source_version})});
    state.prediction = prediction; $('show-model').checked = true; $('model-preview').hidden = false;
    $('model-status').textContent = `${prediction.model_name}: ${prediction.boxes.length} bird boxes · ${prediction.elapsed_seconds.toFixed(1)} s · minimum score ${prediction.confidence_threshold}. ${prediction.boxes.length ? 'Preview only; use them to start a draft.' : 'No birds detected; check the image yourself before confirming a negative.'}`;
    $('canvas').scrollIntoView({behavior: 'smooth', block: 'start'});
  } catch (error) {$('model-status').textContent = error.message;}
  finally {state.predicting = false; state.locked = false; $('run-model').textContent = 'Run YOLO'; renderInspector();}
}
start();
setInterval(() => {if (!document.hidden && !state.loading) refreshQueue();}, 15000);
