const $ = (id) => document.getElementById(id);
const state = {photos: [], reviews: {}, limit: 24, filter: 'all', showBoxes: false, undo: [], current: null, browseIds: [], busy: false};
const observer = new ResizeObserver((entries) => entries.forEach(({target}) => positionBoxes(target)));

function positionBoxes(media) {
  const img = media.querySelector('img'), svg = media.querySelector('svg');
  if (!img.naturalWidth) return;
  const scale = Math.min(media.clientWidth / img.naturalWidth, media.clientHeight / img.naturalHeight);
  const width = img.naturalWidth * scale, height = img.naturalHeight * scale;
  Object.assign(svg.style, {width: `${width}px`, height: `${height}px`, left: `${(media.clientWidth-width)/2}px`, top: `${(media.clientHeight-height)/2}px`});
}

function photoMedia(photo, large = false) {
  const media = document.createElement('div'); media.className = 'media';
  const img = document.createElement('img'); img.alt = photo.title || 'Bird photograph';
  img.loading = large ? 'eager' : 'lazy'; img.referrerPolicy = 'no-referrer';
  img.src = large ? photo.image_url : photo.thumbnail_url;
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 1 1'); svg.setAttribute('preserveAspectRatio', 'none'); svg.setAttribute('aria-hidden', 'true'); svg.style.display = state.showBoxes ? 'block' : 'none';
  for (const box of photo.boxes) {
    const rect = document.createElementNS(svg.namespaceURI, 'rect'), b = box.bounding_box;
    for (const [key, value] of Object.entries({x: +b.xmin, y: +b.ymin, width: +b.xmax - +b.xmin, height: +b.ymax - +b.ymin})) rect.setAttribute(key, value);
    rect.setAttribute('class', `box${box.attributes.includes('GroupOf') || box.text.includes('Group') ? ' group' : ''}`);
    svg.append(rect);
  }
  const error = document.createElement('div'); error.className = 'image-error'; error.hidden = true; error.textContent = 'Photo unavailable. Open the source to check it.';
  let fallback = false;
  img.onerror = () => {if (!fallback) {fallback = true; img.src = large ? photo.thumbnail_url : photo.image_url;} else {error.hidden = false;}};
  img.onload = () => positionBoxes(media);
  media.append(img, svg, error); observer.observe(media);
  return media;
}

function filteredPhotos() {
  return state.photos.filter(photo => state.filter === 'all' || (state.reviews[photo.id]?.decision || 'unreviewed') === state.filter);
}

function updateCounts() {
  const kept = state.photos.filter(photo => state.reviews[photo.id]?.decision === 'keep').length;
  $('picks').textContent = `My picks (${kept})`; $('undo').disabled = !state.undo.length || state.busy;
}

function render() {
  observer.disconnect();
  const photos = filteredPhotos(), grid = $('grid'); grid.replaceChildren();
  for (const photo of photos.slice(0, state.limit)) {
    const card = document.createElement('article'); card.className = `card${state.reviews[photo.id]?.decision === 'keep' ? ' kept' : ''}`; card.dataset.id = photo.id;
    const media = photoMedia(photo); media.tabIndex = 0; media.setAttribute('role', 'button'); media.setAttribute('aria-label', `Preview ${photo.title || photo.id}`);
    media.onclick = () => openPreview(photo.id); media.onkeydown = event => {if (event.key === 'Enter' || event.key === ' ') {event.preventDefault(); openPreview(photo.id);}};
    const actions = document.createElement('div'); actions.className = 'card-actions';
    const preview = document.createElement('button'); preview.className = 'preview'; preview.textContent = 'Preview'; preview.onclick = () => openPreview(photo.id);
    const keep = document.createElement('button'); keep.className = 'keep'; keep.textContent = state.reviews[photo.id]?.decision === 'keep' ? 'Kept' : 'Keep'; keep.setAttribute('aria-pressed', String(state.reviews[photo.id]?.decision === 'keep'));
    keep.disabled = state.busy; keep.onclick = () => save(photo.id, state.reviews[photo.id]?.decision === 'keep' ? 'unreviewed' : 'keep');
    actions.append(preview, keep); card.append(media, actions); grid.append(card);
  }
  $('more').hidden = state.limit >= photos.length; $('empty').hidden = photos.length !== 0;
  $('count').textContent = `Showing ${Math.min(state.limit, photos.length)} of ${photos.length} photos`;
  updateCounts();
}

async function save(id, decision, notes = state.reviews[id]?.notes || '', remember = true) {
  if (state.busy) return false;
  state.busy = true; updateCounts();
  document.querySelectorAll('.keep, [data-decision]').forEach(button => button.disabled = true);
  const previous = state.reviews[id] || {id, decision: 'unreviewed', notes: ''};
  try {
    const response = await fetch('/api/review', {method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({id, decision, notes})});
    const review = await response.json(); if (!response.ok) throw new Error(review.error);
    state.reviews[id] = review; if (remember) state.undo.push(previous);
    $('status').textContent = decision === 'unreviewed' ? 'Selection updated.' : `${decision === 'keep' ? 'Kept' : decision === 'skip' ? 'Skipped' : 'Marked unsure'}. Saved automatically.`;
    return true;
  } catch (error) {$('status').textContent = `Could not save: ${error.message}`; return false;}
  finally {state.busy = false; render(); document.querySelectorAll('[data-decision]').forEach(button => button.disabled = false);}
}

function openPreview(id, retainQueue = false) {
  if (!retainQueue) state.browseIds = filteredPhotos().map(photo => photo.id);
  const photo = state.photos.find(photo => photo.id === id); state.current = id;
  $('preview-title').textContent = photo.title || 'Photo preview'; $('large-photo').replaceChildren(photoMedia(photo, true));
  $('attribution').textContent = `${photo.attribution} · ${photo.boxes.length} source bird box${photo.boxes.length === 1 ? '' : 'es'} · Boxes need review`;
  $('original').href = photo.source_url; $('notes').value = state.reviews[id]?.notes || '';
  const index = state.browseIds.indexOf(id); $('previous').disabled = index <= 0; $('next').disabled = index >= state.browseIds.length - 1;
  if (!$('preview').open) $('preview').showModal();
}

function move(delta) {
  const index = state.browseIds.indexOf(state.current), id = state.browseIds[index + delta];
  if (id) openPreview(id, true);
}

async function decide(decision) {
  if (await save(state.current, decision, $('notes').value)) {
    const index = state.browseIds.indexOf(state.current);
    if (index < state.browseIds.length - 1) move(1); else $('preview').close();
  }
}

$('filter').onchange = () => {state.filter = $('filter').value; state.limit = 24; render();};
$('picks').onclick = () => {$('filter').value = 'keep'; $('filter').dispatchEvent(new Event('change'));};
$('show-boxes').onchange = () => {state.showBoxes = $('show-boxes').checked; document.querySelectorAll('.media svg').forEach(svg => svg.style.display = state.showBoxes ? 'block' : 'none');};
$('more').onclick = () => {state.limit += 24; render();};
$('undo').onclick = async () => {const previous = state.undo.at(-1); if (previous && await save(previous.id, previous.decision, previous.notes, false)) {state.undo.pop(); updateCounts();}};
$('close').onclick = () => $('preview').close(); $('previous').onclick = () => move(-1); $('next').onclick = () => move(1);
document.querySelectorAll('[data-decision]').forEach(button => button.onclick = () => decide(button.dataset.decision));
$('preview').addEventListener('keydown', event => {
  if (['TEXTAREA', 'INPUT', 'SELECT'].includes(event.target.tagName) || state.busy) return;
  const decisions = {k: 'keep', s: 'skip', u: 'unsure'};
  if (decisions[event.key.toLowerCase()]) {event.preventDefault(); decide(decisions[event.key.toLowerCase()]);}
  if (event.key === 'ArrowRight') {event.preventDefault(); move(1);}
  if (event.key === 'ArrowLeft') {event.preventDefault(); move(-1);}
});

async function start() {
  try {
    const responses = await Promise.all([fetch('/api/catalog'), fetch('/api/state'), fetch('/api/catalog-info')]);
    if (responses.some(response => !response.ok)) throw new Error('The local picker could not load.');
    const [photos, reviews, info] = await Promise.all(responses.map(response => response.json()));
    state.photos = photos; state.reviews = reviews;
    $('catalog-info').textContent = `Open Images · ${info.full_catalog ? 'Full Bird training catalog' : 'Bird gallery sample'} · ${info.total.toLocaleString()} photos · ${info.hidden.toLocaleString()} Approved / Needs work hidden · Camera and location unverified`;
    $('status').textContent = ''; render();
  } catch (error) {$('status').textContent = error.message;}
}
start();
