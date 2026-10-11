import {clamp, imagePoint, pinchView, constrainView, moveBox, resizeBox} from './geometry.js';

const CORNERS = ['nw', 'ne', 'se', 'sw'];
const distance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const midpoint = (a, b) => ({x: (a.x + b.x) / 2, y: (a.y + b.y) / 2});

export class BoxCanvas {
  constructor(canvas, callbacks) {
    this.canvas = canvas;
    this.context = canvas.getContext('2d');
    this.callbacks = callbacks;
    this.image = null;
    this.boxes = [];
    this.selected = null;
    this.mode = 'select';
    this.editable = true;
    this.visible = true;
    this.original = false;
    this.proposals = [];
    this.predictions = [];
    this.points = new Map();
    this.gesture = null;
    this.view = {x: 0, y: 0, scale: 1};
    this.fitScale = 1;
    this.frame = null;
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas.parentElement);
    canvas.addEventListener('pointerdown', event => this.pointerDown(event));
    canvas.addEventListener('pointermove', event => this.pointerMove(event));
    canvas.addEventListener('pointerup', event => this.pointerEnd(event));
    canvas.addEventListener('pointercancel', event => this.pointerEnd(event, true));
    canvas.addEventListener('lostpointercapture', event => {if (this.points.has(event.pointerId)) this.pointerEnd(event, true);});
    canvas.addEventListener('wheel', event => {
      event.preventDefault(); this.zoom(Math.exp(-event.deltaY * .002), this.point(event));
    }, {passive: false});
    canvas.addEventListener('dblclick', event => {event.preventDefault(); this.zoom(2, this.point(event));});
  }

  resize() {
    const rect = this.canvas.parentElement.getBoundingClientRect(), ratio = devicePixelRatio || 1;
    this.viewport = {width: rect.width, height: rect.height};
    this.canvas.width = Math.round(rect.width * ratio);
    this.canvas.height = Math.round(rect.height * ratio);
    this.context.setTransform(ratio, 0, 0, ratio, 0, 0);
    if (this.image) {
      const nextFit = Math.min(rect.width / this.image.width, rect.height / this.image.height);
      const oldFit = this.fitScale;
      this.fitScale = nextFit;
      if (Math.abs(this.view.scale / oldFit - 1) < .02) this.fit();
      else this.view = constrainView(this.view, this.image, this.viewport);
    }
    this.render();
  }

  async load(url) {
    this.points.clear(); this.gesture = null; this.image = null; this.render();
    const image = new Image();
    await new Promise((resolve, reject) => {image.onload = resolve; image.onerror = () => reject(new Error('The photo could not load. Retry this image.')); image.src = url;});
    this.image = image; this.fit();
    return {width: image.width, height: image.height};
  }

  setReview(boxes, selected, proposals, editable) {
    this.boxes = boxes; this.selected = selected; this.proposals = proposals; this.editable = editable; this.render();
  }

  point(event) {
    const rect = this.canvas.getBoundingClientRect();
    return {x: event.clientX - rect.left, y: event.clientY - rect.top};
  }

  normalized(point) {
    const world = imagePoint(point, this.view);
    return {x: clamp(world.x / this.image.width, 0, 1), y: clamp(world.y / this.image.height, 0, 1)};
  }

  fit() {
    if (!this.image || !this.viewport) return;
    this.fitScale = Math.min(this.viewport.width / this.image.width, this.viewport.height / this.image.height);
    this.view = constrainView({scale: this.fitScale, x: 0, y: 0}, this.image, this.viewport);
    this.render();
  }

  zoom(ratio, anchor = {x: this.viewport.width / 2, y: this.viewport.height / 2}) {
    if (!this.image) return;
    this.view = constrainView(pinchView(this.view, anchor, anchor, ratio, this.fitScale, this.fitScale * 20), this.image, this.viewport);
    this.render();
  }

  focus() {
    const box = this.boxes.find(box => box.id === this.selected);
    if (!box || !this.image) return;
    const scale = clamp(Math.min(this.viewport.width / (box.width * this.image.width * 1.5), this.viewport.height / (box.height * this.image.height * 1.5)), this.fitScale, this.fitScale * 20);
    this.view = constrainView({scale,
      x: this.viewport.width / 2 - (box.x + box.width / 2) * this.image.width * scale,
      y: this.viewport.height / 2 - (box.y + box.height / 2) * this.image.height * scale}, this.image, this.viewport);
    this.render();
  }

  screenBox(box) {
    return {x: this.view.x + box.x * this.image.width * this.view.scale,
      y: this.view.y + box.y * this.image.height * this.view.scale,
      width: box.width * this.image.width * this.view.scale,
      height: box.height * this.image.height * this.view.scale};
  }

  hit(point) {
    if (!this.visible) return null;
    const selected = this.boxes.find(box => box.id === this.selected);
    if (selected && this.editable) {
      const rect = this.screenBox(selected);
      const corners = [{x: rect.x, y: rect.y}, {x: rect.x + rect.width, y: rect.y},
        {x: rect.x + rect.width, y: rect.y + rect.height}, {x: rect.x, y: rect.y + rect.height}];
      const hits = corners.map((corner, index) => ({distance: distance(point, corner), corner: CORNERS[index]})).sort((a, b) => a.distance - b.distance);
      if (hits[0].distance < 22) return {id: selected.id, corner: hits[0].corner};
    }
    const pixel = imagePoint(point, this.view);
    if (pixel.x < 0 || pixel.y < 0 || pixel.x > this.image.width || pixel.y > this.image.height) return null;
    const world = this.normalized(point);
    const matches = this.boxes.filter(box => world.x >= box.x && world.x <= box.x + box.width && world.y >= box.y && world.y <= box.y + box.height);
    matches.sort((a, b) => a.width * a.height - b.width * b.height);
    return matches.length ? {id: matches[0].id} : null;
  }

  pointerDown(event) {
    if (!this.image || (event.pointerType === 'mouse' && event.button !== 0)) return;
    event.preventDefault(); this.canvas.setPointerCapture(event.pointerId);
    const point = this.point(event); this.points.set(event.pointerId, point);
    if (this.points.size === 2) {
      if (this.gesture?.before) this.boxes = structuredClone(this.gesture.before);
      const [first, second] = [...this.points.values()];
      this.gesture = {type: 'pinch', view: {...this.view}, midpoint: midpoint(first, second), distance: Math.max(1, distance(first, second))};
      this.render(); return;
    }
    if (this.points.size > 2 || this.gesture?.type === 'wait') return;
    const hit = this.mode === 'select' ? this.hit(point) : null;
    if (hit) {this.selected = hit.id; this.callbacks.select(hit.id);}
    const type = this.mode === 'add' && this.editable ? 'draw' : hit && this.editable ? hit.corner ? 'resize' : 'move' : 'pan';
    this.gesture = {type, point, view: {...this.view}, before: structuredClone(this.boxes),
      box: hit ? structuredClone(this.boxes.find(box => box.id === hit.id)) : null,
      corner: hit?.corner, start: this.normalized(point), changed: false};
    this.render();
  }

  pointerMove(event) {
    if (!this.points.has(event.pointerId) || !this.gesture || !this.image) return;
    event.preventDefault(); const point = this.point(event); this.points.set(event.pointerId, point);
    const gesture = this.gesture;
    if (gesture.type === 'pinch' && this.points.size >= 2) {
      const [first, second] = [...this.points.values()];
      this.view = constrainView(pinchView(gesture.view, gesture.midpoint, midpoint(first, second), distance(first, second) / gesture.distance, this.fitScale, this.fitScale * 20), this.image, this.viewport);
    } else if (gesture.type === 'pan') {
      this.view = constrainView({...gesture.view, x: gesture.view.x + point.x - gesture.point.x, y: gesture.view.y + point.y - gesture.point.y}, this.image, this.viewport);
    } else if (gesture.type === 'move' || gesture.type === 'resize') {
      if (distance(point, gesture.point) < 3 && !gesture.changed) return;
      const next = gesture.type === 'move'
        ? moveBox(gesture.box, (point.x - gesture.point.x) / this.view.scale / this.image.width, (point.y - gesture.point.y) / this.view.scale / this.image.height)
        : resizeBox(gesture.box, gesture.corner, this.normalized(point), 1 / this.image.width, 1 / this.image.height);
      this.boxes = gesture.before.map(box => box.id === next.id ? next : box);
      gesture.changed = true;
    } else if (gesture.type === 'draw') {
      gesture.end = this.normalized(point); gesture.changed = distance(point, gesture.point) >= 5;
    }
    this.render();
  }

  pointerEnd(event, cancelled = false) {
    if (!this.points.has(event.pointerId)) return;
    this.points.delete(event.pointerId);
    const gesture = this.gesture;
    if (gesture?.type === 'pinch' || gesture?.type === 'wait') {
      this.gesture = this.points.size ? {type: 'wait'} : null; this.render(); return;
    }
    if (cancelled && gesture?.before) this.boxes = gesture.before;
    else if (gesture?.changed && gesture.type === 'draw') {
      const start = gesture.start, end = gesture.end;
      const box = {id: crypto.randomUUID ? crypto.randomUUID() : `box-${Date.now()}-${Math.random().toString(16).slice(2)}`,
        label: 'bird', x: Math.min(start.x, end.x), y: Math.min(start.y, end.y), width: Math.abs(start.x - end.x), height: Math.abs(start.y - end.y), flags: []};
      if (box.width * this.image.width >= 1 && box.height * this.image.height >= 1) {
        this.boxes = [...gesture.before, box]; this.selected = box.id; this.callbacks.change(this.boxes, box.id);
      }
    } else if (gesture?.changed && ['move', 'resize'].includes(gesture.type)) this.callbacks.change(this.boxes, this.selected);
    this.gesture = null; this.render();
  }

  render() {
    if (this.frame) return;
    this.frame = requestAnimationFrame(() => {this.frame = null; this.draw();});
  }

  drawBox(box, index, proposal = false, prediction = false) {
    const context = this.context, rect = this.screenBox(box), selected = box.id === this.selected && !proposal && !prediction;
    context.strokeStyle = prediction ? '#c7a7ff' : proposal ? '#ffcd75' : selected ? '#42ece4' : '#96f4ed';
    context.lineWidth = selected ? 2.5 : 1.5;
    context.setLineDash(proposal || prediction ? [6, 4] : box.flags?.includes('groupof') || box.flags?.includes('group') ? [6, 3] : []);
    context.strokeRect(rect.x, rect.y, rect.width, rect.height); context.setLineDash([]);
    context.font = '600 13px Arial';
    const text = prediction ? `YOLO ${index + 1} · ${Math.round(box.confidence * 100)}%` : proposal ? `source ${index + 1}` : `bird ${index + 1}`;
    const width = context.measureText(text).width + 12;
    const labelX = clamp(rect.x, 0, Math.max(0, this.viewport.width - width));
    const labelY = clamp(rect.y - 25, 0, this.viewport.height - 23);
    context.fillStyle = prediction ? '#604396' : proposal ? '#6a4d23' : '#087e81'; context.fillRect(labelX, labelY, width, 23);
    context.fillStyle = '#fff'; context.fillText(text, labelX + 6, labelY + 16);
    if (selected && this.editable) {
      context.fillStyle = '#087e81'; context.strokeStyle = '#fff'; context.lineWidth = 2;
      for (const point of [{x: rect.x, y: rect.y}, {x: rect.x + rect.width, y: rect.y}, {x: rect.x + rect.width, y: rect.y + rect.height}, {x: rect.x, y: rect.y + rect.height}]) {
        context.fillRect(point.x - 6, point.y - 6, 12, 12); context.strokeRect(point.x - 6, point.y - 6, 12, 12);
      }
    }
  }

  draw() {
    if (!this.viewport) return;
    const context = this.context;
    context.clearRect(0, 0, this.viewport.width, this.viewport.height);
    context.fillStyle = '#13232e'; context.fillRect(0, 0, this.viewport.width, this.viewport.height);
    if (!this.image) return;
    context.drawImage(this.image, this.view.x, this.view.y, this.image.width * this.view.scale, this.image.height * this.view.scale);
    if (this.visible) this.boxes.forEach((box, index) => this.drawBox(box, index));
    if (this.original) this.proposals.forEach((box, index) => this.drawBox(box, index, true));
    this.predictions.forEach((box, index) => this.drawBox(box, index, false, true));
    if (this.gesture?.type === 'draw' && this.gesture.end) {
      const start = this.gesture.start, end = this.gesture.end;
      this.drawBox({id: 'drawing', x: Math.min(start.x, end.x), y: Math.min(start.y, end.y), width: Math.abs(start.x-end.x), height: Math.abs(start.y-end.y)}, this.boxes.length);
    }
    this.callbacks.zoom(Math.round(this.view.scale / this.fitScale * 100));
  }
}
