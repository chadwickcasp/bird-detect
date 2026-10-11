import test from 'node:test';
import assert from 'node:assert/strict';
import {imagePoint, pinchView, constrainView, moveBox, resizeBox, nudgeBox} from '../review_ui/geometry.js';
import {BoxCanvas} from '../review_ui/canvas.js';

test('pinch keeps the image point under the midpoint and clamps zoom', () => {
  const view = {scale: 1, x: 10, y: 20}, center = {x: 100, y: 100};
  const result = pinchView(view, center, center, 2, .5, 20);
  assert.deepEqual(imagePoint(center, result), imagePoint(center, view));
  assert.equal(pinchView(view, center, center, 100, .5, 20).scale, 20);
});

test('movement, resizing, and pixel nudges stay inside image bounds', () => {
  const box = {id: 'bird', x: .2, y: .2, width: .3, height: .4};
  assert.equal(moveBox(box, 2, 2).x, .7);
  const resized = resizeBox(box, 'nw', {x: .9, y: .9}, .01, .01);
  assert.ok(resized.width > 0 && resized.height > 0);
  const moved = nudgeBox(box, 'move', 1, 0, {width: 100, height: 100});
  assert.ok(Math.abs(moved.x - .21) < 1e-10);
  assert.equal(constrainView({scale: 1, x: -1000, y: 1000}, {width: 100, height: 100}, {width: 50, height: 50}).x, -50);
});

function setupCanvas() {
  globalThis.ResizeObserver = class {observe() {}};
  globalThis.devicePixelRatio = 1;
  globalThis.requestAnimationFrame = () => 1;
  const noop = () => {};
  const element = {getContext: () => ({}), addEventListener: noop, setPointerCapture: noop,
    getBoundingClientRect: () => ({left: 0, top: 0, width: 100, height: 100}),
    parentElement: {getBoundingClientRect: () => ({width: 100, height: 100})}};
  const changes = [];
  const canvas = new BoxCanvas(element, {select: noop, zoom: noop, change: (boxes) => changes.push(structuredClone(boxes))});
  canvas.viewport = {width: 100, height: 100}; canvas.image = {width: 100, height: 100}; canvas.fitScale = 1;
  canvas.boxes = [{id: 'bird', label: 'bird', x: .1, y: .1, width: .6, height: .6, flags: []}]; canvas.selected = 'bird';
  return {canvas, changes};
}

const pointer = (id, x, y) => ({pointerId: id, clientX: x, clientY: y, pointerType: 'touch', preventDefault() {}});

test('starting a pinch cancels a provisional box drag and never commits a box edit', () => {
  const {canvas, changes} = setupCanvas(), original = structuredClone(canvas.boxes);
  canvas.pointerDown(pointer(1, 40, 40)); canvas.pointerMove(pointer(1, 45, 45));
  assert.notDeepEqual(canvas.boxes, original);
  canvas.pointerDown(pointer(2, 70, 70));
  assert.deepEqual(canvas.boxes, original);
  canvas.pointerMove(pointer(2, 95, 95));
  assert.ok(canvas.view.scale > 1);
  canvas.pointerEnd(pointer(2, 95, 95)); canvas.pointerEnd(pointer(1, 45, 45));
  assert.equal(changes.length, 0); assert.deepEqual(canvas.boxes, original);
});

test('corner drag commits once while pointer cancellation restores original geometry', () => {
  const {canvas, changes} = setupCanvas();
  canvas.pointerDown(pointer(1, 10, 10)); canvas.pointerMove(pointer(1, 20, 20)); canvas.pointerEnd(pointer(1, 20, 20));
  assert.equal(changes.length, 1); assert.ok(Math.abs(changes[0][0].x - .2) < 1e-10);
  const saved = structuredClone(canvas.boxes);
  canvas.pointerDown(pointer(1, 20, 20)); canvas.pointerMove(pointer(1, 30, 30)); canvas.pointerEnd(pointer(1, 30, 30), true);
  assert.deepEqual(canvas.boxes, saved); assert.equal(changes.length, 1);
});
