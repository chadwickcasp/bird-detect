export const clamp = (value, low, high) => Math.max(low, Math.min(high, value));

export function imagePoint(point, view) {
  return {x: (point.x - view.x) / view.scale, y: (point.y - view.y) / view.scale};
}

export function pinchView(view, startMidpoint, midpoint, ratio, minimum, maximum) {
  const anchor = imagePoint(startMidpoint, view);
  const scale = clamp(view.scale * ratio, minimum, maximum);
  return {scale, x: midpoint.x - anchor.x * scale, y: midpoint.y - anchor.y * scale};
}

export function constrainView(view, image, viewport) {
  const width = image.width * view.scale, height = image.height * view.scale;
  return {...view,
    x: width <= viewport.width ? (viewport.width - width) / 2 : clamp(view.x, viewport.width - width, 0),
    y: height <= viewport.height ? (viewport.height - height) / 2 : clamp(view.y, viewport.height - height, 0)};
}

export function moveBox(box, dx, dy) {
  return {...box, x: clamp(box.x + dx, 0, 1 - box.width), y: clamp(box.y + dy, 0, 1 - box.height)};
}

export function resizeBox(box, corner, point, minimumWidth, minimumHeight) {
  let left = box.x, top = box.y, right = box.x + box.width, bottom = box.y + box.height;
  if (corner.includes('w')) left = clamp(point.x, 0, right - minimumWidth);
  if (corner.includes('e')) right = clamp(point.x, left + minimumWidth, 1);
  if (corner.includes('n')) top = clamp(point.y, 0, bottom - minimumHeight);
  if (corner.includes('s')) bottom = clamp(point.y, top + minimumHeight, 1);
  return {...box, x: left, y: top, width: right - left, height: bottom - top};
}

export function nudgeBox(box, operation, dx, dy, image) {
  if (operation === 'move') return moveBox(box, dx / image.width, dy / image.height);
  if (operation === 'left') return resizeBox(box, 'w', {x: box.x + dx / image.width, y: box.y}, 1 / image.width, 1 / image.height);
  if (operation === 'right') return resizeBox(box, 'e', {x: box.x + box.width + dx / image.width, y: box.y}, 1 / image.width, 1 / image.height);
  if (operation === 'top') return resizeBox(box, 'n', {x: box.x, y: box.y + dy / image.height}, 1 / image.width, 1 / image.height);
  return resizeBox(box, 's', {x: box.x, y: box.y + box.height + dy / image.height}, 1 / image.width, 1 / image.height);
}
