"use strict";
// Smooth zoom and pan for one photo: wheel, trackpad or touch pinch, drag, double-click, and the returned controls.
function attachZoom(stage, img, onChange) {
  const MAX_SCALE = 5, DOUBLE_CLICK_SCALE = 2.5, SMOOTHING_MS = 70;
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const pointers = new Map();
  const target = {scale: 1, x: 0, y: 0}, shown = {...target};
  let frame = null, lastTime = 0, pinchDistance = 0;

  function draw() {
    img.style.transform = `translate(${shown.x}px, ${shown.y}px) scale(${shown.scale})`;
  }

  function animate(time) {
    const blend = 1 - Math.exp(-(time - lastTime) / SMOOTHING_MS);  // the same speed at any refresh rate
    lastTime = time;
    for (const key in target) shown[key] += (target[key] - shown[key]) * blend;
    const settled = Math.abs(target.scale - shown.scale) < 0.001 && Math.abs(target.x - shown.x) < 0.1 && Math.abs(target.y - shown.y) < 0.1;
    if (settled) Object.assign(shown, target);
    draw();
    frame = settled ? null : requestAnimationFrame(animate);
  }

  function update({immediate = false} = {}) {
    clampPan();
    onChange(target.scale);
    if (immediate || reducedMotion) {
      Object.assign(shown, target);
      return draw();
    }
    if (frame === null) {
      lastTime = performance.now();
      frame = requestAnimationFrame(animate);
    }
  }

  function clampPan() {
    if (!img.naturalWidth) return;
    const {width, height} = stage.getBoundingClientRect();
    const fit = Math.min(width / img.naturalWidth, height / img.naturalHeight);
    const limitX = Math.max(0, (img.naturalWidth * fit * target.scale - width) / 2);
    const limitY = Math.max(0, (img.naturalHeight * fit * target.scale - height) / 2);
    target.x = Math.min(limitX, Math.max(-limitX, target.x));
    target.y = Math.min(limitY, Math.max(-limitY, target.y));
  }

  function zoomAt(factor, clientX, clientY) {
    const box = stage.getBoundingClientRect();
    const scale = Math.min(MAX_SCALE, Math.max(1, target.scale * factor));
    const applied = scale / target.scale;
    // keep the point under the cursor in place
    const px = clientX - box.left - box.width / 2, py = clientY - box.top - box.height / 2;
    target.x = px - (px - target.x) * applied;
    target.y = py - (py - target.y) * applied;
    target.scale = scale;
    update();
  }

  function zoomFromCenter(factor) {
    const box = stage.getBoundingClientRect();
    zoomAt(factor, box.left + box.width / 2, box.top + box.height / 2);
  }

  function reset() {
    Object.assign(target, {scale: 1, x: 0, y: 0});
    update();
  }

  stage.addEventListener("wheel", e => {
    e.preventDefault();
    const pixels = e.deltaMode === WheelEvent.DOM_DELTA_LINE ? e.deltaY * 16 : e.deltaY;
    zoomAt(Math.exp(-pixels * (e.ctrlKey ? 0.012 : 0.002)), e.clientX, e.clientY);  // ctrlKey marks a trackpad pinch
  }, {passive: false});

  stage.addEventListener("dblclick", e => {
    if (e.target.closest(".zoom-controls")) return;
    target.scale > 1 ? reset() : zoomAt(DOUBLE_CLICK_SCALE, e.clientX, e.clientY);
  });

  stage.addEventListener("pointerdown", e => {
    if (e.target.closest(".zoom-controls")) return;
    stage.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, {x: e.clientX, y: e.clientY});
    stage.classList.add("is-dragging");
    pinchDistance = 0;
  });

  stage.addEventListener("pointermove", e => {
    const previous = pointers.get(e.pointerId);
    if (!previous) return;
    const point = {x: e.clientX, y: e.clientY};
    pointers.set(e.pointerId, point);
    if (pointers.size === 1) {
      target.x += point.x - previous.x;
      target.y += point.y - previous.y;
      update({immediate: true});  // the photo stays under the finger
    } else if (pointers.size === 2) {
      const [a, b] = pointers.values();
      const distance = Math.hypot(a.x - b.x, a.y - b.y);
      if (pinchDistance) zoomAt(distance / pinchDistance, (a.x + b.x) / 2, (a.y + b.y) / 2);
      pinchDistance = distance;
    }
  });

  const release = e => {
    pointers.delete(e.pointerId);
    pinchDistance = 0;
    if (!pointers.size) stage.classList.remove("is-dragging");
  };
  stage.addEventListener("pointerup", release);
  stage.addEventListener("pointercancel", release);
  new ResizeObserver(() => update()).observe(stage);

  return {zoomIn: () => zoomFromCenter(1.5), zoomOut: () => zoomFromCenter(1 / 1.5), reset};
}
