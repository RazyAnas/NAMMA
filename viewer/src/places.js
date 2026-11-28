// NAMMA demo viewer — places, search and 3D help cards.
//
// Reuses the production POI engine (poi_core.js: SpatialIndex, makePoi, CATEGORIES — pure logic, no DOM) and adds a thin
// Three.js overlay: pins projected over the canvas, a search box, and the 3D help cards (category 'help'): tap a help pin and a
// menu card opens anchored over it, tracking the pin as you move. Admin (?admin=1) places and edits points; everything persists
// to localStorage (the demo has no backend). Points are stored in viewer-world coordinates (three.js units).
import * as THREE from 'three';
import { SpatialIndex, makePoi, CATEGORIES } from '../../colab/real_scenes/final/poi_core.js';
import './places.css';

const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const el = (tag, attrs = {}, html) => { const n = document.createElement(tag); Object.assign(n, attrs); if (html != null) n.innerHTML = html; return n; };

// ctx = { camera, canvas, metersPerUnit, frame?, layers?, goTo(THREE.Vector3), seed? [{name,pos,category,card,note}] }
export function initPlaces(ctx) {
  const { camera, frame, layers } = ctx;
  const mpu = ctx.metersPerUnit || 1;
  const admin = new URLSearchParams(location.search).get('admin') === '1';
  const KEY = 'namma-demo-pois:' + location.pathname;

  // ---- data ---------------------------------------------------------------
  const index = new SpatialIndex();
  let pois = [];
  const persist = () => { try { localStorage.setItem(KEY, JSON.stringify({ version: 1, pois })); } catch (e) {} };
  const reindex = () => { index.clear(); pois.forEach((p) => index.insert({ id: p.id, pos: p.pos, poi: p })); };
  const byId = (id) => pois.find((p) => p.id === id);
  { let raw = null; try { raw = JSON.parse(localStorage.getItem(KEY) || 'null'); } catch (e) {}
    const src = raw && Array.isArray(raw.pois) ? raw.pois : (ctx.seed || []);
    pois = src.map((p) => makePoi(p, null)); reindex(); }
  function addPoi(p) { const poi = makePoi(p, null); const old = byId(poi.id);
    if (old) { index.remove(old.id); pois[pois.indexOf(old)] = poi; } else pois.push(poi);
    index.insert({ id: poi.id, pos: poi.pos, poi }); persist(); return poi; }
  function removePoi(id) { const p = byId(id); if (!p) return; index.remove(id); pois.splice(pois.indexOf(p), 1); persist(); if (cardId === id) closeCard(); }

  // ---- DOM ---------------------------------------------------------------
  const pinsEl = el('div', { id: 'pl-pins' });
  const cardEl = el('div', { id: 'pl-card', className: 'pl-panel' }, '<button class="pl-btn x" title="Close">×</button><h2></h2><div class="bd"></div>');
  const findEl = el('div', { id: 'pl-find' }, '<div id="pl-hits"></div><input type="search" placeholder="Search the room: sofa, door, help …" autocomplete="off">');
  document.body.append(pinsEl, cardEl, findEl);
  const cardTitle = cardEl.querySelector('h2'), cardBody = cardEl.querySelector('.bd');
  const q = findEl.querySelector('input'), hitsEl = findEl.querySelector('#pl-hits');
  ['pointerdown', 'pointerup', 'click', 'wheel', 'keydown', 'keyup'].forEach((t) => { cardEl.addEventListener(t, (e) => e.stopPropagation()); findEl.addEventListener(t, (e) => e.stopPropagation()); });

  // ---- projection: viewer world point -> screen pixels -------------------
  const v = new THREE.Vector3(), camPos = new THREE.Vector3();
  function project(pos) { v.set(pos[0], pos[1], pos[2]).project(camera);
    const infront = v.z < 1 && v.z > -1; camera.getWorldPosition(camPos);
    return { x: (v.x * 0.5 + 0.5) * innerWidth, y: (-v.y * 0.5 + 0.5) * innerHeight, infront, dist: camPos.distanceTo({ x: pos[0], y: pos[1], z: pos[2] }) * mpu }; }

  // ---- pins --------------------------------------------------------------
  const pinEl = new Map();
  function activate(p) { if (p.category === 'help') openCard(p); else if (ctx.goTo) { ctx.goTo(new THREE.Vector3(p.pos[0], p.pos[1], p.pos[2])); closeCard(); } }
  function drawPins() { const seen = new Set();
    pois.forEach((p) => { seen.add(p.id); let b = pinEl.get(p.id);
      if (!b) { b = el('button', { className: 'pl-pin' }, '<span></span><i></i>'); pinsEl.append(b); pinEl.set(p.id, b);
        b.onclick = (e) => { e.stopPropagation(); if (admin) editSel(p.id); else activate(p); }; }
      b.firstChild.textContent = p.name; b.title = p.category === 'help' ? p.name + ' — tap for help' : p.name;
      b.classList.toggle('help', p.category === 'help'); });
    pinEl.forEach((b, id) => { if (!seen.has(id)) { b.remove(); pinEl.delete(id); } }); }
  function placePins() { pinEl.forEach((b, id) => { const p = byId(id); const s = p && project(p.pos);
    if (!s || !s.infront || s.dist > 60) { b.style.transform = 'translate(-9999px,0)'; return; }
    b.classList.toggle('far', s.dist > 7 && sel !== id); b.classList.toggle('sel', sel === id);
    b.style.opacity = Math.max(0.5, Math.min(1, 1.3 - s.dist / 22));
    b.style.transform = `translate(${Math.round(s.x - b.offsetWidth / 2)}px,${Math.round(s.y - b.offsetHeight)}px)`; }); }

  // ---- help card ---------------------------------------------------------
  let cardId = null;
  function openCard(p) { if (!p) return; cardId = p.id; cardTitle.textContent = p.name; cardBody.textContent = (p.card || p.note || '').trim() || 'No help text yet.'; cardEl.style.display = 'flex'; positionCard(); }
  function closeCard() { cardId = null; cardEl.style.display = 'none'; cardEl.style.transform = 'translate(-9999px,0)'; }
  function positionCard() { if (cardId == null) return; const p = byId(cardId); if (!p) return closeCard();
    const s = project(p.pos); if (!s || !s.infront || s.dist > 60) { cardEl.style.transform = 'translate(-9999px,0)'; return; }
    const w = cardEl.offsetWidth, h = cardEl.offsetHeight, m = 8;
    const x = Math.max(m, Math.min(innerWidth - w - m, s.x - 24)), y = Math.max(m, Math.min(innerHeight - h - m, s.y - h - 22));
    cardEl.style.transform = `translate(${Math.round(x)}px,${Math.round(y)}px)`; }
  cardEl.querySelector('.x').onclick = (e) => { e.stopPropagation(); closeCard(); };
  addEventListener('keydown', (e) => { if (e.key === 'Escape' && cardId != null) closeCard(); });

  // ---- search ------------------------------------------------------------
  function render(list) { hitsEl.textContent = ''; const s = (q.value || '').trim().toLowerCase();
    const items = (s ? list.filter((p) => (p.name + ' ' + p.category + ' ' + (p.note || '') + ' ' + (p.card || '')).toLowerCase().includes(s)) : list).slice(0, 8);
    if (!items.length) { hitsEl.append(el('div', { className: 'empty' }, s ? 'Nothing matches here.' : 'No places yet.' + (admin ? ' Press P to drop a help point where you look.' : ''))); return; }
    items.forEach((p) => { const b = el('button', {}, `<span class="dot ${p.category === 'help' ? 'help' : ''}"></span><span>${esc(p.name)}</span><span class="tag">${esc(p.category)}</span>`);
      b.onclick = () => { q.value = ''; render(pois); activate(p); }; hitsEl.append(b); }); }
  let open = false; const showHits = (on) => { open = on; hitsEl.style.display = on ? 'flex' : 'none'; if (on) render(pois); };
  q.onfocus = () => showHits(true); q.oninput = () => render(pois);
  q.onkeydown = (e) => { e.stopPropagation(); if (e.key === 'Enter') { const first = hitsEl.querySelector('button'); if (first) first.click(); } else if (e.key === 'Escape') { q.blur(); showHits(false); } };
  document.addEventListener('pointerdown', (e) => { if (open && !findEl.contains(e.target)) showHits(false); });
  showHits(false);

  // ---- admin (?admin=1): place with P, edit, delete; saved to this browser ----
  let sel = null, placing = false;
  const ahead = new THREE.Vector3(), fwd = new THREE.Vector3();
  let placeDist = 2.0;                                                           // metres in front of the look point
  function pointAhead() { camera.getWorldPosition(ahead); fwd.set(0, 0, -1).applyQuaternion(camera.quaternion); ahead.addScaledVector(fwd, placeDist / mpu); return [ahead.x, ahead.y, ahead.z]; }
  function dropHere() { const p = addPoi({ name: 'Help point', category: 'help', card: '', pos: pointAhead() }); drawPins(); editSel(p.id); }

  let panel = null, nameI, catI, noteI, cardI, cardL, listEl, dirtyEl;
  function editSel(id) { const p = byId(id); if (!p || !panel) return; sel = id; form.style.display = 'flex';
    nameI.value = p.name; catI.value = p.category; noteI.value = p.note || ''; cardI.value = p.card || ''; showCardField(p.category); renderList(); drawPins(); }
  let form = null, showCardField = () => {};
  if (admin) {
    document.body.append(el('div', { id: 'pl-aim-ring' }));
    panel = el('div', { id: 'pl-admin', className: 'pl-panel' });
    panel.innerHTML = `<h1><span>Places <small id="pl-dirty"></small></span></h1>
      <p class="hint">Look where it should go and press <b>P</b> (or the button) to drop a help point. <b>[</b> / <b>]</b> set how far ahead.</p>
      <div class="row"><button class="pl-pill on" id="pl-add">+ Add help point (P)</button></div>
      <div id="pl-list"></div>
      <div id="pl-form" style="display:none;flex-direction:column;gap:8px">
        <label>Name<input id="pl-name" maxlength="80"></label>
        <label>Kind<select id="pl-cat">${CATEGORIES.map((c) => `<option value="${c}">${c[0].toUpperCase() + c.slice(1)}</option>`).join('')}</select></label>
        <label>Note<textarea id="pl-note" maxlength="400" style="min-height:40px"></textarea></label>
        <label id="pl-cardl">Help card text<textarea id="pl-card-in" maxlength="1500" placeholder="What this point explains …"></textarea></label>
        <div class="row"><button class="pl-pill" id="pl-preview">Preview card</button><button class="pl-pill" id="pl-reaim">Re-aim here</button><button class="pl-pill warn" id="pl-del">Delete</button></div>
      </div>`;
    document.body.append(panel);
    ['pointerdown', 'pointerup', 'click', 'wheel', 'keydown', 'keyup'].forEach((t) => panel.addEventListener(t, (e) => e.stopPropagation()));
    form = panel.querySelector('#pl-form'); nameI = panel.querySelector('#pl-name'); catI = panel.querySelector('#pl-cat');
    noteI = panel.querySelector('#pl-note'); cardI = panel.querySelector('#pl-card-in'); cardL = panel.querySelector('#pl-cardl');
    listEl = panel.querySelector('#pl-list'); dirtyEl = panel.querySelector('#pl-dirty');
    showCardField = (cat) => { cardL.style.display = cat === 'help' ? '' : 'none'; };
    const save = () => { const p = byId(sel); if (!p) return; addPoi({ ...p, name: nameI.value.trim() || 'Place', category: catI.value, note: noteI.value.trim(), card: cardI.value.trim() }); drawPins(); renderList(); if (cardId === sel) openCard(byId(sel)); };
    nameI.oninput = save; noteI.oninput = save; cardI.oninput = save; catI.onchange = () => { showCardField(catI.value); save(); };
    panel.querySelector('#pl-add').onclick = dropHere;
    panel.querySelector('#pl-preview').onclick = () => { const p = byId(sel); if (p) openCard(p); };
    panel.querySelector('#pl-reaim').onclick = () => { const p = byId(sel); if (p) { addPoi({ ...p, pos: pointAhead() }); drawPins(); } };
    panel.querySelector('#pl-del').onclick = () => { const p = byId(sel); if (p && confirm(`Delete "${p.name}"?`)) { removePoi(p.id); sel = null; form.style.display = 'none'; drawPins(); renderList(); } };
    function renderList() { listEl.textContent = ''; dirtyEl.textContent = pois.length ? pois.length + ' saved' : '';
      pois.forEach((p) => { const b = el('button', { className: p.id === sel ? 'on' : '' }, `<span>${esc(p.name)}</span><small>${esc(p.category)}</small>`); b.onclick = () => editSel(p.id); listEl.append(b); });
      if (!pois.length) listEl.append(el('div', { className: 'hint' }, 'No points yet.')); }
    panel._renderList = renderList; renderList();
    addEventListener('keydown', (e) => { if (/INPUT|TEXTAREA|SELECT/.test((e.target || {}).tagName || '')) return;
      if (e.code === 'KeyP') { e.preventDefault(); dropHere(); }
      else if (e.key === '[') { placeDist = Math.max(0.5, placeDist - 0.5); }
      else if (e.key === ']') { placeDist = Math.min(8, placeDist + 0.5); } });
  }
  const renderList = () => { if (panel && panel._renderList) panel._renderList(); };

  // ---- AR button (top-right): visual parity with the tour; real AR lives on the published tour --------------------------
  const arEl = el('button', { id: 'pl-ar', className: 'pl-panel' }, '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M12 3l8 4.5v9L12 21l-8-4.5v-9L12 3z"/><path d="M12 12l8-4.5M12 12v9M12 12L4 7.5"/></svg><span>AR</span>');
  arEl.onclick = () => toast('On-site AR works on the published tour (open it on an Android phone standing in the room). This is the local demo.');
  document.body.append(arEl);

  // ---- toast ------------------------------------------------------------------
  let toastEl = null, toastT = 0;
  function toast(msg) { if (!toastEl) { toastEl = el('div', { className: 'pl-panel' }); Object.assign(toastEl.style, { position: 'fixed', left: '50%', top: '18%', transform: 'translateX(-50%)', zIndex: 35, padding: '13px 18px', maxWidth: 'min(88vw,420px)', textAlign: 'center', font: '14px Arial,sans-serif', display: 'none' }); document.body.append(toastEl); }
    toastEl.textContent = msg; toastEl.style.display = 'block'; clearTimeout(toastT); toastT = setTimeout(() => { toastEl.style.display = 'none'; }, 4500); }

  // ---- floor-plan minimap (top-left): walkable area, POIs, and the camera cone ----------------------------------------
  let plan = null;
  if (frame && layers && layers.standable) { const n = frame.navigation, cell = n.cell_size_m, [x0, z0] = n.origin_floor_coordinates, W = n.width, H = n.height, st = layers.standable;
    let minx = W, miny = H, maxx = 0, maxy = 0, any = false;
    for (let gy = 0; gy < H; gy++) for (let gx = 0; gx < W; gx++) if (st[gy * W + gx]) { any = true; if (gx < minx) minx = gx; if (gx > maxx) maxx = gx; if (gy < miny) miny = gy; if (gy > maxy) maxy = gy; }
    if (any) { const cols = maxx - minx + 1, rows = maxy - miny + 1, pad = 1;
      const svgns = 'http://www.w3.org/2000/svg';
      const svg = document.createElementNS(svgns, 'svg'); svg.setAttribute('viewBox', `${-pad} ${-pad} ${cols + 2 * pad} ${rows + 2 * pad}`);
      const mk = (name, attrs, parent) => { const e = document.createElementNS(svgns, name); for (const k in attrs) e.setAttribute(k, attrs[k]); (parent || svg).append(e); return e; };
      const floorG = mk('g', { class: 'floor' }); for (let gy = miny; gy <= maxy; gy++) for (let gx = minx; gx <= maxx; gx++) if (st[gy * W + gx]) mk('rect', { x: gx - minx, y: gy - miny, width: 1.02, height: 1.02 }, floorG);
      const dotsG = mk('g', {}); const cone = mk('polygon', { class: 'cone', points: '' }); const me = mk('circle', { class: 'me', r: 1.1, cx: -99, cy: -99 });
      const box = el('div', { id: 'pl-plan', className: 'pl-panel' }); const wrap = el('div', { className: 'wrap' }); wrap.append(svg); box.append(wrap, el('div', { className: 'bar' }, '<span>Floor plan</span>')); document.body.append(box);
      ['pointerdown', 'pointerup', 'click', 'wheel'].forEach((t) => box.addEventListener(t, (e) => e.stopPropagation()));
      const toPlan = (nx, nz) => [(nx - x0) / cell - minx + 0.5, (nz - z0) / cell - miny + 0.5];
      const nav3 = new THREE.Vector3(), fwdv = new THREE.Vector3();
      plan = { update() {
        dotsG.textContent = '';
        pois.forEach((p) => { const nv = frame.viewerToNav(p.pos); const [px, py] = toPlan(nv[0], nv[2]); const c = mk('circle', { class: 'poi' + (p.category === 'help' ? ' help' : ''), r: 1, cx: px, cy: py }, dotsG); c.onclick = (e) => { e.stopPropagation(); if (admin) editSel(p.id); else activate(p); }; c.style.cursor = 'pointer'; });
        camera.getWorldPosition(nav3); const nv = frame.viewerToNav([nav3.x, nav3.y, nav3.z]); const [mx, my] = toPlan(nv[0], nv[2]); me.setAttribute('cx', mx); me.setAttribute('cy', my);
        fwdv.set(0, 0, -1).applyQuaternion(camera.quaternion); const nd = frame.viewerDirToNav([fwdv.x, fwdv.y, fwdv.z]); const a = Math.atan2(nd[2], nd[0]), L = 2.6, w = 0.5;
        const pt = (ang) => `${(mx + Math.cos(ang) * L).toFixed(2)},${(my + Math.sin(ang) * L).toFixed(2)}`; cone.setAttribute('points', `${mx.toFixed(2)},${my.toFixed(2)} ${pt(a - w)} ${pt(a + w)}`);
      } };
    }
  }

  // ---- how-to card, shown once on entry (like the tour's "How to explore") -------------------------------------------
  const howto = el('div', { id: 'pl-howto' });
  howto.innerHTML = `<div class="box pl-panel"><h1>How to explore</h1><p class="sub">Walk through the reconstructed room like a game.</p>
    <h3>Walk</h3>
    <div class="row"><span>Move</span><span class="keys"><kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd></span></div>
    <div class="row"><span>Look around</span><span>click, then mouse</span></div>
    <div class="row"><span>Run</span><span class="keys"><kbd>Shift</kbd></span></div>
    <div class="row"><span>Fly / walk</span><span class="keys"><kbd>F</kbd></span></div>
    <h3>Find a place</h3>
    <div class="row"><span>Help points</span><span>tap an amber “?”</span></div>
    <div class="row"><span>Search</span><span>the box at the bottom</span></div>
    ${admin ? '<div class="row"><span>Add a help point</span><span class="keys"><kbd>P</kbd></span></div>' : ''}
    <button class="go">Start exploring</button></div>`;
  document.body.append(howto);
  ['pointerdown', 'pointerup', 'click', 'wheel'].forEach((t) => howto.addEventListener(t, (e) => e.stopPropagation()));
  let howtoSeen = false;
  const showHowTo = () => { if (howtoSeen) return; howtoSeen = true; howto.style.display = 'flex'; };
  howto.querySelector('.go').onclick = () => { howto.style.display = 'none'; };
  howto.addEventListener('click', (e) => { if (e.target === howto) howto.style.display = 'none'; });

  drawPins();
  // ---- per-frame: keep pins, the open card, and the minimap glued to the 3D view ----
  function update() { placePins(); if (cardId != null) positionCard(); if (plan) plan.update(); }

  return { update, showHowTo, api: { list: () => pois.slice(), add: (p) => { const q = addPoi(p); drawPins(); renderList(); return q; }, open: openCard, close: closeCard } };
}
