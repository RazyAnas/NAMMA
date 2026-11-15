// Places in the tour page: pins in the 3D view and on the floor plan, search, a walkable route with the distance left, the
// arrival message, and the admin mode (?admin=1) that adds / edits / saves them. make_tour.py puts this file inside OVERLAY's
// module script, in one block after poi_core.js, so it uses that script's o, P, el, mini, big, camState, go, obs, q, find,
// draw3d ... and core's SpatialIndex, decodeNav, route, makePoi ... directly. Nothing here runs for a tour without
// nav.json / pois.json except the (empty) hooks, and every step is guarded: a fault here must never break the tour.
const $ = (id) => document.getElementById(id), qs = new URLSearchParams(location.search);
const store = { get: (k) => { try { return localStorage.getItem(k); } catch (e) { return null; } }, set: (k, v) => { try { v == null ? localStorage.removeItem(k) : localStorage.setItem(k, v); } catch (e) {} } };
// admin mode: ?admin=1. The Hugging Face sign-in comes back to this page with code + state; should the admin flag be lost on
// the way, a mark left in this tab before leaving brings it back for that one load.
let admin = qs.get('admin') === '1';
try { if (!admin && qs.has('code') && qs.has('state') && sessionStorage.getItem('nammaAdmin') === '1') admin = true; sessionStorage.removeItem('nammaAdmin'); } catch (e) {}
const getJson = async (u) => { try { const r = await fetch(u + '?v=' + Date.now(), { cache: 'no-store' }); return r.ok ? await r.json() : null; } catch (e) { return null; } };
const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

// ---- data: the walkable grid, the places, one spatial index over places and object boxes ----
let nav = null; { const j = await getJson('./nav.json'); if (j && j.grid && typeof DecompressionStream !== 'undefined') { try { nav = await decodeNav(j); } catch (e) { console.warn('nav.json unreadable: no routes', e); } } }
const live = await getJson('./pois.json'), DRAFT = 'nammaPois:' + location.pathname;
const index = new SpatialIndex(); let pois = [], dirty = false, base = (live && live.updated) || null, removed = (live && Array.isArray(live.removed) ? live.removed.filter((x) => typeof x === 'string') : []);
obs.forEach((x) => { if (x.lo && x.hi) index.insert({ id: 'o:' + x.id, lo: x.lo, hi: x.hi, obj: x }); });
{ let list = readPois(live);
  if (admin) { try { const d = JSON.parse(store.get(DRAFT) || 'null'); if (d && d.dirty && Array.isArray(d.pois)) { list = readPois(d); dirty = true; base = d.base || base; if (Array.isArray(d.removed)) removed = d.removed; } } catch (e) {} }   // unsaved edits survive a reload
  pois = list; pois.forEach((p) => index.insert({ id: p.id, pos: p.pos, poi: p })); }
const keep = () => { if (admin) store.set(DRAFT, dirty ? JSON.stringify({ dirty: true, base, pois, removed }) : null); };
const byId = (id) => pois.find((p) => p.id === id);
const ends = (t) => t.approach || [t.pos[0], t.pos[2]];                                      // where a route to it stops
const feet = (ix, iz) => (nav ? floorAt(nav, ix, iz) : 0);
const listeners = []; const changed = () => { drawDots(); drawPins(); listeners.forEach((f) => { try { f(pois); } catch (e) {} }); if (panel) panel.list(); };

// ---- toast (arrival and plain messages) ----
const toastEl = $('nm-toast'); let toastT = 0;
const toast = (msg, ok, ms = 4500) => { toastEl.textContent = msg; toastEl.classList.toggle('ok', !!ok); toastEl.style.display = 'block'; clearTimeout(toastT); toastT = setTimeout(() => { toastEl.style.display = 'none'; }, ms); };

// ---- 3D view: world point -> screen (the camera's own projection), for pins and the route line ----
const camC = () => (window.app && window.app.root ? window.app.root.findComponents('camera')[0] : null);
const view = () => (window.app && window.app.graphicsDevice ? window.app.graphicsDevice.canvas : null);          // the viewer's canvas (not the 3D floor plan's)
let V3 = null, vin = null, vout = null;
const project = (x, y, z) => { const c = camC(); if (!c) return null; const e = c.entity, p = e.getPosition(), fw = e.forward;
  const d = (x - p.x) * fw.x + (y - p.y) * fw.y + (z - p.z) * fw.z; if (d < 0.12) return null;                 // behind the camera
  if (!V3) { V3 = p.constructor; vin = new V3(); vout = new V3(); } vin.set(x, y, z); c.worldToScreen(vin, vout);
  return Number.isFinite(vout.x) && Number.isFinite(vout.y) ? [vout.x, vout.y, d] : null; };
// a screen point -> ray from the camera (as the object click of OVERLAY)
const rayAt = (sx, sy) => { const c = camC(); if (!c) return null; const a = c.entity.getPosition(), p = c.screenToWorld(sx, sy, 10);
  const d = [p.x - a.x, p.y - a.y, p.z - a.z], L = Math.hypot(d[0], d[1], d[2]) || 1; return { o: [a.x, a.y, a.z], d: [d[0] / L, d[1] / L, d[2] / L] }; };

// ---- pins ----
const pinsEl = $('nm-pins'), pinEl = new Map(); let goal = null, sel = null;
function drawPins() { const seen = new Set();
  pois.forEach((p) => { seen.add(p.id); let b = pinEl.get(p.id);
    if (!b) { b = document.createElement('button'); b.className = 'nm-pin'; b.innerHTML = '<span></span><i></i>'; pinsEl.appendChild(b); pinEl.set(p.id, b);
      b.onclick = (e) => { e.stopPropagation(); if (admin && panel) panel.edit(p.id); else routeTo(p.id); }; }
    b.firstChild.textContent = p.name; b.title = p.note || p.name; b.classList.toggle('ai', p.source === 'ai'); });
  pinEl.forEach((b, id) => { if (!seen.has(id)) { b.remove(); pinEl.delete(id); } }); }
function placePins() { pinEl.forEach((b, id) => { const p = byId(id); const s = p && project(p.pos[0], p.pos[1], p.pos[2]);
    const isGoal = !!(goal && goal.id === id); b.classList.toggle('goal', isGoal || (arrived && arrived.id === id)); b.classList.toggle('sel', sel === id);
    if (!s || (s[2] > 30 && !isGoal)) { b.style.transform = 'translate(-9999px,0)'; return; }
    b.classList.toggle('far', s[2] > 9 && !isGoal && sel !== id); b.style.opacity = isGoal ? 1 : Math.max(0.45, Math.min(1, 1.25 - s[2] / 20));
    b.style.transform = `translate(${Math.round(s[0] - b.offsetWidth / 2)}px,${Math.round(s[1] - b.offsetHeight)}px)`; }); }

// ---- floor plan: dots for places, the route line (both the minimap and the big plan; the 3D plan through extra3d) ----
const plans = [mini, big].map((m) => ({ m, g: el('g', {}, null), line: el('polyline', { class: 'route', 'stroke-width': Math.max(3, 0.09 * f.px) }, null) }));
plans.forEach((pl) => { pl.m.sv.insertBefore(pl.line, pl.m.cone); pl.m.sv.insertBefore(pl.g, pl.m.cone); });
function drawDots() { plans.forEach((pl, k) => { pl.g.textContent = ''; const r = Math.max(5, 0.11 * f.px) * (k ? 1 : 1.25);
    pois.forEach((p) => { const [x, y] = P(p.pos[0], p.pos[2]);
      const c = el('circle', { cx: x, cy: y, r, 'stroke-width': Math.max(2, r * 0.45), class: 'poi' + (p.source === 'ai' ? ' ai' : '') + (goal && goal.id === p.id ? ' goal' : '') }, pl.g);
      c.onclick = (e) => { e.stopPropagation(); closeFp(); if (admin && panel) panel.edit(p.id); else routeTo(p.id); };
      const t = el('title', {}, c); t.textContent = p.name;
      if (k) { const lb = el('text', { x, y: y - r - 5, 'text-anchor': 'middle', class: 'lbl' }, pl.g); lb.textContent = p.name; } }); }); }
extra3d = (g, pr) => { if (R && R.pts && R.pts.length > 1) { g.beginPath(); R.pts.forEach((p, i) => { const s = pr(p[0], 0.05, p[2]); i ? g.lineTo(s[0], s[1]) : g.moveTo(s[0], s[1]); }); g.lineWidth = 4; g.strokeStyle = '#e8a96b'; g.stroke(); }
  pois.forEach((p) => { const s = pr(p.pos[0], 1.0, p.pos[2]); g.beginPath(); g.arc(s[0], s[1], 6, 0, 7); g.fillStyle = goal && goal.id === p.id ? '#e8a96b' : '#fff'; g.fill(); g.lineWidth = 2.5; g.strokeStyle = p.source === 'ai' ? '#7fc8ff' : '#e8a96b'; g.stroke();
    g.font = 'bold 12px Arial, sans-serif'; g.textAlign = 'center'; g.lineWidth = 3; g.strokeStyle = '#000b'; g.strokeText(p.name, s[0], s[1] - 11); g.fillStyle = '#fff'; g.fillText(p.name, s[0], s[1] - 11); }); };

// ---- route: A* on the walkable grid from where the camera is, redone as the visitor moves; distance left; arrival ----
// goal: {id, name, pos, approach?, view?: annotation index of an object}. R: {pts, fine, len} of the current route.
const bar = $('nm-route'), rname = $('nm-rname'), rdist = $('nm-rdist'), rarr = $('nm-rarr'), rview = $('nm-rview'), rgo = $('nm-rgo'), pathEl = $('nm-path'), ribH = pathEl.children[0], chH = pathEl.children[1], ribV = pathEl.children[2], chV = pathEl.children[3], behind = $('nm-behind');
let R = null, arrived = null, lastPlan = 0, lastAt = null;
const ARRIVE = 1.0, BAND = 0.55;                                                           // metres: arrival distance, the band's width
//                                                                         // metres from the end point
function stopRoute(silent) { stopAuto(); goal = null; R = null; bar.style.display = 'none'; [ribH, chH, ribV, chV].forEach((x) => x.setAttribute('d', '')); behind.style.display = 'none'; plans.forEach((pl) => pl.line.setAttribute('points', '')); drawDots(); if (!silent && fp.style.display === 'flex' && mode === '3d') draw3d(); }
function startRoute(t) { arrived = null; goal = t; R = null; lastPlan = 0; lastAt = null; rname.textContent = t.name; rdist.textContent = ''; rview.style.display = t.view != null ? '' : 'none';
  bar.style.display = 'flex'; drawDots(); if (!tick(true)) return false;
  const v = window.viewer, st = v && v.state;
  if (st && st.cameraMode === 'anim') { try { st.cameraMode = 'walk'; } catch (e) {} }
  if (R && R.pts && st && st.cameraMode !== 'walk') toast('Follow the line. Walk mode: the person button.', false, 3500);
  return true; }
function fineOf(pts) { const out = []; for (let i = 1; i < pts.length; i++) { const a = pts[i - 1], b = pts[i], n = Math.max(1, Math.ceil(Math.hypot(b[0] - a[0], b[2] - a[2]) / 0.25));
    for (let k = i === 1 ? 0 : 1; k <= n; k++) out.push([a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n + 0.04, a[2] + (b[2] - a[2]) * k / n]); } return out; }
// one step of the route loop; false when the route had to be given up
function tick(force) { if (!goal) return false; const c = camState(); if (!c) return true; const e = ends(goal), now = Date.now();
  const moved = !lastAt || Math.hypot(c.x - lastAt[0], c.z - lastAt[1]) > 0.25;
  if (nav && (force || (moved && now - lastPlan > 400))) { lastPlan = now; lastAt = [c.x, c.z]; const r = route(nav, [c.x, c.z], e);
    if (r.ok) { const fine = fineOf(r.points); R = { pts: r.points, fine, len: r.length + r.from, rb: ribbon(fine, BAND), hid: null }; rgo.style.display = canAuto() ? '' : 'none'; plans.forEach((pl) => pl.line.setAttribute('points', r.points.map((p) => P(p[0], p[2]).map((v) => v.toFixed(1)).join(',')).join(' '))); }
    else if (!R) { const nm = goal.name; stopRoute();
      toast(r.reason === 'unreachable' ? `No walkable way to ${nm} from here.` : r.reason === 'target-off-floor' ? `${nm} is not next to any floor that was filmed: no route.` : 'No walkable floor in this tour: no route.', false, 6000); return false; } }
  else if (!nav) R = { pts: null, fine: null, len: Math.hypot(c.x - e[0], c.z - e[1]) };          // no grid: straight-line distance only
  const d = Math.hypot(c.x - e[0], c.z - e[1]), len = R ? R.len : d;
  rdist.textContent = (len < 10 ? len.toFixed(1) : Math.round(len)) + ' m' + (nav ? ' · ' + walkTime(len) : ' (straight line)');
  if (R && R.fine) R.hid = hiddenParts(R.fine, [c.x, c.z], walls, nav);                     // which of it is behind a wall from here
  const nx = R && R.pts ? (R.pts.find((p) => Math.hypot(p[0] - c.x, p[2] - c.z) > 0.6) || R.pts[R.pts.length - 1]) : [e[0], 0, e[1]];      // the next turn: where the arrow points
  rarr.style.transform = `rotate(${(Math.atan2(nx[2] - c.z, nx[0] - c.x) - Math.atan2(c.fz, c.fx)) * 180 / Math.PI}deg)`;
  let level = true; if (nav && nav.floor && R && R.pts && window.viewer && window.viewer.state.cameraMode === 'walk') { const w = window.nammaWalk, eh = w ? (w.eyeHeight || 1.3) + (w.hoverHeight || 0.2) : 1.5;
    level = Math.abs(c.y - eh - R.pts[R.pts.length - 1][1]) < 1.2; }                                              // standing a floor above or below it is not there
  if (d <= ARRIVE && len <= ARRIVE + 1.5 && level) { const t = goal; stopRoute(true); arrived = t; toast(`You have reached your destination: ${t.name}`, true, 6000); setTimeout(() => { if (arrived === t) arrived = null; }, 6000);
    api.onArrive.forEach((fn) => { try { fn(t); } catch (er) {} }); return false; }
  return true; }
setInterval(() => { try { if (goal) tick(false); } catch (e) { console.warn(e); } }, 200);
// "Take me there": walks the route for the visitor through the viewer's own controls, the way holding the stick forward
// would (viewer.setMoveInput, every frame) while turning to the next point of the route (the walk controller's target yaw).
// The viewer's collision and floor stay in charge, so it cannot pass a wall; it stops on arrival, on any input of the
// visitor's, when leaving walk mode, and when 3 s of pushing forward make no headway (something the grid did not know).
let auto = false, autoT = 0, autoPush = 0, autoFrom = null, autoOn = 0;
const canAuto = () => { const v = window.viewer, w = window.nammaWalk; return !!(nav && v && typeof v.setMoveInput === 'function' && w && w._targetAngles && typeof w._targetAngles.y === 'number'); };
function stopAuto(why) { if (!auto) return; auto = false; rgo.classList.remove('on'); rgo.textContent = 'Take me there'; try { window.viewer.setMoveInput(0, 0); } catch (e) {} if (why) toast(why, false, 4000); }
function startAuto() { if (!goal || !canAuto()) return; const st = window.viewer.state; if (st.cameraMode !== 'walk') { try { st.cameraMode = 'walk'; } catch (e) {} }
  auto = true; autoOn = autoT = performance.now(); autoPush = 0; autoFrom = null; rgo.classList.add('on'); rgo.textContent = 'Stop'; }
function autoStep() { const v = window.viewer, w = window.nammaWalk, c = camState(), now = performance.now(), dt = Math.min(0.25, (now - autoT) / 1000); autoT = now;
  if (!goal || !R || !R.pts || !c || !canAuto()) { stopAuto(); return; }
  if (v.state.cameraMode !== 'walk') { if (now - autoOn > 2500) stopAuto(); return; }                             // (walk mode takes a moment to start)
  const e = ends(goal), nx = R.pts.find((p) => Math.hypot(p[0] - c.x, p[2] - c.z) > 0.5) || [e[0], 0, e[1]];
  const turn = ((Math.atan2(-(nx[0] - c.x), -(nx[2] - c.z)) * 180 / Math.PI - w._targetAngles.y) % 360 + 540) % 360 - 180;   // shortest way round
  w._targetAngles.y += Math.max(-360 * dt, Math.min(360 * dt, turn)); if (Math.abs(w._targetAngles.x) > 12) w._targetAngles.x *= 0.9;   // a full turn a second at most; look level
  const push = Math.abs(turn) < 40; v.setMoveInput(0, push ? 0.45 : 0);                                         // turn first, then walk
  if (!autoFrom || Math.hypot(c.x - autoFrom[0], c.z - autoFrom[1]) > 0.3) { autoFrom = [c.x, c.z]; autoPush = 0; }
  else if (push && (autoPush += Math.min(dt, 0.1)) > 3) stopAuto('Something is in the way here. Walk around it, the line will follow.'); }   // 3 s of pushing, no headway
rgo.onclick = (e) => { e.stopPropagation(); auto ? stopAuto() : startAuto(); };
addEventListener('keydown', (e) => { if (auto && /^(Key[WASD]|Arrow|Space|Escape)/.test(e.code || '')) stopAuto(); }, true);
addEventListener('pointerdown', (e) => { if (auto && e.target && e.target.tagName === 'CANVAS') stopAuto(); }, true);
$('nm-rx').onclick = (e) => { e.stopPropagation(); stopRoute(); }; rview.onclick = (e) => { e.stopPropagation(); if (goal && goal.view != null) go(goal.view); };
// the 3D view, every frame: the route as a band on the floor with arrow heads sliding the way to go, drawn in perspective (each
// edge point through the camera's own projection); the pins. What the visitor cannot see from where they stand (a wall in
// between) is still drawn, since it is a guide, but as a dashed outline with hollow arrows and a tag where it goes out of sight.
function drawBand() { const { L, R: Rr, s: dist } = R.rb, n = R.fine.length, hid = R.hid || [], W = innerWidth, H = innerHeight;
  const ok = (q) => q && q[2] < 45 && q[0] > -W && q[0] < 2 * W && q[1] > -H && q[1] < 2 * H, xy = (q) => q[0].toFixed(1) + ' ' + q[1].toFixed(1);
  const pl = new Array(n), pr = new Array(n);
  for (let i = 0; i < n; i++) { const a = project(L[i][0], L[i][1], L[i][2]), b = project(Rr[i][0], Rr[i][1], Rr[i][2]); pl[i] = ok(a) ? a : null; pr[i] = ok(b) ? b : null; }
  let dv = '', dh = '', tag = null, i = 0;
  while (i < n) { if (!pl[i] || !pr[i]) { i++; continue; } const h = !!hid[i]; let j = i; while (j + 1 < n && pl[j + 1] && pr[j + 1] && !!hid[j + 1] === h) j++;
    const e = j + 1 < n && pl[j + 1] && pr[j + 1] ? j + 1 : j;                                    // up to the first point of the next part: no gap between them
    if (e > i) { let d = 'M' + xy(pl[i]); for (let k = i + 1; k <= e; k++) d += 'L' + xy(pl[k]); for (let k = e; k >= i; k--) d += 'L' + xy(pr[k]); d += 'Z'; if (h) dh += d; else dv += d; }
    if (h && !tag && i > 0 && !hid[i - 1]) tag = [(pl[i][0] + pr[i][0]) / 2, (pl[i][1] + pr[i][1]) / 2];
    i = j + 1; }
  ribV.setAttribute('d', dv); ribH.setAttribute('d', dh);
  let cv = '', ch = '';
  for (const c of chevrons(R.fine, dist, BAND, 1.0, (performance.now() / 1000 * 0.5) % 1.0)) { const q = c.pts.map((p) => project(p[0], p[1], p[2])); if (!q.every(ok)) continue;
    const d = 'M' + q.map(xy).join('L') + 'Z'; if (hid[c.i]) ch += d; else cv += d; }
  chV.setAttribute('d', cv); chH.setAttribute('d', ch);
  if (tag) { behind.style.display = 'block'; behind.style.transform = `translate(${tag[0].toFixed(0)}px,${tag[1].toFixed(0)}px) translate(-50%,-160%)`; } else behind.style.display = 'none'; }
function frame() { try { placePins(); if (auto) autoStep(); if (R && R.fine && R.rb) drawBand(); } catch (e) {} requestAnimationFrame(frame); }
requestAnimationFrame(frame);

function routeTo(id) { const p = byId(id); if (!p) return false; return startRoute(p); }
function routeToPoint(name, pos, view) { let approach = null; if (nav) { const s = snap(nav, pos[0], pos[2], 3); if (s) approach = centre(nav, s.ix, s.iz); }
  return startRoute({ id: null, name, pos, approach, view }); }

// ---- search: places first, then the object layer; choosing one starts the route (objects: fly there when there is no grid) ----
const hitOf = new Map(), catName = (p) => p.category.charAt(0).toUpperCase() + p.category.slice(1);
extra = () => pois.map((p) => { let h = hitOf.get(p); const size = catName(p) + (p.note ? ' · ' + esc(p.note).slice(0, 60) : '') + (p.source === 'ai' ? ' · found by AI' : '');
  if (!h || h.label !== esc(p.name) || h.size !== size) { h = { poi: p, label: esc(p.name), size, words: poiWords(p), thumb: p.thumb }; hitOf.set(p, h); } return h; });
pick = (x) => { if (x.ai) aiFind(x.ai); else if (x.poi) routeTo(x.poi.id); else if (nav && x.lo && x.hi) routeToPoint(x.label, [(x.lo[0] + x.hi[0]) / 2, (x.lo[1] + x.hi[1]) / 2, (x.lo[2] + x.hi[2]) / 2], x.i); else go(x.i); };
const searchOn = () => { find.style.display = 'block'; q.placeholder = aiMeta ? 'Search, or ask for anything ...' : obs.length ? 'Search places and things ...' : 'Search places ...'; };

// ---- AI find: type anything; the tour's own frames are searched for it (regions.json / regions.f16 from ai_find.py), and it
// is marked only when two filmed frames agree on where it is. The words are embedded in the browser with CLIP's text half
// (transformers.js, ~65 MB once, then cached); nothing is sent anywhere.
const aiMeta = await getJson('./regions.json'); let aiR = null, aiEmbed = null, aiBusy = false;
const TJS = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';
const slug = (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 40) || 'thing';
async function aiLoad() {
  if (!aiR) { const r = await fetch('./regions.f16'); if (!r.ok) throw new Error('no search index in this tour'); aiR = decodeRegions(aiMeta, await r.arrayBuffer()); }
  if (!aiEmbed) { const T = await import(TJS), id = 'Xenova/' + String(aiMeta.model || 'openai/clip-vit-base-patch32').split('/').pop();
    const tok = await T.AutoTokenizer.from_pretrained(id), model = await T.CLIPTextModelWithProjection.from_pretrained(id, { dtype: 'q8' });
    aiEmbed = async (text) => { const out = await model(tok([text], { padding: true, truncation: true })), e = Array.from(out.text_embeds.data).slice(0, aiR.dim);
      const n = Math.hypot(...e) || 1; return e.map((v) => v / n); }; }
}
async function aiFind(text) { text = String(text || '').trim(); if (!text || !aiMeta) return null; if (aiBusy) { toast('Still looking ...'); return null; }
  aiBusy = true; toast(aiEmbed ? 'Looking for “' + text + '” ...' : 'Getting the AI model (about 65 MB, the first time only) ...', false, 60000);
  try { await aiLoad(); const r = aiSearch(await aiEmbed(aiR.prompt.replace('{}', text)), aiR); api.lastAi = Object.assign({ query: text }, r);
    if (!r.found) { toast('“' + text + '” was not found in this tour (nothing was seen from two sides).'); return r; }
    const p = addPoi({ id: 'ai-' + slug(text), name: text.charAt(0).toUpperCase() + text.slice(1), pos: r.found.pos, source: 'ai', category: 'other',
      confidence: r.found.frames.length >= 3 ? 0.8 : 0.6, note: 'found by AI in ' + r.found.frames.length + ' filmed views' });
    toast('Found: ' + p.name, true); routeTo(p.id); return r;
  } catch (e) { console.warn('AI find', e); toast('AI find could not run here: ' + (e && e.message ? e.message : e)); return null; } finally { aiBusy = false; } }
if (aiMeta && aiMeta.count) more = (s) => (s.length >= 3 ? [{ ai: s, label: 'Find “' + esc(s) + '” with AI', size: 'looks through the filmed frames', words: '' }] : []);
// places the find agent marked on Kaggle (pois_ai.json): part of the tour unless the admin removed or re-placed them
{ const a = await getJson('./pois_ai.json'); ((a && a.pois) || []).forEach((p) => { if (p && p.id && Array.isArray(p.pos) && !byIdEarly(p.id) && !removed.includes(p.id))
    { const poi = makePoi({ id: p.id, name: p.name, pos: p.pos, source: 'ai', category: 'other', thumb: p.thumb, confidence: p.confidence === 'high' ? 0.9 : p.confidence === 'medium' ? 0.7 : 0.4,
        note: p.review ? 'found by AI in one view: check it' : 'found by AI' }, nav); pois.push(poi); index.insert({ id: poi.id, pos: poi.pos, poi }); } }); }
function byIdEarly(id) { return pois.find((p) => p.id === id); }
if (pois.length || admin || aiMeta) searchOn();

// ---- hooks (AI find, tests, the console) ----
function addPoi(p, quiet) { const poi = makePoi(p, nav); const old = byId(poi.id); if (old) { index.remove(old.id); pois[pois.indexOf(old)] = poi; } else pois.push(poi);
  index.insert({ id: poi.id, pos: poi.pos, poi }); removed = removed.filter((x) => x !== poi.id); if (admin) { dirty = true; keep(); } searchOn(); if (!quiet) changed(); return poi; }
function removePoi(id) { const p = byId(id); if (!p) return false; index.remove(id); pois.splice(pois.indexOf(p), 1); if (!removed.includes(id)) removed.push(id); if (goal && goal.id === id) stopRoute(); if (admin) { dirty = true; keep(); } changed(); return true; }
const api = window.nammaPoi = { version: 1, admin, onArrive: [],
  list: () => pois.slice(), get: byId, nav: () => nav, index,
  aiFind, lastAi: null, add: (p) => addPoi(p),                                                                    // {name, pos [x, y, z], category?, note?, thumb?}
  addAi: (p) => addPoi(Object.assign({}, p, { source: 'ai' })),                               // + confidence 0..1; shown blue, "found by AI"
  remove: removePoi, routeTo, routeToPoint, clearRoute: () => stopRoute(), takeMeThere: (on) => (on === false ? stopAuto() : startAuto()), walking: () => auto,
  nearest: (pos, max) => index.nearest(pos, max == null ? Infinity : max), within: (pos, r) => index.within(pos, r),
  pick: (sx, sy) => { const r = rayAt(sx, sy); return r ? index.rayPick(r.o, r.d, 30) : null; },
  onChange: (fn) => { listeners.push(fn); }, route: () => (goal ? { goal, length: R ? R.len : null, points: R ? R.pts : null } : null), file: () => poisFile(pois, Date.now(), removed) };

// ---- admin (?admin=1): drop / edit / move / delete, save to the live tour ----
let panel = null;
{ const b = $('nm-adminbtn'); if (b) { b.style.color = admin ? 'var(--nm-acc)' : ''; b.title = admin ? 'Leave admin mode' : 'Admin: places and collision';
    b.onclick = (e) => { e.stopPropagation(); const u = new URL(location.href); if (admin) u.searchParams.delete('admin'); else u.searchParams.set('admin', '1'); location.href = u.toString(); }; } }
if (admin) {
  const HUB = 'https://cdn.jsdelivr.net/npm/@huggingface/hub@2.11.2/+esm';                    // as v3/collision_editor.html
  const d = document.createElement('div'); d.id = 'nm-admin'; d.className = 'nm-panel';
  d.innerHTML = `<h1><span>Places <small id="nm-adirty"></small></span><button class="nm-btn" id="nm-ax" title="Leave admin mode"><svg width="20" height="20" viewBox="0 0 24 24"><g class="s"><use href="#nmCloseIcon"/></g><g class="f"><use href="#nmCloseIcon"/></g></svg></button></h1>
    <div class="r"><button class="nm-pill" id="nm-atap">Tap to place</button><button class="nm-pill" id="nm-actr" title="Pin where the middle of the screen points (P)">Pin at centre</button></div>
    <div class="list" id="nm-alist"></div>
    <div class="sec" id="nm-aform" style="display:none"><label>Name<input id="nm-aname" maxlength="80"></label>
      <label>Kind<select id="nm-acat">${CATEGORIES.map((c) => `<option value="${c}">${c.charAt(0).toUpperCase() + c.slice(1)}</option>`).join('')}</select></label>
      <label>Note<textarea id="nm-anote" maxlength="400"></textarea></label>
      <div class="r"><button class="nm-pill" id="nm-amove">Move</button><button class="nm-pill" id="nm-atest">Route</button><button class="nm-pill" id="nm-adel">Delete</button></div></div>
    <div class="sec"><b style="color:var(--nm-light)">Collision</b>
      <label style="flex-direction:row;align-items:center;gap:8px;cursor:pointer"><input type="checkbox" id="nm-acol" style="width:auto;flex:none">Show what walk mode bumps into (objects and walls)</label>
      <div class="r"><button class="nm-pill" id="nm-aedit">Collision editor ↗</button></div>
      <div class="st">The editor works on the published tour (it signs in with Hugging Face and saves to the same link). Republish with "collision" afterwards so routes follow the new shape.</div></div>
    <div class="sec"><div class="r"><button class="nm-pill on" id="nm-apub">Save to the live tour</button><button class="nm-pill" id="nm-adl">Download</button></div>
      <div class="r" id="nm-aauth"><button class="nm-pill" id="nm-ain">Sign in with Hugging Face</button></div>
      <label id="nm-atokl" style="display:none">Or a write token (kept in this browser only)<input id="nm-atok" type="password" autocomplete="off" placeholder="hf_..."></label>
      <label>Space<input id="nm-aspace" placeholder="owner/namma-tour"></label>
      <div class="st" id="nm-ast"></div></div>`;
  document.querySelector('.nm-ui').appendChild(d);
  const st = $('nm-ast'), say = (s) => { st.textContent = s; }, form = $('nm-aform'), nameI = $('nm-aname'), catI = $('nm-acat'), noteI = $('nm-anote');
  d.querySelectorAll('input,select,textarea').forEach((x) => ['keydown', 'keyup', 'keypress'].forEach((t) => x.addEventListener(t, (e) => e.stopPropagation())));   // typing must not walk
  ['pointerdown', 'pointerup', 'click', 'wheel'].forEach((t) => d.addEventListener(t, (e) => e.stopPropagation()));
  const folder = location.pathname.split('/').filter(Boolean).filter((x) => !/\.html?$/.test(x)).join('/'), file = (folder ? folder + '/' : '') + 'pois.json';
  const spaceI = $('nm-aspace'), hv = (window.huggingface && window.huggingface.variables) || {};
  spaceI.value = store.get('nammaSpace') || hv.SPACE_ID || ''; spaceI.onchange = () => store.set('nammaSpace', spaceI.value.trim());
  let token = null; try { const t = JSON.parse(store.get('nammaHf') || 'null'); if (t && t.t && (!t.exp || t.exp > Date.now() + 60000)) token = t.t; } catch (e) {}
  const mark = () => { $('nm-adirty').textContent = dirty ? 'unsaved' : ''; $('nm-ain').textContent = token ? 'Signed in · sign out' : 'Sign in with Hugging Face'; $('nm-atokl').style.display = token ? 'none' : ''; };
  // place = where the ray first meets something: an object box or another pin (the spatial index), else the collision mesh
  // (the walk controller's capsule test, as a 3 cm ball marched along the ray), else the floor, else 2 m ahead
  const hitOfRay = (r) => { const hit = index.rayPick(r.o, r.d, 15, 0.2, (it) => !!it.obj); let t = null;
    const w = window.nammaWalk, col = w && w.collision, out = { x: 0, y: 0, z: 0 };
    if (col && typeof col.queryCapsule === 'function') { try { for (let k = 0.3; k <= 15; k += 0.04) if (col.queryCapsule(r.o[0] + r.d[0] * k, r.o[1] + r.d[1] * k, r.o[2] + r.d[2] * k, 0, 0.03, out)) { t = k; break; } } catch (e) { t = null; } }
    if (hit && (t == null || hit.t <= t + 0.05)) { const x = hit.item.obj; return { pos: [(x.lo[0] + x.hi[0]) / 2, (x.lo[1] + x.hi[1]) / 2, (x.lo[2] + x.hi[2]) / 2], name: x.label, thumb: x.thumb || null, how: 'object' }; }
    if (t != null) return { pos: [r.o[0] + r.d[0] * t, r.o[1] + r.d[1] * t, r.o[2] + r.d[2] * t], how: 'surface' };
    if (r.d[1] < -0.05) { let fy = 0; for (let n = 0; n < 3; n++) { const k = (fy - r.o[1]) / r.d[1], c = nav ? cellOf(nav, r.o[0] + r.d[0] * k, r.o[2] + r.d[2] * k) : null; fy = c ? feet(c[0], c[1]) : 0; }
      const k = (fy - r.o[1]) / r.d[1]; if (k > 0.2 && k < 15) return { pos: [r.o[0] + r.d[0] * k, fy, r.o[2] + r.d[2] * k], how: 'floor' }; }
    return { pos: [r.o[0] + r.d[0] * 2, r.o[1] + r.d[1] * 2, r.o[2] + r.d[2] * 2], how: 'ahead' }; };
  let placing = false, moving = null;
  const setPlacing = (on, mv) => { placing = on; moving = on ? mv || null : null; document.body.classList.toggle('nm-place', on); $('nm-atap').classList.toggle('on', on && !mv); $('nm-amove').classList.toggle('on', on && !!mv);
    if (on) say(mv ? 'Tap where it should be.' : 'Tap the spot in the 3D view.'); };
  const dropAt = (sx, sy) => { const r = rayAt(sx, sy); if (!r) { say('The 3D view is not ready yet.'); return; } const h = hitOfRay(r);
    if (moving) { const p = byId(moving); if (p) { addPoi(Object.assign({}, p, { pos: h.pos }), true); changed(); say(`Moved (${h.how}).`); panel.edit(p.id); } }
    else { const p = addPoi({ name: h.name || 'New place', pos: h.pos, thumb: h.thumb }); say(`Placed on ${h.how === 'ahead' ? 'nothing (2 m ahead): move it if needed' : 'the ' + h.how}${p.approach ? '' : nav ? '. No walkable floor within 3 m: no route to it.' : ''}.`); panel.edit(p.id); nameI.focus(); nameI.select(); }
    setPlacing(false); mark();
    // the viewer takes the same tap as "walk to this point" (its NavInteraction, on the canvas after us): call that walk off,
    // or the camera leaves while the admin types the name. A phone reports its tap a little later, hence the repeats.
    const ev = window.viewer && window.viewer.events; if (ev) [0, 60, 400].forEach((ms) => setTimeout(() => { try { ev.fire('navigateCancel'); } catch (e) {} }, ms)); };
  let dn = null;
  window.addEventListener('pointerdown', (e) => { dn = placing && e.target.tagName === 'CANVAS' ? [e.clientX, e.clientY, Date.now()] : null; }, true);
  window.addEventListener('pointerup', (e) => { if (!dn || !placing) return; const ok = Math.hypot(e.clientX - dn[0], e.clientY - dn[1]) <= 8 && Date.now() - dn[2] < 600; dn = null; if (!ok) return;
    const r = e.target.getBoundingClientRect(); dropAt(e.clientX - r.left, e.clientY - r.top); }, true);
  const centre2 = () => { const c = view(); if (!c) return; const r = c.getBoundingClientRect(); dropAt(r.width / 2, r.height / 2); };
  $('nm-atap').onclick = () => setPlacing(!(placing && !moving)); $('nm-actr').onclick = centre2;
  addEventListener('keydown', (e) => { if (/INPUT|TEXTAREA|SELECT/.test((e.target || {}).tagName || '')) return; if (e.code === 'KeyP') centre2(); else if (e.key === 'Escape' && placing) setPlacing(false); });
  const save = () => { const p = byId(sel); if (!p) return; const n = { ...p, name: nameI.value.trim() || 'Place', category: catI.value, note: noteI.value.trim() }; addPoi(n, true); changed(); mark(); };
  nameI.oninput = save; catI.onchange = save; noteI.oninput = save;
  $('nm-amove').onclick = () => { if (sel) setPlacing(!(placing && moving), sel); };
  $('nm-atest').onclick = () => { if (sel) routeTo(sel); };
  $('nm-adel').onclick = () => { const p = byId(sel); if (p && confirm(`Delete "${p.name}"?`)) { removePoi(p.id); sel = null; form.style.display = 'none'; mark(); say('Deleted.'); } };
  $('nm-ax').onclick = () => { const u = new URL(location.href); u.searchParams.delete('admin'); location.href = u.toString(); };
  // collision: the view of what walk mode bumps into (off for visitors since 1 Nov; on for the admin, who edits against it) and the
  // editor for it (v3/collision_editor.html, published beside the tours as ../editor/)
  const colI = $('nm-acol'), setCol = (on) => { const vs = window.viewer && window.viewer.state; if (!vs || !vs.hasCollisionOverlay) return false; try { vs.collisionOverlayEnabled = on; } catch (e) { return false; } return true; };
  colI.onchange = () => { if (!setCol(colI.checked)) { colI.checked = false; say('This tour has no collision view.'); } };
  { const t0 = Date.now(), k = setInterval(() => { const vs = window.viewer && window.viewer.state; if (vs && vs.loaded && vs.hasCollisionOverlay) { clearInterval(k); colI.checked = setCol(true); } else if (Date.now() - t0 > 90000) clearInterval(k); }, 500); }
  // the editor lives beside the tours on the Space and is told which tour this is (its folder, its run and copy: overlay.json src)
  $('nm-aedit').onclick = () => { if (!/\.hf\.space$/.test(location.hostname)) { say('The collision editor works on the published tour only: it signs in with Hugging Face and loads the mesh from your private dataset. Open this tour from its link and use this button there.'); return; }
    const u = new URL('../editor/index.html', location.href), fo = location.pathname.split('/').filter(Boolean).filter((x) => !/\.html?$/.test(x)).pop() || '';
    u.searchParams.set('folder', fo); u.searchParams.set('run', (o.src && o.src.run) || fo.replace(/-[0-9a-f]{20}$/, '')); if (o.src && o.src.dst) u.searchParams.set('dst', o.src.dst);
    window.open(u.toString(), '_blank', 'noopener'); };
  panel = { list() { const L = $('nm-alist'); L.textContent = ''; pois.forEach((p) => { const b = document.createElement('button'); b.className = p.id === sel ? 'on' : '';
        b.innerHTML = `<span>${esc(p.name)}</span><small>${esc(p.category)}${p.source === 'ai' ? ' · AI' : ''}${nav && !p.approach ? ' · no route' : ''}</small>`; b.onclick = () => panel.edit(p.id); L.appendChild(b); });
      if (!pois.length) L.innerHTML = '<div class="st">No places yet. Tap to place the first one.</div>'; mark(); },
    edit(id) { const p = byId(id); if (!p) return; sel = id; form.style.display = 'flex'; if (document.activeElement !== nameI) nameI.value = p.name; catI.value = p.category; if (document.activeElement !== noteI) noteI.value = p.note; panel.list(); } };
  // save. 1) straight to the live tour: one commit of pois.json to the Space through the Hub API (@huggingface/hub in the page,
  //    the way v3/collision_editor.html applies collision edits), signed in with Hugging Face or with a pasted write token;
  // 2) always possible: download pois.json (see the note it shows).
  const out = () => new Blob([JSON.stringify(poisFile(pois, Date.now(), removed), null, 1)], { type: 'application/json' });
  $('nm-adl').onclick = () => { const a = document.createElement('a'); a.href = URL.createObjectURL(out()); a.download = 'pois.json'; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 5000);
    say(`Downloaded. To put it live without signing in here: upload it as ${file} in the tour Space (Files tab), or commit it as colab/real_scenes/v3/pois/<run>.json and republish the tour.`); };
  const hub = () => import(HUB);
  $('nm-ain').onclick = async () => { if (token) { token = null; store.set('nammaHf', null); mark(); say('Signed out.'); return; }
    try { say('Opening the Hugging Face sign-in ...'); const h = await hub(), url = await h.oauthLoginUrl({ scopes: hv.OAUTH_SCOPES || 'openid profile write-repos' });
      try { sessionStorage.setItem('nammaAdmin', '1'); } catch (e) {} location.href = url; }
    catch (e) { say('Sign-in is not available here (' + (e.message || e) + '). Paste a write token below, or download the file.'); } };
  $('nm-atok').onchange = (e) => { const v = e.target.value.trim(); e.target.value = ''; if (!v) return; token = v; store.set('nammaHf', JSON.stringify({ t: v })); mark(); say('Token kept in this browser only.'); };
  $('nm-apub').onclick = async () => { if (!pois.length && !live) { say('Nothing to save yet.'); return; } if (!token) { say('Sign in first (or paste a write token), or use Download.'); return; }
    const b = $('nm-apub'); b.disabled = true;
    try { const h = await hub(); let name = spaceI.value.trim();
      if (!name) { const me = await h.whoAmI({ accessToken: token }), sub = location.hostname.split('.')[0], u = (me.name || '').toLowerCase();    // owner-space.static.hf.space
        if (u && sub.toLowerCase().startsWith(u + '-')) name = me.name + '/' + sub.slice(u.length + 1); }
      if (!/^[\w.-]+\/[\w.-]+$/.test(name)) throw new Error('fill in the Space as owner/name');
      spaceI.value = name; store.set('nammaSpace', name);
      const now = await getJson('./pois.json'); if (now && now.updated && base && now.updated !== base && !confirm('The live places changed since you opened this page. Replace them with yours?')) { say('Not saved.'); return; }
      say('Saving ...'); const body = poisFile(pois, Date.now(), removed);
      await h.uploadFiles({ repo: { type: 'space', name }, accessToken: token, files: [{ path: file, content: new Blob([JSON.stringify(body, null, 1)], { type: 'application/json' }) }],
        commitTitle: `Places: ${pois.length} in ${folder || 'tour'}` });
      dirty = false; base = body.updated; keep(); mark(); say(`Saved ${pois.length} places to ${name}/${file}. Visitors see them after the Space refreshes (about a minute).`); }
    catch (e) { say('Not saved: ' + (e.message || e) + '. Your edits are kept in this browser; Download still works.'); }
    finally { b.disabled = false; } };
  // back from the Hugging Face sign-in (code + state in the address): take the token, keep it in this browser only
  if (qs.has('code') && qs.has('state')) { try { const h = await hub(), a = await h.oauthHandleRedirectIfPresent();
      if (a && a.accessToken) { token = a.accessToken; store.set('nammaHf', JSON.stringify({ t: token, exp: a.accessTokenExpiresAt ? +new Date(a.accessTokenExpiresAt) : Date.now() + 8 * 3600e3 })); say('Signed in.'); } }
    catch (e) { say('Sign-in did not finish: ' + (e.message || e)); } }
  panel.list(); mark(); if (!st.textContent) say(dirty ? 'Unsaved edits from last time were restored.' : nav ? '' : 'This tour has no walkable grid (nav.json): places work, routes do not.');
}
drawPins(); drawDots();
