// Explore (#/explore): the public tours from gallery.json. Works without sign-in.
import { h, replace, icon, extLink, stateBox, fmtDate, fmtCount, fmtMb } from '../ui.js';

function card(e) {
  const facts = [
    e.splats ? `${fmtCount(e.splats)} splats` : '',
    e.area ? `${Math.round(e.area)} m²` : '',
    e.sizeMb ? fmtMb(e.sizeMb) : '',
  ].filter(Boolean);
  return h('li', { class: 'grid__item' }, extLink(e.url, { class: 'tour' },
    h('span', { class: 'tour__cover' }, e.cover ? h('img', { src: e.cover, alt: '', loading: 'lazy', decoding: 'async', referrerpolicy: 'no-referrer' }) : h('span', { class: 'tour__nocover' }, icon('cube', 30))),
    h('span', { class: 'tour__body' },
      h('span', { class: 'tour__title' }, e.title),
      h('span', { class: 'tour__meta' }, [e.kind, fmtDate(e.created), e.by ? `by ${e.by}` : ''].filter(Boolean).join(' · ')),
      facts.length ? h('span', { class: 'tour__meta' }, facts.join(' · ')) : null,
      h('span', { class: 'tour__open' }, 'Open tour', icon('open', 18), h('span', { class: 'sr-only' }, ' (opens in a new tab)')))));
}

export function render(ctx) {
  let all = [];
  let pick = '';          // chosen chip: '' = all
  let query = '';

  const chips = h('div', { class: 'chips', role: 'group', 'aria-label': 'Filter by kind' });
  const count = h('p', { class: 'muted', role: 'status' }, '');
  const body = h('div', { class: 'explore__body' });
  const search = h('input', { type: 'search', class: 'input input--search', placeholder: 'Filter by name', 'aria-label': 'Filter tours by name', autocomplete: 'off',
    oninput: () => { query = search.value.trim().toLowerCase(); draw(); } });

  function labels(e) {
    return [e.kind, ...e.tags].filter(Boolean);
  }

  function drawChips() {
    const seen = new Map();
    for (const e of all) for (const l of labels(e)) seen.set(l, (seen.get(l) || 0) + 1);
    const names = ['', ...[...seen.keys()].sort()];
    replace(chips, names.length > 2 ? names.map((name) => h('button', { type: 'button', class: 'chip', 'aria-pressed': String(pick === name),
      onclick: () => { pick = name; drawChips(); draw(); } }, name || 'All', name ? h('span', { class: 'chip__n' }, String(seen.get(name))) : null)) : null);
  }

  function draw() {
    const list = all.filter((e) => (!pick || labels(e).includes(pick)) && (!query || `${e.title} ${e.kind} ${e.by} ${e.tags.join(' ')}`.toLowerCase().includes(query)));
    count.textContent = all.length ? `${list.length} of ${all.length} tours` : '';
    if (!all.length) {
      replace(body, stateBox({ kind: 'empty', iconName: 'cube', title: 'No public tours yet', body: 'Tours show up here after their makers submit them and the site owner accepts them.',
        actions: [h('a', { class: 'btn btn--primary', href: '#/create' }, 'Make your own')] }));
    } else if (!list.length) {
      replace(body, stateBox({ kind: 'empty', iconName: 'search', title: 'Nothing matches this filter',
        actions: [h('button', { type: 'button', class: 'btn btn--line', onclick: () => { pick = ''; query = ''; search.value = ''; drawChips(); draw(); } }, 'Clear the filter')] }));
    } else {
      replace(body, h('ul', { class: 'grid' }, list.map(card)));
    }
  }

  async function load(force) {
    replace(body, h('ul', { class: 'grid', 'aria-busy': 'true', 'aria-label': 'Loading tours' }, [0, 1, 2, 3, 4, 5].map(() => h('li', { class: 'grid__item' }, h('div', { class: 'tour tour--ghost' })))));
    try {
      all = await ctx.gallery(force);
      drawChips();
      draw();
    } catch (err) {
      console.warn('[namma] gallery', err);
      count.textContent = '';
      replace(body, stateBox({ kind: 'error', title: 'The tour list could not be loaded', body: 'Check your connection and try again.',
        actions: [h('button', { type: 'button', class: 'btn btn--line', onclick: () => load(true) }, 'Try again')] }));
    }
  }

  load(false);
  return h('section', { class: 'wrap page' },
    h('h1', { class: 'title title--page' }, h('span', null, 'Public works'), h('span', { class: 'title__dim' }, 'places people filmed and shared')),
    h('div', { class: 'toolbar' }, h('label', { class: 'searchbox' }, icon('search', 18), search), chips),
    count,
    body);
}
