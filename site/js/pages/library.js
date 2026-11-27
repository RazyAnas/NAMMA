// Library (#/library, needs sign-in): the user's tours, uploaded splat files and running jobs.
import { h, replace, icon, extLink, stateBox, spinner, notice, toast, errText, confirmDialog, copyText, fmtDate, fmtMb, safeUrl } from '../ui.js';
import { PRIVACY_NOTE, PRIVATE_NOTE } from './home.js';

const VISIBILITY = [
  ['private', 'Private', 'lock', 'The site does not show or share the link.'],
  ['link', 'Link-only', 'link', 'Anyone who has the link.'],
  ['public', 'Public', 'globe', 'Anyone. It can be shown in the public works.'],
];

const STAGE_TEXT = {
  uploading: 'Upload not finished',
  uploaded: 'Uploaded, waiting to start',
  training: 'Training on your Kaggle GPU',
  publishing: 'Building the tour',
  done: 'Done',
  failed: 'Stopped',
};

function jobRow(j) {
  const id = String(j.job);
  const failed = j.stage === 'failed';
  return h('li', { class: failed ? 'row row--bad' : 'row' },
    h('div', { class: 'row__main' },
      h('a', { class: 'row__title', href: `#/job/${encodeURIComponent(id)}` }, String(j.title || 'Untitled')),
      h('p', { class: 'row__meta' }, [STAGE_TEXT[j.stage] || String(j.stage || ''), j.preset ? `${j.preset} setting` : '', fmtDate(j.created)].filter(Boolean).join(' · ')),
      failed && j.reason ? h('p', { class: 'row__reason' }, String(j.reason)) : null),
    h('a', { class: 'btn btn--line btn--sm', href: `#/job/${encodeURIComponent(id)}` }, 'See progress', icon('arrow', 18)));
}

function itemCard(item, ctx, onRemoved) {
  const { jobs } = ctx.api;
  const id = String(item.id);
  let visibility = ['private', 'link', 'public'].includes(item.visibility) ? item.visibility : 'private';
  const url = safeUrl(item.url);
  const cover = safeUrl(item.cover);
  const status = h('p', { class: 'item__status', role: 'status' }, '');
  const submit = h('button', { type: 'button', class: 'btn btn--line btn--sm' }, icon('plus', 18), 'Submit to public works');
  const copy = url ? h('button', { type: 'button', class: 'btn btn--line btn--sm' }, icon('copy', 18), 'Copy link') : null;
  const group = `vis-${id.replace(/[^a-z0-9]/gi, '_')}`;
  const radios = [];

  const sync = () => {
    for (const r of radios) r.checked = r.value === visibility;
    submit.disabled = visibility !== 'public';
    submit.title = visibility === 'public' ? '' : 'Set the tour to Public first';
    if (copy) copy.hidden = visibility === 'private';
  };

  const seg = h('fieldset', { class: 'seg' },
    h('legend', { class: 'seg__legend' }, 'Who can open it'),
    h('div', { class: 'seg__row' }, VISIBILITY.map(([value, label, ico, hint]) => {
      const input = h('input', { type: 'radio', name: group, value, class: 'seg__input', title: hint });
      radios.push(input);
      input.addEventListener('change', async () => {
        if (!input.checked || value === visibility) return;
        const before = visibility;
        if (value === 'public' && !(await confirmDialog({ title: 'Make this tour public?', body: 'Anyone will be able to open it, and you can submit it to the public works. Share only places you have the right to show.', confirmLabel: 'Make public' }))) {
          sync();
          return;
        }
        seg.disabled = true;
        status.textContent = 'Saving...';
        try {
          await jobs.setVisibility(id, value);
          visibility = value;
          status.textContent = `Now ${label.toLowerCase()}.`;
        } catch (err) {
          visibility = before;
          status.textContent = '';
          toast('The setting was not changed. ' + errText(err, ''), 'error');
        } finally {
          seg.disabled = false;
          sync();
        }
      });
      return h('label', { class: 'seg__opt' }, input, h('span', { class: 'seg__face' }, icon(ico, 16), label));
    })));

  if (copy) {
    copy.addEventListener('click', async () => {
      const ok = await copyText(url);
      toast(ok ? 'Link copied.' : 'Copying did not work. Open the tour and copy the address.', ok ? 'ok' : 'error');
    });
  }

  submit.addEventListener('click', async () => {
    const yes = await confirmDialog({ title: 'Submit to the public works?', body: 'This opens a request in your name. The site owner looks at it, and the tour appears in the public list once it is accepted.', confirmLabel: 'Submit' });
    if (!yes) return;
    submit.disabled = true;
    status.textContent = 'Sending the request...';
    try {
      const res = await jobs.submitToGallery(id);
      const link = extLink(res?.prUrl, null, 'See the request');
      replace(status, 'Request sent. ', link);
    } catch (err) {
      status.textContent = '';
      toast('The request was not sent. ' + errText(err, ''), 'error');
    } finally {
      sync();
    }
  });

  const remove = h('button', { type: 'button', class: 'btn btn--ghost btn--sm btn--bad', onclick: async () => {
    const yes = await confirmDialog({ title: `Delete "${String(item.title || 'this item')}"?`, body: 'Its files are removed from your tour Space and its link stops working. This cannot be undone. The training results stay in your private namma-v3-out dataset.', confirmLabel: 'Delete', danger: true });
    if (!yes) return;
    remove.disabled = true;
    status.textContent = 'Deleting...';
    try {
      await jobs.remove(id);
      toast('Deleted.', 'ok');
      onRemoved(card);
    } catch (err) {
      remove.disabled = false;
      status.textContent = '';
      toast('It was not deleted. ' + errText(err, ''), 'error');
    }
  } }, icon('trash', 18), 'Delete');

  const kind = item.kind === 'splat' ? 'Splat file' : 'Tour';
  const card = h('li', { class: 'item' },
    h('div', { class: 'item__cover' }, cover ? h('img', { src: cover, alt: '', loading: 'lazy', decoding: 'async', referrerpolicy: 'no-referrer' }) : h('span', { class: 'tour__nocover' }, icon('cube', 28))),
    h('div', { class: 'item__body' },
      h('h3', { class: 'item__title' }, String(item.title || 'Untitled')),
      h('p', { class: 'row__meta' }, [kind, fmtDate(item.created), fmtMb(Number(item.sizeMb))].filter(Boolean).join(' · ')),
      seg,
      h('div', { class: 'item__actions' },
        extLink(url, { class: 'btn btn--primary btn--sm' }, 'Open', icon('open', 18), h('span', { class: 'sr-only' }, ' (opens in a new tab)')),
        copy, submit, remove),
      status));
  sync();
  return card;
}

export function render(ctx) {
  const { jobs } = ctx.api;
  const running = h('div', { class: 'block' });
  const items = h('div', { class: 'block' });
  const root = h('section', { class: 'wrap page' },
    h('div', { class: 'section__head' },
      h('h1', { class: 'title title--page' }, h('span', null, 'Your library'), h('span', { class: 'title__dim' }, 'kept in your own account')),
      h('a', { class: 'btn btn--primary', href: '#/create' }, icon('plus', 18), 'Make a tour')),
    running, items);

  if (!jobs) {
    replace(items, stateBox({ kind: 'error', title: 'Your library could not be opened', body: 'A part of this site did not load. Reload the page and try again.',
      actions: [h('button', { type: 'button', class: 'btn btn--line', onclick: () => window.location.reload() }, 'Reload')] }));
    return root;
  }

  async function loadJobs() {
    try {
      const list = (await jobs.list()) || [];
      const open = list.filter((j) => j && j.stage !== 'done');
      replace(running, open.length ? [h('h2', { class: 'block__title' }, 'In progress'), h('ul', { class: 'rows' }, open.map(jobRow))] : null);
    } catch (err) {
      replace(running, h('h2', { class: 'block__title' }, 'In progress'),
        notice('error', 'Your running jobs could not be loaded. ', errText(err, ''), ' ', h('button', { type: 'button', class: 'linkbtn', onclick: loadJobs }, 'Try again')));
    }
  }

  async function loadItems() {
    replace(items, h('h2', { class: 'block__title' }, 'Tours and splat files'), spinner('Loading your library'));
    try {
      const list = (await jobs.library()) || [];
      if (!list.length) {
        replace(items, stateBox({ kind: 'empty', iconName: 'cube', title: 'Nothing here yet', body: 'Film a room and turn it into a tour, or upload a splat file you already have.',
          actions: [h('a', { class: 'btn btn--primary', href: '#/create' }, 'Make a tour')] }));
        return;
      }
      const ul = h('ul', { class: 'items' });
      const onRemoved = (card) => { card.remove(); if (!ul.children.length) loadItems(); };
      for (const it of list) if (it && it.id != null) ul.appendChild(itemCard(it, ctx, onRemoved));
      replace(items,
        h('h2', { class: 'block__title' }, 'Tours and splat files'),
        h('p', { class: 'privacy-note' }, icon('info', 18), h('span', null, PRIVACY_NOTE, ' ', PRIVATE_NOTE)),
        ul);
    } catch (err) {
      replace(items, stateBox({ kind: 'error', title: 'Your library could not be loaded', body: errText(err),
        actions: [h('button', { type: 'button', class: 'btn btn--line', onclick: loadItems }, 'Try again')] }));
    }
  }

  loadJobs();
  loadItems();
  return root;
}
