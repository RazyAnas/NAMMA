// Home (#/) and the privacy note (#/privacy). Works without sign-in.
import { h, replace, icon, extLink, fmtDate } from '../ui.js';

export const PRIVACY_NOTE = 'Link-only tours sit in a public Space with a random folder name. Anyone who knows how can list the folder names, so do not treat link-only as private.';
export const PRIVATE_NOTE = 'Private means this site does not show or share the link. The files stay in the same public Space until you delete the tour.';

const STEPS = [
  { n: '01', icon: 'phone', label: 'Record on your phone', text: 'Walk slowly through the room for 2 to 4 minutes. The normal camera app is enough.' },
  { n: '02', icon: 'chip', label: 'Trained on your own free GPU', text: 'Your Kaggle account does the heavy work. The fast setting takes about 55 minutes.', hot: true },
  { n: '03', icon: 'walk', label: 'Walk it in the browser', text: 'Open a link and walk around. It runs on phones and computers, with nothing to install.' },
  { n: '04', icon: 'lock', label: 'Yours: stored in your account', text: 'The video and the tour live in your Hugging Face account. You can delete them at any time.' },
];

const FEATURES = [
  ['Walk mode', 'Move at eye level like in a game. Walls and furniture stop you, so you stay inside the room.'],
  ['Floor plan', 'A small map shows the room from above and where you are standing.'],
  ['Real filmed spots', 'Jump to the exact places the phone filmed. These views are the sharpest.'],
  ['Places and routes', 'Search for a place in the tour and follow a route to it, with the distance left.'],
  ['AR on Android', 'Stand in the real room and see its places and routes over the camera. Early version, still being tested.'],
  ['Adaptive quality', 'The tour loads a lighter version on slower devices and more detail on faster ones.'],
];

function chip(text) {
  return h('span', { class: 'chip-inline' }, text);
}

function hero() {
  return h('section', { class: 'hero' },
    h('div', { class: 'hero__light', 'aria-hidden': 'true' }, h('span', { class: 'hero__orb' }), h('span', { class: 'hero__orb hero__orb--b' })),
    h('div', { class: 'wrap hero__in' },
      h('p', { class: 'eyebrow' }, h('span', { class: 'dot' }), 'Free and open source'),
      h('h1', { class: 'display' },
        h('span', { class: 'display__line' }, 'Turn a phone video ', chip('2 to 4 minutes of filming')),
        h('span', { class: 'display__line display__line--dim' }, 'into a place you can walk through ', chip('in about an hour'))),
      h('p', { class: 'lead' }, 'Film a room with your phone. NAMMA builds a 3D tour from it on your own free Kaggle GPU and saves it in your own Hugging Face account. This site keeps nothing.'),
      h('div', { class: 'hero__cta' },
        h('a', { class: 'btn btn--primary btn--lg', href: '#/explore' }, 'Explore tours', icon('arrow', 20)),
        h('a', { class: 'btn btn--line btn--lg', href: '#/create' }, 'Make your own')),
      h('ul', { class: 'hero__facts' },
        h('li', null, 'No payment'),
        h('li', null, 'No app to install'),
        h('li', null, 'About an hour per room'))));
}

function steps() {
  return h('section', { class: 'wrap section', 'aria-labelledby': 'steps-title' },
    h('h2', { id: 'steps-title', class: 'title' }, h('span', null, 'Four steps'), h('span', { class: 'title__dim' }, 'from a video to a walk')),
    h('ol', { class: 'steps' }, STEPS.map((s) => h('li', { class: s.hot ? 'step step--hot' : 'step' },
      h('div', { class: 'step__top' }, h('span', { class: 'step__n' }, s.n), h('span', { class: 'step__icon' }, icon(s.icon, 22))),
      h('div', { class: 'step__bottom' }, h('h3', { class: 'step__label' }, s.label), h('p', { class: 'step__text' }, s.text))))));
}

function tourCard(e) {
  const meta = [e.kind, fmtDate(e.created)].filter(Boolean).join(' · ');
  return extLink(e.url, { class: 'tour' },
    h('span', { class: 'tour__cover' }, e.cover ? h('img', { src: e.cover, alt: '', loading: 'lazy', decoding: 'async', referrerpolicy: 'no-referrer' }) : null),
    h('span', { class: 'tour__body' },
      h('span', { class: 'tour__title' }, e.title),
      meta ? h('span', { class: 'tour__meta' }, meta) : null,
      h('span', { class: 'tour__open' }, 'Open tour', icon('open', 18), h('span', { class: 'sr-only' }, ' (opens in a new tab)'))));
}

function featured(ctx) {
  const strip = h('div', { class: 'strip' }, [0, 1, 2, 3].map(() => h('div', { class: 'tour tour--ghost', 'aria-hidden': 'true' })));
  const section = h('section', { class: 'wrap section', 'aria-labelledby': 'feat-title' },
    h('div', { class: 'section__head' },
      h('h2', { id: 'feat-title', class: 'title' }, h('span', null, 'Featured tours'), h('span', { class: 'title__dim' }, 'made with NAMMA')),
      h('a', { class: 'btn btn--ghost', href: '#/explore' }, 'See all', icon('arrow', 18))),
    strip);
  ctx.gallery().then((list) => {
    const picks = list.filter((e) => e.cover).slice(0, 4);
    if (!picks.length) {
      replace(strip, h('p', { class: 'strip__note' }, 'No public tours are listed yet. Yours could be the first.'));
      return;
    }
    replace(strip, picks.map(tourCard));
  }).catch(() => {
    replace(strip, h('p', { class: 'strip__note' }, 'The tour list could not be loaded right now. ', h('a', { href: '#/explore' }, 'Try the Explore page.')));
  });
  return section;
}

function features() {
  return h('section', { class: 'wrap section', 'aria-labelledby': 'what-title' },
    h('h2', { id: 'what-title', class: 'title' }, h('span', null, 'What a tour can do'), h('span', { class: 'title__dim' }, 'once it is built')),
    h('ul', { class: 'features' }, FEATURES.map(([word, text], i) => h('li', { class: 'feature' },
      h('span', { class: 'feature__n', 'aria-hidden': 'true' }, String(i + 1).padStart(2, '0')),
      h('span', { class: 'feature__word' }, word),
      h('span', { class: 'feature__text' }, text)))));
}

function costs() {
  const how = [
    ['Sign in with Hugging Face.', 'The site asks for permission to create and fill the NAMMA storage in your account.'],
    ['Paste your Kaggle key.', 'It lets the site start a private notebook on your Kaggle account. The notebook gets your Hugging Face pass, so it can read the video and save the tour.'],
    ['Upload the video.', 'It goes to a private dataset in your Hugging Face account. If the upload breaks off, choose the same file again and it continues.'],
    ['Wait about an hour.', 'The job runs on Kaggle, not in this page.'],
    ['Open your tour.', 'You get a link. You decide whether to share it.'],
  ];
  return h('section', { class: 'wrap section', 'aria-labelledby': 'how-title' },
    h('div', { class: 'plain' },
      h('div', { class: 'plain__col' },
        h('h2', { id: 'how-title', class: 'plain__title' }, 'How it works and what it costs'),
        h('ol', { class: 'plain__list' }, how.map(([a, b]) => h('li', null, h('b', null, a), ' ', b)))),
      h('div', { class: 'plain__col plain__col--cost' },
        h('p', { class: 'plain__big' }, 'Free'),
        h('ul', { class: 'plain__facts' },
          h('li', null, h('b', null, 'The GPU'), ' comes from your Kaggle account. Kaggle gives phone-verified accounts free GPU hours every week.'),
          h('li', null, h('b', null, 'The storage'), ' is your Hugging Face account, within its free limits.'),
          h('li', null, h('b', null, 'A room'), ' takes about an hour: 55 minutes on Fast, 90 minutes on Quality.'),
          h('li', null, h('b', null, 'This site'), ' has no server and no account system. It is a plain page with open code.')),
        h('a', { class: 'btn btn--primary', href: '#/create' }, 'Make your own', icon('arrow', 18)))));
}

function wordmark() {
  return h('section', { class: 'mark', 'aria-hidden': 'true' },
    h('span', { class: 'mark__glow' }),
    h('span', { class: 'mark__word' }, 'NAMMA'));
}

export function render(ctx) {
  return h('div', { class: 'home' }, hero(), steps(), featured(ctx), features(), costs(), wordmark());
}

export function renderPrivacy() {
  const rows = [
    ['This site', 'It is a static page. It has no server of its own, no database, no analytics and no cookies. It talks only to Hugging Face and Kaggle, straight from your browser.'],
    ['Sign-in', 'Hugging Face gives this page a pass that ends after 8 hours. It is kept in this browser tab. When you start a job, the pass is also written into the private notebook that the site starts on your Kaggle account, so the job can read your video and save your tour. Signing out removes the pass from this browser. The copy in your Kaggle notebook stops working when the 8 hours are over.'],
    ['Your Kaggle key', 'It is kept for this browser tab only. If you tick "Remember on this device" it is saved in this browser until you sign out. It is sent to Kaggle and nowhere else.'],
    ['Your video', 'It is uploaded to a private dataset in your own Hugging Face account. Your Kaggle job reads it from there.'],
    ['Your tours', 'They are saved in a public Space in your own Hugging Face account, each in a folder with a random name. Private, Link-only and Public decide what this site shows and shares. They do not lock the files.'],
    ['Link-only', PRIVACY_NOTE],
    ['Private', PRIVATE_NOTE],
    ['Public works', 'When you submit a tour, the site opens a request in your name. The tour appears in the public list only after the site owner accepts it.'],
    ['Deleting', 'Delete in your library removes the files of that tour from your tour Space. The uploaded video (namma-inbox) and the training results (namma-v3-out) stay in your private datasets until you delete them on huggingface.co. You can delete the NAMMA datasets and the Space there at any time.'],
  ];
  return h('section', { class: 'wrap page page--narrow' },
    h('h1', { class: 'title title--page' }, h('span', null, 'Privacy'), h('span', { class: 'title__dim' }, 'in plain words')),
    h('dl', { class: 'facts' }, rows.map(([k, v]) => [h('dt', null, k), h('dd', null, v)])),
    h('p', { class: 'muted' }, 'The code of this site is open: ',
      extLink('https://github.com/RazyAnas/NAMMA', null, 'github.com/RazyAnas/NAMMA'), '.'));
}
