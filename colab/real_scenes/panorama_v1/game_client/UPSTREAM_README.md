# 3rd-Person Character Controller — Spark 2.0 LoD + Rapier + Three.js

👉 [Try the live URL link here](https://third-person-character-controller.netlify.app/)


> A ready-to-fork template for building 3D games with Gaussian splat environments, a kinematic character controller, and real-time physics.

**Stack**
| Layer | Library |
|---|---|
| Renderer | [Three.js r180](https://threejs.org/) |
| 3D World | [World Labs Marble](https://marble.worldlabs.ai/)|
| Gaussian splats | [@sparkjsdev/spark 2.0](https://sparkjs.dev/) with LoD |
| Physics | [@dimforge/rapier3d-compat 0.12](https://rapier.rs/) — KCC + dynamic rigid bodies |
| Post-processing | [pmndrs/postprocessing](https://github.com/pmndrs/postprocessing) — bloom, vignette, brightness/contrast |
| GUI | [lil-gui](https://lil-gui.georgealways.com/) |
| Build | [Vite 6](https://vitejs.dev/) + TypeScript |

**Controls**
| Key | Action |
|---|---|
| W / A / S / D | Move |
| Shift | Sprint |
| Space | Jump |
| 1 | Trick animation |
| Right-click drag | Orbit camera |
| Scroll | Zoom |

---

## Quick start

### Requirements
- [Node.js](https://nodejs.org/) **18.18+** (20 LTS recommended)
- npm 9+

### Install & run

```bash
npm ci          # clean, lockfile-exact install
npm run dev     # dev server → http://localhost:5173
```

### Build for production

```bash
npm run build
npm run preview
```

### Verify before sharing

```bash
npm run verify  # TypeScript + production build check
```

---

## How to use this as a template

### Swap the world (Gaussian splat + collider)

Open `src/main.ts` and find the `WORLD_ASSETS` block near the top:

```ts
const WORLD_ASSETS = {
  splatSpz: '/attic.spz',      // ← your .spz Gaussian splat file
  colliderGlb: '/collider.glb', // ← invisible collision mesh
  colliderGlbUniformScale: 5,  // ← scale to match splat units
} as const;
```

1. Drop your `.spz` and collider `.glb` into `public/`.
2. Update the paths and scale.
3. The rest of the scene, character, and physics adapt automatically.

**Tip:** Export the collider as a simple, low-poly mesh from Blender. The code bakes it into a Rapier `trimesh` at startup.

### Swap the character model

Find the `CHARACTER_ASSETS` block right below `WORLD_ASSETS`:

```ts
const CHARACTER_ASSETS = {
  glb: '/dog.glb', // ← your animated character GLB
} as const;
```

Then update the clip name constants to match your model's animation names:

```ts
const CLIP_HAPPY_IDLE = 'Happy Idle'; // ground idle
const CLIP_AIR_IDLE   = 'Idle';       // airborne idle
const CLIP_WALK       = 'Brutal To';  // walk / run
const CLIP_TRICK      = 'Step Hip Hop'; // press-1 emote (optional)
```

The character system auto-scales the model to match the capsule height and handles idle → walk → air crossfades.

### Tune the controller

All gameplay values live in the `tuning` object and are exposed live in the GUI panel (top-right corner). Key parameters:

| `tuning` key | What it does |
|---|---|
| `moveSpeed` | Base movement speed (m/s) |
| `jumpSpeed` | Jump launch velocity |
| `gravityY` | World gravity (negative = down) |
| `characterHeight` | Auto-scales the mesh to this height |
| `characterTurnSpeed` | How fast the character yaws toward input |
| `splatUniformScale` | Global scale of the splat world |

Changes in the GUI update in real time — useful for dialing in feel without restarting.

---

## Project layout

```
character-controller-final/
├── public/              # Runtime assets (served at /)
│   ├── attic.spz        # Gaussian splat world
│   ├── collider.glb     # Invisible physics mesh for the world
│   ├── dog.glb          # Animated character
│   └── pano.jpg         # Panorama for character/ball reflections
├── src/
│   └── main.ts          # Entire app — scene, physics, KCC, GUI
├── index.html
├── vite.config.ts
├── tsconfig.json
└── package.json
```

> **Why one file?** `src/main.ts` is intentionally self-contained so the flow is easy to read top-to-bottom. Split it as you see fit for your own project.

---

## Sharing / zipping

Generate a zip (excludes `node_modules` and `dist`):

```bash
npm run package:zip
# → ../character-controller-share.zip
```

Custom filename:

```bash
npm run package:zip -- my-game-demo
# → ../my-game-demo.zip
```

**Recipient steps after unzip:**

```bash
rm -rf node_modules dist   # if replacing an older copy
npm ci
npm run dev
```

---

## Version notes

- **Three.js is pinned to `0.180.0`** — do not bump without testing. Later releases can break `@sparkjsdev/spark` typings or behavior.
- `package-lock.json` is committed so `npm ci` always installs the exact tested versions.

---

## License

MIT — fork freely and build something fun.
