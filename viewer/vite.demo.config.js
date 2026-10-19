import { defineConfig } from 'vite';
import { resolve } from 'node:path';

// Static walking demo: `npm run build:demo` writes dist-demo/, which runs from
// any static file server once a data/ folder (splat.ply, navmesh.json,
// collision.glb) is placed next to its index.html.
export default defineConfig({
  base: './',
  build: {
    outDir: resolve(import.meta.dirname, 'dist-demo'),
    emptyOutDir: true,
    chunkSizeWarningLimit: 4000,
    rollupOptions: { input: { index: resolve(import.meta.dirname, 'demo.html') } },
  },
});
