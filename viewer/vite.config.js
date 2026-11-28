import { defineConfig } from 'vite';
import { resolve } from 'node:path';

export default defineConfig({
  base: '/viewer/',
  // places.js imports the shared POI engine from ../colab/real_scenes/final/poi_core.js (one level above this root).
  server: { host: '127.0.0.1', fs: { allow: [resolve(import.meta.dirname, '..')] } },
  build: {
    outDir: resolve(import.meta.dirname, '../app/static/viewer'),
    emptyOutDir: true,
    chunkSizeWarningLimit: 900,
  },
});
