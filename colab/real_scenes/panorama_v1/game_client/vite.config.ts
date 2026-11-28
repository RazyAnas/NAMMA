import { defineConfig } from 'vite';

export default defineConfig({
  /** NAMMA: relative asset paths so a build can live in any folder of the HF Space. */
  base: './',
  server: {
    open: true,
  },
  esbuild: {
    target: 'es2022',
  },
  build: {
    /** Required for top-level `await RAPIER.init()` (WASM physics bootstrap). */
    target: 'es2022',
  },
});
