import { defineConfig } from 'vite';
import { resolve } from 'node:path';

export default defineConfig({
  base: '/viewer/',
  build: {
    outDir: resolve(import.meta.dirname, '../app/static/viewer'),
    emptyOutDir: true,
    chunkSizeWarningLimit: 900,
  },
});
