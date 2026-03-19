import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/frontend/app/',
  build: {
    outDir: '../hcultctrl/hcultctrl/frontend/app',
    emptyOutDir: true,
  },
})
