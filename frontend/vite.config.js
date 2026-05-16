import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/frontend/app/',
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      '/plants': 'http://localhost:8000',
      '/plant_sensors': 'http://localhost:8000',
      '/timeseries': 'http://localhost:8000',
      '/observations': 'http://localhost:8000',
      '/auth': 'http://localhost:8000',
      '/water_calibration': 'http://localhost:8000',
      '/drying_rate': 'http://localhost:8000',
      '/swc_timeseries': 'http://localhost:8000',
      '/chords': 'http://localhost:8000',
      '/prior': 'http://localhost:8000',
    },
  },
  test: {
    globals: true,
    environment: 'jsdom',
  },
})
