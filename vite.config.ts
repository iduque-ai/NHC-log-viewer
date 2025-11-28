
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Check API Key presence during build
if (process.env.API_KEY) {
  console.log('✅ API_KEY found in environment variables.');
} else {
  console.log('⚠️ API_KEY NOT found in environment variables. The AI features will not work.');
}

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  define: {
    // This replaces process.env.API_KEY in the source code with the actual string value
    'process.env.API_KEY': JSON.stringify(process.env.API_KEY)
  },
  build: {
    rollupOptions: {
      // These libraries are loaded via CDN in index.html, so we exclude them from the bundle
      external: ['@google/genai', '@mlc-ai/web-llm'],
    }
  }
})
