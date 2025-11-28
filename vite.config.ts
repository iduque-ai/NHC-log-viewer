import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig(() => {
  // Log the presence of the API Key during build for debugging
  const hasKey = !!process.env.API_KEY;
  console.log(`[Vite Build] API_KEY environment variable is: ${hasKey ? 'PRESENT' : 'MISSING'}`);

  return {
    plugins: [react()],
    // This replaces process.env.API_KEY in the source code with the string value
    define: {
      'process.env.API_KEY': JSON.stringify(process.env.API_KEY),
    },
  };
})