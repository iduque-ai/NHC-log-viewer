
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig(() => {
  // Check for both legacy API_KEY and new VITE_API_KEY
  const apiKey = process.env.API_KEY || process.env.VITE_API_KEY;
  const hasKey = !!apiKey;
  console.log(`[Vite Build] API Key is: ${hasKey ? 'PRESENT' : 'MISSING'}`);

  return {
    plugins: [react()],
    // Define the VITE_API_KEY global to ensure it exists even if using the legacy API_KEY env var
    define: {
      'import.meta.env.VITE_API_KEY': JSON.stringify(apiKey),
    },
  };
})
