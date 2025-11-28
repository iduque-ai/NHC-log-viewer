// Fixed: Removed missing 'vite/client' reference and added process.env type definition.
declare var process: {
  env: {
    API_KEY: string;
    [key: string]: string | undefined;
  }
};
