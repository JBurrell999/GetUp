import type { NextConfig } from "next";

// npm run build:pages sets these: a static export for GitHub Pages, served from /<repo>.
const staticSite = process.env.NEXT_PUBLIC_STATIC === "1";
const basePath = process.env.NEXT_PUBLIC_BASE_PATH || undefined;

const config: NextConfig = {
  poweredByHeader: false,
  // Let the browser load dev assets and hot reload through an ngrok tunnel.
  allowedDevOrigins: ["*.ngrok-free.app"],
  // pin the workspace root to this folder (a stray lockfile higher up confuses detection)
  turbopack: { root: __dirname },
  ...(staticSite && {
    output: "export",
    basePath,
    trailingSlash: true,
    images: { unoptimized: true },
    // build-pages.mjs type-checks before it moves the API routes aside
    typescript: { ignoreBuildErrors: true },
  }),
};

export default config;
