import type { NextConfig } from "next";

const config: NextConfig = {
  poweredByHeader: false,
  // Let the browser load dev assets and hot reload through an ngrok tunnel.
  allowedDevOrigins: ["*.ngrok-free.app"],
  // pin the workspace root to this folder (a stray lockfile higher up confuses detection)
  turbopack: { root: __dirname },
};

export default config;
