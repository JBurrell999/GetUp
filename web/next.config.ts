import type { NextConfig } from "next";

const config: NextConfig = {
  poweredByHeader: false,
  // pin the workspace root to this folder (a stray lockfile higher up confuses detection)
  turbopack: { root: __dirname },
};

export default config;
