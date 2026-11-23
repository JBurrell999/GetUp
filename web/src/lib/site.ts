// Two builds of the same site:
// - the normal one (npm run dev / next start) next to motion_mirror/, with uploads and "Try it";
// - a static one for GitHub Pages (npm run build:pages): the demos and gallery only, served under a sub-path.
export const STATIC = process.env.NEXT_PUBLIC_STATIC === "1";
export const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
export const REPO_URL = "https://github.com/JBurrell999/GetUp";

/** A file in public/, with the sub-path the site is served under. */
export const asset = (p: string) => BASE_PATH + p;
