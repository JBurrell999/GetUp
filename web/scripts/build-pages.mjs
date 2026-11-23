// Static build for GitHub Pages. The API routes need the Python pipeline and can't be exported, so they are
// moved into a private folder (_api, ignored by the router) for the build and always moved back.
import { execSync } from "node:child_process";
import { existsSync, renameSync, writeFileSync } from "node:fs";

const api = "src/app/api";
const parked = "src/app/_api";
const env = {
  ...process.env,
  NEXT_PUBLIC_STATIC: "1",
  NEXT_PUBLIC_BASE_PATH: process.env.NEXT_PUBLIC_BASE_PATH ?? "/GetUp",
};

if (existsSync(parked)) throw new Error(`${parked} exists: a previous build was interrupted. Move it back to ${api}.`);
// Type-check with the API routes in place (the dev server's generated types refer to them); the build skips it.
execSync("tsc --noEmit", { stdio: "inherit" });
renameSync(api, parked);
try {
  execSync("next build", { stdio: "inherit", env });
  writeFileSync("out/.nojekyll", ""); // serve the _next/ folder as-is
} finally {
  renameSync(parked, api);
}
