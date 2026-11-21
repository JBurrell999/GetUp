// Copies the task list from the repo root (the one session.py uses) into the web app.
// Runs before dev and build. If the root file isn't there (e.g. the web folder was
// deployed on its own), the committed copy in src/data is used as is.
import { copyFileSync, existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const src = resolve(here, "../../tasks.json");
const dst = resolve(here, "../src/data/tasks.json");

if (existsSync(src)) {
  JSON.parse(readFileSync(src, "utf8")); // fail loudly on a broken file
  copyFileSync(src, dst);
  console.log("[sync] tasks.json copied from the repo root");
} else if (existsSync(dst)) {
  console.log("[sync] repo-root tasks.json not found, using src/data/tasks.json");
} else {
  console.error("[sync] no tasks.json found");
  process.exit(1);
}
