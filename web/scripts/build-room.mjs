// Turns the MuJoCo room's layout.json into the small floor-plan file the site draws.
//   node scripts/build-room.mjs /path/to/g1_room_mujoco/layout.json
// Only footprints are kept: [x, y, hx, hy] in metres (centre and half-extents, z up).
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const input = process.argv[2];
if (!input) {
  console.error("usage: node scripts/build-room.mjs <layout.json>");
  process.exit(1);
}
const layout = JSON.parse(readFileSync(input, "utf8"));
const r = (v) => Math.round(v * 1000) / 1000;
const box = (b) => [r(b.x), r(b.y), r(b.hx), r(b.hy)];
const boxes = layout.render_boxes;
const pick = (fn) => boxes.filter(fn).map(box);

const room = {
  source: "g1_room_mujoco/layout.json",
  floors: pick((b) => b.role === "floor"),
  walls: pick((b) => b.role === "wall" || (b.role === "wood" && b.collision)),
  columns: pick((b) => b.role === "column"),
  stage: pick((b) => b.name === "stage_platform"),
  screen: pick((b) => b.name === "stage_display_surface"),
  tables: pick((b) => b.role === "table" && b.name.endsWith("_top")),
  bar: pick((b) => b.name === "annex_counter_top"),
  chairs: pick((b) => b.role === "chair" && /_(seat|back)$/.test(b.name)),
  spawn: layout.spawn_xy,
};
const out = resolve(here, "../src/data/room.json");
writeFileSync(out, JSON.stringify(room) + "\n");
console.log(`[room] ${Object.entries(room).map(([k, v]) => `${k}:${Array.isArray(v) ? v.length : "-"}`).join(" ")} -> ${out}`);
