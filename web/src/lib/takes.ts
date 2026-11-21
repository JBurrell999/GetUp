// Where uploads live in the Blob store, shared by the browser and the API routes.
//
//   takes/<judge>/<env>/<what>/<yyyymmdd-hhmmss>-<rand>.<ext>   the video
//   takes/<judge>/<env>/<what>/<yyyymmdd-hhmmss>-<rand>.json    who, where, what, consent, file details
//
// <judge> is a slug of the person's name plus a random suffix, so two people called Sam
// don't land in the same folder. <env> is where they filmed (see environments.ts). <what> is
// a slug of what they said they were doing ("Move a chair" -> move_a_chair); the JSON keeps
// their exact words. Older uploads used a task id from the old fixed task list in the <what>
// slot, and some have no <env> level; both are still read.

import { ENV_IDS, type EnvId } from "./environments";

export const VIDEO_EXTS = ["mp4", "mov", "m4v", "webm"] as const;
export type VideoExt = (typeof VIDEO_EXTS)[number];

export const MAX_BYTES = 2 * 1024 ** 3;

const JUDGE = "[a-z0-9]+(?:-[a-z0-9]+)*";
const WHAT = "[a-z0-9]+(?:_[a-z0-9]+)*";
const WHAT_MAX = 60;
const STAMP = "\\d{8}-\\d{6}-[a-z0-9]{4}";
const ENV = ENV_IDS.join("|");
const TAKE_RE = new RegExp(`^takes/(${JUDGE})/(?:(${ENV})/)?(${WHAT})/(${STAMP})\\.(${VIDEO_EXTS.join("|")}|json)$`);
const JUDGE_RE = new RegExp(`^${JUDGE}$`);

export const isJudgeId = (s: string) => s.length <= 40 && JUDGE_RE.test(s);

export interface TakePath {
  judge: string;
  /** null for takes uploaded before the site asked where people were filming */
  env: EnvId | null;
  what: string;
  stamp: string;
  ext: string;
}

export function parseTakePath(pathname: string): TakePath | null {
  const m = TAKE_RE.exec(pathname);
  if (!m || m[1].length > 40 || m[3].length > WHAT_MAX) return null;
  return { judge: m[1], env: (m[2] as EnvId | undefined) ?? null, what: m[3], stamp: m[4], ext: m[5] };
}

const rand = (n: number) =>
  Array.from(crypto.getRandomValues(new Uint8Array(n)), (b) => "abcdefghijklmnopqrstuvwxyz0123456789"[b % 36]).join("");

export function makeJudgeId(name: string) {
  const slug = name
    .normalize("NFKD")
    .replace(/[̀-ͯ]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 24)
    .replace(/-+$/, "");
  return `${slug || "judge"}-${rand(4)}`;
}

/** "Move a chair!" -> "move_a_chair". Falls back to "video" when nothing usable is left. */
export function slugify(text: string) {
  const s = text
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 48)
    .replace(/_+$/, "");
  return s || "video";
}

/** "move_a_chair" -> "Move a chair", for listing uploads without reading their details. */
export const humanize = (slug: string) => {
  const s = slug.replace(/_/g, " ");
  return s[0].toUpperCase() + s.slice(1);
};

export function takePathname(judge: string, env: EnvId, what: string, ext: VideoExt, now = new Date()) {
  const p = (n: number) => String(n).padStart(2, "0");
  const stamp =
    `${now.getUTCFullYear()}${p(now.getUTCMonth() + 1)}${p(now.getUTCDate())}-` +
    `${p(now.getUTCHours())}${p(now.getUTCMinutes())}${p(now.getUTCSeconds())}-${rand(4)}`;
  return `takes/${judge}/${env}/${slugify(what)}/${stamp}.${ext}`;
}

export const metaPathname = (videoPathname: string) => videoPathname.replace(/\.[a-z0-9]+$/, ".json");

const TYPE_EXT: Record<string, VideoExt> = {
  "video/mp4": "mp4",
  "video/quicktime": "mov",
  "video/x-m4v": "m4v",
  "video/webm": "webm",
};

/** The extension to store a file under, or null if it isn't a video we take. */
export function videoExt(file: { name: string; type: string }): VideoExt | null {
  const fromName = file.name.split(".").pop()?.toLowerCase();
  if (fromName && (VIDEO_EXTS as readonly string[]).includes(fromName)) return fromName as VideoExt;
  return TYPE_EXT[file.type] ?? null;
}

export const CONTENT_TYPE: Record<VideoExt, string> = {
  mp4: "video/mp4",
  mov: "video/quicktime",
  m4v: "video/x-m4v",
  webm: "video/webm",
};

// The robot's copy of a take, written by the worker in motion_mirror (site_worker.py):
//   outputs/<judge>/<env>/<what>/<stamp>/{result.json, robot.mp4, side_by_side.mp4, wide.mp4}
export const ROBOT_FILES = { side_by_side: "side_by_side.mp4", robot: "robot.mp4", wide: "wide.mp4" } as const;
export type RobotFile = keyof typeof ROBOT_FILES;
export const isRobotFile = (s: unknown): s is RobotFile => typeof s === "string" && Object.hasOwn(ROBOT_FILES, s);

/** Where a take's robot results go, or null for takes that don't say which room they were filmed in. */
export const outputPrefix = (t: TakePath) => (t.env ? `outputs/${t.judge}/${t.env}/${t.what}/${t.stamp}/` : null);

export interface RobotResult {
  status: "processing" | "complete" | "failed";
  /** The videos that are ready, in the order to offer them. */
  files: RobotFile[];
  /** Gemini's 0-10 score for how closely the robot's motion matches, when complete. */
  score: number | null;
}

/** What the list endpoint returns for each take. */
export interface TakeSummary {
  pathname: string;
  env: EnvId | null;
  what: string;
  uploadedAt: string;
  size: number;
  /** null until the robot worker has picked the take up. */
  robot?: RobotResult | null;
}

export function formatBytes(b: number) {
  if (b >= 1e9) return `${(b / 1e9).toFixed(2)} GB`;
  if (b >= 1e6) return `${(b / 1e6).toFixed(b < 1e7 ? 1 : 0)} MB`;
  return `${Math.max(1, Math.round(b / 1e3))} KB`;
}

/** Compares a typed event code with the configured one, forgiving case and spaces. */
export function codeMatches(given: unknown, expected: string | undefined) {
  if (!expected) return true;
  return typeof given === "string" && given.trim().toLowerCase() === expected.trim().toLowerCase();
}
