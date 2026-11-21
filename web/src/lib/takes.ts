// Where uploads live in the Blob store, shared by the browser and the API routes.
//
//   takes/<judge>/<task_id>/<yyyymmdd-hhmmss>-<rand>.<ext>   the video
//   takes/<judge>/<task_id>/<yyyymmdd-hhmmss>-<rand>.json    who, consent, file details
//
// <judge> is a slug of the person's name plus a random suffix, so two people called Sam
// don't land in the same folder.

export const VIDEO_EXTS = ["mp4", "mov", "m4v", "webm"] as const;
export type VideoExt = (typeof VIDEO_EXTS)[number];

export const MAX_BYTES = 2 * 1024 ** 3;

const JUDGE = "[a-z0-9]+(?:-[a-z0-9]+)*";
const TASK = "[a-z0-9_]+";
const STAMP = "\\d{8}-\\d{6}-[a-z0-9]{4}";
const TAKE_RE = new RegExp(`^takes/(${JUDGE})/(${TASK})/(${STAMP})\\.(${VIDEO_EXTS.join("|")}|json)$`);
const JUDGE_RE = new RegExp(`^${JUDGE}$`);

export const isJudgeId = (s: string) => s.length <= 40 && JUDGE_RE.test(s);

export interface TakePath {
  judge: string;
  task: string;
  stamp: string;
  ext: string;
}

export function parseTakePath(pathname: string): TakePath | null {
  const m = TAKE_RE.exec(pathname);
  if (!m || m[1].length > 40) return null;
  return { judge: m[1], task: m[2], stamp: m[3], ext: m[4] };
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

export function takePathname(judge: string, task: string, ext: VideoExt, now = new Date()) {
  const p = (n: number) => String(n).padStart(2, "0");
  const stamp =
    `${now.getUTCFullYear()}${p(now.getUTCMonth() + 1)}${p(now.getUTCDate())}-` +
    `${p(now.getUTCHours())}${p(now.getUTCMinutes())}${p(now.getUTCSeconds())}-${rand(4)}`;
  return `takes/${judge}/${task}/${stamp}.${ext}`;
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

/** What the list endpoint returns for each take. */
export interface TakeSummary {
  pathname: string;
  task: string;
  uploadedAt: string;
  size: number;
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
