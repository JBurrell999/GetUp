import "server-only";

// Runs the motion_mirror pipeline (Gemini + MuJoCo, Python) on an uploaded video. This only works where the site runs
// on the same machine as the pipeline (npm run dev / next start next to the repo). On Vercel it reports "unavailable".
import { spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import fs from "node:fs";
import path from "node:path";

export const MIRROR_DIR = process.env.MIRROR_DIR ?? path.resolve(process.cwd(), "..", "motion_mirror");
const MAX_RUNNING = 2;
const ID = /^[a-z0-9]{10}$/;

const uploads = () => path.join(MIRROR_DIR, "videos", "web_uploads");
const runDir = (id: string) => path.join(MIRROR_DIR, "runs", "mirror2", `web_${id}`);
const logFile = (id: string) => path.join(MIRROR_DIR, "runs", `web_${id}.log`);
const jobFile = (id: string) => path.join(MIRROR_DIR, "runs", `web_${id}.job.json`);

export function available(): boolean {
  if (process.env.VERCEL) return false;
  return fs.existsSync(path.join(MIRROR_DIR, "mm")) && fs.existsSync(path.join(MIRROR_DIR, ".env"));
}

export const validId = (id: string) => ID.test(id);

function alive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

function running(): number {
  const dir = path.join(MIRROR_DIR, "runs");
  if (!fs.existsSync(dir)) return 0;
  return fs
    .readdirSync(dir)
    .filter((f) => f.endsWith(".job.json"))
    .map((f) => {
      try {
        return JSON.parse(fs.readFileSync(path.join(dir, f), "utf8")) as { pid: number };
      } catch {
        return null;
      }
    })
    .filter((j): j is { pid: number } => Boolean(j && alive(j.pid))).length;
}

export async function startJob(file: File, what: string): Promise<{ id?: string; error?: string }> {
  if (running() >= MAX_RUNNING) return { error: `Already running ${MAX_RUNNING} videos. Try again in a couple of minutes.` };
  const ext = (file.name.split(".").pop() || "mp4").toLowerCase().replace(/[^a-z0-9]/g, "") || "mp4";
  const id = randomBytes(8).toString("hex").slice(0, 10);
  fs.mkdirSync(uploads(), { recursive: true });
  fs.mkdirSync(path.join(MIRROR_DIR, "runs"), { recursive: true });
  const video = path.join(uploads(), `${id}.${ext}`);
  fs.writeFileSync(video, Buffer.from(await file.arrayBuffer()));
  const log = fs.openSync(logFile(id), "w");
  const child = spawn(path.join(MIRROR_DIR, "mm"), ["mirror", video, "--name", `web_${id}`, "--no-bench"], {
    cwd: MIRROR_DIR,
    detached: true,
    stdio: ["ignore", log, log],
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
  });
  child.unref();
  fs.writeFileSync(jobFile(id), JSON.stringify({ id, pid: child.pid, what, started: Date.now(), file: file.name }));
  return { id };
}

export interface JobStatus {
  id: string;
  state: "running" | "done" | "error" | "unknown";
  stage: string;
  progress: number;
  elapsed_s: number;
  understood?: string;
  scores?: { scene?: number; motion?: number };
  video?: string;
  error?: string;
}

export function jobStatus(id: string): JobStatus {
  const base: JobStatus = { id, state: "unknown", stage: "", progress: 0, elapsed_s: 0 };
  if (!fs.existsSync(jobFile(id))) return base;
  const job = JSON.parse(fs.readFileSync(jobFile(id), "utf8")) as { pid: number; started: number };
  base.elapsed_s = Math.round((Date.now() - job.started) / 1000);
  const log = fs.existsSync(logFile(id)) ? fs.readFileSync(logFile(id), "utf8") : "";
  const dir = runDir(id);

  // progress from the pipeline's own log lines (scene phase, then parallel motion attempts)
  const envRounds = (log.match(/^\[env \d+\]/gm) ?? []).length;
  let motionRounds = 0;
  if (fs.existsSync(dir)) {
    for (const f of fs.readdirSync(dir).filter((x) => /^attempt\d+\.log$/.test(x))) {
      const n = (fs.readFileSync(path.join(dir, f), "utf8").match(/^\[motion \d+\]/gm) ?? []).length;
      motionRounds = Math.max(motionRounds, n);
    }
  }
  if (/\[env\] locked/.test(log)) base.stage = "Gemini is refining the motion against the video";
  else if (/== PHASE 1/.test(log)) base.stage = "Gemini is rebuilding the scene from the video";
  else base.stage = "Reading the video";
  base.progress = Math.min(0.95, 0.05 + 0.3 * Math.min(envRounds, 2) / 2 + 0.6 * Math.min(motionRounds, 4) / 4);

  const final = path.join(dir, "final.json");
  if (fs.existsSync(final)) {
    try {
      base.understood = (JSON.parse(fs.readFileSync(final, "utf8")) as { summary?: string }).summary;
    } catch {}
  } else if (fs.existsSync(path.join(dir, "environment_locked.json"))) {
    try {
      base.understood = (JSON.parse(fs.readFileSync(path.join(dir, "environment_locked.json"), "utf8")) as { summary?: string }).summary;
    } catch {}
  }
  const sbs = path.join(dir, "best_side_by_side.mp4");
  if (fs.existsSync(sbs) && /\[done\]/.test(log)) {
    base.state = "done";
    base.stage = "Done";
    base.progress = 1;
    base.video = `/api/mirror/${id}/video`;
    try {
      const S = JSON.parse(fs.readFileSync(path.join(dir, "summary.json"), "utf8"));
      base.scores = { scene: S.env?.best_score, motion: S.motion?.best_score };
      base.elapsed_s = Math.round(S.total_s ?? base.elapsed_s);
    } catch {}
    return base;
  }
  if (alive(job.pid)) {
    base.state = "running";
    return base;
  }
  base.state = "error";
  const tb = log.split("\n").filter((l) => /Error|Exception/.test(l)).slice(-1)[0];
  base.error = tb ? tb.trim().slice(0, 300) : "The run stopped before finishing.";
  return base;
}

export const resultVideo = (id: string) => path.join(runDir(id), "best_side_by_side.mp4");
