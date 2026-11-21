// GET  /api/takes?judge=<id>  the takes one person has uploaded (paths and sizes, no URLs), and how far
//                             the robot's copy of each has got
// POST /api/takes             records who uploaded a video, where, what it shows, and on what terms, next to it
import { get, head, list, put } from "@vercel/blob";
import { blobAccess, notConfigured } from "@/lib/config";
import { ENV_BY_ID } from "@/lib/environments";
import { EXAMPLES } from "@/lib/examples";
import {
  codeMatches,
  isJudgeId,
  metaPathname,
  outputPrefix,
  parseTakePath,
  ROBOT_FILES,
  type RobotFile,
  type RobotResult,
  type TakeSummary,
} from "@/lib/takes";

async function listAll(prefix: string) {
  const out: { pathname: string; size: number; uploadedAt: Date }[] = [];
  let cursor: string | undefined;
  do {
    const page = await list({ prefix, cursor, limit: 1000 });
    out.push(...page.blobs);
    cursor = page.hasMore ? page.cursor : undefined;
  } while (cursor);
  return out;
}

/** The robot's results for each take folder under outputs/<judge>/, keyed by folder. */
async function robotResults(judge: string) {
  const folders = new Map<string, Set<string>>();
  for (const b of await listAll(`outputs/${judge}/`)) {
    const cut = b.pathname.lastIndexOf("/") + 1;
    const folder = b.pathname.slice(0, cut);
    if (!folders.has(folder)) folders.set(folder, new Set());
    folders.get(folder)!.add(b.pathname.slice(cut));
  }
  const results = new Map<string, RobotResult>();
  await Promise.all(
    [...folders].map(async ([folder, names]) => {
      const files = (Object.keys(ROBOT_FILES) as RobotFile[]).filter((f) => names.has(ROBOT_FILES[f]));
      let status: RobotResult["status"] = files.length ? "complete" : "processing";
      let score: number | null = null;
      if (names.has("result.json")) {
        try {
          const r = await get(`${folder}result.json`, { access: blobAccess(), useCache: false });
          if (r?.statusCode === 200) {
            const j = (await new Response(r.stream).json()) as { status?: string; scores?: { motion?: unknown } };
            if (j.status === "processing" || j.status === "complete" || j.status === "failed") status = j.status;
            if (typeof j.scores?.motion === "number") score = j.scores.motion;
          }
        } catch (err) {
          console.error("[takes] result.json unreadable", folder, err);
        }
      }
      // while a take is being redone, its old videos are still there: don't offer them as the new result
      results.set(folder, { status, files: status === "complete" ? files : [], score });
    }),
  );
  return results;
}

export async function GET(request: Request) {
  const judge = new URL(request.url).searchParams.get("judge") ?? "";
  if (!isJudgeId(judge)) return Response.json({ error: "Missing or malformed judge id." }, { status: 400 });
  if (!process.env.BLOB_READ_WRITE_TOKEN) return Response.json({ takes: [] });

  const takes: TakeSummary[] = [];
  try {
    const [blobs, robot] = await Promise.all([listAll(`takes/${judge}/`), robotResults(judge)]);
    for (const b of blobs) {
      const p = parseTakePath(b.pathname);
      if (!p || p.ext === "json") continue;
      const prefix = outputPrefix(p);
      takes.push({
        pathname: b.pathname,
        env: p.env,
        what: p.what,
        uploadedAt: new Date(b.uploadedAt).toISOString(),
        size: b.size,
        robot: (prefix && robot.get(prefix)) || null,
      });
    }
  } catch (err) {
    console.error("[takes] list failed", err);
    return Response.json({ error: "Storage didn't answer." }, { status: 502 });
  }

  takes.sort((a, b) => a.uploadedAt.localeCompare(b.uploadedAt));
  return Response.json({ takes }, { headers: { "Cache-Control": "no-store" } });
}

interface MetaBody {
  pathname?: string;
  name?: unknown;
  code?: unknown;
  consent?: unknown;
  what?: unknown;
  example?: unknown;
  video?: { duration?: unknown; width?: unknown; height?: unknown; size?: unknown; type?: unknown; filename?: unknown };
}

const num = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : null);
const str = (v: unknown, max = 200) => (typeof v === "string" ? v.slice(0, max) : null);

export async function POST(request: Request) {
  if (!process.env.BLOB_READ_WRITE_TOKEN) return notConfigured();
  const body = (await request.json().catch(() => ({}))) as MetaBody;

  const take = parseTakePath(body.pathname ?? "");
  if (!take || take.ext === "json") return Response.json({ error: "Unknown video." }, { status: 400 });
  const what = str(body.what, 120)?.trim();
  if (!what) return Response.json({ error: "Say what the video shows." }, { status: 400 });
  const example = EXAMPLES.find((e) => e.id === body.example);
  if (!codeMatches(body.code, process.env.UPLOAD_CODE)) return Response.json({ error: "Wrong event code." }, { status: 403 });
  if (body.consent !== true) return Response.json({ error: "Consent is required." }, { status: 400 });

  let blob;
  try {
    blob = await head(body.pathname!);
  } catch {
    return Response.json({ error: "The video isn't in storage yet." }, { status: 404 });
  }

  const v = body.video ?? {};
  const env = take.env ? ENV_BY_ID.get(take.env) : undefined;
  const meta = {
    version: 3,
    judge: { id: take.judge, name: str(body.name, 80) },
    // where it was filmed, and the MuJoCo scene (relative to the repo root) to copy it in
    environment: env ? { id: env.id, name: env.name, scene: env.scene } : null,
    // what the person said they were doing, in their words; `example` when they picked one of the site's ideas
    activity: { description: what, slug: take.what, example: example ? { id: example.id, name: example.name } : null },
    video: {
      pathname: blob.pathname,
      size: blob.size,
      content_type: blob.contentType,
      uploaded_at: new Date(blob.uploadedAt).toISOString(),
      duration_s: num(v.duration),
      width: num(v.width),
      height: num(v.height),
      original_filename: str(v.filename),
    },
    consent: { given: true, at: new Date().toISOString() },
    user_agent: str(request.headers.get("user-agent"), 300),
  };

  await put(metaPathname(blob.pathname), JSON.stringify(meta, null, 2), {
    access: blobAccess(),
    contentType: "application/json",
    addRandomSuffix: false,
    allowOverwrite: true,
  });
  return Response.json({ ok: true });
}
