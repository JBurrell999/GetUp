// GET  /api/takes?judge=<id>  the takes one person has uploaded (paths and sizes, no URLs)
// POST /api/takes             records who uploaded a take and on what terms, next to the video
import { head, list, put } from "@vercel/blob";
import { blobAccess, notConfigured } from "@/lib/config";
import { TASK_BY_ID } from "@/lib/tasks";
import { codeMatches, isJudgeId, metaPathname, parseTakePath, type TakeSummary } from "@/lib/takes";

export async function GET(request: Request) {
  const judge = new URL(request.url).searchParams.get("judge") ?? "";
  if (!isJudgeId(judge)) return Response.json({ error: "Missing or malformed judge id." }, { status: 400 });
  if (!process.env.BLOB_READ_WRITE_TOKEN) return Response.json({ takes: [] });

  const takes: TakeSummary[] = [];
  let cursor: string | undefined;
  try {
    do {
      const page = await list({ prefix: `takes/${judge}/`, cursor, limit: 1000 });
      for (const b of page.blobs) {
        const p = parseTakePath(b.pathname);
        if (!p || p.ext === "json") continue;
        takes.push({ pathname: b.pathname, task: p.task, uploadedAt: new Date(b.uploadedAt).toISOString(), size: b.size });
      }
      cursor = page.hasMore ? page.cursor : undefined;
    } while (cursor);
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
  video?: { duration?: unknown; width?: unknown; height?: unknown; size?: unknown; type?: unknown; filename?: unknown };
}

const num = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : null);
const str = (v: unknown, max = 200) => (typeof v === "string" ? v.slice(0, max) : null);

export async function POST(request: Request) {
  if (!process.env.BLOB_READ_WRITE_TOKEN) return notConfigured();
  const body = (await request.json().catch(() => ({}))) as MetaBody;

  const take = parseTakePath(body.pathname ?? "");
  const task = take && TASK_BY_ID.get(take.task);
  if (!take || take.ext === "json" || !task) return Response.json({ error: "Unknown take." }, { status: 400 });
  if (!codeMatches(body.code, process.env.UPLOAD_CODE)) return Response.json({ error: "Wrong event code." }, { status: 403 });
  if (body.consent !== true) return Response.json({ error: "Consent is required." }, { status: 400 });

  let blob;
  try {
    blob = await head(body.pathname!);
  } catch {
    return Response.json({ error: "The video isn't in storage yet." }, { status: 404 });
  }

  const v = body.video ?? {};
  const meta = {
    version: 1,
    judge: { id: take.judge, name: str(body.name, 80) },
    task: { id: task.id, no: task.no, name: task.name, category: task.category, tier: task.tier, stop: task.stop, max_s: task.max_s, still_s: task.still_s },
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
