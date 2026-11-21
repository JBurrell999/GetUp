// Runs the Gemini -> robot pipeline (motion_mirror) on a video, when the site runs next to it (local / team laptop).
import { available, startJob } from "@/lib/mirror";
import { codeMatches } from "@/lib/takes";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const MAX_MB = 300;

export async function GET() {
  return Response.json({ available: available() });
}

export async function POST(request: Request) {
  if (!available()) {
    return Response.json({ error: "The robot pipeline isn't running on this server." }, { status: 503 });
  }
  const form = await request.formData().catch(() => null);
  const file = form?.get("video");
  if (!(file instanceof File) || file.size === 0) {
    return Response.json({ error: "Attach a video." }, { status: 400 });
  }
  if (file.size > MAX_MB * 1e6) {
    return Response.json({ error: `Keep it under ${MAX_MB} MB (a short clip is best).` }, { status: 413 });
  }
  if (!codeMatches(form?.get("code"), process.env.UPLOAD_CODE)) {
    return Response.json({ error: "The event code is wrong." }, { status: 403 });
  }
  const what = String(form?.get("what") ?? "").slice(0, 200);
  const r = await startJob(file, what);
  if (r.error) return Response.json({ error: r.error }, { status: 429 });
  return Response.json({ id: r.id });
}
