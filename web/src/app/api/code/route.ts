// Lets the name dialog check the event code before anyone spends time filming.
import { codeMatches } from "@/lib/takes";

export async function POST(request: Request) {
  const { code } = (await request.json().catch(() => ({}))) as { code?: unknown };
  const ok = codeMatches(code, process.env.UPLOAD_CODE);
  return Response.json({ ok });
}
