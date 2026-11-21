import { jobStatus, validId } from "@/lib/mirror";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(_req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  if (!validId(id)) return Response.json({ error: "Unknown run." }, { status: 404 });
  return Response.json(jobStatus(id));
}
