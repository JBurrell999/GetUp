// Hands the browser a short-lived token so the video goes straight to Vercel Blob.
// Serverless functions can't take request bodies over ~4.5 MB, so the file never passes
// through here; this route only decides whether the upload is allowed.
import { handleUpload, type HandleUploadBody } from "@vercel/blob/client";
import { notConfigured } from "@/lib/config";
import { TASK_BY_ID } from "@/lib/tasks";
import { codeMatches, MAX_BYTES, parseTakePath } from "@/lib/takes";

export async function POST(request: Request) {
  if (!process.env.BLOB_READ_WRITE_TOKEN) return notConfigured();
  const body = (await request.json()) as HandleUploadBody;
  try {
    const result = await handleUpload({
      body,
      request,
      onBeforeGenerateToken: async (pathname, clientPayload) => {
        const take = parseTakePath(pathname);
        if (!take || take.ext === "json" || !TASK_BY_ID.has(take.task)) {
          throw new Error("That upload path isn't one this site creates.");
        }
        let payload: { code?: unknown } = {};
        try {
          payload = JSON.parse(clientPayload ?? "{}");
        } catch {}
        if (!codeMatches(payload.code, process.env.UPLOAD_CODE)) {
          throw new Error("The event code is wrong. Ask the team for the current one.");
        }
        return {
          allowedContentTypes: ["video/*"],
          maximumSizeInBytes: MAX_BYTES,
          addRandomSuffix: false,
          allowOverwrite: false,
        };
      },
    });
    return Response.json(result);
  } catch (err) {
    return Response.json({ error: err instanceof Error ? err.message : "Upload refused." }, { status: 400 });
  }
}
