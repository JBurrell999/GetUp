// GET /api/robot?take=<take video pathname>&file=side_by_side|robot|wide
// Streams one of the robot's videos for a take from the private Blob store (see outputPrefix in lib/takes.ts).
// Byte ranges are passed through, which Safari needs to play video at all.
//
// Like GET /api/takes, this trusts whoever knows the take's path: it holds the uploader's random judge id and
// a random suffix, and the page only learns it from the uploader's own list. side_by_side shows the person.
import { get } from "@vercel/blob";
import { blobAccess } from "@/lib/config";
import { isRobotFile, outputPrefix, parseTakePath, ROBOT_FILES } from "@/lib/takes";

const PASS = ["content-type", "content-length", "content-range", "etag", "last-modified"];

export async function GET(request: Request) {
  if (!process.env.BLOB_READ_WRITE_TOKEN) return new Response("Storage isn't connected.", { status: 503 });
  const q = new URL(request.url).searchParams;
  const take = parseTakePath(q.get("take") ?? "");
  const file = q.get("file") ?? "side_by_side";
  const prefix = take && take.ext !== "json" ? outputPrefix(take) : null;
  if (!prefix || !isRobotFile(file)) return new Response("Unknown video.", { status: 400 });

  const range = request.headers.get("range");
  let result;
  try {
    result = await get(prefix + ROBOT_FILES[file], {
      access: blobAccess(),
      useCache: false,
      ...(range ? { headers: { range } } : {}),
    });
  } catch (err) {
    console.error("[robot] get failed", err);
    return new Response("Storage didn't answer.", { status: 502 });
  }
  if (!result || result.statusCode !== 200) return new Response("Not ready yet.", { status: 404 });

  const headers = new Headers({ "accept-ranges": "bytes", "cache-control": "private, no-store" });
  for (const h of PASS) {
    const v = result.headers.get(h);
    if (v) headers.set(h, v);
  }
  if (!headers.has("content-type")) headers.set("content-type", "video/mp4");
  // The robot's videos are a few MB, so read the (partial) body and send it with an exact length:
  // Safari won't play a range response it can't check, and streamed responses can lose the header.
  const body = await new Response(result.stream).arrayBuffer();
  headers.set("content-length", String(body.byteLength));
  return new Response(body, { status: headers.has("content-range") ? 206 : 200, headers });
}
