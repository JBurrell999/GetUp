#!/usr/bin/env node
// The robot worker's access to the Blob store. motion_mirror/src/motion_mirror/site_worker.py runs it
// on the computer that does the MuJoCo rendering; the site itself never uses it.
//
//   node scripts/blob.mjs list <prefix>                         one JSON object per line: pathname, size, uploadedAt
//   node scripts/blob.mjs get  <pathname> <dest-file>           download, bypassing the CDN cache
//   node scripts/blob.mjs put  <pathname> <file> <content-type> upload, replacing what was there
//
// put only writes under outputs/, so the worker can't change an upload or its details under takes/.
// Needs BLOB_READ_WRITE_TOKEN in the environment, and BLOB_ACCESS=public only if the store is public.
import { createReadStream, createWriteStream, statSync } from "node:fs";
import { mkdir, rename, rm } from "node:fs/promises";
import { dirname } from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { get, list, put } from "@vercel/blob";

const access = process.env.BLOB_ACCESS === "public" ? "public" : "private";
const MULTIPART_BYTES = 32 * 1024 ** 2;

function fail(message) {
  console.error(message);
  process.exit(1);
}

async function listCmd(prefix) {
  let cursor;
  do {
    const page = await list({ prefix, cursor, limit: 1000 });
    for (const b of page.blobs) {
      console.log(JSON.stringify({ pathname: b.pathname, size: b.size, uploadedAt: new Date(b.uploadedAt).toISOString() }));
    }
    cursor = page.hasMore ? page.cursor : undefined;
  } while (cursor);
}

async function getCmd(pathname, dest) {
  const result = await get(pathname, { access, useCache: false });
  if (!result || result.statusCode !== 200) fail(`not found: ${pathname}`);
  await mkdir(dirname(dest), { recursive: true });
  // Write next to the destination, then rename, so an interrupted download never looks complete.
  const part = `${dest}.part`;
  try {
    await pipeline(Readable.fromWeb(result.stream), createWriteStream(part));
    await rename(part, dest);
  } catch (err) {
    await rm(part, { force: true });
    throw err;
  }
  console.log(JSON.stringify({ pathname, size: statSync(dest).size }));
}

async function putCmd(pathname, file, contentType) {
  if (!pathname.startsWith("outputs/") || pathname.split("/").some((p) => p === ".." || p === "")) {
    fail(`refusing to write outside outputs/: ${pathname}`);
  }
  const size = statSync(file).size;
  const blob = await put(pathname, createReadStream(file), {
    access,
    contentType,
    addRandomSuffix: false,
    allowOverwrite: true,
    multipart: size > MULTIPART_BYTES,
    // result.json changes from "processing" to "complete", so don't let caches hold it long
    ...(contentType === "application/json" ? { cacheControlMaxAge: 60 } : {}),
  });
  console.log(JSON.stringify({ pathname: blob.pathname, size }));
}

const [cmd, ...args] = process.argv.slice(2);
if (!process.env.BLOB_READ_WRITE_TOKEN) fail("BLOB_READ_WRITE_TOKEN is not set");
try {
  if (cmd === "list" && args.length === 1) await listCmd(args[0]);
  else if (cmd === "get" && args.length === 2) await getCmd(args[0], args[1]);
  else if (cmd === "put" && args.length === 3) await putCmd(args[0], args[1], args[2]);
  else fail("usage: blob.mjs list <prefix> | get <pathname> <dest> | put <pathname> <file> <content-type>");
} catch (err) {
  fail(err instanceof Error ? err.message : String(err));
}
