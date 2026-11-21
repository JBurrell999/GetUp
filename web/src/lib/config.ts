import "server-only";

export type BlobAccess = "public" | "private";

export interface PublicConfig {
  /** A Blob store is connected; without it the page works but can't upload. */
  uploadsReady: boolean;
  /** UPLOAD_CODE is set, so people are asked for it once. */
  codeRequired: boolean;
  access: BlobAccess;
}

export const blobAccess = (): BlobAccess => (process.env.BLOB_ACCESS === "public" ? "public" : "private");

export function publicConfig(): PublicConfig {
  return {
    uploadsReady: Boolean(process.env.BLOB_READ_WRITE_TOKEN),
    codeRequired: Boolean(process.env.UPLOAD_CODE),
    access: blobAccess(),
  };
}

export const notConfigured = () =>
  Response.json(
    { error: "Uploads aren't switched on for this site yet. Connect a Vercel Blob store and redeploy." },
    { status: 503 },
  );
