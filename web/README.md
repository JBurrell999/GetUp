# GetUp capture site

The page people use to film themselves doing the tasks in `../tasks.json` in the event room and
upload one video per task. Next.js on Vercel; videos go straight from the browser to a
Vercel Blob store (the site's own functions never carry the file, so there's no 4.5 MB limit).

## Deploy

1. In Vercel, **Add New → Project**, import this repo, and set **Root Directory** to `web`.
   Leave "Include files outside the root directory" on so the build can read `../tasks.json`.
2. In the project, **Storage → Create → Blob**. Choose **private** access (the videos show
   people's faces) and connect it to the project. That sets `BLOB_READ_WRITE_TOKEN`.
3. Optional environment variables:
   - `UPLOAD_CODE`: people must type this once before uploading. Matching ignores case.
   - `BLOB_ACCESS=public`: only if you created a public store instead. Must match the store.
4. Redeploy. Until a store is connected the site works but shows an "Uploads off" strip.

## Local

```bash
npm install
cp .env.example .env.local   # paste the store's token, or: vercel env pull .env.local
npm run dev
```

`npm run dev` and `npm run build` first copy `../tasks.json` into `src/data/`, so the list on
the site is always the one `session.py` uses. Commit the copy along with task changes.

The floor plan is drawn from the MuJoCo room. After changing the room, regenerate it:

```bash
node scripts/build-room.mjs /path/to/g1_room_mujoco/layout.json
```

The capture area and camera spot shown on the plan are set in `src/lib/capture.ts`.

## What lands in the store

```
takes/<judge>/<task_id>/<yyyymmdd-hhmmss>-<rand>.<mp4|mov|m4v|webm>   the video
takes/<judge>/<task_id>/<yyyymmdd-hhmmss>-<rand>.json                 details, below
```

`<judge>` is a slug of the name the person typed plus four random characters, so takes from
two people with the same name stay apart. The JSON next to each video records the judge's
name, the task (id, number, tier, stop rule, max length), the video's size, duration and
resolution as read in the browser, the original filename, and when consent was given. Times
in paths are UTC.

Every upload is checked on the server: the path must match the pattern above, the task must
exist in `tasks.json`, the file must be a video under 2 GB, existing files are never
overwritten, and the event code must match if one is set.

Browse or download takes from the store's page in the Vercel dashboard, or with the
`@vercel/blob` SDK's `list({ prefix: "takes/" })` using the same token.
