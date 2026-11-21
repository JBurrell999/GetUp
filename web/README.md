# GetUp capture site

The page people use to film themselves doing the tasks in `../tasks.json` in the event room
and upload one video per task. The top of the page is the project's own MuJoCo scene running
in the browser: the Unitree G1 walking a loop through the room, and a "Where to stand" view
that flies to the filming spot.

Next.js on Vercel. Videos go straight from the browser to a private Vercel Blob store; the
site's functions only hand out short-lived upload tokens, so there's no 4.5 MB body limit.

## Deploy

From this folder:

```bash
vercel link                                   # or import the repo with Root Directory = web
vercel blob create-store getup-takes --access private --yes   # sets BLOB_READ_WRITE_TOKEN
vercel deploy --prod
```

Optional environment variables:
- `UPLOAD_CODE`: people must type this once before uploading. Matching ignores case.
- `BLOB_ACCESS=public`: only if the store was created public. Must match the store.

Until a store is connected the site works but says uploads are off.

## Local

```bash
npm install
vercel env pull .env.local   # or copy .env.example and paste the store's token
npm run dev
```

`npm run dev` and `npm run build` first copy `../tasks.json` into `src/data/`, so the list on
the site is always the one `session.py` uses. When deploying this folder on its own, the
committed copy is used, so commit it with task changes.

## The 3D scene

`public/sim/` is generated from the scene XML by `scripts/sim/export.py`, using the repo's
Python environment (MuJoCo, NumPy, OpenCV):

```bash
.venv/bin/python web/scripts/sim/export.py \
    g1_room_mujoco/Hackathon_G1_Ultra_Detail_V2/generated/G1_HACKATHON_V2.xml
```

It writes the room's geoms, materials and lights as MuJoCo compiled them, the textures as
WebP, the G1's body tree with its meshes decimated (197k to 52k vertices), and one lap of the
walk. Commit the output; Vercel doesn't run Python.

The walk (`scripts/sim/walk.py`) is kinematic, because there's no walking controller for the
G1 in this project yet. It plans 70 footsteps around an oval in the centre aisle, moves the
pelvis along it with some sway and bob, rolls each foot onto its toe before lift-off, and
solves each leg's six joints with damped least squares on MuJoCo's own Jacobians. Planted
feet don't slide and the soles stay on the floor. The browser applies those joint angles with
the same forward kinematics MuJoCo uses, so the pose on screen is the sim's pose. When a
trained policy exists, export its rollout to `walk.bin` instead.

The filming area and tripod spot drawn in the room are set in `src/lib/capture.ts`.

## What lands in the store

```
takes/<judge>/<task_id>/<yyyymmdd-hhmmss>-<rand>.<mp4|mov|m4v|webm>   the video
takes/<judge>/<task_id>/<yyyymmdd-hhmmss>-<rand>.json                 details, below
```

`<judge>` is a slug of the name the person typed plus four random characters, so two people
with the same name stay apart. The JSON next to each video has the judge's name, the task
(id, number, tier, stop rule, max length), the video's size, duration and resolution as read
in the browser, the original filename, and when consent was given. Times in paths are UTC.

Every upload is checked on the server: the path must match the pattern above, the task must
exist in `tasks.json`, the file must be a video under 2 GB, nothing is ever overwritten, and
the event code must match if one is set.

Browse or download takes with `vercel blob list` / `vercel blob get`, the store's page in the
Vercel dashboard, or the `@vercel/blob` SDK's `list({ prefix: "takes/" })`.
