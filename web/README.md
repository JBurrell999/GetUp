# GetUp capture site

The page people use to upload videos of themselves doing anything they like, one thing per video,
for the simulated Unitree G1 to learn to copy. There's no fixed task list: a few examples (in
`src/lib/examples.ts`) are offered as ideas, and people describe each video in their own words
when they upload it. The top of the page is the project's own MuJoCo scene running
in the browser: the Unitree G1 walking a loop through the room, and a "Where to stand" view
that flies to the filming spot.

Next.js on Vercel. Videos go straight from the browser to a private Vercel Blob store; the
site's functions only hand out short-lived upload tokens, so there's no 4.5 MB body limit.

## Robot section ("From a phone video to the robot")

Plays the best side-by-side results (`public/mirror/*.mp4`, human | robot) and, when the site runs on the same machine
as the pipeline, lets anyone upload a video and run it: `POST /api/mirror` saves the file and starts
`motion_mirror/mm mirror <video>` in the background, `GET /api/mirror/<id>` reports progress and what Gemini understood,
and `GET /api/mirror/<id>/video` streams the finished side-by-side. It needs `motion_mirror/.env` (Gemini key) and its
Python environment, so it only works locally (`npm run dev` next to the repo, shareable with ngrok); on Vercel the
section shows the gallery and says the pipeline runs on the team laptop. At most 2 runs at once; a run takes ~2-3 min.
If `UPLOAD_CODE` is set, the same event code is required. `MIRROR_DIR` overrides where the pipeline lives.

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

## Share with ngrok

Keep `npm run dev` running, then run this in another terminal:

```bash
ngrok http 3000
```

Share the HTTPS forwarding URL ngrok prints. Keep both processes running and this computer
awake while others use the link. If ngrok shows a browser warning, visitors can select
**Visit Site** to continue. The development server allows `*.ngrok-free.app` origins so
assets and hot reload work through the tunnel. If your ngrok URL uses a different domain,
add its hostname to `allowedDevOrigins` in `next.config.ts`.

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

## Where people are filming

Before a first upload, people pick where they're filming. They choose from the "Where are you
filming?" section or the first step of the upload panel, which hides the drop zone until a place
is chosen. The choices are the three MuJoCo scenes in `../g1_room_mujoco/`:

| id | Place | Scene the sim uses |
|---|---|---|
| `hackathon` | Hackathon room (the event room) | `Hackathon_G1_Ultra_Detail_V2/generated/G1_HACKATHON_V2.xml` |
| `kitchen` | Home kitchen | `Kitchen_G1_Detail/generated/G1_KITCHEN.xml` |
| `warehouse` | Distribution warehouse | `Warehouse_G1_Detail/generated/G1_WAREHOUSE.xml` |

How the choice is handled:

- **Remembered:** the choice is stored on the device and can be changed per video.
- **Link preset:** `?env=kitchen` on the link preselects it, for example from a QR code posted in
  that room.
- **Source:** names, blurbs and scene paths are in `src/lib/environments.ts`.

The screenshots in `public/envs/` are rendered from the generated G1 scenes, with the robot at its
start pose. Re-render them after changing a scene:

```bash
.venv/bin/python web/scripts/sim/screenshots.py            # or: ... screenshots.py kitchen
```

The 3D view at the top still shows the event room.

## What lands in the store

```
takes/<judge>/<env>/<what>/<yyyymmdd-hhmmss>-<rand>.<mp4|mov|m4v|webm>   the video
takes/<judge>/<env>/<what>/<yyyymmdd-hhmmss>-<rand>.json                 details, below
```

**Path parts:**

- `<judge>` is a slug of the name the person typed plus four random characters, so two people
  with the same name stay apart.
- `<env>` is where they filmed: `hackathon`, `kitchen` or `warehouse`.
- `<what>` is a slug of what they said the video shows ("Move a chair" becomes `move_a_chair`).
  The JSON has their exact words.
- **Older uploads:** they used a task id from the old fixed list in the `<what>` slot, and some
  have no `<env>` level. Both shapes are still listed and still accepted, so a page opened before
  a change can finish its upload.
- Times in paths are UTC.

**The JSON next to each video** (`version: 3`) has:

- the judge's name
- `environment`: `{ id, name, scene }`, where `scene` is the MuJoCo file to copy the video in,
  relative to the repo root
- `activity`: `{ description, slug, example }`, where `description` is what they typed and
  `example` is `{ id, name }` if they picked one of the site's ideas, otherwise `null`
- the video's size, duration and resolution as read in the browser
- the original filename
- when consent was given

**Every upload is checked on the server:**

- the path must match the pattern above, and any `<env>` must be one of the three
- the details must include a description of what the video shows
- the file must be a video under 2 GB
- nothing is ever overwritten
- the event code must match, if one is set

Browse or download takes with `vercel blob list` / `vercel blob get`, the store's page in the
Vercel dashboard, or the `@vercel/blob` SDK's `list({ prefix: "takes/" })`.
