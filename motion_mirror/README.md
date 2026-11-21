# Motion Mirror: casual video → Unitree G1

Record anyone on a phone, from any angle and with no calibration, doing something like picking up a bottle or moving a chair. **Gemini** works out what they are doing, the environment, and the object. A Unitree G1 then reproduces it in MuJoCo, from the same camera viewpoint, with three-finger hands that actually grasp.

```
video ──► Gemini: environment ──► still from same camera ──► Gemini judges ──► (repeat) ──► LOCKED scene
                                                                                              │
          Gemini: motion ──► kinematic replay on G1 ──► collisions / grasp / placement checks ◄─┘
                 ▲                        │
                 └──── Gemini judges ◄── side-by-side video (human | robot)     (repeat until equivalent)

stage 2 (in progress): physics + RL policy + domain randomization  →  sim-to-real
```

## Quick start

```bash
cd motion_mirror
uv sync                                   # Python 3.12
echo "GEMINI_API_KEY=..." > .env          # not committed

# stage 1: map a video onto the robot (about 1-2 min per round)
./mm mirror videos/IMG_6520.mp4 --name bottle \
    --fact "The person uses his RIGHT hand (his arm comes from the LEFT side of the image)"
./mm mirror videos/IMG_6521.mp4 --name chair \
    --fact "The person lifts the chair with BOTH hands on the top of the backrest"
```

Results go to `runs/mirror2/<name>/`:

| Path | Contents |
|---|---|
| `env/iterK/still.jpg` | Phase 1: first frame next to the sim recreation |
| `env/iterK/judge.json` | Gemini's score and differences for that round |
| `environment_locked.json` | The accepted scene |
| `motion/iterK/side_by_side.mp4` | Phase 2: human \| robot, time-synced |
| `motion/iterK/telemetry.json` | Collisions, grasp accuracy, placements |
| `motion/iterK/judge.json` | Score, differences, revised motion |
| `final.json` | The best complete interpretation (scene + motion) |

The best side-by-side and its JSON are also copied to `benchmarks/`.

Videos longer or larger than about 1080p should be transcoded first (the pipeline uses about 854 px):
`ffmpeg -i in.MOV -vf "scale='if(gt(iw,ih),854,-2)':'if(gt(iw,ih),-2,854)'" -r 30 -an out.mp4`

## Capture-site uploads → robot videos on Vercel

`site_worker` reads uploads from the site's Vercel Blob store, runs each one in the room its
uploader picked, and puts the robot videos back in the same store under `outputs/`:

```bash
./mm site_worker --list          # every take and its status; changes nothing
./mm site_worker                 # process every new take, then exit
./mm site_worker --watch 60      # keep running, checking for new takes every minute
./mm site_worker --take takes/<judge>/<env>/<what>/<stamp>.mp4 [--redo]
```

For each take it writes:

```
outputs/<judge>/<env>/<what>/<stamp>/result.json        status, room, scores, placement, collisions
outputs/<judge>/<env>/<what>/<stamp>/robot.mp4          the G1 in the room, from the filming camera
outputs/<judge>/<env>/<what>/<stamp>/side_by_side.mp4   the person | the robot, time-synced
outputs/<judge>/<env>/<what>/<stamp>/wide.mp4           a wider view of the room
```

- **Same path as the upload:** `takes/<x>.mp4` gets its results in `outputs/<x>/`.
- **Status:** `result.json` is written first with `"status": "processing"`, then `"complete"` or
  `"failed"` (with the error). A take that has a `result.json` is skipped; `--retry` picks up failed
  ones. If Gemini had already finished a failed take, it is rendered again without new Gemini calls.
  `--redo` starts one take over.
- **Never touches `takes/`:** the Blob helper (`../web/scripts/blob.mjs`) refuses to write outside
  `outputs/`.
- **Token:** `BLOB_READ_WRITE_TOKEN` from the environment, or else `../web/.env.local`
  (`vercel env pull`). Needs Node and `npm install` in `../web`.
- **One at a time:** run a single worker at a time.

**The room is always the one the uploader chose.** It comes from the `<env>` in the take's path,
which the site also writes into the take's JSON. If the two disagree, the take fails before any
Gemini call. Uploads from before the site asked where people film have no room. They're skipped
until you run one with `--take ... --env kitchen`. A take filmed in one room can't be run in another.

The room is used throughout, not only for the final render. Every phase 2 round, the one Gemini
judges, is replayed inside the room:

- The robot is placed on clear floor there and rendered from Gemini's estimated camera.
- If a wall or fridge is in the way, the camera is pulled in.
- Collisions with the room's furniture are reported to Gemini like any other.

Phase 1 (matching Gemini's reconstructed table, object and camera to the first frame) stays in the
plain scene, because unrelated room furniture would only confuse that comparison. `./mm mirror
... --room kitchen` does the same outside the worker.

The rooms are `../g1_room_mujoco/*/*_VISUAL.xml`, the same rooms as the site's `generated/G1_*.xml`
scenes. Those files contain the stock G1 with rubber hands. The replay merges in the G1 with Dex3
hands it needs for grasping.

### One local file

`site_worker` downloads to `runs/site_downloads/` and calls `site_job`, which works on local files:

```bash
./mm site_job /path/to/take.mp4 --metadata /path/to/take.json --name uploaded_sample
```

The metadata selects `hackathon`, `kitchen` or `warehouse`, and `--env` overrides it. Without
either, the job is refused. Scene paths in uploaded JSON are recorded as provenance; rendering uses
the project's known room XML files.

It runs the interpretation/judging loop in that room, then writes
`runs/mirror2/uploaded_sample/environment/robot_env.mp4`, a time-synced
`side_by_side_env.mp4`, a wider `robot_wide.mp4`, and a reloadable `scene.xml` with
absolute asset paths for this checkout. `job.json` records the interpretation engine, source SHA-256,
upload metadata, room placement, scores, collision telemetry, and output paths. These are
kinematic MuJoCo replays.

Choose a unique name for each source. If interpretation finishes but room rendering is
interrupted, repeat the same command with `--resume` to render `final.json` without new
Gemini calls. `--env-iters`, `--iters`, `--model`, `--judge-model`, and `--fact` work as in
`mirror`; `--no-wide` skips the second room camera.

For a local workflow, first extract MediaPipe world landmarks using an existing model file.
The default model is `../models/mediapipe/pose_landmarker_heavy.task`; the extractor never
downloads a model. Crop and subject-center coordinates are pixels in the source video.
These values select the central subject in the uploaded hackathon sample:

```bash
.venv/bin/python scripts/extract_local_pose.py /path/to/take.mp4 \
    --out runs/local_landmarks --crop 245 90 610 480 --subject-center 392 263
./mm local_pose /path/to/take.mp4 --landmarks runs/local_landmarks/local_pose.npz \
    --out runs/mirror2/local_sample
./mm site_job /path/to/take.mp4 --metadata /path/to/take.json \
    --name local_sample --resume
```

For another video, choose its crop and shoulder center, or omit both to use the full frame
and frame center. `--model /path/to/existing.task` selects another installed model. The
extraction directory contains `local_pose.npz`, an overlay video, snapshots, and detection
statistics. The `local_pose` stage writes `final.json` and `summary.json`; the room worker
preserves the declared interpretation engine, extraction details, and assumptions in `job.json`.
The local retarget preset assumes this seated water-bottle action: legs, table dimensions
and contact times are estimates. Adjust those annotations before using it for another activity.

## How stage 1 works (`src/motion_mirror/mirror.py`, `replay.py`)

**Phase 1, environment.** Gemini 4 Hydrogen reads the video at 2 fps and recreates the scene:
- surfaces (tables are drawn as a top with legs; "solid" for counters and stools),
- the object (type, size, colour, start pose, grasp style),
- the camera (distance, height, azimuth, field of view),
- the person's starting posture.

One still is rendered from the estimated camera and placed next to the video's first frame, along with any collisions (for example, legs inside the table). Gemini corrects the scene until it matches, and then the scene is locked.

**Phase 2, motion.** Gemini reads the video at 5 fps and writes timed keyframes inside the locked scene:
- pelvis position and heading, trunk lean, side bend and twist, thighs and knees, and four angles per arm,
- grasp and release times and which hand is used,
- where the object ends up.

The replay is purely kinematic, with no physics:
- **Smooth motion:** a shape-preserving spline (PCHIP) runs through the keyframes, so the robot doesn't stop at each one, plus light Gaussian smoothing.
- **Feet on the floor:** the root height is solved every frame.
- **Grasping:** 7-DoF arm IK puts the palm on the object at the right wrist orientation. The style is a side grip for bottles, overhand for a chair backrest, or palms on the faces for boxes. The correction blends in before the grasp and out after it.
- **Fingers:** the Dex3 fingers curl until their tips touch the object.
- **Carrying:** a one-handed object is held rigidly in the hand and tilts when drinking. With two hands the object stays upright and turns with the body, and both hands are re-solved onto it every frame.
- **Placing:** the arm is solved so the object lands exactly at its end pose, upright on the surface below it.
- **Collision checks:** these catch the robot or the object going into the table or floor, and the robot going through the object. They are reported back to Gemini as, for example, "right hand inside table 4 cm, 4.2-5.7 s".

Gemini then judges a single side-by-side video (4 fps, with timestamps burned in) together with the telemetry, and revises the motion. If a round scores worse, the loop reverts to the best motion so far and tells Gemini which changes to avoid.

`--fact` locks a user-verified fact into every prompt, for example which hand the person uses.

## Stage 2: physics, RL, walking (in progress)

| File | Role |
|---|---|
| `src/motion_mirror/skill_env.py` | Whole-body skill policy env. Commands: walking (vx, vy, wz), pelvis height (squat), 17 upper-body joint targets. Episodes alternate between walk and manip modes, with a random payload in the hands. |
| `src/motion_mirror/g1_env.py` | Base env: the actor sees only real-robot sensors (pelvis IMU, joint encoders). The critic also gets privileged sim state. |
| `src/motion_mirror/config.py` | Joint order, default pose, PD gains (close to the real G1's; **verify before hardware**), domain-randomization ranges |
| `src/motion_mirror/ppo.py`, `vec_env.py`, `scripts/train.py` | PPO across CPU processes (about 13k steps/s on an M5 Pro) |
| `scripts/evaluate.py` | Score a checkpoint and render it |
| `src/motion_mirror/plan.py`, `ik.py`, `gemini_loop.py` | Earlier executor: run command plans through the policy with physics |

Domain randomization covers friction, link masses, a torso payload and centre-of-mass shift, PD gains, torque limits, joint friction and armature, encoder offsets, 0–20 ms latency, pushes up to 120 N, and IMU and encoder noise.

```bash
uv run scripts/train.py --env skill --name skill --init runs/skill_v2/latest.pt --leg-std 1.0 \
    --workers 17 --envs-per-worker 10        # checkpoints -> runs/skill/ckpts (not committed)
uv run scripts/evaluate.py runs/skill/latest.pt --video out.mp4
```

**Status:** the policy balances, squats and tracks arm motion with a payload under pushes (about 870 of 1,000 steps per episode). Walking is only starting to emerge: the velocity error fell from 0.32 to 0.23 m/s. Resetting the leg exploration noise to 1.0 (`--leg-std`) is what got it moving. Possible next steps:
- a walking-only run that is then combined with or switched with the manipulation policy,
- a feet-air-time reward,
- GPU training (MJX / MuJoCo Playground).

## Layout

```
motion_mirror/
  mm                    run a module:  ./mm mirror ... | ./mm plan ...   (sets PYTHONPATH, loads .env)
  view                  live 3D viewer for a command plan (mjpython)
  src/motion_mirror/    all code (see above)
  scripts/              train / evaluate / compare_models (Gemini model comparison)
  assets/unitree_g1/    G1 MJCF models, with and without Dex3 hands (MuJoCo Menagerie)
  videos/               test clips (bottle: IMG_6520, chair: IMG_6521)
  benchmarks/           best results so far (JSON; videos are regenerated)
  runs/                 outputs, not committed
```

On macOS, `uv`'s editable-install `.pth` file can get a "hidden" flag that Python 3.12+ skips. `./mm` sets `PYTHONPATH=src` so this doesn't matter.

## Disclosures

All pipeline code was written during the hackathon. The G1 models come from MuJoCo Menagerie (copyright Unitree). See `DISCLOSURES.md`.
