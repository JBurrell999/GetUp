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
