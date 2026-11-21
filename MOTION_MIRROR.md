# Motion Mirror: casual video → Unitree G1

Gemini watches a phone video of a person (picking up a bottle, moving a chair…) and maps it onto a Unitree G1 in MuJoCo.

## Setup

```bash
uv sync                                     # Python 3.12; mujoco, torch, google-genai, opencv
echo "GEMINI_API_KEY=..." > .env            # not committed
```

Run modules through `./mm <module> ...`. The script sets `PYTHONPATH=src` and loads `.env`. On macOS the editable-install `.pth` file can get a "hidden" flag that Python then skips, and the wrapper avoids that problem.

## Stage 1: Gemini mapping (kinematic, no RL). Status: working, being refined

```bash
./mm mirror videos/clip.mp4 --name bottle --fact "uses his RIGHT hand"
```

`src/motion_mirror/mirror.py` runs two phases, both on `gemini-4-hydrogen`:

1. **Environment.** Gemini recreates the surfaces, the object, the camera and the person's start pose. Each round renders one still from the estimated camera and puts it next to the video's first frame. Gemini judges the pair and revises. The accepted result is locked.
2. **Motion.** Inside the locked scene, Gemini writes the keyframes, the grasp and release times, and where the object ends up. Each round:
   - kinematic replay on the G1 with Dex3 hands (`replay.py`),
   - collision, grasp and placement checks,
   - a time-synced side-by-side video,
   - Gemini judges it and revises the motion.

   If a round scores worse, the loop reverts to the best motion so far.

What `replay.py` handles:
- smooth PCHIP splines through the keyframes,
- 7-DoF arm IK for exact palm position and wrist orientation at each grasp,
- fingers curl until they touch the object,
- two-handed carries solved as a closed chain,
- exact placement at the object's end position,
- interpenetration reports that are fed back to Gemini.

Outputs go to `runs/mirror2/<name>/` (not committed). The best result is also copied to `benchmarks/`.

`--fact` adds a user-verified fact that Gemini may not contradict, for example which hand the person uses.

## Stage 2: physics / sim-to-real (RL). Status: in progress. This is where walking goes

- **`skill_env.py`**: the whole-body skill policy environment. It takes commands for walking velocity (vx, vy, wz), pelvis height (squat) and 17 upper-body joint targets.
  - The policy sees only what the real robot has: the pelvis IMU, joint encoders and the commands.
  - The critic also gets privileged sim state (asymmetric actor-critic).
  - Domain randomization covers friction, masses, a torso payload, PD gains, torque limits, joint friction, 0–20 ms latency, pushes up to 120 N and sensor noise. Episodes alternate between "walk" and "manip".
- **`config.py`**: joint order, the default pose, PD gains (close to the real G1's, but verify them before running on hardware) and the DR ranges.
- **`ppo.py`**, **`vec_env.py`**, **`scripts/train.py`**: PPO across CPU worker processes, about 13k steps/s on an M5 Pro.

  ```bash
  uv run scripts/train.py --env skill --name skill --init runs/skill_v2/latest.pt --leg-std 1.0 --workers 17 --envs-per-worker 10
  ```

- **`scripts/evaluate.py`** scores a checkpoint and renders it. **`plan.py`** executes a command plan with the policy and renders it.

**Where walking is now:** the policy balances, squats and tracks arm motion with a payload under pushes (about 870 of 1,000 steps per episode, 23–30% falls with exploration noise). Walking is only starting to emerge: the velocity error dropped from 0.32 to 0.23 m/s over about 3,500 iterations. Restarting with leg exploration std 1.0 (`--leg-std`) is what got it moving.

Ideas to speed this up:
- a dedicated walking-only run (no arm gestures or squats) that is merged or switched with the manipulation policy,
- a feet-air-time reward,
- GPU training (MJX / MuJoCo Playground).

Checkpoints live in `runs/` and are not committed.

## Assets / disclosures

The G1 MJCF models come from MuJoCo Menagerie (copyright Unitree, see `assets/unitree_g1/LICENSE`). The full list is in `DISCLOSURES.md`.
