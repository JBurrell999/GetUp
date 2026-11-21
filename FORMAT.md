# Judge recordings: data format

Each bundle is one recording session: one judge performing a suite of everyday tasks,
filmed by a single camera. Every take carries the judge's 3D motion as the 24 joints of
the SMPL body model, ready to compare with a simulated humanoid doing the same task.

```
<session>/
  session.json              judge, camera, body measurements, tasks and accepted takes
  FORMAT.md                 this file
  NN_<task_id>/take<k>/
    joints3d.npz            3D motion: what to compare against the sim
    joints24.npz            2D image-space joints the 3D was derived from
    take.json               task, timing, why recording stopped, quality checks
    raw.mp4, capture_24pt.mp4   videos (only if exported with --with-video)
```

## joints3d.npz (load with `numpy.load`)

| key | shape | meaning |
|---|---|---|
| `xyz` | (T, 24, 3) float32 | joint positions in metres, world frame (below) |
| `conf` | (T, 24) | visibility 0..1; low = the joint was hidden and estimated |
| `t` | (T,) | seconds since the take started (camera timestamps, ~30 fps) |
| `joint_names`, `parents` | (24,) | SMPL joint order and kinematic tree (`-1` = root) |
| `mp_world`, `mp_vis` | (T, 33, 3), (T, 33) | raw MediaPipe landmarks the joints came from |
| `up` | (3,) | floor normal used, in the camera's axes |

Frames where the judge wasn't found are `NaN`.

**World frame:** right-handed, z up, floor at z = 0. y is the direction the camera was
looking (projected onto the floor) and x is the camera's right. The pelvis starts at
x = y = 0. To compare with a sim, align the two by the pelvis position and heading at
the first frame (or by a best-fit rigid transform), not by absolute coordinates.

**Joint order** (index: name), and the parent of each joint:

```
 0 pelvis      1 l_hip       2 r_hip       3 spine1      4 l_knee      5 r_knee
 6 spine2      7 l_ankle     8 r_ankle     9 spine3     10 l_foot     11 r_foot
12 neck       13 l_collar   14 r_collar   15 head       16 l_shoulder 17 r_shoulder
18 l_elbow    19 r_elbow    20 l_wrist    21 r_wrist    22 l_hand     23 r_hand
parents = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21]
```

`l_`/`r_` are the judge's own left and right. Hips, knees, ankles, shoulders, elbows and
wrists are measured directly. The others are derived from them: the spine joints are
interpolated between pelvis and the mid-shoulders, `head` is between the ears, `l_foot`/
`r_foot` is the ball of the foot, and `l_hand`/`r_hand` is the knuckles.

## How the 3D was made, and its limits

- **Single camera.** Depth (distance from the camera) is the least accurate direction, and
  limbs hidden behind the body are estimated rather than seen; check `conf`.
- **Body pose** comes from MediaPipe's pose landmarker (heavy model), run on the judge
  only. Bone lengths are not forced to be constant.
- **Position in the room is inferred, not measured.** In each frame the lowest visible
  point of the body is placed on the floor, and the body moves horizontally so that
  whatever is planted on the floor (feet, hands, knees) doesn't slide. Standing tasks
  stay put and walking advances step by step. On synthetic walking with 1 cm of noise,
  the distance walked came out within about 5%. Height during a jump is not recoverable;
  the lowest point is always on the floor.
- **Floor orientation** is estimated from hands and toes planted on the floor when there
  are any (push-ups, planks). Otherwise the camera is assumed level.

## take.json

| field | meaning |
|---|---|
| `task` | `id`, `category`, `tier` (core/extended/edge), `name`, `instruction` the judge was given, `props` used, and `note`: known limitations for this task (e.g. jumps, objects hiding limbs) |
| `stop_reason` | `finished` (judge moved, then held still; the hold is trimmed to 0.5 s), `time` (fixed length), `max duration`, or `stopped by operator` |
| `quality.status` | `ok`, or `check` with reasons in `quality.flags` |
| `quality.frames_with_3d_pct`, `visibility`, `bone_length_cv` | coverage, mean `conf` per body part, variation of each bone's length (high = tracking jumped) |

## session.json

`judge` (an ID code), `camera` (resolution and fps), `setup` (camera placement used),
`body` (the judge's segment lengths in metres and approximate height, from the neutral
stance `calibrate` task, for scaling a sim humanoid or normalizing comparisons), and
`tasks`: for each task its folder (`dir`), `status` (`accepted`/`skipped`/`pending`),
`accepted_take`, and every take with its quality status.
