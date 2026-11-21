"""Stage 1 harness (no RL): Gemini maps a casual human video onto the G1 in MuJoCo, in two phases.

Phase 1 - ENVIRONMENT: Gemini recreates the scene (surfaces, object, camera, the person's start pose).
          A still from the estimated camera is compared with the video's first frame; Gemini corrects it.
          The accepted environment is then LOCKED.
Phase 2 - MOTION: inside the locked scene Gemini writes the motion (keyframes, grasp/release, where the object ends).
          Each round: kinematic replay -> collision / grasp / placement checks -> side-by-side video ->
          Gemini judges and revises, until the robot does what the human does.

./mm mirror videos/IMG_6520.mp4 --name bottle --fact "uses his RIGHT hand"
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from google import genai
from google.genai import types

from . import config as C
from .replay import Replay

# ---------------------------------------------------------------- shared conventions
CONVENTIONS = """COORDINATES: the person's pelvis starts at the origin facing +forward; "left" is the PERSON's left. Meters, degrees.
LEFT/RIGHT RULE: always use the person's own anatomical side. If the person faces the camera, their LEFT hand and
everything on their left appear on the RIGHT side of the image (and vice versa). To tell which hand is used, follow
the ARM back to its shoulder - do not go by which side of the image the object is on (people reach across their body).
- pose fields: "pelvis": {"forward_m", "left_m", "facing_left_deg"};
  "trunk": {"lean_forward_deg" (from vertical, 90 = bent over), "side_bend_left_deg", "twist_left_deg"};
  "left_leg"/"right_leg": {"thigh_forward_deg" (thigh from vertical, knee in front > 0; sitting ~ 90, standing 0),
                           "knee_flexion_deg" (0 = straight; sitting ~ 90), "hip_abduction_deg"};
  "left_arm"/"right_arm" (relative to the trunk): {"shoulder_flexion_deg" (0 hanging, 90 forward, 180 up),
     "shoulder_abduction_deg" (0 at side, 90 sideways), "elbow_flexion_deg" (0 straight),
     "shoulder_external_rotation_deg" (+90 = forearm rotated up/out)}.
  Feet are kept flat on the floor automatically (a seated pose lowers the pelvis by itself).
- camera: where the video was filmed from, relative to the person's START pose. azimuth_deg 0 = camera directly in
  front of the person looking at them, 90 = on the person's left, 180 = behind. height_m = lens height above floor.
  look_at_* = the point the camera centers on. vertical_fov_deg: phone ~ 45-60 (portrait videos ~ 60-70).
- The robot is a """ + C.ROBOTS[C.ROBOT]["desc"] + """. At a grasp, the arm is solved so the palm lands on the
  grasp point facing palm_dir (fingers curl around it if the robot has fingers)."""

ENV_SCHEMA = """Return JSON:
{
 "summary": str,                          # what happens in the video, step by step
 "environment": {"description": str,
     "surfaces": [{"name": str, "kind": "table|solid", "forward_m": f, "left_m": f,
                   "depth_m": f, "width_m": f, "height_m": f}],   # table = top + legs; solid = counter/stool/block
     "furniture": [{"kind": "chair|stool|table|backpack|laptop|...", "forward_m": f, "left_m": f, "yaw_deg": f}]},
         # other furniture / items near the person that you can see (where they are), so the scene can be arranged
 "object": null | {"name": str, "description": str, "mass_kg": f,
     "parts": [{"shape": "box|cylinder|sphere|capsule", "size": [...], "pos": [x, y, z], "euler_deg": [r, p, y] (optional),
                "color_rgb": [r, g, b] (0-1)}],
         # build the object's real shape from a few primitives, in the OBJECT frame: origin = center of its base on the
         # surface it rests on, +x = its front, +z up. box size = [x, y, z] extents; cylinder = [diameter, height] (axis z,
         # tilt with euler_deg); sphere = [diameter]; capsule = [diameter, length]. pos = part center. 1-12 parts.
     "start": {"forward_m": f, "left_m": f, "on_surface_height_m": f, "yaw_deg": f}},
 "camera": {"horizontal_distance_m": f, "height_m": f, "azimuth_deg": f, "vertical_fov_deg": f,
            "look_at_forward_m": f, "look_at_left_m": f, "look_at_height_m": f},
 "start_pose": {<pose fields>},           # the person's posture in the FIRST frame (e.g. seated, standing)
 "duration_s": f
}"""

MOTION_SCHEMA = """Return JSON:
{
 "contacts": [{"t_grasp": f, "t_release": f|null, "hand": "left|right|both"}],
 "grasp_points": {"left": {"pos": [x, y, z], "palm_dir": [x, y, z], "finger_dir": [x, y, z]}, "right": {...}},
     # where each hand that grasps touches the object, in the OBJECT frame (see object.parts): pos = point on the
     # object's surface under the palm center; palm_dir = direction the palm faces (into the object);
     # finger_dir = direction the fingers point (optional). Only for the hand(s) that grasp.
 "camera_path": [{"t": f, <any camera fields that change>}],
     # if the camera MOVES (handheld, pans to follow the person, walks), its pose over time in the same camera fields as
     # environment.camera; [] for a static camera.
 "footsteps": [{"foot": "left|right", "t_lift": f, "t_land": f, "land": {"forward_m": f, "left_m": f} | null}],
     # EVERY step the person takes (walking, turning, side steps, small shuffles), in video time, with where the foot
     # lands if you can tell (else null = under the hip). Feet stay planted between steps. [] if they never step.
 "object_end": null | {"forward_m": f, "left_m": f, "on_surface_height_m": f, "yaw_deg": f},  # where it is put down
 "keyframes": [{"t": f, <pose fields>}]   # first keyframe at t=0 = start_pose; every ~0.3-0.7 s, more in fast motion
}"""

ENV_PROMPT = f"""Watch this casual, uncalibrated video. Recreate its ENVIRONMENT for a simulator: the surfaces
(tables, counters...), the object the person manipulates (type, size, color, where it starts), the camera viewpoint,
and the person's posture in the first frame. Estimate metric sizes from human proportions (an adult is ~1.7 m tall,
shoulders ~0.4 m wide) and the typical real-world size of the objects you see.
{CONVENTIONS}
{ENV_SCHEMA}"""

ENV_JUDGE = f"""This image is side by side. LEFT: the first frame of the original video. RIGHT: the simulator
recreation of the scene with the robot (""" + C.ROBOTS[C.ROBOT]["desc"] + """) in the person's start pose, rendered from the
estimated camera. Ignore the background, textures, lighting and other people; judge the camera viewpoint and framing
of the subject, the surfaces the person actually uses (height, position relative to them), the object, and the
person's position and posture.
Compare: camera viewpoint and framing (angle, height, distance, where things appear in the image), every relevant
surface (position, size, height), the object (type, size, color, position), the person's position and posture.
COLLISIONS lists interpenetrations in the recreation (e.g. legs inside the table): physically impossible, fix them.
Score 0-10 and return a corrected environment.
{CONVENTIONS}
Return JSON: {{"score": f, "differences": [str], "revised": <full environment JSON>}}
{ENV_SCHEMA}"""

MOTION_PROMPT = """Watch the video and describe the person's MOTION precisely, inside this LOCKED environment
(do not change it): what they do, when, with which hand, posture over time, and where the object ends up.
LOCKED ENVIRONMENT:
{env}
{conv}
{schema}"""

MOTION_JUDGE = """This is ONE side-by-side video. LEFT half: the original human video. RIGHT half: the robot replaying
the current motion in the simulator, in the locked environment, from the same camera. Time-synced; timestamp top-left.
The robot (""" + C.ROBOTS[C.ROBOT]["desc"] + """) is not the person: ignore its looks, the background and visual style.
Compare frame by frame: posture over time (trunk, legs, arms), steps (which foot, when, where), which hand does what,
timing of each phase, the object
(when grasped, how it moves, where/when it is put down).
TELEMETRY.collisions = interpenetrations detected by the simulator (e.g. "right hand inside table, 4 cm"): physically
impossible, they MUST be fixed (raise the arm/trunk, retime). TELEMETRY.grasps / placements say whether the hand could
reach the object / put it down at object_end; if not, move the body closer/lower at that moment.
TELEMETRY.pose_match is an OBJECTIVE measurement: the person's 2D pose detected in the video vs the robot's joints seen
through the same camera, per limb, with the worst time windows, plus framing (robot's on-screen size relative to the
person, center offset, % of frames off-screen) and hard failures (body flipped, object above the head). Trust it over
your visual impression; fix the worst windows first. If the robot is off-screen or framed very differently, fix the
camera / camera_path.
IMPORTANT: the robot's VISIBLE limbs are driven directly by video body tracking, frame by frame (TELEMETRY.retarget
says how many frames). Your keyframe arm/leg/trunk angles only matter where the camera cannot see the limb (occluded,
out of frame). So put your patches where they matter: the object (where it starts / ends, grasp_points), contacts
(which hand, grasp / release times), footsteps, pelvis position / facing, camera, and occluded limbs.
TELEMETRY.auto_fixed lists problems already corrected automatically (e.g. a hand lifted out of the table): don't redo them.
Cite timestamps; be quantitative. Score 0-10 and return a PATCH to the motion (the environment stays locked):
only the keyframes you add or change (each with its "t" and ONLY the fields that change), keyframe times to remove,
and contacts / object_end / grasp_points / footsteps / camera_path only if they change.
LOCKED ENVIRONMENT:
{env}
{conv}
Return JSON: {{"score": f, "differences": [str],
  "patch": {{"keyframes": [{{"t": f, <changed pose fields only>}}], "remove_keyframe_t": [f], "contacts": <optional>, "object_end": <optional>, "grasp_points": <optional>, "footsteps": <optional>, "camera_path": <optional>}}}}
Motion schema (for reference):
{schema}"""


# ---------------------------------------------------------------- helpers
class DualJudge:
    """Two different Gemini models judge the same round in parallel. Score = mean (less noise); the patch of the
    stricter judge (lower score, i.e. it found more problems) is applied; both critiques are kept."""

    def __init__(self, a, b):
        self.a, self.b = a, b
        self.model = f"{a.model}+{b.model}"

    def ask(self, parts, **kw):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(2) as ex:
            ja, jb = ex.map(lambda g: g.ask(parts, **kw), (self.a, self.b))
        sa, sb = float(ja.get("score") or 0), float(jb.get("score") or 0)
        strict, other = (ja, jb) if sa <= sb else (jb, ja)
        out = dict(strict)
        out["score"] = round((sa + sb) / 2, 2)
        out["scores"] = {self.a.model: sa, self.b.model: sb}
        out["differences"] = list(strict.get("differences") or []) + \
            [f"[2nd judge] {x}" for x in (other.get("differences") or [])[:3]]
        return out


class Gemini:
    def __init__(self, model):
        self.client, self.model = genai.Client(), model

    def ask(self, parts, retries=2, temperature=0.3, thinking="LOW", low_res=False):
        cfg = dict(response_mime_type="application/json", temperature=temperature)
        if thinking:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_level=thinking)
        if low_res:
            cfg["media_resolution"] = "MEDIA_RESOLUTION_LOW"
        for k in range(retries + 1):
            try:
                r = self.client.models.generate_content(
                    model=self.model, contents=parts, config=types.GenerateContentConfig(**cfg))
                return json.loads(r.text)
            except Exception as e:  # noqa: BLE001
                if k == retries:
                    raise
                print(f"  retry: {str(e)[:120]}", flush=True)
                time.sleep(2)


def vpart(path, fps=4.0):
    """Video part with an explicit sampling rate (the API default is 1 frame/s, too coarse for motion)."""
    return types.Part(inline_data=types.Blob(data=Path(path).read_bytes(), mime_type="video/mp4"),
                      video_metadata=types.VideoMetadata(fps=fps))


def ipart(img_bgr):
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return types.Part.from_bytes(data=buf.tobytes(), mime_type="image/jpeg")


def video_info(path):
    c = cv2.VideoCapture(str(path))
    w, h, fps = int(c.get(3)), int(c.get(4)), c.get(5) or 30.0
    return w - w % 2, h - h % 2, fps


def first_frame(path):
    ok, im = cv2.VideoCapture(str(path)).read()
    return im


def label(img, text, pos):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def subject_crop(human, pose_dir, margin=0.45):
    """Same zoom for both halves: the region the person occupies over the whole clip (+ head + margin), with the
    frame's aspect ratio. None if the person already fills most of the frame."""
    try:
        from .posematch import human_pose
        h2 = human_pose(human, pose_dir)
    except Exception:  # noqa: BLE001
        return None
    W, H, _ = video_info(human)
    pts = h2[:, :, :2][h2[:, :, 2] > 0.5]
    if len(pts) < 20:
        return None
    (x0, y0), (x1, y1) = np.percentile(pts, 2, axis=0), np.percentile(pts, 98, axis=0)
    y0 -= 0.3 * (y1 - y0)                                   # landmarks start at the shoulders: room for the head
    cx, cy = (x0 + x1) / 2 * W, (y0 + y1) / 2 * H
    w, h = (x1 - x0) * W * (1 + margin), (y1 - y0) * H * (1 + margin)
    ar = W / H
    w, h = (h * ar, h) if w / h < ar else (w, w / ar)
    if w >= 0.85 * W or h >= 0.85 * H:
        return None
    x = int(np.clip(cx - w / 2, 0, W - w))
    y = int(np.clip(cy - h / 2, 0, H - h))
    return x, y, int(w), int(h)


def robot_crop(robot, like, margin=0.2):
    """Box around the robot in its render (light grey on the dark scene), same aspect as `like`=(w, h)."""
    c = cv2.VideoCapture(str(robot))
    n = int(c.get(7))
    W, H = int(c.get(3)), int(c.get(4))
    pts = []
    for i in np.linspace(0, max(n - 1, 0), 12).astype(int):
        c.set(1, i)
        ok, im = c.read()
        if not ok:
            continue
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        m = ((hsv[:, :, 2] > 120) & (hsv[:, :, 1] < 50)).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))   # drop the floor's thin grid lines
        ys, xs = np.nonzero(m)
        if len(xs) > 200:
            pts.append([np.percentile(xs, 2), np.percentile(ys, 2), np.percentile(xs, 98), np.percentile(ys, 98)])
    if not pts:
        return None
    P = np.array(pts)
    x0, y0, x1, y1 = P[:, 0].min(), P[:, 1].min(), P[:, 2].max(), P[:, 3].max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = (x1 - x0) * (1 + margin), (y1 - y0) * (1 + margin)
    ar = like[0] / like[1]
    w, h = (h * ar, h) if w / h < ar else (w, w / ar)
    if w > W:
        w, h = W, W / ar
    if h > H:
        w, h = H * ar, H
    if w * h > 0.8 * W * H:
        return None
    return int(np.clip(cx - w / 2, 0, W - w)), int(np.clip(cy - h / 2, 0, H - h)), int(w), int(h)


def side_by_side(human, robot, out, text, crop=None, robot_crop_box=None):
    """Human | robot, time-synced, with timestamp and labels. crop=(x, y, w, h) zooms both halves identically."""
    import imageio
    a, b = cv2.VideoCapture(str(human)), cv2.VideoCapture(str(robot))
    fps = video_info(human)[2]
    frames = []
    while True:
        ok1, f1 = a.read()
        ok2, f2 = b.read()
        if not (ok1 and ok2):
            break
        size = (f1.shape[1], f1.shape[0])
        if robot_crop_box:                              # each half framed on its own subject (display)
            x, y, w, h = robot_crop_box
            f2 = cv2.resize(f2[y:y + h, x:x + w], size, interpolation=cv2.INTER_CUBIC)
        else:
            f2 = cv2.resize(f2, size)
        if crop:
            x, y, w, h = crop
            f1 = cv2.resize(f1[y:y + h, x:x + w], size, interpolation=cv2.INTER_CUBIC)
            if not robot_crop_box:
                f2 = cv2.resize(f2[y:y + h, x:x + w], size, interpolation=cv2.INTER_CUBIC)
        stack = f1.shape[1] > f1.shape[0]          # landscape clips: person on top, robot below (fits a page/phone)
        f = np.vstack([f1, f2]) if stack else np.hstack([f1, f2])
        label(f, f"t={len(frames) / fps:.1f}s  {text}", (12, 30))
        label(f, "HUMAN", (12, f1.shape[0] - 14))
        if stack:
            label(f, "ROBOT", (12, f.shape[0] - 14))
        else:
            label(f, "ROBOT", (f1.shape[1] + 12, f.shape[0] - 14))
        frames.append(f[:, :, ::-1])
    if frames:
        imageio.mimsave(out, frames, fps=fps, macro_block_size=8)


def _deep(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = _deep(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def apply_patch(motion, patch, tol=0.06):
    """Apply the judge's patch: merge changed keyframes by time, add new ones, drop removed ones."""
    if not patch:
        return motion
    m = json.loads(json.dumps(motion))
    kfs = m.get("keyframes", [])
    rm = patch.get("remove_keyframe_t") or []
    kfs = [k for k in kfs if not any(abs(k["t"] - r) < tol for r in rm)]
    for pk in patch.get("keyframes") or []:
        hit = next((k for k in kfs if abs(k["t"] - pk.get("t", -9)) < tol), None)
        if hit is not None:
            hit.update(_deep(hit, pk))
        else:
            # new keyframe: start from the nearest earlier one so unspecified fields stay sensible
            prev = max((k for k in kfs if k["t"] <= pk.get("t", 0)), key=lambda k: k["t"], default=kfs[0] if kfs else {})
            kfs.append(_deep(prev, pk))
    m["keyframes"] = sorted(kfs, key=lambda k: k["t"])
    for key in ("contacts", "object_end", "grasp_points", "footsteps", "camera_path"):
        if patch.get(key) is not None:
            m[key] = patch[key]
    return m


_TORSO_LEN = [None]


def _surfaces_under_hands(it, ofh, margin=0.06):
    """The hands (from the video) say where the object starts and ends. Move the surface that supports it - sideways,
    as little as possible - so both spots are on it. Gemini's surface placement is an estimate; the hands are data."""
    obj = it.get("object")
    if not ofh or not obj or ofh.get("start") is None:
        return it
    surfs = [dict(x) for x in (it.get("environment") or {}).get("surfaces") or []]
    if not surfs:
        return it
    pts = [np.asarray(ofh["start"], float)] + ([np.asarray(ofh["end"], float)] if ofh.get("end") is not None else [])
    st_h = (obj.get("start") or {}).get("on_surface_height_m", 0.0)
    cands = [s for s in surfs if abs(s.get("height_m", 0.75) - st_h) < 0.15]
    if not cands or st_h < 0.05:                          # floor objects: nothing to move
        return it
    s = min(cands, key=lambda x: np.hypot(x.get("forward_m", 0) - pts[0][0], x.get("left_m", 0) - pts[0][1]))
    c = np.array([s.get("forward_m", 0.0), s.get("left_m", 0.0)])
    half = np.array([s.get("depth_m", 0.6), s.get("width_m", 0.8)]) / 2 - margin
    lo = np.max([p[:2] - half for p in pts], 0)           # feasible centers contain every point
    hi = np.min([p[:2] + half for p in pts], 0)
    newc = np.clip(c, lo, hi) if np.all(lo <= hi) else np.mean([p[:2] for p in pts], 0)
    s.update(forward_m=float(newc[0]), left_m=float(newc[1]))
    if ofh.get("implied_z") is not None:   # the surface is where the robot's hand actually picks the object up
        h0 = s.get("height_m", st_h)
        s["height_m"] = float(np.clip(ofh["implied_z"], h0 - 0.15, h0 + 0.35))   # lowering it risks the legs
    new_h = s.get("height_m", st_h)
    it = dict(it)
    env = dict(it.get("environment") or {})
    env["surfaces"] = surfs
    it["environment"] = env
    o = dict(obj)
    o["start"] = dict(o.get("start") or {}, forward_m=float(pts[0][0]), left_m=float(pts[0][1]), on_surface_height_m=float(new_h))
    if len(pts) > 1:
        o["end"] = dict(o.get("end") or o["start"], forward_m=float(pts[1][0]), left_m=float(pts[1][1]),
                        on_surface_height_m=float(new_h))
    it["object"] = o
    it["_surface_moved_m"] = round(float(np.linalg.norm(newc - c)), 3)
    return it


def replay_matched(interp, video, out, W, H, fps, video_path):
    """Replay with posture retargeted from the video and a camera that reproduces the video's perspective."""
    from .posematch import human_pose
    from .retarget import camera_from_perspective
    human = None
    if C.USE_VIDEO_POSE:
        try:
            w2, w3 = human_pose(video, out, full=True)
            human = (w2, w3)
        except Exception as e:  # noqa: BLE001
            print(f"  (pose unavailable: {str(e)[:80]})", flush=True)
    it = dict(interp)
    cam_info = None
    if human is not None:
        rp0 = Replay(it, W, H, human=human)
        rp0.run(fps, None)
        if _TORSO_LEN[0] is None:
            import mujoco as _mj
            d = _mj.MjData(rp0.m)
            d.qpos[3] = 1.0
            d.qpos[rp0.jq] = C.DEFAULT_Q
            _mj.mj_kinematics(rp0.m, d)
            b = lambda n: _mj.mj_name2id(rp0.m, _mj.mjtObj.mjOBJ_BODY, n)
            hip = (d.xpos[b("left_hip_roll_link")] + d.xpos[b("right_hip_roll_link")]) / 2
            sho = (d.xpos[b("left_shoulder_roll_link")] + d.xpos[b("right_shoulder_roll_link")]) / 2
            _TORSO_LEN[0] = float(np.linalg.norm(sho - hip))
        fovy = float((it.get("camera") or {}).get("vertical_fov_deg") or (62.0 if H > W else 50.0))
        cam_info = camera_from_perspective(rp0.rel, w2, rp0.pelvis_pose, _TORSO_LEN[0], fovy, W, H, fps) if C.PERSPECTIVE_CAMERA else None
        if cam_info:
            it["_camera_fit"] = dict(fovy=fovy, static=[x.tolist() for x in cam_info["static"]],
                                     path=[(tt, p.tolist(), l.tolist()) for tt, p, l in cam_info["path"]])
            from .retarget import root_from_video
            T = len(rp0.pelvis_pose)
            start = rp0.pelvis_pose[0][1][:2]
            ro = root_from_video(rp0.rel, w2, cam_info["static"], _TORSO_LEN[0], fovy, W, H, fps, start, T) if C.VIDEO_ROOT else None
            if ro is not None:
                it["_root_override"] = ro.tolist()
        rp1 = Replay(it, W, H, human=human)                 # with the video's heading + movement applied
        rp1.run(fps, None)
        if C.MOVE_SURFACES:
            it = _surfaces_under_hands(it, getattr(rp1, "obj_from_hands", None))
    rp = Replay(it, W, H, human=human)
    tel = rp.run(fps, video_path)
    if cam_info:
        tel["camera_from_perspective"] = dict(moving=bool(cam_info["path"]), spread_m=cam_info["spread_m"])
    return rp, tel


def _pose_match(video, out, rp, fps, W, H):
    from .posematch import compare, human_pose
    try:
        human = human_pose(video, out)
        return compare(human, np.array(rp.robot_kp), fps, W / H, extra=dict(rp.flags))
    except Exception as e:  # noqa: BLE001
        return dict(error=str(e)[:200])


def quality(score, tel):
    """Judge score, penalised by what the simulator says is physically wrong."""
    coll = tel.get("collisions", [])
    sustained = sum(1 for c in coll if c["t_end"] - c["t_start"] > 0.1 and c["max_depth_cm"] > 2)
    grasp = max((x["grip_pos_err_m"] for g in tel.get("grasps", []) for x in g["detail"]), default=0.0)
    place = max((p["placement_err_m"] for p in tel.get("placements", [])), default=0.0)
    q = score - 0.5 * sustained - 0.1 * (len(coll) - sustained) - 2.0 * (grasp > 0.03) - 1.0 * (place > 0.05)
    pm = tel.get("pose_match") or {}
    if pm.get("limb_angle_err_deg") is not None:
        q -= max(0.0, pm["limb_angle_err_deg"] - 20.0) / 10.0         # each 10 deg of limb error beyond 20 costs 1 point
    q -= 4.0 * pm.get("robot_offscreen_frac", 0.0) + 2.0 * (pm.get("flipped_frames", 0) > 0) + 1.0 * (pm.get("object_above_head_frames", 0) > 3)
    return q


def merge(env, motion):
    it = {k: v for k, v in env.items() if k != "start_pose"}
    it["object"] = dict(env["object"]) if env.get("object") else None
    if it["object"] is not None and motion.get("object_end"):
        it["object"]["end"] = motion["object_end"]
    if it["object"] is not None and motion.get("grasp_points"):
        it["object"]["grasp_points"] = motion["grasp_points"]
    it["contacts"] = motion.get("contacts", [])
    it["footsteps"] = motion.get("footsteps", [])
    it["camera_path"] = motion.get("camera_path", [])
    it["keyframes"] = motion.get("keyframes") or [dict(t=0.0, **env.get("start_pose", {}))]
    return it


# ---------------------------------------------------------------- phases
STATS = {}


def phase_env(g, video, out, facts, iters, threshold, W, H, gj=None):
    gj = gj or g
    print("== PHASE 1: environment", flush=True)
    t0 = time.time()
    env = g.ask([vpart(video, fps=2.0), ENV_PROMPT + facts])
    print(f"[env] {time.time()-t0:.0f}s  {env.get('summary', '')[:150]}", flush=True)
    human0 = first_frame(video)
    best = None
    for k in range(iters):
        d = out / "env" / f"iter{k}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "environment.json").write_text(json.dumps(env, indent=1))
        img, coll = Replay(merge(env, {}), W, H).still()
        img = cv2.resize(img[:, :, ::-1], (human0.shape[1], human0.shape[0]))
        sbs = np.hstack([human0, img])
        label(sbs, "HUMAN (first frame)", (12, sbs.shape[0] - 14))
        label(sbs, f"SIM env iter {k}", (human0.shape[1] + 12, sbs.shape[0] - 14))
        cv2.imwrite(str(d / "still.jpg"), sbs)
        t1 = time.time()
        j = gj.ask([ipart(sbs), f"CURRENT ENVIRONMENT:\n{json.dumps(env)}\n\nCOLLISIONS:\n{json.dumps(coll)}\n{facts}\n{ENV_JUDGE}"])
        (d / "judge.json").write_text(json.dumps(j, indent=1))
        sc = float(j.get("score") or 0)
        print(f"[env {k}] judge {time.time()-t1:.0f}s  score {sc}/10  collisions {len(coll)}", flush=True)
        for x in (j.get("differences") or [])[:3]:
            print(f"   - {x[:150]}", flush=True)
        key = sc - 0.5 * len(coll)
        if best is None or key > best[0]:
            best = (key, k, env, sc)
        if sc >= threshold and not coll:
            break
        if k - best[1] >= 2:
            break
        env = j.get("revised") or env
    env = best[2]
    (out / "environment_locked.json").write_text(json.dumps(env, indent=1))
    print(f"[env] locked iter {best[1]} ({best[3]}/10)", flush=True)
    STATS["env"] = dict(rounds=k + 1, best_iter=best[1], best_score=best[3], seconds=round(time.time() - t0, 1))
    return env


def phase_motion(g, video, out, env, facts, iters, threshold, W, H, fps, gj=None):
    gj = gj or g
    print("== PHASE 2: motion (environment locked)", flush=True)
    envs = json.dumps(env)
    t0 = time.time()
    from .posematch import hand_activity, hand_activity_text, human_pose
    try:
        _, w3 = human_pose(video, out, full=True)
        ha_txt = hand_activity_text(hand_activity(w3, fps))
    except Exception:  # noqa: BLE001
        w3, ha_txt = None, ""
    motion = g.ask([vpart(video, fps=5.0), MOTION_PROMPT.format(env=envs, conv=CONVENTIONS, schema=MOTION_SCHEMA) + facts + "\n" + ha_txt])
    print(f"[motion] {time.time()-t0:.0f}s  {len(motion.get('keyframes', []))} keyframes, contacts {motion.get('contacts')}", flush=True)
    judge = MOTION_JUDGE.format(env=envs, conv=CONVENTIONS, schema=MOTION_SCHEMA)
    best = None
    for k in range(iters):
        d = out / "motion" / f"iter{k}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "motion.json").write_text(json.dumps(motion, indent=1))
        t1 = time.time()
        rp, tel = replay_matched(merge(env, motion), video, out, W, H, fps, d / "robot.mp4")
        tel["pose_match"] = _pose_match(video, out, rp, fps, W, H)
        from .posematch import feedback as _pm_text
        tel_short = {k_: v for k_, v in tel.items() if k_ not in ("samples", "pose_match")}
        tel_short["pose_match"] = _pm_text(tel.get("pose_match"))
        try:
            tel_short["video_tracking"] = hand_activity_text(hand_activity(w3, fps, motion.get("contacts")))
        except Exception:  # noqa: BLE001
            pass
        tel_short["samples"] = tel.get("samples", [])[::2]
        (d / "telemetry.json").write_text(json.dumps(tel, indent=1, default=float))
        side_by_side(video, d / "robot.mp4", d / "side_by_side.mp4", f"motion iter {k}", crop=subject_crop(video, out))
        t_rep = time.time() - t1
        t1 = time.time()
        j = gj.ask([vpart(d / "side_by_side.mp4", fps=3.0),
                   f"CURRENT MOTION:\n{json.dumps(motion)}\n\nTELEMETRY:\n{json.dumps(tel_short, default=float)}\n{facts}\n{judge}"],
                  low_res=True)
        (d / "judge.json").write_text(json.dumps(j, indent=1))
        sc = float(j.get("score") or 0)
        nc = len(tel.get("collisions", []))
        print(f"[motion {k}] replay {t_rep:.1f}s  judge {time.time()-t1:.0f}s  score {sc}/10  collisions {nc}  "
              f"quality {quality(sc, tel):.1f}", flush=True)
        for x in (j.get("differences") or [])[:4]:
            print(f"   - {x[:150]}", flush=True)
        key = quality(sc, tel)
        STATS.setdefault("motion_rounds", []).append(dict(iter=k, score=sc, collisions=nc, replay_s=round(t_rep, 1),
                                                          judge_s=round(time.time() - t1, 1)))
        if best is None or key > best[0]:
            best = (key, k, motion, j, sc)
        if sc >= threshold and key >= sc - 0.3:      # good score and physically clean
            break
        if k - best[1] >= 2:
            print("  no improvement for 2 rounds; stopping", flush=True)
            break
        if key < best[0]:
            print(f"  regression; reverting to motion iter {best[1]}", flush=True)
            r = gj.ask([vpart(out / "motion" / f"iter{best[1]}" / "side_by_side.mp4", fps=3.0),
                       f"CURRENT MOTION:\n{json.dumps(best[2])}\n\nA previous attempt to fix it made things WORSE. Its problems: "
                       f"{json.dumps(j.get('differences', []))}\nAvoid those changes. Fix these instead: "
                       f"{json.dumps(best[3].get('differences', []))}\n{facts}\n{judge}"], low_res=True)
            motion = apply_patch(best[2], r.get("patch"))
        else:
            motion = apply_patch(motion, j.get("patch"))
    (out / "motion_best.json").write_text(json.dumps(best[2], indent=1))
    (out / "final.json").write_text(json.dumps(merge(env, best[2]), indent=1))
    sbs = out / "motion" / f"iter{best[1]}" / "side_by_side.mp4"
    print(f"[motion] best iter {best[1]} ({best[4]}/10)  ->  {sbs}", flush=True)
    STATS["motion"] = dict(best_iter=best[1], best_score=best[4], seconds=round(time.time() - t0, 1),
                           telemetry=json.loads((out / "motion" / f"iter{best[1]}" / "telemetry.json").read_text()))
    return best, sbs


def prepare_video(video, out):
    """Accept any upload: convert MOV / large / odd-codec videos to a ~854 px, 30 fps H.264 mp4."""
    video = Path(video)
    W, H, _ = video_info(video)
    if video.suffix.lower() == ".mp4" and max(W, H) <= 1000 and W > 0:
        return video
    import subprocess
    import imageio_ffmpeg
    dst = out / "input.mp4"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(video), "-vf",
                    "scale='if(gt(iw,ih),854,-2)':'if(gt(iw,ih),-2,854)'", "-r", "30", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-an", str(dst)], check=True)
    print(f"[video] converted {video.name} -> {dst.name}", flush=True)
    return dst


def parallel_attempts(video, name, out, env, facts, n, model, judge_model, iters, threshold, judge_model2=None):
    """Run the motion phase n times at once (Gemini is the bottleneck, so this costs no extra wall time) and keep the
    attempt with the best physical-quality score. Results vary a lot between runs, so this is a cheap win."""
    import subprocess
    locked = out / "environment_locked.json"
    locked.write_text(json.dumps(env, indent=1))
    procs = []
    for i in range(n):
        cmd = [str(C.ROOT / "mm"), "mirror", str(video), "--name", f"{name}/attempt{i}", "--env-from", str(locked),
               "--attempts", "1", "--iters", str(iters), "--threshold", str(threshold), "--model", model, "--no-bench"]
        if judge_model:
            cmd += ["--judge-model", judge_model]
        if judge_model2:
            cmd += ["--judge-model2", judge_model2]
        for f in facts or []:
            cmd += ["--fact", f]
        procs.append(subprocess.Popen(cmd, cwd=C.ROOT, stdout=open(out / f"attempt{i}.log", "w"), stderr=subprocess.STDOUT))
    for p in procs:
        p.wait()
    cands = []
    for i in range(n):
        sf = out / f"attempt{i}" / "summary.json"
        if sf.exists():
            S = json.loads(sf.read_text())
            q = quality(S["motion"]["best_score"], S["best_telemetry"])
            cands.append((q, i, S))
            print(f"[attempt {i}] score {S['motion']['best_score']}  quality {q:.1f}", flush=True)
    q, i, S = max(cands)
    a = out / f"attempt{i}"
    shutil.copy(a / "final.json", out / "final.json")
    STATS.update(attempts=[dict(attempt=j, quality=round(qq, 2), score=SS["motion"]["best_score"]) for qq, j, SS in cands],
                 motion=S["motion"], motion_rounds=S["motion_rounds"])
    STATS["motion"]["telemetry"] = dict(S["best_telemetry"])
    sbs = a / "motion" / f"iter{S['motion']['best_iter']}" / "side_by_side.mp4"
    print(f"[attempts] best = attempt {i} (score {S['motion']['best_score']}, quality {q:.1f})", flush=True)
    return (q, S["motion"]["best_iter"], None, None, S["motion"]["best_score"]), sbs


def run(video, name, model, env_iters, env_threshold, iters, threshold, facts, env_from=None, judge_model=None,
        attempts=1, bench=True, judge_model2=None):
    out = C.ROOT / "runs" / "mirror2" / name
    if env_from:
        env_from = Path(env_from)
        keep = json.loads(env_from.read_text())
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    video = prepare_video(video, out)
    W, H, fps = video_info(video)
    g = Gemini(model)
    gj = Gemini(judge_model) if judge_model else g
    if judge_model2:
        gj = DualJudge(gj, Gemini(judge_model2))
    fx = ("\nUSER-VERIFIED FACTS (ground truth; never contradict or change these):\n- " + "\n- ".join(facts) + "\n") if facts else ""
    t0 = time.time()
    env = keep if env_from else phase_env(g, video, out, fx, env_iters, env_threshold, W, H, gj)
    (out / "environment_locked.json").write_text(json.dumps(env, indent=1))
    if attempts > 1:
        best, sbs = parallel_attempts(video, name, out, env, facts, attempts, model, judge_model, iters, threshold,
                                      judge_model2)
    else:
        best, sbs = phase_motion(g, video, out, env, fx, iters, threshold, W, H, fps, gj)
    shutil.copy(sbs, out / "best_side_by_side.mp4")
    if bench:
        bdir = C.ROOT / "benchmarks"
        bdir.mkdir(exist_ok=True)
        shutil.copy(sbs, bdir / f"{name}_score{best[4]}.mp4")
        shutil.copy(out / "final.json", bdir / f"{name}_score{best[4]}.json")
    print(f"[done] {time.time()-t0:.0f}s total", flush=True)
    STATS.update(total_s=round(time.time() - t0, 1), model=model, judge_model=gj.model, facts=facts)
    tel = STATS["motion"].pop("telemetry")
    STATS["best_telemetry"] = {k: v for k, v in tel.items() if k != "samples"}
    (out / "summary.json").write_text(json.dumps(STATS, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--name", default=None)
    ap.add_argument("--model", default="gemini-4-hydrogen")
    ap.add_argument("--env-iters", type=int, default=2)
    ap.add_argument("--env-threshold", type=float, default=8.0)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--judge-model", default=None, help="model for the judge rounds (default: --model), e.g. gemini-3.6-flash")
    ap.add_argument("--threshold", type=float, default=9.0)
    ap.add_argument("--fact", action="append", default=[], help="locked fact, e.g. --fact 'uses his RIGHT hand'")
    ap.add_argument("--env-from", default=None, help="skip phase 1 and use this locked environment JSON")
    ap.add_argument("--attempts", type=int, default=2, help="parallel motion attempts on the locked scene; best is kept")
    ap.add_argument("--no-bench", action="store_true", help="don't copy the result into benchmarks/")
    ap.add_argument("--judge-model2", default=None, help="second judge model (dual judge), e.g. gemini-4-carbon")
    a = ap.parse_args()
    run(a.video, a.name or Path(a.video).stem.replace(" ", "_"), a.model, a.env_iters, a.env_threshold,
        a.iters, a.threshold, a.fact, a.env_from, a.judge_model, a.attempts, not a.no_bench, a.judge_model2)


if __name__ == "__main__":
    main()
