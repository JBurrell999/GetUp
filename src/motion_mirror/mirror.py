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
- The robot is a Unitree G1 (1.3 m tall) with 3-finger hands. Hand placement, wrist orientation and finger closure
  at a grasp are solved automatically from the object geometry and grasp_style."""

ENV_SCHEMA = """Return JSON:
{
 "summary": str,                          # what happens in the video, step by step
 "environment": {"description": str,
     "surfaces": [{"name": str, "kind": "table|solid", "forward_m": f, "left_m": f,
                   "depth_m": f, "width_m": f, "height_m": f}]},   # table = top + legs; solid = counter/stool/block
 "object": null | {"type": "bottle|chair|box|other", "description": str, "color_rgb": [r, g, b] (0-1),
     "grasp_style": "side|top",           # side = hand wraps around (bottle); top = overhand grip (chair backrest)
     "size_m": [depth, width, height], "seat_height_m": f (chairs only), "grasp_height_frac": f,
     "start": {"forward_m": f, "left_m": f, "on_surface_height_m": f, "yaw_deg": f}},
 "camera": {"horizontal_distance_m": f, "height_m": f, "azimuth_deg": f, "vertical_fov_deg": f,
            "look_at_forward_m": f, "look_at_left_m": f, "look_at_height_m": f},
 "start_pose": {<pose fields>},           # the person's posture in the FIRST frame (e.g. seated, standing)
 "duration_s": f
}"""

MOTION_SCHEMA = """Return JSON:
{
 "contacts": [{"t_grasp": f, "t_release": f|null, "hand": "left|right|both"}],
 "object_end": null | {"forward_m": f, "left_m": f, "on_surface_height_m": f, "yaw_deg": f},  # where it is put down
 "keyframes": [{"t": f, <pose fields>}]   # first keyframe at t=0 = start_pose; every ~0.3-0.7 s, more in fast motion
}"""

ENV_PROMPT = f"""Watch this casual, uncalibrated video. Recreate its ENVIRONMENT for a simulator: the surfaces
(tables, counters...), the object the person manipulates (type, size, color, where it starts), the camera viewpoint,
and the person's posture in the first frame. Estimate metric sizes from human proportions (adult ~1.7 m,
shoulders ~0.4 m, table ~0.75 m high, chair seat ~0.45 m, water bottle ~0.22 m).
{CONVENTIONS}
{ENV_SCHEMA}"""

ENV_JUDGE = f"""This image is side by side. LEFT: the first frame of the original video. RIGHT: the simulator
recreation of the scene with the robot (1.3 m, shorter than the person) in the person's start pose, rendered from the
estimated camera. Ignore textures, lighting, background clutter, people other than the subject.
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
The robot is shorter (1.3 m): ignore that and visual style.
Compare frame by frame: posture over time (trunk, legs, arms), which hand does what, timing of each phase, the object
(when grasped, how it moves, where/when it is put down).
TELEMETRY.collisions = interpenetrations detected by the simulator (e.g. "right hand inside table, 4 cm"): physically
impossible, they MUST be fixed (raise the arm/trunk, retime). TELEMETRY.grasps / placements say whether the hand could
reach the object / put it down at object_end; if not, move the body closer/lower at that moment.
Cite timestamps; be quantitative. Score 0-10 and return a corrected MOTION (the environment stays locked).
LOCKED ENVIRONMENT:
{env}
{conv}
Return JSON: {{"score": f, "differences": [str], "revised": <full motion JSON>}}
{schema}"""


# ---------------------------------------------------------------- helpers
class Gemini:
    def __init__(self, model):
        self.client, self.model = genai.Client(), model

    def ask(self, parts, retries=2, temperature=0.3):
        for k in range(retries + 1):
            try:
                r = self.client.models.generate_content(
                    model=self.model, contents=parts,
                    config=types.GenerateContentConfig(response_mime_type="application/json", temperature=temperature))
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


def side_by_side(human, robot, out, text):
    """Human | robot, time-synced, with timestamp and labels."""
    import imageio
    a, b = cv2.VideoCapture(str(human)), cv2.VideoCapture(str(robot))
    fps = video_info(human)[2]
    frames = []
    while True:
        ok1, f1 = a.read()
        ok2, f2 = b.read()
        if not (ok1 and ok2):
            break
        f2 = cv2.resize(f2, (f1.shape[1], f1.shape[0]))
        f = np.hstack([f1, f2])
        label(f, f"t={len(frames) / fps:.1f}s  {text}", (12, 30))
        label(f, "HUMAN", (12, f.shape[0] - 14))
        label(f, "ROBOT", (f1.shape[1] + 12, f.shape[0] - 14))
        frames.append(f[:, :, ::-1])
    if frames:
        imageio.mimsave(out, frames, fps=fps, macro_block_size=8)


def merge(env, motion):
    it = {k: v for k, v in env.items() if k != "start_pose"}
    it["object"] = dict(env["object"]) if env.get("object") else None
    if it["object"] is not None and motion.get("object_end"):
        it["object"]["end"] = motion["object_end"]
    it["contacts"] = motion.get("contacts", [])
    it["keyframes"] = motion.get("keyframes") or [dict(t=0.0, **env.get("start_pose", {}))]
    return it


# ---------------------------------------------------------------- phases
def phase_env(g, video, out, facts, iters, threshold, W, H):
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
        j = g.ask([ipart(sbs), f"CURRENT ENVIRONMENT:\n{json.dumps(env)}\n\nCOLLISIONS:\n{json.dumps(coll)}\n{facts}\n{ENV_JUDGE}"])
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
        env = j.get("revised") or env
    env = best[2]
    (out / "environment_locked.json").write_text(json.dumps(env, indent=1))
    print(f"[env] locked iter {best[1]} ({best[3]}/10)", flush=True)
    return env


def phase_motion(g, video, out, env, facts, iters, threshold, W, H, fps):
    print("== PHASE 2: motion (environment locked)", flush=True)
    envs = json.dumps(env)
    t0 = time.time()
    motion = g.ask([vpart(video, fps=5.0), MOTION_PROMPT.format(env=envs, conv=CONVENTIONS, schema=MOTION_SCHEMA) + facts])
    print(f"[motion] {time.time()-t0:.0f}s  {len(motion.get('keyframes', []))} keyframes, contacts {motion.get('contacts')}", flush=True)
    judge = MOTION_JUDGE.format(env=envs, conv=CONVENTIONS, schema=MOTION_SCHEMA)
    best = None
    for k in range(iters):
        d = out / "motion" / f"iter{k}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "motion.json").write_text(json.dumps(motion, indent=1))
        t1 = time.time()
        tel = Replay(merge(env, motion), W, H).run(fps, d / "robot.mp4")
        tel_short = {k_: v for k_, v in tel.items() if k_ != "samples"}
        tel_short["samples"] = tel.get("samples", [])[::2]
        (d / "telemetry.json").write_text(json.dumps(tel, indent=1, default=float))
        side_by_side(video, d / "robot.mp4", d / "side_by_side.mp4", f"motion iter {k}")
        t_rep = time.time() - t1
        t1 = time.time()
        j = g.ask([vpart(d / "side_by_side.mp4", fps=4.0),
                   f"CURRENT MOTION:\n{json.dumps(motion)}\n\nTELEMETRY:\n{json.dumps(tel_short, default=float)}\n{facts}\n{judge}"])
        (d / "judge.json").write_text(json.dumps(j, indent=1))
        sc = float(j.get("score") or 0)
        nc = len(tel.get("collisions", []))
        print(f"[motion {k}] replay {t_rep:.1f}s  judge {time.time()-t1:.0f}s  score {sc}/10  collisions {nc}", flush=True)
        for x in (j.get("differences") or [])[:4]:
            print(f"   - {x[:150]}", flush=True)
        key = sc - 0.25 * nc
        if best is None or key > best[0]:
            best = (key, k, motion, j, sc)
        if sc >= threshold and nc == 0:
            break
        if key < best[0]:
            print(f"  regression; reverting to motion iter {best[1]}", flush=True)
            r = g.ask([vpart(out / "motion" / f"iter{best[1]}" / "side_by_side.mp4", fps=4.0),
                       f"CURRENT MOTION:\n{json.dumps(best[2])}\n\nA previous attempt to fix it made things WORSE. Its problems: "
                       f"{json.dumps(j.get('differences', []))}\nAvoid those changes. Fix these instead: "
                       f"{json.dumps(best[3].get('differences', []))}\n{facts}\n{judge}"])
            motion = r.get("revised") or best[2]
        else:
            motion = j.get("revised") or motion
    (out / "motion_best.json").write_text(json.dumps(best[2], indent=1))
    (out / "final.json").write_text(json.dumps(merge(env, best[2]), indent=1))
    sbs = out / "motion" / f"iter{best[1]}" / "side_by_side.mp4"
    print(f"[motion] best iter {best[1]} ({best[4]}/10)  ->  {sbs}", flush=True)
    return best, sbs


def run(video, name, model, env_iters, env_threshold, iters, threshold, facts, env_from=None):
    out = C.ROOT / "runs" / "mirror2" / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    W, H, fps = video_info(video)
    g = Gemini(model)
    fx = ("\nUSER-VERIFIED FACTS (ground truth; never contradict or change these):\n- " + "\n- ".join(facts) + "\n") if facts else ""
    t0 = time.time()
    env = json.loads(Path(env_from).read_text()) if env_from else phase_env(g, video, out, fx, env_iters, env_threshold, W, H)
    best, sbs = phase_motion(g, video, out, env, fx, iters, threshold, W, H, fps)
    bench = C.ROOT / "benchmarks"
    bench.mkdir(exist_ok=True)
    shutil.copy(sbs, bench / f"{name}_v2_score{best[4]}.mp4")
    shutil.copy(out / "final.json", bench / f"{name}_v2_score{best[4]}.json")
    print(f"[done] {time.time()-t0:.0f}s total", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--name", default=None)
    ap.add_argument("--model", default="gemini-4-hydrogen")
    ap.add_argument("--env-iters", type=int, default=4)
    ap.add_argument("--env-threshold", type=float, default=8.5)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=9.0)
    ap.add_argument("--fact", action="append", default=[], help="locked fact, e.g. --fact 'uses his RIGHT hand'")
    ap.add_argument("--env-from", default=None, help="skip phase 1 and use this locked environment JSON")
    a = ap.parse_args()
    run(a.video, a.name or Path(a.video).stem.replace(" ", "_"), a.model, a.env_iters, a.env_threshold,
        a.iters, a.threshold, a.fact, a.env_from)


if __name__ == "__main__":
    main()
