"""Gemini-in-the-loop: video -> plan -> MuJoCo rollout video -> Gemini judges + revises -> ... -> robustness check.

uv run python -m motion_mirror.gemini_loop videos/clip.mp4 --name demo1
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from google import genai
from google.genai import types

from . import config as C
from .plan import execute, load_policy

ROBOT_SPEC = """ROBOT: Unitree G1 humanoid (1.3 m tall, 35 kg). It is driven by a trained whole-body controller that accepts:
- walk_forward_mps in [-0.4, 0.8], walk_left_mps in [-0.3, 0.3], turn_left_radps in [-0.8, 0.8] (robot frame).
  Walking is only reliable with pelvis_height_m >= 0.65. Turning 90 degrees at 0.8 rad/s takes ~2 s.
- pelvis_height_m in [0.45, 0.78] (0.78 = standing tall, 0.45 = deep squat). Height changes at most 0.3 m/s.
- arm keyframes per arm: shoulder_flexion_deg (0 = arm down at side, 90 = pointing forward, 180 = straight up),
  shoulder_abduction_deg (0 = at side, 90 = out sideways), elbow_flexion_deg (0 = straight, 90 = right angle),
  shoulder_external_rotation_deg (upper-arm rotation in [-80, 85]: with the elbow bent, 0 = forearm points forward,
  +90 = forearm rotated up/outward as in a bicep flex, negative = forearm rotated inward across the body);
  torso_lean_forward_deg in [-5, 30], torso_twist_left_deg in [-35, 35]. Keyframes are smoothly interpolated.
- hands: no fingers. To pick up an object it squeezes it between both hands: plan "hands": "grasp" while both hands
  are pressed on the sides of the object (shoulder_abduction ~10-15 deg, arms forward). "open" releases it.
- Lowest reachable hand height is ~0.34 m above the floor (deep squat + lean). Objects lower than that are unreachable:
  place them on a surface (surface_height_m) so the object's mid-height is >= 0.4 m, and say so in "adaptations".
- Distances: the robot's shoulder width is ~0.3 m; arm length ~0.5 m. To grasp, the object center should end up
  ~0.35-0.45 m in front of the pelvis.
"""

PLAN_SCHEMA = """Return JSON:
{
 "summary": str,                 # what the human is doing
 "environment": str,             # the scene: surfaces/furniture the object sits on, layout, distances
 "adaptations": [str],   # where the robot must deviate from the human (reach limits, speed limits, ...)
 "object": null | {"description": str, "size_m": [depth, width, height], "mass_kg": float,
                   "start": {"forward_m": float, "left_m": float, "surface_height_m": float}},
 "segments": [{"t_start": float, "t_end": float, "action": "stand|walk|turn|squat|reach|grasp|lift|carry|place|release|gesture",
               "walk_forward_mps": float, "walk_left_mps": float, "turn_left_radps": float,
               "pelvis_height_m": float, "hands": "open|grasp"}],
 "keyframes": [{"t": float, "left_arm": {"shoulder_flexion_deg": f, "shoulder_abduction_deg": f, "elbow_flexion_deg": f, "shoulder_external_rotation_deg": f},
                "right_arm": {...}, "torso_lean_forward_deg": f, "torso_twist_left_deg": f}]
}
You only decide WHAT happens and WHEN; exact arm angles for grasping are computed automatically from the object's
size and position, so get the object geometry (size, distance, surface height) and the timing right.
Segments must tile time without gaps, starting at 0. Object position is relative to where the robot starts
(forward_m ahead, left_m to its left). Use the PERSON's left/right. Plan the same task, order and rough timing as the video.
"""

PLAN_PROMPT = f"""You convert a casual, uncalibrated video of a person into an executable plan for a humanoid robot.
Watch the video carefully: what the person does, in what order, how long each part takes, where objects are,
how they hold them, and their arm posture. Estimate distances from human body proportions (a person is ~1.7 m tall).

{ROBOT_SPEC}
{PLAN_SCHEMA}"""

JUDGE_PROMPT = f"""You are evaluating whether a simulated robot faithfully reproduces a human's movement.
Video 1 is the original human video. Video 2 is the robot executing the current plan in the MuJoCo simulator.
You also get the plan and the simulator telemetry (robot/object positions, grasp events, falls).

Score the robot rollout 0-10 for fidelity to the human video:
task outcome (e.g. object picked up and moved to the right place), order of actions, timing, posture/arm motion, stability.
Then fix the plan so the next rollout is closer. Use the telemetry: e.g. if the hands never got within 0.07 m of the
object, adjust walking distance or arm reach; if the robot fell, slow down or reduce lean; if timing is off, retime segments.

{ROBOT_SPEC}
Return JSON: {{"score": float, "what_matches": [str], "issues": [str], "changes": [str], "revised_plan": <full plan JSON, same schema>}}
Plan schema:
{PLAN_SCHEMA}"""


class Gemini:
    def __init__(self, model):
        self.client = genai.Client()
        self.model = model

    def ask(self, parts, retries=2):
        for k in range(retries + 1):
            try:
                r = self.client.models.generate_content(
                    model=self.model, contents=parts,
                    config=types.GenerateContentConfig(response_mime_type="application/json"))
                return json.loads(r.text)
            except Exception as e:  # noqa: BLE001
                if k == retries:
                    raise
                print(f"  gemini retry ({e.__class__.__name__}: {str(e)[:120]})")
                time.sleep(2)


def video_part(path):
    return types.Part.from_bytes(data=Path(path).read_bytes(), mime_type="video/mp4")


def success(tel, plan):
    ok = tel["fell_at_s"] is None
    if plan.get("object"):
        ok = ok and tel["grasped_at_s"] is not None
    return ok


def robustness(plan, policy, n=8):
    """Same plan under randomized physics (friction, masses, payload, gains, latency, pushes, sensor noise)."""
    res = []
    for s in range(n):
        tel = execute(plan, policy, None, randomize=True, seed=1000 + s)
        res.append(dict(seed=s, success=success(tel, plan), fell_at_s=tel["fell_at_s"], grasped_at_s=tel["grasped_at_s"]))
    return dict(success_rate=float(np.mean([r["success"] for r in res])), trials=res)


def run(video, name, model, iters, threshold, ckpt, robust_n):
    out = C.ROOT / "runs" / "mirror" / name
    out.mkdir(parents=True, exist_ok=True)
    g = Gemini(model)
    policy = load_policy(ckpt)
    human = video_part(video)

    t0 = time.time()
    print(f"[plan] {model} watching {video} ...")
    plan = g.ask([human, PLAN_PROMPT])
    print(f"  {time.time()-t0:.0f}s: {plan.get('summary')}")
    for a in plan.get("adaptations", []):
        print(f"  adaptation: {a}")
    history = []
    for k in range(iters):
        d = out / f"iter{k}"
        d.mkdir(exist_ok=True)
        (d / "plan.json").write_text(json.dumps(plan, indent=1))
        t1 = time.time()
        tel = execute(plan, policy, d / "sim.mp4")
        (d / "telemetry.json").write_text(json.dumps(tel, indent=1))
        print(f"[iter {k}] sim {time.time()-t1:.0f}s: fell={tel['fell_at_s']} grasped={tel['grasped_at_s']} "
              f"min_hand_obj={tel.get('min_hand_object_dist_m')} final_obj={tel.get('object_final_xyz')}")
        t1 = time.time()
        j = g.ask([human, video_part(d / "sim.mp4"),
                   f"CURRENT PLAN:\n{json.dumps(plan)}\n\nTELEMETRY:\n{json.dumps(tel)}\n\n{JUDGE_PROMPT}"])
        (d / "judge.json").write_text(json.dumps(j, indent=1))
        print(f"  judge {time.time()-t1:.0f}s: score {j.get('score')}/10")
        for s in j.get("issues", [])[:4]:
            print(f"   - {s}")
        history.append(dict(iter=k, score=j.get("score"), success=success(tel, plan)))
        if (j.get("score") or 0) >= threshold and success(tel, plan):
            print("  accepted.")
            break
        plan = j.get("revised_plan") or plan
    best = max(range(len(history)), key=lambda i: ((history[i]["success"]), history[i]["score"] or 0))
    final = json.loads((out / f"iter{best}" / "plan.json").read_text())
    (out / "final_plan.json").write_text(json.dumps(final, indent=1))
    print(f"[robustness] best = iter {best}; running {robust_n} randomized-physics trials ...")
    rob = robustness(final, policy, robust_n)
    print(f"  success rate under domain randomization: {rob['success_rate']:.0%}")
    summary = dict(video=str(video), model=model, history=history, best_iter=best, robustness=rob,
                   wall_s=round(time.time() - t0, 1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--name", default=None)
    ap.add_argument("--model", default="gemini-4-hydrogen")
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=8.0)
    ap.add_argument("--ckpt", default=str(C.ROOT / "runs/skill/latest.pt"))
    ap.add_argument("--robust-n", type=int, default=8)
    a = ap.parse_args()
    run(a.video, a.name or Path(a.video).stem.replace(" ", "_"), a.model, a.iters, a.threshold, a.ckpt, a.robust_n)


if __name__ == "__main__":
    main()
