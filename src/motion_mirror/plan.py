"""Execute a Gemini motion plan on the G1 in MuJoCo: scene with the object, timeline, grasp, video + telemetry.

Plan JSON (written by Gemini; see gemini_loop.PLAN_PROMPT):
{
  "summary": str,
  "object": null | {"description": str, "size_m": [x, y, z], "mass_kg": float,
                    "start": {"forward_m": float, "left_m": float, "surface_height_m": float}},
  "segments": [{"t_start": s, "t_end": s, "action": str,
                "walk_forward_mps": f, "walk_left_mps": f, "turn_left_radps": f,
                "pelvis_height_m": f, "hands": "open" | "grasp"}],
  "keyframes": [{"t": s, "left_arm": {"shoulder_flexion_deg", "shoulder_abduction_deg", "elbow_flexion_deg", "shoulder_external_rotation_deg"},
                 "right_arm": {...}, "torso_lean_forward_deg": f, "torso_twist_left_deg": f}]
}
Velocities are in the robot's own frame. Positions are relative to the robot's start pose.
"""
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from . import config as C
from .skill_env import CMD_RANGES, H_MIN, SkillEnv

ELBOW_STRAIGHT = 1.45
GRASP_DIST = 0.05  # m, palm-to-box-surface distance that counts as touching
PALM = np.array([0.10, 0.0, 0.0])  # palm center in the wrist_yaw_link frame (the hand mesh is ~10 cm past the wrist)


# ---------------- human angles -> G1 joints ----------------
def arm_to_g1(arm: dict, side: str) -> np.ndarray:
    f = np.radians(arm.get("shoulder_flexion_deg", 10))
    a = np.radians(arm.get("shoulder_abduction_deg", 10))
    e = np.radians(arm.get("elbow_flexion_deg", 25))
    r = np.radians(arm.get("shoulder_external_rotation_deg", 0))
    sgn = 1 if side == "left" else -1
    q = np.array([-f, sgn * a, sgn * r, ELBOW_STRAIGHT - e, 0.0, 0.0, 0.0])
    return q


def keyframe_to_upper(kf: dict) -> np.ndarray:
    up = np.zeros(C.NU)
    up[0] = np.radians(kf.get("torso_twist_left_deg", 0.0))
    up[2] = np.radians(kf.get("torso_lean_forward_deg", 0.0))
    up[3:10] = arm_to_g1(kf.get("left_arm", {}), "left")
    up[10:17] = arm_to_g1(kf.get("right_arm", {}), "right")
    lo, hi = C.UPPER_SAMPLE_RANGE[:, 0], C.UPPER_SAMPLE_RANGE[:, 1]
    lims = np.stack([lo, hi], 1)
    lims[0] = [-0.6, 0.6]
    lims[2] = [-0.1, 0.52]
    return np.clip(up, lims[:, 0], lims[:, 1])


def _min_jerk(s):
    return s ** 3 * (10 - 15 * s + 6 * s ** 2)


def build_timeline(plan: dict, base_height: float):
    segs = sorted(plan.get("segments", []), key=lambda s: s["t_start"])
    T_end = max([s["t_end"] for s in segs] + [k["t"] for k in plan.get("keyframes", [])] + [1.0]) + 1.0
    n = int(T_end / C.CONTROL_DT)
    t = np.arange(n) * C.CONTROL_DT
    cmd = np.zeros((n, 4))
    cmd[:, 3] = base_height
    grasp = np.zeros(n, bool)
    for s in segs:
        idx = (t >= s["t_start"]) & (t < s["t_end"])
        cmd[idx, 0] = np.clip(s.get("walk_forward_mps", 0.0), *CMD_RANGES["vx"])
        cmd[idx, 1] = np.clip(s.get("walk_left_mps", 0.0), *CMD_RANGES["vy"])
        cmd[idx, 2] = np.clip(s.get("turn_left_radps", 0.0), *CMD_RANGES["wz"])
        h = s.get("pelvis_height_m")
        cmd[idx, 3] = base_height if h is None else np.clip(h, H_MIN, base_height)
        grasp[idx] = s.get("hands", "open") == "grasp"
    # carry the last grasp state to the end
    # upper body: min-jerk between keyframes, starting from the default pose
    kfs = sorted(plan.get("keyframes", []), key=lambda k: k["t"])
    times = [0.0] + [k["t"] for k in kfs]
    poses = [C.DEFAULT_Q[C.UPPER].copy()] + [keyframe_to_upper(k) for k in kfs]
    upper = np.repeat(poses[-1][None], n, 0)
    for i in range(len(times) - 1):
        t0, t1 = times[i], max(times[i + 1], times[i] + 0.3)
        idx = (t >= t0) & (t < t1)
        s = _min_jerk(np.clip((t[idx] - t0) / (t1 - t0), 0, 1))[:, None]
        upper[idx] = poses[i] + s * (poses[i + 1] - poses[i])
    upper[t < times[0]] = poses[0]
    if plan.get("object") and grasp.any():
        ground_grasp(plan, t, cmd, upper, grasp, base_height)
    return t, cmd, upper, grasp


def _dead_reckon(t, cmd, upto):
    """Robot (x, y, yaw) at time `upto` if it follows the commanded velocities."""
    x = y = yaw = 0.0
    for i in range(int(upto / C.CONTROL_DT)):
        vx, vy, wz = cmd[i, :3]
        x += (np.cos(yaw) * vx - np.sin(yaw) * vy) * C.CONTROL_DT
        y += (np.sin(yaw) * vx + np.cos(yaw) * vy) * C.CONTROL_DT
        yaw += wz * C.CONTROL_DT
    return x, y, yaw


def ground_grasp(plan, t, cmd, upper, grasp, base_height):
    """Replace Gemini's guessed arm angles around the grasp with exact IK (kinematics only, no simulation)."""
    from .ik import grasp_targets, solve_palms
    obj = plan["object"]
    st = obj.get("start", {})
    size = np.array(obj.get("size_m", [0.3, 0.25, 0.2]))
    g0 = int(np.argmax(grasp))                              # first grasp step
    g1 = g0 + int(np.argmin(grasp[g0:])) if not grasp[g0:].all() else len(t)  # release step
    x, y, yaw = _dead_reckon(t, cmd, t[g0])
    dx, dy = st.get("forward_m", 1.0) - x, st.get("left_m", 0.0) - y
    rel = np.array([np.cos(yaw) * dx + np.sin(yaw) * dy, -np.sin(yaw) * dx + np.cos(yaw) * dy])
    half_lat = abs(size[1] / 2 * np.cos(yaw)) + abs(size[0] / 2 * np.sin(yaw))
    box_z = st.get("surface_height_m", 0.4) + size[2] / 2
    h_plan = float(np.clip(cmd[g0, 3], H_MIN, base_height))
    up0 = upper[max(g0 - 1, 0)]
    # where to stand: allow stepping up to 0.4 m closer (never closer than 0.25 m from the box center)
    stance = (0.0, max(0.0, min(0.4, rel[0] - 0.25)))
    g_up, h_g, step, err = solve_palms(*grasp_targets(rel, box_z, half_lat), up0, base_height, (H_MIN, base_height), stance_range=stance)
    rel_s = rel - np.array([step, 0.0])
    pre_up, _, _, _ = solve_palms(*grasp_targets(rel_s, box_z, half_lat, out=0.08, up=0.05), g_up, h_g, (h_g, h_g))
    plan.setdefault("_grounding", {}).update(grasp_ik_err_m=round(err, 3), grasp_pelvis_h=round(h_g, 3),
                                              step_closer_m=round(step, 3), object_rel_at_grasp=rel.round(3).tolist())
    if step > 0.02:  # add the approach step, before reaching (like the human would)
        k_pre0 = max(g0 - int(1.2 / C.CONTROL_DT), 0)
        win = max(int(1.0 / C.CONTROL_DT), 1)
        a = max(k_pre0 - win - int(0.6 / C.CONTROL_DT), 0)
        cmd[a:a + win, 0] += step / (win * C.CONTROL_DT)
    # timeline: reach pre-grasp 1.2 s before, close onto the sides by g0, hold the arm pose while carrying
    k_pre, k_close = max(g0 - int(1.2 / C.CONTROL_DT), 0), max(g0 - int(0.4 / C.CONTROL_DT), 0)
    for a, b, p0, p1 in ((k_pre, k_close, upper[k_pre].copy(), pre_up), (k_close, g0, pre_up, g_up)):
        if b > a:
            s = _min_jerk(np.linspace(0, 1, b - a))[:, None]
            upper[a:b] = p0 + s * (p1 - p0)
    upper[g0:g1] = g_up
    # squat to the solved height for the reach and the grasp itself; Gemini's lift/carry heights follow after
    first_seg_end = g0 + int(np.argmax(cmd[g0:, 3] != cmd[g0, 3])) if (cmd[g0:, 3] != cmd[g0, 3]).any() else g1
    cmd[k_pre - int(0.6 / C.CONTROL_DT) if k_pre > 30 else 0:first_seg_end, 3] = h_g


# ---------------- scene ----------------
def build_scene(plan: dict) -> Path:
    obj = plan.get("object")
    extra, eq = "", ""
    if obj:
        sx, sy, sz = (np.array(obj.get("size_m", [0.3, 0.25, 0.2])) / 2).tolist()
        st = obj.get("start", {})
        x, y = st.get("forward_m", 1.0), st.get("left_m", 0.0)
        h = max(st.get("surface_height_m", 0.4), 0.0)
        mass = obj.get("mass_kg", 1.5)
        # pedestal narrower than the box, set back from the robot, so knees/shins have clearance when squatting
        stand = (f'<geom name="stand" type="box" size="{sx*0.6:.3f} {sy*0.8:.3f} {h/2:.3f}" pos="{x + sx*0.4:.3f} {y:.3f} {h/2:.3f}" '
                 f'rgba="0.45 0.35 0.25 1"/>') if h > 0.01 else ""
        extra = f'''{stand}
    <body name="object" pos="{x:.3f} {y:.3f} {h + sz + 0.002:.3f}">
      <freejoint/>
      <geom name="object" type="box" size="{sx:.3f} {sy:.3f} {sz:.3f}" mass="{mass:.3f}" rgba="0.9 0.5 0.1 1" friction="1.0 0.02 0.001"/>
    </body>'''
        eq = '''<equality>
    <connect name="grasp_left" body1="left_wrist_yaw_link" body2="object" anchor="0 0 0" active="false"/>
    <connect name="grasp_right" body1="right_wrist_yaw_link" body2="object" anchor="0 0 0" active="false"/>
  </equality>'''
    src = (C.SCENE_XML).read_text()
    src = src.replace('<geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>',
                      '<geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>\n    ' + extra)
    src = src.replace("</mujoco>", eq + "\n</mujoco>")
    out = C.SCENE_XML.parent / "_task_scene.xml"
    out.write_text(src)
    return out


# ---------------- grasp ----------------
class Grasp:
    """Simplified, disclosed grasp: when the plan says 'grasp' and both hands touch the object, attach it."""

    def __init__(self, env: SkillEnv):
        m = env.m
        self.env = env
        self.ok = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object") >= 0
        if not self.ok:
            return
        self.obj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
        self.geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "object")
        self.eq = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, n) for n in ("grasp_left", "grasp_right")]
        self.hands = env.wrists
        self.held = False

    def palm(self, h):
        d = self.env.d
        return d.xpos[h] + d.xmat[h].reshape(3, 3) @ PALM

    def hand_dists(self):
        d, m = self.env.d, self.env.m
        R = d.xmat[self.obj].reshape(3, 3)
        half = m.geom_size[self.geom]
        out = []
        for h in self.hands:
            p = R.T @ (self.palm(h) - d.xpos[self.obj])
            out.append(float(np.linalg.norm(p - np.clip(p, -half, half))))
        return out

    def update(self, want: bool):
        if not self.ok:
            return
        d, m = self.env.d, self.env.m
        if want and not self.held and max(self.hand_dists()) < GRASP_DIST:
            R = d.xmat[self.obj].reshape(3, 3)
            for e, h in zip(self.eq, self.hands):
                m.eq_data[e, 0:3] = PALM
                m.eq_data[e, 3:6] = R.T @ (self.palm(h) - d.xpos[self.obj])
                d.eq_active[e] = 1
            self.held = True
        elif not want and self.held:
            for e in self.eq:
                d.eq_active[e] = 0
            self.held = False


# ---------------- execution ----------------
def load_policy(path):
    from .ppo import ActorCritic
    ck = torch.load(path, weights_only=False)
    ac = ActorCritic(ck["obs_dim"], ck["priv_dim"], C.NJ)
    ac.load_state_dict(ck["model"])
    ac.eval()
    return ac


def execute(plan: dict, policy, video_path=None, randomize=False, seed=0, width=640, height=480, viewer=None):
    xml = build_scene(plan)
    env = SkillEnv(seed=seed, randomize=randomize, xml_path=xml)
    if not randomize:
        env.push_scale = 0.0
    obs, _ = env.reset()
    t, cmd, upper, grasp_tl = build_timeline(plan, env._base_height)
    n = len(t)
    env.ref = np.vstack([upper, np.repeat(upper[-1:], max(C.REF_LOOKAHEAD) + 2, 0)])
    env.plan = lambda i: (cmd[min(i, n - 1)], env.ref[i + 1])
    env.h_smooth = env._base_height
    env.d.qpos[7 + C.UPPER] = upper[0]
    mujoco.mj_forward(env.m, env.d)
    obs, _ = env._obs()
    grasp = Grasp(env)

    renderer = cam = None
    frames = []
    if video_path:
        renderer = mujoco.Renderer(env.m, height, width)
        cam = mujoco.MjvCamera()
        cam.distance, cam.azimuth, cam.elevation = 3.2, 135, -18

    if viewer is not None:  # live view: swap in this run's model and data
        viewer = viewer(env.m, env.d)
    tel = dict(fell_at_s=None, grasped_at_s=None, released_at_s=None, min_hand_object_dist_m=None,
               samples=[])
    min_d = np.inf
    for i in range(n - 1):
        a = policy.act_deterministic(torch.from_numpy(obs)[None])[0].numpy()
        (obs, _), _, fell, _, info = env.step(a)
        was = grasp.held if grasp.ok else False
        grasp.update(bool(grasp_tl[i]))
        if grasp.ok:
            hd = grasp.hand_dists()
            min_d = min(min_d, max(hd))
            if grasp.held and not was:
                tel["grasped_at_s"] = round(t[i], 2)
            if was and not grasp.held:
                tel["released_at_s"] = round(t[i], 2)
        if i % 25 == 0:
            s = dict(t=round(t[i], 1), pelvis_xy=env.d.qpos[:2].round(2).tolist(), pelvis_h=round(float(env.d.qpos[2]), 2),
                     cmd=cmd[i].round(2).tolist(), arm_err_rad=round(info["joint_err"], 3))
            if grasp.ok:
                s.update(object_xyz=env.d.xpos[grasp.obj].round(2).tolist(), hand_obj_dist=[round(x, 2) for x in hd],
                         holding=grasp.held)
            tel["samples"].append(s)
        if renderer is not None:
            cam.lookat[:] = env.d.qpos[:3] * np.array([1, 1, 0]) + np.array([0.3, 0, 0.6])
            renderer.update_scene(env.d, cam)
            frames.append(renderer.render())
        if viewer is not None:
            import time as _t
            if not viewer.is_running():
                break
            viewer.sync()
            _t.sleep(C.CONTROL_DT)
        if fell:
            tel["fell_at_s"] = round(t[i], 2)
            break
    if grasp.ok:
        tel["min_hand_object_dist_m"] = round(float(min_d), 3)
        tel["object_final_xyz"] = env.d.xpos[grasp.obj].round(2).tolist()
    tel["robot_final_xy"] = env.d.qpos[:2].round(2).tolist()
    tel["duration_s"] = round(t[-1], 1)
    if video_path:
        import imageio
        imageio.mimsave(video_path, frames, fps=int(1 / C.CONTROL_DT), macro_block_size=8)
    return tel


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("--ckpt", default=str(C.ROOT / "runs/skill/latest.pt"))
    ap.add_argument("--video", default=None)
    ap.add_argument("--randomize", action="store_true")
    ap.add_argument("--live", action="store_true", help="interactive 3D viewer (run with mjpython on macOS)")
    args = ap.parse_args()
    viewer = None
    if args.live:
        import mujoco.viewer
        viewer = lambda m, d: mujoco.viewer.launch_passive(m, d)
    tel = execute(json.loads(Path(args.plan).read_text()), load_policy(args.ckpt), args.video, args.randomize,
                  viewer=viewer)
    if args.live:
        input("Done. Press Enter to close the viewer...")
    print(json.dumps({k: v for k, v in tel.items() if k != "samples"}, indent=1))


if __name__ == "__main__":
    main()
