"""
Renders the environment screenshots the site shows in "Where are you filming?".

    .venv/bin/python web/scripts/sim/screenshots.py            # all environments
    .venv/bin/python web/scripts/sim/screenshots.py kitchen    # just one

Each environment is rendered from its generated G1 scene, with the robot standing at its start
pose, from three viewpoints. Writes web/public/envs/<env>-<n>.webp (1280 wide) and
<env>-<n>-sm.webp (640 wide). Build the G1 scenes first (each folder's BUILD_G1.py).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[2] / "public" / "envs"
W, H = 1600, 900

# camera position, look-at point, vertical field of view (deg)
ENVS = {
    "hackathon": dict(
        xml="g1_room_mujoco/Hackathon_G1_Ultra_Detail_V2/generated/G1_HACKATHON_V2.xml",
        views=[((3.2, -7.0, 2.4), (-0.8, -2.4, 0.8), 66),
               ((0.0, -3.4, 1.9), (0.0, 4.0, 1.4), 70),
               ((-8.7, -6.6, 1.8), (-10.2, -1.0, 1.0), 70)],
    ),
    "kitchen": dict(
        xml="g1_room_mujoco/Kitchen_G1_Detail/generated/G1_KITCHEN.xml",
        views=[((3.55, -3.35, 2.25), (-1.1, 0.7, 0.75), 70),
               ((0.35, 1.75, 1.6), (0.45, 3.45, 1.05), 72),
               ((-1.1, 1.55, 1.55), (-3.6, 1.6, 1.15), 72)],
    ),
    "warehouse": dict(
        xml="g1_room_mujoco/Warehouse_G1_Detail/generated/G1_WAREHOUSE.xml",
        views=[((1.3, -9.4, 2.1), (-0.2, -1.0, 1.1), 62),
               ((13.8, -11.3, 7.0), (-3.0, 3.5, 0.0), 74),
               ((7.0, -5.2, 1.7), (7.0, 14.0, 2.6), 66)],
    ),
}


def free_camera(pos, target):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    p, t = np.asarray(pos, float), np.asarray(target, float)
    d = t - p
    cam.lookat[:] = t
    cam.distance = float(np.linalg.norm(d))
    cam.azimuth = math.degrees(math.atan2(d[1], d[0]))
    cam.elevation = math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))
    return cam


def render(env: str):
    spec = ENVS[env]
    xml = REPO / spec["xml"]
    if not xml.exists():
        raise SystemExit(f"{xml} is missing: run that environment's BUILD_G1.py first")
    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
    mujoco.mj_forward(m, d)

    opt = mujoco.MjvOption()
    if env == "hackathon":
        # This room keeps its tabletop props in group 3, which MuJoCo hides by default but the
        # site's 3D view shows. Show them, and move the robot's own collision meshes (also
        # group 3) out of the way.
        pelvis = m.body("pelvis").id
        robot = {b for b in range(m.nbody) if m.body_rootid[b] == pelvis}
        for g in range(m.ngeom):
            if m.geom_group[g] == 3 and m.geom_bodyid[g] in robot:
                m.geom_group[g] = 5
        opt.geomgroup[3] = 1
        # MuJoCo 4.0.1 drops all ambient light outside a non-shadow spotlight's cone, which
        # leaves this room's upper walls black; shadow-casting spots don't have the problem.
        for i in range(m.nlight):
            if m.light_type[i] != mujoco.mjtLightType.mjLIGHT_DIRECTIONAL:
                m.light_castshadow[i] = 1

    m.vis.global_.offwidth = max(m.vis.global_.offwidth, W)
    m.vis.global_.offheight = max(m.vis.global_.offheight, H)
    r = mujoco.Renderer(m, H, W)
    OUT.mkdir(parents=True, exist_ok=True)
    for i, (pos, target, fovy) in enumerate(spec["views"], 1):
        m.vis.global_.fovy = fovy
        r.update_scene(d, camera=free_camera(pos, target), scene_option=opt)
        img = Image.fromarray(r.render())
        for suffix, width, q in (("", 1280, 80), ("-sm", 640, 78)):
            out = OUT / f"{env}-{i}{suffix}.webp"
            img.resize((width, width * H // W), Image.LANCZOS).save(out, quality=q, method=6)
            print(f"wrote {out.relative_to(REPO)} ({out.stat().st_size / 1e3:.0f} kB)")
    r.close()


if __name__ == "__main__":
    for env in sys.argv[1:] or ENVS:
        render(env)
