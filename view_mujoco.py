#!/usr/bin/env python3
"""
view_mujoco.py - replay a capture's 3D joints (joints3d.npz from lift3d.py) in MuJoCo.

    mjpython view_mujoco.py outputs/<capture>                     # interactive viewer, loops
    python view_mujoco.py outputs/<capture> --render replay.mp4   # render a video instead

On macOS the interactive viewer has to be started with mjpython (installed with mujoco).
The 24 SMPL joints are mocap spheres with bones drawn between them, over a floor at z = 0.
This is a kinematic replay for checking the 3D; driving a humanoid model means solving IK
for its joint angles against these positions, which is the next step.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

C_LEFT, C_RIGHT, C_MID = (1.0, 0.55, 0.0), (0.0, 0.67, 1.0), (0.92, 0.92, 0.92)
HIDDEN = np.array([0.0, 0.0, -10.0])         # where joints with no data are parked


def load(cap_dir):
    z = np.load(Path(cap_dir) / "joints3d.npz")
    names = [str(n) for n in z["joint_names"]]
    colors = [C_LEFT if n.startswith("l_") else C_RIGHT if n.startswith("r_") else C_MID
              for n in names]
    return z["xyz"].astype(np.float64), z["t"], names, [int(p) for p in z["parents"]], colors


def build_model(names, colors):
    bodies = "".join(
        f'<body name="{n}" mocap="true"><geom type="sphere" size=".028" contype="0" '
        f'conaffinity="0" rgba="{c[0]} {c[1]} {c[2]} 1"/></body>' for n, c in zip(names, colors))
    return mujoco.MjModel.from_xml_string(f"""
<mujoco model="capture replay">
  <visual>
    <headlight diffuse=".6 .6 .6" ambient=".35 .35 .35"/>
    <global offwidth="1920" offheight="1080"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".2 .22 .25" rgb2=".28 .3 .33"
             width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="10 10" reflectance=".08"/>
    <texture type="skybox" builtin="gradient" rgb1=".45 .55 .65" rgb2=".05 .05 .08"
             width="512" height="512"/>
  </asset>
  <worldbody>
    <light pos="1 -1 4" dir="-.2 .2 -1" directional="true"/>
    <geom name="floor" type="plane" size="6 6 .1" material="grid"/>
    {bodies}
  </worldbody>
</mujoco>""")


def set_frame(d, p):
    d.mocap_pos[:] = np.where(np.isfinite(p).all(1, keepdims=True), p, HIDDEN)


def add_bones(scn, p, parents, colors, width=0.014):
    for j, par in enumerate(parents):
        if par < 0 or scn.ngeom >= scn.maxgeom:
            continue
        if not (np.isfinite(p[j]).all() and np.isfinite(p[par]).all()):
            continue
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                            np.eye(3).ravel(), np.array([*colors[j], 1.0], np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, p[par], p[j])
        scn.ngeom += 1


def framing(xyz):
    """Fixed look-at point and distance that keep the whole clip in view."""
    pts = xyz[np.isfinite(xyz).all(2)]
    if not len(pts):
        return np.array([0.0, 0.0, 0.5]), 3.0
    lo, hi = np.percentile(pts, 2, 0), np.percentile(pts, 98, 0)
    return (lo + hi) / 2, max(2.2, 1.6 * float(np.max(hi - lo)))


def render(cap_dir, out, height=720):
    xyz, t, names, parents, colors = load(cap_dir)
    m = build_model(names, colors)
    d = mujoco.MjData(m)
    w = height * 8 // 9                       # two panels side by side, 16:9 overall
    r = mujoco.Renderer(m, height, w)
    center, dist = framing(xyz)
    views = []
    for az, el, label in ((90, -8, "original camera's view"), (20, -18, "from the side")):
        cam = mujoco.MjvCamera()
        cam.type, cam.azimuth, cam.elevation = mujoco.mjtCamera.mjCAMERA_FREE, az, el
        cam.lookat[:], cam.distance = center, dist
        views.append((cam, label))
    fps = (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 else 30.0
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"avc1"), fps, (2 * w, height))
    if not writer.isOpened():
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (2 * w, height))
    for i in range(len(t)):
        set_frame(d, xyz[i])
        mujoco.mj_forward(m, d)
        panels = []
        for cam, label in views:
            r.update_scene(d, cam)
            add_bones(r.scene, xyz[i], parents, colors)
            img = np.ascontiguousarray(r.render()[:, :, ::-1])
            cv2.putText(img, f"{label}   t={t[i]:5.2f}s", (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)
            panels.append(img)
        writer.write(np.hstack(panels))
    writer.release()
    r.close()
    print(f"[view_mujoco] rendered {len(t)} frames -> {out}")


def interactive(cap_dir):
    import mujoco.viewer

    xyz, t, names, parents, colors = load(cap_dir)
    m = build_model(names, colors)
    d = mujoco.MjData(m)
    center, dist = framing(xyz)
    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.lookat[:], v.cam.distance, v.cam.azimuth, v.cam.elevation = center, dist, 90, -8
        t0, span = time.monotonic(), float(t[-1]) + 0.5
        while v.is_running():
            i = min(int(np.searchsorted(t, (time.monotonic() - t0) % span)), len(t) - 1)
            set_frame(d, xyz[i])
            mujoco.mj_forward(m, d)
            with v.lock():
                v.user_scn.ngeom = 0
                add_bones(v.user_scn, xyz[i], parents, colors)
            v.sync()
            time.sleep(1 / 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Replay a capture's 3D joints in MuJoCo")
    ap.add_argument("capture", help="capture folder containing joints3d.npz")
    ap.add_argument("--render", metavar="OUT.mp4", help="render to a video instead of a window")
    a = ap.parse_args()
    render(a.capture, a.render) if a.render else interactive(a.capture)
