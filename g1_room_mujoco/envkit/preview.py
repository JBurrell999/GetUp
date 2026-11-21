"""Compile a scene, settle its free props, and render every named camera to PNG."""
from __future__ import annotations

from pathlib import Path


def check(xml: Path, settle_s: float = 1.0) -> dict:
    """Compile and step the scene; report how far free props moved (they should rest in place)."""
    import mujoco
    import numpy as np

    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    q0 = d.qpos.copy()
    for _ in range(int(settle_s / m.opt.timestep)):
        mujoco.mj_step(m, d)
    drift = 0.0
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            a = m.jnt_qposadr[j]
            drift = max(drift, float(np.linalg.norm(d.qpos[a:a + 3] - q0[a:a + 3])))
    return dict(ngeom=m.ngeom, nbody=m.nbody, ncam=m.ncam, nlight=m.nlight, nq=m.nq,
                contacts_after_settle=int(d.ncon), max_prop_drift_m=round(drift, 4))


def render_all(xml: Path, out_dir: Path, width=1280, height=720, cameras=None, hide_ceiling_for=("top",)):
    import mujoco
    from PIL import Image

    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    out_dir.mkdir(parents=True, exist_ok=True)
    r = mujoco.Renderer(m, height, width)
    opt = mujoco.MjvOption()
    written = []
    for i in range(m.ncam):
        name = m.camera(i).name
        if cameras and name not in cameras:
            continue
        opt.geomgroup[1] = 0 if any(k in name for k in hide_ceiling_for) else 1
        r.update_scene(d, camera=i, scene_option=opt)
        p = out_dir / f"{name}.jpg"
        Image.fromarray(r.render()).save(p, quality=88)
        written.append(p)
    r.close()
    return written
