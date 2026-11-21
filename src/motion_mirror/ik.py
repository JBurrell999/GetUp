"""Kinematics-only grounding of Gemini plans (milliseconds, no physics simulation).

Gemini says *what* to do (grasp the box at t, object size/position); this solves *exactly how*:
the arm joints, waist lean and squat height that put both palms on the object's sides.
"""
import mujoco
import numpy as np

from . import config as C

PALM = np.array([0.10, 0.0, 0.0])          # palm center in wrist_yaw_link frame
ARM_L, ARM_R = [15, 16, 17, 18], [22, 23, 24, 25]   # shoulder pitch/roll/yaw, elbow
WAIST_PITCH = 14

_m = _d = None


def _model():
    global _m, _d
    if _m is None:
        _m = mujoco.MjModel.from_xml_path(str(C.SCENE_XML))
        _d = mujoco.MjData(_m)
    return _m, _d


def solve_palms(left_target, right_target, upper_init, h_init, h_range, iters=200, stance_range=(0.0, 0.0)):
    """Find upper-body joints, pelvis height and (optionally) a forward stance shift so the palms reach the targets.

    Robot starts at the origin facing +x. Prefers an upright, balanced posture: arms do the work first,
    then stepping closer, and only then leaning/squatting.
    Returns (upper17, pelvis_height, stance_shift_x, max_palm_error).
    """
    m, d = _model()
    wl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
    wr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    joints = [WAIST_PITCH] + ARM_L + ARM_R
    q = C.DEFAULT_Q.copy()
    q[C.UPPER] = upper_init
    z = h_init
    sx = 0.0
    lo = m.jnt_range[1:1 + C.NJ, 0].copy()
    hi = m.jnt_range[1:1 + C.NJ, 1].copy()
    lo[WAIST_PITCH], hi[WAIST_PITCH] = -0.1, 0.52
    tgt = np.concatenate([left_target, right_target])
    jacp = np.zeros((3, m.nv))
    err = np.inf
    for _ in range(iters):
        d.qpos[:] = 0
        d.qpos[0], d.qpos[2], d.qpos[3] = sx, z, 1.0
        d.qpos[7:7 + C.NJ] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        palms, J = [], []
        for b in (wl, wr):
            p = d.xpos[b] + d.xmat[b].reshape(3, 3) @ PALM
            mujoco.mj_jac(m, d, jacp, None, p, b)
            palms.append(p)
            J.append(np.concatenate([jacp[:, [6 + j for j in joints]], jacp[:, [2, 0]]], 1))
        e = tgt - np.concatenate(palms)
        err = np.linalg.norm(e.reshape(2, 3), axis=1).max()
        if err < 0.005:
            break
        J = np.vstack(J)
        W = np.ones(J.shape[1])
        W[0] = 0.3                      # waist lean: last resort
        W[-2] = 0.3                     # squat: last resort
        W[-1] = 1.5 if stance_range[1] > stance_range[0] else 0.0   # stepping closer: preferred
        Jw = J * W
        dx = W * (Jw.T @ np.linalg.solve(Jw @ Jw.T + 1e-3 * np.eye(6), e))
        # pull toward upright/standing tall in the null space (balance)
        dx[0] -= 0.02 * q[WAIST_PITCH]
        dx[-2] += 0.02 * (h_range[1] - z)
        dx = np.clip(dx, -0.15, 0.15)
        for k, j in enumerate(joints):
            q[j] = np.clip(q[j] + dx[k], lo[j], hi[j])
        z = float(np.clip(z + dx[-2], *h_range))
        sx = float(np.clip(sx + dx[-1], *stance_range))
    return q[C.UPPER].copy(), z, sx, float(err)


def grasp_targets(box_rel, box_z, half_lateral, clearance=0.02, out=0.0, up=0.0):
    """Palm targets just outside the left/right faces of the box (robot frame)."""
    x, y = box_rel
    w = half_lateral + clearance + out
    return np.array([x, y + w, box_z + up]), np.array([x, y - w, box_z + up])
