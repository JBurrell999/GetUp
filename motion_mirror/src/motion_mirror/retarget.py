"""Per-frame body posture from the video (MediaPipe 3D world landmarks) retargeted onto the G1, plus a camera that
reproduces the video's perspective.

Posture: limb DIRECTIONS are matched (scale-free, so a 1.3 m robot copies a 1.7 m person): upper arm / forearm in the
torso frame, thigh / shin in the pelvis frame, torso relative to pelvis (waist), pelvis tilt relative to gravity.
Low-visibility limbs (occluded, out of frame) fall back to Gemini's keyframes, blended smoothly.

Perspective: MediaPipe's world landmarks are camera-aligned (x right, y down, z away from the camera), so they say
exactly how the body is oriented relative to the camera in every frame. Placing the simulator camera at the same pose
relative to the robot's body reproduces the video's viewpoint by construction.
"""
import mujoco
import numpy as np

from . import config as C

L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI = 11, 12, 13, 14, 15, 16
L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK = 23, 24, 25, 26, 27, 28
ARM = {"left": (list(range(15, 19)), L_SHO, L_ELB, L_WRI, "left"), "right": (list(range(22, 26)), R_SHO, R_ELB, R_WRI, "right")}
LEG = {"left": ([0, 1, 2, 3], L_HIP, L_KNE, L_ANK), "right": ([6, 7, 8, 9], R_HIP, R_KNE, R_ANK)}
WAIST = [12, 13, 14]


def _n(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def _frame(left_pt, right_pt, low, high):
    """Body frame (x forward, y left, z up) from a left-right point pair and a low->high axis."""
    y = _n(left_pt - right_pt)
    z = high - low
    z = _n(z - (z @ y) * y)
    x = np.cross(y, z)
    return np.stack([x, y, z], 1)


def _smooth(X, sigma):
    if sigma <= 0 or len(X) < 3:
        return X
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    P = np.pad(X, [(r, r)] + [(0, 0)] * (X.ndim - 1), mode="edge")
    return np.apply_along_axis(lambda c: np.convolve(c, k, mode="valid"), 0, P)


class Retargeter:
    def __init__(self, replay):
        self.rp = replay
        m = replay.m
        self.m, self.d = m, mujoco.MjData(m)
        b = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        self.B = dict(pelvis=b("pelvis"), torso=b("torso_link"),
                      l_sho=b("left_shoulder_roll_link"), l_elb=b("left_elbow_link"), l_wri=b("left_wrist_roll_link"),
                      r_sho=b("right_shoulder_roll_link"), r_elb=b("right_elbow_link"), r_wri=b("right_wrist_roll_link"),
                      l_hip=b("left_hip_roll_link"), l_kne=b("left_knee_link"), l_ank=b("left_ankle_roll_link"),
                      r_hip=b("right_hip_roll_link"), r_kne=b("right_knee_link"), r_ank=b("right_ankle_roll_link"))
        self.jq, self.lo, self.hi = replay.jq, replay.lo, replay.hi

    def _fk(self, q):
        d = self.d
        d.qpos[:] = 0
        d.qpos[3] = 1.0
        d.qpos[self.jq] = q
        mujoco.mj_kinematics(self.m, d)
        return d

    def _dirs(self, q, side, kind):
        d = self._fk(q)
        B = self.B
        if kind == "arm":
            s = side[0]
            R = d.xmat[B["torso"]].reshape(3, 3)
            a, b, c = d.xpos[B[f"{s}_sho"]], d.xpos[B[f"{s}_elb"]], d.xpos[B[f"{s}_wri"]]
        else:
            s = side[0]
            R = d.xmat[B["pelvis"]].reshape(3, 3)
            a, b, c = d.xpos[B[f"{s}_hip"]], d.xpos[B[f"{s}_kne"]], d.xpos[B[f"{s}_ank"]]
        return np.concatenate([_n(R.T @ (b - a)), _n(R.T @ (c - b))])

    def _waist_rot(self, q):
        d = self._fk(q)
        return d.xmat[self.B["pelvis"]].reshape(3, 3).T @ d.xmat[self.B["torso"]].reshape(3, 3)

    def _solve(self, q, js, err_fn, iters=12, eps=1e-3, reg=0.02, q_ref=None):
        """Damped Gauss-Newton on a few joints with a numeric Jacobian (FK only)."""
        for _ in range(iters):
            e = err_fn(q)
            if np.linalg.norm(e) < 1e-3:
                break
            J = np.zeros((len(e), len(js)))
            for k, j in enumerate(js):
                q2 = q.copy()
                q2[j] += eps
                J[:, k] = (err_fn(q2) - e) / eps
            g = J.T @ e
            if q_ref is not None:                          # stay near Gemini's pose where the data is ambiguous
                g += reg * (q[js] - q_ref[js])
            dq = -np.linalg.solve(J.T @ J + (1e-3 + reg) * np.eye(len(js)), g)
            q[js] = np.clip(q[js] + np.clip(dq, -0.3, 0.3), self.lo[js], self.hi[js])
        return q

    def run(self, traj, w3, t, fps, cam_pitch_deg):
        """Overwrite body joints (and pelvis tilt) in traj with the retargeted posture. Returns traj, info, rel."""
        T = min(len(traj), len(w3))
        P = _smooth(w3[:T, :, :3].astype(float), 1.5)
        V = w3[:T, :, 3]
        e = np.radians(cam_pitch_deg)
        g_C = np.array([0.0, np.cos(e), np.sin(e)])           # gravity in camera coords (phone held upright)
        used = {"arms": 0, "legs": 0}
        rel = []                                               # person pelvis pose relative to the camera per frame
        q_prev = None
        for f in range(T):
            p = P[f]
            if np.abs(p).sum() == 0:
                rel.append(None)
                continue
            mid_hip, mid_sho = (p[L_HIP] + p[R_HIP]) / 2, (p[L_SHO] + p[R_SHO]) / 2
            Rp = _frame(p[L_HIP], p[R_HIP], mid_hip, mid_sho)   # person pelvis frame in camera coords
            Rt = _frame(p[L_SHO], p[R_SHO], mid_hip, mid_sho)   # person torso frame in camera coords
            rel.append(Rp)
            q_g = traj[f, :C.NJ].copy()                          # Gemini's pose for this frame
            q = (q_prev if q_prev is not None else q_g).copy()
            q[12:] = np.where(np.isnan(q[12:]), q_g[12:], q[12:])
            hips_seen = min(V[f, L_HIP], V[f, R_HIP]) > 0.7    # hidden hips (under a table, out of frame) are guesses
            # waist: torso orientation relative to pelvis (only when the pelvis is actually seen)
            if hips_seen:
                R_pt = Rp.T @ Rt
                q = self._solve(q, WAIST, lambda x: (self._waist_rot(x) - R_pt).ravel(), q_ref=q_g, reg=0.05)
            else:
                q[WAIST] = q_g[WAIST]
            # arms in the torso frame, legs in the pelvis frame
            for side, (js, a, b, c, _) in ARM.items():
                vis = V[f, [a, b, c]].min()
                w = float(np.clip((vis - 0.5) / 0.25, 0, 1))
                if w > 0:
                    tgt = np.concatenate([_n(Rt.T @ (p[b] - p[a])), _n(Rt.T @ (p[c] - p[b]))])
                    qa = self._solve(q.copy(), js, lambda x: self._dirs(x, side, "arm") - tgt, q_ref=q_g)
                    q[js] = w * qa[js] + (1 - w) * q_g[js]
                    used["arms"] += w > 0.5
                else:
                    q[js] = q_g[js]
            leg_w = {}
            for side, (js, a, b, c) in LEG.items():
                vis = V[f, [a, b, c]].min()
                w = float(np.clip((vis - 0.6) / 0.25, 0, 1))
                leg_w[side] = w
                if w > 0:
                    tgt = np.concatenate([_n(Rp.T @ (p[b] - p[a])), _n(Rp.T @ (p[c] - p[b]))])
                    ql = self._solve(q.copy(), js, lambda x: self._dirs(x, side, "leg") - tgt, q_ref=q_g)
                    q[js] = w * ql[js] + (1 - w) * q_g[js]
                    used["legs"] += w > 0.5
                else:
                    q[js] = q_g[js]
            q_prev = q.copy()
            traj[f, :C.NJ] = q
            # pelvis tilt relative to gravity (bending over, leaning); heading stays Gemini's (world path)
            if hips_seen:
                g_P = Rp.T @ g_C
                pitch = np.arctan2(g_P[0], -g_P[2]) if g_P[2] < 0 else traj[f, C.NJ + 1]
                roll = np.arctan2(-g_P[1], -g_P[2]) if g_P[2] < 0 else traj[f, C.NJ + 2]
                pitch, roll = float(np.clip(pitch, -0.3, 1.4)), float(np.clip(roll, -0.4, 0.4))
                if min(V[f, L_KNE], V[f, R_KNE]) < 0.5:     # legs hidden (e.g. under a table): hips are partly guessed
                    g0 = traj[f, C.NJ + 1]
                    pitch = float(np.clip(pitch, g0 - 0.5, g0 + 0.5))
                dp, dr = pitch - traj[f, C.NJ + 1], roll - traj[f, C.NJ + 2]
                # legs that are NOT tracked (hidden) keep their world angle: counter-rotate the hips by the tilt change,
                # otherwise a seated person leaning forward would stand up out of the chair
                for side, (js, *_r) in LEG.items():
                    k = 1.0 - leg_w.get(side, 0.0)
                    q[js[0]] -= k * dp                     # hip pitch
                    q[js[1]] -= k * dr                     # hip roll
                traj[f, :C.NJ] = q
                traj[f, C.NJ + 1] = pitch
                traj[f, C.NJ + 2] = roll
        # light smoothing of the retargeted joints and tilt
        traj[:T, :C.NJ + 3] = _smooth(traj[:T, :C.NJ + 3], 1.0)
        info = dict(frames=T, arm_frames_from_video=used["arms"] // 2, leg_frames_from_video=used["legs"] // 2)
        return traj, info, rel


def camera_from_perspective(rel, w2, robot_pelvis_pose, robot_torso_len, fovy, W, H, fps):
    """Camera that sees the robot's pelvis exactly as the phone saw the person's: same relative orientation
    (from the camera-aligned 3D landmarks), position from where the hips are in the image and how big the torso is
    (using the ROBOT's torso length, so the robot fills the frame like the person did).
    robot_pelvis_pose: list of (R_WP, p_WP) per frame. Returns dict(static=(pos, look), path=[...], spread_m)."""
    t = np.tan(np.radians(fovy) / 2)
    aspect = W / H
    poses = []
    for f, Rp in enumerate(rel):
        if Rp is None or f >= len(robot_pelvis_pose) or f >= len(w2):
            continue
        k = w2[f]
        if min(k[6, 2], k[7, 2], k[0, 2], k[1, 2]) < 0.5:     # hips + shoulders visible in 2D
            continue
        hip2d = (k[6, :2] + k[7, :2]) / 2
        sho2d = (k[0, :2] + k[1, :2]) / 2
        up_C = Rp[:, 2]                                         # torso up direction in camera coords
        l2d = np.linalg.norm((sho2d - hip2d) * np.array([aspect, 1.0])) * 2 * t   # image-plane length / depth
        proj = np.linalg.norm(up_C[:2]) * robot_torso_len         # foreshortened robot torso length
        if l2d < 1e-3 or proj < 1e-3:
            continue
        z = proj / l2d
        X = (hip2d[0] - 0.5) * 2 * z * t * aspect
        Yd = (hip2d[1] - 0.5) * 2 * z * t
        t_C = np.array([X, Yd, z])                              # robot pelvis in camera coords
        R_WP, p_WP = robot_pelvis_pose[f]
        R_WC = R_WP @ Rp.T                                      # camera axes in world (x right, y down, z forward)
        p_WC = p_WP - R_WC @ t_C
        poses.append((f / fps, p_WC, R_WC[:, 2]))
    if not poses:
        return None
    Pm = np.array([p for _, p, _ in poses])
    Fm = _n(np.array([fw for _, _, fw in poses]))
    pos, fwd = np.median(Pm, 0), _n(np.median(Fm, 0))
    spread = float(np.median(np.linalg.norm(Pm - pos, axis=1)))
    out = dict(static=(pos, pos + fwd * 2.0), path=[], spread_m=round(spread, 2))
    if False:                                                   # moving camera disabled: always one static camera
        sm_p, sm_f = _smooth(Pm, 6.0), _n(_smooth(Fm, 6.0))
        out["path"] = [(tt, p, p + fw * 2.0) for (tt, _, _), p, fw in zip(poses, sm_p, sm_f)]
    return out


def root_from_video(rel, w2, cam_static, robot_torso_len, fovy, W, H, fps, start_xy, T):
    """With the (static) camera known, the person's heading and movement in every frame follow from the video:
    heading = camera orientation x the body orientation relative to the camera; movement = change of the hips'
    3D position seen through the camera. Returns per-frame (yaw, x, y) or None."""
    pos, look = np.asarray(cam_static[0], float), np.asarray(cam_static[1], float)
    z_c = _n(look - pos)
    x_c = _n(np.cross(z_c, np.array([0, 0, 1.0])))
    y_c = np.cross(z_c, x_c)
    R_WC = np.stack([x_c, y_c, z_c], 1)
    t = np.tan(np.radians(fovy) / 2)
    aspect = W / H
    yaw, xy, ok = np.full(T, np.nan), np.full((T, 2), np.nan), np.zeros(T, bool)
    for f in range(min(T, len(rel), len(w2))):
        Rp = rel[f]
        if Rp is None:
            continue
        R_WP = R_WC @ Rp
        yaw[f] = np.arctan2(R_WP[1, 0], R_WP[0, 0])
        k = w2[f]
        if min(k[6, 2], k[7, 2], k[0, 2], k[1, 2]) < 0.5:
            continue
        hip2d, sho2d = (k[6, :2] + k[7, :2]) / 2, (k[0, :2] + k[1, :2]) / 2
        l2d = np.linalg.norm((sho2d - hip2d) * np.array([aspect, 1.0])) * 2 * t
        proj = np.linalg.norm(Rp[:2, 2]) * robot_torso_len
        if l2d < 1e-3 or proj < 1e-3:
            continue
        z = proj / l2d
        t_C = np.array([(hip2d[0] - 0.5) * 2 * z * t * aspect, (hip2d[1] - 0.5) * 2 * z * t, z])
        xy[f] = (pos + R_WC @ t_C)[:2]
        ok[f] = True
    if ok.sum() < 5:
        return None
    idx = np.arange(T)
    good = ~np.isnan(yaw)
    yaw = np.interp(idx, idx[good], np.unwrap(yaw[good]))
    for c in range(2):
        xy[:, c] = np.interp(idx, idx[ok], xy[ok, c])
    yaw = _smooth(yaw, 3.0)
    xy = _smooth(xy, 6.0)
    xy = xy - xy[0] + np.asarray(start_xy)                 # movement relative to the scene's start position
    return np.column_stack([yaw, xy])
