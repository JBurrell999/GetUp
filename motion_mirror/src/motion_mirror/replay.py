"""Stage 1: kinematic replay of Gemini's interpretation on the G1 (no RL, no physics control).

Gemini's interpretation (JSON) -> G1 joint/root trajectory + object trajectory -> MuJoCo render from the
camera Gemini estimated, at the input video's resolution and fps.

Interpretation schema: see mirror.INTERPRET_SCHEMA.
Conventions: person starts at the origin facing +x; "left" = +y. Angles in degrees.
"""
import numpy as np
import mujoco

from . import config as C

ELBOW_STRAIGHT = 1.45
FOOT_R = 0.005


# ---------------- scene ----------------
def _obj_geoms(obj):
    """MJCF geoms for the manipulated object, in its own frame (origin at its base center)."""
    kind = obj.get("type", "box").lower()
    sz = np.array(obj.get("size_m", [0.3, 0.3, 0.3]), float)
    col = 'rgba="0.95 0.55 0.1 1"'
    if "bottle" in kind or "cyl" in kind or "can" in kind or "cup" in kind:
        r, h = max(sz[0], sz[1]) / 2, sz[2]
        return (f'<geom type="cylinder" size="{r:.3f} {h/2*0.8:.3f}" pos="0 0 {h*0.4:.3f}" rgba="0.3 0.6 0.95 0.9"/>'
                f'<geom type="cylinder" size="{r*0.45:.3f} {h*0.1:.3f}" pos="0 0 {h*0.9:.3f}" rgba="0.2 0.3 0.8 1"/>')
    if "chair" in kind:   # frame chair: thin legs + uprights, seat, backrest pad at the top
        d, w, h = sz      # depth, width, total height (backrest top)
        seat = obj.get("seat_height_m", h * 0.55)
        rgb = obj.get("color_rgb") or [0.75, 0.35, 0.3]
        c = f'rgba="{rgb[0]:.2f} {rgb[1]:.2f} {rgb[2]:.2f} 1"'
        metal = 'rgba="0.8 0.8 0.82 1"'
        tr, pad = 0.008, min(0.2, (h - seat) * 0.5)
        g = [f'<geom type="box" size="{d/2:.3f} {w/2:.3f} 0.02" pos="0 0 {seat:.3f}" {c}/>',
             f'<geom type="box" size="0.02 {w/2:.3f} {pad/2:.3f}" pos="{-d/2+0.02:.3f} 0 {h-pad/2:.3f}" {c}/>']
        for sx in (-1, 1):
            for sy in (-1, 1):
                g.append(f'<geom type="cylinder" size="{tr} {seat/2:.3f}" pos="{sx*(d/2-0.02):.3f} {sy*(w/2-0.02):.3f} {seat/2:.3f}" {metal}/>')
        for sy in (-1, 1):  # backrest uprights
            g.append(f'<geom type="cylinder" size="{tr} {(h-seat)/2:.3f}" pos="{-d/2+0.02:.3f} {sy*(w/2-0.03):.3f} {seat+(h-seat)/2:.3f}" {metal}/>')
        return "".join(g)
    return f'<geom type="box" size="{sz[0]/2:.3f} {sz[1]/2:.3f} {sz[2]/2:.3f}" pos="0 0 {sz[2]/2:.3f}" {col}/>'


def build_scene(interp) -> str:
    env = interp.get("environment", {}) or {}
    surf = []
    for i, s in enumerate(env.get("surfaces", []) or []):
        x, y = s.get("forward_m", 1.0), s.get("left_m", 0.0)
        sx, sy, h = s.get("depth_m", 0.6), s.get("width_m", 0.8), s.get("height_m", 0.75)
        if (s.get("kind") or "table") == "solid":     # counter, stool, block
            surf.append(f'<geom name="surface{i}" type="box" size="{sx/2:.3f} {sy/2:.3f} {h/2:.3f}" pos="{x:.3f} {y:.3f} {h/2:.3f}" '
                        f'rgba="0.55 0.42 0.3 1"/>')
        else:                                          # table: top slab + 4 legs (legs fit underneath)
            surf.append(f'<geom name="surface{i}" type="box" size="{sx/2:.3f} {sy/2:.3f} 0.02" pos="{x:.3f} {y:.3f} {h-0.02:.3f}" '
                        f'rgba="0.25 0.25 0.27 1"/>')
            for lx in (-1, 1):
                for ly in (-1, 1):
                    surf.append(f'<geom type="cylinder" size="0.02 {(h-0.04)/2:.3f}" pos="{x+lx*(sx/2-0.04):.3f} {y+ly*(sy/2-0.04):.3f} '
                                f'{(h-0.04)/2:.3f}" rgba="0.2 0.2 0.2 1"/>')
    obj = interp.get("object")
    body = ""
    if obj:
        body = f'<body name="object" mocap="true">{_obj_geoms(obj)}</body>'
    src = (C.SCENE_XML.parent / "scene_with_hands.xml").read_text()
    src = src.replace('<geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>',
                      '<geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>\n' + "\n".join(surf) + body)
    return src


# ---------------- pose mapping ----------------
def _arm(a, side):
    f, ab = np.radians(a.get("shoulder_flexion_deg", 5)), np.radians(a.get("shoulder_abduction_deg", 8))
    e, r = np.radians(a.get("elbow_flexion_deg", 15)), np.radians(a.get("shoulder_external_rotation_deg", 0))
    s = 1 if side == "left" else -1
    return np.array([-f, s * ab, s * r, ELBOW_STRAIGHT - e, 0.0, 0.0, 0.0])


def keyframe_to_pose(kf):
    """Human keyframe -> (G1 joints[29], root yaw, root pitch, root roll, root xy)."""
    q = np.zeros(C.NJ)
    tr = kf.get("trunk", {})
    lean = np.radians(tr.get("lean_forward_deg", 0.0))
    side = np.radians(tr.get("side_bend_left_deg", 0.0))
    twist = np.radians(tr.get("twist_left_deg", 0.0))
    waist_p = np.clip(lean * 0.25, -0.1, 0.5)          # a bit of the lean in the waist, the rest at the hips
    p = lean - waist_p                                  # pelvis pitch
    q[12], q[13], q[14] = np.clip(twist, -1.0, 1.0), np.clip(side * 0.5, -0.5, 0.5), waist_p
    for s, off in (("left", 0), ("right", 6)):
        leg = kf.get(f"{s}_leg", {})
        th = np.radians(leg.get("thigh_forward_deg", 0.0))      # thigh angle from vertical, knee forward > 0
        k = np.radians(leg.get("knee_flexion_deg", 5.0))
        ab = np.radians(leg.get("hip_abduction_deg", 0.0))
        q[off + 0] = -th - p
        q[off + 1] = ab if s == "left" else -ab
        q[off + 3] = k
        q[off + 4] = th - k                                      # keep the foot flat
    q[15:22] = _arm(kf.get("left_arm", {}), "left")
    q[22:29] = _arm(kf.get("right_arm", {}), "right")
    pel = kf.get("pelvis", {})
    yaw = np.radians(pel.get("facing_left_deg", 0.0))
    xy = np.array([pel.get("forward_m", 0.0), pel.get("left_m", 0.0)])
    roll = side * 0.5
    return q, yaw, p, roll, xy


def _quat(yaw, pitch, roll):
    qz, qy, qx, out = np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)
    mujoco.mju_axisAngle2Quat(qz, np.array([0, 0, 1.0]), yaw)
    mujoco.mju_axisAngle2Quat(qy, np.array([0, 1.0, 0]), pitch)
    mujoco.mju_axisAngle2Quat(qx, np.array([1.0, 0, 0]), roll)
    tmp = np.zeros(4)
    mujoco.mju_mulQuat(tmp, qz, qy)
    mujoco.mju_mulQuat(out, tmp, qx)
    return out


def _smooth(s):
    return s ** 3 * (10 - 15 * s + 6 * s ** 2)


def _pchip(x, y, xq):
    """Shape-preserving cubic (PCHIP) interpolation, per column: smooth, passes through every keyframe,
    keeps moving through intermediate keyframes, never overshoots."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) == 1:
        return np.repeat(y[:1], len(xq), 0)
    h = np.diff(x)[:, None]
    dlt = np.diff(y, axis=0) / h
    m = np.zeros_like(y)
    m[0], m[-1] = dlt[0], dlt[-1]
    for k in range(1, len(x) - 1):
        a, b = dlt[k - 1], dlt[k]
        w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
        same = (a * b) > 0
        m[k] = np.where(same, (w1 + w2) / (w1 / np.where(same, a, 1) + w2 / np.where(same, b, 1)), 0.0)
    # ease in/out at the very start and end
    m[0] = 0
    m[-1] = 0
    xq = np.clip(xq, x[0], x[-1])
    i = np.clip(np.searchsorted(x, xq) - 1, 0, len(x) - 2)
    hh = (x[i + 1] - x[i])[:, None]
    tt = ((xq - x[i])[:, None]) / hh
    h00, h10 = 2 * tt**3 - 3 * tt**2 + 1, tt**3 - 2 * tt**2 + tt
    h01, h11 = -2 * tt**3 + 3 * tt**2, tt**3 - tt**2
    return h00 * y[i] + h10 * hh * m[i] + h01 * y[i + 1] + h11 * hh * m[i + 1]


def _gauss_smooth(Y, sigma):
    """Light temporal smoothing along axis 0 (edge-padded)."""
    if sigma <= 0:
        return Y
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    P = np.pad(Y, ((r, r), (0, 0)), mode="edge")
    return np.stack([np.convolve(P[:, c], k, mode="valid") for c in range(Y.shape[1])], 1)


def interpolate(kfs, t):
    """Keyframes -> smooth per-frame (joints[29], yaw, pitch, roll, xy[2]) as one array [T, 34]."""
    kfs = sorted(kfs, key=lambda k: k["t"])
    times = np.array([k["t"] for k in kfs], float)
    times = np.maximum.accumulate(times + np.arange(len(times)) * 1e-4)   # strictly increasing
    rows = []
    for k in kfs:
        q, yaw, p, r, xy = keyframe_to_pose(k)
        rows.append(np.concatenate([q, [yaw, p, r], xy]))
    return _pchip(times, np.array(rows), np.asarray(t, float))


def _split(row):
    return row[:C.NJ].copy(), float(row[C.NJ]), float(row[C.NJ + 1]), float(row[C.NJ + 2]), row[C.NJ + 3:C.NJ + 5].copy()


# ---------------- replay ----------------
HANDS_SCENE = C.SCENE_XML.parent / "scene_with_hands.xml"
# Dex3 hand, in the wrist_yaw_link frame: fingers along +x, spread along z, thumb opposes toward -y (left) / +y (right),
# so the palm faces -y on the left hand and +y on the right hand.
GRIP_X = 0.125
HAND_J = ["thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"]
FINGER_CLOSE = {"left": np.array([0.0, 0.3, 1.4, -1.4, -1.5, -1.4, -1.5]),
                "right": np.array([0.0, -0.3, -1.4, 1.4, 1.5, 1.4, 1.5])}
CHAINS = [(1, 2), (3, 4), (5, 6)]   # thumb, middle, index
PALM_SIGN = {"left": -1.0, "right": 1.0}


class Replay:
    def __init__(self, interp, width=848, height=480):
        self.interp = interp
        xml = C.SCENE_XML.parent / "_replay_scene.xml"
        xml.write_text(build_scene(interp))
        self.m = mujoco.MjModel.from_xml_path(str(xml))
        self.m.vis.global_.offwidth = max(self.m.vis.global_.offwidth, width)
        self.m.vis.global_.offheight = max(self.m.vis.global_.offheight, height)
        self.d = mujoco.MjData(self.m)
        self.W, self.H = width, height
        m = self.m
        b = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.jq = np.array([m.jnt_qposadr[jid(n + "_joint")] for n in C.JOINTS])
        self.jv = np.array([m.jnt_dofadr[jid(n + "_joint")] for n in C.JOINTS])
        self.lo = np.array([m.jnt_range[jid(n + "_joint"), 0] for n in C.JOINTS])
        self.hi = np.array([m.jnt_range[jid(n + "_joint"), 1] for n in C.JOINTS])
        self.hq = {s: np.array([m.jnt_qposadr[jid(f"{s}_hand_{n}_joint")] for n in HAND_J]) for s in ("left", "right")}
        self.chain_bodies = {s: [{b(f"{s}_hand_{HAND_J[i]}_link") for i in ch} for ch in CHAINS] for s in ("left", "right")}
        self.hand = {"left": b("left_wrist_yaw_link"), "right": b("right_wrist_yaw_link")}
        self.feet = [b("left_ankle_roll_link"), b("right_ankle_roll_link")]
        self.foot_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in self.feet and m.geom_contype[g]]
        self.obj_body = b("object")
        self.obj_mocap = m.body_mocapid[self.obj_body] if self.obj_body >= 0 else -1
        self.obj_geoms = set(np.where(m.geom_bodyid == self.obj_body)[0]) if self.obj_body >= 0 else set()
        self.fingers = {"left": np.zeros(7), "right": np.zeros(7)}
        surfs = (interp.get("environment", {}) or {}).get("surfaces", []) or []
        self.env_names = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"surface{i}"): (sf.get("name") or "table")
                          for i, sf in enumerate(surfs)}
        self.self_ok = self._self_contact_whitelist()
        for g in range(m.ngeom):   # unnamed world geoms (table legs) count as "table"
            if m.geom_bodyid[g] == 0 and g not in self.env_names and m.geom(g).name != "floor":
                self.env_names[g] = "table leg"

    def _resolve_collisions(self, traj, t, grasps, fps, iters=12, rounds=2):
        """Kinematic clean-up: push penetrating arm parts out of each other / the torso / the table / the floor along
        the contact normal (Jacobian step on the arm joints), smoothed over time. No physics, a few ms per frame."""
        m, d = self.m, self.d
        ARMS = {"left": list(range(15, 22)), "right": list(range(22, 29))}
        jp = np.zeros((3, m.nv))
        robot_bodies = set(range(1, m.nbody)) - {self.obj_body}
        fixed_frames = 0
        for _ in range(rounds):
            corr = np.zeros((len(t), C.NJ))
            for f in range(len(t)):
                q, yaw, pitch, roll, xy = _split(traj[f])
                root = (yaw, pitch, roll, xy)
                busy = {h for gr in grasps if gr["g"] - 3 <= f < gr["r"] for h in gr["hands"]}
                arm = [j for side, js in ARMS.items() if side not in busy for j in js]   # never move a holding arm
                if not arm:
                    continue
                dofs = self.jv[arm]
                q0 = q.copy()
                for _i in range(iters):
                    self._set(q, *root)
                    mujoco.mj_collision(m, d)
                    dq = np.zeros(len(arm))
                    hit = False
                    for k in range(d.ncon):
                        c = d.contact[k]
                        if c.dist > -0.004:
                            continue
                        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
                        if c.geom1 in self.obj_geoms or c.geom2 in self.obj_geoms:
                            continue
                        if b1 in robot_bodies and b2 in robot_bodies and (min(b1, b2), max(b1, b2)) in self.self_ok:
                            continue
                        if (b1 == 0 or b2 == 0) and any(b in self.feet for b in (b1, b2)):
                            continue
                        n = c.frame[:3]                                     # normal from geom1 to geom2
                        for b, sgn in ((b1, -1.0), (b2, 1.0)):
                            if b not in robot_bodies:
                                continue
                            mujoco.mj_comPos(m, d)
                            mujoco.mj_jac(m, d, jp, None, c.pos, b)
                            J = jp[:, dofs]
                            if not J.any():
                                continue
                            dq += J.T @ (sgn * n * (-c.dist + 0.005)) / (np.sum(J * J) + 1e-3) * 1.5
                            hit = True
                    if not hit:
                        break
                    q[arm] = np.clip(q[arm] + np.clip(dq, -0.1, 0.1), self.lo[arm], self.hi[arm])
                corr[f] = q - q0
                if np.abs(corr[f]).max() > 1e-4:
                    fixed_frames += 1
            if not np.abs(corr).any():
                break
            traj[:, :C.NJ] += _gauss_smooth(corr, 0.05 * fps) * 1.2
        if not fixed_frames:
            return traj, None
        return traj, [f"arm/hand collisions with the body, the other arm, the table or the floor were pushed apart "
                      f"automatically in {fixed_frames} frames (kinematics)"]

    def _arm_in_body(self):
        m, d = self.m, self.d
        mujoco.mj_collision(m, d)
        for k in range(d.ncon):
            c = d.contact[k]
            if c.dist > -0.005:
                continue
            p1, p2 = self._part(m.geom_bodyid[c.geom1]), self._part(m.geom_bodyid[c.geom2])
            if {"torso", "pelvis"} & {p1, p2} and any(x in p1 + p2 for x in ("hand", "forearm")):
                return True
        return False

    def _self_contact_whitelist(self):
        m = self.m
        def chain(b):
            out = [b]
            while b > 0:
                b = m.body_parentid[b]
                out.append(b)
            return out
        ok = set()
        robot = [b for b in range(1, m.nbody) if b != self.obj_body]
        ch = {b: chain(b) for b in robot}
        for i, a in enumerate(robot):
            for b in robot[i + 1:]:
                ca, cb = ch[a], ch[b]
                common = next(x for x in ca if x in cb)
                if ca.index(common) + cb.index(common) <= 3:          # close relatives in the kinematic tree
                    ok.add((min(a, b), max(a, b)))
        d = mujoco.MjData(m)                                         # pairs already touching when standing neutrally
        d.qpos[3], d.qpos[2] = 1.0, 0.8
        d.qpos[self.jq] = C.DEFAULT_Q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_collision(m, d)
        for i in range(d.ncon):
            c = d.contact[i]
            a, b = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if a > 0 and b > 0:
                ok.add((min(a, b), max(a, b)))
        return ok

    # ---- posing ----
    def _set(self, q, yaw, pitch, roll, xy):
        d = self.d
        d.qpos[self.jq] = np.clip(q, self.lo, self.hi)
        for s in ("left", "right"):
            d.qpos[self.hq[s]] = self.fingers[s]
        d.qpos[3:7] = _quat(yaw, pitch, roll)
        d.qpos[0:2] = xy
        d.qpos[2] = 1.0
        mujoco.mj_kinematics(self.m, d)
        low = min(d.geom_xpos[g][2] for g in self.foot_geoms) - FOOT_R
        d.qpos[2] = 1.0 - low                           # lowest foot point on the floor
        mujoco.mj_kinematics(self.m, d)

    def _set_obj(self, pos, quat):
        if self.obj_mocap >= 0:
            self.d.mocap_pos[self.obj_mocap] = pos
            self.d.mocap_quat[self.obj_mocap] = quat
            mujoco.mj_kinematics(self.m, self.d)

    def grip_point(self, side, c_w):
        h = self.hand[side]
        return self.d.xpos[h] + self.d.xmat[h].reshape(3, 3) @ c_w

    # ---- grasp geometry (what a person does: palm on the object, oriented to it) ----
    def _grip_specs(self, obj, pos, quat, sz, hands, yaw):
        """For each hand: (target point, target wrist rotation, grip center in the wrist frame)."""
        kind = obj.get("type", "box").lower()
        style = obj.get("grasp_style") or ("top" if "chair" in kind else "side")
        Ro = np.zeros(9)
        mujoco.mju_quat2Mat(Ro, quat)
        Ro = Ro.reshape(3, 3)
        fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        up = np.array([0, 0, 1.0])
        frac = obj.get("grasp_height_frac", 0.5)
        specs = {}
        for side in hands:
            sg = PALM_SIGN[side]
            if "chair" in kind:           # overhand grip on the top of the backrest
                d_, w_, h_ = sz
                ys = (0.28 * w_) * (1 if side == "left" else -1) if len(hands) == 2 else 0.0
                bar = pos + Ro @ np.array([-d_ / 2 + 0.02, ys, h_ - 0.03])
                f = Ro[:, 0] if Ro[:, 0] @ fwd >= 0 else -Ro[:, 0]
                f = f - (f @ up) * up
                f /= np.linalg.norm(f)
                x_w, y_w = f, (up if side == "left" else -up)    # palm (sg*y_w) faces down
                c_w = np.array([GRIP_X, sg * (0.02 + 0.012), 0.0])
                tgt = bar
            elif len(hands) == 2:         # box: palms pressed on the left/right faces
                lat = np.cross(up, fwd)
                face = pos + Ro @ np.array([0, 0, sz[2] * frac]) + lat * (sz[1] / 2) * (1 if side == "left" else -1)
                x_w, y_w = fwd, lat
                c_w = np.array([GRIP_X - 0.02, sg * 0.012, 0.0])
                tgt = face
            else:                         # bottle / cup: side grip around it, thumb up
                axis = pos + Ro @ np.array([0, 0, sz[2] * frac])
                hdir = axis - self.grip_point(side, np.zeros(3))
                hdir[2] = 0
                hdir = hdir / (np.linalg.norm(hdir) + 1e-9)
                r = max(sz[0], sz[1]) / 2
                x_w = hdir
                y_w = np.cross(up, x_w)
                c_w = np.array([GRIP_X - 0.02, sg * (r + 0.022), 0.0])
                tgt = axis
                if style == "top":
                    x_w, y_w = hdir, (up if side == "left" else -up)
            z_w = np.cross(x_w, y_w)
            specs[side] = (tgt, np.stack([x_w, y_w, z_w], 1), c_w)
        return specs

    def _ik6(self, q, root, side, tgt, Rt, c_w, iters=120):
        """7-DoF arm IK: grip center to tgt and wrist orientation to Rt (kinematics only)."""
        js = list(range(15, 22)) if side == "left" else list(range(22, 29))
        dofs = self.jv[js]
        jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        ep = er = np.ones(3)
        for _ in range(iters):
            self._set(q, *root)
            h = self.hand[side]
            Rw = self.d.xmat[h].reshape(3, 3)
            p = self.d.xpos[h] + Rw @ c_w
            ep = tgt - p
            er = 0.5 * sum(np.cross(Rw[:, i], Rt[:, i]) for i in range(3))
            if np.linalg.norm(ep) < 0.003 and np.linalg.norm(er) < 0.03:
                break
            mujoco.mj_comPos(self.m, self.d)
            mujoco.mj_jac(self.m, self.d, jp, jr, p, h)
            J = np.vstack([jp[:, dofs], 0.4 * jr[:, dofs]])
            e = np.concatenate([ep, 0.4 * er])
            dq = J.T @ np.linalg.solve(J @ J.T + 2e-3 * np.eye(6), e)
            q[js] = np.clip(q[js] + np.clip(dq, -0.15, 0.15), self.lo[js], self.hi[js])
        self._set(q, *root)
        return q, float(np.linalg.norm(ep)), float(np.degrees(np.linalg.norm(er) * 2))

    def _obj_pen(self):
        """Deepest penetration of each finger link into the object (link body id -> depth m)."""
        mujoco.mj_collision(self.m, self.d)
        pen = {}
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            o = c.geom2 if c.geom1 in self.obj_geoms else c.geom1 if c.geom2 in self.obj_geoms else None
            if o is None:
                continue
            bo = self.m.geom_bodyid[o]
            pen[bo] = max(pen.get(bo, 0.0), -c.dist)
        return pen

    def _close_fingers(self, q, root, side):
        """Curl the fingers around the object like a real grasp: each finger keeps closing until its tip segment
        touches the object; a base segment only stops if it presses deeper than where it started."""
        f = np.zeros(7)
        self.fingers[side] = f
        self._set(q, *root)
        base = self._obj_pen()
        prox_done, tip_done = [False] * 3, [False] * 3
        close = FINGER_CLOSE[side]
        steps = 40
        for _ in range(steps):
            for ci, (j0, j1) in enumerate(CHAINS):
                if not prox_done[ci]:
                    f[j0] += close[j0] / steps
                if not tip_done[ci]:
                    f[j1] += close[j1] / steps
            self.fingers[side] = f
            self._set(q, *root)
            pen = self._obj_pen()
            for ci, (j0, j1) in enumerate(CHAINS):
                b0 = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_hand_{HAND_J[j0]}_link")
                b1 = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_hand_{HAND_J[j1]}_link")
                if pen.get(b1, 0) > 0.001 and not tip_done[ci]:
                    tip_done[ci] = prox_done[ci] = True
                    f[j0] -= close[j0] / steps          # back off one step: touching, not inside
                    f[j1] -= close[j1] / steps
                elif pen.get(b0, 0) > base.get(b0, 0) + 0.004:
                    prox_done[ci] = True
            if all(tip_done):
                break
        self.fingers[side] = f
        return f.copy(), sum(tip_done)

    # ---- main ----
    def run(self, fps=30.0, video_path=None, smooth_s=0.06, blend_in=0.7, blend_out=0.8, finger_s=0.3):
        it = self.interp
        dur = float(it.get("duration_s") or max(k["t"] for k in it["keyframes"]) + 0.5)
        t = np.arange(0, dur, 1.0 / fps)
        T = len(t)
        traj = interpolate(it["keyframes"], t)                     # smooth spline through Gemini's keyframes
        obj = it.get("object")
        events = sorted(it.get("contacts", []) or [], key=lambda c: c["t_grasp"])
        tel = dict(grasps=[])
        grasps = []
        if obj:
            st = obj.get("start", {})
            obj_pos = np.array([st.get("forward_m", 0.6), st.get("left_m", 0.0), st.get("on_surface_height_m", 0.0)])
            obj_quat = _quat(np.radians(st.get("yaw_deg", 0.0)), 0, 0)
            sz = np.array(obj.get("size_m", [0.3, 0.3, 0.3]))
            self._set_obj(obj_pos, obj_quat)
            # ---- pass 1: grasp poses (exact palm placement + orientation + finger closure) ----
            for ev in events[:1] if len(events) else []:
                hands = ["left", "right"] if ev.get("hand") == "both" else [ev.get("hand", "right")]
                g = int(np.clip(round(ev["t_grasp"] * fps), 0, T - 1))
                r = int(round(ev["t_release"] * fps)) if ev.get("t_release") is not None else T + 10
                q, yaw, pitch, roll, xy = _split(traj[g])
                root = (yaw, pitch, roll, xy)
                self.fingers = {"left": np.zeros(7), "right": np.zeros(7)}
                self._set(q, *root)
                specs = self._grip_specs(obj, obj_pos, obj_quat, sz, hands, yaw)
                info = []
                for _ in range(2):          # two sweeps so both arms settle with each other's pose
                    for h in hands:
                        q, pe, re_ = self._ik6(q, root, h, *specs[h])
                info = []
                closed = {}
                for h in hands:
                    _, pe, re_ = self._ik6(q, root, h, *specs[h], iters=1)
                    closed[h], nf = self._close_fingers(q, root, h)
                    info.append(dict(hand=h, grip_pos_err_m=round(pe, 3), grip_rot_err_deg=round(re_, 1), fingers_touching=nf))
                delta = q - traj[g, :C.NJ]
                grasps.append(dict(g=g, r=r, hands=hands, delta=delta, closed=closed, c_w={h: specs[h][2] for h in hands}))
                bad = any(x["grip_pos_err_m"] > 0.03 for x in info)
                tel["grasps"].append(dict(t=round(float(t[g]), 2), hands=hands, detail=info,
                                          note="arm cannot reach the object here: move the body closer / lean / squat" if bad else "ok"))
                # blend the arm correction in before the grasp; keep it while carrying with two hands; fade after
                for f in range(T):
                    dt = t[f] - t[g]
                    if -blend_in <= dt <= 0:
                        w = _smooth(1 + dt / blend_in)
                    elif dt > 0 and len(hands) == 2 and f < r:
                        w = 1.0
                    elif dt > 0:
                        t_ref = t[g] if len(hands) == 1 else t[min(r, T - 1)]
                        w = 1 - _smooth(np.clip((t[f] - t_ref) / blend_out, 0, 1))
                    else:
                        continue
                    traj[f, :C.NJ] += w * delta
                end = obj.get("end")
                if end and r < T:
                    self._place(obj, end, sz, hands, q, root, obj_pos, obj_quat, closed, traj, t, r, blend_in, blend_out, tel)
        traj = _gauss_smooth(traj, smooth_s * fps)
        traj, fixed = self._resolve_collisions(traj, t, grasps, fps)
        if fixed:
            tel["auto_fixed"] = fixed

        # ---- pass 2: pose, fingers, carry (closed chain for two hands), render ----
        renderer = None
        if video_path:
            renderer = mujoco.Renderer(self.m, self.H, self.W)
            cam = self._camera()
        frames = []
        coll = {}
        held, rel = None, {}
        nfin = max(int(finger_s * fps), 1)
        for f in range(T):
            q, yaw, pitch, roll, xy = _split(traj[f])
            root = (yaw, pitch, roll, xy)
            # fingers: open -> closed in the last moments before the grasp, open again after release
            self.fingers = {"left": np.zeros(7), "right": np.zeros(7)}
            for gr in grasps:
                for h in gr["hands"]:
                    if gr["g"] - nfin <= f < gr["r"]:
                        w = _smooth(np.clip((f - (gr["g"] - nfin)) / nfin, 0, 1))
                    elif gr["r"] <= f < gr["r"] + nfin:
                        w = 1 - _smooth((f - gr["r"]) / nfin)
                    else:
                        continue
                    self.fingers[h] = w * gr["closed"][h]
            self._set(q, *root)
            if obj:
                for gr in grasps:
                    if f == gr["g"] and held is None:
                        held = gr
                        h0 = gr["hands"][0]
                        Rh = self.d.xmat[self.hand[h0]].reshape(3, 3)
                        Ro = np.zeros(9)
                        mujoco.mju_quat2Mat(Ro, obj_quat)
                        rel = dict(p=Rh.T @ (obj_pos - self.d.xpos[self.hand[h0]]), R=Rh.T @ Ro.reshape(3, 3),
                                   yaw0=yaw, quat0=obj_quat.copy(),
                                   mid_off=obj_pos - np.mean([self.grip_point(h, gr["c_w"][h]) for h in gr["hands"]], 0))
                    if held is gr and f >= gr["r"]:
                        held = None
                        yaw_o = np.arctan2(2 * (obj_quat[0] * obj_quat[3] + obj_quat[1] * obj_quat[2]),
                                           1 - 2 * (obj_quat[2] ** 2 + obj_quat[3] ** 2))
                        obj_quat = _quat(yaw_o, 0, 0)                      # set down upright
                        obj_pos = obj_pos.copy()
                        obj_pos[2] = self._support_height(obj_pos)
                if held is not None:
                    hands = held["hands"]
                    if len(hands) == 1:     # rigidly in the hand (tilts with the wrist, e.g. when drinking)
                        h0 = hands[0]
                        Rh = self.d.xmat[self.hand[h0]].reshape(3, 3)
                        obj_pos = self.d.xpos[self.hand[h0]] + Rh @ rel["p"]
                        mujoco.mju_mat2Quat(obj_quat, (Rh @ rel["R"]).flatten())
                    else:                   # two hands: object upright, turning with the body; hands re-solved onto it
                        dy = yaw - rel["yaw0"]
                        Rz = np.array([[np.cos(dy), -np.sin(dy), 0], [np.sin(dy), np.cos(dy), 0], [0, 0, 1]])
                        obj_pos = np.mean([self.grip_point(h, held["c_w"][h]) for h in hands], 0) + Rz @ rel["mid_off"]
                        qz = np.zeros(4)
                        mujoco.mju_axisAngle2Quat(qz, np.array([0, 0, 1.0]), dy)
                        mujoco.mju_mulQuat(obj_quat, qz, rel["quat0"])
                        obj_pos[2] = max(obj_pos[2], self._support_height(obj_pos))
                        fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
                        for _k in range(8):        # carried object pressed into the body? hold it further out
                            self._set_obj(obj_pos, obj_quat)
                            specs = self._grip_specs(obj, obj_pos, obj_quat, sz, hands, yaw)
                            for h in hands:
                                q, _, _ = self._ik6(q, root, h, *specs[h], iters=25)
                            if not self._arm_in_body():
                                break
                            obj_pos = obj_pos + 0.02 * fwd
                            rel["mid_off"] = rel["mid_off"] + 0.02 * fwd
                    if len(hands) == 1:
                        obj_pos = obj_pos.copy()
                self._set_obj(obj_pos, obj_quat)
            near = any(abs(f - gr["g"]) <= nfin or abs(f - gr["r"]) <= nfin for gr in grasps)
            for pair, depth in self._collisions(f, held is not None, near):
                coll.setdefault(pair, []).append((f, depth))
            if renderer is not None:
                renderer.update_scene(self.d, cam)
                frames.append(renderer.render())
            if f % max(int(fps / 2), 1) == 0:
                smp = dict(t=round(float(t[f]), 1), pelvis=self.d.qpos[:3].round(2).tolist())
                if obj:
                    smp.update(object=np.asarray(obj_pos).round(2).tolist(), held=held is not None)
                tel.setdefault("samples", []).append(smp)
        tel["collisions"] = []
        for (a, b_), hits in coll.items():           # group into time ranges
            hits.sort()
            start = prev = hits[0][0]
            dmax = 0
            for fr, dep in hits + [(10 ** 9, 0)]:
                if fr > prev + 2:
                    tel["collisions"].append(dict(t_start=round(float(t[start]), 2), t_end=round(float(t[min(prev, T - 1)]), 2),
                                                  what=f"{a} inside {b_}" if b_ in ("floor", "table") or b_ in self.env_names.values() else f"{a} / {b_}",
                                                  max_depth_cm=round(dmax * 100, 1)))
                    start, dmax = fr, 0
                prev, dmax = fr, max(dmax, dep)
        tel["collisions"].sort(key=lambda c: c["t_start"])
        if video_path:
            import imageio
            imageio.mimsave(video_path, frames, fps=fps, macro_block_size=8)
        return tel

    def _place(self, obj, end, sz, hands, q_g, root_g, obj_pos, obj_quat, closed, traj, t, r, blend_in, blend_out, tel):
        """Solve the arm(s) at release so the carried object lands exactly where Gemini says it ends up."""
        self.fingers = {"left": np.zeros(7), "right": np.zeros(7), **closed}
        self._set(q_g, *root_g)
        h0 = hands[0]
        Rh0 = self.d.xmat[self.hand[h0]].reshape(3, 3).copy()
        Ro0 = np.zeros(9)
        mujoco.mju_quat2Mat(Ro0, obj_quat)
        rel_p, rel_R = Rh0.T @ (obj_pos - self.d.xpos[self.hand[h0]]), Rh0.T @ Ro0.reshape(3, 3)
        tgt = np.array([end.get("forward_m", obj_pos[0]), end.get("left_m", obj_pos[1]), 0.0])
        tgt[2] = end.get("on_surface_height_m", self._support_height(tgt))
        tq = _quat(np.radians(end.get("yaw_deg", 0.0)), 0, 0)
        q, yaw, pitch, roll, xy = _split(traj[r])
        root = (yaw, pitch, roll, xy)
        errs = []
        if len(hands) == 1:
            Rt = np.zeros(9)
            mujoco.mju_quat2Mat(Rt, tq)
            Rw = Rt.reshape(3, 3) @ rel_R.T              # wrist orientation that holds the object upright
            pw = tgt - Rw @ rel_p                        # wrist position that puts the object on the target
            q, pe, _ = self._ik6(q, root, h0, pw, Rw, np.zeros(3))
            errs.append(pe)
        else:
            specs = self._grip_specs(obj, tgt, tq, sz, hands, yaw)
            for _ in range(2):
                for h in hands:
                    q, pe, _ = self._ik6(q, root, h, *specs[h])
            errs = [self._ik6(q, root, h, *specs[h], iters=1)[1] for h in hands]
        delta = q - traj[r, :C.NJ]
        for f in range(len(t)):
            dt = t[f] - t[r]
            if -blend_in <= dt <= 0:
                w = _smooth(1 + dt / blend_in)
            elif 0 < dt <= blend_out:
                w = 1 - _smooth(dt / blend_out)
            else:
                continue
            traj[f, :C.NJ] += w * delta
        tel.setdefault("placements", []).append(dict(t=round(float(t[r]), 2), target=tgt.round(2).tolist(),
                                                     placement_err_m=round(max(errs), 3),
                                                     note="arm cannot reach the place target: move closer/lean" if max(errs) > 0.03 else "ok"))

    def _collisions(self, f, held, near_grasp):
        """Interpenetrations this frame: robot vs table/floor, robot vs object, object vs table/floor,
        and robot self-collisions (arms through each other / the torso / the legs)."""
        m, d = self.m, self.d
        mujoco.mj_collision(m, d)
        out = []
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist > -0.01:
                continue
            names = []
            for g in (c.geom1, c.geom2):
                b = m.geom_bodyid[g]
                if g in self.obj_geoms:
                    names.append("object")
                elif b == 0:
                    names.append(self.env_names.get(g, "floor"))
                else:
                    names.append(self._part(b))
            a, b_ = sorted(names)
            env = {"floor"} | set(self.env_names.values())
            if a in env and b_ in env:
                continue                                              # env-env
            ba, bb = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if a not in env and b_ not in env and "object" not in (a, b_):
                # robot self-collision (arm through arm / torso / leg ...), minus pairs that always touch
                if (min(ba, bb), max(ba, bb)) in self.self_ok or a == b_:
                    continue
            if "foot" in a + b_ and "floor" in (a, b_):
                continue                                              # standing on the floor
            if "object" in (a, b_):
                other = b_ if a == "object" else a
                if other in env and not held:
                    continue                                          # object resting on its surface
                if "hand" in other and (held or near_grasp):
                    continue                                          # the grip itself
            out.append(((a, b_), -c.dist))
        return out

    def _part(self, b):
        n = self.m.body(b).name
        for k, v in (("hand", "hand"), ("wrist", "hand"), ("elbow", "forearm"), ("shoulder", "upper arm"),
                     ("ankle", "foot"), ("knee", "shin"), ("hip", "thigh"), ("torso", "torso"), ("pelvis", "pelvis"),
                     ("waist", "torso")):
            if k in n:
                side = "left " if n.startswith("left") else "right " if n.startswith("right") else ""
                return side + v
        return n

    def still(self):
        """Phase 1: the scene with the robot in the person's start pose, from the estimated camera (+ collisions)."""
        kf = sorted(self.interp["keyframes"], key=lambda k: k["t"])[0]
        q, yaw, p, r, xy = keyframe_to_pose(kf)
        self.fingers = {"left": np.zeros(7), "right": np.zeros(7)}
        self._set(q, yaw, p, r, xy)
        obj = self.interp.get("object")
        if obj:
            st = obj.get("start", {})
            self._set_obj(np.array([st.get("forward_m", 0.6), st.get("left_m", 0.0), st.get("on_surface_height_m", 0.0)]),
                          _quat(np.radians(st.get("yaw_deg", 0.0)), 0, 0))
        coll = [dict(what=f"{a} inside {b}" if b in ("floor",) or b in self.env_names.values() else f"{a} / {b}",
                     depth_cm=round(dep * 100, 1)) for (a, b), dep in self._collisions(0, False, False)]
        renderer = mujoco.Renderer(self.m, self.H, self.W)
        renderer.update_scene(self.d, self._camera())
        return renderer.render(), coll

    def _support_height(self, pos):
        best = 0.0
        for s in (self.interp.get("environment", {}) or {}).get("surfaces", []) or []:
            if abs(pos[0] - s.get("forward_m", 0)) <= s.get("depth_m", 0.6) / 2 and \
               abs(pos[1] - s.get("left_m", 0)) <= s.get("width_m", 0.8) / 2:
                best = max(best, s.get("height_m", 0.75))
        return best

    def _camera(self):
        c = self.interp.get("camera", {}) or {}
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        r = c.get("horizontal_distance_m", 3.0)
        h = c.get("height_m", 1.4)
        look = np.array([c.get("look_at_forward_m", 0.0), c.get("look_at_left_m", 0.0), c.get("look_at_height_m", 0.8)])
        az = c.get("azimuth_deg", 0.0)   # 0 = camera in front of the person, 90 = on the person's left
        cam.lookat[:] = look
        cam.azimuth = 180.0 + az
        cam.elevation = -np.degrees(np.arctan2(h - look[2], r))
        cam.distance = float(np.hypot(r, h - look[2]))
        self.m.vis.global_.fovy = c.get("vertical_fov_deg", 50.0)
        return cam

