"""Stage 1: kinematic replay of Gemini's interpretation on the G1 (no RL, no physics control).

Gemini's interpretation (JSON) -> G1 joint/root trajectory + object trajectory -> MuJoCo render from the
camera Gemini estimated, at the input video's resolution and fps.

Interpretation schema: see mirror.INTERPRET_SCHEMA.
Conventions: person starts at the origin facing +x; "left" = +y. Angles in degrees.
"""
import re

import numpy as np
import mujoco

from . import config as C

ELBOW_STRAIGHT = 1.45
FOOT_R = 0.005


# ---------------- scene ----------------
def scale_world(interp, k):
    """Scale everything Gemini placed (surfaces, object, its parts/grasp points, pelvis path, footholds) by k around
    the person's start, so a smaller robot relates to the scene exactly like the person."""
    if abs(k - 1.0) < 1e-6 or interp.get("_world_scaled"):
        return interp
    import copy
    it = copy.deepcopy(interp)
    L = ("forward_m", "left_m", "depth_m", "width_m", "height_m", "on_surface_height_m", "seat_height_m")
    for sf in (it.get("environment") or {}).get("surfaces") or []:
        for f in L:
            if isinstance(sf.get(f), (int, float)):
                sf[f] *= k
    for fz in (it.get("environment") or {}).get("furniture") or []:
        for f in ("forward_m", "left_m"):
            if isinstance(fz.get(f), (int, float)):
                fz[f] *= k
    o = it.get("object")
    if o:
        for key in ("start", "end"):
            for f in L:
                if isinstance((o.get(key) or {}).get(f), (int, float)):
                    o[key][f] *= k
        if o.get("size_m"):
            o["size_m"] = [x * k for x in o["size_m"]]
        for p in o.get("parts") or []:
            p["size"] = [x * k for x in p.get("size") or []]
            p["pos"] = [x * k for x in p.get("pos") or []]
        for gp in (o.get("grasp_points") or {}).values():
            if gp.get("pos") is not None:
                gp["pos"] = [x * k for x in gp["pos"]]
    for kf in it.get("keyframes") or []:
        pel = kf.get("pelvis") or {}
        for f in ("forward_m", "left_m"):
            if isinstance(pel.get(f), (int, float)):
                pel[f] *= k
    for st in it.get("footsteps") or []:
        for f in ("forward_m", "left_m"):
            if isinstance((st.get("land") or {}).get(f), (int, float)):
                st["land"][f] *= k
    it["_world_scaled"] = k
    return it


def _rgba(c, default=(0.8, 0.5, 0.2)):
    c = list(c or default)[:3]
    return f'rgba="{c[0]:.2f} {c[1]:.2f} {c[2]:.2f} 1"'


def _obj_geoms(obj):
    """Any object, as Gemini describes it: a set of primitive parts in the object frame
    (origin = center of its base, z up). Falls back to one box of size_m."""
    parts = obj.get("parts") or [dict(shape="box", size=obj.get("size_m", [0.3, 0.3, 0.3]),
                                      pos=[0, 0, obj.get("size_m", [0.3, 0.3, 0.3])[2] / 2], color_rgb=obj.get("color_rgb"))]
    out = []
    for p in parts:
        shape = (p.get("shape") or "box").lower()
        sz = [max(float(x), 0.004) for x in (p.get("size") or [0.1, 0.1, 0.1])]
        pos = " ".join(f"{float(x):.3f}" for x in (p.get("pos") or [0, 0, 0]))
        eul = p.get("euler_deg")
        ori = f' euler="{" ".join(f"{np.radians(float(x)):.4f}" for x in eul)}"' if eul else ""
        if p.get("quat"):
            ori = f' quat="{" ".join(f"{float(x):.6f}" for x in p["quat"])}"'
        col = f'material="{p["material"]}"' if p.get("material") else _rgba(p.get("color_rgb") or obj.get("color_rgb"))
        if shape == "cylinder":          # size = [diameter, height], axis along z
            g = f'type="cylinder" size="{sz[0]/2:.3f} {sz[-1]/2:.3f}"'
        elif shape == "capsule":         # size = [diameter, length]
            g = f'type="capsule" size="{sz[0]/2:.3f} {max(sz[-1] - sz[0], 0.002)/2:.3f}"'
        elif shape == "sphere":          # size = [diameter]
            g = f'type="sphere" size="{sz[0]/2:.3f}"'
        else:                            # box: size = [x, y, z] full extents
            sz = (sz + sz[-1:] * 3)[:3]
            g = f'type="box" size="{sz[0]/2:.3f} {sz[1]/2:.3f} {sz[2]/2:.3f}"'
        out.append(f'<geom {g} pos="{pos}"{ori} {col} density="300"/>')
    return "".join(out)


def _obj_bbox(obj):
    """Object-frame bounding box (lo, hi) of its parts."""
    parts = obj.get("parts")
    if not parts:
        sz = np.asarray(obj.get("size_m", [0.3, 0.3, 0.3]), float)
        return np.array([-sz[0] / 2, -sz[1] / 2, 0.0]), np.array([sz[0] / 2, sz[1] / 2, sz[2]])
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for p in parts:
        c = np.asarray(p.get("pos") or [0, 0, 0], float)
        sz = np.asarray(p.get("size") or [0.1], float)
        shape = (p.get("shape") or "box").lower()
        if shape in ("cylinder", "capsule"):
            h = np.array([sz[0] / 2, sz[0] / 2, sz[-1] / 2])
        elif shape == "sphere":
            h = np.full(3, sz[0] / 2)
        else:
            h = np.resize(sz, 3) / 2
        if p.get("euler_deg"):
            h = np.full(3, np.linalg.norm(h))                     # rotated part: conservative
        lo, hi = np.minimum(lo, c - h), np.maximum(hi, c + h)
    return lo, hi


def build_scene(interp, room=None) -> str:
    env = interp.get("environment", {}) or {}
    surf = []
    for i, s in enumerate(env.get("surfaces", []) or []):
        x, y = s.get("forward_m", 1.0), s.get("left_m", 0.0)
        sx, sy, h = s.get("depth_m", 0.6), s.get("width_m", 0.8), s.get("height_m", 0.75)
        if s.get("hidden"):                            # matched to real room furniture: collide, don't draw
            solid = (s.get("kind") or "table") == "solid"
            hz, cz = (h / 2, h / 2) if solid else (0.02, h - 0.02)
            surf.append(f'<geom name="surface{i}" type="box" size="{sx/2:.3f} {sy/2:.3f} {hz:.3f}" pos="{x:.3f} {y:.3f} {cz:.3f}" '
                        f'rgba="0 0 0 0" group="4"/>')
        elif (s.get("kind") or "table") == "solid":     # counter, stool, block
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
    src = C.ROBOTS[C.ROBOT]["scene"].read_text()
    src = re.sub(r'<texture type="skybox"[^>]*/>',
                 '<texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>', src)
    floor = '<geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>'
    if room is not None:                  # known room as the background; our floor stays as an invisible collider
        from .room import room_xml_parts
        r_asset, r_body = room_xml_parts(*room)
        src = re.sub(r'<texture type="skybox"[^>]*/>', "", src)
        src = re.sub(r'<headlight [^>]*/>', '<headlight ambient=".42 .41 .39" diffuse=".85 .85 .8" specular=".1 .1 .1"/>', src)
        src = src.replace('<light pos="1 0 3.5" dir="0 0 -1" directional="true"/>', "")
        src = src.replace("</asset>", r_asset + "</asset>", 1)
        src = src.replace(floor, '<geom name="floor" size="0 0 0.05" type="plane" rgba="0 0 0 0"/>\n' + r_body + "\n" + floor.replace(floor, ""))
        floor = '<geom name="floor" size="0 0 0.05" type="plane" rgba="0 0 0 0"/>'
    src = src.replace(floor, floor + "\n" + "\n".join(surf) + body)
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
MJ_PARENT = [None]
# Dex3 hand, in the wrist_yaw_link frame: fingers along +x, spread along z, thumb opposes toward -y (left) / +y (right),
# so the palm faces -y on the left hand and +y on the right hand.
GRIP_X = 0.125
HAND_J = ["thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"]
FINGER_CLOSE = {"left": np.array([0.0, 0.3, 1.4, -1.4, -1.5, -1.4, -1.5]),
                "right": np.array([0.0, -0.3, -1.4, 1.4, 1.5, 1.4, 1.5])}
CHAINS = [(1, 2), (3, 4), (5, 6)]   # thumb, middle, index
PALM_SIGN = {"left": -1.0, "right": 1.0}


class Replay:
    def __init__(self, interp, width=848, height=480, room=None, human=None):
        room = C.USE_ROOM if room is None else room
        interp = scale_world(interp, getattr(C, "WORLD_SCALE", 1.0))
        self.human = human                    # (w2 [T,12,3], w3 [T,33,4]) from posematch.human_pose(full=True)
        from . import room as R
        hide = set()
        reg = None
        moves = {}
        if room and R.available():
            yaw, spawn, interp, hide = R.register(interp)
            reg = (yaw, spawn)
            moves, item = R.arrange(interp, yaw, spawn)
            if item is not None and interp.get("object"):
                lo0, hi0 = _obj_bbox(interp["object"])
                ratio = (item["hi"] - item["lo"]) / np.maximum(hi0 - lo0, 1e-3)
                if np.any(ratio < 0.75) or np.any(ratio > 1.33):  # sizes differ too much: keep Gemini's shape
                    item = None
            if item is not None and interp.get("object"):     # use the room's own model of this object
                interp = dict(interp)
                o = dict(interp["object"])
                lo0, hi0 = _obj_bbox(o)
                o_yaw = yaw + np.radians((o.get("start") or {}).get("yaw_deg", 0.0))
                o["parts"] = R.item_parts(item, o_yaw, align_to=o.get("parts"))
                lo1, hi1 = _obj_bbox(o)
                gps = {}
                for h, gp in (o.get("grasp_points") or {}).items():   # same relative spot on the real-shaped object
                    if gp.get("pos") is not None:
                        u = (np.asarray(gp["pos"], float) - lo0) / np.maximum(hi0 - lo0, 1e-3)
                        gps[h] = dict(gp, pos=(lo1 + u * (hi1 - lo1)).tolist())
                if gps:
                    o["grasp_points"] = gps
                o["_room_item"] = item["name"]
                interp["object"] = o
                rm, aabb, _ = R.room_model()
                hide = set(hide) | set(item["geoms"])
        self.interp = interp
        xml = C.ROBOTS[C.ROBOT]["scene"].parent / "_replay_scene.xml"
        xml.write_text(build_scene(interp, reg))
        self.m = mujoco.MjModel.from_xml_path(str(xml))
        MJ_PARENT[0] = self.m.body_parentid
        self.m.vis.global_.offwidth = max(self.m.vis.global_.offwidth, width)
        self.m.vis.global_.offheight = max(self.m.vis.global_.offheight, height)
        self.d = mujoco.MjData(self.m)
        self.W, self.H = width, height
        m = self.m
        if reg is not None:               # room geoms are added in file order inside body "room": hide the ones in the way
            rb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "room")
            sub = [g for g in range(m.ngeom) if self._in_subtree(m.geom_bodyid[g], rb)]
            for k, g in enumerate(sub):
                if k in hide:
                    m.geom_rgba[g, 3] = 0.0
                if k in moves:                       # furniture rearranged to where it is in the video
                    m.geom_pos[g], m.geom_quat[g] = moves[k]
        b = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.jq = np.array([m.jnt_qposadr[jid(n + "_joint")] for n in C.JOINTS])
        self.jv = np.array([m.jnt_dofadr[jid(n + "_joint")] for n in C.JOINTS])
        self.lo = np.array([m.jnt_range[jid(n + "_joint"), 0] for n in C.JOINTS])
        self.hi = np.array([m.jnt_range[jid(n + "_joint"), 1] for n in C.JOINTS])
        self.has_hands = C.ROBOTS[C.ROBOT]["hands"]
        if self.has_hands:
            self.hq = {s: np.array([m.jnt_qposadr[jid(f"{s}_hand_{n}_joint")] for n in HAND_J]) for s in ("left", "right")}
            self.chain_bodies = {s: [{b(f"{s}_hand_{HAND_J[i]}_link") for i in ch} for ch in CHAINS] for s in ("left", "right")}
        else:
            self.hq = {s: np.zeros(0, int) for s in ("left", "right")}
            self.chain_bodies = {s: [set(), set(), set()] for s in ("left", "right")}
        self.hand = {"left": b("left_wrist_yaw_link"), "right": b("right_wrist_yaw_link")}
        self.feet = [b(n) for n in C.ROBOTS[C.ROBOT]["feet"]]
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

    # ---- walking: footholds fixed in the world, swing arcs, exact leg IK ----
    def _palm_at(self, traj, f, hands):
        q, yaw, pitch, roll, xy = _split(traj[min(f, len(traj) - 1)])
        self.fingers = {"left": np.zeros(7), "right": np.zeros(7)}
        self._set(q, yaw, pitch, roll, xy)
        return np.mean([self.grip_point(h, np.array([GRIP_X - 0.025, PALM_SIGN[h] * 0.03, 0.0])) for h in hands], 0)

    def _object_from_hands(self, obj, obj_pos, obj_quat, ev, traj, t, fps, tel, max_shift=2.0):
        """The object is in the person's hand at the grasp and released where the hand puts it down. With the body
        retargeted from the video, the robot's hand is at the person's hand position, so place the object there."""
        hands = ["left", "right"] if ev.get("hand") == "both" else [ev.get("hand", "right")]
        Ro = np.zeros(9)
        mujoco.mju_quat2Mat(Ro, obj_quat)
        Ro = Ro.reshape(3, 3)
        lo, hi = _obj_bbox(obj)
        gps = obj.get("grasp_points") or {}
        gp = np.mean([np.asarray(gps[h]["pos"], float) if (gps.get(h) or {}).get("pos") is not None
                      else np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2] + 0.5 * (hi[2] - lo[2])]) for h in hands], 0)
        g = int(np.clip(round(ev["t_grasp"] * fps), 0, len(t) - 1))
        palm = self._palm_at(traj, g, hands)
        want = palm - Ro @ gp
        shift = want[:2] - obj_pos[:2]
        if np.linalg.norm(shift) > max_shift:
            shift *= max_shift / np.linalg.norm(shift)
        new = obj_pos.copy()
        new[:2] += shift
        sup = self._support_height(new)
        new[2] = sup if abs(obj_pos[2] - sup) < 0.15 else obj_pos[2]      # keep its support level (floor stays floor)
        info = dict(start_shift_m=round(float(np.linalg.norm(shift)), 3))
        self.obj_from_hands = dict(start=new.copy(), end=None, implied_z=float(want[2]))
        end = None
        if ev.get("t_release") is not None:
            r0 = int(np.clip(round(ev["t_release"] * fps), 0, len(t) - 1))
            win = range(max(g + 1, r0 - int(0.6 * fps)), min(len(t), r0 + int(0.3 * fps)))
            palms = {f: self._palm_at(traj, f, hands) for f in win}
            def score(f):                                    # the object touches down where the hand is low and still
                p0, p1 = palms.get(f - 1, palms[f]), palms.get(f + 1, palms[f])
                return palms[f][2] + 0.5 * np.linalg.norm(p1 - p0) * fps / 2
            r = min(win, key=score) if len(win) else r0
            pr = palms.get(r, self._palm_at(traj, r, hands))
            e0 = obj.get("end") or {}
            yaw_e = np.radians(e0.get("yaw_deg", (obj.get("start") or {}).get("yaw_deg", 0.0)))
            Re = np.array([[np.cos(yaw_e), -np.sin(yaw_e), 0], [np.sin(yaw_e), np.cos(yaw_e), 0], [0, 0, 1]])
            ep = pr - Re @ gp
            self.obj_from_hands["end"] = ep.copy()
            sup = self._support_height(ep)
            want_h = e0.get("on_surface_height_m", sup)
            if abs(sup - want_h) < 0.15:                     # there is a surface at the intended height under the hand
                ep[2] = sup
                end = dict(forward_m=float(ep[0]), left_m=float(ep[1]), on_surface_height_m=float(ep[2]), yaw_deg=float(np.degrees(yaw_e)))
                info["end_from_hand"] = [round(float(x), 2) for x in ep]
                info["release_frame_t"] = round(float(t[r]), 2)
            else:
                info["end_from_hand"] = "kept Gemini's (no surface under the hand)"
        tel["object_from_video"] = info
        return new, end

    def _foot_pose(self, side):
        b = self.feet[0] if side == "left" else self.feet[1]
        return self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3).copy()

    def _seated_fraction(self):
        """Share of Gemini's keyframes where the person sits (thighs well forward, knees bent)."""
        kfs = self.interp.get("keyframes") or []
        if not kfs:
            return 0.0
        n = sum(1 for k in kfs if (k.get("left_leg") or {}).get("thigh_forward_deg", 0) > 60
                and (k.get("left_leg") or {}).get("knee_flexion_deg", 0) > 60)
        return n / len(kfs)

    def _auto_footsteps(self, traj, t, fps):
        """General fallback when Gemini lists no steps but the pelvis travels: alternate feet along the path."""
        xy = traj[:, C.NJ + 3:C.NJ + 5]
        yaw = traj[:, C.NJ]
        steps, foot, last = [], "left", 0
        moving_since = None
        for f in range(1, len(t)):
            d_lin = np.linalg.norm(xy[f] - xy[last])
            d_yaw = abs(yaw[f] - yaw[last])
            if d_lin > 0.14 or d_yaw > 0.35:        # ~ every 0.14 m of travel (or a 20 deg turn) one foot steps
                t1 = t[f]
                steps.append(dict(foot=foot, t_lift=max(t1 - 0.4, t[last]), t_land=t1))
                foot = "right" if foot == "left" else "left"
                last = f
        if steps:                                   # bring the trailing foot alongside at the end
            steps.append(dict(foot=foot, t_lift=steps[-1]["t_land"], t_land=min(steps[-1]["t_land"] + 0.4, t[-1])))
        return steps

    def _gait(self, traj, t, footsteps, fps, lift_h=0.06, iters=25):
        T = len(t)
        steps = sorted([dict(s) for s in footsteps if s.get("foot") in ("left", "right")], key=lambda s: s["t_lift"])
        auto = False
        seated = self._seated_fraction() > 0.5
        if not steps and not seated:                    # seated people don't walk: never invent steps for them
            steps = self._auto_footsteps(traj, t, fps)
            auto = bool(steps)
        # where the feet are at the start (Gemini's first pose), and the flat-foot height / orientation
        q, yaw, pitch, roll, xy = _split(traj[0])
        self._set(q, yaw, pitch, roll, xy)
        hold = {s: self._foot_pose(s) for s in ("left", "right")}
        z0 = {s: hold[s][0][2] for s in hold}
        Rz = lambda a: np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])
        R_flat = {s: Rz(-yaw) @ hold[s][1] for s in hold}            # foot orientation relative to heading
        yaw_hold = {s: yaw for s in hold}
        # per-frame target for each foot
        tgt = {s: [None] * T for s in ("left", "right")}
        for s in ("left", "right"):
            mine = [st for st in steps if st["foot"] == s]
            pos, fy = hold[s][0].copy(), yaw_hold[s]
            k = 0
            for f in range(T):
                while k < len(mine) and t[f] > mine[k]["t_land"]:
                    st = mine[k]
                    pos, fy = self._landing(st, traj, t, s, z0[s])
                    k += 1
                if k < len(mine) and mine[k]["t_lift"] <= t[f] <= mine[k]["t_land"]:
                    st = mine[k]
                    land, ly = self._landing(st, traj, t, s, z0[s])
                    u = _smooth(np.clip((t[f] - st["t_lift"]) / max(st["t_land"] - st["t_lift"], 1e-3), 0, 1))
                    p = pos + u * (land - pos)
                    p[2] = z0[s] + lift_h * np.sin(np.pi * u)
                    tgt[s][f] = (p, Rz(fy + u * (ly - fy)) @ R_flat[s])
                else:
                    tgt[s][f] = (pos.copy(), Rz(fy) @ R_flat[s])
        if getattr(self, "_video_legs", False):        # legs tracked in the video: feet follow them, pinned only in stance
            tgt = self._video_foot_targets(traj, t, fps, z0, R_flat)
            steps, auto = [], False
        # leg IK every frame (6 joints per leg), pelvis from the (smoothed) body motion
        legs = {"left": list(range(0, 6)), "right": list(range(6, 12))}
        jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        worst = 0.0
        for f in range(T):
            q, yaw, pitch, roll, xy = _split(traj[f])
            root = (yaw, pitch, roll, xy)
            # pelvis height: from Gemini's legs (squat / sit / stand) with the planted foot on the floor
            self._set(q, *root)
            zr = self.d.qpos[2]
            for s in ("left", "right"):
                js, dofs = legs[s], self.jv[legs[s]]
                b = self.feet[0] if s == "left" else self.feet[1]
                p_t, R_t = tgt[s][f]
                for _ in range(iters):
                    self._set_fixed_z(q, root, zr)
                    pc, Rc = self.d.xpos[b], self.d.xmat[b].reshape(3, 3)
                    ep = p_t - pc
                    er = 0.5 * sum(np.cross(Rc[:, i], R_t[:, i]) for i in range(3))
                    if np.linalg.norm(ep) < 0.002 and np.linalg.norm(er) < 0.02:
                        break
                    mujoco.mj_comPos(self.m, self.d)
                    mujoco.mj_jac(self.m, self.d, jp, jr, pc, b)
                    J = np.vstack([jp[:, dofs], 0.3 * jr[:, dofs]])
                    e = np.concatenate([ep, 0.3 * er])
                    dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), e)
                    q[js] = np.clip(q[js] + np.clip(dq, -0.2, 0.2), self.lo[js], self.hi[js])
                worst = max(worst, float(np.linalg.norm(ep)))
            traj[f, :12] = q[:12]
        self._fixed_z = None
        info = dict(steps=len(steps), auto_generated=auto, worst_foot_err_m=round(worst, 3))
        if worst > 0.05:
            info["note"] = "a foot could not reach its foothold: the pelvis is too far from the feet (step or move the body)"
        return traj, info

    def _video_foot_targets(self, traj, t, fps, z0, R_flat, v_stance=0.3, h_stance=0.04):
        """Foot targets from the tracked legs: where a foot is (nearly) still and low it is in stance -> pinned at its
        median spot, flat on the floor (no sliding); otherwise it follows the tracked motion, offset smoothly so it
        leaves and arrives exactly at its stance spots."""
        T = len(t)
        Rz = lambda a: np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])
        Pf = {s: np.zeros((T, 3)) for s in ("left", "right")}
        Yf = {s: np.zeros(T) for s in ("left", "right")}
        for f in range(T):
            q, yaw, pitch, roll, xy = _split(traj[f])
            self._set(q, yaw, pitch, roll, xy)
            for s in ("left", "right"):
                p, R = self._foot_pose(s)
                Pf[s][f] = p
                Yf[s][f] = np.arctan2(R[1, 0], R[0, 0])
        tgt = {}
        for s in ("left", "right"):
            P = Pf[s]
            k = max(int(0.1 * fps), 1)
            Ps = np.array([P[max(0, i - k):i + k + 1].mean(0) for i in range(T)])
            v = np.r_[0, np.linalg.norm(np.diff(Ps[:, :2], axis=0), axis=1) * fps]
            stance = (v < v_stance) & (P[:, 2] - P[:, 2].min() < h_stance)
            # stance runs -> pinned spots
            runs, i = [], 0
            while i < T:
                if stance[i]:
                    j = i
                    while j + 1 < T and stance[j + 1]:
                        j += 1
                    runs.append((i, j, np.r_[np.median(P[i:j + 1, :2], 0), z0[s]], float(np.median(Yf[s][i:j + 1]))))
                    i = j + 1
                else:
                    i += 1
            out = [None] * T
            for (a, b, spot, fy) in runs:
                for f in range(a, b + 1):
                    out[f] = (spot.copy(), Rz(fy) @ R_flat[s])
            # swings between stance runs: tracked motion + an offset that blends from one spot to the next
            prev = None
            bounds = [(-1, -1, None, None)] + runs + [(T, T, None, None)]
            for (a0, b0, sp0, y0), (a1, b1, sp1, y1) in zip(bounds[:-1], bounds[1:]):
                lo, hi = b0 + 1, a1 - 1
                if hi < lo:
                    continue
                d0 = (sp0 - P[b0]) if sp0 is not None else np.zeros(3)
                d1 = (sp1 - P[a1]) if sp1 is not None else d0
                for f in range(lo, hi + 1):
                    u = _smooth((f - lo + 1) / (hi - lo + 2))
                    p = P[f] + (1 - u) * d0 + u * d1
                    p[2] = max(p[2], z0[s])
                    out[f] = (p, Rz(Yf[s][f]) @ R_flat[s])
            tgt[s] = out
        return tgt

    def _landing(self, st, traj, t, side, z):
        """Where a step lands: Gemini's foothold if given, else under the hip at landing time."""
        f = int(np.clip(round(st["t_land"] / (t[1] - t[0])), 0, len(t) - 1))
        _, yaw, _, _, xy = _split(traj[f])
        if st.get("land") and st["land"].get("forward_m") is not None:
            p = np.array([st["land"]["forward_m"], st["land"].get("left_m", 0.0), z])
        else:
            lat = np.array([-np.sin(yaw), np.cos(yaw)])
            fwd = np.array([np.cos(yaw), np.sin(yaw)])
            p2 = xy + lat * (0.1 if side == "left" else -0.1) + fwd * 0.03
            p = np.array([p2[0], p2[1], z])
        return p, yaw

    def _set_fixed_z(self, q, root, z):
        d = self.d
        yaw, pitch, roll, xy = root
        d.qpos[self.jq] = np.clip(q, self.lo, self.hi)
        if self.has_hands:
            for s in ("left", "right"):
                d.qpos[self.hq[s]] = self.fingers[s]
        d.qpos[3:7] = _quat(yaw, pitch, roll)
        d.qpos[0:2] = xy
        d.qpos[2] = z
        mujoco.mj_kinematics(self.m, d)

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
        """During a two-hand carry: is the object, or a hand/forearm, pushed into the robot's body?"""
        m, d = self.m, self.d
        mujoco.mj_collision(m, d)
        for k in range(d.ncon):
            c = d.contact[k]
            if c.dist > -0.005:
                continue
            g1, g2 = c.geom1, c.geom2
            p1, p2 = self._part(m.geom_bodyid[g1]), self._part(m.geom_bodyid[g2])
            if g1 in self.obj_geoms or g2 in self.obj_geoms:
                other = p2 if g1 in self.obj_geoms else p1
                if m.geom_bodyid[g2 if g1 in self.obj_geoms else g1] > 0 and "hand" not in other:
                    return True                                   # object inside torso / legs / arms
                continue
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

    @staticmethod
    def _in_subtree(b, root):
        while b > 0:
            if b == root:
                return True
            b = MJ_PARENT[0][b]
        return False

    # ---- posing ----
    def _set(self, q, yaw, pitch, roll, xy):
        d = self.d
        d.qpos[self.jq] = np.clip(q, self.lo, self.hi)
        if self.has_hands:
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
        """For each hand: (target point, target wrist rotation, grip center in the wrist frame).

        General: Gemini says where on the object each hand touches (a surface point in the object frame) and which way
        the palm faces there. The palm center goes on that point, the palm faces that direction, and the fingers then
        curl until they touch. Without a grasp point, the hand goes to the nearest side of the object's bounding box."""
        Ro = np.zeros(9)
        mujoco.mju_quat2Mat(Ro, quat)
        Ro = Ro.reshape(3, 3)
        fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        up = np.array([0, 0, 1.0])
        lat = np.cross(up, fwd)                                   # robot's left
        lo, hi = _obj_bbox(obj)
        center = pos + Ro @ ((lo + hi) / 2)
        gps = obj.get("grasp_points") or {}
        pts = {}
        for side in hands:
            gp = gps.get(side) or {}
            if gp.get("pos") is not None:
                p = pos + Ro @ np.asarray(gp["pos"], float)
                n = Ro @ np.asarray(gp.get("palm_dir") or (center - p), float)
            else:                                                 # nearest side of the bounding box
                sh = self.grip_point(side, np.zeros(3))
                to = center - sh
                to[2] = 0
                to /= np.linalg.norm(to) + 1e-9
                half = (hi - lo) / 2
                p = center - to * (np.abs(Ro.T @ to) @ half)
                if len(hands) == 2:
                    p = center + lat * (1 if side == "left" else -1) * (np.abs(Ro.T @ lat) @ half)
                n = center - p
            n = n / (np.linalg.norm(n) + 1e-9)
            f = Ro @ np.asarray(gp["finger_dir"], float) if gp.get("finger_dir") else None
            pts[side] = [p, n, f]
        # anatomy: the left hand can't hold the point that is to the right of the right hand's point
        if len(hands) == 2 and (pts["left"][0] - pts["right"][0]) @ lat < 0:
            pts["left"], pts["right"] = pts["right"], pts["left"]
        specs = {}
        for side in hands:
            p, n, f = pts[side]
            sg = PALM_SIGN[side]
            if f is None or abs(f @ n) > 0.9:                     # fingers point forward/away from the body
                f = fwd - (fwd @ n) * n
                if np.linalg.norm(f) < 0.3:
                    f = up - (up @ n) * n
            f = f - (f @ n) * n
            f /= np.linalg.norm(f) + 1e-9
            y_w = sg * n                                          # the palm faces +n
            x_w = f
            z_w = np.cross(x_w, y_w)
            specs[side] = (p, np.stack([x_w, y_w, z_w], 1), np.array([GRIP_X - 0.025, sg * 0.012, 0.0]))
        return specs

    def _ik6(self, q, root, side, tgt, Rt, c_w, iters=120):
        """7-DoF arm IK: grip center to tgt and wrist orientation to Rt (kinematics only). Position has priority:
        if the hand can't get there with the requested orientation, the orientation is relaxed step by step."""
        for w_rot in (0.4, 0.12, 0.03):
            q, pe, re_ = self._ik6_w(q, root, side, tgt, Rt, c_w, iters, w_rot)
            if pe < 0.02:
                break
        return q, pe, re_

    def _ik6_w(self, q, root, side, tgt, Rt, c_w, iters, w_rot):
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
            J = np.vstack([jp[:, dofs], w_rot * jr[:, dofs]])
            e = np.concatenate([ep, w_rot * er])
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
        touches the object; a base segment only stops if it presses deeper than where it started.
        Robots without fingers: palm grasp, nothing to close."""
        if not self.has_hands:
            return np.zeros(0), 1
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
        tel = dict(grasps=[])
        traj = interpolate(it["keyframes"], t)                     # smooth spline through Gemini's keyframes
        traj = _gauss_smooth(traj, smooth_s * fps)
        ro = it.get("_root_override")
        if ro is not None:                    # heading + movement measured from the video through the static camera
            ro = np.asarray(ro, float)
            n_ = min(len(ro), T)
            traj[:n_, C.NJ] = ro[:n_, 0]
            traj[:n_, C.NJ + 3:C.NJ + 5] = ro[:n_, 1:3]
        if self.human is not None:            # posture from the video itself, per frame (Gemini fills occluded limbs)
            from .retarget import Retargeter
            c0 = self._cam_params(0)
            pitch = np.degrees(np.arctan2(c0.get("height_m", 1.4) - c0.get("look_at_height_m", 0.8),
                                          c0.get("horizontal_distance_m", 3.0)))
            traj, rinfo, self.rel = Retargeter(self).run(traj, self.human[1], t, fps, pitch)
            tel["retarget"] = rinfo
            self._video_legs = rinfo["leg_frames_from_video"] > 0.3 * len(t)
        traj, gait_info = self._gait(traj, t, it.get("footsteps") or [], fps)   # planted feet, swing arcs, leg IK
        tel["gait"] = gait_info
        obj = it.get("object")
        events = sorted(it.get("contacts", []) or [], key=lambda c: c["t_grasp"])
        grasps = []
        if obj:
            st = obj.get("start", {})
            obj_pos = np.array([st.get("forward_m", 0.6), st.get("left_m", 0.0), st.get("on_surface_height_m", 0.0)])
            obj_quat = _quat(np.radians(st.get("yaw_deg", 0.0)), 0, 0)
            _lo, _hi = _obj_bbox(obj)
            sz = _hi - _lo
            if self.human is not None and events and getattr(C, "OBJECT_FROM_HANDS", True):
                ev_se = dict(events[0], t_release=events[-1].get("t_release"))
                obj_pos, obj_end = self._object_from_hands(obj, obj_pos, obj_quat, ev_se, traj, t, fps, tel)
                if obj_end is not None:
                    obj = dict(obj, end=obj_end)
                    it["object"] = obj
            self._set_obj(obj_pos, obj_quat)
            # ---- pass 1: grasp poses (exact palm placement + orientation + finger closure) ----
            for ei, ev in enumerate(events):
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
                # pre-grasp: same hand orientation, palm backed off along its normal -> the hand comes in from outside
                q_pre = q.copy()
                for h in hands:
                    tgt_h, Rt_h, c_h = specs[h]
                    back = tgt_h - Rt_h[:, 1] * PALM_SIGN[h] * 0.07
                    q_pre, _, _ = self._ik6(q_pre, root, h, back, Rt_h, c_h, iters=60)
                delta_pre = q_pre - traj[g, :C.NJ]
                grasps.append(dict(g=g, r=r, hands=hands, delta=delta, closed=closed, c_w={h: specs[h][2] for h in hands}))
                bad = any(x["grip_pos_err_m"] > 0.03 for x in info)
                tel["grasps"].append(dict(t=round(float(t[g]), 2), hands=hands, detail=info,
                                          note="arm cannot reach the object here: move the body closer / lean / squat" if bad else "ok"))
                # blend the arm correction in before the grasp; keep it while carrying with two hands; fade after
                k_pre = 0.3                                            # last 0.3 s: from the pre-grasp onto the object
                for f in range(T):
                    dt = t[f] - t[g]
                    if -blend_in <= dt < -k_pre:
                        w = _smooth((dt + blend_in) / (blend_in - k_pre))
                        traj[f, :C.NJ] += w * delta_pre
                        continue
                    if -k_pre <= dt <= 0:
                        u = _smooth((dt + k_pre) / k_pre)
                        traj[f, :C.NJ] += (1 - u) * delta_pre + u * delta
                        continue
                    if False:
                        pass
                    elif dt > 0 and len(hands) == 2 and f < r:
                        w = 1.0
                    elif dt > 0:
                        t_ref = t[g] if len(hands) == 1 else t[min(r, T - 1)]
                        w = 1 - _smooth(np.clip((t[f] - t_ref) / blend_out, 0, 1))
                    else:
                        continue
                    traj[f, :C.NJ] += w * delta
                end = obj.get("end")
                last = ei == len(events) - 1
                if end and r < T and last:
                    self._place(obj, end, sz, hands, q, root, obj_pos, obj_quat, closed, traj, t, r, blend_in, blend_out, tel)
                elif r < T:                    # released mid-clip: it stays where the hand put it, for the next grasp
                    self.fingers = {"left": np.zeros(7), "right": np.zeros(7), **closed}
                    self._set(q, *root)
                    h0 = hands[0]
                    Rg, pg = self.d.xmat[self.hand[h0]].reshape(3, 3).copy(), self.d.xpos[self.hand[h0]].copy()
                    rel_p = Rg.T @ (obj_pos - pg)
                    qr, yr, pr_, rr, xyr = _split(traj[min(r, T - 1)])
                    self._set(qr, yr, pr_, rr, xyr)
                    Rr, prr = self.d.xmat[self.hand[h0]].reshape(3, 3), self.d.xpos[self.hand[h0]]
                    new = prr + Rr @ rel_p
                    new[2] = self._support_height(new) if abs(self._support_height(new) - obj_pos[2]) < 0.2 else obj_pos[2]
                    obj_pos = new
                    self._set_obj(obj_pos, obj_quat)
        traj[:, 15:29] = _gauss_smooth(traj[:, 15:29], smooth_s * fps)   # arms only: the legs are exact now
        traj, fixed = self._resolve_collisions(traj, t, grasps, fps)
        if fixed:
            tel["auto_fixed"] = fixed

        # ---- pass 2: pose, fingers, carry (closed chain for two hands), render ----
        renderer = None
        if video_path:
            renderer = mujoco.Renderer(self.m, self.H, self.W)
            cam = self._camera(0.0)
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
            cam = self._camera(float(t[f]))
            self._record_pose2d(f)
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

    ROBOT_H, HUMAN_H = 1.32, 1.70

    def _cam_params(self, t=0.0):
        """Camera at time t: Gemini's static camera, or its camera_path (pans / handheld motion), interpolated."""
        c = dict(self.interp.get("camera", {}) or {})
        path = sorted(self.interp.get("camera_path") or [], key=lambda k: k.get("t", 0))
        keys = ("horizontal_distance_m", "height_m", "azimuth_deg", "vertical_fov_deg",
                "look_at_forward_m", "look_at_left_m", "look_at_height_m")
        if len(path) >= 1:
            for k in keys:
                pts = [(p["t"], p[k]) for p in path if p.get(k) is not None]
                if pts:
                    ts, vs = zip(*pts)
                    c[k] = float(np.interp(t, ts, vs))
        return c

    def _camera(self, t=0.0):
        fit = self.interp.get("_camera_fit")
        if fit:                                  # camera fitted to the person's 2D pose (posematch.fit_camera_path)
            fovy = float(fit["fovy"])
            if fit.get("path"):
                ts = [p[0] for p in fit["path"]]
                P = np.array([np.concatenate([p[1], p[2]]) for p in fit["path"]])
                q = np.array([np.interp(t, ts, P[:, k]) for k in range(6)])
            else:
                q = np.concatenate(fit["static"])
            pos, look = q[:3], q[3:]
            d = look - pos
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = look
            cam.distance = float(np.linalg.norm(d))
            cam.azimuth = float(np.degrees(np.arctan2(d[1], d[0])))
            cam.elevation = float(np.degrees(np.arcsin(d[2] / np.linalg.norm(d))))
            self.m.vis.global_.fovy = fovy
            self._cam_geom = (pos, d / np.linalg.norm(d), fovy)
            return cam
        c = self._cam_params(t)
        r = c.get("horizontal_distance_m", 3.0)
        h = c.get("height_m", 1.4)
        look = np.array([c.get("look_at_forward_m", 0.0), c.get("look_at_left_m", 0.0), c.get("look_at_height_m", 0.8)])
        az = np.radians(c.get("azimuth_deg", 0.0))   # 0 = camera in front of the person, 90 = on the person's left
        pos = look + np.array([r * np.cos(az), r * np.sin(az), 0.0])
        pos[2] = h
        if self.interp.get("camera_scale_match", False):
            # the camera was placed for a ~1.7 m person; scale the rig about the person's start so the 1.3 m robot is
            # framed like the person (same on-screen size and angle)
            k = self.ROBOT_H / self.HUMAN_H
            pos, look = pos * k, look * k
        d = look - pos
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = look
        cam.distance = float(np.linalg.norm(d))
        cam.azimuth = float(np.degrees(np.arctan2(d[1], d[0])))
        cam.elevation = float(np.degrees(np.arcsin(d[2] / np.linalg.norm(d))))
        fovy = c.get("vertical_fov_deg", 50.0)
        self.m.vis.global_.fovy = fovy
        self._cam_geom = (pos, d / np.linalg.norm(d), fovy)
        return cam

    def _record_pose2d(self, f):
        """Robot joints projected through the camera (for the objective pose match) + hard checks."""
        from .posematch import ROBOT_BODY, KEYS, project
        if not hasattr(self, "_kp_ids"):
            self._kp_ids = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, ROBOT_BODY[k]) for k in KEYS]
            self._pelvis = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
            self._torso = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
            self.robot_kp, self.flags = [], {"flipped_frames": 0, "object_above_head_frames": 0}
        if not hasattr(self, "robot_kp3d"):
            self.robot_kp3d, self.pelvis_pose = [], []
        self.pelvis_pose.append((self.d.xmat[self._pelvis].reshape(3, 3).copy(), self.d.xpos[self._pelvis].copy()))
        self.robot_kp3d.append(self.d.xpos[self._kp_ids].copy())
        pos, fwd, fovy = self._cam_geom
        self.robot_kp.append(project(self.d.xpos[self._kp_ids], pos, fwd, np.array([0, 0, 1.0]), fovy, self.W, self.H))
        if self.d.xpos[self._torso][2] < self.d.xpos[self._pelvis][2] - 0.05:
            self.flags["flipped_frames"] += 1                       # torso below the pelvis: body upside down
        if self.obj_mocap >= 0 and self.d.mocap_pos[self.obj_mocap][2] > self.d.xpos[self._torso][2] + 0.55:
            self.flags["object_above_head_frames"] += 1

