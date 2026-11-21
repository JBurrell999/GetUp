"""Put the replay inside a known room (g1_room_mujoco) instead of an empty floor.

General scene registration: Gemini recreates the surfaces it sees in the video (in the person's frame). We find the
room surface that best matches the one the object starts on, and the rigid transform (heading + position) that puts
that surface onto it with the object on top, the person standing clear of the room's furniture. Matched surfaces use
the room's real geometry; unmatched ones stay as Gemini drew them. Room props occupying the object's spot or the
robot's space are hidden. The room itself is background only (it never collides)."""
from functools import lru_cache
from pathlib import Path
import re

import mujoco
import numpy as np

from . import config as C

ROOM_XML = C.ROOT.parent / "g1_room_mujoco" / "Hackathon_G1_Ultra_Detail_V2" / "HACKATHON_V2_VISUAL.xml"
DEFAULT_SPAWN = (np.array([0.0, -5.65]), np.pi / 2)     # open aisle, facing the stage (room's own G1 spawn)


def available():
    return ROOM_XML.exists()


@lru_cache(1)
def room_model():
    m = mujoco.MjModel.from_xml_path(str(ROOM_XML))
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    aabb = []
    for g in range(m.ngeom):
        R = d.geom_xmat[g].reshape(3, 3)
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            aabb.append(None)
            continue
        ext = np.abs(R) @ (m.geom_aabb[g][3:] if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH else _half(m, g))
        c = d.geom_xpos[g]
        aabb.append((c - ext, c + ext))
    tops = []
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX or aabb[g] is None:
            continue
        lo, hi = aabb[g]
        ext = (hi - lo) / 2
        if ext[2] < 0.05 and 0.35 < hi[2] < 1.25 and ext[0] * ext[1] * 4 > 0.15:
            R = d.geom_xmat[g].reshape(3, 3)
            tops.append(dict(geom=g, center=d.geom_xpos[g][:2].copy(), top=float(hi[2]),
                             half=m.geom_size[g][:2].copy(), yaw=float(np.arctan2(R[1, 0], R[0, 0]))))
    return m, aabb, tops


def _half(m, g):
    t, s = m.geom_type[g], m.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        return s.copy()
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        return np.full(3, s[0])
    if t in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
        return np.array([s[0], s[0], s[1] + (s[0] if t == mujoco.mjtGeom.mjGEOM_CAPSULE else 0)])
    return np.full(3, s.max())


def _rot(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def _clear(aabb, tops, spawn, r=0.3, skip=()):
    """How much room furniture overlaps a person-sized column at spawn (0 = clear)."""
    pen = 0.0
    for g, bb in enumerate(aabb):
        if bb is None or g in skip:
            continue
        lo, hi = bb
        if hi[2] < 0.05 or lo[2] > 1.6 or (hi - lo).max() > 6:       # floor slabs, ceiling, walls handled by extent
            continue
        dx = max(lo[0] - spawn[0], 0, spawn[0] - hi[0])
        dy = max(lo[1] - spawn[1], 0, spawn[1] - hi[1])
        if np.hypot(dx, dy) < r:
            pen += r - np.hypot(dx, dy)
    return pen


def register(interp):
    """Returns (yaw, spawn_xy, interp_in_room, hidden_room_geoms). Pure geometry, no Gemini."""
    m, aabb, tops = room_model()
    it = dict(interp)
    env = dict(it.get("environment") or {})
    surfs = [dict(s) for s in env.get("surfaces") or []]
    obj = it.get("object")
    anchor = None
    if obj and surfs:
        st = obj.get("start", {})
        o = np.array([st.get("forward_m", 0.5), st.get("left_m", 0.0)])
        h = st.get("on_surface_height_m", 0.0)
        under = [s for s in surfs if abs(o[0] - s.get("forward_m", 0)) <= s.get("depth_m", 0.6) / 2 + 0.05
                 and abs(o[1] - s.get("left_m", 0)) <= s.get("width_m", 0.8) / 2 + 0.05 and abs(s.get("height_m", 0.75) - h) < 0.2]
        anchor = under[0] if under else None
    if anchor is None and surfs:
        anchor = max(surfs, key=lambda s: s.get("depth_m", 0.6) * s.get("width_m", 0.8))
    best = None
    if anchor is not None and tops:
        a_c = np.array([anchor.get("forward_m", 0.6), anchor.get("left_m", 0.0)])
        a_h = anchor.get("height_m", 0.75)
        a_half = np.array([anchor.get("depth_m", 0.6), anchor.get("width_m", 0.8)]) / 2
        o = np.array([obj["start"].get("forward_m"), obj["start"].get("left_m")]) if obj else a_c
        for tp in tops:
            for k in range(4):
                yaw = tp["yaw"] + k * np.pi / 2                    # person heading in the room
                R = _rot(yaw)
                th = tp["half"] if k % 2 == 0 else tp["half"][::-1] # table half extents in the person frame
                # slide along the table so the object lands on it at the same place relative to the near edge
                rel_o = o - a_c                                     # object relative to the anchor center
                rel_o = np.clip(rel_o, -th + 0.08, th - 0.08)
                # keep the person's distance to the table's near edge the same as in the video
                near = a_c[0] - a_half[0]                           # person -> near edge, along the person's forward
                c_person = np.array([near + th[0], a_c[1] + (o - a_c)[1] - rel_o[1]])
                spawn = tp["center"] - R @ c_person
                clear = _clear(aabb, tops, spawn, skip={tp["geom"]})
                score = 3.0 * abs(tp["top"] - a_h) + 0.5 * np.abs(th - a_half).sum() + 10.0 * clear
                if best is None or score < best[0]:
                    best = (score, yaw, spawn, tp, k, th, c_person)
    if best is None:
        yaw, spawn = DEFAULT_SPAWN[1], DEFAULT_SPAWN[0]
        it["_room"] = dict(matched=None, spawn=spawn.round(2).tolist(), yaw_deg=round(np.degrees(yaw)))
        return yaw, spawn, it, set(_hide(m, aabb, it, yaw, spawn, set()))
    _, yaw, spawn, tp, k, th, c_person = best
    R = _rot(yaw)
    keep_hidden = {tp["geom"]}
    new_surfs = []
    for s in surfs:
        c = np.array([s.get("forward_m", 0), s.get("left_m", 0)])
        if s is anchor:
            s2 = dict(s, forward_m=float(c_person[0]), left_m=float(c_person[1]), depth_m=float(2 * th[0]),
                      width_m=float(2 * th[1]), height_m=tp["top"], kind="table", hidden=True, room_match=True)
            new_surfs.append(s2)
            continue
        w = tp["center"] + R @ (c - np.array([anchor.get("forward_m"), anchor.get("left_m")]) + c_person) - tp["center"]
        world = spawn + R @ c
        match = next((t2 for t2 in tops if np.all(np.abs(_rot(-t2["yaw"]) @ (world - t2["center"])) <= t2["half"] + 0.05)
                      and abs(t2["top"] - s.get("height_m", 0.75)) < 0.12), None)
        if match is not None:
            s = dict(s, height_m=match["top"], hidden=True, room_match=True)
        new_surfs.append(s)
    env["surfaces"] = new_surfs
    it["environment"] = env
    if obj:                                                    # objects that were on the anchor now sit on the room table
        o2 = dict(obj)
        for key in ("start", "end"):
            if o2.get(key) and abs(o2[key].get("on_surface_height_m", 0) - anchor.get("height_m", 0.75)) < 0.2:
                o2[key] = dict(o2[key], on_surface_height_m=tp["top"],
                               forward_m=float(np.clip(o2[key].get("forward_m", 0), c_person[0] - th[0] + 0.06, c_person[0] + th[0] - 0.06)))
        it["object"] = o2
    it["_room"] = dict(matched=m.geom(tp["geom"]).name, spawn=spawn.round(2).tolist(), yaw_deg=round(np.degrees(yaw)))
    return yaw, spawn, it, set(_hide(m, aabb, it, yaw, spawn, keep_hidden))


def _hide(m, aabb, it, yaw, spawn, never):
    """Room geoms in the way: on the object's start/end spot, or where the robot stands."""
    R = _rot(yaw)
    boxes = []
    obj = it.get("object")
    if obj:
        from .replay import _obj_bbox
        lo, hi = _obj_bbox(obj)
        for key in ("start", "end"):
            st = obj.get(key)
            if st:
                c = spawn + R @ np.array([st.get("forward_m", 0), st.get("left_m", 0)])
                r = np.linalg.norm((hi - lo)[:2]) / 2 + 0.05
                z0 = st.get("on_surface_height_m", 0) + 0.005
                boxes.append((c, r, z0, z0 + (hi - lo)[2] + 0.05))
    boxes.append((spawn, 0.35, 0.0, 1.8))                      # the robot's own space at the start
    out = []
    for g, bb in enumerate(aabb):
        if bb is None or g in never:
            continue
        lo, hi = bb
        if (hi - lo).max() > 3:                                # walls, floor, big tables: never hide
            continue
        for c, r, z0, z1 in boxes:
            dx = max(lo[0] - c[0], 0, c[0] - hi[0])
            dy = max(lo[1] - c[1], 0, c[1] - hi[1])
            if np.hypot(dx, dy) < r and hi[2] > z0 and lo[2] < z1:
                out.append(g)
                break
    return out


def room_xml_parts(yaw, spawn):
    """(asset xml, worldbody xml) of the room, transformed into the person frame, background-only."""
    src = ROOM_XML.read_text()
    base = ROOM_XML.parent
    asset = src[src.index("<asset>") + 7:src.index("</asset>")]
    asset = re.sub(r'file="([^"]+)"', lambda mm: f'file="{(base / mm.group(1)).resolve()}"', asset)
    world = src[src.index("<worldbody>") + 11:src.index("</worldbody>")]
    world = re.sub(r"<light [^>]*/>", "", world)            # too many lights breaks the classic renderer's shading
    world = re.sub(r'\s(contype|conaffinity)="[^"]*"', "", world)       # background only: never collides
    world = re.sub(r"<geom ", '<geom contype="0" conaffinity="0" ', world)
    # person frame = R(yaw)^-1 (room - spawn)
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.array([0, 0, 1.0]), -yaw)
    p = -_rot(-yaw) @ spawn
    lights = ('<light pos="0 0 3" dir="0.3 0.2 -1" directional="true" castshadow="false" diffuse=".45 .45 .43"/>'
              '<light pos="0 0 3" dir="-0.4 -0.3 -1" directional="true" castshadow="false" diffuse=".25 .25 .25"/>')
    body = lights + f'<body name="room" pos="{p[0]:.4f} {p[1]:.4f} 0" quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}">{world}</body>'
    return asset, body


# ---------------------------------------------------------------- movable room items (chairs, stools, laptops ...)
KINDS = ("chair", "stool", "table", "laptop", "cup", "bottle", "can", "rucksack", "backpack", "bag", "phone",
         "notebook", "bowl", "drink", "badge", "urn", "snack")
_PART = re.compile(r"_(seat|back|leg|top|lid|screen|base|body|cap|frame|strap|handle|shell|keyboard|pad|rim)(_\d+)*$")


@lru_cache(1)
def room_items():
    """Group the room's named geoms into items (by name prefix) with a kind, center, base height and yaw."""
    m, aabb, tops = room_model()
    groups = {}
    for g in range(m.ngeom):
        n = m.geom(g).name
        if not n or aabb[g] is None:
            continue
        base = _PART.sub("", n)
        kind = next((k for k in KINDS if k in base.lower()), None)
        if kind is None or base.endswith("_top") or "worktable" in base and kind == "table":
            continue
        groups.setdefault(base, dict(name=base, kind="backpack" if kind == "rucksack" else kind, geoms=[]))["geoms"].append(g)
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    items = []
    for it in groups.values():
        los = np.array([aabb[g][0] for g in it["geoms"]])
        his = np.array([aabb[g][1] for g in it["geoms"]])
        lo, hi = los.min(0), his.max(0)
        if (hi - lo).max() > 2.5:                                  # not a movable item
            continue
        main = max(it["geoms"], key=lambda g: np.prod(np.maximum(aabb[g][1] - aabb[g][0], 1e-3)))
        R = d.geom_xmat[main].reshape(3, 3)
        it.update(center=((lo + hi) / 2)[:2], base_z=float(lo[2]), height=float(hi[2] - lo[2]),
                  yaw=float(np.arctan2(R[1, 0], R[0, 0])), lo=lo, hi=hi)
        items.append(it)
    return items


def arrange(interp, yaw, spawn, matched_geom=None):
    """Move room items so the furniture Gemini saw is where it is in the video, and pick a room item to BE the
    manipulated object when the room has one of that kind. Returns (moves {room_geom: (pos, quat)}, object_item)."""
    m, aabb, _ = room_model()
    items = room_items()
    R = _rot(yaw)
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    used, moves = set(), {}
    want = list((interp.get("environment") or {}).get("furniture") or [])
    obj = interp.get("object") or {}
    okind = next((k for k in KINDS if k in (obj.get("name", "") + " " + obj.get("description", "")).lower()), None)
    obj_item = None
    if okind:                                                       # the manipulated object itself, if the room has one
        st = obj.get("start", {})
        tgt = spawn + R @ np.array([st.get("forward_m", 0.6), st.get("left_m", 0.0)])
        cands = [it for it in items if it["kind"] == okind]
        if cands:
            obj_item = min(cands, key=lambda it: np.linalg.norm(it["center"] - tgt))
            used.add(obj_item["name"])
    for fz in want:
        kind = (fz.get("kind") or "").lower()
        cands = [it for it in items if it["kind"] == kind and it["name"] not in used]
        if not cands:
            continue
        tgt = spawn + R @ np.array([fz.get("forward_m", 1.0), fz.get("left_m", 0.0)])
        it = min(cands, key=lambda c: np.linalg.norm(c["center"] - tgt))
        used.add(it["name"])
        dyaw = (yaw + np.radians(fz.get("yaw_deg", 0.0))) - it["yaw"] if fz.get("yaw_deg") is not None else 0.0
        Rz = _rot(dyaw)
        for g in it["geoms"]:
            p = d.geom_xpos[g].copy()
            p[:2] = tgt + Rz @ (p[:2] - it["center"])
            q, qz = np.zeros(4), np.zeros(4)
            mujoco.mju_axisAngle2Quat(qz, np.array([0, 0, 1.0]), dyaw)
            mujoco.mju_mulQuat(q, qz, m.geom_quat[g])
            moves[g] = (p, q)
    return moves, obj_item


def item_parts(item, obj_yaw_world, align_to=None):
    """Room item geoms as object parts in the object frame (origin = base center), exact quaternions and the room's
    own materials. align_to: Gemini's parts for this object; the item is turned (about z) so its tall side (e.g. a
    chair's backrest) is on the same side as in Gemini's description."""
    m, aabb, _ = room_model()
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    T = {mujoco.mjtGeom.mjGEOM_BOX: "box", mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder",
         mujoco.mjtGeom.mjGEOM_SPHERE: "sphere", mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule"}

    def raw(yaw):
        out = []
        Rz = _rot(-yaw)
        R3 = np.eye(3)
        R3[:2, :2] = Rz
        for g in item["geoms"]:
            t = T.get(int(m.geom_type[g]))
            if t is None:
                continue
            s = m.geom_size[g]
            p = d.geom_xpos[g]
            xy = Rz @ (p[:2] - item["center"])
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, (R3 @ d.geom_xmat[g].reshape(3, 3)).flatten())
            size = {"box": [2 * s[0], 2 * s[1], 2 * s[2]], "cylinder": [2 * s[0], 2 * s[1]], "sphere": [2 * s[0]],
                    "capsule": [2 * s[0], 2 * s[1] + 2 * s[0]]}[t]
            part = dict(shape=t, size=[float(x) for x in size], pos=[float(xy[0]), float(xy[1]), float(p[2] - item["base_z"])],
                        quat=[float(x) for x in q])
            if m.geom_matid[g] >= 0:
                part["material"] = m.material(m.geom_matid[g]).name
            else:
                part["color_rgb"] = [float(x) for x in m.geom_rgba[g][:3]]
            out.append(part)
        return out

    def tall_side(parts):
        pts = [(np.asarray(p["pos"], float), float(np.prod(np.resize(p.get("size", [0.1]), 3)))) for p in parts]
        if not pts:
            return None
        zs = np.array([p[0][2] for p in pts])
        top = [p for p in pts if p[0][2] >= zs.min() + 0.7 * (zs.max() - zs.min())]
        allc = np.mean([p[0] for p in pts], 0)
        c = np.mean([p[0] for p in top], 0)
        v = (c - allc)[:2]
        return v if np.linalg.norm(v) > 0.03 else None

    parts = raw(obj_yaw_world)
    if align_to:
        a, b = tall_side(align_to), tall_side(parts)
        if a is not None and b is not None:
            dyaw = np.arctan2(a[1], a[0]) - np.arctan2(b[1], b[0])
            parts = raw(obj_yaw_world - dyaw)
    return parts
