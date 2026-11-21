"""Replay a Stage 1 interpretation inside one of the project's detailed rooms.

The kinematic replay (replay.py) works in the person's frame: they start at the origin facing +x.
Instead of moving the robot, the whole room is wrapped in one static body and placed so that a
chosen spot in it lands on that origin. Gemini's reconstructed table(s) and object are added in the
room's own materials.

Placement: candidate spots and headings in each room's clear areas are tested so that the robot's
path, the reconstructed surfaces and the object don't overlap the room's furniture, and so that the
camera Gemini estimated isn't behind a wall or inside a fridge. If the camera would be, it is pulled
in toward the subject (and its field of view widened to keep the framing).

    from motion_mirror.in_env import render_in_env
    render_in_env(final_json, "kitchen", "input.mp4", out_dir)
"""
from __future__ import annotations

import copy
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from . import config as C
from .replay import Replay, _obj_geoms, keyframe_to_pose

REPO = C.ROOT.parent

ENVS = {
    "hackathon": dict(
        name="hackathon room", xml="g1_room_mujoco/Hackathon_G1_Ultra_Detail_V2/HACKATHON_V2_VISUAL.xml",
        prefer=(0.0, -4.2, 90.0),                        # x, y, heading (deg) in the room
        regions=[(-1.7, 1.7, -6.6, 4.8)],                # x0, x1, y0, y1 of open floor to search
        top="v2_table", legs="room_frame", solid="v2_darkwood", seat=("v2_chair", "room_frame", True)),
    "kitchen": dict(
        name="kitchen", xml="g1_room_mujoco/Kitchen_G1_Detail/KITCHEN_VISUAL.xml",
        prefer=(-1.35, -2.2, 90.0),
        regions=[(-3.4, 0.7, -3.4, -0.6)],
        top="kt_oak", legs="kt_black_matte", solid="kt_quartz", seat=("kt_linen", "kt_oak", True)),
    "warehouse": dict(
        name="warehouse", xml="g1_room_mujoco/Warehouse_G1_Detail/WAREHOUSE_VISUAL.xml",
        prefer=(0.0, -5.65, 90.0),
        regions=[(-1.8, 1.8, -11.0, 15.0), (-14.5, 14.5, -7.7, -4.3)],
        top="wh_butcher", legs="wh_bench_grey", solid="wh_galvanized", seat=("wh_seat_black", "wh_chrome", False)),
}

ROBOT_HALF = 0.38          # m around each pelvis keyframe the robot's body and arms can occupy
HIDE = {"robot_spawning_disk"}  # room geoms that make no sense in a result (the hackathon room's spawn marker)
ROBOT_H = 1.4
MARGIN = 0.04


def _rz(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])


def _quat_z(deg):
    a = math.radians(deg) / 2
    return np.array([math.cos(a), 0, 0, math.sin(a)])


# ---------------------------------------------------------------- room geometry (cached per env)
_ROOM: dict[str, dict] = {}


def room(env_id):
    """The room alone, compiled: obstacle boxes (room frame) and a model for camera ray tests."""
    if env_id in _ROOM:
        return _ROOM[env_id]
    spec = ENVS[env_id]
    path = REPO / spec["xml"]
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    boxes = []
    for g in range(m.ngeom):
        if not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
        R = d.geom_xmat[g].reshape(3, 3)
        ctr, half = d.geom_xpos[g] + R @ c, np.abs(R) @ h
        lo, hi = ctr - half, ctr + half
        if hi[2] <= 0.02 or lo[2] >= 2.2 or (hi - lo)[:2].max() > 25:   # floors, ceilings, the ground outside
            continue
        boxes.append(np.concatenate([lo, hi]))
    _ROOM[env_id] = dict(path=path, m=m, d=d, boxes=np.array(boxes))
    return _ROOM[env_id]


# ---------------------------------------------------------------- what the interpretation needs, in the person frame
def _seated(kf):
    legs = [kf.get(f"{s}_leg", {}) or {} for s in ("left", "right")]
    return np.mean([l.get("thigh_forward_deg", 0.0) for l in legs]) > 50 and np.mean([l.get("knee_flexion_deg", 0.0) for l in legs]) > 50


def seat_for(interp):
    """If the person sits at some point, a seat under the robot's thighs in that pose (person frame):
    dict(x, y, yaw_deg, top_z). The replay keeps the feet on the floor, so the seat height comes from the pose."""
    kfs = sorted(interp.get("keyframes", []), key=lambda k: k["t"])
    kf = next((k for k in kfs if _seated(k)), None)
    if kf is None:
        return None
    rp = Replay(interp, 64, 64)
    q, yaw, pitch, roll, xy = keyframe_to_pose(kf)
    rp.fingers = {"left": np.zeros(7), "right": np.zeros(7)}
    rp._set(q, yaw, pitch, roll, xy)
    m, d = rp.m, rp.d
    lows = []
    for g in range(m.ngeom):
        if "hip_yaw" in m.body(m.geom_bodyid[g]).name:     # the thigh links
            c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
            R = d.geom_xmat[g].reshape(3, 3)
            lows.append((d.geom_xpos[g] + R @ c - np.abs(R) @ h)[2])
    pel = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")]
    fwd = np.array([math.cos(yaw), math.sin(yaw)])
    ctr = pel[:2] + 0.1 * fwd
    return dict(x=float(ctr[0]), y=float(ctr[1]), yaw_deg=math.degrees(yaw), top_z=float(max(0.3, min(lows) - 0.005)))


def footprint(interp):
    """Boxes [x0 y0 z0 x1 y1 z1] the robot, the surfaces and the object occupy (person frame)."""
    out = []
    for k in interp.get("keyframes", []):
        p = k.get("pelvis", {}) or {}
        x, y = p.get("forward_m", 0.0), p.get("left_m", 0.0)
        out.append([x - ROBOT_HALF, y - ROBOT_HALF, 0.02, x + ROBOT_HALF, y + ROBOT_HALF, ROBOT_H])
    for s in (interp.get("environment", {}) or {}).get("surfaces", []) or []:
        x, y = s.get("forward_m", 1.0), s.get("left_m", 0.0)
        dx, dy = s.get("depth_m", 0.6) / 2, s.get("width_m", 0.8) / 2
        out.append([x - dx, y - dy, 0.02, x + dx, y + dy, s.get("height_m", 0.75)])
    obj = interp.get("object")
    if obj:
        sz = np.array(obj.get("size_m", [0.3, 0.3, 0.3]), float)
        r = max(sz[0], sz[1]) / 2 + 0.05
        for key in ("start", "end"):
            p = obj.get(key)
            if p:
                z0 = p.get("on_surface_height_m", 0.0)
                out.append([p.get("forward_m", 0) - r, p.get("left_m", 0) - r, max(z0, 0.02),
                            p.get("forward_m", 0) + r, p.get("left_m", 0) + r, z0 + sz[2]])
    return np.array(out, float)


def gemini_camera(interp):
    """Look-at point, camera position, distance and fovy, as replay.Replay._camera sets them (person frame)."""
    c = interp.get("camera", {}) or {}
    r, h = c.get("horizontal_distance_m", 3.0), c.get("height_m", 1.4)
    look = np.array([c.get("look_at_forward_m", 0.0), c.get("look_at_left_m", 0.0), c.get("look_at_height_m", 0.8)])
    az = math.radians(180.0 + c.get("azimuth_deg", 0.0))
    el = -math.atan2(h - look[2], r)
    dist = math.hypot(r, h - look[2])
    fwd = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    return look, look - dist * fwd, dist, c.get("vertical_fov_deg", 50.0)


def _to_room(boxes, ax, ay, yaw):
    """Person-frame boxes -> room-frame AABBs for a person standing at (ax, ay) facing yaw (deg)."""
    R = _rz(yaw)[:2, :2]
    corners = np.stack([boxes[:, [0, 1]], boxes[:, [3, 1]], boxes[:, [0, 4]], boxes[:, [3, 4]]], 1)   # B,4,2
    xy = corners @ R.T + np.array([ax, ay])
    return np.concatenate([xy.min(1), boxes[:, 2:3], xy.max(1), boxes[:, 5:6]], 1)


def _ray(env, p0, p1):
    """Free distance from p0 toward p1 (room frame), capped at |p1 - p0|."""
    r = room(env)
    v = np.asarray(p1, float) - np.asarray(p0, float)
    L = float(np.linalg.norm(v))
    gid = np.zeros(1, np.int32)
    hit = mujoco.mj_ray(r["m"], r["d"], np.asarray(p0, float), v / L, None, 1, -1, gid)
    return L if hit < 0 else min(L, float(hit))


def choose_place(interp, env_id):
    """Pick the spot and heading in the room for the person's start, plus camera adjustments."""
    spec, r = ENVS[env_id], room(env_id)
    fp = footprint(interp)
    obst = r["boxes"]
    cands = []
    for x0, x1, y0, y1 in spec["regions"]:
        for x in np.arange(x0, x1 + 1e-6, 0.25):
            for y in np.arange(y0, y1 + 1e-6, 0.25):
                for yaw in range(0, 360, 45):
                    cands.append((x, y, yaw))
    cands = np.array(cands, float)
    # overlap of every candidate's footprint with every obstacle (vectorised AABB test). Candidates come in
    # grid order, so a small chunk covers a narrow strip of floor: only obstacles that touch the strip's
    # bounding box can overlap any footprint in it, which keeps the large warehouse fast (exact, not approximate).
    free = np.ones(len(cands), bool)
    step = 128
    for i in range(0, len(cands), step):
        chunk = cands[i:i + step]
        fb = np.stack([_to_room(fp, *c) for c in chunk])                         # N,B,6
        lo, hi = fb[:, :, None, :3] - MARGIN, fb[:, :, None, 3:] + MARGIN
        near = obst[np.all((lo.min((0, 1, 2)) < obst[:, 3:]) & (hi.max((0, 1, 2)) > obst[:, :3]), 1)]
        if len(near):
            ov = np.all((lo < near[None, None, :, 3:]) & (hi > near[None, None, :, :3]), -1)
            free[i:i + step] = ~ov.any((1, 2))
    px, py, pyaw = spec["prefer"]
    dyaw = np.abs((cands[:, 2] - pyaw + 180) % 360 - 180)
    pref = -0.08 * np.hypot(cands[:, 0] - px, cands[:, 1] - py) - 0.002 * dyaw
    order = np.argsort(-(pref + 10 * free))[:80]
    look, cam, dist, fovy = gemini_camera(interp)
    best = None
    for i in order:
        x, y, yaw = cands[i]
        R = _rz(yaw)
        lr, cr = R @ look + [x, y, 0], R @ cam + [x, y, 0]
        clear = _ray(env_id, lr, cr)
        score = (3.0 * min(1.0, clear / dist)) + pref[i] + (0 if free[i] else -20)
        if best is None or score > best[0]:
            best = (score, (float(x), float(y), float(yaw)), clear, bool(free[i]))
    _, anchor, clear, ok = best
    match_dist = dist if clear >= dist else max(1.0, clear - 0.2)
    wide = wide_camera(env_id, anchor, interp, look)
    return dict(env=env_id, anchor=anchor, clear_of_furniture=ok, camera_distance=round(dist, 2),
                match_distance=round(match_dist, 2), match_fovy=round(_widen(fovy, dist, match_dist), 1), wide=wide)


def _widen(fovy, d0, d1):
    """Field of view that keeps the subject the same size when the camera moves from d0 to d1."""
    if d1 >= d0:
        return fovy
    return min(95.0, math.degrees(2 * math.atan(math.tan(math.radians(fovy) / 2) * d0 / d1)))


def wide_camera(env_id, anchor, interp, look):
    """An establishing view: further back and higher, from whichever side the room leaves open."""
    x, y, yaw = anchor
    R = _rz(yaw)
    pel = np.array([[(k.get("pelvis") or {}).get("forward_m", 0.0), (k.get("pelvis") or {}).get("left_m", 0.0)]
                    for k in interp.get("keyframes", [])] or [[0.0, 0.0]])
    centre = np.array([*pel.mean(0), 0.8])
    c_az = (interp.get("camera", {}) or {}).get("azimuth_deg", 0.0)
    best = None
    for k in range(16):
        az = c_az + 22.5 * k
        a = math.radians(az)
        for dist, h in ((4.2, 2.1), (3.4, 1.9)):
            p = centre + np.array([dist * math.cos(a), dist * math.sin(a), h - 0.8])
            clear = _ray(env_id, R @ centre + [x, y, 0], R @ p + [x, y, 0])
            ok = clear >= dist * 0.98
            score = (2 if ok else clear / dist) + 0.4 * (dist > 4) - 0.004 * min(abs(22.5 * k), abs(360 - 22.5 * k))
            if best is None or score > best[0]:
                best = (score, dict(lookat=centre.round(3).tolist(), azimuth=180.0 + az,
                                    elevation=-math.degrees(math.atan2(h - 0.8, dist)),
                                    distance=round(math.hypot(dist, h - 0.8) if ok else max(1.5, clear - 0.2), 2),
                                    fovy=58.0))
    return best[1]


# ---------------------------------------------------------------- the merged scene
def _seat_geoms(wb, seat, spec):
    """A chair (or a stool, in the warehouse) under the seated robot, in the room's style. Visual only."""
    top, frame, back = spec["seat"]
    R = _rz(seat["yaw_deg"])
    c = np.array([seat["x"], seat["y"], 0.0])
    q = " ".join(f"{v:.6f}" for v in _quat_z(seat["yaw_deg"]))
    z = seat["top_z"]
    vis = dict(contype="0", conaffinity="0", group="2")

    def box(local, half, mat):
        p = c + R @ np.array(local)
        ET.SubElement(wb, "geom", type="box", pos=f"{p[0]:.3f} {p[1]:.3f} {local[2]:.3f}", quat=q,
                      size=" ".join(f"{v:.3f}" for v in half), material=mat, **vis)

    if back:
        box([0, 0, z - 0.03], [0.22, 0.22, 0.03], top)
        for lx in (-0.18, 0.18):
            for ly in (-0.18, 0.18):
                box([lx, ly, (z - 0.06) / 2], [0.018, 0.018, (z - 0.06) / 2], frame)
        box([-0.24, 0, z + 0.25], [0.03, 0.22, 0.22], top)
        for ly in (-0.18, 0.18):
            box([-0.24, ly, z + 0.02], [0.018, 0.018, 0.04], frame)
    else:
        ET.SubElement(wb, "geom", type="cylinder", pos=f"{c[0]:.3f} {c[1]:.3f} {z - 0.03:.3f}", size="0.19 0.03",
                      material=top, **vis)
        ET.SubElement(wb, "geom", type="cylinder", pos=f"{c[0]:.3f} {c[1]:.3f} {(z - 0.06) / 2:.3f}",
                      size=f"0.025 {(z - 0.06) / 2:.3f}", material=frame, **vis)
        ET.SubElement(wb, "geom", type="cylinder", pos=f"{c[0]:.3f} {c[1]:.3f} 0.01", size="0.25 0.01",
                      material=frame, **vis)


def scene_xml(interp, env_id, anchor):
    """Menagerie G1-with-hands scene + the room (placed so `anchor` is the origin, facing +x)
    + Gemini's surfaces (in the room's materials) + the object."""
    spec = ENVS[env_id]
    base = ET.parse(C.SCENE_XML.parent / "scene_with_hands.xml").getroot()
    rpath = REPO / spec["xml"]
    rroot = ET.parse(rpath).getroot()

    vis = rroot.find("visual")
    if vis is not None:
        vis = copy.deepcopy(vis)
        g = vis.find("global")
        if g is not None:
            g.attrib.pop("cameraid", None)
        old = base.find("visual")
        if old is not None:
            base.remove(old)
        base.insert(1, vis)
    if rroot.find("statistic") is not None:
        base.append(copy.deepcopy(rroot.find("statistic")))

    asset = base.find("asset")
    for t in list(asset):
        if t.tag == "texture" and t.get("type") == "skybox":
            asset.remove(t)
    for a in rroot.find("asset"):
        a = copy.deepcopy(a)
        if a.get("file"):
            a.set("file", str((rpath.parent / a.get("file")).resolve()))
        asset.append(a)

    wb = base.find("worldbody")
    for light in wb.findall("light"):
        wb.remove(light)
    floor = wb.find("geom[@name='floor']")
    if floor is not None:          # kept (invisible) so the replay's foot/floor bookkeeping still works
        floor.attrib.pop("material", None)
        floor.set("rgba", "0 0 0 0")
        floor.set("group", "5")
    ax, ay, yaw = anchor
    p = _rz(-yaw) @ np.array([-ax, -ay, 0.0])
    q = _quat_z(-yaw)
    body = ET.SubElement(wb, "body", name="furniture", pos=f"{p[0]:.4f} {p[1]:.4f} 0",
                         quat=" ".join(f"{v:.6f}" for v in q))
    for ch in rroot.find("worldbody"):
        if ch.get("name") in HIDE:
            continue
        ch = copy.deepcopy(ch)
        if ch.tag == "body":
            for fj in ch.findall("freejoint"):
                ch.remove(fj)                         # room props stay put during the replay
        body.append(ch)

    for i, s in enumerate((interp.get("environment", {}) or {}).get("surfaces", []) or []):
        x, y = s.get("forward_m", 1.0), s.get("left_m", 0.0)
        sx, sy, h = s.get("depth_m", 0.6), s.get("width_m", 0.8), s.get("height_m", 0.75)
        if (s.get("kind") or "table") == "solid":
            ET.SubElement(wb, "geom", name=f"surface{i}", type="box", size=f"{sx/2:.3f} {sy/2:.3f} {h/2:.3f}",
                          pos=f"{x:.3f} {y:.3f} {h/2:.3f}", material=spec["solid"])
        else:
            ET.SubElement(wb, "geom", name=f"surface{i}", type="box", size=f"{sx/2:.3f} {sy/2:.3f} 0.02",
                          pos=f"{x:.3f} {y:.3f} {h-0.02:.3f}", material=spec["top"])
            for lx in (-1, 1):
                for ly in (-1, 1):
                    ET.SubElement(wb, "geom", type="box", size=f"0.022 0.022 {(h-0.04)/2:.3f}",
                                  pos=f"{x+lx*(sx/2-0.05):.3f} {y+ly*(sy/2-0.05):.3f} {(h-0.04)/2:.3f}",
                                  material=spec["legs"])
    seat = seat_for(interp)
    if seat:
        _seat_geoms(wb, seat, spec)
    obj = interp.get("object")
    if obj:
        # a little self-lit, so the object reads clearly in the room's lighting
        ET.SubElement(asset, "material", name="mm_object", emission="0.3", specular="0.6", shininess="0.6")
        ob = ET.fromstring(f'<body name="object" mocap="true">{_obj_geoms(obj)}</body>')
        for g in ob.findall("geom"):
            g.set("material", "mm_object")
        wb.append(ob)
    return ET.tostring(base, encoding="unicode")


class EnvReplay(Replay):
    """Replay inside a room. view="match": Gemini's camera (pulled in if the room is in the way);
    view="wide": an establishing shot from the most open side."""

    def __init__(self, interp, place, width=848, height=480, view="match"):
        self.place, self.view = place, view
        super().__init__(interp, width, height, scene_xml=scene_xml(interp, place["env"], place["anchor"]))
        # MuJoCo 4.0.1: a spotlight without shadows drops ambient light outside its cone (black ceilings and
        # upper walls). The hackathon room has such lights; shadow-casting ones render correctly.
        for i in range(self.m.nlight):
            if self.m.light_type[i] != mujoco.mjtLightType.mjLIGHT_DIRECTIONAL:
                self.m.light_castshadow[i] = 1

    def _camera(self, t=0.0):   # same signature as Replay._camera (the replay may pass the time)
        cam = super()._camera()
        if self.view == "match":
            cam.distance = self.place["match_distance"]
            self.m.vis.global_.fovy = self.place["match_fovy"]
        else:
            w = self.place["wide"]
            cam.lookat[:] = w["lookat"]
            cam.azimuth, cam.elevation, cam.distance = w["azimuth"], w["elevation"], w["distance"]
            self.m.vis.global_.fovy = w["fovy"]
        return cam


def render_in_env(interp, env_id, video, out_dir, wide=True, log=print):
    """Place, replay and render. Writes robot_env.mp4, side_by_side_env.mp4 (+ robot_wide.mp4) and place.json."""
    from .mirror import side_by_side, video_info

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(interp, (str, Path)):
        interp = json.loads(Path(interp).read_text())
    W, H, fps = video_info(video)
    place = choose_place(interp, env_id)
    (out_dir / "place.json").write_text(json.dumps(place, indent=1))
    log(f"[site] placed in the {ENVS[env_id]['name']} at {place['anchor']} "
        f"(clear of furniture: {place['clear_of_furniture']}, camera {place['camera_distance']} -> {place['match_distance']} m)")
    log("[site] rendering the robot in the room", )
    replay = EnvReplay(interp, place, W, H, "match")
    # Keep a reloadable copy of the actual merged scene alongside the video. Meshes
    # stay in the repository; room textures already have absolute paths.
    scene_path = out_dir / "scene.xml"
    mujoco.mj_saveLastXML(str(scene_path), replay.m)
    scene = ET.parse(scene_path)
    compiler = scene.getroot().find("compiler")
    compiler.set("meshdir", str((C.SCENE_XML.parent / "assets").resolve()))
    scene.write(scene_path, encoding="unicode")
    mujoco.MjModel.from_xml_path(str(scene_path))
    tel = replay.run(fps, out_dir / "robot_env.mp4")
    side_by_side(video, out_dir / "robot_env.mp4", out_dir / "side_by_side_env.mp4", f"in the {ENVS[env_id]['name']}")
    if wide:
        log("[site] rendering the wide view")
        EnvReplay(interp, place, W, H, "wide").run(fps, out_dir / "robot_wide.mp4")
    tel_short = {k: v for k, v in tel.items() if k != "samples"}
    (out_dir / "telemetry_env.json").write_text(json.dumps(tel_short, indent=1, default=float))
    return place, tel_short
