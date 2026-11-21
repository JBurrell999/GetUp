"""
Merge the official Menagerie Unitree G1 into one of the generated environments.

Produces <env>/generated/<NAME>.xml plus generated/assets/ (room textures and the G1 STLs),
so the generated folder can be dragged into MuJoCo on its own.
"""
from __future__ import annotations

import argparse
import copy
import math
import shutil
from pathlib import Path

from lxml import etree as E

REPO = Path(__file__).resolve().parents[2]
DEFAULT_G1 = REPO / "models" / "mujoco_menagerie" / "unitree_g1" / "g1.xml"
PELVIS_Z = 0.793


def merge(env_dir: Path, room_xml: Path, out_name: str, g1_xml: Path, spawn_xy, yaw_deg: float, model_name: str) -> Path:
    g1_xml = g1_xml.expanduser().resolve()
    if not g1_xml.is_file():
        raise SystemExit(f"G1 XML missing: {g1_xml}\nUse --g1-xml /absolute/path/to/mujoco_menagerie/unitree_g1/g1.xml")
    g1_assets = g1_xml.parent / "assets"
    if not g1_assets.is_dir():
        raise SystemExit(f"G1 mesh assets missing: {g1_assets}")

    room = E.parse(str(room_xml)).getroot()
    robot = E.parse(str(g1_xml)).getroot()
    robot.set("model", model_name)
    compiler = robot.find("compiler")
    if compiler is None:
        compiler = E.SubElement(robot, "compiler")
    compiler.set("meshdir", "assets/g1")
    compiler.set("autolimits", "true")
    opt = robot.find("option")
    if opt is not None:
        opt.set("timestep", "0.002")

    assets = robot.find("asset")
    if assets is None:
        assets = E.SubElement(robot, "asset")
    for a in room.find("asset"):
        a = copy.deepcopy(a)
        if a.tag == "mesh" and a.get("file", "").startswith("assets/"):
            a.set("file", "../" + a.get("file")[len("assets/"):])  # meshdir becomes assets/g1 below
        assets.append(a)
    world = robot.find("worldbody")
    extra_qpos: list[str] = []
    for child in room.find("worldbody"):
        world.append(copy.deepcopy(child))
        if child.tag == "body" and child.find("freejoint") is not None:
            extra_qpos += (child.get("pos", "0 0 0").split() + child.get("quat", "1 0 0 0").split())

    half = math.radians(yaw_deg) / 2
    qw, qz = f"{math.cos(half):.10f}", f"{math.sin(half):.10f}"
    x, y = (f"{v:.4f}" for v in spawn_xy)
    pelvis = world.find("body[@name='pelvis']")
    if pelvis is None:
        raise SystemExit("Unexpected G1 model: missing pelvis body")
    pelvis.set("pos", f"{x} {y} {PELVIS_Z}")
    pelvis.set("quat", f"{qw} 0 0 {qz}")

    keys = robot.find("keyframe")
    if keys is not None:
        for key in keys.findall("key"):
            q = key.get("qpos")
            if q:
                v = q.split()
                if len(v) >= 7:
                    v[:7] = [x, y, f"{PELVIS_Z}", qw, "0", "0", qz]
                key.set("qpos", " ".join(v + extra_qpos))
            qv = key.get("qvel")
            if qv:
                key.set("qvel", " ".join(qv.split() + ["0"] * (len(extra_qpos) // 7 * 6)))

    for tag in ("visual", "statistic"):
        src = room.find(tag)
        dst = robot.find(tag)
        if src is not None:
            if dst is not None:
                robot.remove(dst)
            robot.insert(0 if tag == "visual" else 1, copy.deepcopy(src))
    # aim the default free camera at the robot
    stat = robot.find("statistic")
    if stat is not None:
        stat.set("center", f"{x} {y} 0.9")

    out_dir = env_dir / "generated"
    (out_dir / "assets").mkdir(parents=True, exist_ok=True)
    shutil.copytree(g1_assets, out_dir / "assets" / "g1", dirs_exist_ok=True)
    baked = out_dir / "assets" / "baked"
    if baked.is_dir():
        shutil.rmtree(baked)
    shutil.copytree(env_dir / "assets", out_dir / "assets", dirs_exist_ok=True)
    n = sum(1 for p in (env_dir / "assets").rglob("*") if p.is_file())
    # keep the viewer's default camera pointing at the same named room camera
    g = robot.find("visual/global")
    if g is not None and g.get("cameraid") is not None:
        room_cams = [c.get("name") for c in room.find("worldbody") if c.tag == "camera"]
        all_cams = [c.get("name") for c in world.iter("camera")]
        g.set("cameraid", str(all_cams.index(room_cams[int(g.get("cameraid"))])))
    target = out_dir / out_name
    E.ElementTree(robot).write(str(target), encoding="utf-8", xml_declaration=True, pretty_print=True)
    print(f"Created: {target}")
    print(f"Copied: G1 meshes and {n} scene asset files (textures + baked meshes) to {out_dir / 'assets'}")
    try:
        import mujoco
        m = mujoco.MjModel.from_xml_path(str(target))
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
        mujoco.mj_forward(m, d)
        print(f"COMPILATION OK: nq={m.nq}, nv={m.nv}, geoms={m.ngeom}, cameras={m.ncam}, "
              f"free props={len(extra_qpos) // 7}, contacts at 'stand'={d.ncon}")
    except ImportError:
        print("MuJoCo not installed in this Python; compilation not checked")
    except Exception as e:
        raise SystemExit(f"MuJoCo compilation error: {e}")
    print(f"Drag {target.relative_to(env_dir)} from Finder into MuJoCo (keep generated/assets next to it).")
    return target


def cli(env_dir: Path, nav_xml: str, visual_xml: str, out_name: str, spawn_xy, yaw_deg: float, model_name: str):
    pa = argparse.ArgumentParser(description=f"Place the Unitree G1 into {env_dir.name}")
    pa.add_argument("--g1-xml", type=Path, default=DEFAULT_G1, help="path to the official Menagerie g1.xml")
    pa.add_argument("--visual-only-collision", action="store_true",
                    help="use the detailed furniture colliders instead of the simplified NAV blockers")
    pa.add_argument("--spawn", type=float, nargs=3, metavar=("X", "Y", "YAW_DEG"), default=None,
                    help=f"robot start pose (default {spawn_xy[0]} {spawn_xy[1]} {yaw_deg})")
    a = pa.parse_args()
    if a.spawn:
        spawn_xy, yaw_deg = a.spawn[:2], a.spawn[2]
    room = env_dir / (visual_xml if a.visual_only_collision else nav_xml)
    merge(env_dir, room, out_name, a.g1_xml, spawn_xy, yaw_deg, model_name)
