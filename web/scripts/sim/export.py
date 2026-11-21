"""
Exports the MuJoCo room + G1 + looping walk for the site's 3D view.

    .venv/bin/python web/scripts/sim/export.py \
        g1_room_mujoco/Hackathon_G1_Ultra_Detail_V2/generated/G1_HACKATHON_V2.xml

Writes web/public/sim/:
    scene.json   room geoms, materials, lights, robot body tree, walk metadata, footsteps
    robot.bin    decimated robot meshes (float32 positions, uint16/uint32 indices)
    walk.bin     float32 qpos per frame (frames x nq), one seamless lap
    tex/*.webp   room textures, resized
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from walk import Walker  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "public" / "sim"
MESH_CELL = 0.0035      # vertex-clustering grid for robot meshes, metres
TEX_MAX = {"v2_floor_tex": 1024}  # everything else is capped at 512 px


def r(v, n=4):
    return [round(float(x), n) for x in np.ravel(v)]


def mat_to_quat(mat):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(mat, dtype=np.float64).ravel())
    return q


def decimate(verts: np.ndarray, faces: np.ndarray, cell: float):
    """Vertex clustering: merge vertices in the same grid cell, drop collapsed triangles."""
    key = np.floor(verts / cell).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    n = inv.max() + 1
    out = np.zeros((n, 3))
    np.add.at(out, inv, verts)
    out /= np.bincount(inv, minlength=n)[:, None]
    f = inv[faces]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    _, first = np.unique(np.sort(f, axis=1), axis=0, return_index=True)
    f = f[np.sort(first)]
    used = np.unique(f)
    remap = -np.ones(n, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return out[used].astype(np.float32), remap[f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xml", type=Path)
    a = ap.parse_args()
    xml = a.xml.resolve()
    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
    mujoco.mj_forward(m, d)
    pelvis = m.body("pelvis").id
    robot_bodies = [b for b in range(m.nbody) if m.body_rootid[b] == pelvis]
    in_robot = set(robot_bodies)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tex").mkdir(exist_ok=True)

    # ------------------------------------------------------------ textures
    files = {t.get("name"): t.get("file") for t in ET.parse(xml).getroot().iter("texture") if t.get("file")}
    tex_out = {}
    for name, rel in files.items():
        img = cv2.imread(str(xml.parent / rel), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[tex] can't read {rel}")
            continue
        cap = TEX_MAX.get(name, 512)
        h, w = img.shape[:2]
        s = min(1.0, cap / max(h, w))
        if s < 1:
            img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        fn = f"{name}.webp"
        cv2.imwrite(str(OUT / "tex" / fn), img, [cv2.IMWRITE_WEBP_QUALITY, 82])
        tex_out[name] = fn

    # ------------------------------------------------------------ materials
    materials = []
    for i in range(m.nmat):
        tex = [int(x) for x in m.mat_texid[i] if x >= 0]
        tname = m.texture(tex[0]).name if tex else None
        materials.append(dict(
            name=m.material(i).name,
            rgba=r(m.mat_rgba[i], 3),
            emission=round(float(m.mat_emission[i]), 3),
            specular=round(float(m.mat_specular[i]), 3),
            reflectance=round(float(m.mat_reflectance[i]), 3),
            tex=tex_out.get(tname) if tname else None,
            repeat=r(m.mat_texrepeat[i], 3),
            uniform=bool(m.mat_texuniform[i]),
        ))

    default_rgba = np.array([0.5, 0.5, 0.5, 1.0], dtype=np.float32)

    def colour(g):
        if m.geom_matid[g] >= 0 and np.allclose(m.geom_rgba[g], default_rgba):
            return None
        return r(m.geom_rgba[g], 3)

    # ------------------------------------------------------------ room geoms
    TYPES = {int(mujoco.mjtGeom.mjGEOM_BOX): "box", int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
             int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule", int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
             int(mujoco.mjtGeom.mjGEOM_PLANE): "plane", int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid"}
    room = []
    skipped = 0
    for g in range(m.ngeom):
        if m.geom_bodyid[g] in in_robot:
            continue
        rgba = m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 and colour(g) is None else m.geom_rgba[g]
        if rgba[3] <= 0.01 or m.geom_group[g] == 4 or int(m.geom_type[g]) not in TYPES:
            skipped += 1
            continue
        room.append(dict(
            t=TYPES[int(m.geom_type[g])],
            m=int(m.geom_matid[g]),
            c=colour(g),
            s=r(m.geom_size[g]),
            p=r(d.geom_xpos[g]),
            q=r(mat_to_quat(d.geom_xmat[g]), 5),
            n=m.geom(g).name or None,
        ))

    lights = [dict(name=m.light(i).name, pos=r(m.light_pos[i], 3), dir=r(m.light_dir[i], 3),
                   diffuse=r(m.light_diffuse[i], 3), cutoff=round(float(m.light_cutoff[i]), 2),
                   directional=bool(m.light_type[i] == mujoco.mjtLightType.mjLIGHT_DIRECTIONAL))
              for i in range(m.nlight)]

    # ------------------------------------------------------------ robot
    index = {b: i for i, b in enumerate(robot_bodies)}
    bodies = []
    for b in robot_bodies:
        jnt = None
        for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]):
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                jnt = dict(q=int(m.jnt_qposadr[j]), axis=r(m.jnt_axis[j], 5), name=m.joint(j).name)
        bodies.append(dict(
            name=m.body(b).name,
            parent=index.get(int(m.body_parentid[b]), -1),
            pos=r(m.body_pos[b], 5),
            quat=r(m.body_quat[b], 6),
            joint=jnt,
            free=b == pelvis,
        ))

    blobs, meshes, mesh_ids, geoms = [], [], {}, []
    offset = 0
    total_in = total_out = 0
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        if b not in in_robot or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_group[g] != 2:
            continue
        mid = int(m.geom_dataid[g])
        if mid not in mesh_ids:
            va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
            v, f = decimate(m.mesh_vert[va:va + vn].astype(np.float64), m.mesh_face[fa:fa + fn].astype(np.int64), MESH_CELL)
            total_in += vn
            total_out += len(v)
            idx = f.astype(np.uint16 if len(v) < 65536 else np.uint32).ravel()
            pos = v.astype(np.float32).ravel()
            entry = dict(name=m.mesh(mid).name, verts=len(v), tris=len(f), pos=offset, idx=offset + pos.nbytes,
                         wide=bool(idx.dtype == np.uint32))
            blobs += [pos.tobytes(), idx.tobytes()]
            offset += pos.nbytes + idx.nbytes
            pad = (-offset) % 4
            if pad:
                blobs.append(b"\0" * pad)
                offset += pad
            mesh_ids[mid] = len(meshes)
            meshes.append(entry)
        mat = int(m.geom_matid[g])
        geoms.append(dict(body=index[b], mesh=mesh_ids[mid], pos=r(m.geom_pos[g], 5), quat=r(m.geom_quat[g], 6),
                          rgba=r(m.mat_rgba[mat], 3) if mat >= 0 else r(m.geom_rgba[g], 3)))
    (OUT / "robot.bin").write_bytes(b"".join(blobs))

    # ------------------------------------------------------------ walk
    walker = Walker(m)
    Q, info = walker.run()
    if info["worst_ik_error"] > 1e-4:
        raise SystemExit(f"walk IK didn't converge: {info['worst_ik_error']:.2e}")
    (OUT / "walk.bin").write_bytes(Q.astype(np.float32).tobytes())

    scene = dict(
        source=xml.name,
        materials=materials,
        room=room,
        lights=lights,
        robot=dict(bodies=bodies, geoms=geoms, meshes=meshes, nq=int(m.nq)),
        walk=dict((k, v) for k, v in info.items() if k != "worst_ik_error"),
    )
    (OUT / "scene.json").write_text(json.dumps(scene, separators=(",", ":")))

    size = lambda p: f"{p.stat().st_size / 1e3:.0f} kB"
    print(f"[room] {len(room)} geoms ({skipped} invisible skipped), {len(tex_out)} textures, {len(lights)} lights")
    print(f"[robot] {len(bodies)} bodies, {len(geoms)} mesh geoms, verts {total_in} -> {total_out}")
    print(f"[walk] {info['frames']} frames, {info['duration']:.1f} s lap, {info['steps_per_lap']} steps, "
          f"{info['speed']:.2f} m/s, IK err {info['worst_ik_error']:.1e}")
    print(f"[out] scene.json {size(OUT / 'scene.json')}, robot.bin {size(OUT / 'robot.bin')}, "
          f"walk.bin {size(OUT / 'walk.bin')}, tex {sum(f.stat().st_size for f in (OUT / 'tex').iterdir()) / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
