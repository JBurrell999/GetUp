"""
Bake many small static primitives into a few triangle meshes (one per material and geom group).

MuJoCo's classic renderer issues one draw call per geom per pass, so a scene with thousands of
small decorative geoms renders slowly (~8 us per geom per pass on an Apple-silicon Mac). Baking
them into one mesh per material keeps every detail but cuts the draw calls to about a hundred.

UVs reproduce what the primitive would have shown, with one deliberate difference: every box
face is mapped so its texture reads correctly (not mirrored) from outside, with "up" = +z on
vertical faces and +y on top faces. Materials with a uniform tile size use world coordinates,
so textures continue seamlessly across neighbouring pieces.

Meshes are written in MuJoCo's binary .msh format:
    int32 nvert, nnormal, ntexcoord, nface; float32 vert[3n], normal[3n], texcoord[2n]; int32 face[3m]
"""
from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np


def quat_to_mat(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def frame_from_axis(a, b):
    """Rotation whose z axis points from a to b, plus centre and half length."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L = float(np.linalg.norm(d))
    z = d / L if L > 1e-12 else np.array([0.0, 0.0, 1.0])
    ref = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(ref, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1), (a + b) / 2, L / 2


def segments(r: float) -> int:
    return int(np.clip(round(6 + 120 * r), 8, 36))


class Mesh:
    def __init__(self):
        self.v, self.n, self.t, self.f = [], [], [], []
        self.count = 0
        self.nv = 0

    def _add(self, V, N, TC, F):
        self.v.append(np.asarray(V, np.float32))
        self.n.append(np.asarray(N, np.float32))
        self.t.append(np.asarray(TC, np.float32))
        self.f.append(np.asarray(F, np.int64) + self.nv)
        self.nv += len(V)
        self.count += 1

    # -------------------------------------------------- primitives (world transform R, c)
    def box(self, R, c, h, uv):
        V, N, TC, F = [], [], [], []
        faces = [((0, -1, 0), (1, 0, 0), (0, 0, 1)), ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
                 ((-1, 0, 0), (0, -1, 0), (0, 0, 1)), ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                 ((0, 0, 1), (1, 0, 0), (0, 1, 0)), ((0, 0, -1), (1, 0, 0), (0, -1, 0))]
        for n, u, v in faces:
            n, u, v = np.array(n, float), np.array(u, float), np.array(v, float)
            hn, hu, hv = abs(n @ h), abs(u @ h), abs(v @ h)
            o = len(V)
            for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = R @ (n * hn + u * hu * su + v * hv * sv) + c
                V.append(p)
                N.append(R @ n)
                TC.append(uv.face(p, R @ u, R @ v, su, sv, hu, hv))
            F += [(o, o + 1, o + 2), (o, o + 2, o + 3)]
        self._add(V, N, TC, F)

    def cylinder(self, R, c, r, hh, uv):
        n = segments(r)
        V, N, TC, F = [], [], [], []
        for j in range(n + 1):
            a = 2 * math.pi * j / n
            ca, sa = math.cos(a), math.sin(a)
            for z, tv in ((-hh, 1.0), (hh, 0.0)):
                V.append(R @ np.array([r * ca, r * sa, z]) + c)
                N.append(R @ np.array([ca, sa, 0.0]))
                TC.append(uv.wrap(j / n, tv, a * r, z))
        for j in range(n):
            b0, t0, b1, t1 = 2 * j, 2 * j + 1, 2 * j + 2, 2 * j + 3
            F += [(b0, b1, t1), (b0, t1, t0)]
        for z, s in ((hh, 1), (-hh, -1)):
            o = len(V)
            V.append(R @ np.array([0, 0, z]) + c)
            N.append(R @ np.array([0, 0, s]))
            TC.append(uv.disc(0, 0))
            for j in range(n + 1):
                a = 2 * math.pi * j / n
                V.append(R @ np.array([r * math.cos(a), r * math.sin(a), z]) + c)
                N.append(R @ np.array([0, 0, s]))
                TC.append(uv.disc(math.cos(a), math.sin(a)))
            for j in range(n):
                F.append((o, o + 1 + j, o + 2 + j) if s > 0 else (o, o + 2 + j, o + 1 + j))
        self._add(V, N, TC, F)

    def ellipsoid(self, R, c, radii, uv, half_len=0.0):
        """Ellipsoid (or a capsule when half_len > 0: hemispheres pushed apart along z)."""
        nlon = min(20, segments(float(max(radii[0], radii[1]))))
        nlat = max(6, nlon // 2)
        V, N, TC, F = [], [], [], []
        if half_len:  # duplicate the equator so the straight section is a clean cylinder band
            k = nlat // 2 + 1
            rows = [(th, half_len) for th in np.linspace(0, math.pi / 2, k)] + \
                   [(th, -half_len) for th in np.linspace(math.pi / 2, math.pi, k)]
        else:
            rows = [(math.pi * i / nlat, 0.0) for i in range(nlat + 1)]
        nr = len(rows)
        for i, (th, dz) in enumerate(rows):
            st, ct = math.sin(th), math.cos(th)
            for j in range(nlon + 1):
                ph = 2 * math.pi * j / nlon
                unit = np.array([st * math.cos(ph), st * math.sin(ph), ct])
                p = unit * radii + np.array([0, 0, dz])
                nn = unit / np.maximum(radii, 1e-9)
                V.append(R @ p + c)
                N.append(R @ (nn / np.linalg.norm(nn)))
                TC.append(uv.wrap(j / nlon, i / (nr - 1), ph * radii[0], -p[2]))
        w = nlon + 1
        for i in range(nr - 1):
            for j in range(nlon):
                a, b, cc, d = i * w + j, (i + 1) * w + j, (i + 1) * w + j + 1, i * w + j + 1
                F += [(a, b, cc), (a, cc, d)]
        self._add(V, N, TC, F)

    # -------------------------------------------------- output
    def write(self, path: Path):
        V = np.concatenate(self.v)
        N = np.concatenate(self.n)
        TC = np.concatenate(self.t)
        F = np.concatenate(self.f).astype(np.int32)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(struct.pack("4i", len(V), len(N), len(TC), len(F)))
            fh.write(V.astype(np.float32).tobytes())
            fh.write(N.astype(np.float32).tobytes())
            fh.write(TC.astype(np.float32).tobytes())
            fh.write(F.tobytes())
        return len(V), len(F)


class UV:
    """Texture coordinates for one material: per-face stretch (repeat) or world-uniform tiles."""

    def __init__(self, mat_attrs: dict):
        rep = [float(x) for x in mat_attrs.get("texrepeat", "1 1").split()]
        self.uniform = mat_attrs.get("texuniform") == "true"
        self.ru, self.rv = rep[0], rep[1]
        self.tu, self.tv = (2.0 / rep[0], 2.0 / rep[1]) if self.uniform else (1, 1)

    def face(self, p, U, V, su, sv, hu, hv):
        if self.uniform:
            return (float(p @ U) / self.tu, -float(p @ V) / self.tv)
        return ((su + 1) / 2 * self.ru, (1 - sv) / 2 * self.rv)

    def wrap(self, fu, fv, arc, z):
        if self.uniform:
            return (arc / self.tu, -z / self.tv)
        return (fu * self.ru, fv * self.rv)

    def disc(self, x, y):
        return ((x + 1) / 2 * self.ru, (1 - y) / 2 * self.rv)


def add_geom(mesh: Mesh, g: dict, uv: UV):
    t, size = g["type"], g["size"]
    if "fromto" in g:
        R, c, hl = frame_from_axis(g["fromto"][:3], g["fromto"][3:])
        if t == "cylinder":
            mesh.cylinder(R, c, size[0], hl, uv)
        elif t == "capsule":
            mesh.ellipsoid(R, c, np.array([size[0]] * 3), uv, half_len=hl)
        else:
            raise ValueError(f"fromto not supported for {t}")
        return
    R, c = quat_to_mat(g["quat"]), np.asarray(g["pos"], float)
    if t == "box":
        mesh.box(R, c, np.asarray(size, float), uv)
    elif t == "cylinder":
        mesh.cylinder(R, c, size[0], size[1], uv)
    elif t == "sphere":
        mesh.ellipsoid(R, c, np.array([size[0]] * 3), uv)
    elif t == "ellipsoid":
        mesh.ellipsoid(R, c, np.asarray(size, float), uv)
    elif t == "capsule":
        mesh.ellipsoid(R, c, np.array([size[0]] * 3), uv, half_len=size[1])
    else:
        raise ValueError(f"cannot bake geom type {t}")
