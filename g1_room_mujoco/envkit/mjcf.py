"""
A small MJCF scene builder for static environments made of primitive geoms.

One scene description is written out twice:

    VISUAL  every piece of furniture collides with its real geoms
    NAV     the same render geometry, but each furniture *item* collides through one
            invisible bounding box (group 4), which is easier for G1 path planning

Geoms are placed through a Python-side transform stack (`frame`), so furniture can be
written once in a canonical frame (front faces -y) and dropped anywhere with a yaw.
Everything ends up as flat geoms in <worldbody>, except free-floating `dynamic` bodies.

Roles decide group and collision:

    struct   group 0, collides in both files      (floor, walls, built-ins, seats)
    solid    group 0, collides in VISUAL; in NAV its item's proxy box collides instead
    deco     group 2, never collides               (all the small detail)
    ceiling  group 1, never collides               (press 1 in the viewer to hide for top views)
    dyn      group 0, collides in both             (geoms of free bodies you can push/lift)
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from pathlib import Path

import numpy as np

GROUP = {"struct": 0, "solid": 0, "deco": 2, "ceiling": 1, "dyn": 0, "proxy": 4}


def fmt(v, nd=4) -> str:
    s = f"{float(v):.{nd}f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def vec(v, nd=4) -> str:
    return " ".join(fmt(x, nd) for x in v)


def rot(rx=0.0, ry=0.0, rz=0.0) -> np.ndarray:
    """Rotation matrix: about x, then y, then z (fixed axes), angles in radians."""
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def quat(R: np.ndarray) -> np.ndarray:
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q) * (1 if q[0] >= 0 else -1)


# axis name -> rotation taking the local z axis onto it (for cylinders/capsules)
AXIS = {"z": np.eye(3), "x": rot(ry=math.pi / 2), "y": rot(rx=-math.pi / 2)}


class Scene:
    def __init__(self, model: str, prefix: str):
        self.model = model
        self.p = prefix
        self.textures: list[dict] = []
        self.materials: dict[str, dict] = {}
        self.world: list = []          # emitted elements (dicts) in worldbody
        self.lights: list[dict] = []
        self.cameras: list[dict] = []
        self.proxies: list[tuple] = []  # (name, lo, hi)
        self._T = [(np.eye(3), np.zeros(3))]
        self._container = self.world
        self._in_body = False
        self._item: dict | None = None
        self._count: dict[str, int] = {}
        self.visual = dict(azimuth=150, elevation=-25)
        self.statistic = dict(center=(0, 0, 1), extent=10)
        self.headlight = dict(ambient=(.3, .3, .3), diffuse=(.35, .35, .35), specular=(.05, .05, .05))
        self.skybox = dict(rgb1=(.62, .76, .92), rgb2=(.93, .95, .97))
        self.haze = (.85, .87, .9, 1)
        self.meta: dict = {}
        self.default_camera: str | None = None  # viewer opens on this camera instead of the free camera

    # ---------------------------------------------------------------- assets
    def tex(self, name: str, file: str):
        self.textures.append(dict(name=self.p + name, file=file))

    def mat(self, name: str, rgba=(1, 1, 1, 1), tex: str | None = None, tile=None, repeat=None,
            spec=0.3, shin=0.3, refl=0.0, emission=0.0):
        """tile=(u, v) metres per texture repeat (spatially uniform); repeat=(u, v) per face instead."""
        a = dict(name=self.p + name, rgba=vec(rgba, 3), specular=fmt(spec, 3), shininess=fmt(shin, 3))
        if refl:
            a["reflectance"] = fmt(refl, 3)
        if emission:
            a["emission"] = fmt(emission, 3)
        if tex:
            a["texture"] = self.p + tex
            if tile is not None:
                tu, tv = (tile, tile) if np.isscalar(tile) else tile
                a["texuniform"] = "true"
                a["texrepeat"] = vec((2.0 / tu, 2.0 / tv), 4)  # measured: one repeat = 2 m / texrepeat
            else:
                a["texrepeat"] = vec(repeat or (1, 1), 3)
        self.materials[name] = a
        return name

    # ---------------------------------------------------------------- transforms
    @contextmanager
    def frame(self, pos=(0, 0, 0), yaw: float = 0.0, R: np.ndarray | None = None):
        R0, t0 = self._T[-1]
        Rl = rot(rz=yaw) if R is None else R
        self._T.append((R0 @ Rl, R0 @ np.asarray(pos, float) + t0))
        try:
            yield
        finally:
            self._T.pop()

    def to_world(self, p) -> np.ndarray:
        R, t = self._T[-1]
        return R @ np.asarray(p, float) + t

    @contextmanager
    def item(self, name: str, proxy: bool = True, pad: float = 0.0):
        """Group the furniture geoms of one object; NAV gets one bounding-box blocker for it."""
        prev = self._item
        self._item = dict(name=name, lo=np.full(3, np.inf), hi=np.full(3, -np.inf), proxy=proxy, n=0)
        try:
            yield
        finally:
            it = self._item
            self._item = prev
            if it["proxy"] and it["n"]:
                lo, hi = it["lo"] - pad, it["hi"] + pad
                lo[2] = max(lo[2], 0.0)
                self.proxies.append((self._uniq(f"proxy_{it['name']}"), lo, hi))

    @contextmanager
    def dynamic(self, name: str, pos, yaw: float = 0.0):
        """A free body (pushable, liftable). Geoms inside are relative to the body."""
        assert not self._in_body, "nested dynamic bodies are not supported"
        R0, t0 = self._T[-1]
        Rb = R0 @ rot(rz=yaw)
        body = dict(tag="body", name=self._uniq(name), pos=R0 @ np.asarray(pos, float) + t0, quat=quat(Rb), children=[])
        self.world.append(body)
        saved_T, saved_c = self._T, self._container
        self._T, self._container, self._in_body = [(np.eye(3), np.zeros(3))], body["children"], True
        try:
            yield
        finally:
            self._T, self._container, self._in_body = saved_T, saved_c, False

    def _uniq(self, base: str) -> str:
        base = self.p + base
        n = self._count.get(base, 0)
        self._count[base] = n + 1
        return base if n == 0 else f"{base}_{n}"

    # ---------------------------------------------------------------- geoms
    def geom(self, type: str, size, pos=(0, 0, 0), mat: str | None = None, role: str = "deco",
             R: np.ndarray | None = None, rgba=None, name: str | None = None, fromto=None, **extra):
        if self._in_body and role in ("struct", "solid"):
            role = "dyn"
        Rw, t = self._T[-1]
        g = dict(tag="geom", type=type, role=role, size=np.atleast_1d(np.asarray(size, float)))
        if fromto is not None:
            a, b = self.to_world(fromto[0]), self.to_world(fromto[1])
            g["fromto"] = np.concatenate([a, b])
            ext_lo = np.minimum(a, b) - g["size"][0]
            ext_hi = np.maximum(a, b) + g["size"][0]
        else:
            Rg = Rw @ (np.eye(3) if R is None else R)
            c = Rw @ np.asarray(pos, float) + t
            g["pos"], g["quat"] = c, quat(Rg)
            sz = g["size"]
            if type in ("box", "ellipsoid"):
                h = sz
            elif type == "cylinder":
                h = np.array([sz[0], sz[0], sz[1]])
            elif type == "capsule":
                h = np.array([sz[0], sz[0], sz[1] + sz[0]])
            elif type == "sphere":
                h = np.repeat(sz[0], 3)
            else:
                h = np.zeros(3)
            e = np.abs(Rg) @ h
            ext_lo, ext_hi = c - e, c + e
        if mat is not None:
            assert mat in self.materials, f"unknown material {mat}"
            g["material"] = self.p + mat
        if rgba is not None:
            g["rgba"] = rgba
        g["name"] = self._uniq(name or (self._item["name"] if self._item else type))
        g.update(extra)
        if self._item is not None and role == "solid":
            self._item["lo"] = np.minimum(self._item["lo"], ext_lo)
            self._item["hi"] = np.maximum(self._item["hi"], ext_hi)
            self._item["n"] += 1
        g["in_item"] = self._item is not None and self._item["proxy"]
        self._container.append(g)
        return g

    def box(self, lo, hi, mat=None, role="deco", **kw):
        """Axis-aligned (in the current frame) box between two corners."""
        a, b = np.asarray(lo, float), np.asarray(hi, float)
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        if np.any(hi - lo <= 0):
            raise ValueError(f"degenerate box {a} {b}")
        return self.geom("box", (hi - lo) / 2, (hi + lo) / 2, mat, role, **kw)

    def boxc(self, c, half, mat=None, role="deco", R=None, **kw):
        return self.geom("box", half, c, mat, role, R=R, **kw)

    def cyl(self, c, r, hh, mat=None, role="deco", axis="z", R=None, **kw):
        return self.geom("cylinder", (r, hh), c, mat, role, R=AXIS[axis] if R is None else R, **kw)

    def cyl_ft(self, a, b, r, mat=None, role="deco", **kw):
        return self.geom("cylinder", (r,), mat=mat, role=role, fromto=(a, b), **kw)

    def cap_ft(self, a, b, r, mat=None, role="deco", **kw):
        return self.geom("capsule", (r,), mat=mat, role=role, fromto=(a, b), **kw)

    def sphere(self, c, r, mat=None, role="deco", **kw):
        return self.geom("sphere", (r,), c, mat, role, **kw)

    def ell(self, c, radii, mat=None, role="deco", R=None, **kw):
        return self.geom("ellipsoid", radii, c, mat, role, R=R, **kw)

    def proxy(self, name, lo, hi):
        """Explicit NAV-only blocker in world coordinates."""
        self.proxies.append((self._uniq(f"proxy_{name}"), np.asarray(lo, float), np.asarray(hi, float)))

    # ---------------------------------------------------------------- lights & cameras
    def light(self, name, pos, dir=(0, 0, -1), diffuse=(.5, .5, .5), cutoff=60, exponent=8,
              castshadow=None, directional=False, attenuation=None, specular=(.1, .1, .1), ambient=None):
        """Directional lights never cast shadows here; spotlights always do.

        MuJoCo 4.0.1's classic renderer drops ALL ambient light on surfaces outside the cone of a
        spotlight with castshadow="false" (ceilings and upper walls render black), so such lights
        are refused. Each shadow-casting spotlight costs one extra pass over every geom, so use few.
        """
        if castshadow is None:
            castshadow = not directional
        if not directional and not castshadow:
            raise ValueError(f"light {name}: non-shadow spotlights black out ambient light in MuJoCo 4.0.1")
        a = dict(name=self.p + name, pos=vec(self.to_world(pos), 3), dir=vec(self._T[-1][0] @ np.asarray(dir, float), 4),
                 diffuse=vec(diffuse, 3), specular=vec(specular, 3), castshadow="true" if castshadow else "false")
        if directional:
            a["directional"] = "true"
        else:
            a["cutoff"] = fmt(cutoff, 2)
            a["exponent"] = fmt(exponent, 2)
        if attenuation:
            a["attenuation"] = vec(attenuation, 4)
        if ambient:
            a["ambient"] = vec(ambient, 3)
        self.lights.append(a)

    def camera(self, name, pos, lookat, fovy=60, ortho=False):
        """fovy is degrees, or the visible height in metres when ortho=True."""
        p, l = np.asarray(pos, float), np.asarray(lookat, float)
        f = (l - p) / np.linalg.norm(l - p)
        x = np.cross(f, [0, 0, 1])
        x = x / np.linalg.norm(x)
        y = np.cross(x, f)
        cam = dict(name=self.p + name, pos=vec(p, 3), xyaxes=vec(np.concatenate([x, y]), 4), fovy=fmt(fovy, 2))
        if ortho:
            cam["projection"] = "orthographic"
        self.cameras.append(cam)

    # ---------------------------------------------------------------- output
    def _geom_xml(self, g: dict, variant: str, collision_only: bool = False) -> str | None:
        """collision_only: emit an untextured group-3 stand-in (its visual was baked into a mesh)."""
        role = g["role"]
        collide = role in ("struct", "dyn") or (role == "solid" and (variant == "visual" or not g["in_item"]))
        a = [f'name="{g["name"]}"', f'type="{g["type"]}"']
        if "fromto" in g:
            a.append(f'fromto="{vec(g["fromto"])}"')
        else:
            a.append(f'pos="{vec(g["pos"])}"')
            if not np.allclose(g["quat"], [1, 0, 0, 0], atol=1e-7):
                a.append(f'quat="{vec(g["quat"], 6)}"')
        a.append(f'size="{vec(g["size"])}"')
        if collision_only:
            a.append('rgba="0.9 0.5 0.1 0.4" group="3"')
        else:
            if "material" in g:
                a.append(f'material="{g["material"]}"')
            if "rgba" in g:
                a.append(f'rgba="{vec(g["rgba"], 3)}"')
            a.append(f'group="{GROUP[role]}"')
        a.append('contype="1" conaffinity="1"' if collide else 'contype="0" conaffinity="0"')
        for k, v in g.items():
            if k in ("mass", "density", "friction", "condim", "priority", "solref", "solimp", "margin"):
                a.append(f'{k}="{v if isinstance(v, str) else (vec(v) if np.ndim(v) else fmt(v, 5))}"')
        return "<geom " + " ".join(a) + "/>"

    def to_xml(self, variant: str = "visual", bake_dir: Path | None = None, bake_rel: str = "assets/baked") -> str:
        """bake_dir: write deco/ceiling/solid visuals as merged meshes into this folder (see bake.py)."""
        from .bake import UV, Mesh, add_geom

        assert variant in ("visual", "nav")
        by_full = {m["name"]: m for m in self.materials.values()}
        world, buckets = [], {}
        for e in self.world:
            if e["tag"] == "geom":
                bakeable = bake_dir is not None and e["role"] in ("deco", "ceiling", "solid") and "material" in e
                if bakeable:
                    key = (e["material"], GROUP[e["role"]])
                    if key not in buckets:
                        buckets[key] = (Mesh(), UV(by_full[e["material"]]))
                    add_geom(buckets[key][0], e, buckets[key][1])
                    if e["role"] == "solid" and (variant == "visual" or not e["in_item"]):
                        world.append("    " + self._geom_xml(e, variant, collision_only=True))
                    continue
                world.append("    " + self._geom_xml(e, variant))
            else:
                world.append(f'    <body name="{e["name"]}" pos="{vec(e["pos"])}" quat="{vec(e["quat"], 6)}">')
                world.append(f'      <freejoint name="{e["name"]}_free"/>')
                for g in e["children"]:
                    world.append("      " + self._geom_xml(g, variant))
                world.append("    </body>")
        if variant == "nav":
            for name, lo, hi in self.proxies:
                c, h = (lo + hi) / 2, np.maximum((hi - lo) / 2, 0.005)
                world.append(f'    <geom name="{name}" type="box" pos="{vec(c)}" size="{vec(h)}" rgba="1 0.15 0.1 0.3" '
                             f'group="4" contype="1" conaffinity="1"/>')
        mesh_assets, baked_mats, baked_geoms = [], {}, []
        self.bake_info = dict(meshes=0, vertices=0, triangles=0, geoms_baked=0)
        for (mat, grp), (mesh, _) in sorted(buckets.items()):
            stem = f"{mat}_g{grp}"
            nv, nf = mesh.write(Path(bake_dir) / f"{stem}.msh")
            self.bake_info["meshes"] += 1
            self.bake_info["vertices"] += nv
            self.bake_info["triangles"] += nf
            self.bake_info["geoms_baked"] += mesh.count
            src = by_full[mat]
            bmat = mat
            if "texture" in src:  # UVs already carry the tiling: same look, repeat 1 1
                bmat = mat.replace(self.p, self.p + "bk_", 1)
                attrs = {k: v for k, v in src.items() if k not in ("texuniform", "texrepeat")}
                attrs["name"] = bmat
                baked_mats[bmat] = attrs
            mesh_assets.append(f'    <mesh name="{self.p}bk_{stem[len(self.p):]}" file="{bake_rel}/{stem}.msh" inertia="shell"/>')
            baked_geoms.append(f'    <geom name="{self.p}baked_{stem[len(self.p):]}" type="mesh" mesh="{self.p}bk_{stem[len(self.p):]}" '
                               f'material="{bmat}" group="{grp}" contype="0" conaffinity="0"/>')

        hl = self.headlight
        cam_attr = ""
        if self.default_camera:
            names = [c["name"] for c in self.cameras]
            cam_attr = f' cameraid="{names.index(self.p + self.default_camera)}"'
        out = [f'<mujoco model="{self.model}_{variant}">',
               '  <compiler angle="radian" autolimits="true"/>',
               '  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>',
               '  <visual>',
               f'    <headlight ambient="{vec(hl["ambient"], 3)}" diffuse="{vec(hl["diffuse"], 3)}" specular="{vec(hl["specular"], 3)}"/>',
               f'    <global azimuth="{self.visual["azimuth"]}" elevation="{self.visual["elevation"]}" offwidth="1920" offheight="1080"{cam_attr}/>',
               '    <quality shadowsize="4096" offsamples="8" numslices="40" numstacks="24"/>',
               '    <map znear="0.002" zfar="40" haze="0.12" shadowclip="1.5"/>',
               f'    <rgba haze="{vec(self.haze, 3)}"/>',
               '  </visual>',
               f'  <statistic center="{vec(self.statistic["center"], 3)}" extent="{fmt(self.statistic["extent"], 2)}"/>',
               '  <asset>',
               f'    <texture name="{self.p}sky" type="skybox" builtin="gradient" rgb1="{vec(self.skybox["rgb1"], 3)}" '
               f'rgb2="{vec(self.skybox["rgb2"], 3)}" width="512" height="3072"/>']
        for t in self.textures:
            out.append(f'    <texture name="{t["name"]}" type="2d" file="{t["file"]}"/>')
        for m in list(self.materials.values()) + list(baked_mats.values()):
            out.append("    <material " + " ".join(f'{k}="{v}"' for k, v in m.items()) + "/>")
        out += mesh_assets
        out += ['  </asset>', '  <worldbody>']
        for l in self.lights:
            out.append("    <light " + " ".join(f'{k}="{v}"' for k, v in l.items()) + "/>")
        for c in self.cameras:
            out.append("    <camera " + " ".join(f'{k}="{v}"' for k, v in c.items()) + "/>")
        out += baked_geoms + world
        out += ['  </worldbody>', '</mujoco>', '']
        return "\n".join(out)

    def write(self, path: Path, variant: str, bake: bool = True):
        path = Path(path)
        bake_dir = path.parent / "assets" / "baked" if bake else None
        path.write_text(self.to_xml(variant, bake_dir=bake_dir))

    def stats(self) -> dict:
        geoms = [e for e in self.world if e["tag"] == "geom"]
        bodies = [e for e in self.world if e["tag"] == "body"]
        by_role: dict[str, int] = {}
        for g in geoms + [c for b in bodies for c in b["children"]]:
            by_role[g["role"]] = by_role.get(g["role"], 0) + 1
        return dict(geoms=sum(by_role.values()), by_role=by_role, nav_proxies=len(self.proxies),
                    dynamic_bodies=[b["name"] for b in bodies], materials=len(self.materials),
                    textures=len(self.textures), lights=len(self.lights), cameras=[c["name"] for c in self.cameras])
