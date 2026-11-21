#!/usr/bin/env python3
"""
Generate the detailed distribution warehouse for MuJoCo + Unitree G1.

    python g1_room_mujoco/Warehouse_G1_Detail/build_warehouse.py            # textures (if missing) + XMLs
    python g1_room_mujoco/Warehouse_G1_Detail/build_warehouse.py --textures # regenerate textures too
    python g1_room_mujoco/Warehouse_G1_Detail/build_warehouse.py --preview  # also render previews/*.png

Layout (metres, z up, interior x -15..15, y -12..16, eaves 9 m):
    south      4 dock doors (dock 2 open onto a loaded trailer), drive-in door, staging lanes,
               forklift, pallet jacks, stretch wrapper, empty-pallet stacks, cardboard bales
    centre     4 m main aisle (x -2..2) kept clear; painted ROBOT TEST ZONE around the spawn
               (0, -5.65) facing +y; lift station with liftable cartons and a tote just east of it
    east       three runs of selective pallet racking, 5 load levels, location labels
    west       4 pack stations + roller conveyor, 3 rows of bin shelving, glass office, break area
    overhead   roof deck, girders, bar joists, high-bay LEDs, skylights, duct, sprinkler mains
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from envkit import props as P  # noqa: E402
from envkit import textures as T  # noqa: E402
from envkit.mjcf import Scene, rot  # noqa: E402

SPAWN = (0.0, -5.65, 90.0)
X0, X1, Y0, Y1 = -15.0, 15.0, -12.0, 16.0
HW, WT = 9.0, 0.25          # wall height, wall thickness
ROOF = 9.6
PI = math.pi
DOCKS = [(-11.0, "1"), (-7.0, "2"), (7.0, "3"), (11.0, "4")]
OPEN_DOCK = -7.0
COLS_X, COLS_Y = (-9.75, -2.5, 2.5, 9.75), (-6.0, 4.0, 14.0)

# ---------------------------------------------------------------- textures


def _cork(w=640, h=420):
    from PIL import ImageDraw
    img = T.plaster(w, h, (.70, .52, .34), seed=90, amount=.18)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w - 1, h - 1], outline=(120, 85, 50), width=14)
    rng = np.random.default_rng(91)
    for k in range(9):
        x, y = 30 + (k % 3) * 200 + rng.integers(0, 30), 30 + (k // 3) * 125 + rng.integers(0, 20)
        col = [(250, 250, 245), (255, 240, 120), (180, 220, 255), (255, 200, 200)][k % 4]
        d.rectangle([x, y, x + 150, y + 100], fill=col)
        for j in range(4):
            d.line([x + 10, y + 20 + 18 * j, x + 40 + rng.integers(40, 120), y + 20 + 18 * j], fill=(70, 70, 80), width=3)
        d.ellipse([x + 70, y - 6, x + 82, y + 6], fill=(200, 40, 40))
    return img


def _bin(color):
    from PIL import ImageDraw
    img = T.plaster(256, 256, color, seed=92, amount=.03)
    d = ImageDraw.Draw(img)
    d.rectangle([60, 70, 196, 150], fill=(248, 248, 245))
    T.fit_text(d, (66, 74, 190, 110), "B-14-03", (20, 20, 20))
    T.barcode(d, (72, 114, 184, 144), np.random.default_rng(93))
    d.rectangle([0, 0, 255, 20], fill=tuple(int(c * 200) for c in color))
    return img


def _carton_stack(seed):
    """Side view of a pallet load of cartons: 2 wide x 3 high with seams and labels."""
    from PIL import ImageDraw
    img = T.cardboard(512, 512, seed=seed, label=False, arrows=False)
    d = ImageDraw.Draw(img)
    rng = np.random.default_rng(seed)
    for k in (1, 2):
        d.line([0, 512 * k / 3, 512, 512 * k / 3], fill=(95, 70, 45), width=6)
    d.line([256, 0, 256, 512], fill=(95, 70, 45), width=6)
    for r in range(3):
        for c in range(2):
            x0, y0 = c * 256 + 30, r * 171 + 95
            d.rectangle([x0, y0, x0 + 90, y0 + 50], fill=(240, 238, 230))
            T.barcode(d, (x0 + 6, y0 + 8, x0 + 84, y0 + 42), rng)
            d.rectangle([c * 256, r * 171 + 70, c * 256 + 256, r * 171 + 84], fill=(205, 170, 115))
    return img


TEXTURES = {
    "floor_concrete": lambda: T.concrete(1024, 1024, seed=1, tone=(.66, .66, .64), stains=7),
    "wall_panel": lambda: T.ribbed(512, 512, (.86, .87, .88), ribs=6, seed=2),
    "precast": lambda: T.concrete(512, 512, seed=3, tone=(.58, .57, .55), joints=False, stains=3, polish=False),
    "roof_deck": lambda: T.ribbed(512, 512, (.80, .81, .83), ribs=8, seed=4, depth=.3, joint=False),
    "upright": lambda: T.perforated(128, 128, (.12, .30, .62), seed=5),
    "pallet_top": lambda: T.pallet_top(512, 512, seed=6),
    "pallet_wood": lambda: T.wood_grain(512, 256, (.80, .68, .50), (.58, .44, .28), seed=7, rings=12, knots=2),
    "carton_a": lambda: T.cardboard(512, 512, seed=8, logo="BLUEPEAK SUPPLY"),
    "carton_b": lambda: T.cardboard(512, 512, seed=9, tone=(.78, .62, .42), logo="FRAGILE"),
    "carton_c": lambda: T.cardboard(512, 512, seed=10, tone=(.62, .46, .30), tape=(.55, .70, .85), logo="NORTHWIND"),
    "carton_stack_a": lambda: _carton_stack(11),
    "carton_stack_b": lambda: _carton_stack(12),
    "hazard": lambda: T.hazard(512, 128, stripes=4),
    "diamond_plate": lambda: T.diamond_plate(512, 512, n=14, seed=13),
    "dock_door": lambda: T.sectional_door(512, 640, sections=5, windows_row=1, seed=14),
    "wire_deck": lambda: T.grille(256, 256, color=(.55, .57, .60), slats=10, dark=(.25, .26, .28)),
    "butcher": lambda: T.wood_grain(512, 512, (.86, .70, .50), (.64, .46, .28), seed=15, rings=14, knots=0),
    "plywood": lambda: T.wood_grain(512, 512, (.82, .70, .52), (.66, .52, .34), seed=16, rings=5, knots=4),
    "rubber_mat": lambda: T.rubber_mat(256, 256),
    "screen_a": lambda: T.screen_ui(640, 400, seed=17),
    "screen_b": lambda: T.screen_ui(640, 400, seed=18, title="Inventory  |  Cycle count  |  Zone B", accent=(.15, .55, .35)),
    "vending": lambda: T.vending_front(400, 800, seed=19),
    "bin_blue": lambda: _bin((.15, .35, .70)),
    "bin_red": lambda: _bin((.75, .18, .15)),
    "bin_yellow": lambda: _bin((.92, .72, .12)),
    "cork": _cork,
    "test_zone": lambda: T.floor_stencil(1024, 320, "ROBOT TEST ZONE", (.12, .30, .62), (.96, .96, .96)),
    "whiteboard": lambda: T.whiteboard(800, 480, seed=20, lines=[("SHIFT 2  -  OUTBOUND", (30, 60, 170)), ("Wave 14: 212 orders", (30, 30, 30)),
                                                                   ("Dock 2 loading  - trailer 4471", (30, 30, 30)), ("Cycle count: aisle 02", (200, 40, 40)),
                                                                   ("Robot trial: test zone 14:00", (30, 120, 60))]),
    "banner": lambda: T.sign(2048, 256, [("GETUP  ·  ROBOT TEST FACILITY", 1.0)], (.10, .12, .16), (.96, .96, .96), stripe=((.95, .70, .10), .07)),
    "sign_exit": lambda: T.sign(512, 192, [("EXIT", 1.0)], (.08, .45, .20), (1, 1, 1)),
    "sign_forklift": lambda: T.sign(768, 512, [("CAUTION", .28), ("FORKLIFT TRAFFIC", .22), ("PEDESTRIANS USE", .17), ("MARKED WALKWAYS", .17)],
                                    (.98, .82, .10), (.08, .08, .08), border=(.08, .08, .08)),
    "sign_speed": lambda: T.sign(400, 520, [("SPEED", .2), ("LIMIT", .2), ("5", .38), ("MPH", .14)], (1, 1, 1), (.05, .05, .05), border=(.05, .05, .05)),
    "sign_ppe": lambda: T.sign(640, 400, [("NOTICE", .25), ("HI-VIS VEST &", .2), ("SAFETY SHOES", .2), ("REQUIRED", .2)], (.12, .35, .70), (1, 1, 1),
                               border=(1, 1, 1)),
    "sign_fire": lambda: T.sign(300, 300, [("FIRE", .3), ("EXTINGUISHER", .2)], (.80, .10, .08), (1, 1, 1)),
    "sign_incident": lambda: T.sign(640, 420, [("SAFETY FIRST", .2), ("DAYS WITHOUT", .17), ("AN INCIDENT", .17), ("128", .36)], (.10, .45, .22), (1, 1, 1),
                                    border=(1, 1, 1)),
    "sign_aisle_02": lambda: T.sign(640, 300, [("AISLE 02", .6), ("A-07 … A-12", .3)], (.12, .30, .62), (1, 1, 1), border=(1, 1, 1)),
    "sign_aisle_03": lambda: T.sign(640, 300, [("AISLE 03", .6), ("A-13 … A-18", .3)], (.12, .30, .62), (1, 1, 1), border=(1, 1, 1)),
    "sign_aisle_01": lambda: T.sign(640, 300, [("AISLE 01", .6), ("A-01 … A-06", .3)], (.12, .30, .62), (1, 1, 1), border=(1, 1, 1)),
    "sign_pack": lambda: T.sign(800, 260, [("PACK & SHIP", .7)], (.10, .12, .16), (.98, .82, .10)),
    "sign_pick": lambda: T.sign(800, 260, [("PICK  ZONE  B", .7)], (.10, .12, .16), (.98, .82, .10)),
    **{f"dock_{n}": (lambda n=n: T.sign(512, 256, [(f"DOCK {n}", .8)], (.98, .82, .10), (.05, .05, .05), border=(.05, .05, .05))) for _, n in DOCKS},
    **{f"labels_{k}": (lambda k=k: T.rack_labels(1024, 64, [f"A-{k * 2 + 1:02d}-{j}" for j in range(1, 3)], seed=30 + k)) for k in range(4)},
    **{f"placard_{k}": (lambda k=k: T.sign(256, 384, [("ROW", .15), (f"A{k + 1}", .45), ("NORTH", .12)], (.12, .30, .62), (1, 1, 1), border=(1, 1, 1))) for k in range(5)},
}


def make_textures(assets: Path, force: bool):
    assets.mkdir(parents=True, exist_ok=True)
    for name, fn in TEXTURES.items():
        p = assets / f"{name}.png"
        if force or not p.exists():
            T.save(fn(), p)


# ---------------------------------------------------------------- materials

def materials(s: Scene):
    for name in TEXTURES:
        s.tex(name, f"assets/{name}.png")
    P.base_materials(s)
    m = s.mat
    m("floor", tex="floor_concrete", tile=6.0, spec=.35, shin=.5)
    m("wall_panel", tex="wall_panel", tile=1.5, spec=.3, shin=.4)
    m("precast", tex="precast", tile=3.0, spec=.05, shin=.1)
    m("roof_deck", tex="roof_deck", tile=1.2, spec=.2, shin=.3)
    m("upright", tex="upright", tile=0.12, spec=.5, shin=.5)
    m("beam_orange", (.90, .40, .10, 1), spec=.5, shin=.5)
    m("pallet_top", tex="pallet_top", spec=.05, shin=.1)
    m("pallet_wood", tex="pallet_wood", spec=.05, shin=.1)
    m("wire_deck", tex="wire_deck", repeat=(6, 3), spec=.4, shin=.4)
    m("wrap", (.92, .94, .97, .38), spec=1, shin=.9)
    m("hazard", tex="hazard", spec=.2, shin=.3)
    m("hazard_tile", tex="hazard", tile=(0.8, 0.2), spec=.2, shin=.3)
    m("diamond_plate", tex="diamond_plate", tile=0.6, spec=.6, shin=.5)
    m("dock_door", tex="dock_door", spec=.4, shin=.4)
    m("butcher", tex="butcher", spec=.2, shin=.3)
    m("plywood", tex="plywood", tile=1.2, spec=.1, shin=.2)
    m("rubber_mat", tex="rubber_mat", tile=0.25, spec=.05, shin=.1)
    m("test_zone", tex="test_zone", spec=.2, shin=.3)
    m("cork", tex="cork", spec=.05, shin=.1)
    for t in ("carton_a", "carton_b", "carton_c", "carton_stack_a", "carton_stack_b", "vending", "bin_blue", "bin_red",
              "bin_yellow", "whiteboard", "banner", "sign_forklift", "sign_speed", "sign_ppe", "sign_fire", "sign_incident",
              "sign_aisle_01", "sign_aisle_02", "sign_aisle_03", "sign_pack", "sign_pick",
              *[f"dock_{n}" for _, n in DOCKS], *[f"labels_{k}" for k in range(4)], *[f"placard_{k}" for k in range(5)]):
        m(t, tex=t, spec=.15, shin=.2)
    m("sign_exit", tex="sign_exit", emission=.8, spec=0)
    m("screen_a", tex="screen_a", emission=.7, spec=.2)
    m("screen_b", tex="screen_b", emission=.7, spec=.2)
    for name, c, sp in [
        ("steel", (.62, .64, .67, 1), .6), ("steel_dark", (.30, .32, .35, 1), .5), ("galvanized", (.72, .74, .76, 1), .5),
        ("column", (.58, .60, .63, 1), .4), ("safety_yellow", (.97, .76, .06, 1), .5), ("safety_orange", (.95, .42, .08, 1), .5),
        ("paint_yellow", (.98, .80, .10, 1), .2), ("paint_white", (.95, .95, .93, 1), .2), ("paint_green", (.20, .58, .32, 1), .2),
        ("paint_blue", (.12, .30, .62, 1), .2), ("paint_red", (.80, .12, .10, 1), .2),
        ("forklift_yellow", (.96, .70, .05, 1), .7), ("counterweight", (.16, .16, .17, 1), .4), ("seat_black", (.08, .08, .09, 1), .3),
        ("tire", (.05, .05, .05, 1), .1), ("drum_blue", (.10, .30, .65, 1), .6), ("sack", (.88, .84, .74, 1), .05),
        ("tote_grey", (.56, .59, .63, 1), .3), ("tote_blue", (.18, .38, .70, 1), .4), ("bench_grey", (.50, .53, .56, 1), .4),
        ("dark_glass", (.08, .10, .12, .85), .9), ("office_glass", (.75, .85, .90, .18), 1), ("aluminum", (.78, .79, .80, 1), .7),
        ("ceiling_tile", (.93, .93, .91, 1), .05), ("carpet", (.28, .30, .33, 1), .02), ("desk_top", (.78, .70, .58, 1), .3),
        ("chair_fabric", (.15, .17, .20, 1), .1), ("water_jug", (.45, .70, .95, .55), .9), ("duct", (.80, .81, .83, 1), .7),
        ("sprinkler_red", (.75, .10, .08, 1), .5), ("asphalt", (.22, .22, .23, 1), .05), ("trailer_wall", (.86, .87, .88, 1), .4),
        ("strap", (.15, .15, .15, 1), .2), ("coffee", (.17, .09, .04, 1), .6), ("terracotta", (.72, .40, .26, 1), .1),
        ("charger_grey", (.70, .72, .74, 1), .4), ("cable", (.08, .08, .08, 1), .2),
    ]:
        m(name, c, spec=sp, shin=sp)
    m("led_panel", (1, .98, .94, 1), emission=1.0, spec=0)
    m("skylight", (.92, .96, 1, 1), emission=.9, spec=0)
    m("clerestory", (.85, .92, .98, 1), emission=.55, spec=.5)
    m("amber_light", (1, .65, .1, 1), emission=1.0, spec=0)
    m("blue_light", (.2, .45, 1, 1), emission=1.0, spec=0)


# ---------------------------------------------------------------- shell

def wall_ns(s, ya, yb, x0, x1, openings, h=HW, mat="wall_panel"):
    lo, hi = min(ya, yb), max(ya, yb)
    xs = x0
    for a, b, za, zb in sorted(openings):
        if a > xs:
            s.box((xs, lo, 0), (a, hi, h), mat, "struct", name="wall")
        if za > 0:
            s.box((a, lo, 0), (b, hi, za), mat, "struct", name="wall")
        if zb < h:
            s.box((a, lo, zb), (b, hi, h), mat, "struct", name="wall")
        xs = b
    if xs < x1:
        s.box((xs, lo, 0), (x1, hi, h), mat, "struct", name="wall")


def wall_ew(s, xa, xb, y0, y1, h=HW, mat="wall_panel"):
    s.box((min(xa, xb), y0, 0), (max(xa, xb), y1, h), mat, "struct", name="wall")


def shell(s):
    s.box((X0 - WT, Y0 - WT, -0.2), (X1 + WT, Y1 + WT, 0), "floor", "struct", name="floor")
    south_open = [(x - 1.375, x + 1.375, 0, 3.05) for x, _ in DOCKS] + [(-1.8, 1.8, 0, 4.3), (-4.675, -3.725, 0, 2.15), (3.725, 4.675, 0, 2.15)]
    wall_ns(s, Y0 - WT, Y0, X0 - WT, X1 + WT, south_open)
    wall_ns(s, Y1, Y1 + WT, X0 - WT, X1 + WT, [])
    wall_ew(s, X0 - WT, X0, Y0, Y1)
    wall_ew(s, X1, X1 + WT, Y0, Y1)
    # precast concrete wainscot + clerestory band on the inside faces
    t = 0.03
    for a, b in [(X0, -12.375), (-9.625, -8.375), (-5.625, -4.675), (-3.725, -1.8), (1.8, 3.725), (4.675, 5.625), (8.375, 9.625), (12.375, X1)]:
        s.box((a, Y0, 0), (b, Y0 + t, 1.2), "precast")
    s.box((X0, Y1 - t, 0), (X1, Y1, 1.2), "precast")
    s.box((X0, Y0, 0), (X0 + t, Y1, 1.2), "precast")
    s.box((X1 - t, Y0, 0), (X1, Y1, 1.2), "precast")
    for k in range(10):
        ya = Y0 + 1.0 + k * 2.75
        if ya + 2.2 > Y1:
            break
        s.box((X0, ya, 7.2), (X0 + 0.035, ya + 2.2, 8.1), "clerestory")
        s.box((X1 - 0.035, ya, 7.2), (X1, ya + 2.2, 8.1), "clerestory")
        for xx, sg in ((X0, 1), (X1, -1)):
            s.box((xx, ya - 0.05, 7.15), (xx + sg * 0.06, ya + 2.25, 7.2), "aluminum")
            s.box((xx, ya - 0.05, 8.1), (xx + sg * 0.06, ya + 2.25, 8.15), "aluminum")
    # roof deck (group 1, hidden in top views)
    s.box((X0 - WT, Y0 - WT, ROOF), (X1 + WT, Y1 + WT, ROOF + 0.12), "roof_deck", "ceiling", name="roof")


def columns(s):
    for x in COLS_X:
        for y in COLS_Y:
            with s.item(f"column_{x}_{y}", proxy=False):
                fw, ft, d, wt = 0.15, 0.014, 0.15, 0.008
                s.box((x - fw, y - d, 0), (x + fw, y - d + ft, ROOF - 0.7), "column", "struct", name="column")
                s.box((x - fw, y + d - ft, 0), (x + fw, y + d, ROOF - 0.7), "column", "struct", name="column")
                s.box((x - wt, y - d, 0), (x + wt, y + d, ROOF - 0.7), "column", "struct", name="column")
                s.box((x - 0.25, y - 0.25, 0), (x + 0.25, y + 0.25, 0.02), "steel_dark")
                in_rack = abs(x) > 9 and y > -3
                if not in_rack:  # yellow column guard
                    s.box((x - 0.22, y - 0.22, 0), (x + 0.22, y + 0.22, 1.1), "safety_yellow", "struct", name="column_guard")
                    for k in range(3):
                        s.box((x - 0.225, y - 0.225, 0.2 + 0.32 * k), (x + 0.225, y + 0.225, 0.3 + 0.32 * k), "black_matte")


def roof_structure(s):
    zb = ROOF - 0.7
    for y in COLS_Y:  # girders along x
        s.box((X0, y - 0.12, ROOF - 0.62), (X1, y + 0.12, ROOF - 0.6), "column", "ceiling")
        s.box((X0, y - 0.12, ROOF - 0.02), (X1, y + 0.12, ROOF), "column", "ceiling")
        s.box((X0, y - 0.01, ROOF - 0.6), (X1, y + 0.01, ROOF - 0.02), "column", "ceiling")
    spans = [(Y0, COLS_Y[0]), (COLS_Y[0], COLS_Y[1]), (COLS_Y[1], COLS_Y[2]), (COLS_Y[2], Y1)]
    for x in np.arange(X0 + 1.5, X1, 2.0):  # bar joists along y
        for ya, yb in spans:
            s.box((x - 0.04, ya, ROOF - 0.06), (x + 0.04, yb, ROOF), "steel", "ceiling")
            s.box((x - 0.035, ya + 0.3, zb + 0.08), (x + 0.035, yb - 0.3, zb + 0.13), "steel", "ceiling")
            n = max(2, int((yb - ya) / 1.6))
            for k in range(n):
                y0 = ya + 0.3 + (yb - ya - 0.6) * k / n
                y1 = ya + 0.3 + (yb - ya - 0.6) * (k + 1) / n
                s.cyl_ft((x, y0, zb + 0.12), (x, (y0 + y1) / 2, ROOF - 0.05), 0.012, "steel", "ceiling")
                s.cyl_ft((x, (y0 + y1) / 2, ROOF - 0.05), (x, y1, zb + 0.12), 0.012, "steel", "ceiling")
    # skylights between joists
    for x in (-10.5, -4.5, 4.5, 10.5):
        for y in (-9.0, 9.0):
            s.box((x - 0.6, y - 1.5, ROOF - 0.01), (x + 0.6, y + 1.5, ROOF + 0.005), "skylight", "ceiling")
    # spiral duct with drops, sprinkler mains with branch lines
    s.cyl_ft((-8.0, Y0 + 1, 7.9), (-8.0, Y1 - 1, 7.9), 0.42, "duct", "ceiling")
    for y in np.arange(-9, 15, 5.0):
        s.cyl_ft((-8.0, y, 7.5), (-8.0, y, 7.0), 0.22, "duct", "ceiling")
        s.box((-8.3, y - 0.3, 6.9), (-7.7, y + 0.3, 7.0), "wire_deck", "ceiling")
    for y in (-9.0, -1.0, 7.0, 13.5):
        s.cyl_ft((X0 + 0.5, y, 8.45), (X1 - 0.5, y, 8.45), 0.06, "sprinkler_red", "ceiling")
        for x in np.arange(X0 + 2, X1 - 1, 3.0):
            s.cyl_ft((x, y - 1.4, 8.4), (x, y + 1.4, 8.4), 0.025, "sprinkler_red", "ceiling")
            for dy in (-1.2, 1.2):
                s.cyl((x, y + dy, 8.34), 0.012, 0.04, "brass", "ceiling")
    # high-bay LED fixtures
    spots = [(0.0, y) for y in (-9.5, -4.5, 0.5, 5.5, 10.5, 15.0)] + [(7.0, y) for y in (-1.0, 4.5, 10.0)] + \
            [(12.4, y) for y in (-1.0, 4.5, 10.0)] + [(x, y) for x in (-12.0, -6.5) for y in (-9.5, -4.5, 0.5, 5.5, 10.5)] + \
            [(x, -9.5) for x in (5.0, 10.0)]
    for x, y in spots:
        s.box((x - 0.3, y - 0.18, 7.95), (x + 0.3, y + 0.18, 8.05), "galvanized", "ceiling")
        s.box((x - 0.27, y - 0.15, 7.94), (x + 0.27, y + 0.15, 7.95), "led_panel", "ceiling")
        for dx in (-0.25, 0.25):
            s.cyl_ft((x + dx, y, 8.05), (x + dx, y, ROOF - 0.62), 0.004, "steel_dark", "ceiling")
    # unit heaters on the walls
    for y in (-6.0, 6.0):
        for x, sg in ((X0 + 0.55, 1), (X1 - 0.55, -1)):
            s.box((x - 0.35, y - 0.4, 6.0), (x + 0.35, y + 0.4, 6.7), "steel_dark", "ceiling")
            for k in range(5):
                s.box((x + sg * 0.351 - 0.005, y - 0.35, 6.08 + 0.11 * k), (x + sg * 0.351 + 0.005, y + 0.35, 6.12 + 0.11 * k), "black_matte", "ceiling")


# ---------------------------------------------------------------- floor markings

def floor_markings(s):
    z = 0.0012
    def line(a, b, w=0.1, mat="paint_yellow"):
        (xa, ya), (xb, yb) = a, b
        if xa == xb:
            s.box((xa - w / 2, min(ya, yb), 0), (xa + w / 2, max(ya, yb), z), mat)
        else:
            s.box((min(xa, xb), ya - w / 2, 0), (max(xa, xb), ya + w / 2, z), mat)
    for x in (-2.0, 2.0):  # main aisle
        line((x, -7.9), (x, -4.3))
        line((x, -2.6), (x, 15.9))
    for y in (-8.0, -4.2):  # cross aisle
        line((X0 + 0.1, y), (-2.05, y))
        line((2.05, y), (X1 - 0.1, y))
    # pedestrian walkway from the south personnel door (green with white edges) and a zebra crossing
    s.box((-3.7, Y0 + 0.05, 0), (-2.75, -4.3, z), "paint_green")
    line((-3.75, Y0 + 0.05), (-3.75, -4.3), w=0.08, mat="paint_white")
    line((-2.7, Y0 + 0.05), (-2.7, -4.3), w=0.08, mat="paint_white")
    for k in range(8):
        x = -1.85 + 0.5 * k
        s.box((x, -3.4, 0), (x + 0.28, -2.8, z), "paint_white")
    # staging lanes in front of the docks
    for x, _ in DOCKS:
        for dx in (-1.45, 1.45):
            line((x + dx, Y0 + 0.2), (x + dx, -8.2), w=0.08, mat="paint_white")
        s.box((x - 1.375, Y0 + 0.05, 0), (x + 1.375, Y0 + 0.35, z + 0.0005), "hazard_tile")
    # robot test zone around the spawn: blue field, white border, stencil
    zx0, zx1, zy0, zy1 = -1.5, 1.5, -7.2, -4.1
    s.box((zx0, zy0, 0), (zx1, zy1, z), "paint_blue")
    for a, b in (((zx0, zy0), (zx1, zy0)), ((zx0, zy1), (zx1, zy1)), ((zx0, zy0), (zx0, zy1)), ((zx1, zy0), (zx1, zy1))):
        line(a, b, w=0.08, mat="paint_white")
    s.box((zx0 + 0.15, zy0 + 0.12, 0), (zx1 - 0.15, zy0 + 0.6, z + 0.0004), "test_zone")
    for x in (zx0 - 0.12, zx1 + 0.04):
        s.box((x, zy0 - 0.12, 0), (x + 0.08, zy1 + 0.12, z + 0.0002), "hazard_tile")
    # forklift parking and charger keep-clear hatches
    for (xa, ya, xb, yb) in ((-14.9, -6.9, -12.7, -4.4), (-14.9, -11.9, -13.2, -10.4)):
        s.box((xa, ya, 0), (xb, yb, z), "hazard_tile")
    line((-8.6, -6.9), (-3.85, -6.9))   # forklift parking box
    line((-8.6, -5.4), (-3.85, -5.4))
    line((-8.6, -6.9), (-8.6, -5.4))


# ---------------------------------------------------------------- pallets & loads

def pallet(s, cx, cy, z, role="deco", full=True):
    s.box((cx - 0.5, cy - 0.6, z + 0.122), (cx + 0.5, cy + 0.6, z + 0.144), "pallet_top", role, name="pallet")
    for dx in (-0.46, 0.0, 0.46):
        s.box((cx + dx - 0.04, cy - 0.6, z + 0.022), (cx + dx + 0.04, cy + 0.6, z + 0.122), "pallet_wood", role, name="pallet")
    if full:
        for dy in (-0.55, 0.0, 0.55):
            s.box((cx - 0.5, cy + dy - 0.05, z), (cx + 0.5, cy + dy + 0.05, z + 0.022), "pallet_wood", role, name="pallet")
    return z + 0.144


def load(s, cx, cy, z, rng, role="deco", full=True, max_h=1.4):
    top = pallet(s, cx, cy, z, role, full)
    kind = rng.choice(["wrap", "cartons", "drums", "sacks", "totes"], p=[.36, .34, .1, .1, .1])
    h = float(rng.uniform(0.85, max_h - 0.15))
    if kind == "wrap" or (kind == "cartons" and not full):
        mat = rng.choice(["carton_stack_a", "carton_stack_b"]) if kind == "cartons" or rng.random() < .5 else rng.choice(["carton_a", "carton_b", "carton_c"])
        s.box((cx - 0.48, cy - 0.58, top), (cx + 0.48, cy + 0.58, top + h), mat, role, name="load")
        if kind == "wrap":
            s.box((cx - 0.495, cy - 0.595, top + 0.02), (cx + 0.495, cy + 0.595, top + h + 0.01), "wrap")
            s.box((cx - 0.12, cy - 0.596, top + h * 0.6), (cx + 0.12, cy - 0.594, top + h * 0.6 + 0.16), "paper_white")
    elif kind == "cartons":
        nz = int(rng.integers(2, 4))
        ch = min(0.42, h / nz)
        for k in range(nz):
            for i in range(2):
                for j in range(2):
                    if k == nz - 1 and rng.random() < 0.25:
                        continue
                    x0, y0 = cx - 0.48 + i * 0.48, cy - 0.58 + j * 0.58
                    s.box((x0 + 0.004, y0 + 0.004, top + k * ch), (x0 + 0.476, y0 + 0.576, top + (k + 1) * ch - 0.002),
                          rng.choice(["carton_a", "carton_b", "carton_c"]), role if k == 0 else "deco", name="carton")
    elif kind == "drums":
        for i in (-1, 1):
            for j in (-1, 1):
                x, y = cx + i * 0.245, cy + j * 0.3
                s.cyl((x, y, top + 0.44), 0.24, 0.44, "drum_blue", role, name="drum")
                for zz in (0.29, 0.59):
                    s.cyl((x, y, top + zz), 0.245, 0.012, "drum_blue")
                s.cyl((x + 0.12, y, top + 0.885), 0.03, 0.008, "black_matte")
    elif kind == "sacks":
        for k in range(3):
            for i in range(3):
                s.ell((cx - 0.32 + 0.32 * i, cy, top + 0.08 + 0.15 * k), (0.15, 0.56, 0.085), "sack", role if k == 0 else "deco",
                      R=rot(rz=(0.03 if k % 2 else -0.03)), name="sack")
    else:  # totes
        for k in range(3):
            for i in range(2):
                for j in range(2):
                    x0, y0 = cx - 0.48 + i * 0.48, cy - 0.58 + j * 0.58
                    s.box((x0 + 0.01, y0 + 0.01, top + k * 0.32), (x0 + 0.47, y0 + 0.57, top + k * 0.32 + 0.29), "tote_grey", role if k == 0 else "deco",
                          name="tote")
                    s.box((x0 + 0.008, y0 + 0.008, top + k * 0.32 + 0.29), (x0 + 0.472, y0 + 0.572, top + k * 0.32 + 0.315), "tote_blue")


# ---------------------------------------------------------------- racking

BAY, UPR, DEPTH = 2.75, 0.10, 1.1
LEVELS = (1.75, 3.5, 5.25, 7.0)
UPR_H = 7.9


def rack_side(s, nbays, seed, label_k, placard=None):
    """One single-deep run in canonical frame: runs along +x, aisle at -y, depth 0..DEPTH."""
    rng = np.random.default_rng(seed)
    for k in range(nbays + 1):
        x0 = k * (BAY + UPR)
        for y0 in (0.0, DEPTH - 0.08):
            s.box((x0, y0, 0), (x0 + UPR, y0 + 0.08, UPR_H), "upright", "solid", name="upright")
            s.box((x0 - 0.03, y0 - 0.02, 0), (x0 + UPR + 0.03, y0 + 0.1, 0.008), "galvanized")
        xm = x0 + UPR / 2
        for zz in (0.15, UPR_H - 0.2):
            s.cyl_ft((xm, 0.08, zz), (xm, DEPTH - 0.08, zz), 0.018, "upright")
        zs = np.linspace(0.15, UPR_H - 0.2, 7)
        for a, b in zip(zs[:-1], zs[1:]):
            ya, yb = (0.08, DEPTH - 0.08) if int(a * 10) % 2 == 0 else (DEPTH - 0.08, 0.08)
            s.cyl_ft((xm, ya, a), (xm, yb, b), 0.014, "upright")
        s.box((x0 - 0.02, -0.06, 0), (x0 + UPR + 0.02, 0.0, 0.45), "safety_yellow", "solid", name="upright_guard")
    for b in range(nbays):
        xa, xb = b * (BAY + UPR) + UPR, (b + 1) * (BAY + UPR)
        for li, lv in enumerate(LEVELS):
            s.box((xa, 0.0, lv - 0.12), (xb, 0.05, lv), f"labels_{(label_k + b + li) % 4}")
            s.box((xa, DEPTH - 0.05, lv - 0.12), (xb, DEPTH, lv), "beam_orange")
            s.box((xa + 0.01, 0.05, lv - 0.035), (xb - 0.01, DEPTH - 0.05, lv - 0.03), "wire_deck")
            for yy in (0.25, 0.55, 0.85):
                s.box((xa, yy - 0.02, lv - 0.08), (xb, yy + 0.02, lv - 0.035), "galvanized")
        for li, z in enumerate((0.0,) + LEVELS):
            for p in (0.25, 0.75):
                if li > 0 and rng.random() > 0.8:
                    continue
                cx = xa + (xb - xa) * p
                load(s, cx, DEPTH / 2, z, rng, role="solid" if li == 0 else "deco", full=li <= 1,
                     max_h=1.45 if li < len(LEVELS) else 1.25)
    if placard is not None:
        s.box((-0.02, 0.25, 1.6), (0.0, 0.85, 2.5), f"placard_{placard}", R=None)


def racking(s):
    nb = 6
    L = nb * (BAY + UPR) + UPR
    y_s = -3.05
    runs = [  # (x of aisle face, faces west?)
        (3.0, True), (3.0 + 2 * DEPTH + 0.3, False), (8.5, True), (8.5 + 2 * DEPTH + 0.3, False), (13.85, True)]
    for i, (xf, west) in enumerate(runs):
        with s.item(f"rack_run_{i}"):
            if west:   # aisle on the west (-x): local -y -> world -x, local x runs -y
                with s.frame((xf, y_s + L, 0), -PI / 2):
                    rack_side(s, nb, seed=100 + i, label_k=i)
            else:      # aisle on the east (+x): local x runs +y
                with s.frame((xf, y_s, 0), PI / 2):
                    rack_side(s, nb, seed=100 + i, label_k=i)
        # end-of-row protector and placard at the south end
        if west:
            s.box((xf, y_s - 0.25, 0), (xf + (2 * DEPTH + 0.3 if i < 4 else DEPTH), y_s - 0.1, 0.5), "safety_yellow", "struct", name="row_end_guard")
            with s.frame((xf + (DEPTH + 0.15 if i < 4 else DEPTH / 2), y_s - 0.26, 0)):
                s.box((-0.3, -0.02, 1.4), (0.3, 0.0, 2.3), f"placard_{min(i // 2, 4)}")
    # hanging aisle signs at the south end of each aisle
    for x, tex in ((2.5, "sign_aisle_01"), (7.0, "sign_aisle_02"), (12.4, "sign_aisle_03")):
        s.box((x - 0.8, -3.6, 5.2), (x + 0.8, -3.56, 5.95), tex)
        for dx in (-0.7, 0.7):
            s.cyl_ft((x + dx, -3.58, 5.95), (x + dx, -3.58, ROOF - 0.62), 0.005, "steel_dark", "ceiling")


# ---------------------------------------------------------------- docks

def docks(s):
    for x, n in DOCKS:
        opened = x == OPEN_DOCK
        w2 = 1.375
        with s.frame((x, Y0, 0)):
            # vertical + horizontal tracks, spring shaft
            for sx in (-1, 1):
                s.box((sx * (w2 + 0.02) - 0.04, 0.0, 0), (sx * (w2 + 0.02) + 0.04, 0.09, 3.35), "galvanized")
                s.box((sx * (w2 + 0.02) - 0.04, 0.09, 3.3), (sx * (w2 + 0.02) + 0.04, 3.4, 3.38), "galvanized")
                s.cyl_ft((sx * (w2 + 0.02), 3.4, 3.38), (sx * (w2 + 0.02), 3.4, 4.2), 0.008, "steel_dark", "ceiling")
            s.cyl_ft((-w2 - 0.1, 0.15, 3.45), (w2 + 0.1, 0.15, 3.45), 0.03, "steel_dark")
            for sx in (-1, 1):
                s.cyl_ft((sx * 0.25, 0.15, 3.45), (sx * 1.0, 0.15, 3.45), 0.06, "black_matte")
            if opened:
                s.box((-w2, 0.1, 3.05), (w2, 3.0, 3.3), "dock_door", "ceiling")   # door rolled up under the tracks
            else:
                s.box((-w2, -0.12, 0), (w2, -0.05, 3.05), "dock_door", "struct", name="dock_door")
            # leveler: diamond plate with hazard edges and a lip
            s.box((-1.0, 0.0, 0), (1.0, 2.4, 0.003), "diamond_plate")
            for sx in (-1, 1):
                s.box((sx * 1.0 - 0.08 * (sx > 0), 0.0, 0), (sx * 1.0 + 0.08 * (sx < 0), 2.4, 0.004), "hazard_tile")
            if opened:
                s.box((-1.0, -0.42, -0.01), (1.0, 0.0, 0.003), "diamond_plate", "struct", name="leveler_lip")
            # dock light on a swing arm, control box, sign above
            s.box((w2 + 0.25, 0.0, 2.2), (w2 + 0.35, 0.06, 2.4), "steel_dark")
            s.cap_ft((w2 + 0.3, 0.08, 2.3), (w2 - 0.2, 0.9, 2.4), 0.02, "steel_dark")
            s.cyl((w2 - 0.2, 0.95, 2.4), 0.12, 0.08, "safety_yellow", R=rot(rx=PI / 2 + 0.4))
            s.cyl((w2 - 0.2, 1.02, 2.37), 0.1, 0.005, "bulb", R=rot(rx=PI / 2 + 0.4))
            s.box((-w2 - 0.55, 0.0, 1.2), (-w2 - 0.25, 0.12, 1.65), "charger_grey")
            for k, mat in enumerate(("led_green", "led_red", "black_matte")):
                s.cyl((-w2 - 0.4, 0.125, 1.55 - 0.1 * k), 0.025, 0.008, mat, axis="y")
    # trailer at the open dock
    x = OPEN_DOCK
    tz0, tl = 0.0, 14.0
    ya, yb = Y0 - WT - tl, Y0 - WT - 0.02
    s.box((x - 1.3, ya, tz0 - 0.05), (x + 1.3, yb, tz0), "plywood", "struct", name="trailer_floor")
    s.box((x - 1.35, ya, tz0), (x - 1.3, yb, 2.75), "trailer_wall", "struct", name="trailer_wall")
    s.box((x + 1.3, ya, tz0), (x + 1.35, yb, 2.75), "trailer_wall", "struct", name="trailer_wall")
    s.box((x - 1.35, ya - 0.05, tz0), (x + 1.35, ya, 2.75), "trailer_wall", "struct", name="trailer_wall")
    s.box((x - 1.35, ya, 2.75), (x + 1.35, yb, 2.8), "trailer_wall", "struct", name="trailer_roof")
    for sx in (-1, 1):
        s.box((x + sx * 1.297, ya, 0.1), (x + sx * 1.3, yb, 1.3), "plywood")
        for k in range(8):
            yy = ya + 0.5 + k * 1.6
            s.box((x + sx * 1.294, yy, 0.0), (x + sx * 1.3, yy + 0.08, 2.75), "aluminum")
    rng = np.random.default_rng(5)
    for k in range(5):
        for sx in (-0.6, 0.6):
            load(s, x + sx, ya + 1.0 + k * 1.3, 0.0, rng, full=False)
    s.box((x - 1.0, ya + 0.5, 2.7), (x + 1.0, ya + 13.0, 2.75), "led_panel")
    # outside: dock apron below dock height
    s.box((-40, -60, -1.3), (40, Y0 - WT, -1.2), "asphalt", "struct", name="apron")
    # drive-in door (closed) and personnel doors
    s.box((-1.8, Y0 - 0.12, 0), (1.8, Y0 - 0.05, 4.3), "dock_door", "struct", name="drive_in_door")
    for sx in (-1, 1):
        s.box((sx * 1.85 - 0.05, Y0, 0), (sx * 1.85 + 0.05, Y0 + 0.3, 1.2), "safety_yellow", "struct", name="door_bollard")
    for xd in (-4.2, 4.2):
        s.box((xd - 0.475, Y0 - 0.1, 0), (xd + 0.475, Y0 - 0.05, 2.15), "steel_dark", "struct", name="personnel_door")
        s.box((xd + 0.2, Y0 - 0.05, 0.95), (xd + 0.42, Y0 + 0.02, 1.0), "chrome")
        s.box((xd - 0.15, Y0 - 0.05, 1.2), (xd + 0.15, Y0 - 0.049, 1.75), "dark_glass")


def south_wall_signs(s):
    """Signs on the inside of the south wall must read from +y: rotate so local -y -> world +y."""
    with s.frame((0, Y0 + 0.01, 0), PI):
        for x, n in DOCKS:
            s.box((-x - 0.7, -0.04, 3.6), (-x + 0.7, 0.0, 4.3), f"dock_{n}")
        for xd in (-4.2, 4.2):
            s.box((-xd - 0.35, -0.06, 2.3), (-xd + 0.35, 0.0, 2.55), "sign_exit")
        s.box((-2.9 - 0.72, -0.02, 2.6), (-2.9 + 0.72, 0.0, 3.56), "sign_forklift")
        s.box((9.0 - 0.6, -0.02, 2.0), (9.0 + 0.6, 0.0, 2.75), "sign_ppe")
        s.box((-3.0 - 0.45, -0.02, 4.6), (-3.0 + 0.45, 0.0, 5.75), "sign_speed")


# ---------------------------------------------------------------- vehicles & equipment

def forklift(s, x, y, yaw):
    """Counterbalance forklift; canonical: forks point to -y."""
    with s.item("forklift"), s.frame((x, y, 0), yaw):
        Y = "forklift_yellow"
        s.box((-0.55, -0.2, 0.18), (0.55, 1.5, 0.75), Y, "solid")
        s.box((-0.56, 1.1, 0.18), (0.56, 1.75, 1.05), "counterweight", "solid")
        s.cyl((0, 1.75, 0.6), 0.56, 0.45, "counterweight", "solid")
        s.box((-0.5, 0.2, 0.75), (0.5, 1.1, 0.95), Y, "solid")
        # seat, wheel, dash, levers
        s.box((-0.25, 0.65, 0.95), (0.25, 1.05, 1.05), "seat_black")
        s.box((-0.25, 0.98, 1.05), (0.25, 1.08, 1.45), "seat_black")
        s.box((-0.4, -0.05, 0.75), (0.4, 0.25, 1.15), Y)
        s.cyl_ft((0, 0.2, 1.1), (0, 0.38, 1.38), 0.025, "black_matte")
        with s.frame((0, 0.38, 1.4), R=rot(rx=-0.6)):
            n = 12
            for k in range(n):
                a0, a1 = 2 * PI * k / n, 2 * PI * (k + 1) / n
                s.cap_ft((0.17 * math.cos(a0), 0.17 * math.sin(a0), 0), (0.17 * math.cos(a1), 0.17 * math.sin(a1), 0), 0.014, "black_matte")
            s.cyl((0, 0, 0), 0.04, 0.02, "black_matte")
        for k in range(3):
            s.cap_ft((0.25 + 0.05 * k, 0.05, 1.1), (0.25 + 0.05 * k, 0.1, 1.35), 0.008, "black_matte")
            s.sphere((0.25 + 0.05 * k, 0.1, 1.37), 0.02, "paint_red" if k == 0 else "black_matte")
        # overhead guard
        for px, py in ((-0.5, -0.05), (0.5, -0.05), (-0.5, 1.15), (0.5, 1.15)):
            s.box((px - 0.035, py - 0.035, 0.75), (px + 0.035, py + 0.035, 2.15), "black_matte", "solid")
        s.box((-0.55, -0.1, 2.15), (0.55, 1.2, 2.2), "black_matte", "solid")
        for k in range(6):
            s.box((-0.5, 0.0 + 0.2 * k, 2.2), (0.5, 0.05 + 0.2 * k, 2.23), "black_matte")
        s.cyl((0, 0.95, 2.28), 0.06, 0.05, "amber_light")
        s.cyl((-0.48, 1.22, 2.05), 0.05, 0.02, "blue_light", axis="y")
        # LPG tank on the counterweight
        s.cyl_ft((-0.45, 1.5, 1.25), (0.45, 1.5, 1.25), 0.17, "galvanized")
        s.cyl_ft((-0.47, 1.5, 1.25), (-0.45, 1.5, 1.25), 0.12, "black_matte")
        # mast, carriage, forks, tilt cylinders
        for sx in (-1, 1):
            s.box((sx * 0.42 - 0.07, -0.45, 0.1), (sx * 0.42 + 0.07, -0.33, 2.3), "black_matte", "solid")
            s.box((sx * 0.30 - 0.05, -0.52, 0.12), (sx * 0.30 + 0.05, -0.44, 2.2), "steel_dark")
            s.cap_ft((sx * 0.45, 0.0, 0.85), (sx * 0.45, -0.38, 1.05), 0.035, "chrome")
        s.box((-0.5, -0.33, 2.25), (0.5, -0.25, 2.32), "black_matte")
        s.box((-0.55, -0.62, 0.12), (0.55, -0.52, 0.55), "black_matte")
        for k in range(5):
            xx = -0.45 + 0.225 * k
            s.box((xx - 0.015, -0.6, 0.55), (xx + 0.015, -0.56, 1.35), "black_matte")
        s.box((-0.55, -0.6, 1.33), (0.55, -0.56, 1.38), "black_matte")
        s.cyl_ft((0, -0.4, 2.3), (0, -0.4, 0.4), 0.05, "chrome")
        for sx in (-1, 1):
            s.box((sx * 0.3 - 0.05, -0.66, 0.05), (sx * 0.3 + 0.05, -0.62, 0.6), "steel_dark", "solid")
            s.box((sx * 0.3 - 0.05, -1.73, 0.05), (sx * 0.3 + 0.05, -0.62, 0.1), "steel_dark", "solid")
        # wheels
        for sx in (-1, 1):
            s.cyl((sx * 0.48, -0.15, 0.32), 0.32, 0.11, "tire", "solid", axis="x")
            s.cyl((sx * 0.595, -0.15, 0.32), 0.16, 0.005, "safety_yellow", axis="x")
            s.cyl((sx * 0.45, 1.35, 0.25), 0.25, 0.09, "tire", "solid", axis="x")
            s.cyl((sx * 0.545, 1.35, 0.25), 0.12, 0.005, "safety_yellow", axis="x")
        for sx in (-1, 1):
            s.cyl((sx * 0.4, -0.08, 0.95), 0.05, 0.02, "bulb", axis="y")


def pallet_jack(s, x, y, yaw, powered=False):
    with s.item("pallet_jack"), s.frame((x, y, 0), yaw):
        for sx in (-1, 1):
            s.box((sx * 0.26 - 0.08, -1.15, 0.03), (sx * 0.26 + 0.08, 0.0, 0.085), "safety_orange" if not powered else "paint_blue", "solid")
            s.cyl((sx * 0.26, -1.08, 0.035), 0.035, 0.05, "tire", axis="x")
        s.box((-0.34, -0.05, 0.03), (0.34, 0.25, 0.25 if not powered else 0.85), "safety_orange" if not powered else "paint_blue", "solid")
        s.cyl((0, 0.15, 0.08), 0.08, 0.04, "tire", axis="x")
        s.cyl_ft((0, 0.1, 0.25 if not powered else 0.85), (0, 0.45, 1.15), 0.02, "steel_dark")
        s.cap_ft((-0.18, 0.45, 1.15), (0.18, 0.45, 1.15), 0.025, "black_matte")
        if powered:
            s.box((-0.2, 0.0, 0.85), (0.2, 0.2, 0.95), "black_matte")


def hand_truck(s, x, y, yaw):
    with s.frame((x, y, 0), yaw):
        for sx in (-1, 1):
            s.cap_ft((sx * 0.2, 0.0, 0.15), (sx * 0.2, 0.12, 1.25), 0.016, "paint_red")
            s.cyl((sx * 0.26, 0.15, 0.13), 0.13, 0.03, "tire", axis="x")
        for z in (0.5, 0.9):
            s.cyl_ft((-0.2, 0.05, z), (0.2, 0.05, z), 0.012, "paint_red")
        s.cap_ft((-0.2, 0.12, 1.25), (0.2, 0.12, 1.25), 0.016, "paint_red")
        s.box((-0.22, -0.22, 0.0), (0.22, 0.0, 0.012), "steel_dark")


def stretch_wrapper(s, x, y):
    with s.item("stretch_wrapper"):
        s.cyl((x, y, 0.04), 0.85, 0.04, "galvanized", "solid")
        s.cyl((x, y, 0.085), 0.82, 0.005, "black_matte")
        s.box((x + 0.95, y - 0.2, 0), (x + 1.15, y + 0.2, 2.6), "paint_blue", "solid")
        s.box((x + 0.85, y - 0.25, 1.1), (x + 0.97, y + 0.25, 1.5), "steel_dark")
        s.cyl((x + 0.86, y, 1.3), 0.06, 0.25, "wrap")
        s.box((x + 1.15, y - 0.35, 0.9), (x + 1.45, y + 0.35, 1.5), "charger_grey", "solid")
        s.box((x + 1.451, y - 0.15, 1.2), (x + 1.452, y + 0.15, 1.4), "screen_b")


def floor_staging(s):
    rng = np.random.default_rng(42)
    for x, _ in DOCKS:
        for j, yy in enumerate((-10.6, -9.2)):
            for i, dx in enumerate((-0.6, 0.6)):
                if x == OPEN_DOCK and j == 0 and i == 0:
                    continue  # leave a path for loading
                if rng.random() < 0.8:
                    with s.item("staged_pallet"):
                        load(s, x + dx, yy, 0.0, rng, role="solid", max_h=1.6)
    # empty pallet stacks and cardboard bales
    for (x, y, n) in ((-13.7, -9.0, 11), (13.9, -9.0, 8), (13.9, -6.6, 13)):
        with s.item("pallet_stack"):
            for k in range(n):
                pallet(s, x, y, k * 0.144, "solid", full=True)
    with s.item("bales"):
        for k in range(3):
            for j in range(2):
                bx, by, bz = -14.1 + 0.0 * k, -11.3 + 1.0 * j, k * 0.8
                if k == 2 and j == 1:
                    continue
                s.box((bx - 0.6, by - 0.45, bz), (bx + 0.6, by + 0.45, bz + 0.78), "carton_c", "solid")
                for dx in (-0.35, 0.0, 0.35):
                    s.box((bx + dx - 0.01, by - 0.455, bz), (bx + dx + 0.01, by + 0.455, bz + 0.785), "strap")


def safety_equipment(s):
    # extinguishers + signs on the column faces that look into aisles
    for x in COLS_X:
        for y in COLS_Y:
            if abs(x) > 9 and y > -3:
                continue
            with s.frame((x, y - 0.225, 0.35)):  # on the south face of the column guard
                P.fire_extinguisher(s, 0, 0, 0, wall_yaw=0)
                s.box((-0.15, -0.02, 1.25), (0.15, 0.0, 1.55), "sign_fire")
    # eyewash station + first aid on the west wall
    with s.frame((X0, -2.0, 0), PI / 2):
        s.box((-0.3, -0.02, 0.0), (0.3, 0.0, 2.2), "paint_green")
        s.cyl_ft((0, -0.1, 0.0), (0, -0.1, 1.0), 0.03, "galvanized")
        s.cyl((0, -0.3, 1.0), 0.15, 0.04, "paint_green")
        s.cyl_ft((0, -0.1, 1.0), (0, -0.3, 1.0), 0.02, "galvanized")
        s.box((0.5, -0.12, 1.3), (0.9, 0.0, 1.8), "plastic_white")
        s.box((0.67, -0.121, 1.45), (0.73, -0.12, 1.65), "paint_red")
        s.box((0.6, -0.121, 1.52), (0.8, -0.12, 1.58), "paint_red")
    # flammables cabinet
    with s.item("flammables_cabinet"), s.frame((X0, -9.0, 0), PI / 2):
        s.box((-0.6, -0.55, 0), (0.6, 0.0, 1.65), "safety_yellow", "solid")
        s.box((-0.003, -0.56, 0.05), (0.003, -0.55, 1.6), "black_matte")
        for sx in (-1, 1):
            s.box((sx * 0.06 - 0.01, -0.6, 0.8), (sx * 0.06 + 0.01, -0.55, 1.0), "black_matte")
        s.box((-0.45, -0.551, 1.2), (0.45, -0.55, 1.45), "paint_red")
    # forklift battery chargers along the west wall
    for k, y in enumerate((-6.6, -5.4)):
        with s.item(f"charger_{k}"), s.frame((X0, y, 0), PI / 2):
            s.box((-0.3, -0.45, 0), (0.3, 0.0, 1.3), "charger_grey", "solid")
            s.box((-0.2, -0.451, 0.9), (0.2, -0.45, 1.15), "screen_b")
            s.cyl((0.18, -0.46, 0.75), 0.02, 0.01, "led_green", axis="y")
            for dx in (-0.1, 0.1):
                s.cap_ft((dx, -0.46, 0.4), (dx + 0.3, -1.2, 0.05), 0.015, "cable")
    pallet_jack(s, -14.0, -5.6, PI / 2, powered=True)
    # bollards protecting the office corner and the doors
    for (x, y) in ((-10.0, 11.6), (-9.0, 11.6), (-4.9, -11.6), (-3.5, -11.6), (3.5, -11.6), (4.9, -11.6)):
        with s.item("bollard", proxy=False):
            s.cyl((x, y, 0.6), 0.08, 0.6, "safety_yellow", "struct", name="bollard")
            s.sphere((x, y, 1.2), 0.08, "safety_yellow")
            for z in (0.85, 1.0):
                s.cyl((x, y, z), 0.081, 0.03, "black_matte")
    # banner and safety board
    with s.frame((0, Y1, 0)):
        s.box((-6.0, -0.03, 5.6), (6.0, 0.0, 6.35), "banner")
    with s.frame((X0, 3.0, 0), PI / 2):
        s.box((-0.8, -0.02, 1.7), (0.8, 0.0, 2.75), "sign_incident")


# ---------------------------------------------------------------- pack area, shelving, office, break area

def pack_station(s, cx, cy, k):
    with s.item(f"pack_station_{k}"), s.frame((cx, cy, 0)):
        w, d = 1.0, 0.4
        s.box((-w, -d, 0.86), (w, d, 0.9), "butcher", "solid")
        for sx in (-1, 1):
            for sy in (-1, 1):
                s.box((sx * (w - 0.05) - 0.025, sy * (d - 0.05) - 0.025, 0), (sx * (w - 0.05) + 0.025, sy * (d - 0.05) + 0.025, 0.86), "bench_grey", "solid")
        s.box((-w + 0.05, -d + 0.05, 0.18), (w - 0.05, d - 0.05, 0.2), "bench_grey", "solid")
        for i in range(6):  # flat-packed cartons on the lower shelf
            s.box((-0.8, -0.3, 0.2 + i * 0.012), (0.2, 0.3, 0.211 + i * 0.012), "carton_a")
        # riser with top shelf and light bar
        for sx in (-1, 1):
            s.box((sx * (w - 0.04) - 0.02, d - 0.06, 0.9), (sx * (w - 0.04) + 0.02, d - 0.02, 2.0), "bench_grey", "solid")
        s.box((-w, d - 0.35, 1.6), (w, d, 1.63), "bench_grey")
        s.box((-w + 0.05, d - 0.3, 1.585), (w - 0.05, d - 0.25, 1.6), "led_panel")
        for i, (bw, bh) in enumerate(((0.35, 0.25), (0.3, 0.2), (0.4, 0.3))):
            x0 = -0.9 + i * 0.6
            s.box((x0, d - 0.33, 1.63), (x0 + bw, d - 0.03, 1.63 + bh), ("carton_a", "carton_b", "carton_c")[i])
        # monitor on an arm, keyboard, scanner, label printer, scale, tape gun, open carton
        s.cyl_ft((0.55, d - 0.05, 0.9), (0.55, d - 0.05, 1.25), 0.015, "steel_dark")
        s.box((0.3, d - 0.12, 1.12), (0.8, d - 0.08, 1.45), "black_matte")
        s.box((0.32, d - 0.125, 1.14), (0.78, d - 0.12, 1.43), "screen_a" if k % 2 else "screen_b")
        s.box((0.35, -0.05, 0.9), (0.75, 0.1, 0.915), "black_matte")
        s.box((-0.85, 0.15, 0.9), (-0.55, 0.35, 1.08), "plastic_white")
        s.box((-0.84, 0.149, 1.0), (-0.66, 0.15, 1.05), "black_matte")
        s.box((0.82, -0.3, 0.9), (0.97, -0.08, 0.93), "galvanized")
        s.box((0.85, -0.31, 0.905), (0.94, -0.302, 0.925), "screen_a")
        s.cyl((-0.25, -0.25, 0.94), 0.05, 0.025, "wrap", axis="x")
        s.box((-0.33, -0.28, 0.9), (-0.17, -0.22, 0.93), "paint_red")
        s.cap_ft((-0.3, -0.25, 0.93), (-0.38, -0.25, 0.86 + 0.15), 0.012, "black_matte")
        cx0, cy0, bw, bd, bh = -0.4, 0.05, 0.25, 0.18, 0.2
        for (a, b) in (((cx0 - bw, cy0 - bd), (cx0 + bw, cy0 - bd + 0.006)), ((cx0 - bw, cy0 + bd - 0.006), (cx0 + bw, cy0 + bd)),
                       ((cx0 - bw, cy0 - bd), (cx0 - bw + 0.006, cy0 + bd)), ((cx0 + bw - 0.006, cy0 - bd), (cx0 + bw, cy0 + bd))):
            s.box((a[0], a[1], 0.9), (b[0], b[1], 0.9 + bh), "carton_b")
        s.box((cx0 - bw + 0.006, cy0 - bd + 0.006, 0.9), (cx0 + bw - 0.006, cy0 + bd - 0.006, 0.905), "carton_b")
        with s.frame((cx0, cy0 - bd, 0.9 + bh), R=rot(rx=0.9)):
            s.box((-bw, -0.003, 0), (bw, 0.003, 0.17), "carton_b")
        # scanner in a holder
        s.cap_ft((0.95, 0.25, 0.95), (0.95, 0.2, 1.05), 0.018, "black_matte")
        s.box((0.93, 0.15, 1.04), (0.97, 0.22, 1.08), "safety_yellow")
    # anti-fatigue mat and a stool (real collision, for sitting tasks)
    s.box((cx - 0.9, cy - 1.25, 0), (cx + 0.9, cy - 0.45, 0.012), "rubber_mat")
    with s.item(f"pack_stool_{k}", proxy=False), s.frame((cx + 0.6, cy - 0.8, 0)):
        s.cyl((0, 0, 0.64), 0.19, 0.03, "seat_black", "struct", name="stool_seat")
        s.cyl((0, 0, 0.35), 0.025, 0.28, "chrome", "struct", name="stool_post")
        for a in np.arange(5) * 2 * PI / 5:
            s.cap_ft((0, 0, 0.07), (0.28 * math.cos(a), 0.28 * math.sin(a), 0.03), 0.015, "black_matte")
            s.sphere((0.28 * math.cos(a), 0.28 * math.sin(a), 0.025), 0.025, "black_matte")
        s.cyl((0, 0, 0.3), 0.2, 0.008, "chrome")


def conveyor(s, x0, x1, y, z=0.85, w=0.6):
    rng = np.random.default_rng(8)
    with s.item("conveyor"):
        for sy in (-1, 1):
            s.box((x0, y + sy * w / 2 - 0.04 * (sy > 0), z - 0.12), (x1, y + sy * w / 2 + 0.04 * (sy < 0), z + 0.03), "galvanized", "solid")
        for xl in np.arange(x0 + 0.3, x1, 1.6):
            for sy in (-1, 1):
                s.box((xl - 0.03, y + sy * (w / 2 - 0.02) - 0.03, 0), (xl + 0.03, y + sy * (w / 2 - 0.02) + 0.03, z - 0.12), "galvanized", "solid")
                s.box((xl - 0.08, y + sy * (w / 2 - 0.02) - 0.08, 0), (xl + 0.08, y + sy * (w / 2 - 0.02) + 0.08, 0.01), "steel_dark")
            s.box((xl - 0.02, y - w / 2, 0.25), (xl + 0.02, y + w / 2, 0.29), "galvanized")
    for xr in np.arange(x0 + 0.05, x1 - 0.02, 0.1):
        s.cyl_ft((xr, y - w / 2 + 0.04, z - 0.01), (xr, y + w / 2 - 0.04, z - 0.01), 0.024, "galvanized")
    s.box((x1 - 0.5, y + w / 2, z - 0.4), (x1 - 0.1, y + w / 2 + 0.25, z - 0.12), "paint_blue")
    s.cyl_ft((x0, y - w / 2 - 0.05, z + 0.08), (x1, y - w / 2 - 0.05, z + 0.08), 0.004, "safety_yellow")
    for xe in np.arange(x0 + 1.0, x1, 3.0):
        s.box((xe - 0.05, y - w / 2 - 0.12, z - 0.05), (xe + 0.05, y - w / 2 - 0.04, z + 0.05), "safety_yellow")
        s.cyl((xe, y - w / 2 - 0.13, z), 0.025, 0.012, "paint_red", axis="y")
    for xb in np.arange(x0 + 0.6, x1 - 0.5, 1.3):
        if rng.random() < 0.75:
            bw, bd, bh = rng.uniform(.18, .3), rng.uniform(.15, .25), rng.uniform(.12, .3)
            s.box((xb - bw, y - bd, z + 0.015), (xb + bw, y + bd, z + 0.015 + bh), rng.choice(["carton_a", "carton_b", "carton_c"]))


def shelving_row(s, x0, nunits, y, unit=1.5, depth=0.6, seed=0):
    rng = np.random.default_rng(seed)
    with s.item(f"shelving_{seed}"):
        for k in range(nunits + 1):
            for yy in (y - depth / 2, y + depth / 2 - 0.04):
                s.box((x0 + k * unit - 0.02, yy, 0), (x0 + k * unit + 0.02, yy + 0.04, 2.15), "bench_grey", "solid")
        levels = [0.12, 0.6, 1.08, 1.56, 2.05]
        for k in range(nunits):
            xa, xb = x0 + k * unit + 0.02, x0 + (k + 1) * unit - 0.02
            for lv in levels:
                s.box((xa, y - depth / 2, lv - 0.02), (xb, y + depth / 2, lv), "galvanized", "solid" if lv < 0.2 else "deco")
            for lv in levels[:-1]:
                for j in range(3):
                    if rng.random() < 0.1:
                        continue
                    bx0 = xa + 0.03 + j * (xb - xa - 0.06) / 3
                    bx1 = bx0 + (xb - xa - 0.06) / 3 - 0.03
                    hb = 0.3
                    for side in (-1, 1):
                        yy0 = y + (side < 0) * (-depth / 2 + 0.02) + (side > 0) * 0.01
                        s.box((bx0, yy0, lv), (bx1, yy0 + depth / 2 - 0.03, lv + hb), rng.choice(["bin_blue", "bin_red", "bin_yellow"], p=[.5, .25, .25]))


def office(s):
    xa, xb, ya, yb = X0, -10.6, 12.2, Y1
    hz = 3.0
    with s.item("office_walls", proxy=False):
        for (p0, p1) in (((xa, ya), (xb, ya)), ((xb, ya), (xb, yb))):
            (x0, y0), (x1, y1) = p0, p1
            n = int(max(abs(x1 - x0), abs(y1 - y0)) / 1.1) + 1
            for k in range(n + 1):
                px = x0 + (x1 - x0) * k / n
                py = y0 + (y1 - y0) * k / n
                s.box((px - 0.04, py - 0.04, 0), (px + 0.04, py + 0.04, hz), "aluminum", "struct", name="mullion")
            lo, hi = (min(x0, x1) - 0.04, min(y0, y1) - 0.04), (max(x0, x1) + 0.04, max(y0, y1) + 0.04)
            s.box((lo[0], lo[1], 0), (hi[0], hi[1], 0.15), "aluminum", "struct", name="office_kick")
            s.box((lo[0], lo[1], hz - 0.1), (hi[0], hi[1], hz), "aluminum", "struct", name="office_head")
            s.box((lo[0] + 0.02, lo[1] + 0.02, 0.15), (hi[0] - 0.02, hi[1] - 0.02, hz - 0.1), "office_glass", "struct", name="office_glass")
            s.box((lo[0] - 0.005, lo[1] - 0.005, 1.0), (hi[0] + 0.005, hi[1] + 0.005, 1.05), "aluminum")
        s.box((xa, ya - 0.04, hz), (xb + 0.04, yb, hz + 0.08), "ceiling_tile", "ceiling", name="office_ceiling")
        s.box((xa, ya, hz + 0.08), (xb + 0.04, yb, hz + 1.1), "aluminum", "ceiling", name="office_top")
    s.box((xb - 0.045, 13.0, 0.15), (xb - 0.035, 13.95, 2.15), "dark_glass")
    s.box((xb - 0.06, 13.85, 0.95), (xb - 0.02, 13.9, 1.3), "chrome")
    s.box((xa + 0.05, ya + 0.05, 0), (xb - 0.05, yb - 0.05, 0.006), "carpet")
    # desk with two monitors, chair, filing cabinet, whiteboard, plant, clock
    with s.item("office_desk"):
        s.box((-14.6, 14.6, 0.72), (-12.2, 15.4, 0.75), "desk_top", "solid")
        for x in (-14.55, -12.25):
            s.box((x - 0.02, 14.65, 0), (x + 0.02, 15.35, 0.72), "steel_dark", "solid")
        s.box((-14.55, 15.3, 0.3), (-12.25, 15.33, 0.72), "steel_dark")
    for k, x in enumerate((-13.75, -13.05)):
        with s.frame((x, 15.15, 0.75), R=rot(rz=0.15 * (1 - 2 * k))):
            s.box((-0.3, -0.02, 0.12), (0.3, 0.02, 0.48), "black_matte")
            s.box((-0.29, -0.0205, 0.13), (0.29, -0.02, 0.47), "screen_a" if k == 0 else "screen_b")
            s.cyl((0, 0.03, 0.06), 0.015, 0.06, "steel_dark")
            s.box((-0.1, -0.06, 0.0), (0.1, 0.1, 0.01), "steel_dark")
    s.box((-13.6, 14.75, 0.75), (-13.15, 14.92, 0.765), "black_matte")
    s.ell((-12.95, 14.85, 0.765), (0.03, 0.05, 0.015), "black_matte")
    P.mug(s, -12.6, 15.0, 0.75, "ceramic_white")
    s.cyl((-12.45, 15.25, 0.76), 0.08, 0.01, "black_matte")
    s.cap_ft((-12.45, 15.25, 0.77), (-12.5, 15.2, 1.15), 0.01, "black_matte")
    s.cyl((-12.62, 15.05, 1.15), 0.07, 0.04, "black_matte", R=rot(ry=0.6))
    with s.item("office_chair", proxy=False), s.frame((-13.4, 14.1, 0)):
        for a in np.arange(5) * 2 * PI / 5:
            s.cap_ft((0, 0, 0.1), (0.3 * math.cos(a), 0.3 * math.sin(a), 0.07), 0.018, "black_matte")
            s.sphere((0.3 * math.cos(a), 0.3 * math.sin(a), 0.035), 0.035, "black_matte")
        s.cyl((0, 0, 0.28), 0.03, 0.18, "chrome", "struct", name="chair_post")
        s.box((-0.25, -0.24, 0.45), (0.25, 0.24, 0.52), "chair_fabric", "struct", name="chair_seat")
        s.box((-0.23, -0.32, 0.6), (0.23, -0.26, 1.1), "chair_fabric", "struct", name="chair_back")
        s.cap_ft((0, -0.24, 0.48), (0, -0.29, 0.62), 0.02, "black_matte")
        for sx in (-1, 1):
            s.cap_ft((sx * 0.25, -0.1, 0.52), (sx * 0.25, -0.1, 0.68), 0.015, "black_matte")
            s.box((sx * 0.25 - 0.03, -0.2, 0.68), (sx * 0.25 + 0.03, 0.1, 0.7), "black_matte")
    with s.item("filing_cabinet"):
        s.box((-14.95, 12.4, 0), (-14.45, 13.0, 1.33), "steel_dark", "solid")
        for k in range(3):
            s.box((-14.45, 12.42, 0.05 + 0.43 * k), (-14.44, 12.98, 0.45 + 0.43 * k), "steel_dark")
            s.box((-14.44, 12.6, 0.37 + 0.43 * k), (-14.42, 12.8, 0.39 + 0.43 * k), "chrome")
    with s.frame((-12.8, Y1, 0)):
        s.box((-1.0, -0.03, 1.0), (1.0, 0.0, 2.2), "whiteboard")
        s.box((-1.0, -0.06, 0.98), (1.0, 0.0, 1.0), "aluminum")
    with s.frame((X0, 13.7, 0), PI / 2):
        P.wall_clock(s, 0, 2.3, r=0.17)
    P.plant(s, -10.95, 15.6, 0, pot_r=0.18, pot_h=0.35, kind="snake", seed=70, height=0.9, pot_mat="ceramic_white", role="struct")
    P.trash_can(s, -12.0, 15.6, r=0.14, h=0.4, mat="steel_dark")


def break_area(s):
    with s.item("vending"), s.frame((-9.0, Y1, 0)):
        s.box((-0.45, -0.85, 0), (0.45, 0.0, 1.85), "steel_dark", "solid")
        s.box((-0.43, -0.851, 0.05), (0.43, -0.85, 1.83), "vending")
    with s.item("water_cooler"), s.frame((-7.9, Y1 - 0.25, 0)):
        s.box((-0.16, -0.16, 0), (0.16, 0.16, 1.0), "plastic_white", "solid")
        s.cyl((0, 0, 1.2), 0.14, 0.2, "water_jug")
        s.cyl((0, 0, 1.42), 0.05, 0.03, "water_jug")
        for dx, mat in ((-0.06, "paint_blue"), (0.06, "paint_red")):
            s.box((dx - 0.02, -0.17, 0.8), (dx + 0.02, -0.15, 0.85), mat)
        s.box((-0.12, -0.18, 0.6), (0.12, -0.1, 0.62), "black_matte")
    with s.item("break_counter"), s.frame((-5.0, Y1, 0)):
        s.box((-0.9, -0.6, 0), (0.9, 0.0, 0.9), "bench_grey", "solid")
        s.box((-0.92, -0.62, 0.9), (0.92, 0.0, 0.94), "butcher", "solid")
        s.box((-0.85, -0.5, 0.94), (-0.35, -0.12, 1.24), "steel_dark")
        s.box((-0.82, -0.501, 0.98), (-0.5, -0.5, 1.2), "dark_glass")
        s.box((0.2, -0.4, 0.94), (0.45, -0.1, 1.3), "black_matte")
        P.mug(s, 0.32, -0.32, 0.94, "ceramic_white")
        for k in range(3):
            P.mug(s, 0.6 + 0.1 * k, -0.3, 0.94, "plastic_grey", fill=None)
    with s.item("break_table"):
        s.cyl((-6.7, 14.3, 0.74), 0.45, 0.015, "desk_top", "solid")
        s.cyl((-6.7, 14.3, 0.37), 0.04, 0.36, "steel_dark", "solid")
        s.cyl((-6.7, 14.3, 0.01), 0.3, 0.01, "steel_dark", "solid")
    for k in range(4):
        a = k * PI / 2 + PI / 4
        cx, cy = -6.7 + 0.72 * math.cos(a), 14.3 + 0.72 * math.sin(a)
        with s.item(f"break_chair_{k}", proxy=False), s.frame((cx, cy, 0), a + PI / 2):
            s.box((-0.21, -0.2, 0.43), (0.21, 0.2, 0.46), "tote_blue", "struct", name="break_chair")
            s.box((-0.2, 0.17, 0.46), (0.2, 0.2, 0.85), "tote_blue", "struct", name="break_chair")
            for sx in (-1, 1):
                for sy in (-1, 1):
                    s.cyl_ft((sx * 0.18, sy * 0.17, 0), (sx * 0.18, sy * 0.17, 0.43), 0.012, "chrome", "struct", name="break_chair")
    with s.frame((-6.7, Y1, 0)):
        s.box((-0.7, -0.03, 1.3), (0.7, 0.0, 2.2), "cork")


def lift_station(s):
    """Task props by the test zone: fixed pallet & table, liftable cartons and tote, a stool."""
    with s.item("lift_pallet", proxy=False):
        top = pallet(s, 3.75, -6.9, 0.0, role="struct")
    boxes = [((-0.22, -0.3), (0.2, 0.15, 0.15), 4.0, "carton_a"), ((0.24, -0.3), (0.2, 0.15, 0.125), 3.0, "carton_b"),
             ((-0.2, 0.28), (0.25, 0.2, 0.175), 8.0, "carton_c"), ((0.25, 0.3), (0.15, 0.15, 0.15), 2.5, "carton_a")]
    for i, ((dx, dy), half, mass, mat) in enumerate(boxes):
        with s.dynamic(f"carton_{i}", (3.75 + dx, -6.9 + dy, top + half[2] + 0.0005)):
            s.boxc((0, 0, 0), half, mat, "dyn", mass=mass)
    for i, ((x, y), half, mass) in enumerate((((5.2, -7.3), (0.3, 0.2, 0.2), 10.0), ((5.25, -6.5), (0.2, 0.2, 0.15), 5.0))):
        with s.dynamic(f"floor_carton_{i}", (x, y, half[2] + 0.0005)):
            s.boxc((0, 0, 0), half, "carton_b" if i else "carton_c", "dyn", mass=mass)
    with s.item("lift_table", proxy=False):
        s.box((3.2, -5.4, 0.86), (4.7, -4.65, 0.9), "butcher", "struct", name="lift_table_top")
        for x in (3.25, 4.65):
            for y in (-5.35, -4.7):
                s.box((x - 0.025, y - 0.025, 0), (x + 0.025, y + 0.025, 0.86), "bench_grey", "struct", name="lift_table_leg")
        s.box((3.25, -5.35, 0.15), (4.65, -4.7, 0.17), "bench_grey")
    with s.dynamic("tote", (3.6, -5.02, 0.9 + 0.15 + 0.0005)):
        s.boxc((0, 0, -0.14), (0.28, 0.19, 0.01), "tote_grey", "dyn", mass=0.5)
        for sx in (-1, 1):
            s.boxc((sx * 0.27, 0, 0.0), (0.01, 0.19, 0.15), "tote_grey", "dyn", mass=0.25)
            s.boxc((0, sx * 0.18, 0.0), (0.26, 0.01, 0.15), "tote_grey", "dyn", mass=0.25)
    with s.dynamic("tote_item", (3.6, -5.02, 0.9 + 0.02 + 0.06 + 0.0005)):
        s.boxc((0, 0, 0), (0.1, 0.07, 0.06), "carton_a", "dyn", mass=0.6)
    with s.item("lift_stool", proxy=False), s.frame((5.35, -5.0, 0)):
        s.cyl((0, 0, 0.64), 0.19, 0.03, "seat_black", "struct", name="lift_stool")
        s.cyl((0, 0, 0.35), 0.025, 0.28, "chrome", "struct", name="lift_stool")
        for a in np.arange(5) * 2 * PI / 5:
            s.cap_ft((0, 0, 0.07), (0.28 * math.cos(a), 0.28 * math.sin(a), 0.03), 0.015, "black_matte")
    s.box((3.0, -7.6, 0), (3.08, -4.3, 0.0013), "paint_yellow")
    s.box((5.9, -7.6, 0), (5.98, -4.3, 0.0013), "paint_yellow")


# ---------------------------------------------------------------- lights & cameras

def lights_cameras(s):
    s.light("skylight", (0, 0, 20), dir=(-0.25, -0.3, -0.92), diffuse=(.36, .36, .35), directional=True, specular=(.05, .05, .05))
    s.light("fill", (0, 0, 20), dir=(0.35, 0.4, -0.85), diffuse=(.22, .22, .23), directional=True, specular=(0, 0, 0))
    s.light("bounce", (0, 0, 0.1), dir=(0.1, 0.15, 0.98), diffuse=(.14, .14, .13), directional=True, specular=(0, 0, 0))
    s.light("test_zone_key", (0.4, -5.2, 7.8), dir=(0, 0, -1), diffuse=(.40, .40, .38), cutoff=70, exponent=1, specular=(.05, .05, .05))

    s.camera("entrance", (0.0, -11.3, 2.3), (0.0, 6.0, 0.9), fovy=62)
    s.camera("robot_view", (0.0, -8.6, 1.6), (0.0, 0.0, 0.85), fovy=62)
    s.camera("rack_aisle", (7.0, -5.2, 1.7), (7.0, 14.0, 2.6), fovy=66)
    s.camera("docks", (1.2, -5.0, 3.2), (-8.0, -12.0, 1.2), fovy=72)
    s.camera("forklift_closeup", (-2.6, -4.4, 1.7), (-5.8, -6.6, 0.9), fovy=62)
    s.camera("packing", (-4.2, -3.4, 2.1), (-10.5, 1.8, 0.9), fovy=70)
    s.camera("office_break", (-4.0, 9.5, 2.4), (-11.0, 14.6, 1.0), fovy=70)
    s.camera("overview", (13.8, -11.3, 7.0), (-3.0, 3.5, 0.0), fovy=74)
    s.camera("top", (0.0, 2.0, 42.0), (0.0, 2.0001, 0.0), fovy=30.5, ortho=True)
    s.default_camera = "entrance"


def build() -> Scene:
    s = Scene("g1_detailed_warehouse", "wh_")
    s.statistic = dict(center=(0, 0, 2.0), extent=20.0)
    s.visual = dict(azimuth=120, elevation=-30)
    s.headlight = dict(ambient=(.32, .32, .32), diffuse=(.20, .20, .20), specular=(.03, .03, .03))
    s.skybox = dict(rgb1=(.55, .68, .86), rgb2=(.88, .91, .95))
    materials(s)
    shell(s)
    columns(s)
    roof_structure(s)
    floor_markings(s)
    racking(s)
    docks(s)
    south_wall_signs(s)
    floor_staging(s)
    forklift(s, -6.1, -6.15, PI / 2)
    pallet_jack(s, 4.95, -9.3, 0.3)
    hand_truck(s, -4.9, -9.6, 0.4)
    stretch_wrapper(s, -11.6, -6.1)
    safety_equipment(s)
    for k, x in enumerate((-12.5, -10.0, -7.5, -5.0)):
        pack_station(s, x, 0.0, k)
    conveyor(s, -14.3, -3.6, 1.35)
    with s.frame((-8.9, -1.95, 0)):
        s.box((-0.9, 0.0, 3.0), (0.9, 0.02, 3.55), "sign_pack")
        for dx in (-0.8, 0.8):
            s.cyl_ft((dx, 0.01, 3.55), (dx, 0.01, ROOF - 0.62), 0.005, "steel_dark", "ceiling")
    for k, y in enumerate((5.6, 8.2, 10.8)):
        shelving_row(s, -14.2, 6, y, seed=k)
    with s.frame((-8.9, 4.7, 0)):
        s.box((-0.9, 0.0, 3.0), (0.9, 0.02, 3.55), "sign_pick")
        for dx in (-0.8, 0.8):
            s.cyl_ft((dx, 0.01, 3.55), (dx, 0.01, ROOF - 0.62), 0.005, "steel_dark", "ceiling")
    office(s)
    break_area(s)
    lift_station(s)
    lights_cameras(s)
    s.meta = dict(spawn=dict(x=SPAWN[0], y=SPAWN[1], yaw_deg=SPAWN[2]),
                  clear_floor_area=dict(main_aisle_x=[-2, 2], main_aisle_y=[-11.5, 15.5], test_zone=dict(x=[-1.5, 1.5], y=[-7.2, -4.1])),
                  web_walk_loop=("web/scripts/sim/walk.py's oval (|x|<=1.4, y -5.9..2.6) is inside the clear main aisle: verified IK "
                                 "converges with zero robot-room contacts. web/scripts/sim/export.py reads primitives only, so export "
                                 "from a --no-bake build."),
                  task_props=dict(chairs="pack stools x4, office chair, 4 break chairs, lift-station stool (real collision in both files)",
                                  tables="pack benches 0.9 m, lift table 0.9 m, break table 0.75 m",
                                  liftable=["carton_0..3 on the lift pallet (2.5-8 kg)", "floor_carton_0..1 (5-10 kg)", "tote + tote_item"]))
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--textures", action="store_true", help="regenerate all textures")
    ap.add_argument("--preview", action="store_true", help="render previews/<camera>.png")
    ap.add_argument("--no-bake", action="store_true",
                    help="write every geom as a primitive (slower to render; needed for web/scripts/sim/export.py)")
    a = ap.parse_args()
    make_textures(HERE / "assets", a.textures)
    s = build()
    visual, nav = HERE / "WAREHOUSE_VISUAL.xml", HERE / "WAREHOUSE_G1_NAV.xml"
    baked = HERE / "assets" / "baked"
    for old_mesh in baked.glob("*.msh"):
        old_mesh.unlink()
    s.write(visual, "visual", bake=not a.no_bake)
    s.write(nav, "nav", bake=not a.no_bake)
    from envkit.preview import check, render_all
    manifest = dict(scene="Detailed distribution warehouse (procedural, not a scan)", **s.stats(), **s.meta)
    manifest["baked"] = getattr(s, "bake_info", None) if not a.no_bake else None
    manifest["visual_check"] = check(visual)
    manifest["nav_check"] = check(nav)
    (HERE / "SCENE_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ("geoms", "by_role", "nav_proxies", "baked", "visual_check", "nav_check")}, indent=1))
    if a.preview:
        for p in render_all(visual, HERE / "previews"):
            print("wrote", p.relative_to(HERE))


if __name__ == "__main__":
    main()
