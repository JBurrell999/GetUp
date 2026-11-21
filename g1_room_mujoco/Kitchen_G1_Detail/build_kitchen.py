#!/usr/bin/env python3
"""
Generate the detailed residential kitchen for MuJoCo + Unitree G1.

    python g1_room_mujoco/Kitchen_G1_Detail/build_kitchen.py            # textures (if missing) + XMLs
    python g1_room_mujoco/Kitchen_G1_Detail/build_kitchen.py --textures # regenerate textures too
    python g1_room_mujoco/Kitchen_G1_Detail/build_kitchen.py --preview  # also render previews/*.png

Layout (metres, z up, interior x -4..4, y -3.6..3.6, ceiling 2.75):
    north wall   base + upper cabinets, undermount sink under a window, gas range + chimney hood
    west wall    open pantry shelving, wall-oven/microwave tower, French-door fridge, tall pantry
    centre       2.7 m waterfall island with three stools and pendant lights
    south-east   dining table for six on a rug, chandelier, sideboard
    south-west   clear floor area for the robot (spawn at -1.35, -2.2 facing +y / the island)
    east         sliding patio door and window onto a deck and backyard
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

SPAWN = (-1.35, -2.2, 90.0)          # x, y, yaw (deg): G1 faces +y, toward the island
X0, X1, Y0, Y1, H = -4.0, 4.0, -3.6, 3.6, 2.75
WT = 0.15                            # wall thickness
TK, CT, CB = 0.10, 0.935, 0.895      # toe-kick height, counter top, counter bottom
UZ0, UZ1 = 1.42, 2.34                # upper cabinet bottom / top
PI = math.pi

# ---------------------------------------------------------------- textures

TEXTURES = {
    "floor_oak": lambda: T.wood_planks(1024, 1024, (.80, .63, .45), (.55, .39, .25), rows=10, seed=3),
    "wall_paint": lambda: T.plaster(512, 512, (.90, .885, .855), seed=1, amount=.035),
    "ceiling_paint": lambda: T.plaster(512, 512, (.95, .95, .94), seed=2, amount=.02),
    "quartz": lambda: T.marble(1024, 1024, (.95, .945, .93), (.58, .57, .56), seed=5, freq=2.2, turb=2.6, sharp=26),
    "subway": lambda: T.tiles(512, 512, 4, 8, (.95, .95, .93), (.80, .79, .76), seed=4, offset=.5, gw=.05, bevel=.18),
    "steel": lambda: T.brushed(512, 512, (.79, .80, .81), seed=6, streak=.09),
    "oak": lambda: T.wood_grain(1024, 512, (.78, .60, .40), (.50, .34, .20), seed=7, rings=9, knots=2),
    "walnut": lambda: T.wood_grain(1024, 512, (.42, .28, .18), (.22, .13, .08), seed=8, rings=8, knots=1),
    "butcher": lambda: T.wood_grain(512, 512, (.86, .70, .50), (.64, .46, .28), seed=9, rings=14, knots=0),
    "leather": lambda: T.fabric(256, 256, (.52, .31, .19), seed=10, weave=20, amount=.04),
    "linen": lambda: T.fabric(256, 256, (.84, .80, .73), seed=11, weave=64, amount=.12),
    "rug": lambda: T.rug(1024, 768, seed=12),
    "backyard": lambda: T.backyard(1536, 768, seed=13),
    "deck": lambda: T.wood_planks(512, 512, (.60, .50, .42), (.40, .32, .26), rows=12, seed=14),
    "terracotta_tex": lambda: T.plaster(256, 256, (.72, .40, .26), seed=15, amount=.10),
    "wicker": lambda: T.fabric(256, 256, (.62, .47, .30), seed=16, weave=24, amount=.35),
    "art1": lambda: T.art_abstract(512, 640, seed=17),
    "art2": lambda: T.art_abstract(640, 480, seed=18, palette=[(.92, .90, .85), (.25, .40, .45), (.70, .55, .35), (.85, .45, .35)]),
    "art3": lambda: T.art_abstract(480, 640, seed=19, palette=[(.95, .93, .88), (.45, .55, .40), (.80, .70, .50), (.30, .30, .35)]),
    "books": lambda: T.books(512, 256, n=12, seed=20),
    "calendar": lambda: T.paper_note(256, 320, "calendar", seed=21),
    "drawing": lambda: T.paper_note(256, 200, "drawing", seed=22),
    "photo": lambda: T.paper_note(160, 220, "photo", seed=23),
    "cereal_a": lambda: T.product_box(256, 384, (.85, .15, .12), (.95, .75, .15), "OAT CRUNCH", "whole grain", seed=24),
    "cereal_b": lambda: T.product_box(256, 384, (.15, .35, .70), (.95, .95, .95), "CORN FLAKES", "family size", seed=25),
    "crackers": lambda: T.product_box(256, 384, (.90, .62, .10), (.60, .15, .10), "CRACKERS", "sea salt", seed=26),
    "pasta_box": lambda: T.product_box(256, 384, (.10, .30, .65), (.95, .85, .30), "PENNE", "500 g", seed=27),
    "can_tomato": lambda: T.can_label(512, 256, (.80, .12, .10), (.20, .50, .20), "TOMATOES", seed=28),
    "can_beans": lambda: T.can_label(512, 256, (.25, .45, .20), (.85, .75, .30), "BEANS", seed=29),
    "can_soup": lambda: T.can_label(512, 256, (.95, .95, .92), (.75, .10, .10), "SOUP", seed=30),
    "display_oven": lambda: T.display(256, 96, "350°F", color=(1, .45, .15)),
    "display_clock": lambda: T.display(256, 96, "12:45", color=(.25, .85, 1)),
    "hood_filter": lambda: T.grille(256, 256, color=(.72, .73, .75), slats=14, dark=(.30, .31, .33)),
    "menu_board": lambda: T.sign(640, 480, [("THIS WEEK", .16), ("Mon  — lemon pasta", .1), ("Tue  — tacos", .1),
                                            ("Wed  — salmon + rice", .1), ("Thu  — leftovers", .1), ("Fri  — pizza night", .1),
                                            ("buy: eggs, basil, milk", .1)], (.16, .20, .19), (.93, .93, .88), bold=False),
    "cardboard": lambda: T.cardboard(512, 512, seed=31, logo="FRESH GROCER"),
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
    m("floor", tex="floor_oak", tile=2.0, spec=.35, shin=.45, refl=.05)
    m("wall", tex="wall_paint", tile=1.2, spec=.05, shin=.1)
    m("ceiling", tex="ceiling_paint", tile=1.5, spec=.02, shin=.05)
    m("trim", (.97, .97, .95, 1), spec=.35, shin=.5)
    m("quartz", tex="quartz", tile=1.6, spec=.75, shin=.8)
    m("subway", tex="subway", tile=0.6, spec=.8, shin=.85)
    m("steel", tex="steel", repeat=(1, 1), spec=.8, shin=.7)
    m("steel_dark", (.55, .56, .58, 1), tex="steel", repeat=(1, 1), spec=.7, shin=.6)
    m("paint_white", (.94, .93, .90, 1), spec=.35, shin=.55)
    m("paint_island", (.16, .25, .33, 1), spec=.4, shin=.55)
    m("cab_interior", (.86, .82, .74, 1), tex="oak", repeat=(1, 1), spec=.2, shin=.3)
    m("oak", tex="oak", repeat=(1, 1), spec=.35, shin=.5)
    m("walnut", tex="walnut", repeat=(1, 1), spec=.4, shin=.55)
    m("butcher", tex="butcher", repeat=(1, 1), spec=.2, shin=.3)
    m("leather", tex="leather", repeat=(1, 1), spec=.35, shin=.4)
    m("linen", tex="linen", repeat=(1, 1), spec=.05, shin=.1)
    m("rug", tex="rug", repeat=(1, 1), spec=.02, shin=.05)
    m("backyard", tex="backyard", repeat=(2, 1), spec=0, shin=0, emission=.55)
    m("deck", tex="deck", tile=1.8, spec=.15, shin=.2)
    m("lawn", (.33, .52, .24, 1), spec=.02, shin=.05)
    m("terracotta", tex="terracotta_tex", repeat=(1, 1), spec=.1, shin=.2)
    m("wicker", tex="wicker", repeat=(2, 2), spec=.1, shin=.2)
    m("cooktop", (.055, .055, .06, 1), spec=.9, shin=.9)
    m("cast_iron", (.09, .09, .095, 1), spec=.35, shin=.3)
    m("burner", (.25, .25, .26, 1), spec=.5, shin=.5)
    m("bronze", (.20, .17, .14, 1), spec=.5, shin=.5)
    m("mirror", (.70, .76, .80, 1), spec=1, shin=1)
    m("led_warm", (1, .93, .78, 1), emission=1.0, spec=0)
    m("display_oven", tex="display_oven", emission=.9, spec=0)
    m("display_clock", tex="display_clock", emission=.9, spec=0)
    m("hood_filter", tex="hood_filter", repeat=(3, 1), spec=.7, shin=.6)
    m("menu_board", tex="menu_board", spec=.05, shin=.1)
    m("cardboard", tex="cardboard", spec=.05, shin=.1)
    m("outlet", (.96, .96, .94, 1), spec=.4, shin=.5)
    for a in ("art1", "art2", "art3", "books", "calendar", "drawing", "photo",
              "cereal_a", "cereal_b", "crackers", "pasta_box", "can_tomato", "can_beans", "can_soup"):
        m(a, tex=a, spec=.15, shin=.2)
    # food & small things
    for name, c, sp in [
        ("coffee", (.17, .09, .04, 1), .6), ("apple", (.72, .08, .07, 1), .7), ("apple_green", (.55, .75, .25, 1), .7),
        ("orange", (.95, .50, .06, 1), .4), ("lemon", (.96, .85, .22, 1), .5), ("banana", (.95, .83, .32, 1), .3),
        ("tomato", (.86, .14, .10, 1), .7), ("tomato_in", (.95, .45, .35, 1), .5), ("cucumber", (.22, .45, .16, 1), .5),
        ("cucumber_in", (.80, .88, .62, 1), .3), ("pepper_y", (.97, .78, .10, 1), .8), ("onion", (.75, .55, .40, 1), .4),
        ("bread", (.80, .56, .30, 1), .1), ("potato", (.68, .55, .38, 1), .1), ("pasta", (.95, .85, .55, 1), .2),
        ("rice", (.96, .95, .90, 1), .1), ("beans_red", (.45, .12, .10, 1), .4), ("lentils", (.85, .55, .25, 1), .2),
        ("oil", (.86, .72, .22, .8), .9), ("wine", (.35, .03, .07, .9), .9), ("bottle_green", (.12, .30, .16, .75), .9),
        ("ceramic_blue", (.30, .47, .62, 1), .7), ("ceramic_sage", (.62, .70, .60, 1), .6), ("candle", (.96, .94, .88, 1), .2),
        ("flame", (1, .76, .35, 1), 0), ("flower_pink", (.95, .55, .65, 1), .3), ("flower_white", (.97, .96, .92, 1), .3),
        ("knife_handle", (.08, .07, .06, 1), .5), ("red_appliance", (.70, .10, .10, 1), .8), ("cream_appliance", (.93, .90, .82, 1), .7),
        ("sponge", (.95, .80, .20, 1), .1), ("sponge_green", (.20, .55, .30, 1), .1), ("soap", (.40, .70, .45, .85), .9),
        ("stool_plastic", (.92, .92, .90, 1), .4), ("paper_towel", (.98, .98, .97, 1), .05), ("napkin", (.80, .30, .25, 1), .05),
        ("shade_black", (.07, .07, .08, 1), .3), ("lamp_shade", (.95, .92, .85, 1), .05),
    ]:
        m(name, c, spec=sp, shin=sp)
    s.materials["flame"]["emission"] = "1"
    s.materials["lamp_shade"]["emission"] = "0.6"


# ---------------------------------------------------------------- room shell

def wall_ns(s, ya, yb, x0, x1, openings, h=H, mat="wall"):
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


def wall_ew(s, xa, xb, y0, y1, openings, h=H, mat="wall"):
    lo, hi = min(xa, xb), max(xa, xb)
    ys = y0
    for a, b, za, zb in sorted(openings):
        if a > ys:
            s.box((lo, ys, 0), (hi, a, h), mat, "struct", name="wall")
        if za > 0:
            s.box((lo, a, 0), (hi, b, za), mat, "struct", name="wall")
        if zb < h:
            s.box((lo, a, zb), (hi, b, h), mat, "struct", name="wall")
        ys = b
    if ys < y1:
        s.box((lo, ys, 0), (hi, y1, h), mat, "struct", name="wall")


def casing(s, a, b, za, zb, sill=False, w=0.07, t=0.016):
    """Trim around an opening on a wall at local y=0 (room toward -y). a..b along x, za..zb up."""
    s.box((a - w, -t, za if not sill else za - 0.06), (a, 0, zb + w), "trim")
    s.box((b, -t, za if not sill else za - 0.06), (b + w, 0, zb + w), "trim")
    s.box((a - w - 0.01, -t - 0.006, zb), (b + w + 0.01, 0, zb + w + 0.012), "trim")
    if sill:
        s.box((a - w - 0.02, -0.07, za - 0.02), (b + w + 0.02, 0.06, za), "trim")       # stool
        s.box((a - w, -t, za - 0.11), (b + w, 0, za - 0.02), "trim")                     # apron


def window_unit(s, a, b, za, zb, depth=0.15, mullions=1):
    """Window sashes inside an opening a..b (x), za..zb (z); wall spans local y 0..depth."""
    y = depth * 0.45
    fw = 0.05
    for x0, x1 in ((a, a + 0.03), (b - 0.03, b)):
        s.box((x0, 0.005, za), (x1, depth - 0.005, zb), "trim")
    s.box((a, 0.005, zb - 0.03), (b, depth - 0.005, zb), "trim")
    s.box((a, 0.005, za), (b, depth - 0.005, za + 0.03), "trim")
    n = mullions + 1
    for k in range(n):
        xa = a + 0.03 + (b - a - 0.06) * k / n
        xb = a + 0.03 + (b - a - 0.06) * (k + 1) / n
        s.box((xa, y - 0.02, za + 0.03), (xa + fw, y + 0.02, zb - 0.03), "trim")
        s.box((xb - fw, y - 0.02, za + 0.03), (xb, y + 0.02, zb - 0.03), "trim")
        s.box((xa + fw, y - 0.02, za + 0.03), (xb - fw, y + 0.02, za + 0.03 + fw), "trim")
        s.box((xa + fw, y - 0.02, zb - 0.03 - fw), (xb - fw, y + 0.02, zb - 0.03), "trim")
        s.box((xa + fw, y - 0.004, za + 0.03 + fw), (xb - fw, y + 0.004, zb - 0.03 - fw), "glass", "struct", name="window_glass")
        if k == 0:  # crank handle
            s.box(((xa + xb) / 2 - 0.03, y - 0.035, za + 0.04), ((xa + xb) / 2 + 0.03, y - 0.02, za + 0.06), "bronze")


def shell(s):
    s.box((X0 - WT, Y0 - WT, -0.1), (X1 + WT, Y1 + WT, 0), "floor", "struct", name="floor")
    north_open = [(-0.75, 0.75, 1.05, 2.2)]
    south_open = [(-2.75, -1.85, 0, 2.1)]
    west_open = [(-2.45, -1.55, 0, 2.1)]
    east_open = [(-2.7, -0.5, 0, 2.2), (0.6, 2.0, 1.0, 2.2)]
    wall_ns(s, Y1, Y1 + WT, X0 - WT, X1 + WT, north_open)
    wall_ns(s, Y0 - WT, Y0, X0 - WT, X1 + WT, south_open)
    wall_ew(s, X0 - WT, X0, Y0, Y1, west_open)
    wall_ew(s, X1, X1 + WT, Y0, Y1, east_open)
    s.box((X0, Y0, H), (X1, Y1, H + 0.1), "ceiling", "ceiling", name="ceiling")

    # trim: casings, crown, baseboards (local frames put the wall at y=0, room at -y)
    with s.frame((0, Y1, 0)):
        casing(s, -0.75, 0.75, 1.05, 2.2, sill=True)
        window_unit(s, -0.75, 0.75, 1.05, 2.2)
        # roman shade, partly lowered
        s.box((-0.72, 0.005, 1.86), (0.72, 0.03, 2.17), "linen")
        for k in range(3):
            s.box((-0.72, -0.004, 1.86 + 0.1 * k), (0.72, 0.03, 1.9 + 0.1 * k), "linen")
    with s.frame((0, Y0, 0), PI):         # south wall, room at +y
        casing(s, 1.85, 2.75, 0, 2.1)     # doorway x -2.75..-1.85 (mirrored)
    with s.frame((X0, 0, 0), PI / 2):     # west wall: local x -> +y
        casing(s, -2.45, -1.55, 0, 2.1)
    with s.frame((X1, 0, 0), -PI / 2):    # east wall: local x -> -y
        casing(s, 0.5, 2.7, 0, 2.2)
        casing(s, -2.0, -0.6, 1.0, 2.2, sill=True)
        window_unit(s, -2.0, -0.6, 1.0, 2.2, mullions=1)
    # crown molding around the ceiling
    c = 0.07
    s.box((X0, Y1 - c, H - c), (X1, Y1, H), "trim", "ceiling")
    s.box((X0, Y0, H - c), (X1, Y0 + c, H), "trim", "ceiling")
    s.box((X0, Y0, H - c), (X0 + c, Y1, H), "trim", "ceiling")
    s.box((X1 - c, Y0, H - c), (X1, Y1, H), "trim", "ceiling")
    # baseboards where walls are exposed
    b, t = 0.10, 0.014
    for (xa, xb) in ((X0, -2.82), (-1.78, X1)):
        s.box((xa, Y0, 0), (xb, Y0 + t, b), "trim")
    for (ya, yb) in ((Y0, -2.77), (-0.43, 0.6), (2.0, Y1)):
        s.box((X1 - t, ya, 0), (X1, yb, b), "trim")
    for (ya, yb) in ((Y0, -2.52), (-1.48, 0.38)):
        s.box((X0, ya, 0), (X0 + t, yb, b), "trim")
    s.box((3.0, Y1 - t, 0), (X1, Y1, b), "trim")

    # west door (closed), recessed in the wall
    with s.frame((X0 - 0.06, -2.45, 0), PI / 2):
        s.box((0, -0.005, 0), (0.9, 0.0, 2.1), "trim")
        P.shaker(s, 0.0, 0.9, 1.06, 2.095, "paint_white", t=0.035, rail=0.12)
        P.shaker(s, 0.0, 0.9, 0.005, 1.06, "paint_white", t=0.035, rail=0.12)
        s.cyl((0.8, -0.04, 1.0), 0.03, 0.006, "brass", axis="y")
        s.cap_ft((0.8, -0.05, 1.0), (0.68, -0.06, 1.0), 0.01, "brass")
        for z in (0.25, 1.05, 1.85):
            s.box((0.0, -0.04, z), (0.012, -0.0, z + 0.09), "brass")
    s.box((X0 - 0.1, -2.45, 0), (X0 - 0.062, -1.55, 2.1), "paint_white", "struct", name="west_door_slab")

    # hallway stub behind the south doorway
    s.box((-3.2, -5.45, -0.1), (-1.4, Y0 - WT, 0), "floor", "struct", name="hall_floor")
    s.box((-3.35, -5.6, 0), (-3.2, Y0 - WT, H), "wall", "struct", name="hall_wall")
    s.box((-1.4, -5.6, 0), (-1.25, Y0 - WT, H), "wall", "struct", name="hall_wall")
    s.box((-3.35, -5.6, 0), (-1.25, -5.45, H), "wall", "struct", name="hall_wall")
    s.box((-3.35, -5.6, H), (-1.25, Y0 - WT, H + 0.1), "ceiling", "ceiling", name="hall_ceiling")
    with s.frame((-2.3, -5.45, 0), PI):
        P.framed(s, 0, 1.5, 0.6, 0.8, "art3", "oak")
    s.box((-2.75, -5.3, 0), (-1.85, -3.8, 0.006), "rug")

    # outside: deck, lawn, backyard backdrop
    s.box((-15, -15, -0.6), (20, 20, -0.25), "lawn", "struct", name="lawn")
    s.box((X1 + WT, -4.6, -0.25), (7.6, 3.2, -0.04), "deck", "struct", name="deck")
    for y in np.arange(-4.5, 3.2, 1.2):
        s.cyl((7.5, y, 0.45), 0.04, 0.5, "trim")
    s.box((7.45, -4.6, 0.9), (7.6, 3.2, 0.96), "trim")
    s.box((7.48, -4.6, 0.35), (7.55, 3.2, 0.39), "trim")
    P.plant(s, 5.0, -3.6, -0.04, pot_r=0.25, pot_h=0.45, kind="bush", seed=40, height=1.4)
    P.plant(s, 5.0, 2.4, -0.04, pot_r=0.22, pot_h=0.40, kind="bush", seed=41, height=1.2)
    # patio table outside the slider
    s.cyl((6.3, -1.6, 0.68), 0.45, 0.015, "trim")
    s.cyl((6.3, -1.6, 0.33), 0.03, 0.34, "bronze")
    with s.frame((11, 0, 0), -PI / 2):
        s.box((-12, 0, -0.6), (12, 0.05, 6.5), "backyard")
    with s.frame((0, 9, 0)):
        s.box((-12, 0, -0.6), (12, 0.05, 6.5), "backyard")


def openings_hardware(s):
    # patio slider in the east opening y -2.7..-0.5
    x = X1
    s.box((x, -2.7, 0), (x + WT, -2.66, 2.2), "bronze")
    s.box((x, -0.54, 0), (x + WT, -0.5, 2.2), "bronze")
    s.box((x, -2.7, 2.16), (x + WT, -0.5, 2.2), "bronze")
    s.box((x, -2.7, -0.0), (x + WT, -0.5, 0.025), "bronze", "struct", name="slider_sill")
    for (ya, yb, xx) in ((-2.66, -1.55, x + 0.10), (-1.62, -0.54, x + 0.05)):
        s.box((xx - 0.02, ya, 0.025), (xx + 0.02, ya + 0.06, 2.16), "bronze")
        s.box((xx - 0.02, yb - 0.06, 0.025), (xx + 0.02, yb, 2.16), "bronze")
        s.box((xx - 0.02, ya, 0.025), (xx + 0.02, yb, 0.11), "bronze")
        s.box((xx - 0.02, ya, 2.10), (xx + 0.02, yb, 2.16), "bronze")
        s.box((xx - 0.005, ya + 0.06, 0.11), (xx + 0.005, yb - 0.06, 2.10), "glass", "struct", name="slider_glass")
    s.box((x + 0.01, -1.55, 0.95), (x + 0.025, -1.53, 1.2), "bronze")


# ---------------------------------------------------------------- cabinetry (canonical: front at y=0, faces -y)

def drawer_bank(s, w, zs, paint):
    for z0, z1 in zs:
        ln = min(0.32, w * 0.5)
        P.shaker(s, 0, w, z0, z1, paint, pull=("bar", w / 2, min(z1 - 0.05, (z0 + z1) / 2 + 0.03), ln, "x"))


def base_unit(s, w, kind, paint="paint_white", d=0.62):
    if kind not in ("sink", "dishwasher", "range"):
        s.box((0, 0, TK), (w, d, CB), paint, "struct", name="base_cab")
    s.box((0, 0.07, 0), (w, d, TK), "black_matte", "struct", name="toe_kick")
    top = (CB - 0.005 - 0.165, CB - 0.005)
    if kind == "drawers3":
        drawer_bank(s, w, [top, (0.46, top[0]), (TK + 0.005, 0.46)], paint)
    elif kind in ("door1", "door1L"):
        drawer_bank(s, w, [top], paint)
        px = 0.05 if kind == "door1L" else w - 0.05
        P.shaker(s, 0, w, TK + 0.005, top[0], paint, pull=("bar", px, top[0] - 0.12, 0.14, "z"))
    elif kind == "door2":
        drawer_bank(s, w, [top], paint)
        P.shaker(s, 0, w / 2, TK + 0.005, top[0], paint, pull=("bar", w / 2 - 0.045, top[0] - 0.12, 0.14, "z"))
        P.shaker(s, w / 2, w, TK + 0.005, top[0], paint, pull=("bar", w / 2 + 0.045, top[0] - 0.12, 0.14, "z"))
    elif kind in ("spice", "trash"):
        P.shaker(s, 0, w, TK + 0.005, CB - 0.005, paint, pull=("bar", w / 2, CB - 0.07, min(0.3, w * 0.55), "x"))
    elif kind == "panel":
        for k in range(int(round(w / 0.83))):
            n = int(round(w / 0.83))
            P.shaker(s, w * k / n, w * (k + 1) / n, TK + 0.005, CB - 0.005, paint, rail=0.09)
    elif kind == "sink":
        s.box((0, 0, TK), (w, d, 0.66), paint, "struct", name="sink_cab")
        s.box((0, 0, 0.66), (0.018, d, CB), paint, "struct", name="sink_cab")
        s.box((w - 0.018, 0, 0.66), (w, d, CB), paint, "struct", name="sink_cab")
        s.box((0, d - 0.02, 0.66), (w, d, CB), paint, "struct", name="sink_cab")
        P.shaker(s, 0, w, top[0], top[1], paint)
        P.shaker(s, 0, w / 2, TK + 0.005, top[0], paint, pull=("bar", w / 2 - 0.045, top[0] - 0.12, 0.14, "z"))
        P.shaker(s, w / 2, w, TK + 0.005, top[0], paint, pull=("bar", w / 2 + 0.045, top[0] - 0.12, 0.14, "z"))
    elif kind == "dishwasher":
        s.box((0, 0.01, TK), (w, d, CB), "steel_dark", "struct", name="dishwasher")
        s.box((0.003, -0.022, TK + 0.005), (w - 0.003, 0.01, CB - 0.006), "steel")
        s.box((0.003, -0.023, CB - 0.07), (w - 0.003, -0.021, CB - 0.006), "steel_dark")
        P.bar_pull(s, w / 2, CB - 0.1, w * 0.8, "x", yface=-0.022, mat="chrome", r=0.008)
        s.box((w - 0.06, -0.024, CB - 0.03), (w - 0.05, -0.022, CB - 0.02), "led_cool")


def upper_unit(s, w, kind, paint="paint_white", d=0.34, seed=0):
    z0, z1 = UZ0, UZ1
    rng = np.random.default_rng(seed)
    if kind == "solid":
        s.box((0, 0, z0), (w, d, z1), paint, "struct", name="upper_cab")
        if w > 0.5:
            P.shaker(s, 0, w / 2, z0, z1, paint, pull=("bar", w / 2 - 0.045, z0 + 0.12, 0.14, "z"))
            P.shaker(s, w / 2, w, z0, z1, paint, pull=("bar", w / 2 + 0.045, z0 + 0.12, 0.14, "z"))
        else:
            P.shaker(s, 0, w, z0, z1, paint, pull=("bar", w - 0.045, z0 + 0.12, 0.14, "z"))
    elif kind in ("glass", "open"):
        t = 0.018
        s.box((0, 0, z0), (t, d, z1), paint, "struct", name="upper_cab")
        s.box((w - t, 0, z0), (w, d, z1), paint, "struct", name="upper_cab")
        s.box((0, 0, z0), (w, d, z0 + t), paint, "struct", name="upper_cab")
        s.box((0, 0, z1 - t), (w, d, z1), paint, "struct", name="upper_cab")
        s.box((t, d - 0.012, z0 + t), (w - t, d, z1 - t), "cab_interior")
        shelves = [z0 + t, z0 + 0.31, z0 + 0.61] if kind == "glass" else [z0 + t + (z1 - z0) * k / 4 for k in range(4)]
        for zz in shelves[1:]:
            s.box((t, 0.02, zz - 0.016), (w - t, d - 0.012, zz), "cab_interior")
        if kind == "glass":
            s.box((t + 0.02, 0.03, z1 - t - 0.008), (w - t - 0.02, 0.05, z1 - t), "led_warm")
            P.plate(s, w * 0.3, d * 0.5, shelves[0], r=0.12, stack=7)
            P.bowl(s, w * 0.72, d * 0.5, shelves[0], r=0.085, h=0.055)
            P.bowl(s, w * 0.72, d * 0.5, shelves[0] + 0.06, r=0.085, h=0.055)
            for k in range(4):
                P.tumbler(s, t + 0.06 + (w - 2 * t - 0.12) * k / 3, d * 0.5, shelves[1], fill=0)
            for k in range(3):
                P.wine_glass(s, t + 0.08 + (w - 2 * t - 0.16) * k / 2, d * 0.5, shelves[2])
            P.shaker(s, 0, w / 2, z0, z1, paint, glass=True, pull=("bar", w / 2 - 0.045, z0 + 0.12, 0.14, "z"))
            P.shaker(s, w / 2, w, z0, z1, paint, glass=True, pull=("bar", w / 2 + 0.045, z0 + 0.12, 0.14, "z"))
        else:  # open spice shelf
            for zz in shelves:
                for k in range(3):
                    x = t + 0.045 + (w - 2 * t - 0.09) * k / 2
                    c = ["paprika", "herb", "pepper"][int(rng.integers(3))]
                    s.cyl((x, d * 0.45, zz + 0.045), 0.024, 0.045, "glass")
                    s.cyl((x, d * 0.45, zz + 0.035), 0.021, 0.034, {"paprika": "tomato", "herb": "leaf_light", "pepper": "coffee"}[c])
                    s.cyl((x, d * 0.45, zz + 0.095), 0.025, 0.007, "black_matte")
    # crown + light rail + under-cabinet LED
    s.box((-0.002, -0.03, z1), (w + 0.002, d, z1 + 0.06), paint)
    s.box((-0.004, -0.045, z1 + 0.06), (w + 0.004, d, z1 + 0.08), paint)
    s.box((0, -0.02, z0 - 0.025), (w, 0.02, z0), paint)
    s.box((0.03, 0.03, z0 - 0.004), (w - 0.03, 0.05, z0), "led_warm")


def run(s, units, fn, **kw):
    x = 0.0
    for kind, w in units:
        with s.frame((x, 0, 0)):
            fn(s, w, kind, **kw)
        x += w
    return x


# ---------------------------------------------------------------- appliances

def gas_range(s, w=0.76, d=0.63):
    s.box((0, 0, 0.02), (w, d, 0.89), "steel_dark", "struct", name="range")
    s.box((0.01, 0.02, 0.0), (w - 0.01, d, 0.02), "black_matte")
    s.box((0.005, -0.005, 0.89), (w - 0.005, d - 0.02, 0.913), "cooktop", "struct", name="cooktop")
    # oven door with window and handle
    s.box((0.004, -0.022, 0.13), (w - 0.004, 0.0, 0.80), "steel")
    s.box((0.10, -0.023, 0.30), (w - 0.10, -0.021, 0.64), "black_gloss")
    s.box((0.14, -0.024, 0.33), (w - 0.14, -0.022, 0.61), "glass_dark")
    P.bar_pull(s, w / 2, 0.755, w * 0.82, "x", yface=-0.022, standoff=0.045, r=0.011, mat="chrome")
    s.box((0.004, -0.02, 0.025), (w - 0.004, 0.0, 0.12), "steel")
    P.bar_pull(s, w / 2, 0.095, w * 0.5, "x", yface=-0.02, standoff=0.025, r=0.008, mat="chrome")
    # control strip: four knobs and a clock
    s.box((0.004, -0.02, 0.805), (w - 0.004, 0.0, 0.888), "black_gloss")
    for kx in (0.09, 0.21, w - 0.21, w - 0.09):
        s.cyl((kx, -0.03, 0.846), 0.022, 0.012, "chrome", axis="y")
        s.cyl((kx, -0.022, 0.846), 0.028, 0.002, "chrome", axis="y")
        s.box((kx - 0.002, -0.043, 0.85), (kx + 0.002, -0.041, 0.866), "black_matte")
    s.box((w / 2 - 0.07, -0.021, 0.828), (w / 2 + 0.07, -0.0205, 0.865), "display_oven")
    # burners and continuous cast-iron grates
    zt = 0.913
    for bx, by, r in ((0.18, 0.15, 0.052), (w - 0.18, 0.15, 0.06), (0.18, 0.45, 0.045), (w - 0.18, 0.45, 0.05), (w / 2, 0.30, 0.04)):
        s.cyl((bx, by, zt + 0.006), r, 0.006, "burner")
        s.cyl((bx, by, zt + 0.016), r * 0.72, 0.005, "black_matte")
        for k in range(8):
            a = k / 8 * 2 * PI
            s.sphere((bx + r * 0.98 * math.cos(a), by + r * 0.98 * math.sin(a), zt + 0.012), 0.003, "burner")
    gz = zt + 0.032
    for gx0, gx1 in ((0.02, w / 2 - 0.005), (w / 2 + 0.005, w - 0.02)):
        for yy in (0.02, 0.58):
            s.box((gx0, yy - 0.008, gz - 0.008), (gx1, yy + 0.008, gz + 0.008), "cast_iron")
        for xx in (gx0, gx1):
            s.box((xx - 0.008, 0.02, gz - 0.008), (xx + 0.008, 0.58, gz + 0.008), "cast_iron")
        for k in range(1, 4):
            xx = gx0 + (gx1 - gx0) * k / 4
            s.box((xx - 0.006, 0.02, gz - 0.006), (xx + 0.006, 0.58, gz + 0.006), "cast_iron")
        s.box((gx0, 0.296, gz - 0.006), (gx1, 0.304, gz + 0.006), "cast_iron")
        for xx in (gx0, gx1):
            for yy in (0.02, 0.58):
                s.cyl((xx, yy, zt + 0.012), 0.008, 0.012, "rubber")


def hood(s, cx, w=0.80):
    yb = Y1
    z0 = 1.66
    s.box((cx - w / 2, yb - 0.50, z0), (cx + w / 2, yb, z0 + 0.06), "steel")
    s.box((cx - w / 2 + 0.03, yb - 0.47, z0 - 0.004), (cx + w / 2 - 0.03, yb - 0.03, z0), "hood_filter")
    za = z0 + 0.06
    for dw, dd, dz in ((0.04, 0.06, 0.08), (0.12, 0.13, 0.08), (0.21, 0.20, 0.07)):  # tapered canopy
        s.box((cx - w / 2 + dw, yb - 0.50 + dd, za), (cx + w / 2 - dw, yb, za + dz), "steel")
        za += dz
    s.box((cx - 0.16, yb - 0.27, z0 + 0.29), (cx + 0.16, yb, H), "steel", "ceiling")
    for dx in (-0.2, 0.2):
        s.cyl((cx + dx, yb - 0.38, z0 - 0.003), 0.03, 0.003, "led_warm")
    for k in range(4):
        s.box((cx - 0.09 + 0.05 * k, yb - 0.502, z0 + 0.02), (cx - 0.06 + 0.05 * k, yb - 0.5, z0 + 0.04), "black_matte")


def fridge(s, x0, w=0.91, d=0.70):
    """French-door fridge in canonical frame, spanning x0..x0+w, front proud of y=0."""
    x1 = x0 + w
    s.box((x0, -0.05, 0.02), (x1, d - 0.05, 1.78), "steel_dark", "struct", name="fridge")
    s.box((x0 + 0.01, -0.03, 0.0), (x1 - 0.01, 0.4, 0.02), "black_matte")
    s.box((x0, -0.08, 0.03), (x1, -0.05, 0.79), "steel")
    s.box((x0, -0.08, 0.80), (x0 + w / 2 - 0.003, -0.05, 1.78), "steel")
    s.box((x0 + w / 2 + 0.003, -0.08, 0.80), (x1, -0.05, 1.78), "steel")
    s.box((x0, -0.07, 0.79), (x1, -0.05, 0.80), "rubber")
    s.box((x0 + w / 2 - 0.003, -0.07, 0.80), (x0 + w / 2 + 0.003, -0.05, 1.78), "rubber")
    for xx in (x0 + w / 2 - 0.05, x0 + w / 2 + 0.05):
        P.bar_pull(s, xx, 1.25, 0.70, "z", yface=-0.08, standoff=0.05, r=0.012, mat="chrome")
    P.bar_pull(s, x0 + w / 2, 0.72, w * 0.75, "x", yface=-0.08, standoff=0.05, r=0.012, mat="chrome")
    # water/ice dispenser on the left door
    s.box((x0 + 0.12, -0.085, 1.12), (x0 + 0.34, -0.08, 1.46), "black_gloss")
    s.box((x0 + 0.15, -0.086, 1.38), (x0 + 0.31, -0.085, 1.44), "display_clock")
    s.box((x0 + 0.16, -0.1, 1.17), (x0 + 0.30, -0.085, 1.19), "black_matte")
    # papers and magnets on the right door
    xr = x0 + w / 2 + 0.11
    s.box((xr, -0.0815, 1.30), (xr + 0.2, -0.08, 1.55), "calendar")
    s.box((xr + 0.03, -0.0815, 1.02), (xr + 0.24, -0.08, 1.18), "drawing")
    s.box((xr + 0.24, -0.0815, 1.36), (xr + 0.33, -0.08, 1.49), "photo")
    for (mx, mz, mat) in ((xr + 0.1, 1.54, "red_paint"), (xr + 0.13, 1.17, "ceramic_blue"), (xr + 0.285, 1.48, "lemon"),
                          (xr + 0.3, 1.65, "leaf_light"), (xr + 0.06, 1.7, "orange")):
        s.cyl((mx, -0.088, mz), 0.013, 0.006, mat, axis="y")


def wall_oven_tower(s, x0, w, paint):
    d = 0.65
    s.box((x0, 0, TK), (x0 + w, d, UZ1), paint, "struct", name="oven_tower")
    s.box((x0, 0.07, 0), (x0 + w, d, TK), "black_matte", "struct", name="toe_kick")
    for z0, z1 in ((TK + 0.005, 0.38), (0.38, 0.66)):
        P.shaker(s, x0, x0 + w, z0, z1, paint, pull=("bar", x0 + w / 2, z1 - 0.06, 0.3, "x"))
    # wall oven
    ox0, ox1 = x0 + 0.02, x0 + w - 0.02
    s.box((ox0, -0.025, 0.68), (ox1, 0.0, 1.40), "steel")
    s.box((ox0 + 0.07, -0.026, 0.80), (ox1 - 0.07, -0.024, 1.18), "black_gloss")
    s.box((ox0 + 0.1, -0.027, 0.83), (ox1 - 0.1, -0.025, 1.15), "glass_dark")
    P.bar_pull(s, (ox0 + ox1) / 2, 1.23, (ox1 - ox0) * 0.8, "x", yface=-0.025, standoff=0.045, r=0.011, mat="chrome")
    s.box((ox0, -0.024, 1.29), (ox1, -0.0, 1.40), "black_gloss")
    s.box(((ox0 + ox1) / 2 - 0.08, -0.025, 1.315), ((ox0 + ox1) / 2 + 0.08, -0.024, 1.375), "display_oven")
    # microwave
    s.box((ox0, -0.025, 1.44), (ox1, 0.0, 1.86), "steel")
    s.box((ox0 + 0.03, -0.027, 1.48), (ox1 - 0.2, -0.024, 1.82), "glass_dark")
    s.box((ox1 - 0.17, -0.026, 1.48), (ox1 - 0.03, -0.024, 1.82), "black_gloss")
    s.box((ox1 - 0.15, -0.027, 1.74), (ox1 - 0.05, -0.026, 1.79), "display_clock")
    for r in range(4):
        for c in range(3):
            bx, bz = ox1 - 0.145 + 0.035 * c, 1.69 - 0.04 * r
            s.box((bx, -0.0275, bz), (bx + 0.025, -0.026, bz + 0.025), "plastic_grey")
    P.bar_pull(s, ox1 - 0.2, 1.65, 0.25, "z", yface=-0.025, standoff=0.03, r=0.009, mat="chrome")
    P.shaker(s, x0, x0 + w, 1.90, UZ1, paint, pull=("bar", x0 + w / 2, 1.95, 0.25, "x"))


def open_pantry(s, x0, w, paint):
    d, t = 0.65, 0.02
    rng = np.random.default_rng(77)
    s.box((x0, 0, 0), (x0 + t, d, UZ1), paint, "struct", name="pantry")
    s.box((x0 + w - t, 0, 0), (x0 + w, d, UZ1), paint, "struct", name="pantry")
    s.box((x0, 0, UZ1 - t), (x0 + w, d, UZ1), paint, "struct", name="pantry")
    s.box((x0 + t, 0.05, 0), (x0 + w - t, d, TK), "black_matte", "struct", name="pantry")
    s.box((x0 + t, d - 0.015, TK), (x0 + w - t, d, UZ1 - t), "cab_interior")
    levels = [TK, 0.50, 0.90, 1.30, 1.68, 2.02]
    for z in levels:
        s.box((x0 + t, 0.0, z), (x0 + w - t, d - 0.015, z + 0.022), "cab_interior", "struct", name="pantry_shelf")
    zz = [z + 0.022 for z in levels]
    xa, xb = x0 + t + 0.03, x0 + w - t - 0.03
    # bottom: wicker baskets with potatoes / onions
    for k, (veg, c) in enumerate((("potato", (xa + 0.15, 0.3)), ("onion", (xb - 0.15, 0.3)))):
        s.box((c[0] - 0.14, c[1] - 0.18, zz[0]), (c[0] + 0.14, c[1] + 0.18, zz[0] + 0.2), "wicker")
        for _ in range(9):
            s.ell((c[0] + rng.uniform(-.1, .1), c[1] + rng.uniform(-.13, .13), zz[0] + 0.2 + rng.uniform(0, .03)),
                  (0.04, 0.03, 0.03), veg, R=rot(rz=rng.random() * 3))
    # shelf 1: cereal and cracker boxes
    x = xa
    for mat, bw, bh in (("cereal_a", 0.07, 0.30), ("cereal_b", 0.07, 0.30), ("crackers", 0.06, 0.22), ("pasta_box", 0.06, 0.20),
                        ("cereal_a", 0.07, 0.30)):
        s.box((x, 0.12, zz[1]), (x + bw, 0.32, zz[1] + bh), mat)
        x += bw + 0.025
    # shelf 2: cans
    for row in range(2):
        for k in range(6):
            cx = xa + 0.04 + k * 0.105
            mat = ("can_tomato", "can_beans", "can_soup")[(k + row) % 3]
            s.cyl((cx, 0.15 + row * 0.15, zz[2] + 0.055), 0.037, 0.055, mat)
            s.cyl((cx, 0.15 + row * 0.15, zz[2] + 0.112), 0.035, 0.002, "chrome")
    # shelf 3: glass jars of pasta, rice, beans, lentils
    for k, fill in enumerate(("pasta", "rice", "beans_red", "lentils", "pasta")):
        cx = xa + 0.06 + k * 0.13
        s.cyl((cx, 0.3, zz[3] + 0.11), 0.055, 0.11, "glass")
        s.cyl((cx, 0.3, zz[3] + 0.08), 0.051, 0.075, fill)
        s.cyl((cx, 0.3, zz[3] + 0.228), 0.057, 0.012, "oak")
    # shelf 4: oils, bottles, spare paper towels
    for k, (mat, r, h) in enumerate((("oil", 0.035, 0.25), ("bottle_green", 0.04, 0.26), ("wine", 0.037, 0.26), ("oil", 0.03, 0.2))):
        P.bottle(s, xa + 0.05 + k * 0.1, 0.25, zz[4], r, h, mat, cap="black_matte")
    for k in range(2):
        s.cyl((xb - 0.06 - k * 0.12, 0.35, zz[4] + 0.13), 0.055, 0.13, "paper_towel")
    # top: serving platters and a cake stand
    s.cyl((xa + 0.15, 0.3, zz[5] + 0.12), 0.012, 0.12, "ceramic_white")
    s.cyl((xa + 0.15, 0.3, zz[5] + 0.245), 0.14, 0.006, "ceramic_white")
    s.box((xb - 0.3, 0.05, zz[5]), (xb - 0.02, 0.55, zz[5] + 0.04), "ceramic_sage")
    s.box((xb - 0.3, 0.05, zz[5] + 0.045), (xb - 0.02, 0.55, zz[5] + 0.085), "ceramic_white")


# ---------------------------------------------------------------- counter zones and props

def counters(s):
    q = "quartz"
    yf = 2.98 - 0.025
    for (xa, xb, ya, yb) in ((-3.35, -0.42, yf, Y1), (-0.42, 0.42, yf, 3.05), (-0.42, 0.42, 3.48, Y1),
                             (0.42, 1.10, yf, Y1), (1.86, 3.0, yf, Y1)):
        s.box((xa, ya, CB), (xb, yb, CT), q, "struct", name="counter")
    s.box((3.0, 2.98, 0), (3.025, Y1, CB), "paint_white", "struct", name="end_panel")
    # backsplash
    t = 0.008
    for (xa, xb, za, zb) in ((-3.35, -0.75, CT, UZ0), (-0.75, 0.75, CT, 1.03), (0.75, 1.10, CT, UZ0),
                             (1.10, 1.86, CT, 1.66), (1.86, 3.0, CT, UZ0)):
        s.box((xa, Y1 - t, za), (xb, Y1, zb), "subway")
    for ox in (-2.55, -1.25, 0.95, 2.45):
        s.box((ox - 0.035, Y1 - t - 0.006, 1.06), (ox + 0.035, Y1 - t, 1.175), "outlet")
        for dz in (1.095, 1.14):
            s.box((ox - 0.008, Y1 - t - 0.0075, dz - 0.012), (ox - 0.004, Y1 - t - 0.006, dz + 0.004), "black_matte")
            s.box((ox + 0.004, Y1 - t - 0.0075, dz - 0.012), (ox + 0.008, Y1 - t - 0.006, dz + 0.004), "black_matte")
    # undermount sink
    st = "steel"
    zb = 0.70
    s.box((-0.40, 3.06, zb - 0.01), (0.40, 3.47, zb), st, "struct", name="sink")
    s.box((-0.42, 3.05, zb), (-0.40, 3.48, CB), st, "struct", name="sink")
    s.box((0.40, 3.05, zb), (0.42, 3.48, CB), st, "struct", name="sink")
    s.box((-0.40, 3.05, zb), (0.40, 3.06, CB), st, "struct", name="sink")
    s.box((-0.40, 3.47, zb), (0.40, 3.48, CB), st, "struct", name="sink")
    s.cyl((0.0, 3.30, zb + 0.002), 0.045, 0.002, "chrome")
    s.cyl((0.0, 3.30, zb + 0.003), 0.03, 0.002, "black_matte")
    # gooseneck faucet
    fy = 3.54
    s.cyl((0, fy, CT + 0.01), 0.03, 0.01, "chrome")
    s.cyl((0, fy, CT + 0.17), 0.016, 0.16, "chrome")
    R, cz = 0.11, CT + 0.33
    pts = [(0, fy - R + R * math.cos(a), cz + R * math.sin(a)) for a in np.linspace(0, PI, 9)]
    for p0, p1 in zip(pts[:-1], pts[1:]):
        s.cap_ft(p0, p1, 0.014, "chrome")
    s.cap_ft(pts[-1], (0, pts[-1][1], cz - 0.08), 0.015, "chrome")
    s.cap_ft((0.02, fy, CT + 0.22), (0.09, fy, CT + 0.25), 0.007, "chrome")
    # soap pump, dish soap, sponge, drying rack
    s.cyl((0.22, fy, CT + 0.008), 0.02, 0.008, "chrome")
    s.cyl((0.22, fy, CT + 0.06), 0.008, 0.05, "chrome")
    s.cap_ft((0.22, fy, CT + 0.11), (0.22, fy - 0.05, CT + 0.1), 0.006, "chrome")
    P.bottle(s, -0.56, 3.48, CT, 0.03, 0.19, "soap", cap="plastic_white", neck=0.2)
    s.box((-0.66, 3.02, CT), (-0.56, 3.09, CT + 0.03), "sponge")
    s.box((-0.66, 3.02, CT + 0.03), (-0.56, 3.09, CT + 0.04), "sponge_green")
    s.box((0.55, 3.13, CT), (0.98, 3.47, CT + 0.012), "plastic_white")
    for k in range(8):
        x = 0.58 + 0.05 * k
        s.cyl((x, 3.28, CT + 0.06), 0.003, 0.05, "chrome")
    for k in range(3):
        with s.frame((0.6 + 0.07 * k, 3.30, CT + 0.13), R=rot(rx=PI / 2, rz=PI / 2)):
            s.cyl((0, 0, 0), 0.12, 0.004, "ceramic_white")
    for k in range(2):
        s.cyl((0.85 + 0.07 * k, 3.22, CT + 0.07), 0.035, 0.055, "glass")

    # coffee corner: espresso machine, grinder, cups
    with s.frame((-3.0, 3.32, CT)):
        s.box((-0.17, -0.15, 0), (0.17, 0.2, 0.38), "steel")
        s.box((-0.17, -0.15, 0.36), (0.17, 0.2, 0.39), "black_gloss")
        s.box((-0.14, -0.16, 0.02), (0.14, -0.04, 0.035), "chrome")
        s.cyl((0, -0.12, 0.22), 0.035, 0.02, "chrome")
        s.cyl((0, -0.12, 0.185), 0.03, 0.015, "black_matte")
        s.cap_ft((0, -0.12, 0.185), (0, -0.25, 0.18), 0.011, "black_matte")
        s.cap_ft((0.12, -0.15, 0.25), (0.13, -0.2, 0.06), 0.004, "chrome")
        for k in range(3):
            s.cyl((-0.08 + 0.08 * k, -0.152, 0.31), 0.016, 0.006, "chrome", axis="y")
        P.mug(s, -0.08, 0.05, 0.39, "ceramic_white", fill=None, r=0.03, h=0.06)
        P.mug(s, 0.06, 0.05, 0.39, "ceramic_sage", fill=None, r=0.03, h=0.06)
        P.mug(s, 0.0, -0.1, 0.035, "ceramic_white", r=0.03, h=0.06)
    with s.frame((-2.62, 3.4, CT)):  # grinder
        s.box((-0.06, -0.08, 0), (0.06, 0.08, 0.2), "black_gloss")
        s.cyl((0, 0, 0.27), 0.055, 0.07, "glass")
        s.cyl((0, 0, 0.24), 0.05, 0.04, "coffee")
        s.cyl((0, 0, 0.345), 0.057, 0.008, "black_gloss")
    # knife block, utensil crock, toaster, canisters
    with s.frame((-2.25, 3.42, CT), R=rot(rx=-0.35)):
        s.box((-0.07, -0.06, 0), (0.07, 0.06, 0.2), "walnut")
        for k in range(5):
            x = -0.05 + 0.025 * k
            s.box((x - 0.006, -0.04 + 0.018 * (k % 2), 0.2), (x + 0.006, -0.015 + 0.018 * (k % 2), 0.3), "knife_handle")
            s.cyl((x, -0.0275 + 0.018 * (k % 2), 0.205), 0.004, 0.006, "chrome")
    s.cyl((-1.98, 3.43, CT + 0.08), 0.06, 0.08, "ceramic_sage")
    rng = np.random.default_rng(3)
    for k in range(6):
        a = k / 6 * 2 * PI
        tip = np.array([-1.98 + 0.06 * math.cos(a), 3.43 + 0.06 * math.sin(a), CT + 0.34])
        base = np.array([-1.98 + 0.015 * math.cos(a), 3.43 + 0.015 * math.sin(a), CT + 0.03])
        s.cap_ft(base, tip, 0.006, "oak" if k % 2 else "black_matte")
        s.ell(tip + [0, 0, 0.03], (0.025, 0.012, 0.04), "oak" if k % 2 else "black_matte", R=rot(rz=a))
    with s.frame((-1.62, 3.38, CT)):  # toaster
        s.box((-0.14, -0.09, 0.01), (0.14, 0.09, 0.19), "cream_appliance")
        s.box((-0.135, -0.085, 0.0), (0.135, 0.085, 0.01), "chrome")
        for dy in (-0.035, 0.035):
            s.box((-0.11, dy - 0.012, 0.188), (0.11, dy + 0.012, 0.192), "black_matte")
        s.box((0.14, -0.015, 0.08), (0.16, 0.015, 0.1), "chrome")
        s.box((-0.03, -0.092, 0.04), (0.03, -0.09, 0.06), "chrome")
    for k, (r, h) in enumerate(((0.065, 0.2), (0.055, 0.16), (0.045, 0.12))):
        x = -1.25 + 0.15 * k
        s.cyl((x, 3.45, CT + h / 2), r, h / 2, "ceramic_white")
        s.cyl((x, 3.45, CT + h + 0.012), r * 1.02, 0.012, "oak")
        s.sphere((x, 3.45, CT + h + 0.03), 0.012, "oak")
    # right of the range: stand mixer, oil & vinegar, grinders, paper towel, boards
    with s.frame((2.2, 3.33, CT)):
        s.box((-0.1, -0.16, 0), (0.1, 0.12, 0.04), "red_appliance")
        s.box((-0.07, 0.04, 0.04), (0.07, 0.12, 0.3), "red_appliance")
        s.ell((0, -0.04, 0.33), (0.08, 0.2, 0.08), "red_appliance")
        s.cyl((0, -0.06, 0.12), 0.1, 0.08, "steel")
        s.cyl((0, -0.06, 0.04), 0.05, 0.01, "steel")
        s.cap_ft((0, -0.06, 0.25), (0, -0.06, 0.14), 0.012, "chrome")
        s.cyl((0, 0.18, 0.33), 0.02, 0.02, "chrome", axis="y")
    P.bottle(s, 1.98, 3.48, CT, 0.03, 0.26, "oil", cap="black_matte")
    P.bottle(s, 2.06, 3.50, CT, 0.028, 0.24, "wine", cap="chrome")
    for k, mat in enumerate(("walnut", "oak")):
        s.cyl((1.95 + 0.06 * k, 3.30, CT + 0.08), 0.022, 0.08, mat)
        s.sphere((1.95 + 0.06 * k, 3.30, CT + 0.165), 0.02, "chrome")
    s.cyl((2.65, 3.47, CT + 0.008), 0.07, 0.008, "walnut")
    s.cyl((2.65, 3.47, CT + 0.17), 0.008, 0.17, "walnut")
    s.cyl((2.65, 3.47, CT + 0.14), 0.06, 0.13, "paper_towel")
    for k, (w, hgt) in enumerate(((0.34, 0.45), (0.28, 0.38))):
        with s.frame((2.85 - 0.02 * k, Y1 - 0.075 - 0.03 * k, CT), R=rot(rx=-0.12)):
            s.box((-w / 2, -0.012, 0.0), (w / 2, 0.012, hgt), "butcher")
            s.cyl((0, -0.013, hgt - 0.05), 0.02, 0.001, "black_matte", axis="y")
    # on the range: stock pot, frying pan, kettle
    zt = 0.913 + 0.04
    s.cyl((1.28, 3.40, zt + 0.1), 0.12, 0.1, "steel")
    s.cyl((1.28, 3.40, zt + 0.205), 0.122, 0.006, "glass")
    s.sphere((1.28, 3.40, zt + 0.22), 0.016, "black_matte")
    for dx in (-1, 1):
        s.box((1.28 + dx * 0.12, 3.39, zt + 0.15), (1.28 + dx * 0.15, 3.41, zt + 0.17), "black_matte")
    s.cyl((1.29, 3.12, zt + 0.02), 0.13, 0.02, "cast_iron")
    s.cyl((1.29, 3.12, zt + 0.037), 0.118, 0.003, "black_gloss")
    s.cap_ft((1.16, 3.12, zt + 0.03), (0.98, 3.10, zt + 0.06), 0.012, "cast_iron")
    s.ell((1.68, 3.40, zt + 0.1), (0.1, 0.1, 0.1), "red_appliance")
    s.cyl((1.68, 3.40, zt + 0.19), 0.045, 0.012, "red_appliance")
    s.sphere((1.68, 3.40, zt + 0.21), 0.015, "black_matte")
    s.cap_ft((1.77, 3.40, zt + 0.08), (1.83, 3.40, zt + 0.16), 0.012, "red_appliance")
    for a0, a1 in ((0.0, 0.8), (0.8, 1.6), (1.6, 2.4), (2.4, 3.14)):
        p0 = (1.68 - 0.07 * math.cos(a0), 3.40, zt + 0.19 + 0.08 * math.sin(a0))
        p1 = (1.68 - 0.07 * math.cos(a1), 3.40, zt + 0.19 + 0.08 * math.sin(a1))
        s.cap_ft(p0, p1, 0.009, "black_matte")

    # windowsill herbs
    for k, x in enumerate((-0.45, 0.0, 0.45)):
        P.plant(s, x, Y1 - 0.035, 1.05, pot_r=0.045, pot_h=0.09, kind="herb", seed=50 + k)
    # on top of the uppers: baskets and a pitcher
    for x in (-3.0, -1.6):
        s.box((x - 0.18, Y1 - 0.3, UZ1 + 0.08), (x + 0.18, Y1 - 0.04, UZ1 + 0.3), "wicker")
    s.cyl((2.4, Y1 - 0.17, UZ1 + 0.2), 0.08, 0.12, "ceramic_white")
    s.ell((2.4, Y1 - 0.17, UZ1 + 0.33), (0.07, 0.07, 0.03), "ceramic_white")


def island(s):
    paint = "paint_island"
    s.box((-1.25, 0.25, TK), (1.25, 1.20, CB), paint, "struct", name="island")
    s.box((-1.2, 0.32, 0), (1.2, 1.13, TK), "black_matte", "struct", name="island")
    s.box((-1.35, -0.05, CB), (1.35, 1.23, CT), "quartz", "struct", name="island_top")
    s.box((-1.35, -0.05, 0), (-1.31, 1.23, CB), "quartz", "struct", name="island_waterfall")
    s.box((1.31, -0.05, 0), (1.35, 1.23, CB), "quartz", "struct", name="island_waterfall")
    with s.frame((1.25, 1.20, 0), PI):
        run(s, [("drawers3", 0.6), ("door2", 0.8), ("drawers3", 0.6), ("trash", 0.5)], base_unit, paint=paint)
    with s.frame((-1.25, 0.25, 0)):
        base_unit(s, 2.5, "panel", paint=paint)
    for x in (-0.6, 0.6):  # outlets under the overhang
        s.box((x - 0.035, 0.24, 0.72), (x + 0.035, 0.25, 0.835), "outlet")

    # cutting board with a knife and vegetables
    s.box((-0.78, 0.55, CT), (-0.33, 0.82, CT + 0.022), "butcher")
    z = CT + 0.022
    s.box((-0.72, 0.60, z), (-0.53, 0.625, z + 0.002), "chrome")
    s.cap_ft((-0.53, 0.6125, z + 0.008), (-0.42, 0.6125, z + 0.008), 0.008, "knife_handle")
    for k in range(4):
        s.cyl((-0.70 + 0.045 * k, 0.70, z + 0.004), 0.03, 0.004, "tomato")
        s.cyl((-0.70 + 0.045 * k, 0.70, z + 0.0085), 0.022, 0.0006, "tomato_in")
    for k in range(6):
        s.cyl((-0.48 + 0.022 * k, 0.76, z + 0.004), 0.017, 0.004, "cucumber")
        s.cyl((-0.48 + 0.022 * k, 0.76, z + 0.0085), 0.014, 0.0006, "cucumber_in")
    s.cap_ft((-0.36, 0.68, z + 0.02), (-0.36, 0.5, z + 0.02), 0.02, "cucumber")
    s.ell((-0.62, 0.9, CT + 0.045), (0.045, 0.045, 0.045), "pepper_y")
    s.cyl((-0.62, 0.9, CT + 0.095), 0.006, 0.01, "stem")
    s.ell((-0.5, 0.95, CT + 0.035), (0.035, 0.035, 0.035), "onion")
    # fruit bowl
    P.bowl(s, 0.55, 0.75, CT, r=0.16, h=0.07, mat="ceramic_blue")
    rng = np.random.default_rng(9)
    for k, (mat, r) in enumerate((("apple", .038), ("apple", .037), ("apple_green", .037), ("orange", .04), ("orange", .039), ("lemon", .033))):
        a = k / 6 * 2 * PI
        s.sphere((0.55 + 0.08 * math.cos(a), 0.75 + 0.08 * math.sin(a), CT + 0.09 + 0.01 * (k % 2)), r, mat)
    for b in range(3):
        pts = [(0.48 + 0.04 * math.cos(t) + 0.02 * b, 0.70 + 0.09 * math.sin(t) + 0.01 * b, CT + 0.14 + 0.012 * b) for t in np.linspace(-1.1, 1.1, 5)]
        for p0, p1 in zip(pts[:-1], pts[1:]):
            s.cap_ft(p0, p1, 0.017, "banana")
    # vase with flowers
    s.cyl((0.05, 0.95, CT + 0.13), 0.05, 0.13, "glass")
    s.cyl((0.05, 0.95, CT + 0.07), 0.045, 0.065, "water")
    for k in range(9):
        a = k / 9 * 2 * PI
        tip = (0.05 + 0.1 * math.cos(a) * rng.uniform(.5, 1), 0.95 + 0.1 * math.sin(a) * rng.uniform(.5, 1), CT + 0.42 + rng.uniform(0, .1))
        s.cyl_ft((0.05, 0.95, CT + 0.02), tip, 0.003, "stem")
        s.sphere(tip, rng.uniform(.025, .04), "flower_pink" if k % 3 else "flower_white")
        s.ell(((0.05 + tip[0]) / 2, (0.95 + tip[1]) / 2, CT + 0.3), (0.03, 0.01, 0.015), "leaf_light", R=rot(rz=a, ry=0.4))
    # open cookbook on a stand
    with s.frame((0.95, 0.95, CT)):
        s.box((-0.12, -0.02, 0), (0.12, 0.08, 0.012), "walnut")
        with s.frame((0, 0.03, 0.012), R=rot(rx=-0.35)):
            s.box((-0.17, -0.005, 0), (0.17, 0.005, 0.24), "walnut")
            s.box((-0.16, -0.012, 0.01), (-0.005, -0.006, 0.23), "paper_white")
            s.box((0.005, -0.012, 0.01), (0.16, -0.006, 0.23), "paper_white")
            for k in range(8):
                for side in (-1, 1):
                    s.box((side * 0.085 - 0.06, -0.0125, 0.2 - 0.02 * k), (side * 0.085 + 0.06 - 0.02 * (k % 3), -0.012, 0.204 - 0.02 * k), "black_matte")
    # pendant lights
    for x in (-0.85, 0.0, 0.85):
        y = 0.6
        s.cyl((x, y, H - 0.012), 0.06, 0.012, "brass", "ceiling")
        s.cyl_ft((x, y, H - 0.02), (x, y, 1.95), 0.004, "black_matte", "ceiling")
        for r, z, hh in ((0.04, 1.94, 0.02), (0.08, 1.915, 0.012), (0.12, 1.895, 0.01), (0.155, 1.875, 0.01), (0.18, 1.855, 0.01)):
            s.cyl((x, y, z), r, hh, "shade_black", "ceiling")
        s.cyl((x, y, 1.846), 0.182, 0.003, "brass", "ceiling")
        s.cyl((x, y, 1.842), 0.165, 0.002, "lamp_shade", "ceiling")
        s.sphere((x, y, 1.86), 0.04, "bulb", "ceiling")


def stools(s):
    for i, x in enumerate((-0.85, 0.0, 0.85)):
        with s.item(f"island_stool_{i}"), s.frame((x, -0.36, 0)):
            sz = 0.66
            s.cyl((0, 0, sz), 0.19, 0.03, "leather", "solid")
            s.cyl((0, 0, sz - 0.04), 0.17, 0.01, "black_matte", "solid")
            # low curved back
            for k in range(7):
                a = PI / 2 + (k - 3) * 0.21
                s.boxc((0.17 * math.cos(a), 0.17 * math.sin(a), sz + 0.115), (0.019, 0.012, 0.085), "leather", "solid", R=rot(rz=a + PI / 2))
            feet = [(0.21 * math.cos(a), 0.21 * math.sin(a)) for a in (PI / 4, 3 * PI / 4, 5 * PI / 4, 7 * PI / 4)]
            tops = [(0.13 * math.cos(a), 0.13 * math.sin(a)) for a in (PI / 4, 3 * PI / 4, 5 * PI / 4, 7 * PI / 4)]
            for (fx, fy), (tx, ty) in zip(feet, tops):
                s.cyl_ft((fx, fy, 0.0), (tx, ty, sz - 0.04), 0.013, "black_matte", "solid")
                s.cyl((fx, fy, 0.004), 0.016, 0.004, "rubber")
            ring = [(0.185 * math.cos(a), 0.185 * math.sin(a), 0.28) for a in (PI / 4, 3 * PI / 4, 5 * PI / 4, 7 * PI / 4)]
            for k in range(4):
                s.cyl_ft(ring[k], ring[(k + 1) % 4], 0.009, "brass", "solid")


# ---------------------------------------------------------------- dining

def dining_chair(s, x, y, facing_deg, name):
    yaw = math.radians(facing_deg) + PI / 2   # local -y faces the table
    with s.frame((x, y, 0), yaw):
        sz = 0.46
        s.box((-0.22, -0.21, sz - 0.04), (0.22, 0.21, sz), "oak", "struct", name=name)
        s.box((-0.2, -0.19, sz), (0.2, 0.17, sz + 0.03), "linen", "struct", name=name)
        for lx, ly in ((-0.19, -0.18), (0.19, -0.18)):
            s.box((lx - 0.018, ly - 0.018, 0), (lx + 0.018, ly + 0.018, sz - 0.04), "oak", "struct", name=name)
        for lx in (-0.19, 0.19):
            with s.frame((lx, 0.18, 0), R=rot(rx=-0.08)):
                s.box((-0.018, -0.018, 0), (0.018, 0.018, 0.95), "oak", "struct", name=name)
        s.box((-0.2, 0.21, 0.82), (0.2, 0.25, 0.92), "oak", "struct", name=name)
        for k in range(4):
            xx = -0.12 + 0.08 * k
            s.cyl_ft((xx, 0.2, sz), (xx, 0.235, 0.82), 0.009, "oak")
        for a, b in (((-0.19, -0.18, 0.15), (-0.19, 0.18, 0.15)), ((0.19, -0.18, 0.15), (0.19, 0.18, 0.15)),
                     ((-0.19, 0.0, 0.15), (0.19, 0.0, 0.15))):
            s.cyl_ft(a, b, 0.011, "oak")


def dining(s):
    cx, cy = 2.275, -1.35
    hw, hl = 0.475, 0.9
    s.box((1.0, -2.95, 0), (3.55, -0.2, 0.006), "rug", name="rug")
    with s.item("dining_table", pad=0.02):
        s.box((cx - hw, cy - hl, 0.72), (cx + hw, cy + hl, 0.76), "oak", "solid")
        s.box((cx - hw + 0.06, cy - hl + 0.06, 0.62), (cx + hw - 0.06, cy - hl + 0.085, 0.72), "oak", "solid")
        s.box((cx - hw + 0.06, cy + hl - 0.085, 0.62), (cx + hw - 0.06, cy + hl - 0.06, 0.72), "oak", "solid")
        s.box((cx - hw + 0.06, cy - hl + 0.06, 0.62), (cx - hw + 0.085, cy + hl - 0.06, 0.72), "oak", "solid")
        s.box((cx + hw - 0.085, cy - hl + 0.06, 0.62), (cx + hw - 0.06, cy + hl - 0.06, 0.72), "oak", "solid")
        for sx in (-1, 1):
            for sy in (-1, 1):
                lx, ly = cx + sx * (hw - 0.1), cy + sy * (hl - 0.1)
                s.box((lx - 0.035, ly - 0.035, 0.35), (lx + 0.035, ly + 0.035, 0.72), "oak", "solid")
                s.box((lx - 0.026, ly - 0.026, 0.0), (lx + 0.026, ly + 0.026, 0.35), "oak", "solid")
    seats = [(cx - 0.72, cy - 0.42, 0), (cx - 0.72, cy + 0.42, 0), (cx + 0.72, cy - 0.42, 180), (cx + 0.72, cy + 0.42, 180),
             (cx, cy - 1.2, 90), (cx, cy + 1.2, -90)]
    for i, (x, y, f) in enumerate(seats):
        dining_chair(s, x, y, f, f"dining_chair_{i}")
        # place setting in front of each chair
        d = np.array([math.cos(math.radians(f)), math.sin(math.radians(f))])
        p = np.array([x, y]) + d * (0.45 if i < 4 else 0.55)
        with s.frame((p[0], p[1], 0.76), math.radians(f) + PI / 2):
            s.box((-0.22, -0.16, 0), (0.22, 0.16, 0.003), "linen")
            P.plate(s, 0, 0, 0.003, r=0.13)
            P.plate(s, 0, 0, 0.018, r=0.1, mat="ceramic_sage")
            s.box((-0.18, -0.08, 0.003), (-0.165, 0.08, 0.006), "chrome")
            s.box((0.165, -0.08, 0.003), (0.18, 0.08, 0.006), "chrome")
            s.box((-0.215, -0.07, 0.003), (-0.19, 0.06, 0.012), "napkin")
            P.tumbler(s, 0.17, -0.12, 0.003, r=0.03, h=0.1, fill=0.55)
    # centrepiece: tray with candles, a bowl of lemons
    s.box((cx - 0.12, cy - 0.35, 0.76), (cx + 0.12, cy + 0.35, 0.785), "walnut")
    for k, h in enumerate((0.12, 0.18, 0.09)):
        y = cy - 0.2 + 0.2 * k
        s.cyl((cx, y, 0.785 + h / 2), 0.03, h / 2, "candle")
        s.ell((cx, y, 0.785 + h + 0.018), (0.007, 0.007, 0.016), "flame")
    P.bowl(s, cx + 0.3, cy + 0.0, 0.76, r=0.1, h=0.05, mat="ceramic_white")
    for k in range(4):
        s.sphere((cx + 0.3 + 0.04 * math.cos(k * 1.6), cy + 0.04 * math.sin(k * 1.6), 0.83), 0.032, "lemon")
    # chandelier
    zc = 1.78
    s.cyl((cx, cy, H - 0.012), 0.07, 0.012, "brass", "ceiling")
    s.cyl_ft((cx, cy, H - 0.02), (cx, cy, zc + 0.05), 0.006, "brass", "ceiling")
    s.sphere((cx, cy, zc + 0.04), 0.03, "brass", "ceiling")
    n, r = 8, 0.38
    for k in range(n):
        a0, a1 = 2 * PI * k / n, 2 * PI * (k + 1) / n
        p0 = (cx + r * math.cos(a0), cy + r * math.sin(a0), zc)
        p1 = (cx + r * math.cos(a1), cy + r * math.sin(a1), zc)
        s.cap_ft(p0, p1, 0.008, "brass", "ceiling")
        s.cap_ft((cx, cy, zc + 0.04), p0, 0.005, "brass", "ceiling")
        s.cyl((p0[0], p0[1], zc + 0.05), 0.012, 0.05, "candle", "ceiling")
        s.ell((p0[0], p0[1], zc + 0.125), (0.014, 0.014, 0.025), "bulb", "ceiling")


def sideboard(s):
    with s.item("sideboard"), s.frame((3.8, -3.15, 0), PI):
        w, d, h = 1.6, 0.45, 0.82
        s.box((0, 0, 0.16), (w, d, h), "walnut", "solid")
        s.box((-0.01, -0.01, h), (w + 0.01, d, h + 0.03), "walnut", "solid")
        for lx in (0.05, w - 0.05):
            for ly in (0.05, d - 0.05):
                s.cyl_ft((lx, ly, 0.16), (lx + (0.02 if lx < 1 else -0.02), ly, 0), 0.016, "walnut", "solid")
        for k in range(4):
            x0 = w * k / 4
            s.box((x0 + 0.004, -0.018, 0.18), (x0 + w / 4 - 0.004, 0, h - 0.02), "walnut")
            P.knob(s, x0 + (w / 4 - 0.04 if k % 2 == 0 else 0.04), 0.62, yface=-0.018, r=0.012)
        # lamp, books, bowl, small plant
        s.cyl((0.25, 0.22, h + 0.03 + 0.012), 0.09, 0.012, "ceramic_white")
        s.ell((0.25, 0.22, h + 0.03 + 0.15), (0.08, 0.08, 0.13), "ceramic_sage")
        s.cyl((0.25, 0.22, h + 0.03 + 0.33), 0.008, 0.06, "brass")
        s.cyl((0.25, 0.22, h + 0.03 + 0.46), 0.17, 0.11, "lamp_shade")
        s.box((0.7, 0.1, h + 0.03), (0.98, 0.32, h + 0.08), "books")
        s.box((0.73, 0.12, h + 0.08), (0.95, 0.30, h + 0.115), "art2")
        P.bowl(s, 1.25, 0.22, h + 0.03, r=0.13, h=0.06, mat="walnut")
        P.plant(s, 1.48, 0.25, h + 0.03, pot_r=0.07, pot_h=0.12, kind="snake", seed=60, height=0.45, pot_mat="ceramic_white")
    with s.frame((3.0, Y0, 0), PI):
        P.framed(s, 0, 1.62, 1.0, 0.75, "mirror", "brass", mat_board=False)


def wall_dressing(s):
    # west wall: art and the menu board, light switches
    with s.frame((X0, -0.55, 0), -PI / 2 + PI):
        P.framed(s, 0, 1.55, 0.6, 0.75, "art1", "black_matte")
    with s.frame((X0, -3.05, 0), PI / 2):
        P.framed(s, 0, 1.5, 0.45, 0.6, "art3", "oak")
    with s.frame((-0.7, Y0, 0), PI):
        P.framed(s, 0, 1.5, 1.1, 0.8, "menu_board", "oak", mat_board=False)
    with s.frame((-1.65, Y0, 0), PI):
        s.box((-0.04, -0.008, 1.15), (0.04, 0, 1.27), "outlet")
        s.box((-0.008, -0.012, 1.19), (0.008, -0.008, 1.23), "plastic_white")
    with s.frame((X1, 2.65, 0), -PI / 2):
        P.wall_clock(s, 0, 1.95, r=0.18)
    with s.frame((X1, -3.15, 0), -PI / 2):
        P.framed(s, 0, 1.55, 0.5, 0.65, "art2", "walnut")
    P.plant(s, 3.55, 3.2, 0, pot_r=0.2, pot_h=0.38, kind="fig", seed=61, height=1.5, role="struct")
    P.plant(s, 1.72, -3.3, 0, pot_r=0.15, pot_h=0.32, kind="snake", seed=62, height=0.9, role="struct")
    P.trash_can(s, -3.72, 0.05, mat="steel")
    # ceiling: recessed cans, smoke detector, supply vents
    for x in (-2.8, -1.2, 0.4, 2.0, 3.3):
        for y in (-2.6, -0.4, 2.2):
            if abs(x - 2.275) < 0.6 and abs(y + 1.35) < 1.2:
                continue
            s.cyl((x, y, H - 0.004), 0.075, 0.004, "trim", "ceiling")
            s.cyl((x, y, H - 0.006), 0.055, 0.003, "led_warm", "ceiling")
    s.cyl((-1.5, -1.2, H - 0.02), 0.07, 0.02, "plastic_white", "ceiling")
    s.sphere((-1.5, -1.2, H - 0.042), 0.006, "led_green", "ceiling")
    for (x, y) in ((-3.0, -1.0), (3.2, 1.0)):
        s.box((x - 0.15, y - 0.1, H - 0.006), (x + 0.15, y + 0.1, H), "hood_filter", "ceiling")


# ---------------------------------------------------------------- free props (dynamic)

def dynamic_props(s):
    with s.dynamic("grocery_box", (-3.55, -0.55, 0.1255)):
        s.boxc((0, 0, 0), (0.2, 0.15, 0.125), "cardboard", "dyn", mass=3.0)
    with s.dynamic("step_stool", (-3.6, -1.15, 0.1255)):
        s.boxc((0, 0, 0.1), (0.2, 0.15, 0.025), "stool_plastic", "dyn", mass=0.6)
        for sx in (-1, 1):
            s.boxc((sx * 0.17, 0, -0.025), (0.03, 0.15, 0.1), "stool_plastic", "dyn", mass=0.45)
        s.boxc((0, 0, 0.126), (0.17, 0.12, 0.002), "rubber")
    with s.dynamic("apple", (0.30, 0.16, CT + 0.0385)):
        s.sphere((0, 0, 0), 0.038, "apple", "dyn", mass=0.18)
        s.cyl((0, 0, 0.04), 0.003, 0.008, "trunk")
    with s.dynamic("mug", (-1.0, 0.18, CT + 0.0476)):
        s.cyl((0, 0, 0), 0.042, 0.0475, "ceramic_white", "dyn", mass=0.3)
        s.cyl((0, 0, 0.034), 0.037, 0.002, "coffee")
        s.boxc((0.055, 0, 0), (0.014, 0.006, 0.03), "ceramic_white")
    with s.dynamic("water_bottle", (-0.25, 0.14, CT + 0.1105)):
        s.cyl((0, 0, 0), 0.035, 0.11, "water", "dyn", mass=0.5)
        s.cyl((0, 0, 0.0), 0.0355, 0.04, "ceramic_blue")
        s.cyl((0, 0, 0.12), 0.016, 0.01, "plastic_white")
    with s.dynamic("cereal_box", (1.0, 0.25, CT + 0.1505)):
        s.boxc((0, 0, 0), (0.095, 0.035, 0.15), "cereal_b", "dyn", mass=0.45)


# ---------------------------------------------------------------- lights & cameras

def lights_cameras(s):
    # Directional fill (no shadows) + one shadow-casting key light; fixtures glow via emissive materials.
    s.light("sun", (0, 0, 8), dir=(-0.35, -0.55, -0.76), diffuse=(.30, .29, .27), directional=True, specular=(.05, .05, .05))
    s.light("sky_fill", (0, 0, 8), dir=(0.3, 0.45, -0.85), diffuse=(.20, .20, .22), directional=True, specular=(0, 0, 0))
    s.light("bounce", (0, 0, 0.1), dir=(0.15, -0.2, 0.97), diffuse=(.16, .15, .13), directional=True, specular=(0, 0, 0))
    s.light("key", (-0.5, -0.4, 2.7), dir=(0.05, 0.1, -1), diffuse=(.42, .39, .34), cutoff=88, exponent=1,
            specular=(.08, .08, .08))

    s.camera("entrance", (-2.3, -3.35, 1.62), (0.0, 1.8, 0.95), fovy=68)
    s.camera("robot_view", (-1.35, -3.45, 1.45), (-0.9, 1.5, 0.85), fovy=62)
    s.camera("island_closeup", (1.55, -0.55, 1.45), (-0.2, 0.75, 0.95), fovy=60)
    s.camera("sink_and_range", (0.35, 1.75, 1.6), (0.45, 3.45, 1.05), fovy=72)
    s.camera("fridge_pantry", (-1.1, 1.55, 1.55), (-3.6, 1.6, 1.15), fovy=72)
    s.camera("dining", (0.55, -3.25, 1.65), (2.4, -1.2, 0.75), fovy=66)
    s.camera("wide_corner", (3.75, -3.4, 2.45), (-1.0, 1.4, 0.5), fovy=78)
    s.camera("top", (0, 0, 11.5), (0, 0.0001, 0), fovy=8.4, ortho=True)
    s.default_camera = "entrance"


# ---------------------------------------------------------------- build

def build() -> Scene:
    s = Scene("g1_detailed_kitchen", "kt_")
    s.statistic = dict(center=(0, 0, 1.0), extent=6.0)
    s.visual = dict(azimuth=120, elevation=-30)
    s.headlight = dict(ambient=(.30, .30, .29), diffuse=(.20, .20, .19), specular=(.03, .03, .03))
    s.skybox = dict(rgb1=(.55, .70, .90), rgb2=(.90, .93, .96))
    materials(s)
    shell(s)
    openings_hardware(s)
    with s.frame((-3.35, 2.98, 0)):  # north base run
        run(s, [("drawers3", 0.6), ("door1L", 0.6), ("spice", 0.45), ("trash", 0.6), ("dishwasher", 0.6), ("sink", 1.0),
                ("drawers3", 0.6), ("range_gap", 0.76), ("door2", 0.6), ("door1", 0.54)],
            lambda s, w, k, **kw: gas_range(s, w) if k == "range_gap" else base_unit(s, w, k, **kw))
    with s.frame((-3.35, Y1 - 0.34, 0)):  # north uppers
        run(s, [("solid", 0.625), ("glass", 0.625), ("glass", 0.625), ("solid", 0.625)], upper_unit)
    with s.frame((0.85, Y1 - 0.34, 0)):
        run(s, [("open", 0.25)], upper_unit)
    with s.frame((1.86, Y1 - 0.34, 0)):
        run(s, [("solid", 0.57), ("solid", 0.57)], upper_unit)
    hood(s, 1.48)
    counters(s)
    with s.frame((-3.35, 0.40, 0), PI / 2):  # west tall bank: local x runs +y, fronts face +x
        paint = "paint_white"
        s.box((-0.02, 0, 0), (0, 0.65, UZ1), paint, "struct", name="tall_end_panel")
        open_pantry(s, 0.0, 0.75, paint)
        wall_oven_tower(s, 0.75, 0.80, paint)
        s.box((1.55, -0.07, 0), (1.595, 0.65, UZ1), paint, "struct", name="fridge_panel")
        s.box((2.505, -0.07, 0), (2.55, 0.65, UZ1), paint, "struct", name="fridge_panel")
        fridge(s, 1.595)
        s.box((1.595, 0.0, 1.82), (2.505, 0.65, UZ1), paint, "struct", name="bridge_cab")
        P.shaker(s, 1.595, 2.05, 1.82, UZ1, paint, pull=("bar", 2.0, 1.88, 0.14, "z"))
        P.shaker(s, 2.05, 2.505, 1.82, UZ1, paint, pull=("bar", 2.1, 1.88, 0.14, "z"))
        s.box((2.55, 0, TK), (3.2, 0.65, UZ1), paint, "struct", name="tall_pantry")
        s.box((2.55, 0.07, 0), (3.2, 0.65, TK), "black_matte", "struct", name="toe_kick")
        P.shaker(s, 2.55, 3.2, TK + 0.005, 1.62, paint, pull=("bar", 2.6, 1.05, 0.3, "z"))
        P.shaker(s, 2.55, 3.2, 1.62, UZ1, paint, pull=("bar", 2.6, 1.72, 0.14, "z"))
        s.box((-0.02, -0.03, UZ1), (3.2, 0.65, UZ1 + 0.06), paint)
        s.box((-0.03, -0.045, UZ1 + 0.06), (3.2, 0.65, UZ1 + 0.08), paint)
    island(s)
    stools(s)
    dining(s)
    sideboard(s)
    wall_dressing(s)
    dynamic_props(s)
    lights_cameras(s)
    s.meta = dict(spawn=dict(x=SPAWN[0], y=SPAWN[1], yaw_deg=SPAWN[2]),
                  clear_floor_area=dict(x=[-3.3, 0.6], y=[-3.3, -0.8]),
                  task_props=dict(chairs="6 dining chairs (real collision in both files), 3 island stools",
                                  tables="island top 0.935 m, dining table 0.76 m",
                                  liftable=["grocery_box 3 kg", "step_stool 1.5 kg", "apple", "mug", "water_bottle", "cereal_box"]))
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
    visual, nav = HERE / "KITCHEN_VISUAL.xml", HERE / "KITCHEN_G1_NAV.xml"
    baked = HERE / "assets" / "baked"
    for old_mesh in baked.glob("*.msh"):
        old_mesh.unlink()
    s.write(visual, "visual", bake=not a.no_bake)
    s.write(nav, "nav", bake=not a.no_bake)
    from envkit.preview import check, render_all
    manifest = dict(scene="Detailed residential kitchen (procedural, not a scan)", **s.stats(), **s.meta)
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
