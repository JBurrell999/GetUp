"""
Reusable props built from primitives, written in a canonical frame:
x = width, z = up, and the visible front faces -y (the viewer stands at -y).
Place them with `with s.frame(pos, yaw): ...`.
"""
from __future__ import annotations

import math

import numpy as np

from .mjcf import Scene, rot


def base_materials(s: Scene):
    """Plain materials every environment shares (names are used by the props below)."""
    m = s.mat
    m("chrome", (.86, .87, .89, 1), spec=1, shin=.95)
    m("brass", (.76, .60, .32, 1), spec=.9, shin=.8)
    m("black_matte", (.045, .045, .05, 1), spec=.15, shin=.2)
    m("black_gloss", (.03, .03, .035, 1), spec=.9, shin=.9)
    m("rubber", (.035, .035, .035, 1), spec=.05, shin=.05)
    m("plastic_white", (.93, .93, .91, 1), spec=.4, shin=.5)
    m("plastic_grey", (.42, .44, .46, 1), spec=.3, shin=.4)
    m("glass", (.80, .90, .95, .16), spec=1, shin=1)
    m("glass_dark", (.04, .05, .06, .88), spec=1, shin=1)
    m("water", (.62, .80, .95, .35), spec=1, shin=1)
    m("leaf_dark", (.13, .32, .12, 1), spec=.25, shin=.4)
    m("leaf_light", (.30, .52, .20, 1), spec=.25, shin=.4)
    m("stem", (.30, .42, .18, 1), spec=.1, shin=.2)
    m("soil", (.22, .15, .10, 1), spec=.02, shin=.1)
    m("trunk", (.38, .27, .18, 1), spec=.05, shin=.1)
    m("ceramic_white", (.95, .95, .93, 1), spec=.7, shin=.8)
    m("paper_white", (.96, .96, .94, 1), spec=.05, shin=.1)
    m("red_paint", (.78, .08, .07, 1), spec=.6, shin=.6)
    m("bulb", (1, .93, .78, 1), emission=1.0, spec=0)
    m("led_cool", (.92, .96, 1, 1), emission=1.0, spec=0)
    m("led_green", (.15, .95, .35, 1), emission=1.0, spec=0)
    m("led_red", (1, .12, .08, 1), emission=1.0, spec=0)


def bar_pull(s: Scene, x, z, length, axis="x", yface=0.0, standoff=0.03, r=0.0065, mat="brass"):
    yb = yface - standoff
    inset = min(0.025, length * 0.15)
    if axis == "x":
        s.cyl_ft((x - length / 2, yb, z), (x + length / 2, yb, z), r, mat)
        for sx in (-1, 1):
            xx = x + sx * (length / 2 - inset)
            s.cyl_ft((xx, yface, z), (xx, yb, z), r * 0.8, mat)
    else:
        s.cyl_ft((x, yb, z - length / 2), (x, yb, z + length / 2), r, mat)
        for sz in (-1, 1):
            zz = z + sz * (length / 2 - inset)
            s.cyl_ft((x, yface, zz), (x, yb, zz), r * 0.8, mat)


def knob(s: Scene, x, z, yface=0.0, r=0.014, mat="brass"):
    s.cyl_ft((x, yface, z), (x, yface - 0.022, z), r * 0.45, mat)
    s.sphere((x, yface - 0.026, z), r, mat)


def shaker(s: Scene, x0, x1, z0, z1, mat, t=0.019, rail=0.064, pull=None, glass=False, gap=0.0016):
    """Recessed-panel door/drawer front between x0..x1, z0..z1, protruding from y=0 to y=-t.

    pull: None, ('bar', x, z, length, axis) or ('knob', x, z). Coordinates relative to the front.
    """
    x0, x1, z0, z1 = x0 + gap, x1 - gap, z0 + gap, z1 - gap
    rail = min(rail, (x1 - x0) * 0.24, (z1 - z0) * 0.3)
    if glass:
        s.box((x0 + rail, -t * 0.5, z0 + rail), (x1 - rail, -t * 0.35, z1 - rail), "glass")
    else:
        s.box((x0 + rail, -t + 0.007, z0 + rail), (x1 - rail, 0, z1 - rail), mat)
        # thin bevel lines inside the frame
        s.box((x0 + rail, -t + 0.0035, z0 + rail), (x1 - rail, -t + 0.007, z0 + rail + 0.008), mat)
    s.box((x0, -t, z0), (x0 + rail, 0, z1), mat)
    s.box((x1 - rail, -t, z0), (x1, 0, z1), mat)
    s.box((x0 + rail, -t, z0), (x1 - rail, 0, z0 + rail), mat)
    s.box((x0 + rail, -t, z1 - rail), (x1 - rail, 0, z1), mat)
    if pull:
        if pull[0] == "bar":
            _, px, pz, ln, ax = pull
            bar_pull(s, px, pz, ln, ax, yface=-t)
        else:
            knob(s, pull[1], pull[2], yface=-t)


def framed(s: Scene, cx, cz, w, h, art_mat, frame_mat="black_matte", depth=0.025, border=0.035, mat_board=True, yface=0.0):
    """Picture on a wall whose surface is at y=yface (picture hangs toward -y)."""
    y0 = yface - depth
    s.box((cx - w / 2, y0, cz - h / 2), (cx + w / 2, yface, cz + h / 2), frame_mat)
    inner = border + (0.04 if mat_board else 0)
    if mat_board:
        s.box((cx - w / 2 + border, y0 - 0.001, cz - h / 2 + border), (cx + w / 2 - border, y0 + 0.004, cz + h / 2 - border), "paper_white")
    s.box((cx - w / 2 + inner, y0 - 0.002, cz - h / 2 + inner), (cx + w / 2 - inner, y0 + 0.004, cz + h / 2 - inner), art_mat)


def wall_clock(s: Scene, cx, cz, r=0.16, yface=0.0, rim="black_matte", t_hours=10.15):
    s.cyl((cx, yface - 0.025, cz), r, 0.025, rim, axis="y")
    s.cyl((cx, yface - 0.051, cz), r * 0.9, 0.002, "paper_white", axis="y")
    for k in range(12):
        a = k / 12 * 2 * math.pi
        L = 0.022 if k % 3 == 0 else 0.01
        p = np.array([math.sin(a), 0, math.cos(a)])
        c0 = np.array([cx, yface - 0.054, cz]) + p * (r * 0.8)
        s.cap_ft(c0, c0 + p * L, 0.0025 if k % 3 == 0 else 0.0015, "black_matte")
    hh, mm = t_hours % 12, (t_hours % 1) * 60
    for ang, L, w in ((hh / 12 * 2 * math.pi, r * 0.5, 0.004), (mm / 60 * 2 * math.pi, r * 0.75, 0.003)):
        p = np.array([math.sin(ang), 0, math.cos(ang)])
        c0 = np.array([cx, yface - 0.056, cz])
        s.cap_ft(c0, c0 + p * L, w, "black_matte")
    s.cyl((cx, yface - 0.058, cz), 0.007, 0.003, "red_paint", axis="y")


def plant(s: Scene, x, y, z=0.0, pot_r=0.16, pot_h=0.30, kind="bush", seed=0, pot_mat="terracotta", height=1.0, role="deco"):
    rng = np.random.default_rng(seed)
    s.cyl((x, y, z + pot_h / 2), pot_r, pot_h / 2, pot_mat, role=role)
    s.cyl((x, y, z + pot_h - 0.01), pot_r * 1.06, 0.015, pot_mat)
    s.cyl((x, y, z + pot_h - 0.012), pot_r * 0.94, 0.006, "soil")
    top = z + pot_h
    if kind == "bush":
        for _ in range(int(14 * height) + 6):
            a, rr = rng.random() * 2 * math.pi, rng.random() * pot_r * 1.6
            zz = top + 0.05 + rng.random() * 0.35 * height
            r3 = (0.06 + 0.06 * rng.random()) * np.array([1.2, 1.0, 0.7])
            s.ell((x + rr * math.cos(a), y + rr * math.sin(a), zz), r3, "leaf_dark" if rng.random() < .5 else "leaf_light",
                  R=rot(rx=rng.random(), ry=rng.random(), rz=rng.random() * 6))
    elif kind == "fig":  # fiddle-leaf fig: trunk and big leaves
        s.cyl_ft((x, y, top), (x + 0.02, y - 0.01, top + height * 0.85), 0.018, "trunk")
        for k in range(int(20 * height)):
            zz = top + height * (0.3 + 0.7 * rng.random())
            a = rng.random() * 2 * math.pi
            rr = 0.05 + 0.18 * rng.random()
            c = (x + rr * math.cos(a), y + rr * math.sin(a), zz)
            s.ell(c, (0.09, 0.012, 0.13), "leaf_dark" if k % 3 else "leaf_light",
                  R=rot(rx=0.5 * rng.random() - 0.25, ry=0.8 * rng.random() - 0.4, rz=a + math.pi / 2))
    elif kind == "snake":
        for k in range(11):
            a = rng.random() * 2 * math.pi
            rr = rng.random() * pot_r * 0.6
            hgt = height * (0.45 + 0.5 * rng.random())
            c = np.array([x + rr * math.cos(a), y + rr * math.sin(a), top + hgt / 2])
            s.ell(c, (0.035, 0.006, hgt / 2), "leaf_dark" if k % 2 else "leaf_light",
                  R=rot(rx=(rng.random() - .5) * .3, ry=(rng.random() - .5) * .3, rz=rng.random() * 3))
    elif kind == "herb":
        for k in range(16):
            a, rr = rng.random() * 2 * math.pi, rng.random() * pot_r * 0.8
            zz = top + 0.02 + rng.random() * 0.12 * height
            s.ell((x + rr * math.cos(a), y + rr * math.sin(a), zz), (0.022, 0.016, 0.008), "leaf_light",
                  R=rot(rx=rng.random(), ry=rng.random(), rz=rng.random() * 6))


def bottle(s: Scene, x, y, z, r, h, body, label=None, cap="black_matte", neck=0.35):
    hb = h * (1 - neck)
    s.cyl((x, y, z + hb / 2), r, hb / 2, body)
    if label:
        s.cyl((x, y, z + hb * 0.45), r * 1.012, hb * 0.28, label)
    s.ell((x, y, z + hb), (r, r, r * 0.6), body)
    s.cyl((x, y, z + hb + h * neck * 0.5), r * 0.32, h * neck * 0.5, body)
    s.cyl((x, y, z + h + 0.008), r * 0.36, 0.01, cap)


def mug(s: Scene, x, y, z, mat="ceramic_white", yaw=0.0, fill="coffee", r=0.042, h=0.095):
    s.cyl((x, y, z + h / 2), r, h / 2, mat)
    if fill:
        s.cyl((x, y, z + h - 0.012), r * 0.88, 0.002, fill)
    d = np.array([math.cos(yaw), math.sin(yaw), 0])
    a = np.array([x, y, z]) + d * r
    s.cap_ft(a + [0, 0, h * 0.75], a + d * 0.028 + [0, 0, h * 0.62], 0.007, mat)
    s.cap_ft(a + d * 0.028 + [0, 0, h * 0.62], a + d * 0.028 + [0, 0, h * 0.35], 0.007, mat)
    s.cap_ft(a + d * 0.028 + [0, 0, h * 0.35], a + [0, 0, h * 0.25], 0.007, mat)


def tumbler(s: Scene, x, y, z, r=0.035, h=0.11, fill=0.6, liquid="water"):
    s.cyl((x, y, z + h / 2), r, h / 2, "glass")
    s.cyl((x, y, z + 0.004), r * 0.9, 0.004, "glass")
    if fill:
        s.cyl((x, y, z + 0.008 + h * fill / 2), r * 0.9, h * fill / 2 - 0.004, liquid)


def wine_glass(s: Scene, x, y, z, liquid=None):
    s.cyl((x, y, z + 0.002), 0.034, 0.002, "glass")
    s.cyl((x, y, z + 0.055), 0.004, 0.053, "glass")
    s.ell((x, y, z + 0.15), (0.042, 0.042, 0.055), "glass")
    if liquid:
        s.ell((x, y, z + 0.128), (0.034, 0.034, 0.02), liquid)


def plate(s: Scene, x, y, z, r=0.13, mat="ceramic_white", stack=1):
    for k in range(stack):
        zz = z + k * 0.014
        s.cyl((x, y, zz + 0.004), r * 0.62, 0.004, mat)
        s.cyl((x, y, zz + 0.011), r, 0.003, mat)


def bowl(s: Scene, x, y, z, r=0.09, h=0.06, mat="ceramic_white", fill=None):
    s.cyl((x, y, z + 0.006), r * 0.45, 0.006, mat)
    n = 4
    for k in range(n):
        f = (k + 1) / n
        s.cyl((x, y, z + 0.012 + h * (k + 0.5) / n), r * (0.55 + 0.45 * f), h / n / 2, mat)
    if fill:
        s.cyl((x, y, z + h * 0.92), r * 0.9, 0.003, fill)


def fire_extinguisher(s: Scene, x, y, z=0.0, wall_yaw=0.0):
    """Wall-hung extinguisher; the wall is at local +y."""
    with s.frame((x, y, z), wall_yaw):
        s.box((-0.05, 0.0, 0.95), (0.05, 0.02, 1.0), "black_matte")
        s.cyl((0, -0.07, 0.62), 0.075, 0.23, "red_paint")
        s.ell((0, -0.07, 0.85), (0.075, 0.075, 0.05), "red_paint")
        s.cyl((0, -0.07, 0.92), 0.018, 0.03, "chrome")
        s.box((-0.05, -0.08, 0.94), (0.04, -0.06, 0.955), "black_matte")
        s.cap_ft((0.02, -0.07, 0.93), (0.09, -0.11, 0.75), 0.008, "rubber")
        s.cyl((0, -0.146, 0.62), 0.045, 0.06, "paper_white", axis="y")


def trash_can(s: Scene, x, y, r=0.15, h=0.62, mat="steel", role="solid"):
    s.cyl((x, y, h / 2), r, h / 2, mat, role=role)
    s.ell((x, y, h), (r, r, 0.03), mat)
    s.box((x - 0.05, y - r - 0.04, 0.015), (x + 0.05, y - r + 0.01, 0.03), "black_matte")
    s.cyl((x, y, 0.01), r * 1.02, 0.01, "black_matte")
