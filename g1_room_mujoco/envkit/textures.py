"""
Procedural textures (numpy + PIL only), all deterministic from a seed.

Surface textures (floors, walls, counters) are tileable so they can repeat across large
geoms. Signage/label textures are drawn so that they read correctly on a box's local -y
face, which is how envkit places signs.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ---------------------------------------------------------------- basics

_FONTS = {
    False: ["/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf"],
    True: ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Supplemental/Arial Black.ttf",
           "/Library/Fonts/Arial Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVuSans-Bold.ttf"],
}
_MONO = ["/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/SFNSMono.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "DejaVuSansMono.ttf"]


def font(size: int, bold: bool = False, mono: bool = False):
    for path in (_MONO if mono else _FONTS[bold]):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def rgb(c) -> np.ndarray:
    return np.asarray(c, np.float32).reshape(1, 1, 3)


def to_img(a: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(a * 255 if a.max() <= 1.001 else a, 0, 255).astype(np.uint8))


def noise(h: int, w: int, ky: int, kx: int, rng) -> np.ndarray:
    """Tileable value noise with ky x kx lattice cells, smoothstep interpolation, range 0..1."""
    g = rng.random((ky, kx)).astype(np.float32)
    ys, xs = np.arange(h) * ky / h, np.arange(w) * kx / w
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    fy, fx = ys - y0, xs - x0
    fy, fx = fy * fy * (3 - 2 * fy), fx * fx * (3 - 2 * fx)
    y1, x1 = (y0 + 1) % ky, (x0 + 1) % kx
    a, b = g[y0][:, x0], g[y0][:, x1]
    c, d = g[y1][:, x0], g[y1][:, x1]
    fx, fy = fx[None, :], fy[:, None]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def fbm(h: int, w: int, rng, base=(4, 4), octaves=5, persistence=0.5) -> np.ndarray:
    out, amp, tot = np.zeros((h, w), np.float32), 1.0, 0.0
    ky, kx = base
    for _ in range(octaves):
        out += amp * noise(h, w, max(1, min(ky, h)), max(1, min(kx, w)), rng)
        tot += amp
        amp *= persistence
        ky, kx = ky * 2, kx * 2
    return out / tot


def save(img: Image.Image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, optimize=True)


def fit_text(draw, box, text, color, bold=True, max_size=400, mono=False, anchor="mm"):
    """Draw text centred in box (x0, y0, x1, y1), shrinking to fit."""
    x0, y0, x1, y1 = box
    size = min(max_size, int((y1 - y0) * 0.9))
    while size > 6:
        f = font(size, bold, mono)
        l, t, r, b = draw.textbbox((0, 0), text, font=f)
        if r - l <= (x1 - x0) * 0.94 and b - t <= (y1 - y0) * 0.94:
            break
        size = int(size * 0.92)
    f = font(size, bold, mono)
    draw.text(((x0 + x1) / 2, (y0 + y1) / 2), text, fill=color, font=f, anchor=anchor)


# ---------------------------------------------------------------- natural materials

def wood_grain(w, h, light, dark, seed=0, rings=7.0, knots=2) -> Image.Image:
    """Grain runs along x. light/dark are 0..1 RGB."""
    rng = np.random.default_rng(seed)
    turb = fbm(h, w, rng, base=(6, 2), octaves=5)
    fine = fbm(h, w, rng, base=(64, 3), octaves=3)
    y = np.linspace(0, 1, h, endpoint=False)[:, None]
    v = np.sin((y * rings + turb * 1.6) * 2 * math.pi) * 0.5 + 0.5
    v = 0.55 * v ** 2.2 + 0.45 * fine
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    for _ in range(knots):
        cx, cy, r = rng.integers(0, w), rng.integers(0, h), rng.integers(h // 40 + 2, h // 18 + 4)
        d = np.sqrt(((xx - cx) / 2.5) ** 2 + (yy - cy) ** 2)
        v = np.where(d < r * 3, v * (0.75 + 0.25 * np.clip(d / (r * 3), 0, 1)) + 0.18 * np.exp(-d / r), v)
    v = np.clip(v, 0, 1)[..., None]
    return to_img(rgb(light) * (1 - v) + rgb(dark) * v)


def wood_planks(w, h, light, dark, rows=10, seed=0, seam=(0.12, 0.08, 0.05), vary=0.10) -> Image.Image:
    """Tileable plank floor, planks along x. `rows` planks across the tile height."""
    rng = np.random.default_rng(seed)
    base = np.asarray(wood_grain(w, h * 2, light, dark, seed=seed + 1, rings=26, knots=6), np.float32) / 255
    out = np.zeros((h, w, 3), np.float32)
    ph = h / rows
    for r in range(rows):
        y0, y1 = int(round(r * ph)), int(round((r + 1) * ph))
        n_cuts = rng.integers(1, 3)
        cuts = np.sort((rng.random(n_cuts) * w).astype(int))
        bounds = list(cuts) + [cuts[0] + w]
        for i in range(len(cuts)):
            a, b = bounds[i], bounds[i + 1]
            src_y = rng.integers(0, h * 2 - (y1 - y0))
            src_x = rng.integers(0, w)
            hue = rng.normal(0, vary * 0.25)
            tint = ((1 + rng.normal(0, vary)) * (1 + hue * np.array([1.0, 0.25, -0.8]))).astype(np.float32)
            xs = np.arange(a, b) % w
            patch = base[src_y:src_y + (y1 - y0)][:, (np.arange(b - a) + src_x) % w] * tint
            out[y0:y1][:, xs] = patch
            out[y0:y1, a % w] = seam
            out[y0:y1, (a + 1) % w] = np.asarray(seam) * 1.6
        out[y0:min(y0 + 2, h)] = np.asarray(seam) * 1.3
    return to_img(np.clip(out, 0, 1))


def marble(w, h, base, vein, seed=0, freq=3.0, turb=2.2, sharp=18, speckle=0.0) -> Image.Image:
    rng = np.random.default_rng(seed)
    n = fbm(h, w, rng, base=(3, 3), octaves=6)
    n2 = fbm(h, w, rng, base=(5, 5), octaves=5)
    xx, yy = np.meshgrid(np.linspace(0, 1, w, endpoint=False), np.linspace(0, 1, h, endpoint=False))
    v = np.exp(-sharp * np.abs(np.sin(((xx * 0.6 + yy) * freq + n * turb) * math.pi)))
    v2 = 0.5 * np.exp(-sharp * 1.6 * np.abs(np.sin(((xx - yy * 0.4) * freq * 1.7 + n2 * turb) * math.pi)))
    cloud = (fbm(h, w, rng, base=(4, 4), octaves=4) - 0.5) * 0.08
    t = np.clip(v + v2, 0, 1)[..., None]
    img = rgb(base) * (1 - t) + rgb(vein) * t + cloud[..., None]
    if speckle:
        sp = rng.random((h, w)) < speckle
        img[sp] = img[sp] * 0.6 + np.asarray(vein, np.float32) * 0.4
    return to_img(np.clip(img, 0, 1))


def concrete(w, h, seed=0, tone=(.62, .62, .60), joints=True, stains=6, polish=True) -> Image.Image:
    rng = np.random.default_rng(seed)
    n = fbm(h, w, rng, base=(4, 4), octaves=7)
    n2 = fbm(h, w, rng, base=(16, 16), octaves=3)
    img = rgb(tone) * (0.86 + 0.26 * n[..., None]) * (0.95 + 0.1 * n2[..., None])
    sp = rng.random((h, w))
    img[sp < 0.012] *= 0.78
    img[sp > 0.992] *= 1.12
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    for _ in range(stains):
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        r = rng.integers(w // 30, w // 9)
        d = np.sqrt(np.minimum(np.abs(xx - cx), w - np.abs(xx - cx)) ** 2 + np.minimum(np.abs(yy - cy), h - np.abs(yy - cy)) ** 2)
        m = np.clip(1 - d / r, 0, 1) ** 1.5 * (0.5 + 0.5 * fbm(h, w, rng, base=(8, 8), octaves=3))
        img *= (1 - 0.16 * m[..., None])
    # hairline crack
    for _ in range(2):
        x, y = rng.integers(0, w), rng.integers(0, h)
        ang = rng.random() * math.pi
        for _ in range(int(w * 0.35)):
            ang += rng.normal(0, 0.25)
            x, y = (x + math.cos(ang)) % w, (y + math.sin(ang)) % h
            img[int(y), int(x)] *= 0.7
    if joints:  # saw-cut control joints on the tile border
        for k in range(3):
            img[k, :] *= 0.55 + 0.1 * k
            img[:, k] *= 0.55 + 0.1 * k
    if polish:
        img = img * 0.96 + 0.04
    return to_img(np.clip(img, 0, 1))


def plaster(w, h, color, seed=0, amount=0.05) -> Image.Image:
    rng = np.random.default_rng(seed)
    n = fbm(h, w, rng, base=(6, 6), octaves=6)
    return to_img(np.clip(rgb(color) * (1 - amount + 2 * amount * n[..., None]), 0, 1))


def fabric(w, h, color, seed=0, weave=96, amount=0.10) -> Image.Image:
    rng = np.random.default_rng(seed)
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    t = np.sin(xx * 2 * math.pi * weave / w) * np.sin(yy * 2 * math.pi * weave / h)
    n = fbm(h, w, rng, base=(8, 8), octaves=4)
    v = 1 + amount * t + 0.12 * (n - 0.5)
    return to_img(np.clip(rgb(color) * v[..., None], 0, 1))


def brushed(w, h, color, seed=0, streak=0.10, horizontal=True) -> Image.Image:
    rng = np.random.default_rng(seed)
    n = fbm(h, w, rng, base=(h // 2, 3), octaves=3) if horizontal else fbm(h, w, rng, base=(3, w // 2), octaves=3)
    m = fbm(h, w, rng, base=(3, 3), octaves=3)
    v = 1 + streak * (n - 0.5) * 2 + 0.06 * (m - 0.5)
    return to_img(np.clip(rgb(color) * v[..., None], 0, 1))


# ---------------------------------------------------------------- tiles & panels

def tiles(w, h, cols, rows, color, grout, seed=0, offset=0.5, gw=0.035, bevel=0.10, vary=0.05, gloss=True) -> Image.Image:
    """Running-bond (offset=0.5) or grid (offset=0) tiles; tileable when cols, rows integers and rows even."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ty, tx = yy / h * rows, xx / w * cols
    row = np.floor(ty).astype(int)
    tx = tx + (row % 2) * offset
    col = np.floor(tx).astype(int) % cols
    fy, fx = ty - np.floor(ty), tx - np.floor(tx)
    edge = np.minimum(np.minimum(fx, 1 - fx) * w / cols, np.minimum(fy, 1 - fy) * h / rows)
    gpx = gw * min(w / cols, h / rows)
    tint = 1 + rng.normal(0, vary, (rows, cols, 1)).astype(np.float32)
    img = rgb(color) * tint[row % rows, col]
    shade = np.clip(edge / (gpx * 3 + 1e-6), 0, 1)
    img = img * (1 - bevel + bevel * shade[..., None])
    if gloss:
        g = fbm(h, w, rng, base=(2, 2), octaves=3)
        img = img + 0.05 * (g[..., None] - 0.5)
    img = np.where((edge < gpx)[..., None], rgb(grout), img)
    return to_img(np.clip(img, 0, 1))


def ribbed(w, h, color, ribs=8, seed=0, depth=0.25, joint=True) -> Image.Image:
    """Insulated metal wall panel: vertical trapezoidal ribs."""
    rng = np.random.default_rng(seed)
    x = np.arange(w) / w * ribs
    f = x - np.floor(x)
    prof = np.where(f < 0.15, 1 - depth, np.where(f < 0.2, 1 + depth * 0.6, np.where(f < 0.95, 1.0, 1 - depth * 0.5)))
    n = fbm(h, w, rng, base=(4, 2), octaves=4)
    img = rgb(color) * prof[None, :, None] * (0.95 + 0.1 * n[..., None])
    if joint:
        img[:3] *= 0.6
    return to_img(np.clip(img, 0, 1))


def hazard(w, h, a=(.96, .78, .05), b=(.08, .08, .08), stripes=6, worn=0.15, seed=0) -> Image.Image:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    s = ((xx / w + yy / h) * stripes) % 1.0 < 0.5
    img = np.where(s[..., None], rgb(a), rgb(b))
    n = fbm(h, w, rng, base=(8, 8), octaves=4)
    img = img * (1 - worn * (n[..., None] > 0.62))
    return to_img(np.clip(img, 0, 1))


def diamond_plate(w, h, color=(.62, .63, .65), n=16, seed=0) -> Image.Image:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u, v = xx / w * n, yy / h * n
    row = np.floor(v).astype(int)
    u = u + (row % 2) * 0.5
    fu, fv = u - np.floor(u) - 0.5, v - np.floor(v) - 0.5
    a = np.where(row % 2 == 0, 1, -1)
    d = np.abs((fu * 0.8 + a * fv * 0.6)) + np.abs(fu * 0.6 - a * fv * 0.8) * 4
    ridge = np.clip(1 - d / 0.45, 0, 1)
    br = brushed(w, h, color, seed=seed, streak=0.06)
    img = np.asarray(br, np.float32) / 255 * (0.85 + 0.35 * ridge[..., None])
    return to_img(np.clip(img, 0, 1))


def perforated(w, h, color, hole=(.06, .06, .07), seed=0) -> Image.Image:
    """Rack upright face: one teardrop slot centred in a square tile."""
    img = Image.fromarray((np.asarray(brushed(w, h, color, seed=seed, streak=0.05, horizontal=False)))).convert("RGB")
    d = ImageDraw.Draw(img)
    hc = tuple(int(c * 255) for c in hole)
    d.rounded_rectangle([w * 0.38, h * 0.18, w * 0.62, h * 0.55], radius=w * 0.1, fill=hc)
    d.ellipse([w * 0.40, h * 0.62, w * 0.60, h * 0.82], fill=hc)
    return img


def grille(w, h, color=(.85, .85, .85), slats=12, dark=(.25, .25, .27)) -> Image.Image:
    img = Image.new("RGB", (w, h), tuple(int(c * 255) for c in color))
    d = ImageDraw.Draw(img)
    for i in range(slats):
        y = (i + 0.5) * h / slats
        d.rectangle([w * 0.05, y - h / slats * 0.2, w * 0.95, y + h / slats * 0.2], fill=tuple(int(c * 255) for c in dark))
    d.rectangle([0, 0, w - 1, h - 1], outline=tuple(int(c * 200) for c in color), width=max(2, w // 40))
    return img


# ---------------------------------------------------------------- printed / signage

def _c(c):
    return tuple(int(x * 255) for x in c) if max(c) <= 1 else tuple(c)


def sign(w, h, lines, bg, fg, border=None, icon=None, bold=True, radius=0.04, stripe=None) -> Image.Image:
    """lines: list of (text, rel_height). icon: callable(draw, box) drawn on the left third."""
    img = Image.new("RGB", (w, h), _c(bg))
    d = ImageDraw.Draw(img)
    if stripe:
        d.rectangle([0, 0, w, int(h * stripe[1])], fill=_c(stripe[0]))
    if border:
        bw = max(3, int(min(w, h) * 0.035))
        d.rounded_rectangle([bw // 2, bw // 2, w - bw // 2 - 1, h - bw // 2 - 1], radius=int(min(w, h) * radius),
                            outline=_c(border), width=bw)
    x0 = int(w * 0.06)
    if icon:
        icon(d, (int(w * 0.04), int(h * 0.12), int(w * 0.04 + h * 0.76), int(h * 0.88)))
        x0 = int(w * 0.06 + h * 0.8)
    total = sum(r for _, r in lines)
    y = h * (0.5 - total / 2) if total < 1 else 0
    for text, rel in lines:
        hh = h * rel
        fit_text(d, (x0, y, w - int(w * 0.05), y + hh), text, _c(fg), bold=bold)
        y += hh
    return img


def barcode(d: ImageDraw.ImageDraw, box, rng, color=(20, 20, 20)):
    x0, y0, x1, y1 = box
    x = x0
    while x < x1:
        bw = rng.choice([1, 1, 2, 3]) * max(1, (x1 - x0) // 120)
        if rng.random() < 0.55:
            d.rectangle([x, y0, min(x + bw - 1, x1), y1], fill=color)
        x += bw


def rack_labels(w, h, codes, seed=0, color=(.98, .83, .10)) -> Image.Image:
    """A beam-face strip with one location label per code, evenly spaced."""
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (w, h), (232, 108, 30))
    d = ImageDraw.Draw(img)
    n = len(codes)
    for i, code in enumerate(codes):
        cx = (i + 0.5) * w / n
        lw = min(w / n * 0.6, h * 3.2)
        bx = (int(cx - lw / 2), int(h * 0.1), int(cx + lw / 2), int(h * 0.9))
        d.rectangle(bx, fill=_c(color) if i % 2 == 0 else (245, 245, 240))
        fit_text(d, (bx[0], bx[1], bx[0] + (bx[2] - bx[0]) * 0.55, bx[3]), code, (15, 15, 15), bold=True)
        barcode(d, (int(bx[0] + (bx[2] - bx[0]) * 0.58), int(bx[1] + h * 0.1), int(bx[2] - 4), int(bx[3] - h * 0.1)), rng)
    return img


def cardboard(w, h, seed=0, tone=(.70, .53, .34), tape=(.80, .66, .44), label=True, arrows=True, logo=None) -> Image.Image:
    rng = np.random.default_rng(seed)
    n = fbm(h, w, rng, base=(6, 6), octaves=5)
    flutes = 1 + 0.025 * np.sin(np.arange(w) / w * 2 * math.pi * 90)[None, :]
    a = rgb(tone) * (0.9 + 0.2 * n[..., None]) * flutes[..., None]
    img = to_img(np.clip(a, 0, 1))
    d = ImageDraw.Draw(img)
    # packing tape across the middle
    ty = h * 0.5
    d.rectangle([0, ty - h * 0.07, w, ty + h * 0.07], fill=_c(tape))
    d.line([0, ty - h * 0.07, w, ty - h * 0.07], fill=_c(np.asarray(tape) * 0.85), width=2)
    if label:
        lx, ly = w * (0.08 + 0.4 * rng.random()), h * 0.62
        d.rectangle([lx, ly, lx + w * 0.42, ly + h * 0.3], fill=(244, 243, 236))
        d.rectangle([lx + w * 0.02, ly + h * 0.02, lx + w * 0.4, ly + h * 0.07], fill=(30, 30, 30))
        for k in range(3):
            d.line([lx + w * 0.02, ly + h * (0.11 + 0.045 * k), lx + w * (0.18 + 0.15 * rng.random()), ly + h * (0.11 + 0.045 * k)],
                   fill=(60, 60, 60), width=max(2, h // 160))
        barcode(d, (int(lx + w * 0.22), int(ly + h * 0.1), int(lx + w * 0.4), int(ly + h * 0.27)), rng)
    if arrows:
        ink = _c(np.asarray(tone) * 0.35)
        for k in range(2):
            ax, ay = w * (0.1 + 0.08 * k), h * 0.12
            d.polygon([(ax, ay + h * 0.16), (ax + w * 0.025, ay + h * 0.16), (ax + w * 0.025, ay + h * 0.05),
                       (ax + w * 0.045, ay + h * 0.05), (ax + w * 0.0125, ay), (ax - w * 0.02, ay + h * 0.05),
                       (ax, ay + h * 0.05)], fill=ink)
        fit_text(d, (w * 0.25, h * 0.12, w * 0.62, h * 0.24), "THIS SIDE UP", ink)
    if logo:
        fit_text(d, (w * 0.6, h * 0.1, w * 0.95, h * 0.3), logo, _c(np.asarray(tone) * 0.3))
    return img


def product_box(w, h, color, accent, name, sub="", seed=0) -> Image.Image:
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (w, h), _c(color))
    d = ImageDraw.Draw(img)
    d.rectangle([0, int(h * 0.62), w, h], fill=_c(accent))
    d.ellipse([w * 0.2, h * 0.36, w * 0.8, h * 0.86], fill=_c(np.clip(np.asarray(accent) * 1.25, 0, 1)))
    for _ in range(14):
        cx, cy, r = rng.random() * w * 0.5 + w * 0.25, h * (0.5 + 0.25 * rng.random()), w * (0.03 + 0.03 * rng.random())
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=_c((0.85, 0.65, 0.3)))
    fit_text(d, (w * 0.05, h * 0.06, w * 0.95, h * 0.30), name, (255, 255, 255))
    if sub:
        fit_text(d, (w * 0.1, h * 0.30, w * 0.9, h * 0.38), sub, (255, 255, 235), bold=False)
    return img


def can_label(w, h, color, band, name, seed=0) -> Image.Image:
    img = Image.new("RGB", (w, h), _c(color))
    d = ImageDraw.Draw(img)
    d.rectangle([0, int(h * 0.15), w, int(h * 0.3)], fill=_c(band))
    d.rectangle([0, int(h * 0.75), w, int(h * 0.85)], fill=_c(band))
    for k in range(2):
        fit_text(d, (w * (0.05 + 0.5 * k), h * 0.38, w * (0.45 + 0.5 * k), h * 0.68), name, (255, 255, 255))
    return img


def books(w, h, n=14, seed=0, palette=None) -> Image.Image:
    rng = np.random.default_rng(seed)
    palette = palette or [(.55, .12, .10), (.10, .25, .45), (.18, .38, .22), (.85, .75, .55), (.22, .22, .25),
                          (.75, .45, .15), (.45, .30, .55), (.92, .92, .88), (.12, .45, .48)]
    img = Image.new("RGB", (w, h), (40, 32, 26))
    d = ImageDraw.Draw(img)
    x = 0
    while x < w:
        bw = int(w / n * (0.6 + 0.8 * rng.random()))
        top = int(h * rng.random() * 0.18)
        c = palette[rng.integers(len(palette))]
        d.rectangle([x, top, x + bw - 2, h], fill=_c(c))
        gold = (200, 170, 90)
        for k in range(rng.integers(1, 3)):
            yb = top + (h - top) * (0.1 + 0.75 * k)
            d.rectangle([x + 2, yb, x + bw - 4, yb + h * 0.015], fill=gold)
        d.rectangle([x + bw * 0.3, top + (h - top) * 0.3, x + bw * 0.7, top + (h - top) * 0.65],
                    fill=_c(np.clip(np.asarray(c) * 1.5 + .1, 0, 1)))
        x += bw
    return img


def display(w, h, text, color=(.2, 1, .45), bg=(.02, .03, .03)) -> Image.Image:
    img = Image.new("RGB", (w, h), _c(bg))
    d = ImageDraw.Draw(img)
    fit_text(d, (w * 0.08, h * 0.1, w * 0.92, h * 0.9), text, _c(color), mono=True)
    return img


def screen_ui(w, h, seed=0, title="WMS  |  Wave 14  |  Outbound", accent=(.12, .45, .85)) -> Image.Image:
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (w, h), (238, 241, 245))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w, h * 0.1], fill=_c(accent))
    fit_text(d, (w * 0.02, 0, w * 0.7, h * 0.1), title, (255, 255, 255), anchor="mm")
    d.rectangle([0, h * 0.1, w * 0.18, h], fill=(214, 220, 228))
    for k in range(7):
        d.rectangle([w * 0.02, h * (0.15 + 0.1 * k), w * 0.16, h * (0.2 + 0.1 * k)], fill=(180, 190, 205))
    for r in range(9):
        y = h * (0.14 + 0.055 * r)
        d.rectangle([w * 0.21, y, w * 0.62, y + h * 0.04], fill=(255, 255, 255) if r % 2 else (246, 248, 251))
        for c in range(4):
            d.rectangle([w * (0.22 + 0.1 * c), y + h * 0.012, w * (0.22 + 0.1 * c + 0.04 + 0.04 * rng.random()), y + h * 0.028],
                        fill=(90, 98, 110))
        st = [(60, 170, 90), (235, 170, 40), (210, 70, 60)][rng.integers(3)]
        d.ellipse([w * 0.6, y + h * 0.01, w * 0.612, y + h * 0.03], fill=st)
    d.rectangle([w * 0.65, h * 0.14, w * 0.97, h * 0.62], fill=(255, 255, 255))
    for b in range(8):
        bh = h * (0.08 + 0.3 * rng.random())
        d.rectangle([w * (0.67 + 0.037 * b), h * 0.6 - bh, w * (0.67 + 0.037 * b + 0.025), h * 0.6], fill=_c(accent))
    d.rectangle([w * 0.21, h * 0.68, w * 0.97, h * 0.95], fill=(255, 255, 255))
    pts = [(w * (0.23 + 0.72 * i / 20), h * (0.9 - 0.18 * rng.random() - 0.004 * i)) for i in range(21)]
    d.line(pts, fill=(40, 160, 120), width=max(2, w // 300))
    return img


def whiteboard(w, h, seed=0, lines=None) -> Image.Image:
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (w, h), (246, 247, 248))
    d = ImageDraw.Draw(img)
    lines = lines or []
    y = h * 0.06
    for text, col in lines:
        d.text((w * 0.05, y), text, fill=col, font=font(int(h * 0.065), bold=False))
        y += h * 0.095
    # a doodled bar chart
    for b in range(6):
        bh = h * (0.08 + 0.22 * rng.random())
        d.rectangle([w * (0.62 + 0.055 * b), h * 0.9 - bh, w * (0.62 + 0.055 * b + 0.035), h * 0.9], outline=(30, 80, 170), width=3)
    d.line([w * 0.6, h * 0.9, w * 0.96, h * 0.9], fill=(30, 30, 30), width=3)
    return img


def art_abstract(w, h, seed=0, palette=None) -> Image.Image:
    rng = np.random.default_rng(seed)
    palette = palette or [(.93, .89, .82), (.80, .45, .30), (.20, .33, .38), (.85, .70, .40), (.55, .62, .55)]
    img = Image.new("RGB", (w, h), _c(palette[0]))
    d = ImageDraw.Draw(img)
    for _ in range(7):
        c = _c(palette[rng.integers(1, len(palette))])
        if rng.random() < 0.5:
            r = w * (0.08 + 0.22 * rng.random())
            cx, cy = rng.random() * w, rng.random() * h
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=c)
        else:
            x0, y0 = rng.random() * w * 0.7, rng.random() * h * 0.7
            d.rectangle([x0, y0, x0 + w * (0.1 + 0.4 * rng.random()), y0 + h * (0.05 + 0.3 * rng.random())], fill=c)
    return img.filter(ImageFilter.GaussianBlur(1))


def backyard(w, h, seed=0) -> Image.Image:
    """View through a window: sky, clouds, trees, fence, lawn."""
    rng = np.random.default_rng(seed)
    yy = np.linspace(0, 1, h)[:, None, None]
    sky = rgb((.45, .66, .90)) * (1 - yy) + rgb((.80, .88, .96)) * yy
    a = np.repeat(sky, w, axis=1)
    cl = fbm(h, w, rng, base=(3, 6), octaves=5)
    cm = np.clip((cl - 0.55) * 3, 0, 1)[..., None] * (yy < 0.5)
    a = a * (1 - cm) + cm * 0.97
    img = to_img(np.clip(a, 0, 1))
    d = ImageDraw.Draw(img)
    for _ in range(18):  # distant trees
        cx, r = rng.random() * w, w * (0.04 + 0.06 * rng.random())
        cy = h * (0.42 + 0.06 * rng.random())
        g = (int(40 + 40 * rng.random()), int(85 + 50 * rng.random()), int(40 + 30 * rng.random()))
        d.ellipse([cx - r, cy - r * 1.3, cx + r, cy + r], fill=g)
    d.rectangle([0, h * 0.55, w, h], fill=(86, 140, 62))
    for k in range(int(w / (w * 0.035))):  # fence
        x = k * w * 0.035
        d.rectangle([x, h * 0.46, x + w * 0.03, h * 0.62], fill=(205, 190, 165))
        d.polygon([(x, h * 0.46), (x + w * 0.015, h * 0.44), (x + w * 0.03, h * 0.46)], fill=(205, 190, 165))
    d.rectangle([0, h * 0.5, w, h * 0.51], fill=(170, 155, 130))
    d.rectangle([0, h * 0.58, w, h * 0.59], fill=(170, 155, 130))
    lawn = np.asarray(img, np.float32) / 255
    n = fbm(h, w, rng, base=(20, 60), octaves=3)
    lawn[int(h * 0.62):] *= (0.85 + 0.3 * n[int(h * 0.62):, :, None])
    return to_img(np.clip(lawn, 0, 1)).filter(ImageFilter.GaussianBlur(1.2))


def rug(w, h, seed=0, field=(.62, .22, .18), border=(.18, .20, .30), accent=(.88, .80, .62)) -> Image.Image:
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (w, h), _c(border))
    d = ImageDraw.Draw(img)
    m = int(min(w, h) * 0.09)
    d.rectangle([m, m, w - m, h - m], fill=_c(field))
    d.rectangle([m * 0.45, m * 0.45, w - m * 0.45, h - m * 0.45], outline=_c(accent), width=max(3, m // 6))
    cx, cy = w / 2, h / 2
    for k in range(5, 0, -1):
        r = min(w, h) * 0.06 * k
        d.polygon([(cx, cy - r), (cx + r * 1.4, cy), (cx, cy + r), (cx - r * 1.4, cy)],
                  fill=_c(accent if k % 2 else border))
    for i in range(12):
        x = m + (w - 2 * m) * (i + 0.5) / 12
        for y in (m * 1.6, h - m * 1.6):
            d.polygon([(x, y - m * 0.3), (x + m * 0.3, y), (x, y + m * 0.3), (x - m * 0.3, y)], fill=_c(accent))
    a = np.asarray(img, np.float32) / 255
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    t = 1 + 0.06 * np.sin(xx * 0.9) * np.sin(yy * 0.9)
    n = fbm(h, w, rng, base=(10, 10), octaves=4)
    a = a * t[..., None] * (0.92 + 0.16 * n[..., None])
    return to_img(np.clip(a, 0, 1))


def paper_note(w, h, kind="calendar", seed=0) -> Image.Image:
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (w, h), (250, 250, 246))
    d = ImageDraw.Draw(img)
    if kind == "calendar":
        d.rectangle([0, 0, w, h * 0.18], fill=(200, 60, 55))
        fit_text(d, (0, 0, w, h * 0.18), "NOVEMBER", (255, 255, 255))
        for r in range(5):
            for c in range(7):
                x0, y0 = w * (0.04 + 0.13 * c), h * (0.22 + 0.15 * r)
                d.rectangle([x0, y0, x0 + w * 0.12, y0 + h * 0.13], outline=(170, 170, 170))
                d.text((x0 + 3, y0 + 2), str(r * 7 + c + 1), fill=(60, 60, 60), font=font(max(8, h // 30)))
                if rng.random() < 0.18:
                    d.line([x0 + 4, y0 + h * 0.09, x0 + w * 0.1, y0 + h * 0.09], fill=(40, 80, 190), width=2)
    elif kind == "drawing":
        d.rectangle([0, h * 0.75, w, h], fill=(120, 190, 90))
        d.ellipse([w * 0.7, h * 0.05, w * 0.92, h * 0.27], fill=(255, 210, 40))
        d.rectangle([w * 0.2, h * 0.45, w * 0.55, h * 0.78], fill=(220, 80, 70))
        d.polygon([(w * 0.15, h * 0.45), (w * 0.375, h * 0.22), (w * 0.6, h * 0.45)], fill=(140, 90, 60))
        d.rectangle([w * 0.33, h * 0.6, w * 0.42, h * 0.78], fill=(90, 60, 40))
    else:  # photo
        a = np.asarray(art_abstract(w, h, seed=seed, palette=[(.6, .75, .9), (.95, .8, .65), (.3, .5, .3), (.85, .55, .45)]))
        img = Image.fromarray(a).filter(ImageFilter.GaussianBlur(3))
        ImageDraw.Draw(img).rectangle([0, 0, w - 1, h - 1], outline=(255, 255, 255), width=max(4, w // 20))
    return img


def vending_front(w, h, seed=0) -> Image.Image:
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (w, h), (30, 32, 38))
    d = ImageDraw.Draw(img)
    d.rectangle([w * 0.05, h * 0.05, w * 0.72, h * 0.78], fill=(70, 80, 92))
    for r in range(6):
        for c in range(6):
            x0, y0 = w * (0.07 + 0.108 * c), h * (0.07 + 0.118 * r)
            col = [(220, 60, 50), (240, 190, 40), (60, 120, 220), (60, 170, 90), (240, 120, 40), (150, 80, 170)][rng.integers(6)]
            d.rectangle([x0, y0 + h * 0.03, x0 + w * 0.09, y0 + h * 0.1], fill=col)
            d.line([x0, y0 + h * 0.105, x0 + w * 0.1, y0 + h * 0.105], fill=(180, 180, 190), width=2)
    d.rectangle([w * 0.76, h * 0.1, w * 0.95, h * 0.2], fill=(20, 60, 40))
    fit_text(d, (w * 0.76, h * 0.1, w * 0.95, h * 0.2), "1.75", (90, 255, 140), mono=True)
    for r in range(4):
        for c in range(3):
            d.rectangle([w * (0.77 + 0.06 * c), h * (0.25 + 0.06 * r), w * (0.81 + 0.06 * c), h * (0.29 + 0.06 * r)], fill=(190, 190, 195))
    d.rectangle([w * 0.1, h * 0.84, w * 0.68, h * 0.95], fill=(15, 15, 18))
    fit_text(d, (w * 0.05, h * 0.78, w * 0.72, h * 0.84), "SNACKS  ·  DRINKS", (235, 235, 240))
    return img


def sectional_door(w, h, color=(.86, .87, .88), sections=5, windows_row=1, seed=0) -> Image.Image:
    rng = np.random.default_rng(seed)
    a = np.asarray(brushed(w, h, color, seed=seed, streak=0.04), np.float32) / 255
    img = to_img(a)
    d = ImageDraw.Draw(img)
    sh = h / sections
    for s in range(sections):
        y0 = s * sh
        d.line([0, y0, w, y0], fill=(110, 112, 116), width=max(3, h // 150))
        for k in (0.33, 0.66):
            d.line([0, y0 + sh * k, w, y0 + sh * k], fill=(185, 187, 190), width=max(1, h // 400))
        if s == windows_row:
            for i in range(4):
                x0 = w * (0.08 + 0.23 * i)
                d.rectangle([x0, y0 + sh * 0.25, x0 + w * 0.16, y0 + sh * 0.75], fill=(55, 65, 75), outline=(40, 40, 40), width=3)
    return img


def rubber_mat(w, h, color=(.10, .10, .11), holes=12) -> Image.Image:
    img = Image.new("RGB", (w, h), _c(color))
    d = ImageDraw.Draw(img)
    s = w / holes
    for i in range(holes):
        for j in range(int(h / s)):
            cx, cy = (i + 0.5) * s, (j + 0.5) * s
            d.ellipse([cx - s * 0.28, cy - s * 0.28, cx + s * 0.28, cy + s * 0.28], fill=_c(np.asarray(color) * 0.45))
    return img


def pallet_top(w, h, seed=0, boards=7) -> Image.Image:
    wood = np.asarray(wood_grain(w, h, (.80, .68, .50), (.58, .44, .28), seed=seed, rings=18, knots=4), np.float32) / 255
    out = np.full_like(wood, 0.08)
    bw = h / (boards * 1.35)
    gap = (h - boards * bw) / (boards - 1)
    for b in range(boards):
        y0 = int(b * (bw + gap))
        out[y0:int(y0 + bw)] = wood[y0:int(y0 + bw)] * (0.9 + 0.2 * np.random.default_rng(seed + b).random())
    return to_img(np.clip(out, 0, 1))


def floor_stencil(w, h, text, bg, fg, border=None) -> Image.Image:
    img = Image.new("RGB", (w, h), _c(bg))
    d = ImageDraw.Draw(img)
    if border:
        d.rectangle([0, 0, w - 1, h - 1], outline=_c(border), width=max(6, h // 14))
    fit_text(d, (w * 0.06, h * 0.12, w * 0.94, h * 0.88), text, _c(fg))
    a = np.asarray(img, np.float32) / 255
    rng = np.random.default_rng(7)
    n = fbm(h, w, rng, base=(6, 12), octaves=4)
    a = a * (0.9 + 0.12 * n[..., None])
    return to_img(np.clip(a, 0, 1))
