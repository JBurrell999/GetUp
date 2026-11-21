#!/usr/bin/env python3
"""
stream24.py - stream video, run OpenPose BODY_25 through OpenCV's DNN module
(no OpenPose/Caffe build needed), convert to the 24-joint SMPL skeleton, and
draw ONLY those 24 joints + 23 bones on the live picture.

    python stream24.py                         # default webcam
    python stream24.py --source 1              # another camera
    python stream24.py --source clip.mp4       # a video file
    python stream24.py --source http://PHONE_IP:8080/video   # phone stream (e.g. "IP Webcam" app)
    python stream24.py --countdown 10 --duration 20         # more time to get into position
    python stream24.py --auto                  # record push-ups automatically

Capturing an action (e.g. a push-up), hands-free:
    manual: press r -> beeping countdown -> records every camera frame at full camera fps
        for --duration seconds (or until r).
    auto (--auto or press a): put the camera side-on to you with your whole body in view.
        Recording starts (beep) once you hold a push-up/plank position for ~0.5 s, includes
        the --preroll seconds before that, and stops (beep) --auto-stop seconds after you
        leave the position. It re-arms once you're up, ready for the next set.
    Either way, full-quality pose then runs on EVERY recorded frame, using several
    OpenPose processes in parallel (--workers). The live preview uses a smaller network
    (--live-height) just so you can line yourself up. A video file as --source is
    processed the same way (e.g. a 60 fps clip from your phone's camera app).

Keys:  q / Esc quit  |  r start capture (r again: cancel countdown / stop early)
       a auto-capture on/off  |  s smoothing on/off  |  l joint numbers  |  m mirror
Mouse: by default the closest (biggest) person is shown. Click a person to lock onto
       them; click empty space to go back to auto.

Each capture writes outputs/<timestamp>/ with raw.mp4 (camera only; live captures),
capture_24pt.mp4 (skeleton drawn, same frame rate), and joints24.npz:
    uv          (T, 24, 2)  pixel coords in the original (unmirrored) frame, NaN = not seen
    conf        (T, 24)     OpenPose heatmap peak, 0 = not seen
    t           (T,)        seconds since recording started
    body25_uv   (T, 25, 2)  raw OpenPose keypoints, body25_conf (T, 25)
    joint_names, parents, image_size (w, h)
These points are 2D (OpenPose has no depth). After that, lift3d.py adds joints3d.npz: the
same 24 joints in 3D metres (z up, floor at z = 0) from MediaPipe, for MuJoCo. Replay it
with view_mujoco.py. Turn this off with --no-3d.

License note: the OpenPose model is CMU's, free for non-commercial research use only.
"""
from __future__ import annotations

import argparse
import copy
import math
import multiprocessing as mp
import os
import queue
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------- model files
HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "models" / "body_25"
PROTO = MODEL_DIR / "pose_deploy.prototxt"
WEIGHTS = MODEL_DIR / "pose_iter_584000.caffemodel"
PROTO_URLS = [
    "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/"
    "models/pose/body_25/pose_deploy.prototxt",
]
WEIGHT_URLS = [  # CMU's own server first, then public mirrors of the same file
    "http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/body_25/pose_iter_584000.caffemodel",
    "https://huggingface.co/eelxpeng/openpose/resolve/main/pose/body_25/pose_iter_584000.caffemodel",
    "https://huggingface.co/camenduru/openpose/resolve/f67df11200f96068406fe707dd5fd6b70ba0e346/"
    "pose_iter_584000.caffemodel",
]

# --------------------------------------------------------------------------- skeletons
# OpenPose BODY_25 keypoint order ("L"/"R" = the person's own left/right)
(NOSE, NECK, RSHO, RELB, RWRI, LSHO, LELB, LWRI, MIDHIP, RHIP, RKNE, RANK, LHIP, LKNE,
 LANK, REYE, LEYE, REAR, LEAR, LBTOE, LSTOE, LHEEL, RBTOE, RSTOE, RHEEL) = range(25)

SMPL24 = ["pelvis", "l_hip", "r_hip", "spine1", "l_knee", "r_knee", "spine2", "l_ankle",
          "r_ankle", "spine3", "l_foot", "r_foot", "neck", "l_collar", "r_collar", "head",
          "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
          "l_hand", "r_hand"]
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

# BGR colors: person's left = orange, right = blue, spine/head = white
C_LEFT, C_RIGHT, C_MID = (0, 140, 255), (255, 170, 0), (235, 235, 235)
COLORS = [C_LEFT if n.startswith("l_") else C_RIGHT if n.startswith("r_") else C_MID
          for n in SMPL24]


# --------------------------------------------------------------------------- 25 -> 24 mapping
# Each node maps (kp[25,2], conf[25]) -> (xy[2], conf). conf 0 means "not available".
def P(i):
    return lambda kp, c: (kp[i], c[i])


def AVG(*nodes):  # mean of nodes, all required
    def f(kp, c):
        vals = [n(kp, c) for n in nodes]
        if any(v[1] <= 0 for v in vals):
            return np.array([np.nan, np.nan]), 0.0
        return np.mean([v[0] for v in vals], axis=0), min(v[1] for v in vals)
    return f


def ANY(*nodes):  # mean of whichever nodes are available
    def f(kp, c):
        vals = [v for v in (n(kp, c) for n in nodes) if v[1] > 0]
        if not vals:
            return np.array([np.nan, np.nan]), 0.0
        return np.mean([v[0] for v in vals], axis=0), float(np.mean([v[1] for v in vals]))
    return f


def FIRST(*nodes):  # first available option
    def f(kp, c):
        for n in nodes:
            xy, cf = n(kp, c)
            if cf > 0:
                return xy, cf
        return np.array([np.nan, np.nan]), 0.0
    return f


def LERP(a, b, t):  # point fraction t of the way from a to b
    def f(kp, c):
        (pa, ca), (pb, cb) = a(kp, c), b(kp, c)
        if ca <= 0 or cb <= 0:
            return np.array([np.nan, np.nan]), 0.0
        return (1 - t) * pa + t * pb, min(ca, cb)
    return f


def EXTEND(a, b, t):  # past b along a->b by t * |ab|  (used for hands: elbow->wrist->hand)
    def f(kp, c):
        (pa, ca), (pb, cb) = a(kp, c), b(kp, c)
        if ca <= 0 or cb <= 0:
            return np.array([np.nan, np.nan]), 0.0
        return pb + t * (pb - pa), min(ca, cb)
    return f


_pelvis = FIRST(P(MIDHIP), AVG(P(LHIP), P(RHIP)))
_neck_base = FIRST(P(NECK), AVG(P(LSHO), P(RSHO)))
_head = FIRST(AVG(P(LEAR), P(REAR)), ANY(P(LEAR), P(REAR), P(LEYE), P(REYE), P(NOSE)))
_lfoot = FIRST(LERP(P(LHEEL), ANY(P(LBTOE), P(LSTOE)), 0.65), ANY(P(LBTOE), P(LSTOE)))
_rfoot = FIRST(LERP(P(RHEEL), ANY(P(RBTOE), P(RSTOE)), 0.65), ANY(P(RBTOE), P(RSTOE)))

RECIPES = [
    _pelvis,                           # 0  pelvis
    P(LHIP), P(RHIP),                  # 1  l_hip      2  r_hip
    LERP(_pelvis, _neck_base, 0.20),   # 3  spine1
    P(LKNE), P(RKNE),                  # 4  l_knee     5  r_knee
    LERP(_pelvis, _neck_base, 0.45),   # 6  spine2
    P(LANK), P(RANK),                  # 7  l_ankle    8  r_ankle
    LERP(_pelvis, _neck_base, 0.70),   # 9  spine3
    _lfoot, _rfoot,                    # 10 l_foot     11 r_foot  (ball of foot)
    LERP(_neck_base, _head, 0.25),     # 12 neck
    LERP(_neck_base, P(LSHO), 0.45),   # 13 l_collar
    LERP(_neck_base, P(RSHO), 0.45),   # 14 r_collar
    _head,                             # 15 head
    P(LSHO), P(RSHO),                  # 16 l_shoulder 17 r_shoulder
    P(LELB), P(RELB),                  # 18 l_elbow    19 r_elbow
    P(LWRI), P(RWRI),                  # 20 l_wrist    21 r_wrist
    EXTEND(P(LELB), P(LWRI), 0.30),    # 22 l_hand  (BODY_25 has no hand point: extrapolated)
    EXTEND(P(RELB), P(RWRI), 0.30),    # 23 r_hand
]
assert len(RECIPES) == 24


def body25_to_smpl24(kp25: np.ndarray, conf25: np.ndarray):
    uv = np.full((24, 2), np.nan, np.float32)
    conf = np.zeros(24, np.float32)
    for j, rec in enumerate(RECIPES):
        xy, cf = rec(kp25, conf25)
        if cf > 0:
            uv[j], conf[j] = xy, cf
    return uv, conf


# --------------------------------------------------------------------------- OpenPose inference
def _download(urls, dest: Path, min_bytes: int):
    dest.parent.mkdir(parents=True, exist_ok=True)
    for url in urls:
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            print(f"[stream24] downloading {dest.name} from {url.split('/')[2]} ...", flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": "stream24"})
            with urllib.request.urlopen(req, timeout=30) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                done = 0
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
            print()
            if tmp.stat().st_size >= min_bytes:
                tmp.replace(dest)
                return
            print(f"[stream24] file too small ({tmp.stat().st_size} bytes), trying next source")
        except Exception as e:  # noqa: BLE001 - try every mirror
            print(f"\n[stream24] failed: {e}")
        tmp.unlink(missing_ok=True)
    sys.exit(f"[stream24] could not download {dest.name}. Download it manually into {dest.parent}/ "
             f"from one of:\n  " + "\n  ".join(urls))


def load_net(target: str):
    if not PROTO.exists():
        _download(PROTO_URLS, PROTO, 10_000)
    if not WEIGHTS.exists():
        _download(WEIGHT_URLS, WEIGHTS, 90_000_000)
    net = cv2.dnn.readNetFromCaffe(str(PROTO), str(WEIGHTS))
    if target == "cuda":  # needs an OpenCV build with CUDA (the pip wheel has none)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    elif target == "opencl":
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
    return net


# BODY_25 limbs (part a, part b) and their PAF (x, y) channels, as in OpenPose's
# poseParameters.cpp. The first 24 form a tree rooted at the neck; the last 2 are redundant.
LIMBS = [(1, 8), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (8, 9), (9, 10), (10, 11),
         (8, 12), (12, 13), (13, 14), (1, 0), (0, 15), (15, 17), (0, 16), (16, 18),
         (14, 19), (19, 20), (14, 21), (11, 22), (22, 23), (11, 24), (2, 17), (5, 18)]
PAF_IDX = [(0, 1), (14, 15), (22, 23), (16, 17), (18, 19), (24, 25), (26, 27), (6, 7), (2, 3),
           (4, 5), (8, 9), (10, 11), (12, 13), (30, 31), (32, 33), (36, 37), (34, 35), (38, 39),
           (40, 41), (42, 43), (44, 45), (46, 47), (48, 49), (50, 51), (20, 21), (28, 29)]


def _peaks(m, thr):
    """Local maxima of one heatmap above thr -> [(x, y, score)] in map coords, sub-pixel."""
    h, w = m.shape
    ys, xs = np.nonzero((m >= cv2.dilate(m, np.ones((3, 3), np.uint8))) & (m > thr))
    out = []
    for x, y in zip(xs, ys):
        v = m[y, x]
        dx = dy = 0.0             # parabola fit through the neighbours
        if 0 < x < w - 1:
            den = m[y, x - 1] - 2 * v + m[y, x + 1]
            if den < 0:
                dx = 0.5 * (m[y, x - 1] - m[y, x + 1]) / den
        if 0 < y < h - 1:
            den = m[y - 1, x] - 2 * v + m[y + 1, x]
            if den < 0:
                dy = 0.5 * (m[y - 1, x] - m[y + 1, x]) / den
        out.append((x + dx, y + dy, float(v)))
    return out


def _limb_score(pafx, pafy, pa, pb, n=10):
    """How well the PAF field points from pa to pb (OpenPose's line integral). <= 0 = reject."""
    h, w = pafx.shape
    d = np.array([pb[0] - pa[0], pb[1] - pa[1]])
    norm = float(np.hypot(*d))
    if norm < 1e-3:
        return 0.0
    ts = np.linspace(0, 1, n)
    xs = np.clip(np.round(pa[0] + ts * d[0]).astype(int), 0, w - 1)
    ys = np.clip(np.round(pa[1] + ts * d[1]).astype(int), 0, h - 1)
    s = (pafx[ys, xs] * d[0] + pafy[ys, xs] * d[1]) / norm
    if (s > 0.05).mean() < 0.8:
        return 0.0
    return float(s.mean() + min(0.5 * h / norm - 1, 0))


def infer_people(net, frame: np.ndarray, net_h: int, thr: float):
    """Multi-person BODY_25 (heatmap peaks grouped by part affinity fields).
    Returns a list of (kp (25,2) px, conf (25,)) per person; missing parts are NaN / 0."""
    h, w = frame.shape[:2]
    net_w = max(16, int(round(net_h * w / h / 16)) * 16)
    # OpenPose preprocessing: img/256 - 0.5, BGR order
    blob = cv2.dnn.blobFromImage(frame, 1 / 256, (net_w, net_h), (128, 128, 128),
                                 swapRB=False, crop=False)
    net.setInput(blob)
    out = net.forward()            # (1, 78, net_h/8, net_w/8): 25 parts + bg, then 52 PAF
    hms, pafs = out[0, :25], out[0, 26:]
    oh, ow = hms.shape[1:]
    peaks = [_peaks(hms[i], thr) for i in range(25)]

    persons = []                   # each: int array (25,) of peak index per part, -1 = none
    for k, ((a, b), (cx, cy)) in enumerate(zip(LIMBS, PAF_IDX)):
        if not peaks[a] or not peaks[b]:
            continue
        cands = sorted(((s, i, j) for i, pa in enumerate(peaks[a]) for j, pb in enumerate(peaks[b])
                        if (s := _limb_score(pafs[cx], pafs[cy], pa, pb)) > 0), reverse=True)
        used_a, used_b = set(), set()
        for _, i, j in cands:      # greedy one-to-one matching, best score first
            if i in used_a or j in used_b:
                continue
            used_a.add(i)
            used_b.add(j)
            hits = [p for p in persons if p[a] == i or p[b] == j]
            if len(hits) == 1:
                p = hits[0]
                if p[a] < 0:
                    p[a] = i
                if p[b] < 0:
                    p[b] = j
            elif len(hits) == 2:   # this limb joins two partial skeletons
                p, q = hits
                if not ((p >= 0) & (q >= 0)).any():
                    p[q >= 0] = q[q >= 0]
                    persons = [r for r in persons if r is not q]
            elif k < 24:
                p = np.full(25, -1)
                p[a], p[b] = i, j
                persons.append(p)

    sx, sy = w / ow, h / oh
    people = []
    for p in persons:
        if (p >= 0).sum() < 3:
            continue
        kp = np.full((25, 2), np.nan, np.float32)
        conf = np.zeros(25, np.float32)
        for part in np.nonzero(p >= 0)[0]:
            x, y, s = peaks[part][p[part]]
            kp[part] = ((x + 0.5) * sx - 0.5, (y + 0.5) * sy - 0.5)
            conf[part] = s
        people.append((kp, conf))
    return people


# --------------------------------------------------------------------------- who to show
def person_scale(kp, conf):
    """Rough head-width-sized estimate in px from whatever is visible (bigger = closer)."""
    est = [k * float(np.hypot(*(kp[a] - kp[b])))
           for a, b, k in ((LEAR, REAR, 1.0), (LEYE, REYE, 2.4), (LSHO, RSHO, 0.4),
                           (NOSE, NECK, 0.9), (NECK, MIDHIP, 0.3))
           if conf[a] > 0 and conf[b] > 0]
    if est:
        return float(np.median(est))
    pts = kp[conf > 0]
    return 0.1 * float((pts.max(0) - pts.min(0)).max()) if len(pts) > 1 else 0.0


class TargetPicker:
    """Picks the one person to draw. Auto: the biggest (closest) person, sticking with the
    current one unless someone is clearly bigger. Clicking a person locks onto them;
    clicking empty space goes back to auto."""

    def __init__(self):
        self.center, self.scale, self.locked, self.lost = None, 0.0, False, 0
        self.click = None          # (x, y) in original-frame px, set by the mouse callback

    def _nearest(self, infos, pt):
        d = [np.hypot(*(c - pt)) / max(s, 1.0) for _, _, c, s in infos]
        i = int(np.argmin(d))
        return i, d[i]

    def pick(self, people):
        """Returns (index into people or None, switched_person: bool)."""
        infos = [(kp, cf, np.nanmean(kp[cf > 0], 0), person_scale(kp, cf)) for kp, cf in people]
        prev = self.center
        if self.click is not None:
            pt, self.click = np.array(self.click, np.float32), None
            hit = None
            for i, (kp, cf, _, s) in enumerate(infos):
                d = np.hypot(*(kp[cf > 0] - pt).T).min()
                if d < max(2.5 * s, 60) and (hit is None or d < hit[1]):
                    hit = (i, d)
            self.locked = hit is not None
            if hit is not None:
                kp, cf, c, s = infos[hit[0]]
                self.center, self.scale, self.lost = c, s, 0
                return hit[0], True
            prev = None

        if not infos:
            self.lost += 1
            if self.lost > 45:     # ~gone for a while: forget them
                self.locked, self.center = False, None
            return None, False

        near = self._nearest(infos, prev) if prev is not None else None
        tracked = near[0] if near is not None and near[1] < 3.0 else None
        if self.locked:
            if tracked is None:
                self.lost += 1
                if self.lost > 45:
                    self.locked, self.center = False, None
                return None, False
            choice = tracked
        else:
            biggest = int(np.argmax([s for *_, s in infos]))
            choice = tracked if tracked is not None and \
                infos[tracked][3] >= 0.7 * infos[biggest][3] else biggest
        self.lost = 0
        switched = choice != tracked
        self.center, self.scale = infos[choice][2], infos[choice][3]
        return choice, switched


# --------------------------------------------------------------------------- smoothing
class OneEuro:
    """One-euro filter per joint; joints that disappear are reset."""

    def __init__(self, n, min_cutoff=1.2, beta=0.01, d_cutoff=1.0):
        self.mc, self.beta, self.dc = min_cutoff, beta, d_cutoff
        self.x = np.full((n, 2), np.nan, np.float32)
        self.dx = np.zeros((n, 2), np.float32)
        self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        return 1.0 / (1.0 + 1.0 / (2 * np.pi * cutoff * dt))

    def __call__(self, x, t):
        out = x.copy()
        if self.t is not None:
            dt = max(t - self.t, 1e-3)
            upd = np.isfinite(x).all(1) & np.isfinite(self.x).all(1)
            if upd.any():
                dx = (x[upd] - self.x[upd]) / dt
                ad = self._alpha(self.dc, dt)
                dxh = ad * dx + (1 - ad) * self.dx[upd]
                a = self._alpha(self.mc + self.beta * np.linalg.norm(dxh, axis=1, keepdims=True), dt)
                out[upd] = a * x[upd] + (1 - a) * self.x[upd]
                self.dx[upd] = dxh
            self.dx[~upd] = 0
        self.x, self.t = out.copy(), t
        return out


# --------------------------------------------------------------------------- drawing
def draw_skeleton(img, uv, conf, labels=False, mirror=False):
    if mirror:
        img = cv2.flip(img, 1)
        uv = uv.copy()
        uv[:, 0] = img.shape[1] - 1 - uv[:, 0]
    th = max(2, round(min(img.shape[:2]) / 240))
    ok = conf > 0
    for j, p in enumerate(PARENTS):
        if p >= 0 and ok[j] and ok[p]:
            cv2.line(img, tuple(np.round(uv[p]).astype(int)), tuple(np.round(uv[j]).astype(int)),
                     COLORS[j], th, cv2.LINE_AA)
    for j in range(24):
        if ok[j]:
            c = tuple(np.round(uv[j]).astype(int))
            cv2.circle(img, c, th + 3, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, c, th + 2, COLORS[j], -1, cv2.LINE_AA)
            if labels:
                cv2.putText(img, str(j), (c[0] + th + 4, c[1] - th - 2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4 + th * 0.05, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def draw_hud(img, text, recording):
    cv2.rectangle(img, (0, 0), (img.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    if recording:
        cv2.circle(img, (img.shape[1] - 20, 14), 7, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(img, "REC", (img.shape[1] - 62, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 1, cv2.LINE_AA)


# --------------------------------------------------------------------------- capture helpers
AUTO_MAX_S = 120                   # an auto-capture never runs longer than this


def draw_banner(img, text, frac=0.25, color=(255, 255, 255)):
    """Big centred outlined text (countdown digits)."""
    h, w = img.shape[:2]
    font, th = cv2.FONT_HERSHEY_DUPLEX, max(2, round(h * frac / 40))
    scale = cv2.getFontScaleFromHeight(font, int(h * frac), th)
    (tw, tht), _ = cv2.getTextSize(text, font, scale, th)
    if tw > 0.9 * w:
        scale *= 0.9 * w / tw
        (tw, tht), _ = cv2.getTextSize(text, font, scale, th)
    org = ((w - tw) // 2, (h + tht) // 2)
    cv2.putText(img, text, org, font, scale, (0, 0, 0), th * 3, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, color, th, cv2.LINE_AA)


def beep(sound="Tink"):
    """Audible cue so a capture works hands-free (you're on the floor, not at the keyboard)."""
    if sys.platform == "darwin":
        try:
            subprocess.Popen(["afplay", f"/System/Library/Sounds/{sound}.aiff"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except OSError:
            pass
    print("\a", end="", flush=True)


def open_writer(path: Path, fps: float, size):
    """H.264 (plays in QuickTime, browsers, phones) when available, else MPEG-4 Part 2."""
    for fourcc in ("avc1", "mp4v"):
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if w.isOpened():
            return w
    sys.exit(f"[stream24] could not open a video writer for {path}")


class AsyncWriter:
    """Video writer on its own thread, so encoding never stalls reading the camera."""

    def __init__(self, path: Path, fps: float, size):
        self.w, self.q = open_writer(path, fps, size), queue.Queue()
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self):
        while (f := self.q.get()) is not None:
            self.w.write(f)
        self.w.release()

    def write(self, f):
        self.q.put(f)

    def release(self):
        self.q.put(None)
        self.th.join()


def new_capture_dir(root: Path) -> Path:
    d = root / datetime.now().strftime("%Y%m%d_%H%M%S")
    d.mkdir(parents=True, exist_ok=True)
    return d


def in_pushup_position(kp, conf, max_tilt=35.0):
    """Plank / push-up seen side-on: shoulders -> hips -> feet roughly along a horizontal line,
    with the hips between the two. False standing or sitting (torso upright) or when the
    hips or legs aren't visible. Lying flat also counts."""
    def mean(*idx):
        v = [kp[i] for i in idx if conf[i] > 0]
        return np.mean(v, 0) if v else None

    sho = mean(NECK, LSHO, RSHO)
    hip = mean(MIDHIP, LHIP, RHIP)
    feet = mean(LANK, RANK, LHEEL, RHEEL, LBTOE, RBTOE, LSTOE, RSTOE)
    if feet is None:
        feet = mean(LKNE, RKNE)    # knee push-ups / feet out of frame
    if sho is None or hip is None or feet is None:
        return False

    def tilt(a, b):                # degrees away from horizontal, 0..90
        return float(np.degrees(np.arctan2(abs(b[1] - a[1]), abs(b[0] - a[0]))))

    stretched_out = (sho[0] - hip[0]) * (feet[0] - hip[0]) < 0
    return stretched_out and tilt(sho, hip) < max_tilt and tilt(hip, feet) < max_tilt


# --------------------------------------------------------------------------- video sources
class LatestFrameReader:
    """Live sources: read in a background thread and always hand back the newest frame,
    so slow inference never builds up lag. During a capture it also writes EVERY camera
    frame to disk (with its timestamp), so the capture runs at the camera's full frame rate
    no matter how slow inference is. With a pre-roll set, it keeps the last few seconds of
    frames so an auto-triggered capture can start slightly in the past."""

    def __init__(self, cap):
        self.cap, self.frame, self.seq, self.alive = cap, None, 0, True
        self.fps = 0.0             # measured camera frame rate
        self.writer, self.stamps = None, []
        self.preroll_s, self.preroll = 0.0, deque()
        self.cv = threading.Condition()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        fails, last = 0, None
        while self.alive:
            ok, f = self.cap.read()
            now = time.monotonic()
            if not ok:
                fails += 1
                if fails > 100:
                    break
                time.sleep(0.02)
                continue
            fails = 0
            if last is not None and now > last:
                inst = 1.0 / (now - last)
                self.fps = inst if self.fps == 0 else 0.95 * self.fps + 0.05 * inst
            last = now
            with self.cv:
                self.frame, self.seq = f, self.seq + 1
                if self.writer is not None:
                    self.writer.write(f)
                    self.stamps.append(now)
                elif self.preroll_s > 0:
                    self.preroll.append((now, f))
                    while now - self.preroll[0][0] > self.preroll_s:
                        self.preroll.popleft()
                self.cv.notify_all()
        with self.cv:
            self.alive = False
            self.cv.notify_all()

    def read(self, last_seq):
        with self.cv:
            self.cv.wait_for(lambda: self.seq > last_seq or not self.alive, timeout=10)
            if self.seq <= last_seq:
                return None, last_seq
            return self.frame.copy(), self.seq

    def set_preroll(self, seconds: float):
        with self.cv:
            self.preroll_s = max(0.0, seconds)
            if self.preroll_s == 0:
                self.preroll.clear()

    def start_capture(self, path: Path, preroll=False) -> float:
        with self.cv:
            h, w = self.frame.shape[:2]
            fps = min((15.0, 24.0, 25.0, 30.0, 50.0, 60.0), key=lambda r: abs(r - self.fps))
            if self.fps >= 5 and abs(fps - self.fps) > 0.08 * fps:
                fps = float(round(self.fps))   # an unusual camera rate: use it as measured
            self.writer, self.stamps = AsyncWriter(path, fps, (w, h)), []
            if preroll:
                for t, f in self.preroll:
                    self.writer.write(f)
                    self.stamps.append(t)
            self.preroll.clear()
        return fps

    def stop_capture(self) -> np.ndarray:
        with self.cv:
            writer, stamps, self.writer = self.writer, self.stamps, None
        if writer is not None:
            writer.release()
        return np.asarray(stamps, np.float64)

    def stop(self):
        self.alive = False


class Recorder:
    """Writes the annotated video + joints24.npz for one capture."""

    def __init__(self, out_dir: Path, size, fps):
        self.dir, self.size = out_dir, size
        self.writer = AsyncWriter(out_dir / "capture_24pt.mp4", max(1.0, fps), size)
        self.uv, self.conf, self.t, self.b_uv, self.b_conf = [], [], [], [], []
        self.t0 = None

    def add(self, img, uv, conf, kp25, conf25, t):
        self.t0 = t if self.t0 is None else self.t0
        self.writer.write(img)
        self.uv.append(uv)
        self.conf.append(conf)
        self.b_uv.append(kp25)
        self.b_conf.append(conf25)
        self.t.append(t - self.t0)

    def close(self):
        self.writer.release()
        np.savez_compressed(
            self.dir / "joints24.npz",
            uv=np.asarray(self.uv, np.float32).reshape(-1, 24, 2),
            conf=np.asarray(self.conf, np.float32).reshape(-1, 24),
            t=np.asarray(self.t, np.float64),
            body25_uv=np.asarray(self.b_uv, np.float32).reshape(-1, 25, 2),
            body25_conf=np.asarray(self.b_conf, np.float32).reshape(-1, 25),
            joint_names=np.array(SMPL24), parents=np.array(PARENTS),
            image_size=np.array(self.size), source=np.array("openpose_body25"))
        return self.dir


class Follower:
    """One frame's detected people -> the target's BODY_25 + (smoothed) SMPL-24."""

    def __init__(self, picker=None):
        self.picker = picker if picker is not None else TargetPicker()
        self.smoother = OneEuro(24)

    def __call__(self, people, t, smooth=True):
        idx, switched = self.picker.pick(people)
        if idx is None:
            kp25, conf25 = np.full((25, 2), np.nan, np.float32), np.zeros(25, np.float32)
        else:
            kp25, conf25 = people[idx]
        if switched:               # don't smear the smoothing from one person to another
            self.smoother = OneEuro(24)
        uv, conf = body25_to_smpl24(kp25, conf25)
        if smooth:
            uv = self.smoother(uv, t)
        return uv, conf, kp25, conf25


# --------------------------------------------------------------------------- processing
_worker_net = None


def _worker_init(threads, target):
    global _worker_net
    cv2.setNumThreads(threads)
    _worker_net = load_net(target)


def _worker_infer(frame, net_h, thr):
    return infer_people(_worker_net, frame, net_h, thr)


class PosePool:
    """Several OpenPose copies in separate processes. One forward pass doesn't use many cores
    well: on a 14-core M4 Pro, 4 workers give ~1.6x the frames/s of one process."""

    def __init__(self, workers, target):
        threads = max(1, (os.cpu_count() or 4) // workers)
        self.workers = workers
        self.pool = mp.get_context("spawn").Pool(workers, _worker_init, (threads, target))

    def submit(self, frame, net_h, thr):
        return self.pool.apply_async(_worker_infer, (frame, net_h, thr))

    def close(self):
        self.pool.terminate()


def make_pool(args):
    workers = args.workers or min(4, max(1, (os.cpu_count() or 4) // 3))
    return PosePool(workers, args.target) if workers > 1 else None


def _poses(cap, net, pool, net_h, thr):
    """(frame, people) for every frame of cap, in order. With a pool, several frames are
    in flight at once."""
    if pool is None:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield frame, infer_people(net, frame, net_h, thr)
    pending, more = deque(), True
    while more or pending:
        while more and len(pending) < 2 * pool.workers:
            ok, frame = cap.read()
            if not ok:
                more = False
                break
            pending.append((frame, pool.submit(frame, net_h, thr)))
        if pending:
            frame, res = pending.popleft()
            yield frame, res.get()


def process_video(net, src, out_dir: Path, args, picker=None, stamps=None, show=None,
                  pool=None, until=None):
    """Full-quality pose on EVERY frame of a video file -> capture_24pt.mp4 + joints24.npz.
    stamps: per-frame capture times (live captures); until: skip frames captured after this
    time. show(img, i, total) -> False stops early."""
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        sys.exit(f"[stream24] could not open video {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if stamps is not None and len(stamps) > 1:
        fps = (len(stamps) - 1) / (stamps[-1] - stamps[0])   # the camera's real rate
        if until is not None:
            total = int(np.searchsorted(stamps, until, side="right"))
    follow = Follower(picker)
    rec, n, tracked, t_start = None, 0, 0, time.monotonic()
    try:
        for frame, people in _poses(cap, net, pool, args.net_height, args.thr):
            if stamps is not None and n < len(stamps):
                if until is not None and stamps[n] > until:
                    break
                t = stamps[n] - stamps[0]
            else:
                t = n / fps
            uv, conf, kp25, conf25 = follow(people, t, args.smooth)
            tracked += bool((conf > 0).any())
            if rec is None:
                rec = Recorder(out_dir, (frame.shape[1], frame.shape[0]), fps)
            img = draw_skeleton(frame.copy(), uv, conf, args.labels)
            rec.add(img, uv, conf, kp25, conf25, t)
            n += 1
            if show is not None:
                if show(img, n, total) is False:
                    print("[stream24] processing stopped early")
                    break
            elif n % 30 == 0:
                print(f"[stream24] processed {n}/{total or '?'} frames", flush=True)
            if args.max_frames and n >= args.max_frames:
                break
    finally:
        cap.release()
    if rec is None:
        print(f"[stream24] no frames in {src}")
        return None
    rec.close()
    print(f"[stream24] {n} frames ({n / fps:.1f} s at {fps:.1f} fps) processed in "
          f"{time.monotonic() - t_start:.0f} s; person tracked in {tracked}/{n} frames\n"
          f"[stream24] saved -> {out_dir}")
    return out_dir


def progress_window(win, label="Processing capture at full quality"):
    t0 = time.monotonic()

    def show(img, i, total):
        view = img.copy()
        left = (time.monotonic() - t0) / i * (total - i) if total else 0
        draw_hud(view, f"{label}: frame {i}/{total or '?'}"
                 + (f" | ~{left:.0f} s left" if total else "") + " | q = stop early", False)
        cv2.imshow(win, view)
        return (cv2.waitKey(1) & 0xFF) not in (ord("q"), 27)
    return show


def lift_3d(out_dir, video, args, win):
    """Single-camera 3D for a processed capture (lift3d.py, MediaPipe) -> joints3d.npz."""
    if out_dir is None or not args.lift3d:
        return
    try:
        import lift3d
        lift3d.lift_capture(out_dir, video, show=None if args.headless
                            else progress_window(win, "3D pose (MediaPipe)"))
    except ImportError as e:
        print(f"[stream24] skipping 3D: {e} (pip install -r requirements.txt)")


# --------------------------------------------------------------------------- main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Live OpenPose -> 24 SMPL joints overlay + capture")
    ap.add_argument("--source", default="0", help="camera index, video file, or stream URL")
    ap.add_argument("--net-height", type=int, default=368,
                    help="network height for processing captures/files (368 = OpenPose default)")
    ap.add_argument("--live-height", type=int, default=256,
                    help="network height for the live preview (smaller = higher preview fps)")
    ap.add_argument("--thr", type=float, default=0.1, help="min heatmap peak to accept a keypoint")
    ap.add_argument("--target", choices=["cpu", "opencl", "cuda"], default="cpu")
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel OpenPose processes for processing (0 = auto, 1 = off)")
    ap.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=None,
                    help="mirror the display (default: on for webcams, off otherwise)")
    ap.add_argument("--smooth", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--labels", action="store_true", help="show joint numbers 0-23")
    ap.add_argument("--record", action="store_true", help="start the capture countdown at launch")
    ap.add_argument("--countdown", type=float, default=5, help="seconds before a capture starts")
    ap.add_argument("--duration", type=float, default=15,
                    help="capture length in seconds (0 = until you press r)")
    ap.add_argument("--auto", action="store_true",
                    help="auto-capture: record when you get into push-up position (side-on to "
                         "the camera), stop when you leave it. Toggle with a")
    ap.add_argument("--preroll", type=float, default=2.0,
                    help="auto mode: seconds of video kept from before the trigger")
    ap.add_argument("--auto-stop", type=float, default=2.0,
                    help="auto mode: stop after this many seconds out of position")
    ap.add_argument("--3d", dest="lift3d", action=argparse.BooleanOptionalAction, default=True,
                    help="after processing, add single-camera 3D joints (joints3d.npz, lift3d.py)")
    ap.add_argument("--headless", action="store_true",
                    help="no window: process a file, or do one live capture (--record/--auto) and exit")
    ap.add_argument("--width", type=int, default=1280, help="requested camera width")
    ap.add_argument("--height", type=int, default=720, help="requested camera height")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = no limit)")
    ap.add_argument("--out", default=str(HERE / "outputs"))
    return ap.parse_args(argv)


def main(argv=None, net=None):
    args = parse_args(argv)
    src = int(args.source) if args.source.isdigit() else args.source
    is_camera = isinstance(src, int)
    is_file = not is_camera and Path(str(src)).exists()
    mirror = is_camera if args.mirror is None else args.mirror
    win = "stream24 - OpenPose -> SMPL-24"

    if is_file:                    # a recorded clip: process every frame at full quality
        net = net if net is not None else load_net(args.target)
        pool = make_pool(args)
        try:
            out = process_video(net, src, new_capture_dir(Path(args.out)), args, pool=pool,
                                show=None if args.headless else progress_window(win))
            lift_3d(out, src, args, win)
        finally:
            if pool is not None:
                pool.close()
            if not args.headless:
                cv2.destroyAllWindows()
        return out

    cap = cv2.VideoCapture(src)
    if is_camera:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        sys.exit(f"[stream24] could not open source {args.source!r}"
                 + (" (on macOS, allow camera access for your terminal)" if is_camera else ""))

    net = net if net is not None else load_net(args.target)
    pool = make_pool(args)         # workers start loading now, so the first capture won't wait
    reader = LatestFrameReader(cap)
    follow = Follower()
    labels, smooth, auto = args.labels, args.smooth, args.auto
    seq, n, fps, t_start = 0, 0, 0.0, time.monotonic()
    state, mark, tick, cap_dir, seed, last_out = "live", 0.0, None, None, None, None
    rec_auto, armed, pos_since, last_pos = False, False, None, -1e9
    if auto:
        reader.set_preroll(args.preroll)
    if args.record:
        state, mark = "countdown", time.monotonic() + args.countdown
    frame_w = [0]                  # for un-mirroring clicks

    if not args.headless:
        cv2.namedWindow(win)

        def on_mouse(event, x, y, *_):
            if event == cv2.EVENT_LBUTTONDOWN:
                follow.picker.click = (frame_w[0] - 1 - x if mirror else x, y)
        cv2.setMouseCallback(win, on_mouse)

    try:
        while True:
            frame, seq = reader.read(seq)
            if frame is None:
                print("[stream24] stream ended or stalled")
                break
            now = time.monotonic()
            frame_w[0] = frame.shape[1]
            t0 = time.perf_counter()
            people = infer_people(net, frame, args.live_height, args.thr)
            uv, conf, kp25, conf25 = follow(people, now - t_start, smooth)
            dt = time.perf_counter() - t0
            fps = 1 / dt if fps == 0 else 0.9 * fps + 0.1 / dt

            in_pos = in_pushup_position(kp25, conf25)
            if in_pos:
                last_pos = now
                pos_since = now if pos_since is None else pos_since
            else:
                pos_since = None

            # capture: live -> (countdown | auto trigger) -> recording -> processing -> live
            stop = quit_after = begin = False
            until = None
            if state == "countdown":
                if now >= mark:
                    begin, rec_auto = True, False
                elif math.ceil(mark - now) != tick:
                    tick = math.ceil(mark - now)
                    beep("Tink")
            elif state == "recording":
                if rec_auto:
                    if now - last_pos >= args.auto_stop or now - mark >= AUTO_MAX_S:
                        stop, until = True, last_pos + 0.7   # trim the getting-up part
                elif args.duration > 0 and now - mark >= args.duration:
                    stop = True
            elif auto:                 # live: watch for the push-up position
                if not in_pos and now - last_pos > 1.0:
                    armed = True       # only re-arm once you've been out of position
                if armed and pos_since is not None and now - pos_since >= 0.5:
                    begin, rec_auto, armed = True, True, False

            if begin:
                cap_dir = new_capture_dir(Path(args.out))
                cam_fps = reader.start_capture(cap_dir / "raw.mp4", preroll=rec_auto)
                seed = copy.deepcopy(follow.picker)   # process starting on the same person
                state, mark = "recording", now
                beep("Glass")
                print(f"[stream24] {'auto-' if rec_auto else ''}recording at the camera's "
                      f"~{cam_fps:.0f} fps ...")

            seen = int((conf > 0).sum())
            if state == "recording":
                status = (f"{'AUTO ' if rec_auto else ''}REC {now - mark:4.1f}"
                          + (f"/{args.duration:g}" if args.duration > 0 and not rec_auto else "")
                          + f" s @ {reader.fps:.0f} fps")
            elif state == "countdown":
                status = "get ready"
            elif auto:
                status = ("AUTO: in position..." if in_pos else
                          "AUTO: get into push-up position, side-on" if armed else
                          "AUTO: get up to re-arm")
            else:
                status = "r = capture | a = auto"
            hud = (f"live {fps:4.1f} fps | {len(people)} people | "
                   f"target: {'locked' if follow.picker.locked else 'closest'} | "
                   f"{seen}/24 joints | {status}")

            if args.headless:
                if n % 30 == 0:
                    print(f"[stream24] {hud}", flush=True)
            else:
                view = draw_skeleton(frame.copy(), uv, conf, labels, mirror)
                draw_hud(view, hud, state == "recording")
                if state == "countdown":
                    draw_banner(view, str(max(1, math.ceil(mark - now))))
                cv2.imshow(win, view)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    if state != "recording":
                        break
                    stop = quit_after = True       # keep what was captured
                elif key == ord("r"):
                    if state == "live":
                        state, mark, tick = "countdown", now + args.countdown, None
                    elif state == "countdown":
                        state = "live"
                        print("[stream24] capture cancelled")
                    else:
                        stop = True
                elif key == ord("a"):
                    auto, armed = not auto, False
                    reader.set_preroll(args.preroll if auto else 0)
                    print(f"[stream24] auto-capture {'on' if auto else 'off'}")
                elif key == ord("s"):
                    smooth = not smooth
                elif key == ord("l"):
                    labels = not labels
                elif key == ord("m"):
                    mirror = not mirror

            if stop:
                stamps = reader.stop_capture()
                state, was_auto, rec_auto = "live", rec_auto, False
                beep("Hero")
                if len(stamps) < 2:
                    print("[stream24] nothing captured")
                else:
                    print(f"[stream24] captured {len(stamps)} frames in "
                          f"{stamps[-1] - stamps[0]:.1f} s -> processing every frame ...")
                    opts = argparse.Namespace(**{**vars(args), "smooth": smooth, "labels": labels})
                    last_out = process_video(net, cap_dir / "raw.mp4", cap_dir, opts, picker=seed,
                                             stamps=stamps, pool=pool,
                                             until=until if was_auto else None,
                                             show=None if args.headless else progress_window(win))
                    lift_3d(last_out, cap_dir / "raw.mp4", opts, win)
                if args.headless or quit_after:
                    break

            n += 1
            if args.max_frames and n >= args.max_frames:
                break
    finally:
        if state == "recording":   # interrupted mid-capture: raw.mp4 is still finalized
            reader.stop_capture()
        reader.stop()
        cap.release()
        if pool is not None:
            pool.close()
        if not args.headless:
            cv2.destroyAllWindows()
    return last_out


if __name__ == "__main__":
    main()
