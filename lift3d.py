#!/usr/bin/env python3
"""
lift3d.py - single-camera 3D for a capture. Runs MediaPipe's 3D pose model on the person
stream24.py tracked, maps it to the same 24 SMPL joints, and places it in a gravity-aligned
world (metres, z up, floor at z = 0) ready for MuJoCo.

    python lift3d.py outputs/<capture>          # (re)lift a capture; stream24.py does this itself

Reads <capture>/joints24.npz (the OpenPose 2D track, used to keep MediaPipe on the right
person in a crowded room) and the capture's video. Writes <capture>/joints3d.npz:
    xyz        (T, 24, 3)  SMPL-24 joint positions in metres: z up, floor z = 0,
                           y = the original camera's viewing direction, x = its right
    conf       (T, 24)     MediaPipe visibility (0..1); low = occluded and guessed
    t          (T,)        seconds, the same frames as joints24.npz
    mp_world   (T, 33, 3)  raw MediaPipe world landmarks (hip-centred, camera axes:
                           x right, y down, z away from the camera), mp_vis (T, 33)
    up         (3,)        floor normal (in MediaPipe camera axes) that was used
    joint_names, parents
Frames where MediaPipe found nobody are NaN.

One camera can't measure where the body is in the room, so it's placed by contact: in
each frame the lowest visible point touches the floor, and the body moves horizontally so
that whatever is planted (feet, hands, knees) doesn't slide (walking advances it step by
step). Height during a jump isn't recoverable this way: the lowest point is always put on
the floor. MediaPipe's pose model is Apache-2.0 licensed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
MODEL = HERE / "models" / "mediapipe" / "pose_landmarker_heavy.task"
MODEL_URLS = ["https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
              "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"]

# MediaPipe's 33 landmarks (L/R = the person's own left/right)
(NOSE, L_EAR, R_EAR, L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI, L_PINKY, R_PINKY, L_INDEX,
 R_INDEX, L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK, L_HEEL, R_HEEL, L_TOE, R_TOE) = (
    0, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32)
FEET = [L_HEEL, R_HEEL, L_TOE, R_TOE]
PLANTED = [L_PINKY, R_PINKY, L_INDEX, R_INDEX, L_TOE, R_TOE]   # on the floor in a push-up

# OpenPose BODY_25 -> MediaPipe pairs, for checking MediaPipe found the tracked person
_B25_MP = [(0, NOSE), (2, R_SHO), (5, L_SHO), (3, R_ELB), (6, L_ELB), (9, R_HIP), (12, L_HIP),
           (10, R_KNE), (13, L_KNE), (11, R_ANK), (14, L_ANK)]


def mp33_to_smpl24(P, V):
    """P (T,33,3), V (T,33) -> xyz (T,24,3), conf (T,24). Same recipes as stream24's 2D."""
    def pt(*idx):
        return P[:, idx].mean(1), V[:, idx].min(1)

    def lerp(a, b, t):
        return (1 - t) * a[0] + t * b[0], np.minimum(a[1], b[1])

    pelvis, neck_base, head = pt(L_HIP, R_HIP), pt(L_SHO, R_SHO), pt(L_EAR, R_EAR)
    joints = [
        pelvis, pt(L_HIP), pt(R_HIP),                        # 0 pelvis 1 l_hip 2 r_hip
        lerp(pelvis, neck_base, 0.20),                       # 3 spine1
        pt(L_KNE), pt(R_KNE),                                # 4 l_knee 5 r_knee
        lerp(pelvis, neck_base, 0.45),                       # 6 spine2
        pt(L_ANK), pt(R_ANK),                                # 7 l_ankle 8 r_ankle
        lerp(pelvis, neck_base, 0.70),                       # 9 spine3
        lerp(pt(L_HEEL), pt(L_TOE), 0.65),                   # 10 l_foot (ball of foot)
        lerp(pt(R_HEEL), pt(R_TOE), 0.65),                   # 11 r_foot
        lerp(neck_base, head, 0.25),                         # 12 neck
        lerp(neck_base, pt(L_SHO), 0.45),                    # 13 l_collar
        lerp(neck_base, pt(R_SHO), 0.45),                    # 14 r_collar
        head,                                                # 15 head
        pt(L_SHO), pt(R_SHO), pt(L_ELB), pt(R_ELB),          # 16-19 shoulders, elbows
        pt(L_WRI), pt(R_WRI),                                # 20 l_wrist 21 r_wrist
        pt(L_PINKY, L_INDEX), pt(R_PINKY, R_INDEX),          # 22 l_hand 23 r_hand (knuckles)
    ]
    return np.stack([j[0] for j in joints], 1), np.stack([j[1] for j in joints], 1)


# --------------------------------------------------------------------------- MediaPipe
def make_landmarker(model: Path = MODEL):
    if not model.exists():
        from stream24 import _download
        _download(MODEL_URLS, model, 10_000_000)
    from mediapipe.tasks.python import BaseOptions, vision
    return vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO, num_poses=2))


def crop_box(b_uv, b_conf, w, h, margin=0.3):
    """One fixed box around everywhere the tracked person goes (robust to stray frames)."""
    boxes = []
    for uv, c in zip(b_uv, b_conf):
        p = uv[c > 0]
        if len(p) >= 3:
            boxes.append([*p.min(0), *p.max(0)])
    if not boxes:
        return 0, 0, w, h
    b = np.asarray(boxes)
    x0, y0 = np.percentile(b[:, 0], 2), np.percentile(b[:, 1], 2)
    x1, y1 = np.percentile(b[:, 2], 98), np.percentile(b[:, 3], 98)
    mx, my = margin * (x1 - x0) + 40, margin * (y1 - y0) + 40
    return (int(max(0, x0 - mx)), int(max(0, y0 - my)),
            int(min(w, x1 + mx)), int(min(h, y1 + my)))


def _pick(result, box, target_uv, target_conf, prev_c):
    """Index of the MediaPipe pose that is the tracked person, or None."""
    if not result.pose_landmarks:
        return None, prev_c
    x0, y0, x1, y1 = box
    ref, scale = prev_c, None
    pairs = [(b, m) for b, m in _B25_MP if target_conf[b] > 0]
    if len(pairs) >= 2:
        pts = target_uv[[b for b, _ in pairs]]
        ref, scale = pts.mean(0), max(np.ptp(pts, 0).max(), 80.0)
    best, best_d, best_c = None, np.inf, prev_c
    for k, lms in enumerate(result.pose_landmarks):
        mp_pts = np.array([[lms[m].x * (x1 - x0) + x0, lms[m].y * (y1 - y0) + y0]
                           for _, m in (pairs if len(pairs) >= 2 else _B25_MP)])
        c = mp_pts.mean(0)
        d = 0.0 if ref is None else float(np.hypot(*(c - ref)))
        if d < best_d:
            best, best_d, best_c = k, d, c
    if scale is not None and best_d > 0.6 * scale:   # MediaPipe is on someone else
        return None, prev_c
    return best, best_c


# --------------------------------------------------------------------------- world placement
def estimate_up(P, V, max_tilt=35.0):
    """Floor normal (camera axes) from frames with hands and toes planted, e.g. a push-up:
    the normal of the plane through them. Falls back to 'camera is level'."""
    cam_up = np.array([0.0, -1.0, 0.0])
    normals = []
    for p, v in zip(P, V):
        q = p[PLANTED]
        if not np.isfinite(q).all() or v[PLANTED].min() < 0.3:
            continue
        q = q - q.mean(0)
        _, s, vt = np.linalg.svd(q)
        if s[0] < 0.5 or s[1] < 0.08:                # hands+feet must span a real plane
            continue
        n = vt[2] if vt[2] @ cam_up > 0 else -vt[2]
        if np.degrees(np.arccos(np.clip(n @ cam_up, -1, 1))) < max_tilt:
            normals.append(n)
    if len(normals) < 10:
        return cam_up, 0
    n = np.median(normals, 0)
    return n / np.linalg.norm(n), len(normals)


def _smooth(x, k=7):
    """Moving median over valid values, then fill gaps by interpolation."""
    out, ok = x.copy(), np.isfinite(x)
    if not ok.any():
        return np.zeros_like(x)
    for i in np.nonzero(ok)[0]:
        w = x[max(0, i - k // 2): i + k // 2 + 1]
        out[i] = np.nanmedian(w)
    if ok.any() and not ok.all():
        idx = np.arange(len(x))
        out[~ok] = np.interp(idx[~ok], idx[ok], out[ok])
    return out


CONTACTS = [L_HEEL, R_HEEL, L_TOE, R_TOE, L_INDEX, R_INDEX, L_KNE, R_KNE]  # can be planted
CONTACT_Z = 0.05                                  # m above the floor that still counts as planted


def contact_odometry(Wp, V):
    """Horizontal path of the hips through the room, from hip-centred, floor-grounded
    landmarks: between consecutive frames the body moves so that whatever is planted on
    the floor (feet, hands, knees) doesn't slide. Standing still -> stays put; walking ->
    advances one planted foot at a time. With nothing planted (mid-jump, or the feet just
    not visible) it coasts, slowing to a stop within ~0.5 s so it can't drift away."""
    T = len(Wp)
    root, vel = np.zeros((T, 2)), np.zeros(2)
    for i in range(1, T):
        a, b = Wp[i - 1], Wp[i]
        root[i] = root[i - 1]
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            continue
        low = [c for c in CONTACTS if a[c, 2] < CONTACT_Z and b[c, 2] < CONTACT_Z
               and V[i - 1, c] > 0.3 and V[i, c] > 0.3]
        if low:
            # A planted point stays put in the room, so relative to the hips it moves by
            # minus the body's motion. A swinging foot skimming the floor moves the other
            # way: keep only the points consistent with the body's current speed.
            d = b[low, :2] - a[low, :2]
            miss = np.linalg.norm(d + vel, axis=1)
            keep = miss < max(0.02, 1.5 * miss.min())
            vel = -np.median(d[keep], 0)
        else:
            vel = 0.85 * vel
        root[i] = root[i - 1] + vel
    return root


def to_world(P, V, up):
    """Hip-centred camera-axes landmarks -> z-up world: lowest point on the floor, horizontal
    position from contact_odometry, starting at the origin."""
    z = up
    x = np.array([1.0, 0.0, 0.0]) - up[0] * up       # camera's right, made horizontal
    x /= np.linalg.norm(x)
    y = np.cross(z, x)                               # the camera's viewing direction
    Wp = P @ np.stack([x, y, z]).T                   # (T, 33, 3)
    vis = V >= 0.5
    # Ground on what's actually visible. With the feet out of view, use MediaPipe's
    # full-body guess instead, or the unseen legs would hang through the floor.
    feet_seen = vis[:, [L_ANK, R_ANK] + FEET].any(1)
    use = np.where(feet_seen[:, None], vis, V > 0)
    has = use.any(1)
    low = np.full(len(P), np.nan)
    low[has] = np.nanmin(np.where(use, Wp[..., 2], np.nan)[has], 1)
    Wp[..., 2] -= _smooth(low)[:, None]
    Wp[..., :2] += contact_odometry(Wp, V)[:, None, :]
    return Wp


# --------------------------------------------------------------------------- main entry
def lift_capture(cap_dir, video=None, show=None, model: Path = MODEL):
    """3D for a capture folder (needs joints24.npz). show(img, i, total) -> False stops."""
    import mediapipe as mp
    from stream24 import PARENTS, SMPL24

    cap_dir = Path(cap_dir)
    z2 = np.load(cap_dir / "joints24.npz")
    b_uv, b_conf, t = z2["body25_uv"], z2["body25_conf"], z2["t"]
    T = len(t)
    video = Path(video) if video else cap_dir / "raw.mp4"
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f"[lift3d] could not open {video}")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    box = crop_box(b_uv, b_conf, w, h)
    x0, y0, x1, y1 = box
    P = np.full((T, 33, 3), np.nan, np.float32)
    V = np.zeros((T, 33), np.float32)
    prev_c, last_ms, found = None, -1, 0
    with make_landmarker(model) as lm:
        for i in range(T):
            ok, frame = cap.read()
            if not ok:
                break
            rgb = np.ascontiguousarray(frame[y0:y1, x0:x1, ::-1])
            ms = max(last_ms + 1, int(round(t[i] * 1000)))
            last_ms = ms
            res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ms)
            k, prev_c = _pick(res, box, b_uv[i], b_conf[i], prev_c)
            if k is not None:
                P[i] = [[p.x, p.y, p.z] for p in res.pose_world_landmarks[k]]
                V[i] = [p.visibility for p in res.pose_world_landmarks[k]]
                found += 1
            if show is not None:
                img = frame.copy()
                cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), (90, 90, 90), 1)
                if k is not None:
                    for p in res.pose_landmarks[k]:
                        cv2.circle(img, (int(p.x * (x1 - x0) + x0), int(p.y * (y1 - y0) + y0)),
                                   4, (0, 220, 120), -1, cv2.LINE_AA)
                if show(img, i + 1, T) is False:
                    break
            elif (i + 1) % 60 == 0:
                print(f"[lift3d] {i + 1}/{T} frames", flush=True)
    cap.release()

    up, n_up = estimate_up(P, V)
    tilt = float(np.degrees(np.arccos(np.clip(up @ [0, -1, 0], -1, 1))))
    Wp = to_world(P, V, up)
    xyz, conf = mp33_to_smpl24(Wp, V)
    np.savez_compressed(cap_dir / "joints3d.npz", xyz=xyz.astype(np.float32),
                        conf=conf.astype(np.float32), t=t, mp_world=P, mp_vis=V,
                        up=up, joint_names=np.array(SMPL24), parents=np.array(PARENTS))
    print(f"[lift3d] 3D in {found}/{T} frames | floor: "
          + (f"from planted hands/feet in {n_up} frames (camera tilted {tilt:.0f} deg)" if n_up
             else "camera assumed level (no hands+feet-on-floor frames)")
          + f"\n[lift3d] saved -> {cap_dir / 'joints3d.npz'}")
    return cap_dir / "joints3d.npz"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Single-camera 3D (MediaPipe) for a stream24 capture")
    ap.add_argument("capture", help="capture folder containing joints24.npz")
    ap.add_argument("--video", help="video the capture came from (default: <capture>/raw.mp4)")
    a = ap.parse_args()
    lift_capture(a.capture, a.video)
