"""Objective 'does the robot match the human?' measurement (no Gemini).

Human: 2D body landmarks per frame from the video (MediaPipe pose landmarker, cached next to the run).
Robot: the same joints projected through the simulator camera.
Metrics per frame:
  - limb direction error (deg) for upper arms, forearms, thighs, shins, torso: scale-free, robust to the robot being smaller
  - framing: where the subject is on screen (center offset, fraction of image) and how big (height ratio)
  - hard checks: robot off-screen, body flipped / lying, held object above the head
"""
from pathlib import Path

import numpy as np

from . import config as C

MODEL = C.ROOT / "assets" / "mediapipe" / "pose_landmarker_full.task"
# MediaPipe landmark indices (person's anatomical left/right)
MP = dict(l_sho=11, r_sho=12, l_elb=13, r_elb=14, l_wri=15, r_wri=16, l_hip=23, r_hip=24, l_kne=25, r_kne=26, l_ank=27, r_ank=28)
ROBOT_BODY = dict(l_sho="left_shoulder_roll_link", r_sho="right_shoulder_roll_link", l_elb="left_elbow_link",
                  r_elb="right_elbow_link", l_wri="left_wrist_yaw_link", r_wri="right_wrist_yaw_link",
                  l_hip="left_hip_roll_link", r_hip="right_hip_roll_link", l_kne="left_knee_link", r_kne="right_knee_link",
                  l_ank="left_ankle_roll_link", r_ank="right_ankle_roll_link")
KEYS = list(MP)
BONES = {"left upper arm": ("l_sho", "l_elb"), "left forearm": ("l_elb", "l_wri"),
         "right upper arm": ("r_sho", "r_elb"), "right forearm": ("r_elb", "r_wri"),
         "left thigh": ("l_hip", "l_kne"), "left shin": ("l_kne", "l_ank"),
         "right thigh": ("r_hip", "r_kne"), "right shin": ("r_kne", "r_ank"), "torso": ("hip_mid", "sho_mid")}


def human_pose(video, cache_dir, full=False):
    """2D: [T, 12, 3] normalized (x, y, visibility) for KEYS. With full=True also returns the 3D world landmarks
    [T, 33, 4] (meters, hip-centered, camera-aligned axes: x right, y down, z away from the camera; + visibility).
    Cached per run."""
    c2, c3 = Path(cache_dir) / "human_pose.npy", Path(cache_dir) / "human_pose3d.npy"
    if c2.exists() and (c3.exists() or not full):
        return (np.load(c2), np.load(c3)) if full else np.load(c2)
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision
    opts = vision.PoseLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path=str(MODEL)),
                                        running_mode=vision.RunningMode.VIDEO, num_poses=1)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(5) or 30.0
    out2, out3 = [], []
    with vision.PoseLandmarker.create_from_options(opts) as lm:
        i = 0
        while True:
            ok, im = cap.read()
            if not ok:
                break
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=im[:, :, ::-1].copy())
            r = lm.detect_for_video(img, int(i * 1000 / fps))
            row, w3 = np.zeros((len(KEYS), 3)), np.zeros((33, 4))
            if r.pose_landmarks:
                p = r.pose_landmarks[0]
                for k, key in enumerate(KEYS):
                    q = p[MP[key]]
                    row[k] = (q.x, q.y, q.visibility if q.visibility is not None else 1.0)
                if r.pose_world_landmarks:
                    for j, q in enumerate(r.pose_world_landmarks[0]):
                        w3[j] = (q.x, q.y, q.z, p[j].visibility if p[j].visibility is not None else 1.0)
            out2.append(row)
            out3.append(w3)
            i += 1
    np.save(c2, np.array(out2))
    np.save(c3, np.array(out3))
    return (np.array(out2), np.array(out3)) if full else np.array(out2)


def project(points, cam_pos, cam_fwd, cam_up, fovy_deg, W, H):
    """World points -> normalized image coords (x right, y down, 0..1); z<=0 means behind the camera."""
    f = cam_fwd / np.linalg.norm(cam_fwd)
    r = np.cross(f, cam_up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    rel = points - cam_pos
    z = rel @ f
    t = np.tan(np.radians(fovy_deg) / 2)
    x = (rel @ r) / np.maximum(z, 1e-6) / (t * W / H)
    y = (rel @ u) / np.maximum(z, 1e-6) / t
    return np.stack([0.5 + 0.5 * x, 0.5 - 0.5 * y, z], -1)


def _with_mids(P):
    """Add hip/shoulder midpoints to a [12, k] keypoint array -> dict."""
    d = {k: P[i] for i, k in enumerate(KEYS)}
    d["hip_mid"] = (d["l_hip"] + d["r_hip"]) / 2
    d["sho_mid"] = (d["l_sho"] + d["r_sho"]) / 2
    return d


def compare(human, robot, fps, aspect, extra=None):
    """human [T,12,3] (x,y,vis), robot [T,12,3] (x,y,depth). Returns a summary dict for telemetry / benchmarks."""
    T = min(len(human), len(robot))
    errs = {b: np.full(T, np.nan) for b in BONES}
    center, size_ratio, offscreen = [], [], 0
    for f in range(T):
        h, r = _with_mids(human[f]), _with_mids(robot[f])
        vis = human[f][:, 2]
        if vis.mean() < 0.3:
            continue
        for b, (a, c) in BONES.items():
            va = np.array([h[c][0] - h[a][0], (h[c][1] - h[a][1]) / aspect])
            vb = np.array([r[c][0] - r[a][0], (r[c][1] - r[a][1]) / aspect])
            hv = min(h[a][2], h[c][2]) if b != "torso" else vis[[0, 1, 6, 7]].min()
            if hv < 0.5 or np.linalg.norm(va) < 1e-3 or np.linalg.norm(vb) < 1e-3:
                continue
            ang = np.degrees(np.arccos(np.clip(va @ vb / np.linalg.norm(va) / np.linalg.norm(vb), -1, 1)))
            errs[b][f] = ang
        hv_keys = vis > 0.5
        rp = robot[f][:, :2]
        on = (robot[f][:, 2] > 0) & (rp[:, 0] > 0) & (rp[:, 0] < 1) & (rp[:, 1] > 0) & (rp[:, 1] < 1)
        if on.mean() < 0.5:
            offscreen += 1
        if hv_keys.sum() >= 4:
            hc, rc = human[f][hv_keys, :2].mean(0), rp[hv_keys].mean(0)
            center.append(np.linalg.norm(hc - rc))
            hs = np.ptp(human[f][hv_keys, 1])
            rs = np.ptp(rp[hv_keys, 1])
            if hs > 0.05:
                size_ratio.append(rs / hs)
    per_bone = {b: float(np.nanmean(e)) for b, e in errs.items() if np.isfinite(e).any()}
    all_e = np.nanmean(np.stack([e for e in errs.values()]), 0) if per_bone else np.array([])
    # worst time windows (0.5 s) for feedback
    worst = []
    if all_e.size:
        w = max(int(0.5 * fps), 1)
        for s in range(0, T, w):
            seg = {b: np.nanmean(e[s:s + w]) for b, e in errs.items() if np.isfinite(e[s:s + w]).any()}
            if seg:
                b, v = max(seg.items(), key=lambda kv: kv[1])
                worst.append((round(s / fps, 1), round((s + w) / fps, 1), b, round(float(v))))
        worst = sorted(worst, key=lambda x: -x[3])[:5]
    out = dict(limb_angle_err_deg=round(float(np.nanmean(all_e)), 1) if all_e.size else None,
               per_limb_deg={b: round(v, 1) for b, v in sorted(per_bone.items(), key=lambda kv: -kv[1])},
               worst_windows=[dict(t_start=a, t_end=b, limb=c, err_deg=d) for a, b, c, d in worst],
               framing_center_offset=round(float(np.mean(center)), 3) if center else None,
               framing_size_ratio=round(float(np.median(size_ratio)), 2) if size_ratio else None,
               robot_offscreen_frac=round(offscreen / max(T, 1), 3))
    if extra:
        out.update(extra)
    return out


def feedback(pm):
    """Short, quantitative text for the judge."""
    if not pm or pm.get("limb_angle_err_deg") is None:
        return "pose_match unavailable (the person was not detected)."
    s = [f"Measured pose match (2D, human video vs robot from the same camera): mean limb-direction error "
         f"{pm['limb_angle_err_deg']} deg; per limb {pm['per_limb_deg']}."]
    for w in pm["worst_windows"][:4]:
        s.append(f"  worst: {w['t_start']}-{w['t_end']} s {w['limb']} off by {w['err_deg']} deg")
    if pm.get("framing_size_ratio") is not None:
        s.append(f"Framing: robot appears {pm['framing_size_ratio']}x the person's on-screen height, center offset "
                 f"{pm['framing_center_offset']} of the image; robot off-screen in {pm['robot_offscreen_frac']*100:.0f}% of frames.")
    for k in ("flipped_frames", "object_above_head_frames"):
        if pm.get(k):
            s.append(f"{k}: {pm[k]}")
    return "\n".join(s)


# ---------------------------------------------------------------- camera fitting from body correspondences
def _proj_params(p, pts, fovy, W, H):
    pos, look = p[:3], p[3:6]
    return project(pts, pos, look - pos, np.array([0, 0, 1.0]), fovy, W, H)


TORSO = [KEYS.index(k) for k in ("l_sho", "r_sho", "l_hip", "r_hip")]


def _resid(p, robot3d, human, fovy, W, H, prior, w_prior):
    """Torso-only reprojection residual (shoulders + hips): the robot reproduces the torso reliably, so the camera is
    not bent to hide arm/leg errors. Only confidently detected points count."""
    T = len(robot3d)
    r = []
    for f in range(T):
        pr = _proj_params(p, robot3d[f][TORSO], fovy, W, H)
        hf = human[f][TORSO]
        v = hf[:, 2] > 0.85
        e = (pr[:, :2] - hf[:, :2]) * np.array([W / H, 1.0])
        r.append((e * v[:, None]).ravel())
    r.append(w_prior * (p - prior))
    return np.concatenate(r)


def fit_camera(robot3d, human, fovy, W, H, init_pos, init_look, iters=25, w_prior=0.15):
    """Gauss-Newton fit of camera position + look-at so the robot's projected joints land on the person's detected
    2D joints. Gemini's camera is the prior; returns (pos, look, rms_before, rms_after)."""
    p = np.concatenate([init_pos, init_look]).astype(float)
    prior = p.copy()
    def rms(x):
        r = _resid(x, robot3d, human, fovy, W, H, prior, 0.0)
        n = max(int(sum((h[TORSO, 2] > 0.85).sum() for h in human)), 1)
        return float(np.sqrt((r ** 2).sum() / n))
    r0 = rms(p)
    lam = 1e-2
    for _ in range(iters):
        r = _resid(p, robot3d, human, fovy, W, H, prior, w_prior)
        J = np.zeros((len(r), 6))
        for k in range(6):
            dp = np.zeros(6)
            dp[k] = 1e-3
            J[:, k] = (_resid(p + dp, robot3d, human, fovy, W, H, prior, w_prior) - r) / 1e-3
        step = np.linalg.solve(J.T @ J + lam * np.eye(6), -J.T @ r)
        p_new = p + np.clip(step, -0.5, 0.5)
        if (_resid(p_new, robot3d, human, fovy, W, H, prior, w_prior) ** 2).sum() < (r ** 2).sum():
            p, lam = p_new, lam * 0.5
        else:
            lam *= 4
        if np.abs(step).max() < 1e-4:
            break
    return p[:3], p[3:], r0, rms(p)


def fit_camera_path(robot3d, human, fovy, W, H, init_pos, init_look, fps, window_s=1.0, moving=None):
    """Static fit; if the camera clearly moves (Gemini says so, or the static fit leaves a large residual that drifts
    over time), refine per window with a strong pull to the static camera, then smooth -> list of (t, pos, look)."""
    pos, look, r0, r1 = fit_camera(robot3d, human, fovy, W, H, init_pos, init_look)
    T = min(len(robot3d), len(human))
    w = max(int(window_s * fps), 2)
    win_rms = []
    for s in range(0, T, w):
        seg = slice(s, min(s + w, T))
        if (human[seg][:, TORSO, 2] > 0.85).sum() >= 8:
            rr = _resid(np.concatenate([pos, look]), robot3d[seg], human[seg], fovy, W, H, np.zeros(6), 0.0)
            win_rms.append(np.sqrt((rr ** 2).sum() / max((human[seg][:, TORSO, 2] > 0.85).sum(), 1)))
    drift = bool(win_rms) and max(win_rms) > 0.12 and max(win_rms) > 2.0 * min(win_rms)
    path = []
    if moving or (moving is None and drift):
        for s in range(0, T, w):
            seg = slice(s, min(s + w, T))
            if (human[seg][:, TORSO, 2] > 0.85).sum() < 8:
                continue
            p2, l2, _, _ = fit_camera(robot3d[seg], human[seg], fovy, W, H, pos, look, iters=12, w_prior=0.6)
            path.append(((s + min(s + w, T)) / 2 / fps, p2, l2))
        if len(path) >= 3:
            P = np.array([np.concatenate([p, l]) for _, p, l in path])
            Ps = P.copy()
            for i in range(len(P)):
                Ps[i] = P[max(i - 1, 0):i + 2].mean(0)
            path = [(t, q[:3], q[3:]) for (t, _, _), q in zip(path, Ps)]
    return dict(static=(pos, look), path=path, rms_gemini=round(r0, 3), rms_fit=round(r1, 3), moving=bool(path))


def hand_activity(w3, fps, contacts=None):
    """Which hand reaches when: arm extension (shoulder -> wrist distance, from the 3D landmarks). A grasping arm
    stretches toward the object; the idle one stays close to the body. Anatomical sides."""
    if w3 is None or len(w3) < 3:
        return {}
    P = w3[:, :, :3].astype(float)
    out = {}
    ext = {}
    for name, (sh, wr) in (("left", (11, 15)), ("right", (12, 16))):
        e = np.linalg.norm(P[:, wr] - P[:, sh], axis=1)
        k = 4
        e = np.array([e[max(0, i - k):i + k + 1].mean() for i in range(len(e))])
        ext[name] = e
    arm_len = max(float(np.percentile(ext["left"], 98)), float(np.percentile(ext["right"], 98)), 0.3)
    for name, e in ext.items():
        act = e > 0.82 * arm_len
        ivs, start = [], None
        for i, a in enumerate(act):
            if a and start is None:
                start = i
            if (not a or i == len(act) - 1) and start is not None:
                if i - start > 0.15 * fps:
                    ivs.append((round(start / fps, 1), round(i / fps, 1)))
                start = None
        out[name] = dict(reaching_intervals_s=ivs, max_extension_m=round(float(e.max()), 2))
    if contacts:
        chk = []
        for c in contacts:
            g = int(c.get("t_grasp", 0) * fps)
            win = slice(max(0, g - int(0.5 * fps)), g + int(0.5 * fps))
            l, r = float(ext["left"][win].max()), float(ext["right"][win].max())
            meas = "both" if abs(l - r) < 0.06 else ("right" if r > l else "left")
            chk.append(dict(t_grasp=c.get("t_grasp"), planned=c.get("hand"), measured=meas,
                            left_arm_extension_m=round(l, 2), right_arm_extension_m=round(r, 2)))
        out["grasp_hand_check"] = chk
    return out


def hand_activity_text(ha):
    if not ha:
        return ""
    s = ("VIDEO TRACKING (from the person's 3D body landmarks, anatomical sides; an arm that reaches out stretches): " +
         "; ".join(f"{h} arm reaching during {v['reaching_intervals_s'] or 'never'} (max extension {v['max_extension_m']} m)"
                   for h, v in ha.items() if h in ("left", "right")))
    for c in ha.get("grasp_hand_check", []):
        flag = "" if c["planned"] == c["measured"] or c["measured"] == "both" else "  <-- MISMATCH, check the hand"
        s += (f"\n grasp at {c['t_grasp']} s: planned hand {c['planned']}, measured {c['measured']} "
              f"(left arm {c['left_arm_extension_m']} m, right arm {c['right_arm_extension_m']} m){flag}")
    return s
