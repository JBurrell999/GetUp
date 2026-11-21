"""Local fallback for the seated water-bottle sample, without external APIs.

    ./mm local_pose video.mp4 --landmarks local_pose.npz --out runs/local

The landmark file contains MediaPipe world_landmarks (T,33,3), visibility (T,33),
and t (T,). This module never downloads a model or uploads the source video.
This sample-specific fallback fits the two arms to observed segment directions.
Seated legs, bottle/table dimensions and contact events are annotated estimates;
it does not infer arbitrary objects, activities, scenes or contacts from poses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

from . import config as C
from .replay import ELBOW_STRAIGHT, _arm


def _unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-9)


def _smooth(a, sigma=2.0):
    radius = max(1, int(np.ceil(3 * sigma)))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(a, [(radius, radius)] + [(0, 0)] * (a.ndim - 1), mode="edge")
    return np.stack([np.tensordot(kernel, padded[i:i + len(kernel)], axes=(0, 0))
                     for i in range(len(a))])


def _clean_landmarks(p, visibility, times):
    """Interpolate missing/very occluded points, median-filter, then smooth."""
    clean = np.empty_like(p, dtype=float)
    for joint in range(33):
        good = np.isfinite(p[:, joint]).all(1) & (visibility[:, joint] >= 0.25)
        if not good.any():
            # Hidden feet are never used for fitting; retain finite model guesses.
            good = np.isfinite(p[:, joint]).all(1)
        if not good.any():
            clean[:, joint] = 0
            continue
        for dim in range(3):
            clean[:, joint, dim] = np.interp(times, times[good], p[good, joint, dim])
    median = np.stack([np.median(clean[max(0, i-2):min(len(clean), i+3)], axis=0)
                       for i in range(len(clean))])
    return _smooth(median, sigma=2.0)


def torso_frames(p):
    """Columns: person's forward, anatomical left, up in camera coordinates."""
    up = _unit(p[:, [11, 12]].mean(1) - p[:, [23, 24]].mean(1))
    left = p[:, 11] - p[:, 12]
    left = _unit(left - (left * up).sum(1)[:, None] * up)
    forward = _unit(np.cross(left, up))
    return np.stack([forward, left, up], axis=-1)


class ArmFitter:
    """Damped least-squares fit of both observed bone vectors to G1 geometry."""

    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(str(C.SCENE_XML))
        self.d = mujoco.MjData(self.m)
        self.d.qpos[:] = 0
        self.d.qpos[2:4] = [0.8, 1.0]
        self.jid = np.array([self.m.joint(name + "_joint").id for name in C.JOINTS])
        self.qadr = self.m.jnt_qposadr[self.jid]
        self.dadr = self.m.jnt_dofadr[self.jid]
        self.lo, self.hi = self.m.jnt_range[self.jid].T
        self.arm_indices = {"left": np.arange(15, 19), "right": np.arange(22, 26)}
        self.bodies = {side: [self.m.body(f"{side}_{name}_link").id for name in
                              ["shoulder_roll", "elbow", "wrist_yaw"]]
                       for side in self.arm_indices}
        self.jac = np.zeros((3, self.m.nv))
        self.lengths = {}
        self._set(C.DEFAULT_Q)
        for side in self.arm_indices:
            shoulder, elbow, wrist = self.d.xpos[self.bodies[side]]
            self.lengths[side] = np.array([np.linalg.norm(elbow - shoulder),
                                           np.linalg.norm(wrist - elbow)])

    def _set(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)

    def segments(self, side, q):
        self._set(q)
        positions = self.d.xpos[self.bodies[side]]
        return np.diff(positions, axis=0)

    def fit(self, side, directions, weights, previous, first=False):
        indices = self.arm_indices[side]
        dofs = self.dadr[indices]
        q = previous.copy()
        target = self.lengths[side][:, None] * directions
        row_weight = np.repeat(np.sqrt(np.clip(weights, 0.15, 1.0)), 3)
        prior = 1e-7 if first else 2e-5
        for _ in range(80):
            self._set(q)
            pos = self.d.xpos[self.bodies[side]].copy()
            vectors = np.diff(pos, axis=0)
            jacobians = []
            for body in self.bodies[side]:
                mujoco.mj_jac(self.m, self.d, self.jac, None, self.d.xpos[body], body)
                jacobians.append(self.jac[:, dofs].copy())
            J = np.vstack([jacobians[1] - jacobians[0], jacobians[2] - jacobians[1]])
            error = (target - vectors).ravel()
            J = row_weight[:, None] * J
            error = row_weight * error
            step = np.linalg.solve(J.T @ J + (1e-5 + prior) * np.eye(4),
                                   J.T @ error + prior * (previous[indices] - q[indices]))
            step = np.clip(step, -0.18, 0.18)
            q[indices] = np.clip(q[indices] + step, self.lo[indices], self.hi[indices])
            if np.linalg.norm(step) < 1e-5:
                break
        return q


def arm_schema(q, side):
    """Exact inverse of replay._arm, preserving fitted G1 joint angles."""
    sign = 1 if side == "left" else -1
    a = {
        "shoulder_flexion_deg": float(np.degrees(-q[0])),
        "shoulder_abduction_deg": float(np.degrees(sign * q[1])),
        "shoulder_external_rotation_deg": float(np.degrees(sign * q[2])),
        "elbow_flexion_deg": float(np.degrees(ELBOW_STRAIGHT - q[3])),
    }
    assert np.allclose(_arm(a, side)[:4], q, atol=1e-8)
    return {key: round(value, 4) for key, value in a.items()}


def retarget(video, landmarks, out, grasp=2.1, release=5.9):
    video, landmarks, out = map(Path, [video, landmarks, out])
    data = np.load(landmarks)
    raw = np.asarray(data["world_landmarks"], dtype=float)
    visibility = np.asarray(data["visibility"], dtype=float)
    times = np.asarray(data["t"], dtype=float)
    if raw.shape != (len(times), 33, 3) or visibility.shape != (len(times), 33):
        raise ValueError("Expected world_landmarks (T,33,3), visibility (T,33), t (T,)")
    if len(times) < 2 or not np.all(np.diff(times) > 0):
        raise ValueError("At least two strictly increasing source timestamps required")
    required = [11, 12, 13, 14, 15, 16, 23, 24]
    if not np.isfinite(raw[:, required]).any():
        raise ValueError("No usable upper-body landmark observations")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(3)), int(cap.get(4))
    cap.release()
    duration = frame_count / fps
    if grasp < times[0] or release <= grasp or release >= duration:
        raise ValueError("Contact times must lie within the video and release must follow grasp")
    p = _clean_landmarks(raw, visibility, times)
    bases = torso_frames(p)
    fitter = ArmFitter()
    directions, weights = {}, {}
    for side, ids in [("left", [11, 13, 15]), ("right", [12, 14, 16])]:
        segments = np.diff(p[:, ids], axis=1)
        directions[side] = _unit(np.einsum("tij,tkj->tki", bases.transpose(0, 2, 1), segments))
        weights[side] = np.stack([visibility[:, ids[:2]].min(1),
                                  visibility[:, ids[1:]].min(1)], axis=1)
    fitted = np.repeat(C.DEFAULT_Q[None], len(times), axis=0)
    previous = C.DEFAULT_Q.copy()
    for i in range(len(times)):
        for side in ["left", "right"]:
            previous = fitter.fit(side, directions[side][i], weights[side][i], previous, first=(i == 0))
        fitted[i] = previous
    fitted = _smooth(fitted, sigma=1.5)
    fitted = np.clip(fitted, fitter.lo, fitter.hi)
    # The occluded hips make depth/forward lean uncertain. Preserve conservative
    # source-derived changes relative to the opening half-second, not a new motion.
    baseline = bases[:max(1, round(0.5 / np.median(np.diff(times))))].mean(0)
    u, _, vt = np.linalg.svd(baseline)
    baseline = u @ vt
    relative = np.einsum("ji,tjk->tik", baseline, bases)
    lean_measured = np.degrees(np.arctan2(relative[:, 0, 2], relative[:, 2, 2]))
    side_measured = np.degrees(np.arctan2(-relative[:, 1, 2], relative[:, 2, 2]))
    twist_measured = np.degrees(np.arctan2(relative[:, 1, 0], relative[:, 0, 0]))
    lean = np.clip(5 + 0.35 * _smooth(lean_measured), -5, 20)
    side_bend = np.clip(0.5 * _smooth(side_measured), -15, 15)
    twist = np.clip(_smooth(twist_measured), -35, 35)
    keyframes = []
    # 10 Hz keyframes preserve the measured trajectory without source frame noise.
    wanted = np.r_[np.arange(times[0], times[-1], 0.1), grasp, release, times[-1]]
    indices = sorted(set(np.abs(times[:, None] - wanted[None]).argmin(0).tolist()))
    seated = {"thigh_forward_deg": 88.0, "knee_flexion_deg": 90.0, "hip_abduction_deg": 5.0}
    for i in indices:
        keyframes.append({
            "t": round(float(times[i]), 6),
            "pelvis": {"forward_m": 0.0, "left_m": 0.0, "facing_left_deg": 0.0},
            "trunk": {"lean_forward_deg": round(float(lean[i]), 3),
                      "side_bend_left_deg": round(float(side_bend[i]), 3),
                      "twist_left_deg": round(float(twist[i]), 3)},
            "left_leg": dict(seated), "right_leg": dict(seated),
            "left_arm": arm_schema(fitted[i, 15:19], "left"),
            "right_arm": arm_schema(fitted[i, 22:26], "right"),
        })
    # Metric scene estimates are separate from fitted arm observations. The room
    # renderer may adjust this interaction patch to its navigable filming area.
    interp = {
        "summary": "The seated subject reaches across with the right hand, picks up the water bottle, holds it upright near chest height, and sets it down nearer the table center.",
        "interpretation_engine": "local_mediapipe_pose_ik",
        "environment": {"description": "Seated bottle transfer at a black covered hackathon table; metric dimensions estimated for G1 proportions.",
                        "surfaces": [{"name": "table", "kind": "table", "forward_m": 0.57, "left_m": 0.05,
                                      "depth_m": 0.50, "width_m": 1.35, "height_m": 0.54}]},
        "object": {"type": "bottle", "description": "Clear plastic water bottle with white cap", "color_rgb": [0.8, 0.9, 0.95],
                   "grasp_style": "side", "keep_upright": True,
                   "size_m": [0.065, 0.065, 0.21], "grasp_height_frac": 0.48,
                   "start": {"forward_m": 0.40, "left_m": 0.20, "on_surface_height_m": 0.54, "yaw_deg": 0.0},
                   "end": {"forward_m": 0.40, "left_m": 0.12, "on_surface_height_m": 0.54, "yaw_deg": 0.0}},
        "object_end": {"forward_m": 0.40, "left_m": 0.12, "on_surface_height_m": 0.54, "yaw_deg": 0.0},
        "camera": {"horizontal_distance_m": 1.4, "height_m": 1.12, "azimuth_deg": 5.0,
                   "vertical_fov_deg": 48.0, "look_at_forward_m": 0.20, "look_at_left_m": 0.05, "look_at_height_m": 0.75},
        "duration_s": duration, "start_pose": {key: value for key, value in keyframes[0].items() if key != "t"},
        "contacts": [{"t_grasp": grasp, "t_release": release, "hand": "right"}],
        "keyframes": keyframes,
    }
    arm_reports = {}
    errors, target_vectors = [], []
    for side in ["left", "right"]:
        actual = np.stack([fitter.segments(side, q) for q in fitted])
        target = fitter.lengths[side][None, :, None] * directions[side]
        errors.append(actual - target)
        target_vectors.append(target)
        angles = np.degrees(np.arccos(np.clip((_unit(actual) * directions[side]).sum(-1), -1, 1)))
        wrist_error = np.linalg.norm((actual - target).sum(1), axis=-1)
        arm_reports[side] = {
            "robot_segment_lengths_m": fitter.lengths[side].round(6).tolist(),
            "median_visibility": {"shoulder": float(np.median(visibility[:, 11 if side == "left" else 12])),
                                  "elbow": float(np.median(visibility[:, 13 if side == "left" else 14])),
                                  "wrist": float(np.median(visibility[:, 15 if side == "left" else 16]))},
            "segment_angle_error_deg": {"median": np.median(angles, axis=0).round(3).tolist(),
                                         "p95": np.percentile(angles, 95, axis=0).round(3).tolist()},
            "wrist_vector_error_m": {"median": float(np.median(wrist_error)), "p95": float(np.percentile(wrist_error, 95))},
        }
    report = {
        "interpretation_engine": "local_mediapipe_pose_ik",
        "source_video": str(video.resolve()), "source_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "landmarks": str(landmarks.resolve()), "external_video_transfer": False,
        "frames": len(times), "fps": fps, "duration_s": duration, "resolution": [width, height],
        "keyframes": len(keyframes), "arm_fit": arm_reports,
        "method": "MediaPipe heavy 3D segment directions in per-frame torso frame; fit G1 shoulder pitch/roll/yaw and elbow with MuJoCo point Jacobians, true link lengths, joint limits, damped least squares and temporal continuity. Convert joints via exact inverse of replay._arm.",
        "fit_metric_scope": "Residuals measure the numerical fit to smoothed MediaPipe estimates after scaling each segment to G1 length. They do not measure true human 3D accuracy or contact accuracy.",
        "smoothing": {"landmark_median_frames": 5, "landmark_gaussian_sigma_frames": 2,
                      "joint_gaussian_sigma_frames": 1.5, "keyframe_hz": 10},
        "assumptions": [
            "This is kinematic retargeting; no physics controller or sim-to-real validation.",
            "Feet and knees are occluded. Both legs use a constant inferred seated pose (88 deg thigh, 90 deg knee); no measured leg motion is claimed.",
            "Source camera is uncalibrated. Torso orientation is relative to the opening half-second. Forward lean changes are attenuated to 35%, side bend to 50%, with conservative clamps because hip depth is inferred.",
            "Left wrist is partially occluded. Confidence below 0.25 is interpolated before smoothing; its reconstruction is less certain than the right arm.",
            f"Right-hand grasp at {grasp:.2f}s and release at {release:.2f}s are estimated by local visual inspection of the source; no object detector or hand-contact sensor was used.",
            "Table size/height, bottle dimensions/placement, camera pose and a stationary pelvis are metric reconstruction assumptions and may be adjusted for the target XML scene.",
            "Wrist joints are neutral in keyframes; the existing replay solver supplies grasp orientation and finger closure from bottle geometry.",
            "The source bottle stays upright. Its orientation is constrained upright during carry because the pose model does not measure wrist rotation. Table/bottle placement is adjusted to G1 reach and checked for collisions.",
        ],
        "source_phases_s": {"rest": [0, 1.0], "reach": [1.1, 2.1], "lift": [2.2, 3.2],
                            "hold": [3.2, 4.6], "lower": [4.7, 5.7], "retract": [6.1, 6.7]},
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "final.json").write_text(json.dumps(interp, indent=2))
    (out / "extraction_report.json").write_text(json.dumps(report, indent=2))
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(out / "retarget.npz", t=times, q=fitted, clean_world_landmarks=p,
                        torso_frames=bases, target_vectors=np.stack(target_vectors, axis=1),
                        residual_vectors=np.stack(errors, axis=1), visibility=visibility)
    return interp, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--landmarks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--grasp", type=float, default=2.1, help="Visually estimated right-hand grasp time")
    parser.add_argument("--release", type=float, default=5.9, help="Visually estimated release time")
    args = parser.parse_args()
    _, report = retarget(args.video, args.landmarks, args.out, args.grasp, args.release)
    print(json.dumps({"out": str(args.out), "frames": report["frames"], "arm_fit": report["arm_fit"]}, indent=2))


if __name__ == "__main__":
    main()
