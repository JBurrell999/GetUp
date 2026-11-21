"""Extract tracked MediaPipe world landmarks using an already installed model.

No network calls or model downloads. Crop and subject coordinates are source-video pixels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO / "models/mediapipe/pose_landmarker_heavy.task"
EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
         (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
         (24, 26), (26, 28), (15, 17), (15, 19), (16, 18), (16, 20)]


def extract(video, out, *, model=DEFAULT_MODEL, crop=None, subject_center=None, subject_radius=95.0):
    video = Path(video).expanduser().resolve(strict=True)
    model = Path(model).expanduser().resolve(strict=True)
    out = Path(out).expanduser().resolve()
    if not video.is_file() or not model.is_file():
        raise ValueError("Video and preinstalled model must both be files; this command never downloads models")
    for filename in ("local_pose.npz", "local_pose_summary.json", "local_pose_overlay.mp4"):
        if (out / filename).exists():
            raise FileExistsError(f"Output already exists: {out / filename}; choose another --out")
    capture = cv2.VideoCapture(str(video))
    writer = None
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width, height = (int(capture.get(key)) for key in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
        if not capture.isOpened() or width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Cannot open video with a valid frame rate and dimensions: {video}")
        box = tuple(crop) if crop is not None else (0, 0, width, height)
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(f"Crop must satisfy 0 <= x0 < x1 <= {width} and 0 <= y0 < y1 <= {height}")
        target = np.array(subject_center if subject_center is not None else ((x0 + x1) / 2, (y0 + y1) / 2), float)
        if not np.isfinite(target).all() or not (x0 <= target[0] < x1 and y0 <= target[1] < y1):
            raise ValueError("Subject center must be a finite pixel coordinate inside the crop")
        if not math.isfinite(subject_radius) or subject_radius <= 0:
            raise ValueError("Subject radius must be positive")

        # Import only after local path/input validation; model_asset_path never downloads a model.
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model), delegate=BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO, num_poses=2,
            min_pose_detection_confidence=0.4, min_pose_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        worlds, visibilities, images, times, selected, pose_counts = [], [], [], [], [], []
        previous = target.copy()
        with vision.PoseLandmarker.create_from_options(options) as detector:
            out.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(out / "local_pose_overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not writer.isOpened():
                raise OSError("Could not open the MP4 overlay encoder")
            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
                result = detector.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)),
                    round(frame_index * 1000 / fps))
                world = np.full((33, 3), np.nan)
                visibility = np.zeros(33)
                image = np.full((33, 3), np.nan)
                chosen = -1
                candidates = []
                for index, landmarks in enumerate(result.pose_landmarks):
                    uv = np.array([[point.x * (x1 - x0) + x0, point.y * (y1 - y0) + y0] for point in landmarks])
                    shoulders = uv[[11, 12]].mean(0)
                    score = np.linalg.norm(shoulders - target) + 0.5 * np.linalg.norm(shoulders - previous)
                    candidates.append((score, index, uv, shoulders))
                if candidates:
                    _, index, uv, shoulders = min(candidates, key=lambda candidate: candidate[0])
                    if np.linalg.norm(shoulders - target) < subject_radius:
                        chosen = index
                        previous = shoulders
                        world = np.array([[point.x, point.y, point.z] for point in result.pose_world_landmarks[index]])
                        visibility = np.array([point.visibility for point in result.pose_world_landmarks[index]])
                        image = np.array([[point.x * (x1 - x0) + x0, point.y * (y1 - y0) + y0,
                                           point.z * (x1 - x0)] for point in result.pose_landmarks[index]])
                        for a, b in EDGES:
                            color = (50, 220, 50) if min(visibility[a], visibility[b]) > 0.6 else (0, 180, 240)
                            cv2.line(frame, tuple(np.rint(uv[a]).astype(int)), tuple(np.rint(uv[b]).astype(int)), color, 2)
                        for joint in (11, 12, 13, 14, 15, 16):
                            point = tuple(np.rint(uv[joint]).astype(int))
                            cv2.circle(frame, point, 4, (255, 70, 70), -1)
                            cv2.putText(frame, f"{joint}:{visibility[joint]:.2f}", (point[0] + 5, point[1] - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                cv2.putText(frame, f"LOCAL MediaPipe | {frame_index / fps:.2f}s | pose {chosen}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                writer.write(frame)
                if frame_index % 30 == 0:
                    cv2.imwrite(str(out / f"pose_{frame_index:03d}.jpg"), frame)
                worlds.append(world)
                visibilities.append(visibility)
                images.append(image)
                times.append(frame_index / fps)
                selected.append(chosen)
                pose_counts.append(len(result.pose_landmarks))
                frame_index += 1
                if frame_index % 30 == 0:
                    print(f"Extracted {frame_index} frames", flush=True)
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if not times:
        raise ValueError("No video frames decoded")
    worlds, visibilities, images = map(np.array, (worlds, visibilities, images))
    np.savez_compressed(out / "local_pose.npz", world_landmarks=worlds, visibility=visibilities,
                        image_landmarks=images, t=np.array(times), selected=np.array(selected),
                        pose_counts=np.array(pose_counts), crop=np.array(box), fps=fps)
    with video.open("rb") as source:
        source_hash = hashlib.file_digest(source, "sha256").hexdigest()
    summary = dict(
        interpretation_engine="MediaPipe Pose Landmarker (local CPU)",
        source_video=str(video), source_sha256=source_hash,
        frames=len(times), detected_frames=int(np.sum(np.array(selected) >= 0)), fps=fps,
        crop=box, subject_center=target.tolist(), subject_radius_px=subject_radius, local_model=str(model),
        median_visibility={str(joint): float(np.median(visibilities[:, joint]))
                           for joint in (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)},
        world_axes="MediaPipe: x image right, y down, z away from camera; meters, hip centered",
    )
    (out / "local_pose_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return out / "local_pose.npz"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="Source video to process locally")
    parser.add_argument("--out", required=True, help="Directory for landmarks, overlay video, snapshots and summary")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Existing MediaPipe .task model; no downloads")
    parser.add_argument("--crop", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"), help="Subject crop in source pixels; default: full frame")
    parser.add_argument("--subject-center", type=float, nargs=2, metavar=("X", "Y"), help="Expected shoulder center in source pixels; default: crop center")
    parser.add_argument("--subject-radius", type=float, default=95.0, help="Maximum shoulder distance from subject center in pixels (default: 95)")
    args = parser.parse_args()
    try:
        extract(args.video, args.out, model=args.model, crop=args.crop,
                subject_center=args.subject_center, subject_radius=args.subject_radius)
    except (ValueError, OSError) as error:
        parser.exit(1, f"extract_local_pose: {error}\n")


if __name__ == "__main__":
    main()
