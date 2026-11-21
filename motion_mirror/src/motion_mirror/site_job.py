"""Run a downloaded capture-site upload through interpretation and room rendering.

    ./mm site_job /path/to/take.mp4 --metadata /path/to/take.json --name sample

The room comes from the upload's details (environment.id: hackathon, kitchen or warehouse), or --env. A take
that names no room is refused rather than guessed. Every judged motion round is replayed inside that room
(mirror.run(room=...)), and the final videos are rendered there too.

This runs on local files. site_worker.py fetches uploads from the Blob store, runs this, and puts the videos
back under outputs/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2

from . import config as C
from . import mirror
from .in_env import ENVS, render_in_env


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _video_details(path):
    capture = cv2.VideoCapture(str(path))
    try:
        ok, frame = capture.read()
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not ok or frame is None or fps <= 0 or frames <= 0:
            raise ValueError(f"Cannot decode video: {path}")
        return dict(width=frame.shape[1], height=frame.shape[0], fps=fps,
                    frames=frames, duration_s=frames / fps, size_bytes=path.stat().st_size)
    finally:
        capture.release()


def _validate_render(path):
    """Decode every frame, so a successful encoder exit alone cannot mark a job complete."""
    capture = cv2.VideoCapture(str(path))
    frames = 0
    first = None
    changed = False
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        reported = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames += 1
            small = cv2.resize(frame, (64, 64))
            if first is None:
                first = small
                height, width = frame.shape[:2]
            elif not changed and cv2.absdiff(first, small).mean() > 0.1:
                changed = True
        if frames == 0 or fps <= 0 or frames != reported:
            raise ValueError(f"Rendered video failed full decoding: {path} ({frames}/{reported} frames)")
        return dict(decoded_frames=frames, width=width, height=height, fps=fps,
                    duration_s=frames / fps, size_bytes=path.stat().st_size,
                    has_visual_change=changed)
    finally:
        capture.release()


def run(video, *, name=None, metadata=None, env=None, resume=False,
        model="gemini-4-hydrogen", judge_model=None, env_iters=3,
        env_threshold=8.0, iters=10, threshold=9.0, facts=(), wide=True):
    video = Path(video).expanduser().resolve(strict=True)
    if not video.is_file():
        raise ValueError(f"Video must be a file: {video}")
    details = _video_details(video)
    source_hash = _sha256(video)
    meta = {}
    metadata_path = None
    if metadata:
        metadata_path = Path(metadata).expanduser().resolve(strict=True)
        meta = json.loads(metadata_path.read_text())
        if not isinstance(meta, dict):
            raise ValueError("Upload metadata must be a JSON object")
    selected = env
    if selected is None:
        selected = (meta.get("environment") or {}).get("id") if isinstance(meta.get("environment"), dict) else None
    if selected is None:
        raise ValueError("The upload doesn't say which room it was filmed in; pass --env "
                         f"({', '.join(ENVS)})")
    if selected not in ENVS:
        raise ValueError(f"Unknown environment {selected!r}; choose one of {', '.join(ENVS)}")
    if env_iters < 1 or iters < 1:
        raise ValueError("Environment and motion iterations must each be at least 1")

    if name is None:
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", video.stem).strip("._-")[:70] or "upload"
        name = f"{stem}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", name):
        raise ValueError("Run name must be 1–120 letters, numbers, dots, underscores or hyphens, starting with a letter or number")
    runs = (C.ROOT / "runs" / "mirror2").resolve()
    out = runs / name
    if out.resolve().parent != runs:
        raise ValueError("Run directory must stay directly inside runs/mirror2")
    if out.exists() and not resume:
        raise FileExistsError(f"Run already exists: {out}. Choose another --name or use --resume to render its final.json")
    if resume and not (out / "final.json").is_file():
        raise FileNotFoundError(f"--resume requires a completed interpretation: {out / 'final.json'}")
    report_path = out / "job.json"
    if resume and report_path.is_file():
        previous = json.loads(report_path.read_text())
        if previous.get("source", {}).get("sha256") != source_hash:
            raise ValueError("Source video differs from the original run; choose another --name")
        if previous.get("environment", {}).get("id") != selected:
            raise ValueError("Environment differs from the original run; choose another --name")

    started = time.time()
    if not resume:
        mirror.STATS.clear()
        mirror.run(str(video), name, model, env_iters, env_threshold, iters,
                   threshold, list(facts), judge_model=judge_model, bench=False, room=selected)
    prepared = out / "input.mp4"
    if not prepared.is_file():
        prepared = mirror.prepare_video(video, out)
    interpretation_stats = {}
    if (out / "summary.json").is_file():
        interpretation_stats = json.loads((out / "summary.json").read_text())
    report = dict(
        version=1, status="rendering", mode="kinematic",
        interpretation_engine=interpretation_stats.get("interpretation_engine", "Gemini"),
        source=dict(path=str(video), sha256=source_hash, **details),
        metadata_path=str(metadata_path) if metadata_path else None,
        upload_metadata=meta,
        activity=meta.get("activity"),
        environment=dict(id=selected, name=ENVS[selected]["name"],
                         room_xml=ENVS[selected]["xml"]),
        prepared_video=str(prepared), interpretation=str(out / "final.json"),
        resumed_interpretation=resume,
    )
    report["interpretation_stats"] = interpretation_stats
    report_path.write_text(json.dumps(report, indent=2))
    render_started = time.time()
    room_out = out / "environment"
    place, telemetry = render_in_env(out / "final.json", selected, prepared, room_out, wide=wide)
    outputs = {key: str(room_out / filename) for key, filename in (
        ("robot", "robot_env.mp4"), ("comparison", "side_by_side_env.mp4"),
        ("scene", "scene.xml"), ("placement", "place.json"), ("telemetry", "telemetry_env.json"),
    )}
    if wide:
        outputs["wide"] = str(room_out / "robot_wide.mp4")
    validation = {key: _validate_render(Path(path)) for key, path in outputs.items()
                  if Path(path).suffix == ".mp4"}
    report.update(status="complete", outputs=outputs, placement=place, telemetry=telemetry,
                  validation=validation,
                  render_seconds=round(time.time() - render_started, 2),
                  job_seconds=round(time.time() - started, 2),
                  completed_at=datetime.now(timezone.utc).isoformat())
    report_path.write_text(json.dumps(report, indent=2, default=float))
    print(f"[site] complete: {outputs['robot']}", flush=True)
    print(f"[site] comparison: {outputs['comparison']}", flush=True)
    return report_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="Downloaded source video")
    parser.add_argument("--metadata", help="Downloaded JSON next to the upload")
    parser.add_argument("--env", choices=sorted(ENVS), help="Override the upload's room; required if its details name none")
    parser.add_argument("--name", help="Unique run name under runs/mirror2")
    parser.add_argument("--resume", action="store_true", help="Render existing final.json without calling Gemini again")
    parser.add_argument("--model", default="gemini-4-hydrogen")
    parser.add_argument("--judge-model")
    parser.add_argument("--env-iters", type=int, default=3)
    parser.add_argument("--env-threshold", type=float, default=8.0)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=9.0)
    parser.add_argument("--fact", action="append", default=[], help="User-verified motion fact")
    parser.add_argument("--no-wide", action="store_true", help="Skip the second, wider camera render")
    args = parser.parse_args()
    try:
        run(args.video, name=args.name, metadata=args.metadata, env=args.env,
            resume=args.resume, model=args.model, judge_model=args.judge_model,
            env_iters=args.env_iters, env_threshold=args.env_threshold,
            iters=args.iters, threshold=args.threshold, facts=args.fact, wide=not args.no_wide)
    except (ValueError, OSError, json.JSONDecodeError) as error:
        parser.exit(1, f"site_job: {error}\n")


if __name__ == "__main__":
    main()
