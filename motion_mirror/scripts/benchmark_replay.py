"""Offline standard-scene replay benchmark; no renderer, network, or model calls.

    .venv/bin/python scripts/benchmark_replay.py
    .venv/bin/python scripts/benchmark_replay.py path/to/final.json --out runs/custom
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from motion_mirror.replay import Replay  # noqa: E402

THRESHOLDS = dict(grasp_position_cm_lt=3.0, placement_position_cm_lt=3.0,
                  fingers_touching_min=1, sustained_duration_s_gt=0.1,
                  sustained_depth_cm_gt=2.0, sustained_events_max=0,
                  deterministic_required=True)
LIMITATIONS = [
    "Cached interpretation replay in the standard generated scene, not room replay or end-to-end inference.",
    "Kinematic replay attaches objects and constrains release; no physics, grasp-force, or hardware success is measured.",
    "Grasp and placement values are IK target residuals, not human tracking accuracy or independent ground truth.",
    "Collision metrics use Replay's sampled geometry checks and contact exclusions, not continuous collision detection.",
    "Replay currently processes at most one object contact event; incomplete contact coverage fails this benchmark.",
    "Replay-only timing includes scene construction and kinematic computation; excludes rendering and source interpretation.",
    "Default fixtures are two selected cached examples, not a held-out dataset or a population accuracy estimate.",
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(path, fps):
    result = dict(input=str(path), input_sha256=None, interpretation_engine="unknown/cached",
                  duration_s=None, fps=fps, grasp_position_cm=None, placement_position_cm=None,
                  fingers_touching_min=None, collision_events=None, sustained_collision_events=None,
                  replay_only_seconds=None, repeat_replay_only_seconds=None, deterministic=None,
                  declared_contact_events=0, replayed_contact_events=0, checks={}, status="error", error=None)
    try:
        result["input_sha256"] = sha256(path)
        interp = json.loads(path.read_text())
        result["interpretation_engine"] = str(interp.get("interpretation_engine") or "unknown/cached")
        duration = float(interp.get("duration_s") or max(k["t"] for k in interp["keyframes"]) + 0.5)
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("Replay duration must be finite and positive")
        result["duration_s"] = duration
        has_contacts = bool(interp.get("object") and interp.get("contacts"))
        result["declared_contact_events"] = len(interp.get("contacts") or []) if has_contacts else 0
        telemetry = []
        for key in ("replay_only_seconds", "repeat_replay_only_seconds"):
            start = time.perf_counter()
            telemetry.append(Replay(interp, 64, 64).run(fps=fps, video_path=None))
            result[key] = round(time.perf_counter() - start, 6)
        encoded = [json.dumps(t, sort_keys=True, default=float, allow_nan=False) for t in telemetry]
        result["deterministic"] = encoded[0] == encoded[1]
        tel = telemetry[0]
        grasps = [d for grasp in tel.get("grasps", []) for d in grasp.get("detail", [])]
        placements = tel.get("placements", [])
        collisions = tel.get("collisions", [])
        result["replayed_contact_events"] = len(tel.get("grasps", []))
        if grasps:
            result["grasp_position_cm"] = round(max(d["grip_pos_err_m"] for d in grasps) * 100, 6)
            result["fingers_touching_min"] = min(d["fingers_touching"] for d in grasps)
        if placements:
            result["placement_position_cm"] = round(max(p["placement_err_m"] for p in placements) * 100, 6)
        sustained = [c for c in collisions
                     if c["t_end"] - c["t_start"] > THRESHOLDS["sustained_duration_s_gt"]
                     and c["max_depth_cm"] > THRESHOLDS["sustained_depth_cm_gt"]]
        result["collision_events"] = len(collisions)
        result["sustained_collision_events"] = len(sustained)
        needs_place = bool(has_contacts and interp["object"].get("end") and any(
            c.get("t_release") is not None and 0 <= round(c["t_release"] * fps) < len(np.arange(0, duration, 1 / fps))
            for c in interp["contacts"]))
        result["checks"] = dict(
            grasp_position=(bool(grasps) and result["grasp_position_cm"] < THRESHOLDS["grasp_position_cm_lt"])
            if has_contacts else None,
            fingers_touching=(bool(grasps) and result["fingers_touching_min"] >= THRESHOLDS["fingers_touching_min"])
            if has_contacts else None,
            placement_position=(bool(placements) and result["placement_position_cm"] < THRESHOLDS["placement_position_cm_lt"])
            if needs_place else None,
            contacts_covered=(result["replayed_contact_events"] == result["declared_contact_events"])
            if has_contacts else None,
            no_sustained_collisions=len(sustained) == 0,
            deterministic=result["deterministic"],
        )
        result["telemetry"] = {key: value for key, value in tel.items() if key != "samples"}
        result["status"] = "fail" if False in result["checks"].values() else "pass"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="*")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "replay_benchmark")
    parser.add_argument("--fps", type=float, default=30)
    args = parser.parse_args()
    if not np.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be finite and positive")
    inputs = args.inputs or [ROOT / "benchmarks" / name for name in
                            ("bottle_v2_score8.5.json", "chair_v2_score9.0.json")]
    cases = [evaluate(path.resolve(), args.fps) for path in inputs]
    now = datetime.now(timezone.utc)
    report = dict(version=1, benchmark="standard_scene_kinematic_replay", generated_at=now.isoformat(),
                  fps=args.fps, thresholds=THRESHOLDS, limitations=LIMITATIONS,
                  environment=dict(python=platform.python_version(), mujoco=mujoco.__version__,
                                   numpy=np.__version__, platform=platform.platform(),
                                   replay_source_sha256=sha256(ROOT / "src" / "motion_mirror" / "replay.py")),
                  counts={status: sum(c["status"] == status for c in cases) for status in ("pass", "fail", "error")},
                  cases=cases)
    args.out.mkdir(parents=True, exist_ok=True)
    stem = "report_" + now.strftime("%Y%m%dT%H%M%S_%fZ")
    json_path, csv_path = args.out / f"{stem}.json", args.out / f"{stem}.csv"
    report["outputs"] = dict(json=str(json_path.resolve()), csv=str(csv_path.resolve()))
    json_path.write_text(json.dumps(report, indent=2, default=float, allow_nan=False) + "\n")
    columns = [key for key in cases[0] if key not in ("checks", "telemetry")]
    columns += [f"check_{key}" for key in ("grasp_position", "fingers_touching", "placement_position",
                                          "contacts_covered", "no_sustained_collisions", "deterministic")]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for case in cases:
            row = {key: case[key] for key in columns if key in case}
            row.update({f"check_{key}": value for key, value in case["checks"].items()})
            writer.writerow(row)
    print("Standard-scene kinematic replay only; residuals are not human accuracy or physics success.")
    print("case                               grasp cm  place cm  sustained  deterministic  quality")
    for case in cases:
        def value(key):
            return "N/A" if case[key] is None else str(case[key])
        print(f"{Path(case['input']).stem:34} {value('grasp_position_cm'):>8}  "
              f"{value('placement_position_cm'):>8}  {value('sustained_collision_events'):>9}  "
              f"{value('deterministic'):>13}  {case['status']}")
        failed = [key for key, ok in case["checks"].items() if ok is False]
        if case["error"] or failed:
            print("  " + (case["error"] or "Failed: " + ", ".join(failed)))
    print(f"JSON: {json_path.resolve()}\nCSV:  {csv_path.resolve()}")
    return int(any(case["status"] != "pass" for case in cases))


if __name__ == "__main__":
    sys.exit(main())
