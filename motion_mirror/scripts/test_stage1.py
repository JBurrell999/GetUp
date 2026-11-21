"""Offline checks for the stage-1 pipeline (no Gemini calls).   uv run scripts/test_stage1.py"""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from motion_mirror import config as C  # noqa: E402
from motion_mirror.mirror import apply_patch, merge, prepare_video, replay_matched, video_info  # noqa: E402
from motion_mirror.replay import Replay  # noqa: E402

# latest full-pipeline results (interpretation + the converted input video)
def _latest(clip):
    runs = sorted((ROOT / "runs/mirror2").glob(f"*{clip}*/final.json"), key=lambda p: p.stat().st_mtime)
    runs = [r for r in runs if (r.parent / "input.mp4").exists()]
    return runs[-1] if runs else None


BENCH = {c: _latest(c) for c in ("bottle", "chair")}
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


# 1. no false-positive collisions: robot standing neutrally in an empty scene, and seated pose clear of a table
neutral = dict(environment={"surfaces": []}, object=None, camera={}, duration_s=1.0,
               keyframes=[dict(t=0.0, left_arm={"shoulder_flexion_deg": 5, "shoulder_abduction_deg": 12},
                               right_arm={"shoulder_flexion_deg": 5, "shoulder_abduction_deg": 12})])
_, coll = Replay(neutral, 320, 240).still()
check("neutral standing pose has no collisions", not coll, f"{coll}")

# 2. patch merge: change one field, add one keyframe, remove one, replace contacts
motion = dict(contacts=[dict(t_grasp=1.0, t_release=3.0, hand="right")],
              keyframes=[dict(t=0.0, trunk={"lean_forward_deg": 0}, left_arm={"shoulder_flexion_deg": 5, "elbow_flexion_deg": 10}),
                         dict(t=1.0, trunk={"lean_forward_deg": 10}, left_arm={"shoulder_flexion_deg": 40}),
                         dict(t=2.0, trunk={"lean_forward_deg": 5})])
patched = apply_patch(motion, dict(keyframes=[dict(t=1.02, left_arm={"elbow_flexion_deg": 50}), dict(t=1.5, trunk={"lean_forward_deg": 20})],
                                   remove_keyframe_t=[2.0], contacts=[dict(t_grasp=1.2, t_release=3.0, hand="left")]))
ts = [k["t"] for k in patched["keyframes"]]
kf1 = patched["keyframes"][1]
check("patch: merges by time, keeps untouched fields", kf1["left_arm"] == {"shoulder_flexion_deg": 40, "elbow_flexion_deg": 50}, f"{kf1}")
check("patch: adds / removes / retimes keyframes, sorted", ts == [0.0, 1.02, 1.5], f"{ts}")
check("patch: replaces contacts", patched["contacts"][0]["hand"] == "left")
check("patch: original motion not mutated", motion["keyframes"][1]["left_arm"] == {"shoulder_flexion_deg": 40})

# 3. benchmark interpretations: grasp / placement accuracy, deterministic replay, speed
for name, path in BENCH.items():
    if path is None:
        check(f"{name}: a full-pipeline result exists", False)
        continue
    it = json.loads(path.read_text())
    vid = path.parent / "input.mp4"
    W, H, fps = video_info(vid)
    t0 = time.time()
    rp, tel = replay_matched(it, vid, path.parent, W, H, fps, None)
    dt = time.time() - t0
    _, tel2 = replay_matched(it, vid, path.parent, W, H, fps, None)
    print(f"   ({name}: {path.parent.name})")
    g = tel["grasps"][0]["detail"] if tel["grasps"] else []
    check(f"{name}: grasp reached (<3 cm)", bool(g) and all(x["grip_pos_err_m"] < 0.03 for x in g),
          f"{[(x['hand'], x['grip_pos_err_m'], x['grip_rot_err_deg'], x['fingers_touching']) for x in g]}")
    check(f"{name}: fingers touch the object", bool(g) and all(x["fingers_touching"] >= 1 for x in g))
    if tel.get("placements"):
        check(f"{name}: placed at target (<3 cm)", all(p["placement_err_m"] < 0.03 for p in tel["placements"]), f"{tel['placements']}")
    long_coll = [c for c in tel["collisions"] if c["t_end"] - c["t_start"] > 0.1 and c["max_depth_cm"] > 2.0]
    check(f"{name}: no sustained collisions (>0.1 s, >2 cm)", not long_coll, f"{long_coll[:3]}")
    check(f"{name}: posture retargeted from the video", (tel.get("retarget") or {}).get("arm_frames_from_video", 0) > 0, f"{tel.get('retarget')}")
    check(f"{name}: static camera", not (tel.get("camera_from_perspective") or {}).get("moving", False))
    check(f"{name}: body never flipped", not rp.flags.get("flipped_frames"), f"{rp.flags}")
    check(f"{name}: replay deterministic", json.dumps(tel, default=float) == json.dumps(tel2, default=float))
    check(f"{name}: matched replay fast (<10 s, no render)", dt < 10.0, f"{dt:.2f}s")

# 4. any upload: MOV / 1080p is converted to ~854 px mp4
src = ROOT / "videos/IMG_6520.MOV"
if src.exists():
    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.time()
        out = prepare_video(src, Path(tmp))
        w, h, fps = video_info(out)
        check("upload: MOV converted to mp4 <=1000 px", out.suffix == ".mp4" and max(w, h) <= 1000, f"{w}x{h} @{fps:.0f}fps in {time.time()-t0:.1f}s")

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
