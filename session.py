#!/usr/bin/env python3
"""
session.py - guided recording of a task suite by a judge, for comparison with the sim.

    python session.py --judge J01                         # every task in tasks.json
    python session.py --judge J01 --only sit_down,get_up  # just some tasks
    python session.py --judge J01 --tier core             # the essential ~25 tasks
    python session.py --judge J01 --category "lifting & carrying"
    python session.py --list                              # what's in tasks.json
    python session.py --judge J01 --hands-free            # no keyboard needed between takes

For each task the screen shows the instruction. The operator (or the judge) presses SPACE,
a beeping countdown runs, and recording starts on the "go" beep. Recording stops by itself
once the judge has moved and then held still for a moment (or after the task's max_s;
"stop": "time" tasks just run for max_s). The take is processed (2D, then 3D), checked, and
shown for review: accept it or redo it.

Keys   brief:     SPACE start | N skip task | B previous task | Q quit
       countdown: SPACE cancel
       recording: SPACE stop now | X discard this take
       review:    ENTER/SPACE accept | R redo | Q quit
Mouse: click a person to lock onto them (bystanders), click empty space for auto.

Output: outputs/sessions/<date>_<judge>/
    session.json                 judge, camera, body measurements (from the calibrate
                                 task), and every task's takes + which one was accepted
    NN_<task>/take<k>/           raw.mp4, capture_24pt.mp4, joints24.npz (2D),
                                 joints3d.npz (3D, for the sim), take.json (task, timing,
                                 quality flags)
Bundle it for sending with export_session.py. What each file contains: FORMAT.md.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
import shutil
import sys
import time
import warnings
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import lift3d
import stream24 as s

HERE = Path(__file__).resolve().parent
MOVE, STILL = 1.0, 0.5         # body-size units (~head widths) per second
STILL_S = 1.5                  # held still this long after moving = the action is done
MIN_MOVED_S = 0.4              # must have moved at least this long before stillness counts
WINDOW = 1.0                   # s of live skeleton the motion meter looks at

# SMPL-24 bones/parts used for quality checks and body measurements
BONES = {"shoulder_width": (16, 17), "hip_width": (1, 2),
         "upper_arm_l": (16, 18), "upper_arm_r": (17, 19),
         "forearm_l": (18, 20), "forearm_r": (19, 21),
         "thigh_l": (1, 4), "thigh_r": (2, 5), "shin_l": (4, 7), "shin_r": (5, 8)}
PARTS = {"head_torso": [0, 3, 6, 9, 12, 15], "arms": [16, 17, 18, 19, 20, 21, 22, 23],
         "legs": [1, 2, 4, 5, 7, 8, 10, 11]}


def to_json(obj) -> str:
    """JSON that other languages can read: NaN/inf become null."""
    def clean(o):
        if isinstance(o, float):
            return o if math.isfinite(o) else None
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        return o
    return json.dumps(clean(obj), indent=2, allow_nan=False)


# --------------------------------------------------------------------------- motion
class Motion:
    """How much the tracked person is moving, in body-size units (~head widths) per second:
    how far each joint of the smoothed live skeleton ranged over the last WINDOW seconds. A range,
    not start-vs-end, so back-and-forth motion like waving counts even when it returns to
    where it was. Uses the second-fastest joint: a wave moves only the wrist and hand, while
    one jittery joint on its own isn't enough to count as moving."""

    def __init__(self, window=None):
        window = WINDOW if window is None else window
        self.window, self.hist = window, deque()

    def __call__(self, t, uv, conf, scale):
        self.hist.append((t, uv.copy(), conf.copy()))
        while len(self.hist) > 2 and t - self.hist[1][0] >= self.window:
            self.hist.popleft()
        span = t - self.hist[0][0]
        pts = np.stack([u for _, u, _ in self.hist])
        ok = np.all([c > 0 for _, _, c in self.hist], 0)
        if span < 0.5 * self.window or scale <= 0 or ok.sum() < 4:
            return None
        rng = np.sort(np.hypot(*np.ptp(pts[:, ok], axis=0).T))
        return float(rng[-2] / scale / span)


# --------------------------------------------------------------------------- quality
def assess(take_dir: Path) -> dict:
    """Checks a processed take; status 'ok' or 'check' with human-readable flags."""
    z3, z2 = np.load(take_dir / "joints3d.npz"), np.load(take_dir / "joints24.npz")
    X, C, t = z3["xyz"], z3["conf"], z3["t"]
    T = len(X)
    dur = float(t[-1]) if T > 1 else 0.0
    has3d = np.isfinite(X).all((1, 2))
    vis = {p: round(float(C[has3d][:, idx].mean()), 2) if has3d.any() else 0.0
           for p, idx in PARTS.items()}
    cv = {}
    for name, (a, b) in BONES.items():
        ln = np.linalg.norm(X[:, a] - X[:, b], axis=1)
        if np.isfinite(ln).sum() > 2:
            cv[name] = round(float(np.nanstd(ln) / np.nanmean(ln)), 3)
    w, h = z2["image_size"]
    uv, c2 = z2["body25_uv"], z2["body25_conf"]
    seen2d = (c2 > 0).any(1)
    m = 0.02 * min(w, h)
    near = [(c > 0).any() and ((u[c > 0] < m) | (u[c > 0] > [w - m, h - m])).any()
            for u, c in zip(uv, c2)]
    flags = []
    if dur < 1.0:
        flags.append(f"very short take ({dur:.1f} s)")
    if has3d.mean() < 0.9:
        flags.append(f"no 3D in {100 * (1 - has3d.mean()):.0f}% of frames")
    if seen2d.mean() < 0.9:
        flags.append(f"lost track of the judge in {100 * (1 - seen2d.mean()):.0f}% of frames")
    if np.mean(near) > 0.2:
        flags.append(f"body at the edge of the picture in {100 * np.mean(near):.0f}% of frames")
    for p in ("legs", "arms"):
        if vis[p] < 0.4:
            flags.append(f"{p} poorly visible (mean visibility {vis[p]:.2f})")
    if cv and max(cv.values()) > 0.15:
        flags.append(f"tracking jumped (bone length varies {100 * max(cv.values()):.0f}%)")
    return {"status": "check" if flags else "ok", "flags": flags, "frames": T,
            "duration_s": round(dur, 2), "frames_with_3d_pct": round(100 * float(has3d.mean()), 1),
            "visibility": vis, "bone_length_cv": cv}


def body_measurements(take_dir: Path) -> dict:
    """Judge's segment lengths (m) from the neutral-stance take, for scaling/normalizing."""
    X = np.load(take_dir / "joints3d.npz")["xyz"]
    with warnings.catch_warnings():            # all-NaN columns just give null
        warnings.simplefilter("ignore", RuntimeWarning)
        out = {name: round(float(np.nanmedian(np.linalg.norm(X[:, a] - X[:, b], axis=1))), 3)
               for name, (a, b) in BONES.items()}
        out["height_approx"] = round(float(np.nanmedian(X[:, 15, 2])) + 0.11, 2)  # ears + ~11 cm
    return out


# --------------------------------------------------------------------------- drawing
def _wrap(text, font, scale, th, max_w):
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if cv2.getTextSize(trial, font, scale, th)[0][0] <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    return lines + ([cur] if cur else [])


def draw_card(img, title, body=(), footer="", color=(255, 255, 255)):
    """Instruction card across the bottom of the picture."""
    h, w = img.shape[:2]
    font, sc = cv2.FONT_HERSHEY_SIMPLEX, max(0.55, h / 900)
    th, lh, pad = max(1, round(sc * 1.6)), int(34 * sc), int(16 * sc)
    rows = [(title, sc * 1.35, color, th + 1)]
    for para in body:
        rows += [(ln, sc, (235, 235, 235), th) for ln in _wrap(para, font, sc, th, w - 2 * pad)]
    if footer:
        rows.append((footer, sc * 0.85, (150, 220, 255), th))
    top = h - pad - lh * len(rows) - int(lh * 0.4)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, top - pad), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    y = top + int(lh * 0.6)
    for text, scale, col, thick in rows:
        cv2.putText(img, text, (pad, y), font, scale, col, thick, cv2.LINE_AA)
        y += int(lh * scale / sc) if scale > sc else lh


# --------------------------------------------------------------------------- session
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Guided task-suite recording for sim comparison")
    ap.add_argument("--judge", help="judge ID (use a code, not a full name)")
    ap.add_argument("--tasks", default=str(HERE / "tasks.json"))
    ap.add_argument("--only", help="comma-separated task ids to record")
    ap.add_argument("--category", help="comma-separated categories to record (see --list)")
    ap.add_argument("--tier", help="comma-separated tiers: core, extended, edge")
    ap.add_argument("--list", action="store_true", help="list the tasks by category and exit")
    ap.add_argument("--source", default="0", help="camera index or stream URL")
    ap.add_argument("--countdown", type=float, default=3)
    ap.add_argument("--hands-free", action="store_true",
                    help="start each task after --brief-s and accept takes after --review-s")
    ap.add_argument("--brief-s", type=float, default=8, help="hands-free: time to read a task")
    ap.add_argument("--review-s", type=float, default=5, help="hands-free: review time")
    ap.add_argument("--live-height", type=int, default=256)
    ap.add_argument("--net-height", type=int, default=368)
    ap.add_argument("--thr", type=float, default=0.1)
    ap.add_argument("--target", choices=["cpu", "opencl", "cuda"], default="cpu")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--headless", action="store_true", help="no window (implies --hands-free)")
    ap.add_argument("--out", default=str(HERE / "outputs"))
    a = ap.parse_args(argv)
    if not a.list and not a.judge:
        ap.error("--judge is required")
    a.hands_free = a.hands_free or a.headless
    return a


def list_tasks(tasks):
    cats = {}
    for t in tasks:
        cats.setdefault(t.get("category", "other"), []).append(t)
    for cat, ts in cats.items():
        print(f"\n{cat} ({len(ts)})")
        for t in ts:
            extra = ", ".join(filter(None, [f"{t.get('max_s')} s max",
                                            "fixed length" if t.get("stop") == "time" else "",
                                            f"needs {', '.join(t['props'])}" if t.get("props") else ""]))
            print(f"  {t['id']:24s} {t.get('tier', ''):9s} {t['name']}  ({extra})")
    print(f"\n{len(tasks)} tasks")


def main(argv=None, net=None):
    args = parse_args(argv)
    spec = json.loads(Path(args.tasks).read_text())
    tasks = spec["tasks"]
    if args.tier:
        want = {x.strip() for x in args.tier.split(",") if x.strip()}
        unknown = want - {t.get("tier", "extended") for t in tasks}
        if unknown:
            sys.exit(f"[session] unknown tier(s): {', '.join(sorted(unknown))}")
        tasks = [t for t in tasks if t.get("tier", "extended") in want]
    if args.category:
        want = {x.strip() for x in args.category.split(",") if x.strip()}
        unknown = want - {t.get("category", "other") for t in tasks}
        if unknown:
            sys.exit(f"[session] unknown categor(ies): {', '.join(sorted(unknown))} (see --list)")
        tasks = [t for t in tasks if t.get("category", "other") in want]
    if args.only:
        want = [x.strip() for x in args.only.split(",") if x.strip()]
        unknown = set(want) - {t["id"] for t in spec["tasks"]}
        if unknown:
            sys.exit(f"[session] unknown task id(s): {', '.join(sorted(unknown))} (see --list)")
        tasks = [t for t in tasks if t["id"] in want]
    if args.list:
        list_tasks(tasks)
        return None
    if not tasks:
        sys.exit("[session] no tasks selected")
    judge = re.sub(r"[^A-Za-z0-9_-]+", "_", args.judge).strip("_") or "judge"
    sess_dir = Path(args.out) / "sessions" / f"{datetime.now():%Y%m%d_%H%M%S}_{judge}"
    sess_dir.mkdir(parents=True)
    session = {"judge": args.judge, "started": datetime.now().isoformat(timespec="seconds"),
               "tasks_file": Path(args.tasks).name, "setup": spec.get("setup"),
               "camera": None, "body": None, "format": "FORMAT.md",
               "tasks": [{"id": t["id"], "name": t["name"], "dir": f"{i + 1:02d}_{t['id']}",
                          "status": "pending", "accepted_take": None, "takes": []}
                         for i, t in enumerate(tasks)]}

    def save():
        (sess_dir / "session.json").write_text(to_json(session))
    save()

    src = int(args.source) if args.source.isdigit() else args.source
    mirror = isinstance(src, int) if args.mirror is None else args.mirror
    cap = cv2.VideoCapture(src)
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        sys.exit(f"[session] could not open source {args.source!r}")
    net = net if net is not None else s.load_net(args.target)
    pool = s.make_pool(args)
    reader = s.LatestFrameReader(cap)
    follow, motion = s.Follower(), Motion()
    proc = argparse.Namespace(net_height=args.net_height, thr=args.thr, smooth=True,
                              labels=False, max_frames=0)
    win = "session - task recording"
    frame_w = [0]
    if not args.headless:
        cv2.namedWindow(win)

        def on_mouse(event, x, y, *_):
            if event == cv2.EVENT_LBUTTONDOWN:
                follow.picker.click = (frame_w[0] - 1 - x if mirror else x, y)
        cv2.setMouseCallback(win, on_mouse)

    k, state, state_t, seq, last = 0, "brief", time.monotonic(), 0, time.monotonic()
    tick = take_dir = seed = review = None
    moved = still_since = rec_t0 = None
    print(f"[session] {len(tasks)} tasks for judge {args.judge} -> {sess_dir}")

    def go(new_state):
        nonlocal state, state_t
        state, state_t = new_state, time.monotonic()
        if new_state == "brief" and k < len(tasks):
            print(f"[session] task {k + 1}/{len(tasks)}: {tasks[k]['name']}")

    def finish_take(reason, until=None):
        nonlocal review
        stamps = reader.stop_capture()
        s.beep("Hero")
        print(f"[session] take stopped ({reason}); processing ...")
        if len(stamps) < 2:
            shutil.rmtree(take_dir, ignore_errors=True)
            go("brief")
            return
        show2d = None if args.headless else s.progress_window(win, "Processing take (2D)")
        s.process_video(net, take_dir / "raw.mp4", take_dir, proc, picker=seed, stamps=stamps,
                        show=show2d, pool=pool, until=until)
        lift3d.lift_capture(take_dir, take_dir / "raw.mp4", show=None if args.headless
                            else s.progress_window(win, "Processing take (3D)"))
        task, entry = tasks[k], session["tasks"][k]
        q = assess(take_dir)
        take = {"task": {key: task.get(key) for key in ("id", "category", "tier", "name", "instruction",
                                                        "props", "note")},
                "judge": args.judge, "take": take_dir.name,
                "recorded_at": datetime.now().isoformat(timespec="seconds"),
                "stop_reason": reason, "fps": round(float(q["frames"] / max(q["duration_s"], 1e-6)), 2),
                "quality": q,
                "files": {"joints3d": "joints3d.npz", "joints2d": "joints24.npz",
                          "video": "raw.mp4", "video_overlay": "capture_24pt.mp4"},
                "software": {"pose2d": "OpenPose BODY_25 (OpenCV DNN)",
                             "pose3d": "MediaPipe pose_landmarker_heavy",
                             "opencv": cv2.__version__}}
        (take_dir / "take.json").write_text(to_json(take))
        entry["takes"].append({"take": take_dir.name, "status": q["status"], "flags": q["flags"],
                               "stop_reason": reason, "duration_s": q["duration_s"]})
        save()
        review = q
        print(f"[session] {take_dir.name}: {q['status'].upper()}"
              + (" - " + "; ".join(q["flags"]) if q["flags"] else ""))
        go("review")

    try:
        while k < len(tasks):
            frame, seq = reader.read(seq)
            if frame is None:
                print("[session] camera stopped")
                break
            now = time.monotonic()
            dt, last = now - last, now
            frame_w[0] = frame.shape[1]
            if session["camera"] is None and reader.fps > 0 and now - state_t > 1:
                session["camera"] = {"size": [frame.shape[1], frame.shape[0]],
                                     "fps": round(reader.fps, 1), "mirrored_preview": mirror}
                save()
            people = s.infer_people(net, frame, args.live_height, args.thr)
            uv, conf, kp25, conf25 = follow(people, now, True)
            speed = motion(now, uv, conf, s.person_scale(kp25, conf25))
            task, key = tasks[k], -1

            view = None if args.headless else s.draw_skeleton(frame.copy(), uv, conf, False, mirror)
            hud = (f"judge {args.judge} | task {k + 1}/{len(tasks)}: {task['name']} | "
                   f"{len(people)} people | target: {'locked' if follow.picker.locked else 'closest'}")

            if state == "brief":
                props = f"You need: {', '.join(task['props'])}." if task.get("props") else ""
                left = args.brief_s - (now - state_t)
                footer = (f"starting in {max(0, math.ceil(left))} s  (SPACE now | N skip | Q quit)"
                          if args.hands_free else "SPACE start | N skip | B previous | Q quit")
                if view is not None:
                    draw_card(view, f"Task {k + 1}/{len(tasks)}: {task['name']}",
                              [task["instruction"]] + ([props] if props else []), footer)
                if args.hands_free and left <= 0:
                    tick = None
                    go("countdown")
            elif state == "countdown":
                left = args.countdown - (now - state_t)
                if left <= 0:
                    entry = session["tasks"][k]
                    take_dir = sess_dir / entry["dir"] / f"take{len(entry['takes']) + 1}"
                    take_dir.mkdir(parents=True, exist_ok=True)
                    reader.start_capture(take_dir / "raw.mp4")
                    seed = copy.deepcopy(follow.picker)
                    moved, still_since, rec_t0 = 0.0, None, now
                    s.beep("Glass")
                    go("recording")
                else:
                    if math.ceil(left) != tick:
                        tick = math.ceil(left)
                        s.beep("Tink")
                    if view is not None:
                        s.draw_banner(view, str(tick))
                        draw_card(view, task["name"], [task["instruction"]], "SPACE cancel")
            elif state == "recording":
                el = now - rec_t0
                if task.get("stop") == "time":
                    status = f"{el:4.1f}/{task['max_s']} s"
                    if el >= task["max_s"]:
                        finish_take("time")
                        continue
                else:
                    if speed is not None and speed > MOVE:
                        moved, still_since = moved + dt, None
                    elif speed is not None and speed < STILL:
                        still_since = still_since or now
                    else:
                        still_since = None
                    held = now - still_since if still_since else 0.0
                    status = (f"{el:4.1f} s | " + ("waiting for movement" if moved < MIN_MOVED_S
                              else f"still {held:.1f}/{task.get('still_s', STILL_S)} s" if held
                              else "moving") + f" | speed {speed or 0:.1f}")
                    if moved >= MIN_MOVED_S and held >= task.get("still_s", STILL_S):
                        # stillness registers ~WINDOW after the movement ended: keep 0.5 s of hold
                        finish_take("finished", until=still_since - WINDOW + 0.5)
                        continue
                    if el >= task["max_s"]:
                        finish_take("max duration")
                        continue
                hud += f" | REC {status}"
            elif state == "review":
                left = args.review_s - (now - state_t)
                footer = (f"accepting in {max(0, math.ceil(left))} s  (R redo)" if args.hands_free
                          else "ENTER accept | R redo | Q quit")
                if view is not None:
                    title = f"{take_dir.name}: " + ("looks good" if review["status"] == "ok"
                                                    else "check this take")
                    body = [f"{review['duration_s']:.1f} s, 3D in {review['frames_with_3d_pct']:.0f}% "
                            f"of frames"] + review["flags"]
                    draw_card(view, title, body, footer,
                              (120, 230, 120) if review["status"] == "ok" else (80, 200, 255))
                if args.hands_free and left <= 0:
                    key = 13               # accept

            if view is not None:
                s.draw_hud(view, hud, state == "recording")
                cv2.imshow(win, view)
                key = cv2.waitKey(1) & 0xFF if key < 0 else key
            elif args.headless and seq % 15 == 0:
                print(f"[session] {state}: {hud}", flush=True)

            # ------------------------------------------------------------ keys
            if key in (ord("q"), 27) and state in ("brief", "review"):
                break
            if state == "brief":
                if key == ord(" "):
                    tick = None
                    go("countdown")
                elif key == ord("n"):
                    session["tasks"][k]["status"] = "skipped"
                    save()
                    k += 1
                    go("brief")
                elif key == ord("b") and k > 0:
                    k -= 1
                    go("brief")
            elif state == "countdown" and key == ord(" "):
                go("brief")
            elif state == "recording":
                if key == ord(" "):
                    finish_take("stopped by operator")
                elif key == ord("x"):
                    reader.stop_capture()
                    shutil.rmtree(take_dir, ignore_errors=True)
                    print("[session] take discarded")
                    go("brief")
            elif state == "review":
                if key in (13, 10, ord(" ")):
                    entry = session["tasks"][k]
                    entry["status"], entry["accepted_take"] = "accepted", take_dir.name
                    if tasks[k]["id"] == "calibrate":
                        session["body"] = body_measurements(take_dir)
                    save()
                    k += 1
                    go("brief")
                elif key == ord("r"):
                    go("brief")
    finally:
        if state == "recording":
            reader.stop_capture()
        reader.stop()
        cap.release()
        if pool is not None:
            pool.close()
        if not args.headless:
            cv2.destroyAllWindows()
        session["finished"] = datetime.now().isoformat(timespec="seconds")
        save()
    done = sum(t["status"] == "accepted" for t in session["tasks"])
    print(f"[session] {done}/{len(tasks)} tasks accepted -> {sess_dir}\n"
          f"[session] to send: python export_session.py {sess_dir}")
    return sess_dir


if __name__ == "__main__":
    main()
