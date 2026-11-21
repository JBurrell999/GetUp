#!/usr/bin/env python3
"""
export_session.py - bundle a judge session (from session.py) for the sim comparison.

    python export_session.py outputs/sessions/<session>                # -> <session>.zip
    python export_session.py outputs/sessions/<session> --with-video   # include the videos
    python export_session.py outputs/sessions/<session> --all-takes    # not just accepted

By default: the accepted take of each task, joints + metadata only. The comparison needs
the joint data, and the videos show the judge's face and anyone else in the room, so they
are only included on request. FORMAT.md, describing every file, goes in the bundle.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = ["joints3d.npz", "joints24.npz", "take.json"]
VIDEO = ["raw.mp4", "capture_24pt.mp4"]


def export(sess_dir, with_video=False, all_takes=False) -> Path:
    sess_dir = Path(sess_dir)
    session = json.loads((sess_dir / "session.json").read_text())
    out = sess_dir.parent / f"{sess_dir.name}.zip"
    root, n_takes = sess_dir.name, 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(sess_dir / "session.json", f"{root}/session.json")
        if (HERE / "FORMAT.md").exists():
            z.write(HERE / "FORMAT.md", f"{root}/FORMAT.md")
        for task in session["tasks"]:
            takes = ([t["take"] for t in task["takes"]] if all_takes
                     else [task["accepted_take"]] if task.get("accepted_take") else [])
            for take in takes:
                d = sess_dir / task["dir"] / take
                for f in DATA + (VIDEO if with_video else []):
                    if (d / f).exists():
                        z.write(d / f, f"{root}/{task['dir']}/{take}/{f}")
                n_takes += 1
    missing = [t["id"] for t in session["tasks"] if not t.get("accepted_take")]
    print(f"[export] {n_takes} takes -> {out} ({out.stat().st_size / 1e6:.1f} MB)"
          + (f"\n[export] no accepted take for: {', '.join(missing)}" if missing else ""))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Bundle a judge session for the sim comparison")
    ap.add_argument("session", help="outputs/sessions/<session> folder")
    ap.add_argument("--with-video", action="store_true", help="include raw and overlay videos")
    ap.add_argument("--all-takes", action="store_true", help="include rejected takes too")
    a = ap.parse_args()
    export(a.session, a.with_video, a.all_takes)
