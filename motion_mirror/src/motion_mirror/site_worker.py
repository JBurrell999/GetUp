"""Turn capture-site uploads into robot videos and put them back in the Vercel Blob store.

    ./mm site_worker --list           # every take and how far it has got; changes nothing
    ./mm site_worker                  # process each take that has its details but no outputs yet, then exit
    ./mm site_worker --watch 60       # keep going, looking for new takes every 60 s
    ./mm site_worker --take takes/<judge>/<env>/<what>/<stamp>.mp4 [--redo]

Every take runs in the room its uploader picked: the <env> in its path, which the site also writes into
its JSON (the two must agree). The Gemini motion rounds are replayed and judged in that room, and the
final videos are rendered there (site_job.py). Takes from before the site asked "where are you filming?"
have no room and are skipped; run one with --take ... --env <room>.

What this writes, next to nothing else (takes/ is never changed):

    outputs/<judge>/<env>/<what>/<stamp>/result.json        status, room, scores, placement, collisions
    outputs/<judge>/<env>/<what>/<stamp>/robot.mp4          the G1 in the room, from the filming camera
    outputs/<judge>/<env>/<what>/<stamp>/side_by_side.mp4   the person | the robot, time-synced
    outputs/<judge>/<env>/<what>/<stamp>/wide.mp4           a wider view of the robot in the room

result.json is written first with status "processing", then "complete" or "failed". A take with any
result.json is left alone; --retry also picks up "failed" ones and runs a crashed one's interpretation
again only if it never finished (a finished one is re-rendered without new Gemini calls). Run one
worker at a time.

The store token comes from BLOB_READ_WRITE_TOKEN, or else ../web/.env.local (`vercel env pull`).
Blob calls go through ../web/scripts/blob.mjs, so ../web needs `npm install`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import config as C
from . import site_job
from .in_env import ENVS

REPO = C.ROOT.parent
WEB = REPO / "web"
HELPER = WEB / "scripts" / "blob.mjs"
DOWNLOADS = C.ROOT / "runs" / "site_downloads"

# The site's scene file for each room (web/src/lib/environments.ts), recorded in result.json.
SITE_SCENES = {
    "hackathon": "g1_room_mujoco/Hackathon_G1_Ultra_Detail_V2/generated/G1_HACKATHON_V2.xml",
    "kitchen": "g1_room_mujoco/Kitchen_G1_Detail/generated/G1_KITCHEN.xml",
    "warehouse": "g1_room_mujoco/Warehouse_G1_Detail/generated/G1_WAREHOUSE.xml",
}

# Same shape as web/src/lib/takes.ts
_JUDGE = r"[a-z0-9]+(?:-[a-z0-9]+)*"
_WHAT = r"[a-z0-9]+(?:_[a-z0-9]+)*"
_STAMP = r"\d{8}-\d{6}-[a-z0-9]{4}"
VIDEO_EXTS = ("mp4", "mov", "m4v", "webm")
TAKE_RE = re.compile(rf"^takes/({_JUDGE})/(?:({'|'.join(ENVS)})/)?({_WHAT})/({_STAMP})\.({'|'.join(VIDEO_EXTS)}|json)$")

OUTPUT_FILES = {   # site_job output key -> name in the store
    "robot": "robot.mp4",
    "comparison": "side_by_side.mp4",
    "wide": "wide.mp4",
}


@dataclass(frozen=True)
class Take:
    judge: str
    env: str | None          # None for uploads from before the site asked where people film
    what: str
    stamp: str
    ext: str

    @classmethod
    def parse(cls, pathname):
        m = TAKE_RE.match(pathname)
        if not m or len(m[1]) > 40 or len(m[3]) > 60:
            return None
        return cls(m[1], m[2], m[3], m[4], m[5])

    @property
    def base(self):
        """Path inside takes/ without the extension."""
        return "/".join(p for p in (self.judge, self.env, self.what, self.stamp) if p)

    @property
    def video(self):
        return f"takes/{self.base}.{self.ext}"

    @property
    def details(self):
        return f"takes/{self.base}.json"

    def output_prefix(self, env):
        return f"outputs/{self.judge}/{env}/{self.what}/{self.stamp}/"

    def run_name(self, env):
        """Run directory name under runs/mirror2 (site_job allows 120 characters)."""
        return f"site_{self.judge}_{env}_{self.what[:40]}_{self.stamp}"


def _env_file(path):
    out = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            m = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", line)
            if m and not line.lstrip().startswith("#"):
                out[m[1]] = m[2].strip().strip("'\"")
    return out


class Blob:
    """The Blob store, through web/scripts/blob.mjs."""

    def __init__(self):
        node = shutil.which("node")
        if not node:
            raise RuntimeError("node is not installed; the Blob store is reached through web/scripts/blob.mjs")
        if not (WEB / "node_modules" / "@vercel" / "blob").is_dir():
            raise RuntimeError(f"run `npm install` in {WEB} first")
        local = _env_file(WEB / ".env.local")
        token = os.environ.get("BLOB_READ_WRITE_TOKEN") or local.get("BLOB_READ_WRITE_TOKEN")
        if not token:
            raise RuntimeError("no BLOB_READ_WRITE_TOKEN: set it, or run `vercel env pull .env.local` in web/")
        self.node = node
        self.env = {**os.environ, "BLOB_READ_WRITE_TOKEN": token}
        access = os.environ.get("BLOB_ACCESS") or local.get("BLOB_ACCESS")
        if access:
            self.env["BLOB_ACCESS"] = access

    def _call(self, *args, timeout=1800):
        r = subprocess.run([self.node, str(HELPER), *map(str, args)], env=self.env, cwd=WEB,
                           capture_output=True, text=True, timeout=timeout)
        if r.returncode:
            raise RuntimeError(f"blob {args[0]} {args[1]}: {r.stderr.strip()[-400:] or 'failed'}")
        return r.stdout

    def list(self, prefix):
        return [json.loads(line) for line in self._call("list", prefix, timeout=300).splitlines() if line.strip()]

    def get(self, pathname, dest):
        self._call("get", pathname, dest)

    def get_json(self, pathname):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "blob.json"
            self.get(pathname, dest)
            return json.loads(dest.read_text())

    def put(self, pathname, path, content_type):
        if not pathname.startswith("outputs/"):
            raise ValueError(f"the worker only writes under outputs/: {pathname}")
        self._call("put", pathname, path, content_type)

    def put_json(self, pathname, obj):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            path.write_text(json.dumps(obj, indent=2, default=float))
            self.put(pathname, path, "application/json")


# ---------------------------------------------------------------- what's in the store
@dataclass
class Entry:
    take: Take
    size: int
    has_details: bool
    result: str | None       # pathname of an existing result.json, if any


def scan(blob):
    """Every uploaded video, whether its details have arrived, and whether it has outputs."""
    videos, details = {}, set()
    for item in blob.list("takes/"):
        take = Take.parse(item["pathname"])
        if take is None:
            continue
        if take.ext == "json":
            details.add(take.base)
        else:
            videos[take.base] = (take, item["size"])
    results = {item["pathname"] for item in blob.list("outputs/") if item["pathname"].endswith("/result.json")}
    out = []
    for base, (take, size) in sorted(videos.items(), key=lambda kv: kv[1][0].stamp):
        result = take.output_prefix(take.env) + "result.json" if take.env else None
        out.append(Entry(take, size, base in details, result if result in results else None))
    return out


def ready_takes(blob):
    """Takes whose details have arrived and that name a room, from one list call (the cheap check
    the watch loop makes; Blob list calls count against the store's operation limits)."""
    videos, details = {}, set()
    for item in blob.list("takes/"):
        take = Take.parse(item["pathname"])
        if take is None:
            continue
        if take.ext == "json":
            details.add(take.base)
        else:
            videos[take.base] = take
    return [t for b, t in videos.items() if b in details and t.env]


def status_of(blob, entry):
    if not entry.has_details:
        return "waiting for details"
    if entry.take.env is None:
        return "no room chosen (use --take ... --env)"
    if entry.result is None:
        return "pending"
    try:
        return blob.get_json(entry.result).get("status", "unknown")
    except (RuntimeError, ValueError) as err:
        return f"unreadable result.json ({err})"


# ---------------------------------------------------------------- one take
def _scrub(text):
    """Error text for the store, without this computer's paths."""
    return str(text).replace(str(REPO), "<repo>").replace(str(Path.home()), "~")[:800]


def _now():
    return datetime.now(timezone.utc).isoformat()


def _summary(report):
    stats = report.get("interpretation_stats") or {}
    tel = report.get("telemetry") or {}
    place = report.get("placement") or {}
    robot = (report.get("validation") or {}).get("robot") or {}
    return dict(
        interpretation_engine=report.get("interpretation_engine"),
        judged_in_room=stats.get("room") == report["environment"]["id"],
        scores=dict(environment=(stats.get("env") or {}).get("best_score"),
                    motion=(stats.get("motion") or {}).get("best_score"),
                    motion_rounds=len(stats.get("motion_rounds") or [])),
        placement=dict(spot=place.get("anchor"), clear_of_furniture=place.get("clear_of_furniture"),
                       camera_distance_m=place.get("camera_distance"), camera_used_m=place.get("match_distance")),
        collisions=tel.get("collisions", []),
        grasps=tel.get("grasps", []),
        placements=tel.get("placements", []),
        video=dict(duration_s=robot.get("duration_s"), fps=robot.get("fps"),
                   width=robot.get("width"), height=robot.get("height")),
        job_seconds=report.get("job_seconds"),
    )


def process(blob, take, *, env=None, redo=False, settings=None, log=print):
    """Download one take, run it in its room, upload the videos. Returns the final result.json contents."""
    settings = dict(settings or {})
    if take.env and env and env != take.env:
        raise ValueError(f"{take.video} was filmed in the {take.env}; it can't be run in the {env}")
    room = take.env or env
    if room is None:
        raise ValueError(f"{take.video} doesn't say which room it was filmed in; pass --env ({', '.join(ENVS)})")
    prefix = take.output_prefix(room)
    result = dict(version=1, status="processing", mode="kinematic", take=take.video,
                  environment=dict(id=room, name=ENVS[room]["name"], scene=SITE_SCENES[room],
                                   room_xml=ENVS[room]["xml"]),
                  settings=dict(settings, facts=list(settings.get("facts", ()))),
                  started_at=_now())

    local = DOWNLOADS / take.base
    meta_path = local.with_suffix(".json")
    video = local.with_suffix(f".{take.ext}")
    try:
        blob.get(take.details, meta_path)
        meta = json.loads(meta_path.read_text())
        named = (meta.get("environment") or {}).get("id") if isinstance(meta.get("environment"), dict) else None
        if named and named != room:
            raise ValueError(f"the take's path says {room} but its details say {named}")
        result.update(activity=meta.get("activity"), judge=(meta.get("judge") or {}).get("id"))
        blob.put_json(prefix + "result.json", result)
        log(f"[worker] {take.video} -> {prefix} ({ENVS[room]['name']})")
        if not video.is_file():
            blob.get(take.video, video)

        name = take.run_name(room)
        run_dir = C.ROOT / "runs" / "mirror2" / name
        resume = False
        if run_dir.exists():
            if not redo and (run_dir / "final.json").is_file():
                resume = True          # Gemini already finished for this take: render again without new calls
                log(f"[worker] reusing the finished interpretation in {run_dir.name}")
            else:
                shutil.rmtree(run_dir)  # an unfinished (or --redo) run; mirror.run would start it over anyway
        report_path = site_job.run(video, name=name, metadata=meta_path, env=room, resume=resume, **settings)
        report = json.loads(Path(report_path).read_text())
        if report.get("environment", {}).get("id") != room:
            raise RuntimeError(f"rendered in {report.get('environment')}, expected {room}")

        files = {}
        for key, filename in OUTPUT_FILES.items():
            path = (report.get("outputs") or {}).get(key)
            if path:
                blob.put(prefix + filename, path, "video/mp4")
                files[key] = filename
        result.update(status="complete", files=files, completed_at=_now(), **_summary(report))
        blob.put_json(prefix + "result.json", result)
        log(f"[worker] done: {prefix} ({', '.join(files.values())})")
        return result
    except BaseException as err:  # noqa: BLE001 - recorded in the store, then re-raised or reported
        result.update(status="failed", error=_scrub(f"{type(err).__name__}: {err}"), failed_at=_now())
        try:
            blob.put_json(prefix + "result.json", result)
        except Exception as put_err:  # noqa: BLE001
            log(f"[worker] couldn't record the failure: {put_err}")
        if isinstance(err, (KeyboardInterrupt, SystemExit)):
            raise
        log(f"[worker] FAILED {take.video}: {result['error']}")
        return result


def run_pending(blob, *, retry=False, settings=None, log=print):
    """Process every take that's ready and has no outputs yet (and failed ones, with retry)."""
    done = []
    for entry in scan(blob):
        if not entry.has_details or entry.take.env is None:
            continue
        if entry.result is not None:
            if not retry or status_of(blob, entry) not in ("failed", "processing"):
                continue
        done.append(process(blob, entry.take, settings=settings, log=log))
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="show every take and its status, then exit")
    parser.add_argument("--take", help="process this one take (its video pathname under takes/)")
    parser.add_argument("--env", choices=sorted(ENVS), help="room for a --take that predates the room choice")
    parser.add_argument("--redo", action="store_true", help="with --take: start over, including new Gemini calls")
    parser.add_argument("--retry", action="store_true", help="also pick up takes whose last attempt failed")
    parser.add_argument("--watch", type=float, metavar="SECONDS", help="keep checking for new takes this often")
    parser.add_argument("--model", default="gemini-4-hydrogen")
    parser.add_argument("--judge-model")
    parser.add_argument("--env-iters", type=int, default=3)
    parser.add_argument("--env-threshold", type=float, default=8.0)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=9.0)
    parser.add_argument("--fact", action="append", default=[], help="with --take: a user-verified fact about the motion")
    parser.add_argument("--no-wide", action="store_true", help="skip the wide room view")
    args = parser.parse_args()
    if (args.env or args.redo or args.fact) and not args.take:
        parser.error("--env, --redo and --fact apply to one --take")
    settings = dict(model=args.model, judge_model=args.judge_model, env_iters=args.env_iters,
                    env_threshold=args.env_threshold, iters=args.iters, threshold=args.threshold,
                    facts=tuple(args.fact), wide=not args.no_wide)
    try:
        blob = Blob()
        if args.list:
            for entry in scan(blob):
                print(f"{status_of(blob, entry):<40} {entry.take.video}  ({entry.size / 1e6:.1f} MB)")
            return
        if args.take:
            take = Take.parse(args.take)
            if take is None or take.ext == "json":
                parser.error(f"not a take video pathname: {args.take}")
            result = process(blob, take, env=args.env, redo=args.redo, settings=settings)
            sys.exit(0 if result["status"] == "complete" else 1)
        if not args.watch:
            done = run_pending(blob, retry=args.retry, settings=settings)
            if not done:
                print("[worker] nothing to do", flush=True)
            sys.exit(0 if all(r["status"] == "complete" for r in done) else 1)
        # Watch: one list call per check; the full scan (and any work) only when a new take appears.
        # Each take is looked at once per worker start, so a failing one isn't retried in a loop.
        seen = set()
        print(f"[worker] watching for new takes every {args.watch:g} s", flush=True)
        while True:
            try:
                fresh = [t for t in ready_takes(blob) if t.video not in seen]
                if fresh:
                    run_pending(blob, retry=args.retry, settings=settings)
                    seen.update(t.video for t in fresh)
                    print(f"[worker] watching for new takes every {args.watch:g} s", flush=True)
            except Exception as err:  # noqa: BLE001 - a demo worker must outlive network/store hiccups
                print(f"[worker] {_scrub(err)}; trying again in {args.watch:g} s", flush=True)
            time.sleep(args.watch)
    except (RuntimeError, ValueError) as err:
        parser.exit(1, f"site_worker: {err}\n")


if __name__ == "__main__":
    main()
