"""Offline checks for the Blob worker: an in-memory store stands in for Vercel, and site_job is faked,
so there are no Gemini, network or MuJoCo calls.

    .venv/bin/python scripts/test_site_worker.py
"""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from motion_mirror import site_worker as W  # noqa: E402

TAKE = "takes/sam-ab12/kitchen/lift_a_bottle/20261121-192941-x9k7.mp4"
OLD = "takes/getup-test-zz01/sit_down/20261121-175420-420m.mp4"
NO_DETAILS = "takes/sam-ab12/warehouse/carry_a_box/20261121-200000-q1w2.mp4"


class FakeBlob:
    def __init__(self, files):
        self.files = dict(files)
        self.puts = []

    def list(self, prefix):
        return [dict(pathname=p, size=len(b)) for p, b in sorted(self.files.items()) if p.startswith(prefix)]

    def get(self, pathname, dest):
        if pathname not in self.files:
            raise RuntimeError(f"blob get {pathname}: not found")
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(self.files[pathname])

    def get_json(self, pathname):
        return json.loads(self.files[pathname])

    def put(self, pathname, path, content_type):
        assert pathname.startswith("outputs/"), pathname
        self.files[pathname] = Path(path).read_bytes()
        self.puts.append((pathname, content_type))

    def put_json(self, pathname, obj):
        assert pathname.startswith("outputs/"), pathname
        self.files[pathname] = json.dumps(obj, default=float).encode()
        self.puts.append((pathname, "application/json"))


def details(env):
    return json.dumps(dict(version=3, judge=dict(id="sam-ab12", name="Sam"),
                           environment=dict(id=env, name=env, scene="whatever.xml") if env else None,
                           activity=dict(description="Lift a bottle", slug="lift_a_bottle", example=None))).encode()


class WorkerTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        for target, value in [(W.C, "ROOT"), (W, "DOWNLOADS")]:
            p = patch.object(target, value, self.root if value == "ROOT" else self.root / "runs" / "site_downloads")
            p.start()
            self.addCleanup(p.stop)
        self.blob = FakeBlob({
            TAKE: b"video", TAKE.replace(".mp4", ".json"): details("kitchen"),
            OLD: b"old", OLD.replace(".mp4", ".json"): details(None),
            NO_DETAILS: b"pending upload",
        })
        self.calls = []

    def fake_site_job(self, video, *, name, metadata, env, resume, **settings):
        """Writes what site_job.run writes, with this computer's paths in it."""
        self.calls.append(dict(video=Path(video), name=name, env=env, resume=resume, settings=settings))
        out = W.C.ROOT / "runs" / "mirror2" / name
        (out / "environment").mkdir(parents=True, exist_ok=True)
        outputs = {}
        for key in ("robot", "comparison", "wide"):
            path = out / "environment" / f"{key}.mp4"
            path.write_bytes(f"{key} video".encode())
            outputs[key] = str(path)
        report = dict(status="complete", environment=dict(id=env), outputs=outputs,
                      source=dict(path=str(Path.home() / "secret" / "take.mp4")),
                      interpretation_engine="Gemini",
                      interpretation_stats=dict(room=env, env=dict(best_score=8.0),
                                                motion=dict(best_score=8.5), motion_rounds=[{}, {}]),
                      placement=dict(anchor=[0.0, -2.0, 90.0], clear_of_furniture=True,
                                     camera_distance=2.0, match_distance=1.8),
                      telemetry=dict(collisions=[], grasps=[dict(t=1.0, note="ok")], placements=[]),
                      validation=dict(robot=dict(duration_s=2.0, fps=30.0, width=848, height=480)),
                      job_seconds=12.0)
        (out / "job.json").write_text(json.dumps(report))
        (out / "final.json").write_text("{}")
        return out / "job.json"

    def result(self, env="kitchen", take=TAKE):
        t = W.Take.parse(take)
        return json.loads(self.blob.files[t.output_prefix(env) + "result.json"])

    def test_paths_match_the_site(self):
        t = W.Take.parse(TAKE)
        self.assertEqual((t.judge, t.env, t.what, t.stamp, t.ext),
                         ("sam-ab12", "kitchen", "lift_a_bottle", "20261121-192941-x9k7", "mp4"))
        self.assertEqual(t.output_prefix("kitchen"), "outputs/sam-ab12/kitchen/lift_a_bottle/20261121-192941-x9k7/")
        self.assertIsNone(W.Take.parse(OLD).env)
        for bad in ["takes/../x/a/20261121-192941-x9k7.mp4", "takes/sam/garden/a/20261121-192941-x9k7.mp4",
                    "outputs/sam/kitchen/a/20261121-192941-x9k7.mp4", "takes/sam/kitchen/a/20261121-192941-x9k7.exe"]:
            self.assertIsNone(W.Take.parse(bad), bad)
        long_take = W.Take("j" * 40, "warehouse", "w" * 60, "20261121-192941-x9k7", "mp4")
        self.assertLessEqual(len(long_take.run_name("warehouse")), 120)

    def test_pending_take_runs_in_its_room_and_uploads_outputs(self):
        with patch.object(W.site_job, "run", side_effect=self.fake_site_job):
            done = W.run_pending(self.blob, settings=dict(iters=2), log=lambda *_: None)
        # only the take with details and a room was processed: not the old one, not the one still uploading
        self.assertEqual(len(done), 1)
        self.assertEqual([c["env"] for c in self.calls], ["kitchen"])
        self.assertEqual(self.calls[0]["video"].read_bytes(), b"video")
        self.assertEqual(self.calls[0]["settings"], dict(iters=2))
        prefix = "outputs/sam-ab12/kitchen/lift_a_bottle/20261121-192941-x9k7/"
        for name in ("robot.mp4", "side_by_side.mp4", "wide.mp4"):
            self.assertIn(prefix + name, self.blob.files)
        self.assertEqual(self.blob.files[prefix + "side_by_side.mp4"], b"comparison video")
        # result.json said "processing" before any video went up, and "complete" last
        self.assertEqual(self.blob.puts[0], (prefix + "result.json", "application/json"))
        self.assertEqual(self.blob.puts[-1], (prefix + "result.json", "application/json"))
        res = self.result()
        self.assertEqual(res["status"], "complete")
        self.assertEqual(res["environment"]["id"], "kitchen")
        self.assertEqual(res["environment"]["scene"], W.SITE_SCENES["kitchen"])
        self.assertTrue(res["judged_in_room"])
        self.assertEqual(res["scores"], dict(environment=8.0, motion=8.5, motion_rounds=2))
        self.assertEqual(res["files"], dict(robot="robot.mp4", comparison="side_by_side.mp4", wide="wide.mp4"))
        self.assertEqual(res["activity"]["description"], "Lift a bottle")
        self.assertNotIn(str(Path.home()), self.blob.files[prefix + "result.json"].decode())
        self.assertFalse(any(p.startswith("takes/") for p, _ in self.blob.puts))

        # a second pass leaves finished takes alone
        with patch.object(W.site_job, "run", side_effect=self.fake_site_job):
            self.assertEqual(W.run_pending(self.blob, log=lambda *_: None), [])
        self.assertEqual(len(self.calls), 1)

    def test_room_is_never_guessed_or_changed(self):
        with patch.object(W.site_job, "run", side_effect=self.fake_site_job):
            with self.assertRaises(ValueError):          # old take, no room given
                W.process(self.blob, W.Take.parse(OLD), log=lambda *_: None)
            with self.assertRaises(ValueError):          # filmed in the kitchen, asked for the warehouse
                W.process(self.blob, W.Take.parse(TAKE), env="warehouse", log=lambda *_: None)
            self.assertEqual(self.calls, [])
            res = W.process(self.blob, W.Take.parse(OLD), env="hackathon", log=lambda *_: None)
        self.assertEqual(res["status"], "complete")
        self.assertEqual(self.calls[0]["env"], "hackathon")
        self.assertIn("outputs/getup-test-zz01/hackathon/sit_down/20261121-175420-420m/robot.mp4", self.blob.files)

    def test_details_that_disagree_with_the_path_fail_before_any_run(self):
        self.blob.files[TAKE.replace(".mp4", ".json")] = details("warehouse")
        with patch.object(W.site_job, "run", side_effect=self.fake_site_job):
            res = W.process(self.blob, W.Take.parse(TAKE), log=lambda *_: None)
        self.assertEqual(res["status"], "failed")
        self.assertIn("warehouse", res["error"])
        self.assertEqual(self.calls, [])
        self.assertEqual(self.result()["status"], "failed")

    def test_failure_is_recorded_and_retry_reuses_finished_interpretation(self):
        def crash(video, *, name, **kwargs):
            out = W.C.ROOT / "runs" / "mirror2" / name
            out.mkdir(parents=True)
            (out / "final.json").write_text("{}")         # Gemini finished, then rendering died
            raise RuntimeError(f"renderer crashed in {W.REPO}/motion_mirror")

        with patch.object(W.site_job, "run", side_effect=crash):
            res = W.run_pending(self.blob, log=lambda *_: None)[0]
        self.assertEqual(res["status"], "failed")
        self.assertIn("<repo>/motion_mirror", res["error"])
        self.assertNotIn(str(W.REPO), res["error"])
        with patch.object(W.site_job, "run", side_effect=self.fake_site_job):
            self.assertEqual(W.run_pending(self.blob, log=lambda *_: None), [])   # failed ones wait for --retry
            res = W.run_pending(self.blob, retry=True, log=lambda *_: None)[0]
        self.assertEqual(res["status"], "complete")
        self.assertTrue(self.calls[0]["resume"])          # no new Gemini calls for a finished interpretation

    def test_redo_and_unfinished_runs_start_over(self):
        take = W.Take.parse(TAKE)
        run_dir = W.C.ROOT / "runs" / "mirror2" / take.run_name("kitchen")
        run_dir.mkdir(parents=True)
        (run_dir / "half.txt").write_text("unfinished")
        with patch.object(W.site_job, "run", side_effect=self.fake_site_job):
            W.process(self.blob, take, log=lambda *_: None)
            self.assertFalse(self.calls[-1]["resume"])
            self.assertFalse((run_dir / "half.txt").exists())
            W.process(self.blob, take, redo=True, log=lambda *_: None)
            self.assertFalse(self.calls[-1]["resume"])
            W.process(self.blob, take, log=lambda *_: None)
            self.assertTrue(self.calls[-1]["resume"])

    def test_interrupt_marks_failed_and_propagates(self):
        with patch.object(W.site_job, "run", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                W.process(self.blob, W.Take.parse(TAKE), log=lambda *_: None)
        self.assertEqual(self.result()["status"], "failed")

    def test_env_file_parsing(self):
        f = self.root / ".env.local"
        f.write_text('# Created by Vercel CLI\nBLOB_READ_WRITE_TOKEN="vercel_blob_rw_abc"\nexport BLOB_ACCESS=private\n')
        self.assertEqual(W._env_file(f), dict(BLOB_READ_WRITE_TOKEN="vercel_blob_rw_abc", BLOB_ACCESS="private"))


if __name__ == "__main__":
    unittest.main()
