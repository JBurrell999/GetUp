"""Offline worker orchestration checks; no Gemini or Blob calls.

    .venv/bin/python scripts/test_site_job.py
"""
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from motion_mirror import site_job  # noqa: E402


class SiteJobTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.video = self.root / "source.mp4"
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*"mp4v"), 4, (64, 64))
        for index in range(4):
            writer.write(np.full((64, 64, 3), index * 60, dtype=np.uint8))
        writer.release()
        self.meta = self.root / "metadata.json"
        self.meta.write_text(json.dumps(dict(environment=dict(id="kitchen", scene="/untrusted/path.xml"),
                                             activity=dict(description="Lift a bottle"))))
        self.root_patch = patch.object(site_job.C, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def fake_mirror(self, video, name, *args, **kwargs):
        self.assertFalse(kwargs["bench"])
        self.assertEqual(kwargs.get("room", "kitchen"), "kitchen")   # judged in the upload's room
        out = self.root / "runs" / "mirror2" / name
        out.mkdir(parents=True)
        (out / "final.json").write_text('{"duration_s": 1, "keyframes": [{"t": 0}]}')
        (out / "summary.json").write_text('{"model": "offline-test", "motion": {"best_score": 8}}')

    def fake_render(self, interp, env, video, out, *, wide):
        self.assertTrue(Path(interp).is_file())
        self.assertEqual(env, "kitchen")
        out.mkdir(parents=True, exist_ok=True)
        for filename in ["robot_env.mp4", "side_by_side_env.mp4"] + (["robot_wide.mp4"] if wide else []):
            shutil.copyfile(video, out / filename)
        placement = dict(env=env, clear_of_furniture=True)
        telemetry = dict(collisions=[])
        (out / "place.json").write_text(json.dumps(placement))
        (out / "telemetry_env.json").write_text(json.dumps(telemetry))
        return placement, telemetry

    def test_end_to_end_routing_and_decoded_artifacts(self):
        with patch.object(site_job.mirror, "run", side_effect=self.fake_mirror) as mirror_run, \
                patch.object(site_job, "render_in_env", side_effect=self.fake_render):
            report_path = site_job.run(self.video, name="sample", metadata=self.meta)
        report = json.loads(report_path.read_text())
        mirror_run.assert_called_once()
        self.assertEqual(mirror_run.call_args.kwargs["room"], "kitchen")
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["mode"], "kinematic")
        self.assertEqual(report["source"]["sha256"], hashlib.sha256(self.video.read_bytes()).hexdigest())
        self.assertEqual(report["environment"]["id"], "kitchen")
        self.assertNotEqual(report["environment"]["room_xml"], "/untrusted/path.xml")
        self.assertEqual(report["validation"]["robot"]["decoded_frames"], 4)
        self.assertTrue(report["validation"]["robot"]["has_visual_change"])
        self.assertIn("wide", report["outputs"])

    def test_existing_run_is_preserved(self):
        out = self.root / "runs" / "mirror2" / "existing"
        out.mkdir(parents=True)
        sentinel = out / "keep.txt"
        sentinel.write_text("preserve me")
        with patch.object(site_job.mirror, "run") as mirror_run, self.assertRaises(FileExistsError):
            site_job.run(self.video, name="existing", metadata=self.meta)
        mirror_run.assert_not_called()
        self.assertEqual(sentinel.read_text(), "preserve me")

    def test_invalid_routing_and_traversal_never_call_model(self):
        with patch.object(site_job.mirror, "run") as mirror_run:
            for kwargs in [dict(name="../outside", env="kitchen"), dict(name="ok", env="unknown"),
                           dict(name="ok", env="kitchen", iters=0), dict(name="ok", env="kitchen", resume=True),
                           dict(name="no_room")]:   # no metadata and no --env: refused, never guessed
                with self.assertRaises((ValueError, FileNotFoundError)):
                    site_job.run(self.video, **kwargs)
        mirror_run.assert_not_called()

    def test_resume_renders_without_model_and_checks_source(self):
        with patch.object(site_job.mirror, "run", side_effect=self.fake_mirror), \
                patch.object(site_job, "render_in_env", side_effect=self.fake_render):
            site_job.run(self.video, name="sample", metadata=self.meta, wide=False)
        with patch.object(site_job.mirror, "run") as mirror_run, \
                patch.object(site_job, "render_in_env", side_effect=self.fake_render):
            report = site_job.run(self.video, name="sample", metadata=self.meta, resume=True, wide=False)
            self.assertTrue(json.loads(report.read_text())["resumed_interpretation"])
            with self.assertRaises(ValueError):
                site_job.run(self.video, name="sample", env="hackathon", resume=True)
            with self.video.open("ab") as changed:
                changed.write(b"different source")
            with self.assertRaises(ValueError):
                site_job.run(self.video, name="sample", metadata=self.meta, resume=True)
        mirror_run.assert_not_called()

    def test_resume_preserves_local_extraction_provenance(self):
        self.fake_mirror(str(self.video), "local", bench=False)
        stats = dict(interpretation_engine="MediaPipe local pose", assumptions=["Root translation is approximate"],
                     tracked_frames=4)
        summary = self.root / "runs" / "mirror2" / "local" / "summary.json"
        summary.write_text(json.dumps(stats))
        with patch.object(site_job.mirror, "run") as mirror_run, \
                patch.object(site_job, "render_in_env", side_effect=self.fake_render):
            report_path = site_job.run(self.video, name="local", metadata=self.meta, resume=True)
        report = json.loads(report_path.read_text())
        self.assertEqual(report["interpretation_engine"], "MediaPipe local pose")
        self.assertEqual(report["interpretation_stats"], stats)
        mirror_run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
