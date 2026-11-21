"""Offline room and object carry regressions; no renderer or Gemini calls.

Run from motion_mirror: .venv/bin/python scripts/test_in_env.py
"""
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from motion_mirror import config as C  # noqa: E402
from motion_mirror.in_env import EnvReplay  # noqa: E402
from motion_mirror.replay import Replay, keyframe_to_pose  # noqa: E402


NEUTRAL = dict(
    environment={"surfaces": []}, object=None, camera={}, duration_s=0.2,
    keyframes=[dict(t=0.0,
                    left_arm={"shoulder_flexion_deg": 5, "shoulder_abduction_deg": 12},
                    right_arm={"shoulder_flexion_deg": 5, "shoulder_abduction_deg": 12})],
)


class RoomReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clear = EnvReplay(NEUTRAL, dict(env="hackathon", anchor=(0.0, -4.2, 90)), 64, 64)

    def test_nested_room_geometry_is_environment(self):
        replay = self.clear
        m = replay.m
        furniture = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "furniture")
        self.assertGreater(furniture, 0)
        room_bodies = {furniture}
        for body in range(1, m.nbody):
            if m.body_parentid[body] in room_bodies:
                room_bodies.add(body)
        self.assertGreater(len(room_bodies), 1, "fixture must include nested room props")
        self.assertTrue(room_bodies.isdisjoint(replay.robot_bodies))
        room_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in room_bodies}
        self.assertGreater(len(room_geoms), 100)
        self.assertTrue(room_geoms <= replay.env_names.keys())
        self.assertTrue(all(a in replay.robot_bodies and b in replay.robot_bodies
                            for a, b in replay.self_ok))

    def test_clear_room_pose_has_no_collisions(self):
        replay = self.clear
        replay._set(*keyframe_to_pose(NEUTRAL["keyframes"][0]))
        self.assertEqual(replay._collisions(0, False, False), [])

    def test_robot_penetrating_nested_wall_is_reported(self):
        # Deliberately anchor the robot inside the east wall, bypassing placement search.
        # The wall lives under furniture, and must not be whitelisted as robot self-contact.
        replay = EnvReplay(NEUTRAL, dict(env="hackathon", anchor=(6.13, 0.0, 0)), 64, 64)
        replay._set(*keyframe_to_pose(NEUTRAL["keyframes"][0]))
        wall = mujoco.mj_name2id(replay.m, mujoco.mjtObj.mjOBJ_GEOM, "east_wall")
        self.assertGreater(replay.m.geom_bodyid[wall], 0)
        hits = replay._collisions(0, False, False)
        wall_hits = [(parts, depth) for parts, depth in hits if "east_wall" in parts]
        self.assertTrue(wall_hits, f"robot/wall penetration was omitted: {hits}")
        self.assertTrue(any(depth > 0.01 for _, depth in wall_hits))
        environment = set(replay.env_names.values())
        self.assertTrue(all(sum(part in environment for part in parts) == 1
                            for parts, _ in wall_hits))

    def test_concurrent_replays_use_distinct_cleaned_up_xml(self):
        make_temp = tempfile.NamedTemporaryFile
        paths = []
        barrier = threading.Barrier(2)
        shared = C.SCENE_XML.parent / "_replay_scene.xml"
        before = shared.read_bytes() if shared.exists() else None

        def tracked_temp(*args, **kwargs):
            handle = make_temp(*args, **kwargs)
            paths.append(Path(handle.name))
            try:
                barrier.wait(timeout=10)
            except BaseException:
                handle.close()
                raise
            return handle

        def compile_box(depth):
            interp = copy.deepcopy(NEUTRAL)
            interp["object"] = dict(type="box", size_m=[depth, 0.2, 0.3])
            replay = Replay(interp, 64, 64)
            geom = min(replay.obj_geoms)
            return replay.m.geom_size[geom].copy()

        with patch("motion_mirror.replay.tempfile.NamedTemporaryFile", side_effect=tracked_temp):
            with ThreadPoolExecutor(max_workers=2) as pool:
                sizes = list(pool.map(compile_box, (0.1, 0.4)))
        np.testing.assert_allclose(sizes, [[0.05, 0.1, 0.15], [0.2, 0.1, 0.15]])
        self.assertEqual(len(paths), 2)
        self.assertEqual(len(set(paths)), 2)
        self.assertTrue(all(not path.exists() for path in paths))
        self.assertEqual(shared.read_bytes() if shared.exists() else None, before)


class ObjectCarryTests(unittest.TestCase):
    def test_one_hand_upright_carry_is_opt_in(self):
        fixture = json.loads((ROOT / "benchmarks/bottle_v2_score8.5.json").read_text())
        for upright in (False, True):
            with self.subTest(keep_upright=upright):
                interp = copy.deepcopy(fixture)
                # Stop during the right-hand grasp, before release resets the bottle upright.
                interp["duration_s"] = 3.5
                self.assertEqual(interp["contacts"][0]["hand"], "right")
                if upright:
                    interp["object"]["keep_upright"] = True
                replay = Replay(interp, 64, 64)
                telemetry = replay.run(fps=15, video_path=None)
                self.assertTrue(telemetry["samples"][-1]["held"])
                start = interp["object"]["start"]
                start_pos = np.array([start["forward_m"], start["left_m"], start["on_surface_height_m"]])
                final_pos = replay.d.mocap_pos[replay.obj_mocap]
                self.assertGreater(np.linalg.norm(final_pos - start_pos), 0.05,
                                   "upright carry must still move the grasped object")
                rotation = np.zeros(9)
                mujoco.mju_quat2Mat(rotation, replay.d.mocap_quat[replay.obj_mocap])
                axis = rotation.reshape(3, 3)[:, 2]
                if upright:
                    np.testing.assert_allclose(axis, [0, 0, 1], atol=1e-10)
                else:
                    self.assertLess(axis[2], 0.95, "default drinking motion must retain wrist tilt")


if __name__ == "__main__":
    unittest.main(verbosity=2)
