#!/usr/bin/env python3
"""
Put the official Menagerie Unitree G1 into the detailed kitchen.

    python g1_room_mujoco/Kitchen_G1_Detail/BUILD_G1.py
    python g1_room_mujoco/Kitchen_G1_Detail/BUILD_G1.py --g1-xml /path/to/unitree_g1/g1.xml
    python g1_room_mujoco/Kitchen_G1_Detail/BUILD_G1.py --spawn -1.35 -2.2 90   # x y yaw(deg)

Writes generated/G1_KITCHEN.xml plus generated/assets/ (textures + G1 meshes).
By default the G1 collides with the simplified NAV blockers; --visual-only-collision uses
the detailed furniture colliders instead.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from envkit.g1merge import cli  # noqa: E402

if __name__ == "__main__":
    cli(HERE, nav_xml="KITCHEN_G1_NAV.xml", visual_xml="KITCHEN_VISUAL.xml", out_name="G1_KITCHEN.xml",
        spawn_xy=(-1.35, -2.2), yaw_deg=90.0, model_name="unitree_g1_detailed_kitchen")
