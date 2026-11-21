#!/usr/bin/env python3
"""
Put the official Menagerie Unitree G1 into the detailed warehouse.

    python g1_room_mujoco/Warehouse_G1_Detail/BUILD_G1.py
    python g1_room_mujoco/Warehouse_G1_Detail/BUILD_G1.py --g1-xml /path/to/unitree_g1/g1.xml
    python g1_room_mujoco/Warehouse_G1_Detail/BUILD_G1.py --spawn 0 -5.65 90   # x y yaw(deg)

Writes generated/G1_WAREHOUSE.xml plus generated/assets/ (textures, baked meshes, G1 meshes).
The default spawn matches the hackathon room (0, -5.65, facing +y), inside the ROBOT TEST ZONE.
By default the G1 collides with the simplified NAV blockers; --visual-only-collision uses
the detailed furniture colliders instead.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from envkit.g1merge import cli  # noqa: E402

if __name__ == "__main__":
    cli(HERE, nav_xml="WAREHOUSE_G1_NAV.xml", visual_xml="WAREHOUSE_VISUAL.xml", out_name="G1_WAREHOUSE.xml",
        spawn_xy=(0.0, -5.65), yaw_deg=90.0, model_name="unitree_g1_detailed_warehouse")
