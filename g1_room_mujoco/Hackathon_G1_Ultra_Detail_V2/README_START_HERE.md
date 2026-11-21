# High-detail hackathon environment for Unitree G1 / MuJoCo 4.x

This package is a **photo-inspired approximate reconstruction**, not a laser-scanned digital twin. All attendees are removed. The stage, dark wood refreshment bar, worktables, red chairs, structural columns and open robot aisle are preserved.

## Fastest way to see the upgraded room

1. Extract the entire ZIP, keeping the `assets` folder next to the XML files.
2. **Drag `HACKATHON_V2_VISUAL.xml` from Finder onto the open blue MuJoCo viewport**.
3. Select an eye-level camera in MuJoCo's Rendering > Camera menu: `v2_from_entrance`, `v2_robot_aisle`, `v2_stage_facing_audience`, `v2_side_wide`, or `v2_refreshment_walkthrough`.
4. `v2_clean_topdown` is the unobstructed overhead camera. The overhead grid was completely removed.

This file is the environment alone, without the G1 robot.

## Real G1 inside the detailed room (Mac, your existing setup)

**Recommended for your existing GetUp project:** Place the extracted `Hackathon_G1_Ultra_Detail_V2` folder inside your existing `g1_room_mujoco` folder. From `/Users/jjburrell/GetUp/GetUp`, run:

```bash
python g1_room_mujoco/Hackathon_G1_Ultra_Detail_V2/BUILD_G1.py \
  --g1-xml /Users/jjburrell/GetUp/GetUp/models/mujoco_menagerie/unitree_g1/g1.xml
open g1_room_mujoco/Hackathon_G1_Ultra_Detail_V2/generated
```

Alternatively, if you unzip it directly into Downloads:

```bash
python ~/Downloads/Hackathon_G1_Ultra_Detail_V2/BUILD_G1.py \
  --g1-xml /Users/jjburrell/GetUp/GetUp/models/mujoco_menagerie/unitree_g1/g1.xml
open ~/Downloads/Hackathon_G1_Ultra_Detail_V2/generated
```

Drag **`generated/G1_HACKATHON_V2.xml`** from Finder onto your existing MuJoCo window. This generated folder includes the G1 STL meshes and visual texture files in `generated/assets`; keep them beside the XML.

The G1 starts near `(0,-5.65)` facing the stage. **Do not press Run expecting it to balance or walk**—you still need your G1 stabilization/locomotion controller.

## Files

- `HACKATHON_V2_VISUAL.xml`: detailed scene with original solid furniture collision shapes and non-colliding microprops
- `HACKATHON_V2_G1_NAV.xml`: identical render geometry, simpler invisible table/chair proxy blockers for G1 path planning. Does not use invisible proxies for walls, pillars, stage (their real colliders are retained)
- `BUILD_G1.py`: merges real Menagerie G1 and all relevant meshes/textures with G1 navigation scene
- `assets/*.png`: photo-inspired synthetic surfaces, stage QR-like resource graphic (decorative, not real QR links), laptop screen UIs, signage, notebook, keyboard
- `approx_floorplan.png`: earlier approximate layout illustration; not an updated screenshot or exact scan

## Visual upgrades

- No obstructing overhead ceiling grid.
- Higher-detail wood, stone, cloth, chair upholstery and dark marble textures.
- 15 desktop/table laptops (varied IDE, terminal, simulation telemetry, dashboard screens); keyboards, touchpads, screens and webcams are geometry, not a flat icon.
- Coffee cups with lids/sleeves, soda and energy cans, translucent water bottles, snack bags, open notebooks with diagrams, phones, power strips, cords, stickers, badges, backpacks.
- More stocked bar with coffee urns, cups, beverages and fruit bowls; wall sconces, stage spots and decorative trim.
- 6 extra camera viewpoints including unobstructed top down and 5 eye-level angles.

## Simulation fidelity / important limitations

- The reconstruction's walls and dimensions are visual **estimates from photos**, not measured dimensions or a geometry survey.
- Small visual clutter is intentionally non-colliding for stable locomotion testing; floor backpacks have simplified invisible collision proxies in G1 nav. Table props are not suitable for grasping/manipulation contact until dedicated colliders/dynamics are added.
- G1 and simplified large obstacle collision geometry are preserved; **robot-specific clearance and locomotion are not validated** by image inspection or XML parsing.
- MuJoCo is a physics-first renderer: this is richer procedural geometry and textures, *not true photorealism*. For film-quality images, use Blender/Unreal alongside the physics model.
- You can toggle geometry group 3 (microprops) / 4 (invisible collision proxies) in the MuJoCo viewer for debugging.
- The only G1 model source used is the official local copy already on your Mac; this ZIP does not redistribute third-party G1 meshes.

## If a texture can't be opened

Keep the `assets` folder directly beside the source XML, or beside the generated XML under `generated/assets`. Drag the XML *from that folder*. Do not copy the XML by itself.
