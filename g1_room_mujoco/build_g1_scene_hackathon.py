#!/usr/bin/env python3
from pathlib import Path
from lxml import etree as ET
import argparse, copy, shutil
HERE=Path(__file__).resolve().parent

def build(g1xml:Path, out:Path, room_file:Path):
    g1xml=g1xml.resolve(); room_file=room_file.resolve()
    if not g1xml.exists(): raise SystemExit(f"Missing {g1xml}")
    if not (g1xml.parent/'assets').is_dir(): raise SystemExit(f"Missing robot STL meshes in {g1xml.parent/'assets'}")
    room_root=ET.parse(str(room_file)).getroot()
    robot_root=ET.parse(str(g1xml)).getroot()
    robot_root.set('model','g1_in_hackathon_room')
    compiler=robot_root.find('compiler')
    if compiler is None: compiler=ET.SubElement(robot_root,'compiler')
    compiler.set('meshdir','assets/g1'); compiler.set('autolimits','true')
    option=robot_root.find('option')
    if option is not None: option.set('timestep','0.002')
    robot_assets=robot_root.find('asset')
    for a in room_root.find('asset'): robot_assets.append(copy.deepcopy(a))
    worldbody=robot_root.find('worldbody')
    for child in room_root.find('worldbody'): worldbody.append(copy.deepcopy(child))
    robot=worldbody.find("body[@name='pelvis']")
    if robot is None: raise ValueError('No pelvis body found in supplied G1 model.')
    robot.set('pos','0 -5.65 0.793'); robot.set('quat','.7071067812 0 0 .7071067812')
    keyframes=robot_root.find('keyframe')
    if keyframes is not None:
        for key in keyframes.findall('key'):
            raw=key.get('qpos')
            if not raw: continue
            vals=raw.split()
            if len(vals)<7: continue
            vals[:7]=['0','-5.65','0.79','.7071067812','0','0','.7071067812']
            key.set('qpos',' '.join(vals))
    old_visual=robot_root.find('visual')
    if old_visual is None: robot_root.insert(0, copy.deepcopy(room_root.find('visual')))
    out.parent.mkdir(parents=True, exist_ok=True)
    mesh_dest=out.parent/'assets'/'g1'
    shutil.copytree(g1xml.parent/'assets', mesh_dest, dirs_exist_ok=True)
    screen_dest=out.parent/'assets'; screen_dest.mkdir(exist_ok=True, parents=True)
    shutil.copy2(HERE/'assets'/'stage_screen.png', screen_dest/'stage_screen.png')
    out.write_bytes(ET.tostring(robot_root, pretty_print=True, xml_declaration=True, encoding='UTF-8'))
    print(f'Created {out} from {room_file.name}')
    try:
        import mujoco
        model=mujoco.MjModel.from_xml_path(str(out))
        print(f'MuJoCo compiled: nq={model.nq}, nv={model.nv}, geoms={model.ngeom}')
    except Exception as e:
        print(f'Compilation check skipped/failed: {e}')

if __name__=='__main__':
    pa=argparse.ArgumentParser()
    pa.add_argument('--g1-xml', type=Path, required=True)
    pa.add_argument('--room-xml', type=Path, default=HERE/'room_hackathon_nav.xml')
    pa.add_argument('--output', type=Path, default=HERE/'generated'/'g1_hackathon_room.xml')
    a=pa.parse_args(); build(a.g1_xml, a.output, a.room_xml)
