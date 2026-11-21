#!/usr/bin/env python3
"""Merge official Menagerie Unitree G1 with the high-detail hackathon scene.
Run from anywhere. Generates self-contained generated/G1_HACKATHON_V2.xml & textures/meshes.
"""
from pathlib import Path
import argparse, copy, shutil
from lxml import etree as E

HERE=Path(__file__).resolve().parent
DEFAULT_G1=Path.home()/"GetUp/GetUp/models/mujoco_menagerie/unitree_g1/g1.xml"

def main():
    pa=argparse.ArgumentParser()
    pa.add_argument('--g1-xml',type=Path,default=DEFAULT_G1,help='Path to official G1 g1.xml')
    pa.add_argument('--visual-only-collision',action='store_true',help='Use detailed visual scene colliders rather than simplified nav proxy obstacles')
    args=pa.parse_args()
    model_path=args.g1_xml.expanduser().resolve()
    if not model_path.is_file():
        raise SystemExit(f'G1 XML missing: {model_path}\nUse --g1-xml /absolute/path/to/g1.xml')
    source_assets=model_path.parent/'assets'
    if not source_assets.is_dir():raise SystemExit(f'G1 mesh assets missing: {source_assets}')
    source=HERE/('HACKATHON_V2_VISUAL.xml' if args.visual_only_collision else 'HACKATHON_V2_G1_NAV.xml')
    room=E.parse(str(source)).getroot()
    robot=E.parse(str(model_path)).getroot()
    robot.set('model','unitree_g1_hyper_detailed_hackathon')
    compiler=robot.find('compiler')
    if compiler is None: compiler=E.SubElement(robot,'compiler')
    compiler.set('meshdir','assets/g1')
    compiler.set('autolimits','true')
    robot_assets=robot.find('asset')
    if robot_assets is None:robot_assets=E.SubElement(robot,'asset')
    for a in room.find('asset'):robot_assets.append(copy.deepcopy(a))
    scene=robot.find('worldbody')
    for child in room.find('worldbody'):scene.append(copy.deepcopy(child))
    pelvis=scene.find("body[@name='pelvis']")
    if pelvis is None:raise SystemExit('Unexpected G1 model: missing pelvis body')
    pelvis.set('pos','0 -5.65 0.793')
    pelvis.set('quat','.7071067812 0 0 .7071067812')
    # stand keyframe initial floating base
    keys=robot.find('keyframe')
    if keys is not None:
        for key in keys.findall('key'):
            qpos=key.get('qpos')
            if not qpos:continue
            v=qpos.split()
            if len(v)<7:continue
            v[:7]=['0','-5.65','0.793','.7071067812','0','0','.7071067812']
            key.set('qpos',' '.join(v))
    opt=robot.find('option')
    if opt is not None: opt.set('timestep','0.002')
    visual=robot.find('visual')
    src_v=room.find('visual')
    if visual is None and src_v is not None:
        robot.insert(0,copy.deepcopy(src_v))
    elif visual is not None and src_v is not None:
        g=visual.find('global'); sg=src_v.find('global')
        if g is not None and sg is not None:
            for k in ['azimuth','elevation']:
                if sg.get(k):g.set(k,sg.get(k))
    stat=robot.find('statistic'); sr=room.find('statistic')
    if stat is None and sr is not None:robot.append(copy.deepcopy(sr))
    elif stat is not None and sr is not None:
        stat.set('center',sr.get('center'));stat.set('extent',sr.get('extent'))
    folder=HERE/'generated'; folder.mkdir(parents=True,exist_ok=True)
    assets=folder/'assets'; assets.mkdir(parents=True,exist_ok=True)
    shutil.copytree(source_assets,assets/'g1',dirs_exist_ok=True)
    for p in (HERE/'assets').iterdir():
        if p.is_file():shutil.copy2(p,assets/p.name)
    target=folder/'G1_HACKATHON_V2.xml'
    E.ElementTree(robot).write(str(target),encoding='utf-8',xml_declaration=True,pretty_print=True)
    print(f'Created: {target}')
    print(f'Copied: G1 meshes and {len(list((HERE/"assets").iterdir()))} texture files')
    try:
        import mujoco
        m=mujoco.MjModel.from_xml_path(str(target))
        print(f'COMPILATION OK: nq={m.nq}, nv={m.nv}, geoms={m.ngeom}, cameras={m.ncam}')
    except ImportError:
        print('MuJoCo not installed in this Python environment; compilation not checked here')
    except Exception as e:
        raise SystemExit(f'MuJoCo compilation error: {e}')
    print('Open Finder and drag generated/G1_HACKATHON_V2.xml into MuJoCo.')

if __name__=='__main__':main()
