"""Evaluate a checkpoint deterministically (no exploration noise), optionally render a video.

uv run scripts/evaluate.py runs/general/latest.pt --episodes 5
uv run scripts/evaluate.py runs/general/latest.pt --episodes 1 --video out.mp4 [--clip clip.npy] [--no-randomize]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import argparse

import numpy as np
import torch

from motion_mirror import config as C
from motion_mirror.g1_env import OBS_DIM, PRIV_DIM, G1Env
from motion_mirror.ppo import ActorCritic
from motion_mirror.reference import load_clip


def load_policy(path):
    ck = torch.load(path, weights_only=False)
    ac = ActorCritic(OBS_DIM, PRIV_DIM, C.NJ)
    ac.load_state_dict(ck["model"])
    ac.eval()
    return ac, ck


def run(ac, env, steps, renderer=None, cam=None, frames=None):
    obs, _ = env.reset()
    je, he, n, fell = [], [], 0, False
    for _ in range(steps):
        a = ac.act_deterministic(torch.from_numpy(obs)[None])[0].numpy()
        (obs, _), _, fell, trunc, info = env.step(a)
        je.append(info["joint_err"]); he.append(info["hand_err"]); n += 1
        if renderer is not None:
            cam.lookat[:] = env.d.qpos[:3] + np.array([0, 0, 0.1])
            renderer.update_scene(env.d, cam)
            frames.append(renderer.render())
        if fell or trunc:
            break
    return dict(len=n, fell=fell, joint_err=float(np.mean(je)), hand_err=float(np.mean(he)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--clip", default=None)
    ap.add_argument("--no-randomize", action="store_true")
    ap.add_argument("--amp", type=float, default=1.0)
    ap.add_argument("--video", default=None)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    ac, ck = load_policy(args.ckpt)
    clips = [load_clip(args.clip)] if args.clip else []
    env = G1Env(seed=args.seed, clips=clips, clip_prob=1.0 if clips else 0.0, randomize=not args.no_randomize)
    env.gen_amplitude = args.amp
    renderer = cam = frames = None
    if args.video:
        import mujoco
        renderer = mujoco.Renderer(env.m, 480, 640)
        cam = mujoco.MjvCamera()
        cam.distance, cam.azimuth, cam.elevation = 2.8, 150, -15
        frames = []
    res = [run(ac, env, args.steps, renderer, cam, frames) for _ in range(args.episodes)]
    print(f"ckpt iter {ck['iter']} ({ck['env_steps']/1e6:.1f}M steps)")
    for r in res:
        print(f"  len {r['len']:4d} fell {r['fell']!s:5} joint_err {r['joint_err']:.3f} rad  hand_err {r['hand_err']*100:.1f} cm")
    print(f"  mean: survive {np.mean([not r['fell'] for r in res]):.0%}  joint_err {np.mean([r['joint_err'] for r in res]):.3f}"
          f"  hand_err {np.mean([r['hand_err'] for r in res])*100:.1f} cm")
    if args.video:
        import imageio
        imageio.mimsave(args.video, frames, fps=int(1 / C.CONTROL_DT))
        print("wrote", args.video)


if __name__ == "__main__":
    main()
