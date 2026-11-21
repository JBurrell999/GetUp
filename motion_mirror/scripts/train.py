"""Train (or fine-tune) the G1 motion-tracking policy.

Long general run:   uv run scripts/train.py --name general --iters 3000
Fine-tune on clip:  uv run scripts/train.py --name judge --init runs/general/latest.pt --clips clip.npy --iters 100
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from motion_mirror import config as C
from motion_mirror.g1_env import OBS_DIM, PRIV_DIM
from motion_mirror.skill_env import SKILL_OBS_DIM, SKILL_PRIV_DIM
from motion_mirror.ppo import PPO, ActorCritic
from motion_mirror.reference import load_clip
from motion_mirror.vec_env import VecEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="general")
    ap.add_argument("--env", choices=["gesture", "skill"], default="gesture")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--envs-per-worker", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=48)
    ap.add_argument("--init", default=None, help="checkpoint to start from")
    ap.add_argument("--clips", nargs="*", default=[], help="retargeted .npy clips to mix in")
    ap.add_argument("--clip-prob", type=float, default=0.7)
    ap.add_argument("--curriculum-iters", type=int, default=600, help="iters to ramp gesture size + pushes to full")
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--leg-std", type=float, default=None, help="reset exploration std of the 12 leg actions")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(3)
    run = C.ROOT / "runs" / args.name
    (run / "ckpts").mkdir(parents=True, exist_ok=True)
    (run / "args.json").write_text(json.dumps(vars(args), indent=2))

    clips = [load_clip(p) for p in args.clips]
    obs_dim, priv_dim = (SKILL_OBS_DIM, SKILL_PRIV_DIM) if args.env == "skill" else (OBS_DIM, PRIV_DIM)
    venv = VecEnv(args.workers, args.envs_per_worker, seed=args.seed * 100, env_name=args.env, clips=clips,
                  clip_prob=args.clip_prob if clips else 0.0)
    ac = ActorCritic(obs_dim, priv_dim, C.NJ)
    start_it = 0
    if args.init:
        ck = torch.load(args.init, weights_only=False)
        if ck.get("env", "gesture") == args.env:
            ac.load_state_dict(ck["model"])
            ac.log_std.data.clamp_(max=np.log(0.5))  # fine-tunes start from a confident policy
            args.curriculum_iters = 0
        else:
            warm_start(ac, ck["model"])
            print(f"warm-started {args.env} policy from {args.init} ({ck.get('env', 'gesture')})")
    if args.leg_std:
        with torch.no_grad():
            ac.log_std[C.LEG] = float(np.log(args.leg_std))
    ppo = PPO(ac)

    N, T = venv.num_envs, args.horizon
    obs_np, priv_np = venv.reset()
    log_f = open(run / "progress.csv", "a", newline="")
    log = csv.writer(log_f)
    if (run / "progress.csv").stat().st_size == 0:
        log.writerow(["iter", "env_steps", "wall_s", "ep_return", "ep_len", "fall_rate", "joint_err", "hand_err",
                      "vel_err", "height_err", "amplitude", "push_scale", "std", "lr", "kl", "fps"])
    t0, env_steps, recent = time.time(), 0, []

    for it in range(start_it, start_it + args.iters):
        ramp = 1.0 if args.curriculum_iters == 0 else min(1.0, it / args.curriculum_iters)
        amp, push = 0.3 + 0.7 * ramp, ramp
        venv.set_attr("push_scale", push)
        venv.set_attr("gen_amplitude", amp)
        if args.env == "skill":
            venv.set_attr("cmd_scale", 0.4 + 0.6 * ramp)

        buf = {k: [] for k in ("obs", "priv", "act", "logp", "val", "rew", "done")}
        errs = []
        ti = time.time()
        for _ in range(T):
            obs, priv = torch.from_numpy(obs_np), torch.from_numpy(priv_np)
            with torch.no_grad():
                d = ac.dist(obs)
                a = d.sample()
                logp, v = d.log_prob(a).sum(-1), ac.value(priv)
            obs_np, priv_np, rew, done, trunc, term_priv, finished, e = venv.step(a.numpy())
            rew = torch.from_numpy(rew).float()
            if trunc.any():  # bootstrap timeouts instead of treating them as failures
                with torch.no_grad():
                    rew += ppo.gamma * torch.from_numpy(trunc).float() * ac.value(torch.from_numpy(term_priv))
            for k, x in zip(buf, (obs, priv, a, logp, v, rew, torch.from_numpy(done).float())):
                buf[k].append(x)
            recent.extend(finished)
            errs.append(e)
        with torch.no_grad():
            last_v = ac.value(torch.from_numpy(priv_np))
        B = {k: torch.stack(x) for k, x in buf.items()}
        adv, ret = ppo.gae(B["rew"], B["done"], B["val"], last_v)
        ac.obs_norm.update(B["obs"].flatten(0, 1))
        ac.priv_norm.update(B["priv"].flatten(0, 1))
        flat = {k: B[k].flatten(0, 1) for k in ("obs", "priv", "act", "logp", "val")}
        flat.update(adv=adv.flatten(), ret=ret.flatten())
        stats = ppo.update(flat)

        env_steps += N * T
        fps = N * T / (time.time() - ti)
        recent = recent[-200:]
        ep_ret = np.mean([r for r, _, _ in recent]) if recent else float("nan")
        ep_len = np.mean([l for _, l, _ in recent]) if recent else float("nan")
        fall = np.mean([f for _, _, f in recent]) if recent else float("nan")
        je, he, ve, hte = np.mean(errs, 0)
        log.writerow([it, env_steps, round(time.time() - t0, 1), ep_ret, ep_len, fall, je, he, ve, hte, amp, push,
                      stats["std"], stats["lr"], stats["kl"], fps])
        log_f.flush()
        if it % 5 == 0:
            print(f"it {it:5d} | steps {env_steps/1e6:6.2f}M | ret {ep_ret:7.1f} len {ep_len:6.1f} fall {fall:.2f} | "
                  f"joint_err {je:.3f} rad hand_err {he*100:.1f} cm vel_err {ve:.2f} m/s h_err {hte*100:.1f} cm | amp {amp:.2f} push {push:.2f} | "
                  f"std {stats['std']:.2f} kl {stats['kl']:.4f} | {fps:.0f} sps", flush=True)
        if it % args.save_every == 0 or it == start_it + args.iters - 1:
            ck = dict(model=ac.state_dict(), iter=it, env_steps=env_steps, wall=time.time() - t0,
                      obs_dim=obs_dim, priv_dim=priv_dim, env=args.env, amp=amp, push=push)
            torch.save(ck, run / "ckpts" / f"it{it:05d}.pt")
            torch.save(ck, run / "latest.pt")
    venv.close()


def warm_start(ac, sd):
    """Copy weights from a policy with fewer inputs; new input columns start at zero."""
    own = ac.state_dict()
    for k, v in sd.items():
        t = own[k]
        if t.shape == v.shape:
            t.copy_(v)
        elif t.dim() == 2 and t.shape[0] == v.shape[0]:      # first Linear layer: extra input columns
            t.zero_()
            t[:, :v.shape[1]] = v
        elif t.dim() == 1 and k.endswith(("mean", "var")):   # normalizer stats
            t[:v.shape[0]] = v
    ac.log_std.data.fill_(np.log(0.6))


if __name__ == "__main__":
    main()
