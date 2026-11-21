"""Runs many G1Envs across CPU processes, stepping them in lockstep."""
import multiprocessing as mp

import numpy as np


def _worker(remote, seed, n_envs, env_kwargs, env_name="gesture"):
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from .g1_env import G1Env
    from .skill_env import SkillEnv
    Env = SkillEnv if env_name == "skill" else G1Env

    envs = [Env(seed=seed * 1000 + i, **env_kwargs) for i in range(n_envs)]
    ep_ret = np.zeros(n_envs)
    ep_len = np.zeros(n_envs, int)
    while True:
        cmd, data = remote.recv()
        if cmd == "reset":
            out = [e.reset() for e in envs]
            remote.send((np.stack([o for o, _ in out]), np.stack([p for _, p in out])))
        elif cmd == "step":
            obs, priv, rew, done, trunc = [], [], np.zeros(n_envs), np.zeros(n_envs, bool), np.zeros(n_envs, bool)
            term_priv = np.zeros((n_envs, envs[0]._obs()[1].shape[0]), np.float32)
            finished, errs = [], []
            for i, e in enumerate(envs):
                (o, p), r, fell, tr, info = e.step(data[i])
                rew[i], done[i], trunc[i] = r, fell or tr, tr
                ep_ret[i] += r
                ep_len[i] += 1
                errs.append((info["joint_err"], info["hand_err"], info.get("vel_err", 0.0), info.get("height_err", 0.0)))
                if fell or tr:
                    term_priv[i] = p
                    finished.append((ep_ret[i], ep_len[i], fell))
                    ep_ret[i], ep_len[i] = 0, 0
                    o, p = e.reset()
                obs.append(o)
                priv.append(p)
            remote.send((np.stack(obs), np.stack(priv), rew, done, trunc, term_priv, finished, np.mean(errs, 0)))
        elif cmd == "set":
            for e in envs:
                setattr(e, data[0], data[1])
            remote.send(None)
        elif cmd == "close":
            remote.close()
            return


class VecEnv:
    def __init__(self, n_workers, envs_per_worker, seed=0, env_name="gesture", **env_kwargs):
        ctx = mp.get_context("spawn")
        self.remotes, self.procs = [], []
        for w in range(n_workers):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, seed + w, envs_per_worker, env_kwargs, env_name), daemon=True)
            p.start()
            self.remotes.append(a)
            self.procs.append(p)
        self.epw = envs_per_worker
        self.num_envs = n_workers * envs_per_worker

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        res = [r.recv() for r in self.remotes]
        return np.concatenate([o for o, _ in res]), np.concatenate([p for _, p in res])

    def step(self, actions):
        for i, r in enumerate(self.remotes):
            r.send(("step", actions[i * self.epw:(i + 1) * self.epw]))
        res = [r.recv() for r in self.remotes]
        cat = lambda k: np.concatenate([x[k] for x in res])
        finished = [f for x in res for f in x[6]]
        errs = np.mean([x[7] for x in res], 0)
        return cat(0), cat(1), cat(2), cat(3), cat(4), cat(5), finished, errs

    def set_attr(self, name, value):
        for r in self.remotes:
            r.send(("set", (name, value)))
        for r in self.remotes:
            r.recv()

    def close(self):
        for r in self.remotes:
            r.send(("close", None))
        for p in self.procs:
            p.join(timeout=2)
