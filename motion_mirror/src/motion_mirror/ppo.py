"""PPO with an asymmetric actor-critic (actor: real sensors, critic: privileged sim state)."""
import numpy as np
import torch
import torch.nn as nn


def mlp(i, o, hidden=(512, 256, 128)):
    layers, d = [], i
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ELU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class Normalizer(nn.Module):
    def __init__(self, dim, clip=5.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))
        self.clip = clip

    @torch.no_grad()
    def update(self, x):
        bm, bv, bc = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta, tot = bm - self.mean, self.count + bc
        self.mean += delta * bc / tot
        self.var = (self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def forward(self, x):
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + 1e-8), -self.clip, self.clip)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, priv_dim, act_dim, init_std=0.8):
        super().__init__()
        self.obs_norm, self.priv_norm = Normalizer(obs_dim), Normalizer(priv_dim)
        self.actor, self.critic = mlp(obs_dim, act_dim), mlp(priv_dim, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(np.log(init_std))))

    def dist(self, obs):
        mu = self.actor(self.obs_norm(obs))
        return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))

    def value(self, priv):
        return self.critic(self.priv_norm(priv)).squeeze(-1)

    @torch.no_grad()
    def act_deterministic(self, obs):
        return self.actor(self.obs_norm(obs))


class PPO:
    def __init__(self, ac: ActorCritic, lr=3e-4, gamma=0.99, lam=0.95, clip=0.2, epochs=5, minibatches=4,
                 ent_coef=0.003, vf_coef=1.0, max_grad=1.0, desired_kl=0.01):
        self.ac = ac
        self.opt = torch.optim.Adam(ac.parameters(), lr=lr)
        self.lr, self.gamma, self.lam, self.clip = lr, gamma, lam, clip
        self.epochs, self.mb, self.ent, self.vf, self.max_grad, self.kl_t = epochs, minibatches, ent_coef, vf_coef, max_grad, desired_kl

    def update(self, buf):
        obs, priv, act, logp_old, adv, ret, val_old = (buf[k] for k in ("obs", "priv", "act", "logp", "adv", "ret", "val"))
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = obs.shape[0]
        stats = dict(pi_loss=0.0, v_loss=0.0, kl=0.0, std=0.0)
        k = 0
        for _ in range(self.epochs):
            for idx in torch.randperm(n).chunk(self.mb):
                d = self.ac.dist(obs[idx])
                logp = d.log_prob(act[idx]).sum(-1)
                with torch.no_grad():
                    kl = (logp_old[idx] - logp).mean().clamp(min=0)
                    if kl > self.kl_t * 2:
                        self.lr = max(1e-5, self.lr / 1.5)
                    elif kl < self.kl_t / 2:
                        self.lr = min(1e-3, self.lr * 1.5)
                    for g in self.opt.param_groups:
                        g["lr"] = self.lr
                ratio = torch.exp(logp - logp_old[idx])
                pi_loss = -torch.min(ratio * adv[idx], ratio.clamp(1 - self.clip, 1 + self.clip) * adv[idx]).mean()
                v = self.ac.value(priv[idx])
                v_clip = val_old[idx] + (v - val_old[idx]).clamp(-self.clip, self.clip)
                v_loss = torch.max((v - ret[idx]) ** 2, (v_clip - ret[idx]) ** 2).mean()
                loss = pi_loss + self.vf * v_loss - self.ent * d.entropy().sum(-1).mean()
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad)
                self.opt.step()
                stats["pi_loss"] += pi_loss.item(); stats["v_loss"] += v_loss.item(); stats["kl"] += kl.item()
                k += 1
        stats = {a: b / k for a, b in stats.items()}
        stats["std"] = self.ac.log_std.exp().mean().item()
        stats["lr"] = self.lr
        return stats

    def gae(self, rew, done, val, last_val):
        T = rew.shape[0]
        adv = torch.zeros_like(rew)
        gae = 0
        for t in reversed(range(T)):
            nv = last_val if t == T - 1 else val[t + 1]
            nonterm = 1.0 - done[t]
            delta = rew[t] + self.gamma * nv * nonterm - val[t]
            gae = delta + self.gamma * self.lam * nonterm * gae
            adv[t] = gae
        return adv, adv + val
