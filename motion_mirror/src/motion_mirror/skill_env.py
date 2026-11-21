"""Whole-body skill policy: the executor for Gemini's plans.

Commands: walk velocity (vx, vy), turn rate (wz), pelvis height (squat), upper-body pose (17 joints).
Trained once; Gemini plans are sequences of these commands. Random payloads in the hands teach it to
carry objects. All sim-to-real randomization from G1Env still applies.
"""
import mujoco
import numpy as np

from . import config as C
from .g1_env import EP_STEPS, OBS_DIM, PRIV_DIM, UPPER_W, G1Env

THIGH, SHIN = 0.3409, 0.300
H_MIN = 0.45
GAIT_PERIOD = 0.8
H_RATE = 0.3  # m/s, max squat speed
CMD_RANGES = dict(vx=(-0.4, 0.8), vy=(-0.3, 0.3), wz=(-0.8, 0.8))
CMD_SCALE = np.array([2.0, 2.0, 1.0, 4.0])


SKILL_OBS_DIM, SKILL_PRIV_DIM = OBS_DIM + 6, PRIV_DIM + 7


def squat_leg_pose(dh):
    """Nominal 12 leg joints for a pelvis height change dh (<= 0), torso upright, feet flat."""
    d0 = np.sqrt(THIGH**2 + SHIN**2 + 2 * THIGH * SHIN * np.cos(0.3))
    D = np.clip(d0 + dh, 0.25, d0)
    knee = np.arccos(np.clip((D**2 - THIGH**2 - SHIN**2) / (2 * THIGH * SHIN), -1, 1))
    a = np.arcsin(SHIN * np.sin(knee) / D)
    q = C.DEFAULT_Q[C.LEG].copy()
    for s in (0, 6):
        q[s + 0], q[s + 3], q[s + 4] = -a, knee, -(knee - a)
    return q


class SkillEnv(G1Env):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        bid = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n)
        self.wrists = [bid("left_wrist_yaw_link"), bid("right_wrist_yaw_link")]
        self.nom["wrist_mass"] = self.m.body_mass[self.wrists].copy()
        self.cmd_scale = 1.0  # curriculum on walking speed
        self.payload_max = 3.0
        self.plan = None      # when set: callable t -> (cmd[4], upper[17]) for executing Gemini plans
        self.amp_full = 1.0
        self.mode = "manip"

    # Training alternates "walk" episodes (small arm motion, always moving) and "manip" episodes
    # (stand/squat with full arm motion), so walking isn't drowned out by the other tasks.
    @property
    def gen_amplitude(self):
        return self.amp_full

    @gen_amplitude.setter
    def gen_amplitude(self, v):
        self.amp_full = v

    # ----- commands -----
    def _resample_cmd(self):
        r, u = self.rng.random(), self.rng.uniform
        v = np.zeros(3)
        walk_prob = 0.95 if self.mode == "walk" else 0.25
        if r < walk_prob:  # otherwise stand still
            v = np.array([u(*CMD_RANGES[k]) for k in ("vx", "vy", "wz")]) * self.cmd_scale
            if self.mode == "manip":
                v *= 0.5
            if self.rng.random() < 0.3:
                v[1] = 0.0
            if np.linalg.norm(v[:2]) < 0.1:
                v[:2] = 0
        h = self._base_height
        if self.mode == "manip" and self.rng.random() < 0.5:  # squat; only while (nearly) standing still
            h = u(H_MIN, self._base_height)
            v *= 0.0 if h < 0.65 else 0.5
        self.cmd = np.array([*v, h])
        self.next_cmd = self.t + int(u(3.0, 7.0) / C.CONTROL_DT)

    def _cmd_now(self):
        cmd, ref = (self.plan(self.t) if self.plan is not None else (self.cmd, self.ref[self.t + 1]))
        cmd = np.array(cmd, float)
        cmd[3] = getattr(self, "h_smooth", cmd[3])
        return cmd, ref

    def _update_height(self):
        goal = (self.plan(self.t)[0][3] if self.plan is not None else self.cmd[3])
        step = H_RATE * C.CONTROL_DT
        self.h_smooth += np.clip(goal - self.h_smooth, -step, step)

    # ----- episode -----
    def reset(self):
        if self.randomize:
            payload = self.rng.uniform(0, self.payload_max) if self.rng.random() < 0.5 else 0.0
        else:
            payload = 0.0
        self.payload = payload
        self.mode = "walk" if (self.randomize and self.rng.random() < 0.5) else "manip"
        self.gen.amplitude = 0.25 * self.amp_full if self.mode == "walk" else self.amp_full
        out = super().reset()
        if self.mode == "walk":
            self.push_scale_ep = 0.6
        else:
            self.push_scale_ep = 1.0
        self.m.body_mass[self.wrists] += payload / 2  # object carried in the hands
        mujoco.mj_setConst(self.m, self.d)
        self.phase = 0.0
        self.h_smooth = self._base_height
        self._resample_cmd()
        self.feet_air = np.zeros(2)
        return self._obs()

    def step(self, action):
        d, m = self.d, self.m
        if self.plan is None and self.t >= self.next_cmd:
            self._resample_cmd()
        self._update_height()
        cmd, ref_next = self._cmd_now()
        a = np.clip(action, -4.0, 4.0)
        target = np.empty(C.NJ)
        target[C.LEG] = squat_leg_pose(cmd[3] - self._base_height) + C.ACTION_SCALE * a[C.LEG]
        target[C.UPPER] = ref_next + C.ACTION_SCALE * a[C.UPPER]

        if self.randomize and self.t >= self.next_push:
            ang = self.rng.uniform(0, 2 * np.pi)
            mag = self.rng.uniform(*C.DR["push_force"]) * self.push_scale * getattr(self, "push_scale_ep", 1.0)
            self.push_force = mag * np.array([np.cos(ang), np.sin(ang), 0.0])
            self.push_until = self.t + max(1, int(self.rng.uniform(*C.DR["push_duration"]) / C.CONTROL_DT))
            self._schedule_push()
        if self.t >= self.push_until:
            self.push_force[:] = 0
        d.xfrc_applied[self.pelvis, :3] = self.push_force
        for k in range(C.DECIMATION):
            d.ctrl[:] = self.prev_target if k < self.latency else target
            mujoco.mj_step(m, d)
        self.prev_target = target
        self.t += 1
        self.phase = (self.phase + C.CONTROL_DT / GAIT_PERIOD) % 1.0

        rew, info = self._reward(a, cmd)
        self.last_action = a
        g = self._gravity()
        fell = d.qpos[2] < 0.3 or g[2] > -0.5
        if fell:
            rew -= 10.0
        truncated = (not fell) and self.t >= EP_STEPS
        return self._obs(), rew, fell, truncated, info

    # ----- reward -----
    def _yaw_frame_vel(self):
        d = self.d
        R = d.xmat[self.pelvis].reshape(3, 3)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        v = d.qvel[0:3]
        return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1], v[2]]), R.T @ d.qvel[3:6]

    def _reward(self, a, cmd=None):
        d = self.d
        if cmd is None:
            cmd = self.cmd
        q, dq = self._q(), self._dq()
        ref = self.ref[self.t] if self.plan is None else self._cmd_now()[1]
        err = q[C.UPPER] - ref
        mse = np.mean(UPPER_W * err ** 2)
        kp_err = np.mean(np.sum((self._keypoints(d) - self._ref_keypoints(ref)) ** 2, axis=1))
        v, w = self._yaw_frame_vel()
        g = self._gravity()
        contacts = np.array(self._foot_contacts())
        feet = d.xpos[self.feet]
        slip = sum(np.sum(((feet[k, :2] - self.prev_feet[k]) / C.CONTROL_DT) ** 2) for k in (0, 1) if contacts[k])
        self.prev_feet = feet[:, :2].copy()
        moving = np.linalg.norm(cmd[:3]) > 0.05
        if moving:  # alternate stance/swing from the gait clock
            stance = np.array([((self.phase + off) % 1.0) < 0.55 for off in (0.0, 0.5)])
            gait = np.mean(stance == contacts)
            swing_h = np.sum(((feet[:, 2] - 0.09) ** 2) * (~stance))
            still = 0.0
        else:
            gait = float(contacts.all())
            swing_h = 0.0
            still = slip
        lo, hi = self.m.jnt_range[1:1 + C.NJ, 0], self.m.jnt_range[1:1 + C.NJ, 1]
        mid, half = (lo + hi) / 2, (hi - lo) / 2 * 0.95
        limit = np.sum(np.clip(np.abs(q - mid) - half, 0, None))
        leg_nom = squat_leg_pose(cmd[3] - self._base_height)
        terms = dict(
            lin_vel=4.0 * np.exp(-np.sum((cmd[:2] - v[:2]) ** 2) / 0.15),
            yaw_rate=2.0 * np.exp(-(cmd[2] - w[2]) ** 2 / 0.15),
            height=1.5 * np.exp(-(cmd[3] - d.qpos[2]) ** 2 / 0.004),
            track_joints=2.0 * np.exp(-mse / 0.0625),
            track_hands=1.0 * np.exp(-kp_err / 0.01),
            gait=0.6 * gait,
            alive=0.3,
            swing_height=-15.0 * swing_h,
            upright=-2.0 * np.sum(g[:2] ** 2),
            vel_z=-1.0 * v[2] ** 2,
            ang_vel_xy=-0.05 * np.sum(w[:2] ** 2),
            hip_rollyaw=-0.5 * np.sum(q[[1, 2, 7, 8]] ** 2),
            leg_pose=-0.05 * np.sum((q[C.LEG] - leg_nom) ** 2),
            feet_slip=-0.3 * slip - 0.5 * still,
            action_rate=-0.01 * np.sum((a - self.last_action) ** 2),
            joint_vel=-1e-4 * np.sum(dq ** 2),
            torque=-1e-5 * np.sum(d.actuator_force ** 2),
            joint_limit=-2.0 * limit,
        )
        info = dict(joint_err=float(np.sqrt(np.mean(err ** 2))), hand_err=float(np.sqrt(kp_err)),
                    vel_err=float(np.linalg.norm(cmd[:2] - v[:2])), height_err=float(abs(cmd[3] - d.qpos[2])), **terms)
        return float(sum(terms.values())), info

    # ----- observations -----
    def _obs(self):
        obs, priv = super()._obs()
        cmd = self._cmd_now()[0] if hasattr(self, "cmd") else np.array([0, 0, 0, self._base_height])
        phase = getattr(self, "phase", 0.0)
        extra = np.concatenate([(cmd - np.array([0, 0, 0, self._base_height])) * CMD_SCALE,
                                [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)]]).astype(np.float32)
        payload = np.array([getattr(self, "payload", 0.0) / 3.0], np.float32)
        return np.concatenate([obs, extra]), np.concatenate([priv, extra, payload])

