"""G1 upper-body motion tracking environment with balance + domain randomization.

The policy only observes what the real robot measures (pelvis IMU, joint encoders), plus the
reference motion. The critic additionally gets privileged sim state (asymmetric actor-critic).
"""
import mujoco
import numpy as np

from . import config as C
from .reference import GestureGenerator

EP_STEPS = 1000  # 20 s
OBS_DIM = 3 + 3 + C.NJ * 3 + C.NU * len(C.REF_LOOKAHEAD)
PRIV_DIM = OBS_DIM + 13
UPPER_W = np.array([1, 1, 1] + [1, 1, 1, 1, .5, .5, .5] * 2)
UPPER_W = UPPER_W / UPPER_W.mean()


class G1Env:
    def __init__(self, seed=0, clips=None, clip_prob=0.0, randomize=True, xml_path=None):
        self.rng = np.random.default_rng(seed)
        self.m = mujoco.MjModel.from_xml_path(str(xml_path or C.SCENE_XML))
        self.d = mujoco.MjData(self.m)
        self.ref_d = mujoco.MjData(self.m)
        self.m.opt.timestep = C.SIM_DT
        self.randomize = randomize
        self.clips = clips or []
        self.clip_prob = clip_prob
        self.gen = GestureGenerator(self.rng)
        self.push_scale = 1.0

        m = self.m
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        self.pelvis, self.torso = bid("pelvis"), bid("torso_link")
        self.feet = [bid("left_ankle_roll_link"), bid("right_ankle_roll_link")]
        self.hands = [bid("left_wrist_yaw_link"), bid("right_wrist_yaw_link"),
                      bid("left_elbow_link"), bid("right_elbow_link")]
        self.floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.foot_geoms = [set(np.where(m.geom_bodyid == f)[0]) for f in self.feet]
        self.gyro_adr = m.sensor("imu-pelvis-angular-velocity").adr[0]
        self.qadr, self.vadr = 7, 6

        # Nominal values that domain randomization perturbs.
        self.nom = dict(mass=m.body_mass.copy(), inertia=m.body_inertia.copy(), ipos=m.body_ipos.copy(),
                        friction=m.geom_friction.copy(), frictionloss=m.dof_frictionloss.copy(),
                        armature=m.dof_armature.copy(), forcerange=m.actuator_forcerange.copy())
        self._base_height = self._standing_height()

    @property
    def gen_amplitude(self):
        return self.gen.amplitude

    @gen_amplitude.setter
    def gen_amplitude(self, v):
        self.gen.amplitude = v

    # ---------- setup ----------
    def _standing_height(self):
        d = self.ref_d
        d.qpos[:] = 0
        d.qpos[3] = 1
        d.qpos[self.qadr:self.qadr + C.NJ] = C.DEFAULT_Q
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_collision(self.m, d)
        foot_bottom = min(d.geom_xpos[g][2] - self.m.geom_size[g][0] * (self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE)
                          for gs in self.foot_geoms for g in gs)
        return -foot_bottom + 0.002

    def _apply_dr(self):
        m, n, R, u = self.m, self.nom, C.DR, self.rng.uniform
        if not self.randomize:
            self.kp, self.kd = C.KP.copy(), C.KD.copy()
            self.enc_offset = np.zeros(C.NJ)
            self.latency = 0
            self.fric = 1.0
            self.added_mass = 0.0
            self._set_gains()
            return
        s = u(*R["body_mass_scale"], m.nbody)
        m.body_mass[:] = n["mass"] * s
        m.body_inertia[:] = n["inertia"] * s[:, None]
        self.added_mass = u(*R["torso_added_mass"])
        m.body_mass[self.torso] += self.added_mass
        m.body_ipos[:] = n["ipos"]
        m.body_ipos[self.torso] += u(-1, 1, 3) * R["torso_com_offset"]
        self.fric = u(*R["friction"])
        m.geom_friction[:] = n["friction"]
        m.geom_friction[:, 0] *= self.fric
        m.dof_frictionloss[:] = n["frictionloss"] * u(*R["joint_friction_scale"], m.nv)
        m.dof_armature[:] = n["armature"] * u(*R["armature_scale"], m.nv)
        m.actuator_forcerange[:] = n["forcerange"] * u(*R["torque_limit_scale"], (m.nu, 1))
        self.kp = C.KP * u(*R["kp_scale"], C.NJ)
        self.kd = C.KD * u(*R["kd_scale"], C.NJ)
        self._set_gains()
        self.enc_offset = u(-1, 1, C.NJ) * R["encoder_offset"]
        self.latency = self.rng.integers(R["latency_substeps"][0], R["latency_substeps"][1] + 1)
        mujoco.mj_setConst(m, self.d)

    def _set_gains(self):
        self.m.actuator_gainprm[:, 0] = self.kp
        self.m.actuator_biasprm[:, 1] = -self.kp
        self.m.actuator_biasprm[:, 2] = -self.kd

    def _new_reference(self, n):
        if self.clips and self.rng.random() < self.clip_prob:
            clip = self.clips[self.rng.integers(len(self.clips))]
            # Loop the clip, blending smoothly from its last frame back to its first.
            s = (lambda x: x ** 3 * (10 - 15 * x + 6 * x ** 2))(np.linspace(0, 1, 27)[1:-1])[:, None]
            loop = np.vstack([clip, clip[-1] + s * (clip[0] - clip[-1])])
            seq = np.vstack([loop] * (n // len(loop) + 2))
            start = self.rng.integers(len(loop))
            return seq[start:start + n]
        return self.gen.sample(n)

    def _schedule_push(self):
        self.next_push = self.t + int(self.rng.uniform(*C.DR["push_interval"]) / C.CONTROL_DT)

    # ---------- API ----------
    def reset(self):
        self._apply_dr()
        d = self.d
        mujoco.mj_resetData(self.m, d)
        self.ref = self._new_reference(EP_STEPS + max(C.REF_LOOKAHEAD) + 2)
        q = C.DEFAULT_Q + self.rng.normal(0, 0.03, C.NJ) * self.randomize
        q[C.UPPER] = self.ref[0]
        d.qpos[2] = self._base_height
        d.qpos[3] = 1
        d.qpos[self.qadr:self.qadr + C.NJ] = q
        d.ctrl[:] = q
        mujoco.mj_forward(self.m, d)
        self.t = 0
        self.last_action = np.zeros(C.NJ)
        self.prev_target = q.copy()
        self.push_force = np.zeros(3)
        self.push_until = -1
        self._schedule_push()
        self.prev_feet = d.xpos[self.feet, :2].copy()
        return self._obs()

    def step(self, action):
        d, m = self.d, self.m
        a = np.clip(action, -4.0, 4.0)
        ref_next = self.ref[self.t + 1]
        target = np.empty(C.NJ)
        target[C.LEG] = C.DEFAULT_Q[C.LEG] + C.ACTION_SCALE * a[C.LEG]
        target[C.UPPER] = ref_next + C.ACTION_SCALE * a[C.UPPER]

        # Random pushes on the pelvis.
        if self.randomize and self.t >= self.next_push:
            ang = self.rng.uniform(0, 2 * np.pi)
            mag = self.rng.uniform(*C.DR["push_force"]) * self.push_scale
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

        rew, info = self._reward(a)
        self.last_action = a
        g = self._gravity()
        fell = d.qpos[2] < 0.5 or g[2] > -0.6
        if fell:
            rew -= 20.0
        truncated = (not fell) and self.t >= EP_STEPS
        return self._obs(), rew, fell, truncated, info

    # ---------- internals ----------
    def _gravity(self):
        return self.d.xmat[self.pelvis].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])

    def _q(self):
        return self.d.qpos[self.qadr:self.qadr + C.NJ]

    def _dq(self):
        return self.d.qvel[self.vadr:self.vadr + C.NJ]

    def _keypoints(self, d):
        R = d.xmat[self.torso].reshape(3, 3)
        return (d.xpos[self.hands] - d.xpos[self.torso]) @ R

    def _ref_keypoints(self, ref_upper):
        rd = self.ref_d
        rd.qpos[:] = 0
        rd.qpos[3] = 1
        rd.qpos[self.qadr:self.qadr + C.NJ] = C.DEFAULT_Q
        rd.qpos[self.qadr + C.UPPER] = ref_upper
        mujoco.mj_kinematics(self.m, rd)
        return self._keypoints(rd)

    def _foot_contacts(self):
        c = [False, False]
        d = self.d
        for i in range(d.ncon):
            g1, g2 = d.contact[i].geom1, d.contact[i].geom2
            other = g2 if g1 == self.floor else g1 if g2 == self.floor else None
            if other is None:
                continue
            for k in (0, 1):
                if other in self.foot_geoms[k]:
                    c[k] = True
        return c

    def _reward(self, a):
        d = self.d
        q, dq = self._q(), self._dq()
        ref = self.ref[self.t]
        err = q[C.UPPER] - ref
        mse = np.mean(UPPER_W * err ** 2)
        kp_err = np.mean(np.sum((self._keypoints(d) - self._ref_keypoints(ref)) ** 2, axis=1))
        g = self._gravity()
        contacts = self._foot_contacts()
        feet = d.xpos[self.feet, :2]
        slip = sum(np.sum(((feet[k] - self.prev_feet[k]) / C.CONTROL_DT) ** 2) for k in (0, 1) if contacts[k])
        self.prev_feet = feet.copy()
        lo, hi = self.m.jnt_range[1:1 + C.NJ, 0], self.m.jnt_range[1:1 + C.NJ, 1]
        mid, half = (lo + hi) / 2, (hi - lo) / 2 * 0.95
        limit = np.sum(np.clip(np.abs(q - mid) - half, 0, None))

        terms = dict(
            track_joints=3.0 * np.exp(-mse / 0.0625),
            track_hands=1.5 * np.exp(-kp_err / 0.01),
            alive=0.5,
            upright=-2.0 * np.sum(g[:2] ** 2),
            height=-10.0 * (d.qpos[2] - self._base_height) ** 2,
            leg_pose=-0.1 * np.sum((q[C.LEG] - C.DEFAULT_Q[C.LEG]) ** 2),
            feet_slip=-0.5 * slip,
            both_feet_down=-0.3 * (not all(contacts)),
            action_rate=-0.01 * np.sum((a - self.last_action) ** 2),
            joint_vel=-1e-4 * np.sum(dq ** 2),
            torque=-1e-5 * np.sum(d.actuator_force ** 2),
            joint_limit=-2.0 * limit,
        )
        info = dict(joint_err=float(np.sqrt(np.mean(err ** 2))), hand_err=float(np.sqrt(kp_err)), **terms)
        return float(sum(terms.values())), info

    def _obs(self):
        d = self.d
        q, dq = self._q(), self._dq()
        gyro = d.sensordata[self.gyro_adr:self.gyro_adr + 3]
        g = self._gravity()
        refs = [self.ref[self.t + 1 + L] - C.DEFAULT_Q[C.UPPER] for L in C.REF_LOOKAHEAD]
        clean = np.concatenate([gyro * 0.25, g, q - C.DEFAULT_Q, dq * 0.05, self.last_action, *refs])
        obs = clean.copy()
        if self.randomize:
            N, n = C.OBS_NOISE, self.rng.normal
            obs[0:3] += n(0, N["gyro"] * 0.25, 3)
            obs[3:6] += n(0, N["gravity"], 3)
            obs[6:6 + C.NJ] += self.enc_offset + n(0, N["q"], C.NJ)
            obs[6 + C.NJ:6 + 2 * C.NJ] += n(0, N["dq"] * 0.05, C.NJ)
        R = d.xmat[self.pelvis].reshape(3, 3)
        lin_vel = R.T @ d.qvel[0:3]
        priv = np.concatenate([clean, lin_vel * 2.0, [d.qpos[2] - self._base_height, self.fric - 1.0,
                               self.added_mass / 3.0, self.kp.mean() / C.KP.mean() - 1.0],
                               self.push_force / 100.0, np.array(self._foot_contacts(), float),
                               [self.latency / 10.0]])
        return obs.astype(np.float32), priv.astype(np.float32)
