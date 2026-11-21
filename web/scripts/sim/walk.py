"""
A looping walk for the Menagerie Unitree G1, generated kinematically inside the MuJoCo scene.

There is no walking controller for the G1 in this project, so this doesn't simulate physics.
Instead it plans footsteps around an oval in the room's centre aisle, moves the pelvis along
the oval with a little sway and bob, and solves each leg's six joints with damped least
squares on MuJoCo's own Jacobians, so a planted foot stays exactly where it landed.

The result is one qpos per frame (36 numbers: free joint + 29 hinges) that repeats seamlessly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

# ---------------------------------------------------------------- the loop (metres, room frame)
R_TURN = 0.95          # radius of the two U-turns
Y0, Y1 = -4.5, 1.25    # straights run between these y values, at x = +/- R_TURN
L_STRAIGHT = Y1 - Y0
L_TOTAL = 2 * L_STRAIGHT + 2 * math.pi * R_TURN

# ---------------------------------------------------------------- gait
FPS = 30
SPEED = 0.45           # m/s, adjusted slightly so the loop is a whole number of frames
STEP = 0.25            # m per step, adjusted so the loop is an even number of steps
FOOT_HALF_WIDTH = 0.105
DOUBLE_SUPPORT = 0.22  # fraction of each step with both feet down
LIFT = 0.07            # swing foot clearance
PELVIS_Z = 0.768       # standing height is 0.793 with straight legs; a little lower keeps knees soft
SWAY = 0.028
BOB = 0.008
PELVIS_YAW = 0.05
LEAN = 0.03
ARM_SWING = 0.32
ANKLE_ABOVE_SOLE = 0.035
TOE_X = 0.12           # toe contact, ahead of the ankle
TOE_OFF = 0.3          # rad of heel lift just before the foot leaves the floor
TOE_OFF_TIME = 0.35    # fraction of a step spent peeling the heel up

# joint ranges the IK may use: the model's limits allow poses no walk needs (and a straight
# or backwards knee lets the solver jump to the wrong branch)
IK_RANGE = {
    "hip_pitch": (-1.6, 0.8), "hip_roll": (-0.4, 0.4), "hip_yaw": (-0.5, 0.5),
    "knee": (0.05, 2.2), "ankle_pitch": (-0.87, 0.52), "ankle_roll": (-0.26, 0.26),
}
LEG_SEED = [-0.25, 0.0, 0.0, 0.55, -0.3, 0.0]


def path(s: float) -> tuple[np.ndarray, float]:
    """Point and heading at arclength s, counter-clockwise seen from above."""
    s %= L_TOTAL
    if s < L_STRAIGHT:
        return np.array([R_TURN, Y0 + s]), math.pi / 2
    s -= L_STRAIGHT
    if s < math.pi * R_TURN:
        a = s / R_TURN
        return np.array([R_TURN * math.cos(a), Y1 + R_TURN * math.sin(a)]), a + math.pi / 2
    s -= math.pi * R_TURN
    if s < L_STRAIGHT:
        return np.array([-R_TURN, Y1 - s]), -math.pi / 2
    s -= L_STRAIGHT
    a = math.pi + s / R_TURN
    return np.array([R_TURN * math.cos(a), Y0 + R_TURN * math.sin(a)]), a + math.pi / 2


def left_of(h: float) -> np.ndarray:
    return np.array([-math.sin(h), math.cos(h)])


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def yaw_quat(h: float) -> np.ndarray:
    return np.array([math.cos(h / 2), 0, 0, math.sin(h / 2)])


def min_jerk(u: float) -> float:
    return u * u * u * (10 - 15 * u + 6 * u * u)


@dataclass
class Plan:
    frames: int
    speed: float
    step: float
    steps: int
    period: float  # seconds per step

    @property
    def duration(self) -> float:
        return self.frames / FPS


def make_plan() -> Plan:
    steps = int(round(L_TOTAL / STEP / 2)) * 2
    frames = int(round(L_TOTAL / SPEED * FPS))
    speed = L_TOTAL / (frames / FPS)
    step = L_TOTAL / steps
    return Plan(frames, speed, step, steps, step / speed)


def footprint(plan: Plan, k: int) -> tuple[np.ndarray, float, int]:
    """Where step k lands: position, yaw, side (+1 left, -1 right)."""
    side = 1 if k % 2 == 0 else -1
    p, h = path(k * plan.step)
    return p + side * FOOT_HALF_WIDTH * left_of(h), h, side


def land_time(plan: Plan, k: int) -> float:
    # the pelvis is halfway between the two feet when a foot lands
    return (k - 0.5) * plan.period


def smooth(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def ankle_on_toe(p: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
    """Ankle position for a foot at footprint p, rolled heel-up by pitch about its toe."""
    c, s = math.cos(yaw), math.sin(yaw)
    fwd = np.array([c, s])
    toe = p + TOE_X * fwd
    back = -TOE_X * math.cos(pitch) + ANKLE_ABOVE_SOLE * math.sin(pitch)
    up = TOE_X * math.sin(pitch) + ANKLE_ABOVE_SOLE * math.cos(pitch)
    xy = toe + back * fwd
    return np.array([xy[0], xy[1], up])


def foot_target(plan: Plan, t: float, side: int) -> tuple[np.ndarray, float, float, bool]:
    """Ankle position, yaw, heel-up pitch and whether the foot is on the floor, at time t."""
    k = math.floor(t / plan.period + 0.5)        # most recent landing, either foot
    if (k % 2 == 0) != (side == 1):
        k -= 1                                    # most recent landing of this foot
    p0, h0, _ = footprint(plan, k)
    swing = plan.period * (1 - DOUBLE_SUPPORT)
    lift_off = land_time(plan, k + 2) - swing
    if t < lift_off:
        pitch = TOE_OFF * smooth(1 - (lift_off - t) / (TOE_OFF_TIME * plan.period))
        return ankle_on_toe(p0, h0, pitch), h0, pitch, True
    u = (t - lift_off) / swing
    p1, h1, _ = footprint(plan, k + 2)
    start = ankle_on_toe(p0, h0, TOE_OFF)
    end = np.array([p1[0], p1[1], ANKLE_ABOVE_SOLE])
    a = min_jerk(u)
    pos = start + (end - start) * a
    pos[2] += LIFT * (1 - math.cos(2 * math.pi * u)) / 2
    pitch = TOE_OFF * (1 - smooth(u / 0.5))
    return pos, h0 + wrap(h1 - h0) * a, pitch, False


class Walker:
    def __init__(self, model: mujoco.MjModel):
        self.m = model
        self.d = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, self.d, model.key("stand").id)
        self.q_stand = self.d.qpos.copy()
        self.plan = make_plan()
        jq = lambda n: model.joint(n).qposadr[0]
        jd = lambda n: model.joint(n).dofadr[0]
        self.legs = {}
        for side, name in ((1, "left"), (-1, "right")):
            joints = [f"{name}_{j}_joint" for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
            self.legs[side] = dict(
                body=model.body(f"{name}_ankle_roll_link").id,
                qadr=np.array([jq(j) for j in joints]),
                dofs=np.array([jd(j) for j in joints]),
                lo=np.array([IK_RANGE[j][0] for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]),
                hi=np.array([IK_RANGE[j][1] for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]),
            )
        self.waist_yaw = jq("waist_yaw_joint")
        self.waist_pitch = jq("waist_pitch_joint")
        self.shoulder = {1: jq("left_shoulder_pitch_joint"), -1: jq("right_shoulder_pitch_joint")}
        self.elbow = {1: jq("left_elbow_joint"), -1: jq("right_elbow_joint")}
        self.arm_sign = self._arm_forward_sign()

    def _arm_forward_sign(self) -> float:
        """+1 if raising shoulder pitch swings the hand forward (+x of the robot)."""
        m, d = self.m, self.d
        hand = m.body("left_wrist_yaw_link").id
        d.qpos[:] = self.q_stand
        d.qpos[3:7] = [1, 0, 0, 0]
        xs = []
        for dq in (-0.2, 0.2):
            d.qpos[self.shoulder[1]] = self.q_stand[self.shoulder[1]] + dq
            mujoco.mj_kinematics(m, d)
            xs.append(d.xpos[hand][0])
        return 1.0 if xs[1] > xs[0] else -1.0

    def _ik(self, side: int, target_p: np.ndarray, target_yaw: float, pitch: float) -> float:
        err = self._solve(side, target_p, target_yaw, pitch, 60)
        if err > 1e-5:   # stuck: start again from the bent-knee stance
            self.d.qpos[self.legs[side]["qadr"]] = LEG_SEED
            err = self._solve(side, target_p, target_yaw, pitch, 200)
        return err

    def _solve(self, side: int, target_p: np.ndarray, target_yaw: float, pitch: float, iters: int) -> float:
        m, d, leg = self.m, self.d, self.legs[side]
        target_q = np.zeros(4)
        mujoco.mju_mulQuat(target_q, yaw_quat(target_yaw), np.array([math.cos(pitch / 2), 0, math.sin(pitch / 2), 0]))
        jp = np.zeros((3, m.nv))
        jr = np.zeros((3, m.nv))
        res = np.zeros(3)
        err = 1.0
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            b = leg["body"]
            mujoco.mju_subQuat(res, target_q, d.xquat[b])
            e = np.concatenate([target_p - d.xpos[b], d.xmat[b].reshape(3, 3) @ res])
            err = float(np.linalg.norm(e))
            if err < 1e-7:
                break
            mujoco.mj_jacBody(m, d, jp, jr, b)
            J = np.vstack([jp[:, leg["dofs"]], jr[:, leg["dofs"]]])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
            dq *= min(1.0, 0.15 / max(1e-9, float(np.abs(dq).max())))   # small steps, no flips
            d.qpos[leg["qadr"]] = np.clip(d.qpos[leg["qadr"]] + dq, leg["lo"], leg["hi"])
        return err

    def pose(self, t: float) -> tuple[np.ndarray, dict]:
        """qpos at time t, plus diagnostics."""
        m, d, plan = self.m, self.d, self.plan
        k_next = math.floor(t / plan.period + 0.5) + 1          # the step in progress
        step_start = land_time(plan, k_next) - plan.period
        swing_side = 1 if k_next % 2 == 0 else -1
        u = (t - step_start) / plan.period                        # 0..1 through this step
        stride = math.pi * (t / plan.period + 0.5)                # 0 when the left foot lands

        p, h = path(plan.speed * t)
        sway = SWAY * math.sin(math.pi * u) * (-swing_side)       # toward the planted foot
        xy = p + sway * left_of(h)
        z = PELVIS_Z - BOB * math.cos(2 * math.pi * u)
        yaw = h - PELVIS_YAW * math.cos(stride)
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, yaw_quat(yaw), np.array([math.cos(LEAN / 2), 0, math.sin(LEAN / 2), 0]))

        # keep the previous frame's legs as the IK starting point; the first frame starts
        # from Unitree's usual bent-knee stance (straight legs are a singular pose for IK)
        if hasattr(self, "_warm"):
            q = d.qpos.copy()
        else:
            q = self.q_stand.copy()
            for leg in self.legs.values():
                q[leg["qadr"]] = LEG_SEED
            self._warm = True
        q[0:3] = [xy[0], xy[1], z]
        q[3:7] = quat
        q[self.waist_yaw] = 0.8 * PELVIS_YAW * math.cos(stride)
        q[self.waist_pitch] = 0.02
        for side in (1, -1):
            # an arm swings forward with the opposite leg
            q[self.shoulder[side]] = self.q_stand[self.shoulder[side]] + self.arm_sign * side * ARM_SWING * math.cos(stride) * -1
            q[self.elbow[side]] = self.q_stand[self.elbow[side]] - 0.15
        d.qpos[:] = q

        diag = {}
        for side in (1, -1):
            fp, fyaw, pitch, planted = foot_target(plan, t, side)
            diag[side] = dict(err=self._ik(side, fp, fyaw, pitch), planted=planted, target=fp)
        return d.qpos.copy(), diag

    def run(self) -> tuple[np.ndarray, dict]:
        plan = self.plan
        # one lap to settle the warm start, then the lap we keep
        for i in range(plan.frames):
            self.pose(i / FPS)
        out = np.zeros((plan.frames, self.m.nq))
        worst = 0.0
        for i in range(plan.frames):
            out[i], diag = self.pose(i / FPS)
            worst = max(worst, diag[1]["err"], diag[-1]["err"])
        steps = []
        for k in range(plan.steps):
            p, h, side = footprint(plan, k)
            steps.append(dict(x=round(float(p[0]), 4), y=round(float(p[1]), 4), yaw=round(float(h), 4), side=side, t=round(land_time(plan, k) % plan.duration, 4)))
        info = dict(
            fps=FPS, frames=plan.frames, duration=plan.duration, speed=plan.speed, step=plan.step,
            steps_per_lap=plan.steps, step_period=plan.period, lap_length=L_TOTAL, worst_ik_error=worst,
            footsteps=steps,
        )
        return out, info
