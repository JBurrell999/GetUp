"""Robot + training constants for the Unitree G1 (29-DoF, no hands)."""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCENE_XML = ROOT / "assets" / "unitree_g1" / "scene.xml"

JOINTS = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
    "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]
NJ = len(JOINTS)                      # 29, same order as the MJCF actuators
LEG = np.arange(0, 12)
UPPER = np.arange(12, 29)             # waist + arms: what we retarget from video
NU = len(UPPER)                       # 17

# Standing pose: slight knee bend (more robust than locked knees), relaxed arms.
DEFAULT_Q = np.zeros(NJ)
DEFAULT_Q[[0, 6]] = -0.1              # hip pitch
DEFAULT_Q[[3, 9]] = 0.3               # knee
DEFAULT_Q[[4, 10]] = -0.2             # ankle pitch
DEFAULT_Q[[15, 22]] = 0.2             # shoulder pitch
DEFAULT_Q[16], DEFAULT_Q[23] = 0.2, -0.2  # shoulder roll (outward)
DEFAULT_Q[[18, 25]] = 1.0             # elbow

# PD gains close to what is used on the real G1's low-level controller.
# Verify against the deployment stack before running on hardware.
_KP = {"hip": 100, "knee": 150, "ankle": 40, "waist": 200, "shoulder": 60, "elbow": 50,
       "wrist_roll": 30, "wrist_pitch": 20, "wrist_yaw": 20}
_KD = {"hip": 2, "knee": 4, "ankle": 2, "waist": 5, "shoulder": 2, "elbow": 2,
       "wrist_roll": 1, "wrist_pitch": 1, "wrist_yaw": 1}


def _group(name: str) -> str:
    for g in ("wrist_roll", "wrist_pitch", "wrist_yaw", "hip", "knee", "ankle", "waist", "shoulder", "elbow"):
        if g in name:
            return g
    raise KeyError(name)


KP = np.array([_KP[_group(j)] for j in JOINTS], dtype=float)
KD = np.array([_KD[_group(j)] for j in JOINTS], dtype=float)

# Ranges the gesture generator samples upper-body targets from (subset of hard limits,
# biased away from arm-through-torso poses).
UPPER_SAMPLE_RANGE = np.array([
    [-0.6, 0.6], [-0.25, 0.25], [-0.1, 0.45],                          # waist yaw/roll/pitch
    [-3.0, 1.0], [-0.15, 2.0], [-1.5, 1.5], [-0.8, 2.0], [-1.5, 1.5], [-1.0, 1.0], [-1.0, 1.0],   # left arm
    [-3.0, 1.0], [-2.0, 0.15], [-1.5, 1.5], [-0.8, 2.0], [-1.5, 1.5], [-1.0, 1.0], [-1.0, 1.0],   # right arm
])

CONTROL_DT = 0.02                     # 50 Hz policy, as on the real robot
SIM_DT = 0.002
DECIMATION = int(round(CONTROL_DT / SIM_DT))
ACTION_SCALE = 0.25
REF_LOOKAHEAD = (0, 5, 15)            # reference frames shown to the policy: now, +0.1 s, +0.3 s

# Domain randomization (sim-to-real).
DR = dict(
    friction=(0.4, 1.25),
    body_mass_scale=(0.9, 1.1),
    torso_added_mass=(-1.0, 3.0),
    torso_com_offset=0.03,
    kp_scale=(0.85, 1.15),
    kd_scale=(0.8, 1.2),
    torque_limit_scale=(0.85, 1.0),
    joint_friction_scale=(0.5, 1.5),
    armature_scale=(0.8, 1.2),
    encoder_offset=0.02,              # rad, per-joint calibration error
    latency_substeps=(0, 10),         # 0-20 ms action delay
    push_interval=(3.0, 8.0),         # s
    push_force=(0.0, 120.0),          # N, horizontal on pelvis
    push_duration=(0.05, 0.2),        # s
)

# Per-observation Gaussian noise (std), in obs units.
OBS_NOISE = dict(gyro=0.15, gravity=0.05, q=0.01, dq=1.5)
