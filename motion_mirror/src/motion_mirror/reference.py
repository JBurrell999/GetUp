"""Upper-body reference motions: random gestures for general training, and clips from video."""
import numpy as np

from .config import CONTROL_DT, DEFAULT_Q, UPPER, UPPER_SAMPLE_RANGE, NU

MAX_JOINT_SPEED = 4.0  # rad/s, keeps gestures within what the arm motors can do


def _min_jerk(s):
    return s * s * s * (10 - 15 * s + 6 * s * s)


class GestureGenerator:
    """Endless stream of smooth random upper-body gestures (17 joint targets at 50 Hz)."""

    def __init__(self, rng: np.random.Generator, amplitude: float = 1.0):
        self.rng = rng
        self.amplitude = amplitude
        self.default = DEFAULT_Q[UPPER].copy()

    def _sample_pose(self):
        lo, hi = UPPER_SAMPLE_RANGE[:, 0], UPPER_SAMPLE_RANGE[:, 1]
        pose = self.rng.uniform(lo, hi)
        pose = self.default + self.amplitude * (pose - self.default)
        # Often leave an arm (or the waist) relaxed, like real gestures.
        r = self.rng.random()
        if r < 0.2:
            pose[3:10] = self.default[3:10]
        elif r < 0.4:
            pose[10:17] = self.default[10:17]
        elif r < 0.5:  # mirrored, symmetric gesture
            pose[10:17] = pose[3:10] * np.array([1, -1, -1, 1, -1, 1, -1])
        if self.rng.random() < 0.5:
            pose[0:3] = self.default[0:3]
        return pose

    def sample(self, n_steps: int, start=None) -> np.ndarray:
        out = np.empty((n_steps, NU))
        cur = self.default.copy() if start is None else start.copy()
        i = 0
        while i < n_steps:
            nxt = self._sample_pose() if self.rng.random() > 0.15 else self.default.copy()
            dur = self.rng.uniform(0.4, 1.5)
            dur = max(dur, np.abs(nxt - cur).max() / MAX_JOINT_SPEED * 1.875)  # min-jerk peak speed factor
            hold = self.rng.uniform(0.0, 0.8)
            k = max(1, int(dur / CONTROL_DT))
            s = _min_jerk(np.linspace(0, 1, k + 1)[1:])[:, None]
            seg = cur + s * (nxt - cur)
            seg = np.vstack([seg, np.repeat(nxt[None], int(hold / CONTROL_DT), 0)])
            m = min(len(seg), n_steps - i)
            out[i:i + m] = seg[:m]
            i += m
            cur = nxt
        return out


def load_clip(path) -> np.ndarray:
    """A retargeted clip saved by the video pipeline: (T, 17) upper-body joints at 50 Hz."""
    clip = np.load(path)
    assert clip.ndim == 2 and clip.shape[1] == NU, clip.shape
    return clip
